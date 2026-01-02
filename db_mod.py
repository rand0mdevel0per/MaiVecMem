import uuid
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union

import asyncpg
import numpy as np
import openai


@dataclass
class MemorySearchConfig:
    dropout_rate: float = 0.3  # 随机丢弃 30% 的弱边
    min_edge_weight: float = 0.2  # 最小边权重阈值
    max_depth: int = 5
    strengthen_boost: float = 0.1
    similarity_threshold: float = 0.75  # topic 相似度阈值
    auto_link: bool = True  # 自动链接相似 topic


class GraphMemoryDB:
    def __init__(self, embedder: str, db_conn: asyncpg.Connection, emb_api_sk: str, emb_model: str):
        self.embedder = openai.OpenAI(api_key=emb_api_sk, base_url=embedder)
        self.conn = db_conn
        self.emb_model = emb_model

    async def _embed(self, input_data: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        生成 embedding 向量 (支持 Batch 处理)
        优化：合并 HTTP 请求，大幅减少网络延迟
        """
        # 确保输入不为空
        if not input_data:
            return [] if isinstance(input_data, list) else np.array([])

        resp = self.embedder.embeddings.create(input=input_data, model=self.emb_model, timeout=30)

        if isinstance(input_data, str):
            return np.array(resp.data[0].embedding)
        else:
            return [np.array(d.embedding) for d in resp.data]

    # ========== 写入函数 ==========

    async def add_mem(self, topic: str, instances: List[str], config: Optional[MemorySearchConfig] = None) -> bool:
        """
        添加记忆：topic -> [instance1, instance2, ...]
        优化：全链路 Batch 处理
        """
        if not instances:
            return False

        config = config or MemorySearchConfig()

        try:
            # 1. 生成 topic embedding
            topic_emb = await self._embed(topic)

            # 2. 检查或创建 Topic (复用逻辑保持不变，但SQL微调)
            existing_topic = await self.conn.fetchrow(
                """
                SELECT id, topic_name, 
                       1 - (topic_embedding <=> $1) as similarity
                FROM memory_topics
                ORDER BY topic_embedding <=> $1
                LIMIT 1
            """,
                topic_emb.tolist(),
            )

            if existing_topic and existing_topic["similarity"] > 0.9:
                topic_id = existing_topic["id"]
                # print(f"[INFO] Reusing existed topic: {existing_topic['topic_name']}")
            else:
                topic_id = await self.conn.fetchval(
                    """
                    INSERT INTO memory_topics (topic_name, topic_embedding)
                    VALUES ($1, $2)
                    RETURNING id
                """,
                    topic,
                    topic_emb.tolist(),
                )

                if config.auto_link and existing_topic:
                    similarity = existing_topic["similarity"]
                    if similarity > config.similarity_threshold:
                        await self._create_edge(existing_topic["id"], topic_id, similarity)

            # 3. 批量处理 Instances (核心优化点)

            # A. 批量去重查询
            # 一次性查出该 Topic 下已存在的这些内容
            existing_contents = await self.conn.fetch(
                """
                SELECT content FROM memory_instances 
                WHERE topic_id = $1 AND content = ANY($2::text[])
            """,
                topic_id,
                instances,
            )

            existing_set = {r["content"] for r in existing_contents}
            new_instances = [inst for inst in instances if inst not in existing_set]

            if not new_instances:
                return True

            # B. 批量生成 Embedding (一次 HTTP 请求代替 N 次)
            new_embeddings = await self._embed(new_instances)

            # C. 准备批量插入数据
            instance_data = []
            for content, content_emb in zip(new_instances, new_embeddings, strict=True):
                # 计算与 Topic 的相关度 (Distance 越小越相关)
                # Cosine Distance = 1 - Cosine Similarity
                # 这里简单估算：直接用 1 - distance 作为相关度
                dist = np.linalg.norm(topic_emb - content_emb)
                relevance = max(0.0, 1.0 - float(dist / 2))  # 简单的归一化

                instance_data.append((str(uuid.uuid4()), topic_id, content, content_emb.tolist(), float(relevance)))

            # D. 批量插入
            if instance_data:
                await self.conn.executemany(
                    """
                    INSERT INTO memory_instances 
                    (id, topic_id, content, content_embedding, relevance_score)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                    instance_data,
                )

            # print(f"[INFO] Added topic: {topic}, {len(instance_data)} instances")
            return True

        except Exception as e:
            print(f"Error in add_mem: {e}")
            return False

    async def batch_add_mem(
        self, batch: List[Tuple[str, List[str]]], config: Optional[MemorySearchConfig] = None
    ) -> bool:
        """批量添加记忆"""
        if not batch:
            return False
        config = config or MemorySearchConfig()

        success_count = 0
        # 使用 transaction 保证数据一致性
        async with self.conn.transaction():
            for topic, instances in batch:
                if await self.add_mem(topic, instances, config):
                    success_count += 1
        return success_count == len(batch)

    async def _create_edge(self, from_topic: str, to_topic: str, weight: float):
        """创建 topic 之间的边 (Upsert)"""
        await self.conn.execute(
            """
            INSERT INTO topic_edges (from_topic, to_topic, weight)
            VALUES ($1, $2, $3)
            ON CONFLICT (from_topic, to_topic) 
            DO UPDATE SET weight = GREATEST(topic_edges.weight, $3)
        """,
            from_topic,
            to_topic,
            weight,
        )

    # ========== 读取函数 ==========

    async def read_mem(
        self, query: str, config: Optional[MemorySearchConfig] = None, top_k: int = 5
    ) -> List[Tuple[str, List[str]]]:
        """
        读取记忆
        优化：使用 LATERAL JOIN 将 N+1 次查询压缩为 2 次 (Search + Fetch)
        """
        config = config or MemorySearchConfig()
        query_emb = await self._embed(query)

        # 1. 图遍历找相关 topics
        relevant_topics = await self._search_topics_with_dropout(query_emb, config)

        if not relevant_topics:
            return []

        # 提取前 top_k 的 topic IDs
        top_topics = relevant_topics[:top_k]
        top_topic_ids = [t["id"] for t in top_topics]

        # 建立 ID 到 Name 的映射，方便最后组装
        topic_map = {t["id"]: t["topic_name"] for t in top_topics}

        # 2. 批量增强 Topic 活跃度 (一次更新)
        await self.conn.execute(
            """
            UPDATE memory_topics 
            SET access_count = access_count + 1, last_accessed = NOW(), strength = LEAST(strength + $2, 1.0)
            WHERE id = ANY($1::text[])
        """,
            top_topic_ids,
            config.strengthen_boost,
        )

        # 3. 超级 SQL：一次性抓取所有 Topic 下的最佳 Instances
        # 使用 LATERAL JOIN 对每个 topic 取 top 10
        rows = await self.conn.fetch(
            """
            SELECT t.id as topic_id, i.content
            FROM unnest($1::text[]) WITH ORDINALITY t(id, ord)
            CROSS JOIN LATERAL (
                SELECT content
                FROM memory_instances
                WHERE topic_id = t.id
                ORDER BY (1 - (content_embedding <=> $2)) * relevance_score * importance DESC
                LIMIT 10
            ) i
            ORDER BY t.ord
        """,
            top_topic_ids,
            query_emb.tolist(),
        )

        # 4. 在内存中组装结果
        results_map = {tid: [] for tid in top_topic_ids}
        for row in rows:
            results_map[row["topic_id"]].append(row["content"])

        # 保持排序顺序返回
        final_results = []
        for tid in top_topic_ids:
            if results_map[tid]:  # 只返回有内容的 topic
                final_results.append((topic_map[tid], results_map[tid]))

        return final_results

    async def _search_topics_with_dropout(
        self, query_emb: np.ndarray, config: MemorySearchConfig
    ) -> List[Dict[str, Any]]:
        """使用 dropout 的图遍历搜索 topics"""
        # 1. 动态入口点
        start_topics = await self.conn.fetch(
            """
            SELECT id, topic_name, strength, 1 - (topic_embedding <=> $1) as similarity
            FROM memory_topics
            ORDER BY topic_embedding <=> $1 LIMIT 3
        """,
            query_emb.tolist(),
        )

        queue = deque()
        for topic in start_topics:
            queue.append((topic["id"], 0, float(topic["similarity"])))

        visited = set()
        results = []

        # 缓存更新队列
        topics_to_strengthen = []
        edges_to_update = []

        while queue:
            topic_id, depth, path_strength = queue.popleft()

            if topic_id in visited or depth > config.max_depth:
                continue
            visited.add(topic_id)

            # 获取 Topic 详情 (这里仍需单次查询，因为是 BFS 动态的)
            topic = await self.conn.fetchrow(
                """
                SELECT id, topic_name, strength, 1 - (topic_embedding <=> $1) as similarity
                FROM memory_topics WHERE id = $2
            """,
                query_emb.tolist(),
                topic_id,
            )

            if not topic:
                continue

            # 综合打分
            score = topic["similarity"] * topic["strength"] * path_strength

            topics_to_strengthen.append(topic_id)

            if score > 0.25:  # 稍微降低阈值，保证召回
                results.append(
                    {
                        "id": topic["id"],
                        "topic_name": topic["topic_name"],
                        "score": score,
                        "strength": topic["strength"],
                    }
                )

            # 获取边
            edges = await self.conn.fetch(
                """
                SELECT to_topic, weight FROM topic_edges
                WHERE from_topic = $1 AND weight >= $2
                ORDER BY weight DESC LIMIT 8
            """,
                topic_id,
                config.min_edge_weight,
            )

            for edge in edges:
                if edge["weight"] < 0.5 and np.random.random() < config.dropout_rate:
                    continue

                edges_to_update.append((topic_id, edge["to_topic"]))
                queue.append((edge["to_topic"], depth + 1, path_strength * edge["weight"]))

        # 批量写回状态
        if topics_to_strengthen:
            await self.conn.execute(
                """
                UPDATE memory_topics 
                SET strength = LEAST(strength + $2, 1.0), last_accessed = NOW()
                WHERE id = ANY($1::text[])
            """,
                topics_to_strengthen,
                config.strengthen_boost * 0.5,
            )  # 搜索引起的增强稍微弱一点

        if edges_to_update:
            await self.conn.executemany(
                """
                UPDATE topic_edges 
                SET weight = LEAST(weight + 0.02, 1.0), last_activated = NOW()
                WHERE from_topic = $1 AND to_topic = $2
            """,
                edges_to_update,
            )

        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ========== 维护任务 ==========

    async def _update_importance_scores(self):
        """
        更新所有 instances 的重要性评分
        优化：O(N) 算法替代 O(N^2)
        原理：Instance 的独特性 ≈ 它离 Topic 中心的距离。离中心越远，越是 Outlier/独特。
        """
        print("[INFO] Updating importance scores...")

        # 直接使用 SQL 计算，无需把数据拉回 Python
        # 1. Recency: exp(-days / 30)
        # 2. Relevance: 字段本身
        # 3. Uniqueness: distance(instance_embedding, topic_embedding)
        #    memory_topics 表里已经存了 topic_embedding (即中心点)

        await self.conn.execute("""
            UPDATE memory_instances i
            SET importance = (
                0.3 * EXP(-EXTRACT(DAY FROM NOW() - i.last_accessed) / 30.0) +
                0.4 * i.relevance_score +
                0.3 * (
                    SELECT LEAST((i.content_embedding <=> t.topic_embedding) * 1.5, 1.0)
                    FROM memory_topics t 
                    WHERE t.id = i.topic_id
                )
            )
        """)

        print("[INFO] Importance scores updated via SQL.")

    async def cron(self) -> bool:
        """定时维护任务"""
        try:
            print("[INFO] Cron job begin...")

            # 1. 衰减边
            await self.conn.execute("UPDATE topic_edges SET weight = weight * 0.95 WHERE weight > 0.1")

            # 2. 清理极弱的边
            await self.conn.execute("DELETE FROM topic_edges WHERE weight < 0.05")

            # 3. 自动发现新 Topic
            await self._discover_topics()

            # 4. 合并相似 Topics (阈值提高防止误合并)
            await self._merge_similar_topics(threshold=0.92)

            # 5. 更新评分
            await self._update_importance_scores()

            # 6. 清理孤立 Topic
            await self.conn.execute("""
                DELETE FROM memory_topics
                WHERE id NOT IN (SELECT from_topic FROM topic_edges)
                  AND id NOT IN (SELECT to_topic FROM topic_edges)
                  AND strength < 0.1
                  AND access_count < 2
            """)

            print("[INFO] Cron job completed.")
            return True
        except Exception as e:
            print(f"Cron error: {e}")
            return False

    async def _discover_topics(self, min_instances: int = 5):
        """自动发现新 topics（从高相关度的 instances 聚类）"""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            print("[WARNING] scikit-learn lib not installed, skipping topic discovery")
            return

        # 获取所有 instances 的 embeddings
        instances = await self.conn.fetch("""
            SELECT id, content, content_embedding, topic_id
            FROM memory_instances
        """)

        if len(instances) < min_instances * 2:
            return

        # 按 topic 分组，找出实例数量少的 topics
        topic_counts = {}
        for inst in instances:
            topic_counts[inst["topic_id"]] = topic_counts.get(inst["topic_id"], 0) + 1

        # 对小 topics 的 instances 进行重新聚类
        small_topic_instances = [inst for inst in instances if topic_counts.get(inst["topic_id"], 0) < 3]

        if len(small_topic_instances) < min_instances:
            return

        embeddings = np.array([inst["content_embedding"] for inst in small_topic_instances])

        clustering = DBSCAN(eps=0.3, min_samples=min_instances, metric="cosine").fit(embeddings)

        # 为每个簇创建新 topic
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue

            cluster_instances = [
                small_topic_instances[i]
                for i in range(len(small_topic_instances))
                if clustering.labels_[i] == cluster_id
            ]

            # 生成 topic 名称（使用最中心的实例）
            center_content = cluster_instances[0]["content"]
            topic_name = f"[INFO] Automatically discovered: {center_content[:30]}"

            # 创建新 topic 并重新分配 instances
            topic_emb = embeddings[clustering.labels_ == cluster_id].mean(axis=0)

            new_topic_id = await self.conn.fetchval(
                """
                INSERT INTO memory_topics (topic_name, topic_embedding)
                VALUES ($1, $2)
                RETURNING id
            """,
                topic_name,
                topic_emb.tolist(),
            )

            # 更新 instances 的 topic_id
            instance_ids = [inst["id"] for inst in cluster_instances]
            await self.conn.execute(
                """
                UPDATE memory_instances
                SET topic_id = $1
                WHERE id = ANY($2::text[])
            """,
                new_topic_id,
                instance_ids,
            )

        print(f"[INFO] {len(set(clustering.labels_)) - 1} new topics discovered from clustering")

    async def _merge_similar_topics(self, threshold: float = 0.85):
        """合并过于相似的 topics"""
        pairs = await self.conn.fetch(
            """
            SELECT t1.id as id1, t1.topic_name as name1,
                   t2.id as id2, t2.topic_name as name2,
                   1 - (t1.topic_embedding <=> t2.topic_embedding) as similarity
            FROM memory_topics t1
            CROSS JOIN memory_topics t2
            WHERE t1.id < t2.id
              AND 1 - (t1.topic_embedding <=> t2.topic_embedding) > $1
            ORDER BY similarity DESC
            LIMIT 10
        """,
            threshold,
        )

        merged_count = 0
        for pair in pairs:
            # 合并：将 topic2 的 instances 移到 topic1
            await self.conn.execute(
                """
                UPDATE memory_instances
                SET topic_id = $1
                WHERE topic_id = $2
            """,
                pair["id1"],
                pair["id2"],
            )

            # 更新边
            await self.conn.execute(
                """
                UPDATE topic_edges
                SET to_topic = $1
                WHERE to_topic = $2
            """,
                pair["id1"],
                pair["id2"],
            )

            await self.conn.execute(
                """
                UPDATE topic_edges
                SET from_topic = $1
                WHERE from_topic = $2
            """,
                pair["id1"],
                pair["id2"],
            )

            # 删除 topic2
            await self.conn.execute(
                """
                DELETE FROM memory_topics WHERE id = $1
            """,
                pair["id2"],
            )

            merged_count += 1

        if merged_count > 0:
            print(f"[INFO] {merged_count} similar topics merged")

    # ========== 导出函数 ==========

    async def export_graph(self) -> Dict[str, Any]:
        """
        导出图结构（用于可视化）

        Returns:
            {
                'nodes': [...],
                'edges': [...],
                'stats': {...}
            }
        """
        # 获取所有 topics
        topics = await self.conn.fetch("""
            SELECT id, topic_name, strength, access_count,
                   (SELECT COUNT(*) FROM memory_instances 
                    WHERE topic_id = memory_topics.id) as instance_count
            FROM memory_topics
        """)

        # 获取所有边
        edges = await self.conn.fetch("""
            SELECT from_topic, to_topic, weight, activation_count
            FROM topic_edges
            WHERE weight > 0.2
        """)

        # 统计信息
        stats = await self.conn.fetchrow("""
            SELECT 
                COUNT(DISTINCT id) as total_topics,
                COUNT(DISTINCT id) as total_instances,
                AVG(strength) as avg_topic_strength
            FROM memory_topics
        """)

        total_instances = await self.conn.fetchval("""
            SELECT COUNT(*) FROM memory_instances
        """)

        return {
            "nodes": [
                {
                    "id": str(t["id"]),
                    "label": t["topic_name"],
                    "strength": float(t["strength"]),
                    "access_count": t["access_count"],
                    "instance_count": t["instance_count"],
                    "size": t["strength"] * (1 + np.log(1 + t["access_count"])),
                }
                for t in topics
            ],
            "edges": [
                {
                    "source": str(e["from_topic"]),
                    "target": str(e["to_topic"]),
                    "weight": float(e["weight"]),
                    "activation_count": e["activation_count"],
                }
                for e in edges
            ],
            "stats": {
                "total_topics": stats["total_topics"],
                "total_instances": total_instances,
                "total_edges": len(edges),
                "avg_topic_strength": float(stats["avg_topic_strength"] or 0),
            },
        }
