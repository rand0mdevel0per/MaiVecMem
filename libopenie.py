"""
OpenIE 知识库转换器
将 OpenIE 格式的三元组转换为 topic-instance 格式

支持的 OpenIE 格式：
1. Stanford OpenIE: {sentences: [{triples: [...]}]}
2. AllenNLP OpenIE: [{arg1, rel, arg2, confidence}, ...]
3. ClausIE: [{subject, predicate, object}, ...]
"""

import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class OpenIEConverter:
    """OpenIE 格式转换器"""

    @staticmethod
    def load_openie_json(filepath: str) -> Any:
        """加载 OpenIE JSON 文件"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def detect_format(data: Any) -> str:
        """自动检测 OpenIE 格式"""
        if isinstance(data, dict):
            if "sentences" in data:
                return "stanford"
            elif "triples" in data or "extractions" in data:
                return "generic"

        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if "arg1" in first_item and "rel" in first_item:
                return "allennlp"
            elif "subject" in first_item and "predicate" in first_item:
                return "clausie"

        return "unknown"

    @staticmethod
    def normalize_triples(data: Any) -> List[Tuple[str, str, str]]:
        """
        标准化三元组为统一格式: [(subject, relation, object), ...]
        """
        triples = []
        format_type = OpenIEConverter.detect_format(data)

        if format_type == "stanford":
            # Stanford OpenIE format
            for sentence in data.get("sentences", []):
                for triple in sentence.get("triples", []):
                    triples.append((triple.get("subject", ""), triple.get("relation", ""), triple.get("object", "")))

        elif format_type == "allennlp":
            # AllenNLP OpenIE format
            for item in data:
                triples.append((item.get("arg1", ""), item.get("rel", ""), item.get("arg2", "")))

        elif format_type == "clausie":
            # ClausIE format
            for item in data:
                triples.append((item.get("subject", ""), item.get("predicate", ""), item.get("object", "")))

        elif format_type == "generic":
            # Generic format
            extractions = data.get("triples", data.get("extractions", []))
            for item in extractions:
                triples.append(
                    (
                        item.get("subject", item.get("arg1", "")),
                        item.get("relation", item.get("predicate", item.get("rel", ""))),
                        item.get("object", item.get("arg2", "")),
                    )
                )

        # 过滤空值
        return [(s.strip(), r.strip(), o.strip()) for s, r, o in triples if s and r and o]

    # ========== 策略 1: 按主语聚合 ==========

    @staticmethod
    def by_subject(triples: List[Tuple[str, str, str]]) -> Dict[str, List[str]]:
        """
        策略1: 按主语(subject)聚合成 topic

        Example:
            Input: [("Apple", "founded by", "Steve Jobs"),
                    ("Apple", "headquartered in", "Cupertino")]
            Output: {
                "Apple": [
                    "founded by Steve Jobs",
                    "headquartered in Cupertino"
                ]
            }
        """
        result = defaultdict(list)

        for subject, relation, obj in triples:
            # 将 relation + object 组合成 instance
            instance = f"{relation} {obj}"
            result[subject].append(instance)

        return dict(result)

    # ========== 策略 2: 按关系聚合 ==========

    @staticmethod
    def by_relation(triples: List[Tuple[str, str, str]]) -> Dict[str, List[str]]:
        """
        策略2: 按关系(relation)聚合成 topic

        Example:
            Input: [("Apple", "founded by", "Steve Jobs"),
                    ("Microsoft", "founded by", "Bill Gates")]
            Output: {
                "founded by": [
                    "Apple founded by Steve Jobs",
                    "Microsoft founded by Bill Gates"
                ]
            }
        """
        result = defaultdict(list)

        for subject, relation, obj in triples:
            # 将完整三元组作为 instance
            instance = f"{subject} {relation} {obj}"
            result[relation].append(instance)

        return dict(result)

    # ========== 策略 3: 混合策略 ==========

    @staticmethod
    def hybrid(triples: List[Tuple[str, str, str]], min_instances: int = 2) -> Dict[str, List[str]]:
        """
        策略3: 混合策略 - 智能选择聚合方式

        - 如果主语出现频率高，按主语聚合
        - 否则按关系聚合
        - 过滤掉实例数量少于 min_instances 的 topic

        Args:
            triples: 三元组列表
            min_instances: topic 最少包含的实例数
        """
        # 统计主语和关系的频率
        subject_freq = defaultdict(int)
        relation_freq = defaultdict(int)

        for subject, relation, _obj in triples:
            subject_freq[subject] += 1
            relation_freq[relation] += 1

        result = defaultdict(list)

        for subject, relation, obj in triples:
            # 如果主语频率高，作为 topic
            if subject_freq[subject] >= min_instances:
                instance = f"{relation} {obj}"
                result[subject].append(instance)
            # 否则如果关系频率高，按关系聚合
            elif relation_freq[relation] >= min_instances:
                instance = f"{subject} {relation} {obj}"
                result[relation].append(instance)

        # 过滤
        return {topic: instances for topic, instances in result.items() if len(instances) >= min_instances}

    # ========== 策略 4: 语义聚类 (需要 embeddings) ==========

    @staticmethod
    def by_semantic_clustering(
        triples: List[Tuple[str, str, str]], embedder_func=None, n_clusters: int = 10
    ) -> Dict[str, List[str]]:
        """
        策略4: 语义聚类 - 使用 embeddings 聚类

        需要传入 embedder_func(text) -> vector

        Args:
            triples: 三元组列表
            embedder_func: 生成 embedding 的函数
            n_clusters: 聚类数量
        """
        if embedder_func is None:
            raise ValueError("Embedder Function Required for Semantic Clustering Strategy")

        try:
            from sklearn.cluster import KMeans
            import numpy as np
        except ImportError:
            raise ImportError("scikit-learn and numpy are required for semantic clustering strategy") from None

        # 生成三元组的文本表示
        triple_texts = [f"{s} {r} {o}" for s, r, o in triples]

        # 生成 embeddings
        embeddings = np.array([embedder_func(text) for text in triple_texts])

        # KMeans 聚类
        kmeans = KMeans(n_clusters=min(n_clusters, len(triples)), random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # 聚合结果
        result = defaultdict(list)

        for idx, (subject, relation, obj) in enumerate(triples):
            cluster_id = labels[idx]

            # 使用聚类中心最近的三元组作为 topic 名称
            if f"Cluster_{cluster_id}" not in result or len(result[f"Cluster_{cluster_id}"]) == 0:
                # 第一个元素，用主语作为 topic
                topic_name = subject
            else:
                topic_name = f"Cluster_{cluster_id}"

            instance = f"{subject} {relation} {obj}"
            result[topic_name].append(instance)

        return dict(result)

    # ========== 策略 5: 实体中心 ==========

    @staticmethod
    def entity_centric(triples: List[Tuple[str, str, str]]) -> Dict[str, List[str]]:
        """
        策略5: 实体中心 - 同时考虑主语和宾语

        将同一实体作为主语或宾语的所有关系聚合在一起

        Example:
            Input: [("Apple", "founded by", "Steve Jobs"),
                    ("Steve Jobs", "co-founded", "Apple")]
            Output: {
                "Apple": [
                    "founded by Steve Jobs",
                    "Steve Jobs co-founded Apple"
                ],
                "Steve Jobs": [
                    "co-founded Apple",
                    "Apple founded by Steve Jobs"
                ]
            }
        """
        result = defaultdict(list)

        for subject, relation, obj in triples:
            # 主语作为 topic
            result[subject].append(f"{relation} {obj}")

            # 宾语也作为 topic（反向关系）
            result[obj].append(f"{subject} {relation}")

        return dict(result)

    # ========== 便捷函数 ==========

    @staticmethod
    def convert(openie_file: str, strategy: str = "subject", **kwargs) -> Dict[str, List[str]]:
        """
        一站式转换函数

        Args:
            openie_file: OpenIE JSON 文件路径
            strategy: 聚合策略
                - 'subject': 按主语聚合
                - 'relation': 按关系聚合
                - 'hybrid': 混合策略
                - 'semantic': 语义聚类（需要 embedder_func）
                - 'entity': 实体中心
            **kwargs: 策略特定参数

        Returns:
            {"topic": ["instance1", "instance2", ...], ...}
        """
        # 加载数据
        data = OpenIEConverter.load_openie_json(openie_file)

        # 标准化三元组
        triples = OpenIEConverter.normalize_triples(data)

        if not triples:
            print("[ERROR] No valid triples found in the OpenIE data.")
            return {}

        print(f"[INFO] {len(triples)} trples loaded from {openie_file}.")

        # 应用策略
        strategies = {
            "subject": OpenIEConverter.by_subject,
            "relation": OpenIEConverter.by_relation,
            "hybrid": OpenIEConverter.hybrid,
            "semantic": OpenIEConverter.by_semantic_clustering,
            "entity": OpenIEConverter.entity_centric,
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy} Available strategies: {list(strategies.keys())}")

        result = strategies[strategy](triples, **kwargs)

        print(f"[INFO] {len(result)} Topics generated using '{strategy}' strategy.")
        return result
