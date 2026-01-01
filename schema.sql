-- ============================================
-- Graph Memory System Schema
-- Topic-Instance 分离架构 + 遗忘机制
-- ============================================

-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- 主题层（Topic Layer）
-- ============================================
CREATE TABLE memory_topics
(
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    topic_name      TEXT         NOT NULL,
    topic_embedding vector(1536) NOT NULL,

    -- 记忆强度机制
    strength        FLOAT            DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
    access_count    INTEGER          DEFAULT 0,
    last_accessed   TIMESTAMP        DEFAULT NOW(),
    created_at      TIMESTAMP        DEFAULT NOW(),

    -- 遗忘曲线参数
    decay_rate      FLOAT            DEFAULT 0.1,
    half_life_days  FLOAT            DEFAULT 7.0,

    -- 元数据
    metadata        JSONB            DEFAULT '{}'::jsonb
);

-- HNSW 索引（高性能向量搜索）
CREATE INDEX memory_topics_embedding_idx ON memory_topics
    USING hnsw (topic_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 其他索引
CREATE INDEX memory_topics_strength_idx ON memory_topics (strength DESC);
CREATE INDEX memory_topics_accessed_idx ON memory_topics (last_accessed DESC);

-- ============================================
-- 实例层（Instance Layer）
-- ============================================
CREATE TABLE memory_instances
(
    id                TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    topic_id          TEXT         NOT NULL REFERENCES memory_topics (id) ON DELETE CASCADE,

    content           TEXT         NOT NULL,
    content_embedding vector(1536) NOT NULL,

    -- 实例特有属性
    relevance_score   FLOAT            DEFAULT 1.0 CHECK (relevance_score >= 0 AND relevance_score <= 1),
    importance        FLOAT            DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),

    -- 时间戳
    created_at        TIMESTAMP        DEFAULT NOW(),
    last_accessed     TIMESTAMP        DEFAULT NOW(),

    -- 元数据
    metadata          JSONB            DEFAULT '{}'::jsonb
);

-- HNSW 索引
CREATE INDEX memory_instances_embedding_idx ON memory_instances
    USING hnsw (content_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 其他索引
CREATE INDEX memory_instances_topic_idx ON memory_instances (topic_id);
CREATE INDEX memory_instances_relevance_idx ON memory_instances (topic_id, relevance_score DESC);
CREATE INDEX memory_instances_importance_idx ON memory_instances (importance DESC);

-- ============================================
-- 关系层（Topic Graph）
-- ============================================
CREATE TABLE topic_edges
(
    id               TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    from_topic       TEXT NOT NULL REFERENCES memory_topics (id) ON DELETE CASCADE,
    to_topic         TEXT NOT NULL REFERENCES memory_topics (id) ON DELETE CASCADE,

    -- 边权重（连接强度）
    weight           FLOAT            DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),
    edge_type        TEXT             DEFAULT 'semantic',

    -- 激活统计
    activation_count INTEGER          DEFAULT 0,
    last_activated   TIMESTAMP        DEFAULT NOW(),

    -- 衰减参数
    decay_rate       FLOAT            DEFAULT 0.05,

    -- 唯一约束
    UNIQUE (from_topic, to_topic)
);

CREATE INDEX topic_edges_from_idx ON topic_edges (from_topic, weight DESC);
CREATE INDEX topic_edges_to_idx ON topic_edges (to_topic, weight DESC);
CREATE INDEX topic_edges_weight_idx ON topic_edges (weight DESC);

-- ============================================
-- 遗忘机制函数
-- ============================================

-- 计算记忆强度衰减（基于遗忘曲线）
CREATE OR REPLACE FUNCTION decay_memory_strength(node_id TEXT)
    RETURNS FLOAT AS
$$
DECLARE
    current_strength   FLOAT;
    days_elapsed       FLOAT;
    current_decay_rate FLOAT;
    current_half_life  FLOAT;
    new_strength       FLOAT;
BEGIN
    SELECT strength,
           EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0,
           decay_rate,
           half_life_days
    INTO current_strength, days_elapsed, current_decay_rate, current_half_life
    FROM memory_topics
    WHERE id = node_id;

    -- 指数衰减公式: S(t) = S₀ × (0.5)^(t/t_half)
    new_strength := current_strength * POWER(0.5, days_elapsed / current_half_life);

    UPDATE memory_topics
    SET strength = GREATEST(new_strength, 0.01) -- 最低保留 0.01
    WHERE id = node_id;

    RETURN new_strength;
END;
$$ LANGUAGE plpgsql;

-- 批量衰减所有边权重
CREATE OR REPLACE FUNCTION decay_edge_weights()
    RETURNS INTEGER AS
$$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE topic_edges
    SET weight = GREATEST(
            weight * POWER(0.5,
                           EXTRACT(EPOCH FROM (NOW() - last_activated)) / (86400.0 * 7.0)
                     ),
            0.01
                 )
    WHERE last_activated < NOW() - INTERVAL '1 day';

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- 增强记忆（访问时调用）
CREATE OR REPLACE FUNCTION strengthen_memory(node_id TEXT, boost FLOAT DEFAULT 0.1)
    RETURNS void AS
$$
BEGIN
    UPDATE memory_topics
    SET strength       = LEAST(strength + boost, 1.0),
        access_count   = access_count + 1,
        last_accessed  = NOW(),
        half_life_days = half_life_days * 1.1 -- 延长半衰期 10%
    WHERE id = node_id;
END;
$$ LANGUAGE plpgsql;

-- Evict 弱记忆
CREATE OR REPLACE FUNCTION evict_weak_memories(threshold FLOAT DEFAULT 0.05)
    RETURNS INTEGER AS
$$
DECLARE
    evicted_count INTEGER;
BEGIN
    -- 先全局衰减
    PERFORM decay_memory_strength(id) FROM memory_topics;

    -- 删除过弱且长期未访问的记忆
    WITH deleted AS (
        DELETE FROM memory_topics
            WHERE strength < threshold
                AND last_accessed < NOW() - INTERVAL '30 days'
                AND access_count < 5
            RETURNING id)
    SELECT COUNT(*)
    INTO evicted_count
    FROM deleted;

    RETURN evicted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 统计视图
-- ============================================

CREATE OR REPLACE VIEW memory_stats AS
SELECT (SELECT COUNT(*) FROM memory_topics)                      as total_topics,
       (SELECT COUNT(*) FROM memory_instances)                   as total_instances,
       (SELECT COUNT(*) FROM topic_edges)                        as total_edges,
       (SELECT AVG(strength) FROM memory_topics)                 as avg_topic_strength,
       (SELECT AVG(weight) FROM topic_edges)                     as avg_edge_weight,
       (SELECT COUNT(*) FROM memory_topics WHERE strength > 0.5) as strong_topics,
       (SELECT COUNT(*) FROM memory_topics WHERE strength < 0.2) as weak_topics;

-- ============================================
-- 实用查询函数
-- ============================================

-- 查找孤立 topics
CREATE OR REPLACE FUNCTION find_orphan_topics()
    RETURNS TABLE
            (
                id         TEXT,
                topic_name TEXT,
                strength   FLOAT
            )
AS
$$
BEGIN
    RETURN QUERY
        SELECT t.id, t.topic_name, t.strength
        FROM memory_topics t
        WHERE t.id NOT IN (SELECT DISTINCT from_topic FROM topic_edges)
          AND t.id NOT IN (SELECT DISTINCT to_topic FROM topic_edges)
          AND t.strength < 0.3
        ORDER BY t.strength ASC;
END;
$$ LANGUAGE plpgsql;

-- 获取 topic 的邻居
CREATE OR REPLACE FUNCTION get_topic_neighbors(topic_id TEXT, min_weight FLOAT DEFAULT 0.3)
    RETURNS TABLE
            (
                neighbor_id   TEXT,
                neighbor_name TEXT,
                edge_weight   FLOAT
            )
AS
$$
BEGIN
    RETURN QUERY
        SELECT t.id, t.topic_name, e.weight
        FROM topic_edges e
                 JOIN memory_topics t ON e.to_topic = t.id
        WHERE e.from_topic = topic_id
          AND e.weight >= min_weight
        ORDER BY e.weight DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 触发器：自动更新时间戳
-- ============================================

CREATE OR REPLACE FUNCTION update_accessed_timestamp()
    RETURNS TRIGGER AS
$$
BEGIN
    NEW.last_accessed = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 在 instances 被访问时更新 topic 的时间戳
CREATE TRIGGER update_topic_on_instance_access
    AFTER UPDATE
    ON memory_instances
    FOR EACH ROW
    WHEN (NEW.last_accessed > OLD.last_accessed)
EXECUTE FUNCTION update_accessed_timestamp();

-- ============================================
-- 初始化脚本（可选）
-- ============================================

-- 插入示例数据
-- INSERT INTO memory_topics (topic_name, topic_embedding)
-- VALUES ('示例主题', array_fill(0, ARRAY[1536])::vector);

-- 查看统计信息
-- SELECT * FROM memory_stats;