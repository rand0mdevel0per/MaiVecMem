# PostgreSQL 配置教程

本教程将指导您配置 PostgreSQL 以支持 MaiVecMem 插件。

## 前提条件

- 已完成 PostgreSQL 安装（参考 [安装教程](postgresql_installation.md)）
- 已安装 pgvector 扩展（用于向量存储）

## 基本配置

### 1. 连接到 PostgreSQL

```bash
# 使用 postgres 用户连接
psql -U postgres
```

### 2. 创建专用数据库

```sql
-- 创建数据库用户
CREATE USER mai_user WITH PASSWORD 'your_secure_password';

-- 创建数据库
CREATE DATABASE mai_memory OWNER mai_user;

-- 授予权限
GRANT ALL PRIVILEGES ON DATABASE mai_memory TO mai_user;
```

### 3. 安装 pgvector 扩展

pgvector 是 PostgreSQL 的向量相似度搜索扩展，MaiVecMem 需要它来存储和检索向量。

#### Windows 安装

1. 下载预编译的 pgvector 扩展：
   - 访问：https://github.com/pgvector/pgvector/releases
   - 下载适合您 PostgreSQL 版本的 `.zip` 文件

2. 解压文件到 PostgreSQL 的 lib 目录：
   ```
   C:\Program Files\PostgreSQL\<版本号>\lib
   ```

3. 将 `.dll` 文件复制到 lib 目录

4. 在 PostgreSQL 中启用扩展：
   ```sql
   CREATE EXTENSION vector;
   ```

#### macOS 安装（使用 Homebrew）

```bash
# 安装 pgvector
brew install pgvector

# 在 PostgreSQL 中启用扩展
psql -U postgres -d mai_memory -c "CREATE EXTENSION vector;"
```

#### Linux 安装

```bash
# Ubuntu/Debian
sudo apt install postgresql-<版本号>-pgvector

# 或者从源码编译
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# 在 PostgreSQL 中启用扩展
sudo -u postgres psql -d mai_memory -c "CREATE EXTENSION vector;"
```

### 4. 配置 MaiVecMem 数据库

```sql
-- 连接到 mai_memory 数据库
\c mai_memory

-- 确保扩展已启用
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建向量存储表（参考 schema.sql）
CREATE TABLE IF NOT EXISTS memory_vectors (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768), -- 根据实际向量维度调整
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引以优化搜索性能
CREATE INDEX idx_memory_vectors_topic ON memory_vectors(topic);
CREATE INDEX idx_memory_vectors_embedding ON memory_vectors USING ivfflat (embedding vector_l2_ops);

-- 授予用户权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mai_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mai_user;
```

## MaiVecMem 配置

### 1. 配置数据库连接

在第一次运行MaiCore生成配置文件后，编辑 `config.toml` 文件，更新数据库连接信息

### 2. 初始化数据库结构

使用 CLI 工具初始化数据库：

```bash
python cli_tool.py
```
选择`Initialize Graph Memory Tables`选项

### 3. 导入知识库数据

```bash
python cli_tool.py
```
选择`Load Dataset from Local .json File`选项，提供知识库 JSON 文件路径

## 性能优化配置

### 1. 调整 PostgreSQL 配置

编辑 `postgresql.conf` 文件：

```conf
# 增加共享缓冲区（根据系统内存调整）
shared_buffers = 256MB

# 增加工作内存
work_mem = 4MB

# 增加维护工作内存
maintenance_work_mem = 64MB

# 启用并行查询
max_parallel_workers_per_gather = 2

# 调整 wal 设置
wal_buffers = 16MB
checkpoint_completion_target = 0.9
```

### 2. 向量索引优化

```sql
-- 重建索引以优化性能
REINDEX INDEX idx_memory_vectors_embedding;

-- 分析表以更新统计信息
ANALYZE memory_vectors;
```

### 3. 连接池配置

考虑使用连接池工具如 `pgBouncer` 来管理数据库连接：

```bash
# Ubuntu/Debian
sudo apt install pgbouncer

# 配置 /etc/pgbouncer/pgbouncer.ini
[databases]
mai_memory = host=localhost port=5432 dbname=mai_memory

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
```

## 安全配置

### 1. 修改默认端口（可选）

编辑 `postgresql.conf`：
```conf
port = 5433  # 或其他非默认端口
```

### 2. 配置 pg_hba.conf

编辑 `pg_hba.conf` 文件以限制访问：

```conf
# 只允许本地连接
local   all             all                                     peer
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5

# 拒绝其他所有连接
host    all             all             0.0.0.0/0               reject
```

### 3. 启用 SSL（生产环境推荐）

```conf
# 在 postgresql.conf 中
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

## 备份和恢复

### 备份数据库

```bash
# 备份整个数据库
pg_dump -U mai_user -h localhost mai_memory > mai_memory_backup.sql

# 备份特定表
pg_dump -U mai_user -h localhost -t memory_vectors mai_memory > vectors_backup.sql
```

### 恢复数据库

```bash
# 恢复整个数据库
psql -U mai_user -h localhost mai_memory < mai_memory_backup.sql

# 恢复特定表
psql -U mai_user -h localhost mai_memory < vectors_backup.sql
```

## 监控和维护

### 1. 查看数据库状态

```sql
-- 查看活跃连接
SELECT * FROM pg_stat_activity;

-- 查看表大小
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 查看索引使用情况
SELECT 
    indexrelname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE relname = 'memory_vectors';
```

### 2. 定期维护

```bash
# 定期执行 vacuum 和 analyze
psql -U mai_user -h localhost mai_memory -c "VACUUM ANALYZE;"

# 或设置自动维护
-- 创建维护脚本
CREATE OR REPLACE FUNCTION maintain_memory_vectors()
RETURNS void AS $$
BEGIN
    VACUUM ANALYZE memory_vectors;
    REINDEX TABLE memory_vectors;
END;
$$ LANGUAGE plpgsql;

-- 设置定时任务（需要 pg_cron 扩展）
SELECT cron.schedule('weekly-maintenance', '0 2 * * 0', 'SELECT maintain_memory_vectors();');
```

## 故障排除

### 问题1：无法连接到数据库

```bash
# 检查服务状态
sudo systemctl status postgresql

# 检查端口监听
netstat -tlpn | grep 5432

# 检查防火墙规则
sudo ufw status
```

### 问题2：pgvector 扩展加载失败

```sql
-- 检查扩展是否已安装
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- 查看错误日志
SHOW log_destination;
SHOW log_directory;
```

### 问题3：向量搜索性能差

```sql
-- 检查索引
\d memory_vectors

-- 重建索引
REINDEX INDEX idx_memory_vectors_embedding;

-- 分析查询计划
EXPLAIN ANALYZE SELECT * FROM memory_vectors 
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector 
LIMIT 10;
```

## 下一步

配置完成后，您可以：
1. 阅读 [README.md](../README.md) 了解插件使用方法
2. 使用 CLI 工具导入和管理知识库数据
3. 集成到 MaiBot 主程序中

## 参考资源

- PostgreSQL 官方文档：https://www.postgresql.org/docs/
- pgvector 文档：https://github.com/pgvector/pgvector
- MaiVecMem 项目：https://github.com/Mai-with-u/MaiVecMem