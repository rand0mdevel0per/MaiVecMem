# Contributing to MaiVecMem

感谢你对 MaiVecMem 的贡献！本手册面向开发者，介绍如何在本仓库中进行开发、生成 migration、运行静态检查与格式化、以及提交 PR 的规范流程。

> 目录
> - 开发前准备
> - 代码风格和静态检查（ruff）
> - 生成 DB migration 与 migration-scripts（使用 iflow 自动化流程）
> - 本地测试
> - 提交与发布分支策略


## 开发前准备

1. 克隆仓库并创建虚拟环境：

```powershell
git clone <repo-url>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. 推荐安装开发工具（可选）：
- ruff（用于 lint/format）
- black（可与 ruff 一起使用）

你可以通过 pip 安装：

```powershell
pip install ruff black
```


## 代码风格与静态检查（ruff）

本仓库推荐使用 `ruff` 做快速静态检查和自动修复（包括格式化）。

常用命令：

```powershell
# 检查错误
ruff check .

# 自动修复大部分问题（包括格式化）
ruff check --fix .

# 或者使用 ruff format（如果你想仅仅格式化）
ruff format .
```

建议在每次 commit 前运行一次 `ruff check --fix .`。


## 生成 DB migration 与 migration-scripts（iflow 辅助）

本项目采用 SQL migration 脚本管理数据库 schema 的变更。以下是推荐的自动化流程：

1. 在本地修改 `plugins/MaiVecMem/schema.sql`（或在插件代码中变更向量维度）并验证无误。
2. 使用 `git` 在 feature 分支上提交修改：

```powershell
git checkout -b feat/your-feature
git add plugins/MaiVecMem/schema.sql
git commit -m "chore(migration): update schema for ..."
```

3. 生成 migration-scripts（自动化建议：使用 `iflow` 执行本地脚本来生成 migration 文件并推送）：

iflow 被设计为一个可以运行命令并将结果作为 commit/PR 的工具。我们建议使用 `shutil` 或 shell 调用 iflow（如果系统支持）。示例命令（本地运行）：

```powershell
# iflow 执行示例（yolo 模式将赋予模型文件与终端访问权限）
iflow --yolo -p "Generate SQL migration script for schema changes in plugins/MaiVecMem/schema.sql and produce file migration-scripts/<commit-id>.sql"
```

4. 本地生成 migration-scripts 后，手动或自动将其加入到 `migration-scripts` 分支或目录：

- 推荐存放位置：`migration-scripts/<commit-id>.sql`，`commit-id` 为触发变更的主分支提交哈希。
- 在 PR 中引用该 migration 文件以便 CI 或运维能自动应用。

5. 自动决定推送到 `rel` 分支：

- 你的 CI/自动化脚本可以基于分支策略决定是否把 migration 推送到 `rel` 分支（例如当 PR 被 merge 后，release 流程可在 `rel` 分支自动触发并 bump 版本号）。
- 为避免频繁发布，建议在 `rel` 上合并经过验证的 migration，并在 `rel` 上运行一次完整的迁移验证流程。


## 使用 iflow 自动化生成并提交 migration（建议脚本示例）

下面是一个建议的 Python 脚本片段（放在插件的工具脚本中，如 `iflow_action_generate_migration.py`），它会用 `shutil` 调用 `iflow`：

```python
import shutil
import subprocess

prompt = "Generate SQL migration file for schema changes in plugins/MaiVecMem/schema.sql and write result to migration-scripts/<commit-id>.sql"
cmd = ["iflow", "--yolo", "-p", prompt]
subprocess.run(cmd, check=True)
```

注意：在 `--yolo` 下模型拥有文件与终端访问能力，这会带来安全风险，请仅在受信任的环境下使用。


## 本地测试

- 单元测试（如果已有测试）：

```powershell
pytest -q
```

- 对于与 PostgreSQL 交互的测试，建议使用测试数据库实例或在 CI 中使用容器化数据库（例如 GitHub Actions 的 services 或 Docker Compose）。


## 提交与 PR 流程

1. 在本地运行 `ruff check --fix .` 并执行测试。
2. 按分支策略提交：`feat/*`、`fix/*`、`chore/*`。
3. 推送并在 PR 描述中包含：
   - 变更摘要
   - 是否包含 DB schema 变更，以及 migration-scripts 的路径（如有）
   - 测试覆盖/手动验证步骤

合并策略：
- PR 需至少包含一个代码审查者的通过。
- 包含 DB 变更的 PR 应当在合并前完成 migration 文件并通过迁移验证。


## 版本与发布（简述）

- 推荐分支策略：`dev`（开发） -> `main`（主） -> `rel`（发布/稳定）
- 在合并到 `rel` 时，自动化脚本可更新插件版本号、生成发布说明并触发 CI 发布流程。


## 联系与帮助

在贡献过程中遇到问题，请在仓库中提交 Issue，或直接在 PR 中标注需要帮助的地方。感谢你的贡献！

