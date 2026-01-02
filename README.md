# MaiVecMem

MaiVecMem 是 [MaiBot](https://github.com/Mai-with-u/MaiBot) 的一个附属插件，用于向量化的记忆存储和检索。

## 功能特点

- 基于 PostgreSQL 的向量化存储
- 高效的记忆检索功能
- 支持 CLI 工具进行数据库管理

## 目录结构

```
MaiVecMem/
├── cli_tool.py     # 命令行工具
├── db_mod.py       # 数据库操作模块
├── hf_converter.py # 向量化转换器
├── plugin.py       # 插件主程序
├── schema.sql      # 数据库结构定义
├── README.md       # 说明文档
├── LICENSE         # MIT 许可证
└── manuals/        # 使用手册目录
    ├── postgresql_installation.md  # PostgreSQL 安装教程
    └── postgresql_configuration.md  # PostgreSQL 配置教程
```

## 安装说明

1. 确保已安装 PostgreSQL 数据库
2. 创建一个新的数据库用于存储向量化记忆
3. 执行 `schema.sql` 文件初始化数据库结构
4. 在 `plugin.py` 中配置数据库连接信息

详细的 PostgreSQL 安装和配置教程请参考 `manuals` 目录下的文档。

## 使用说明

### CLI 工具使用（推荐）

MaiVecMem 提供命令行工具 `cli_tool.py` 来管理数据库和导入工作。插件设计为独立插件（independent），所以建议在插件所在目录或在包含插件包的 Python 环境中运行命令。

重要：
- CLI 需要两个地方的配置：
  1. 仓库根（或运行时可访问的）`config.toml` 中的 `postgresql` 段（仅用于数据库连接凭据）。
  2. `plugins/MaiVecMem/model_info.json`（必需）——该文件包含插件初始化时导出的 embedding 配置与 model 信息（例如 embedding 服务地址、API Key、模型名、向量维度等）。
- 如果 `model_info.json` 不存在，CLI 会立刻报错并退出（这是为了避免向量维度或凭据不匹配导致的数据损坏）。

运行方式（示例）：

- 从仓库根以模块方式运行（推荐，能保证 Python path 正确）：

```powershell
python -m plugins.MaiVecMem.cli_tool [命令] [参数]
```

- 或直接运行插件目录下的脚本（在插件目录内）：

```powershell
python plugins\MaiVecMem\cli_tool.py [命令] [参数]
```

可用命令（主要）：
- `init`：初始化数据库结构（会尝试使用 `plugins/MaiVecMem/model_info.json` 中的向量维度来替换 `schema.sql` 中的 vector(N) 定义）。
- `import <file_path>`：从标准 topic-instance JSON 导入数据（见下文 JSON 格式）。
- `import-openie <file_path> --strategy <strategy>`：导入 OpenIE 格式的 JSON 文件并按指定聚合策略转换后导入。策略：`subject`、`relation`、`hybrid`、`semantic`、`entity`。
- `import-openie-dir <dir_path> --pattern <pattern> --strategy <strategy>`：遍历目录并按后缀/模式匹配逐文件导入（默认 `--pattern "-openie.json"`）。
- `search <query>`：在记忆库中检索相关 topic/instances。
- `export <file_path>`：导出数据库内容为 JSON。
- `interactive`：进入交互式菜单进行操作（带提示）。

示例：
- 单文件 OpenIE（按主语聚合）：

```powershell
python -m plugins.MaiVecMem.cli_tool import-openie data\example-openie.json --strategy subject
```

- 目录批量导入（匹配 `*-openie.json`）：

```powershell
python -m plugins.MaiVecMem.cli_tool import-openie-dir data\openie --pattern -openie.json --strategy semantic
```

- 初始化数据库（将使用 `model_info.json` 中的向量维度替换 schema 中的 vector 定义）：

```powershell
python -m plugins.MaiVecMem.cli_tool init
```

### 知识库 JSON 格式（标准 topic-instance）

MaiVecMem 使用以下 JSON 格式来表示要导入的主题和上下文：

```json
{
  "topic1": ["ctx1", "ctx2"],
  "topic2": ["ctx1", "ctx2", "ctx3"]
}
```

字段说明：
- 键（如 `topic1`）表示主题名。
- 值是字符串数组，包含该主题下的实例/上下文。

### OpenIE 导入说明

- `import-openie` / `import-openie-dir` 复用 `plugins/MaiVecMem/libopenie.py` 中的转换逻辑。
- 非 `semantic` 策略（如 `subject`/`relation`/`hybrid`）会直接调用转换器生成 topic->instances map 并逐批写入 DB。
- `semantic` 策略会使用 embedding 来做聚类；embedding 凭据必须来自 `model_info.json` 中的 `plugin_config_snapshot.openai_embedding`（或可由 CLI 所在的环境覆盖）。

### 故障排查要点

- 报错：找不到 `model_info.json` —— 请先在插件初始化阶段运行生成 model 信息，或将有效的 `model_info.json` 放在 `plugins/MaiVecMem/` 下。
- 报错：数据库连接失败 —— 检查 `config.toml` 中 `postgresql` 段的 host/port/user/password/database 是否正确，并确保网络与 PostgreSQL 服务可达。
- 向量维度冲突（例如在 `schema.sql` 中 vector 大小不匹配）——请确保 `model_info.json` 中的 `dimension` 与数据库 schema 中的 vector 大小一致，或在 `init` 时让 CLI 用 `model_info.json` 自动替换 schema 中的 vector(N)。

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 贡献（简短）

欢迎提交 Issue 与 Pull Request；插件代码位于 `plugins/MaiVecMem/`。更详细的贡献指南请查看 `manuals/CONTRIBUTING.md`。

## 联系方式

如有问题，请在 GitHub 上创建 Issue。