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

### CLI 工具使用

MaiVecMem 提供了命令行工具 `cli_tool.py` 来管理数据库：

```bash
python cli_tool.py [命令] [参数]
```

可用命令：
- `init`: 初始化数据库结构
- `import [文件路径]`: 导入知识库 JSON 文件
- `search [关键词]`: 搜索相关记忆
- `export [文件路径]`: 导出数据库内容到 JSON 文件

### 知识库 JSON 格式

MaiVecMem 使用标准的 JSON 格式存储知识库：

```json
{
  "topic1": ["ctx1", "ctx2"],
  "topic2": ["ctx1", "ctx2", "ctx3"]
}
```

其中：
- 键（如 `topic1`, `topic2`）表示主题名称
- 值是一个字符串数组，包含该主题相关的上下文内容

### 示例

导入知识库：
```bash
python cli_tool.py import knowledge_base.json
```

搜索相关记忆：
```bash
python cli_tool.py search "关键词"
```

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请在 GitHub 上创建 Issue。