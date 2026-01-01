import db_mod

import sys
import os
import asyncio
import asyncpg
import timeit
import ujson

current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.abspath(os.path.join(current_dir, "../.."))
if target_path not in sys.path:
    sys.path.insert(0, target_path)

from typing import List, Tuple, Type
from src.plugin_system import BasePlugin, register_plugin, ComponentInfo, ConfigField, BaseTool

dbman = None
cron_task = None
config = None
initialized = False


@register_plugin  # 注册插件
class PgVecMemPlugin(BasePlugin):
    plugin_name = "pgvec_mem_plugin"
    enable_plugin = True
    dependencies = []
    python_dependencies = ["uuid", "dataclasses", "typing", "asyncpg", "numpy", "openai", "timeit", "ujson"]
    config_file_name = "config.toml"  # 配置文件名
    config_schema = \
        {
            "postgresql":
                {
                    "host": ConfigField(type=str,
                                        default="localhost",
                                        description="PostgreSQL服务器地址",
                                        hint="你的PostgreSQL服务器地址，通常是localhost或IP地址"),
                    "port": ConfigField(type=int,
                                        default=5432,
                                        description="PostgreSQL服务器端口",
                                        min=1, max=65535,
                                        hint="你的PostgreSQL服务器端口，默认是5432"),
                    "user": ConfigField(type=str,
                                        default="postgres",
                                        description="PostgreSQL用户名",
                                        hint="用于连接PostgreSQL数据库的用户名"),
                    "password": ConfigField(type=str,
                                            default="yourpassword",
                                            description="PostgreSQL密码",
                                            input_type="password",
                                            hint="用于连接PostgreSQL数据库的密码"),
                    "database": ConfigField(type=str,
                                            default="pgvec_maimem_db",
                                            description="PostgreSQL数据库名",
                                            hint="用于存储记忆数据的数据库名称"),
                    "ssl": ConfigField(type=bool,
                                       default=False,
                                       description="是否启用SSL连接",
                                       hint="如果你的PostgreSQL服务器启用了SSL连接，请设置为True"),
                },
            "openai_embedding":
                {
                    "api_key": ConfigField(type=str,
                                           default="sk-xxx",
                                           description="OpenAI API密钥",
                                           input_type="password",
                                           hint="用于访问OpenAI嵌入模型的API密钥"),
                    "model": ConfigField(type=str,
                                         default="baai/bge-m3",
                                         description="OpenAI嵌入模型名称",
                                         hint="用于生成文本嵌入的OpenAI模型名称，例如：text-embedding-ada-002"),
                    "base_url": ConfigField(type=str,
                                            default="https://openrouter.ai/api/v1",
                                            description="OpenAI API基础URL",
                                            hint="如果你使用的是OpenAI的自定义部署或代理，请在此处指定基础URL"), },
            "generic_cfg":
                {
                    "cron_interval": ConfigField(type=int,
                                                 default=10,
                                                 description="定时任务间隔（分钟）",
                                                 min=1, max=1440,
                                                 hint="定时任务执行的时间间隔，单位为分钟"),
                    "dropout_rate": ConfigField(type=float,
                                                default=0.3,
                                                description="Dropout率",
                                                min=0.0, max=1.0,
                                                hint="查询时的Dropout率"),
                    "min_edge_weight": ConfigField(type=float,
                                                   default=0.2,
                                                   description="最小边权重",
                                                   min=0.0, max=1.0,
                                                   hint="在图数据库中考虑的最小边权重"),
                    "max_depth": ConfigField(type=int,
                                             default=5,
                                             description="最大深度",
                                             min=1, max=20,
                                             hint="在图数据库中搜索的最大深度"),
                    "strengthen_boost": ConfigField(type=float,
                                                    default=0.1,
                                                    description="强化系数",
                                                    min=0.0, max=1.0,
                                                    hint="用于强化记忆连接的系数"),
                    "similarity_threshold": ConfigField(type=float,
                                                        default=0.75,
                                                        description="相似度阈值",
                                                        min=0.0, max=1.0,
                                                        hint="在查询时考虑的最小相似度阈值"),
                    "auto_link": ConfigField(type=bool,
                                             default=True,
                                             description="自动链接相邻Topics",
                                             hint="是否自动链接相邻的主题以构建记忆图谱"),
                }
        }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        global dbman, cron_task, config, initialized
        # Initialize database module with config
        loop = asyncio.get_event_loop()
        cron_interval = self.get_config("generic_cfg.cron_interval")
        if not initialized:
            initialized = True

            async def init():
                global dbman, cron_task, config
                # 连接到PostgreSQL数据库
                db_conn = await asyncpg.connect(
                    host=self.get_config("postgresql.host"),
                    port=self.get_config("postgresql.port"),
                    user=self.get_config("postgresql.user"),
                    password=self.get_config("postgresql.password"),
                    database=self.get_config("postgresql.database"),
                    ssl=self.get_config("postgresql.ssl")
                )
                config = db_mod.MemorySearchConfig(
                    dropout_rate=self.get_config("generic_cfg.dropout_rate"),
                    min_edge_weight=self.get_config("generic_cfg.min_edge_weight"),
                    max_depth=self.get_config("generic_cfg.max_depth"),
                    strengthen_boost=self.get_config("generic_cfg.strengthen_boost"),
                    similarity_threshold=self.get_config("generic_cfg.similarity_threshold"),
                    auto_link=self.get_config("generic_cfg.auto_link"),
                )
                dbman = db_mod.GraphMemoryDB(
                    self.get_config("openai_embedding.base_url"),
                    db_conn,
                    self.get_config("openai_embedding.api_key"),
                    self.get_config("openai_embedding.model"),
                )

                async def cron_task_():
                    while True:
                        await dbman.cron()
                        await asyncio.sleep(cron_interval * 60)

                cron_task = loop.create_task(cron_task_())

            loop.create_task(init())
        # 获取插件组件
        return [
            (ReadMem.get_tool_info(), ReadMem),
            (WriteMem.get_tool_info(), WriteMem),
        ]


class ReadMem(BaseTool):
    name = "read_mem"  # 工具名称

    # 工具描述，告诉LLM这个工具的用途
    description = "这个工具用来读取记忆库中的信息"

    parameters = [
        ("query", "string", "查询参数", True),
        ("limit", "integer", "返回结果的数量限制，默认为无", False),
    ]

    available_for_llm = True  # 是否对LLM可用

    async def execute(self, function_args: Dict[str, Any]):
        """执行工具逻辑"""
        s = timeit.default_timer()
        if dbman is None:
            raise RuntimeError("Database manager is not initialized yet.")
        result = await dbman.read_mem(function_args.get("query"), config)
        result = result[: function_args.get("limit")] if function_args.get("limit") else result
        result_txt = ''
        for (tp, inst) in result:
            result_txt += f"- Topic: {tp}, Content: {inst}\n"
        e = timeit.default_timer()
        res = ujson.dumps(
            {
                "elapsed_time": f"{round((e - s) * 1e6, 3)} µs",
                "result": f"Results:\n\n-----------------------------------------------\n\n{result_txt}",
            }
        )

        return {"name": self.name, "content": res}


class WriteMem(BaseTool):
    name = "write_mem"  # 工具名称

    # 工具描述，告诉LLM这个工具的用途
    description = "这个工具用来写入记忆库中的信息"

    parameters = [
        ("topic", "string", "记忆主题", True),
        ("instance", "string", "记忆内容", True),
    ]

    available_for_llm = True  # 是否对LLM可用

    async def execute(self, function_args: Dict[str, Any]):
        """执行工具逻辑"""
        s = timeit.default_timer()
        if dbman is None:
            raise RuntimeError("Database manager is not initialized yet.")
        await dbman.write_mem(function_args.get("topic"), function_args.get("instance"), config)
        e = timeit.default_timer()
        res = ujson.dumps(
            {
                "elapsed_time": f"{round((e - s) * 1e6, 3)} µs",
                "result": f"Memory written successfully for topic '{function_args.get('topic')}'.",
            }
        )

        return {"name": self.name, "content": res}
