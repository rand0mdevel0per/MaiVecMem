import asyncio
import tomllib
import argparse
from typing import Dict, Any

import asyncpg
import ujson
import re
import os

from . import db_mod, hf_converter


def load_config(file_path: str) -> dict:
    """加载配置文件"""
    with open(file_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


async def load_dataset_from_json(
    dbman: db_mod.GraphMemoryDB, file_path: str, memory_search_config: db_mod.MemorySearchConfig
):
    """从JSON文件加载数据集到数据库"""
    json_raw: Dict[str, Any] = ujson.load(open(file_path, "r", encoding="utf-8"))
    batch = []
    for entry in json_raw.keys():
        batch.append((entry, list(json_raw[entry])))
    await dbman.batch_add_mem(batch, memory_search_config)


async def initialize_database(db_conn: asyncpg.Connection, cfg: Dict[str, Any] | None = None):
    """初始化数据库表结构

    Read `model_info.json` produced by the plugin init to detect embedding dimension.
    If missing, fall back to original `schema.sql` without modification.
    """
    plugin_dir = os.path.dirname(__file__)

    # Try to load model_info.json (produced by plugin init). This avoids probing from CLI.
    model_info = None
    model_info_path = os.path.join(plugin_dir, "model_info.json")
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, "r", encoding="utf-8") as mf:
                model_info = ujson.load(mf)
            if model_info and "dimension" in model_info:
                print(f"[INFO] Loaded model_info.json with dimension {model_info['dimension']}")
            else:
                print(f"[WARN] model_info.json does not contain dimension: {model_info}")
        except Exception as e:
            print(f"[WARN] Failed to read model_info.json: {e}")
    else:
        print("[WARN] model_info.json not found; will use schema.sql as-is")

    # Read schema and patch vector size if we got a dimension
    schema_path = os.path.join(plugin_dir, "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        init_script = f.read()

    if model_info and isinstance(model_info, dict) and model_info.get("dimension"):
        dim_val = model_info.get("dimension")
        try:
            dim = int(dim_val)
        except Exception:
            dim = None
        if dim:

            def _replacer(m):
                return f"vector({dim})"

            patched_script = re.sub(r"vector\(\s*\d+\s*\)", _replacer, init_script)
            # Also save patched script for debugging
            gen_path = os.path.join(plugin_dir, "schema.generated.sql")
            try:
                with open(gen_path, "w", encoding="utf-8") as gf:
                    gf.write(patched_script)
                print(f"[INFO] Wrote generated schema to {gen_path}")
            except Exception:
                pass
            script_to_run = patched_script
        else:
            script_to_run = init_script
    else:
        script_to_run = init_script

    # Execute SQL within a transaction
    try:
        with db_conn.transaction():
            await db_conn.execute(script_to_run)
        print("[INFO] Graph memory tables initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize database schema: {e}")
        raise


async def export_graph_memory(dbman: db_mod.GraphMemoryDB, export_path: str):
    """导出图记忆到JSON文件"""
    data = await dbman.export_graph()
    with open(export_path, "w", encoding="utf-8") as f:
        ujson.dump(data, f)
    print(f"[INFO] Graph memory exported to '{export_path}' successfully.")


async def search_memory(dbman: db_mod.GraphMemoryDB, query: str, memory_search_config: db_mod.MemorySearchConfig):
    """搜索记忆"""
    results = await dbman.read_mem(query, memory_search_config)
    print(f"[INFO] Search results for query '{query}':")
    for topic, contents in results:
        print(f"\nTopic: {topic}")
        for content in contents:
            print(f"- {content}")


async def interactive_mode(dbman: db_mod.GraphMemoryDB, memory_search_config: db_mod.MemorySearchConfig):
    """交互式模式"""
    while True:
        print("""
        MaiBot Postgresql Memory Plugin Cli Tool
        ==================================================================
        [1] Initialize Graph Memory Tables
        [2] Load Dataset from HuggingFace
        [3] Load Dataset from Local .json File
        [4] Test Memory Search
        [5] Manually Add Memory Entry
        [6] Export Graph Memory to JSON
        [0] Exit
        ==================================================================
        """)
        option = input("Please select an option (0-6): ").strip()
        if option == "1":
            await initialize_database(dbman.db_conn, cfg=None)
        elif option == "2":
            import datasets

            dataset_name = input("Enter HuggingFace dataset name (e.g., 'ag_news'): ").strip()
            if not dataset_name:
                print("Dataset name cannot be empty.")
                continue
            dataset_cfgs = datasets.get_dataset_config_names(dataset_name)
            if not dataset_cfgs:
                print(f"[ERROR] Dataset '{dataset_name}' not found on HuggingFace.")
                continue
            print(f"Available configurations for '{dataset_name}':")
            for idx, cfg in enumerate(dataset_cfgs):
                print(f"[{idx}] {cfg}")
            cfg_option = input(f"Select configuration (0-{len(dataset_cfgs) - 1}) or press ENTER for default: ").strip()
            if cfg_option.isdigit() and 0 <= int(cfg_option) < len(dataset_cfgs):
                selected_cfg = dataset_cfgs[int(cfg_option)]
            else:
                selected_cfg = None
            print(f"[INFO] Loading dataset '{dataset_name}' with configuration '{selected_cfg}'...")
            dataset_raw = datasets.load_dataset(dataset_name, selected_cfg, split="train")
            max_samples = input("Enter maximum number of samples to load (or press ENTER for all): ").strip()
            await hf_converter.convert_dataset_to_memory_json(
                dataset_raw, "tmp_dataset.json", int(max_samples) if max_samples.isdigit() else None
            )
            await load_dataset_from_json(dbman, "tmp_dataset.json", memory_search_config)
            print(f"[INFO] Dataset '{dataset_name}' loaded successfully.")
        elif option == "3":
            file_path = input("Enter local JSON file path: ").strip()
            await load_dataset_from_json(dbman, file_path, memory_search_config)
            print(f"[INFO] Dataset from '{file_path}' loaded successfully.")
        elif option == "4":
            query = input("Enter search query: ").strip()
            await search_memory(dbman, query, memory_search_config)
        elif option == "5":
            topic = input("Enter memory topic: ").strip()
            content = []
            while True:
                line = input("Enter memory content line (Press ENTER without input to break): ").strip()
                if line == "":
                    break
                content.append(line)
            await dbman.add_mem(topic, content, memory_search_config)
            print(f"[INFO] Memory entry added under topic '{topic}'.")
        elif option == "6":
            export_path = input("Enter export JSON file path: ").strip()
            await export_graph_memory(dbman, export_path)
        elif option == "0":
            print("Exiting interactive mode.")
            break
        else:
            print("Invalid option. Please try again.")


async def setup_database_manager(cfg: dict) -> db_mod.GraphMemoryDB:
    """设置数据库管理器"""
    db_conn = await asyncpg.connect(
        host=cfg["postgresql"]["host"],
        port=cfg["postgresql"]["port"],
        user=cfg["postgresql"]["user"],
        password=cfg["postgresql"]["password"],
        database=cfg["postgresql"]["database"],
        ssl=cfg["postgresql"].get("ssl", False),
    )

    memory_search_config = db_mod.MemorySearchConfig(
        dropout_rate=cfg["generic_cfg"]["dropout_rate"],
        min_edge_weight=cfg["generic_cfg"]["min_edge_weight"],
        max_depth=cfg["generic_cfg"]["max_depth"],
        strengthen_boost=cfg["generic_cfg"]["strengthen_boost"],
        similarity_threshold=cfg["generic_cfg"]["similarity_threshold"],
        auto_link=cfg["generic_cfg"]["auto_link"],
    )

    # Prefer model_info.json credentials if available (written by plugin init)
    plugin_dir = os.path.dirname(__file__)
    model_info_path = os.path.join(plugin_dir, "model_info.json")
    emb_base = cfg["openai_embedding"]["base_url"]
    emb_key = cfg["openai_embedding"]["api_key"]
    emb_model = cfg["openai_embedding"]["model"]
    if os.path.exists(model_info_path):
        try:
            mi = ujson.load(open(model_info_path, "r", encoding="utf-8"))
            pit = mi.get("plugin_config_snapshot", {}).get("openai_embedding", {})
            emb_base = pit.get("base_url") or emb_base
            emb_key = pit.get("api_key") or emb_key
            emb_model = pit.get("model") or emb_model
            print(f"[INFO] Using embedding credentials from model_info.json: model={emb_model}, base={emb_base}")
        except Exception as e:
            print(f"[WARN] Failed to read model_info.json for embedding creds: {e}")

    return (
        db_mod.GraphMemoryDB(
            emb_base,
            db_conn,
            emb_key,
            emb_model,
        ),
        memory_search_config,
        db_conn,
    )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MaiBot Postgresql Memory Plugin Cli Tool", formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init 命令
    subparsers.add_parser("init", help="初始化数据库结构")
    # apply-update 命令
    apply_parser = subparsers.add_parser("apply-update", help="Apply available plugin update (reads update_available.json)")
    apply_parser.add_argument("--auto", action="store_true", help="Automatically apply the update (backup+pull+migration)")

    # import 命令
    import_parser = subparsers.add_parser("import", help="导入知识库 JSON 文件")
    import_parser.add_argument("file_path", help="JSON 文件路径")

    # search 命令
    search_parser = subparsers.add_parser("search", help="搜索相关记忆")
    search_parser.add_argument("query", help="搜索关键词")

    # export 命令
    export_parser = subparsers.add_parser("export", help="导出数据库内容到 JSON 文件")
    export_parser.add_argument("file_path", help="导出文件路径")

    # interactive 命令
    subparsers.add_parser("interactive", help="进入交互式模式")

    return parser.parse_args()


async def main():
    """主函数"""
    config = load_config("config.toml")
    args = parse_arguments()

    dbman, memory_search_config, db_conn = await setup_database_manager(config)
    print("[INFO] Connected to database successfully")

    try:
        if args.command == "init":
            await initialize_database(db_conn, cfg=config)
        elif args.command == "apply-update":
            # call the helper script
            script_path = os.path.join(os.path.dirname(__file__), "apply_update.py")
            cmd = ["python", script_path]
            if hasattr(args, 'auto') and args.auto:
                cmd.append("--auto")
            import subprocess
            print(f"[INFO] Running apply_update: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            print(proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr)
        elif args.command == "import":
            await load_dataset_from_json(dbman, args.file_path, memory_search_config)
            print(f"[INFO] Dataset from '{args.file_path}' loaded successfully.")
        elif args.command == "search":
            await search_memory(dbman, args.query, memory_search_config)
        elif args.command == "export":
            await export_graph_memory(dbman, args.file_path)
        elif args.command == "interactive":
            await interactive_mode(dbman, memory_search_config)
        else:
            await interactive_mode(dbman, memory_search_config)
    finally:
        await db_conn.close()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
