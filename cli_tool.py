# Ensure running as a script sets package context before relative imports
if __name__ == "__main__" and __package__ is None:
    import sys
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    __package__ = "plugins.MaiVecMem"

import asyncio
import tomllib
import argparse
from typing import Dict, Any

import asyncpg
import re
import os
import numpy as np

from . import db_mod, hf_converter
from . import libopenie

# try to import secret_store decrypt helper
try:
    from .secret_store import decrypt_for_service, key_available as secret_key_available
except Exception:
    decrypt_for_service = None

    def secret_key_available():
        return False


# Prefer ujson for speed but fall back to stdlib json if unavailable
try:
    import ujson as ujson  # type: ignore
except Exception:
    import json as ujson  # type: ignore


def _ensure_plugin_modules() -> None:
    """Ensure plugin and repo paths are on sys.path so local imports work when running as a script.

    This is intentionally minimal and only mutates sys.path in-process. It is safe to call repeatedly.
    """
    import sys
    plugin_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(plugin_dir, "..", ".."))

    # prepend plugin_dir first so local package imports resolve
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    # also ensure repo root is available for importing shared modules
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_config(file_path: str) -> dict:
    """加载配置文件"""
    with open(file_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


async def load_dataset_from_json(
    dbman: db_mod.GraphMemoryDB, file_path: str, memory_search_config: db_mod.MemorySearchConfig
):
    """从JSON文件加载数据集到数据库"""
    _ensure_plugin_modules()
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
        async with db_conn.transaction():
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
    """Interactive mode (ASCII-only English prompts)

    Reordered options for a more logical workflow.
    """
    _ensure_plugin_modules()
    while True:
        print(
            "\nMaiVecMem CLI - Interactive Menu\n========================================\n[1] Initialize graph memory tables\n[2] Import OpenIE directory (batch)\n[3] Import OpenIE file (single)\n[4] Load dataset from HuggingFace\n[5] Load dataset from local JSON file\n[6] Test memory search\n[7] Manually add memory entry\n[8] Export graph memory to JSON\n[0] Exit\n========================================"
        )
        option = input("Select an option (0-8): ").strip()

        if option == "1":
            await initialize_database(dbman.db_conn, cfg=None)
        elif option == "2":
            dir_path = input("Enter directory path containing OpenIE JSON files: ").strip()
            strategy = input("Strategy (subject/relation/hybrid/semantic/entity) [semantic]: ").strip() or "semantic"
            min_inst = None
            if strategy == "hybrid":
                v = input("min_instances for hybrid (default 2): ").strip()
                min_inst = int(v) if v.isdigit() else 2
            kw = {}
            if strategy == "semantic":
                v = input("similarity_threshold (0.0-1.0, default 0.7): ").strip()
                try:
                    simt = float(v) if v else 0.7
                except Exception:
                    simt = 0.7
                kw = {
                    "similarity_threshold": simt,
                    "min_instances": int(input("min_instances for clusters (default 1): ").strip() or 1),
                }
            else:
                if min_inst is not None:
                    kw["min_instances"] = min_inst

            if not os.path.isdir(dir_path):
                print(f"[ERROR] Directory not found: {dir_path}")
            else:
                for fname in sorted(os.listdir(dir_path)):
                    if fname.endswith("-openie.json"):
                        fpath = os.path.join(dir_path, fname)
                        print(f"[INFO] Importing from '{fpath}'...")
                        await load_openie_to_db(dbman, fpath, memory_search_config, strategy=strategy, **kw)
        elif option == "3":
            file_path = input("Enter OpenIE JSON file path: ").strip()
            strategy = input("Strategy (subject/relation/hybrid/semantic/entity) [semantic]: ").strip() or "semantic"
            if strategy == "hybrid":
                mi = input("min_instances for hybrid (default 2): ").strip()
                kw = {"min_instances": int(mi) if mi.isdigit() else 2}
            elif strategy == "semantic":
                v = input("similarity_threshold (0.0-1.0, default 0.7): ").strip()
                try:
                    simt = float(v) if v else 0.7
                except Exception:
                    simt = 0.7
                kw = {
                    "similarity_threshold": simt,
                    "min_instances": int(input("min_instances for clusters (default 1): ").strip() or 1),
                }
            else:
                kw = {}
            await load_openie_to_db(dbman, file_path, memory_search_config, strategy=strategy, **kw)
        elif option == "4":
            import datasets

            dataset_name = input("Enter HuggingFace dataset name (e.g., ag_news): ").strip()
            if not dataset_name:
                print("[ERROR] Dataset name cannot be empty.")
                continue
            dataset_cfgs = datasets.get_dataset_config_names(dataset_name)
            if not dataset_cfgs:
                print(f"[ERROR] Dataset '{dataset_name}' not found on HuggingFace.")
                continue
            print("Available configurations:")
            for idx, cfg in enumerate(dataset_cfgs):
                print(f"[{idx}] {cfg}")
            cfg_option = input(f"Select configuration (0-{len(dataset_cfgs) - 1}) or press ENTER for default: ").strip()
            if cfg_option.isdigit() and 0 <= int(cfg_option) < len(dataset_cfgs):
                selected_cfg = dataset_cfgs[int(cfg_option)]
            else:
                selected_cfg = None
            print(f"[INFO] Loading dataset '{dataset_name}' config='{selected_cfg}'...")
            dataset_raw = datasets.load_dataset(dataset_name, selected_cfg, split="train")
            max_samples = input("Enter maximum number of samples to load (or press ENTER for all): ").strip()
            await hf_converter.convert_dataset_to_memory_json(
                dataset_raw, "tmp_dataset.json", int(max_samples) if max_samples.isdigit() else None
            )
            await load_dataset_from_json(dbman, "tmp_dataset.json", memory_search_config)
            print(f"[INFO] Dataset '{dataset_name}' loaded successfully.")
        elif option == "5":
            file_path = input("Enter local JSON file path: ").strip()
            await load_dataset_from_json(dbman, file_path, memory_search_config)
            print(f"[INFO] Dataset from '{file_path}' loaded successfully.")
        elif option == "6":
            query = input("Enter search query: ").strip()
            await search_memory(dbman, query, memory_search_config)
        elif option == "7":
            topic = input("Enter memory topic: ").strip()
            content = []
            while True:
                line = input("Enter memory content line (empty line to finish): ").strip()
                if line == "":
                    break
                content.append(line)
            await dbman.add_mem(topic, content, memory_search_config)
            print(f"[INFO] Memory entry added under topic '{topic}'.")
        elif option == "8":
            export_path = input("Enter export JSON file path: ").strip()
            await export_graph_memory(dbman, export_path)
        elif option == "0":
            print("Exiting interactive mode.")
            break
        else:
            print("Invalid option. Please try again.")


async def load_openie_to_db(
    dbman: db_mod.GraphMemoryDB,
    file_path: str,
    memory_search_config: db_mod.MemorySearchConfig,
    strategy: str = "subject",
    **kwargs,
):
    _ensure_plugin_modules()
    """从 OpenIE JSON 文件加载并导入到 GraphMemoryDB

    - file_path: OpenIE JSON 文件路径
    - strategy: 聚合策略（subject/relation/hybrid/semantic/entity）
    - kwargs: 传递给转换器的策略特定参数（例如 hybrid 的 min_instances 或 semantic 的 similarity_threshold）
    """
    # For non-semantic strategies we defer to the converter which already aggregates topics
    if strategy != "semantic":
        try:
            converted = libopenie.OpenIEConverter.convert(file_path, strategy, **kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to convert OpenIE file: {e}")
            return

        if not converted:
            print("[WARN] No topics produced from OpenIE converter; nothing to import.")
            return

        # 转换为 batch 格式: List[Tuple[topic, List[str]]]
        batch = [(topic, list(instances)) for topic, instances in converted.items()]

        try:
            success = await dbman.batch_add_mem(batch, memory_search_config)
            if success:
                print(f"[INFO] Imported {len(batch)} topics from OpenIE file '{file_path}' successfully.")
            else:
                print("[WARN] Import may have partially failed or no new items were added.")
        except Exception as e:
            print(f"[ERROR] Exception while importing into DB: {e}")
        return

    # ----------------------------
    # semantic 聚类实现
    # ----------------------------
    try:
        converted_raw = libopenie.OpenIEConverter.load_raw(file_path)
    except Exception:
        # fallback: try convert to get raw mapping
        try:
            converted_raw = libopenie.OpenIEConverter.convert(file_path, "relation", **kwargs)
        except Exception as e2:
            print(f"[ERROR] Failed to read OpenIE file for semantic processing: {e2}")
            return

    # Flatten instances: converted_raw may be dict(topic->instances) or list of triples; try to produce list
    all_instances = []
    # If converter returned dict-like
    if isinstance(converted_raw, dict):
        for insts in converted_raw.values():
            for s in insts:
                all_instances.append(s)
    elif isinstance(converted_raw, list):
        # list of triplets or entries
        for item in converted_raw:
            if isinstance(item, dict):
                # try keys 'sentence' or 'text'
                txt = item.get("sentence") or item.get("text") or item.get("surface")
                if txt:
                    all_instances.append(txt)
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                all_instances.append(item[0])
    else:
        print("[WARN] Unrecognized OpenIE raw format for semantic strategy")
        return

    # deduplicate while preserving order
    seen = set()
    uniq_instances = []
    for s in all_instances:
        if s is None:
            continue
        key = s.strip()
        if key and key not in seen:
            seen.add(key)
            uniq_instances.append(key)

    if not uniq_instances:
        print("[WARN] No textual instances found for semantic clustering")
        return

    # get embeddings using dbman internal embedder (run in batch)
    try:
        embs = await dbman._embed(uniq_instances)
    except Exception as e:
        print(f"[ERROR] Failed to compute embeddings for semantic clustering: {e}")
        return

    # ensure embeddings as numpy arrays
    emb_arrays = []
    if isinstance(embs, list):
        emb_arrays = embs
    else:
        emb_arrays = [embs]

    # greedy clustering by cosine similarity to centroids
    sim_threshold = float(kwargs.get("similarity_threshold", 0.7))
    clusters = []  # list of dict: {centroid: np.array, items: [indices]}

    def cosine(a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    for idx, v in enumerate(emb_arrays):
        assigned = False
        for cl in clusters:
            sim = cosine(v, cl["centroid"])
            if sim >= sim_threshold:
                # assign
                cl["indices"].append(idx)
                # update centroid
                pts = [emb_arrays[i] for i in cl["indices"]]
                cl["centroid"] = np.mean(np.stack(pts, axis=0), axis=0)
                assigned = True
                break
        if not assigned:
            clusters.append({"centroid": v.copy(), "indices": [idx]})

    # build batch: pick representative sentence as topic name (closest to centroid)
    batch = []
    min_instances = int(kwargs.get("min_instances", 1))
    for ci, cl in enumerate(clusters):
        if len(cl["indices"]) < min_instances:
            # skip small clusters
            continue
        indices = cl["indices"]
        centroid = cl["centroid"]
        # find representative
        best_i = None
        best_s = -1.0
        for i in indices:
            s = cosine(emb_arrays[i], centroid)
            if s > best_s:
                best_s = s
                best_i = i
        topic_name = uniq_instances[best_i][:80] if best_i is not None else f"semantic_cluster_{ci}"
        instances = [uniq_instances[i] for i in indices]
        batch.append((topic_name, instances))

    if not batch:
        print("[WARN] No clusters met min_instances threshold; nothing to import.")
        return

    try:
        success = await dbman.batch_add_mem(batch, memory_search_config)
        if success:
            print(f"[INFO] Imported {len(batch)} semantic clusters from OpenIE file '{file_path}' successfully.")
        else:
            print("[WARN] Semantic import may have partially failed.")
    except Exception as e:
        print(f"[ERROR] Exception while importing semantic clusters into DB: {e}")


async def setup_database_manager(cfg: dict) -> db_mod.GraphMemoryDB:
    """设置数据库管理器"""
    _ensure_plugin_modules()
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
            with open(model_info_path, "r", encoding="utf-8") as mf:
                mi = ujson.load(mf)
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
    apply_parser = subparsers.add_parser(
        "apply-update", help="Apply available plugin update (reads update_available.json)"
    )
    apply_parser.add_argument(
        "--auto", action="store_true", help="Automatically apply the update (backup+pull+migration)"
    )

    # import 命令
    import_parser = subparsers.add_parser("import", help="导入知识库 JSON 文件")
    import_parser.add_argument("file_path", help="JSON 文件路径")

    # import-openie 命令: 导入 OpenIE 格式的 JSON 文件
    import_oi_parser = subparsers.add_parser("import-openie", help="导入 OpenIE JSON 文件并转换为 topic-instance 格式")
    import_oi_parser.add_argument("file_path", help="OpenIE JSON 文件路径")
    import_oi_parser.add_argument(
        "--strategy",
        choices=["subject", "relation", "hybrid", "semantic", "entity"],
        default="subject",
        help="聚合策略，默认 subject",
    )
    import_oi_parser.add_argument(
        "--min-instances", type=int, default=2, help="仅对 hybrid 策略有效：topic 最少包含实例数（默认 2)"
    )

    # 新增：import-openie-dir 命令: 导入目录中所有 OpenIE JSON 文件
    import_oi_dir_parser = subparsers.add_parser(
        "import-openie-dir", help="导入目录下的所有 OpenIE JSON 文件并逐个转换导入"
    )
    import_oi_dir_parser.add_argument("dir_path", help="包含 OpenIE JSON 文件的目录路径")
    import_oi_dir_parser.add_argument(
        "--pattern", default="-openie.json", help="匹配文件后缀或模式（默认 '-openie.json'，也可以使用 '.json'）"
    )
    import_oi_dir_parser.add_argument(
        "--strategy",
        choices=["subject", "relation", "hybrid", "semantic", "entity"],
        default="semantic",
        help="聚合策略，默认 semantic",
    )
    import_oi_dir_parser.add_argument(
        "--min-instances", type=int, default=2, help="仅对 hybrid 策略有效：topic 最少包含实例数（默认 2)"
    )

    # search 命令
    search_parser = subparsers.add_parser("search", help="搜索相关记忆")
    search_parser.add_argument("query", help="搜索关键词")

    # export 命令
    export_parser = subparsers.add_parser("export", help="导出数据库内容到 JSON 文件")
    export_parser.add_argument("file_path", help="导出文件路径")

    # interactive 命令
    subparsers.add_parser("interactive", help="进入交互式模式")

    return parser.parse_args()


def load_cli_config() -> dict:
    """Load minimal config for CLI:
    - PostgreSQL creds are read from repo `config.toml` if present (only postgresql section used)
    - Other settings (openai_embedding, generic_cfg) are read from plugins/MaiVecMem/model_info.json (must exist)
    This satisfies: CLI only needs pg creds from repo config and everything else from model_info.json.
    """
    plugin_dir = os.path.dirname(__file__)

    # Load PG creds from repo config.toml if available (only use 'postgresql' section)
    repo_cfg = {}
    try:
        repo_cfg = load_config("config.toml")
    except Exception:
        repo_cfg = {}
    pg = repo_cfg.get("postgresql", {})

    # Require model_info.json in plugin dir
    model_info_path = os.path.join(plugin_dir, "model_info.json")
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"model_info.json not found in plugin directory: {model_info_path}")

    try:
        mi = ujson.load(open(model_info_path, "r", encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load model_info.json: {e}") from None

    snapshot = mi.get("plugin_config_snapshot", {})
    openai_embedding = snapshot.get("openai_embedding", {})

    # Provide reasonable defaults for generic_cfg if not present
    generic_defaults = {
        "dropout_rate": 0.3,
        "min_edge_weight": 0.2,
        "max_depth": 5,
        "strengthen_boost": 0.1,
        "similarity_threshold": 0.75,
        "auto_link": True,
    }
    generic_cfg = snapshot.get("generic_cfg", generic_defaults)

    return {
        "postgresql": pg,
        "openai_embedding": openai_embedding,
        "generic_cfg": generic_cfg,
    }


async def main():
    """主函数"""
    args = parse_arguments()

    # Load minimal CLI config: pg creds from repo config.toml and other settings from model_info.json
    try:
        config = load_cli_config()
    except Exception as e:
        print(f"[ERROR] Failed to load CLI config (pg creds + model_info.json): {e}")
        return

    dbman, memory_search_config, db_conn = await setup_database_manager(config)
    print("[INFO] Connected to database successfully")

    try:
        if args.command == "init":
            await initialize_database(db_conn, cfg=config)
        elif args.command == "apply-update":
            # call the helper script
            script_path = os.path.join(os.path.dirname(__file__), "apply_update.py")
            cmd = ["python", script_path]
            if hasattr(args, "auto") and args.auto:
                cmd.append("--auto")
            import subprocess

            print(f"[INFO] Running apply_update: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            print(proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr)
        elif args.command == "import-openie":
            # Use OpenIE converter and import into DB
            strategy = getattr(args, "strategy", "subject")
            min_instances = getattr(args, "min_instances", None) or getattr(args, "min-instances", None)
            kw = {}
            if min_instances is not None:
                kw["min_instances"] = int(min_instances)
            await load_openie_to_db(dbman, args.file_path, memory_search_config, strategy=strategy, **kw)
        elif args.command == "import-openie-dir":
            # Import all matching OpenIE JSON files from a directory
            dir_path = getattr(args, "dir_path", None)
            if not dir_path or not os.path.isdir(dir_path):
                print(f"[ERROR] Provided path is not a directory: {dir_path}")
            else:
                pattern = getattr(args, "pattern", "-openie.json")
                strategy = getattr(args, "strategy", "semantic")
                min_instances = getattr(args, "min_instances", None) or getattr(args, "min-instances", None)
                kw = {}
                if min_instances is not None:
                    kw["min_instances"] = int(min_instances)
                print(f"[INFO] Scanning directory '{dir_path}' for files matching '{pattern}'")
                for fname in sorted(os.listdir(dir_path)):
                    if (
                        pattern == ".json"
                        and fname.endswith(".json")
                        or (pattern != ".json" and fname.endswith(pattern))
                    ):
                        fpath = os.path.join(dir_path, fname)
                        print(f"[INFO] Importing from '{fpath}'...")
                        await load_openie_to_db(dbman, fpath, memory_search_config, strategy=strategy, **kw)
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
