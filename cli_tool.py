import asyncio
import tomllib
from typing import Dict

import asyncpg
import ujson

import db_mod
import hf_converter


def load_config(file_path):
    with open(file_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


async def load_dataset_from_json(dbman: db_mod.GraphMemoryDB, file_path: str,
                                 memory_search_config: db_mod.MemorySearchConfig):
    json_raw: Dict[str, Any] = ujson.load(open(file_path, "r", encoding="utf-8"))
    batch = []
    for entry in json_raw.keys():
        batch.append((entry, list(json_raw[entry])))
    await dbman.batch_add_mem(batch, memory_search_config)


async def main(cfg: dict):
    db_conn = await asyncpg.connect(
        host=cfg["postgresql"]["host"],
        port=cfg["postgresql"]["port"],
        user=cfg["postgresql"]["user"],
        password=cfg["postgresql"]["password"],
        database=cfg["postgresql"]["database"],
        ssl=cfg["postgresql"]["ssl"]
    )
    memory_search_config = db_mod.MemorySearchConfig(
        dropout_rate=cfg["generic_cfg"]["dropout_rate"],
        min_edge_weight=cfg["generic_cfg"]["min_edge_weight"],
        max_depth=cfg["generic_cfg"]["max_depth"],
        strengthen_boost=cfg["generic_cfg"]["strengthen_boost"],
        similarity_threshold=cfg["generic_cfg"]["similarity_threshold"],
        auto_link=cfg["generic_cfg"]["auto_link"],
    )
    dbman = db_mod.GraphMemoryDB(
        cfg["openai_embedding"]["base_url"],
        db_conn,
        cfg["openai_embedding"]["api_key"],
        cfg["openai_embedding"]["model"],
    )
    print("[INFO] Connected to database successfully")
    while True:
        print(
            """
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
        Please select an option (0-5):
        """, end=""
        )
        option = input("").strip()
        if option == "1":
            with open("./schema.sql", "r", encoding="utf-8") as f:
                init_script = f.read()
                with db_conn.transaction():
                    await db_conn.execute(init_script)
            print("[INFO] Graph memory tables initialized successfully.")
        elif option == "2":
            dataset_name = input("Enter HuggingFace dataset name (e.g., 'ag_news'): ").strip()
            import datasets
            dataset_cfgs = datasets.get_dataset_config_names(dataset_name)
            if dataset_cfgs:
                if len(dataset_cfgs) == 1:
                    print(
                        f"[INFO] Only one configuration '{dataset_cfgs[0]}' available for '{dataset_name}'. Using it by default.")
                    dataset = datasets.load_dataset(dataset_name, dataset_cfgs[0])
                else:
                    print(f"[INFO] Available configurations for '{dataset_name}': {dataset_cfgs}")
                    dataset_config = input("Enter dataset configuration name (or press Enter to skip): ").strip()
                    if dataset_config:
                        dataset = datasets.load_dataset(dataset_name, dataset_config)
                    else:
                        dataset = datasets.load_dataset(dataset_name)
            else:
                dataset = datasets.load_dataset(dataset_name)
            print(f"[INFO] Dataset '{dataset_name}' loaded successfully.")
            opt = input(
                f"[EVALUATION] Do you want to proceed with loading the dataset {dataset_name} into memory? [Y/n]: ").trim()
            if opt.lower() == "n":
                print("[INFO] Dataset loading cancelled by user.")
                continue
            elif opt.lower() != "y" and opt != "":
                print("[WARNING] Invalid input. Dataset loading cancelled.")
                continue
            max_samples = input("Enter maximum number of samples to load (or press Enter for all): ").strip()
            print(f"[INFO] Loading dataset into graph memory...")
            hf_converter.convert_dataset_to_memory_json(dataset, "tmp_memory.json",
                                                        int(max_samples) if max_samples.isdigit() else None)
            await load_dataset_from_json(dbman, "tmp_memory.json", memory_search_config)
            print(f"[INFO] Dataset '{dataset_name}' loaded into graph memory successfully.")
        elif option == "3":
            file_path = input("Enter local .json file path: ").strip()
            await load_dataset_from_json(dbman, file_path, memory_search_config)
            print(f"[INFO] Local dataset from '{file_path}' loaded successfully.")
        elif option == "4":
            query = input("Enter search query: ").strip()
            results = await dbman.read_mem(query, memory_search_config)
            print(f"[INFO] Search results for query '{query}':")
            for res in results:
                print(res)
        elif option == "5":
            topic = input("Enter memory entry topic: ").strip()
            content = []
            while True:
                content_ = input("Enter memory entry content (Press ENTER to break): \n").strip()
                if content_ == "":
                    break
                content += [content_]
            await dbman.add_mem(topic, content, memory_search_config)
            print("[INFO] Memory entry added successfully.")
        elif option == "6":
            export_path = input("Enter export file path (e.g., 'exported_memory.json'): ").strip()
            ujson.dump(await dbman.export_graph(), open(export_path, "w", encoding="utf-8"))
            print(f"[INFO] Graph memory exported to '{export_path}' successfully.")
        elif option == "0":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")
    await db_conn.close()


if __name__ == "__main__":
    config = load_config("config.toml")
    print("[INFO] Loaded configuration Successfully")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(config))
