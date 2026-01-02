from . import db_mod

import sys
import os
import asyncio
import asyncpg
import timeit
import ujson
import time
import urllib.request
import json
import subprocess
import shutil
from typing import List, Tuple, Type, Dict, Any

from .secret_store import encrypt_for_service, key_available  # moved up to top-level imports  # noqa: E402

# attempt to import plugin management wrapper for hot reload
try:
    from . import plugin_manage_wrapper as _pmw
except Exception:
    try:
        import plugin_manage_wrapper as _pmw
    except Exception:
        _pmw = None

current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.abspath(os.path.join(current_dir, "../.."))
if target_path not in sys.path:
    sys.path.insert(0, target_path)

from src.plugin_system import BasePlugin, register_plugin, ComponentInfo, ConfigField, BaseTool, config_api  # noqa: E402

dbman = None
cron_task = None
config = None
initialized = False


async def _background_manifest_check(plugin_dir: str, db_conn=None, timeout: int = 10, auto_apply: bool = False):
    """
    Background non-blocking task that checks remote manifest and attempts to fetch migration scripts if schema changed.
    - plugin_dir: local plugin directory path
    - db_conn: optional asyncpg.Connection to execute migration SQL
    - timeout: network timeout
    - auto_apply: whether to attempt automatic git pull and run migration

    Process:
    1. Fetch remote rel/_manifest.json (raw or via gh-proxy fallback)
    2. If remote version > local version: fetch remote 'rel/schema.sql' raw and compare with local schema.sql
    3. If schema differs: find commit id for latest change to schema.sql on rel branch via GitHub API
    4. Fetch migration script from migration-scripts branch: migration-scripts/sql/<commit-id>.sql (raw or gh-proxy)
    5. If migration script exists and db_conn provided, execute it.
    6. Write update_available.json with details.
    7. If auto_apply true, attempt backup + git pull + run migration, with rollback on failure.
    """
    import hashlib

    def _fetch_url_text(url: str, timeout: int):
        req = urllib.request.Request(url, headers={"User-Agent": "MaiVecMem-Updater/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")

    def _run_subprocess(cmd: list):
        return subprocess.run(cmd, capture_output=True, text=True)

    manifest_path = os.path.join(plugin_dir, "_manifest.json")
    if not os.path.exists(manifest_path):
        print("[INFO] No local _manifest.json found for update check.")
        return

    try:
        local = json.load(open(manifest_path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read local manifest: {e}")
        return

    repo_url = local.get("repository_url")
    local_ver = local.get("version")
    if not repo_url or not repo_url.startswith("https://github.com/"):
        print(f"[INFO] Repository url unsupported for auto-update: {repo_url}")
        return

    repo = repo_url.rstrip(".git").split("github.com/")[-1]
    raw_manifest_rel = f"https://raw.githubusercontent.com/{repo}/rel/_manifest.json"
    gh_proxy_manifest = f"https://gh-proxy.org/https://raw.githubusercontent.com/{repo}/rel/_manifest.json"

    remote = None
    # fetch manifest (in thread)
    try:
        remote_txt = await asyncio.to_thread(_fetch_url_text, raw_manifest_rel, timeout)
        remote = json.loads(remote_txt)
    except Exception as e:
        print(f"[WARN] Direct raw.githubusercontent fetch failed: {e}; trying gh-proxy mirror")
        try:
            remote_txt = await asyncio.to_thread(_fetch_url_text, gh_proxy_manifest, timeout)
            remote = json.loads(remote_txt)
        except Exception as e2:
            print(f"[WARN] gh-proxy fetch also failed: {e2}; skipping update check")
            return

    remote_ver = remote.get("version")

    def _parts(v):
        return [int(x) if x.isdigit() else 0 for x in str(v).split(".")]

    def _semver_compare(a, b):
        pa = _parts(a)
        pb = _parts(b)
        L = max(len(pa), len(pb))
        pa += [0] * (L - len(pa))
        pb += [0] * (L - len(pb))
        for i in range(L):
            if pa[i] > pb[i]:
                return 1
            if pa[i] < pb[i]:
                return -1
        return 0

    if _semver_compare(remote_ver, local_ver) <= 0:
        print(f"[INFO] Plugin is up-to-date (local={local_ver}, remote={remote_ver})")
        return

    # remote is newer -> attempt to fetch remote schema and diff
    local_schema_path = os.path.join(plugin_dir, "schema.sql")
    local_schema = None
    try:
        with open(local_schema_path, "r", encoding="utf-8") as f:
            local_schema = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read local schema.sql: {e}")

    raw_schema_rel = f"https://raw.githubusercontent.com/{repo}/rel/schema.sql"
    gh_proxy_schema = f"https://gh-proxy.org/https://raw.githubusercontent.com/{repo}/rel/schema.sql"

    remote_schema = None
    try:
        remote_schema = await asyncio.to_thread(_fetch_url_text, raw_schema_rel, timeout)
    except Exception:
        try:
            remote_schema = await asyncio.to_thread(_fetch_url_text, gh_proxy_schema, timeout)
        except Exception as e2:
            print(f"[WARN] Failed to fetch remote schema via raw and gh-proxy: {e2}")
            remote_schema = None

    schema_changed = False
    if remote_schema is not None and local_schema is not None:
        if (
            hashlib.sha256(remote_schema.encode("utf-8")).hexdigest()
            != hashlib.sha256(local_schema.encode("utf-8")).hexdigest()
        ):
            schema_changed = True
            print("[INFO] Detected change in schema.sql between local and remote rel branch.")
        else:
            print("[INFO] No change in schema.sql detected.")
    else:
        print("[INFO] Could not compare schema files; proceeding to record update info only.")

    # write update_available.json with details
    update_info = {
        "local_version": local_ver,
        "remote_version": remote_ver,
        "schema_changed": schema_changed,
        "update_time": int(time.time()),
        "remote_manifest": remote,
    }
    try:
        with open(os.path.join(plugin_dir, "update_available.json"), "w", encoding="utf-8") as f:
            ujson.dump(update_info, f)
        print("[INFO] update_available.json written; review before applying update")
    except Exception as e:
        print(f"[WARN] Failed to write update_available.json: {e}")

    # If no schema change, optional auto_apply will still attempt git pull if enabled
    if not schema_changed and not auto_apply:
        print("[INFO] No schema change and auto_apply disabled; not applying update automatically")
        return

    if not auto_apply:
        print("[INFO] auto_apply disabled; skipping automatic apply of remote update")
        return

    # Attempt to get commit id for latest change of schema.sql on rel branch using GitHub API
    commit_sha = None
    try:
        commits_api = f"https://api.github.com/repos/{repo}/commits?path=schema.sql&sha=rel&per_page=1"
        commits_txt = await asyncio.to_thread(_fetch_url_text, commits_api, timeout)
        commits_json = json.loads(commits_txt)
        if isinstance(commits_json, list) and len(commits_json) > 0:
            commit_sha = commits_json[0].get("sha")
            print(f"[INFO] Found commit SHA for schema change: {commit_sha}")
    except Exception as e:
        print(f"[WARN] Failed to get commit SHA via GitHub API: {e}")

    migration_sql = None
    files_changed = None
    if commit_sha:
        migration_raw = f"https://raw.githubusercontent.com/{repo}/migration-scripts/sql/{commit_sha}.sql"
        migration_gh_proxy = (
            f"https://gh-proxy.org/https://raw.githubusercontent.com/{repo}/migration-scripts/sql/{commit_sha}.sql"
        )
        try:
            migration_sql = await asyncio.to_thread(_fetch_url_text, migration_raw, timeout)
        except Exception:
            try:
                migration_sql = await asyncio.to_thread(_fetch_url_text, migration_gh_proxy, timeout)
            except Exception:
                migration_sql = None

        # Fetch commit details to list files_changed
        try:
            commit_api = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
            commit_txt = await asyncio.to_thread(_fetch_url_text, commit_api, timeout)
            commit_json = json.loads(commit_txt)
            files_changed = [f.get("filename") for f in commit_json.get("files", [])]
        except Exception:
            files_changed = None

    else:
        files_changed = None

    # write update_available.json with details
    update_info = {
        "local_version": local_ver,
        "remote_version": remote_ver,
        "schema_changed": schema_changed,
        "commit_sha": commit_sha,
        "files_changed": files_changed,
        "update_time": int(time.time()),
        "remote_manifest": remote,
    }
    try:
        with open(os.path.join(plugin_dir, "update_available.json"), "w", encoding="utf-8") as f:
            ujson.dump(update_info, f)
        print("[INFO] update_available.json written; review before applying update")
    except Exception as e:
        print(f"[WARN] Failed to write update_available.json: {e}")

    # If migration SQL found and db_conn provided, execute it
    migration_applied = False
    if migration_sql and db_conn is not None:
        try:
            print("[INFO] Executing migration script against DB...")
            # run migration inside a transaction
            try:
                async with db_conn.transaction():
                    await db_conn.execute(migration_sql)
            except Exception:
                # if transaction fails, re-raise to outer handler
                raise
            migration_applied = True
            print("[INFO] Migration script applied successfully.")
        except Exception as e:
            print(f"[WARN] Migration execution failed: {e}")

    # If auto_apply True, attempt to pull plugin code (git) and keep backup+restore safety
    backup_dir = None
    try:
        # create backup copy
        backup_dir = f"{plugin_dir}_backup_{int(time.time())}"
        await asyncio.to_thread(shutil.copytree, plugin_dir, backup_dir)
        print(f"[INFO] Backup created at: {backup_dir}")
    except Exception as e:
        print(f"[WARN] Failed to create backup before updating: {e}")
        backup_dir = None

    git_ok = await asyncio.to_thread(_run_subprocess, ["git", "--version"])
    if git_ok.returncode != 0:
        print("[WARN] git not found on system; cannot auto-update plugin")
        return

    try:
        fetch = await asyncio.to_thread(_run_subprocess, ["git", "-C", plugin_dir, "fetch"])
        if fetch.returncode != 0:
            raise RuntimeError(f"git fetch failed: {fetch.stderr}")
        pull = await asyncio.to_thread(_run_subprocess, ["git", "-C", plugin_dir, "pull", "--ff-only"])
        if pull.returncode != 0:
            raise RuntimeError(f"git pull failed: {pull.stderr}")
        print(f"[INFO] Plugin files updated via git pull (rel->local) from {local_ver} to {remote_ver}")

        # after pulling, if migration wasn't applied earlier but migration SQL exists, try applying again
        if migration_sql and not migration_applied and db_conn is not None:
            try:
                async with db_conn.transaction():
                    await db_conn.execute(migration_sql)
                print("[INFO] Migration script applied after pulling successfully.")
            except Exception as e:
                print(f"[WARN] Migration after pull failed: {e}")
                # attempt restore
                if backup_dir and os.path.exists(backup_dir):
                    try:
                        await asyncio.to_thread(lambda: shutil.rmtree(plugin_dir))
                        await asyncio.to_thread(lambda: shutil.copytree(backup_dir, plugin_dir))
                        print("[INFO] Restored plugin from backup due to migration failure.")
                    except Exception as re:
                        print(f"[WARN] Failed to restore plugin from backup: {re}")
        # Attempt to hot-reload the plugin via plugin management API if available
        try:
            if _pmw is not None:
                try:
                    reload_res = await _pmw.reload_plugin("pgvec_mem_plugin")
                    print(f"[INFO] Plugin reload attempted, result: {reload_res}")
                except Exception as rexc:
                    print(f"[WARN] Plugin reload failed: {rexc}")
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] Error while attempting git update: {e}")
        # restore from backup if update fails
        if backup_dir and os.path.exists(backup_dir):
            try:
                await asyncio.to_thread(lambda: shutil.rmtree(plugin_dir))
                await asyncio.to_thread(lambda: shutil.copytree(backup_dir, plugin_dir))
                print(f"[INFO] Restored from backup: {backup_dir} to {plugin_dir}")
            except Exception as restore_e:
                print(f"[WARN] Restore from backup failed: {restore_e}")
    finally:
        # optional: keep backup for manual inspection
        pass


@register_plugin  # 注册插件
class PgVecMemPlugin(BasePlugin):
    plugin_name = "pgvec_mem_plugin"
    enable_plugin = True
    dependencies = []
    python_dependencies = [
        "uuid",
        "dataclasses",
        "typing",
        "asyncpg",
        "numpy",
        "openai",
        "timeit",
        "ujson",
        "cryptography",
        "keyring",
    ]
    config_file_name = "config.toml"  # 配置文件名
    config_schema = {
        "postgresql": {
            "host": ConfigField(
                type=str,
                default="localhost",
                description="PostgreSQL服务器地址",
                hint="你的PostgreSQL服务器地址，通常是localhost或IP地址",
            ),
            "port": ConfigField(
                type=int,
                default=5432,
                description="PostgreSQL服务器端口",
                min=1,
                max=65535,
                hint="你的PostgreSQL服务器端口，默认是5432",
            ),
            "user": ConfigField(
                type=str, default="postgres", description="PostgreSQL用户名", hint="用于连接PostgreSQL数据库的用户名"
            ),
            "password": ConfigField(
                type=str,
                default="yourpassword",
                description="PostgreSQL密码",
                input_type="password",
                hint="用于连接PostgreSQL数据库的密码",
            ),
            "database": ConfigField(
                type=str,
                default="pgvec_maimem_db",
                description="PostgreSQL数据库名",
                hint="用于存储记忆数据的数据库名称",
            ),
            "ssl": ConfigField(
                type=bool,
                default=False,
                description="是否启用SSL连接",
                hint="如果你的PostgreSQL服务器启用了SSL连接，请设置为True",
            ),
        },
        "openai_embedding": {
            "api_key": ConfigField(
                type=str,
                default="sk-xxx",
                description="OpenAI API密钥",
                input_type="password",
                hint="用于访问OpenAI嵌入模型的API密钥，如果在全局配置中已设置，则可以留空",
            ),
            "model": ConfigField(
                type=str,
                default="baai/bge-m3",
                description="OpenAI嵌入模型名称",
                hint="用于生成文本嵌入的OpenAI模型名称，例如：text-embedding-ada-002，baai/bge-m3等，如果在全局配置中已设置，则可以留空",
            ),
            "base_url": ConfigField(
                type=str,
                default="https://openrouter.ai/api/v1",
                description="OpenAI API基础URL",
                hint="如果你使用的是OpenAI的自定义部署或代理，请在此处指定基础URL，如果在全局配置中已设置，则可以留空",
            ),
        },
        "generic_cfg": {
            "cron_interval": ConfigField(
                type=int,
                default=10,
                description="定时任务间隔（分钟）",
                min=1,
                max=1440,
                hint="定时任务执行的时间间隔，单位为分钟",
            ),
            "dropout_rate": ConfigField(
                type=float, default=0.3, description="Dropout率", min=0.0, max=1.0, hint="查询时的Dropout率"
            ),
            "min_edge_weight": ConfigField(
                type=float, default=0.2, description="最小边权重", min=0.0, max=1.0, hint="在图数据库中考虑的最小边权重"
            ),
            "max_depth": ConfigField(
                type=int, default=5, description="最大深度", min=1, max=20, hint="在图数据库中搜索的最大深度"
            ),
            "strengthen_boost": ConfigField(
                type=float, default=0.1, description="强化系数", min=0.0, max=1.0, hint="用于强化记忆连接的系数"
            ),
            "similarity_threshold": ConfigField(
                type=float,
                default=0.75,
                description="相似度阈值",
                min=0.0,
                max=1.0,
                hint="在查询时考虑的最小相似度阈值",
            ),
            "auto_link": ConfigField(
                type=bool, default=True, description="自动链接相邻Topics", hint="是否自动链接相邻的主题以构建记忆图谱"
            ),
            "auto_update": ConfigField(
                type=bool,
                default=False,
                description="自动应用远端更新（谨慎，默认关闭）",
                hint="如果启用，发现远端新版本会在后台备份并尝试 git pull --ff-only 来应用更新",
            ),
        },
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
                    ssl=self.get_config("postgresql.ssl"),
                )
                config = db_mod.MemorySearchConfig(
                    dropout_rate=self.get_config("generic_cfg.dropout_rate"),
                    min_edge_weight=self.get_config("generic_cfg.min_edge_weight"),
                    max_depth=self.get_config("generic_cfg.max_depth"),
                    strengthen_boost=self.get_config("generic_cfg.strengthen_boost"),
                    similarity_threshold=self.get_config("generic_cfg.similarity_threshold"),
                    auto_link=self.get_config("generic_cfg.auto_link"),
                )
                model_burl = ""
                model_id = ""
                model_sk = ""
                if len(list(config_api.get_global_config("model_task_config.embedding.model_list"))) > 0:
                    model_name = list(config_api.get_global_config("model_task_config.embedding.model_list"))[0]
                    model_cfgs = list(config_api.get_global_config("models"))
                    model_provider = ""
                    for md in model_cfgs:
                        if md.get("name") == model_name:
                            model_id = md.get("model_identifier", "")
                            model_provider = md.get("api_provider", "")
                            break
                    if model_provider == "" or model_id == "":
                        model_burl = self.get_config("openai_embedding.base_url")
                        model_sk = self.get_config("openai_embedding.api_key")
                        model_id = self.get_config("openai_embedding.model")
                    else:
                        providers = list(config_api.get_global_config("api_providers"))
                        for pd in providers:
                            if pd.get("name") == model_provider:
                                model_burl = pd.get("base_url", "")
                                model_sk = pd.get("api_key", "")
                                break
                else:
                    model_burl = self.get_config("openai_embedding.base_url")
                    model_id = self.get_config("openai_embedding.model")
                    model_sk = self.get_config("openai_embedding.api_key")
                dbman = db_mod.GraphMemoryDB(
                    model_burl,
                    db_conn,
                    model_sk,
                    model_id,
                )

                # Probe model info asynchronously (non-blocking for plugin init path)
                async def _probe_and_write():
                    try:
                        probe = await db_mod.probe_model_info(model_burl, model_sk, model_id)
                        if probe and probe.get("dimension"):
                            # collect sanitized global config excerpt so CLI can read metadata without contacting global config
                            global_snapshot = {}
                            try:
                                # try to include useful non-secret parts
                                g = config_api.get_global_config()
                                # include model list and models/providers mapping but mask api_key
                                global_snapshot["model_task_config"] = g.get("model_task_config", {})
                                # include models but only identifier/provider
                                global_snapshot["models"] = [
                                    {
                                        "name": m.get("name"),
                                        "model_identifier": m.get("model_identifier"),
                                        "api_provider": m.get("api_provider"),
                                    }
                                    for m in g.get("models", [])
                                ]
                                global_snapshot["api_providers"] = [
                                    {"name": p.get("name"), "base_url": p.get("base_url")}
                                    for p in g.get("api_providers", [])
                                ]
                            except Exception:
                                global_snapshot = {}

                            # Prepare plugin_config_snapshot: encrypt api_key if possible
                            plugin_cfg_snap = {
                                "openai_embedding": {
                                    "model": model_id,
                                    "base_url": model_burl,
                                }
                            }
                            try:
                                if model_sk:
                                    if key_available():
                                        try:
                                            enc = encrypt_for_service(model_sk)
                                            plugin_cfg_snap["openai_embedding"]["api_key_encrypted"] = enc
                                            plugin_cfg_snap["openai_embedding"]["api_key_masked"] = (
                                                model_sk[:4] + "..." + model_sk[-4:]
                                            )
                                        except Exception as e:
                                            # fallback to not storing key
                                            plugin_cfg_snap["openai_embedding"]["api_key"] = None
                                            print(f"[WARN] Failed to encrypt api_key for model_info.json: {e}")
                                    else:
                                        # no key storage available; avoid writing cleartext API key
                                        plugin_cfg_snap["openai_embedding"]["api_key"] = None
                                        print(
                                            "[WARN] secret_store unavailable; model_info.json will not contain api_key"
                                        )
                            except Exception:
                                pass

                            model_info = {
                                "model": probe.get("model"),
                                "dimension": probe.get("dimension"),
                                "raw": probe.get("raw"),
                                "checked_at": int(time.time()),
                                "plugin_config_snapshot": plugin_cfg_snap,
                                "global_config_excerpt": global_snapshot,
                            }
                            try:
                                with open(os.path.join(current_dir, "model_info.json"), "w", encoding="utf-8") as f:
                                    ujson.dump(model_info, f)
                                print(f"[INFO] Wrote model_info.json at plugin init: {current_dir}")
                            except Exception as e:
                                print(f"[WARN] Failed to write model_info.json: {e}")
                        else:
                            print(f"[WARN] probe_model_info returned: {probe}")
                    except Exception as e:
                        print(f"[WARN] Exception while probing model info on plugin init: {e}")

                # Schedule probe task (non-blocking)
                loop.create_task(_probe_and_write())

                # Schedule non-blocking manifest/background update check; pass whether to auto-apply from config
                try:
                    auto_apply = bool(self.get_config("generic_cfg.auto_update"))
                except Exception:
                    auto_apply = False
                loop.create_task(_background_manifest_check(current_dir, db_conn, timeout=10, auto_apply=auto_apply))

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
        result_txt = ""
        for tp, inst in result:
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
