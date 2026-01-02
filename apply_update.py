"""
Manual/automatic apply-update helper.
Reads update_available.json and, optionally, applies update by running git pull and migration.
Usage:
  python apply_update.py [--auto] [--plugin-dir PATH]

If --auto is set, the script will attempt to git pull and apply migration (if available), otherwise it will just print the update info.
The script expects to be run from repository root or pass plugin-dir explicitly.
"""

import argparse
import os
import ujson
import subprocess
import shutil
import time
import asyncio

# try to import plugin management wrapper to support hot-reload after apply
_pmw = None
try:
    from . import plugin_manage_wrapper as _pmw
except Exception:
    try:
        import plugin_manage_wrapper as _pmw
    except Exception:
        _pmw = None

PLUGIN_NAME = "MaiVecMem"
DEFAULT_PLUGIN_DIR = os.path.join(os.path.dirname(__file__))


def run(cmd, cwd=None):
    print(f"RUN: {' '.join(cmd)} (cwd={cwd})")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Attempt to auto-apply the update")
    parser.add_argument("--plugin-dir", default=DEFAULT_PLUGIN_DIR, help="Path to plugin directory")
    args = parser.parse_args()

    plugin_dir = args.plugin_dir
    ua_path = os.path.join(plugin_dir, "update_available.json")
    if not os.path.exists(ua_path):
        print("No update_available.json found. Nothing to do.")
        return

    info = ujson.load(open(ua_path, "r", encoding="utf-8"))
    print("Update info:")
    print(ujson.dumps(info, indent=2, ensure_ascii=False))

    if not args.auto:
        print("Run with --auto to apply the update")
        return

    # attempt backup
    backup_dir = f"{plugin_dir}_apply_backup_{int(time.time())}"
    try:
        shutil.copytree(plugin_dir, backup_dir)
        print(f"Backup created at {backup_dir}")
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return

    # try git pull
    try:
        run(["git", "-C", plugin_dir, "fetch"])
        run(["git", "-C", plugin_dir, "pull", "--ff-only"])
    except Exception as e:
        print(f"git pull failed: {e}")
        print("Attempting to restore from backup")
        try:
            shutil.rmtree(plugin_dir)
            shutil.copytree(backup_dir, plugin_dir)
            print("Restored from backup")
        except Exception as re:
            print(f"Restore failed: {re}")
        return

    # attempt hot-reload of plugin via plugin_manage_wrapper if available
    if _pmw is not None:
        try:
            print("[INFO] Attempting to hot-reload plugin 'pgvec_mem_plugin'...")
            reload_result = asyncio.run(_pmw.reload_plugin("pgvec_mem_plugin"))
            print(f"[INFO] reload_plugin result: {reload_result}")
        except Exception as e:
            print(f"[WARN] plugin reload failed: {e}")
    else:
        print("[INFO] plugin_manage_wrapper not available; please reload plugin manually if needed.")

    # attempt to apply migration if file exists
    commit_sha = info.get("commit_sha")
    if commit_sha:
        migration_file = os.path.join(plugin_dir, "migration-scripts", "sql", f"{commit_sha}.sql")
        if os.path.exists(migration_file):
            print(f"Applying migration: {migration_file}")
            # run psql - but we don't know credentials here; expect environment or user will run migration
            # For safety, we just print path and return success
            print(
                "Migration file present. Please apply it with proper DB credentials (psql) or enable auto_apply in plugin config to let plugin do it."
            )
        else:
            print("No migration file found in plugin dir; plugin may fetch remote migration during background check.")

    print("Update applied (files pulled). If plugin requires reload, use plugin_manage_api to reload or restart host.")


if __name__ == "__main__":
    main()
