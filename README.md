MaiVecMem plugin — developer notes

What this plugin adds
- Probes embedding model on plugin init and writes `model_info.json` with metadata (model, dimension, plugin_config_snapshot, global_config_excerpt).
- Background update checker that compares `rel/_manifest.json` and `schema.sql`, fetches migration scripts from `migration-scripts/sql/<commit>.sql`, and writes `update_available.json`.
- Optional `generic_cfg.auto_update` to auto-apply updates (disabled by default). Auto-apply backs up plugin dir, pulls, attempts migration, and tries hot-reload.
- CLI `plugins/MaiVecMem/cli_tool.py` reads `model_info.json` and generates `schema.generated.sql` used to initialize DB.
- `apply_update.py` helper script to apply updates manually (with `--auto` to apply).
- `iflow_action_generate_migration.py` calls `iflow --yolo` to let the model write migration SQL and a marker file; the script then commits/pushes/merges as instructed by the marker.

Quick usage

1) Let plugin probe and write model info (plugin init in host application)
   - Ensure your host loads the plugin so `plugin.get_plugin_components()` runs.
   - After probe, check `plugins/MaiVecMem/model_info.json`.

2) Generate schema and init DB (CLI)
   - The CLI reads `model_info.json` and writes `schema.generated.sql`:
     ```powershell
     python .\plugins\MaiVecMem\cli_tool.py init
     ```
   - If you want to apply changes without DB execution, see `tests/test_schema_generation.py`.

3) Create migration using iflow
   - Make sure `iflow` is installed and accessible in PATH and model credentials are set (or present in `model_info.json`).
   - Run:
     ```powershell
     python .\plugins\MaiVecMem\iflow_action_generate_migration.py --desc "add new column"
     ```
   - The model will write files under the plugin directory and create `migration_action.json` to instruct the script what to commit/push.

4) Apply updates (manual)
   - Inspect `plugins/MaiVecMem/update_available.json` then run:
     ```powershell
     python .\plugins\MaiVecMem\apply_update.py --auto --plugin-dir .\plugins\MaiVecMem
     ```

Developer test: generate schema without Postgres
- There's a lightweight test harness: `tests/test_schema_generation.py` which creates a fake `model_info.json` and uses a fake DB connection to test schema generation and writing `schema.generated.sql`.

Security notes
- `model_info.json` currently contains `api_key` in `plugin_config_snapshot.openai_embedding` so CLI tools can reuse it. If you prefer to avoid storing secrets on disk, set OPENAI_API_KEY in the environment and remove the key from `model_info.json`.

Files to review
- `plugins/MaiVecMem/plugin.py` (probe + background update logic)
- `plugins/MaiVecMem/cli_tool.py` (CLI init reads model_info.json)
- `plugins/MaiVecMem/apply_update.py` (apply helper + hot-reload)
- `plugins/MaiVecMem/iflow_action_generate_migration.py` (iflow-driven migration generator)

If you want, I can add an automated unit test suite (pytest) that mocks network, OpenAI, and git interactions.

