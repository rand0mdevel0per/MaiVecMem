"""
Automated iflow CLI action script.

Behavior:
1) Call `iflow --yolo -p <prompt>` and instruct the model to:
   - write a migration SQL file under `migration-scripts/sql/<id>.sql`
   - optionally update `_manifest.json` to a new version
   - write a JSON marker `migration_action.json` in the plugin dir with the following keys:
       {
         "migration_file": "migration-scripts/sql/<id>.sql",
         "commit_message": "...",
         "bump_version": true|false,
         "new_version": "x.y.z" or null,
         "merge_rel": true|false
       }
   The model has full file/terminal access under --yolo and should perform file writes itself.

2) After iflow returns, this script reads `migration_action.json` (if present) and commits/pushes/merges accordingly.

3) If the marker is missing, fallback to capturing stdout as the migration SQL and behave as previous implementation.
"""
import os
import subprocess
import time
import ujson

PLUGIN_DIR = os.path.dirname(__file__)
MODEL_INFO = os.path.join(PLUGIN_DIR, "model_info.json")
MANIFEST = os.path.join(PLUGIN_DIR, "_manifest.json")
ACTION_MARKER = os.path.join(PLUGIN_DIR, "migration_action.json")


def run(cmd, cwd=None, env=None):
    print("RUN:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("ERR:", r.stderr)
        raise RuntimeError("Command failed: " + " ".join(cmd))
    return r


def build_iflow_prompt_for_file_write(change_desc: str, files_changed=None) -> str:
    files_section = ""
    if files_changed:
        files_section = "\nFiles changed: " + ", ".join(files_changed) + "\n"

    # The prompt instructs the model in --yolo mode to write files directly in the repository.
    prompt = f"""
You are an expert PostgreSQL DBA and release engineer with direct file and terminal access.
You will perform the following actions by writing files under the current working directory.

Primary goal: produce a single PostgreSQL migration SQL file and, if appropriate, update the plugin manifest and request a rel release.

Requirements (very strict):
- Create a migration SQL file at 'migration-scripts/sql/<id>.sql'. The file must contain only valid PostgreSQL SQL statements.
- Wrap the migration in a transaction using BEGIN; and COMMIT;.
- Do not print any SQL or other content to stdout besides diagnostic logs. Instead, write the files to disk.
- After writing files, create a JSON marker file at '{os.path.basename(ACTION_MARKER)}' (in plugin root) with the following structure:
  {{
    "migration_file": "migration-scripts/sql/<id>.sql",
    "commit_message": "Add migration <id> - <short description>",
    "bump_version": true|false,
    "new_version": "x.y.z" | null,
    "merge_rel": true|false
  }}
- If you decide NOT to create a migration, write migration_file as null and set appropriate fields.
- If you choose to bump version, update '_manifest.json' on disk to the new version string (exact JSON update). Otherwise don't modify it.
- Use conservative, reversible SQL where possible (IF NOT EXISTS / IF EXISTS).
- This is running in --yolo mode so you may create directories and files under the repository.

Change description: {change_desc}
{files_section}

Write the migration SQL file and the marker file exactly as specified. Do not ask for confirmation.
"""
    return prompt


def fallback_generate_stub(change_desc: str) -> str:
    ts = int(time.time())
    return f"-- migration {ts}\n-- desc: {change_desc}\nBEGIN;\n-- add your ALTER TABLE / data migrations here\nCOMMIT;\n"


def git_commit_and_push(files: list, branch: str = "migration-scripts", commit_message: str = None):
    # ensure branch
    run(["git", "-C", PLUGIN_DIR, "checkout", "-B", branch])
    if files:
        run(["git", "-C", PLUGIN_DIR, "add"] + files)
    else:
        # add all changes if no specific files listed
        run(["git", "-C", PLUGIN_DIR, "add", "-A"])
    if not commit_message:
        commit_message = f"Update by iflow action {int(time.time())}"
    run(["git", "-C", PLUGIN_DIR, "commit", "-m", commit_message])
    run(["git", "-C", PLUGIN_DIR, "push", "--set-upstream", "origin", branch])


def git_merge_to_rel(from_branch: str = "migration-scripts"):
    run(["git", "-C", PLUGIN_DIR, "checkout", "rel"])
    run(["git", "-C", PLUGIN_DIR, "merge", "--no-ff", from_branch, "-m", f"merge {from_branch} into rel"])
    run(["git", "-C", PLUGIN_DIR, "push", "origin", "rel"])


def main(change_desc: str = "auto-migration"):
    prompt = build_iflow_prompt_for_file_write(change_desc)

    # Run iflow in --yolo mode and instruct it to write files to disk
    iflow_cmd = ["iflow", "--yolo", "-p", prompt]
    print("Invoking iflow --yolo to generate migration and action marker (model will write files)")
    proc = subprocess.run(iflow_cmd, cwd=PLUGIN_DIR, env=os.env, capture_output=True, text=True)
    print("iflow stdout:")
    print(proc.stdout)
    if proc.returncode != 0:
        print("iflow stderr:")
        print(proc.stderr)
        print("iflow returned non-zero; falling back to generating stub migration and marker")
        # write fallback migration and marker
        os.makedirs(os.path.join(PLUGIN_DIR, "migration-scripts", "sql"), exist_ok=True)
        commit_id = str(int(time.time()))
        mig_path = os.path.join("migration-scripts", "sql", f"{commit_id}.sql")
        with open(os.path.join(PLUGIN_DIR, mig_path), "w", encoding="utf-8") as f:
            f.write(fallback_generate_stub(change_desc))
        marker = {
            "migration_file": mig_path,
            "commit_message": f"Add migration {commit_id} - {change_desc}",
            "bump_version": False,
            "new_version": None,
            "merge_rel": False,
        }
        ujson.dump(marker, open(ACTION_MARKER, "w", encoding="utf-8"))

    # After iflow returns, read the marker file if present
    if os.path.exists(ACTION_MARKER):
        try:
            marker = ujson.load(open(ACTION_MARKER, "r", encoding="utf-8"))
        except Exception as e:
            print(f"Failed to read marker file: {e}")
            marker = None
    else:
        marker = None

    # If marker present, use it to drive git operations
    if marker:
        migration_file = marker.get("migration_file")
        commit_message = marker.get("commit_message") or f"Add migration - {int(time.time())}"
        bump_version = bool(marker.get("bump_version"))
        marker.get("new_version")
        merge_rel = bool(marker.get("merge_rel"))

        files_to_commit = []
        if migration_file:
            files_to_commit.append(migration_file)
        # If manifest was modified (if model wrote new manifest), include it
        if bump_version and os.path.exists(MANIFEST):
            files_to_commit.append(os.path.relpath(MANIFEST, PLUGIN_DIR))

        try:
            git_commit_and_push(files_to_commit, branch="migration-scripts", commit_message=commit_message)
        except Exception as e:
            print(f"Git commit/push failed: {e}")
            return

        if merge_rel:
            try:
                git_merge_to_rel(from_branch="migration-scripts")
            except Exception as e:
                print(f"Git merge to rel failed: {e}")
                return

        print("Action marker applied successfully.")
        return

    # Fallback: if no marker, attempt earlier behavior: take stdout as SQL
    out = proc.stdout.strip()
    if out:
        migration_sql = out
    else:
        migration_sql = fallback_generate_stub(change_desc)

    # Write migration file and commit as before
    os.makedirs(os.path.join(PLUGIN_DIR, "migration-scripts", "sql"), exist_ok=True)
    commit_id = str(int(time.time()))
    target_path = os.path.join("migration-scripts", "sql", f"{commit_id}.sql")
    with open(os.path.join(PLUGIN_DIR, target_path), "w", encoding="utf-8") as f:
        f.write(migration_sql)

    try:
        git_commit_and_push([target_path], branch="migration-scripts", commit_message=f"Add migration {commit_id} - {change_desc}")
        print("Migration published using fallback stdout path.")
    except Exception as e:
        print(f"Failed to publish migration: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str, default="auto-migration")
    args = parser.parse_args()
    main(change_desc=args.desc)
