"""Link existing demo run_logs / call_decisions to a user-owned run.

Runs tools/backfill_runs.sql via the InsForge CLI. Using a subprocess argv
(not a shell string) keeps Postgres dollar-quoting ($$) intact.

Usage:
    python tools/backfill_runs.py

Idempotent — safe to re-run after each demo to link new runs.
"""

import pathlib
import subprocess
import sys

SQL_FILE = pathlib.Path(__file__).with_name("backfill_runs.sql")


def main() -> int:
    sql = SQL_FILE.read_text()
    # `--` stops option parsing so the SQL (which starts with `--`, a comment)
    # isn't mistaken for a CLI flag.
    cmd = ["npx", "@insforge/cli", "db", "query", "--unrestricted", "--", sql]
    print(f"Running backfill from {SQL_FILE.name} ...")
    result = subprocess.run(cmd, cwd=SQL_FILE.parent.parent)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
