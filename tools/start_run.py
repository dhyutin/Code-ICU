#!/usr/bin/env python3
"""Pre-register an owned run so a training script's logs are RLS-visible on the
dashboard *live* (not just after a backfill).

It creates a `runs` row for your profile and prints shell `export`s for
CODE_ICU_RUN_ID / CODE_ICU_RUN_REF, which log_pusher + the example scripts honor.

Usage:
    eval "$(python tools/start_run.py text-fail)"
    python examples/text_sentiment_fail.py      # now linked + visible live
    # copy the printed RUN_ID into the monitor in another terminal
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import trace


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    run_id = f"{name}-{datetime.now().strftime('%H%M%S')}"

    profiles = trace._get("profiles", {"limit": 1})
    if not profiles:
        sys.stderr.write("No profile found — sign into the dashboard once (Google) first.\n")
        sys.exit(1)

    row = trace._insert("runs", {"user_id": profiles[0]["user_id"], "name": run_id, "status": "running"})
    print(f"export CODE_ICU_RUN_ID={run_id}")
    print(f"export CODE_ICU_RUN_REF={row['id']}")
    sys.stderr.write(
        f"Registered run {run_id} (ref {row['id']}).\n"
        f"Run your training script in THIS shell, and use RUN_ID={run_id} for the monitor.\n"
    )


if __name__ == "__main__":
    main()
