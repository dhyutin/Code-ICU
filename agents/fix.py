#!/usr/bin/env python3
"""Self-healing loop (agent 5) with safe rollback.

propose_fix  -> Conversation & Fix agent proposes a minimal code edit
apply_fix    -> backs up the original, applies the edit, appends changes.txt,
                records a fix_attempts row
spawn_rerun  -> registers a corrected re-run (new runs row, parent_run_id linked)
rollback     -> restores a change from its backup and marks it rolled_back

Every applied change is human-readably logged to changes.txt so you can always
see what was touched and revert it.

CLI:
    python agents/fix.py list
    python agents/fix.py rollback [change_id]
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import trace
from agents.runtime import run_agent_json

BACKUP_DIR = ROOT / "backups"
CHANGES_LOG = ROOT / "changes.txt"
MAX_CHARS = 12000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def propose_fix(error_report: dict, code_context: str, *, file: str, transcript: str | None = None) -> dict:
    """Agent 5 proposes a minimal edit to fix the failure. Returns the fix dict."""
    p = ROOT / file
    source = p.read_text()[:MAX_CHARS]
    prompt = (
        "A training run failed. Propose the SMALLEST code edit that addresses the "
        "root cause. Respond with JSON only, keys: file (string, keep it as given), "
        "edits (array of {find, replace} where `find` is an EXACT substring of the "
        "current file), rationale (one sentence). Prefer one or two edits.\n\n"
        f"File: {file}\n"
        f"Error report: {json.dumps(error_report)}\n"
        f"Code context:\n{code_context}\n"
        + (f"Call transcript: {transcript}\n" if transcript else "")
        + f"\nCurrent file content:\n```python\n{source}\n```"
    )
    fix = run_agent_json("conversation_fix", prompt)
    fix.setdefault("file", file)
    return fix


def apply_fix(fix: dict, run_id: str) -> dict:
    """Back up, apply edits, log to changes.txt, record fix_attempts."""
    file = fix["file"]
    p = ROOT / file
    original = p.read_text()

    edits = fix.get("edits", [])
    updated = original
    applied_edits = []
    for e in edits:
        find, replace = e.get("find"), e.get("replace")
        if find and find in updated:
            updated = updated.replace(find, replace)
            applied_edits.append(e)
    if not applied_edits:
        raise RuntimeError("No proposed edits matched the current file; nothing applied.")

    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = BACKUP_DIR / f"{file.replace('/', '_')}.{ts}.bak"
    shutil.copy2(p, backup)
    p.write_text(updated)

    run = trace.resolve_run(run_id)
    row = trace._insert(
        "fix_attempts",
        {
            "run_ref": run["id"],
            "run_id": run_id,
            "parent_run_id": run["id"],
            "file": file,
            "rationale": fix.get("rationale", ""),
            "patch": {"edits": applied_edits},
            "backup_path": str(backup.relative_to(ROOT)),
            "status": "applied",
        },
    )

    _append_log(
        f"[{_now()}] APPLIED  change_id={row.get('id')}  run={run_id}  file={file}\n"
        f"  reason: {fix.get('rationale', '')}\n"
        f"  backup: {backup.relative_to(ROOT)}\n"
        + "".join(f"  edit: {e['find']!r} -> {e['replace']!r}\n" for e in applied_edits)
    )
    trace.log_event(run_id, "conversation_fix", "note", f"Applied fix to {file}: {fix.get('rationale', '')}")
    return {"change_id": row.get("id"), "backup": str(backup.relative_to(ROOT)), "edits": applied_edits, "file": file}


def spawn_rerun(run_id: str, fix_change_id: str | None = None) -> str:
    """Register a corrected re-run linked to the original via parent_run_id."""
    parent = trace.resolve_run(run_id)
    new_run_id = f"{run_id}-fix-{datetime.now().strftime('%H%M%S')}"
    trace._insert(
        "runs",
        {
            "user_id": parent["user_id"],
            "name": new_run_id,
            "status": "queued",
            "parent_run_id": parent["id"],
        },
    )
    if fix_change_id:
        trace.patch("fix_attempts", fix_change_id, {"new_run_id": new_run_id})
    _append_log(f"  re-run registered: {new_run_id} (parent {run_id})\n")
    trace.log_event(run_id, "conversation_fix", "note", f"Corrected re-run registered: {new_run_id}")
    return new_run_id


def rollback(change_id: str | None = None) -> dict:
    """Restore a change from its backup. Defaults to the latest applied change."""
    if change_id:
        rows = trace._get("fix_attempts", {"id": f"eq.{change_id}", "limit": 1})
    else:
        rows = trace._get("fix_attempts", {"status": "eq.applied", "order": "created_at.desc", "limit": 1})
    if not rows:
        raise RuntimeError("No applied change found to roll back.")
    row = rows[0]

    backup = ROOT / row["backup_path"]
    target = ROOT / row["file"]
    if not backup.exists():
        raise RuntimeError(f"Backup missing: {backup}")
    shutil.copy2(backup, target)

    trace.patch("fix_attempts", row["id"], {"status": "rolled_back"})
    _append_log(
        f"[{_now()}] ROLLBACK change_id={row['id']}  file={row['file']}  "
        f"restored from {row['backup_path']}\n"
    )
    print(f"Rolled back {row['file']} from {row['backup_path']}")
    return row


def _append_log(text: str) -> None:
    with CHANGES_LOG.open("a") as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))


def _cli() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "rollback"):
        print(__doc__)
        return
    if sys.argv[1] == "list":
        rows = trace._get("fix_attempts", {"order": "created_at.desc", "limit": 50})
        if not rows:
            print("No fix attempts yet.")
            return
        for r in rows:
            print(f"{r['created_at']}  {r['id']}  {r['status']:11}  {r['file']}  -> {r.get('new_run_id') or '-'}")
            print(f"    {r.get('rationale', '')}")
    elif sys.argv[1] == "rollback":
        rollback(sys.argv[2] if len(sys.argv) > 2 else None)


if __name__ == "__main__":
    _cli()
