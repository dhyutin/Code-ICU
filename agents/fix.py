#!/usr/bin/env python3
"""Self-healing loop (agent 5) with safe rollback.

propose_fix  -> Conversation & Fix agent proposes a minimal code edit
apply_fix    -> backs up the original, applies the edit, logs the change,
                records a fix_attempts row
spawn_rerun  -> registers a corrected re-run (new runs row, parent_run_id linked)
                and launches it
rollback     -> restores a change from its backup and marks it rolled_back

Each run gets its own folder under `runs/<run_id>/` holding that run's
`changes.txt`, `backups/`, and (for a launched re-run) `rerun.log` — so every
demo run's history lives in one place and any change is reversible.

CLI:
    python agents/fix.py list
    python agents/fix.py rollback [change_id]
    python agents/fix.py watch        # apply restore requests from the dashboard
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import trace
from agents.runtime import run_agent_json

RUNS_DIR = ROOT / "runs"        # per-run artifact folders
MAX_CHARS = 12000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(run_id: str) -> str:
    return run_id.replace("/", "_")


def run_dir(run_id: str) -> Path:
    """Return (and create) the artifact folder for a run."""
    d = RUNS_DIR / _safe(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_log(run_id: str, text: str) -> None:
    """Append a human-readable entry to that run's own changes.txt."""
    log = run_dir(run_id) / "changes.txt"
    with log.open("a") as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))


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


def apply_fix(fix: dict, run_id: str, error: dict | None = None) -> dict:
    """Back up, apply edits, log to changes.txt, record fix_attempts.

    `error` ({type, brief}) is stored alongside the patch so the dashboard can
    show exactly which failure each change was made for.
    """
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

    backups = run_dir(run_id) / "backups"
    backups.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = backups / f"{_safe(file)}.{ts}.bak"
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
            "patch": {"edits": applied_edits, "error": error or {}},
            "backup_path": str(backup.relative_to(ROOT)),
            "status": "applied",
        },
    )

    _append_log(
        run_id,
        f"[{_now()}] APPLIED  change_id={row.get('id')}  file={file}\n"
        f"  reason: {fix.get('rationale', '')}\n"
        f"  backup: {backup.relative_to(ROOT)}\n"
        + "".join(f"  edit: {e['find']!r} -> {e['replace']!r}\n" for e in applied_edits),
    )
    trace.log_event(run_id, "conversation_fix", "note", f"Applied fix to {file}: {fix.get('rationale', '')}")
    return {"change_id": row.get("id"), "backup": str(backup.relative_to(ROOT)), "edits": applied_edits, "file": file}


def spawn_rerun(run_id: str, fix_change_id: str | None = None, *, file: str | None = None,
                launch: bool = True) -> str:
    """Register a corrected re-run (linked via parent_run_id) and actually launch it.

    The corrected `file` is run as a detached subprocess with CODE_ICU_RUN_ID /
    CODE_ICU_RUN_REF set, so it trains under the new run id and streams live to the
    dashboard. Set env CODE_ICU_AUTORUN=0 (or launch=False) to only register.
    """
    parent = trace.resolve_run(run_id)
    new_run_id = f"{run_id}-fix-{datetime.now().strftime('%H%M%S')}"
    new_run = trace._insert(
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
    # log the lineage on both ends: parent points forward, child points back
    _append_log(run_id, f"[{_now()}] DIRECTED TO NEXT -> {new_run_id} (corrected re-run)")
    _append_log(new_run_id, f"[{_now()}] CORRECTED RE-RUN of {run_id} (fix change_id={fix_change_id})")
    trace.log_event(run_id, "conversation_fix", "note", f"Corrected re-run registered: {new_run_id}")

    autorun = os.getenv("CODE_ICU_AUTORUN", "1") != "0"
    if launch and file and autorun:
        try:
            _launch_rerun(file, new_run_id, new_run.get("id"))
            trace.patch("runs", new_run["id"], {"status": "running"})
            trace.log_event(run_id, "conversation_fix", "note",
                            f"Corrected re-run launched: python {file} as {new_run_id}")
        except Exception as exc:
            print(f"  [agent 5] re-run launch failed: {exc}", flush=True)
            _append_log(new_run_id, f"[{_now()}] LAUNCH FAILED: {exc}")
    return new_run_id


def _launch_rerun(file: str, new_run_id: str, run_ref: str | None) -> None:
    """Run the corrected training file as a detached process under the new run id."""
    target = ROOT / file
    if not target.exists():
        raise FileNotFoundError(f"cannot launch re-run: {target} not found")
    log_path = run_dir(new_run_id) / "rerun.log"

    env = {**os.environ, "CODE_ICU_RUN_ID": new_run_id}
    if run_ref:
        env["CODE_ICU_RUN_REF"] = str(run_ref)

    log_file = open(log_path, "w")
    subprocess.Popen(
        [sys.executable, str(target)],
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # survive after the monitor exits
    )
    _append_log(new_run_id, f"[{_now()}] LAUNCHED: python {file} (log: {log_path.relative_to(ROOT)})")
    print(f"  [agent 5] re-run LAUNCHED: python {file} as {new_run_id}", flush=True)
    print(f"  [agent 5] watch it live on the dashboard (run {new_run_id})", flush=True)


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
        row.get("run_id", "unknown"),
        f"[{_now()}] ROLLBACK change_id={row['id']}  file={row['file']}  "
        f"restored from {row['backup_path']}",
    )
    print(f"Rolled back {row['file']} from {row['backup_path']}")
    return row


def watch(interval: int = 4) -> None:
    """Poll for dashboard-requested restores and perform the actual file rollback.

    The browser can't touch the local filesystem, so a 'Restore' click only flips
    the row to `rollback_requested`; this loop does the real restore + marks it
    `rolled_back`. Leave it running during a demo.
    """
    print(f"[fix-watch] polling for restore requests every {interval}s (Ctrl-C to stop)", flush=True)
    while True:
        try:
            pending = trace._get(
                "fix_attempts",
                {"status": "eq.rollback_requested", "order": "updated_at.asc", "limit": 10},
            )
            for row in pending:
                try:
                    rollback(row["id"])
                except Exception as exc:
                    trace.patch("fix_attempts", row["id"], {"status": "failed"})
                    print(f"  [fix-watch] rollback {row['id']} failed: {exc}", flush=True)
        except Exception as exc:
            print(f"  [fix-watch] poll error: {exc}", flush=True)
        time.sleep(interval)


def _cli() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "rollback", "watch"):
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
    elif sys.argv[1] == "watch":
        watch()


if __name__ == "__main__":
    _cli()
