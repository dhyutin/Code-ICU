#!/usr/bin/env python3
"""Phase 10 — decision dataset export.

Turns the judgment calls code_icu has logged into fine-tune-ready supervised
data for two future autonomous agents:

  triage.jsonl  — anomaly context  -> decision (kill | pause | continue | fix)
  fix.jsonl     — failing code + error -> JSON edits that repair it

Both are written in OpenAI/Nebius chat fine-tuning format
({"messages": [system, user, assistant]}). A raw `records.jsonl` with the full
joined rows is also written for analysis.

Reads everything through the service key (so it sees all runs), joining
call_decisions + fix_attempts with run_logs (recent metrics) and agent_events
(error-detection severity / code context).

    python agents/export_dataset.py                       # -> dataset/
    python agents/export_dataset.py --out data            # custom output dir
    python agents/export_dataset.py --fine-tune triage    # export, then start a Nebius job
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import trace

DECISION_CLASSES = {"kill", "pause", "continue", "fix"}

TRIAGE_SYSTEM = (
    "You are code_icu's triage agent. Given an ML training anomaly and the recent "
    "metric history, respond with exactly one word — one of: kill, pause, continue, fix."
)
FIX_SYSTEM = (
    "You are code_icu's fix agent. Given a failing training file and the error, "
    "respond with JSON only: {\"edits\": [{\"find\": <exact substring>, \"replace\": "
    "<new code>}], \"rationale\": <one sentence>}. Make the smallest edit that fixes "
    "the root cause."
)


def _recent_logs(run_id: str, upto_step: int | None, n: int = 15) -> list[dict]:
    rows = trace._get(
        "run_logs",
        {"run_id": f"eq.{run_id}", "order": "step.desc", "limit": 60},
    )
    rows = list(reversed(rows))
    if upto_step is not None:
        rows = [r for r in rows if r.get("step") is not None and r["step"] <= upto_step]
    return rows[-n:]


def _fmt_logs(rows: list[dict]) -> str:
    if not rows:
        return "(no metrics)"
    return "\n".join(
        f"  step {r['step']:>4}  loss={r['loss']}  grad_norm={r.get('grad_norm')}  lr={r.get('lr')}"
        for r in rows
    )


def _error_detection_index() -> dict[str, dict]:
    """run_id -> parsed latest error_detection output (severity, error_type, ...)."""
    out: dict[str, dict] = {}
    events = trace._get(
        "agent_events",
        {"agent_key": "eq.error_detection", "kind": "eq.output", "order": "created_at.asc", "limit": 1000},
    )
    for e in events:
        try:
            out[e.get("run_id")] = json.loads(e.get("content") or "{}")
        except Exception:
            pass
    return out


def _code_context() -> str:
    try:
        from agents.code_context import context_to_text, load_cached

        return context_to_text(load_cached() or {})
    except Exception:
        return ""


def build_triage(decisions: list[dict], err_idx: dict[str, dict], ctx: str) -> list[dict]:
    examples = []
    for d in decisions:
        decision = (d.get("decision") or "").strip().lower()
        if decision not in DECISION_CLASSES:
            continue
        run_id = d.get("run_id")
        report = err_idx.get(run_id, {})
        logs = _recent_logs(run_id, d.get("anomaly_step"))
        user = (
            f"Run: {run_id}\n"
            f"Error type: {d.get('anomaly_type')}\n"
            f"Anomaly step: {d.get('anomaly_step')}\n"
            f"Severity: {report.get('severity', 'unknown')}\n"
            f"Evidence: {report.get('evidence', report.get('summary', '')) or 'n/a'}\n"
            f"Recent metrics:\n{_fmt_logs(logs)}\n"
            + (f"\nCode context:\n{ctx[:1500]}\n" if ctx else "")
            + "\nDecision?"
        )
        examples.append({
            "messages": [
                {"role": "system", "content": TRIAGE_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": decision},
            ]
        })
    return examples


def build_fix(fixes: list[dict], ctx: str) -> list[dict]:
    examples = []
    for f in fixes:
        if (f.get("status") or "").lower() not in {"applied", "rolled_back"}:
            continue
        patch = f.get("patch") or {}
        edits = patch.get("edits") if isinstance(patch, dict) else None
        if not edits:
            continue
        err = patch.get("error", {}) if isinstance(patch, dict) else {}
        buggy = "\n".join(f"  {e.get('find')}" for e in edits)
        user = (
            f"File: {f.get('file')}\n"
            f"Error type: {err.get('type', 'unknown')}\n"
            f"Error: {err.get('brief', '') or 'n/a'}\n"
            f"Offending code:\n{buggy}\n"
            + (f"\nCode context:\n{ctx[:1500]}\n" if ctx else "")
            + "\nPropose the fix as JSON edits."
        )
        assistant = json.dumps({
            "edits": [{"find": e.get("find"), "replace": e.get("replace")} for e in edits],
            "rationale": f.get("rationale", ""),
        })
        examples.append({
            "messages": [
                {"role": "system", "content": FIX_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })
    return examples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


MIN_FINE_TUNE_EXAMPLES = 10


def start_fine_tune(path: Path, count: int) -> None:
    """Upload a JSONL and kick off an OpenAI-compatible fine-tune job on Nebius."""
    from agents.nebius_client import NEBIUS_MODEL, get_client

    if count == 0:
        print(f"  [fine-tune] {path.name} is empty — nothing to train on. Skipping.")
        return
    if count < MIN_FINE_TUNE_EXAMPLES:
        print(
            f"  [fine-tune] WARNING: only {count} examples in {path.name}; most fine-tune "
            f"endpoints require >= {MIN_FINE_TUNE_EXAMPLES}. Submitting anyway — it may be rejected."
        )

    client = get_client()
    print(f"  [fine-tune] uploading {path.name} ({count} examples)...")
    f = client.files.create(file=open(path, "rb"), purpose="fine-tune")
    print(f"  [fine-tune] file id: {f.id}")
    job = client.fine_tuning.jobs.create(training_file=f.id, model=NEBIUS_MODEL)
    print(f"  [fine-tune] job started: {job.id}  (base model: {NEBIUS_MODEL})")
    print(f"  [fine-tune] check status: client.fine_tuning.jobs.retrieve('{job.id}')")
    print(
        "  [fine-tune] when it finishes, point the agent's `model` (its row in the InsForge "
        "`agents` table) at the tuned model id — no code change."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset", help="output directory")
    ap.add_argument(
        "--fine-tune",
        choices=["triage", "fix"],
        help="after exporting, upload that dataset and start a Nebius fine-tune job",
    )
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    decisions = trace._get("call_decisions", {"order": "created_at.asc", "limit": 1000})
    fixes = trace._get("fix_attempts", {"order": "created_at.asc", "limit": 1000})
    err_idx = _error_detection_index()
    ctx = _code_context()

    triage = build_triage(decisions, err_idx, ctx)
    fix = build_fix(fixes, ctx)

    _write_jsonl(out_dir / "triage.jsonl", triage)
    _write_jsonl(out_dir / "fix.jsonl", fix)
    _write_jsonl(out_dir / "records.jsonl", [
        {"kind": "decision", **d} for d in decisions
    ] + [
        {"kind": "fix", **f} for f in fixes
    ])

    stats = {
        "decisions_total": len(decisions),
        "triage_examples": len(triage),
        "fix_examples": len(fix),
        "decision_breakdown": dict(Counter(
            (d.get("decision") or "").strip().lower() for d in decisions
        )),
        "error_breakdown": dict(Counter(d.get("anomaly_type") for d in decisions)),
        "fix_status_breakdown": dict(Counter((f.get("status") or "").lower() for f in fixes)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"Exported to {out_dir.relative_to(ROOT)}/")
    print(f"  triage.jsonl : {len(triage)} examples")
    print(f"  fix.jsonl    : {len(fix)} examples")
    print(f"  records.jsonl: {len(decisions) + len(fixes)} raw rows")
    print(json.dumps(stats, indent=2))

    if args.fine_tune:
        which = args.fine_tune
        counts = {"triage": len(triage), "fix": len(fix)}
        print(f"\nStarting fine-tune on {which}.jsonl...")
        start_fine_tune(out_dir / f"{which}.jsonl", counts[which])


if __name__ == "__main__":
    main()
