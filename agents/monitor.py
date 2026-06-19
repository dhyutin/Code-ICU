#!/usr/bin/env python3
"""Agent-driven monitor (Phase 6-7).

Pipeline per anomaly:
    heuristic flag  -> Error Detection agent (3)  -> structured report
                    -> Call Decision agent (4)    -> {call, reason}
                    -> if call: Vapi outbound (Conversation & Fix agent, 5)
                    -> transcript -> decision -> InsForge call_decisions

Run:  python agents/monitor.py <RUN_ID>

The Conversation & Fix agent prompt is configured on the Vapi assistant via
agents/sync_vapi.py. Writes use the service key (bypasses RLS).
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from anomaly_detector import check_anomaly
from agents import fix as fix_mod
from agents import trace
from agents.runtime import decide_call, detect_error
from agents.code_context import context_to_text, load_cached
from agents.nebius_client import extract_decision

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "demo-run-001"
POLL_INTERVAL = 15
CALL_POLL_INTERVAL = 10

# Agent 1 output (run `python agents/code_context.py <script>` to generate it).
_CTX = load_cached() or {}
CODE_CTX = context_to_text(_CTX)
TARGET_FILE = _CTX.get("_script_path", "demo/dummy_training.py")

VAPI_API_KEY = os.getenv("VAPI_API_KEY", "").strip('"')
VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID", "").strip('"')
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID", "").strip('"')
MY_PHONE_NUMBER = os.getenv("MY_PHONE_NUMBER", "").strip('"')

INSFORGE_URL = os.getenv("INSFORGE_URL", "").rstrip("/")
INSFORGE_SERVICE_KEY = os.getenv("INSFORGE_SERVICE_KEY", "").strip('"')
_IF_HEADERS = {
    "Authorization": f"Bearer {INSFORGE_SERVICE_KEY}",
    "Content-Type": "application/json",
}
_VAPI_HEADERS = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json",
}

# Optional: load the caller's saved preferences to inform the Call Decision agent.
USER_PREFS = json.loads(os.getenv("CALL_PREFS", "{}"))


def _trigger_call(anomaly: dict, report: dict) -> str:
    recent = anomaly.get("recent_logs", [])
    recent_str = json.dumps(
        [{"step": r["step"], "loss": r["loss"], "grad_norm": r.get("grad_norm")} for r in recent[-20:]],
        indent=2,
    )
    brief = f"{report.get('error_type', anomaly['type'])} (severity {report.get('severity', '?')}) — {anomaly['brief']}"
    body = {
        "assistantId": VAPI_ASSISTANT_ID,
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {"number": MY_PHONE_NUMBER},
        "assistantOverrides": {
            "variableValues": {
                "run_id": RUN_ID,
                "anomaly_brief": brief,
                "recent_logs": recent_str,
                "code_context": CODE_CTX or "(no code context available)",
            }
        },
    }
    resp = httpx.post("https://api.vapi.ai/call", json=body, headers=_VAPI_HEADERS, timeout=15)
    resp.raise_for_status()
    call_id = resp.json()["id"]
    print(f"  Call triggered -> call_id={call_id}", flush=True)
    return call_id


def _wait_for_call(call_id: str) -> dict:
    print("  Waiting for call to end...", end="", flush=True)
    while True:
        time.sleep(CALL_POLL_INTERVAL)
        resp = httpx.get(f"https://api.vapi.ai/call/{call_id}", headers=_VAPI_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(".", end="", flush=True)
        if data.get("status") == "ended":
            print()
            return data


def _log_decision(anomaly: dict, report: dict, call_data: dict | None, call_decision: dict) -> str:
    transcript = (call_data or {}).get("transcript", "") or ""
    decision = extract_decision(transcript) if transcript else "not_called"
    print(f"  Decision: {decision}", flush=True)
    if transcript:
        trace.log_event(RUN_ID, "conversation_fix", "transcript", transcript, echo=False)
    payload = {
        "run_id": RUN_ID,
        "anomaly_type": report.get("error_type", anomaly["type"]),
        "anomaly_step": report.get("step", anomaly.get("step")),
        "decision": decision,
        "transcript": transcript or f"[call skipped] {call_decision.get('reason', '')}",
    }
    resp = httpx.post(
        f"{INSFORGE_URL}/api/database/records/call_decisions",
        json=[payload],
        headers=_IF_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    print("  Logged to call_decisions.", flush=True)
    return decision


def _run_fix_loop(report: dict, transcript: str, anomaly: dict) -> None:
    """Agent 5 self-healing: propose -> apply -> register corrected re-run."""
    print(f"  [agent 5] proposing a fix for {TARGET_FILE}...", flush=True)
    error = {
        "type": report.get("error_type") or anomaly.get("type"),
        "brief": anomaly.get("brief", ""),
    }
    try:
        fix = fix_mod.propose_fix(report, CODE_CTX, file=TARGET_FILE, transcript=transcript)
        trace.log_event(RUN_ID, "conversation_fix", "output", json.dumps(fix), echo=True)
        result = fix_mod.apply_fix(fix, RUN_ID, error=error)
        print(f"  [agent 5] applied fix change_id={result['change_id']} (backup {result['backup']})", flush=True)
        new_run = fix_mod.spawn_rerun(RUN_ID, result["change_id"], file=TARGET_FILE)
        print(f"  [agent 5] corrected re-run: {new_run}", flush=True)
        print(f"  Rollback anytime: python agents/fix.py rollback {result['change_id']}", flush=True)
    except Exception as exc:
        print(f"  [agent 5] fix loop failed: {exc}", flush=True)


def main() -> None:
    print(f"[agent-monitor] Watching run {RUN_ID} (every {POLL_INTERVAL}s)", flush=True)
    handled = False

    while True:
        result = check_anomaly(RUN_ID)

        if not result.get("detected"):
            print(f"  step={result.get('step', '?')} loss={result.get('loss', '?')} OK", flush=True)
            time.sleep(POLL_INTERVAL)
            continue

        print(f"\n  HEURISTIC FLAG: {result['brief']}", flush=True)
        if handled:
            time.sleep(POLL_INTERVAL)
            continue
        handled = True

        trace.log_event(RUN_ID, "monitor", "note", result["brief"], echo=False)

        recent = result.get("recent_logs", [])
        report = detect_error(result, recent, code_context=CODE_CTX or None)
        print(f"  [agent 3] error report: {report}", flush=True)
        trace.log_event(RUN_ID, "error_detection", "output", json.dumps(report), echo=False)

        call_decision = decide_call(report, prefs=USER_PREFS)
        print(f"  [agent 4] call decision: {call_decision}", flush=True)
        trace.log_event(RUN_ID, "call_decision", "output", json.dumps(call_decision), echo=False)

        if call_decision.get("call"):
            call_id = _trigger_call(result, report)
            call_data = _wait_for_call(call_id)
            decision = _log_decision(result, report, call_data, call_decision)
            if decision == "fix":
                _run_fix_loop(report, (call_data or {}).get("transcript", "") or "", result)
        else:
            print(f"  [agent 4] no call — {call_decision.get('reason', '')}", flush=True)
            _log_decision(result, report, None, call_decision)

        print("\n[agent-monitor] done.", flush=True)
        break


if __name__ == "__main__":
    main()
