#!/usr/bin/env python3
"""Poll for anomalies every 15s. Trigger a Vapi call when one is detected. Log the decision."""

import os
import sys
import time
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly_detector import check_anomaly
from agents.nebius_client import extract_decision

load_dotenv(Path(__file__).parent.parent / ".env")

# Accept run ID as CLI arg — must match what dummy_training.py printed
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "demo-run-001"
POLL_INTERVAL = 15
CALL_POLL_INTERVAL = 10

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


def _trigger_call(anomaly: dict) -> str:
    recent = anomaly.get("recent_logs", [])
    recent_str = json.dumps(
        [{"step": r["step"], "loss": r["loss"], "grad_norm": r.get("grad_norm")} for r in recent[-20:]],
        indent=2,
    )
    body = {
        "assistantId": VAPI_ASSISTANT_ID,
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {"number": MY_PHONE_NUMBER},
        "assistantOverrides": {
            "variableValues": {
                "run_id": RUN_ID,
                "anomaly_brief": anomaly["brief"],
                "recent_logs": recent_str,
            }
        },
    }
    resp = httpx.post("https://api.vapi.ai/call", json=body, headers=_VAPI_HEADERS, timeout=15)
    resp.raise_for_status()
    call_id = resp.json()["id"]
    print(f"  Call triggered → call_id={call_id}", flush=True)
    return call_id


def _wait_for_call(call_id: str) -> dict:
    print("  Waiting for call to end...", end="", flush=True)
    while True:
        time.sleep(CALL_POLL_INTERVAL)
        resp = httpx.get(f"https://api.vapi.ai/call/{call_id}", headers=_VAPI_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        print(".", end="", flush=True)
        if status == "ended":
            print()
            return data


def _log_decision(anomaly: dict, call_data: dict) -> None:
    transcript = call_data.get("transcript", "") or ""
    decision = extract_decision(transcript)
    print(f"  Decision: {decision}", flush=True)
    payload = {
        "run_id": RUN_ID,
        "anomaly_type": anomaly["type"],
        "anomaly_step": anomaly["step"],
        "decision": decision,
        "transcript": transcript,
    }
    resp = httpx.post(
        f"{INSFORGE_URL}/api/database/records/call_decisions",
        json=[payload],
        headers=_IF_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    print("  Logged to call_decisions.", flush=True)


def main() -> None:
    print(f"Monitoring run {RUN_ID} (polling every {POLL_INTERVAL}s)", flush=True)
    called = False

    while True:
        result = check_anomaly(RUN_ID)

        if not result.get("detected"):
            step = result.get("step", "?")
            loss = result.get("loss", "?")
            print(f"  step={step}  loss={loss}  OK", flush=True)
        else:
            print(f"\n  ANOMALY: {result['brief']}", flush=True)
            if not called:
                called = True
                call_id = _trigger_call(result)
                call_data = _wait_for_call(call_id)
                _log_decision(result, call_data)
                print("\nMonitor done — call complete and decision logged.")
                break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
