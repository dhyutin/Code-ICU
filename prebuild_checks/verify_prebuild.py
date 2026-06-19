#!/usr/bin/env python3
"""Verify all pre-build checklist items from Plan/README.md Section 2.

Run from anywhere:  python prebuild_checks/verify_prebuild.py
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

REQUIRED_FILES = [
    "log_pusher.py",
    "anomaly_detector.py",
    "agents/nebius_client.py",
    "requirements.txt",
    "migrations/20260619183440_init-schema.sql",
    ".env",
    "demo/dummy_training.py",
    "demo/monitor.py",
]

REQUIRED_ENV = [
    "INSFORGE_URL",
    "INSFORGE_ANON_KEY",
    "INSFORGE_SERVICE_KEY",
    "VAPI_API_KEY",
    "VAPI_ASSISTANT_ID",
    "VAPI_PHONE_NUMBER_ID",
    "MY_PHONE_NUMBER",
    "NEBIUS_API_KEY",
    "NEBIUS_BASE_URL",
    "NEBIUS_MODEL",
]


def check_files() -> list[str]:
    return [f for f in REQUIRED_FILES if not (ROOT / f).exists()]


def check_env() -> list[str]:
    bad = []
    for key in REQUIRED_ENV:
        val = os.getenv(key, "").strip().strip('"')
        if not val or val.startswith("<"):
            bad.append(key)
    if os.getenv("MY_PHONE_NUMBER", "").startswith("+1") is False:
        bad.append("MY_PHONE_NUMBER (not E.164)")
    return bad


def check_insforge() -> None:
    url = os.getenv("INSFORGE_URL", "").rstrip("/")
    key = os.getenv("INSFORGE_SERVICE_KEY", "")
    headers = {"Authorization": f"Bearer {key}"}
    for table in ("run_logs", "call_decisions"):
        r = httpx.get(
            f"{url}/api/database/records/{table}",
            params={"limit": 1},
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
    print("  Insforge tables OK")


def check_push_and_detect() -> None:
    from log_pusher import push_log
    from anomaly_detector import check_anomaly

    run_id = "build-verify"
    for step in range(1, 6):
        push_log(run_id, step, 1.0, 1.0, 3e-4)
    push_log(run_id, 6, 1.0, 150.0, 3e-4)
    result = check_anomaly(run_id)
    if not result.get("detected"):
        raise RuntimeError("Anomaly detector did not flag grad explosion")
    print(f"  push_log + anomaly_detector OK ({result['type']})")


def check_nebius() -> None:
    from agents.nebius_client import chat

    reply = chat("Reply with exactly: ok", temperature=0)
    if "ok" not in reply.lower():
        raise RuntimeError(f"Unexpected Nebius reply: {reply[:50]}")
    print("  Nebius API OK")


def main() -> None:
    print("=== Pre-build verification ===\n")

    missing = check_files()
    if missing:
        print(f"FAIL missing files: {', '.join(missing)}")
        sys.exit(1)
    print(f"Files ({len(REQUIRED_FILES)}): OK")

    bad_env = check_env()
    if bad_env:
        print(f"FAIL missing/placeholder env: {', '.join(bad_env)}")
        sys.exit(1)
    print(f"Env ({len(REQUIRED_ENV)} vars): OK")

    if not (ROOT / "venv").exists():
        print("WARN venv/ not found — run: python3 -m venv venv && pip install -r requirements.txt")
    else:
        print("venv/: OK")

    print("\nLive checks:")
    check_insforge()
    check_push_and_detect()
    check_nebius()

    print("\n=== Pre-build PASS ===")
    print("Run demo: python demo/dummy_training.py   (terminal 1, copy the printed RUN_ID)")
    print("          python demo/monitor.py <RUN_ID> (terminal 2)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)
