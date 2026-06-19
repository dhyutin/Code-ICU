#!/usr/bin/env python3
"""Complete pre-build Vapi configuration: Nebius LLM, system prompt, phone check.

Run from anywhere:  python prebuild_checks/setup_prebuild.py
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

VAPI_BASE = "https://api.vapi.ai"
SYSTEM_PROMPT = """You are code_icu, an ML experiment monitor.
You are calling a researcher about a training anomaly.

Run ID: {{run_id}}
Anomaly: {{anomaly_brief}}
Recent logs (last 20 steps): {{recent_logs}}

Rules:
- Open: Hey, your run {{run_id}} just [anomaly]. Want me to walk through it?
- Answer questions using only the log data provided.
- Be concise. Researchers are busy.
- When they say kill it / pause / continue: confirm and end the call.
- Never invent data not in the logs."""


def _env(name: str) -> str:
    return os.getenv(name, "").strip().strip('"')


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_env('VAPI_API_KEY')}",
        "Content-Type": "application/json",
    }


def configure_assistant() -> None:
    assistant_id = _env("VAPI_ASSISTANT_ID")
    nebius_key = _env("NEBIUS_API_KEY")
    nebius_model = _env("NEBIUS_MODEL")
    nebius_url = _env("NEBIUS_BASE_URL").rstrip("/")

    cred_resp = httpx.post(
        f"{VAPI_BASE}/credential",
        json={"provider": "custom-llm", "apiKey": nebius_key},
        headers=_headers(),
        timeout=15,
    )
    if cred_resp.status_code in (200, 201):
        print(f"  credential OK: {cred_resp.json().get('id')}")
    else:
        print(f"  credential: {cred_resp.status_code} (may already exist)")

    patch_body = {
        "model": {
            "provider": "custom-llm",
            "model": nebius_model,
            "url": f"{nebius_url}/chat/completions",
            "temperature": 0.6,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        }
    }
    resp = httpx.patch(
        f"{VAPI_BASE}/assistant/{assistant_id}",
        json=patch_body,
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    model = resp.json().get("model", {})
    print(f"  assistant: {model.get('provider')} / {model.get('model')}")


def check_phone() -> None:
    phone_id = _env("VAPI_PHONE_NUMBER_ID")
    resp = httpx.get(f"{VAPI_BASE}/phone-number/{phone_id}", headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "unknown")
    print(f"  phone: {data.get('number')} status={status}")
    if status != "active":
        raise RuntimeError(f"Phone number not active: {status}")


def main() -> None:
    print("Configuring Vapi...")
    configure_assistant()
    print("Checking phone...")
    check_phone()
    print("Pre-build Vapi setup OK.")


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPStatusError as exc:
        print(f"HTTP error: {exc.response.status_code} {exc.response.text[:300]}", file=sys.stderr)
        sys.exit(1)
