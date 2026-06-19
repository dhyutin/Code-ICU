"""Push the Conversation & Fix agent (agent 5) from the InsForge registry to Vapi.

The Vapi assistant's system prompt is sourced from the `conversation_fix` row in
the `agents` table, then extended with the live per-call context block that
agents/monitor.py fills via assistantOverrides.variableValues. Re-run this
whenever the agent's prompt changes in the registry.

    python agents/sync_vapi.py
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agents.runtime import load_agent

VAPI_BASE = "https://api.vapi.ai"

# Live context Vapi injects per call (filled from monitor's variableValues).
CONTEXT_TEMPLATE = """

--- Current run context ---
Run ID: {{run_id}}
Anomaly: {{anomaly_brief}}
Recent logs (last 20 steps): {{recent_logs}}

Code context (from the Code Context agent):
{{code_context}}

Open the call naturally, e.g. "Hey, your run {{run_id}} just hit {{anomaly_brief}}.
Want me to walk through it?" Answer only from the logs and code context above."""


def _env(name: str) -> str:
    return os.getenv(name, "").strip().strip('"')


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_env('VAPI_API_KEY')}",
        "Content-Type": "application/json",
    }


def main() -> None:
    assistant_id = _env("VAPI_ASSISTANT_ID")
    nebius_key = _env("NEBIUS_API_KEY")
    nebius_model = _env("NEBIUS_MODEL")
    nebius_url = _env("NEBIUS_BASE_URL").rstrip("/")

    agent = load_agent("conversation_fix", force=True)
    system_prompt = agent["system_prompt"] + CONTEXT_TEMPLATE
    print(f"Sourcing prompt from registry agent: {agent['name']} ({agent['model']})")

    # Ensure Nebius custom-llm credential exists (idempotent).
    cred = httpx.post(
        f"{VAPI_BASE}/credential",
        json={"provider": "custom-llm", "apiKey": nebius_key},
        headers=_headers(),
        timeout=15,
    )
    print(f"  credential: {cred.status_code}")

    patch_body = {
        "model": {
            "provider": "custom-llm",
            "model": nebius_model,
            "url": f"{nebius_url}/chat/completions",
            "temperature": (agent.get("params") or {}).get("temperature", 0.5),
            "messages": [{"role": "system", "content": system_prompt}],
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
    print(f"  assistant updated: {model.get('provider')} / {model.get('model')}")
    print("Vapi now speaks as the registry's Conversation & Fix agent.")


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPStatusError as exc:
        print(f"HTTP error: {exc.response.status_code} {exc.response.text[:300]}", file=sys.stderr)
        sys.exit(1)
