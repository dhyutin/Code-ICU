"""Registry-driven Nebius agent runtime.

The five code_icu agents live as rows in the InsForge `agents` table
(key, role, model, system_prompt, params). This module loads those definitions
and runs them through Nebius. Tuning or adding an agent is a data change in the
table, not a code change here.

High-level helpers (monitor / detect_error / decide_call) wrap the raw agents
with the structured JSON contracts the orchestrator expects.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agents.nebius_client import get_client

_URL = os.getenv("INSFORGE_URL", "").rstrip("/")
_KEY = os.getenv("INSFORGE_SERVICE_KEY", "").strip('"')
_HEADERS = {"Authorization": f"Bearer {_KEY}"}

_CACHE_TTL = 60.0  # seconds; agent defs rarely change mid-run
_cache: dict[str, tuple[float, dict]] = {}


class AgentError(RuntimeError):
    pass


def load_agent(key: str, *, force: bool = False) -> dict:
    """Fetch an agent definition from the InsForge registry (cached)."""
    now = time.time()
    if not force and key in _cache:
        ts, agent = _cache[key]
        if now - ts < _CACHE_TTL:
            return agent

    resp = httpx.get(
        f"{_URL}/api/database/records/agents",
        params={"key": f"eq.{key}", "limit": 1},
        headers=_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise AgentError(f"No agent with key '{key}' in the registry")
    agent = rows[0]
    if not agent.get("enabled", True):
        raise AgentError(f"Agent '{key}' is disabled")
    _cache[key] = (now, agent)
    return agent


def run_agent(
    key: str,
    user_content: str,
    *,
    context: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Run a registry agent against Nebius and return its text response."""
    agent = load_agent(key)
    params = agent.get("params") or {}
    if temperature is None:
        temperature = params.get("temperature", 0.3)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": agent["system_prompt"]}
    ]
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": user_content})

    kwargs: dict[str, Any] = {
        "model": agent["model"],
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    client = get_client()
    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    if not content:
        raise AgentError(f"Agent '{key}' returned an empty response")
    return content.strip()


def run_agent_json(key: str, user_content: str, *, context: str | None = None) -> dict:
    """Run an agent expecting JSON output; parse with a salvage fallback."""
    raw = run_agent(key, user_content, context=context, temperature=0, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise AgentError(f"Agent '{key}' did not return valid JSON:\n{raw}")


# ---------------------------------------------------------------------------
# High-level agent helpers (structured contracts for the orchestrator)
# ---------------------------------------------------------------------------

def _logs_to_text(recent_logs: list[dict]) -> str:
    return json.dumps(
        [
            {
                "step": r.get("step"),
                "loss": r.get("loss"),
                "grad_norm": r.get("grad_norm"),
                "lr": r.get("lr"),
            }
            for r in recent_logs[-20:]
        ],
        indent=2,
    )


def monitor_health(recent_logs: list[dict], *, code_context: str | None = None) -> str:
    """Agent 2 — Monitor: one-line health read of the live metric stream."""
    prompt = (
        "Here is the most recent window of training metrics. In one or two "
        "sentences, describe the current health of the run and flag anything "
        "deviating from a healthy trajectory.\n\n"
        f"{_logs_to_text(recent_logs)}"
    )
    return run_agent("monitor", prompt, context=code_context, temperature=0.2)


def detect_error(
    anomaly: dict, recent_logs: list[dict], *, code_context: str | None = None
) -> dict:
    """Agent 3 — Error Detection: structured failure report."""
    prompt = (
        "Classify the training failure from the heuristic anomaly flag and the "
        "recent metric window. Respond with JSON only, with keys: "
        "error_type (string), step (number), severity (one of "
        "low|medium|high|critical), evidence (string).\n\n"
        f"Heuristic flag: {json.dumps({k: anomaly.get(k) for k in ('type', 'step', 'brief')})}\n\n"
        f"Recent metrics:\n{_logs_to_text(recent_logs)}"
    )
    return run_agent_json("error_detection", prompt, context=code_context)


def decide_call(error_report: dict, *, prefs: dict | None = None) -> dict:
    """Agent 4 — Call Decision: should we phone the researcher?

    A high/critical failure is worth a call by default — missing preferences mean
    "call me on serious problems", not "stay silent". A deterministic safety net
    guarantees the call on severe failures unless prefs explicitly forbid it.
    """
    prefs = prefs or {}
    prompt = (
        "Given the error report and the user's preferences, decide whether to "
        "interrupt the researcher with a phone call. Respond with JSON only, "
        "with keys: call (boolean), reason (one short sentence). "
        "Call when the failure is severe or actionable. If severity is high or "
        "critical, you should call unless preferences explicitly say otherwise. "
        "Treat missing preferences as 'call me on serious problems'.\n\n"
        f"Error report: {json.dumps(error_report)}\n\n"
        f"User preferences: {json.dumps(prefs)}"
    )
    result = run_agent_json("call_decision", prompt)

    # Deterministic safety net: never silently swallow a severe failure.
    severity = str(error_report.get("severity", "")).lower()
    if severity in ("high", "critical") and not prefs.get("never_call") and not result.get("call"):
        result = {"call": True, "reason": f"Severity {severity}: calling on a serious failure."}
    return result


if __name__ == "__main__":
    # Quick visibility into what's loaded.
    for k in ("code_context", "monitor", "error_detection", "call_decision", "conversation_fix"):
        a = load_agent(k)
        print(f"{k:18} -> {a['name']} | {a['model']} | enabled={a.get('enabled')}")
