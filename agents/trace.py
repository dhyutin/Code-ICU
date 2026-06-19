"""Agent traceability — persist every agent input/output + human transcript.

Writes to InsForge `agent_events` (service key, bypasses RLS) and echoes to the
backend terminal so you can hear/see the agents reasoning during a run.

Also resolves the UUID `runs` row for a text run_id (creating it if needed) so
agent_events / fix_attempts can be owner-scoped via RLS.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

_URL = os.getenv("INSFORGE_URL", "").rstrip("/")
_KEY = os.getenv("INSFORGE_SERVICE_KEY", "").strip('"')
_HEADERS = {"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"}

_run_cache: dict[str, dict] = {}


def _get(table: str, params: dict) -> list[dict]:
    r = httpx.get(f"{_URL}/api/database/records/{table}", params=params, headers=_HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def _insert(table: str, row: dict) -> dict:
    headers = {**_HEADERS, "Prefer": "return=representation"}
    r = httpx.post(f"{_URL}/api/database/records/{table}", json=[row], headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = data.get("records", data.get("data", data))
    return data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})


def patch(table: str, row_id: str, row: dict) -> None:
    r = httpx.patch(
        f"{_URL}/api/database/records/{table}",
        params={"id": f"eq.{row_id}"},
        json=row,
        headers=_HEADERS,
        timeout=10,
    )
    r.raise_for_status()


def resolve_run(run_id: str) -> dict:
    """Return the `runs` row ({id, user_id, ...}) for a text run_id, creating it."""
    if run_id in _run_cache:
        return _run_cache[run_id]

    rows = _get("runs", {"name": f"eq.{run_id}", "limit": 1})
    if rows:
        _run_cache[run_id] = rows[0]
        return rows[0]

    # No run yet — attach it to a user (single-tenant hackathon: first profile).
    profiles = _get("profiles", {"limit": 1})
    user_id = profiles[0]["user_id"] if profiles else None
    if not user_id:
        raise RuntimeError("No user/profile found to own the run")
    run = _insert("runs", {"user_id": user_id, "name": run_id, "status": "running"})
    _run_cache[run_id] = run
    return run


def log_event(run_id: str, agent_key: str, kind: str, content: str, *, echo: bool = True) -> None:
    """Persist an agent event and (optionally) echo it to the terminal."""
    try:
        run = resolve_run(run_id)
        _insert(
            "agent_events",
            {
                "run_ref": run["id"],
                "run_id": run_id,
                "agent_key": agent_key,
                "kind": kind,
                "content": content,
            },
        )
    except Exception as exc:  # tracing must never break the pipeline
        if echo:
            print(f"  [trace error] {exc}", flush=True)

    if echo:
        label = f"[{agent_key}:{kind}]"
        snippet = content if len(content) <= 600 else content[:600] + "…"
        print(f"  {label} {snippet}", flush=True)
