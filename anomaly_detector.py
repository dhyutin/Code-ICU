"""Fetch the last N steps from InsForge and detect training anomalies."""

import os
import statistics
import httpx
from dotenv import load_dotenv

load_dotenv()

_URL = os.getenv("INSFORGE_URL", "").rstrip("/")
_KEY = os.getenv("INSFORGE_SERVICE_KEY", "")
_HEADERS = {
    "Authorization": f"Bearer {_KEY}",
}

WINDOW = 30
SPIKE_RATIO = 2.0
GRAD_LIMIT = 100.0
PLATEAU_STEPS = 20


def _fetch_recent(run_id: str, limit: int) -> list[dict]:
    resp = httpx.get(
        f"{_URL}/api/database/records/run_logs",
        params={"run_id": f"eq.{run_id}", "order": "step.desc", "limit": limit},
        headers=_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    return list(reversed(resp.json()))


def check_anomaly(run_id: str) -> dict:
    rows = _fetch_recent(run_id, WINDOW)
    if len(rows) < 5:
        return {"detected": False}

    losses = [r["loss"] for r in rows]
    grad_norms = [r["grad_norm"] for r in rows if r.get("grad_norm") is not None]
    latest = rows[-1]

    # Grad explosion
    if grad_norms and grad_norms[-1] > GRAD_LIMIT:
        return {
            "detected": True,
            "type": "grad_explosion",
            "step": latest["step"],
            "brief": f"Gradient norm hit {grad_norms[-1]:.1f} (limit {GRAD_LIMIT}) at step {latest['step']}",
            "recent_logs": rows,
        }

    # Loss spike — latest loss vs mean of everything before it
    if len(losses) >= 2:
        baseline = statistics.mean(losses[:-1])
        if baseline > 0 and losses[-1] > baseline * SPIKE_RATIO:
            return {
                "detected": True,
                "type": "loss_spike",
                "step": latest["step"],
                "brief": f"Loss jumped to {losses[-1]:.4f} (baseline {baseline:.4f}) at step {latest['step']}",
                "recent_logs": rows,
            }

    # Plateau — no improvement over last PLATEAU_STEPS
    if len(losses) >= PLATEAU_STEPS:
        window = losses[-PLATEAU_STEPS:]
        if max(window) - min(window) < 0.001:
            return {
                "detected": True,
                "type": "plateau",
                "step": latest["step"],
                "brief": f"Loss stuck at ~{statistics.mean(window):.4f} for {PLATEAU_STEPS} steps",
                "recent_logs": rows,
            }

    return {"detected": False, "step": latest["step"], "loss": latest["loss"]}
