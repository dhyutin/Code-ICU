"""Fetch the last N steps from InsForge and detect training anomalies."""

import math
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
    latest = rows[-1]

    # Grad explosion — any step in the recent window over the limit (also
    # catches non-finite values, which scripts log as a large sentinel).
    grad_rows = [r for r in rows if r.get("grad_norm") is not None]
    if grad_rows:
        worst = max(grad_rows, key=lambda r: r["grad_norm"])
        if not math.isfinite(worst["grad_norm"]) or worst["grad_norm"] > GRAD_LIMIT:
            return {
                "detected": True,
                "type": "grad_explosion",
                "step": worst["step"],
                "brief": f"Gradient norm hit {worst['grad_norm']:.1f} (limit {GRAD_LIMIT}) at step {worst['step']}",
                "recent_logs": rows,
            }

    # Loss spike — a sudden jump UP between consecutive steps (a healthy run
    # only trends down, so this won't fire on strong convergence).
    if len(losses) >= 2:
        median = statistics.median(losses)
        for i in range(1, len(losses)):
            prev, cur = losses[i - 1], losses[i]
            if prev > 0 and cur > prev * SPIKE_RATIO and cur > median:
                return {
                    "detected": True,
                    "type": "loss_spike",
                    "step": rows[i]["step"],
                    "brief": f"Loss jumped to {cur:.4f} (from {prev:.4f}) at step {rows[i]['step']}",
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
