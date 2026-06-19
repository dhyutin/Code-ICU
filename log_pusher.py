"""Push a training step to InsForge run_logs."""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

_URL = os.getenv("INSFORGE_URL", "").rstrip("/")
_KEY = os.getenv("INSFORGE_SERVICE_KEY", "")
_HEADERS = {
    "Authorization": f"Bearer {_KEY}",
    "Content-Type": "application/json",
}


def push_log(run_id: str, step: int, loss: float, grad_norm: float = None, lr: float = None) -> None:
    payload = {"run_id": run_id, "step": step, "loss": loss}
    if grad_norm is not None:
        payload["grad_norm"] = grad_norm
    if lr is not None:
        payload["lr"] = lr

    resp = httpx.post(f"{_URL}/api/database/records/run_logs", json=[payload], headers=_HEADERS, timeout=10)
    resp.raise_for_status()
