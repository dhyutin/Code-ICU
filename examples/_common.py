"""Shared helpers for the example training tasks.

Every example streams real metrics (loss, gradient norm, lr) into the same
InsForge `run_logs` pipeline that code_icu monitors — so the agents watch these
exactly like a real run. Each script prints `RUN_ID=...`; pass it to the monitor:

    python agents/monitor.py <RUN_ID>
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from log_pusher import push_log


def new_run_id(tag: str) -> str:
    # A code_icu-launched corrected re-run pins the id so its logs match the
    # registered run; otherwise generate a fresh timestamped id.
    return os.getenv("CODE_ICU_RUN_ID") or f"{tag}-{datetime.now().strftime('%H%M%S')}"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def grad_global_norm(model: torch.nn.Module) -> float:
    """L2 norm of all gradients — the real signal code_icu watches for explosions."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item()) ** 2
    return total ** 0.5


def hashing_vectorize(texts: list[str], dim: int = 4096) -> np.ndarray:
    """Lightweight bag-of-words via the hashing trick (no sklearn needed)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def log_step(
    run_id: str,
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    *,
    delay: float = 0.35,
    echo: bool = True,
) -> None:
    """Push one training step and pace it so the monitor has time to react."""
    # Diverged runs overflow to inf/NaN; log a large finite sentinel so the row
    # is valid JSON and still reads as an explosion downstream.
    loss = float(loss) if math.isfinite(loss) else 1e6
    grad_norm = float(grad_norm) if math.isfinite(grad_norm) else 1e6
    push_log(run_id, step, round(loss, 4), round(grad_norm, 4), lr)
    if echo:
        print(f"  step {step:3d}  loss={loss:.4f}  grad_norm={grad_norm:.2f}", flush=True)
    time.sleep(delay)
