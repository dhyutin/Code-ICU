#!/usr/bin/env python3
"""Simulated training run. Loss spikes to 5.5 at step 50."""

import sys
import time
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from log_pusher import push_log

RUN_ID = f"demo-run-{datetime.now().strftime('%H%M%S')}"
STEPS = 100
SPIKE_STEP = 50
STEP_DELAY = 0.8
LR = 3e-4


def main() -> None:
    print(f"RUN_ID={RUN_ID}")
    print(f"Starting run {RUN_ID} ({STEPS} steps, spike at step {SPIKE_STEP})")
    loss = 1.0

    for step in range(1, STEPS + 1):
        if step == SPIKE_STEP:
            loss = 5.5
            grad_norm = 150.0
        else:
            loss = max(0.1, loss * 0.98 + random.gauss(0, 0.02))
            grad_norm = random.uniform(0.5, 2.5)

        push_log(RUN_ID, step, round(loss, 4), round(grad_norm, 4), LR)
        print(f"  step {step:3d}  loss={loss:.4f}  grad_norm={grad_norm:.4f}", flush=True)
        time.sleep(STEP_DELAY)

    print("Run complete.")


if __name__ == "__main__":
    main()
