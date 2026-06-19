#!/usr/bin/env python3
"""Digit classifier on `mnist` (FAILING — pixels are never normalized).

Identical to mnist_mlp.py except the raw 0-255 pixel values are fed straight
into the network. The huge input scale blows up activations and gradients, so
the loss diverges to NaN. The fix is to normalize the pixels (divide by 255).

    python examples/mnist_mlp_fail.py
    python agents/monitor.py <RUN_ID>
"""

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from _common import grad_global_norm, log_step, new_run_id, set_seed

TAG = "mnist-mlp-fail"
HIDDEN = 256
BATCH = 64
STEPS = 28
LR = 0.1
DELAY = 1.1
SUBSET = 4000


def load_pixels(ds) -> np.ndarray:
    return np.stack([np.array(im, dtype=np.float32) for im in ds["image"]]).reshape(len(ds), 784)


def main() -> None:
    set_seed(0)
    run_id = new_run_id(TAG)
    print(f"RUN_ID={run_id}")
    print(f"Loading mnist (subset {SUBSET})...", flush=True)

    ds = load_dataset("ylecun/mnist", split="train").shuffle(seed=0).select(range(SUBSET))
    pixels = load_pixels(ds)  # BUG: raw 0-255 pixels, never normalized
    X = torch.from_numpy(pixels)
    y = torch.tensor(ds["label"], dtype=torch.long)

    model = nn.Sequential(nn.Linear(784, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 10))
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training MLP ({STEPS} steps, lr={LR})", flush=True)
    n = X.shape[0]
    for step in range(1, STEPS + 1):
        idx = torch.from_numpy(np.random.randint(0, n, size=BATCH))
        logits = model(X[idx])
        loss = loss_fn(logits, y[idx])

        opt.zero_grad()
        loss.backward()
        gnorm = grad_global_norm(model)
        opt.step()

        log_step(run_id, step, loss.item(), gnorm, LR, delay=DELAY)

    print("Run complete.", flush=True)


if __name__ == "__main__":
    main()
