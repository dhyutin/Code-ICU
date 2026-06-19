#!/usr/bin/env python3
"""Sentiment classifier on `rotten_tomatoes` (FAILING — learning rate too high).

Identical to text_sentiment.py except the learning rate is set far too high, so
SGD overshoots and the loss/gradients explode within a few steps. This is the
classic bug the Code Context + Error Detection agents catch, and the fix is to
lower the learning rate.

    python examples/text_sentiment_fail.py
    python agents/monitor.py <RUN_ID>
"""

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from _common import grad_global_norm, hashing_vectorize, log_step, new_run_id, set_seed

TAG = "text-sentiment-fail"
DIM = 4096
HIDDEN = 128
BATCH = 32
STEPS = 28
LR = 30.0  # BUG: learning rate is orders of magnitude too high
DELAY = 1.1
SUBSET = 3000


def main() -> None:
    set_seed(0)
    run_id = new_run_id(TAG)
    print(f"RUN_ID={run_id}")
    print(f"Loading rotten_tomatoes (subset {SUBSET})...", flush=True)

    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train").shuffle(seed=0).select(range(SUBSET))
    X = torch.from_numpy(hashing_vectorize(ds["text"], DIM))
    y = torch.tensor(ds["label"], dtype=torch.long)

    model = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))
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
