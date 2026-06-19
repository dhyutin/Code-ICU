#!/usr/bin/env python3
"""Sentiment classifier on the Hugging Face `rotten_tomatoes` dataset (WORKING).

A 2-layer MLP over hashed bag-of-words. Sane learning rate → loss decreases
smoothly. Streams real metrics to code_icu.

    python examples/text_sentiment.py
    # copy the printed RUN_ID, then in another terminal:
    python agents/monitor.py <RUN_ID>
"""

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from _common import grad_global_norm, hashing_vectorize, log_step, new_run_id, set_seed

TAG = "text-sentiment"
DIM = 4096
HIDDEN = 128
BATCH = 32
STEPS = 60
LR = 1e-3
DELAY = 0.5
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
    opt = torch.optim.Adam(model.parameters(), lr=LR)
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
