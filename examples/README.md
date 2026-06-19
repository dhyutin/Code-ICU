# Examples — real training tasks

Real PyTorch training jobs on Hugging Face datasets. Each script streams **real**
metrics (loss, gradient norm, learning rate) into the same InsForge `run_logs`
pipeline that code_icu monitors, so the agents watch these exactly like a
production run.

There are two domains, each with a **working** and a **failing** (`_fail.py`)
version. The failing versions contain a single realistic bug that the Code
Context + Error Detection agents diagnose — and the Fix agent can repair.

| Script | Data (HF) | Status | Bug | Symptom code_icu detects |
| --- | --- | --- | --- | --- |
| `text_sentiment.py` | `rotten_tomatoes` | works | — | loss decreases smoothly |
| `text_sentiment_fail.py` | `rotten_tomatoes` | fails | learning rate `30.0` (way too high) | loss spikes within a few steps |
| `mnist_mlp.py` | `mnist` | works | — | loss decreases steadily |
| `mnist_mlp_fail.py` | `mnist` | fails | pixels never normalized (raw 0-255) | activations blow up → loss → NaN |

## Run one

```bash
source venv/bin/activate           # if not already active
python examples/mnist_mlp_fail.py  # prints RUN_ID=mnist-mlp-fail-HHMMSS
```

Then point the agent monitor at the run (in a second terminal) to get live
anomaly detection, the voice call, and the self-healing fix loop:

```bash
python agents/monitor.py <RUN_ID>
```

The dashboard (`dashboard/index.html`) shows the live loss/grad curves, agent
activity, and any applied fixes for that run.

## Notes

- First run downloads the dataset to the local Hugging Face cache (a few MB) — be
  online for that.
- Subsets are small (3-4k rows) and steps are paced so the monitor has time to
  react; this is tuned for a live demo, not for accuracy.
- The two bugs are intentionally the two most common real-world training
  failures: a bad learning rate and missing input normalization.
