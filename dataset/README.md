# Decision dataset (Phase 10)

The data flywheel: every anomaly code_icu calls about and every fix it applies is
logged, then exported here as supervised training data for two future autonomous
agents. Over time this is what you'd fine-tune to take the human out of the loop.

Generate / refresh:

```bash
source venv/bin/activate
python agents/export_dataset.py        # writes to dataset/
```

## Files

| File | What it is |
| --- | --- |
| `triage.jsonl` | anomaly context → decision (`kill`/`pause`/`continue`/`fix`) |
| `fix.jsonl` | failing code + error → JSON edits that repair it |
| `records.jsonl` | raw joined `call_decisions` + `fix_attempts` rows (for analysis) |
| `stats.json` | counts: decisions, examples, class/error/status breakdowns |

`triage.jsonl` and `fix.jsonl` are in OpenAI/Nebius **chat fine-tuning format** —
one JSON object per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "kill"}]}
```

## Fine-tuning (Nebius / OpenAI-compatible)

One command exports **and** starts the job:

```bash
python agents/export_dataset.py --fine-tune triage   # or: fix
```

It uploads the chosen JSONL and creates a fine-tune job on Nebius (base model =
`NEBIUS_MODEL`), printing the job id. It warns if there are fewer than 10
examples (most endpoints reject smaller sets) — collect more real decisions
first.

Under the hood it's just the standard OpenAI-compatible flow, using the same
`openai` client the project already uses:

```python
from openai import OpenAI
import os

client = OpenAI(base_url=os.environ["NEBIUS_BASE_URL"], api_key=os.environ["NEBIUS_API_KEY"])

f = client.files.create(file=open("dataset/triage.jsonl", "rb"), purpose="fine-tune")
job = client.fine_tuning.jobs.create(training_file=f.id, model=os.environ["NEBIUS_MODEL"])
print(job.id)
```

Then point `agents.runtime` / the Call Decision agent's `model` (in the InsForge
`agents` table) at the resulting fine-tuned model id — no code change, just a row
update.

> Collect a few dozen real decisions first; a handful of demo rows is enough to
> validate the pipeline but not to train a useful model.
