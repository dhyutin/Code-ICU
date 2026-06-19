#!/usr/bin/env python3
"""Code Context Agent (agent 1).

Reads a training script and produces structured context that the Monitor (2),
Error Detection (3), Call Decision (4), and Conversation & Fix (5) agents use to
reason about *this* codebase — not just raw metrics.

Output is cached to code_context.json (at the project root) so the monitor can
load it cheaply.

    python agents/code_context.py demo/dummy_training.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.runtime import run_agent_json

CACHE_FILE = ROOT / "code_context.json"
MAX_CHARS = 12000  # keep the prompt bounded for big scripts


def analyze_script(path: str | Path) -> dict:
    """Run agent 1 over a training script → structured context dict."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    source = p.read_text()[:MAX_CHARS]

    prompt = (
        "Analyze this ML training script and return JSON only with keys: "
        "summary (string, 1-2 sentences), "
        "architecture (string), "
        "hyperparams (object of name -> value you can find), "
        "data_flow (string), "
        "likely_failure_points (array of short strings). "
        "Answer only from the code provided.\n\n"
        f"File: {p.name}\n```python\n{source}\n```"
    )
    ctx = run_agent_json("code_context", prompt)
    ctx["_script_path"] = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    return ctx


def context_to_text(ctx: dict) -> str:
    """Compact human/agent-readable view for feeding downstream agents."""
    if not ctx:
        return ""
    lines = [
        f"Script: {ctx.get('_script_path', '?')}",
        f"Summary: {ctx.get('summary', '')}",
        f"Architecture: {ctx.get('architecture', '')}",
        f"Hyperparams: {json.dumps(ctx.get('hyperparams', {}))}",
        f"Data flow: {ctx.get('data_flow', '')}",
        "Likely failure points: " + "; ".join(ctx.get("likely_failure_points", []) or []),
    ]
    return "\n".join(lines)


def save(ctx: dict) -> None:
    CACHE_FILE.write_text(json.dumps(ctx, indent=2))


def load_cached() -> dict | None:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python agents/code_context.py <training_script.py>", file=sys.stderr)
        sys.exit(1)
    ctx = analyze_script(sys.argv[1])
    save(ctx)
    print(context_to_text(ctx))
    print(f"\nSaved -> {CACHE_FILE.name}")


if __name__ == "__main__":
    main()
