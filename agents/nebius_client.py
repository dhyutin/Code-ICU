"""Nebius AI Studio client (OpenAI-compatible) — the agents' LLM gateway."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

DECISION_PROMPT = """You classify a researcher's decision from a phone call transcript about an ML training run.
Reply with exactly one word:
- kill      (stop the run)
- pause     (hold the run)
- continue  (let it keep running)
- fix       (the researcher wants code_icu to apply a code fix and re-run)
If unclear, reply: unknown

Transcript:
{transcript}"""


def get_client() -> OpenAI:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key or api_key.startswith("<"):
        raise ValueError(
            "NEBIUS_API_KEY is missing or still a placeholder. "
            "Create a key at https://studio.nebius.ai/settings/api-keys"
        )
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)


def chat(prompt: str, *, temperature: float = 0.6) -> str:
    client = get_client()
    completion = client.chat.completions.create(
        model=NEBIUS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Nebius returned an empty response")
    return content.strip()


def extract_decision(transcript: str) -> str:
    """Return kill, pause, continue, or unknown using Nebius with keyword fallback."""
    try:
        raw = chat(
            DECISION_PROMPT.format(transcript=transcript),
            temperature=0,
        ).lower()
        for word in ("kill", "pause", "continue", "fix", "unknown"):
            if word in raw.split():
                return word
    except Exception:
        pass

    text = transcript.lower()
    if re.search(r"\b(fix|patch|repair|correct|lower the lr|clip)\b", text):
        return "fix"
    if re.search(r"\b(kill|stop|abort|terminate)\b", text):
        return "kill"
    if re.search(r"\b(pause|hold|wait)\b", text):
        return "pause"
    if re.search(r"\b(continue|keep|resume|go on)\b", text):
        return "continue"
    return "unknown"
