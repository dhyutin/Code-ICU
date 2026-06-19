"""Smoke-test the registry-driven agents end to end against Nebius."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.runtime import decide_call, detect_error, load_agent, monitor_health

SAMPLE_LOGS = [
    {"step": 45, "loss": 2.11, "grad_norm": 8.2, "lr": 3e-4},
    {"step": 46, "loss": 2.09, "grad_norm": 9.0, "lr": 3e-4},
    {"step": 47, "loss": 2.05, "grad_norm": 12.4, "lr": 3e-4},
    {"step": 48, "loss": 2.40, "grad_norm": 45.0, "lr": 3e-4},
    {"step": 49, "loss": 4.85, "grad_norm": 130.0, "lr": 3e-4},
    {"step": 50, "loss": 9.72, "grad_norm": 410.0, "lr": 3e-4},
]

ANOMALY = {
    "type": "grad_explosion",
    "step": 50,
    "brief": "Gradient norm hit 410.0 (limit 100) at step 50",
}


def main() -> None:
    print("== registry ==")
    for k in ("monitor", "error_detection", "call_decision"):
        a = load_agent(k)
        print(f"  {k:16} -> {a['name']} | {a['model']}")

    print("\n== monitor (agent 2) ==")
    print(" ", monitor_health(SAMPLE_LOGS))

    print("\n== error detection (agent 3) ==")
    report = detect_error(ANOMALY, SAMPLE_LOGS)
    print(" ", report)

    print("\n== call decision (agent 4) ==")
    decision = decide_call(report, prefs={"call_threshold": "high"})
    print(" ", decision)


if __name__ == "__main__":
    main()
