"""code_icu agent layer.

The five specialized agents (Code Context, Monitor, Error Detection, Call
Decision, Conversation & Fix) live as rows in the InsForge `agents` table and
run through Nebius. This package holds the runtime, the orchestrator, the Code
Context agent, the Nebius client, and the Vapi sync helper.

Modules are runnable directly, e.g.:
    python agents/code_context.py demo/dummy_training.py
    python agents/monitor.py <RUN_ID>
    python agents/sync_vapi.py
    python agents/test_agents.py
"""
