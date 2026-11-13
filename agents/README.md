# Condrox Agents

Specialist agents that extend the main cognitive pipeline
(`core/brain/brain_controller.py`).

Two kinds live here:

## In-pipeline mode agents (`core/agents/`)

Default, code, analyze, architect, and explain agents. These run inside
pipeline stage 08b and shape the response style for their intent class.

## Standalone external agents (this directory)

Full agents with their own API/CLI entry points. They delegate all cognition
to `BrainController.process_request()` — they add domain tools and workflows,
never a parallel intelligence.

| Agent | Purpose |
|---|---|
| `DeveloperAgent/` | Code analysis, generation, refactoring, and verification engine |
| `AdminAgent/` | Administrative operations agent |
| `PersonalAgent/` | Personal-assistant workflows on top of the main brain |
| `PreReleaseAgent/` | Hardened public-demo API — per-session isolation, IP bans, strict limits |

Each agent directory contains its own README, API, and CLI where applicable.
