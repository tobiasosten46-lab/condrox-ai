# DeveloperAgent

Code-specialist extension of the main brain (`BrainController`).

An **external agent**: it takes the main `BrainController` as input and uses
`BrainController.process_request()` for all intelligence. It adds
developer-specific tools, workflows, and quality gates on top.

## Capabilities

- AST-based code analysis (`src/analysis/`)
- Code generation with templates and pattern library (`src/generation/`)
- Test generation, refactoring engine, diff engine
- Policy enforcement on generated code (`src/verifier/`)
- Ruleset knowledge ingestion (`src/ingestion/`)

## Entry points

- `api/` — FastAPI service exposing the agent over HTTP
- `cli/` — command-line interface
- `condrox_agent_coder.py` — main agent class (`CondroxAgentCoder`)

## Design rules

- No own intelligence: all reasoning flows through the main brain.
- Generated code passes through the policy enforcer before delivery.
- Ingestion sources are configurable via `CONDROX_RULESET_PATH`.
