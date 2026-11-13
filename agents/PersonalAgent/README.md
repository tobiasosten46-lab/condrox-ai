# PersonalAgent

Extended personal-assistant agent built on the main brain
(`BrainController`).

An **external agent**: it delegates all cognition to
`BrainController.process_request()` and adds PA-specific tools, workflows,
and quality gates.

## Design rule

PersonalAgent has **no intelligence of its own** — no separate pipeline,
identity, reasoning, or context. Everything flows main brain → agent →
main brain.
