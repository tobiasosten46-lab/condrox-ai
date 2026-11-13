# Copyright 2026 Tobias Østen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Print detailed evidence breakdown from latest report."""
import json
import os
from pathlib import Path

reports_dir = Path(os.environ.get("CONDROX_REPORTS_DIR", "Reports/evidence"))
latest = max(reports_dir.glob("evidence_*.json"), key=lambda p: p.stat().st_mtime)
print(f"Reading: {latest.name}")
print()

with open(latest, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Overall: {data['overall']}  Score: {round(data['score'], 2)}")
print(f"Version: {data['version']}")
print(f"Deviations: {len(data.get('deviations', []))}")
for d in data.get('deviations', []):
    print(f"  - {d}")
print()

print("=== Conversation sessions ===")
for s in data.get('conversation_sessions', []):
    summary = s.get('summary', {})
    metrics = summary.get('metric_averages', {})
    print(f"  {s['name']}: score={summary.get('score')}, status={summary.get('status')}")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    for t in s.get('turns', [])[:2]:
        print(f"    Turn: input={t.get('input','')[:60]}")
        print(f"           output={t.get('output','')[:80]}")
print()

print("=== Reasoning sessions ===")
for s in data.get('reasoning_sessions', []):
    summary = s.get('summary', {})
    metrics = summary.get('metric_averages', {})
    print(f"  {s['name']}: score={summary.get('score')}, status={summary.get('status')}")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    for t in s.get('tasks', [])[:2]:
        print(f"    Task: input={t.get('input','')[:60]}")
        print(f"          output={t.get('output','')[:80]}")
print()

print("=== Agent pipeline sessions ===")
for s in data.get('agent_pipeline_sessions', []):
    summary = s.get('summary', {})
    metrics = summary.get('metric_averages', {})
    print(f"  {s['name']}: score={summary.get('score')}, status={summary.get('status')}")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
print()

print("=== Full system sessions ===")
for s in data.get('full_system_sessions', []):
    summary = s.get('summary', {})
    print(f"  {s['name']}: score={summary.get('score')}, status={summary.get('status')}")
    print(f"    release_status: {summary.get('release_status')}")
    print(f"    critical_issues: {summary.get('critical_issues')}")
    print(f"    conversation_turns: {summary.get('conversation_turns')}")
    print(f"    reasoning_tasks: {summary.get('reasoning_tasks')}")
    print(f"    agent_invocations: {summary.get('agent_invocations')}")
    print(f"    rag_hits: {summary.get('rag_hits')}")
