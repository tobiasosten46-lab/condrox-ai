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

"""AI-OS real runtime — boots full system with services, channels, and live monitoring."""

from __future__ import annotations

import os
import time

os.environ.setdefault("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")

from core.brain.brain_controller import BrainController


def main() -> None:
    sep = "=" * 64
    print(sep)
    print("  CONDROX AI-OS — REAL RUNTIME")
    print(sep)

    bc = BrainController()

    print("\n[1] Booting full system (BrainController + AIOSKernel)...")
    t0 = time.time()
    bc.boot_system()
    boot_time = time.time() - t0
    print(f"    Boot complete in {boot_time:.3f}s")

    status = bc.aios_kernel.get_status()

    print(f"\n[2] AI-OS Kernel:")
    print(f"    Booted:       {status['booted']}")
    print(f"    Boot time:    {status['boot_time']}s")
    print(f"    Events total: {status['events']['total']}")

    print(f"\n[3] Registered Services ({len(status['services'])}):")
    for svc in status["services"]:
        state = svc.get("state", "unknown")
        name = svc.get("name", "?")
        print(f"    {name:20s}  state={state}")

    print(f"\n[4] Health:")
    h = status["health"]
    print(f"    Overall score:    {h['overall_score']}")
    print(f"    Alert count:      {h['alert_count']}")
    print(f"    Degraded services:{h['degraded_services']}")

    print(f"\n[5] Data Flow Channels ({status['data_flow']['channels']}):")
    for ch_name, ch_info in status["data_flow"]["channel_metrics"].items():
        print(f"    {ch_name:12s}  state={ch_info['state']}  size={ch_info['size']}/{ch_info['capacity']}")
    print(f"    Routing rules:    {status['data_flow']['routing_rules']}")
    print(f"    Router running:   {status['data_flow']['router_running']}")

    print(f"\n[6] Subsystem Status:")
    print(f"    Monitor:        {status['monitor']['running']} (interval={status['monitor']['interval']}s)")
    print(f"    Task queue:     {status['task_queue']['running']}")
    print(f"    Self-repair:    {status['self_repair']['breaker_count']} breakers, {status['self_repair']['open_circuits']} open")
    print(f"    Hot reloader:   {status['hot_reloader']}")
    print(f"    State manager:  {status['state']['state_keys']} keys, last_ckpt={status['state']['last_checkpoint']}")

    print(f"\n[7] Recent Events ({len(status['events']['recent'])}):")
    for e in status["events"]["recent"][:8]:
        print(f"    [{e['severity']:5s}] {e['source']:20s} {e['message']}")

    print(f"\n[8] Processing requests through the 18-stage pipeline...")
    queries = [
        "What is the capital of France?",
        "Explain how recursion works.",
        "Write a Python function to reverse a list.",
    ]
    for q in queries:
        t0 = time.time()
        result = bc.process_request(q)
        dt = time.time() - t0
        out = result.final_output
        if len(out) > 100:
            out = out[:100] + "..."
        print(f"    [{dt:.3f}s] intent={result.intent:12s} mode={result.mode:10s}")
        print(f"           Q: {q}")
        print(f"           A: {out}")

    print(f"\n[9] Post-processing health:")
    health = bc.aios_kernel.get_health()
    if health:
        print(f"    Overall score: {health.overall_score}")
        print(f"    Alerts:        {health.alert_count}")
        svc_health = health.service_health
        for name, (ok, msg) in svc_health.items():
            print(f"    {name:20s} healthy={ok}  msg={msg}")

    print(f"\n[10] Data flow metrics after processing:")
    for m in bc.aios_kernel.data_flow.get_all_metrics():
        print(f"    {m.channel_name:12s}  put={m.total_put}  get={m.total_get}  dropped={m.total_dropped}  latency={m.avg_latency_ms}ms")

    print(f"\n[11] Event log (last 10):")
    events = bc.aios_kernel.get_events(limit=10)
    for e in events:
        print(f"    [{e.severity:5s}] {e.source:20s} {e.message}")

    print(f"\n[12] Graceful shutdown...")
    t0 = time.time()
    bc.close()
    dt = time.time() - t0
    print(f"    Shutdown complete in {dt:.3f}s")
    print(f"    AIOSKernel booted: {bc.aios_kernel.is_booted}")

    print(f"\n{sep}")
    print("  CONDROX AI-OS — RUNTIME COMPLETE")
    print(sep)


if __name__ == "__main__":
    main()
