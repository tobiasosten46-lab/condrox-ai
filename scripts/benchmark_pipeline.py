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

"""
Benchmark suite for the BrainController pipeline.

Measures per-stage latency, total request latency, memory allocations,
and throughput. Identifies the top bottlenecks.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

from core import BrainController


def benchmark_stage_latency(
    controller: BrainController,
    inputs: List[str],
    warmup: int = 5,
) -> Dict[str, List[float]]:
    """Measure per-stage latency by wrapping each stage callable.

    Complexity: T(n) ∈ O(r * s) for r runs, s stages.
    """
    # Warmup
    for inp in inputs[:warmup]:
        controller.process_request(inp)

    stage_times: Dict[str, List[float]] = {}
    pipeline = controller.pipeline
    original_stages = list(pipeline._stages)

    # Wrap each stage with timing
    wrapped: List = []
    for i, stage in enumerate(original_stages):
        stage_name = getattr(stage, "stage_name", f"stage_{i:02d}")

        def make_wrapper(s, name: str):
            def wrapper(ctrl, state):
                t0 = time.perf_counter()
                result = s(ctrl, state)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if name not in stage_times:
                    stage_times[name] = []
                stage_times[name].append(elapsed_ms)
                return result
            return wrapper

        wrapped.append(make_wrapper(stage, stage_name))

    pipeline._stages = wrapped
    try:
        for inp in inputs:
            controller.process_request(inp)
    finally:
        pipeline._stages = original_stages

    return stage_times


def benchmark_total_latency(
    controller: BrainController,
    inputs: List[str],
    warmup: int = 5,
) -> List[float]:
    """Measure total request latency.

    Complexity: T(n) ∈ O(r) for r runs.
    """
    # Warmup
    for inp in inputs[:warmup]:
        controller.process_request(inp)

    latencies: List[float] = []
    for inp in inputs:
        gc.collect()
        start = time.perf_counter()
        controller.process_request(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

    return latencies


def benchmark_memory(
    controller: BrainController,
    inputs: List[str],
    warmup: int = 3,
) -> List[Tuple[str, float, float]]:
    """Measure memory usage per request.

    Returns list of (input_snippet, current_mb, peak_mb).

    Complexity: T(n) ∈ O(r) for r runs.
    """
    # Warmup
    for inp in inputs[:warmup]:
        controller.process_request(inp)

    results: List[Tuple[str, float, float]] = []
    for inp in inputs:
        gc.collect()
        tracemalloc.start()
        controller.process_request(inp)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append((
            inp[:40],
            current / (1024 * 1024),
            peak / (1024 * 1024),
        ))

    return results


def benchmark_throughput(
    controller: BrainController,
    input_text: str,
    duration_seconds: float = 5.0,
) -> Tuple[int, float]:
    """Measure throughput (requests per second) for a fixed duration.

    Returns (total_requests, rps).

    Complexity: T(n) ∈ O(d * p) for d duration, p pipeline cost.
    """
    gc.collect()
    count = 0
    start = time.perf_counter()
    end_time = start + duration_seconds
    while time.perf_counter() < end_time:
        controller.process_request(input_text)
        count += 1
    elapsed = time.perf_counter() - start
    rps = count / elapsed
    return count, rps


def print_stats(label: str, values: List[float]) -> None:
    """Print statistics for a list of values."""
    if not values:
        print(f"  {label}: no data")
        return
    avg = mean(values)
    med = median(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    mn = min(values)
    mx = max(values)
    print(f"  {label}:")
    print(f"    avg={avg:.3f}ms  median={med:.3f}ms  std={sd:.3f}ms")
    print(f"    min={mn:.3f}ms  max={mx:.3f}ms  n={len(values)}")


def run_benchmark() -> None:
    """Run the full benchmark suite and print results."""
    print("=" * 70)
    print("  Condrox AI v2.0 + Condrox — Pipeline Benchmark")
    print("=" * 70)
    print()

    controller = BrainController()
    controller.boot_system()
    print(f"System: {controller.config.system_name} v{controller.config.version}")
    print(f"Pipeline stages: {controller.pipeline.stage_count()}")
    print()

    # Test inputs — varied intents
    inputs = [
        "Write a Python function that sorts a list.",
        "Explain what a closure is in JavaScript.",
        "What is a Python decorator?",
        "Fix this bug: my function returns None.",
        "Explain Python list comprehensions.",
    ] * 20  # 100 requests total

    # --- 1. Total latency ---
    print("─" * 50)
    print("1. Total Request Latency (100 requests)")
    print("─" * 50)
    latencies = benchmark_total_latency(controller, inputs)
    print_stats("Total latency", latencies)
    print()

    # --- 2. Per-stage latency ---
    print("─" * 50)
    print("2. Per-Stage Latency")
    print("─" * 50)
    stage_times = benchmark_stage_latency(controller, inputs)
    # Sort stages by average latency descending
    stage_avgs = [(name, mean(times)) for name, times in stage_times.items()]
    stage_avgs.sort(key=lambda x: -x[1])
    for name, avg in stage_avgs:
        times = stage_times[name]
        print_stats(f"Stage '{name}'", times)
    print()

    # --- 3. Top 5 bottlenecks ---
    print("─" * 50)
    print("3. Top 5 Bottlenecks (by avg latency)")
    print("─" * 50)
    for i, (name, avg) in enumerate(stage_avgs[:5], 1):
        pct = (avg / mean(latencies)) * 100 if latencies else 0
        print(f"  #{i}: {name} — {avg:.3f}ms ({pct:.1f}% of total)")
    print()

    # --- 4. Memory usage ---
    print("─" * 50)
    print("4. Memory Usage Per Request (5 samples)")
    print("─" * 50)
    mem_results = benchmark_memory(controller, inputs[:5])
    for snippet, current_mb, peak_mb in mem_results:
        print(f"  '{snippet}...': current={current_mb:.3f}MB  peak={peak_mb:.3f}MB")
    avg_peak = mean([r[2] for r in mem_results])
    print(f"  Avg peak memory: {avg_peak:.3f}MB")
    print()

    # --- 5. Throughput ---
    print("─" * 50)
    print("5. Throughput (5 second sustained)")
    print("─" * 50)
    count, rps = benchmark_throughput(
        controller,
        "Write a Python function that sorts a list.",
        duration_seconds=3.0,
    )
    print(f"  Requests: {count}")
    print(f"  Throughput: {rps:.2f} req/s")
    print(f"  Avg latency: {1000.0 / rps:.2f}ms" if rps > 0 else "")
    print()

    # --- 6. Summary ---
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total avg latency: {mean(latencies):.3f}ms")
    print(f"  Total median latency: {median(latencies):.3f}ms")
    print(f"  Throughput: {rps:.2f} req/s")
    print(f"  Avg peak memory: {avg_peak:.3f}MB")
    print(f"  Slowest stage: {stage_avgs[0][0]} ({stage_avgs[0][1]:.3f}ms)" if stage_avgs else "  No stage data")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
