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
Run school-domain training scenarios through the live BrainController.

Produces a TrainingReport showing which scenarios and gates PASS/FAIL.
Use CONDROX_SKIP_KNOWLEDGE_LOAD=1 to bypass disk knowledge loading.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CONDROX_SKIP_KNOWLEDGE_LOAD"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Reconfigure stdout for UTF-8 so mathematical symbols in reports print cleanly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.brain.brain_config import BrainConfig
from core.brain.brain_controller import BrainController
from core.training.training_console import TrainingConsole


def main() -> None:
    """Run school training and print a human-readable report."""
    config = BrainConfig(
        enable_single_instance_lock=False,
        persistence_path=None,
    )
    controller = BrainController(config=config)
    controller.boot_system()

    report = controller.run_training("school")
    report_text = TrainingConsole.format_report(report)
    print(report_text)

    out_path = PROJECT_ROOT / "Report" / "reports" / "school_training_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
        fh.write(f"\n\nSUMMARY: {report.failed}/{report.total} school scenarios failed.\n")
    print("\nReport saved to", out_path, flush=True)

    if report.passed == report.total:
        print("\nAll school scenarios passed quality gates.")
    else:
        print(f"\n{report.failed}/{report.total} school scenarios failed.")


if __name__ == "__main__":
    main()
