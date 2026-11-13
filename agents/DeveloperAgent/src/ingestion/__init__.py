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
DeveloperAgent ingestion subpackage — ruleset knowledge inheritance.

Ingests and upgrades knowledge from a ruleset directory (Condrox AI)
into the DeveloperAgent knowledge backend.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from agents.DeveloperAgent.src.ingestion.ruleset_ingestion import (
    IngestionResult,
    IngestionSource,
    IngestionStats,
    RulesetIngester,
)

__all__ = [
    "IngestionResult",
    "IngestionSource",
    "IngestionStats",
    "RulesetIngester",
]
