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

"""Pre-defined training scenarios per domain."""

from core.training.scenarios.business import BUSINESS_SCENARIOS
from core.training.scenarios.health import HEALTH_SCENARIOS
from core.training.scenarios.public import PUBLIC_SCENARIOS
from core.training.scenarios.school import SCHOOL_SCENARIOS

ALL_SCENARIOS = SCHOOL_SCENARIOS + BUSINESS_SCENARIOS + PUBLIC_SCENARIOS + HEALTH_SCENARIOS

__all__ = [
    "ALL_SCENARIOS",
    "BUSINESS_SCENARIOS",
    "HEALTH_SCENARIOS",
    "PUBLIC_SCENARIOS",
    "SCHOOL_SCENARIOS",
]
