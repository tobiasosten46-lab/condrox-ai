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

"""Word Identity System for Condrox AI.

Provides semantic word lookup, intent classification, emotion detection,
and domain routing based on structured word identity data.

Modules:
    WordIdentityEngine: Main engine class for word lookup and classification.
    WordIdentity: Frozen dataclass for a single word's identity record.
    ClassificationResult: Frozen dataclass for classification output.
"""

from core.word_identity.word_identity_engine import (
    ClassificationResult,
    WordIdentity,
    WordIdentityEngine,
)

__all__ = ["WordIdentityEngine", "WordIdentity", "ClassificationResult"]
