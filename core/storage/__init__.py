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
Storage backend abstraction for persistence.

Provides a pluggable interface for saving and loading memory and knowledge
items. The default implementation uses SQLite for zero-dependency persistence.

Mathematical Foundation:
- Serialization is O(n) for n items.
- Deserialization is O(n) for n items.
"""

from __future__ import annotations

from core.storage.storage_interface import StorageBackend
from core.storage.sqlite_storage import SQLiteStorage

__all__ = ["StorageBackend", "SQLiteStorage"]
