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
Single Instance Lock — ensures only one Condrox pipeline runs at a time
per process.

Uses a threading.Lock for thread-level safety and a named mutex (Windows)
or file lock (Linux) for process-level safety.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional


class SingleInstanceLock:
    """
    Reentrant lock that prevents concurrent pipeline execution.

    Thread-level: a `threading.RLock` ensures only one thread runs the
    pipeline at a time.

    Process-level: on Windows, a named mutex is used; on Linux, a file lock
    via `fcntl`. If the platform lock cannot be acquired, `acquire()` returns
    False.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_name", "_thread_lock", "_owned", "_os_lock", "_os_handle")

    def __init__(self, name: str = "condrox_single_instance") -> None:
        """Initialize the lock.

        Args:
            name: Unique name for the process-level lock.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._name = name
        self._thread_lock = threading.RLock()
        self._owned = False
        self._os_lock: Optional[object] = None
        self._os_handle: Optional[object] = None

    def _acquire_os_lock(self, blocking: bool) -> bool:
        """Acquire the OS-level lock.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                # Create or open a named mutex.
                kernel32 = ctypes.windll.kernel32
                mutex_name = f"Global\\{self._name}"
                handle = kernel32.CreateMutexW(None, False, mutex_name)
                if not handle:
                    return False
                # Check if already exists
                last_error = kernel32.GetLastError()
                if last_error == 183:  # ERROR_ALREADY_EXISTS
                    kernel32.CloseHandle(handle)
                    return False
                self._os_handle = handle
                return True
            except Exception:
                return False
        else:
            try:
                import fcntl

                lock_path = f"/tmp/{self._name}.lock"
                fh = open(lock_path, "w")
                flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(fh.fileno(), flags)
                self._os_lock = fh
                return True
            except Exception:
                return False

    def _release_os_lock(self) -> None:
        """Release the OS-level lock.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self._os_handle is not None:
            try:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(self._os_handle)
            except Exception:
                pass
            self._os_handle = None
        if self._os_lock is not None:
            try:
                import fcntl

                fcntl.flock(self._os_lock.fileno(), fcntl.LOCK_UN)
                self._os_lock.close()
            except Exception:
                pass
            self._os_lock = None

    def acquire(self, blocking: bool = False) -> bool:
        """Acquire both thread and OS-level locks.

        Args:
            blocking: If True, wait for the thread lock. OS lock is always
                      non-blocking (returns False if another process holds it).

        Returns:
            True if both locks were acquired, False otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not self._thread_lock.acquire(blocking=blocking):
            return False
        if not self._acquire_os_lock(blocking=False):
            self._thread_lock.release()
            return False
        self._owned = True
        return True

    def release(self) -> None:
        """Release both thread and OS-level locks.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self._owned:
            self._release_os_lock()
            self._thread_lock.release()
            self._owned = False

    def is_locked(self) -> bool:
        """Return True if the lock is currently held.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._owned

    def __enter__(self) -> bool:
        """Context manager entry. Non-blocking acquire.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self.acquire(blocking=False)

    def __exit__(self, *exc: object) -> None:
        """Context manager exit.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.release()
