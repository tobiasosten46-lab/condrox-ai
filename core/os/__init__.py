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
AI-OS — Operating System layer for Condrox AI.

Provides boot management, service lifecycle, event bus, monitoring,
state persistence, task scheduling, self-repair, and knowledge hot-reload
for the entire AI system.

Components:
- AIOSKernel: Central orchestrator
- BootManager: Staged boot with progress tracking
- ServiceManager: Service lifecycle with dependency resolution
- EventBus: Pub/sub event system with ring buffer log
- MonitorLoop: Always-on background health monitoring
- StateManager: State checkpointing and recovery
- TaskQueue: Priority task queue with scheduler
- SelfRepairManager: Circuit breakers and auto-recovery
- KnowledgeHotReloader: Incremental knowledge file watcher

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from core.os.aios_kernel import AIOSKernel
from core.os.boot_manager import BootManager, BootProgress, BootStage
from core.os.data_flow_manager import ChannelState, DataChannel, DataFlowManager, FlowMetrics, RoutingRule
from core.os.event_bus import AIEvent, EventBus, EventType
from core.os.knowledge_hot_reload import FileTrackingRecord, KnowledgeHotReloader
from core.os.monitor_loop import MonitorLoop, SystemHealth
from core.os.self_repair import CircuitBreaker, CircuitState, RecoveryAction, SelfRepairManager
from core.os.service_manager import ServiceDescriptor, ServiceManager, ServiceState
from core.os.state_manager import StateManager, StateSnapshot
from core.os.task_queue import Task, TaskQueue, TaskState

__all__ = [
    "AIOSKernel",
    "BootManager",
    "BootProgress",
    "BootStage",
    "ChannelState",
    "DataChannel",
    "DataFlowManager",
    "FlowMetrics",
    "RoutingRule",
    "AIEvent",
    "EventBus",
    "EventType",
    "FileTrackingRecord",
    "KnowledgeHotReloader",
    "MonitorLoop",
    "SystemHealth",
    "CircuitBreaker",
    "CircuitState",
    "RecoveryAction",
    "SelfRepairManager",
    "ServiceDescriptor",
    "ServiceManager",
    "ServiceState",
    "StateManager",
    "StateSnapshot",
    "Task",
    "TaskQueue",
    "TaskState",
]
