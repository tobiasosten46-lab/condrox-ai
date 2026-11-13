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
Ingest downloaded knowledge from data/knowledge into a persistent
SQLite backend so the Brain can use it on future boots without reloading.

This is the "training" step after datasets have been downloaded: it loads
chunks from disk into the KnowledgeBackend, builds the graph if the store is
small enough, and persists everything to SQLite.

Usage:
    python scripts/ingest_downloaded_knowledge.py
    python scripts/ingest_downloaded_knowledge.py --db Databases/knowledge_ingestion.db
    python scripts/ingest_downloaded_knowledge.py --max-chunks 20000 --max-files 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.brain.brain_config import BrainConfig
from core.brain.brain_controller import BrainController
from core.knowledge.knowledge_loader import KnowledgeLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load downloaded knowledge into a persisted Brain state."""
    parser = argparse.ArgumentParser(
        description="Ingest downloaded knowledge into a persistent SQLite backend."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("CONDROX_KNOWLEDGE_ROOT", r"data/knowledge"),
        help="Knowledge root directory",
    )
    parser.add_argument(
        "--db",
        default="Databases/knowledge_ingestion.db",
        help="SQLite persistence path",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10000,
        help="Maximum chunks to load per domain",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum files to sample per domain",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Build knowledge graph after ingestion (only if <= 5000 items)",
    )
    args = parser.parse_args()

    os.environ["CONDROX_KNOWLEDGE_ROOT"] = str(args.root)
    # Skip the boot-time loader so we can control the ingestion manually
    # and avoid the hot-reloader racing with save_state.
    os.environ["CONDROX_SKIP_KNOWLEDGE_LOAD"] = "1"

    # Make loader tunables visible via env.
    os.environ["CONDROX_KL_MAX_CHUNKS"] = str(args.max_chunks)
    os.environ["CONDROX_KL_MAX_FILES"] = str(args.max_files)

    db_path = Path(args.db)
    if db_path.exists():
        logger.info("Removing existing persistence file %s", db_path)
        db_path.unlink()

    config = BrainConfig(
        enable_single_instance_lock=False,
        persistence_path=str(db_path),
    )

    logger.info("Booting BrainController with persistence=%s", db_path)
    controller = BrainController(config=config)
    controller.boot_system()

    logger.info("Loading knowledge from %s", args.root)
    loader = KnowledgeLoader(
        knowledge_root=args.root,
    )
    loaded = loader.load_into(controller.knowledge_router.backend)
    logger.info("Loaded %d knowledge items from disk", loaded)

    kb_count = controller.knowledge_router.backend.count()
    logger.info("Knowledge items in backend after ingestion: %d", kb_count)

    if args.build_graph and kb_count <= 5000:
        logger.info("Building knowledge graph...")
        try:
            controller.knowledge_router.build_graph()
            logger.info("Graph built.")
        except Exception as exc:
            logger.warning("Graph build failed: %s", exc)

    logger.info("Saving state to %s", args.db)
    saved = controller.save_state()
    logger.info("Saved %d items total (memory + knowledge + profiles + skills)", saved)

    controller.close()
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    logger.info("Total time: %.1f seconds", elapsed)
