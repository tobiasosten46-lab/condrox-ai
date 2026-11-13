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
Tests for admin API endpoints.

Tests CRUD operations for memory and knowledge, pipeline info,
agents listing, config viewing, and state save/load.

Run with:
    $env:PYTHONPATH = "."; python -m pytest tests/test_admin_api.py -v --tb=short
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["CONDROX_AUTH_ENABLED"] = "1"
os.environ["CONDROX_API_KEY"] = "test-secret-key-12345"
os.environ["CONDROX_RATE_LIMIT"] = "1000/minute"
os.environ["CONDROX_RATE_LIMIT_PROCESS"] = "1000/minute"
os.environ["CONDROX_RATE_LIMIT_CREATIVE"] = "1000/minute"

from api.main import API_KEY, app  # noqa: E402

TEST_API_KEY = API_KEY


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Provide authentication headers."""
    return {"X-API-Key": TEST_API_KEY}


# â”€â”€â”€ Admin Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminHealth:
    """Tests for GET /admin/health."""

    def test_admin_health(self, client: TestClient, auth_headers: dict) -> None:
        """Admin health returns detailed health info."""
        response = client.get("/admin/health", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "aggregate_score" in data
        assert "subsystem" in data


# â”€â”€â”€ Admin Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminConfig:
    """Tests for GET /admin/config."""

    def test_get_config(self, client: TestClient, auth_headers: dict) -> None:
        """Config endpoint returns configuration."""
        response = client.get("/admin/config", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "3.4.0"
        assert data["pipeline_stages"] == 20
        assert "max_input_length" in data


# â”€â”€â”€ Memory CRUD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminMemory:
    """Tests for memory CRUD endpoints."""

    def test_list_memory_empty(self, client: TestClient, auth_headers: dict) -> None:
        """List memory returns empty initially."""
        response = client.get("/admin/memory", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["items"] == []

    def test_add_and_list_memory(self, client: TestClient, auth_headers: dict) -> None:
        """Add memory then list shows it."""
        # Add
        response = client.post(
            "/admin/memory",
            json={"content": "Test memory item", "source": "test", "tags": ["unit"]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        item_id = response.json()["item_id"]

        # List
        response = client.get("/admin/memory", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1
        assert any(item["item_id"] == item_id for item in data["items"])

    def test_search_memory(self, client: TestClient, auth_headers: dict) -> None:
        """Search memory returns results."""
        client.post(
            "/admin/memory",
            json={"content": "Python programming language"},
            headers=auth_headers,
        )
        response = client.post(
            "/admin/memory/search",
            json={"query": "Python", "top_k": 5},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1

    def test_delete_memory(self, client: TestClient, auth_headers: dict) -> None:
        """Delete memory removes the item."""
        # Add
        response = client.post(
            "/admin/memory",
            json={"content": "To be deleted"},
            headers=auth_headers,
        )
        item_id = response.json()["item_id"]

        # Delete
        response = client.delete(f"/admin/memory/{item_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["message"] == "Memory deleted"

        # Verify gone
        response = client.get("/admin/memory", headers=auth_headers)
        assert not any(item["item_id"] == item_id for item in response.json()["items"])

    def test_delete_memory_not_found(self, client: TestClient, auth_headers: dict) -> None:
        """Delete non-existent memory returns 404."""
        response = client.delete("/admin/memory/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    def test_clear_all_memory(self, client: TestClient, auth_headers: dict) -> None:
        """Clear all memory removes everything."""
        client.post(
            "/admin/memory",
            json={"content": "Item 1"},
            headers=auth_headers,
        )
        client.post(
            "/admin/memory",
            json={"content": "Item 2"},
            headers=auth_headers,
        )
        response = client.delete("/admin/memory", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["cleared"] >= 2

        response = client.get("/admin/memory", headers=auth_headers)
        assert response.json()["count"] == 0


# â”€â”€â”€ Knowledge CRUD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminKnowledge:
    """Tests for knowledge CRUD endpoints."""

    def test_list_knowledge_initial(self, client: TestClient, auth_headers: dict) -> None:
        """List knowledge returns valid response initially."""
        response = client.get("/admin/knowledge", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "items" in data

    def test_ingest_and_list_knowledge(self, client: TestClient, auth_headers: dict) -> None:
        """Ingest knowledge then list shows it."""
        response = client.post(
            "/admin/knowledge",
            json={"content": "Python uses def for functions.", "domain": "programming"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        item_id = response.json()["item_id"]

        response = client.get("/admin/knowledge", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1
        assert any(item["item_id"] == item_id for item in data["items"])

    def test_search_knowledge(self, client: TestClient, auth_headers: dict) -> None:
        """Search knowledge returns results."""
        client.post(
            "/admin/knowledge",
            json={"content": "Machine learning models require training data."},
            headers=auth_headers,
        )
        response = client.post(
            "/admin/knowledge/search",
            json={"query": "machine learning", "top_k": 5},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1

    def test_delete_knowledge(self, client: TestClient, auth_headers: dict) -> None:
        """Delete knowledge removes the item."""
        response = client.post(
            "/admin/knowledge",
            json={"content": "To be deleted knowledge"},
            headers=auth_headers,
        )
        item_id = response.json()["item_id"]

        response = client.delete(f"/admin/knowledge/{item_id}", headers=auth_headers)
        assert response.status_code == 200

        response = client.get("/admin/knowledge", headers=auth_headers)
        assert not any(item["item_id"] == item_id for item in response.json()["items"])

    def test_delete_knowledge_not_found(self, client: TestClient, auth_headers: dict) -> None:
        """Delete non-existent knowledge returns 404."""
        response = client.delete("/admin/knowledge/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    def test_clear_all_knowledge(self, client: TestClient, auth_headers: dict) -> None:
        """Clear all knowledge removes everything."""
        client.post(
            "/admin/knowledge",
            json={"content": "Knowledge 1"},
            headers=auth_headers,
        )
        response = client.delete("/admin/knowledge", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["cleared"] >= 1


# â”€â”€â”€ Agents â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminAgents:
    """Tests for GET /admin/agents."""

    def test_list_agents(self, client: TestClient, auth_headers: dict) -> None:
        """List agents returns registered agents."""
        response = client.get("/admin/agents", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 7
        assert "code" in data["modes"]
        assert "analyze" in data["modes"]


# â”€â”€â”€ Pipeline Info â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminPipeline:
    """Tests for GET /admin/pipeline."""

    def test_pipeline_info(self, client: TestClient, auth_headers: dict) -> None:
        """Pipeline info returns stage count and names."""
        response = client.get("/admin/pipeline", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["stage_count"] == 20
        assert len(data["stages"]) == 20


# â”€â”€â”€ State Save/Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminStateSaveLoad:
    """Tests for state save/load endpoints."""

    def test_save_state_no_backend(self, client: TestClient, auth_headers: dict) -> None:
        """Save state returns 0 when no persistence backend configured."""
        response = client.post("/admin/state/save", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["items_saved"] == 0

    def test_load_state_no_backend(self, client: TestClient, auth_headers: dict) -> None:
        """Load state returns 400 when no persistence backend configured."""
        response = client.post("/admin/state/load", headers=auth_headers)
        assert response.status_code == 400


# â”€â”€â”€ Auth Required â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAdminAuthRequired:
    """Tests that admin endpoints require authentication."""

    @pytest.mark.parametrize("endpoint", [
        "/admin/health",
        "/admin/config",
        "/admin/memory",
        "/admin/knowledge",
        "/admin/agents",
        "/admin/pipeline",
    ])
    def test_admin_requires_auth(self, client: TestClient, endpoint: str) -> None:
        """Admin endpoints return 401 without API key."""
        response = client.get(endpoint)
        assert response.status_code == 401
