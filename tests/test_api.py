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
Tests for the FastAPI REST API.

Tests authentication, rate limiting, CORS, and all endpoints.
Uses TestClient to exercise /health, /process, /metrics, /status, /creative.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Set test environment variables BEFORE importing the app
os.environ["CONDROX_AUTH_ENABLED"] = "1"
os.environ["CONDROX_API_KEY"] = "test-secret-key-12345"
os.environ["CONDROX_RATE_LIMIT"] = "1000/minute"
os.environ["CONDROX_RATE_LIMIT_PROCESS"] = "1000/minute"
os.environ["CONDROX_RATE_LIMIT_CREATIVE"] = "1000/minute"

from api.main import API_KEY, app  # noqa: E402

# Reset API_KEY to match what we set above (import may have cached the old one)
# The import above reads the env var, so API_KEY should be correct.
TEST_API_KEY = API_KEY


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient for the FastAPI app, triggering lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Provide authentication headers with a valid API key."""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def bad_auth_headers() -> dict[str, str]:
    """Provide authentication headers with an invalid API key."""
    return {"X-API-Key": "wrong-key-xyz"}


# â”€â”€â”€ Health Endpoint (no auth required) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200 with health status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"healthy", "unhealthy"}
        assert 0.0 <= data["score"] <= 1.0

    def test_health_works_without_auth(self, client: TestClient) -> None:
        """Health endpoint works without API key (public endpoint)."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_after_process(self, client: TestClient) -> None:
        """Health endpoint still works after processing requests."""
        client.post(
            "/process",
            json={"input": "Explain Python."},
            headers={"X-API-Key": TEST_API_KEY},
        )
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"healthy", "unhealthy"}


# â”€â”€â”€ Metrics Endpoint (no auth required) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_returns_200(self, client: TestClient) -> None:
        """Metrics endpoint returns Prometheus-format text."""
        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text
        assert "condrox_" in text
        assert "condrox_pipeline_stages" in text

    def test_metrics_works_without_auth(self, client: TestClient) -> None:
        """Metrics endpoint works without API key (Prometheus scraping)."""
        response = client.get("/metrics")
        assert response.status_code == 200


# â”€â”€â”€ Process Endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestProcessEndpoint:
    """Tests for the /process endpoint."""

    def test_process_with_valid_auth(self, client: TestClient, auth_headers: dict) -> None:
        """Process endpoint works with valid API key."""
        response = client.post(
            "/process",
            json={"input": "Explain Python."},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["final_output"]
        assert data["mode"] == "explain"
        assert data["request_id"]

    def test_process_without_auth_returns_401(self, client: TestClient) -> None:
        """Process endpoint rejects requests without API key."""
        response = client.post("/process", json={"input": "Explain Python."})
        assert response.status_code == 401
        assert "API key" in response.json()["detail"]

    def test_process_with_invalid_auth_returns_401(self, client: TestClient, bad_auth_headers: dict) -> None:
        """Process endpoint rejects requests with invalid API key."""
        response = client.post(
            "/process",
            json={"input": "Explain Python."},
            headers=bad_auth_headers,
        )
        assert response.status_code == 401
        assert "Invalid" in response.json()["detail"]

    def test_process_empty_input_returns_422(self, client: TestClient, auth_headers: dict) -> None:
        """Empty input is rejected by Pydantic validation."""
        response = client.post(
            "/process",
            json={"input": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_process_increments_request_counter(self, client: TestClient, auth_headers: dict) -> None:
        """Processing a request increments the Prometheus request counter."""
        client.post(
            "/process",
            json={"input": "Explain Python."},
            headers=auth_headers,
        )
        metrics_response = client.get("/metrics")
        text = metrics_response.text
        assert "condrox_requests_total" in text


# â”€â”€â”€ Status Endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_with_valid_auth(self, client: TestClient, auth_headers: dict) -> None:
        """Status endpoint returns full system snapshot with valid auth."""
        response = client.get("/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_stages"] == 20
        assert "state_query_handler" in data
        assert "priority_engine" in data

    def test_status_without_auth_returns_401(self, client: TestClient) -> None:
        """Status endpoint rejects requests without API key."""
        response = client.get("/status")
        assert response.status_code == 401


# â”€â”€â”€ Creative Endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestCreativeEndpoint:
    """Tests for the /creative endpoint."""

    def test_creative_with_valid_auth(self, client: TestClient, auth_headers: dict) -> None:
        """Creative endpoint returns a creative package with valid auth."""
        response = client.post(
            "/creative",
            json={"input": "I'm so happy! I just graduated!", "include_lyrics": True},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["emotion"] == "happy"
        assert data["genre"] == "indie pop"
        assert data["tempo_bpm"] > 100
        assert data["music_prompt"]
        assert data["style_tags"]
        assert data["lyrics"]  # include_lyrics=True
        assert data["negative_prompt"]
        assert data["request_id"]

    def test_creative_without_lyrics(self, client: TestClient, auth_headers: dict) -> None:
        """Creative endpoint respects include_lyrics=False."""
        response = client.post(
            "/creative",
            json={"input": "I'm sad. My dog died.", "include_lyrics": False},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["emotion"] == "sad"
        assert data["genre"] == "ambient"
        assert data["lyrics"] == ""

    def test_creative_without_auth_returns_401(self, client: TestClient) -> None:
        """Creative endpoint rejects requests without API key."""
        response = client.post(
            "/creative",
            json={"input": "I'm happy!"},
        )
        assert response.status_code == 401

    def test_creative_sad_input(self, client: TestClient, auth_headers: dict) -> None:
        """Creative endpoint maps sad emotion to ambient palette."""
        response = client.post(
            "/creative",
            json={"input": "I lost my father. I'm devastated."},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["emotion"] == "sad"
        assert data["genre"] == "ambient"
        assert data["energy"] < 0.3
        assert data["valence"] < 0.2

    def test_creative_angry_input(self, client: TestClient, auth_headers: dict) -> None:
        """Creative endpoint maps angry emotion to industrial metal."""
        response = client.post(
            "/creative",
            json={"input": "I'm furious! Someone stole my work!"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["emotion"] == "angry"
        assert data["genre"] == "industrial metal"
        assert data["energy"] > 0.8


# â”€â”€â”€ Security & Edge Case Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAPISecurity:
    """Security-focused API tests."""

    def test_missing_api_key_header_returns_401(self, client: TestClient) -> None:
        """Request without X-API-Key header returns 401."""
        response = client.post("/process", json={"input": "test"})
        assert response.status_code == 401

    def test_empty_api_key_returns_401(self, client: TestClient) -> None:
        """Empty API key string returns 401."""
        response = client.post(
            "/process",
            json={"input": "test"},
            headers={"X-API-Key": ""},
        )
        assert response.status_code == 401

    def test_wrong_api_key_format_returns_401(self, client: TestClient) -> None:
        """Random string API key returns 401."""
        response = client.post(
            "/process",
            json={"input": "test"},
            headers={"X-API-Key": "completely-wrong-key-xyz"},
        )
        assert response.status_code == 401

    def test_api_key_in_wrong_header_returns_401(self, client: TestClient) -> None:
        """API key in wrong header (Authorization) returns 401."""
        response = client.post(
            "/process",
            json={"input": "test"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert response.status_code == 401

    def test_health_does_not_require_auth(self, client: TestClient) -> None:
        """Health endpoint is publicly accessible without auth."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_metrics_does_not_require_auth(self, client: TestClient) -> None:
        """Metrics endpoint is publicly accessible for Prometheus scraping."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_status_requires_auth(self, client: TestClient) -> None:
        """Status endpoint requires authentication."""
        response = client.get("/status")
        assert response.status_code == 401

    def test_creative_requires_auth(self, client: TestClient) -> None:
        """Creative endpoint requires authentication."""
        response = client.post("/creative", json={"input": "test"})
        assert response.status_code == 401

    def test_auth_failure_counted_in_metrics(self, client: TestClient) -> None:
        """Authentication failures are counted in Prometheus metrics."""
        # Make a failed auth request
        client.post("/process", json={"input": "test"})
        # Check metrics
        metrics_response = client.get("/metrics")
        text = metrics_response.text
        assert "condrox_auth_failures_total" in text

    def test_process_with_oversized_input_returns_422(self, client: TestClient, auth_headers: dict) -> None:
        """Input exceeding max_length (4096) is rejected."""
        response = client.post(
            "/process",
            json={"input": "x" * 5000},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_creative_with_oversized_input_returns_422(self, client: TestClient, auth_headers: dict) -> None:
        """Creative input exceeding max_length is rejected."""
        response = client.post(
            "/creative",
            json={"input": "x" * 5000},
            headers=auth_headers,
        )
        assert response.status_code == 422
