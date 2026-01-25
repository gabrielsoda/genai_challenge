"""
Pytest configuration and shared fixtures
"""

import pytest
from fastapi.testclient import TestClient

from genai_challenge.main import app


@pytest.fixture
def client():
    """FastAPI test client for integration test."""
    return TestClient(app)
