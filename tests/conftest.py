"""Shared pytest fixtures for the npcpy test suite."""

import threading
from http.server import ThreadingHTTPServer

import pytest

from tests.ollama_mock import MockOllamaHandler


@pytest.fixture
def mock_ollama():
    """Run a mock Ollama server on an ephemeral port and yield its base URL."""
    MockOllamaHandler.requests_seen = []
    MockOllamaHandler.version = "0.35.1"
    MockOllamaHandler.models = ["nimble:latest", "qwen3:1.7b"]
    MockOllamaHandler.fail_status = None
    MockOllamaHandler.fail_body = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def mock_decisions():
    """Run a mock hosted-Decisions server on an ephemeral port and yield its base URL."""
    from tests.decisions_mock import MockDecisionsHandler

    MockDecisionsHandler.requests_seen = []
    MockDecisionsHandler.fail_status = None
    MockDecisionsHandler.fail_body = None
    MockDecisionsHandler.refusal_question = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockDecisionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
