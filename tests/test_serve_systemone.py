"""End-to-end tests: Flask serve endpoints backed by the Ollama System One client."""

import pytest

from tests.ollama_mock import MockOllamaHandler


TICKET = {"ticket": "I was charged twice. Please refund the extra payment."}
QUEUE_CRITERIA = {
    "billing": "Payments and refunds",
    "technical": "Bugs and integrations",
    "other": "None of the above",
}


@pytest.fixture
def client():
    from npcpy.serve import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_system1_predict_endpoint_with_ollama(client, mock_ollama):
    response = client.post(
        "/api/system1/predict",
        json={
            "state": TICKET,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": QUEUE_CRITERIA,
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Does the customer explicitly ask for a refund?",
                },
            },
            "backend": "ollama",
            "model": "nimble",
            "base_url": mock_ollama,
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["answers"]["team"]["choice"] == "billing"
    assert data["answers"]["refund"]["noul"] == pytest.approx(0.997)
    posts = [r for r in MockOllamaHandler.requests_seen if r["path"] == "/v1/systemone"]
    assert len(posts) == 1


def test_system1_choice_endpoint_with_ollama(client, mock_ollama):
    response = client.post(
        "/api/system1/choice",
        json={
            "state": TICKET,
            "instructions": "Which team should handle this ticket?",
            "criteria": QUEUE_CRITERIA,
            "backend": "ollama",
            "base_url": mock_ollama,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["choice"] == "billing"


def test_system1_score_endpoint_with_ollama(client, mock_ollama):
    response = client.post(
        "/api/system1/score",
        json={
            "state": TICKET,
            "instructions": "How urgent is this ticket?",
            "criteria": ["Routine", "Soon", "Urgent"],
            "backend": "ollama",
            "base_url": mock_ollama,
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["score"] == pytest.approx(0.815)
    assert body["legend"]["2"] == "Urgent"


def test_system1_noul_endpoint_with_ollama(client, mock_ollama):
    response = client.post(
        "/api/system1/noul",
        json={
            "state": TICKET,
            "instructions": "Does the customer explicitly ask for a refund?",
            "backend": "ollama",
            "base_url": mock_ollama,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["noul"] == pytest.approx(0.997)


def test_decision_choice_endpoint_with_ollama(client, mock_ollama):
    response = client.post(
        "/api/decision/choice",
        json={
            "state": TICKET,
            "instructions": "Which team should handle this ticket?",
            "criteria": QUEUE_CRITERIA,
            "backend": "ollama",
            "model": "nimble",
            "base_url": mock_ollama,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["choice"] == "billing"


def test_decision_predict_endpoint_with_ollama(client, mock_ollama):
    response = client.post(
        "/api/decision/predict",
        json={
            "state": TICKET,
            "questions": [
                {
                    "name": "team",
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": QUEUE_CRITERIA,
                }
            ],
            "backend": "ollama",
            "base_url": mock_ollama,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["answers"]["team"]["choice"] == "billing"


def test_system1_endpoint_reports_unavailable_backend(client):
    response = client.post(
        "/api/system1/predict",
        json={
            "state": TICKET,
            "questions": {"refund": {"type": "noul", "instructions": "Refund?"}},
            "backend": "ollama",
            "base_url": "http://127.0.0.1:59997",
        },
    )
    assert response.status_code == 500
    assert "ollama serve" in response.get_json()["error"]
