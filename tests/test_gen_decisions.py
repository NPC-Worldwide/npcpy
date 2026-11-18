"""Tests for the hosted Decisions API client (npcpy.gen.decisions)."""

import json

import pytest

from npcpy.ft.system1 import ChoiceResult, NoulResult, ScoreResult, System1Result
from npcpy.gen.decisions import (
    DEFAULT_MODELS,
    PROVIDERS,
    DecisionsClient,
    DecisionsConfig,
    DecisionsError,
    build_payload,
    decisions,
    load_decisions,
    parse_answers,
    resolve_provider,
)

from tests.decisions_mock import MockDecisionsHandler


TICKET = {"ticket": "I was charged twice. Please refund the extra payment."}
QUEUE_CRITERIA = {
    "billing": "Payments and refunds",
    "technical": "Bugs and integrations",
}
SEVERITY = ["Cosmetic", "Workaround available", "Fully blocked"]


# --------------------------------------------------------------------------- #
# provider + payload normalization
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("openai", "openai"),
        ("OAI", "openai"),
        ("gemini", "gemini"),
        ("google", "gemini"),
        ("perplexity", "perplexity"),
        ("pplx", "perplexity"),
    ],
)
def test_resolve_provider_aliases(raw, expected):
    assert resolve_provider(raw) == expected


def test_resolve_provider_rejects_unknown():
    with pytest.raises(DecisionsError):
        resolve_provider("nope")


def test_resolve_provider_env_fallback(monkeypatch):
    monkeypatch.setenv("NPCPY_DECISIONS_PROVIDER", "gemini")
    assert resolve_provider(None) == "gemini"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_default_models_are_set(provider):
    assert DEFAULT_MODELS[provider]


def test_payload_openai_shape():
    body = build_payload(
        "openai",
        "charged twice",
        {"dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA}},
    )
    assert body["model"] == DEFAULT_MODELS["openai"]
    assert body["input"] == "charged twice"
    assert isinstance(body["questions"], list)
    question = body["questions"][0]
    assert question["type"] == "choice"
    assert question["name"] == "dept"
    assert [c["value"] for c in question["choices"]] == list(QUEUE_CRITERIA)


def test_payload_gemini_shape():
    body = build_payload(
        "gemini",
        "charged twice",
        {"dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA}},
    )
    assert body["model"] == DEFAULT_MODELS["gemini"]
    assert body["input"] == "charged twice"
    assert isinstance(body["questions"], dict)
    assert body["questions"]["dept"]["criteria"] == QUEUE_CRITERIA


def test_payload_perplexity_shape_uses_state():
    body = build_payload(
        "perplexity",
        TICKET,
        {"dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA}},
    )
    assert body["state"] == TICKET
    assert isinstance(body["questions"], dict)


def test_payload_maps_noul_to_predicate_for_openai():
    body = build_payload("openai", "text", {"q": {"type": "noul", "instructions": "yes?"}})
    assert body["questions"][0]["type"] == "predicate"


def test_payload_accepts_predicate_alias_everywhere():
    for provider in PROVIDERS:
        body = build_payload(provider, "text", {"q": {"type": "predicate", "instructions": "?"}})
        questions = body["questions"]
        qtype = questions[0]["type"] if isinstance(questions, list) else questions["q"]["type"]
        assert qtype in ("predicate", "noul")


def test_payload_score_levels_are_split_for_openai():
    body = build_payload(
        "openai", "text", {"sev": {"type": "score", "instructions": "?", "criteria": SEVERITY}}
    )
    levels = body["questions"][0]["levels"]
    assert [level["label"] for level in levels] == SEVERITY


def test_payload_json_state_for_openai_uses_input_json_part():
    body = build_payload("openai", TICKET, {"q": {"type": "noul", "instructions": "?"}})
    assert body["input"][0]["content"][0]["type"] == "input_json"


def test_payload_json_state_for_gemini_is_serialized_text():
    body = build_payload("gemini", TICKET, {"q": {"type": "noul", "instructions": "?"}})
    assert json.loads(body["input"]) == TICKET


def test_payload_preserves_dict_score_descriptions():
    body = build_payload(
        "openai",
        "text",
        {
            "sev": {
                "type": "score",
                "instructions": "?",
                "criteria": [
                    {"label": "Low", "description": "minor issue"},
                    {"label": "High", "description": "blocks work"},
                ],
            }
        },
    )
    assert body["questions"][0]["levels"] == [
        {"label": "Low", "description": "minor issue"},
        {"label": "High", "description": "blocks work"},
    ]


def test_payload_normalizes_choice_list_criteria():
    body = build_payload(
        "gemini",
        "text",
        {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}},
    )
    assert body["questions"]["q"]["criteria"] == {"a": "", "b": ""}


def test_payload_noul_true_false_criteria_for_gemini():
    body = build_payload(
        "gemini",
        "text",
        {
            "q": {
                "type": "noul",
                "instructions": "?",
                "criteria": {"true": "broken", "false": "fine"},
            }
        },
    )
    assert body["questions"]["q"]["criteria"] == {
        "true_description": "broken",
        "false_description": "fine",
    }


def test_payload_noul_criteria_folded_into_openai_instructions():
    body = build_payload(
        "openai",
        "text",
        {
            "q": {
                "type": "noul",
                "instructions": "Is it bad?",
                "criteria": {"true": "broken", "false": "fine"},
            }
        },
    )
    instructions = body["questions"][0]["instructions"]
    assert "Is it bad?" in instructions
    assert "Answer yes when: broken" in instructions
    assert "Answer no when: fine" in instructions


def test_payload_rejects_empty_questions():
    with pytest.raises(DecisionsError):
        build_payload("openai", "text", {})


def test_payload_rejects_unknown_question_type():
    with pytest.raises(DecisionsError):
        build_payload("openai", "text", {"q": {"type": "locate", "instructions": "?"}})


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #

def test_parse_answers_map_shape():
    payload = {
        "model": "m",
        "answers": {
            "yes": {"type": "noul", "noul": 0.8},
            "pick": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.9, "technical": 0.1},
            },
            "sev": {
                "type": "score",
                "score": 1.5,
                "confidence": 0.4,
                "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
            },
        },
    }
    result = parse_answers(payload, provider="gemini")
    assert isinstance(result, System1Result)
    assert isinstance(result.answers["yes"], NoulResult)
    assert result.answers["yes"].noul == pytest.approx(0.8)
    assert isinstance(result.answers["pick"], ChoiceResult)
    assert result.answers["pick"].choice == "billing"
    assert isinstance(result.answers["sev"], ScoreResult)
    assert result.answers["sev"].legend == {"0": "low", "1": "mid", "2": "high"}
    assert result.routing["provider"] == "gemini"


def test_parse_answers_list_shape_with_probability_records():
    payload = {
        "answers": [
            {"type": "predicate", "name": "dmg", "probability": 0.42},
            {
                "type": "choice",
                "name": "dept",
                "choice": "billing",
                "confidence": 0.7,
                "probabilities": [
                    {"value": "billing", "probability": 0.7},
                    {"value": "technical", "probability": 0.3},
                ],
            },
            {
                "type": "score",
                "name": "sev",
                "score": 1.1,
                "confidence": 0.55,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    {"value": 1, "label": "Workaround", "probability": 0.7},
                    {"value": 2, "label": "Blocked", "probability": 0.2},
                ],
            },
        ]
    }
    result = parse_answers(payload, provider="openai")
    assert result.answers["dmg"].noul == pytest.approx(0.42)
    assert result.answers["dept"].probabilities == {"billing": 0.7, "technical": 0.3}
    assert result.answers["sev"].legend["2"] == "Blocked"
    assert result.answers["sev"].probabilities == {"0": 0.1, "1": 0.7, "2": 0.2}


def test_parse_answers_raises_on_refusal():
    payload = {"answers": [{"type": "refusal", "name": "q", "refusal": "policy"}]}
    with pytest.raises(DecisionsError, match="refused"):
        parse_answers(payload)


def test_parse_answers_rejects_missing_answers():
    with pytest.raises(DecisionsError):
        parse_answers({"model": "m"})


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def test_config_auth_headers_per_provider(monkeypatch):
    openai_cfg = DecisionsConfig(provider="openai", api_key="sk-1")
    assert openai_cfg.auth_headers()["Authorization"] == "Bearer sk-1"

    gemini_cfg = DecisionsConfig(provider="gemini", api_key="g-1")
    assert gemini_cfg.auth_headers()["x-goog-api-key"] == "g-1"

    pplx_cfg = DecisionsConfig(provider="perplexity", api_key="p-1")
    assert pplx_cfg.auth_headers()["Authorization"] == "Bearer p-1"


def test_config_requires_api_key(monkeypatch):
    for env_name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(DecisionsError, match="No API key"):
        DecisionsConfig(provider="openai").auth_headers()


def test_config_endpoints():
    assert DecisionsConfig(provider="openai").endpoint() == "https://api.openai.com/v1/decisions"
    assert (
        DecisionsConfig(provider="gemini").endpoint()
        == "https://generativelanguage.googleapis.com/v1beta/decisions"
    )
    assert (
        DecisionsConfig(provider="perplexity").endpoint()
        == "https://api.perplexity.ai/v1/decisions"
    )


def test_config_model_env_override(monkeypatch):
    monkeypatch.setenv("NPCPY_DECISIONS_OPENAI_MODEL", "custom-model")
    assert DecisionsConfig(provider="openai").resolved_model() == "custom-model"


# --------------------------------------------------------------------------- #
# client against the mock server
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("provider", PROVIDERS)
def test_client_choice_noul_score(provider, mock_decisions):
    client = DecisionsClient(
        DecisionsConfig(provider=provider, api_key="test-key", base_url=mock_decisions)
    )
    assert client.noul("text", instructions="Did it break?").noul > 0.5
    assert client.choice("text", instructions="Where?", criteria=QUEUE_CRITERIA).choice == "billing"
    severity = client.score("text", instructions="How bad?", criteria=SEVERITY)
    assert severity.score > 0
    assert severity.legend is not None


@pytest.mark.parametrize("provider", PROVIDERS)
def test_client_predict_multi_question(provider, mock_decisions):
    client = DecisionsClient(
        DecisionsConfig(provider=provider, api_key="test-key", base_url=mock_decisions)
    )
    result = client.predict(
        TICKET,
        {
            "refund": {"type": "noul", "instructions": "Refund requested?"},
            "dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA},
            "sev": {"type": "score", "instructions": "How bad?", "criteria": SEVERITY},
        },
    )
    assert set(result.answers) == {"refund", "dept", "sev"}
    assert isinstance(result.answers["refund"], NoulResult)
    assert isinstance(result.answers["dept"], ChoiceResult)
    assert isinstance(result.answers["sev"], ScoreResult)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_client_sends_request_to_decisions_path(provider, mock_decisions):
    client = DecisionsClient(
        DecisionsConfig(provider=provider, api_key="test-key", base_url=mock_decisions)
    )
    client.noul("text", instructions="?")
    posts = [r for r in MockDecisionsHandler.requests_seen if r["path"].endswith("/decisions")]
    assert len(posts) == 1


def test_client_uses_provider_request_shape(mock_decisions):
    DecisionsClient(
        DecisionsConfig(provider="openai", api_key="k", base_url=mock_decisions)
    ).noul("text", instructions="?")
    openai_body = MockDecisionsHandler.requests_seen[-1]["body"]
    assert isinstance(openai_body["questions"], list)
    assert "input" in openai_body

    DecisionsClient(
        DecisionsConfig(provider="perplexity", api_key="k", base_url=mock_decisions)
    ).noul("text", instructions="?")
    pplx_body = MockDecisionsHandler.requests_seen[-1]["body"]
    assert "state" in pplx_body
    assert isinstance(pplx_body["questions"], dict)


def test_client_refusal_raises(mock_decisions):
    MockDecisionsHandler.refusal_question = "q"
    client = DecisionsClient(
        DecisionsConfig(provider="openai", api_key="k", base_url=mock_decisions)
    )
    with pytest.raises(DecisionsError, match="refused"):
        client.noul("text", instructions="?", question_name="q")


def test_client_error_includes_provider_hint(mock_decisions):
    MockDecisionsHandler.fail_status = 401
    MockDecisionsHandler.fail_body = {"error": {"message": "bad key"}}
    client = DecisionsClient(
        DecisionsConfig(provider="gemini", api_key="k", base_url=mock_decisions)
    )
    with pytest.raises(DecisionsError, match="rejected the API key"):
        client.noul("text", instructions="?")


def test_client_400_message_matches_provider_request_shape(mock_decisions):
    MockDecisionsHandler.fail_status = 400
    MockDecisionsHandler.fail_body = {"error": {"message": "bad question"}}
    with pytest.raises(DecisionsError, match="rejected the decisions request"):
        decisions("text", {"q": {"type": "noul", "instructions": "?"}},
                  provider="openai", api_key="k", base_url=mock_decisions)


def test_decisions_returns_raw_payload(mock_decisions):
    payload = decisions(
        "text",
        {"q": {"type": "noul", "instructions": "?"}},
        provider="gemini",
        api_key="k",
        base_url=mock_decisions,
    )
    assert "answers" in payload
    assert payload["model"] == DEFAULT_MODELS["gemini"]


def test_load_decisions_returns_client(mock_decisions):
    client = load_decisions(provider="openai", api_key="k", base_url=mock_decisions)
    assert isinstance(client, DecisionsClient)
    assert client.to_dict()["backend"] == "openai"
