"""Decisions APIs: fast typed decisions from hosted decision models.

The hosted counterpart to :mod:`npcpy.gen.systemone` (which speaks to a local
Ollama decision model). OpenAI, Gemini, and Perplexity all expose a decision
endpoint that takes shared evidence plus named questions and returns typed
answers with probabilities instead of generated text. Each spells the request
differently; this module normalizes all three onto the surface used by
:mod:`npcpy.ft.system1` and :mod:`npcpy.gen.decision`:

* question types are always ``noul`` / ``choice`` / ``score``
  (``predicate`` is accepted because that is OpenAI's name for ``noul``),
* results are always :class:`~npcpy.ft.system1.ChoiceResult`,
  :class:`~npcpy.ft.system1.NoulResult`, or
  :class:`~npcpy.ft.system1.ScoreResult`,
* every client exposes ``choice`` / ``noul`` / ``score`` / ``predict``.

The provider and model are always explicit. API keys come from the same
environment variables the rest of npcpy uses.

Example
-------
>>> from npcpy.gen.decisions import DecisionsClient
>>> client = DecisionsClient(provider="gemini", model="gemini-decision-1")
>>> client.choice(
...     "I was charged twice for my order.",
...     instructions="Which department should handle this?",
...     criteria={"billing": "Payments and refunds.", "technical": "Bugs."},
... ).choice
'billing'
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

from npcpy.ft.system1 import (
    ChoiceResult,
    NoulResult,
    ScoreResult,
    System1Result,
)

PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "api_url": "https://api.openai.com/v1",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "api_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "perplexity": {
        "api_key_env": "PERPLEXITY_API_KEY",
        "api_url": "https://api.perplexity.ai/v1",
    },
}

DEFAULT_TIMEOUT = 60

# ``predicate`` is OpenAI's spelling of a yes/no question.
_QUESTION_TYPES = {"noul", "predicate", "choice", "score"}


class DecisionsError(RuntimeError):
    """Raised when a hosted Decisions API request fails or is misconfigured."""


def _check_provider(provider: str) -> str:
    if provider not in PROVIDERS:
        raise DecisionsError(
            f"Unknown Decisions API provider: {provider!r}. "
            f"Expected one of: {', '.join(sorted(PROVIDERS))}."
        )
    return provider


def _check_question_type(question_type: str) -> str:
    if question_type not in _QUESTION_TYPES:
        raise DecisionsError(
            f"Unknown question type: {question_type!r}. "
            f"Expected one of: {', '.join(sorted(_QUESTION_TYPES))}."
        )
    return question_type


# --------------------------------------------------------------------------- #
# question normalization
# --------------------------------------------------------------------------- #

def _normalize_questions(
    questions: Union[Dict[str, Any], List[Any]],
) -> Dict[str, Dict[str, Any]]:
    """Coerce questions into the canonical ``{name: {type, instructions, criteria}}`` shape."""
    if isinstance(questions, list):
        normalized: Dict[str, Dict[str, Any]] = {}
        for index, question in enumerate(questions):
            name = (
                question.get("name")
                if isinstance(question, dict)
                else getattr(question, "name", None)
            ) or f"question_{index}"
            normalized[str(name)] = _normalize_question(question)
    elif isinstance(questions, dict):
        normalized = {
            str(name): _normalize_question(question)
            for name, question in questions.items()
        }
    else:
        raise DecisionsError(
            "questions must be a mapping of name -> question or a list of questions"
        )
    if not normalized:
        raise DecisionsError("questions must contain at least one question")
    return normalized


def _normalize_question(question: Any) -> Dict[str, Any]:
    if isinstance(question, dict):
        qtype = question.get("type", "choice")
        instructions = question.get("instructions", "")
        criteria = question.get("criteria")
    else:
        qtype = getattr(question, "type", "choice")
        instructions = getattr(question, "instructions", "")
        criteria = getattr(question, "criteria", None)

    qtype = _check_question_type(str(qtype).strip().lower())
    payload: Dict[str, Any] = {"type": qtype, "instructions": instructions or ""}

    if criteria is None:
        if qtype == "choice":
            raise DecisionsError(
                f"choice question {instructions!r} requires criteria: a mapping of "
                f"label -> description."
            )
        if qtype == "score":
            raise DecisionsError(
                f"score question {instructions!r} requires criteria: an ordered list "
                f"of level descriptions."
            )
        return payload

    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise DecisionsError("choice criteria must be a mapping of label -> description")
        payload["criteria"] = {str(k): "" if v is None else str(v) for k, v in criteria.items()}
    elif qtype == "score":
        if not isinstance(criteria, list):
            raise DecisionsError("score criteria must be an ordered list of level descriptions")
        payload["criteria"] = [str(level) for level in criteria]
    else:
        if not isinstance(criteria, dict):
            raise DecisionsError(
                "noul criteria must be a mapping like {'true': '...', 'false': '...'}"
            )
        payload["criteria"] = criteria
    return payload


# --------------------------------------------------------------------------- #
# per-provider request building
# --------------------------------------------------------------------------- #

def _openai_input(state: Union[str, Dict[str, Any], List[Any]]) -> Any:
    if isinstance(state, str):
        return state
    if isinstance(state, list) and state and all(
        isinstance(item, dict) and "role" in item for item in state
    ):
        return state
    return [{"role": "user", "content": [{"type": "input_json", "data": state}]}]


def _gemini_input(state: Union[str, Dict[str, Any], List[Any]]) -> Any:
    if isinstance(state, str):
        return state
    if isinstance(state, list) and state and all(
        isinstance(item, dict) and "type" in item for item in state
    ):
        return state
    # Gemini reads structured evidence as text.
    return json.dumps(state, ensure_ascii=False, default=str)


def _openai_question(name: str, question: Dict[str, Any]) -> Dict[str, Any]:
    qtype = question["type"]
    instructions = question["instructions"]
    criteria = question.get("criteria")

    if qtype in ("noul", "predicate"):
        # OpenAI predicates take no criteria; fold the definitions into the prompt.
        if isinstance(criteria, dict):
            hints = []
            if criteria.get("true"):
                hints.append(f"Answer yes when: {criteria['true']}")
            if criteria.get("false"):
                hints.append(f"Answer no when: {criteria['false']}")
            if hints:
                instructions = "\n\n".join([instructions] + hints)
        return {"type": "predicate", "name": name, "instructions": instructions}

    if qtype == "choice":
        return {
            "type": "choice",
            "name": name,
            "instructions": instructions,
            "choices": [
                {"value": label, "description": description}
                for label, description in criteria.items()
            ],
        }

    levels = []
    for level in criteria:
        label, sep, description = level.partition(":")
        levels.append(
            {
                "label": label.strip(),
                "description": description.strip() if sep else label.strip(),
            }
        )
    return {"type": "score", "name": name, "instructions": instructions, "levels": levels}


def _gemini_question(question: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": question["type"],
        "instructions": question["instructions"],
    }
    criteria = question.get("criteria")
    if criteria is not None:
        if question["type"] == "noul" and isinstance(criteria, dict):
            criteria = {
                "true_description": criteria.get("true", ""),
                "false_description": criteria.get("false", ""),
            }
        payload["criteria"] = criteria
    return payload


def _perplexity_question(question: Dict[str, Any]) -> Dict[str, Any]:
    # Perplexity documents the same shape this module uses internally.
    payload: Dict[str, Any] = {
        "type": question["type"],
        "instructions": question["instructions"],
    }
    if question.get("criteria") is not None:
        payload["criteria"] = question["criteria"]
    return payload


def build_payload(
    provider: str,
    model: str,
    state: Union[str, Dict[str, Any], List[Any]],
    questions: Union[Dict[str, Any], List[Any]],
) -> Dict[str, Any]:
    """Build the provider-specific JSON body for a decisions request."""
    _check_provider(provider)
    if not model:
        raise DecisionsError("model is required")
    normalized = _normalize_questions(questions)

    if provider == "openai":
        return {
            "model": model,
            "input": _openai_input(state),
            "questions": [
                _openai_question(name, question)
                for name, question in normalized.items()
            ],
        }
    if provider == "gemini":
        return {
            "model": model,
            "input": _gemini_input(state),
            "questions": {
                name: _gemini_question(question)
                for name, question in normalized.items()
            },
        }
    return {
        "model": model,
        "state": state,
        "questions": {
            name: _perplexity_question(question)
            for name, question in normalized.items()
        },
    }


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #

def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _probabilities(raw: Any) -> Dict[str, float]:
    """Normalize a probabilities map or OpenAI's list of records into ``{key: prob}``."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): float(value) for key, value in raw.items()}
    out: Dict[str, float] = {}
    for entry in raw:
        if isinstance(entry, dict) and entry.get("value") is not None:
            out[str(entry["value"])] = float(entry.get("probability", 0.0))
    return out


def _parse_choice(answer: Dict[str, Any]) -> ChoiceResult:
    probabilities = _probabilities(answer.get("probabilities"))
    choice = answer.get("choice")
    if choice is None and probabilities:
        choice = max(probabilities, key=probabilities.get)
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = probabilities.get(str(choice), 0.0)
    return ChoiceResult(
        choice=str(choice if choice is not None else ""),
        probabilities=probabilities,
        confidence=float(confidence or 0.0),
    )


def _parse_noul(answer: Dict[str, Any]) -> NoulResult:
    value = answer.get("noul")
    if value is None:
        value = answer.get("probability")
    if value is None:
        value = _probabilities(answer.get("probabilities")).get("yes", 0.5)
    return NoulResult(noul=float(max(0.0, min(1.0, float(value)))))


def _parse_score(answer: Dict[str, Any]) -> ScoreResult:
    probabilities = _probabilities(answer.get("probabilities"))
    legend = answer.get("legend")
    if isinstance(legend, dict):
        legend = {str(key): str(value) for key, value in legend.items()}
    elif isinstance(answer.get("probabilities"), list):
        legend = {
            str(entry["value"]): str(entry.get("label", entry["value"]))
            for entry in answer["probabilities"]
            if isinstance(entry, dict) and entry.get("value") is not None
        } or None
    else:
        legend = None

    score = answer.get("score")
    if score is None and probabilities:
        total = sum(probabilities.values())
        if total > 0:
            score = sum(
                float(key) * value
                for key, value in probabilities.items()
                if _is_number(key)
            ) / total
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = max(probabilities.values(), default=0.0)
    return ScoreResult(
        score=float(score or 0.0),
        probabilities=probabilities,
        confidence=float(confidence or 0.0),
        legend=legend,
    )


_PARSERS = {"noul": _parse_noul, "predicate": _parse_noul, "choice": _parse_choice, "score": _parse_score}


def parse_answers(payload: Dict[str, Any], provider: Optional[str] = None) -> System1Result:
    """Convert a raw Decisions API payload into a :class:`System1Result`.

    Gemini and Perplexity key answers by question name; OpenAI returns an ordered
    array whose entries carry ``name``. Both are normalized to ``answers[name]``.
    """
    raw = payload.get("answers")
    if raw is None:
        raise DecisionsError(f"Decisions response is missing 'answers': {payload!r}")

    if isinstance(raw, dict):
        entries = {str(name): answer for name, answer in raw.items()}
    elif isinstance(raw, list):
        entries = {}
        for index, answer in enumerate(raw):
            if not isinstance(answer, dict):
                raise DecisionsError(f"Answer at index {index} is not an object: {answer!r}")
            entries[str(answer.get("name") or f"question_{index}")] = answer
    else:
        raise DecisionsError(f"Unexpected 'answers' payload: {raw!r}")

    answers: Dict[str, Any] = {}
    for name, answer in entries.items():
        if not isinstance(answer, dict):
            raise DecisionsError(f"Answer for {name!r} is not an object: {answer!r}")
        if answer.get("type") == "refusal":
            reason = answer.get("refusal") or "no reason given"
            raise DecisionsError(f"Decision model refused question {name!r}: {reason}")
        qtype = _check_question_type(str(answer.get("type", "choice")).strip().lower())
        answers[name] = _PARSERS[qtype](answer)

    return System1Result(
        answers=answers,
        routing={
            "provider": provider,
            "model": payload.get("model"),
            "usage": payload.get("usage"),
        },
    )


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

@dataclass
class DecisionsConfig:
    """Provider, model, and credentials for a hosted Decisions API backend."""

    provider: str
    model: str
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        _check_provider(self.provider)
        if not self.model:
            raise DecisionsError("model is required")

    @property
    def api_key_env(self) -> str:
        return PROVIDERS[self.provider]["api_key_env"]

    def resolved_api_key(self) -> Optional[str]:
        return self.api_key or os.environ.get(self.api_key_env)

    def resolved_api_url(self) -> str:
        return (self.api_url or PROVIDERS[self.provider]["api_url"]).rstrip("/")

    def endpoint(self) -> str:
        return f"{self.resolved_api_url()}/decisions"

    def auth_headers(self) -> Dict[str, str]:
        key = self.resolved_api_key()
        if not key:
            raise DecisionsError(
                f"No API key for provider {self.provider!r}. "
                f"Set {self.api_key_env} or pass api_key=..."
            )
        headers = {"Content-Type": "application/json"}
        if self.provider == "gemini":
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
        headers.update(self.headers)
        return headers


def _extract_error(response: "requests.Response") -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("error"):
        error = payload["error"]
        return str(error.get("message") or error) if isinstance(error, dict) else str(error)
    text = (response.text or "").strip()
    return text[:500] if text else f"HTTP {response.status_code}"


def _raise_for_status(response: "requests.Response", config: DecisionsConfig) -> None:
    if response.ok:
        return
    detail = _extract_error(response)
    if response.status_code in (401, 403):
        raise DecisionsError(
            f"{config.provider} rejected the API key ({response.status_code}: {detail}). "
            f"Check {config.api_key_env}."
        )
    if response.status_code == 400:
        raise DecisionsError(
            f"{config.provider} rejected the decisions request (400: {detail}). "
            f"Confirm model {config.model!r} is a decision model and the question "
            f"shape is valid for this provider."
        )
    if response.status_code == 404:
        raise DecisionsError(
            f"{config.provider} has no decisions endpoint at {response.url} (404: {detail})."
        )
    if response.status_code == 429:
        raise DecisionsError(
            f"{config.provider} rate-limited the decisions request (429: {detail})."
        )
    raise DecisionsError(
        f"{config.provider} decisions request failed ({response.status_code}): {detail}"
    )


def get_decision_response(
    state: Union[str, Dict[str, Any], List[Any]],
    questions: Union[Dict[str, Any], List[Any]],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
    session: Optional["requests.Session"] = None,
) -> Dict[str, Any]:
    """POST one ``/decisions`` request and return the raw provider payload.

    The decision-model counterpart to :func:`npcpy.llm_funcs.get_llm_response`:
    same ``provider`` / ``model`` / ``api_url`` / ``api_key`` arguments, but the
    model returns typed answers with probabilities instead of generated text.
    """
    config = DecisionsConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        api_url=api_url,
        timeout=timeout,
        headers=dict(headers or {}),
    )
    body = json.dumps(
        build_payload(config.provider, config.model, state, questions),
        ensure_ascii=False,
    )
    requester = session or requests
    try:
        response = requester.post(
            config.endpoint(),
            data=body.encode("utf-8"),
            headers=config.auth_headers(),
            timeout=config.timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise DecisionsError(
            f"Could not reach the {config.provider} Decisions API at "
            f"{config.endpoint()} ({exc})."
        ) from exc

    _raise_for_status(response, config)
    try:
        payload = response.json()
    except ValueError as exc:
        raise DecisionsError(f"{config.provider} returned a non-JSON response: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecisionsError(f"Unexpected {config.provider} decisions response: {payload!r}")
    return payload


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #

class DecisionsClient:
    """Decisions predictor backed by a hosted Decisions API.

    Exposes the same ``choice``/``noul``/``score``/``predict`` surface as
    :class:`npcpy.gen.systemone.OllamaSystem1Client` and
    :class:`npcpy.ft.system1.System1Predictor`, so it can be dropped into
    :class:`npcpy.gen.decision.DecisionSystem1`.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Optional[Dict[str, str]] = None,
        session: Optional["requests.Session"] = None,
    ):
        self.config = DecisionsConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            api_url=api_url,
            timeout=timeout,
            headers=dict(headers or {}),
        )
        self.session = session
        self.label_maps: Dict[str, List[str]] = {}

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def _resolve_question_name(self, question_name: str) -> str:
        return question_name

    def _call(self, state: Any, questions: Any) -> System1Result:
        payload = get_decision_response(
            state,
            questions,
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            api_url=self.config.api_url,
            timeout=self.config.timeout,
            headers=self.config.headers,
            session=self.session,
        )
        return parse_answers(payload, provider=self.provider)

    def choice(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: Dict[str, str],
        question_name: str = "default",
    ) -> ChoiceResult:
        result = self._call(
            state,
            {question_name: {"type": "choice", "instructions": instructions, "criteria": criteria}},
        )
        return result.answers[question_name]

    def noul(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: Optional[Dict[str, str]] = None,
        question_name: str = "default",
    ) -> NoulResult:
        question: Dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria
        result = self._call(state, {question_name: question})
        return result.answers[question_name]

    def score(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: List[str],
        question_name: str = "default",
    ) -> ScoreResult:
        result = self._call(
            state,
            {question_name: {"type": "score", "instructions": instructions, "criteria": criteria}},
        )
        return result.answers[question_name]

    def predict(
        self,
        state: Union[str, Dict[str, Any]],
        questions: Union[Dict[str, Any], List[Any]],
    ) -> System1Result:
        return self._call(state, questions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_url": self.config.resolved_api_url(),
            "api_key_env": self.config.api_key_env,
        }


def load_decisions(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> DecisionsClient:
    """Create a :class:`DecisionsClient` for a hosted provider."""
    return DecisionsClient(
        provider=provider,
        model=model,
        api_key=api_key,
        api_url=api_url,
        timeout=timeout,
        headers=headers,
    )


def available(provider: str) -> Dict[str, Any]:
    """Report whether an API key is present for ``provider``.

    A local check only: no request is made and no tokens are spent.
    """
    _check_provider(provider)
    api_key_env = PROVIDERS[provider]["api_key_env"]
    present = bool(os.environ.get(api_key_env))
    return {
        "available": present,
        "provider": provider,
        "api_key_env": api_key_env,
        "api_key_present": present,
        "api_url": PROVIDERS[provider]["api_url"],
        "error": None if present else f"{api_key_env} is not set",
    }


def require_available(provider: str) -> Dict[str, Any]:
    """Like :func:`available` but raises when the provider is not configured."""
    info = available(provider)
    if not info["available"]:
        raise DecisionsError(
            f"No API key for provider {provider!r}. "
            f"Set {info['api_key_env']} or pass api_key=..."
        )
    return info


# --------------------------------------------------------------------------- #
# HuggingFace transformers (local zero-shot classifier)
# --------------------------------------------------------------------------- #

TRANSFORMERS_PROVIDER = "transformers"
DEFAULT_TRANSFORMERS_MODEL = "facebook/bart-large-mnli"


class TransformersDecisionsClient:
    """Decisions predictor backed by a local transformers zero-shot pipeline.

    Uses ``pipeline("zero-shot-classification")``, which returns a probability
    per candidate label -- the same shape a hosted Decisions API returns. That
    maps onto our three question types directly:

    * ``choice`` -- one probability per ``criteria`` label.
    * ``noul``   -- the ``yes`` probability against ``["yes", "no"]``.
    * ``score``  -- the probability-weighted average over ordered ``criteria``.

    Requires the optional ``transformers`` and ``torch`` packages; they are
    imported lazily so importing npcpy stays cheap.
    """

    def __init__(
        self,
        model: str = DEFAULT_TRANSFORMERS_MODEL,
        device: Optional[Any] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not model:
            raise DecisionsError("model is required")
        self.model = model
        self.device = device
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self._pipeline: Optional[Any] = None
        self.label_maps: Dict[str, List[str]] = {}

    @property
    def provider(self) -> str:
        return TRANSFORMERS_PROVIDER

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline as _pipeline
        except ImportError as exc:
            raise DecisionsError(
                "provider='transformers' requires the optional 'transformers' "
                "package. Install it with `pip install transformers torch`."
            ) from exc
        kwargs = dict(self.pipeline_kwargs)
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipeline = _pipeline("zero-shot-classification", model=self.model, **kwargs)
        return self._pipeline

    def _resolve_question_name(self, question_name: str) -> str:
        return question_name

    def _classify(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Return ``{label: probability}`` for the given candidate labels."""
        if len(labels) == 1:
            return {labels[0]: 1.0}
        output = self._get_pipeline()(text, candidate_labels=labels)
        return {
            str(label): float(score)
            for label, score in zip(output["labels"], output["scores"])
        }

    def _call(self, state: Any, questions: Any) -> System1Result:
        normalized = _normalize_questions(questions)
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
        answers: Dict[str, Any] = {}

        for name, question in normalized.items():
            instructions = question["instructions"]
            text = f"{state_text}\n\n{instructions}" if instructions else state_text
            qtype = question["type"]

            if qtype == "choice":
                labels = list(question["criteria"])
                probabilities = self._classify(text, labels)
                choice = max(probabilities, key=probabilities.get) if probabilities else ""
                answers[name] = ChoiceResult(
                    choice=choice,
                    probabilities=probabilities,
                    confidence=probabilities.get(choice, 0.0),
                )
                self.label_maps[name] = labels
            elif qtype in ("noul", "predicate"):
                probabilities = self._classify(text, ["yes", "no"])
                answers[name] = NoulResult(noul=probabilities.get("yes", 0.0))
            else:
                levels = list(question["criteria"])
                probabilities = self._classify(text, levels)
                indexed = {str(index): probabilities.get(label, 0.0) for index, label in enumerate(levels)}
                score = sum(index * probability for index, probability in enumerate(indexed.values()))
                answers[name] = ScoreResult(
                    score=float(score),
                    probabilities=indexed,
                    confidence=max(probabilities.values(), default=0.0),
                    legend={str(index): label for index, label in enumerate(levels)},
                )
                self.label_maps[name] = levels

        return System1Result(
            answers=answers,
            routing={"provider": TRANSFORMERS_PROVIDER, "model": self.model},
        )

    def choice(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: Dict[str, str],
        question_name: str = "default",
    ) -> ChoiceResult:
        result = self._call(
            state,
            {question_name: {"type": "choice", "instructions": instructions, "criteria": criteria}},
        )
        return result.answers[question_name]

    def noul(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: Optional[Dict[str, str]] = None,
        question_name: str = "default",
    ) -> NoulResult:
        question: Dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria
        result = self._call(state, {question_name: question})
        return result.answers[question_name]

    def score(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: List[str],
        question_name: str = "default",
    ) -> ScoreResult:
        result = self._call(
            state,
            {question_name: {"type": "score", "instructions": instructions, "criteria": criteria}},
        )
        return result.answers[question_name]

    def predict(
        self,
        state: Union[str, Dict[str, Any]],
        questions: Union[Dict[str, Any], List[Any]],
    ) -> System1Result:
        return self._call(state, questions)

    def to_dict(self) -> Dict[str, Any]:
        return {"provider": TRANSFORMERS_PROVIDER, "model": self.model}


def load_transformers_decisions(
    model: str = DEFAULT_TRANSFORMERS_MODEL,
    device: Optional[Any] = None,
    **kwargs,
) -> TransformersDecisionsClient:
    """Create a :class:`TransformersDecisionsClient`."""
    return TransformersDecisionsClient(model=model, device=device, **kwargs)


# --------------------------------------------------------------------------- #
# module-level convenience wrappers
# --------------------------------------------------------------------------- #

def choice(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: Dict[str, str],
    provider: str,
    model: str,
    client: Optional[DecisionsClient] = None,
    question_name: str = "default",
    **kwargs,
) -> ChoiceResult:
    client = client or load_decisions(provider=provider, model=model, **kwargs)
    return client.choice(state, instructions, criteria, question_name=question_name)


def noul(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    provider: str,
    model: str,
    client: Optional[DecisionsClient] = None,
    criteria: Optional[Dict[str, str]] = None,
    question_name: str = "default",
    **kwargs,
) -> NoulResult:
    client = client or load_decisions(provider=provider, model=model, **kwargs)
    return client.noul(state, instructions, criteria=criteria, question_name=question_name)


def score(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: List[str],
    provider: str,
    model: str,
    client: Optional[DecisionsClient] = None,
    question_name: str = "default",
    **kwargs,
) -> ScoreResult:
    client = client or load_decisions(provider=provider, model=model, **kwargs)
    return client.score(state, instructions, criteria, question_name=question_name)


def predict(
    state: Union[str, Dict[str, Any]],
    questions: Union[Dict[str, Any], List[Any]],
    provider: str,
    model: str,
    client: Optional[DecisionsClient] = None,
    **kwargs,
) -> System1Result:
    client = client or load_decisions(provider=provider, model=model, **kwargs)
    return client.predict(state, questions)
