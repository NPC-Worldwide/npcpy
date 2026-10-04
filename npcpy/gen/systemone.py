"""System One: fast typed decisions from a local Ollama decision model.

Client for Ollama's ``POST /v1/systemone`` endpoint (Ollama >= 0.35.0, based on
TypeSafe's Jev API). Decision models are **pre-trained** -- you pull one and
call it. There is nothing to fine-tune and no training step.

The endpoint takes a single ``state`` (text or JSON-serializable object) plus a
set of *named questions* and scores every question against the state in one
request. Three question types are supported:

* ``choice`` -- pick one label from a ``criteria`` mapping.
* ``noul``   -- answer a yes/no question; returns ``P(true)`` in ``[0, 1]``.
* ``score``  -- score the state against an ordered rubric of ``criteria``.

No API key is required; the model runs on the user's machine.

Example
-------
>>> from npcpy.gen.systemone import OllamaSystem1Client, OllamaSystem1Config
>>> client = OllamaSystem1Client(OllamaSystem1Config(model="nimble"))
>>> client.choice(
...     {"ticket": "I was charged twice. Please refund the extra payment."},
...     instructions="Which team should handle this ticket?",
...     criteria={"billing": "Payments and refunds", "technical": "Bugs"},
... ).choice
'billing'
"""

import json
import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

from npcpy.ft.system1 import (
    ChoiceResult,
    NoulResult,
    ScoreResult,
    System1Result,
)

DEFAULT_MODEL = "nimble"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "5m"
DEFAULT_TIMEOUT = 120
MIN_OLLAMA_VERSION = (0, 35, 0)
MAX_REQUEST_BYTES = 64 * 1024

_QUESTION_TYPE_ALIASES = {
    "choice": "choice",
    "classify": "choice",
    "classification": "choice",
    "label": "choice",
    "noul": "noul",
    "yesno": "noul",
    "yes_no": "noul",
    "bool": "noul",
    "boolean": "noul",
    "score": "score",
    "scoring": "score",
    "rubric": "score",
}

class OllamaSystem1Error(RuntimeError):
    """Raised when the Ollama System One endpoint fails or is unavailable."""


def _normalize_base_url(url: Optional[str]) -> str:
    """Return an ``http(s)://host:port`` base URL with no endpoint suffix."""
    if not url:
        url = (
            os.environ.get("NPCPY_OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or DEFAULT_BASE_URL
        )
    url = str(url).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme or "http"
    host = parts.hostname or "localhost"
    port = parts.port
    netloc = host
    if port is not None:
        netloc = f"{host}:{port}"
    elif scheme == "http":
        netloc = f"{host}:11434"
    path = parts.path.rstrip("/")
    for suffix in ("/v1/systemone", "/v1", "/api/systemone", "/api"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{scheme}://{netloc}{path}"


def _parse_version(version: Optional[str]) -> Optional[tuple]:
    if not version:
        return None
    cleaned = str(version).strip().lstrip("v")
    pieces = []
    for token in cleaned.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    if not pieces:
        return None
    while len(pieces) < 3:
        pieces.append(0)
    return tuple(pieces[:3])


def _normalize_question_type(question_type: Optional[str]) -> str:
    key = str(question_type or "choice").strip().lower()
    if key not in _QUESTION_TYPE_ALIASES:
        raise ValueError(
            f"Unknown System One question type: {question_type!r}. Expected one of: choice, noul, score"
        )
    return _QUESTION_TYPE_ALIASES[key]


def _serialize_state(state: Union[str, Dict[str, Any], List[Any]]) -> Any:
    """Ollama accepts a string, object, or array directly as ``state``."""
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state must be a non-empty string")
        return state
    if isinstance(state, (dict, list)):
        if not state:
            raise ValueError("state must be a non-empty object or array")
        return state
    return json.dumps(state, ensure_ascii=False, default=str)


def _normalize_questions(
    questions: Union[Dict[str, Any], List[Any]],
) -> Dict[str, Dict[str, Any]]:
    """Coerce questions into the endpoint's ``{name: {type, instructions, ...}}`` shape."""
    if isinstance(questions, list):
        normalized: Dict[str, Dict[str, Any]] = {}
        for index, question in enumerate(questions):
            if isinstance(question, dict):
                name = question.get("name") or f"question_{index}"
            else:
                name = getattr(question, "name", None) or f"question_{index}"
            normalized[name] = _normalize_question(question)
        return normalized
    if not isinstance(questions, dict):
        raise ValueError("questions must be a mapping of name -> question or a list")
    out: Dict[str, Dict[str, Any]] = {}
    for name, question in questions.items():
        out[str(name)] = _normalize_question(question)
    return out


def _normalize_question(question: Any) -> Dict[str, Any]:
    if isinstance(question, dict):
        qtype = question.get("type", "choice")
        instructions = question.get("instructions", "")
        criteria = question.get("criteria")
    else:
        qtype = getattr(question, "type", "choice")
        instructions = getattr(question, "instructions", "")
        criteria = getattr(question, "criteria", None)
    normalized_type = _normalize_question_type(qtype)
    payload: Dict[str, Any] = {
        "type": normalized_type,
        "instructions": instructions or "",
    }
    if criteria is not None:
        if normalized_type == "choice" and not isinstance(criteria, dict):
            raise ValueError("choice questions require a criteria mapping of label -> description")
        if normalized_type == "noul" and isinstance(criteria, dict):
            pass
        payload["criteria"] = criteria
    return payload


@dataclass
class OllamaSystem1Config:
    """Configuration for the Ollama System One backend."""

    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    keep_alive: Optional[str] = DEFAULT_KEEP_ALIVE
    timeout: Optional[float] = DEFAULT_TIMEOUT
    headers: Dict[str, str] = field(default_factory=dict)

    def resolved_base_url(self) -> str:
        return _normalize_base_url(self.base_url)


def _extract_error(response: "requests.Response") -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "error" in payload:
            return str(payload["error"])
    except Exception:
        pass
    text = (response.text or "").strip()
    return text[:500] if text else f"HTTP {response.status_code}"


def _raise_for_status(response: "requests.Response", model: str, base_url: str) -> None:
    if response.ok:
        return
    detail = _extract_error(response)
    if response.status_code == 404:
        raise OllamaSystem1Error(
            f"Ollama System One endpoint not found at {base_url}/v1/systemone (404: {detail}). Ollama >= 0.35.0 is required; upgrade with `ollama update` (or reinstall) and pull the model with `ollama pull {model}`."
        )
    if response.status_code == 413:
        raise OllamaSystem1Error(
            f"Ollama System One request exceeded the 64 KiB limit (413: request body must not exceed 64 KiB). Shorten the state or reduce the number of questions."
        )
    if response.status_code == 400:
        raise OllamaSystem1Error(
            f"Ollama System One rejected the request (400: {detail}). Confirm the model {model!r} is a decision model and that the context window fits the rendered prompt."
        )
    raise OllamaSystem1Error(f"Ollama System One request failed ({response.status_code}): {detail}")


def systemone(
    state: Union[str, Dict[str, Any], List[Any]],
    questions: Union[Dict[str, Any], List[Any]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    keep_alive: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
    session: Optional["requests.Session"] = None,
) -> Dict[str, Any]:
    """Call ``POST /v1/systemone`` and return the raw response payload."""
    effective_model = model or os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL") or DEFAULT_MODEL
    effective_base = _normalize_base_url(base_url)
    payload: Dict[str, Any] = {
        "model": effective_model,
        "state": _serialize_state(state),
        "questions": _normalize_questions(questions),
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    body = json.dumps(payload, ensure_ascii=False)
    if len(body.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise OllamaSystem1Error(
            f"Ollama System One request body exceeds the 64 KiB limit. Shorten the state or reduce the number of questions."
        )

    url = f"{effective_base}/v1/systemone"
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    requester = session or requests
    try:
        response = requester.post(
            url,
            data=body.encode("utf-8"),
            headers=request_headers,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise OllamaSystem1Error(
            f"Could not reach Ollama at {effective_base} ({exc}). Start it with `ollama serve` and confirm the host/port."
        ) from exc

    _raise_for_status(response, effective_model, effective_base)
    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaSystem1Error(f"Ollama returned a non-JSON response: {exc}") from exc
    if not isinstance(data, dict):
        raise OllamaSystem1Error(f"Unexpected Ollama System One response: {data!r}")
    return data


def _response_answers(payload: Dict[str, Any]) -> Dict[str, Any]:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise OllamaSystem1Error(
            f"Ollama System One response is missing the 'answers' object: {payload!r}"
        )
    return answers


def _parse_choice(name: str, answer: Dict[str, Any]) -> ChoiceResult:
    probabilities = {
        str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
    }
    choice = answer.get("choice")
    if choice is None and probabilities:
        choice = max(probabilities, key=probabilities.get)
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = probabilities.get(str(choice), 0.0)
    return ChoiceResult(
        choice=str(choice if choice is not None else ""),
        probabilities=probabilities,
        confidence=float(confidence),
    )


def _parse_noul(name: str, answer: Dict[str, Any]) -> NoulResult:
    value = answer.get("noul")
    if value is None:
        probabilities = answer.get("probabilities") or {}
        value = probabilities.get("yes", 0.5)
    return NoulResult(noul=float(max(0.0, min(1.0, float(value)))))


def _parse_score(name: str, answer: Dict[str, Any]) -> ScoreResult:
    probabilities = {
        str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
    }
    legend = answer.get("legend")
    if legend is not None:
        legend = {str(k): str(v) for k, v in dict(legend).items()}
    score_value = answer.get("score")
    if score_value is None and probabilities:
        total = sum(probabilities.values())
        if total > 0:
            score_value = sum(float(k) * v for k, v in probabilities.items() if _is_number(k)) / total
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = max(probabilities.values(), default=0.0)
    return ScoreResult(
        score=float(score_value or 0.0),
        probabilities=probabilities,
        confidence=float(confidence),
        legend=legend,
    )


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


_PARSERS = {"choice": _parse_choice, "noul": _parse_noul, "score": _parse_score}


def parse_answers(payload: Dict[str, Any]) -> System1Result:
    """Convert a raw ``/v1/systemone`` payload into a :class:`System1Result`."""
    answers: Dict[str, Any] = {}
    for name, answer in _response_answers(payload).items():
        if not isinstance(answer, dict):
            raise OllamaSystem1Error(f"Answer for {name!r} is not an object: {answer!r}")
        qtype = _normalize_question_type(answer.get("type", "choice"))
        answers[str(name)] = _PARSERS[qtype](str(name), answer)
    return System1Result(answers=answers, routing={"usage": payload.get("usage")})


class OllamaSystem1Client:
    """System One predictor backed by a local Ollama server.

    Exposes the same ``choice``/``noul``/``score``/``predict`` surface as
    :class:`npcpy.ft.system1.System1Predictor` so it can be dropped into
    :class:`npcpy.gen.decision.DecisionSystem1`.
    """

    backend = "ollama"

    def __init__(self, config: Optional[OllamaSystem1Config] = None, **kwargs):
        if config is None:
            config = OllamaSystem1Config(**kwargs)
        self.config = config
        self.session: Optional["requests.Session"] = None

    @property
    def base_url(self) -> str:
        return self.config.resolved_base_url()

    @property
    def model(self) -> str:
        return self.config.model

    label_maps: Dict[str, List[str]] = {}

    def _resolve_question_name(self, question_name: str) -> str:
        return question_name

    def _call(self, state: Any, questions: Any) -> System1Result:
        payload = systemone(
            state,
            questions,
            model=self.config.model,
            base_url=self.config.base_url,
            keep_alive=self.config.keep_alive,
            timeout=self.config.timeout,
            headers=self.config.headers,
            session=self.session,
        )
        return parse_answers(payload)

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
        question_name: str = "default",
    ) -> NoulResult:
        result = self._call(
            state,
            {question_name: {"type": "noul", "instructions": instructions}},
        )
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
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "keep_alive": self.config.keep_alive,
        }


def _version_info(base_url: Optional[str] = None, timeout: float = 10) -> Dict[str, Any]:
    base = _normalize_base_url(base_url)
    try:
        response = requests.get(f"{base}/api/version", timeout=timeout)
        response.raise_for_status()
        version = str(response.json().get("version", "")).strip()
    except Exception as exc:
        return {"reachable": False, "version": None, "error": str(exc), "base_url": base}
    return {"reachable": True, "version": version, "error": None, "base_url": base}


def list_models(base_url: Optional[str] = None, timeout: float = 10) -> List[str]:
    base = _normalize_base_url(base_url)
    try:
        response = requests.get(f"{base}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception as exc:
        raise OllamaSystem1Error(f"Could not list Ollama models at {base}: {exc}") from exc
    return [str(m.get("name", "")) for m in models]


def _model_present(model: str, models: List[str]) -> bool:
    model = model.strip()
    if model in models:
        return True
    base = model.split(":")[0]
    return any(m.split(":")[0] == base for m in models)


def available(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 10,
) -> Dict[str, Any]:
    """Report whether a usable System One endpoint/model is available.

    Returns a dict with ``available``, ``reachable``, ``version``,
    ``version_ok``, ``model``, ``model_present``, ``base_url`` and ``error``.
    """
    effective_model = model or os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL") or DEFAULT_MODEL
    info = _version_info(base_url, timeout=timeout)
    result: Dict[str, Any] = {
        "available": False,
        "reachable": info["reachable"],
        "version": info["version"],
        "version_ok": False,
        "model": effective_model,
        "model_present": False,
        "base_url": info["base_url"],
        "error": info["error"],
    }
    if not info["reachable"]:
        return result
    parsed = _parse_version(info["version"])
    result["version_ok"] = bool(parsed and parsed >= MIN_OLLAMA_VERSION)
    try:
        models = list_models(info["base_url"], timeout=timeout)
        result["model_present"] = _model_present(effective_model, models)
    except OllamaSystem1Error as exc:
        result["error"] = str(exc)
    result["available"] = result["version_ok"] and result["model_present"]
    return result


def require_available(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 10,
) -> Dict[str, Any]:
    """Like :func:`available` but raises with a remediation hint when unusable."""
    info = available(model=model, base_url=base_url, timeout=timeout)
    if info["available"]:
        return info
    if not info["reachable"]:
        raise OllamaSystem1Error(
            f"Ollama is not reachable at {info['base_url']} ({info['error']}). Start it with `ollama serve`."
        )
    if not info["version_ok"]:
        raise OllamaSystem1Error(
            f"Ollama {info['version']} does not support System One (requires >= {'.'.join(map(str, MIN_OLLAMA_VERSION))}). Upgrade Ollama."
        )
    raise OllamaSystem1Error(
        f"Decision model {info['model']!r} is not installed. Run `ollama pull {info['model']}`."
    )


def load_ollama_system1(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> OllamaSystem1Client:
    """Create an :class:`OllamaSystem1Client` (no local weights are loaded)."""
    config = OllamaSystem1Config(
        model=model or os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL") or DEFAULT_MODEL,
        base_url=base_url,
        **{k: v for k, v in kwargs.items() if k in {"keep_alive", "timeout", "headers"}},
    )
    return OllamaSystem1Client(config)


_default_client: Optional[OllamaSystem1Client] = None
_default_client_key: Optional[tuple] = None


def _get_default_client() -> OllamaSystem1Client:
    global _default_client, _default_client_key
    key = (
        os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL"),
        os.environ.get("NPCPY_OLLAMA_BASE_URL"),
        os.environ.get("OLLAMA_HOST"),
    )
    if _default_client is None or key != _default_client_key:
        _default_client = load_ollama_system1()
        _default_client_key = key
    return _default_client


def choice(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: Dict[str, str],
    client: Optional[OllamaSystem1Client] = None,
    question_name: str = "default",
    **kwargs,
) -> ChoiceResult:
    client = client or _get_default_client()
    return client.choice(state, instructions, criteria, question_name=question_name)


def noul(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    client: Optional[OllamaSystem1Client] = None,
    question_name: str = "default",
    **kwargs,
) -> NoulResult:
    client = client or _get_default_client()
    return client.noul(state, instructions, question_name=question_name)


def score(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: List[str],
    client: Optional[OllamaSystem1Client] = None,
    question_name: str = "default",
    **kwargs,
) -> ScoreResult:
    client = client or _get_default_client()
    return client.score(state, instructions, criteria, question_name=question_name)


def predict(
    state: Union[str, Dict[str, Any]],
    questions: Union[Dict[str, Any], List[Any]],
    client: Optional[OllamaSystem1Client] = None,
    **kwargs,
) -> System1Result:
    client = client or _get_default_client()
    return client.predict(state, questions)
