import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from npcpy.ft.system1 import (
    ChoiceResult,
    NoulResult,
    ScoreResult,
    System1Config,
    System1Example,
    System1Predictor,
    System1Result,
    choice as _system1_choice,
    load_system1,
    noul as _system1_noul,
    predict as _system1_predict,
    score as _system1_score,
    train_system1,
)
from npcpy.gen.systemone import (
    DEFAULT_MODEL as DEFAULT_OLLAMA_DECISION_MODEL,
    OllamaSystem1Client,
    OllamaSystem1Config,
    load_ollama_system1,
)


@dataclass
class DecisionQuestion:
    name: str
    type: str
    instructions: str
    criteria: Any = None


@dataclass
class DecisionRouter:
    tiers: List[str]
    criteria: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.criteria is None:
            self.criteria = {tier: tier for tier in self.tiers}


@dataclass
class BatchDecisions:
    results: List[System1Result] = field(default_factory=list)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]


def _state_to_text(state: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, default=str)


_OLLAMA_BACKEND_ALIASES = {"ollama", "systemone", "system_one", "system-1", "jev"}
_LOCAL_BACKEND_ALIASES = {"local", "sklearn", "train", "local_embeddings", "embedded"}


def _resolve_backend(backend: Optional[str]) -> str:
    """Resolve the decision backend, falling back to ``NPCPY_SYSTEM1_BACKEND``."""
    if backend is None:
        backend = os.environ.get("NPCPY_SYSTEM1_BACKEND", "local")
    key = str(backend).strip().lower()
    if key in _OLLAMA_BACKEND_ALIASES:
        return "ollama"
    if key in _LOCAL_BACKEND_ALIASES:
        return "local"
    raise ValueError(
        f"Unknown decision backend: {backend!r}. Expected 'local' or 'ollama'."
    )


def _default_ollama_config(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaSystem1Config:
    return OllamaSystem1Config(
        model=model or os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL") or DEFAULT_OLLAMA_DECISION_MODEL,
        base_url=base_url,
    )


def _examples_from_records(
    records: List[Dict[str, Any]],
    state_key: str = "state",
    name_key: str = "name",
    type_key: str = "type",
    instructions_key: str = "instructions",
    criteria_key: str = "criteria",
    answer_key: str = "answer",
) -> List[System1Example]:
    examples = []
    for record in records:
        state = record.get(state_key)
        if state is None:
            continue
        answer = record.get(answer_key)
        question_type = record.get(type_key, "choice")
        if isinstance(answer, str):
            if question_type == "choice":
                answer = {"choice": answer}
            elif question_type == "score":
                answer = {"score": int(answer)}
            elif question_type == "noul":
                answer = {"noul": float(answer)}
        examples.append(
            System1Example(
                state=state,
                question_type=question_type,
                instructions=record.get(instructions_key, ""),
                criteria=record.get(criteria_key),
                answer=answer,
                question_name=record.get(name_key, "default"),
            )
        )
    return examples


def load_decision_examples(
    path: str,
    state_key: str = "state",
    name_key: str = "name",
    type_key: str = "type",
    instructions_key: str = "instructions",
    criteria_key: str = "criteria",
    answer_key: str = "answer",
) -> List[System1Example]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            records = json.load(f)
    elif ext == ".csv":
        import csv
        records = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if not v:
                        continue
                    try:
                        parsed[k] = json.loads(v)
                    except Exception:
                        parsed[k] = v
                records.append(parsed)
    elif ext == ".jsonl":
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported decision examples format: {ext}")
    if not isinstance(records, list):
        raise ValueError("Decision examples file must contain a list of records")
    return _examples_from_records(
        records,
        state_key=state_key,
        name_key=name_key,
        type_key=type_key,
        instructions_key=instructions_key,
        criteria_key=criteria_key,
        answer_key=answer_key,
    )


class DecisionSystem1:
    """System One decision helper.

    ``backend="local"`` (default) trains a small classifier over embeddings
    (see :mod:`npcpy.ft.system1`). ``backend="ollama"`` delegates to a local
    Ollama System One model (``/v1/systemone``, Ollama >= 0.35.0), which needs
    no training.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[System1Config] = None,
        backend: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_config: Optional[OllamaSystem1Config] = None,
    ):
        self.backend = _resolve_backend(backend)
        self.config = config or System1Config()
        self.predictor: Optional[Any] = None
        self.model_path: Optional[str] = None
        self.ollama_config: Optional[OllamaSystem1Config] = None
        if self.backend == "ollama":
            self.ollama_config = ollama_config or _default_ollama_config(
                model=ollama_model, base_url=ollama_base_url
            )
            self.predictor = OllamaSystem1Client(self.ollama_config)
        elif model_path is not None:
            self.load(model_path)

    @classmethod
    def from_ollama(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> "DecisionSystem1":
        """Build a decision helper backed by a local Ollama decision model."""
        return cls(backend="ollama", ollama_model=model, ollama_base_url=base_url, **kwargs)

    def load(self, path: str) -> "DecisionSystem1":
        if self.backend == "ollama":
            raise ValueError(
                f"The Ollama backend does not load local model paths; set ollama_model and run `ollama pull <model>` instead."
            )
        self.predictor = load_system1(path)
        self.model_path = path
        return self

    def train(
        self,
        examples: Union[List[System1Example], List[Dict[str, Any]]],
        output_dir: Optional[str] = None,
    ) -> str:
        if self.backend == "ollama":
            raise ValueError(
                f"The Ollama System One backend is pre-trained; no training is required. Pull a decision model with `ollama pull nimble`."
            )
        if examples and isinstance(examples[0], dict):
            examples = _examples_from_records(examples)
        config = self.config
        if output_dir is not None:
            config = System1Config(
                encoder_name=self.config.encoder_name,
                classifier=self.config.classifier,
                classifier_kwargs=self.config.classifier_kwargs,
                output_dir=output_dir,
                device=self.config.device,
            )
        path = train_system1(examples, config)
        self.load(path)
        return path

    def _require_predictor(self) -> System1Predictor:
        if self.predictor is None:
            raise ValueError("No trained model loaded. Call train() or load() first.")
        return self.predictor

    def choice(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: Dict[str, str],
        question_name: str = "default",
    ) -> ChoiceResult:
        predictor = self._require_predictor()
        return predictor.choice(state, instructions, criteria, question_name=question_name)

    def score(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: List[str],
        question_name: str = "default",
    ) -> ScoreResult:
        predictor = self._require_predictor()
        return predictor.score(state, instructions, criteria, question_name=question_name)

    def noul(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        question_name: str = "default",
    ) -> NoulResult:
        predictor = self._require_predictor()
        return predictor.noul(state, instructions, question_name=question_name)

    def decide(
        self,
        state: Union[str, Dict[str, Any]],
        questions: List[DecisionQuestion],
    ) -> System1Result:
        predictor = self._require_predictor()
        payload = {}
        for question in questions:
            entry: Dict[str, Any] = {
                "type": question.type,
                "instructions": question.instructions,
            }
            if question.criteria is not None:
                entry["criteria"] = question.criteria
            payload[question.name] = entry
        return predictor.predict(state, payload)

    def route(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        router: DecisionRouter,
        threshold: Optional[float] = None,
        question_name: str = "default",
    ) -> Dict[str, Any]:
        predictor = self._require_predictor()
        result = predictor.choice(
            state,
            instructions,
            router.criteria,
            question_name=question_name,
        )
        selected = result.choice
        confidence = result.confidence
        if threshold is not None and confidence < threshold:
            selected = router.tiers[-1]
        return {
            "tier": selected,
            "confidence": confidence,
            "probabilities": result.probabilities,
            "tiers": router.tiers,
        }

    def batch_decide(
        self,
        states: List[Union[str, Dict[str, Any]]],
        questions: List[DecisionQuestion],
    ) -> BatchDecisions:
        batch = BatchDecisions()
        for state in states:
            batch.results.append(self.decide(state, questions))
        return batch

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "backend": self.backend,
            "model_path": self.model_path,
            "config": {
                "encoder_name": self.config.encoder_name,
                "classifier": self.config.classifier,
                "output_dir": self.config.output_dir,
                "device": self.config.device,
            },
        }
        if self.ollama_config is not None:
            payload["ollama"] = {
                "model": self.ollama_config.model,
                "base_url": self.ollama_config.resolved_base_url(),
            }
        return payload


def train_decision_model(
    examples: Union[List[System1Example], List[Dict[str, Any]]],
    config: Optional[System1Config] = None,
    output_dir: Optional[str] = None,
) -> str:
    if examples and isinstance(examples[0], dict):
        examples = _examples_from_records(examples)
    effective_config = config or System1Config()
    if output_dir is not None:
        effective_config = System1Config(
            encoder_name=effective_config.encoder_name,
            classifier=effective_config.classifier,
            classifier_kwargs=effective_config.classifier_kwargs,
            output_dir=output_dir,
            device=effective_config.device,
        )
    return train_system1(examples, effective_config)


def load_decision_model(path: str) -> DecisionSystem1:
    return DecisionSystem1(backend="local").load(path)


_default_decision_model: Optional[DecisionSystem1] = None
_default_decision_model_path: Optional[str] = None
_default_ollama_client: Optional[OllamaSystem1Client] = None
_default_ollama_key: Optional[tuple] = None


def _get_default_decision_model() -> Optional[DecisionSystem1]:
    global _default_decision_model, _default_decision_model_path
    env_path = os.environ.get("NPCPY_DECISION_MODEL")
    if env_path and (env_path != _default_decision_model_path or _default_decision_model is None):
        _default_decision_model = load_decision_model(env_path)
        _default_decision_model_path = env_path
    return _default_decision_model


def _get_ollama_client(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaSystem1Client:
    global _default_ollama_client, _default_ollama_key
    if model is not None or base_url is not None:
        return load_ollama_system1(model=model, base_url=base_url)
    key = (
        os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL"),
        os.environ.get("NPCPY_OLLAMA_BASE_URL"),
        os.environ.get("OLLAMA_HOST"),
    )
    if _default_ollama_client is None or key != _default_ollama_key:
        _default_ollama_client = load_ollama_system1()
        _default_ollama_key = key
    return _default_ollama_client


def decision_choice(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: Dict[str, str],
    model_path: Optional[str] = None,
    question_name: str = "default",
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ChoiceResult:
    if model_path is not None:
        return load_decision_model(model_path).choice(state, instructions, criteria, question_name=question_name)
    if _resolve_backend(backend) == "ollama":
        return _get_ollama_client(model, base_url).choice(
            state, instructions, criteria, question_name=question_name
        )
    default_model = _get_default_decision_model()
    if default_model is not None:
        return default_model.choice(state, instructions, criteria, question_name=question_name)
    return _system1_choice(state, instructions, criteria)


def decision_score(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: List[str],
    model_path: Optional[str] = None,
    question_name: str = "default",
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ScoreResult:
    if model_path is not None:
        return load_decision_model(model_path).score(state, instructions, criteria, question_name=question_name)
    if _resolve_backend(backend) == "ollama":
        return _get_ollama_client(model, base_url).score(
            state, instructions, criteria, question_name=question_name
        )
    default_model = _get_default_decision_model()
    if default_model is not None:
        return default_model.score(state, instructions, criteria, question_name=question_name)
    return _system1_score(state, instructions, criteria)


def decision_noul(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    model_path: Optional[str] = None,
    question_name: str = "default",
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> NoulResult:
    if model_path is not None:
        return load_decision_model(model_path).noul(state, instructions, question_name=question_name)
    if _resolve_backend(backend) == "ollama":
        return _get_ollama_client(model, base_url).noul(
            state, instructions, question_name=question_name
        )
    default_model = _get_default_decision_model()
    if default_model is not None:
        return default_model.noul(state, instructions, question_name=question_name)
    return _system1_noul(state, instructions)


def _questions_payload(questions: List[DecisionQuestion]) -> Dict[str, Dict[str, Any]]:
    payload = {q.name: {"type": q.type, "instructions": q.instructions} for q in questions}
    for q in questions:
        if q.criteria is not None:
            payload[q.name]["criteria"] = q.criteria
    return payload


def decision_predict(
    state: Union[str, Dict[str, Any]],
    questions: List[DecisionQuestion],
    model_path: Optional[str] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> System1Result:
    if model_path is not None:
        return load_decision_model(model_path).decide(state, questions)
    if _resolve_backend(backend) == "ollama":
        return _get_ollama_client(model, base_url).predict(state, _questions_payload(questions))
    default_model = _get_default_decision_model()
    if default_model is not None:
        return default_model.decide(state, questions)
    return _system1_predict(state, _questions_payload(questions))


def _route_result(
    result: ChoiceResult,
    router: DecisionRouter,
    threshold: Optional[float],
) -> Dict[str, Any]:
    selected = result.choice
    if threshold is not None and result.confidence < threshold:
        selected = router.tiers[-1]
    return {
        "tier": selected,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "tiers": router.tiers,
    }


def _route_with_fallback(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    router: DecisionRouter,
    threshold: Optional[float],
    question_name: str,
    predictor: Optional[Any],
) -> Dict[str, Any]:
    if predictor is not None:
        if isinstance(predictor, OllamaSystem1Client):
            result = predictor.choice(
                state, instructions, router.criteria, question_name=question_name
            )
            return _route_result(result, router, threshold)
        label_maps = getattr(predictor, "label_maps", {}) or {}
        criteria_keys = set(str(k) for k in router.criteria.keys())
        try:
            resolved = predictor._resolve_question_name(question_name)
        except Exception:
            resolved = None
        labels = set(label_maps.get(resolved, [])) if resolved else set()
        if resolved and criteria_keys.issubset(labels):
            result = predictor.choice(state, instructions, router.criteria, question_name=question_name)
            return _route_result(result, router, threshold)
    fallback = _system1_choice(state, instructions, router.criteria)
    return _route_result(fallback, router, threshold)


def decision_route(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    tiers: List[str],
    criteria: Optional[Dict[str, str]] = None,
    threshold: Optional[float] = None,
    model_path: Optional[str] = None,
    question_name: str = "default",
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    router = DecisionRouter(tiers=tiers, criteria=criteria)
    predictor: Optional[Any] = None
    if _resolve_backend(backend) == "ollama":
        predictor = _get_ollama_client(model, base_url)
    elif model_path is not None:
        predictor = load_decision_model(model_path).predictor
    else:
        default_model = _get_default_decision_model()
        if default_model is not None:
            predictor = default_model.predictor
    return _route_with_fallback(
        state, instructions, router, threshold, question_name, predictor
    )
