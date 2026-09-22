import json
import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("npcpy.ft.system1")

_sklearn_available = False
try:
    from sklearn.base import BaseEstimator
    _sklearn_available = True
except Exception:
    BaseEstimator = None

_sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except Exception:
    SentenceTransformer = None

_pydantic_available = False
try:
    from pydantic import BaseModel, Field
    _pydantic_available = True
except Exception:
    BaseModel = None
    Field = None


def _state_to_text(state: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, default=str)


def _extract_answer_label(example: "System1Example") -> str:
    answer = example.answer
    if isinstance(answer, dict):
        if "choice" in answer:
            return str(answer["choice"])
        if "score" in answer:
            return str(int(answer["score"]))
        if "noul" in answer:
            return "yes" if float(answer["noul"]) > 0.5 else "no"
    if example.question_type == "noul":
        return "yes" if float(answer) > 0.5 else "no"
    if example.question_type == "score":
        return str(int(answer))
    return str(answer)


def _build_question_text(state_text: str, instructions: str, criteria: Any = None) -> str:
    parts = [state_text, "", f"Question: {instructions}"]
    if criteria is not None:
        if isinstance(criteria, dict):
            parts.append("Options:")
            for key, desc in criteria.items():
                parts.append(f"  {key}: {desc}")
        elif isinstance(criteria, list):
            parts.append("Levels:")
            for idx, level in enumerate(criteria):
                parts.append(f"  {idx}: {level}")
        else:
            parts.append(f"Criteria: {criteria}")
    return "\n".join(parts)


@dataclass
class System1Config:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier: str = "LogisticRegression"
    classifier_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_iter": 1000})
    output_dir: str = "./system1_model"
    device: str = "cpu"


@dataclass
class System1Example:
    state: Union[str, Dict[str, Any], List[Any]]
    question_type: str
    instructions: str
    criteria: Any = None
    answer: Any = None
    question_name: str = "default"


@dataclass
class ChoiceResult:
    choice: str
    probabilities: Dict[str, float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"choice": self.choice, "probabilities": self.probabilities, "confidence": self.confidence}


@dataclass
class NoulResult:
    noul: float

    def to_dict(self) -> Dict[str, Any]:
        return {"noul": self.noul}


@dataclass
class ScoreResult:
    score: float
    probabilities: Dict[str, float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "probabilities": self.probabilities, "confidence": self.confidence}


@dataclass
class System1Result:
    answers: Dict[str, Union[ChoiceResult, NoulResult, ScoreResult]]
    routing: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "routing": self.routing,
        }


class System1Predictor:
    def __init__(self, encoders: Dict[str, Any], classifiers: Dict[str, Any], label_maps: Dict[str, List[str]], config: System1Config):
        self.encoders = encoders
        self.classifiers = classifiers
        self.label_maps = label_maps
        self.config = config

    def _resolve_question_name(self, question_name: str) -> str:
        if question_name in self.classifiers:
            return question_name
        if question_name == "default" and len(self.classifiers) == 1:
            return list(self.classifiers.keys())[0]
        raise ValueError(f"No trained model for question '{question_name}'")

    def _find_question_name(self, question_type: str, criteria: Any) -> str:
        if len(self.classifiers) == 1:
            return list(self.classifiers.keys())[0]
        candidates = []
        if question_type == "noul":
            for name, labels in self.label_maps.items():
                if set(labels) == {"yes", "no"}:
                    candidates.append(name)
        elif question_type == "choice" and isinstance(criteria, dict):
            keys = set(str(k) for k in criteria.keys())
            for name, labels in self.label_maps.items():
                overlap = len(set(labels) & keys)
                if overlap > 0:
                    candidates.append((overlap, name))
            candidates = [name for _, name in sorted(candidates, reverse=True)]
        elif question_type == "score" and isinstance(criteria, list):
            target_len = len(criteria)
            for name, labels in self.label_maps.items():
                try:
                    indices = [int(label) for label in labels]
                    if all(0 <= idx < target_len for idx in indices):
                        candidates.append(name)
                except Exception:
                    pass
        if candidates:
            return candidates[0]
        raise ValueError(f"Could not match {question_type} question to a trained model")

    def _encode(self, question_name: str, texts: List[str]) -> np.ndarray:
        resolved = self._resolve_question_name(question_name)
        encoder = self.encoders.get(resolved)
        if encoder is None:
            raise ValueError(f"No trained model for question '{question_name}'")
        if _sentence_transformers_available and isinstance(encoder, SentenceTransformer):
            embeddings = encoder.encode(texts, device=self.config.device, convert_to_numpy=True, show_progress_bar=False)
            return np.asarray(embeddings)
        return np.asarray(encoder(texts))

    def _classifier_predict(self, question_name: str, features: np.ndarray) -> tuple:
        resolved = self._resolve_question_name(question_name)
        classifier = self.classifiers.get(resolved)
        if classifier is None:
            raise ValueError(f"No trained classifier for question '{question_name}'")
        labels = self.label_maps.get(resolved, [])
        if hasattr(classifier, "predict_proba"):
            prob_matrix = classifier.predict_proba(features)
            pred_idx = int(np.argmax(prob_matrix[0]))
            pred_label = labels[pred_idx] if pred_idx < len(labels) else classifier.classes_[pred_idx]
            probabilities = {}
            classes = getattr(classifier, "classes_", labels)
            for idx, cls in enumerate(classes):
                if idx < prob_matrix.shape[1]:
                    probabilities[str(cls)] = float(prob_matrix[0][idx])
            confidence = float(prob_matrix[0][pred_idx])
            return pred_label, probabilities, confidence
        pred = classifier.predict(features)[0]
        return str(pred), {}, 1.0

    def choice(self, state: Union[str, Dict[str, Any]], instructions: str, criteria: Dict[str, str], question_name: str = "default") -> ChoiceResult:
        effective_name = self._find_question_name("choice", criteria) if question_name == "default" else question_name
        text = _build_question_text(_state_to_text(state), instructions, criteria)
        features = self._encode(effective_name, [text])
        pred_label, probabilities, confidence = self._classifier_predict(effective_name, features)
        probabilities = {str(k): float(v) for k, v in probabilities.items()}
        if pred_label not in criteria:
            pred_label = max(probabilities, key=probabilities.get, default=pred_label)
        return ChoiceResult(choice=str(pred_label), probabilities=probabilities, confidence=confidence)

    def noul(self, state: Union[str, Dict[str, Any]], instructions: str, question_name: str = "default") -> NoulResult:
        effective_name = self._find_question_name("noul", None) if question_name == "default" else question_name
        text = _build_question_text(_state_to_text(state), instructions)
        features = self._encode(effective_name, [text])
        pred_label, probabilities, confidence = self._classifier_predict(effective_name, features)
        if "yes" in probabilities and "no" in probabilities:
            value = probabilities.get("yes", 0.0)
        elif pred_label == "yes":
            value = confidence
        else:
            value = 1.0 - confidence
        return NoulResult(noul=float(np.clip(value, 0.0, 1.0)))

    def score(self, state: Union[str, Dict[str, Any]], instructions: str, criteria: List[str], question_name: str = "default") -> ScoreResult:
        effective_name = self._find_question_name("score", criteria) if question_name == "default" else question_name
        text = _build_question_text(_state_to_text(state), instructions, criteria)
        features = self._encode(effective_name, [text])
        pred_label, probabilities, confidence = self._classifier_predict(effective_name, features)
        probabilities = {str(k): float(v) for k, v in probabilities.items()}
        score_value = 0.0
        if probabilities:
            total = sum(probabilities.values())
            if total > 0:
                score_value = sum(float(k) * v for k, v in probabilities.items()) / total
        else:
            try:
                score_value = float(pred_label)
            except Exception:
                score_value = 0.0
        max_prob = max(probabilities.values(), default=confidence)
        return ScoreResult(score=float(score_value), probabilities=probabilities, confidence=float(max_prob))

    def predict(self, state: Union[str, Dict[str, Any]], questions: Dict[str, Dict[str, Any]]) -> System1Result:
        answers = {}
        for name, question in questions.items():
            qtype = question.get("type", "choice")
            instructions = question.get("instructions", "")
            criteria = question.get("criteria")
            if qtype == "choice":
                answers[name] = self.choice(state, instructions, criteria, question_name=name)
            elif qtype == "noul":
                answers[name] = self.noul(state, instructions, question_name=name)
            elif qtype == "score":
                answers[name] = self.score(state, instructions, criteria, question_name=name)
            else:
                raise ValueError(f"Unknown question type: {qtype}")
        return System1Result(answers=answers)


def _get_sentence_transformer_encoder(model_name: str, device: str):
    if not _sentence_transformers_available:
        raise ImportError("sentence-transformers is required for System 1 training. Install with: pip install sentence-transformers")
    return SentenceTransformer(model_name, device=device)


def _get_sklearn_model(model_name: str, kwargs: Dict[str, Any]):
    if not _sklearn_available:
        raise ImportError("scikit-learn is required for System 1 training. Install with: pip install scikit-learn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    mapping = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "SVC": SVC,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "GaussianNB": GaussianNB,
        "MLPClassifier": MLPClassifier,
    }
    if model_name not in mapping:
        raise ValueError(f"Unsupported classifier: {model_name}")
    return mapping[model_name](**kwargs)


def train_system1(examples: List[System1Example], config: Optional[System1Config] = None) -> str:
    config = config or System1Config()
    if not examples:
        raise ValueError("At least one training example is required")

    by_question: Dict[str, List[System1Example]] = {}
    for ex in examples:
        by_question.setdefault(ex.question_name, []).append(ex)

    encoders = {}
    classifiers = {}
    label_maps = {}

    shared_encoder = _get_sentence_transformer_encoder(config.encoder_name, config.device)

    for question_name, question_examples in by_question.items():
        texts = []
        labels = []
        for ex in question_examples:
            state_text = _state_to_text(ex.state)
            text = _build_question_text(state_text, ex.instructions, ex.criteria)
            texts.append(text)
            labels.append(_extract_answer_label(ex))

        embeddings = shared_encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        classifier = _get_sklearn_model(config.classifier, config.classifier_kwargs)
        classifier.fit(embeddings, labels)

        encoders[question_name] = shared_encoder
        classifiers[question_name] = classifier
        label_maps[question_name] = sorted(set(labels))

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "classifiers.pkl"), "wb") as f:
        pickle.dump(classifiers, f)
    with open(os.path.join(config.output_dir, "label_maps.json"), "w") as f:
        json.dump(label_maps, f, indent=2)
    meta = {
        "encoder_name": config.encoder_name,
        "classifier": config.classifier,
        "classifier_kwargs": config.classifier_kwargs,
        "device": config.device,
        "question_names": list(by_question.keys()),
    }
    with open(os.path.join(config.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shared_encoder.save(os.path.join(config.output_dir, "encoder"))
    return config.output_dir


def load_system1(path: str, device: str = None) -> System1Predictor:
    with open(os.path.join(path, "meta.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(path, "label_maps.json"), "r") as f:
        label_maps = json.load(f)
    with open(os.path.join(path, "classifiers.pkl"), "rb") as f:
        classifiers = pickle.load(f)

    encoder_name = meta.get("encoder_name", os.path.join(path, "encoder"))
    encoder_path = os.path.join(path, "encoder")
    if os.path.isdir(encoder_path):
        encoder_name = encoder_path
    effective_device = device if device is not None else meta.get("device", "cpu")
    encoders = {name: _get_sentence_transformer_encoder(encoder_name, effective_device) for name in classifiers}

    config = System1Config(
        encoder_name=encoder_name,
        classifier=meta.get("classifier", "LogisticRegression"),
        classifier_kwargs=meta.get("classifier_kwargs", {}),
        output_dir=path,
        device=effective_device,
    )
    return System1Predictor(encoders, classifiers, label_maps, config)


class _ChoiceOutput(BaseModel):
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class _NoulOutput(BaseModel):
    noul: float


class _ScoreOutput(BaseModel):
    score: float
    probabilities: Dict[str, float]
    confidence: float


_default_predictor: Optional[System1Predictor] = None
_default_model_path: Optional[str] = None


def _get_default_predictor() -> Optional[System1Predictor]:
    global _default_predictor, _default_model_path
    env_path = os.environ.get("NPCPY_SYSTEM1_MODEL")
    if env_path and (env_path != _default_model_path or _default_predictor is None):
        _default_predictor = load_system1(env_path)
        _default_model_path = env_path
    return _default_predictor


def _llm_choice(state, instructions, criteria, model=None, provider=None, **kwargs) -> ChoiceResult:
    from npcpy.llm_funcs import get_llm_response
    state_text = _state_to_text(state)
    option_lines = "\n".join(f"{k}: {v}" for k, v in criteria.items())
    prompt = f"State:\n{state_text}\n\nQuestion: {instructions}\n\nOptions:\n{option_lines}\n\nReturn the selected option, probability for every option, and confidence."
    output_model = _ChoiceOutput
    response = get_llm_response(prompt, model=model, provider=provider, format=output_model, **kwargs)
    parsed = response.get("response")
    if isinstance(parsed, _ChoiceOutput):
        return ChoiceResult(choice=parsed.choice, probabilities=parsed.probabilities, confidence=parsed.confidence)
    if isinstance(parsed, dict):
        return ChoiceResult(
            choice=str(parsed.get("choice", "")),
            probabilities={str(k): float(v) for k, v in parsed.get("probabilities", {}).items()},
            confidence=float(parsed.get("confidence", 0.0)),
        )
    if isinstance(parsed, str):
        try:
            data = json.loads(parsed)
            return ChoiceResult(
                choice=str(data.get("choice", "")),
                probabilities={str(k): float(v) for k, v in data.get("probabilities", {}).items()},
                confidence=float(data.get("confidence", 0.0)),
            )
        except Exception:
            pass
    return ChoiceResult(choice="", probabilities={}, confidence=0.0)


def _llm_noul(state, instructions, model=None, provider=None, **kwargs) -> NoulResult:
    from npcpy.llm_funcs import get_llm_response
    state_text = _state_to_text(state)
    prompt = f"State:\n{state_text}\n\nQuestion: {instructions}\n\nReturn a single number from 0.0 to 1.0 representing P(true)."
    output_model = _NoulOutput
    response = get_llm_response(prompt, model=model, provider=provider, format=output_model, **kwargs)
    parsed = response.get("response")
    if isinstance(parsed, _NoulOutput):
        return NoulResult(noul=float(np.clip(parsed.noul, 0.0, 1.0)))
    if isinstance(parsed, dict):
        return NoulResult(noul=float(np.clip(parsed.get("noul", 0.5), 0.0, 1.0)))
    if isinstance(parsed, str):
        try:
            data = json.loads(parsed)
            return NoulResult(noul=float(np.clip(data.get("noul", 0.5), 0.0, 1.0)))
        except Exception:
            pass
    return NoulResult(noul=0.5)


def _llm_score(state, instructions, criteria, model=None, provider=None, **kwargs) -> ScoreResult:
    from npcpy.llm_funcs import get_llm_response
    state_text = _state_to_text(state)
    level_lines = "\n".join(f"{i}: {level}" for i, level in enumerate(criteria))
    prompt = f"State:\n{state_text}\n\nQuestion: {instructions}\n\nLevels:\n{level_lines}\n\nReturn the score, probability for every level, and confidence."
    output_model = _ScoreOutput
    response = get_llm_response(prompt, model=model, provider=provider, format=output_model, **kwargs)
    parsed = response.get("response")
    if isinstance(parsed, _ScoreOutput):
        return ScoreResult(score=parsed.score, probabilities=parsed.probabilities, confidence=parsed.confidence)
    if isinstance(parsed, dict):
        return ScoreResult(
            score=float(parsed.get("score", 0.0)),
            probabilities={str(k): float(v) for k, v in parsed.get("probabilities", {}).items()},
            confidence=float(parsed.get("confidence", 0.0)),
        )
    if isinstance(parsed, str):
        try:
            data = json.loads(parsed)
            return ScoreResult(
                score=float(data.get("score", 0.0)),
                probabilities={str(k): float(v) for k, v in data.get("probabilities", {}).items()},
                confidence=float(data.get("confidence", 0.0)),
            )
        except Exception:
            pass
    return ScoreResult(score=0.0, probabilities={}, confidence=0.0)


def choice(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: Dict[str, str],
    predictor: Optional[System1Predictor] = None,
    **kwargs,
) -> ChoiceResult:
    predictor = predictor or _get_default_predictor()
    if predictor is not None:
        return predictor.choice(state, instructions, criteria)
    if not _pydantic_available:
        raise ImportError("pydantic is required for LLM fallback in system1.choice")
    return _llm_choice(state, instructions, criteria, **kwargs)


def noul(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    predictor: Optional[System1Predictor] = None,
    **kwargs,
) -> NoulResult:
    predictor = predictor or _get_default_predictor()
    if predictor is not None:
        return predictor.noul(state, instructions)
    if not _pydantic_available:
        raise ImportError("pydantic is required for LLM fallback in system1.noul")
    return _llm_noul(state, instructions, **kwargs)


def score(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: List[str],
    predictor: Optional[System1Predictor] = None,
    **kwargs,
) -> ScoreResult:
    predictor = predictor or _get_default_predictor()
    if predictor is not None:
        return predictor.score(state, instructions, criteria)
    if not _pydantic_available:
        raise ImportError("pydantic is required for LLM fallback in system1.score")
    return _llm_score(state, instructions, criteria, **kwargs)


def predict(
    state: Union[str, Dict[str, Any]],
    questions: Dict[str, Dict[str, Any]],
    predictor: Optional[System1Predictor] = None,
    **kwargs,
) -> System1Result:
    predictor = predictor or _get_default_predictor()
    if predictor is not None:
        return predictor.predict(state, questions)
    if not _pydantic_available:
        raise ImportError("pydantic is required for LLM fallback in system1.predict")
    answers = {}
    for name, question in questions.items():
        qtype = question.get("type", "choice")
        instructions = question.get("instructions", "")
        criteria = question.get("criteria")
        if qtype == "choice":
            answers[name] = _llm_choice(state, instructions, criteria, **kwargs)
        elif qtype == "noul":
            answers[name] = _llm_noul(state, instructions, **kwargs)
        elif qtype == "score":
            answers[name] = _llm_score(state, instructions, criteria, **kwargs)
        else:
            raise ValueError(f"Unknown question type: {qtype}")
    return System1Result(answers=answers)
