"""A tiny mock of the hosted Decisions APIs for tests.

Covers the three request shapes we support:

* ``openai``     -- ``{model, input, questions: [...]}`` with ``predicate`` questions
* ``gemini``     -- ``{model, input, questions: {name: {...}}}``
* ``perplexity`` -- ``{model, state, questions: {name: {...}}}``

and both response shapes: a map keyed by question name (Gemini, Perplexity) and
an ordered array carrying ``name`` (OpenAI). No network access or API key is
needed, so tests run offline.
"""

import json
from http.server import BaseHTTPRequestHandler


class MockDecisionsHandler(BaseHTTPRequestHandler):
    requests_seen = []
    fail_status = None
    fail_body = None
    refusal_question = None

    def log_message(self, *args):
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {}
        MockDecisionsHandler.requests_seen.append(
            {"path": self.path, "body": body, "headers": dict(self.headers)}
        )
        if MockDecisionsHandler.fail_status is not None:
            self._send(
                MockDecisionsHandler.fail_status,
                MockDecisionsHandler.fail_body or {"error": {"message": "boom"}},
            )
            return
        if not self.path.endswith("/decisions"):
            self._send(404, {"error": {"message": "not found"}})
            return

        questions = body.get("questions")
        provider = self._detect_provider(body, questions)
        if provider == "openai":
            answers = [self._openai_answer(q) for q in questions or []]
            payload = {
                "model": body.get("model"),
                "answers": answers,
                "usage": {"input_tokens": 96, "output_tokens": 0},
            }
        else:
            answers = {
                name: self._map_answer(question)
                for name, question in (questions or {}).items()
            }
            payload = {
                "model": body.get("model"),
                "answers": answers,
                "usage": {"input_tokens": 96, "total_output_tokens": 0},
            }
        self._send(200, payload)

    @staticmethod
    def _detect_provider(body, questions):
        if "input" in body and isinstance(questions, list):
            return "openai"
        if "state" in body:
            return "perplexity"
        return "gemini"

    def _maybe_refuse(self, name):
        if MockDecisionsHandler.refusal_question == name:
            return {"type": "refusal", "name": name, "refusal": "cannot answer"}
        return None

    def _openai_answer(self, question):
        name = question.get("name") or "default"
        refusal = self._maybe_refuse(name)
        if refusal:
            return refusal
        qtype = question.get("type")
        if qtype == "predicate":
            return {"type": "predicate", "name": name, "probability": 0.93}
        if qtype == "choice":
            values = [c.get("value") for c in question.get("choices", [])]
            top = values[0] if values else ""
            return {
                "type": "choice",
                "name": name,
                "choice": top,
                "confidence": 0.9,
                "probabilities": [
                    {"value": value, "probability": 1.0 if value == top else 0.0}
                    for value in values
                ],
            }
        levels = question.get("levels", [])
        return {
            "type": "score",
            "name": name,
            "score": 1.0,
            "confidence": 0.7,
            "probabilities": [
                {
                    "value": index,
                    "label": level.get("label"),
                    "probability": 1.0 if index == 1 else 0.0,
                }
                for index, level in enumerate(levels)
            ],
        }

    def _map_answer(self, question):
        qtype = question.get("type")
        if qtype == "noul":
            return {"type": "noul", "noul": 0.97, "confidence": 0.97}
        if qtype == "choice":
            criteria = question.get("criteria") or {}
            keys = [str(k) for k in criteria] or ["other"]
            top = "billing" if "billing" in keys else keys[0]
            rest = max(1, len(keys) - 1)
            return {
                "type": "choice",
                "choice": top,
                "confidence": 0.8906,
                "probabilities": {
                    key: (0.9781 if key == top else round(0.0219 / rest, 4))
                    for key in keys
                },
            }
        criteria = question.get("criteria") or ["a", "b", "c"]
        return {
            "type": "score",
            "score": 1.25,
            "confidence": 0.62,
            "legend": {str(i): str(level) for i, level in enumerate(criteria)},
            "probabilities": {
                str(i): (0.75 if i == 1 else round(0.25 / max(1, len(criteria) - 1), 4))
                for i in range(len(criteria))
            },
        }
