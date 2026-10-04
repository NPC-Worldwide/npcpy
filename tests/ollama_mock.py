"""A tiny mock of the Ollama HTTP API for System One tests.

Reproduces the documented ``POST /v1/systemone`` request/response shape plus the
``/api/version`` and ``/api/tags`` endpoints used by availability checks, so
tests need neither a real Ollama install nor the ``nimble`` weights.
"""

import json
from http.server import BaseHTTPRequestHandler


class MockOllamaHandler(BaseHTTPRequestHandler):
    requests_seen = []
    version = "0.35.1"
    models = ["nimble:latest", "qwen3:1.7b"]
    fail_status = None
    fail_body = None

    def log_message(self, *args):
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send(200, {"version": self.version})
        elif self.path == "/api/tags":
            self._send(200, {"models": [{"name": n} for n in self.models]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {}
        MockOllamaHandler.requests_seen.append(
            {"path": self.path, "body": body, "headers": dict(self.headers)}
        )
        if MockOllamaHandler.fail_status is not None:
            self._send(
                MockOllamaHandler.fail_status,
                MockOllamaHandler.fail_body or {"error": "boom"},
            )
            return
        if self.path != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        answers = {}
        for name, question in (body.get("questions") or {}).items():
            qtype = question.get("type", "choice")
            if qtype == "choice":
                criteria = question.get("criteria") or {}
                keys = [str(k) for k in criteria.keys()] or ["other"]
                choice = "billing" if "billing" in keys else keys[0]
                rest = max(1, len(keys) - 1)
                probs = {k: (0.9781 if k == choice else round(0.0219 / rest, 4)) for k in keys}
                answers[name] = {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probs,
                    "confidence": 0.8906,
                }
            elif qtype == "noul":
                answers[name] = {"type": "noul", "noul": 0.997}
            elif qtype == "score":
                criteria = question.get("criteria") or ["a", "b", "c"]
                legend = {str(i): str(c) for i, c in enumerate(criteria)}
                probs = {str(i): round(1.0 / len(criteria), 4) for i in range(len(criteria))}
                answers[name] = {
                    "type": "score",
                    "score": 0.815,
                    "legend": legend,
                    "probabilities": probs,
                    "confidence": 0.046,
                }
            else:
                self._send(400, {"error": f"unsupported question type {qtype}"})
                return
        self._send(
            200,
            {
                "model": body.get("model"),
                "answers": answers,
                "usage": {"input_tokens": 174, "output_tokens": len(answers)},
            },
        )
