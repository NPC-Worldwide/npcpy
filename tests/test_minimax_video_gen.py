import importlib.util
from pathlib import Path


def test_minimax_video_generation_and_polling(monkeypatch, tmp_path):
    module_path = Path(__file__).parents[1] / "npcpy" / "gen" / "video_gen.py"
    spec = importlib.util.spec_from_file_location("video_gen", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generate_video_minimax = module.generate_video_minimax

    calls = []

    class Response:
        def __init__(self, payload=None, content=b"video"):
            self._payload = payload
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def post(url, **kwargs):
        calls.append(("post", url, kwargs["json"]))
        return Response({"task_id": "task-1"})

    statuses = iter([
        {"task": {"id": "task-1", "status": "processing"}},
        {"task": {"id": "task-1", "status": "success", "content": {"url": "https://cdn.example/video.mp4"}}},
    ])

    def get(url, **kwargs):
        calls.append(("get", url))
        if url.startswith("https://cdn.example"):
            return Response(content=b"mp4")
        return Response(next(statuses))

    monkeypatch.setenv("MINIMAX_API_KEY", "secret")
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("requests.get", get)
    output = tmp_path / "video.mp4"
    result = generate_video_minimax("A cat runs", output_path=str(output), poll_interval=0)

    assert result == str(output)
    assert output.read_bytes() == b"mp4"
    assert calls[0][2]["model"] == "MiniMax-H3"
    assert calls[0][2]["content"] == [{"type": "text", "text": "A cat runs"}]
    assert any("/v2/query/video_generation/task-1" in call[1] for call in calls if call[0] == "get")
