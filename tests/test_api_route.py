"""API Route resolution and real SDK routing against a local capture server."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from npcpy.gen import response as response_module
from npcpy.gen.response import get_litellm_response
from npcpy.llm_funcs import get_llm_response, resolve_model_provider


@pytest.fixture
def api_server():
    captured = []

    class CaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            captured.append((self.path, self.headers.get('Authorization'), body))
            payload = {
                'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'pong'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), CaptureHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize('provider', ['api-route', 'api_route'])
def test_resolution_defaults(provider, monkeypatch):
    monkeypatch.delenv('API_ROUTE_API_URL', raising=False)
    monkeypatch.setenv('API_ROUTE_API_KEY', 'api-route-test-key')
    assert resolve_model_provider(model='gpt-6.1-sol', provider=provider) == (
        'gpt-6.1-sol', provider, 'https://global.api-route.com/v1', 'api-route-test-key',
    )


def test_resolution_environment_url_override(monkeypatch):
    monkeypatch.setenv('API_ROUTE_API_URL', 'https://custom.invalid/v1')
    monkeypatch.setenv('API_ROUTE_API_KEY', 'env-key')
    assert resolve_model_provider(model='vendor/model', provider='api-route')[2:] == (
        'https://custom.invalid/v1', 'env-key',
    )


def test_explicit_credentials_override_environment_and_npc(monkeypatch):
    monkeypatch.setenv('API_ROUTE_API_URL', 'https://env.invalid/v1')
    monkeypatch.setenv('API_ROUTE_API_KEY', 'env-key')
    npc = SimpleNamespace(api_url='https://npc.invalid/v1', api_key='npc-key')
    assert resolve_model_provider(
        npc=npc, model='gpt-6.1-sol', provider='api-route', api_url='https://explicit.invalid/v1', api_key='explicit-key',
    )[2:] == ('https://explicit.invalid/v1', 'explicit-key')


@pytest.mark.parametrize('call_provider', [get_llm_response, get_litellm_response])
@pytest.mark.parametrize('provider', ['api-route', 'api_route'])
@pytest.mark.parametrize('model', ['gpt-6.1-sol', 'vendor/model'])
def test_transport_preserves_endpoint_key_and_model(call_provider, provider, model, api_server, monkeypatch):
    base_url, captured = api_server
    monkeypatch.setenv('API_ROUTE_API_URL', base_url)
    monkeypatch.setenv('API_ROUTE_API_KEY', 'api-route-test-key')
    result = call_provider('Reply pong.', model=model, provider=provider)
    assert result['response'] == 'pong'
    assert len(captured) == 1
    path, authorization, body = captured[0]
    assert path == '/v1/chat/completions'
    assert authorization == 'Bearer api-route-test-key'
    assert body['model'] == model


@pytest.mark.parametrize('call_provider', [get_llm_response, get_litellm_response])
def test_missing_key_does_not_fall_back_to_openai_credentials(call_provider, monkeypatch):
    monkeypatch.delenv('API_ROUTE_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-sent')
    calls = []
    monkeypatch.setattr(response_module, 'completion', lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match='API_ROUTE_API_KEY'):
        call_provider('Hello', model='gpt-6.1-sol', provider='api-route')
    assert calls == []
