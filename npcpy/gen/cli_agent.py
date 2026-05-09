import hashlib
import json
import os
import subprocess
import uuid
from typing import Dict, Any, List, Optional

_CLI_PROVIDERS = {
    "claude_code", "claude", "opencode", "codex",
    "kimi", "kimi_code", "kilo", "kilo_code",
    "gemini", "amp", "aider", "nanocoder",
}


def _is_cli_provider(provider: str) -> bool:
    return provider in _CLI_PROVIDERS


# ── Stream parsers ──────────────────────────────────────────────────────────

class _ParserResult:
    __slots__ = ("text", "usage", "session_id", "cost")
    def __init__(self, text="", usage=None, session_id=None, cost=0.0):
        self.text = text
        self.usage = usage or {}
        self.session_id = session_id
        self.cost = cost


class _BaseStreamParser:
    """Parse JSONL/JSON streams from CLI tools."""
    def __init__(self):
        self._text_parts: List[str] = []
        self._usage: Dict[str, Any] = {}
        self._session_id: Optional[str] = None
        self._cost: float = 0.0

    def feed(self, line: str) -> None:
        """Process one line of output."""
        raise NotImplementedError

    def finalize(self) -> _ParserResult:
        return _ParserResult(
            text="".join(self._text_parts),
            usage=self._usage,
            session_id=self._session_id,
            cost=self._cost,
        )


class _ClaudeStreamParser(_BaseStreamParser):
    """Parse claude --output-format stream-json."""
    def feed(self, line: str) -> None:
        if not line.strip():
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            self._text_parts.append(line)
            return
        t = obj.get("type")
        if t == "text":
            self._text_parts.append(obj.get("text", ""))
        elif t == "system":
            sid = obj.get("session_id")
            if sid:
                self._session_id = sid
        elif t == "usage":
            self._usage = {
                "input_tokens": obj.get("input_tokens", 0),
                "output_tokens": obj.get("output_tokens", 0),
            }


class _OpencodeStreamParser(_BaseStreamParser):
    """Parse opencode --format json."""
    def feed(self, line: str) -> None:
        if not line.strip():
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        t = obj.get("type")
        if t == "text" or t == "content":
            self._text_parts.append(obj.get("text", obj.get("content", "")))
        elif t == "system":
            sid = obj.get("session_id")
            if sid:
                self._session_id = sid


class _CodexStreamParser(_BaseStreamParser):
    """Parse codex --json output."""
    def feed(self, line: str) -> None:
        if not line.strip():
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        t = obj.get("type")
        if t == "text":
            self._text_parts.append(obj.get("text", ""))
        elif t in ("thread.started", "thread.resumed"):
            tid = obj.get("thread_id") or obj.get("id")
            if tid:
                self._session_id = tid
        elif t == "usage":
            self._usage = {
                "input_tokens": obj.get("input_tokens", 0),
                "output_tokens": obj.get("output_tokens", 0),
            }


class _KimiStreamParser(_BaseStreamParser):
    """Parse kimi --output-format stream-json."""
    def feed(self, line: str) -> None:
        if not line.strip():
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        t = obj.get("type")
        if t == "text":
            self._text_parts.append(obj.get("text", ""))
        elif t == "system":
            sid = obj.get("session_id")
            if sid:
                self._session_id = sid
        elif t == "usage":
            self._usage = {
                "input_tokens": obj.get("input_tokens", 0),
                "output_tokens": obj.get("output_tokens", 0),
            }


_STREAM_PARSERS = {
    "claude": _ClaudeStreamParser,
    "claude_code": _ClaudeStreamParser,
    "opencode": _OpencodeStreamParser,
    "codex": _CodexStreamParser,
    "kimi": _KimiStreamParser,
    "kimi_code": _KimiStreamParser,
}


# ── Session resolvers ───────────────────────────────────────────────────────

def _fetch_opencode_session_id():
    try:
        r = subprocess.run(["opencode", "session", "list"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].startswith("ses_"):
                return parts[0]
    except Exception:
        pass
    return None


def _fetch_kilo_session_id():
    try:
        r = subprocess.run(["kilo", "session", "list"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].startswith("ses_"):
                return parts[0]
    except Exception:
        pass
    return None


def _fetch_kimi_session_id():
    try:
        cwd = os.getcwd()
        cwd_hash = hashlib.md5(cwd.encode()).hexdigest()
        sessions_base = os.path.expanduser("~/.kimi/sessions")
        sessions_dir = os.path.join(sessions_base, cwd_hash)
        if not os.path.isdir(sessions_dir):
            sessions_dir = os.path.join(sessions_base, cwd_hash[:8])
        if not os.path.isdir(sessions_dir):
            return None
        entries = [(os.path.getmtime(os.path.join(sessions_dir, e)), e) for e in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, e))]
        return max(entries)[1] if entries else None
    except Exception:
        return None


_SESSION_RESOLVERS = {
    "opencode": _fetch_opencode_session_id,
    "kilo": _fetch_kilo_session_id,
    "kilo_code": _fetch_kilo_session_id,
    "kimi": _fetch_kimi_session_id,
    "kimi_code": _fetch_kimi_session_id,
}


# ── Usage extractors ────────────────────────────────────────────────────────

def _extract_kimi_usage(session_id: str) -> Dict[str, Any]:
    """Read token usage from kimi wire.jsonl."""
    try:
        cwd = os.getcwd()
        cwd_hash = hashlib.md5(cwd.encode()).hexdigest()
        sessions_base = os.path.expanduser("~/.kimi/sessions")
        wire_path = os.path.join(sessions_base, cwd_hash, session_id, "wire.jsonl")
        if not os.path.exists(wire_path):
            wire_path = os.path.join(sessions_base, cwd_hash[:8], session_id, "wire.jsonl")
        if not os.path.exists(wire_path):
            return {}
        total_input = 0
        total_output = 0
        with open(wire_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage:
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
        }
    except Exception:
        return {}


_USAGE_EXTRACTORS = {
    "kimi": _extract_kimi_usage,
    "kimi_code": _extract_kimi_usage,
}


# ── Command builder ───────────────────────────────────────────────────────────

def _wrap_with_system(prompt, system_prompt, session_id):
    if session_id or not system_prompt:
        return prompt
    return f"<system>\n{system_prompt}\n</system>\n\n{prompt}"


def _build_history_summary(history, max_turns=8):
    if not history:
        return ""
    turns = [m for m in history if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    turns = turns[-(max_turns * 2):]
    lines = ["=== Previous conversation ==="]
    i = 0
    while i < len(turns) - 1:
        u_msg = turns[i]
        a_msg = turns[i + 1]
        if u_msg.get("role") == "user" and a_msg.get("role") == "assistant":
            u = str(u_msg.get("content", ""))[:300]
            a = str(a_msg.get("content", ""))[:400]
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
            lines.append("---")
        i += 2
    lines.append("=== End of context ===")
    return "\n".join(lines)


def _build_cli_cmd(provider, model, prompt, system_prompt=None, session_id=None, history=None, images=None, think=None):
    if provider in ("claude_code", "claude"):
        sid = session_id or str(uuid.uuid4())
        cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--session-id", sid]
        if model:
            cmd += ["--model", model]
        if system_prompt and not session_id:
            cmd += ["--system-prompt", system_prompt]
        if images:
            for img in images:
                cmd += ["--image", img]
        if think:
            cmd += ["--think-mode", "auto"]
        return cmd

    if provider == "opencode":
        full = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ["opencode", "run", full, "--format", "json"]
        if model:
            cmd += ["-m", model]
        if session_id:
            cmd += ["-s", session_id]
        return cmd

    if provider == "codex":
        full = _wrap_with_system(prompt, system_prompt, session_id)
        if session_id == "last":
            cmd = ["codex", "exec", "resume", "--last", "--json", full]
        elif session_id:
            cmd = ["codex", "exec", "resume", session_id, "--json", full]
        else:
            cmd = ["codex", "exec", "--json", full]
        if model:
            cmd += ["--model", model]
        return cmd

    if provider in ("kimi", "kimi_code"):
        full = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ["kimi", "--print", "--output-format", "stream-json", "-p", full]
        if model:
            cmd += ["-m", model]
        if session_id:
            cmd += ["-S", session_id]
        if think:
            cmd += ["--thinking"]
        return cmd

    if provider in ("kilo", "kilo_code"):
        full = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ["kilo", "run", full, "--format", "json"]
        if model:
            cmd += ["-m", model]
        if session_id:
            cmd += ["-s", session_id]
        return cmd

    if provider == "gemini":
        full = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ["gemini", "-p", full]
        if model:
            cmd += ["-m", model]
        if images:
            for img in images:
                cmd += ["--image", img]
        return cmd

    if provider == "amp":
        full = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ["amp", "run", full]
        if model:
            cmd += ["--model", model]
        return cmd

    if provider == "aider":
        summary = _build_history_summary(history) if history and session_id else ""
        base = f"{summary}\n\n{prompt}" if summary else prompt
        full = _wrap_with_system(base, system_prompt, session_id)
        cmd = ["aider", "--message", full, "--no-pretty"]
        if model:
            cmd += ["--model", model]
        return cmd

    if provider == "nanocoder":
        return ["nanocoder", "-p", prompt]

    return None


# ── Subprocess runners ──────────────────────────────────────────────────────

def _run_subprocess(cmd, verbose=False):
    if verbose:
        print(f"[cli_agent] running: {' '.join(cmd)}", file=os.sys.stderr)
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output_lines = []
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                output_lines.append(line)
                if verbose:
                    print(line, file=os.sys.stderr)
        process.wait()
        output = "\n".join(output_lines)
        if process.returncode != 0 and verbose:
            print(f"[cli_agent] exited with code {process.returncode}", file=os.sys.stderr)
        return {"response": output, "output": output, "messages": [], "usage": {}}
    except Exception as e:
        return {"response": f"Error running CLI: {e}", "output": f"Error running CLI: {e}", "messages": [], "usage": {}}


def _run_subprocess_parsed(cmd, parser: _BaseStreamParser, verbose=False):
    """Run subprocess with live stream parsing."""
    if verbose:
        print(f"[cli_agent] running (parsed): {' '.join(cmd)}", file=os.sys.stderr)
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        raw_lines = []
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                raw_lines.append(line)
                parser.feed(line)
                if verbose:
                    print(line, file=os.sys.stderr)
        process.wait()
        result = parser.finalize()
        if process.returncode != 0 and verbose:
            print(f"[cli_agent] exited with code {process.returncode}", file=os.sys.stderr)
        return {
            "response": result.text,
            "output": result.text,
            "messages": [],
            "usage": result.usage,
            "session_id": result.session_id,
            "cost": result.cost,
            "raw": "\n".join(raw_lines),
        }
    except Exception as e:
        return {
            "response": f"Error running CLI: {e}",
            "output": f"Error running CLI: {e}",
            "messages": [],
            "usage": {},
            "session_id": None,
            "cost": 0.0,
            "raw": "",
        }


# ── Public API ──────────────────────────────────────────────────────────────

def run_cli_agent(provider, prompt, model=None, system_prompt=None, session_id=None, history=None, images=None, think=None, n_samples=1, verbose=False):
    if not _is_cli_provider(provider):
        return {"response": f"Unknown CLI provider: {provider}", "output": f"Unknown CLI provider: {provider}", "messages": [], "usage": {}}

    if not session_id:
        resolver = _SESSION_RESOLVERS.get(provider)
        if resolver:
            session_id = resolver()

    # For claude: pre-assign a UUID on first call so subsequent calls reuse it
    if provider in ("claude", "claude_code") and not session_id:
        session_id = str(uuid.uuid4())

    parser_cls = _STREAM_PARSERS.get(provider)

    if n_samples > 1:
        results = []
        for _ in range(n_samples):
            cmd = _build_cli_cmd(provider=provider, model=model, prompt=prompt, system_prompt=system_prompt, session_id=session_id, history=history, images=images, think=think)
            if cmd is None:
                results.append({"response": f"No command builder for CLI provider: {provider}", "output": f"No command builder for CLI provider: {provider}", "messages": [], "usage": {}})
                continue
            if parser_cls:
                result = _run_subprocess_parsed(cmd, parser_cls(), verbose=verbose)
            else:
                result = _run_subprocess(cmd, verbose=verbose)
            # Extract post-run usage for kimi
            if provider in _USAGE_EXTRACTORS and result.get("session_id"):
                extra_usage = _USAGE_EXTRACTORS[provider](result["session_id"])
                if extra_usage:
                    result["usage"] = {**result.get("usage", {}), **extra_usage}
            results.append(result)
        return {"responses": results, "output": "\n---\n".join(r.get("output", "") for r in results), "messages": [], "usage": {}}

    cmd = _build_cli_cmd(provider=provider, model=model, prompt=prompt, system_prompt=system_prompt, session_id=session_id, history=history, images=images, think=think)
    if cmd is None:
        return {"response": f"No command builder for CLI provider: {provider}", "output": f"No command builder for CLI provider: {provider}", "messages": [], "usage": {}}

    if parser_cls:
        result = _run_subprocess_parsed(cmd, parser_cls(), verbose=verbose)
    else:
        result = _run_subprocess(cmd, verbose=verbose)

    # Extract post-run usage for kimi
    if provider in _USAGE_EXTRACTORS and result.get("session_id"):
        extra_usage = _USAGE_EXTRACTORS[provider](result["session_id"])
        if extra_usage:
            result["usage"] = {**result.get("usage", {}), **extra_usage}

    result["session_id"] = result.get("session_id") or session_id
    return result
