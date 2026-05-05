import hashlib
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
        if system_prompt:
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
        cmd = ["kimi", "--print", "--output-format", "text", "-p", full]
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


def run_cli_agent(provider, prompt, model=None, system_prompt=None, session_id=None, history=None, images=None, think=None, verbose=False):
    if not _is_cli_provider(provider):
        return {"response": f"Unknown CLI provider: {provider}", "output": f"Unknown CLI provider: {provider}", "messages": [], "usage": {}}
    if not session_id:
        resolver = _SESSION_RESOLVERS.get(provider)
        if resolver:
            session_id = resolver()
    cmd = _build_cli_cmd(provider=provider, model=model, prompt=prompt, system_prompt=system_prompt, session_id=session_id, history=history, images=images, think=think)
    if cmd is None:
        return {"response": f"No command builder for CLI provider: {provider}", "output": f"No command builder for CLI provider: {provider}", "messages": [], "usage": {}}
    result = _run_subprocess(cmd, verbose=verbose)
    result["session_id"] = session_id
    return result
