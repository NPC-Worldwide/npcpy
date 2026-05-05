from jinja2 import Environment, FileSystemLoader, Undefined
import hashlib
import json
import os
import PIL
import random
import select
import subprocess
import sys
import copy
import itertools
import logging
import threading
import time
import uuid
from typing import List, Dict, Any, Optional, Union, NamedTuple, Callable

try:
    import pty as _pty
except ImportError:
    _pty = None

logger = logging.getLogger("npcpy.llm_funcs")

# ---------------------------------------------------------------------------
# CLI provider routing
# ---------------------------------------------------------------------------

_CLI_PROVIDERS = {
    'opencode', 'claude_code', 'claude', 'codex', 'amp',
    'gemini', 'aider', 'kimi', 'kimi_code', 'kilo',
}

# CLIs that activate a full-screen TUI when stdout is a TTY — use TERM=dumb + pipe.
_TUI_PROVIDERS = {'opencode', 'kilo'}

# CLIs whose output is JSON/JSONL that must be captured and parsed.
# kimi uses --print --output-format text → plain text captured via pipe.
_PIPE_PROVIDERS = {'opencode', 'kilo', 'claude_code', 'claude', 'codex', 'kimi', 'kimi_code'}


def _build_history_summary(history, max_turns=8):
    """Format recent conversation turns as plain text for CLIs without session APIs."""
    if not history:
        return ""
    turns = [m for m in history if isinstance(m, dict) and m.get('role') in ('user', 'assistant')]
    turns = turns[-(max_turns * 2):]
    lines = ["=== Previous conversation ==="]
    i = 0
    while i < len(turns) - 1:
        u_msg = turns[i]
        a_msg = turns[i + 1]
        if u_msg.get('role') == 'user' and a_msg.get('role') == 'assistant':
            u = str(u_msg.get('content', ''))[:300]
            a = str(a_msg.get('content', ''))[:400]
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
            lines.append("---")
        i += 2
    lines.append("=== End of context ===")
    return "\n".join(lines)


def _fetch_opencode_session_id():
    """Return the most recent opencode session ID from `opencode session list`."""
    try:
        r = subprocess.run(
            ['opencode', 'session', 'list'],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].startswith('ses_'):
                return parts[0]
    except Exception:
        pass
    return None


def _fetch_kilo_session_id():
    """Return the most recent kilo session ID from `kilo session list`."""
    try:
        r = subprocess.run(
            ['kilo', 'session', 'list'],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].startswith('ses_'):
                return parts[0]
    except Exception:
        pass
    return None


def _fetch_kimi_session_id():
    """Return the most recently modified Kimi session ID for the current directory."""
    try:
        cwd = os.getcwd()
        # Kimi stores sessions under an MD5 hash of the cwd path.
        cwd_hash = hashlib.md5(cwd.encode()).hexdigest()
        sessions_base = os.path.expanduser('~/.kimi/sessions')
        sessions_dir = os.path.join(sessions_base, cwd_hash)
        if not os.path.isdir(sessions_dir):
            # Some versions use only the first 8 hex chars.
            sessions_dir = os.path.join(sessions_base, cwd_hash[:8])
        if not os.path.isdir(sessions_dir):
            return None
        entries = [
            (os.path.getmtime(os.path.join(sessions_dir, e)), e)
            for e in os.listdir(sessions_dir)
            if os.path.isdir(os.path.join(sessions_dir, e))
        ]
        return max(entries)[1] if entries else None
    except Exception:
        return None


def _wrap_with_system(prompt, system_prompt, session_id):
    """Prepend system_prompt to the user prompt on the first turn (no session_id).

    Used for CLI providers that don't have a native --system-prompt flag.
    Once a session is established, the CLI retains context, so the system
    prompt isn't re-sent.
    """
    if session_id or not system_prompt:
        return prompt
    return f"<system>\n{system_prompt}\n</system>\n\n{prompt}"


def _build_cli_cmd(provider, model, prompt, system_prompt, session_id, history,
                   images=None, think=None):
    """Build the subprocess command list for the given CLI provider."""
    if provider in ('claude_code', 'claude'):
        sid = session_id or str(uuid.uuid4())
        cmd = ['claude', '-p', prompt, '--output-format', 'stream-json', '--verbose',
               '--session-id', sid]
        if model:
            cmd += ['--model', model]
        if system_prompt:
            cmd += ['--system-prompt', system_prompt]
        if images:
            for img in images:
                cmd += ['--image', img]
        if think:
            cmd += ['--think-mode', 'auto']
        return cmd
    if provider == 'opencode':
        full_prompt = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ['opencode', 'run', full_prompt, '--format', 'json']
        if model:
            cmd += ['-m', model]
        if session_id:
            cmd += ['-s', session_id]
        return cmd
    if provider == 'codex':
        full_prompt = _wrap_with_system(prompt, system_prompt, session_id)
        if session_id == 'last':
            cmd = ['codex', 'exec', 'resume', '--last', '--json', full_prompt]
        elif session_id:
            cmd = ['codex', 'exec', 'resume', session_id, '--json', full_prompt]
        else:
            cmd = ['codex', 'exec', '--json', full_prompt]
        if model:
            cmd += ['--model', model]
        return cmd
    if provider in ('kimi', 'kimi_code'):
        full_prompt = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ['kimi', '--print', '--output-format', 'text', '-p', full_prompt]
        if model:
            cmd += ['-m', model]
        if session_id:
            cmd += ['-S', session_id]
        if think:
            cmd += ['--thinking']
        return cmd
    if provider == 'kilo':
        full_prompt = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ['kilo', 'run', full_prompt, '--format', 'json']
        if model:
            cmd += ['-m', model]
        if session_id:
            cmd += ['-s', session_id]
        return cmd
    if provider == 'gemini':
        full_prompt = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ['gemini', '-p', full_prompt]
        if model:
            cmd += ['-m', model]
        if images:
            for img in images:
                cmd += ['--image', img]
        return cmd
    if provider == 'amp':
        full_prompt = _wrap_with_system(prompt, system_prompt, session_id)
        cmd = ['amp', 'run', full_prompt]
        if model:
            cmd += ['--model', model]
        return cmd
    if provider == 'aider':
        summary = _build_history_summary(history) if history and session_id else ''
        base_prompt = f"{summary}\n\n{prompt}" if summary else prompt
        full_prompt = _wrap_with_system(base_prompt, system_prompt, session_id)
        cmd = ['aider', '--message', full_prompt, '--no-pretty']
        if model:
            cmd += ['--model', model]
        return cmd
    return None


# ---------------------------------------------------------------------------
# Incremental output parsers — single source of truth for both live streaming
# (inside _run_cli_provider) and offline parsing (via _parse_*_output wrappers).
# Each subclass consumes a line via feed() and returns the text delta to display
# live (or '' to suppress); finalize() returns the accumulated triplet.
# ---------------------------------------------------------------------------


class _StreamParser:
    """Base class for per-CLI incremental output parsers."""

    def feed(self, line: str) -> str:
        return ''

    def finalize(self):
        # (text, usage_dict, parser_extracted_session_id_or_None)
        return '', {}, None


class _ClaudeStreamParser(_StreamParser):
    """Parses ``claude --output-format stream-json`` JSONL.

    ``assistant`` events carry CUMULATIVE text — feed() returns the delta
    against the previous assistant text. ``result`` event carries final usage.
    """

    def __init__(self):
        self._last_text = ''
        self._final_text = ''
        self._usage = {}

    def feed(self, line):
        s = line.strip()
        if not s:
            return ''
        try:
            ev = json.loads(s)
        except Exception:
            return ''
        t = ev.get('type', '')
        if t == 'assistant':
            new_text = ''
            for c in ev.get('message', {}).get('content', []):
                if c.get('type') == 'text':
                    new_text += c.get('text', '')
            if not new_text:
                return ''
            if new_text.startswith(self._last_text):
                delta = new_text[len(self._last_text):]
            else:
                delta = new_text
            self._last_text = new_text
            self._final_text = new_text
            return delta
        if t == 'result':
            u = ev.get('usage', {})
            self._usage = {
                'input_tokens': u.get('input_tokens', 0),
                'output_tokens': u.get('output_tokens', 0),
                'cache_read_input_tokens': u.get('cache_read_input_tokens', 0),
            }
            cost = ev.get('total_cost_usd')
            if cost is not None:
                self._usage['cost_usd'] = cost
        return ''

    def finalize(self):
        return self._final_text, self._usage, None


class _OpencodeStreamParser(_StreamParser):
    """Parses opencode/kilo ``--format json`` JSONL.

    type:'text' → part.text (text chunk); type:'step_finish' → tokens + cost.
    """

    def __init__(self):
        self._parts = []
        self._usage = {
            'input_tokens': 0, 'output_tokens': 0,
            'cache_read': 0, 'cache_write': 0, 'cost_usd': 0.0,
        }

    def feed(self, line):
        s = line.strip()
        if not s:
            return ''
        try:
            ev = json.loads(s)
        except Exception:
            return ''
        t = ev.get('type', '')
        if t == 'text':
            chunk = ev.get('part', {}).get('text', '')
            if chunk:
                self._parts.append(chunk)
                return chunk
        elif t == 'step_finish':
            part = ev.get('part', {})
            tokens = part.get('tokens', {})
            self._usage['input_tokens'] += tokens.get('input', 0)
            self._usage['output_tokens'] += tokens.get('output', 0)
            cache = tokens.get('cache', {})
            self._usage['cache_read'] += cache.get('read', 0)
            self._usage['cache_write'] += cache.get('write', 0)
            self._usage['cost_usd'] += part.get('cost', 0.0)
        return ''

    def finalize(self):
        return ''.join(self._parts), self._usage, None


class _CodexStreamParser(_StreamParser):
    """Parses codex ``--json`` JSONL.

    thread.started → thread/session id; turn.completed → usage;
    item.completed (type=agent_message) → full message text.

    Codex emits full messages (not chunks) at item.completed. The live delta
    is suppressed (returns '') to preserve the legacy non-streaming UX.
    Final text is exposed via finalize().
    """

    def __init__(self):
        self._parts = []
        self._usage = {}
        self._thread_id = None

    def feed(self, line):
        s = line.strip()
        if not s:
            return ''
        try:
            ev = json.loads(s)
        except Exception:
            return ''
        t = ev.get('type', '')
        if t == 'thread.started':
            self._thread_id = ev.get('thread_id')
        elif t == 'turn.completed':
            u = ev.get('usage', {})
            self._usage = {
                'input_tokens': u.get('input_tokens', 0),
                'cached_input_tokens': u.get('cached_input_tokens', 0),
                'output_tokens': u.get('output_tokens', 0),
                'reasoning_output_tokens': u.get('reasoning_output_tokens', 0),
            }
        elif t == 'item.completed':
            item = ev.get('item', {})
            if item.get('type') == 'agent_message':
                text = item.get('text', '')
                if text:
                    self._parts.append(text)
        return ''

    def finalize(self):
        return ''.join(self._parts), self._usage, self._thread_id


class _KimiStreamParser(_StreamParser):
    """``kimi --print --output-format text`` emits plain text line-by-line.

    Empty lines are accumulated (so finalize() preserves internal blank lines)
    but not emitted live (matches legacy behavior). Usage isn't in the stream
    — see _PROVIDER_CAPS[kimi].usage_enricher.
    """

    def __init__(self):
        self._parts = []

    def feed(self, line):
        self._parts.append(line)
        if not line.strip():
            return ''
        return line

    def finalize(self):
        return ''.join(self._parts).strip(), {}, None


# Backward-compat one-shot wrappers — feed an entire raw buffer at once.
# Used by callers that already have the full output and want the (text, usage)
# tuple without instantiating the parser themselves.

def _parse_claude_output(raw):
    """Parse claude stream-json (with single-blob JSON fallback)."""
    p = _ClaudeStreamParser()
    for line in raw.splitlines():
        p.feed(line)
    text, usage, _ = p.finalize()
    if text or usage:
        return text, usage
    # Fallback: old --output-format json (single JSON blob).
    try:
        data = json.loads(raw.strip())
        text = data.get('result', '')
        u = data.get('usage', {})
        usage = {
            'input_tokens': u.get('input_tokens', 0),
            'output_tokens': u.get('output_tokens', 0),
            'cache_read_input_tokens': u.get('cache_read_input_tokens', 0),
        }
        cost = data.get('total_cost_usd')
        if cost is not None:
            usage['cost_usd'] = cost
        return text, usage
    except Exception:
        return raw, {}


def _parse_opencode_output(raw):
    """Parse opencode/kilo --format json JSONL stream."""
    p = _OpencodeStreamParser()
    for line in raw.splitlines():
        p.feed(line)
    text, usage, _ = p.finalize()
    return text, usage


def _parse_codex_output(raw):
    """Parse codex --json JSONL stream. Returns (text, usage, thread_id)."""
    p = _CodexStreamParser()
    for line in raw.splitlines():
        p.feed(line)
    return p.finalize()


def _parse_kimi_usage(session_id):
    """Sum StatusUpdate token_usage fields from ~/.kimi/sessions/<hash>/<id>/wire.jsonl."""
    if not session_id:
        return {}
    try:
        cwd = os.getcwd()
        cwd_hash = hashlib.md5(cwd.encode()).hexdigest()
        base = os.path.expanduser('~/.kimi/sessions')
        wire = os.path.join(base, cwd_hash, session_id, 'wire.jsonl')
        if not os.path.exists(wire):
            wire = os.path.join(base, cwd_hash[:8], session_id, 'wire.jsonl')
        if not os.path.exists(wire):
            return {}
        totals = {'input_other': 0, 'output': 0, 'input_cache_read': 0, 'input_cache_creation': 0}
        with open(wire) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    msg = ev.get('message', {})
                    if msg.get('type') == 'StatusUpdate':
                        tu = msg.get('payload', {}).get('token_usage', {})
                        for k in totals:
                            totals[k] += tu.get(k, 0)
                except Exception:
                    continue
        return totals
    except Exception:
        return {}


def _parse_kilo_usage(session_id):
    """Query kilo's SQLite database for token usage of the current session."""
    if not session_id:
        return {}
    try:
        r = subprocess.run(['kilo', 'db', 'path'], capture_output=True, text=True, timeout=5)
        db_path = r.stdout.strip()
        if not db_path or not os.path.exists(db_path):
            return {}
        import sqlite3
        con = sqlite3.connect(db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cached_tokens) "
                "FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return {
                    'input_tokens': row[0] or 0,
                    'output_tokens': row[1] or 0,
                    'cached_tokens': row[2] or 0,
                }
        finally:
            con.close()
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Per-provider capability table — single source of truth for which parser
# handles each CLI's output and how its session_id is resolved post-run.
# Adding a new CLI = one entry here + a parser class above + a branch in
# _build_cli_cmd. The table replaces what used to be 4 parallel if/elif
# chains scattered through _run_cli_provider.
# ---------------------------------------------------------------------------


class _ProviderCaps(NamedTuple):
    """Capabilities used by ``_run_cli_provider`` to finalize a CLI run.

    parser_cls
        Incremental output parser (text + usage + optional session id).
    session_resolver
        ``(parser_result, pre_assigned_sid) -> Optional[str]``. Combines
        parser-extracted ids (e.g. codex thread_id) with externally-fetched
        ids (filesystem mtime, ``opencode session list``, …) and the
        pre-assigned UUID claude was called with.
    usage_enricher
        Optional ``(session_id) -> dict``. Called after session resolution
        when usage isn't in the output stream itself (kimi: usage lives in
        ``~/.kimi/sessions/<hash>/<id>/wire.jsonl``).
    """
    parser_cls: type
    session_resolver: Callable
    usage_enricher: Optional[Callable] = None


def _resolve_claude_session(parser_result, pre_assigned_sid):
    return pre_assigned_sid


def _resolve_codex_session(parser_result, pre_assigned_sid):
    _, _, parsed_sid = parser_result
    # Fall back to the 'last' sentinel codex understands when the
    # thread.started event wasn't observed (truncated output, etc.).
    return parsed_sid or 'last'


def _resolve_opencode_session(parser_result, pre_assigned_sid):
    return _fetch_opencode_session_id()


def _resolve_kilo_session(parser_result, pre_assigned_sid):
    return _fetch_kilo_session_id()


def _resolve_kimi_session(parser_result, pre_assigned_sid):
    return _fetch_kimi_session_id()


_PROVIDER_CAPS = {
    'claude_code': _ProviderCaps(_ClaudeStreamParser, _resolve_claude_session),
    'claude':      _ProviderCaps(_ClaudeStreamParser, _resolve_claude_session),
    'opencode':    _ProviderCaps(_OpencodeStreamParser, _resolve_opencode_session),
    'kilo':        _ProviderCaps(_OpencodeStreamParser, _resolve_kilo_session),
    'codex':       _ProviderCaps(_CodexStreamParser, _resolve_codex_session),
    'kimi':        _ProviderCaps(_KimiStreamParser, _resolve_kimi_session, _parse_kimi_usage),
    'kimi_code':   _ProviderCaps(_KimiStreamParser, _resolve_kimi_session, _parse_kimi_usage),
}


def _run_cli_provider(provider, model, prompt, system_prompt,
                      session_id=None, npc_name='', history=None,
                      images=None, think=None, stream=True):
    """Run a CLI agent tool as a subprocess.

    Returns a dict compatible with get_llm_response output:
    {response, response_already_streamed, messages, tool_calls, usage, session_id}.
    Returns None if provider is not in _CLI_PROVIDERS.

    stream
    ------
    - True (default): tokens are written to stdout as they arrive; the spinner
      shows during pauses on the line below the stream cursor; the returned
      `response_already_streamed` is True so callers skip re-rendering.
    - False: output is buffered in full and returned in `response`; stdout is
      untouched during execution; spinner shows inline (as before any chunk);
      `response_already_streamed` is False so callers render markdown normally.

    Subprocess modes
    ----------------
    - _TUI_PROVIDERS (opencode, kilo): TERM=dumb + pipe — suppresses full-screen TUI.
    - _PIPE_PROVIDERS (claude, codex, kilo, opencode, kimi): pipe — output captured.
    - Others (gemini, amp, aider): PTY — real TTY prevents pipe buffering.
    """
    if provider not in _CLI_PROVIDERS:
        return None

    cmd = _build_cli_cmd(provider, model, prompt, system_prompt, session_id, history,
                         images=images, think=think)
    if cmd is None:
        return None

    # Extract pre-assigned session ID for claude (embedded in the command by _build_cli_cmd).
    pre_assigned_sid = None
    if provider in ('claude_code', 'claude') and '--session-id' in cmd:
        pre_assigned_sid = cmd[cmd.index('--session-id') + 1]

    use_pipe = provider in _PIPE_PROVIDERS
    env = dict(os.environ)
    if provider in _TUI_PROVIDERS:
        env['TERM'] = 'dumb'

    output_lines = []
    interrupted = False
    proc = None
    _streamed = threading.Event()  # set once first chunk is written to stdout

    # Per-provider incremental parser — single source of truth for both live
    # streaming (feed returns deltas) and final text/usage extraction
    # (finalize returns the accumulated triplet). PTY-mode providers without
    # a CAPS entry leave parser=None and behave like before (no parsing).
    _caps = _PROVIDER_CAPS.get(provider)
    parser = _caps.parser_cls() if _caps else None

    try:
        if use_pipe:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, text=True, bufsize=1,
            )
            _last_output_t = [time.monotonic()]
            _spinner_done = threading.Event()
            _spinner_visible = [False]
            # Serializes spinner thread vs chunk writers so escape sequences
            # don't interleave (would garble the terminal cursor state).
            _io_lock = threading.Lock()

            def _spin(_prov=provider):
                """Pre-streaming spinner only (matches litellm SpinnerContext).

                Once the first chunk lands the spinner stops drawing — we then
                rely on the post-stream render_markdown for final formatting.
                """
                _frames = itertools.cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
                _t0 = time.monotonic()
                while not _spinner_done.wait(timeout=0.1):
                    # Once streaming has started, no spinner — same as the
                    # litellm path which has nothing visible during streaming.
                    if _streamed.is_set():
                        continue
                    if time.monotonic() - _last_output_t[0] > 0.5:
                        elapsed = int(time.monotonic() - _t0)
                        text = f'{next(_frames)} {_prov} thinking... ({elapsed}s)'
                        with _io_lock:
                            sys.stderr.write(f'\r{text}  ')
                            sys.stderr.flush()
                            _spinner_visible[0] = True
                # Cleanup
                with _io_lock:
                    if _spinner_visible[0]:
                        sys.stderr.write('\r' + ' ' * 60 + '\r')
                        sys.stderr.flush()
                    _spinner_visible[0] = False

            spin_t = threading.Thread(target=_spin, daemon=True)
            spin_t.start()

            # Blank-line normalizer for the live stream.
            # Buffers by line so blank lines adjacent to box-drawing art or
            # 4-space-indented code are suppressed before reaching the scrollback.
            # Same logic as render_markdown._flush_prose (covers the raw stream
            # that cursor-restore can't erase for long responses).
            import re as _sn_re
            _sn_box = _sn_re.compile(r'[─-╿]')  # U+2500–U+257F box drawing block
            def _sn_compact(ln: str) -> bool:
                return bool(_sn_box.search(ln)) or ln.startswith('    ')
            _sn = {'buf': '', 'last': '', 'blanks': 0}

            def _sn_write(text: str) -> None:
                for ch in text:
                    if ch == '\n':
                        line = _sn['buf'].rstrip('\r')
                        _sn['buf'] = ''
                        if not line.strip():
                            _sn['blanks'] += 1
                        else:
                            if _sn['blanks'] > 0:
                                if not (_sn_compact(_sn['last']) or _sn_compact(line)):
                                    sys.stdout.write('\n')
                                _sn['blanks'] = 0
                            sys.stdout.write(line + '\n')
                            _sn['last'] = line
                    else:
                        _sn['buf'] += ch
                sys.stdout.flush()

            def _sn_flush() -> None:
                if _sn['buf']:
                    if _sn['blanks'] > 0 and not _sn_compact(_sn['last']):
                        sys.stdout.write('\n')
                    sys.stdout.write(_sn['buf'])
                    sys.stdout.flush()

            def _write_streamed(chunk):
                """First call: clear pre-stream spinner and save cursor for the
                end-of-stream markdown re-render. Subsequent calls write through
                the blank-line normalizer."""
                with _io_lock:
                    if not _streamed.is_set():
                        if _spinner_visible[0]:
                            sys.stderr.write('\r' + ' ' * 60 + '\r')
                            sys.stderr.flush()
                            _spinner_visible[0] = False
                        # \x1b7 = DEC save cursor — used for the post-stream
                        # \x1b8\x1b[J + render_markdown re-render.
                        sys.stdout.write('\x1b7')
                        _streamed.set()
                    _sn_write(chunk)

            try:
                for line in proc.stdout:
                    _last_output_t[0] = time.monotonic()
                    output_lines.append(line)
                    if parser is None:
                        continue
                    delta = parser.feed(line)
                    if stream and delta:
                        _write_streamed(delta)
                proc.wait()
            except KeyboardInterrupt:
                interrupted = True
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                print()
            finally:
                _spinner_done.set()
                spin_t.join(timeout=1)
                _sn_flush()

        else:
            # PTY mode — gives the CLI a real TTY so it flushes output immediately.
            master_fd = slave_fd = None
            if _pty is not None:
                try:
                    master_fd, slave_fd = _pty.openpty()
                except Exception:
                    pass

            if master_fd is not None:
                proc = subprocess.Popen(
                    cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                    env=env, close_fds=True,
                )
                os.close(slave_fd)
                _pty_last_data = time.monotonic()
                _pty_t0 = time.monotonic()
                _pty_spinner = itertools.cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
                _pty_spinner_last = 0.0
                _pty_spinner_visible = False
                try:
                    while True:
                        try:
                            ready, _, _ = select.select([master_fd], [], [], 0.1)
                        except (ValueError, OSError):
                            break
                        _now = time.monotonic()
                        if ready:
                            try:
                                data = os.read(master_fd, 4096)
                            except OSError:
                                break
                            if not data:
                                break
                            if _pty_spinner_visible:
                                sys.stderr.write('\r' + ' ' * 50 + '\r')
                                sys.stderr.flush()
                                _pty_spinner_visible = False
                            chunk = data.decode('utf-8', errors='replace')
                            output_lines.append(chunk)
                            sys.stdout.write(chunk)
                            sys.stdout.flush()
                            _pty_last_data = _now
                        else:
                            if _now - _pty_last_data > 0.5 and _now - _pty_spinner_last >= 0.1:
                                elapsed = int(_now - _pty_t0)
                                sys.stderr.write(
                                    f'\r{next(_pty_spinner)} {provider} thinking... ({elapsed}s)  '
                                )
                                sys.stderr.flush()
                                _pty_spinner_last = _now
                                _pty_spinner_visible = True
                            if proc.poll() is not None:
                                # Drain any remaining buffered output.
                                try:
                                    while True:
                                        r2, _, _ = select.select([master_fd], [], [], 0.05)
                                        if not r2:
                                            break
                                        data = os.read(master_fd, 4096)
                                        if not data:
                                            break
                                        chunk = data.decode('utf-8', errors='replace')
                                        output_lines.append(chunk)
                                        sys.stdout.write(chunk)
                                        sys.stdout.flush()
                                except OSError:
                                    pass
                                break
                except KeyboardInterrupt:
                    interrupted = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    print()
                finally:
                    if _pty_spinner_visible:
                        sys.stderr.write('\r' + ' ' * 50 + '\r')
                        sys.stderr.flush()
                    try:
                        os.close(master_fd)
                    except OSError:
                        pass
            else:
                # PTY unavailable — fall back to plain pipe.
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, text=True,
                )
                try:
                    for line in proc.stdout:
                        output_lines.append(line)
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    proc.wait()
                except KeyboardInterrupt:
                    interrupted = True
                    proc.terminate()
                    proc.wait()
                    print()

    except Exception as exc:
        logger.error("CLI provider error (%s): %s", provider, exc)
        return {'response': '', 'messages': [], 'tool_calls': [], 'usage': {}, 'session_id': None}

    if interrupted:
        return {'response': '', 'messages': [], 'tool_calls': [], 'usage': {}, 'session_id': None}

    if proc is not None and proc.returncode:
        sys.stderr.write(f'\033[31m✗ {provider} exited with code {proc.returncode}\033[0m\n')
        sys.stderr.flush()

    # --- Parse output and usage ---
    # The parser was fed every line during the streaming loop; finalize()
    # returns the same triplet that the offline _parse_*_output wrappers
    # would return for the buffered case. PTY-mode providers without a
    # CAPS entry get empty values (legacy behavior).
    if parser is not None:
        response_text, usage, parsed_session_id = parser.finalize()
    else:
        response_text, usage, parsed_session_id = '', {}, None
    parser_result = (response_text, usage, parsed_session_id)

    # Final display:
    # - stream=True + streamed chunks: API parity — restore cursor to where
    #   streaming started, clear that area, re-render the full text as markdown
    # - stream=True + nothing was streamed (parser-only providers, or empty
    #   output): plain print as a safety net
    # - stream=False: leave display to the caller (process_result will
    #   render_markdown via its own path)
    if stream and not interrupted and not (proc and proc.returncode):
        if _streamed.is_set() and response_text:
            sys.stdout.write('\x1b8\x1b[J')  # restore cursor + erase to EOS
            sys.stdout.flush()
            render_markdown(response_text)
        elif response_text and not _streamed.is_set():
            render_markdown(response_text)

    # --- Determine returned session ID + enrich usage if needed ---
    if _caps:
        new_session_id = _caps.session_resolver(parser_result, pre_assigned_sid)
        # Some providers' usage isn't in the output stream (kimi: it lives in
        # wire.jsonl under the resolved session dir).
        if _caps.usage_enricher and new_session_id:
            usage = _caps.usage_enricher(new_session_id)
    else:
        new_session_id = None

    return {
        'response': response_text,
        # True when stream=True: we streamed live AND ran the post-stream
        # markdown render (parity with print_and_process_stream_with_markdown).
        # False when stream=False: caller renders the response itself.
        'response_already_streamed': bool(stream),
        'messages': [],
        'tool_calls': [],
        'usage': usage,
        'session_id': new_session_id,
    }

# ---------------------------------------------------------------------------

from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    render_markdown,
    lookup_provider,
    request_user_input, 
    get_system_message
)

from npcpy.gen.response import get_litellm_response
from npcpy.gen.image_gen import generate_image
from npcpy.gen.video_gen import generate_video_diffusers, generate_video_veo3

from datetime import datetime 

def gen_image(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    height: int = 1024,
    width: int = 1024,
    n_images: int=1, 
    input_images: List[Union[str, bytes, PIL.Image.Image]] = None,
    save = False, 
    filename = '',
    api_key: str = None,
):
    """This function generates an image using the specified provider and model.
    Args:
        prompt (str): The prompt for generating the image.
    Keyword Args:
        model (str): The model to use for generating the image.
        provider (str): The provider to use for generating the image.
        filename (str): The filename to save the image to.
        npc (Any): The NPC object.
        api_key (str): The API key for the image generation service.
    Returns:
        List[PIL.Image.Image]: A list of generated PIL Image objects.
    """
    if model is not None and provider is not None:
        pass
    elif model is not None and provider is None:
        provider = lookup_provider(model)
    elif npc is not None:
        if npc.provider is not None:
            provider = npc.provider
        if npc.model is not None:
            model = npc.model
        if npc.api_url is not None:
            api_url = npc.api_url

    images = generate_image(
        prompt=prompt,
        model=model,
        provider=provider,
        height=height,
        width=width, 
        attachments=input_images,
        n_images=n_images, 
        api_key=api_key,
        
    )
    if save:
        if len(filename) == 0 :
            todays_date = datetime.now().strftime("%Y-%m-%d")
            filename = 'vixynt_gen'
        for i, image in enumerate(images):
            
            image.save(filename+'_'+str(i)+'.png')
    return images

def gen_video(
    prompt,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    device: str = "cpu",
    output_path="",
    num_inference_steps=10,
    num_frames=25,
    height=256,
    width=256,
    negative_prompt="",
    messages: list = None,
):
    """
    Function Description:
        This function generates a video using either Diffusers or Veo 3 via Gemini API.
    Args:
        prompt (str): The prompt for generating the video.
    Keyword Args:
        model (str): The model to use for generating the video.
        provider (str): The provider to use for generating the video (gemini for Veo 3).
        device (str): The device to run the model on ('cpu' or 'cuda').
        negative_prompt (str): What to avoid in the video (Veo 3 only).
    Returns:
        dict: Response with output path and messages.
    """
    
    if provider == "gemini":
        
        try:
            output_path = generate_video_veo3(
                prompt=prompt,
                model=model,
                negative_prompt=negative_prompt,
                output_path=output_path,
            )
            return {
                "output": f"High-fidelity video with synchronized audio generated at {output_path}",
                "messages": messages
            }
        except Exception as e:
            print(f"Veo 3 generation failed: {e}")
            print("Falling back to diffusers...")
            provider = "diffusers"
    
    if provider == "diffusers" or provider is None:
      
        output_path = generate_video_diffusers(
            prompt,
            model,
            npc=npc,
            device=device,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        return {
            "output": f"Video generated at {output_path}",
            "messages": messages
        }
    
    return {
        "output": f"Unsupported provider: {provider}",
        "messages": messages
    }

def resolve_model_provider(
    npc=None, team=None, model=None, provider=None,
    api_url=None, api_key=None, images=None, attachments=None,
):
    """Resolve model, provider, api_url, and api_key from npc/team/explicit overrides."""
    m, p, a_url, a_key = model, provider, api_url, api_key
    if m is not None and p is not None:
        pass
    elif p is None and m is not None:
        p = lookup_provider(m)
    elif npc is not None:
        if npc.provider is not None:
            p = npc.provider
        if npc.model is not None:
            m = npc.model
    elif team is not None:
        if team.model is not None:
            m = team.model
        if team.provider is not None:
            p = team.provider
    else:
        p = "ollama"
        if images is not None or attachments is not None:
            m = "llava:7b"
        else:
            m = "llama3.2"
    # Always resolve api_url and api_key from npc/team if not explicitly provided
    if a_url is None and npc is not None and getattr(npc, 'api_url', None) is not None:
        a_url = npc.api_url
    if a_url is None and team is not None and getattr(team, 'api_url', None) is not None:
        a_url = team.api_url
    if a_key is None and npc is not None and getattr(npc, 'api_key', None) is not None:
        a_key = npc.api_key
    if a_key is None and team is not None and getattr(team, 'api_key', None) is not None:
        a_key = team.api_key
    return m, p, a_url, a_key


def get_llm_response(
    prompt: str,
    model: str = None,
    provider: str = None,
    images: List[str] = None,
    npc: Any = None,
    team: Any = None,
    messages: List[Dict[str, str]] = None,
    api_url: str = None,
    api_key: str = None,
    context=None,
    stream: bool = False,
    attachments: List[str] = None,
    include_usage: bool = False,
    n_samples: int = 1,
    matrix: Optional[Dict[str, List[Any]]] = None,
    session_id: Optional[str] = None,
    **kwargs,
):
    """Generate a response using the specified provider and model."""
    logger.debug(
        f"[get_llm_response] {len(messages) if messages else 0} messages, "
        f"prompt_len={len(prompt) if prompt else 0}, "
        f"context={'yes' if context else 'no'}"
    )

    base_model, base_provider, base_api_url, base_api_key = resolve_model_provider(
        npc=npc, team=team, model=model, provider=provider,
        api_url=api_url, api_key=api_key, images=images, attachments=attachments,
    )
    if api_key is None:
        api_key = base_api_key

    # CLI provider routing — intercept before litellm for all paths.
    if base_provider in _CLI_PROVIDERS:
        # Multi-sample / matrix: handled in the loop below for both CLI and
        # litellm. The single-call path runs here.
        if not (matrix is not None and len(matrix) > 0) and not (n_samples and n_samples > 1):
            system_prompt = (
                get_system_message(npc, team) if npc is not None else "You are a helpful assistant."
            )
            full_prompt = prompt or ""
            if context:
                full_prompt = f"{full_prompt}\n\n\nUser Provided Context: {context}" if full_prompt else f"User Provided Context: {context}"
            result = _run_cli_provider(
                base_provider, base_model, full_prompt, system_prompt,
                session_id=session_id, npc_name=getattr(npc, 'name', '') if npc else '',
                history=messages,
                images=images or attachments,
                think=kwargs.get('think'),
                stream=stream,
            )
            if result is not None:
                return result

    use_matrix = matrix is not None and len(matrix) > 0
    multi_sample = n_samples and n_samples > 1

    if not use_matrix and not multi_sample:
        # Simple single-call path
        run_model, run_provider, run_api_url, run_api_key = resolve_model_provider(
            npc=npc, team=team, model=base_model, provider=base_provider,
            api_url=api_url, api_key=api_key, images=images, attachments=attachments,
        )
        tool_capable = bool(kwargs.get("tools"))
        system_message = get_system_message(npc, team, tool_capable=tool_capable) if npc is not None else "You are a helpful assistant."

        # Build messages
        run_messages = copy.deepcopy(messages) if messages else []
        if not run_messages:
            run_messages = [{"role": "system", "content": system_message}]

        # Build full text from prompt and/or context
        full_text = ""
        if prompt and context:
            full_text = f"{prompt}\n\n\nUser Provided Context: {context}"
        elif prompt:
            full_text = prompt
        elif context:
            full_text = f"User Provided Context: {context}"

        if full_text:
            if run_messages[-1]["role"] == "user" and isinstance(run_messages[-1]["content"], str):
                run_messages[-1]["content"] += "\n" + full_text
            else:
                run_messages.append({"role": "user", "content": full_text})

        return get_litellm_response(
            full_text or None,
            messages=run_messages,
            model=run_model,
            provider=run_provider,
            api_url=run_api_url,
            api_key=api_key,
            images=images,
            attachments=attachments,
            stream=stream,
            include_usage=include_usage,
            **kwargs,
        )

    # Matrix / multi-sample path
    combos = []
    if use_matrix:
        keys = list(matrix.keys())
        values = []
        for key in keys:
            val = matrix[key]
            if isinstance(val, (list, tuple, set)):
                values.append(list(val))
            else:
                values.append([val])
        for combo_values in itertools.product(*values):
            combos.append(dict(zip(keys, combo_values)))
    else:
        combos.append({})

    runs = []
    for combo in combos:
        run_npc = combo.get("npc", npc)
        run_team = combo.get("team", team)
        run_context = combo.get("context", context)
        extra_kwargs = dict(kwargs)
        for k, v in combo.items():
            if k not in {"model", "provider", "npc", "context", "team"}:
                extra_kwargs[k] = v

        run_model, run_provider, run_api_url, run_api_key = resolve_model_provider(
            npc=run_npc, team=run_team,
            model=combo.get("model", base_model),
            provider=combo.get("provider", base_provider),
            api_url=api_url, api_key=api_key, images=images, attachments=attachments,
        )

        tool_capable = bool(extra_kwargs.get("tools"))
        system_message = get_system_message(run_npc, run_team, tool_capable=tool_capable) if run_npc is not None else "You are a helpful assistant."

        run_messages = copy.deepcopy(messages) if messages else []
        if not run_messages:
            run_messages = [{"role": "system", "content": system_message}]

        full_text = ""
        if prompt and run_context:
            full_text = f"{prompt}\n\n\nUser Provided Context: {run_context}"
        elif prompt:
            full_text = prompt
        elif run_context:
            full_text = f"User Provided Context: {run_context}"

        if full_text:
            if run_messages[-1]["role"] == "user" and isinstance(run_messages[-1]["content"], str):
                run_messages[-1]["content"] += "\n" + full_text
            else:
                run_messages.append({"role": "user", "content": full_text})

        for sample_idx in range(max(1, n_samples or 1)):
            if run_provider in _CLI_PROVIDERS:
                # Each sample runs in a fresh subprocess and (for sampling) a
                # fresh CLI session — otherwise the CLI treats the calls as a
                # multi-turn conversation rather than independent samples.
                _ctx = run_context
                _full = full_text
                if _ctx and prompt:
                    _full = f"{prompt}\n\n\nUser Provided Context: {_ctx}"
                resp = _run_cli_provider(
                    run_provider, run_model, _full or "", system_message,
                    session_id=None,
                    npc_name=getattr(run_npc, 'name', '') if run_npc else '',
                    history=messages,
                    images=images or attachments,
                    think=extra_kwargs.get('think'),
                    stream=False,  # samples never live-stream — output would interleave
                )
            else:
                resp = get_litellm_response(
                    full_text or None,
                    messages=run_messages,
                    model=run_model,
                    provider=run_provider,
                    api_url=run_api_url,
                    api_key=api_key,
                    images=images,
                    attachments=attachments,
                    stream=stream,
                    include_usage=include_usage,
                    **extra_kwargs,
                )
            runs.append({
                "response": resp.get("response") if isinstance(resp, dict) else resp,
                "raw": resp,
                "combo": combo,
                "sample_index": sample_idx,
            })

    aggregated = {
        "response": runs[0]["response"] if runs else None,
        "runs": runs,
    }
    if runs and isinstance(runs[0]["raw"], dict) and "messages" in runs[0]["raw"]:
        aggregated["messages"] = runs[0]["raw"].get("messages")
    return aggregated

def execute_llm_command(
    command: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: str = None,
    api_key: str = None,
    npc: Optional[Any] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    stream=False,
    context=None,
) -> str:
    """This function executes an LLM command.
    Args:
        command (str): The command to execute.

    Keyword Args:
        model (Optional[str]): The model to use for executing the command.
        provider (Optional[str]): The provider to use for executing the command.
        npc (Optional[Any]): The NPC object.
        messages (Optional[List[Dict[str, str]]): The list of messages.
    Returns:
        str: The result of the LLM command.
    """
    if messages is None:
        messages = []
    max_attempts = 5
    attempt = 0
    subcommands = []

   
    context = ""
    while attempt < max_attempts:
        prompt = f"""
        A user submitted this query: {command}.
        You need to generate a bash command that will accomplish the user's intent.
        Respond ONLY with the bash command that should be executed. 
        Do not include markdown formatting
        """
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=messages,
            npc=npc,
            context=context,
        )

        bash_command = response.get("response", {})
 
        print(f"LLM suggests the following bash command: {bash_command}")
        subcommands.append(bash_command)

        try:
            print(f"Running command: {bash_command}")
            result = subprocess.run(
                bash_command, shell=True, text=True, capture_output=True, check=True
            )
            print(f"Command executed with output: {result.stdout}")

            prompt = f"""
                Here was the output of the result for the {command} inquiry
                which ran this bash command {bash_command}:

                {result.stdout}

                Provide a simple response to the user that explains to them
                what you did and how it accomplishes what they asked for.
                """

            messages.append({"role": "user", "content": prompt})
           
            response = get_llm_response(
                prompt,
                model=model,
                provider=provider,
                api_url=api_url,
                api_key=api_key,
                npc=npc,
                messages=messages,
                context=context,
                stream =stream
            )

            return response
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error:")
            print(e.stderr)

            error_prompt = f"""
            The command '{bash_command}' failed with the following error:
            {e.stderr}
            Please suggest a fix or an alternative command.
            Respond with a JSON object containing the key "bash_command" with the suggested command.
            Do not include any additional markdown formatting.

            """

            fix_suggestion = get_llm_response(
                error_prompt,
                model=model,
                provider=provider,
                npc=npc,
                api_url=api_url,
                api_key=api_key,
                format="json",
                messages=messages,
                context=context,
            )

            fix_suggestion_response = fix_suggestion.get("response", {})

            try:
                if isinstance(fix_suggestion_response, str):
                    fix_suggestion_response = json.loads(fix_suggestion_response)

                if (
                    isinstance(fix_suggestion_response, dict)
                    and "bash_command" in fix_suggestion_response
                ):
                    print(
                        f"LLM suggests fix: {fix_suggestion_response['bash_command']}"
                    )
                    command = fix_suggestion_response["bash_command"]
                else:
                    raise ValueError(
                        "Invalid response format from LLM for fix suggestion"
                    )
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing LLM fix suggestion: {e}")

        attempt += 1

    return {
        "messages": messages,
        "output": "Max attempts reached. Unable to execute the command successfully.",
    }

def handle_request_input(
    context: str,
    model: str ,
    provider: str 
):
    """
    Analyze text and decide what to request from the user
    """
    json_format = """Return a JSON object with:
    {
        "input_needed": boolean,
        "request_reason": string explaining why input is needed,
        "request_prompt": string to show user if input needed
    }

    Do not include any additional markdown formatting or leading ```json tags. Your response
    must be a valid JSON object."""

    prompt = f"""
    Analyze the text:
    {context}
    and determine what additional input is needed.
    {json_format}
    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        messages=[],
        format="json",
    )

    result = response.get("response", {})
    if isinstance(result, str):
        result = json.loads(result)

    user_input = request_user_input(
        {"reason": result["request_reason"], "prompt": result["request_prompt"]},
    )
    return user_input

def _get_jinxes(npc, team):
    """Get available jinxes from npc (already filtered by jinxes_spec)."""
    jinxes = {}
    if npc and hasattr(npc, 'jinxes_dict'):
        jinxes.update(npc.jinxes_dict)
    return jinxes

def _jinxes_to_tools(jinxes):
    """Convert jinxes to OpenAI-style tool definitions."""
    return [jinx.to_tool_def() for jinx in jinxes.values()]

def _execute_jinx(jinx, inputs, npc, team, messages, extra_globals):
    """Execute a jinx and return output."""
    try:
        jinja_env = None
        if npc and hasattr(npc, 'jinja_env'):
            jinja_env = npc.jinja_env
        elif team and hasattr(team, 'forenpc') and team.forenpc:
            jinja_env = getattr(team.forenpc, 'jinja_env', None)

        full_inputs = {}
        for inp in getattr(jinx, 'inputs', []):
            if isinstance(inp, dict):
                key = list(inp.keys())[0]
                default_val = inp[key]
                if isinstance(default_val, str) and default_val.startswith("~"):
                    default_val = os.path.expanduser(default_val)
                full_inputs[key] = default_val
            else:
                full_inputs[inp] = ""
        full_inputs.update(inputs or {})

        result = jinx.execute(
            input_values=full_inputs,
            npc=npc,
            messages=messages,
            extra_globals=extra_globals,
            jinja_env=jinja_env
        )
        if result is None:
            return "Executed with no output."
        if not isinstance(result, dict):
            return str(result)
        return result.get("output", str(result))
    except Exception as e:
        return f"Error: {e}"

def _build_jinx_schema(jinx_obj):
    """Build a self-describing schema string from a jinx's own metadata.

    Only shows params the model must fill: the primary param (first with
    empty default) plus any param with a non-empty default.  Optional
    params with empty defaults are skipped — _execute_jinx fills those
    from the jinx's own defaults so the model can't hallucinate values.
    """
    desc = getattr(jinx_obj, 'description', '') or ''
    inp_list = getattr(jinx_obj, 'inputs', [])
    params = []
    has_primary = False
    for inp in inp_list:
        if isinstance(inp, str):
            params.append(f'"{inp}": "..."')
            has_primary = True
        elif isinstance(inp, dict):
            for k, v in inp.items():
                if isinstance(v, dict) and 'description' in v:
                    params.append(f'"{k}": "...({v["description"]})"')
                    has_primary = True
                elif v:
                    params.append(f'"{k}": "...(default: {v})"')
                else:
                    if not has_primary:
                        params.append(f'"{k}": "..."')
                        has_primary = True
    schema_str = '{' + ', '.join(params) + '}'
    return desc, schema_str


def handle_jinx_call(
    command,
    jinx_name,
    jinxes,
    model=None,
    provider=None,
    api_url=None,
    api_key=None,
    npc=None,
    team=None,
    messages=None,
    stream=False,
    context=None,
    extra_globals=None,
    previous_output=None,
    n_attempts=3,
    attempt=0,
):
    """Resolve inputs for a single jinx and execute it. Retries on failure."""
    if messages is None:
        messages = []

    jinx = jinxes.get(jinx_name)
    if not jinx:
        if attempt < n_attempts:
            available = ", ".join(jinxes.keys())
            return check_llm_command(
                f"""In the previous attempt, the jinx name was: {jinx_name}.
That jinx was not available. Only select from: {available}.
Original request: {command}""",
                model=model, 
                provider=provider, 
                api_url=api_url, 
                api_key=api_key,
                npc=npc, 
                team=team,
                messages=messages, 
                stream=stream,
                context=context, 
                extra_globals=extra_globals,
            )
        return {"output": f"Jinx '{jinx_name}' not found after {n_attempts} attempts.", "messages": messages}

    print(f"[JINX] {jinx.jinx_name}", flush=True)

    # Build example format from jinx inputs
    example_format = {}
    for inp in jinx.inputs:
        if isinstance(inp, str):
            example_format[inp] = "..."
        elif isinstance(inp, dict):
            key = list(inp.keys())[0]
            example_format[key] = "..."
    json_format_str = json.dumps(example_format, indent=4)

    # Show full jinx definition so model understands what it's filling
    prompt = f"""The user wants to use the jinx '{jinx_name}' with the following request:
'{command}'

Here were the previous 5 messages in the conversation: {messages[-5:]}

Here is the jinx file:
```
{jinx.to_dict()}
```

Please determine the required inputs for the jinx as a JSON object.
They must be exactly as they are named in the jinx.
If the jinx requires a file path, you must include an absolute path with extension.
If the jinx requires code, generate it exactly according to the instructions.

Return only the JSON object without any markdown formatting.
The format of the JSON object is:
{json_format_str}"""

    if npc and hasattr(npc, "shared_context"):
        if npc.shared_context.get("dataframes"):
            context_info = "\nAvailable dataframes:\n"
            for df_name in npc.shared_context["dataframes"].keys():
                context_info += f"- {df_name}\n"
            prompt += f"\nContextual info: {context_info}"

    response = get_llm_response(
        prompt,
        format="json",
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        messages=messages[-10:],
        npc=npc,
        team=team,
        context=context,
    )

    response_text = response.get("response", "{}")
    if isinstance(response_text, str):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        try:
            input_values = json.loads(response_text)
        except json.JSONDecodeError as e:
            if attempt < n_attempts:
                return handle_jinx_call(
                    command, 
                    jinx_name, 
                    jinxes,
                    model=model, 
                    provider=provider, 
                    api_url=api_url, 
                    api_key=api_key,
                    npc=npc, 
                    team=team, 
                    messages=messages, 
                    stream=stream,
                    context=f"Previous attempt failed to parse JSON: {e}. Raw: {response_text}",
                    extra_globals=extra_globals, previous_output=previous_output,
                    n_attempts=n_attempts, 
                    attempt=attempt + 1,
                )
            return {"output": f"Error extracting inputs for jinx '{jinx_name}'", "messages": messages}
    elif isinstance(response_text, dict):
        input_values = response_text
    else:
        input_values = {}

    # Validate required inputs
    missing = []
    for inp in jinx.inputs:
        if not isinstance(inp, dict):
            if inp not in input_values or input_values[inp] == "":
                missing.append(inp)
    if missing and attempt < n_attempts:
        return handle_jinx_call(
            command + f". Previous attempt missing inputs: {missing}. Values were: {input_values}.",
            jinx_name, 
            jinxes,
            model=model, 
            provider=provider, 
            api_url=api_url, 
            api_key=api_key,
            npc=npc, team=team, 
            messages=messages, 
            stream=stream,
            context=context, 
            extra_globals=extra_globals, 
            previous_output=previous_output,
            n_attempts=n_attempts, 
            attempt=attempt + 1,
        )
    elif missing:
        return {"output": f"Missing inputs for jinx '{jinx_name}': {missing}", "messages": messages}

    print(f"[INPUTS] {json.dumps({k: str(v)[:200] for k, v in input_values.items()})}", flush=True)

    # Inject previous step output for inter-step data flow
    if previous_output is not None:
        input_values['previous_output'] = previous_output

    # Execute
    output = _execute_jinx(jinx, input_values, npc, team, messages, extra_globals)

    print(f"[RESULT] {str(output)[:300]}", flush=True)

    # Retry on error
    if isinstance(output, str) and output.startswith("Error:") and attempt < n_attempts:
        return handle_jinx_call(
            command, 
            jinx_name, 
            jinxes,
            model=model, 
            provider=provider, 
            api_url=api_url, 
            api_key=api_key,
            npc=npc, team=team, 
            messages=messages, 
            stream=stream,
            context=f"Jinx failed: {output}. Previous inputs: {input_values}",
            extra_globals=extra_globals, 
            previous_output=previous_output,
            n_attempts=n_attempts, 
            attempt=attempt + 1,
        )

    # Collect any stream events produced during jinx execution (e.g. delegation sub-events)
    stream_events = []
    if npc and hasattr(npc, 'shared_context') and npc.shared_context.get('sub_events'):
        stream_events = npc.shared_context.pop('sub_events')

    return {
        "output": output,
        "messages": messages,
        "jinx_calls": [{
            "name": jinx_name,
            "arguments": input_values,
            "result": str(output) if output else "",
        }],
        "stream_events": stream_events,
    }

def handle_action_choice(command: str,
                         action_data: dict, 
                         jinxes: list, 
                         model : str = None,
                         provider: str = None, 
                         api_url:str = None, 
                         api_key: str = None , 
                         npc: Any = None,
                         team: Any = None,
                         messages: list = None,
                         images: list = None,
                         stream: bool = False, 
                         extra_globals: dict = None,
                         last_jinx_output = None,
                         step_outputs: list = None,
                         context: str = '',
                         **kwargs,
                        ):
    action_name = action_data.get("action", "answer")
    if action_name == "invoke_jinx" or 'jinx_name' in action_data:
        jname = action_data.get("jinx_name", "")
        step_context = context or ""
        if step_outputs:
            step_context += f"\nContext from previous steps: {json.dumps(step_outputs)}"

        result = handle_jinx_call(
            command, 
            jname, 
            jinxes,
            model=model, 
            provider=provider, 
            api_url=api_url, 
            api_key=api_key,
            npc=npc, 
            team=team, 
            messages=messages, 
            stream=stream,
            context=step_context, 
            extra_globals=extra_globals,
            previous_output=last_jinx_output,
        )
        output = result.get("output", "")
        current_messages = result.get("messages", messages)
        jinx_calls = result.get("jinx_calls", [])
        stream_events = result.get("stream_events", [])

        # Display
        if output and str(output).strip():
            content = str(output).replace('\\n', '\n').replace('\\t', '\t')
            render_markdown(f"\n⚡ {jname}:")
            lines = content.split('\n')
            if len(lines) > 50:
                render_markdown('\n'.join(lines[:25]))
                print(f"\n... ({len(lines) - 50} lines hidden) ...\n")
                render_markdown('\n'.join(lines[-25:]))
            else:
                render_markdown(content)
    elif action_name != 'answer':
        output = 'INVALID_ACTION'
        jinx_calls = []
        return {'output':output, 'messages':messages, 'jinx_calls': jinx_calls}
    else:
        response = get_llm_response(
            f"""The user asked: {command}

              Provide a direct answer. Do not reference tools or jinxes.""",
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=[],
            npc=npc,
            team=team,
            images=images,
            stream=stream,
            context=context,
            **kwargs,
        )
        output = response.get("response", "")
        current_messages = response.get("messages", messages)
        jinx_calls = []

    return {'output':output, 'messages':current_messages, 'jinx_calls': jinx_calls, 'stream_events': stream_events if 'stream_events' in locals() else []}


def check_llm_command(
    command: str,
    model: str = None,
    provider: str = None,
    api_url: str = None,
    api_key: str = None,
    npc: Any = None,
    team: Any = None,
    messages: List[Dict[str, str]] = None,
    images: list = None,
    stream=False,
    context=None,
    actions: Dict[str, Dict] = None,
    extra_globals=None,
    max_iterations: int = 5,
    jinxes: Dict = None,
    tool_capable: bool = None,
    **kwargs,
):
    """Plan and execute: decide whether to answer directly or use jinxes, then do it.

    When stream=True, returns a generator yielding (event_type, data) tuples.
    When stream=False, returns a dict.
    """
    if messages is None:
        messages = []

    # CLI provider routing — skip the planner/action loop entirely.
    # CLI tools (claude_code, opencode, codex, kimi_code, kilo, ...) have
    # their own internal toolset (Read/Write/Bash/etc.); the npcsh action
    # loop with format="json" planner doesn't apply — they don't honor it
    # and would return raw text that breaks the planner. Route directly to
    # get_llm_response, which itself dispatches to _run_cli_provider.
    if provider and provider in _CLI_PROVIDERS:
        response = get_llm_response(
            command,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=messages[-10:] if messages else [],
            npc=npc,
            team=team,
            images=images,
            stream=stream,
            context=context,
            **kwargs,
        )
        messages.append({"role": "user", "content": command})
        out = response.get("response", "") if isinstance(response, dict) else ""
        if out and isinstance(out, str):
            messages.append({"role": "assistant", "content": out})
        return {
            "messages": messages,
            "output": out,
            "usage": response.get("usage", {}) if isinstance(response, dict) else {},
        }

    if jinxes is None:
        jinxes = _get_jinxes(npc, team)

    # No jinxes — just answer directly
    if not jinxes:
        print('no jinxes detected')

        response = get_llm_response(
            command,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=messages[-10:],
            npc=npc,
            team=team,
            images=images,
            stream=False,
            context=context,
            **kwargs,
        )
        messages.append({"role": "user", "content": command})
        out = response.get("response", "")
        if out and isinstance(out, str):
            messages.append({"role": "assistant", "content": out})
        return {"messages": messages, "output": out, "usage": response.get("usage", {})}


    prompt = f"""

    
          A user submitted this request: {command}
    
      
          Determine the nature of the user's request:
      
          1. Should a jinx be invoked to fulfill the request? A jinx is a jinja-template execution script.
      
          2. Is it a general question that requires an informative answer or a highly specific question that
              requires information on the web?
      
                        
              Use jinxes when it is obvious that the answer needs to be as up-to-date as possible. For example,
                  a question about where mount everest is does not necessarily need to be answered by a jinx call or an agent pass.
          
              If a user asks to explain the plot of the aeneid, this can be answered without a jinx call or agent pass.
              
              If a user were to ask for the current weather in tokyo or the current price of bitcoin or who the mayor of a city is,
                  then a jinx call is appropriate.
          
              If the user wants you to read a file, it must use a jinx to read the file.
          
              If the user asks you to edit a file, you must use a jinx to edit the file.
          
              If the user asks you to take a screenshot, you must to use a jinx to take the screenshot if available
              
              If a user asks you to search or to take a screenshot or to open a program or to write a program most likely it is
              appropriate to use a jinx. 


              remember, in your output, return only the action sequence. do not include and leading ```json or other markdown tags.
              
              """

    if messages:
        prompt += f"\nRecent conversation: {messages[-5:]}"

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        npc=npc,
        team=team,
        format="json",
        messages=[],
        context=context,
        **kwargs,
    )

    actions = response.get("response", {})

    # Some models (notably ollama backends that don't fully honor format="json")
    # return the JSON as a raw string instead of a parsed object. Try to parse;
    # if that fails, fall back to a single "answer" action with the raw text so
    # we never end up iterating characters of a string.
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except (json.JSONDecodeError, ValueError):
            actions = {"action": "answer", "explanation": actions}

    # Display plan
    print(actions)

    if isinstance(actions, dict):
        actions = [actions]
    elif not isinstance(actions, list):
        # Unknown shape — treat as a fallback answer action.
        actions = [{"action": "answer", "explanation": str(actions)}]
    else:
        # List form — coerce any non-dict items into answer actions.
        actions = [
            a if isinstance(a, dict) else {"action": "answer", "explanation": str(a)}
            for a in actions
        ]

    def _execute():
        step_outputs = []
        all_jinx_calls = []
        current_messages = messages.copy()
        last_jinx_output = None

        for i, action_data in enumerate(actions):
            render_markdown(f"- {action_data}")
            action_name = action_data.get("action", "")
            is_jinx = action_name == "invoke_jinx" or "jinx_name" in action_data
            jname = action_data.get("jinx_name", "")

            if stream and is_jinx and jname:
                yield ('tool_start', {'name': jname, 'args': action_data.get('inputs', {})})

            action_result = handle_action_choice(
                         command,
                         action_data,
                         jinxes,
                         model = model,
                         provider = provider,
                         api_url = api_url,
                         api_key = api_key,
                         npc = npc,
                         team = team,
                         messages = current_messages,
                         stream = stream,
                         extra_globals = extra_globals,
                         last_jinx_output = last_jinx_output,
                         step_outputs = step_outputs,
                         context = context,
                         **kwargs,
            )
            current_messages = action_result.get('messages', [])
            output = action_result.get('output', [])
            all_jinx_calls.extend(action_result.get('jinx_calls', []))
            if output == 'INVALID_ACTION':
                for evt in check_llm_command(
                    f"""In the previous attempt, the correct action name was not provided and a jinx could not be deciphered. only select from available jinxes.
                Original request: {command}""",
                    model=model, provider=provider, api_url=api_url, api_key=api_key,
                    npc=npc, team=team, messages=messages, stream=stream,
                    context=context, extra_globals=extra_globals, **kwargs,
                ):
                    yield evt
                return

            if stream:
                # Yield any sub-events from delegation before the tool_result
                for evt in action_result.get('stream_events', []):
                    yield evt
                if is_jinx and jname:
                    yield ('tool_result', {'name': jname, 'result': output, 'args': action_data.get('inputs', {})})
                else:
                    yield ('content', output)

            step_outputs.append(output)
            last_jinx_output = output

        final_output = step_outputs[0] if len(step_outputs) == 1 else "\n".join(str(o) for o in step_outputs)
        yield ('result', {
            "messages": current_messages,
            "output": final_output,
            "usage": response.get("usage", {}),
            "jinx_calls": all_jinx_calls,
        })

    if stream:
        return _execute()

    # Non-streaming: consume the generator and return the result dict
    result = None
    for event_type, data in _execute():
        if event_type == 'result':
            result = data
    return result or {"messages": messages, "output": "", "jinx_calls": []}


def identify_groups(
    facts: List[str],
    model,
    provider,
    npc =  None,
    context: str = None,
    **kwargs
) -> List[str]:
    """Identify natural groups from a list of facts"""

        
    prompt = """What are the main groups these facts could be organized into?
    Express these groups in plain, natural language.

    For example, given:
        - User enjoys programming in Python
        - User works on machine learning projects
        - User likes to play piano
        - User practices meditation daily

    You might identify groups like:
        - Programming
        - Machine Learning
        - Musical Interests
        - Daily Practices

    Return a JSON object with the following structure:
        `{
            "groups": ["list of group names"]
        }`

    Return only the JSON object. Do not include any additional markdown formatting or
    leading json characters.
    """

    response = get_llm_response(
        prompt + f"\n\nFacts: {json.dumps(facts)}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        
        **kwargs
    )
    return response["response"]["groups"]

def get_related_concepts_multi(node_name: str, 
                               node_type: str, 
                               all_concept_names, 
                               model: str = None,
                               provider: str = None,
                               npc=None,
                               context : str = None, 
                               **kwargs):
    """Links any node (fact or concept) to ALL relevant concepts in the entire ontology."""
    json_format = 'Respond with JSON: {"related_concepts": ["Concept A", "Concept B", ...]}'
    prompt = f"""
    Which of the following concepts from the entire ontology relate to the given {node_type}?
    Select all that apply, from the most specific to the most abstract.

    {node_type.capitalize()}: "{node_name}"

    Available Concepts:
    {json.dumps(all_concept_names, indent=2)}

    {json_format}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context, 
                                **kwargs)
    return response["response"].get("related_concepts", [])

def assign_groups_to_fact(
    fact: str,
    groups: List[str],
    model = None,
    provider = None,
    npc = None, 
    context: str = None,
    **kwargs
) -> Dict[str, List[str]]:
    """Assign facts to the identified groups"""
    json_format = """Return a JSON object with the following structure:
        {
            "groups": ["list of group names"]
        }

    Do not include any additional markdown formatting or leading json characters."""

    prompt = f"""Given this fact, assign it to any relevant groups.

    A fact can belong to multiple groups if it fits.

    Here is the fact: {fact}

    Here are the groups: {groups}

    {json_format}
    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        **kwargs
    )
    return response["response"]

def generate_group_candidates(
    items: List[str],
    item_type: str,
    model: str = None,
    provider: str =None,
    npc = None,
    context: str = None,
    n_passes: int = 3,
    subset_size: int = 10, 
    **kwargs
) -> List[str]:
    """Generate candidate groups for items (facts or groups) based on core semantic meaning."""
    all_candidates = []
    
    for pass_num in range(n_passes):
        if len(items) > subset_size:
            item_subset = random.sample(items, min(subset_size, len(items)))
        else:
            item_subset = items
        
      
        prompt = f"""From the following {item_type}, identify specific and relevant conceptual groups.
        Think about the core subject or entity being discussed.
        
        GUIDELINES FOR GROUP NAMES:
        1.  **Prioritize Specificity:** Names should be precise and directly reflect the content.
        2.  **Favor Nouns and Noun Phrases:** Use descriptive nouns or noun phrases.
        3.  **AVOID:**
            *   Gerunds (words ending in -ing when used as nouns, like "Understanding", "Analyzing", "Processing"). If a gerund is unavoidable, try to make it a specific action (e.g., "User Authentication Module" is better than "Authenticating Users").
            *   Adverbs or descriptive adjectives that don't form a core part of the subject's identity (e.g., "Quickly calculating", "Effectively managing").
            *   Overly generic terms (e.g., "Concepts", "Processes", "Dynamics", "Mechanics", "Analysis", "Understanding", "Interactions", "Relationships", "Properties", "Structures", "Systems", "Frameworks", "Predictions", "Outcomes", "Effects", "Considerations", "Methods", "Techniques", "Data", "Theoretical", "Physical", "Spatial", "Temporal").
        4.  **Direct Naming:** If an item is a specific entity or action, it can be a group name itself (e.g., "Earth", "Lamb Shank Braising", "World War I").
        
        EXAMPLE:
        Input {item_type.capitalize()}: ["Self-intersection shocks drive accretion disk formation.", "Gravity stretches star into stream.", "Energy dissipation in shocks influences capture fraction."]
        Desired Output Groups: ["Accretion Disk Formation (Self-Intersection Shocks)", "Stellar Tidal Stretching", "Energy Dissipation from Shocks"]
        
        ---
        
        Now, analyze the following {item_type}:
        {item_type.capitalize()}: {json.dumps(item_subset)}
        
        Return a JSON object:
        """ + '{"groups": ["list of specific, precise, and relevant group names"]}'
      
        
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            format="json",
            npc=npc,
            context=context,
            **kwargs
        )
        
        candidates = response["response"].get("groups", [])
        all_candidates.extend(candidates)

    return list(set(all_candidates))

def remove_idempotent_groups(
    group_candidates: List[str],
    model: str = None,
    provider: str =None,
    npc = None, 
    context : str = None,
    **kwargs: Any
) -> List[str]:
    """Remove groups that are essentially identical in meaning, favoring specificity and direct naming, and avoiding generic structures."""
    
    prompt = f"""Compare these group names. Identify and list ONLY the groups that are conceptually distinct and specific.
    
    GUIDELINES FOR SELECTING DISTINCT GROUPS:
    1.  **Prioritize Specificity and Direct Naming:** Favor precise nouns or noun phrases that directly name the subject.
    2.  **Prefer Concrete Entities/Actions:** If a name refers to a specific entity or action (e.g., "Earth", "Sun", "Water", "France", "User Authentication Module", "Lamb Shank Braising", "World War I"), keep it if it's distinct.
    3.  **Rephrase Gerunds:** If a name uses a gerund (e.g., "Understanding TDEs"), rephrase it to a noun or noun phrase (e.g., "Tidal Disruption Events").
    4.  **AVOID OVERLY GENERIC TERMS:** Do NOT use very broad or abstract terms that don't add specific meaning. Examples to avoid: "Concepts", "Processes", "Dynamics", "Mechanics", "Analysis", "Understanding", "Interactions", "Relationships", "Properties", "Structures", "Systems", "Frameworks", "Predictions", "Outcomes", "Effects", "Considerations", "Methods", "Techniques", "Data", "Theoretical", "Physical", "Spatial", "Temporal". If a group name seems overly generic or abstract, it should likely be removed or refined.
    5.  **Similarity Check:** If two groups are very similar, keep the one that is more descriptive or specific to the domain.

    EXAMPLE 1:
    Groups: ["Accretion Disk Formation", "Accretion Disk Dynamics", "Formation of Accretion Disks"]
    Distinct Groups: ["Accretion Disk Formation", "Accretion Disk Dynamics"] 

    EXAMPLE 2:
    Groups: ["Causes of Events", "Event Mechanisms", "Event Drivers"]
    Distinct Groups: ["Event Causation", "Event Mechanisms"] 

    EXAMPLE 3:
    Groups: ["Astrophysics Basics", "Fundamental Physics", "General Science Concepts"]
    Distinct Groups: ["Fundamental Physics"] 

    EXAMPLE 4:
    Groups: ["Earth", "The Planet Earth", "Sun", "Our Star"]
    Distinct Groups: ["Earth", "Sun"]
    
    EXAMPLE 5:
    Groups: ["User Authentication Module", "Authentication System", "Login Process"]
    Distinct Groups: ["User Authentication Module", "Login Process"]
    
    ---
    
    Now, analyze the following groups:
    Groups: {json.dumps(group_candidates)}
    
    Return JSON:
    """ + '{"distinct_groups": ["list of specific, precise, and distinct group names to keep"]}'
    
    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        **kwargs
    )
    
    return response["response"]["distinct_groups"]

def breathe(
    messages: List[Dict[str, str]],
    model: str = None,
    provider: str = None, 
    npc =  None,
    context: str = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Condense the conversation context into a small set of key extractions."""
    if not messages:
        return {"output": {}, "messages": []}

    if 'stream' in kwargs:
        kwargs['stream'] = False
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    prompt = f'''
    Read the following conversation:

    {conversation_text}

    ''' +'''

    Now identify the following items:

    1. The high level objective
    2. The most recent task
    3. The accomplishments thus far
    4. The failures thus far

    Return a JSON like so:

    {
        "high_level_objective": "the overall goal so far for the user", 
        "most_recent_task": "The currently ongoing task", 
        "accomplishments": ["accomplishment1", "accomplishment2"], 
        "failures": ["falures1", "failures2"], 
    }

    '''

    
    result = get_llm_response(prompt, 
                           model=model, 
                           provider=provider, 
                           npc=npc, 
                           context=context, 
                           format='json', 
                           **kwargs)

    res = result.get('response', {})
    if isinstance(res, str):
        raise Exception
    format_output = f"""Here is a summary of the previous session. 
    The high level objective was: {res.get('high_level_objective')} \n The accomplishments were: {res.get('accomplishments')}, 
    the failures were: {res.get('failures')} and the most recent task was: {res.get('most_recent_task')}   """
    return {'output': format_output,
            'summary': res,
            'messages': [
                         {
                           'content': format_output,
                           'role': 'assistant'}
                           ]
                          }
def abstract(groups, 
             model, 
             provider, 
             npc=None,
             context: str = None, 
             **kwargs):
    """
    Create more abstract terms from groups.
    """
    sample_groups = random.sample(groups, min(len(groups), max(3, len(groups) // 2)))
    
    groups_text_for_prompt = "\n".join([f'- "{g["name"]}"' for g in sample_groups])

    prompt = f"""
        Create more abstract categories from this list of groups.

        Groups:
        {groups_text_for_prompt}

        You will create higher-level concepts that interrelate between the given groups. 

        Create abstract categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. For example, do not try to combine "characters", "settings", and "physical reactions" into a
        compound group like "Characters, Setting, and Physical Reactions". This kind of grouping is not productive and only obfuscates true abstractions. 
        For example, a group that might encompass the three aforermentioned names might be "Literary Themes" or "Video Editing Functionis", depending on the context.
        Your aim is to abstract, not to just arbitrarily generate associations. 

        Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".
        Generate no more than 5 new concepts and no fewer than 2. 

        Respond with JSON:
        """ + '{"groups": [{"name": "abstract category name"}]}'

    response = get_llm_response(prompt,
                                model=model,
                                provider=provider,
                                format="json",
                                npc=npc,
                                context=context,
                                **kwargs)

    return response["response"].get("groups", [])

def get_facts(content_text, 
              model= None,
              provider = None,
              npc=None,
              context : str=None, 
              attempt_number=1,
              n_attempts=3,

              **kwargs):
    """Extract facts from content text"""
    
    prompt = f"""
    Extract facts from this text. A fact is a specific statement that can be sourced from the text.

    Example: if text says "the moon is the earth's only currently known satellite", extract:
    - "The moon is a satellite of earth" 
    - "The moon is the only current satellite of earth"
    - "There may have been other satellites of earth" (inferred from "only currently known")

        A fact is a piece of information that makes a statement about the world.
        A fact is typically a sentence that is true or false.
        Facts may be simple or complex. They can also be conflicting with each other, usually
        because there is some hidden context that is not mentioned in the text.
        In any case, it is simply your job to extract a list of facts that could pertain to
        an individual's personality.
        
        For example, if a message says:
            "since I am a doctor I am often trying to think up new ways to help people.
            Can you help me set up a new kind of software to help with that?"
        You might extract the following facts:
            - The individual is a doctor
            - They are helpful

        Another example:
            "I am a software engineer who loves to play video games. I am also a huge fan of the
            Star Wars franchise and I am a member of the 501st Legion."
        You might extract the following facts:
            - The individual is a software engineer
            - The individual loves to play video games
            - The individual is a huge fan of the Star Wars franchise
            - The individual is a member of the 501st Legion

        Another example:
            "The quantum tunneling effect allows particles to pass through barriers
            that classical physics says they shouldn't be able to cross. This has
            huge implications for semiconductor design."
        You might extract these facts:
            - Quantum tunneling enables particles to pass through barriers that are
              impassable according to classical physics
            - The behavior of quantum tunneling has significant implications for
              how semiconductors must be designed

        Another example:
            "People used to think the Earth was flat. Now we know it's spherical,
            though technically it's an oblate spheroid due to its rotation."
        You might extract these facts:
            - People historically believed the Earth was flat
            - It is now known that the Earth is an oblate spheroid
            - The Earth's oblate spheroid shape is caused by its rotation

        Another example:
            "My research on black holes suggests they emit radiation, but my professor
            says this conflicts with Einstein's work. After reading more papers, I
            learned this is actually Hawking radiation and doesn't conflict at all."
        You might extract the following facts:
            - Black holes emit radiation
            - The professor believes this radiation conflicts with Einstein's work
            - The radiation from black holes is called Hawking radiation
            - Hawking radiation does not conflict with Einstein's work

        Another example:
            "During the pandemic, many developers switched to remote work. I found
            that I'm actually more productive at home, though my company initially
            thought productivity would drop. Now they're keeping remote work permanent."
        You might extract the following facts:
            - The pandemic caused many developers to switch to remote work
            - The individual discovered higher productivity when working from home
            - The company predicted productivity would decrease with remote work
            - The company decided to make remote work a permanent option

        Thus, it is your mission to reliably extract lists of facts.

    Here is the text:
    Text: "{content_text}"

    Facts should never be more than one or two sentences, and they should not be overly complex or literal. They must be explicitly
    derived or inferred from the source text. Do not simply repeat the source text verbatim when stating the fact. 
    
    No two facts should share substantially similar claims. They should be conceptually distinct and pertain to distinct ideas, avoiding lengthy convoluted or compound facts .
    Respond with JSON:
    """ + '{"facts": [{"statement": "fact statement that builds on input text to state a specific claim that can be falsified through reference to the source material", "source_text": "text snippets related to the source text", "type": "explicit or inferred"}]}'
    
    response = get_llm_response(prompt, 
                                model=model,
                                provider=provider, 
                                npc=npc,
                                format="json", 
                                context=context,
                                **kwargs)

    if len(response.get("response", {}).get("facts", [])) == 0 and attempt_number < n_attempts:
        print(f"  Attempt {attempt_number} to extract facts yielded no results. Retrying...")
        return get_facts(content_text, 
                         model=model, 
                         provider=provider, 
                         npc=npc,
                         context=context,
                         attempt_number=attempt_number+1,
                         n_attempts=n_attempts,
                         **kwargs)
    
    return response["response"].get("facts", [])

        

def zoom_in(facts, 
            model= None,
            provider=None, 
            npc=None,
            context: str = None, 
            attempt_number: int = 1,
            n_attempts=3,            
            **kwargs):
    """Infer new implied facts from existing facts"""
    valid_facts = []
    for fact in facts:
        if isinstance(fact, dict) and 'statement' in fact:
            valid_facts.append(fact)
    if not valid_facts:
        return []     

    fact_lines = []
    for fact in valid_facts:
        fact_lines.append(f"- {fact['statement']}")
    facts_text = "\n".join(fact_lines)
    
    prompt = f"""
    Look at these facts and infer new implied facts:

    {facts_text}

    What other facts can be reasonably inferred from these?
    """ +"""
    Respond with JSON:
    {
        "implied_facts": [
            {
                "statement": "new implied fact",
                "inferred_from": ["which facts this comes from"]
            }
        ]
    }
    """
    
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                context=context,
                                npc=npc,
                                **kwargs)

    facts =  response.get("response", {}).get("implied_facts", [])
    if len(facts) == 0:
        return zoom_in(valid_facts, 
                       model=model, 
                       provider=provider, 
                       npc=npc,
                       context=context,
                       attempt_number=attempt_number+1,
                       n_tries=n_attempts,
                       **kwargs)
    return facts
def generate_groups(facts, 
                    model=None,
                    provider=None,
                    npc=None,
                    context: str =None, 
                    **kwargs):
    """Generate conceptual groups for facts"""
    
    facts_text = "\n".join([f"- {fact['statement']}" for fact in facts])
    
    prompt = f"""
    Generate conceptual groups for this group off facts:

    {facts_text}

    Create categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. 
    
    Your aim is to generalize commonly occurring ideas into groups, not to just arbitrarily generate associations. 
    Focus on the key commonly occurring items and expresions.     

    Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".
    Respond with JSON:
    """ + '{"groups": [{"name": "group name"}]}'

    response = get_llm_response(prompt,
                                model=model,
                                provider=provider,
                                format="json",
                                context=context,
                                npc=npc,
                                **kwargs)

    return response["response"].get("groups", [])

def remove_redundant_groups(groups, 
                            model=None,
                            provider=None,
                            npc=None,
                            context: str = None,
                            **kwargs):
    """Remove redundant groups"""
    
    groups_text = "\n".join([f"- {g['name']}" for g in groups])
    
    prompt = f"""
    Remove redundant groups from this list:

    {groups_text}

    Merge similar groups and keep only distinct concepts.
    Create abstract categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. For example, do not try to combine "characters", "settings", and "physical reactions" into a
    compound group like "Characters, Setting, and Physical Reactions". This kind of grouping is not productive and only obfuscates true abstractions. 
    For example, a group that might encompass the three aforermentioned names might be "Literary Themes" or "Video Editing Functionis", depending on the context.
    Your aim is to abstract, not to just arbitrarily generate associations. 

    Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".

    Respond with JSON:
    """ + '{"groups": [{"name": "final group name"}]}'

    response = get_llm_response(prompt,
                                model=model,
                                provider=provider,
                                npc=npc,
                                context=context,
                                **kwargs)

    return response["response"].get("groups", [])

def prune_fact_subset_llm(fact_subset, 
                          concept_name, 
                          model=None,
                          provider=None,
                          npc=None,
                          context : str = None,
                          **kwargs):
    """Identifies redundancies WITHIN a small, topically related subset of facts."""
    print(f"  Step Sleep-A: Pruning fact subset for concept '{concept_name}'...")
    

    prompt = f"""
    The following facts are all related to the concept "{concept_name}".
    Review ONLY this subset and identify groups of facts that are semantically identical.
    Return only the set of facts that are semantically distinct, and archive the rest.

    Fact Subset: {json.dumps(fact_subset, indent=2)}

    Return a json list of groups
    """ + '{"refined_facts": [fact1, fact2, fact3, ...]}'
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                npc=None,
                                format="json", 
                                context=context)
    return response['response'].get('refined_facts', [])

def consolidate_facts_llm(new_fact, 
                          existing_facts, 
                          model, 
                          provider, 
                          npc=None,
                          context: str =None,
                          **kwargs):
    """
    Uses an LLM to decide if a new fact is novel or redundant.
    """
    prompt = f"""
        Analyze the "New Fact" in the context of the "Existing Facts" list.
        Your task is to determine if the new fact provides genuinely new information or if it is essentially a repeat or minor rephrasing of information already present.

        New Fact:
        "{new_fact['statement']}"

        Existing Facts:
        {json.dumps([f['statement'] for f in existing_facts], indent=2)}

        Possible decisions:
        - 'novel': The fact introduces new, distinct information not covered by the existing facts.
        - 'redundant': The fact repeats information already present in the existing facts.

        Respond with a JSON object:
        """ + '{"decision": "novel or redundant", "reason": "A brief explanation for your decision."}'
    response = get_llm_response(prompt,
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)
    return response['response']

def get_related_facts_llm(new_fact_statement, 
                          existing_fact_statements, 
                          model = None, 
                          provider = None,
                          npc = None, 
                          attempt_number = 1,
                          n_attempts = 3,
                          context='', 
                          **kwargs):
    """Identifies which existing facts are causally or thematically related to a new fact."""
    prompt = f"""
    A new fact has been learned: "{new_fact_statement}"

    Which of the following existing facts are directly related to it (causally, sequentially, or thematically)?
    Select only the most direct and meaningful connections.

    Existing Facts:
    {json.dumps(existing_fact_statements, indent=2)}

    Respond with JSON:
    """ + '{"related_facts": ["statement of a related fact", ...]}'
    response = get_llm_response(prompt,
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)   
    if attempt_number <= n_attempts:
        if not response["response"].get("related_facts", []):
            print(f"  Attempt {attempt_number} to find related facts yielded no results. Retrying...")
            return get_related_facts_llm(new_fact_statement, 
                                           existing_fact_statements, 
                                           model=model, 
                                           provider=provider, 
                                           npc=npc,
                                           attempt_number=attempt_number+1,
                                           n_attempts=n_attempts,
                                           context=context,
                                           **kwargs)    

    return response["response"].get("related_facts", [])

def find_best_link_concept_llm(candidate_concept_name, 
                               existing_concept_names, 
                               model = None,
                               provider = None,
                               npc = None,
                               context: str = None,
                               **kwargs   ):
    """
    Finds the best existing concept to link a new candidate concept to.
    This prompt now uses neutral "association" language.
    """
    prompt = f"""
    Here is a new candidate concept: "{candidate_concept_name}"
    
    Which of the following existing concepts is it most closely related to? The relationship could be as a sub-category, a similar idea, or a related domain.

    Existing Concepts:
    {json.dumps(existing_concept_names, indent=2)}

    Respond with the single best-fit concept to link to from the list, or respond with "none" if it is a genuinely new root idea.
    """ + '{"best_link_concept": "The single best concept name OR none"}'
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)
    return response['response'].get('best_link_concept')

def asymptotic_freedom(parent_concept, 
                       supporting_facts, 
                       model=None, 
                       provider=None, 
                       npc = None,
                       context: str = None, 
                       **kwargs):
    """Given a concept and its facts, proposes an intermediate layer of sub-concepts."""
    print(f"  Step Sleep-B: Attempting to deepen concept '{parent_concept['name']}'...")
    fact_statements = []
    for f in supporting_facts:
        fact_statements.append(f['statement'])
        
    prompt = f"""
    The concept "{parent_concept['name']}" is supported by many diverse facts.
    Propose a layer of 2-4 more specific sub-concepts to better organize these facts.
    These new concepts will exist as nodes that link to "{parent_concept['name']}".

    Supporting Facts: {json.dumps(fact_statements, indent=2)}
    Respond with JSON:
    """ + '{"new_sub_concepts": ["sub_layer1", "sub_layer2"]}'
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider,
                                format="json", 
                                context=context, npc=npc,
                                **kwargs)
    return response['response'].get('new_sub_concepts', [])

def bootstrap(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    sample_params: Dict[str, Any] = None,
    sync_strategy: str = "consensus",
    context: str = None,
    n_samples: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """Bootstrap by sampling multiple agents from team or varying parameters"""
    
    if team and hasattr(team, 'npcs') and len(team.npcs) >= n_samples:
      
        sampled_npcs = list(team.npcs.values())[:n_samples]
        results = []
        
        for i, agent in enumerate(sampled_npcs):
            response = get_llm_response(
                f"Sample {i+1}: {prompt}\nContext: {context}",
                npc=agent,
                context=context,
                **kwargs
            )
            results.append({
                'agent': agent.name,
                'response': response.get("response", "")
            })
    else:
      
        if sample_params is None:
            sample_params = {"temperature": [0.3, 0.7, 1.0]}
        
        results = []
        for i in range(n_samples):
            temp = sample_params.get('temperature', [0.7])[i % len(sample_params.get('temperature', [0.7]))]
            response = get_llm_response(
                f"Sample {i+1}: {prompt}\nContext: {context}",
                model=model,
                provider=provider,
                npc=npc,
                temperature=temp,
                context=context,
                **kwargs
            )
            results.append({
                'variation': f'temp_{temp}',
                'response': response.get("response", "")
            })
    
  
    response_texts = [r['response'] for r in results]
    return synthesize(response_texts, sync_strategy, model, provider, npc or (team.forenpc if team else None), context)

def harmonize(
    prompt: str,
    items: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    harmony_rules: List[str] = None,
    context: str = None,
    agent_roles: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Harmonize using multiple specialized agents"""
    
    if team and hasattr(team, 'npcs'):
      
        available_agents = list(team.npcs.values())
        
        if agent_roles:
          
            selected_agents = []
            for role in agent_roles:
                matching_agent = next((a for a in available_agents if role.lower() in a.name.lower() or role.lower() in a.primary_directive.lower()), None)
                if matching_agent:
                    selected_agents.append(matching_agent)
            agents_to_use = selected_agents or available_agents[:len(items)]
        else:
          
            agents_to_use = available_agents[:min(len(items), len(available_agents))]
        
        harmonized_results = []
        for i, (item, agent) in enumerate(zip(items, agents_to_use)):
            harmony_prompt = f"""Harmonize this element: {item}
Task: {prompt}
Rules: {', '.join(harmony_rules or ['maintain_consistency'])}
Context: {context}
Your role in harmony: {agent.primary_directive}"""
            
            response = get_llm_response(
                harmony_prompt,
                npc=agent,
                context=context,
                **kwargs
            )
            harmonized_results.append({
                'agent': agent.name,
                'item': item,
                'harmonized': response.get("response", "")
            })
        
      
        coordinator = team.get_forenpc() if team else npc
        synthesis_prompt = f"""Synthesize these harmonized elements:
{chr(10).join([f"{r['agent']}: {r['harmonized']}" for r in harmonized_results])}
Create unified harmonious result."""
        
        return get_llm_response(synthesis_prompt, npc=coordinator, context=context, **kwargs)
    
    else:
      
        items_text = chr(10).join([f"{i+1}. {item}" for i, item in enumerate(items)])
        harmony_prompt = f"""Harmonize these items: {items_text}
Task: {prompt}
Rules: {', '.join(harmony_rules or ['maintain_consistency'])}
Context: {context}"""
        
        return get_llm_response(harmony_prompt, model=model, provider=provider, npc=npc, context=context, **kwargs)

def orchestrate(
    prompt: str,
    items: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    workflow: str = "sequential_coordination",
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Orchestrate using team.orchestrate method"""
    
    if team and hasattr(team, 'orchestrate'):
      
        orchestration_request = f"""Orchestrate workflow: {workflow}
Task: {prompt}
Items: {chr(10).join([f'- {item}' for item in items])}
Context: {context}"""
        
        return team.orchestrate(orchestration_request)
    
    else:
      
        items_text = chr(10).join([f"{i+1}. {item}" for i, item in enumerate(items)])
        orchestrate_prompt = f"""Orchestrate using {workflow}:
Task: {prompt}
Items: {items_text}
Context: {context}"""
        
        return get_llm_response(orchestrate_prompt, model=model, provider=provider, npc=npc, context=context, **kwargs)

def spread_and_sync(
    prompt: str,
    variations: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    sync_strategy: str = "consensus",
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Spread across agents/variations then sync with distribution analysis"""
    
    if team and hasattr(team, 'npcs') and len(team.npcs) >= len(variations):
      
        agents = list(team.npcs.values())[:len(variations)]
        results = []
        
        for variation, agent in zip(variations, agents):
            variation_prompt = f"""Analyze from {variation} perspective:
Task: {prompt}
Context: {context}
Apply your expertise with {variation} approach."""
            
            response = get_llm_response(variation_prompt, npc=agent, context=context, **kwargs)
            results.append({
                'agent': agent.name,
                'variation': variation,
                'response': response.get("response", "")
            })
    else:
      
        results = []
        agent = npc or (team.get_forenpc() if team else None)
        
        for variation in variations:
            variation_prompt = f"""Analyze from {variation} perspective:
Task: {prompt}
Context: {context}"""
            
            response = get_llm_response(variation_prompt, model=model, provider=provider, npc=agent, context=context, **kwargs)
            results.append({
                'variation': variation,
                'response': response.get("response", "")
            })
    
  
    response_texts = [r['response'] for r in results]
    return synthesize(response_texts, sync_strategy, model, provider, npc or (team.get_forenpc() if team else None), context)

def criticize(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Provide critical analysis and constructive criticism"""
    critique_prompt = f"""
    Provide a critical analysis and constructive criticism of the following:
    {prompt}
    
    Focus on identifying weaknesses, potential improvements, and alternative approaches.
    Be specific and provide actionable feedback.
    """
    
    return get_llm_response(
        critique_prompt,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        context=context,
        **kwargs
    )
def synthesize(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Synthesize information from multiple sources or perspectives"""
    
    responses = kwargs.get('responses', [prompt])
    sync_strategy = kwargs.get('sync_strategy', 'consensus')
    
    if len(responses) > 1:
        synthesis_prompt = f"""Synthesize these multiple perspectives:
        
        {chr(10).join([f'Response {i+1}: {r}' for i, r in enumerate(responses)])}
        
        Synthesis strategy: {sync_strategy}
        Context: {context}
        
        Create a coherent synthesis that incorporates key insights from all perspectives."""
    else:
        synthesis_prompt = f"""Refine and synthesize this content:
        
        {responses[0]}
        
        Context: {context}
        
        Create a clear, concise synthesis that captures the essence of the content."""
    
    return get_llm_response(
        synthesis_prompt,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        context=context,
        **kwargs
    )
