"""Conversion entrypoints between npcpy's canonical formats and adjacent ecosystems.

Canonical source of truth:
  - .jinx files (YAML + embedded Python, Jinja-rendered at load time)
  - .npc files (YAML + Jinja `Jinx()`/`NPC()` macros)

Generated / interoperable formats:
  - skills/<name>/SKILL.md + <step>.py per step — Anthropic-style skill folder.
    Jinja is compiled out during conversion; a consumer reading a skill never
    encounters `{{ }}`. Code snippets become real files the SKILL.md references.
  - agents.md / AGENTS.md / CLAUDE.md — Claude-Code-style markdown agent files.
    Jinja `Jinx('x')` / `NPC('y')` macros collapse to bare names.

Each function has a matching CLI wrapper registered in setup.py console_scripts:
  jinx2skill, skill2jinx, agents2npc, npc2agents.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from npcpy.npc_compiler import load_yaml_file


# ---------------------------------------------------------------------------
# IO + Jinja-compile helpers
# ---------------------------------------------------------------------------


def _read(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _identity_jinja_context() -> Dict[str, Any]:
    """Compile-out context: Jinx()/NPC()/ref() return the name verbatim, so
    `{{ Jinx('chat') }}` renders to the literal string 'chat'. No template
    syntax survives past load time."""
    return {
        'Jinx': lambda name: name,
        'NPC': lambda name: name,
        'ref': lambda name: name,
        'jinxes_list': lambda pattern: [],
    }


def _load_compiled_yaml(path: str) -> Dict[str, Any]:
    """Load a .npc file with Jinja fully resolved. For .npc, `{{ Jinx('x') }}`
    in the jinxes list collapses to the bare name 'x'. For .jinx, we bypass
    Jinja entirely — jinx code bodies carry their own runtime `{{ }}` that
    must survive to load-time untouched."""
    if path.endswith('.jinx'):
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        data = yaml.safe_load(raw)
        return data if isinstance(data, dict) else {}
    data = load_yaml_file(path, jinja_context=_identity_jinja_context())
    return data if isinstance(data, dict) else {}


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split `---\n...\n---\nbody` into (frontmatter_dict, body_str)."""
    if not text.startswith('---'):
        return {}, text
    parts = text.split('---', 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    return fm, parts[2].lstrip('\n')


def _dump_frontmatter(fm: Dict[str, Any], body: str) -> str:
    fm_yaml = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False).strip()
    return f"---\n{fm_yaml}\n---\n\n{body.rstrip()}\n"


def _slugify(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_') or 'step'


# ---------------------------------------------------------------------------
# jinx ↔ skill
# ---------------------------------------------------------------------------


def jinx_to_skill(jinx_path: str, out_dir: str) -> str:
    """Compile a .jinx into a skill folder: <out_dir>/<jinx_name>/ containing
    SKILL.md + one <step>.py per python step. Returns the SKILL.md path.

    Jinja is fully resolved before write — the skill folder never contains
    `{{ }}` syntax. Consumers that don't know Jinja can read it directly.
    """
    data = _load_compiled_yaml(jinx_path)
    name = data.get('jinx_name') or Path(jinx_path).stem
    description = (data.get('description') or '').strip()
    inputs = data.get('inputs', []) or []
    steps = data.get('steps', []) or []

    skill_dir = os.path.join(out_dir, name)

    step_files: List[Tuple[str, str]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        engine = step.get('engine', 'python')
        if engine != 'python':
            continue
        code = (step.get('code') or '').rstrip() + '\n'
        step_name = step.get('name') or f'step_{i + 1}'
        filename = f'{_slugify(step_name)}.py'
        _write(os.path.join(skill_dir, filename), code)
        step_files.append((step_name, filename))

    input_lines: List[str] = []
    for item in inputs:
        if isinstance(item, dict):
            for k, v in item.items():
                input_lines.append(f"- `{k}` (default: `{v!r}`)")
        elif isinstance(item, str):
            input_lines.append(f"- `{item}`")

    body_parts: List[str] = [f"# {name}"]
    if description:
        body_parts.append(description)
    if input_lines:
        body_parts.append("## Inputs\n\n" + "\n".join(input_lines))
    if step_files:
        step_lines = [f"- `{step_name}` → [`{fn}`](./{fn})" for step_name, fn in step_files]
        body_parts.append("## Steps\n\n" + "\n".join(step_lines))

    example_inputs: Dict[str, Any] = {}
    for item in inputs:
        if isinstance(item, dict):
            for k, v in item.items():
                example_inputs[k] = v
        elif isinstance(item, str):
            example_inputs[item] = "<value>"
    body_parts.append(
        "## Usage\n\n```\n"
        f"/run_jinx jinx_ref={name} input_values={json.dumps(example_inputs)}\n"
        "```"
    )

    skill_md = _dump_frontmatter(
        {'name': name, 'description': description},
        "\n\n".join(body_parts),
    )
    out_path = os.path.join(skill_dir, 'SKILL.md')
    _write(out_path, skill_md)
    return out_path


_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_STEP_LINK_RE = re.compile(r"-\s+`([^`]+)`\s*→\s*\[`([^`]+)`\]\(\./([^)]+)\)")
_INPUT_LINE_RE = re.compile(r"^-\s+`([^`]+)`(?:\s*\(default:\s*`([^`]+)`\))?", re.MULTILINE)


def _split_sections(body: str) -> Dict[str, str]:
    """Split markdown body into {heading: section_content} keyed by ## headings."""
    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[heading] = body[start:end].strip()
    return sections


def skill_to_jinx(skill_path: str, out_dir: str) -> str:
    """Read a skill folder's SKILL.md back into a single .jinx file.

    skill_path can point at either `<skill_dir>/SKILL.md` or the skill folder
    itself. Step .py files referenced in the `## Steps` section are inlined
    back into the jinx's steps[].code. Falls back to scanning for .py files
    alongside SKILL.md if no steps section exists.
    """
    if os.path.isdir(skill_path):
        skill_path = os.path.join(skill_path, 'SKILL.md')
    skill_dir = os.path.dirname(os.path.abspath(skill_path))
    raw = _read(skill_path)
    fm, body = _parse_frontmatter(raw)

    name = fm.get('name') or os.path.basename(skill_dir)
    description = fm.get('description') or ''
    if not description:
        first_para = body.strip().split('\n\n', 1)[0]
        description = first_para.lstrip('#').strip()

    sections = _split_sections(body)
    inputs_block = sections.get('Inputs', '')
    inputs: List[Dict[str, str]] = []
    for match in _INPUT_LINE_RE.finditer(inputs_block):
        key = match.group(1).split()[0]
        default = (match.group(2) or '').strip("'\"")
        inputs.append({key: default})

    step_entries: List[Tuple[str, str, str]] = list(_STEP_LINK_RE.findall(body))
    if not step_entries:
        fallback = sorted(
            f for f in os.listdir(skill_dir)
            if f.endswith('.py')
        )
        step_entries = [(Path(f).stem, f, f) for f in fallback]  # (name, display, file)

    steps: List[Dict[str, Any]] = []
    for entry in step_entries:
        step_name, _display, filename = entry
        code_path = os.path.join(skill_dir, filename)
        if not os.path.exists(code_path):
            continue
        code = _read(code_path).rstrip() + '\n'
        steps.append({'name': step_name, 'engine': 'python', 'code': code})

    jinx_data = {
        'jinx_name': name,
        'description': description,
        'inputs': inputs,
        'steps': steps or [{'name': 'run', 'engine': 'python', 'code': '# No steps recovered from skill.\n'}],
    }

    out_path = os.path.join(out_dir, name + '.jinx')
    _write(out_path, yaml.safe_dump(jinx_data, sort_keys=False, default_flow_style=False))
    return out_path


# ---------------------------------------------------------------------------
# agents ↔ npc
# ---------------------------------------------------------------------------


def _iter_inline_agents_md(path: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """Parse an agents.md / AGENTS.md / CLAUDE.md / single .md file into
    (name, frontmatter, body) tuples.

    If the file has H2 headings, each heading defines a separate agent and
    top-of-file frontmatter applies to all of them. If there are no H2
    headings, the whole file is treated as a single-agent definition with
    its frontmatter + body.
    """
    raw = _read(path)
    file_fm, rest = _parse_frontmatter(raw)

    has_h2 = any(line.startswith('## ') for line in rest.splitlines())
    if not has_h2:
        name = file_fm.get('name') or Path(path).stem
        return [(name, file_fm, rest.strip())]

    agents: List[Tuple[str, Dict[str, Any], str]] = []
    current_name: Optional[str] = None
    buf: List[str] = []
    for line in rest.splitlines():
        if line.startswith('## '):
            if current_name is not None:
                agents.append((current_name, dict(file_fm), '\n'.join(buf).strip()))
            current_name = line[3:].strip()
            buf = []
        elif current_name is not None:
            buf.append(line)
    if current_name is not None:
        agents.append((current_name, dict(file_fm), '\n'.join(buf).strip()))
    return agents


def _iter_dir_agents(path: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """Parse a directory of per-agent .md files into (name, frontmatter, body)."""
    agents: List[Tuple[str, Dict[str, Any], str]] = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith('.md'):
            continue
        fm, body = _parse_frontmatter(_read(os.path.join(path, fname)))
        name = fm.get('name') or fname[:-3]
        agents.append((name, fm, body.strip()))
    return agents


def agents_to_npc(agents_path: str, out_dir: str) -> List[str]:
    """Convert markdown agents into one .npc per agent.

    Accepts a single file (H2-delimited) or a directory of per-agent .md.
    Frontmatter `tools:` / `jinxes:` pass through as bare names in the output
    `jinxes:` list — no Jinja wrapping, since the compiler accepts bare names
    as jinxes_spec entries.
    """
    if os.path.isfile(agents_path):
        agents = _iter_inline_agents_md(agents_path)
    elif os.path.isdir(agents_path):
        agents = _iter_dir_agents(agents_path)
    else:
        return []

    written: List[str] = []
    for name, fm, body in agents:
        npc_data: Dict[str, Any] = {
            'name': name,
            'primary_directive': body or f"Agent imported from {agents_path}.",
        }
        if fm.get('model'):
            npc_data['model'] = fm['model']
        if fm.get('provider'):
            npc_data['provider'] = fm['provider']
        spec = fm.get('jinxes', fm.get('tools'))
        if isinstance(spec, list):
            npc_data['jinxes'] = [str(j) for j in spec if j]

        out_path = os.path.join(out_dir, name + '.npc')
        _write(out_path, yaml.safe_dump(npc_data, sort_keys=False, default_flow_style=False))
        written.append(out_path)
    return written


def _iter_npc_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.endswith('.npc') else []
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.npc')]
    return []


def npc_to_agents(npc_path: str, out_dir: str, combined: bool = False) -> List[str]:
    """Convert .npc file(s) into markdown agent files.

    Default: one .md per NPC with frontmatter. If combined=True, writes a
    single agents.md with H2 sections instead.

    The .npc's Jinja macros (`{{ Jinx('x') }}`) are compiled out via
    load_yaml_file with an identity Jinx context, so the output frontmatter
    carries bare names — no `{{ }}` survives.
    """
    agents: List[Tuple[str, Dict[str, Any], str]] = []
    for npc_file in _iter_npc_files(npc_path):
        data = _load_compiled_yaml(npc_file)
        name = data.get('name') or Path(npc_file).stem
        fm: Dict[str, Any] = {'name': name}
        if data.get('model'):
            fm['model'] = data['model']
        if data.get('provider'):
            fm['provider'] = data['provider']
        if data.get('description'):
            fm['description'] = data['description']
        jinxes_spec = data.get('jinxes')
        if isinstance(jinxes_spec, list):
            fm['tools'] = [str(j) for j in jinxes_spec if j]
        body = (data.get('primary_directive') or '').strip()
        agents.append((name, fm, body))

    written: List[str] = []
    if combined:
        parts = [f"## {name}\n\n{body.rstrip()}" for name, _fm, body in agents]
        out_path = os.path.join(out_dir, 'agents.md')
        _write(out_path, '\n\n'.join(parts) + '\n')
        written.append(out_path)
    else:
        for name, fm, body in agents:
            out_path = os.path.join(out_dir, name + '.md')
            _write(out_path, _dump_frontmatter(fm, body))
            written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# CLI wrappers
# ---------------------------------------------------------------------------


def _cli_jinx_to_skill(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='jinx2skill', description='Compile a .jinx into a skill folder (SKILL.md + one .py per step). Accepts a single .jinx file or a directory (recursed for bulk regen).')
    p.add_argument('jinx_path', help='Path to a .jinx file or a directory of jinxes.')
    p.add_argument('-o', '--out-dir', default='skills', help='Output skills root (default: ./skills).')
    args = p.parse_args(argv)

    if os.path.isdir(args.jinx_path):
        wrote, failed = [], []
        for dirpath, _, filenames in os.walk(args.jinx_path):
            for f in filenames:
                if not f.endswith('.jinx'):
                    continue
                src = os.path.join(dirpath, f)
                try:
                    wrote.append(jinx_to_skill(src, args.out_dir))
                except Exception as exc:
                    failed.append((src, str(exc)))
        for path in wrote:
            print(path)
        if failed:
            print(f"\n{len(failed)} failures:", file=sys.stderr)
            for src, err in failed:
                print(f"  {src}: {err}", file=sys.stderr)
            return 1
        return 0

    print(jinx_to_skill(args.jinx_path, args.out_dir))
    return 0


def _cli_skill_to_jinx(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='skill2jinx', description='Bundle a skill folder back into a .jinx.')
    p.add_argument('skill_path', help='Path to a SKILL.md or the skill directory.')
    p.add_argument('-o', '--out-dir', default='jinxes/lib', help='Output directory (default: ./jinxes/lib).')
    args = p.parse_args(argv)
    print(skill_to_jinx(args.skill_path, args.out_dir))
    return 0


def _cli_agents_to_npc(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='agents2npc', description='Convert markdown agent files into .npc files.')
    p.add_argument('agents_path', help='Path to agents.md / AGENTS.md / CLAUDE.md or a directory of agent .md files.')
    p.add_argument('-o', '--out-dir', default='npc_team', help='Output directory (default: ./npc_team).')
    args = p.parse_args(argv)
    for path in agents_to_npc(args.agents_path, args.out_dir):
        print(path)
    return 0


def _cli_npc_to_agents(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='npc2agents', description='Compile .npc files into markdown agent files.')
    p.add_argument('npc_path', help='Path to a .npc file or a directory containing them.')
    p.add_argument('-o', '--out-dir', default='agents', help='Output directory (default: ./agents).')
    p.add_argument('--combined', action='store_true', help='Write one agents.md with H2 sections instead of per-agent .md files.')
    args = p.parse_args(argv)
    for path in npc_to_agents(args.npc_path, args.out_dir, combined=args.combined):
        print(path)
    return 0


if __name__ == '__main__':
    prog = os.path.basename(sys.argv[0]).lower()
    table = {
        'jinx2skill': _cli_jinx_to_skill,
        'skill2jinx': _cli_skill_to_jinx,
        'agents2npc': _cli_agents_to_npc,
        'npc2agents': _cli_npc_to_agents,
    }
    handler = table.get(prog)
    if handler is None and len(sys.argv) > 1 and sys.argv[1] in table:
        handler = table[sys.argv[1]]
        sys.argv = [sys.argv[1]] + sys.argv[2:]
    if handler is None:
        print('Usage: jinx2skill|skill2jinx|agents2npc|npc2agents ...', file=sys.stderr)
        sys.exit(2)
    sys.exit(handler())
