# Skills

Skills are knowledge-content jinxs that provide instructional sections to agents on demand. Unlike code-executing jinxs, skills serve structured knowledge that agents can request when they need guidance on specific topics.

## What is a Skill

A skill is a collection of knowledge sections organized under a single topic. When an agent needs information about code review, debugging, or any domain-specific task, it can call the relevant skill and request specific sections.

Skills use the same jinx pipeline as other workflows. The agent calls `code-review(section=checklist)` the same way it calls `sh(command=ls)` — the skill engine returns the requested knowledge section instead of executing code.

## Creating a Skill

Skills are authored as `SKILL.md` files inside a folder structure:

```
npc_team/
└── jinxs/
    └── skills/
        └── code-review/
            ├── SKILL.md
            ├── scripts/      # Optional helper scripts
            └── references/   # Optional reference files
```

### SKILL.md Format

A skill file has YAML frontmatter followed by markdown sections:

```markdown
---
name: code-review
description: Use when reviewing code for quality, security, and best practices.
---
# Code Review Skill

## checklist
- Check for security vulnerabilities (SQL injection, XSS, etc.)
- Verify error handling and edge cases
- Review naming conventions and code clarity
- Look for performance issues
- Ensure proper input validation

## security
Focus on OWASP top 10 vulnerabilities:
1. Injection attacks
2. Broken authentication
3. Sensitive data exposure
4. XML external entities (XXE)
5. Broken access control

## style
Follow language-specific conventions:
- Python: PEP 8
- JavaScript: ESLint defaults
- Go: gofmt standards

## performance
Look for:
- N+1 query patterns
- Unnecessary memory allocations
- Missing indexes on database queries
- Synchronous operations that could be async
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Skill identifier used to call it |
| `description` | Yes | When/how to use this skill |
| `scripts` | No | List of helper script paths |
| `references` | No | List of reference file paths |
| `assets` | No | List of asset file paths |
| `file_context` | No | Glob patterns for files to include |

### Sections

Each `## heading` becomes a requestable section. The agent can ask for:
- A specific section: `code-review(section=security)`
- All sections: `code-review(section=all)`

Section content is preserved exactly as written, including code blocks, lists, and formatting.

## Assigning Skills to NPCs

Reference skills in your NPC's `jinxs` list:

**reviewer.npc:**
```yaml
name: reviewer
primary_directive: You review code for quality and security issues.
model: llama3.2
provider: ollama
jinxs:
  - skills/code-review
  - skills/debugging
```

The skill becomes available as a callable jinx. The agent's system prompt includes the skill description so it knows when to use it.

## Using Skills Programmatically

Load an NPC with skills and let it use them during responses:

```python
from npcpy.npc_compiler import NPC

reviewer = NPC(file='./npc_team/reviewer.npc')

# The agent can now call skills during get_llm_response
response = reviewer.get_llm_response(
    "Review this function for security issues: def login(user, pwd): ..."
)
print(response['response'])
```

The agent will automatically invoke `code-review(section=security)` if it determines that knowledge is needed.

## Skill with Scripts and References

Skills can include helper scripts and reference files:

```
npc_team/jinxs/skills/testing/
├── SKILL.md
├── scripts/
│   ├── run_tests.sh
│   └── coverage_report.py
└── references/
    ├── pytest_cheatsheet.md
    └── testing_patterns.md
```

**SKILL.md:**
```markdown
---
name: testing
description: Use when writing or running tests.
scripts:
  - scripts/run_tests.sh
  - scripts/coverage_report.py
references:
  - references/pytest_cheatsheet.md
  - references/testing_patterns.md
---
# Testing Skill

## setup
To set up a test environment:
1. Create a `tests/` directory
2. Install pytest: `pip install pytest`
3. Create `conftest.py` for shared fixtures

## patterns
Common testing patterns:
- Arrange-Act-Assert
- Given-When-Then
- Table-driven tests

## mocking
Use `unittest.mock` or `pytest-mock`:
```python
from unittest.mock import Mock, patch

@patch('module.external_api')
def test_with_mock(mock_api):
    mock_api.return_value = {'status': 'ok'}
    result = function_under_test()
    assert result == expected
```
```

## File Context

Skills can automatically include file contents using glob patterns:

```markdown
---
name: project-conventions
description: Use when following project-specific conventions.
file_context:
  - "*.md"
  - "pyproject.toml"
  - ".eslintrc.*"
---
# Project Conventions

## overview
This skill includes the project's configuration files automatically.
The agent sees their contents when the skill is invoked.
```

When the skill is called, matching files are read and included in the context.

## Skills in Teams

Skills defined at the team level are available to all team members:

**team.ctx:**
```yaml
context: A development team with shared knowledge
forenpc: lead
model: llama3.2
provider: ollama
```

Place skills in `npc_team/jinxs/skills/` and they're automatically loaded for all NPCs in the team.

```python
from npcpy.npc_compiler import Team

team = Team(team_path='./npc_team')

# All NPCs in the team can use shared skills
lead = team.get_npc('lead')
response = lead.get_llm_response("Review the authentication module")
```

## How Skills Work Internally

1. **Parsing**: `_parse_skill_md()` reads the SKILL.md file, extracts frontmatter and sections
2. **Compilation**: `_compile_skill_to_jinx()` converts the skill to a Jinx with `engine: skill` steps
3. **Registration**: The skill jinx is added to `jinxs_dict` like any other jinx
4. **Execution**: When called, the skill engine returns the requested section content

The skill content is base64-encoded internally to survive Jinja templating without being mangled.

## Example: Debugging Skill

**npc_team/jinxs/skills/debugging/SKILL.md:**
```markdown
---
name: debugging
description: Use when debugging errors, exceptions, or unexpected behavior.
---
# Debugging Skill

## approach
1. Reproduce the issue consistently
2. Isolate the problem area
3. Form a hypothesis
4. Test the hypothesis
5. Fix and verify

## python-errors
Common Python error patterns:

**AttributeError**: Check if object is None or wrong type
**KeyError**: Verify key exists, use `.get()` with default
**ImportError**: Check module installation and paths
**TypeError**: Verify argument types match function signature

## logging
Add strategic logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Processing item: {item}")
logger.info(f"Completed {count} items")
logger.warning(f"Unexpected value: {value}")
logger.error(f"Failed to process: {e}")
```

## tools
Useful debugging tools:
- `pdb` / `breakpoint()` - Interactive debugger
- `logging` - Structured log output
- `traceback` - Stack trace formatting
- `sys.exc_info()` - Exception details
```

Use it in an NPC:

```yaml
# debugger.npc
name: debugger
primary_directive: You help debug code issues and errors.
model: llama3.2
provider: ollama
jinxs:
  - skills/debugging
```

```python
from npcpy.npc_compiler import NPC

debugger = NPC(file='./npc_team/debugger.npc')
response = debugger.get_llm_response(
    "I'm getting a KeyError in my code: data['user']['email']"
)
```

The agent will invoke `debugging(section=python-errors)` to get relevant guidance.
