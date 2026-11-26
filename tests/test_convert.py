import pytest
import yaml

from npcpy.convert import _cli_agents_to_npc


@pytest.mark.parametrize("description", ["Note with marker", "Note with --- marker"])
def test_agents_to_npc_preserves_frontmatter_and_directive(tmp_path, description):
    source = tmp_path / "agents.md"
    directive = "Check the report.\n\n---\n\nSummarize the findings."
    source.write_text(
        f"---\ndescription: {description}\n"
        "name: reviewer\nmodel: example-model\nprovider: example-provider\n"
        "tools:\n  - search\n  - summarize\n"
        f"---\n\n{directive}\n",
        encoding="utf-8",
    )
    output = tmp_path / "npc_team"

    assert _cli_agents_to_npc([str(source), "--out-dir", str(output)]) == 0

    (npc_file,) = output.glob("*.npc")
    assert yaml.safe_load(npc_file.read_text(encoding="utf-8")) == {
        "name": "reviewer",
        "model": "example-model",
        "provider": "example-provider",
        "jinxes": ["search", "summarize"],
        "primary_directive": directive,
    }
