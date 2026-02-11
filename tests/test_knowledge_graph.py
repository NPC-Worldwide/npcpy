"""Test suite for the Knowledge Graph module.

Tests CRUD operations, search, evolution lifecycle, and sleep/dream processes
using an in-memory SQLite database and mocked LLM calls.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from collections import defaultdict

from sqlalchemy import create_engine, text

from npcpy.memory.command_history import init_kg_schema, load_kg_from_db, save_kg_to_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kg_engine():
    """Create an in-memory SQLite engine with KG schema."""
    engine = create_engine('sqlite:///:memory:')
    init_kg_schema(engine)
    return engine


@pytest.fixture
def sample_kg():
    """Return a sample KG dict for testing."""
    return {
        "generation": 1,
        "facts": [
            {"statement": "Python is a programming language", "source_text": "", "type": "manual", "generation": 0, "origin": "test"},
            {"statement": "Python was created by Guido van Rossum", "source_text": "", "type": "manual", "generation": 0, "origin": "test"},
            {"statement": "JavaScript runs in browsers", "source_text": "", "type": "manual", "generation": 1, "origin": "test"},
        ],
        "concepts": [
            {"name": "Programming Languages", "generation": 0, "origin": "test"},
            {"name": "Web Development", "generation": 1, "origin": "test"},
        ],
        "concept_links": [("Programming Languages", "Web Development")],
        "fact_to_concept_links": {
            "Python is a programming language": ["Programming Languages"],
            "JavaScript runs in browsers": ["Programming Languages", "Web Development"],
        },
        "fact_to_fact_links": [
            ("Python is a programming language", "Python was created by Guido van Rossum")
        ]
    }


TEAM = "test_team"
NPC = "test_npc"
DIR = "/test/dir"


# ---------------------------------------------------------------------------
# Schema & persistence tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_init_kg_schema_creates_tables(self, kg_engine):
        with kg_engine.connect() as conn:
            # Check that all 4 KG tables exist
            for table in ['kg_facts', 'kg_concepts', 'kg_links', 'kg_metadata']:
                result = conn.execute(text(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                ))
                assert result.fetchone() is not None, f"Table {table} should exist"

    def test_load_empty_kg(self, kg_engine):
        kg = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        assert kg['generation'] == 0
        assert kg['facts'] == []
        assert kg['concepts'] == []
        assert kg['concept_links'] == []
        assert kg['fact_to_concept_links'] == {}
        assert kg['fact_to_fact_links'] == []

    def test_save_and_load_roundtrip(self, kg_engine, sample_kg):
        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)
        loaded = load_kg_from_db(kg_engine, TEAM, NPC, DIR)

        assert loaded['generation'] == 1
        assert len(loaded['facts']) == 3
        assert len(loaded['concepts']) == 2

        fact_stmts = {f['statement'] for f in loaded['facts']}
        assert "Python is a programming language" in fact_stmts
        assert "JavaScript runs in browsers" in fact_stmts

        concept_names = {c['name'] for c in loaded['concepts']}
        assert "Programming Languages" in concept_names
        assert "Web Development" in concept_names

    def test_save_ignores_duplicate_facts(self, kg_engine, sample_kg):
        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)
        # Save again with same facts
        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        loaded = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        assert len(loaded['facts']) == 3  # No duplicates

    def test_links_roundtrip(self, kg_engine, sample_kg):
        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)
        loaded = load_kg_from_db(kg_engine, TEAM, NPC, DIR)

        # fact_to_concept_links
        assert "Python is a programming language" in loaded['fact_to_concept_links']
        assert "Programming Languages" in loaded['fact_to_concept_links']["Python is a programming language"]

        # concept_links
        assert len(loaded['concept_links']) == 1

        # fact_to_fact_links
        assert len(loaded['fact_to_fact_links']) == 1

    def test_scope_isolation(self, kg_engine, sample_kg):
        """Facts saved under one scope shouldn't appear in another."""
        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        other_kg = load_kg_from_db(kg_engine, "other_team", "other_npc", "/other/dir")
        assert other_kg['facts'] == []
        assert other_kg['concepts'] == []


# ---------------------------------------------------------------------------
# CRUD operation tests
# ---------------------------------------------------------------------------

class TestCRUD:
    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_add_fact(self, mock_cwd, kg_engine):
        from npcpy.memory.knowledge_graph import kg_add_fact

        class FakeNpc:
            name = NPC
        class FakeTeam:
            name = TEAM

        result = kg_add_fact(kg_engine, "The sky is blue", npc=FakeNpc(), team=FakeTeam())
        assert "Added fact" in result

        kg = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        assert len(kg['facts']) == 1
        assert kg['facts'][0]['statement'] == "The sky is blue"
        assert kg['facts'][0]['type'] == "manual"
        assert kg['facts'][0]['source_text'] == "The sky is blue"

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_search_facts_keyword(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_search_facts

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        results = kg_search_facts(kg_engine, "Python", search_all_scopes=True)
        assert len(results) >= 2
        assert any("Python" in r for r in results)

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_search_facts_no_match(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_search_facts

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        results = kg_search_facts(kg_engine, "nonexistent_xyz", search_all_scopes=True)
        assert results == []

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_remove_fact(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_remove_fact

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        class FakeNpc:
            name = NPC
        class FakeTeam:
            name = TEAM

        result = kg_remove_fact(kg_engine, "JavaScript runs in browsers",
                                npc=FakeNpc(), team=FakeTeam())
        assert "Removed" in result

        kg = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        stmts = [f['statement'] for f in kg['facts']]
        assert "JavaScript runs in browsers" not in stmts
        assert len(kg['facts']) == 2

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_add_concept(self, mock_cwd, kg_engine):
        from npcpy.memory.knowledge_graph import kg_add_concept

        class FakeNpc:
            name = NPC
        class FakeTeam:
            name = TEAM

        result = kg_add_concept(kg_engine, "AI", "Artificial Intelligence",
                                npc=FakeNpc(), team=FakeTeam())
        assert "Added concept" in result

        kg = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        assert len(kg['concepts']) == 1
        assert kg['concepts'][0]['name'] == "AI"

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_remove_concept(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_remove_concept

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        class FakeNpc:
            name = NPC
        class FakeTeam:
            name = TEAM

        result = kg_remove_concept(kg_engine, "Web Development",
                                   npc=FakeNpc(), team=FakeTeam())
        assert "Removed" in result

        kg = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        names = [c['name'] for c in kg['concepts']]
        assert "Web Development" not in names

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_link_fact_to_concept(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_link_fact_to_concept

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        class FakeNpc:
            name = NPC
        class FakeTeam:
            name = TEAM

        result = kg_link_fact_to_concept(
            kg_engine,
            "Python was created by Guido van Rossum",
            "Programming Languages",
            npc=FakeNpc(), team=FakeTeam()
        )
        assert "Linked" in result

        kg = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        links = kg['fact_to_concept_links']
        assert "Python was created by Guido van Rossum" in links
        assert "Programming Languages" in links["Python was created by Guido van Rossum"]

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_get_all_facts(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_get_all_facts

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        facts = kg_get_all_facts(kg_engine, search_all_scopes=True)
        assert len(facts) == 3

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_list_concepts(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_list_concepts

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        concepts = kg_list_concepts(kg_engine, search_all_scopes=True)
        assert len(concepts) == 2
        assert "Programming Languages" in concepts

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_get_stats(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_get_stats

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        class FakeNpc:
            name = NPC
        class FakeTeam:
            name = TEAM

        stats = kg_get_stats(kg_engine, npc=FakeNpc(), team=FakeTeam())
        assert stats['total_facts'] == 3
        assert stats['total_concepts'] == 2
        assert stats['generation'] == 1

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_get_facts_for_concept(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_get_facts_for_concept

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        class FakeNpc:
            name = NPC
        class FakeTeam:
            name = TEAM

        facts = kg_get_facts_for_concept(kg_engine, "Programming Languages",
                                         npc=FakeNpc(), team=FakeTeam())
        assert "Python is a programming language" in facts
        assert "JavaScript runs in browsers" in facts


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearch:
    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_link_search(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_link_search

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        results = kg_link_search(kg_engine, "Python", search_all_scopes=True)
        assert len(results) >= 1
        assert results[0]['type'] == 'fact'
        assert results[0]['depth'] == 0
        assert results[0]['score'] == 1.0

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_link_search_traverses_links(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_link_search

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        results = kg_link_search(kg_engine, "Python", max_depth=2, search_all_scopes=True)
        # Should find seeds + linked concepts/facts
        contents = [r['content'] for r in results]
        assert any("Python" in c for c in contents)

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_link_search_empty(self, mock_cwd, kg_engine):
        from npcpy.memory.knowledge_graph import kg_link_search

        results = kg_link_search(kg_engine, "nonexistent", search_all_scopes=True)
        assert results == []

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_hybrid_search(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_hybrid_search

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        results = kg_hybrid_search(kg_engine, "Python", mode='keyword+link',
                                    search_all_scopes=True)
        assert len(results) >= 1
        assert all('score' in r for r in results)
        assert all('source' in r for r in results)

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_explore_concept(self, mock_cwd, kg_engine, sample_kg):
        from npcpy.memory.knowledge_graph import kg_explore_concept

        save_kg_to_db(kg_engine, sample_kg, TEAM, NPC, DIR)

        result = kg_explore_concept(kg_engine, "Programming Languages")
        assert result['concept'] == "Programming Languages"
        assert "Python is a programming language" in result['direct_facts']
        assert "Web Development" in result['related_concepts']


# ---------------------------------------------------------------------------
# Evolution lifecycle tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestEvolution:
    @patch('npcpy.memory.knowledge_graph.get_related_facts_llm', return_value=[])
    @patch('npcpy.memory.knowledge_graph.get_related_concepts_multi', return_value=["TestConcept"])
    @patch('npcpy.memory.knowledge_graph.generate_groups', return_value=[
        {"name": "TestConcept", "description": "A test concept"}
    ])
    @patch('npcpy.memory.knowledge_graph.zoom_in', return_value=[
        {"statement": "Implied fact 1", "source_text": "", "type": "implied"}
    ])
    @patch('npcpy.memory.knowledge_graph.get_facts', return_value=[
        {"statement": "Fact from content", "source_text": "", "type": "extracted"}
    ])
    def test_kg_initial_from_content(self, mock_get_facts, mock_zoom, mock_groups,
                                      mock_concepts, mock_related):
        from npcpy.memory.knowledge_graph import kg_initial

        result = kg_initial("Some test content", model="test", provider="test")

        assert result['generation'] == 0
        assert len(result['facts']) >= 2  # extracted + implied
        assert len(result['concepts']) == 1
        assert result['concepts'][0]['name'] == "TestConcept"

    @patch('npcpy.memory.knowledge_graph._get_similar_by_embedding',
           side_effect=lambda q, c, *a, **kw: c[:5])
    @patch('npcpy.memory.knowledge_graph.get_related_facts_llm', return_value=[])
    @patch('npcpy.memory.knowledge_graph.get_related_concepts_multi', return_value=[])
    @patch('npcpy.memory.knowledge_graph.generate_groups', return_value=[])
    @patch('npcpy.memory.knowledge_graph.zoom_in', return_value=[])
    def test_kg_initial_from_facts(self, mock_zoom, mock_groups, mock_concepts,
                                    mock_related, mock_embed):
        from npcpy.memory.knowledge_graph import kg_initial

        facts = [
            {"statement": "Fact A", "source_text": "", "type": "test"},
            {"statement": "Fact B", "source_text": "", "type": "test"},
        ]

        result = kg_initial(content=None, facts=facts, model="test", provider="test")

        assert result['generation'] == 0
        assert len(result['facts']) >= 2

    @patch('npcpy.memory.knowledge_graph.get_facts', return_value=[
        {"statement": "New fact from evolve", "source_text": "", "type": "extracted"}
    ])
    def test_kg_evolve_incremental_from_content(self, mock_get_facts, sample_kg):
        from npcpy.memory.knowledge_graph import kg_evolve_incremental

        evolved, _ = kg_evolve_incremental(
            existing_kg=sample_kg,
            new_content_text="New content to process",
            model="test",
            provider="test"
        )

        assert evolved['generation'] == sample_kg['generation'] + 1
        assert len(evolved['facts']) == len(sample_kg['facts']) + 1

        new_stmts = {f['statement'] for f in evolved['facts']}
        assert "New fact from evolve" in new_stmts

    def test_kg_evolve_incremental_with_prefab_facts(self, sample_kg):
        from npcpy.memory.knowledge_graph import kg_evolve_incremental

        new_facts = [
            {"statement": "Rust is a systems language", "source_text": "", "type": "manual"}
        ]

        evolved, _ = kg_evolve_incremental(
            existing_kg=sample_kg,
            new_facts=new_facts,
            model="test",
            provider="test"
        )

        assert evolved['generation'] == 2
        assert len(evolved['facts']) == 4

    def test_kg_evolve_incremental_no_input(self, sample_kg):
        from npcpy.memory.knowledge_graph import kg_evolve_incremental

        result, _ = kg_evolve_incremental(
            existing_kg=sample_kg,
            model="test",
            provider="test"
        )

        # Should return original KG unchanged
        assert result['generation'] == sample_kg['generation']

    @patch('npcpy.memory.knowledge_graph._get_similar_by_embedding',
           side_effect=lambda q, c, *a, **kw: c[:5])
    @patch('npcpy.memory.knowledge_graph.get_related_facts_llm', return_value=["Python is a programming language"])
    @patch('npcpy.memory.knowledge_graph.get_facts', return_value=[
        {"statement": "Python 3.12 adds new features", "source_text": "", "type": "extracted"}
    ])
    def test_kg_evolve_with_fact_linking(self, mock_facts, mock_related, mock_embed, sample_kg):
        from npcpy.memory.knowledge_graph import kg_evolve_incremental

        evolved, _ = kg_evolve_incremental(
            existing_kg=sample_kg,
            new_content_text="Python 3.12 was released",
            model="test",
            provider="test",
            link_facts_facts=True
        )

        assert len(evolved['fact_to_fact_links']) >= len(sample_kg['fact_to_fact_links'])


# ---------------------------------------------------------------------------
# Sleep & Dream tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestSleepDream:
    @patch('npcpy.memory.knowledge_graph.consolidate_facts_llm',
           return_value={'decision': 'unique'})
    @patch('npcpy.memory.knowledge_graph.zoom_in', return_value=[
        {"statement": "Deepened fact", "source_text": "", "type": "implied"}
    ])
    def test_kg_sleep_prune_and_deepen(self, mock_zoom, mock_consolidate, sample_kg):
        from npcpy.memory.knowledge_graph import kg_sleep_process

        result, _ = kg_sleep_process(
            sample_kg,
            model="test",
            provider="test",
            operations_config=['prune', 'deepen']
        )

        assert result['generation'] == sample_kg['generation'] + 1
        # Should have at least the original facts (deepen may add more)
        assert len(result['facts']) >= len(sample_kg['facts'])

    @patch('npcpy.memory.knowledge_graph.consolidate_facts_llm',
           return_value={'decision': 'redundant'})
    def test_kg_sleep_prune_removes_fact(self, mock_consolidate):
        from npcpy.memory.knowledge_graph import kg_sleep_process

        # Prune requires len(facts_map) > 10 or len(concepts_map) > 5
        large_kg = {
            "generation": 1,
            "facts": [
                {"statement": f"Fact number {i}", "source_text": "", "type": "test",
                 "generation": 0, "origin": "test"}
                for i in range(12)
            ],
            "concepts": [{"name": "TestConcept", "generation": 0, "origin": "test"}],
            "concept_links": [],
            "fact_to_concept_links": {},
            "fact_to_fact_links": []
        }

        result, _ = kg_sleep_process(
            large_kg,
            model="test",
            provider="test",
            operations_config=['prune']
        )

        # One fact should be pruned (consolidate returns 'redundant')
        assert len(result['facts']) == len(large_kg['facts']) - 1

    @patch('npcpy.memory.knowledge_graph.kg_evolve_incremental')
    @patch('npcpy.memory.knowledge_graph.get_llm_response', return_value={
        'response': {'dream_text': 'A dream about programming languages connecting the world'}
    })
    def test_kg_dream_process(self, mock_llm, mock_evolve, sample_kg):
        from npcpy.memory.knowledge_graph import kg_dream_process

        # Make evolve return an expanded KG
        dream_kg = dict(sample_kg)
        dream_kg['facts'] = list(sample_kg['facts']) + [
            {"statement": "Dream: languages unite", "source_text": "", "type": "dream", "generation": 2}
        ]
        dream_kg['generation'] = 2
        mock_evolve.return_value = (dream_kg, {})

        result, _ = kg_dream_process(
            sample_kg,
            model="test",
            provider="test",
            num_seeds=2
        )

        assert result['generation'] == 2
        # The new dream fact should be tagged with origin='dream'
        dream_facts = [f for f in result['facts'] if f.get('origin') == 'dream']
        assert len(dream_facts) >= 1

    def test_kg_dream_not_enough_concepts(self):
        from npcpy.memory.knowledge_graph import kg_dream_process

        tiny_kg = {
            "generation": 0,
            "facts": [{"statement": "Only fact", "source_text": "", "type": "test"}],
            "concepts": [{"name": "Only concept"}],
            "concept_links": [],
            "fact_to_concept_links": {},
            "fact_to_fact_links": []
        }

        result, _ = kg_dream_process(tiny_kg, model="test", provider="test", num_seeds=3)
        # Should return unchanged KG
        assert result['generation'] == 0


# ---------------------------------------------------------------------------
# Embedding helper tests
# ---------------------------------------------------------------------------

class TestEmbeddingHelpers:
    @patch('npcpy.gen.embeddings.get_embeddings')
    def test_get_similar_by_embedding(self, mock_get_emb):
        """Test that _get_similar_by_embedding returns top-K similar candidates."""
        from npcpy.memory.knowledge_graph import _get_similar_by_embedding
        import numpy as np

        # Mock embeddings: query=[1,0], candidates with varying similarity
        mock_get_emb.side_effect = lambda texts, *a, **kw: [
            [1.0, 0.0] if t == "query" else
            [0.9, 0.1] if t == "close" else
            [0.5, 0.5] if t == "medium" else
            [0.0, 1.0]  # far
            for t in texts
        ]

        candidates = ["close", "medium", "far"]
        result = _get_similar_by_embedding("query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0] == "close"  # Most similar
        assert result[1] == "medium"  # Second most similar

    def test_get_similar_by_embedding_small_list(self):
        """When candidates <= top_k, return all without embedding."""
        from npcpy.memory.knowledge_graph import _get_similar_by_embedding

        candidates = ["a", "b", "c"]
        result = _get_similar_by_embedding("query", candidates, top_k=5)
        assert result == candidates

    def test_get_similar_by_embedding_empty(self):
        """Empty candidates should return empty list."""
        from npcpy.memory.knowledge_graph import _get_similar_by_embedding

        result = _get_similar_by_embedding("query", [], top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_save_empty_kg(self, kg_engine):
        empty_kg = {
            "generation": 0,
            "facts": [],
            "concepts": [],
            "concept_links": [],
            "fact_to_concept_links": {},
            "fact_to_fact_links": []
        }
        save_kg_to_db(kg_engine, empty_kg, TEAM, NPC, DIR)
        loaded = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        assert loaded['facts'] == []
        assert loaded['concepts'] == []

    def test_special_characters_in_facts(self, kg_engine):
        kg = {
            "generation": 0,
            "facts": [
                {"statement": "He said \"hello world\"", "source_text": "", "type": "test", "generation": 0, "origin": "test"},
                {"statement": "It's a test with apostrophes", "source_text": "", "type": "test", "generation": 0, "origin": "test"},
                {"statement": "Emoji test: Python 🐍", "source_text": "", "type": "test", "generation": 0, "origin": "test"},
            ],
            "concepts": [],
            "concept_links": [],
            "fact_to_concept_links": {},
            "fact_to_fact_links": []
        }
        save_kg_to_db(kg_engine, kg, TEAM, NPC, DIR)
        loaded = load_kg_from_db(kg_engine, TEAM, NPC, DIR)
        stmts = {f['statement'] for f in loaded['facts']}
        assert 'He said "hello world"' in stmts
        assert "It's a test with apostrophes" in stmts
        assert "Emoji test: Python 🐍" in stmts

    def test_multiple_scopes(self, kg_engine, sample_kg):
        """Test that different scopes are fully isolated."""
        save_kg_to_db(kg_engine, sample_kg, "team_a", "npc_a", "/path/a")

        other_kg = {
            "generation": 5,
            "facts": [{"statement": "Totally different fact", "source_text": "", "type": "test", "generation": 5, "origin": "test"}],
            "concepts": [{"name": "Different Concept", "generation": 5, "origin": "test"}],
            "concept_links": [],
            "fact_to_concept_links": {},
            "fact_to_fact_links": []
        }
        save_kg_to_db(kg_engine, other_kg, "team_b", "npc_b", "/path/b")

        loaded_a = load_kg_from_db(kg_engine, "team_a", "npc_a", "/path/a")
        loaded_b = load_kg_from_db(kg_engine, "team_b", "npc_b", "/path/b")

        assert loaded_a['generation'] == 1
        assert loaded_b['generation'] == 5
        assert len(loaded_a['facts']) == 3
        assert len(loaded_b['facts']) == 1

    @patch('npcpy.memory.knowledge_graph.os.getcwd', return_value=DIR)
    def test_kg_evolve_knowledge_integration(self, mock_cwd, kg_engine, sample_kg):
        """Test the high-level kg_evolve_knowledge function."""
        from npcpy.memory.knowledge_graph import kg_evolve_knowledge

        save_kg_to_db(kg_engine, sample_kg, "default_team", "default_npc", DIR)

        with patch('npcpy.memory.knowledge_graph.kg_evolve_incremental') as mock_evolve:
            evolved = dict(sample_kg)
            evolved['generation'] = 2
            evolved['facts'] = list(sample_kg['facts']) + [
                {"statement": "New evolved fact", "type": "test", "generation": 2, "origin": "evolve"}
            ]
            mock_evolve.return_value = (evolved, {})

            result = kg_evolve_knowledge(kg_engine, "New content to evolve")
            assert result == "Knowledge graph evolved with new content"
