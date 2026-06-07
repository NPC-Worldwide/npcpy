#!/usr/bin/env python3
"""
Evolve existing knowledge graphs: fix missing links, sleep, dream.

Usage:
    python scripts/evolve_kg.py [--db-path PATH] [--scope SCOPE] [--op OP]

Operations:
    structure  - Run kg_sleep_process to link orphaned facts to concepts
    sleep      - Run kg_sleep_process (prune + deepen)
    dream      - Run kg_dream_process (creative synthesis)
    all        - structure → sleep → dream
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')

from sqlalchemy import create_engine
from npcsh.history import load_kg_from_db, save_kg_to_db
from npcpy.memory.knowledge_graph import (
    kg_sleep_process, kg_dream_process, kg_initial
)


def get_engine(db_path: str):
    db_path = os.path.expanduser(db_path)
    return create_engine(f'sqlite:///{db_path}')


def print_kg_summary(kg, label=""):
    facts = kg.get('facts', [])
    concepts = kg.get('concepts', [])
    f2c = kg.get('fact_to_concept_links', {})
    c2c = kg.get('concept_links', [])
    f2f = kg.get('fact_to_fact_links', [])
    linked_facts = set(f2c.keys())
    orphaned = [f for f in facts if f['statement'] not in linked_facts]

    print(f"\n{'='*60}")
    print(f"  {label} | Gen {kg.get('generation', '?')}")
    print(f"  Facts: {len(facts)}  Concepts: {len(concepts)}")
    print(f"  Links: {sum(len(v) for v in f2c.values())} fact→concept, "
          f"{len(c2c)} concept→concept, {len(f2f)} fact→fact")
    print(f"  Orphaned facts (no concept): {len(orphaned)}")
    print(f"{'='*60}\n")


def run_structure(engine, team, npc, dirpath, model, provider):
    """Use kg_initial on orphaned facts to generate concept links."""
    kg = load_kg_from_db(engine, team, npc, dirpath)
    print_kg_summary(kg, f"BEFORE structure ({team}/{npc})")

    linked_facts = set(kg.get('fact_to_concept_links', {}).keys())
    orphaned = [f for f in kg['facts'] if f['statement'] not in linked_facts]

    if not orphaned:
        print("  No orphaned facts. Skipping structure step.")
        return kg

    print(f"  Structuring {len(orphaned)} orphaned facts...")

    try:
        new_structure = kg_initial(
            content=None,
            facts=orphaned,
            model=model,
            provider=provider,
            context='',
            generation=kg.get('generation', 0),
            embedding_model='nomic-embed-text',
            embedding_provider='ollama'
        )
    except Exception as e:
        print(f"  WARNING: kg_initial partially failed ({e}), saving what we have...")
        new_structure = {"concepts": [], "fact_to_concept_links": {},
                         "concept_links": [], "fact_to_fact_links": []}

    # Merge new structure into existing KG
    existing_concept_names = {c['name'] for c in kg.get('concepts', [])}
    for concept in new_structure.get('concepts', []):
        if concept['name'] not in existing_concept_names:
            kg['concepts'].append(concept)
            existing_concept_names.add(concept['name'])

    f2c = kg.get('fact_to_concept_links', {})
    for fact_stmt, new_links in new_structure.get('fact_to_concept_links', {}).items():
        existing = set(f2c.get(fact_stmt, []))
        existing.update(new_links)
        f2c[fact_stmt] = list(existing)
    kg['fact_to_concept_links'] = f2c

    c2c = set(tuple(sorted(l)) for l in kg.get('concept_links', []))
    for link in new_structure.get('concept_links', []):
        c2c.add(tuple(sorted(link)))
    kg['concept_links'] = [list(l) for l in c2c]

    f2f = set(tuple(sorted(l)) for l in kg.get('fact_to_fact_links', []))
    for link in new_structure.get('fact_to_fact_links', []):
        f2f.add(tuple(sorted(link)))
    kg['fact_to_fact_links'] = [list(l) for l in f2f]

    save_kg_to_db(engine, kg, team, npc, dirpath)
    print_kg_summary(kg, f"AFTER structure ({team}/{npc})")
    return kg


def run_sleep(engine, team, npc, dirpath, model, provider):
    """Run sleep process (prune + deepen)."""
    kg = load_kg_from_db(engine, team, npc, dirpath)
    print_kg_summary(kg, f"BEFORE sleep ({team}/{npc})")

    result, changelog = kg_sleep_process(
        kg,
        model=model,
        provider=provider,
        embedding_model='nomic-embed-text',
        embedding_provider='ollama'
    )

    save_kg_to_db(engine, result, team, npc, dirpath)
    print_kg_summary(result, f"AFTER sleep ({team}/{npc})")
    return result


def run_dream(engine, team, npc, dirpath, model, provider):
    """Run dream process (creative synthesis)."""
    kg = load_kg_from_db(engine, team, npc, dirpath)
    print_kg_summary(kg, f"BEFORE dream ({team}/{npc})")

    result, changelog = kg_dream_process(
        kg,
        model=model,
        provider=provider,
        num_seeds=3
    )

    save_kg_to_db(engine, result, team, npc, dirpath)
    print_kg_summary(result, f"AFTER dream ({team}/{npc})")
    return result


SCOPES = {
    'guac': ('global_team', 'guac', '/home/caug/npcww/npcsh'),
    'sibiji': ('npcsh', 'sibiji', '/home/caug/npcww/npc-core/npcsh'),
    'corca': ('global_team', 'corca', '/home/caug/npcww/udacity'),
    'sibiji-web': ('npcsh', 'sibiji', '/home/caug/npcww/npc-web-apps/sibiji'),
}


def main():
    parser = argparse.ArgumentParser(description='Evolve knowledge graphs')
    parser.add_argument('--db-path', default='~/npcsh_history.db')
    parser.add_argument('--scope', choices=list(SCOPES.keys()) + ['all'], default='all',
                        help='Which KG scope to evolve')
    parser.add_argument('--op', choices=['structure', 'sleep', 'dream', 'all'], default='all',
                        help='Which operation to run')
    parser.add_argument('--model', default='qwen3:8b', help='LLM model')
    parser.add_argument('--provider', default='ollama', help='LLM provider')

    args = parser.parse_args()
    db_path = os.path.expanduser(args.db_path)
    engine = get_engine(db_path)

    scopes = SCOPES if args.scope == 'all' else {args.scope: SCOPES[args.scope]}

    for scope_name, (team, npc, dirpath) in scopes.items():
        print(f"\n{'#'*60}")
        print(f"# Scope: {scope_name} ({team}/{npc})")
        print(f"{'#'*60}")

        ops = ['structure', 'sleep', 'dream'] if args.op == 'all' else [args.op]

        for op in ops:
            try:
                if op == 'structure':
                    run_structure(engine, team, npc, dirpath, args.model, args.provider)
                elif op == 'sleep':
                    run_sleep(engine, team, npc, dirpath, args.model, args.provider)
                elif op == 'dream':
                    run_dream(engine, team, npc, dirpath, args.model, args.provider)
            except Exception as e:
                print(f"\n  ERROR during '{op}' on {scope_name}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == '__main__':
    main()
