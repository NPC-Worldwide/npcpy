from collections import defaultdict
import datetime
import hashlib
import json
import logging
import os
import random
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple, Any, Set

from npcpy.llm_funcs import (
    abstract,
    consolidate_facts_llm,
    generate_groups,
    get_facts,
    get_llm_response,
    get_related_concepts_multi,
    get_related_facts_llm,
    prune_fact_subset_llm,
    remove_idempotent_groups,
    zoom_in,
    )

from npcpy.memory.command_history import load_kg_from_db, save_kg_to_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_similar_by_embedding(query, candidates, model='nomic-embed-text',
                              provider='ollama', top_k=20):
    """Pre-filter candidates by embedding cosine similarity.

    Returns top-K candidate strings most similar to query.
    Falls back to returning all candidates if embedding fails.
    """
    if not candidates or len(candidates) <= top_k:
        return list(candidates)

    try:
        from npcpy.gen.embeddings import get_embeddings

        query_emb = np.array(get_embeddings([query], model, provider)[0])
        cand_embs = get_embeddings(list(candidates), model, provider)

        similarities = []
        for i, emb in enumerate(cand_embs):
            emb_arr = np.array(emb)
            norm_product = np.linalg.norm(query_emb) * np.linalg.norm(emb_arr)
            if norm_product > 0:
                sim = float(np.dot(query_emb, emb_arr) / norm_product)
            else:
                sim = 0.0
            similarities.append((sim, i))

        similarities.sort(key=lambda x: -x[0])
        return [candidates[idx] for _, idx in similarities[:top_k]]
    except Exception as e:
        logger.warning(f"Embedding pre-filter failed, using all candidates: {e}")
        return list(candidates)


def _sync_kg_facts_to_chroma(engine, chroma_path, embedding_model='nomic-embed-text',
                              embedding_provider='ollama', team_name=None,
                              npc_name=None, directory_path=None):
    """Sync KG facts from SQLAlchemy into a ChromaDB collection for fast ANN search.

    Returns (chroma_client, collection) or (None, None) on failure.
    """
    try:
        from npcpy.memory.command_history import setup_chroma_db
        from npcpy.gen.embeddings import get_embeddings
        from sqlalchemy import text
    except ImportError as e:
        logger.warning(f"Cannot sync to ChromaDB: {e}")
        return None, None

    scope_key = f"{team_name or 'all'}_{npc_name or 'all'}"
    collection_name = f"kg_facts_{scope_key}"

    chroma_client, collection = setup_chroma_db(
        collection_name,
        "KG facts for embedding search",
        chroma_path
    )

    with engine.connect() as conn:
        if team_name and npc_name:
            rows = conn.execute(text("""
                SELECT DISTINCT statement FROM kg_facts
                WHERE team_name = :team AND npc_name = :npc
            """), {"team": team_name, "npc": npc_name}).fetchall()
        else:
            rows = conn.execute(text(
                "SELECT DISTINCT statement FROM kg_facts"
            )).fetchall()

    if not rows:
        return chroma_client, collection

    statements = [r.statement for r in rows]

    # Check what's already in the collection
    existing = collection.get()
    existing_docs = set(existing.get('documents', []))

    new_statements = [s for s in statements if s not in existing_docs]

    if not new_statements:
        return chroma_client, collection

    # Batch embed and upsert
    batch_size = 50
    for i in range(0, len(new_statements), batch_size):
        batch = new_statements[i:i + batch_size]
        try:
            embeddings = get_embeddings(batch, embedding_model, embedding_provider)
            ids = [hashlib.md5(s.encode()).hexdigest() for s in batch]
            metadatas = [{"team": team_name or "all", "npc": npc_name or "all"} for _ in batch]
            collection.add(
                documents=batch,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.warning(f"Failed to embed batch {i}: {e}")

    return chroma_client, collection


# ---------------------------------------------------------------------------
# Chroma helpers (no Kuzu dependency)
# ---------------------------------------------------------------------------

def find_similar_facts_chroma(
    collection,
    query: str,
    query_embedding: List[float],
    n_results: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Find facts similar to the query using pre-generated embedding."""
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter,
        )

        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append(
                {
                    "fact": doc,
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None
                    ),
                }
            )

        return formatted_results
    except Exception as e:
        logger.warning(f"Error searching in Chroma: {e}")
        return []


def store_fact_with_embedding(
    collection, fact: str, metadata: dict, embedding: List[float]
) -> str:
    """Store a fact with its pre-generated embedding in Chroma DB."""
    try:
        fact_id = hashlib.md5(fact.encode()).hexdigest()

        collection.add(
            documents=[fact],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[fact_id],
        )

        return fact_id
    except Exception as e:
        logger.warning(f"Error storing fact in Chroma: {e}")
        return None


# ---------------------------------------------------------------------------
# Core KG lifecycle: initial, evolve, sleep, dream
# ---------------------------------------------------------------------------

def kg_initial(content,
               model=None,
               provider=None,
               npc=None,
               context='',
               facts=None,
               generation=None,
               verbose=True,
               embedding_model=None,
               embedding_provider=None):

    if generation is None:
        CURRENT_GENERATION = 0
    else:
        CURRENT_GENERATION = generation

    logger.info(f"Running KG Structuring Process (Generation: {CURRENT_GENERATION})")

    if facts is None:
        if not content:
            raise ValueError("kg_initial requires either content_text or a list of facts.")
        logger.info("Mode: Deriving new facts from text content...")
        all_facts = []
        if len(content) > 10000:
            for n in range(len(content) // 10000):
                content_to_sample = content[n * 10000:(n + 1) * 10000]
                extracted = get_facts(content_to_sample,
                                model=model,
                                provider=provider,
                                npc=npc,
                                context=context)
                if verbose:
                    logger.debug(f"Extracted {len(extracted)} facts from segment {n+1}")
                all_facts.extend(extracted)
        else:
            all_facts = get_facts(content,
                                  model=model,
                                  provider=provider,
                                  npc=npc,
                                  context=context)
            if verbose:
                logger.debug(f"Extracted {len(all_facts)} facts from content")
        for fact in all_facts:
            fact['generation'] = CURRENT_GENERATION
    else:
        logger.info(f"Mode: Building structure from {len(facts)} pre-existing facts...")
        all_facts = list(facts)

    logger.info("Inferring implied facts (zooming in)...")
    all_implied_facts = []
    if len(all_facts) > 20:
        sampled_facts = random.sample(all_facts, k=20)
        for n in range(len(all_facts) // 20):
            implied_facts = zoom_in(sampled_facts,
                                    model=model,
                                    provider=provider,
                                    npc=npc,
                                    context=context)
            all_implied_facts.extend(implied_facts)
            if verbose:
                logger.debug(f"Inferred {len(implied_facts)} implied facts from sample {n+1}")
    else:
        implied_facts = zoom_in(all_facts,
                                model=model,
                                provider=provider,
                                npc=npc,
                                context=context)
        all_implied_facts.extend(implied_facts)
        if verbose:
            logger.debug(f"Inferred {len(implied_facts)} implied facts from all facts")

    for fact in all_implied_facts:
        fact['generation'] = CURRENT_GENERATION

    all_facts = all_facts + all_implied_facts

    logger.info("Generating concepts from all facts...")
    concepts = generate_groups(all_facts,
                               model=model,
                               provider=provider,
                               npc=npc,
                               context=context)
    for concept in concepts:
        concept['generation'] = CURRENT_GENERATION

    if verbose:
        logger.debug(f"Generated {len(concepts)} concepts")

    logger.info("Linking facts to concepts...")
    fact_to_concept_links = defaultdict(list)
    concept_names = [c['name'] for c in concepts if c and 'name' in c]
    for fact in all_facts:
        fact_to_concept_links[fact['statement']] = get_related_concepts_multi(
            fact['statement'], "fact", concept_names, model, provider, npc, context)

    logger.info("Linking facts to other facts...")
    fact_to_fact_links = []
    fact_statements = [f['statement'] for f in all_facts]

    e_model = embedding_model or 'nomic-embed-text'
    e_provider = embedding_provider or 'ollama'

    for i, fact in enumerate(all_facts):
        other_fact_statements = [s for s in fact_statements if s != fact['statement']]
        if not other_fact_statements:
            continue
        try:
            # Pre-filter by embedding similarity to avoid passing too many to LLM
            candidates = _get_similar_by_embedding(
                fact['statement'], other_fact_statements, e_model, e_provider, top_k=20)
            if candidates:
                related_fact_stmts = get_related_facts_llm(fact['statement'],
                                                           candidates,
                                                           model=model,
                                                           provider=provider,
                                                           npc=npc,
                                                           context=context)
                for related_stmt in related_fact_stmts:
                    fact_to_fact_links.append((fact['statement'], related_stmt))
        except Exception as e:
            logger.warning(f"Failed to link fact {i+1}/{len(all_facts)}: {e}")
            continue

    return {
        "generation": CURRENT_GENERATION,
        "facts": all_facts,
        "concepts": concepts,
        "concept_links": [],
        "fact_to_concept_links": dict(fact_to_concept_links),
        "fact_to_fact_links": fact_to_fact_links
    }


def kg_evolve_incremental(existing_kg,
                          new_content_text=None,
                          new_facts=None,
                          model=None,
                          provider=None,
                          npc=None,
                          context='',
                          get_concepts=False,
                          link_concepts_facts=False,
                          link_concepts_concepts=False,
                          link_facts_facts=False,
                          embedding_model=None,
                          embedding_provider=None):

    current_gen = existing_kg.get('generation', 0)
    next_gen = current_gen + 1

    newly_added_concepts = []
    concept_links = list(existing_kg.get('concept_links', []))
    fact_to_concept_links = defaultdict(list,
                                        existing_kg.get('fact_to_concept_links', {}))
    fact_to_fact_links = list(existing_kg.get('fact_to_fact_links', []))

    existing_facts = existing_kg.get('facts', [])
    existing_concepts = existing_kg.get('concepts', [])
    existing_concept_names = {c['name'] for c in existing_concepts}
    existing_fact_statements = [f['statement'] for f in existing_facts]
    all_concept_names = list(existing_concept_names)

    all_new_facts = []

    if new_facts:
        all_new_facts = new_facts
        logger.info(f'Using pre-approved facts: {len(all_new_facts)}')
    elif new_content_text:
        logger.info('Extracting facts from content...')
        if len(new_content_text) > 10000:
            for n in range(len(new_content_text) // 10000):
                content_to_sample = new_content_text[n * 10000:(n + 1) * 10000]
                facts = get_facts(content_to_sample,
                                model=model,
                                provider=provider,
                                npc=npc,
                                context=context)
                all_new_facts.extend(facts)
        else:
            all_new_facts = get_facts(new_content_text,
                                model=model,
                                provider=provider,
                                npc=npc,
                                context=context)
    else:
        logger.info("No new content or facts provided")
        return existing_kg, {}

    existing_stmts = {f['statement'] for f in existing_facts}
    for fact in all_new_facts:
        fact['generation'] = next_gen

    final_facts = existing_facts + [f for f in all_new_facts if f['statement'] not in existing_stmts]

    if get_concepts:
        logger.info('Generating groups...')
        candidate_concepts = generate_groups(all_new_facts,
                                            model=model,
                                            provider=provider,
                                            npc=npc,
                                            context=context)
        for cand_concept in candidate_concepts:
            cand_name = cand_concept['name']
            if cand_name in existing_concept_names:
                continue
            cand_concept['generation'] = next_gen
            newly_added_concepts.append(cand_concept)
            if link_concepts_concepts:
                related_concepts = get_related_concepts_multi(cand_name,
                                                            "concept",
                                                            all_concept_names,
                                                            model,
                                                            provider,
                                                            npc,
                                                            context)
                for related_name in related_concepts:
                    if related_name != cand_name:
                        concept_links.append((cand_name, related_name))
            all_concept_names.append(cand_name)

        final_concepts = existing_concepts + newly_added_concepts

        if link_concepts_facts:
            for fact in all_new_facts:
                fact_to_concept_links[fact['statement']] = get_related_concepts_multi(
                    fact['statement'], "fact", all_concept_names,
                    model=model, provider=provider, npc=npc, context=context)
    else:
        final_concepts = existing_concepts

    if link_facts_facts and existing_fact_statements:
        e_model = embedding_model or 'nomic-embed-text'
        e_provider = embedding_provider or 'ollama'

        for new_fact in all_new_facts:
            # Pre-filter existing facts by embedding similarity
            candidates = _get_similar_by_embedding(
                new_fact['statement'], existing_fact_statements,
                e_model, e_provider, top_k=20)
            if candidates:
                related_fact_stmts = get_related_facts_llm(new_fact['statement'],
                                                           candidates,
                                                           model=model,
                                                           provider=provider,
                                                           npc=npc,
                                                           context=context)
                for related_stmt in related_fact_stmts:
                    fact_to_fact_links.append((new_fact['statement'], related_stmt))

    final_kg = {
        "generation": next_gen,
        "facts": final_facts,
        "concepts": final_concepts,
        "concept_links": concept_links,
        "fact_to_concept_links": dict(fact_to_concept_links),
        "fact_to_fact_links": fact_to_fact_links
    }
    return final_kg, {}


def kg_sleep_process(existing_kg,
                     model=None,
                     provider=None,
                     npc=None,
                     context='',
                     operations_config=None,
                     embedding_model=None,
                     embedding_provider=None):
    current_gen = existing_kg.get('generation', 0)
    next_gen = current_gen + 1
    logger.info(f"SLEEPING (Evolving Knowledge): Gen {current_gen} -> Gen {next_gen}")

    facts_map = {f['statement']: f for f in existing_kg.get('facts', [])}
    concepts_map = {c['name']: c for c in existing_kg.get('concepts', [])}
    fact_links = defaultdict(list, {k: list(v) for k, v in existing_kg.get('fact_to_concept_links', {}).items()})
    concept_links = set(tuple(sorted(link)) for link in existing_kg.get('concept_links', []))
    fact_to_fact_links = set(tuple(sorted(link)) for link in existing_kg.get('fact_to_fact_links', []))

    # Phase 1: Check for unstructured facts
    logger.info("Phase 1: Checking for unstructured facts...")
    facts_with_concepts = set(fact_links.keys())
    orphaned_fact_statements = list(set(facts_map.keys()) - facts_with_concepts)

    if len(orphaned_fact_statements) > 20:
        logger.info(f"Found {len(orphaned_fact_statements)} orphaned facts. Applying full KG structuring process...")
        orphaned_facts_as_dicts = [facts_map[s] for s in orphaned_fact_statements]

        new_structure = kg_initial(
            content=None,
            facts=orphaned_facts_as_dicts,
            model=model,
            provider=provider,
            npc=npc,
            context=context,
            generation=next_gen,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider
        )

        logger.info("Merging new structure into main KG...")
        for concept in new_structure.get("concepts", []):
            if concept['name'] not in concepts_map:
                concepts_map[concept['name']] = concept

        for fact_stmt, new_links in new_structure.get("fact_to_concept_links", {}).items():
            existing_links = set(fact_links.get(fact_stmt, []))
            existing_links.update(new_links)
            fact_links[fact_stmt] = list(existing_links)

        for f1, f2 in new_structure.get("fact_to_fact_links", []):
            fact_to_fact_links.add(tuple(sorted((f1, f2))))
    else:
        logger.info("Knowledge graph is sufficiently structured. Proceeding to refinement.")

    # Phase 2: Refinement operations
    if operations_config is None:
        possible_ops = ['prune', 'deepen']
        ops_to_run = random.sample(possible_ops, k=random.randint(1, 2))
    else:
        ops_to_run = operations_config

    logger.info(f"Phase 2: Executing refinement operations: {ops_to_run}")

    for op in ops_to_run:
        if op == 'prune' and (len(facts_map) > 10 or len(concepts_map) > 5):
            logger.info("Running 'prune' operation...")
            fact_to_check = random.choice(list(facts_map.values()))
            other_facts = [f for f in facts_map.values() if f['statement'] != fact_to_check['statement']]
            consolidation_result = consolidate_facts_llm(fact_to_check, other_facts, model, provider, npc, context)
            if consolidation_result.get('decision') == 'redundant':
                logger.info(f"Pruning redundant fact: '{fact_to_check['statement'][:80]}...'")
                del facts_map[fact_to_check['statement']]

        elif op == 'deepen' and facts_map:
            logger.info("Running 'deepen' operation...")
            fact_to_deepen = random.choice(list(facts_map.values()))
            implied_facts = zoom_in([fact_to_deepen], model, provider, npc, context)
            new_fact_count = 0
            for fact in implied_facts:
                if fact['statement'] not in facts_map:
                    fact.update({'generation': next_gen, 'origin': 'deepen'})
                    facts_map[fact['statement']] = fact
                    new_fact_count += 1
            if new_fact_count > 0:
                logger.info(f"Inferred {new_fact_count} new fact(s).")

        else:
            logger.debug(f"SKIPPED: Operation '{op}' did not run (conditions not met).")

    new_kg = {
        "generation": next_gen,
        "facts": list(facts_map.values()),
        "concepts": list(concepts_map.values()),
        "concept_links": [list(link) for link in concept_links],
        "fact_to_concept_links": dict(fact_links),
        "fact_to_fact_links": [list(link) for link in fact_to_fact_links]
    }
    return new_kg, {}


def kg_dream_process(existing_kg,
                     model=None,
                     provider=None,
                     npc=None,
                     context='',
                     num_seeds=3):
    current_gen = existing_kg.get('generation', 0)
    next_gen = current_gen + 1
    logger.info(f"DREAMING (Creative Synthesis): Gen {current_gen} -> Gen {next_gen}")
    concepts = existing_kg.get('concepts', [])
    if len(concepts) < num_seeds:
        logger.info(f"Not enough concepts ({len(concepts)}) for dream. Skipping.")
        return existing_kg, {}
    seed_concepts = random.sample(concepts, k=num_seeds)
    seed_names = [c['name'] for c in seed_concepts]
    logger.info(f"Dream seeded with: {seed_names}")
    prompt = f"""
    Write a short, speculative paragraph (a 'dream') that plausibly connects the concepts of {json.dumps(seed_names)}.
    Invent a brief narrative or a hypothetical situation.
    Respond with JSON: {{"dream_text": "A short paragraph..."}}
    """
    response = get_llm_response(prompt,
                                model=model,
                                provider=provider, npc=npc,
                                format="json", context=context)
    dream_text = response['response'].get('dream_text')
    if not dream_text:
        logger.info("Failed to generate a dream narrative. Skipping.")
        return existing_kg, {}
    logger.info(f"Generated Dream: '{dream_text[:150]}...'")

    dream_kg, _ = kg_evolve_incremental(existing_kg, new_content_text=dream_text,
                                         model=model, provider=provider, npc=npc, context=context)

    original_fact_stmts = {f['statement'] for f in existing_kg['facts']}
    for fact in dream_kg['facts']:
        if fact['statement'] not in original_fact_stmts:
            fact['origin'] = 'dream'
    original_concept_names = {c['name'] for c in existing_kg['concepts']}
    for concept in dream_kg['concepts']:
        if concept['name'] not in original_concept_names:
            concept['origin'] = 'dream'
    logger.info("Dream analysis complete. New knowledge integrated.")
    return dream_kg, {}


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_kg_with_pandas(kg, path_prefix="kg_state"):
    generation = kg.get("generation", 0)

    nodes_data = []
    for fact in kg.get('facts', []):
        nodes_data.append({'id': fact['statement'], 'type': 'fact', 'generation': fact.get('generation')})
    for concept in kg.get('concepts', []):
        nodes_data.append({'id': concept['name'], 'type': 'concept', 'generation': concept.get('generation')})
    pd.DataFrame(nodes_data).to_csv(f'{path_prefix}_gen{generation}_nodes.csv', index=False)

    links_data = []
    for fact_stmt, concepts in kg.get("fact_to_concept_links", {}).items():
        for concept_name in concepts:
            links_data.append({'source': fact_stmt, 'target': concept_name, 'type': 'fact_to_concept'})
    for c1, c2 in kg.get("concept_links", []):
        links_data.append({'source': c1, 'target': c2, 'type': 'concept_to_concept'})
    for f1, f2 in kg.get("fact_to_fact_links", []):
        links_data.append({'source': f1, 'target': f2, 'type': 'fact_to_fact'})
    pd.DataFrame(links_data).to_csv(f'{path_prefix}_gen{generation}_links.csv', index=False)
    logger.info(f"Saved KG Generation {generation} to CSV files.")


def save_changelog_to_json(changelog, from_gen, to_gen, path_prefix="changelog"):
    if not changelog:
        return
    with open(f"{path_prefix}_gen{from_gen}_to_{to_gen}.json", 'w', encoding='utf-8') as f:
        json.dump(changelog, f, indent=4)
    logger.info(f"Saved changelog for Gen {from_gen}->{to_gen}.")


# ---------------------------------------------------------------------------
# SQLAlchemy-backed CRUD operations
# ---------------------------------------------------------------------------

def kg_add_fact(
   engine,
   fact_text: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Add a new fact to the knowledge graph"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)

   new_fact = {
       "statement": fact_text,
       "source_text": fact_text,
       "type": "manual",
       "generation": kg_data.get('generation', 0),
       "origin": "manual_add"
   }

   kg_data['facts'].append(new_fact)
   save_kg_to_db(engine, kg_data, team_name, npc_name, directory_path)

   return f"Added fact: {fact_text}"


def kg_search_facts(
   engine,
   query: str,
   npc=None,
   team=None,
   model=None,
   provider=None,
   search_all_scopes=True
):
   """Search facts in the knowledge graph by keyword."""
   from sqlalchemy import text

   directory_path = os.getcwd()
   team_name = getattr(team, 'name', None) if team else None
   npc_name = getattr(npc, 'name', None) if npc else None

   matching_facts = []

   if search_all_scopes and (not team_name or not npc_name):
       with engine.connect() as conn:
           result = conn.execute(text("""
               SELECT DISTINCT statement FROM kg_facts
               WHERE LOWER(statement) LIKE LOWER(:query)
           """), {"query": f"%{query}%"})
           matching_facts = [row.statement for row in result]
   else:
       if not team_name:
           team_name = 'global_team'
       if not npc_name:
           npc_name = 'default_npc'
       kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
       for fact in kg_data.get('facts', []):
           if query.lower() in fact['statement'].lower():
               matching_facts.append(fact['statement'])

   return matching_facts


def kg_remove_fact(
   engine,
   fact_text: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Remove a fact from the knowledge graph"""
   from sqlalchemy import text as sa_text

   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   with engine.begin() as conn:
       result = conn.execute(sa_text("""
           DELETE FROM kg_facts
           WHERE statement = :statement AND team_name = :team_name
           AND npc_name = :npc_name AND directory_path = :directory_path
       """), {
           "statement": fact_text,
           "team_name": team_name,
           "npc_name": npc_name,
           "directory_path": directory_path
       })
       removed_count = result.rowcount

   if removed_count > 0:
       # Also remove any links referencing this fact
       with engine.begin() as conn:
           conn.execute(sa_text("""
               DELETE FROM kg_links
               WHERE (source = :fact OR target = :fact)
               AND team_name = :team_name AND npc_name = :npc_name
               AND directory_path = :directory_path
           """), {
               "fact": fact_text,
               "team_name": team_name,
               "npc_name": npc_name,
               "directory_path": directory_path
           })
       return f"Removed {removed_count} matching fact(s)"

   return "No matching facts found"


def kg_list_concepts(
   engine,
   npc=None,
   team=None,
   model=None,
   provider=None,
   search_all_scopes=True
):
   """List all concepts in the knowledge graph"""
   from sqlalchemy import text

   directory_path = os.getcwd()
   team_name = getattr(team, 'name', None) if team else None
   npc_name = getattr(npc, 'name', None) if npc else None

   if search_all_scopes and (not team_name or not npc_name):
       with engine.connect() as conn:
           result = conn.execute(text("SELECT DISTINCT name FROM kg_concepts"))
           return [row.name for row in result]
   else:
       if not team_name:
           team_name = 'global_team'
       if not npc_name:
           npc_name = 'default_npc'
       kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
       return [c['name'] for c in kg_data.get('concepts', [])]


def kg_get_facts_for_concept(
   engine,
   concept_name: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Get all facts linked to a specific concept"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)

   fact_to_concept_links = kg_data.get('fact_to_concept_links', {})
   linked_facts = []

   for fact_statement, linked_concepts in fact_to_concept_links.items():
       if concept_name in linked_concepts:
           linked_facts.append(fact_statement)

   return linked_facts


def kg_add_concept(
   engine,
   concept_name: str,
   concept_description: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Add a new concept to the knowledge graph"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)

   new_concept = {
       "name": concept_name,
       "description": concept_description,
       "generation": kg_data.get('generation', 0)
   }

   kg_data['concepts'].append(new_concept)
   save_kg_to_db(engine, kg_data, team_name, npc_name, directory_path)

   return f"Added concept: {concept_name}"


def kg_remove_concept(
   engine,
   concept_name: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Remove a concept from the knowledge graph"""
   from sqlalchemy import text as sa_text

   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   with engine.begin() as conn:
       result = conn.execute(sa_text("""
           DELETE FROM kg_concepts
           WHERE name = :name AND team_name = :team_name
           AND npc_name = :npc_name AND directory_path = :directory_path
       """), {
           "name": concept_name,
           "team_name": team_name,
           "npc_name": npc_name,
           "directory_path": directory_path
       })
       removed_count = result.rowcount

   if removed_count > 0:
       # Also remove any links referencing this concept
       with engine.begin() as conn:
           conn.execute(sa_text("""
               DELETE FROM kg_links
               WHERE (source = :concept OR target = :concept)
               AND team_name = :team_name AND npc_name = :npc_name
               AND directory_path = :directory_path
           """), {
               "concept": concept_name,
               "team_name": team_name,
               "npc_name": npc_name,
               "directory_path": directory_path
           })
       return f"Removed concept: {concept_name}"

   return "Concept not found"


def kg_link_fact_to_concept(
   engine,
   fact_text: str,
   concept_name: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Link a fact to a concept in the knowledge graph"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)

   fact_to_concept_links = kg_data.get('fact_to_concept_links', {})

   if fact_text not in fact_to_concept_links:
       fact_to_concept_links[fact_text] = []

   if concept_name not in fact_to_concept_links[fact_text]:
       fact_to_concept_links[fact_text].append(concept_name)
       kg_data['fact_to_concept_links'] = fact_to_concept_links
       save_kg_to_db(engine, kg_data, team_name, npc_name, directory_path)
       return f"Linked fact '{fact_text}' to concept '{concept_name}'"

   return "Fact already linked to concept"


def kg_get_all_facts(
   engine,
   npc=None,
   team=None,
   model=None,
   provider=None,
   search_all_scopes=True
):
   """Get all facts from the knowledge graph"""
   from sqlalchemy import text

   directory_path = os.getcwd()
   team_name = getattr(team, 'name', None) if team else None
   npc_name = getattr(npc, 'name', None) if npc else None

   if search_all_scopes and (not team_name or not npc_name):
       with engine.connect() as conn:
           result = conn.execute(text("SELECT DISTINCT statement FROM kg_facts"))
           return [row.statement for row in result]
   else:
       if not team_name:
           team_name = 'global_team'
       if not npc_name:
           npc_name = 'default_npc'
       kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
       return [f['statement'] for f in kg_data.get('facts', [])]


def kg_get_stats(
   engine,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Get statistics about the knowledge graph"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)

   return {
       "total_facts": len(kg_data.get('facts', [])),
       "total_concepts": len(kg_data.get('concepts', [])),
       "total_fact_concept_links": len(kg_data.get('fact_to_concept_links', {})),
       "generation": kg_data.get('generation', 0)
   }


def kg_evolve_knowledge(
   engine,
   content_text: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Evolve the knowledge graph with new content"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'

   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)

   evolved_kg, _ = kg_evolve_incremental(
       existing_kg=kg_data,
       new_content_text=content_text,
       model=npc.model if npc else model,
       provider=npc.provider if npc else provider,
       npc=npc,
       get_concepts=True,
       link_concepts_facts=False,
       link_concepts_concepts=False,
       link_facts_facts=False
   )

   save_kg_to_db(engine, evolved_kg, team_name, npc_name, directory_path)

   return "Knowledge graph evolved with new content"


# ---------------------------------------------------------------------------
# Search functions (SQLAlchemy-backed)
# ---------------------------------------------------------------------------

def kg_link_search(
    engine,
    query: str,
    npc=None,
    team=None,
    max_depth: int = 2,
    breadth_per_step: int = 5,
    max_results: int = 20,
    strategy: str = 'bfs',
    search_all_scopes: bool = True
):
    """Search KG by traversing links from keyword-matched seeds."""
    from sqlalchemy import text
    from collections import deque

    seeds = kg_search_facts(engine, query, npc=npc, team=team,
                           search_all_scopes=search_all_scopes)

    if not seeds:
        return []

    visited = set(seeds[:breadth_per_step])
    results = [{'content': s, 'type': 'fact', 'depth': 0, 'path': [s], 'score': 1.0}
               for s in seeds[:breadth_per_step]]

    if strategy == 'bfs':
        queue = deque()
        for seed in seeds[:breadth_per_step]:
            queue.append((seed, 'fact', 0, [seed], 1.0))
    else:
        queue = []
        for seed in seeds[:breadth_per_step]:
            queue.append((seed, 'fact', 0, [seed], 1.0))

    with engine.connect() as conn:
        while queue and len(results) < max_results:
            if strategy == 'bfs':
                current, curr_type, depth, path, score = queue.popleft()
            else:
                current, curr_type, depth, path, score = queue.pop()

            if depth >= max_depth:
                continue

            linked = []

            result = conn.execute(text("""
                SELECT target, type FROM kg_links WHERE source = :src
            """), {"src": current})
            for row in result:
                target_type = 'concept' if 'concept' in row.type else 'fact'
                linked.append((row.target, target_type, row.type))

            result = conn.execute(text("""
                SELECT source, type FROM kg_links WHERE target = :tgt
            """), {"tgt": current})
            for row in result:
                source_type = 'fact' if 'fact_to' in row.type else 'concept'
                linked.append((row.source, source_type, f"rev_{row.type}"))

            added = 0
            for item_content, item_type, link_type in linked:
                if item_content in visited or added >= breadth_per_step:
                    continue

                visited.add(item_content)
                new_path = path + [item_content]
                new_score = score * 0.8

                results.append({
                    'content': item_content,
                    'type': item_type,
                    'depth': depth + 1,
                    'path': new_path,
                    'score': new_score,
                    'link_type': link_type
                })

                queue.append((item_content, item_type, depth + 1, new_path, new_score))
                added += 1

    results.sort(key=lambda x: (-x['score'], x['depth']))
    return results[:max_results]


def kg_embedding_search(
    engine,
    query: str,
    npc=None,
    team=None,
    embedding_model: str = None,
    embedding_provider: str = None,
    similarity_threshold: float = 0.6,
    max_results: int = 20,
    include_concepts: bool = True,
    search_all_scopes: bool = True,
    chroma_path: str = None
):
    """Semantic search using embeddings.

    Uses ChromaDB for fast ANN search when available,
    falls back to brute-force cosine similarity.
    """
    from sqlalchemy import text

    try:
        from npcpy.gen.embeddings import get_embeddings
    except ImportError:
        logger.warning("Embeddings not available, falling back to keyword search")
        facts = kg_search_facts(engine, query, npc=npc, team=team,
                               search_all_scopes=search_all_scopes)
        return [{'content': f, 'type': 'fact', 'score': 0.5} for f in facts[:max_results]]

    model = embedding_model or 'nomic-embed-text'
    provider = embedding_provider or 'ollama'

    team_name = getattr(team, 'name', None) if team else None
    npc_name = getattr(npc, 'name', None) if npc else None

    results = []

    # Try ChromaDB fast path first
    if chroma_path:
        try:
            _, collection = _sync_kg_facts_to_chroma(
                engine, chroma_path, model, provider,
                team_name if not search_all_scopes else None,
                npc_name if not search_all_scopes else None
            )
            if collection:
                query_emb = get_embeddings([query], model, provider)[0]
                chroma_results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=max_results
                )
                if chroma_results and chroma_results['documents'] and chroma_results['documents'][0]:
                    for i, doc in enumerate(chroma_results['documents'][0]):
                        dist = chroma_results['distances'][0][i] if 'distances' in chroma_results else 0
                        # ChromaDB returns L2 distance by default; convert to similarity
                        sim = max(0, 1.0 - dist / 2.0)
                        if sim >= similarity_threshold:
                            results.append({'content': doc, 'type': 'fact', 'score': sim})

                    if include_concepts:
                        # Concepts still need brute-force (small set)
                        with engine.connect() as conn:
                            if search_all_scopes:
                                concept_rows = conn.execute(text(
                                    "SELECT DISTINCT name FROM kg_concepts"
                                )).fetchall()
                            else:
                                concept_rows = conn.execute(text("""
                                    SELECT name FROM kg_concepts
                                    WHERE team_name = :team AND npc_name = :npc
                                """), {"team": team_name, "npc": npc_name}).fetchall()

                            if concept_rows:
                                names = [r.name for r in concept_rows]
                                query_embedding = np.array(query_emb)
                                embeddings = get_embeddings(names, model, provider)
                                for i, name in enumerate(names):
                                    emb = np.array(embeddings[i])
                                    norm_p = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                                    if norm_p > 0:
                                        sim = float(np.dot(query_embedding, emb) / norm_p)
                                        if sim >= similarity_threshold:
                                            results.append({'content': name, 'type': 'concept', 'score': sim})

                    results.sort(key=lambda x: -x['score'])
                    return results[:max_results]
        except Exception as e:
            logger.warning(f"ChromaDB search failed, falling back to brute-force: {e}")

    # Brute-force fallback
    query_embedding = np.array(get_embeddings([query], model, provider)[0])

    with engine.connect() as conn:
        if search_all_scopes:
            fact_rows = conn.execute(text(
                "SELECT DISTINCT statement FROM kg_facts"
            )).fetchall()
        else:
            t_name = team_name or 'global_team'
            n_name = npc_name or 'default_npc'
            fact_rows = conn.execute(text("""
                SELECT statement FROM kg_facts
                WHERE team_name = :team AND npc_name = :npc
            """), {"team": t_name, "npc": n_name}).fetchall()

        if fact_rows:
            statements = [r.statement for r in fact_rows]
            embeddings = get_embeddings(statements, model, provider)

            for i, stmt in enumerate(statements):
                emb = np.array(embeddings[i])
                norm_p = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                if norm_p > 0:
                    sim = float(np.dot(query_embedding, emb) / norm_p)
                    if sim >= similarity_threshold:
                        results.append({'content': stmt, 'type': 'fact', 'score': sim})

        if include_concepts:
            if search_all_scopes:
                concept_rows = conn.execute(text(
                    "SELECT DISTINCT name FROM kg_concepts"
                )).fetchall()
            else:
                concept_rows = conn.execute(text("""
                    SELECT name FROM kg_concepts
                    WHERE team_name = :team AND npc_name = :npc
                """), {"team": t_name, "npc": n_name}).fetchall()

            if concept_rows:
                names = [r.name for r in concept_rows]
                embeddings = get_embeddings(names, model, provider)

                for i, name in enumerate(names):
                    emb = np.array(embeddings[i])
                    norm_p = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                    if norm_p > 0:
                        sim = float(np.dot(query_embedding, emb) / norm_p)
                        if sim >= similarity_threshold:
                            results.append({'content': name, 'type': 'concept', 'score': sim})

    results.sort(key=lambda x: -x['score'])
    return results[:max_results]


def kg_hybrid_search(
    engine,
    query: str,
    npc=None,
    team=None,
    mode: str = 'keyword+link',
    max_depth: int = 2,
    breadth_per_step: int = 5,
    max_results: int = 20,
    embedding_model: str = None,
    embedding_provider: str = None,
    similarity_threshold: float = 0.6,
    search_all_scopes: bool = True
):
    """Hybrid search combining multiple methods."""
    all_results = {}

    if 'keyword' in mode or mode == 'link' or mode == 'all':
        keyword_facts = kg_search_facts(engine, query, npc=npc, team=team,
                                        search_all_scopes=search_all_scopes)
        for f in keyword_facts:
            all_results[f] = {'content': f, 'type': 'fact', 'score': 0.7, 'source': 'keyword'}

    if 'embedding' in mode or mode == 'all':
        try:
            emb_results = kg_embedding_search(
                engine, query, npc=npc, team=team,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
                search_all_scopes=search_all_scopes
            )
            for r in emb_results:
                if r['content'] in all_results:
                    all_results[r['content']]['score'] = max(
                        all_results[r['content']]['score'], r['score']
                    ) * 1.1
                    all_results[r['content']]['source'] += '+embedding'
                else:
                    r['source'] = 'embedding'
                    all_results[r['content']] = r
        except Exception as e:
            logger.warning(f"Embedding search failed: {e}")

    if 'link' in mode or mode == 'all':
        link_results = kg_link_search(
            engine, query, npc=npc, team=team,
            max_depth=max_depth,
            breadth_per_step=breadth_per_step,
            max_results=max_results,
            search_all_scopes=search_all_scopes
        )
        for r in link_results:
            if r['content'] in all_results:
                all_results[r['content']]['score'] = max(
                    all_results[r['content']]['score'], r['score']
                ) * 1.05
                all_results[r['content']]['source'] += '+link'
                all_results[r['content']]['depth'] = r.get('depth', 0)
                all_results[r['content']]['path'] = r.get('path', [])
            else:
                r['source'] = 'link'
                all_results[r['content']] = r

    final = sorted(all_results.values(), key=lambda x: -x['score'])
    return final[:max_results]


# ---------------------------------------------------------------------------
# Backfill & explore
# ---------------------------------------------------------------------------

def kg_backfill_from_memories(
    engine,
    model: str = None,
    provider: str = None,
    npc=None,
    get_concepts: bool = True,
    link_concepts_facts: bool = False,
    link_concepts_concepts: bool = False,
    link_facts_facts: bool = False,
    dry_run: bool = False
):
    """Backfill KG from approved memories that haven't been incorporated yet."""
    from sqlalchemy import text

    stats = {
        'scopes_processed': 0,
        'facts_before': 0,
        'facts_after': 0,
        'concepts_before': 0,
        'concepts_after': 0,
        'scopes': []
    }

    with engine.connect() as conn:
        stats['facts_before'] = conn.execute(text("SELECT COUNT(*) FROM kg_facts")).scalar() or 0
        stats['concepts_before'] = conn.execute(text("SELECT COUNT(*) FROM kg_concepts")).scalar() or 0

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT npc, team, directory_path, initial_memory, final_memory
            FROM memory_lifecycle
            WHERE status IN ('human-approved', 'human-edited')
            ORDER BY npc, team, directory_path
        """))

        from collections import defaultdict
        memories_by_scope = defaultdict(list)
        for row in result:
            statement = row.final_memory or row.initial_memory
            scope = (row.npc or 'default', row.team or 'global_team', row.directory_path or os.getcwd())
            memories_by_scope[scope].append({
                'statement': statement,
                'source_text': '',
                'type': 'explicit',
                'generation': 0
            })

    if dry_run:
        for scope, facts in memories_by_scope.items():
            stats['scopes'].append({
                'scope': scope,
                'memory_count': len(facts)
            })
        stats['scopes_processed'] = len(memories_by_scope)
        return stats

    for (npc_name, team_name, directory_path), facts in memories_by_scope.items():
        existing_kg = load_kg_from_db(engine, team_name, npc_name, directory_path)

        existing_statements = {f['statement'] for f in existing_kg.get('facts', [])}
        new_facts = [f for f in facts if f['statement'] not in existing_statements]

        if not new_facts:
            continue

        try:
            evolved_kg, _ = kg_evolve_incremental(
                existing_kg=existing_kg,
                new_facts=new_facts,
                model=model or (npc.model if npc else None),
                provider=provider or (npc.provider if npc else None),
                npc=npc,
                get_concepts=get_concepts,
                link_concepts_facts=link_concepts_facts,
                link_concepts_concepts=link_concepts_concepts,
                link_facts_facts=link_facts_facts
            )
            save_kg_to_db(engine, evolved_kg, team_name, npc_name, directory_path)

            stats['scopes'].append({
                'scope': (npc_name, team_name, directory_path),
                'facts_added': len(new_facts),
                'concepts_added': len(evolved_kg.get('concepts', [])) - len(existing_kg.get('concepts', []))
            })
            stats['scopes_processed'] += 1

        except Exception as e:
            logger.warning(f"Error processing scope {npc_name}/{team_name}: {e}")

    with engine.connect() as conn:
        stats['facts_after'] = conn.execute(text("SELECT COUNT(*) FROM kg_facts")).scalar() or 0
        stats['concepts_after'] = conn.execute(text("SELECT COUNT(*) FROM kg_concepts")).scalar() or 0

    return stats


def kg_explore_concept(
    engine,
    concept_name: str,
    max_depth: int = 2,
    breadth_per_step: int = 10,
    search_all_scopes: bool = True
):
    """Explore all facts and related concepts for a given concept."""
    from sqlalchemy import text

    result = {
        'concept': concept_name,
        'direct_facts': [],
        'related_concepts': [],
        'extended_facts': []
    }

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT source FROM kg_links
            WHERE target = :concept AND type = 'fact_to_concept'
        """), {"concept": concept_name})
        result['direct_facts'] = [r.source for r in rows]

        rows = conn.execute(text("""
            SELECT target FROM kg_links
            WHERE source = :concept AND type = 'concept_to_concept'
            UNION
            SELECT source FROM kg_links
            WHERE target = :concept AND type = 'concept_to_concept'
        """), {"concept": concept_name})
        result['related_concepts'] = [r[0] for r in rows]

        if result['related_concepts'] and max_depth > 0:
            placeholders = ','.join([f':c{i}' for i in range(len(result['related_concepts']))])
            params = {f'c{i}': c for i, c in enumerate(result['related_concepts'])}

            rows = conn.execute(text(f"""
                SELECT DISTINCT source FROM kg_links
                WHERE target IN ({placeholders}) AND type = 'fact_to_concept'
            """), params)
            result['extended_facts'] = [r.source for r in rows
                                        if r.source not in result['direct_facts']]

    return result
