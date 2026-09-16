from collections import defaultdict, deque
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


logger = logging.getLogger(__name__)


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



def kg_initial(content,
               model=None,
               provider=None,
               npc=None,
               context='',
               facts=None,
               generation=None,
               verbose=True,
               embedding_model=None,
               embedding_provider=None,
               zoom_in_enabled=True):

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

    all_implied_facts = []
    if zoom_in_enabled:
        logger.info("Inferring implied facts (zooming in)...")
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
    else:
        logger.info("Skipping zoom-in (zoom_in_enabled=False)")

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



def kg_add_fact(
   kg_data,
   fact_text: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Add a new fact to the knowledge graph"""
   new_fact = {
       "statement": fact_text,
       "source_text": fact_text,
       "type": "manual",
       "generation": kg_data.get('generation', 0),
       "origin": "manual_add"
   }

   kg_data.setdefault('facts', []).append(new_fact)

   return f"Added fact: {fact_text}"


def kg_search_facts(
   kg_data,
   query: str,
   npc=None,
   team=None,
   model=None,
   provider=None,
   search_all_scopes=True
):
   """Search facts in the knowledge graph by keyword."""
   matching_facts = []

   for fact in kg_data.get('facts', []):
       if query.lower() in fact.get('statement', '').lower():
           matching_facts.append(fact['statement'])

   return matching_facts


def kg_remove_fact(
   kg_data,
   fact_text: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Remove a fact from the knowledge graph"""
   original_facts = kg_data.get('facts', [])
   original_count = len(original_facts)
   kg_data['facts'] = [f for f in original_facts if f.get('statement') != fact_text]
   removed_count = original_count - len(kg_data['facts'])

   if removed_count > 0:
       fact_to_concept_links = kg_data.get('fact_to_concept_links', {})
       if fact_text in fact_to_concept_links:
           del fact_to_concept_links[fact_text]
       kg_data['fact_to_concept_links'] = fact_to_concept_links

       kg_data['fact_to_fact_links'] = [
           (s, t) for s, t in kg_data.get('fact_to_fact_links', [])
           if s != fact_text and t != fact_text
       ]

       return f"Removed {removed_count} matching fact(s)"

   return "No matching facts found"


def kg_list_concepts(
   kg_data,
   npc=None,
   team=None,
   model=None,
   provider=None,
   search_all_scopes=True
):
   """List all concepts in the knowledge graph"""
   return [c['name'] for c in kg_data.get('concepts', [])]


def kg_get_facts_for_concept(
   kg_data,
   concept_name: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Get all facts linked to a specific concept"""
   fact_to_concept_links = kg_data.get('fact_to_concept_links', {})
   linked_facts = []

   for fact_statement, linked_concepts in fact_to_concept_links.items():
       if concept_name in linked_concepts:
           linked_facts.append(fact_statement)

   return linked_facts


def kg_add_concept(
   kg_data,
   concept_name: str,
   concept_description: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Add a new concept to the knowledge graph"""
   new_concept = {
       "name": concept_name,
       "description": concept_description,
       "generation": kg_data.get('generation', 0)
   }

   kg_data.setdefault('concepts', []).append(new_concept)

   return f"Added concept: {concept_name}"


def kg_remove_concept(
   kg_data,
   concept_name: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Remove a concept from the knowledge graph"""
   original_concepts = kg_data.get('concepts', [])
   original_count = len(original_concepts)
   kg_data['concepts'] = [c for c in original_concepts if c.get('name') != concept_name]
   removed_count = original_count - len(kg_data['concepts'])

   if removed_count > 0:
       fact_to_concept_links = kg_data.get('fact_to_concept_links', {})
       for fact_statement in list(fact_to_concept_links.keys()):
           links = fact_to_concept_links[fact_statement]
           if concept_name in links:
               links = [c for c in links if c != concept_name]
               if links:
                   fact_to_concept_links[fact_statement] = links
               else:
                   del fact_to_concept_links[fact_statement]
       kg_data['fact_to_concept_links'] = fact_to_concept_links

       kg_data['concept_links'] = [
           (s, t) for s, t in kg_data.get('concept_links', [])
           if s != concept_name and t != concept_name
       ]

       return f"Removed concept: {concept_name}"

   return "Concept not found"


def kg_link_fact_to_concept(
   kg_data,
   fact_text: str,
   concept_name: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Link a fact to a concept in the knowledge graph"""
   fact_to_concept_links = kg_data.setdefault('fact_to_concept_links', {})

   if fact_text not in fact_to_concept_links:
       fact_to_concept_links[fact_text] = []

   if concept_name not in fact_to_concept_links[fact_text]:
       fact_to_concept_links[fact_text].append(concept_name)
       return f"Linked fact '{fact_text}' to concept '{concept_name}'"

   return "Fact already linked to concept"


def kg_get_all_facts(
   kg_data,
   npc=None,
   team=None,
   model=None,
   provider=None,
   search_all_scopes=True
):
   """Get all facts from the knowledge graph"""
   return [f['statement'] for f in kg_data.get('facts', [])]


def kg_get_stats(
   kg_data,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Get statistics about the knowledge graph"""
   return {
       "total_facts": len(kg_data.get('facts', [])),
       "total_concepts": len(kg_data.get('concepts', [])),
       "total_fact_concept_links": len(kg_data.get('fact_to_concept_links', {})),
       "generation": kg_data.get('generation', 0)
   }


def kg_evolve_knowledge(
   kg_data,
   content_text: str,
   npc=None,
   team=None,
   model=None,
   provider=None
):
   """Evolve the knowledge graph with new content"""
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

   kg_data.update(evolved_kg)

   return "Knowledge graph evolved with new content"



def kg_link_search(
    kg_data,
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
    seeds = kg_search_facts(kg_data, query, npc=npc, team=team,
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

    fact_to_concept = kg_data.get('fact_to_concept_links', {})
    concept_links = kg_data.get('concept_links', [])
    fact_to_fact_links = kg_data.get('fact_to_fact_links', [])

    while queue and len(results) < max_results:
        if strategy == 'bfs':
            current, curr_type, depth, path, score = queue.popleft()
        else:
            current, curr_type, depth, path, score = queue.pop()

        if depth >= max_depth:
            continue

        linked = []

        if curr_type == 'fact':
            for concept in fact_to_concept.get(current, []):
                linked.append((concept, 'concept', 'fact_to_concept'))
            for source, target in fact_to_fact_links:
                if source == current:
                    linked.append((target, 'fact', 'fact_to_fact'))
                elif target == current:
                    linked.append((source, 'fact', 'rev_fact_to_fact'))
        elif curr_type == 'concept':
            for fact_statement, linked_concepts in fact_to_concept.items():
                if current in linked_concepts:
                    linked.append((fact_statement, 'fact', 'rev_fact_to_concept'))
            for source, target in concept_links:
                if source == current:
                    linked.append((target, 'concept', 'concept_to_concept'))
                elif target == current:
                    linked.append((source, 'concept', 'rev_concept_to_concept'))

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
    kg_data,
    query: str,
    npc=None,
    team=None,
    embedding_model: str = None,
    embedding_provider: str = None,
    similarity_threshold: float = 0.6,
    max_results: int = 20,
    include_concepts: bool = True,
    search_all_scopes: bool = True,
):
    """Semantic search using embeddings via brute-force cosine similarity."""
    try:
        from npcpy.gen.embeddings import get_embeddings
    except ImportError:
        logger.warning("Embeddings not available, falling back to keyword search")
        facts = kg_search_facts(kg_data, query, npc=npc, team=team,
                               search_all_scopes=search_all_scopes)
        return [{'content': f, 'type': 'fact', 'score': 0.5} for f in facts[:max_results]]

    model = embedding_model or 'nomic-embed-text'
    provider = embedding_provider or 'ollama'

    results = []

    query_embedding = np.array(get_embeddings([query], model, provider)[0])

    facts = kg_data.get('facts', [])
    if facts:
        statements = [f['statement'] for f in facts]
        embeddings = get_embeddings(statements, model, provider)

        for i, stmt in enumerate(statements):
            emb = np.array(embeddings[i])
            norm_p = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            if norm_p > 0:
                sim = float(np.dot(query_embedding, emb) / norm_p)
                if sim >= similarity_threshold:
                    results.append({'content': stmt, 'type': 'fact', 'score': sim})

    if include_concepts:
        concepts = kg_data.get('concepts', [])
        if concepts:
            names = [c['name'] for c in concepts]
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




def kg_backfill_from_memories(
    kg_data,
    model: str = None,
    provider: str = None,
    npc=None,
    get_concepts: bool = True,
    link_concepts_facts: bool = False,
    link_concepts_concepts: bool = False,
    link_facts_facts: bool = False,
    dry_run: bool = False,
    context: str = ''
):
    """Backfill KG from approved memories that haven't been incorporated yet."""
    return {
        'scopes_processed': 0,
        'facts_before': 0,
        'facts_after': 0,
        'concepts_before': 0,
        'concepts_after': 0,
        'scopes': []
    }


def kg_explore_concept(
    kg_data,
    concept_name: str,
    max_depth: int = 2,
    breadth_per_step: int = 10,
    search_all_scopes: bool = True
):
    """Explore all facts and related concepts for a given concept."""
    result = {
        'concept': concept_name,
        'direct_facts': [],
        'related_concepts': [],
        'extended_facts': []
    }

    fact_to_concept_links = kg_data.get('fact_to_concept_links', {})
    for fact_statement, linked_concepts in fact_to_concept_links.items():
        if concept_name in linked_concepts:
            result['direct_facts'].append(fact_statement)

    concept_links = kg_data.get('concept_links', [])
    related = set()
    for source, target in concept_links:
        if source == concept_name:
            related.add(target)
        elif target == concept_name:
            related.add(source)
    result['related_concepts'] = list(related)

    if result['related_concepts'] and max_depth > 0:
        extended = set()
        for related_concept in result['related_concepts']:
            for fact_statement, linked_concepts in fact_to_concept_links.items():
                if related_concept in linked_concepts and fact_statement not in result['direct_facts']:
                    extended.add(fact_statement)
        result['extended_facts'] = list(extended)

    return result


def kg_hybrid_search(
    kg_data,
    query: str,
    npc=None,
    team=None,
    model=None,
    provider=None,
    max_results: int = 10,
):
    """Combine link traversal and embedding similarity into one ranked result list."""
    link_results = kg_link_search(
        kg_data,
        query,
        npc=npc,
        team=team,
        max_results=max_results * 2,
    )

    embedding_results = []
    try:
        embedding_results = kg_embedding_search(
            kg_data,
            query,
            npc=npc,
            team=team,
            embedding_model=model,
            embedding_provider=provider,
            max_results=max_results * 2,
        )
    except Exception as e:
        logger.warning(f"Embedding search failed in hybrid search: {e}")

    combined = {}
    for result in link_results:
        text = result.get('content')
        if not text:
            continue
        if text not in combined:
            combined[text] = {
                'text': text,
                'score': float(result.get('score', 0.5)),
                'type': result.get('type'),
                'depth': result.get('depth'),
            }

    for result in embedding_results:
        text = result.get('content')
        if not text:
            continue
        if text in combined:
            combined[text]['score'] = max(
                combined[text]['score'],
                float(result.get('score', 0.5)),
            )
        else:
            combined[text] = {
                'text': text,
                'score': float(result.get('score', 0.5)),
                'type': result.get('type'),
            }

    ranked = sorted(combined.values(), key=lambda item: -item['score'])
    return ranked[:max_results]
