from collections import defaultdict
import datetime
import json
try:
    import kuzu
except ModuleNotFoundError:
    print("kuzu not installed")
import os
import random 
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

def safe_kuzu_execute(conn, query, error_message="Kuzu query failed"):
    """Execute a Kuzu query with proper error handling"""
    try:
        result = conn.execute(query)
        return result, None
    except Exception as e:
        error = f"{error_message}: {str(e)}"
        print(error)
        return None, error


def create_group(conn, name: str, metadata: str = ""):
    """Create a new group in the database with robust error handling"""
    if conn is None:
        print("Cannot create group: database connection is None")
        return False

    try:
        
        escaped_name = name.replace('"', '\\"')
        escaped_metadata = metadata.replace('"', '\\"')

        query = f"""
        CREATE (g:Groups {{
            name: "{escaped_name}",
            metadata: "{escaped_metadata}"
        }});
        """

        result, error = safe_kuzu_execute(
            conn, query, f"Failed to create group: {name}"
        )
        if error:
            return False

        print(f"Created group: {name}")
        return True
    except Exception as e:
        print(f"Error creating group {name}: {str(e)}")
        traceback.print_exc()
        return False


import traceback
def init_db(db_path: str, drop=False):
    """Initialize Kùzu database and create schema with generational tracking."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
        print("Database connection established successfully")
        
        if drop:
            
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS Contains")
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS EvolvedFrom") 
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS Fact")
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS Groups")

        
        safe_kuzu_execute(
            conn,
            """
            CREATE NODE TABLE IF NOT EXISTS Fact(
              content STRING,
              path STRING,
              recorded_at STRING,
              PRIMARY KEY (content)
            );
            """,
            "Failed to create Fact table",
        )

        
        safe_kuzu_execute(
            conn,
            """
            CREATE NODE TABLE IF NOT EXISTS Groups(
              name STRING,
              metadata STRING,
              generation_created INT64,
              is_active BOOLEAN,
              PRIMARY KEY (name)
            );
            """,
            "Failed to create Groups table",
        )
        print("Groups table (with generation tracking) created or already exists.")
        
        
        safe_kuzu_execute(
            conn,
            "CREATE REL TABLE IF NOT EXISTS Contains(FROM Groups TO Fact);",
            "Failed to create Contains relationship table",
        )
        
        
        safe_kuzu_execute(
            conn,
            """
            CREATE REL TABLE IF NOT EXISTS EvolvedFrom(
                FROM Groups TO Groups,
                event_type STRING,
                generation INT64,
                reason STRING
            );
            """,
            "Failed to create EvolvedFrom relationship table",
        )
        print("EvolvedFrom relationship table created or already exists.")

        return conn
    except Exception as e:
        print(f"Fatal error initializing database: {str(e)}")
        traceback.print_exc()
        return None



def find_similar_groups(
    conn,
    fact: str,  
    model,  
    provider,
    npc =  None,
    context: str = None,
    **kwargs: Any
) -> List[str]:
    """Find existing groups that might contain this fact"""
    response = conn.execute(f"MATCH (g:Groups) RETURN g.name;")  
    
    
    
    groups = response.fetch_as_df()
    
    if not groups:
        return []

    prompt = """Given a fact and a list of groups, determine which groups this fact belongs to.
        A fact should belong to a group if it is semantically related to the group's theme or purpose.
        For example, if a fact is "The user loves programming" and there's a group called "Technical_Interests",
        that would be a match.

    Return a JSON object with the following structure:
        {
            "group_list": "a list containing the names of matching groups"
        }

    Return only the JSON object.
    Do not include any additional markdown formatting.
    """

    response = get_llm_response(
        prompt + f"\n\nFact: {fact}\nGroups: {json.dumps(groups)}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        **kwargs
    )
    response = response["response"]
    return response["group_list"]


def kg_initial(content,  
               model=None,
               provider=None,
               npc=None, 
               context='', 
               facts=None, 
               generation=None, 
               verbose=True,):

    if generation is None:
        CURRENT_GENERATION = 0
    else:
        CURRENT_GENERATION = generation
    
    print(f"--- Running KG Structuring Process (Generation: {CURRENT_GENERATION}) ---")

    if facts is None:
        if not content:
            raise ValueError("kg_initial requires either content_text or a list of facts.")
        print("  - Mode: Deriving new facts from text content...")
        all_facts = []
        print(len(content))
        if len(content)>10000:
            # randomly sub sample 10000 characters
            starting_point = random.randint(0, len(content)-10000)
        
            content_to_sample = content[starting_point:starting_point+10000]

            for n in range(len(content)//10000):
                print(n)
                print(starting_point)
                print(content_to_sample[0:1000])
                facts = get_facts(content_to_sample,
                                model=model, 
                                provider=provider, 
                                npc=npc, 
                                context=context)
                if verbose:
                    print(f"    - Extracted {len(facts)} facts from segment {n+1}")
                    print(facts)
                all_facts.extend(facts)
        else:
            print(content[0:1000]   )
            all_facts = get_facts(content, 
                                  model=model, 
                                  provider=provider, 
                                  npc=npc, 
                                  context=context)
            if verbose:
                print(f"    - Extracted {len(all_facts)} facts from content")
                print(all_facts)            
        for fact in all_facts:
            
            fact['generation'] = CURRENT_GENERATION
    else:
        print(f"  - Mode: Building structure from {len(facts)} pre-existing facts...")

    print("  - Inferring implied facts (zooming in)...")
    all_implied_facts = []
    if len(all_facts) > 20:
        # sub sample facts randomly to generate zoomed in facts
        sampled_facts = random.sample(all_facts, k=20)
        for n in range(len(all_facts) // 20):
            implied_facts = zoom_in(sampled_facts, 
                                    model=model, 
                                    provider=provider,
                                npc=npc, 
                                context=context)
            all_implied_facts.extend(implied_facts)
            if verbose:
                print(f"    - Inferred {len(implied_facts)} implied facts from sample {n+1}")
                print(implied_facts)
    else:
        implied_facts = zoom_in(all_facts, 
                                model=model, 
                                provider=provider,
                                npc=npc, 
                                context=context)
        print(implied_facts)

        all_implied_facts.extend(implied_facts)

        if verbose:
            print(f"    - Inferred {len(implied_facts)} implied facts from all facts")
            print(implied_facts)
    for fact in all_implied_facts:
        fact['generation'] = CURRENT_GENERATION

    all_facts = all_facts + all_implied_facts

    print("  - Generating concepts from all facts...")
    concepts = generate_groups(all_facts, 
                               model=model, 
                               provider=provider, 
                               npc=npc, 
                               context=context)
    for concept in concepts:
        concept['generation'] = CURRENT_GENERATION
    
    if verbose:
        print(f"    - Generated {len(concepts)} concepts")
        print(concepts)
    print("  - Linking facts to concepts...")
    fact_to_concept_links = defaultdict(list)
    concept_names = [c['name'] for c in concepts if c and 'name' in c]
    for fact in all_facts:

        fact_to_concept_links[fact['statement']] = get_related_concepts_multi(fact['statement'], "fact", concept_names, model, provider, npc, context)
        if verbose:
            print(fact_to_concept_links[fact['statement']])
    print("  - Linking facts to other facts...")
    fact_to_fact_links = []
    fact_statements = [f['statement'] for f in all_facts]
    for i, fact in enumerate(all_facts):
        other_fact_statements = fact_statements[all_facts != fact]
        print('checking fact: ', fact)
        if other_fact_statements:
            related_fact_stmts = get_related_facts_llm(fact['statement'], 
                                                       other_fact_statements, 
                                                       model=model, 
                                                       provider=provider, 
                                                       npc=npc, 
                                                       context=context)
            for related_stmt in related_fact_stmts:

                fact_to_fact_links.append((fact['statement'], related_stmt))
                if verbose:
                    print(fact['statement'], related_stmt)

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
                          model = None, 
                          provider=None, 
                          npc=None, 
                          context='', 
                          get_concepts=False,
                          link_concepts_facts = False, 
                          link_concepts_concepts=False, 
                          link_facts_facts = False, 
                          ):

    current_gen = existing_kg.get('generation', 0)
    next_gen = current_gen + 1
    print(f"\n--- ABSORBING INFO: Gen {current_gen} -> Gen {next_gen} ---")

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
    print(npc, npc.model, npc.provider)
    
    if new_facts:
        all_new_facts = new_facts
        print(f'using pre-approved facts: {len(all_new_facts)}')
    elif new_content_text:
        print('extracting facts from content...')
        if len(new_content_text) > 10000:
            starting_point = random.randint(0, len(new_content_text)-10000)
            for n in range(len(new_content_text)//10000):
                content_to_sample = new_content_text[n*10000:(n+1)*10000]
                facts = get_facts(content_to_sample, 
                                model=model,
                                provider=provider,
                                npc = npc, 
                                context=context)
                all_new_facts.extend(facts)
                print(facts)
        else:
            all_new_facts = get_facts(new_content_text, 
                                model=model,
                                provider=provider,
                                npc = npc, 
                                context=context)
            print(all_new_facts)
    else:
        print("No new content or facts provided")
        return existing_kg, {}

    for fact in all_new_facts: 
        fact['generation'] = next_gen

    final_facts = existing_facts + all_new_facts

    if get_concepts:
        print('generating groups...')

        candidate_concepts = generate_groups(all_new_facts, 
                                            model = model, 
                                            provider = provider, 
                                            npc=npc, 
                                            context=context)
        print(candidate_concepts)
        print('checking group uniqueness')
        for cand_concept in candidate_concepts:
            cand_name = cand_concept['name']
            if cand_name in existing_concept_names: 
                continue
            cand_concept['generation'] = next_gen
            newly_added_concepts.append(cand_concept)
            if link_concepts_concepts:
                print('linking concepts and concepts...')

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
            print('linking facts and concepts...')
            for fact in all_new_facts:
                fact_to_concept_links[fact['statement']] = get_related_concepts_multi(fact['statement'], 
                                                                                    "fact", 
                                                                                    all_concept_names, 
                                                                                    model = model,
                                                                                    provider=provider,
                                                                                    npc = npc, 
                                                                                    context= context)
    else:
        final_concepts = existing_concepts
    if link_facts_facts:
        print('linking facts and facts...')

        for new_fact in all_new_facts:
            related_fact_stmts = get_related_facts_llm(new_fact['statement'], 
                                                       existing_fact_statements, 
                                                       model = model,
                                                       provider = provider,
                                                       npc = npc, 
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
                     operations_config=None):
    current_gen = existing_kg.get('generation', 0)
    next_gen = current_gen + 1
    print(f"\n--- SLEEPING (Evolving Knowledge): Gen {current_gen} -> Gen {next_gen} ---")

    
    facts_map = {f['statement']: f for f in existing_kg.get('facts', [])}
    concepts_map = {c['name']: c for c in existing_kg.get('concepts', [])}
    fact_links = defaultdict(list, {k: list(v) for k, v in existing_kg.get('fact_to_concept_links', {}).items()})
    concept_links = set(tuple(sorted(link)) for link in existing_kg.get('concept_links', []))
    fact_to_fact_links = set(tuple(sorted(link)) for link in existing_kg.get('fact_to_fact_links', []))

    
    print("  - Phase 1: Checking for unstructured facts...")
    facts_with_concepts = set(fact_links.keys())
    orphaned_fact_statements = list(set(facts_map.keys()) - facts_with_concepts)

    if len(orphaned_fact_statements) > 20:
        print(f"    - Found {len(orphaned_fact_statements)} orphaned facts. Applying full KG structuring process...")
        orphaned_facts_as_dicts = [facts_map[s] for s in orphaned_fact_statements]
        
        
        new_structure = kg_initial(
            facts=orphaned_facts_as_dicts,
            model=model,
            provider=provider,
            npc=npc,
            context=context,
            generation=next_gen
        )

        
        print("    - Merging new structure into main KG...")
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
        print("    - Knowledge graph is sufficiently structured. Proceeding to refinement.")

    
    if operations_config is None:
        possible_ops = ['prune', 'deepen', 'abstract_link']
        ops_to_run = random.sample(possible_ops, k=random.randint(1, 2))
    else:
        ops_to_run = operations_config
    
    print(f"  - Phase 2: Executing refinement operations: {ops_to_run}")

    for op in ops_to_run:
        
        if op == 'prune' and (len(facts_map) > 10 or len(concepts_map) > 5):
            print("    - Running 'prune' operation using consolidate_facts_llm...")
            fact_to_check = random.choice(list(facts_map.values()))
            other_facts = [f for f in facts_map.values() if f['statement'] != fact_to_check['statement']]
            consolidation_result = consolidate_facts_llm(fact_to_check, other_facts, model, provider, npc, context)
            if consolidation_result.get('decision') == 'redundant':
                print(f"      - Pruning redundant fact: '{fact_to_check['statement'][:80]}...'")
                del facts_map[fact_to_check['statement']]

        
        elif op == 'deepen' and facts_map:
            print("    - Running 'deepen' operation using zoom_in...")
            fact_to_deepen = random.choice(list(facts_map.values()))
            implied_facts = zoom_in([fact_to_deepen], model, provider, npc, context)
            new_fact_count = 0
            for fact in implied_facts:
                if fact['statement'] not in facts_map:
                    fact.update({'generation': next_gen, 'origin': 'deepen'})
                    facts_map[fact['statement']] = fact
                    new_fact_count += 1
            if new_fact_count > 0: print(f"      - Inferred {new_fact_count} new fact(s).")
        
        else:
            print(f"    - SKIPPED: Operation '{op}' did not run (conditions not met).")
        
    
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
                     model = None,
                     provider = None,
                     npc=None,
                     context='', 
                     num_seeds=3):
    current_gen = existing_kg.get('generation', 0)
    next_gen = current_gen + 1
    print(f"\n--- DREAMING (Creative Synthesis): Gen {current_gen} -> Gen {next_gen} ---")
    concepts = existing_kg.get('concepts', [])
    if len(concepts) < num_seeds:
        print(f"  - Not enough concepts ({len(concepts)}) for dream. Skipping.")
        return existing_kg, {}
    seed_concepts = random.sample(concepts, k=num_seeds)
    seed_names = [c['name'] for c in seed_concepts]
    print(f"  - Dream seeded with: {seed_names}")
    prompt = f"""
    Write a short, speculative paragraph (a 'dream') that plausibly connects the concepts of {json.dumps(seed_names)}.
    Invent a brief narrative or a hypothetical situation.
    Respond with JSON: {{"dream_text": "A short paragraph..."}}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, npc = npc,  
                                format="json", context=context)
    dream_text = response['response'].get('dream_text')
    if not dream_text:
        print("  - Failed to generate a dream narrative. Skipping.")
        return existing_kg, {}
    print(f"  - Generated Dream: '{dream_text[:150]}...'")
    
    dream_kg, _ = kg_evolve_incremental(existing_kg, new_content_text=dream_text, model=model, provider=provider, npc=npc, context=context)
    
    original_fact_stmts = {f['statement'] for f in existing_kg['facts']}
    for fact in dream_kg['facts']:
        if fact['statement'] not in original_fact_stmts: fact['origin'] = 'dream'
    original_concept_names = {c['name'] for c in existing_kg['concepts']}
    for concept in dream_kg['concepts']:
        if concept['name'] not in original_concept_names: concept['origin'] = 'dream'
    print("  - Dream analysis complete. New knowledge integrated.")
    return dream_kg, {}


def save_kg_with_pandas(kg, path_prefix="kg_state"):

    generation = kg.get("generation", 0)

    nodes_data = []
    for fact in kg.get('facts', []): nodes_data.append({'id': fact['statement'], 'type': 'fact', 'generation': fact.get('generation')})
    for concept in kg.get('concepts', []): nodes_data.append({'id': concept['name'], 'type': 'concept', 'generation': concept.get('generation')})
    pd.DataFrame(nodes_data).to_csv(f'{path_prefix}_gen{generation}_nodes.csv', index=False)
    
    links_data = []
    for fact_stmt, concepts in kg.get("fact_to_concept_links", {}).items():
        for concept_name in concepts: links_data.append({'source': fact_stmt, 'target': concept_name, 'type': 'fact_to_concept'})
    for c1, c2 in kg.get("concept_links", []): 
        links_data.append({'source': c1, 'target': c2, 'type': 'concept_to_concept'})

    for f1, f2 in kg.get("fact_to_fact_links", []): 
        links_data.append({'source': f1, 'target': f2, 'type': 'fact_to_fact'})
    pd.DataFrame(links_data).to_csv(f'{path_prefix}_gen{generation}_links.csv', index=False)
    print(f"Saved KG Generation {generation} to CSV files.")


def save_changelog_to_json(changelog, from_gen, to_gen, path_prefix="changelog"):
    if not changelog: return
    with open(f"{path_prefix}_gen{from_gen}_to_{to_gen}.json", 'w', encoding='utf-8') as f:
        json.dump(changelog, f, indent=4)
    print(f"Saved changelog for Gen {from_gen}->{to_gen}.")




def store_fact_and_group(conn, fact: str,
                        groups: List[str], path: str) -> bool:
    """Insert a fact into the database along with its groups"""
    if not conn:
        print("store_fact_and_group: Database connection is None")
        return False
    
    print(f"store_fact_and_group: Storing fact: {fact}, with groups:"
          f" {groups}") 
    try:
        
        insert_success = insert_fact(conn, fact, path) 
        if not insert_success:
            print(f"store_fact_and_group: Failed to insert fact: {fact}")
            return False
        
        
        for group in groups:
            assign_success = assign_fact_to_group_graph(conn, fact, group)
            if not assign_success:
                print(f"store_fact_and_group: Failed to assign fact"
                      f" {fact} to group {group}")
                return False
        
        return True
    except Exception as e:
        print(f"store_fact_and_group: Error storing fact and group: {e}")
        traceback.print_exc()
        return False
def insert_fact(conn, fact: str, path: str) -> bool:
    """Insert a fact into the database with robust error handling"""
    if conn is None:
        print("insert_fact: Cannot insert fact:"
              " database connection is None")
        return False
    try:
        
        escaped_fact = fact.replace('"', '\\"')
        escaped_path = os.path.expanduser(path).replace('"', '\\"')

        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"insert_fact: Attempting to insert fact: {fact}") 

        
        safe_kuzu_execute(conn, "BEGIN TRANSACTION")

        
        check_query = f"""
        MATCH (f:Fact {{content: "{escaped_fact}"}})
        RETURN f
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "insert_fact: Failed to check if fact exists"
        )
        if error:
            safe_kuzu_execute(conn, "ROLLBACK")
            print(f"insert_fact: Error checking if fact exists: {error}")
            return False

        
        if not result.has_next():
            insert_query = f"""
            CREATE (f:Fact {{
                content: "{escaped_fact}",
                path: "{escaped_path}",
                recorded_at: "{timestamp}"
            }})
            """

            result, error = safe_kuzu_execute(
                conn, insert_query, "insert_fact: Failed to insert fact"
            )
            if error:
                safe_kuzu_execute(conn, "ROLLBACK")
                print(f"insert_fact: Error inserting fact: {error}")
                return False

        
        safe_kuzu_execute(conn, "COMMIT")
        print(f"insert_fact: Successfully inserted/found fact: {fact}")
        return True
    except Exception as e:
        print(f"insert_fact: Error inserting fact: {str(e)}")
        traceback.print_exc()
        safe_kuzu_execute(conn, "ROLLBACK")
        return False

def assign_fact_to_group_graph(conn, fact: str, group: str) -> bool:
    """Create a relationship between a fact and a group with robust
       error handling"""
    if conn is None:
        print("assign_fact_to_group_graph: Cannot assign fact to group:"
              " database connection is None")
        return False

    try:
        
        escaped_fact = fact.replace('"', '\\"')
        escaped_group = group.replace('"', '\\"')

        print(f"assign_fact_to_group_graph: Assigning fact: {fact} to group:"
              f" {group}") 

        
        check_query = f"""
        MATCH (f:Fact {{content: "{escaped_fact}"}})
        RETURN f
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "assign_fact_to_group_graph: Failed to check"
                               " if fact exists"
        )
        if error or not result.has_next():
            print(f"assign_fact_to_group_graph: Fact not found: {fact}")
            return False

        check_query = f"""
        MATCH (g:Groups {{name: "{escaped_group}"}})
        RETURN g
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "assign_fact_to_group_graph: Failed to check"
                               " if group exists"
        )
        if error or not result.has_next():
            print(f"assign_fact_to_group_graph: Group not found: {group}")
            return False

        
        query = f"""
        MATCH (f:Fact), (g:Groups)
        WHERE f.content = "{escaped_fact}" AND g.name = "{escaped_group}"
        CREATE (g)-[:Contains]->(f)
        """

        result, error = safe_kuzu_execute(
            conn, query, "assign_fact_to_group_graph: Failed to create"
                         " relationship: {error}"
        )
        if error:
            print(f"assign_fact_to_group_graph: Failed to create"
                  f" relationship: {error}")
            return False

        print(f"assign_fact_to_group_graph: Assigned fact to group:"
              f" {group}")
        return True
    except Exception as e:
        print(f"assign_fact_to_group_graph: Error assigning fact to group:"
              f" {str(e)}")
        traceback.print_exc()
        return False


def store_fact_and_group(conn, fact: str, groups: List[str], path: str) -> bool:
    """Insert a fact into the database along with its groups"""
    if not conn:
        print("store_fact_and_group: Database connection is None")
        return False
    
    print(f"store_fact_and_group: Storing fact: {fact}, with groups: {groups}") 
    try:
        
        insert_success = insert_fact(conn, fact, path) 
        if not insert_success:
            print(f"store_fact_and_group: Failed to insert fact: {fact}") 
            return False
        
        
        for group in groups:
            assign_success = assign_fact_to_group_graph(conn, fact, group)
            if not assign_success:
                print(f"store_fact_and_group: Failed to assign fact {fact} to group {group}") 
                return False
        
        return True
    except Exception as e:
        print(f"store_fact_and_group: Error storing fact and group: {e}")
        traceback.print_exc()
        return False
    
        

def safe_kuzu_execute(conn, query, error_message="Kuzu query failed"):
    """Execute a Kuzu query with proper error handling"""
    try:
        result = conn.execute(query)
        return result, None
    except Exception as e:
        error = f"{error_message}: {str(e)}"
        print(error)
        return None, error

def process_text_with_chroma(
    kuzu_db_path: str,
    chroma_db_path: str,
    text: str,
    path: str,
    model: str ,
    provider: str ,
    embedding_model: str ,
    embedding_provider: str ,
    npc = None,
    batch_size: int = 5,
):
    """Process text and store facts in both Kuzu and Chroma DB

    Args:
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        text: Input text to process
        path: Source path or identifier
        model: LLM model to use
        provider: LLM provider
        embedding_model: Model to use for embeddings
        npc: Optional NPC instance
        batch_size: Batch size for processing

    Returns:
        List of extracted facts
    """
    
    kuzu_conn = init_db(kuzu_db_path, drop=False)
    chroma_client, chroma_collection = setup_chroma_db( 
        "knowledge_graph",
        "Facts extracted from various sources",
        chroma_db_path
    )

    
    facts = get_facts(text, model=model, provider=provider, npc=npc)

    
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

        
        from npcpy.llm_funcs import get_embeddings

        batch_embeddings = get_embeddings(
            batch,
        )

        for j, fact in enumerate(batch):
            print(f"Processing fact: {fact}")
            embedding = batch_embeddings[j]

            
            similar_facts = find_similar_facts_chroma(
                chroma_collection, fact, query_embedding=embedding, n_results=3
            )

            if similar_facts:
                print(f"Similar facts found:")
                for result in similar_facts:
                    print(f"  - {result['fact']} (distance: {result['distance']})")
                

            
            metadata = {
                "path": path,
                "timestamp": datetime.now().isoformat(),
                "source_model": model,
                "source_provider": provider,
            }

            
            kuzu_success = insert_fact(kuzu_conn, fact, path)

            
            if kuzu_success:
                chroma_id = store_fact_with_embedding(
                    chroma_collection, fact, metadata, embedding
                )
                if chroma_id:
                    print(f"Successfully saved fact with ID: {chroma_id}")
                else:
                    print(f"Failed to save fact to Chroma")
            else:
                print(f"Failed to save fact to Kuzu graph")

    
    kuzu_conn.close()

    return facts


def hybrid_search_with_chroma(
    kuzu_conn,
    chroma_collection,
    query: str,
    group_filter: Optional[List[str]] = None,
    top_k: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Perform hybrid search using both Chroma vector search and Kuzu graph relationships

    Args:
        kuzu_conn: Connection to Kuzu graph database
        chroma_collection: Chroma collection for vector search
        query: Search query text
        group_filter: Optional list of groups to filter by in graph
        top_k: Number of results to return
        metadata_filter: Optional metadata filter for Chroma search
        embedding_model: Model to use for embeddings
        provider: Provider for embeddings

    Returns:
        List of dictionaries with combined results
    """
    
    from npcpy.llm_funcs import get_embeddings

    query_embedding = get_embeddings([query])[0]

    
    vector_results = find_similar_facts_chroma(
        chroma_collection,
        query,
        query_embedding=query_embedding,
        n_results=top_k,
        metadata_filter=metadata_filter,
    )

    
    vector_facts = [result["fact"] for result in vector_results]

    
    expanded_results = []

    
    for result in vector_results:
        expanded_results.append(
            {
                "fact": result["fact"],
                "source": "vector_search",
                "relevance": "direct_match",
                "distance": result["distance"],
                "metadata": result["metadata"],
            }
        )

    
    for fact in vector_facts:
        try:
            
            escaped_fact = fact.replace('"', '\\"')

            
            group_result = kuzu_conn.execute(
                f"""
                MATCH (g:Groups)-[:Contains]->(f:Fact)
                WHERE f.content = "{escaped_fact}"
                RETURN g.name
                """
            ).get_as_df()

            
            fact_groups = [row["g.name"] for _, row in group_result.iterrows()]

            
            if group_filter:
                fact_groups = [g for g in fact_groups if g in group_filter]

            
            for group in fact_groups:
                escaped_group = group.replace('"', '\\"')

                
                related_facts_result = kuzu_conn.execute(
                    f"""
                    MATCH (g:Groups)-[:Contains]->(f:Fact)
                    WHERE g.name = "{escaped_group}" AND f.content <> "{escaped_fact}"
                    RETURN f.content, f.path, f.recorded_at
                    LIMIT 5
                    """
                ).get_as_df()

                
                for _, row in related_facts_result.iterrows():
                    related_fact = {
                        "fact": row["f.content"],
                        "source": f"graph_relation_via_{group}",
                        "relevance": "group_related",
                        "path": row["f.path"],
                        "recorded_at": row["f.recorded_at"],
                    }

                    
                    if not any(
                        r.get("fact") == related_fact["fact"] for r in expanded_results
                    ):
                        expanded_results.append(related_fact)

        except Exception as e:
            print(f"Error expanding results via graph: {e}")

    
    return expanded_results[:top_k]


def find_similar_facts_chroma(
    collection,
    query: str,
    query_embedding: List[float],
    n_results: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Find facts similar to the query using pre-generated embedding

    Args:
        collection: Chroma collection
        query: Query text (for reference only)
        query_embedding: Pre-generated embedding from get_embeddings
        n_results: Number of results to return
        metadata_filter: Optional filter for metadata fields

    Returns:
        List of dictionaries with results
    """
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
        print(f"Error searching in Chroma: {e}")
        return []



def store_fact_with_embedding(
    collection, fact: str, metadata: dict, embedding: List[float]
) -> str:
    """Store a fact with its pre-generated embedding in Chroma DB

    Args:
        collection: Chroma collection
        fact: The fact text
        metadata: Dictionary with metadata (path, source, timestamp, etc.)
        embedding: Pre-generated embedding vector from get_embeddings

    Returns:
        ID of the stored fact
    """
    try:
        
        import hashlib

        fact_id = hashlib.md5(fact.encode()).hexdigest()

        
        collection.add(
            documents=[fact],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[fact_id],
        )

        return fact_id
    except Exception as e:
        print(f"Error storing fact in Chroma: {e}")
        return None

def save_facts_to_graph_db(
    conn, facts: List[str], path: str, batch_size: int
):
    """Save a list of facts to the database in batches"""
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

        
        for fact in batch:
            try:
                print(f"Inserting fact: {fact}")
                print(f"With path: {path}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"With recorded_at: {timestamp}")

                insert_fact(conn, fact, path)
                print("Success!")
            except Exception as e:
                print(f"Failed to insert fact: {fact}")
                print(f"Error: {e}")
                continue

        print(f"Completed batch {i//batch_size + 1}")



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
   provider=None
):
   """Search facts in the knowledge graph"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'
   
   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
   
   matching_facts = []
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
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'
   
   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
   
   original_count = len(kg_data.get('facts', []))
   kg_data['facts'] = [f for f in kg_data.get('facts', []) if f['statement'] != fact_text]
   removed_count = original_count - len(kg_data['facts'])
   
   if removed_count > 0:
       save_kg_to_db(engine, kg_data, team_name, npc_name, directory_path)
       return f"Removed {removed_count} matching fact(s)"
   
   return "No matching facts found"

def kg_list_concepts(
   engine,
   npc=None, 
   team=None,
   model=None,
   provider=None
):
   """List all concepts in the knowledge graph"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'
   
   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
   
   concepts = [c['name'] for c in kg_data.get('concepts', [])]
   return concepts

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
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'
   
   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
   
   original_count = len(kg_data.get('concepts', []))
   kg_data['concepts'] = [c for c in kg_data.get('concepts', []) if c['name'] != concept_name]
   removed_count = original_count - len(kg_data['concepts'])
   
   if removed_count > 0:
       save_kg_to_db(engine, kg_data, team_name, npc_name, directory_path)
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
   provider=None
):
   """Get all facts from the knowledge graph"""
   directory_path = os.getcwd()
   team_name = getattr(team, 'name', 'default_team') if team else 'default_team'
   npc_name = npc.name if npc else 'default_npc'
   
   kg_data = load_kg_from_db(engine, team_name, npc_name, directory_path)
   
   facts = [f['statement'] for f in kg_data.get('facts', [])]
   return facts

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