"""
Sememolution: Population-based Knowledge Graph Evolution

Implements the population-based evolution layer from the Sememolution paper.
Each individual in the population IS a full knowledge graph that goes through
the waking/sleeping/dreaming lifecycle independently. Stochastic LLM responses
and randomized sleep/dream configurations produce structural diversity across
the population — this IS the mutation. Selection happens through response
ranking: sample a subset, each searches its own graph, each generates a
response, rank the responses, winners propagate.

Uses GeneticEvolver from npcpy.ft.ge as the GA engine.
Uses lifecycle functions from npcpy.memory.knowledge_graph for the KG operations.
Poisson-sampled lambdas control graph traversal depth and breadth at search time.
"""

import os
import json
import random
import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from sqlalchemy import text

from npcpy.ft.ge import GeneticEvolver, GAConfig


@dataclass
class KGGenome:
    """The evolvable parameters of a KG individual.

    lambda_depth and lambda_breadth are the Poisson rate parameters
    for graph traversal at search time. Every search samples:
        actual_depth ~ Poisson(lambda_depth)
        actual_breadth ~ Poisson(lambda_breadth)
    so the same individual produces different traversals each time.

    The sleep/dream parameters control how the graph STRUCTURE evolves
    during the consolidation lifecycle phase. Different configs produce
    different topologies — some dense and shallow, others deep and sparse,
    others speculative and cross-domain.
    """
    lambda_depth: float = 2.0
    lambda_breadth: float = 5.0
    similarity_threshold: float = 0.6
    sleep_ops: List[str] = field(default_factory=lambda: ['prune', 'deepen'])
    dream_seeds: int = 3
    dream_probability: float = 0.3
    get_concepts: bool = True
    link_facts_facts: bool = True
    link_concepts_facts: bool = True
    link_concepts_concepts: bool = False

    def sample_depth(self) -> int:
        """Sample traversal depth from Poisson(lambda_depth)."""
        return max(1, np.random.poisson(self.lambda_depth))

    def sample_breadth(self) -> int:
        """Sample traversal breadth from Poisson(lambda_breadth)."""
        return max(1, np.random.poisson(self.lambda_breadth))


@dataclass
class KGIndividual:
    """A single KG in the population. Contains the genome and the graph."""
    individual_id: str
    genome: KGGenome
    kg_data: Dict = field(default_factory=lambda: {
        'generation': 0, 'facts': [], 'concepts': [],
        'concept_links': [], 'fact_to_concept_links': {},
        'fact_to_fact_links': [],
    })
    fitness: float = 0.0
    wins: int = 0
    total_queries: int = 0


def initialize_individual() -> KGIndividual:
    """Create a random KG individual with empty graph and random genome."""
    genome = KGGenome(
        lambda_depth=max(0.5, np.random.exponential(2.0)),
        lambda_breadth=max(1.0, np.random.exponential(5.0)),
        similarity_threshold=np.random.uniform(0.3, 0.8),
        sleep_ops=random.sample(
            ['prune', 'deepen', 'abstract_link', 'link_facts'],
            k=random.randint(1, 3),
        ),
        dream_seeds=random.randint(2, 6),
        dream_probability=np.random.uniform(0.05, 0.8),
        get_concepts=random.choice([True, False]),
        link_facts_facts=random.choice([True, False]),
        link_concepts_facts=random.choice([True, False]),
        link_concepts_concepts=random.choice([True, False]),
    )
    return KGIndividual(
        individual_id=f"kg_{int(time.time())}_{random.randint(1000,9999)}",
        genome=genome,
    )


def mutate_individual(ind: KGIndividual) -> KGIndividual:
    """Mutate an individual's genome. The graph carries over — only the
    genome changes, which affects future evolution and search behavior."""
    import copy
    new = copy.deepcopy(ind)
    new.individual_id = f"kg_{int(time.time())}_{random.randint(1000,9999)}"
    new.fitness = 0.0
    new.wins = 0
    new.total_queries = 0
    g = new.genome

    # Apply 1-3 random mutations
    mutations = [
        lambda: setattr(g, 'lambda_depth', max(0.5, g.lambda_depth + np.random.normal(0, 0.5))),
        lambda: setattr(g, 'lambda_breadth', max(1.0, g.lambda_breadth + np.random.normal(0, 1.0))),
        lambda: setattr(g, 'similarity_threshold', np.clip(g.similarity_threshold + np.random.normal(0, 0.05), 0.3, 0.9)),
        lambda: setattr(g, 'sleep_ops', random.sample(['prune', 'deepen', 'abstract_link', 'link_facts'], k=random.randint(1, 3))),
        lambda: setattr(g, 'dream_seeds', max(2, min(6, g.dream_seeds + random.randint(-1, 1)))),
        lambda: setattr(g, 'dream_probability', np.clip(g.dream_probability + np.random.normal(0, 0.1), 0.05, 0.9)),
        lambda: setattr(g, 'link_facts_facts', not g.link_facts_facts),
        lambda: setattr(g, 'link_concepts_concepts', not g.link_concepts_concepts),
        lambda: setattr(g, 'get_concepts', not g.get_concepts),
        lambda: setattr(g, 'link_concepts_facts', not g.link_concepts_facts),
    ]
    for mut in random.sample(mutations, k=random.randint(1, 3)):
        mut()

    return new


def crossover_individuals(a: KGIndividual, b: KGIndividual) -> KGIndividual:
    """Cross two individuals. Takes genome parameters from each parent
    randomly, and merges their graph facts (union of both knowledge bases)."""
    import copy

    child_genome = KGGenome(
        lambda_depth=random.choice([a.genome.lambda_depth, b.genome.lambda_depth]),
        lambda_breadth=random.choice([a.genome.lambda_breadth, b.genome.lambda_breadth]),
        similarity_threshold=random.choice([a.genome.similarity_threshold, b.genome.similarity_threshold]),
        sleep_ops=random.choice([a.genome.sleep_ops, b.genome.sleep_ops]),
        dream_seeds=random.choice([a.genome.dream_seeds, b.genome.dream_seeds]),
        dream_probability=random.choice([a.genome.dream_probability, b.genome.dream_probability]),
        get_concepts=random.choice([a.genome.get_concepts, b.genome.get_concepts]),
        link_facts_facts=random.choice([a.genome.link_facts_facts, b.genome.link_facts_facts]),
        link_concepts_facts=random.choice([a.genome.link_concepts_facts, b.genome.link_concepts_facts]),
        link_concepts_concepts=random.choice([a.genome.link_concepts_concepts, b.genome.link_concepts_concepts]),
    )

    # Merge graphs — union of facts from both parents, deduplicated
    a_facts = {f.get('statement', str(f)): f for f in a.kg_data.get('facts', [])}
    b_facts = {f.get('statement', str(f)): f for f in b.kg_data.get('facts', [])}
    merged_facts = list({**a_facts, **b_facts}.values())

    a_concepts = {c.get('name', str(c)): c for c in a.kg_data.get('concepts', [])}
    b_concepts = {c.get('name', str(c)): c for c in b.kg_data.get('concepts', [])}
    merged_concepts = list({**a_concepts, **b_concepts}.values())

    child_kg = {
        'generation': max(a.kg_data.get('generation', 0), b.kg_data.get('generation', 0)),
        'facts': merged_facts,
        'concepts': merged_concepts,
        'concept_links': list(set(
            tuple(l) for l in a.kg_data.get('concept_links', []) + b.kg_data.get('concept_links', [])
        )),
        'fact_to_concept_links': {**a.kg_data.get('fact_to_concept_links', {}),
                                   **b.kg_data.get('fact_to_concept_links', {})},
        'fact_to_fact_links': list(set(
            tuple(l) for l in a.kg_data.get('fact_to_fact_links', []) + b.kg_data.get('fact_to_fact_links', [])
        )),
    }

    return KGIndividual(
        individual_id=f"kg_{int(time.time())}_{random.randint(1000,9999)}",
        genome=child_genome,
        kg_data=child_kg,
    )


class SememolutionPopulation:
    """Population manager for Sememolution.

    Wraps GeneticEvolver with KG-specific fitness, mutation, crossover.
    Manages the lifecycle (assimilate, sleep, dream) for each individual.
    Handles query-time sampling, search, response generation, and ranking.
    """

    def __init__(self, engine=None, model: str = "gemma3:4b",
                 provider: str = "ollama", population_size: int = 100,
                 sample_size: int = 10):
        self.engine = engine
        self.model = model
        self.provider = provider
        self.sample_size = sample_size

        self.ga = GeneticEvolver(
            fitness_fn=lambda ind: ind.fitness,
            mutate_fn=mutate_individual,
            crossover_fn=crossover_individuals,
            initialize_fn=initialize_individual,
            config=GAConfig(
                population_size=population_size,
                mutation_rate=0.15,
                crossover_rate=0.7,
                tournament_size=3,
                elitism_count=max(2, population_size // 10),
            ),
        )

    def initialize(self):
        """Create initial population."""
        self.ga.initialize_population()
        return self.ga.population

    def assimilate_text(self, text: str):
        """Feed new text to every individual in the population.
        Each individual assimilates it according to its own genome,
        producing structural diversity through stochastic LLM responses
        and different linking configurations."""
        from npcpy.memory.knowledge_graph import kg_evolve_incremental

        for ind in self.ga.population:
            ind.kg_data, _ = kg_evolve_incremental(
                ind.kg_data,
                new_content_text=text,
                model=self.model,
                provider=self.provider,
                get_concepts=ind.genome.get_concepts,
                link_facts_facts=ind.genome.link_facts_facts,
                link_concepts_facts=ind.genome.link_concepts_facts,
                link_concepts_concepts=ind.genome.link_concepts_concepts,
            )

    def sleep_cycle(self):
        """Run sleep/dream on every individual using its own genome config.
        This is where structural diversity emerges — different sleep ops
        produce different graph topologies."""
        from npcpy.memory.knowledge_graph import kg_sleep_process, kg_dream_process

        for ind in self.ga.population:
            ind.kg_data, _ = kg_sleep_process(
                ind.kg_data,
                model=self.model,
                provider=self.provider,
                operations_config=ind.genome.sleep_ops,
            )

            if random.random() < ind.genome.dream_probability:
                ind.kg_data, _ = kg_dream_process(
                    ind.kg_data,
                    model=self.model,
                    provider=self.provider,
                    num_seeds=ind.genome.dream_seeds,
                )

    def search_individual(self, ind: KGIndividual, query: str) -> List[str]:
        """Search an individual's graph with Poisson-sampled depth/breadth."""
        depth = ind.genome.sample_depth()
        breadth = ind.genome.sample_breadth()
        threshold = ind.genome.similarity_threshold

        facts = ind.kg_data.get('facts', [])
        if not facts:
            return []

        # Simple keyword + threshold search over this individual's facts
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []
        for f in facts:
            stmt = f.get('statement', str(f)).lower()
            overlap = len(query_words & set(stmt.split()))
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Breadth controls how many results we take
        results = scored[:breadth]

        # Depth controls how many hops we follow through links
        if depth > 1 and results:
            ftc = ind.kg_data.get('fact_to_concept_links', {})
            ftf = ind.kg_data.get('fact_to_fact_links', [])
            seen = set(f.get('statement', '') for _, f in results)

            for hop in range(depth - 1):
                new_facts = []
                for _, f in results:
                    stmt = f.get('statement', '')
                    # Follow fact-to-fact links
                    for link in ftf:
                        if isinstance(link, (list, tuple)) and len(link) == 2:
                            other = link[1] if link[0] == stmt else (link[0] if link[1] == stmt else None)
                            if other and other not in seen:
                                seen.add(other)
                                match = next((ff for ff in facts if ff.get('statement', '') == other), None)
                                if match:
                                    new_facts.append((0.3, match))
                results.extend(new_facts[:breadth])

        return [f.get('statement', str(f)) for _, f in results[:breadth]]

    def query_and_rank(self, query: str) -> List[Dict]:
        """Core sememolution query loop:
        1. Sample individuals from population
        2. Each searches its own graph (Poisson-sampled traversal)
        3. Each generates a response using its retrieved context
        4. Rank responses
        5. Update fitness
        """
        from npcpy.llm_funcs import get_llm_response

        sample = random.sample(
            self.ga.population,
            min(self.sample_size, len(self.ga.population)),
        )

        candidates = []
        for ind in sample:
            context_facts = self.search_individual(ind, query)

            context_str = None
            if context_facts:
                facts_block = "\n".join(f"        - {f}" for f in context_facts)
                context_str = f"""Relevant knowledge from memory:
{facts_block}"""

            try:
                resp = get_llm_response(
                    query, model=self.model, provider=self.provider,
                    context=context_str,
                )
                response_text = resp.get('response', '')
                if isinstance(response_text, dict):
                    response_text = json.dumps(response_text)
            except Exception as e:
                response_text = f"Error: {e}"

            candidates.append({
                'individual': ind,
                'context_facts': context_facts,
                'response': str(response_text),
                'n_facts': len(context_facts),
            })

        # Rank responses
        rankings = self._rank_responses(query, candidates)

        # Update fitness
        for i, c in enumerate(rankings):
            ind = c['individual']
            ind.total_queries += 1
            if i < 3:
                ind.wins += 1
            ind.fitness = ind.wins / max(1, ind.total_queries)
            c['rank'] = i + 1

        return rankings

    def _rank_responses(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rank candidate responses using LLM judge."""
        from npcpy.llm_funcs import get_llm_response

        if len(candidates) <= 1:
            return candidates

        summaries = []
        for i, c in enumerate(candidates):
            preview = str(c['response'])[:200]
            summaries.append(f"        Response {i}: ({c['n_facts']} facts) {preview}")

        summaries_block = "\n\n".join(summaries)
        prompt = f"""Rank these {len(candidates)} responses to: "{query}"

{summaries_block}

        Rank best to worst by accuracy, specificity, and usefulness. Respond with JSON: {{"ranking": [0, 2, 1, ...]}} where numbers are Response indices best to worst."""

        try:
            resp = get_llm_response(
                prompt, model=self.model, provider=self.provider,
                format="json",
            )
            ranking = resp.get('response', {}).get('ranking', [])
            if isinstance(ranking, list) and len(ranking) == len(candidates):
                ordered = []
                for idx in ranking:
                    if 0 <= idx < len(candidates):
                        ordered.append(candidates[idx])
                if len(ordered) == len(candidates):
                    return ordered
        except Exception:
            pass

        # Fallback: rank by number of facts used
        candidates.sort(key=lambda c: c['n_facts'], reverse=True)
        return candidates

    def evolve_generation(self):
        """Run one GA generation: evaluate fitness, select, crossover, mutate."""
        return self.ga.evolve_generation()

    def get_stats(self) -> Dict:
        """Population summary."""
        pop = self.ga.population
        if not pop:
            return {'total': 0}

        fitnesses = [ind.fitness for ind in pop]
        depths = [ind.genome.lambda_depth for ind in pop]
        breadths = [ind.genome.lambda_breadth for ind in pop]
        fact_counts = [len(ind.kg_data.get('facts', [])) for ind in pop]

        return {
            'total': len(pop),
            'avg_fitness': float(np.mean(fitnesses)),
            'max_fitness': float(max(fitnesses)),
            'total_facts': int(sum(fact_counts)),
            'avg_facts_per_individual': float(np.mean(fact_counts)),
            'lambda_depth': {'mean': float(np.mean(depths)), 'std': float(np.std(depths))},
            'lambda_breadth': {'mean': float(np.mean(breadths)), 'std': float(np.std(breadths))},
            'unique_sleep_configs': len(set(tuple(ind.genome.sleep_ops) for ind in pop)),
        }


# ---------------------------------------------------------------------------
# Persistence: SQLite-backed population store. No command_history schema churn.
# ---------------------------------------------------------------------------

def _ensure_population_schema(engine):
    """Create the kg_populations + kg_individuals tables if missing."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS kg_populations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model TEXT,
                provider TEXT,
                sample_size INTEGER NOT NULL DEFAULT 10,
                config_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS kg_individuals (
                individual_id TEXT NOT NULL,
                population_id TEXT NOT NULL,
                genome_json TEXT NOT NULL,
                kg_data_json TEXT NOT NULL,
                fitness REAL NOT NULL DEFAULT 0.0,
                wins INTEGER NOT NULL DEFAULT 0,
                total_queries INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (population_id, individual_id),
                FOREIGN KEY (population_id) REFERENCES kg_populations(id) ON DELETE CASCADE
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_kg_individuals_population ON kg_individuals(population_id)"
        ))


def _genome_to_dict(g: KGGenome) -> Dict:
    return {
        'lambda_depth': float(g.lambda_depth),
        'lambda_breadth': float(g.lambda_breadth),
        'similarity_threshold': float(g.similarity_threshold),
        'sleep_ops': list(g.sleep_ops),
        'dream_seeds': int(g.dream_seeds),
        'dream_probability': float(g.dream_probability),
        'get_concepts': bool(g.get_concepts),
        'link_facts_facts': bool(g.link_facts_facts),
        'link_concepts_facts': bool(g.link_concepts_facts),
        'link_concepts_concepts': bool(g.link_concepts_concepts),
    }


def _genome_from_dict(d: Dict) -> KGGenome:
    default = KGGenome()
    return KGGenome(
        lambda_depth=float(d.get('lambda_depth', default.lambda_depth)),
        lambda_breadth=float(d.get('lambda_breadth', default.lambda_breadth)),
        similarity_threshold=float(d.get('similarity_threshold', default.similarity_threshold)),
        sleep_ops=list(d.get('sleep_ops', default.sleep_ops)),
        dream_seeds=int(d.get('dream_seeds', default.dream_seeds)),
        dream_probability=float(d.get('dream_probability', default.dream_probability)),
        get_concepts=bool(d.get('get_concepts', default.get_concepts)),
        link_facts_facts=bool(d.get('link_facts_facts', default.link_facts_facts)),
        link_concepts_facts=bool(d.get('link_concepts_facts', default.link_concepts_facts)),
        link_concepts_concepts=bool(d.get('link_concepts_concepts', default.link_concepts_concepts)),
    )


def save_population(engine, population_id: str, name: str, mgr: 'SememolutionPopulation'):
    _ensure_population_schema(engine)
    import time as _time

    config_json = json.dumps({
        'population_size': mgr.ga.config.population_size,
        'mutation_rate': mgr.ga.config.mutation_rate,
        'crossover_rate': mgr.ga.config.crossover_rate,
        'tournament_size': mgr.ga.config.tournament_size,
        'elitism_count': mgr.ga.config.elitism_count,
    })

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO kg_populations (id, name, model, provider, sample_size, config_json, updated_at)
            VALUES (:id, :name, :model, :provider, :sample_size, :config_json, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                model = excluded.model,
                provider = excluded.provider,
                sample_size = excluded.sample_size,
                config_json = excluded.config_json,
                updated_at = CURRENT_TIMESTAMP
        """), {
            'id': population_id, 'name': name,
            'model': mgr.model, 'provider': mgr.provider,
            'sample_size': mgr.sample_size, 'config_json': config_json,
        })

        # Drop and re-insert all individuals (simpler than diff).
        conn.execute(text("DELETE FROM kg_individuals WHERE population_id = :pid"),
                     {'pid': population_id})
        for ind in mgr.ga.population:
            conn.execute(text("""
                INSERT INTO kg_individuals
                (individual_id, population_id, genome_json, kg_data_json, fitness, wins, total_queries, updated_at)
                VALUES (:iid, :pid, :genome, :kg, :fit, :wins, :tq, CURRENT_TIMESTAMP)
            """), {
                'iid': ind.individual_id, 'pid': population_id,
                'genome': json.dumps(_genome_to_dict(ind.genome)),
                'kg': json.dumps(ind.kg_data),
                'fit': float(ind.fitness), 'wins': int(ind.wins), 'tq': int(ind.total_queries),
            })


def load_population(engine, population_id: str) -> Optional['SememolutionPopulation']:
    _ensure_population_schema(engine)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT name, model, provider, sample_size, config_json FROM kg_populations WHERE id = :id"),
                           {'id': population_id}).fetchone()
        if not row:
            return None
        cfg = json.loads(row.config_json or '{}')

        inds_rows = conn.execute(text("""
            SELECT individual_id, genome_json, kg_data_json, fitness, wins, total_queries
            FROM kg_individuals WHERE population_id = :pid
            ORDER BY fitness DESC, individual_id ASC
        """), {'pid': population_id}).fetchall()

    mgr = SememolutionPopulation(
        engine=engine,
        model=row.model or "gemma3:4b",
        provider=row.provider or "ollama",
        population_size=int(cfg.get('population_size', max(1, len(inds_rows) or 10))),
        sample_size=int(row.sample_size or 10),
    )
    mgr.ga.config.mutation_rate = float(cfg.get('mutation_rate', mgr.ga.config.mutation_rate))
    mgr.ga.config.crossover_rate = float(cfg.get('crossover_rate', mgr.ga.config.crossover_rate))
    mgr.ga.config.tournament_size = int(cfg.get('tournament_size', mgr.ga.config.tournament_size))
    mgr.ga.config.elitism_count = int(cfg.get('elitism_count', mgr.ga.config.elitism_count))

    mgr.ga.population = [
        KGIndividual(
            individual_id=r.individual_id,
            genome=_genome_from_dict(json.loads(r.genome_json)),
            kg_data=json.loads(r.kg_data_json),
            fitness=float(r.fitness or 0.0),
            wins=int(r.wins or 0),
            total_queries=int(r.total_queries or 0),
        )
        for r in inds_rows
    ]
    return mgr


def list_populations(engine) -> List[Dict]:
    _ensure_population_schema(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.id, p.name, p.model, p.provider, p.sample_size, p.created_at, p.updated_at,
                   COUNT(i.individual_id) AS individual_count,
                   COALESCE(AVG(i.fitness), 0) AS avg_fitness,
                   COALESCE(MAX(i.fitness), 0) AS max_fitness
            FROM kg_populations p
            LEFT JOIN kg_individuals i ON i.population_id = p.id
            GROUP BY p.id
            ORDER BY p.updated_at DESC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def delete_population(engine, population_id: str) -> bool:
    _ensure_population_schema(engine)
    with engine.begin() as conn:
        r = conn.execute(text("DELETE FROM kg_populations WHERE id = :id"),
                         {'id': population_id})
        return (r.rowcount or 0) > 0


def update_individual_genome(engine, population_id: str, individual_id: str, genome: Dict) -> bool:
    _ensure_population_schema(engine)
    with engine.begin() as conn:
        r = conn.execute(text("""
            UPDATE kg_individuals
            SET genome_json = :g, updated_at = CURRENT_TIMESTAMP
            WHERE population_id = :pid AND individual_id = :iid
        """), {'g': json.dumps(_genome_to_dict(_genome_from_dict(genome))),
               'pid': population_id, 'iid': individual_id})
        return (r.rowcount or 0) > 0
