# genetic engineering for using genetic algorithms with LLMs 
from typing import List, Dict, Any, Optional
import json
import copy
from npcpy.llm_funcs import (
    get_llm_response,
    get_facts,
    generate_groups,
    remove_redundant_groups,
    abstract
)


def create_knowledge_graph(
    facts: List[Dict],
    model: str = None,
    provider: str = None
) -> Dict[str, Any]:

    groups = generate_groups(
        facts,
        model=model,
        provider=provider
    )
    
    groups = remove_redundant_groups(
        groups,
        model=model,
        provider=provider
    )
    
    abstractions = abstract(
        groups,
        model=model,
        provider=provider
    )
    
    graph = {
        'facts': facts,
        'groups': groups,
        'abstractions': abstractions,
        'edges': []
    }
    
    for i, fact in enumerate(facts):
        for j, group in enumerate(groups):
            graph['edges'].append({
                'from': f"fact_{i}",
                'to': f"group_{j}",
                'strength': random.random()
            })
    
    return graph


def mutate_knowledge_graph(
    graph: Dict[str, Any],
    mutation_type: str = 'random',
    model: str = None,
    provider: str = None
) -> Dict[str, Any]:

    new_graph = copy.deepcopy(graph)
    
    mutation_types = [
        'regroup',
        'abstract',
        'prune_weak',
        'add_inference'
    ]
    
    if mutation_type == 'random':
        mutation_type = random.choice(mutation_types)
    
    if mutation_type == 'regroup':
        new_graph['groups'] = generate_groups(
            new_graph['facts'],
            model=model,
            provider=provider
        )
    
    elif mutation_type == 'abstract':
        if new_graph['groups']:
            new_abstractions = abstract(
                new_graph['groups'],
                model=model,
                provider=provider
            )
            new_graph['abstractions'].extend(new_abstractions)
    
    elif mutation_type == 'prune_weak':
        threshold = 0.3
        new_graph['edges'] = [
            e for e in new_graph['edges']
            if e['strength'] > threshold
        ]
    
    elif mutation_type == 'add_inference':
        if len(new_graph['facts']) > 2:
            sample_facts = random.sample(
                new_graph['facts'],
                min(3, len(new_graph['facts']))
            )
            
            inferred = infer_from_facts(
                sample_facts,
                model=model,
                provider=provider
            )
            
            new_graph['facts'].extend(inferred)
    
    return new_graph


def crossover_knowledge_graphs(
    graph1: Dict[str, Any],
    graph2: Dict[str, Any]
) -> Dict[str, Any]:

    child = {
        'facts': [],
        'groups': [],
        'abstractions': [],
        'edges': []
    }
    
    child['facts'] = (
        graph1['facts'][:len(graph1['facts'])//2] +
        graph2['facts'][len(graph2['facts'])//2:]
    )
    
    child['groups'] = (
        graph1['groups'] if 
        len(graph1['groups']) > len(graph2['groups'])
        else graph2['groups']
    )
    
    child['abstractions'] = list(
        set(
            [g['name'] for g in graph1['abstractions']] +
            [g['name'] for g in graph2['abstractions']]
        )
    )
    child['abstractions'] = [
        {'name': name} for name in child['abstractions']
    ]
    
    return child


def evaluate_knowledge_graph(
    graph: Dict[str, Any],
    test_queries: List[str],
    model: str = None,
    provider: str = None
) -> float:

    if not graph['facts'] or not graph['groups']:
        return 0.0
    
    graph_context = json.dumps({
        'facts': [
            f['statement'] 
            for f in graph['facts'][:20]
        ],
        'groups': [
            g['name'] 
            for g in graph['groups'][:10]
        ]
    })
    
    scores = []
    
    for query in test_queries:
        prompt = f"""
        Using this knowledge graph context:
        {graph_context}
        
        Answer this query: {query}
        
        Rate your confidence 0-1.
        Return JSON: {{"answer": "...", "confidence": 0.x}}
        """
        
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            format='json'
        )
        
        result = response.get('response', {})
        confidence = result.get('confidence', 0.0)
        scores.append(confidence)
    
    diversity = len(set(
        g['name'] for g in graph['groups']
    )) / max(len(graph['groups']), 1)
    
    coverage = min(
        len(graph['facts']) / 50.0,
        1.0
    )
    
    fitness = (
        np.mean(scores) * 0.5 +
        diversity * 0.3 +
        coverage * 0.2
    )
    
    return fitness


def infer_from_facts(
    facts: List[Dict],
    model: str = None,
    provider: str = None
) -> List[Dict]:

    fact_statements = [f['statement'] for f in facts]
    
    prompt = f"""
    Given these facts:
    {json.dumps(fact_statements)}
    
    Infer 1-2 new facts that logically follow.
    Return JSON: {{"inferred_facts": ["fact1", "fact2"]}}
    """
    
    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format='json'
    )
    
    result = response.get('response', {})
    new_facts = result.get('inferred_facts', [])
    
    return [
        {
            'statement': fact,
            'source_text': 'inferred',
            'type': 'inferred'
        }
        for fact in new_facts
    ]
import random
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GAConfig:
    population_size: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elitism_count: int = 2
    generations: int = 50


class GeneticEvolver:
    """
    Generic GA that takes fitness, mutation, crossover 
    and initialization functions to evolve any population
    """
    def __init__(
        self,
        fitness_fn: Callable,
        mutate_fn: Callable,
        crossover_fn: Callable,
        initialize_fn: Callable,
        config: Optional[GAConfig] = None
    ):
        self.fitness_fn = fitness_fn
        self.mutate_fn = mutate_fn
        self.crossover_fn = crossover_fn
        self.initialize_fn = initialize_fn
        self.config = config or GAConfig()
        self.population = []
        self.history = []
    
    def initialize_population(self):
        self.population = [
            self.initialize_fn() 
            for _ in range(self.config.population_size)
        ]
    
    def evaluate_population(self):
        return [
            self.fitness_fn(individual) 
            for individual in self.population
        ]
    
    def tournament_select(self, fitness_scores):
        indices = random.sample(
            range(len(self.population)),
            self.config.tournament_size
        )
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[
            tournament_fitness.index(max(tournament_fitness))
        ]
        return self.population[winner_idx]
    
    def evolve_generation(self):
        fitness_scores = self.evaluate_population()
        
        sorted_pop = sorted(
            zip(self.population, fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        new_population = [
            ind for ind, _ in sorted_pop[:self.config.elitism_count]
        ]
        
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_select(fitness_scores)
            parent2 = self.tournament_select(fitness_scores)
            
            if random.random() < self.config.crossover_rate:
                child = self.crossover_fn(parent1, parent2)
            else:
                child = parent1
            
            if random.random() < self.config.mutation_rate:
                child = self.mutate_fn(child)
            
            new_population.append(child)
        
        self.population = new_population[:self.config.population_size]
        
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        return {
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'best_individual': sorted_pop[0][0]
        }
    
    def run(self, generations: Optional[int] = None):
        if not self.population:
            self.initialize_population()
        
        gens = generations or self.config.generations
        
        for gen in range(gens):
            gen_stats = self.evolve_generation()
            self.history.append(gen_stats)
            
            if gen % 10 == 0:
                print(
                    f"Gen {gen}: "
                    f"Best={gen_stats['best_fitness']:.3f}, "
                    f"Avg={gen_stats['avg_fitness']:.3f}"
                )
        
        return self.history[-1]['best_individual']