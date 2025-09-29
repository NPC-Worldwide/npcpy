import random
from dataclasses import dataclass
from typing import Callable, Optional, List


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
    
    def evaluate_population(self) -> List[float]:
        return [
            self.fitness_fn(individual) 
            for individual in self.population
        ]
    
    def tournament_select(self, fitness_scores: List[float]):
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