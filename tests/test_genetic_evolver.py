"""Test suite for genetic evolver (ge.py) module."""

import pytest


class TestGAConfig:
    """Test GAConfig dataclass."""

    def test_default_values(self):
        """Test GAConfig has correct defaults"""
        from npcpy.ft.ge import GAConfig

        config = GAConfig()

        assert config.population_size == 20
        assert config.mutation_rate == 0.15
        assert config.crossover_rate == 0.7
        assert config.tournament_size == 3
        assert config.elitism_count == 2
        assert config.generations == 50

    def test_custom_values(self):
        """Test GAConfig with custom values"""
        from npcpy.ft.ge import GAConfig

        config = GAConfig(
            population_size=100,
            mutation_rate=0.25,
            crossover_rate=0.8,
            tournament_size=5,
            elitism_count=5,
            generations=100
        )

        assert config.population_size == 100
        assert config.mutation_rate == 0.25
        assert config.crossover_rate == 0.8
        assert config.tournament_size == 5
        assert config.elitism_count == 5
        assert config.generations == 100


class TestGeneticEvolverInitialization:
    """Test GeneticEvolver initialization."""

    def test_evolver_initialization_with_defaults(self):
        """Test GeneticEvolver initializes with default config"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 0
        )

        assert evolver.config is not None
        assert evolver.config.population_size == 20
        assert evolver.population == []
        assert evolver.history == []

    def test_evolver_initialization_with_custom_config(self):
        """Test GeneticEvolver with custom config"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=50, generations=25)

        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 0,
            config=config
        )

        assert evolver.config.population_size == 50
        assert evolver.config.generations == 25

    def test_evolver_stores_functions(self):
        """Test GeneticEvolver stores provided functions"""
        from npcpy.ft.ge import GeneticEvolver

        fitness = lambda x: x * 2
        mutate = lambda x: x + 1
        crossover = lambda a, b: (a + b) / 2
        initialize = lambda: 5

        evolver = GeneticEvolver(
            fitness_fn=fitness,
            mutate_fn=mutate,
            crossover_fn=crossover,
            initialize_fn=initialize
        )

        assert evolver.fitness_fn is fitness
        assert evolver.mutate_fn is mutate
        assert evolver.crossover_fn is crossover
        assert evolver.initialize_fn is initialize


class TestPopulationInitialization:
    """Test population initialization."""

    def test_initialize_population_size(self):
        """Test population is initialized to correct size"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=10)

        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 1,
            config=config
        )

        evolver.initialize_population()

        assert len(evolver.population) == 10

    def test_initialize_population_uses_initialize_fn(self):
        """Test population uses initialize function"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        counter = {"count": 0}

        def init_fn():
            counter["count"] += 1
            return counter["count"]

        config = GAConfig(population_size=5)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=init_fn,
            config=config
        )

        evolver.initialize_population()

        assert counter["count"] == 5
        assert evolver.population == [1, 2, 3, 4, 5]


class TestFitnessEvaluation:
    """Test fitness evaluation."""

    def test_evaluate_population_returns_scores(self):
        """Test evaluate_population returns list of scores"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=3)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x * 2,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 5,
            config=config
        )

        evolver.population = [1, 2, 3]
        scores = evolver.evaluate_population()

        assert scores == [2, 4, 6]

    def test_evaluate_population_custom_fitness(self):
        """Test evaluate_population with complex fitness function"""
        from npcpy.ft.ge import GeneticEvolver

        def fitness(individual):
            return sum(individual) if isinstance(individual, list) else individual

        evolver = GeneticEvolver(
            fitness_fn=fitness,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: []
        )

        evolver.population = [[1, 2], [3, 4, 5], [10]]
        scores = evolver.evaluate_population()

        assert scores == [3, 12, 10]


class TestTournamentSelection:
    """Test tournament selection."""

    def test_tournament_select_returns_individual(self):
        """Test tournament selection returns an individual from population"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=10, tournament_size=3)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 0,
            config=config
        )

        evolver.population = list(range(10))
        fitness_scores = list(range(10))

        # Run multiple times to test randomness
        for _ in range(20):
            selected = evolver.tournament_select(fitness_scores)
            assert selected in evolver.population

    def test_tournament_select_prefers_higher_fitness(self):
        """Test tournament selection tends to select higher fitness"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=100, tournament_size=5)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 0,
            config=config
        )

        evolver.population = list(range(100))
        fitness_scores = list(range(100))

        selections = [evolver.tournament_select(fitness_scores) for _ in range(100)]
        avg_selection = sum(selections) / len(selections)

        # Average selection should be above median (50) due to selection pressure
        assert avg_selection > 50


class TestEvolveGeneration:
    """Test generation evolution."""

    def test_evolve_generation_returns_stats(self):
        """Test evolve_generation returns statistics dict"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=10, elitism_count=2)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 5,
            config=config
        )

        evolver.initialize_population()
        stats = evolver.evolve_generation()

        assert "best_fitness" in stats
        assert "avg_fitness" in stats
        assert "best_individual" in stats

    def test_evolve_generation_preserves_population_size(self):
        """Test population size is maintained after evolution"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=15)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x + 1,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 0,
            config=config
        )

        evolver.initialize_population()
        assert len(evolver.population) == 15

        evolver.evolve_generation()
        assert len(evolver.population) == 15

    def test_evolve_generation_elitism(self):
        """Test elitism preserves best individuals"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=5, elitism_count=2, crossover_rate=0, mutation_rate=0)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 0,
            config=config
        )

        evolver.population = [1, 2, 3, 4, 5]
        evolver.evolve_generation()

        # Top 2 (4 and 5) should be preserved
        assert 5 in evolver.population
        assert 4 in evolver.population


class TestRun:
    """Test full evolution run."""

    def test_run_initializes_population_if_empty(self):
        """Test run initializes population if not already done"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=5, generations=1)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 1,
            config=config
        )

        assert len(evolver.population) == 0
        evolver.run(generations=1)
        assert len(evolver.population) == 5

    def test_run_records_history(self):
        """Test run records history of generations"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=5, generations=10)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 1,
            config=config
        )

        evolver.run(generations=10)

        assert len(evolver.history) == 10

    def test_run_returns_best_individual(self):
        """Test run returns best individual"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=10, generations=5, mutation_rate=0.5)

        def fitness(x):
            return 100 - abs(x - 50)  # Best fitness at x=50

        evolver = GeneticEvolver(
            fitness_fn=fitness,
            mutate_fn=lambda x: x + (1 if x < 50 else -1),
            crossover_fn=lambda a, b: (a + b) // 2,
            initialize_fn=lambda: 0,
            config=config
        )

        best = evolver.run(generations=5)

        assert best is not None

    def test_run_uses_config_generations(self):
        """Test run uses config generations by default"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig

        config = GAConfig(population_size=3, generations=7)
        evolver = GeneticEvolver(
            fitness_fn=lambda x: x,
            mutate_fn=lambda x: x,
            crossover_fn=lambda a, b: a,
            initialize_fn=lambda: 1,
            config=config
        )

        evolver.run()

        assert len(evolver.history) == 7


class TestGeneticEvolverWithComplexIndividuals:
    """Test GeneticEvolver with complex individual types."""

    def test_list_individuals(self):
        """Test evolution with list-based individuals"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig
        import random

        config = GAConfig(population_size=10, generations=5)

        def fitness(individual):
            return sum(individual)

        def mutate(individual):
            idx = random.randint(0, len(individual) - 1)
            new_ind = individual.copy()
            new_ind[idx] = random.randint(0, 10)
            return new_ind

        def crossover(a, b):
            point = len(a) // 2
            return a[:point] + b[point:]

        def initialize():
            return [random.randint(0, 10) for _ in range(5)]

        evolver = GeneticEvolver(
            fitness_fn=fitness,
            mutate_fn=mutate,
            crossover_fn=crossover,
            initialize_fn=initialize,
            config=config
        )

        best = evolver.run()

        assert isinstance(best, list)
        assert len(best) == 5

    def test_dict_individuals(self):
        """Test evolution with dict-based individuals"""
        from npcpy.ft.ge import GeneticEvolver, GAConfig
        import random

        config = GAConfig(population_size=5, generations=3)

        def fitness(ind):
            return ind.get("score", 0)

        def mutate(ind):
            return {"score": ind["score"] + random.randint(-1, 2)}

        def crossover(a, b):
            return {"score": (a["score"] + b["score"]) // 2}

        def initialize():
            return {"score": random.randint(0, 10)}

        evolver = GeneticEvolver(
            fitness_fn=fitness,
            mutate_fn=mutate,
            crossover_fn=crossover,
            initialize_fn=initialize,
            config=config
        )

        best = evolver.run()

        assert isinstance(best, dict)
        assert "score" in best


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
