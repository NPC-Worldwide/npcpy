"""
Tests for the NEAT module.

Tests the engine abstraction and NEAT evolution on XOR
(the classic NEAT benchmark problem).
"""

import numpy as np
import pytest
from npcpy.ft.engine import get_engine, NumPyEngine, list_engines
from npcpy.ft.neat import (
    NEATEvolver, NEATConfig, NEATNetwork,
    Genome, NodeGene, ConnectionGene, InnovationTracker, Species,
)


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestEngine:
    def test_numpy_basic_ops(self):
        eng = get_engine("numpy", seed=42)
        a = eng.array([1.0, 2.0, 3.0])
        assert eng.to_numpy(eng.sum(a)) == pytest.approx(6.0)
        assert eng.to_numpy(eng.mean(a)) == pytest.approx(2.0)

    def test_numpy_activations(self):
        eng = get_engine("numpy", seed=42)
        x = eng.array([0.0, 1.0, -1.0])

        tanh_out = eng.to_numpy(eng.tanh(x))
        assert tanh_out[0] == pytest.approx(0.0, abs=1e-6)
        assert tanh_out[1] > 0.7
        assert tanh_out[2] < -0.7

        relu_out = eng.to_numpy(eng.relu(x))
        assert relu_out[0] == 0.0
        assert relu_out[1] == 1.0
        assert relu_out[2] == 0.0

        sig_out = eng.to_numpy(eng.sigmoid(x))
        assert sig_out[0] == pytest.approx(0.5, abs=1e-6)

    def test_numpy_random(self):
        eng = get_engine("numpy", seed=42)
        r1 = eng.randn(10)
        assert len(eng.to_numpy(r1)) == 10

    def test_list_engines(self):
        engines = list_engines()
        assert "numpy" in engines
        assert "jax" in engines
        assert "mlx" in engines
        assert "cuda" in engines


# ---------------------------------------------------------------------------
# Genome / Innovation tests
# ---------------------------------------------------------------------------

class TestGenome:
    def test_create_genome(self):
        g = Genome(input_size=3, output_size=2)
        assert len(g.nodes) == 5
        assert sum(1 for n in g.nodes.values() if n.type == "input") == 3
        assert sum(1 for n in g.nodes.values() if n.type == "output") == 2

    def test_copy_genome(self):
        g = Genome(input_size=2, output_size=1)
        g.connections[1] = ConnectionGene(0, 2, 0.5, True, 1)
        g.fitness = 42.0

        g2 = g.copy()
        assert g2.fitness == 42.0
        assert len(g2.connections) == 1
        # Ensure deep copy
        g2.connections[1].weight = 999.0
        assert g.connections[1].weight == 0.5

    def test_serialize_roundtrip(self):
        g = Genome(input_size=2, output_size=1)
        g.connections[1] = ConnectionGene(0, 2, 0.5, True, 1)
        g.fitness = 3.14

        data = g.to_dict()
        g2 = Genome.from_dict(data)
        assert g2.input_size == 2
        assert g2.output_size == 1
        assert len(g2.connections) == 1
        assert g2.fitness == pytest.approx(3.14)


class TestInnovationTracker:
    def test_same_innovation(self):
        tracker = InnovationTracker()
        i1 = tracker.get_innovation(0, 3)
        i2 = tracker.get_innovation(0, 3)
        assert i1 == i2

    def test_different_innovations(self):
        tracker = InnovationTracker()
        i1 = tracker.get_innovation(0, 3)
        i2 = tracker.get_innovation(1, 3)
        assert i1 != i2

    def test_reset(self):
        tracker = InnovationTracker()
        i1 = tracker.get_innovation(0, 3)
        tracker.reset_generation()
        i2 = tracker.get_innovation(0, 3)
        # After reset, same topology gets new number
        assert i2 > i1


# ---------------------------------------------------------------------------
# NEATNetwork tests
# ---------------------------------------------------------------------------

class TestNEATNetwork:
    def test_minimal_network(self):
        """A single input->output connection should produce non-zero output."""
        eng = get_engine("numpy", seed=0)
        g = Genome(input_size=2, output_size=1)
        g.connections[1] = ConnectionGene(0, 2, 1.0, True, 1)
        g.connections[2] = ConnectionGene(1, 2, -1.0, True, 2)

        net = NEATNetwork(g, eng)
        out = net.activate([1.0, 0.5])
        # tanh(1.0 * 1.0 + 0.5 * -1.0) = tanh(0.5) ≈ 0.462
        assert out[0] == pytest.approx(np.tanh(0.5), abs=0.01)

    def test_hidden_node(self):
        """Network with a hidden node should work."""
        eng = get_engine("numpy", seed=0)
        g = Genome(input_size=1, output_size=1)
        # input(0) -> hidden(2) -> output(1)
        g.nodes[2] = NodeGene(id=2, type="hidden")
        g.connections[1] = ConnectionGene(0, 2, 1.0, True, 1)
        g.connections[2] = ConnectionGene(2, 1, 1.0, True, 2)

        net = NEATNetwork(g, eng)
        out = net.activate([0.5])
        # tanh(tanh(0.5)) ≈ tanh(0.462) ≈ 0.432
        expected = np.tanh(np.tanh(0.5))
        assert out[0] == pytest.approx(expected, abs=0.01)

    def test_disabled_connection(self):
        """Disabled connections should not contribute."""
        eng = get_engine("numpy", seed=0)
        g = Genome(input_size=1, output_size=1)
        g.connections[1] = ConnectionGene(0, 1, 100.0, False, 1)  # disabled

        net = NEATNetwork(g, eng)
        out = net.activate([1.0])
        # No enabled connections -> output should be tanh(0) = 0
        assert out[0] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# NEATEvolver tests
# ---------------------------------------------------------------------------

class TestNEATEvolver:
    def test_initialization(self):
        evolver = NEATEvolver(
            input_size=2, output_size=1,
            config=NEATConfig(population_size=20),
            engine="numpy", seed=42,
        )
        evolver.initialize_population()
        assert len(evolver.population) == 20
        for g in evolver.population:
            assert g.input_size == 2
            assert g.output_size == 1
            assert len(g.connections) == 2  # fully connected 2->1

    def test_speciation(self):
        evolver = NEATEvolver(
            input_size=2, output_size=1,
            config=NEATConfig(population_size=20, species_threshold=1.0),
            engine="numpy", seed=42,
        )
        evolver.initialize_population()

        # Assign random fitness
        for g in evolver.population:
            g.fitness = np.random.random()

        evolver.speciate()
        assert len(evolver.species) >= 1
        total_members = sum(len(s.members) for s in evolver.species)
        assert total_members == 20

    def test_mutation_adds_structure(self):
        evolver = NEATEvolver(
            input_size=2, output_size=1,
            config=NEATConfig(
                add_node_rate=1.0,  # always add node
                add_connection_rate=1.0,  # always add connection
            ),
            engine="numpy", seed=42,
        )
        g = evolver._create_initial_genome()
        initial_nodes = len(g.nodes)
        initial_conns = len(g.connections)

        evolver._mutate(g)
        assert len(g.nodes) > initial_nodes  # should have added a hidden node
        assert len(g.connections) > initial_conns

    def test_crossover(self):
        evolver = NEATEvolver(
            input_size=2, output_size=1,
            engine="numpy", seed=42,
        )
        p1 = evolver._create_initial_genome()
        p2 = evolver._create_initial_genome()
        p1.fitness = 1.0
        p2.fitness = 0.5

        child = evolver._crossover(p1, p2)
        assert child.input_size == 2
        assert child.output_size == 1
        assert len(child.connections) > 0

    def test_xor_evolution(self):
        """
        Classic NEAT benchmark: evolve a network to solve XOR.
        XOR requires at least one hidden node, so this tests
        that structural mutations actually work.
        """
        XOR_INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
        XOR_OUTPUTS = [0, 1, 1, 0]

        def xor_fitness(network: NEATNetwork) -> float:
            error = 0.0
            for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
                output = network.activate(inputs)[0]
                error += (output - expected) ** 2
            # Fitness is inverse of error, max 4.0
            return 4.0 - error

        evolver = NEATEvolver(
            input_size=2,
            output_size=1,
            config=NEATConfig(
                population_size=150,
                weight_mutation_rate=0.8,
                add_node_rate=0.03,
                add_connection_rate=0.05,
                species_threshold=3.0,
            ),
            engine="numpy",
            seed=42,
        )

        best = evolver.run(xor_fitness, generations=150, verbose=False)
        best_net = NEATNetwork(best, get_engine("numpy"))

        # Check that it learned something — fitness should be well above random
        final_fitness = xor_fitness(best_net)
        print(f"XOR final fitness: {final_fitness:.3f} (max 4.0)")
        print(f"Topology: {best.num_hidden} hidden, {best.num_enabled_connections} connections")

        # We expect fitness > 3.0 (good but maybe not perfect)
        # NEAT should solve XOR reliably with these settings
        assert final_fitness > 3.0, f"NEAT failed to learn XOR: fitness={final_fitness:.3f}"

        # Verify it added at least one hidden node (XOR requires it)
        assert best.num_hidden >= 1, "XOR requires hidden nodes but NEAT didn't add any"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
