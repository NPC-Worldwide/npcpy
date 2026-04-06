"""
NEAT (NeuroEvolution of Augmenting Topologies) for npcpy.

Evolves neural network topology and weights simultaneously.
Builds on npcpy's GeneticEvolver pattern with multi-backend support
via the engine abstraction (numpy, jax, mlx, cuda).

Usage:
    from npcpy.ft.neat import NEATEvolver, NEATConfig

    evolver = NEATEvolver(
        input_size=12,
        output_size=3,
        config=NEATConfig(population_size=150),
        engine="numpy",  # or "jax", "mlx", "cuda"
    )

    def fitness_fn(network):
        # evaluate network on your task
        return score

    best = evolver.run(fitness_fn, generations=100)

References:
    Stanley, K.O. & Miikkulainen, R. (2002).
    "Evolving Neural Networks through Augmenting Topologies."
    Evolutionary Computation, 10(2), 99-127.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import copy
import random
import json
import pickle

import numpy as np

from npcpy.ft.engine import Engine, get_engine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NEATConfig:
    """Configuration for NEAT evolution."""
    # Population
    population_size: int = 150
    # Selection
    elitism_count: int = 2
    survival_threshold: float = 0.2
    tournament_size: int = 3
    # Mutation rates
    weight_mutation_rate: float = 0.8
    weight_perturbation_std: float = 0.2
    weight_replace_rate: float = 0.1
    add_node_rate: float = 0.03
    add_connection_rate: float = 0.05
    disable_connection_rate: float = 0.01
    # Speciation
    species_threshold: float = 3.0
    disjoint_coefficient: float = 1.0
    excess_coefficient: float = 1.0
    weight_coefficient: float = 0.4
    # Species management
    species_stagnation_limit: int = 15
    species_elitism: int = 1
    min_species_size: int = 2
    # Fitness
    fitness_sharing: bool = True


# ---------------------------------------------------------------------------
# Genome representation
# ---------------------------------------------------------------------------

@dataclass
class NodeGene:
    """A node (neuron) in the network."""
    id: int
    type: str  # "input", "hidden", "output"
    activation: str = "tanh"  # "tanh", "relu", "sigmoid", "identity"
    bias: float = 0.0

    def copy(self):
        return NodeGene(
            id=self.id, type=self.type,
            activation=self.activation, bias=self.bias,
        )


@dataclass
class ConnectionGene:
    """A connection (synapse) between nodes."""
    input_node: int
    output_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0

    def copy(self):
        return ConnectionGene(
            input_node=self.input_node, output_node=self.output_node,
            weight=self.weight, enabled=self.enabled,
            innovation=self.innovation,
        )


class InnovationTracker:
    """
    Global tracker for structural innovations.
    Ensures the same structural change always gets the same innovation number,
    which is critical for meaningful crossover.
    """

    def __init__(self):
        self.counter = 0
        self._history: Dict[Tuple[int, int], int] = {}

    def get_innovation(self, input_node: int, output_node: int) -> int:
        key = (input_node, output_node)
        if key not in self._history:
            self.counter += 1
            self._history[key] = self.counter
        return self._history[key]

    def reset_generation(self):
        """Call between generations to allow re-discovery."""
        self._history.clear()


class Genome:
    """
    A NEAT genome encoding a neural network topology.
    """

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness: float = 0.0
        self._next_node_id = input_size + output_size

        # Create input nodes
        for i in range(input_size):
            self.nodes[i] = NodeGene(id=i, type="input")

        # Create output nodes
        for i in range(output_size):
            nid = input_size + i
            self.nodes[nid] = NodeGene(id=nid, type="output")

    @property
    def num_hidden(self) -> int:
        return sum(1 for n in self.nodes.values() if n.type == "hidden")

    @property
    def num_enabled_connections(self) -> int:
        return sum(1 for c in self.connections.values() if c.enabled)

    def copy(self) -> Genome:
        g = Genome.__new__(Genome)
        g.input_size = self.input_size
        g.output_size = self.output_size
        g.nodes = {k: v.copy() for k, v in self.nodes.items()}
        g.connections = {k: v.copy() for k, v in self.connections.items()}
        g.fitness = self.fitness
        g._next_node_id = self._next_node_id
        return g

    def allocate_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def to_dict(self) -> Dict:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "nodes": {
                str(k): {"id": v.id, "type": v.type, "activation": v.activation, "bias": v.bias}
                for k, v in self.nodes.items()
            },
            "connections": {
                str(k): {
                    "input_node": v.input_node, "output_node": v.output_node,
                    "weight": v.weight, "enabled": v.enabled, "innovation": v.innovation,
                }
                for k, v in self.connections.items()
            },
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Genome:
        g = cls(data["input_size"], data["output_size"])
        g.nodes = {
            int(k): NodeGene(**v)
            for k, v in data["nodes"].items()
        }
        g.connections = {
            int(k): ConnectionGene(**v)
            for k, v in data["connections"].items()
        }
        g.fitness = data.get("fitness", 0.0)
        if g.nodes:
            g._next_node_id = max(g.nodes.keys()) + 1
        return g


# ---------------------------------------------------------------------------
# Network (phenotype) — evaluates a genome using the compute engine
# ---------------------------------------------------------------------------

class NEATNetwork:
    """
    Feed-forward neural network built from a Genome.
    Uses the engine abstraction for compute.
    """

    def __init__(self, genome: Genome, engine: Engine):
        self.genome = genome
        self.engine = engine
        self._eval_order = self._topological_sort()

    def _topological_sort(self) -> List[int]:
        """
        Compute evaluation order via topological sort.
        Only includes nodes reachable from inputs to outputs.
        """
        input_ids = set(range(self.genome.input_size))
        output_ids = set(
            range(self.genome.input_size, self.genome.input_size + self.genome.output_size)
        )

        # Build adjacency: node -> list of (source, weight)
        incoming: Dict[int, List[Tuple[int, float]]] = {
            nid: [] for nid in self.genome.nodes
        }
        for conn in self.genome.connections.values():
            if conn.enabled and conn.input_node in self.genome.nodes and conn.output_node in self.genome.nodes:
                incoming[conn.output_node].append((conn.input_node, conn.weight))

        # Kahn's algorithm
        in_degree = {nid: 0 for nid in self.genome.nodes}
        for nid, sources in incoming.items():
            in_degree[nid] = len(sources) if nid not in input_ids else 0

        queue = list(input_ids)
        order = []
        visited = set()

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            order.append(node)

            for conn in self.genome.connections.values():
                if conn.enabled and conn.input_node == node:
                    out = conn.output_node
                    in_degree[out] -= 1
                    if in_degree[out] <= 0 and out not in visited:
                        queue.append(out)

        # Ensure output nodes are included even if disconnected
        for oid in output_ids:
            if oid not in visited:
                order.append(oid)

        # Return only non-input nodes in order
        return [n for n in order if n not in input_ids]

    def _get_activation_fn(self, name: str):
        fns = {
            "tanh": self.engine.tanh,
            "relu": self.engine.relu,
            "sigmoid": self.engine.sigmoid,
            "identity": lambda x: x,
        }
        return fns.get(name, self.engine.tanh)

    def activate(self, inputs) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            inputs: Array-like of input values (length == genome.input_size)

        Returns:
            numpy array of output values
        """
        eng = self.engine
        inputs = eng.array(inputs)

        # Node activations
        values = {}
        for i in range(self.genome.input_size):
            values[i] = inputs[i] if hasattr(inputs, '__getitem__') else inputs

        # Build incoming connections lookup
        incoming = {}
        for conn in self.genome.connections.values():
            if conn.enabled:
                if conn.output_node not in incoming:
                    incoming[conn.output_node] = []
                incoming[conn.output_node].append(conn)

        # Evaluate in topological order
        for nid in self._eval_order:
            node = self.genome.nodes.get(nid)
            if node is None:
                continue

            acc = node.bias
            for conn in incoming.get(nid, []):
                if conn.input_node in values:
                    v = values[conn.input_node]
                    acc = acc + float(v) * conn.weight

            act_fn = self._get_activation_fn(node.activation)
            activated = act_fn(eng.array([acc]))
            values[nid] = float(eng.to_numpy(activated)[0])

        # Collect outputs
        outputs = []
        for i in range(self.genome.input_size, self.genome.input_size + self.genome.output_size):
            outputs.append(values.get(i, 0.0))

        return np.array(outputs)


# ---------------------------------------------------------------------------
# NEAT Evolver
# ---------------------------------------------------------------------------

class NEATEvolver:
    """
    NEAT evolutionary algorithm.

    Evolves a population of neural network topologies through
    speciation, crossover, and structural/weight mutations.

    Args:
        input_size: Number of input nodes
        output_size: Number of output nodes
        config: NEATConfig with hyperparameters
        engine: Compute backend name ("numpy", "jax", "mlx", "cuda")
        seed: Random seed
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[NEATConfig] = None,
        engine: str = "numpy",
        seed: Optional[int] = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or NEATConfig()
        self.engine = get_engine(engine, seed=seed)
        self.innovations = InnovationTracker()

        self.population: List[Genome] = []
        self.species: List[Species] = []
        self.history: List[Dict[str, Any]] = []
        self.generation = 0

        if seed is not None:
            random.seed(seed)

    # --- Initialization ---

    def initialize_population(self):
        """Create the initial population of minimal genomes."""
        self.population = []
        for _ in range(self.config.population_size):
            genome = self._create_initial_genome()
            self.population.append(genome)

    def _create_initial_genome(self) -> Genome:
        """Create a fully-connected minimal genome (no hidden nodes)."""
        genome = Genome(self.input_size, self.output_size)

        for i in range(self.input_size):
            for j in range(self.output_size):
                out_id = self.input_size + j
                inn = self.innovations.get_innovation(i, out_id)
                genome.connections[inn] = ConnectionGene(
                    input_node=i,
                    output_node=out_id,
                    weight=random.gauss(0, 1),
                    enabled=True,
                    innovation=inn,
                )
        return genome

    # --- Evaluation ---

    def evaluate(self, fitness_fn: Callable[[NEATNetwork], float]):
        """Evaluate all genomes in the population."""
        for genome in self.population:
            network = NEATNetwork(genome, self.engine)
            genome.fitness = fitness_fn(network)

    # --- Speciation ---

    def speciate(self):
        """Divide population into species based on genetic distance."""
        # Clear existing species members but keep representatives
        for sp in self.species:
            sp.representative = random.choice(sp.members) if sp.members else sp.representative
            sp.members = []

        unassigned = list(self.population)

        for genome in unassigned:
            placed = False
            for sp in self.species:
                dist = self._genetic_distance(genome, sp.representative)
                if dist < self.config.species_threshold:
                    sp.members.append(genome)
                    placed = True
                    break
            if not placed:
                new_sp = Species(representative=genome, members=[genome])
                self.species.append(new_sp)

        # Remove empty species
        self.species = [sp for sp in self.species if sp.members]

        # Update stagnation
        for sp in self.species:
            best = max(sp.members, key=lambda g: g.fitness)
            if best.fitness > sp.best_fitness:
                sp.best_fitness = best.fitness
                sp.stagnation = 0
            else:
                sp.stagnation += 1

        # Remove stagnant species (keep at least one)
        if len(self.species) > 1:
            self.species = [
                sp for sp in self.species
                if sp.stagnation < self.config.species_stagnation_limit
            ] or [max(self.species, key=lambda s: s.best_fitness)]

    def _genetic_distance(self, g1: Genome, g2: Genome) -> float:
        """Compute NEAT genetic distance between two genomes."""
        inn1 = set(g1.connections.keys())
        inn2 = set(g2.connections.keys())

        if not inn1 and not inn2:
            return 0.0

        matching = inn1 & inn2
        disjoint_excess = len(inn1.symmetric_difference(inn2))

        weight_diff = 0.0
        if matching:
            weight_diff = sum(
                abs(g1.connections[i].weight - g2.connections[i].weight)
                for i in matching
            ) / len(matching)

        N = max(len(inn1), len(inn2))
        N = 1.0 if N < 20 else float(N)

        c = self.config
        return (c.disjoint_coefficient * disjoint_excess / N) + (c.weight_coefficient * weight_diff)

    # --- Reproduction ---

    def reproduce(self):
        """Create next generation through selection, crossover, and mutation."""
        new_population = []

        # Calculate species offspring allocation
        total_adjusted_fitness = 0.0
        for sp in self.species:
            if self.config.fitness_sharing:
                sp.adjusted_fitness = sum(
                    g.fitness / len(sp.members) for g in sp.members
                )
            else:
                sp.adjusted_fitness = sum(g.fitness for g in sp.members)
            total_adjusted_fitness += sp.adjusted_fitness

        # Elitism: keep best from each species
        for sp in self.species:
            sp.members.sort(key=lambda g: g.fitness, reverse=True)
            for i in range(min(self.config.species_elitism, len(sp.members))):
                new_population.append(sp.members[i].copy())

        # Allocate offspring per species proportionally
        remaining = self.config.population_size - len(new_population)
        if total_adjusted_fitness > 0 and remaining > 0:
            for sp in self.species:
                n_offspring = max(0, int(
                    (sp.adjusted_fitness / total_adjusted_fitness) * remaining
                ))
                for _ in range(n_offspring):
                    if len(new_population) >= self.config.population_size:
                        break
                    child = self._reproduce_from_species(sp)
                    new_population.append(child)

        # Fill remainder
        while len(new_population) < self.config.population_size:
            sp = random.choice(self.species)
            child = self._reproduce_from_species(sp)
            new_population.append(child)

        self.population = new_population[:self.config.population_size]

    def _reproduce_from_species(self, sp: Species) -> Genome:
        """Produce one child from a species."""
        if len(sp.members) == 1:
            child = sp.members[0].copy()
        elif random.random() < 0.75:
            p1 = self._tournament_select(sp.members)
            p2 = self._tournament_select(sp.members)
            child = self._crossover(p1, p2)
        else:
            child = self._tournament_select(sp.members).copy()

        self._mutate(child)
        return child

    def _tournament_select(self, candidates: List[Genome]) -> Genome:
        k = min(self.config.tournament_size, len(candidates))
        tournament = random.sample(candidates, k)
        return max(tournament, key=lambda g: g.fitness)

    # --- Crossover ---

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """
        NEAT crossover: matching genes inherited randomly,
        disjoint/excess genes inherited from fitter parent.
        """
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1

        child = Genome(self.input_size, self.output_size)

        # Inherit nodes from fitter parent, plus any hidden from parent2 that are needed
        child.nodes = {k: v.copy() for k, v in parent1.nodes.items()}
        child._next_node_id = max(
            parent1._next_node_id, parent2._next_node_id
        )

        inn1 = set(parent1.connections.keys())
        inn2 = set(parent2.connections.keys())

        for inn_num in inn1 | inn2:
            if inn_num in inn1 and inn_num in inn2:
                # Matching gene — random inheritance
                conn = random.choice([
                    parent1.connections[inn_num],
                    parent2.connections[inn_num],
                ])
                child.connections[inn_num] = conn.copy()
                # Ensure nodes exist
                for nid in (conn.input_node, conn.output_node):
                    if nid not in child.nodes:
                        src = parent1.nodes.get(nid) or parent2.nodes.get(nid)
                        if src:
                            child.nodes[nid] = src.copy()
            elif inn_num in inn1:
                # Disjoint/excess from fitter parent
                conn = parent1.connections[inn_num]
                child.connections[inn_num] = conn.copy()
                for nid in (conn.input_node, conn.output_node):
                    if nid not in child.nodes:
                        src = parent1.nodes.get(nid)
                        if src:
                            child.nodes[nid] = src.copy()

        return child

    # --- Mutation ---

    def _mutate(self, genome: Genome):
        """Apply mutations to a genome."""
        # Weight mutations
        for conn in genome.connections.values():
            if random.random() < self.config.weight_mutation_rate:
                if random.random() < self.config.weight_replace_rate:
                    conn.weight = random.gauss(0, 1)
                else:
                    conn.weight += random.gauss(0, self.config.weight_perturbation_std)

        # Bias mutations (same rate as weight)
        for node in genome.nodes.values():
            if node.type != "input" and random.random() < self.config.weight_mutation_rate:
                if random.random() < self.config.weight_replace_rate:
                    node.bias = random.gauss(0, 1)
                else:
                    node.bias += random.gauss(0, self.config.weight_perturbation_std)

        # Structural: add node
        if random.random() < self.config.add_node_rate:
            self._mutate_add_node(genome)

        # Structural: add connection
        if random.random() < self.config.add_connection_rate:
            self._mutate_add_connection(genome)

        # Disable a connection
        if random.random() < self.config.disable_connection_rate:
            enabled = [c for c in genome.connections.values() if c.enabled]
            if enabled:
                random.choice(enabled).enabled = False

    def _mutate_add_node(self, genome: Genome):
        """Split an existing connection by inserting a new node."""
        enabled = [c for c in genome.connections.values() if c.enabled]
        if not enabled:
            return

        conn = random.choice(enabled)
        conn.enabled = False

        new_id = genome.allocate_node_id()
        genome.nodes[new_id] = NodeGene(id=new_id, type="hidden")

        # Connection from old input to new node (weight 1.0)
        inn1 = self.innovations.get_innovation(conn.input_node, new_id)
        genome.connections[inn1] = ConnectionGene(
            input_node=conn.input_node, output_node=new_id,
            weight=1.0, enabled=True, innovation=inn1,
        )

        # Connection from new node to old output (original weight)
        inn2 = self.innovations.get_innovation(new_id, conn.output_node)
        genome.connections[inn2] = ConnectionGene(
            input_node=new_id, output_node=conn.output_node,
            weight=conn.weight, enabled=True, innovation=inn2,
        )

    def _mutate_add_connection(self, genome: Genome):
        """Add a new connection between two unconnected nodes."""
        input_candidates = [
            nid for nid, n in genome.nodes.items() if n.type != "output"
        ]
        output_candidates = [
            nid for nid, n in genome.nodes.items() if n.type != "input"
        ]

        if not input_candidates or not output_candidates:
            return

        # Try a few times to find a valid new connection
        for _ in range(20):
            n1 = random.choice(input_candidates)
            n2 = random.choice(output_candidates)
            if n1 == n2:
                continue

            # Check for existing connection
            exists = any(
                c.input_node == n1 and c.output_node == n2
                for c in genome.connections.values()
            )
            if exists:
                continue

            # Check for cycles (we want feed-forward only)
            if self._would_create_cycle(genome, n1, n2):
                continue

            inn = self.innovations.get_innovation(n1, n2)
            genome.connections[inn] = ConnectionGene(
                input_node=n1, output_node=n2,
                weight=random.gauss(0, 1),
                enabled=True, innovation=inn,
            )
            return

    def _would_create_cycle(self, genome: Genome, from_node: int, to_node: int) -> bool:
        """Check if adding from_node -> to_node would create a cycle."""
        if from_node == to_node:
            return True

        visited_rev = set()
        stack_rev = [to_node]

        while stack_rev:
            current = stack_rev.pop()
            if current == from_node:
                return True
            if current in visited_rev:
                continue
            visited_rev.add(current)
            for conn in genome.connections.values():
                if conn.enabled and conn.input_node == current:
                    stack_rev.append(conn.output_node)

        return False

    # --- Main evolution loop ---

    def run(
        self,
        fitness_fn: Callable[[NEATNetwork], float],
        generations: Optional[int] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
        verbose: bool = True,
    ) -> Genome:
        """
        Run NEAT evolution.

        Args:
            fitness_fn: Function that takes a NEATNetwork and returns fitness score.
            generations: Number of generations (defaults to 100).
            callback: Optional callback(generation, stats) called each generation.
            verbose: Print progress.

        Returns:
            Best genome found.
        """
        gens = generations or 100

        if not self.population:
            self.initialize_population()

        best_ever = None
        best_ever_fitness = float("-inf")

        for gen in range(gens):
            self.generation = gen

            # Evaluate
            self.evaluate(fitness_fn)

            # Track best
            best = max(self.population, key=lambda g: g.fitness)
            if best.fitness > best_ever_fitness:
                best_ever_fitness = best.fitness
                best_ever = best.copy()

            # Stats
            fitnesses = [g.fitness for g in self.population]
            stats = {
                "generation": gen,
                "best_fitness": best.fitness,
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "min_fitness": min(fitnesses),
                "best_ever_fitness": best_ever_fitness,
                "num_species": len(self.species),
                "best_hidden_nodes": best.num_hidden,
                "best_connections": best.num_enabled_connections,
            }
            self.history.append(stats)

            if verbose and gen % 10 == 0:
                print(
                    f"Gen {gen}: best={stats['best_fitness']:.3f} "
                    f"avg={stats['avg_fitness']:.3f} "
                    f"species={stats['num_species']} "
                    f"topology={stats['best_hidden_nodes']}h/{stats['best_connections']}c"
                )

            if callback:
                callback(gen, stats)

            # Speciate
            self.speciate()

            # Reset innovation tracker for next generation
            self.innovations.reset_generation()

            # Reproduce
            self.reproduce()

        return best_ever

    # --- Utilities ---

    def get_network(self, genome: Optional[Genome] = None) -> NEATNetwork:
        """Build a NEATNetwork from a genome (or the current best)."""
        if genome is None:
            genome = max(self.population, key=lambda g: g.fitness)
        return NEATNetwork(genome, self.engine)

    def save(self, filepath: str, genome: Optional[Genome] = None):
        """Save a genome to file."""
        g = genome or max(self.population, key=lambda g: g.fitness)
        with open(filepath, "wb") as f:
            pickle.dump(g.to_dict(), f)

    def load(self, filepath: str) -> Genome:
        """Load a genome from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return Genome.from_dict(data)


# ---------------------------------------------------------------------------
# Species container
# ---------------------------------------------------------------------------

@dataclass
class Species:
    """A species of similar genomes."""
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    best_fitness: float = float("-inf")
    stagnation: int = 0
    adjusted_fitness: float = 0.0
