from npcpy.ft.sft import run_sft, load_sft_model, predict_sft, SFTConfig
from npcpy.ft.rl import (
    run_rl_training, 
    collect_traces, 
    RLConfig,
    TaskExecutor
)
from npcpy.ft.ge import GeneticEvolver, GAConfig
from npcpy.ft.model_ensembler import (
    ResponseRouter,
    create_model_genome,
    mutate_model_genome,
    crossover_model_genomes,
    evaluate_model_genome
)
from npcpy.memory.knowledge_graph import (
    kg_initial,
    kg_evolve_incremental,
    kg_sleep_process,
    kg_dream_process
)
from npcpy.npc_compiler import NPC
import random


def demo_sft():
    print("\n=== SFT DEMO ===")
    
    X_train = [
        "What is 2 + 2?",
        "What is 5 + 3?",
        "What is 10 - 4?",
        "What is 7 - 2?",
        "What is 3 * 3?",
    ]
    
    y_train = [
        "4",
        "8", 
        "6",
        "5",
        "9",
    ]
    
    config = SFTConfig(
        base_model_name="Qwen/Qwen3-0.6B",
        output_model_path="models/math_sft",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        logging_steps=1
    )
    
    print("Training SFT model on math examples...")
    model_path = run_sft(X_train, y_train, config, format_style="gemma")
    
    print("\nTesting SFT model...")
    model, tokenizer = load_sft_model(model_path)
    
    test_query = "What is 4 + 4?"
    response = predict_sft(model, tokenizer, test_query, temperature=0.1)
    print(f"Query: {test_query}")
    print(f"Response: {response}")
    
    return model_path


def demo_rl():
    print("\n=== RL DEMO ===")
    
    tasks = [
        {
            'prompt': 'Summarize: The cat sat on the mat.',
            'expected': 'cat on mat'
        },
        {
            'prompt': 'Summarize: The dog ran in the park.',
            'expected': 'dog ran park'
        },
        {
            'prompt': 'Summarize: Birds fly in the sky.',
            'expected': 'birds fly'
        },
    ]
    
    agents = [
        NPC(
            name="agent_verbose",
            primary_directive=(
                "You provide detailed explanations. "
                "Be thorough and complete."
            ),
            model="qwen3:0.6b",
            provider="ollama"
        ),
        NPC(
            name="agent_concise",
            primary_directive=(
                "You provide brief, direct answers. "
                "Be concise and to the point."
            ),
            model="qwen3:0.6b",
            provider="ollama"
        ),
    ]
    
    def reward_fn(trace):
        output = trace['final_output'].lower()
        expected = trace['task_metadata'].get('expected', '').lower()
        
        if not trace['completed']:
            return -0.5
        
        if expected in output:
            return 1.0
        
        word_overlap = sum(
            1 for word in expected.split() 
            if word in output
        )
        return word_overlap / max(len(expected.split()), 1)
    
    config = RLConfig(
        base_model_name="Qwen/Qwen3-0.6B",
        adapter_path="models/rl_adapter",
        max_iterations=3,
        num_train_epochs=5
    )
    
    print("Collecting RL traces...")
    traces = collect_traces(tasks, agents, reward_fn, config)
    
    print("\nTrace rewards:")
    for i, trace in enumerate(traces):
        print(
            f"  Trace {i}: Agent={trace['agent_name']}, "
            f"Reward={trace['reward']:.2f}"
        )
    
    return traces


def demo_genetic_knowledge_graphs():
    print("\n=== GENETIC KNOWLEDGE GRAPH EVOLUTION ===")
    
    initial_content = """
    Paris is the capital of France. France is in Europe.
    The Eiffel Tower is in Paris and is a famous landmark.
    """
    
    kg = kg_initial(
        content=initial_content,
        model="qwen3:0.6b",
        provider="ollama"
    )
    
    print(f"Initial KG: {len(kg['facts'])} facts, "
          f"{len(kg['concepts'])} concepts")
    
    def mutate_kg(kg_data):
        evolved_kg, _ = kg_sleep_process(
            kg_data,
            model="qwen3:0.6b",
            provider="ollama",
            operations_config=['prune', 'deepen']
        )
        return evolved_kg
    
    def crossover_kgs(kg1, kg2):
        child = {
            'facts': kg1['facts'][:len(kg1['facts'])//2] + 
                    kg2['facts'][len(kg2['facts'])//2:],
            'concepts': kg1['concepts'] if len(kg1['concepts']) > 
                       len(kg2['concepts']) else kg2['concepts'],
            'generation': max(kg1['generation'], kg2['generation']) + 1,
            'fact_to_concept_links': kg1.get('fact_to_concept_links', {}),
            'fact_to_fact_links': kg1.get('fact_to_fact_links', [])
        }
        return child
    
    def fitness_kg(kg_data):
        fact_count = len(kg_data.get('facts', []))
        concept_count = len(kg_data.get('concepts', []))
        
        return (fact_count * 0.5 + concept_count * 0.5) / 100.0
    
    def init_kg():
        return kg
    
    config = GAConfig(
        population_size=5,
        generations=3,
        mutation_rate=0.3
    )
    
    evolver = GeneticEvolver(
        fitness_fn=fitness_kg,
        mutate_fn=mutate_kg,
        crossover_fn=crossover_kgs,
        initialize_fn=init_kg,
        config=config
    )
    
    print("Evolving knowledge graphs...")
    best_kg = evolver.run()
    
    print(f"\nBest KG: {len(best_kg['facts'])} facts, "
          f"{len(best_kg['concepts'])} concepts")
    
    return best_kg


def demo_genetic_model_genomes():
    print("\n=== GENETIC MODEL GENOME EVOLUTION ===")
    
    specializations = ['math', 'code', 'factual']
    
    test_cases = [
        {'query': 'What is 2+2?', 'ground_truth': '4'},
        {'query': 'Write hello world', 'ground_truth': 'print'},
        {'query': 'Capital of France?', 'ground_truth': 'Paris'},
    ]
    
    router = ResponseRouter()
    
    def init_genome():
        return create_model_genome(specializations)
    
    def fitness_genome(genome):
        return evaluate_model_genome(genome, test_cases, router)
    
    config = GAConfig(
        population_size=4,
        generations=2,
        mutation_rate=0.4
    )
    
    evolver = GeneticEvolver(
        fitness_fn=fitness_genome,
        mutate_fn=mutate_model_genome,
        crossover_fn=crossover_model_genomes,
        initialize_fn=init_genome,
        config=config
    )
    
    print("Evolving model genomes...")
    best_genome = evolver.run()
    
    print(f"\nBest genome has {len(best_genome)} genes")
    for gene in best_genome:
        print(
            f"  {gene.specialization}: "
            f"triggers={gene.trigger_patterns[:2]}"
        )
    
    return best_genome


def demo_response_router():
    print("\n=== RESPONSE ROUTER DEMO ===")
    
    genome = create_model_genome(['math', 'factual'])
    router = ResponseRouter(
        fast_threshold=0.8,
        ensemble_threshold=0.6
    )
    
    test_queries = [
        "What is 5 + 5?",
        "Who invented the telephone?",
        "Explain quantum mechanics",
    ]
    
    print("Routing queries through system...")
    for query in test_queries:
        result = router.route_query(query, genome)
        
        path = (
            "FAST" if result['used_fast_path'] 
            else "ENSEMBLE" if result.get('used_ensemble')
            else "REASONING"
        )
        
        print(f"\nQuery: {query}")
        print(f"  Path: {path}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Time: {result['response_time']:.3f}s")


if __name__ == "__main__":
    print("NPC Fine-Tuning Demo")
    print("=" * 50)
    
    print("\nThis demo shows:")
    print("1. SFT: Supervised fine-tuning on task data")
    print("2. RL: Trace collection and DPO training")
    print("3. GA: Genetic evolution of knowledge graphs")
    print("4. GA: Genetic evolution of model ensembles")
    print("5. Router: Fast path vs reasoning system")
    
    demo_sft()
    
    demo_rl()
    
    demo_genetic_knowledge_graphs()
    
    demo_genetic_model_genomes()
    
    demo_response_router()
    
    print("\n" + "=" * 50)
    print("Demo complete!")