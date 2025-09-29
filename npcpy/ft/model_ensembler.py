import time
import copy
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from npcpy.llm_funcs import get_llm_response

try:
    from npcpy.ft.sft import predict_sft, load_sft_model
except:
    pass

@dataclass
class ModelGene:
    """
    Represents a specialized model with trigger patterns 
    and confidence threshold
    """
    sft_path: Optional[str] = None
    rl_path: Optional[str] = None
    base_model: str = "Qwen/Qwen3-0.6B"
    specialization: str = "general"
    trigger_patterns: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7


def generate_trigger_patterns(specialization: str) -> List[str]:
    """
    Generate trigger patterns for a given specialization domain
    """
    patterns = {
        'math': ['calculate', 'solve', 'equation', 'number'],
        'code': ['function', 'class', 'bug', 'debug', 'code'],
        'creative': ['story', 'poem', 'creative', 'imagine'],
        'factual': ['what is', 'who is', 'when did', 'where is'],
        'analysis': ['analyze', 'compare', 'evaluate', 'assess']
    }
    
    return patterns.get(specialization, ['general'])


def create_model_genome(
    specializations: List[str],
    base_model: str = "Qwen/Qwen3-0.6B"
) -> List[ModelGene]:
    """
    Initialize a genome of specialized models
    """
    genome = []
    
    for spec in specializations:
        gene = ModelGene(
            base_model=base_model,
            specialization=spec,
            trigger_patterns=generate_trigger_patterns(spec),
            confidence_threshold=random.uniform(0.6, 0.9)
        )
        genome.append(gene)
    
    return genome


def mutate_model_genome(
    genome: List[ModelGene],
    mutation_type: str = 'random'
) -> List[ModelGene]:
    """
    Apply genetic mutation to model genome
    """
    new_genome = copy.deepcopy(genome)
    
    mutations = [
        'adjust_threshold',
        'add_trigger',
        'remove_gene',
        'duplicate_gene'
    ]
    
    if mutation_type == 'random':
        mutation_type = random.choice(mutations)
    
    if mutation_type == 'adjust_threshold':
        gene = random.choice(new_genome)
        gene.confidence_threshold += random.uniform(-0.1, 0.1)
        gene.confidence_threshold = max(
            0.5,
            min(0.95, gene.confidence_threshold)
        )
    
    elif mutation_type == 'add_trigger':
        gene = random.choice(new_genome)
        new_trigger = f"pattern_{random.randint(1, 100)}"
        if new_trigger not in gene.trigger_patterns:
            gene.trigger_patterns.append(new_trigger)
    
    elif mutation_type == 'remove_gene' and len(new_genome) > 1:
        new_genome.pop(random.randint(0, len(new_genome) - 1))
    
    elif mutation_type == 'duplicate_gene':
        gene = random.choice(new_genome)
        new_gene = copy.deepcopy(gene)
        new_gene.specialization = f"{gene.specialization}_variant"
        new_genome.append(new_gene)
    
    return new_genome


def crossover_model_genomes(
    genome1: List[ModelGene],
    genome2: List[ModelGene]
) -> List[ModelGene]:
    """
    Crossover two model genomes to create child genome
    """
    if not genome1 or not genome2:
        return genome1 or genome2
    
    split = random.randint(1, min(len(genome1), len(genome2)) - 1)
    
    child = genome1[:split] + genome2[split:]
    
    return child


def evaluate_model_genome(
    genome: List[ModelGene],
    test_cases: List[Dict[str, Any]],
    router: 'ResponseRouter'
) -> float:
    """
    Evaluate fitness of a model genome based on accuracy, 
    speed and efficiency
    """
    correct = 0
    total_time = 0
    fast_responses = 0
    
    for test_case in test_cases:
        result = router.route_query(
            test_case['query'],
            genome,
            test_case.get('ground_truth')
        )
        
        if result['correct']:
            correct += 1
        
        total_time += result['response_time']
        
        if result['used_fast_path']:
            fast_responses += 1
    
    accuracy = correct / len(test_cases)
    speed_bonus = fast_responses / len(test_cases)
    efficiency = 1.0 / (total_time / len(test_cases))
    
    fitness = (
        accuracy * 0.6 +
        speed_bonus * 0.2 +
        efficiency * 0.2
    )
    
    return fitness


class ResponseRouter:
    """
    Routes queries through fast path, ensemble or full reasoning 
    based on confidence thresholds
    """
    def __init__(
        self,
        fast_threshold: float = 0.8,
        ensemble_threshold: float = 0.6
    ):
        self.fast_threshold = fast_threshold
        self.ensemble_threshold = ensemble_threshold
        self.response_cache = {}
    
    def route_query(
        self,
        query: str,
        genome: List[ModelGene],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route query through system 1 fast path, 
        ensemble or system 2 reasoning
        """
        start_time = time.time()
        
        fast_response = self._try_fast_path(query, genome)
        
        if fast_response and fast_response['confidence'] > (
            self.fast_threshold
        ):
            response_time = time.time() - start_time
            
            return {
                'response': fast_response['answer'],
                'confidence': fast_response['confidence'],
                'used_fast_path': True,
                'response_time': response_time,
                'correct': (
                    ground_truth is None or 
                    self._check_correctness(
                        fast_response['answer'],
                        ground_truth
                    )
                )
            }
        
        ensemble_response = self._try_ensemble(query, genome)
        
        if ensemble_response['confidence'] > (
            self.ensemble_threshold
        ):
            response_time = time.time() - start_time
            
            return {
                'response': ensemble_response['answer'],
                'confidence': ensemble_response['confidence'],
                'used_fast_path': False,
                'used_ensemble': True,
                'response_time': response_time,
                'correct': (
                    ground_truth is None or
                    self._check_correctness(
                        ensemble_response['answer'],
                        ground_truth
                    )
                )
            }
        
        full_response = self._full_reasoning(query)
        response_time = time.time() - start_time
        
        return {
            'response': full_response,
            'confidence': 0.5,
            'used_fast_path': False,
            'used_ensemble': False,
            'response_time': response_time,
            'correct': (
                ground_truth is None or
                self._check_correctness(
                    full_response,
                    ground_truth
                )
            )
        }
    
    def _try_fast_path(
        self,
        query: str,
        genome: List[ModelGene]
    ) -> Optional[Dict[str, Any]]:
        """
        Try fast system 1 gut reaction using pattern matching
        """
        query_lower = query.lower()
        
        for gene in genome:
            if any(
                pattern in query_lower 
                for pattern in gene.trigger_patterns
            ):
                if gene.sft_path:
                    model, tokenizer = load_sft_model(gene.sft_path)
                    
                    response = predict_sft(
                        model,
                        tokenizer,
                        query,
                        temperature=0.1
                    )
                    
                    return {
                        'answer': response,
                        'confidence': gene.confidence_threshold
                    }
        
        return None
    
    def _try_ensemble(
        self,
        query: str,
        genome: List[ModelGene]
    ) -> Dict[str, Any]:
        """
        Try ensemble voting across specialized models
        """
        responses = []
        
        for gene in genome:
            if gene.sft_path or gene.rl_path:
                model_path = gene.rl_path or gene.sft_path
                
                model, tokenizer = load_sft_model(model_path)
                
                response = predict_sft(
                    model,
                    tokenizer,
                    query,
                    temperature=0.3
                )
                
                responses.append({
                    'answer': response,
                    'weight': gene.confidence_threshold
                })
        
        if not responses:
            return {'answer': '', 'confidence': 0.0}
        
        best_response = max(responses, key=lambda x: x['weight'])
        
        avg_confidence = sum(
            r['weight'] for r in responses
        ) / len(responses)
        
        return {
            'answer': best_response['answer'],
            'confidence': avg_confidence
        }
    
    def _full_reasoning(
        self,
        query: str,
        model: str = "qwen3:1.7b",
        provider: str = "ollama"
    ) -> str:
        """
        Fall back to full system 2 reasoning
        """
        response = get_llm_response(
            query,
            model=model,
            provider=provider
        )
        
        return response.get('response', '')
    
    def _check_correctness(
        self,
        response: str,
        ground_truth: str
    ) -> bool:
        """
        Check if response matches ground truth
        """
        response_lower = response.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        return response_lower == truth_lower or (
            truth_lower in response_lower
        )