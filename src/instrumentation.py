"""
ToM-NAS Instrumentation System
Comprehensive logging, tracing, and analysis for dissertation-quality research

Components:
1. TraceLogger - Captures all reasoning steps per agent
2. EmergenceTracker - Monitors communication/norms/object evolution
3. MotifExtractor - Analyzes architectural patterns
4. ZombieInteractionRecorder - Logs detection transcripts
5. SocialWorldVisualizer - Generates network analysis
"""
import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import copy


@dataclass
class BeliefTrajectory:
    """Track belief evolution across orders"""
    order: int
    timesteps: List[int] = field(default_factory=list)
    beliefs: List[torch.Tensor] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)

    def add(self, timestep: int, belief: torch.Tensor, conf: float = 1.0):
        self.timesteps.append(timestep)
        self.beliefs.append(belief.detach().cpu())
        self.confidence.append(conf)

    def to_dict(self):
        return {
            'order': self.order,
            'timesteps': self.timesteps,
            'beliefs': [b.numpy().tolist() for b in self.beliefs],
            'confidence': self.confidence
        }


@dataclass
class ActionWithJustification:
    """Action plus reasoning explanation"""
    timestep: int
    action: str
    action_value: float
    justification: str
    belief_state: Dict
    confidence: float

    def to_dict(self):
        return asdict(self)


@dataclass
class TRNTrace:
    """Transparent RNN specific trace data"""
    hidden_states: List[torch.Tensor] = field(default_factory=list)
    gate_values: List[Dict] = field(default_factory=list)  # update, reset gates
    belief_updates: List[Dict] = field(default_factory=list)
    ontology_activations: List[torch.Tensor] = field(default_factory=list)

    def to_dict(self):
        return {
            'hidden_states': [h.numpy().tolist() for h in self.hidden_states],
            'gate_values': self.gate_values,
            'belief_updates': self.belief_updates,
            'ontology_activations': [o.numpy().tolist() for o in self.ontology_activations]
        }


@dataclass
class RSANTrace:
    """Recursive Self-Attention specific trace data"""
    attention_matrices: List[np.ndarray] = field(default_factory=list)
    recursion_depth: int = 0
    self_other_separation: float = 0.0
    belief_hierarchy: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'attention_matrices': [a.tolist() for a in self.attention_matrices],
            'recursion_depth': self.recursion_depth,
            'self_other_separation': self.self_other_separation,
            'belief_hierarchy': self.belief_hierarchy
        }


@dataclass
class TransformerTrace:
    """Transformer specific trace data"""
    attention_heads: List[np.ndarray] = field(default_factory=list)
    layer_outputs: List[torch.Tensor] = field(default_factory=list)
    communication_tokens: List[int] = field(default_factory=list)

    def to_dict(self):
        return {
            'attention_heads': [a.tolist() for a in self.attention_heads],
            'layer_outputs': [l.numpy().tolist() for l in self.layer_outputs],
            'communication_tokens': self.communication_tokens
        }


@dataclass
class AgentReasoningTrace:
    """Complete reasoning trace for one agent"""
    agent_id: int
    architecture: str
    generation: int

    # Architecture-specific traces
    trn_trace: Optional[TRNTrace] = None
    rsan_trace: Optional[RSANTrace] = None
    transformer_trace: Optional[TransformerTrace] = None

    # Universal traces
    belief_trajectories: Dict[int, BeliefTrajectory] = field(default_factory=dict)
    actions_with_justifications: List[ActionWithJustification] = field(default_factory=list)
    zombie_detection_reasoning: List[Dict] = field(default_factory=list)

    # Summary metrics
    total_timesteps: int = 0
    final_fitness: float = 0.0

    def __post_init__(self):
        if not self.belief_trajectories:
            self.belief_trajectories = {i: BeliefTrajectory(order=i) for i in range(6)}

    def to_dict(self):
        return {
            'agent_id': self.agent_id,
            'architecture': self.architecture,
            'generation': self.generation,
            'trn_trace': self.trn_trace.to_dict() if self.trn_trace else None,
            'rsan_trace': self.rsan_trace.to_dict() if self.rsan_trace else None,
            'transformer_trace': self.transformer_trace.to_dict() if self.transformer_trace else None,
            'belief_trajectories': {k: v.to_dict() for k, v in self.belief_trajectories.items()},
            'actions_with_justifications': [a.to_dict() for a in self.actions_with_justifications],
            'zombie_detection_reasoning': self.zombie_detection_reasoning,
            'total_timesteps': self.total_timesteps,
            'final_fitness': self.final_fitness
        }


class TraceLogger:
    """Captures all reasoning steps for every agent"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.traces_dir = os.path.join(output_dir, 'traces')
        os.makedirs(self.traces_dir, exist_ok=True)

        self.current_traces: Dict[int, AgentReasoningTrace] = {}
        self.generation_traces: List[Dict[int, AgentReasoningTrace]] = []

    def start_agent_trace(self, agent_id: int, architecture: str, generation: int):
        """Begin tracing an agent"""
        trace = AgentReasoningTrace(
            agent_id=agent_id,
            architecture=architecture,
            generation=generation
        )

        if architecture == 'TRN':
            trace.trn_trace = TRNTrace()
        elif architecture == 'RSAN':
            trace.rsan_trace = RSANTrace()
        elif architecture == 'Transformer':
            trace.transformer_trace = TransformerTrace()

        self.current_traces[agent_id] = trace

    def log_forward_pass(self, agent_id: int, model: nn.Module,
                         input_tensor: torch.Tensor, output: Dict,
                         timestep: int):
        """Log a single forward pass with full trace"""
        if agent_id not in self.current_traces:
            return

        trace = self.current_traces[agent_id]
        trace.total_timesteps = timestep + 1

        # Extract architecture-specific data
        if trace.trn_trace is not None and hasattr(model, 'computation_trace'):
            if output.get('final_hidden') is not None:
                trace.trn_trace.hidden_states.append(output['final_hidden'].detach().cpu())
            # Log belief update
            if 'beliefs' in output:
                trace.trn_trace.belief_updates.append({
                    'timestep': timestep,
                    'belief_mean': output['beliefs'].mean().item(),
                    'belief_std': output['beliefs'].std().item()
                })
                trace.trn_trace.ontology_activations.append(output['beliefs'].detach().cpu())

        elif trace.rsan_trace is not None:
            if 'attention_patterns' in output and output['attention_patterns']:
                for attn in output['attention_patterns']:
                    if isinstance(attn, torch.Tensor):
                        trace.rsan_trace.attention_matrices.append(attn.detach().cpu().numpy())
            # Estimate recursion depth from hidden states
            if 'hidden_states' in output:
                trace.rsan_trace.recursion_depth = max(
                    trace.rsan_trace.recursion_depth,
                    output['hidden_states'].shape[1] if len(output['hidden_states'].shape) > 1 else 1
                )

        elif trace.transformer_trace is not None:
            if 'hidden_states' in output:
                trace.transformer_trace.layer_outputs.append(output['hidden_states'].detach().cpu())
            if 'message_tokens' in output:
                trace.transformer_trace.communication_tokens.extend(
                    output['message_tokens'].tolist() if isinstance(output['message_tokens'], torch.Tensor)
                    else [output['message_tokens']]
                )

        # Log belief trajectories (universal)
        if 'beliefs' in output:
            beliefs = output['beliefs'].detach().cpu()
            # Order 0 is just the raw belief
            trace.belief_trajectories[0].add(timestep, beliefs, 1.0)
            # Higher orders show decreasing confidence
            for order in range(1, 6):
                conf = 0.7 ** order
                trace.belief_trajectories[order].add(timestep, beliefs * conf, conf)

    def log_action(self, agent_id: int, timestep: int, action: str,
                   action_value: float, belief_state: Dict, justification: str):
        """Log an action with its justification"""
        if agent_id not in self.current_traces:
            return

        self.current_traces[agent_id].actions_with_justifications.append(
            ActionWithJustification(
                timestep=timestep,
                action=action,
                action_value=action_value,
                justification=justification,
                belief_state=belief_state,
                confidence=action_value
            )
        )

    def log_zombie_detection(self, agent_id: int, suspect_id: int,
                             is_zombie: bool, detected: bool, reasoning: str):
        """Log zombie detection attempt with reasoning"""
        if agent_id not in self.current_traces:
            return

        self.current_traces[agent_id].zombie_detection_reasoning.append({
            'suspect_id': suspect_id,
            'is_zombie': is_zombie,
            'detected': detected,
            'correct': detected == is_zombie,
            'reasoning': reasoning
        })

    def finalize_agent(self, agent_id: int, fitness: float):
        """Finalize trace for an agent"""
        if agent_id in self.current_traces:
            self.current_traces[agent_id].final_fitness = fitness

    def save_generation(self, generation: int):
        """Save all traces for a generation"""
        gen_dir = os.path.join(self.traces_dir, f'generation_{generation}')
        os.makedirs(gen_dir, exist_ok=True)

        # Save as pickle for full fidelity
        traces_data = {aid: trace.to_dict() for aid, trace in self.current_traces.items()}
        with open(os.path.join(gen_dir, 'reasoning_traces.pkl'), 'wb') as f:
            pickle.dump(traces_data, f)

        # Save summary as JSON
        summary = {
            'generation': generation,
            'num_agents': len(self.current_traces),
            'agents': [
                {
                    'id': t.agent_id,
                    'architecture': t.architecture,
                    'fitness': t.final_fitness,
                    'timesteps': t.total_timesteps,
                    'zombie_detections': len(t.zombie_detection_reasoning),
                    'correct_detections': sum(1 for d in t.zombie_detection_reasoning if d['correct'])
                }
                for t in self.current_traces.values()
            ]
        }
        with open(os.path.join(gen_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        self.generation_traces.append(copy.deepcopy(self.current_traces))
        self.current_traces.clear()

        print(f"  [Traces saved: generation {generation}]")


@dataclass
class CommunicationProtocol:
    """Tracks emergent communication patterns"""
    message_types: Dict[str, int] = field(default_factory=dict)
    semantic_drift: List[float] = field(default_factory=list)
    symbolic_conventions: List[str] = field(default_factory=list)
    deception_markers: List[str] = field(default_factory=list)


@dataclass
class ObjectConsensus:
    """Tracks arbitrary interpretable object consensus"""
    object_id: int
    proposed_meanings: Dict[int, str] = field(default_factory=dict)
    consensus_strength: float = 0.0
    consensus_meaning: Optional[str] = None
    generations_to_consensus: int = 0


@dataclass
class EmergentNorm:
    """Tracks norm emergence"""
    norm_description: str
    compliance_rate: float = 0.0
    enforcement_strength: float = 0.0
    violators_punished: int = 0
    generations_active: int = 0


class EmergenceTracker:
    """Monitors emergent phenomena: communication, norms, object meanings"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.emergence_dir = os.path.join(output_dir, 'emergence')
        os.makedirs(self.emergence_dir, exist_ok=True)

        self.communication = CommunicationProtocol()
        self.object_consensuses: Dict[int, ObjectConsensus] = {}
        self.norms: List[EmergentNorm] = []
        self.knowledge_transfers: List[Dict] = []

        self.generation_logs: List[Dict] = []

    def log_communication(self, sender_id: int, receiver_id: int,
                          message: Any, meaning: str, generation: int):
        """Log a communication event"""
        msg_type = str(type(message).__name__)
        self.communication.message_types[msg_type] = \
            self.communication.message_types.get(msg_type, 0) + 1

        # Track symbolic conventions
        if isinstance(message, (int, str)):
            convention = f"agent_{sender_id}_uses_{message}_for_{meaning}"
            if convention not in self.communication.symbolic_conventions:
                self.communication.symbolic_conventions.append(convention)

    def log_deception_attempt(self, agent_id: int, target_id: int,
                              false_message: str, true_state: str, generation: int):
        """Log a deception event"""
        marker = f"gen{generation}_agent{agent_id}_deceived_{target_id}"
        self.communication.deception_markers.append(marker)

    def log_object_interpretation(self, object_id: int, agent_id: int,
                                  meaning: str, generation: int):
        """Log an agent's interpretation of an arbitrary object"""
        if object_id not in self.object_consensuses:
            self.object_consensuses[object_id] = ObjectConsensus(object_id=object_id)

        self.object_consensuses[object_id].proposed_meanings[agent_id] = meaning

        # Check for consensus
        meanings = list(self.object_consensuses[object_id].proposed_meanings.values())
        if meanings:
            from collections import Counter
            most_common = Counter(meanings).most_common(1)[0]
            consensus_strength = most_common[1] / len(meanings)
            self.object_consensuses[object_id].consensus_strength = consensus_strength
            if consensus_strength > 0.7:
                self.object_consensuses[object_id].consensus_meaning = most_common[0]

    def log_norm_observation(self, norm_desc: str, complied: bool,
                             enforced: bool, generation: int):
        """Log norm-related behavior"""
        # Find or create norm
        norm = next((n for n in self.norms if n.norm_description == norm_desc), None)
        if norm is None:
            norm = EmergentNorm(norm_description=norm_desc)
            self.norms.append(norm)

        norm.generations_active = generation - norm.generations_active + 1
        if complied:
            norm.compliance_rate = (norm.compliance_rate + 1) / 2
        if enforced:
            norm.violators_punished += 1
            norm.enforcement_strength = min(1.0, norm.enforcement_strength + 0.1)

    def log_knowledge_transfer(self, parent_id: int, child_id: int,
                               beliefs_transferred: List, accuracy: float):
        """Log intergenerational knowledge transfer"""
        self.knowledge_transfers.append({
            'parent': parent_id,
            'child': child_id,
            'num_beliefs': len(beliefs_transferred),
            'accuracy': accuracy
        })

    def save_generation(self, generation: int):
        """Save emergence data for generation"""
        log = {
            'generation': generation,
            'communication': {
                'message_types': dict(self.communication.message_types),
                'num_conventions': len(self.communication.symbolic_conventions),
                'num_deceptions': len(self.communication.deception_markers)
            },
            'objects': {
                oid: {
                    'consensus_strength': obj.consensus_strength,
                    'consensus_meaning': obj.consensus_meaning,
                    'num_interpretations': len(obj.proposed_meanings)
                }
                for oid, obj in self.object_consensuses.items()
            },
            'norms': [
                {
                    'description': n.norm_description,
                    'compliance': n.compliance_rate,
                    'enforcement': n.enforcement_strength,
                    'active_generations': n.generations_active
                }
                for n in self.norms
            ],
            'knowledge_transfers': len(self.knowledge_transfers)
        }

        self.generation_logs.append(log)

        # Save to file
        with open(os.path.join(self.emergence_dir, f'generation_{generation}.json'), 'w') as f:
            json.dump(log, f, indent=2)


@dataclass
class ToMMotif:
    """A discovered Theory of Mind architectural motif"""
    name: str
    description: str
    architecture: str
    layer_indices: List[int] = field(default_factory=list)
    weight_pattern_hash: str = ""
    activation_signature: List[float] = field(default_factory=list)
    tom_capability: str = ""  # What ToM ability this enables
    effectiveness: float = 0.0


class MotifExtractor:
    """Analyzes and catalogs architectural patterns enabling ToM"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.motifs_dir = os.path.join(output_dir, 'motifs')
        os.makedirs(self.motifs_dir, exist_ok=True)

        self.discovered_motifs: List[ToMMotif] = []
        self.architecture_specializations: Dict[str, Dict] = {
            'TRN': {'strengths': [], 'weaknesses': []},
            'RSAN': {'strengths': [], 'weaknesses': []},
            'Transformer': {'strengths': [], 'weaknesses': []},
            'Hybrid': {'strengths': [], 'weaknesses': []}
        }
        self.generation_analyses: List[Dict] = []

    def analyze_model(self, model: nn.Module, architecture: str,
                      agent_id: int, performance: Dict):
        """Analyze a model for ToM-relevant patterns"""

        # Extract weight statistics
        weight_stats = {}
        for name, param in model.named_parameters():
            weight_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'norm': param.data.norm().item()
            }

        # Look for perspective-taking circuits (high variance in certain layers)
        perspective_layers = []
        for name, stats in weight_stats.items():
            if 'belief' in name.lower() or 'projection' in name.lower():
                if stats['std'] > 0.1:  # Active variance suggests learned patterns
                    perspective_layers.append(name)

        if perspective_layers:
            motif = ToMMotif(
                name=f"PerspectiveTaking_{architecture}_{agent_id}",
                description="Layers showing high variance in belief-related projections",
                architecture=architecture,
                layer_indices=[i for i, n in enumerate(weight_stats.keys()) if n in perspective_layers],
                tom_capability="perspective_taking",
                effectiveness=performance.get('sally_anne', 0.0)
            )
            self.discovered_motifs.append(motif)

        # Track architecture specializations
        if performance.get('sally_anne', 0) > 0.7:
            self.architecture_specializations[architecture]['strengths'].append('false_belief')
        if performance.get('zombie_detection', 0) > 0.8:
            self.architecture_specializations[architecture]['strengths'].append('zombie_detection')
        if performance.get('higher_order', {}).get(5, 0) > 0.5:
            self.architecture_specializations[architecture]['strengths'].append('5th_order_tom')

    def compare_architectures(self, agents: List[Tuple[str, Dict]]):
        """Compare performance across architecture types"""
        arch_performance = defaultdict(list)

        for arch, perf in agents:
            arch_performance[arch].append(perf)

        comparison = {}
        for arch, perfs in arch_performance.items():
            if perfs:
                comparison[arch] = {
                    'avg_fitness': np.mean([p.get('fitness', 0) for p in perfs]),
                    'avg_sally_anne': np.mean([p.get('sally_anne', 0) for p in perfs]),
                    'avg_zombie': np.mean([p.get('zombie_detection', 0) for p in perfs]),
                    'count': len(perfs)
                }

        return comparison

    def save_generation(self, generation: int, agents_data: List):
        """Save motif analysis for generation"""
        analysis = {
            'generation': generation,
            'num_motifs_discovered': len(self.discovered_motifs),
            'architecture_specializations': self.architecture_specializations,
            'motifs': [
                {
                    'name': m.name,
                    'architecture': m.architecture,
                    'capability': m.tom_capability,
                    'effectiveness': m.effectiveness
                }
                for m in self.discovered_motifs[-10:]  # Last 10 motifs
            ]
        }

        self.generation_analyses.append(analysis)

        with open(os.path.join(self.motifs_dir, f'generation_{generation}.json'), 'w') as f:
            json.dump(analysis, f, indent=2)

    def generate_motif_atlas(self):
        """Generate complete motif atlas"""
        atlas = {
            'total_motifs': len(self.discovered_motifs),
            'by_architecture': defaultdict(list),
            'by_capability': defaultdict(list),
            'specializations': self.architecture_specializations
        }

        for motif in self.discovered_motifs:
            atlas['by_architecture'][motif.architecture].append({
                'name': motif.name,
                'capability': motif.tom_capability,
                'effectiveness': motif.effectiveness
            })
            atlas['by_capability'][motif.tom_capability].append({
                'name': motif.name,
                'architecture': motif.architecture,
                'effectiveness': motif.effectiveness
            })

        # Convert defaultdicts
        atlas['by_architecture'] = dict(atlas['by_architecture'])
        atlas['by_capability'] = dict(atlas['by_capability'])

        with open(os.path.join(self.motifs_dir, 'motif_atlas.json'), 'w') as f:
            json.dump(atlas, f, indent=2)

        return atlas


@dataclass
class ZombieInteractionRound:
    """One round of zombie testing interaction"""
    tester_probe: str
    tested_response: str
    tester_reasoning: str
    tested_internal_state: Dict
    detection_confidence: float


@dataclass
class ZombieInteraction:
    """Complete zombie detection interaction"""
    generation: int
    test_type: str
    tester_agent: int
    tested_agent: int
    is_zombie: bool
    rounds: List[ZombieInteractionRound] = field(default_factory=list)
    final_verdict: bool = False
    verdict_correct: bool = False
    detection_strategy: str = ""


class ZombieInteractionRecorder:
    """Records detailed zombie detection transcripts"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.zombie_dir = os.path.join(output_dir, 'zombie_transcripts')
        os.makedirs(self.zombie_dir, exist_ok=True)

        self.interactions: List[ZombieInteraction] = []
        self.generation_interactions: Dict[int, List[ZombieInteraction]] = defaultdict(list)

        self.detection_strategies: Dict[str, int] = defaultdict(int)

    def start_interaction(self, generation: int, test_type: str,
                         tester_id: int, tested_id: int, is_zombie: bool):
        """Start recording a zombie detection interaction"""
        interaction = ZombieInteraction(
            generation=generation,
            test_type=test_type,
            tester_agent=tester_id,
            tested_agent=tested_id,
            is_zombie=is_zombie
        )
        self.interactions.append(interaction)
        return len(self.interactions) - 1  # Return interaction ID

    def add_round(self, interaction_id: int, probe: str, response: str,
                  reasoning: str, internal_state: Dict, confidence: float):
        """Add a round to an interaction"""
        if interaction_id < len(self.interactions):
            self.interactions[interaction_id].rounds.append(
                ZombieInteractionRound(
                    tester_probe=probe,
                    tested_response=response,
                    tester_reasoning=reasoning,
                    tested_internal_state=internal_state,
                    detection_confidence=confidence
                )
            )

    def finalize_interaction(self, interaction_id: int, verdict: bool,
                            strategy: str):
        """Finalize an interaction with verdict"""
        if interaction_id < len(self.interactions):
            interaction = self.interactions[interaction_id]
            interaction.final_verdict = verdict
            interaction.verdict_correct = (verdict == interaction.is_zombie)
            interaction.detection_strategy = strategy

            self.detection_strategies[strategy] += 1
            self.generation_interactions[interaction.generation].append(interaction)

    def save_generation(self, generation: int):
        """Save zombie transcripts for generation"""
        gen_interactions = self.generation_interactions.get(generation, [])

        transcripts = []
        for interaction in gen_interactions:
            transcript = {
                'test_type': interaction.test_type,
                'tester': interaction.tester_agent,
                'tested': interaction.tested_agent,
                'is_zombie': interaction.is_zombie,
                'rounds': [
                    {
                        'probe': r.tester_probe,
                        'response': r.tested_response,
                        'reasoning': r.tester_reasoning,
                        'confidence': r.detection_confidence
                    }
                    for r in interaction.rounds
                ],
                'verdict': interaction.final_verdict,
                'correct': interaction.verdict_correct,
                'strategy': interaction.detection_strategy
            }
            transcripts.append(transcript)

        summary = {
            'generation': generation,
            'total_interactions': len(gen_interactions),
            'correct_detections': sum(1 for i in gen_interactions if i.verdict_correct),
            'accuracy': sum(1 for i in gen_interactions if i.verdict_correct) / max(len(gen_interactions), 1),
            'strategies_used': dict(self.detection_strategies),
            'transcripts': transcripts[:20]  # Save first 20 full transcripts
        }

        with open(os.path.join(self.zombie_dir, f'generation_{generation}.json'), 'w') as f:
            json.dump(summary, f, indent=2)


class SocialWorldVisualizer:
    """Generates social network analysis and visualizations"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'social_viz')
        os.makedirs(self.viz_dir, exist_ok=True)

        self.snapshots: List[Dict] = []

    def capture_snapshot(self, world, generation: int, timestep: int):
        """Capture social world state"""
        snapshot = {
            'generation': generation,
            'timestep': timestep,
            'num_agents': world.num_agents,
            'num_alive': sum(1 for a in world.agents if a.alive),
            'zombies': [a.id for a in world.agents if a.is_zombie],

            # Reputation matrix
            'reputation_matrix': [
                [a.reputation.get(j, 0.5) for j in range(world.num_agents)]
                for a in world.agents
            ],

            # Resource distribution
            'resources': [a.resources for a in world.agents],
            'energy': [a.energy for a in world.agents],

            # Coalition structure
            'coalitions': {
                cid: list(members)
                for cid, members in world.coalitions.items()
            },

            # Agent coalition membership
            'agent_coalitions': [a.coalition for a in world.agents]
        }

        self.snapshots.append(snapshot)
        return snapshot

    def compute_network_metrics(self, snapshot: Dict) -> Dict:
        """Compute network analysis metrics"""
        rep_matrix = np.array(snapshot['reputation_matrix'])
        n = rep_matrix.shape[0]

        metrics = {
            'avg_reputation': float(np.mean(rep_matrix)),
            'reputation_variance': float(np.var(rep_matrix)),
            'reciprocity': float(np.corrcoef(rep_matrix.flatten(), rep_matrix.T.flatten())[0, 1]),
            'clustering': 0.0,  # Would need proper graph computation
            'num_coalitions': len(snapshot['coalitions']),
            'largest_coalition': max([len(m) for m in snapshot['coalitions'].values()], default=0),
            'gini_resources': self._compute_gini(snapshot['resources'])
        }

        return metrics

    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient for inequality"""
        values = sorted(values)
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0.0
        cumulative = np.cumsum(values)
        return float((2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1]))

    def save_generation(self, generation: int):
        """Save visualization data for generation"""
        gen_snapshots = [s for s in self.snapshots if s['generation'] == generation]

        if not gen_snapshots:
            return

        # Compute metrics for each snapshot
        metrics_over_time = [self.compute_network_metrics(s) for s in gen_snapshots]

        viz_data = {
            'generation': generation,
            'num_snapshots': len(gen_snapshots),
            'final_snapshot': gen_snapshots[-1] if gen_snapshots else None,
            'metrics_trajectory': metrics_over_time,
            'summary': {
                'avg_reputation_trend': [m['avg_reputation'] for m in metrics_over_time],
                'coalition_dynamics': [m['num_coalitions'] for m in metrics_over_time],
                'inequality_trend': [m['gini_resources'] for m in metrics_over_time]
            }
        }

        with open(os.path.join(self.viz_dir, f'generation_{generation}.json'), 'w') as f:
            json.dump(viz_data, f, indent=2)


class InstrumentationSuite:
    """Complete instrumentation suite combining all components"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.trace_logger = TraceLogger(output_dir)
        self.emergence_tracker = EmergenceTracker(output_dir)
        self.motif_extractor = MotifExtractor(output_dir)
        self.zombie_recorder = ZombieInteractionRecorder(output_dir)
        self.social_visualizer = SocialWorldVisualizer(output_dir)

        print(f"Instrumentation initialized in: {output_dir}")

    def save_generation(self, generation: int):
        """Save all instrumentation data for generation"""
        self.trace_logger.save_generation(generation)
        self.emergence_tracker.save_generation(generation)
        self.motif_extractor.save_generation(generation, [])
        self.zombie_recorder.save_generation(generation)
        self.social_visualizer.save_generation(generation)

        print(f"  [All instrumentation saved: generation {generation}]")

    def generate_final_report(self):
        """Generate comprehensive final report"""
        report_dir = os.path.join(self.output_dir, 'final_report')
        os.makedirs(report_dir, exist_ok=True)

        # Generate motif atlas
        atlas = self.motif_extractor.generate_motif_atlas()

        # Compile emergence timeline
        emergence_timeline = self.emergence_tracker.generation_logs

        # Compile zombie detection strategies
        zombie_strategies = dict(self.zombie_recorder.detection_strategies)

        final_report = {
            'generated_at': datetime.now().isoformat(),
            'output_directory': self.output_dir,
            'motif_atlas_summary': {
                'total_motifs': atlas['total_motifs'],
                'architectures_analyzed': list(atlas['by_architecture'].keys()),
                'capabilities_discovered': list(atlas['by_capability'].keys())
            },
            'emergence_summary': {
                'total_generations_tracked': len(emergence_timeline),
                'communication_conventions': len(self.emergence_tracker.communication.symbolic_conventions),
                'deception_events': len(self.emergence_tracker.communication.deception_markers),
                'norms_emerged': len(self.emergence_tracker.norms)
            },
            'zombie_detection_summary': {
                'total_interactions': len(self.zombie_recorder.interactions),
                'strategies_discovered': zombie_strategies,
                'overall_accuracy': sum(1 for i in self.zombie_recorder.interactions if i.verdict_correct) /
                                   max(len(self.zombie_recorder.interactions), 1)
            }
        }

        with open(os.path.join(report_dir, 'final_report.json'), 'w') as f:
            json.dump(final_report, f, indent=2)

        print(f"\nFinal report generated in: {report_dir}")
        return final_report
