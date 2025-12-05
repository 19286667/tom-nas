"""
Transparency Tools for ToM-NAS
Trace extraction, belief visualization, and interpretability analysis

These tools ensure we can verify genuine ToM vs pattern matching by:
1. Extracting computation traces from architectures
2. Visualizing belief evolution over time
3. Analyzing attention patterns for ToM reasoning
4. Generating interpretable explanations
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json


@dataclass
class ComputationStep:
    """Single step in computation trace."""
    layer: str
    timestamp: int
    input_state: List[float]
    output_state: List[float]
    gate_values: Optional[Dict[str, List[float]]] = None
    attention_weights: Optional[List[List[float]]] = None
    interpretation: str = ""


@dataclass
class BeliefTrace:
    """Trace of belief evolution."""
    agent_id: int
    order: int
    target_chain: List[int]
    timestamps: List[int]
    belief_vectors: List[List[float]]
    confidences: List[float]
    sources: List[str]


class TraceExtractor:
    """
    Extract computation traces from ToM architectures.

    Works with TRN, RSAN, and Transformer architectures
    to provide full transparency into reasoning process.
    """

    def __init__(self):
        self.traces: List[ComputationStep] = []

    def extract_trn_trace(self, model: nn.Module, input_data: torch.Tensor) -> List[ComputationStep]:
        """Extract trace from Transparent RNN."""
        traces = []

        # Forward with trace recording
        model.eval()
        if hasattr(model, 'record_trace'):
            model.record_trace = True

        with torch.no_grad():
            output = model(input_data, return_trace=True)

        # Get recorded trace
        if hasattr(model, 'computation_trace'):
            for trace in model.computation_trace:
                traces.append(ComputationStep(
                    layer=trace.layer_name,
                    timestamp=trace.timestamp,
                    input_state=trace.input_state.cpu().tolist() if torch.is_tensor(trace.input_state) else [],
                    output_state=trace.output_state.cpu().tolist() if torch.is_tensor(trace.output_state) else [],
                    gate_values={k: v.cpu().tolist() for k, v in trace.gates.items()} if trace.gates else None
                ))

        return traces

    def extract_rsan_trace(self, model: nn.Module, input_data: torch.Tensor) -> List[ComputationStep]:
        """Extract trace from Recursive Self-Attention Network."""
        traces = []

        model.eval()
        if hasattr(model, 'record_attention'):
            model.record_attention = True

        with torch.no_grad():
            output = model(input_data, return_attention=True)

        # Get attention patterns
        if hasattr(model, 'attention_patterns'):
            for depth, attn in enumerate(model.attention_patterns):
                traces.append(ComputationStep(
                    layer=f'recursion_depth_{depth}',
                    timestamp=depth,
                    input_state=[],
                    output_state=[],
                    attention_weights=attn.cpu().tolist() if torch.is_tensor(attn) else attn,
                    interpretation=f"Recursion level {depth} - modeling {'self' if depth == 0 else f'{depth}-order beliefs'}"
                ))

        return traces

    def extract_belief_trace(self, belief_store, order: int,
                           target_chain: List[int]) -> BeliefTrace:
        """Extract belief evolution trace from belief store."""
        trajectory = belief_store.extract_trajectory(order, target_chain)

        return BeliefTrace(
            agent_id=belief_store.owner_id,
            order=order,
            target_chain=target_chain,
            timestamps=[t['timestamp'] for t in trajectory],
            belief_vectors=[t['vec'] for t in trajectory],
            confidences=[t['conf'] for t in trajectory],
            sources=[t['source'] for t in trajectory]
        )

    def to_json(self, traces: List[ComputationStep]) -> str:
        """Convert traces to JSON for export/visualization."""
        return json.dumps([{
            'layer': t.layer,
            'timestamp': t.timestamp,
            'input_size': len(t.input_state),
            'output_size': len(t.output_state),
            'has_gates': t.gate_values is not None,
            'has_attention': t.attention_weights is not None,
            'interpretation': t.interpretation
        } for t in traces], indent=2)


class BeliefVisualizer:
    """
    Visualize belief states and evolution.

    Generates data structures suitable for visualization
    (actual rendering would be done by frontend/GUI).
    """

    def __init__(self, ontology_dim: int = 181):
        self.ontology_dim = ontology_dim
        self.layer_boundaries = [0, 15, 39, 69, 93, 114, 139, 157, 169, 181]
        self.layer_names = [
            'Biological', 'Affective', 'Motivational', 'Cognitive',
            'Self', 'Social Cognition', 'Values/Beliefs', 'Contextual', 'Existential'
        ]

    def prepare_belief_heatmap(self, belief_vector: torch.Tensor) -> Dict[str, Any]:
        """Prepare belief vector for heatmap visualization."""
        if torch.is_tensor(belief_vector):
            belief_vector = belief_vector.cpu().numpy()

        # Organize by ontology layer
        layers_data = {}
        for i, name in enumerate(self.layer_names):
            start = self.layer_boundaries[i]
            end = self.layer_boundaries[i + 1]
            if end <= len(belief_vector):
                layer_values = belief_vector[start:end]
                layers_data[name] = {
                    'values': layer_values.tolist(),
                    'mean': float(np.mean(layer_values)),
                    'std': float(np.std(layer_values)),
                    'max': float(np.max(layer_values)),
                    'min': float(np.min(layer_values))
                }

        return {
            'layers': layers_data,
            'total_dim': len(belief_vector),
            'layer_names': self.layer_names
        }

    def prepare_belief_evolution(self, belief_traces: List[List[float]],
                                timestamps: List[int]) -> Dict[str, Any]:
        """Prepare belief evolution for line chart visualization."""
        if not belief_traces:
            return {'data': [], 'timestamps': []}

        # Convert to numpy for easier processing
        traces_array = np.array(belief_traces)

        # Calculate statistics over time
        evolution = {
            'timestamps': timestamps,
            'layer_means': {},
            'total_mean': traces_array.mean(axis=1).tolist(),
            'total_std': traces_array.std(axis=1).tolist()
        }

        # Per-layer evolution
        for i, name in enumerate(self.layer_names):
            start = self.layer_boundaries[i]
            end = self.layer_boundaries[i + 1]
            if end <= traces_array.shape[1]:
                layer_data = traces_array[:, start:end]
                evolution['layer_means'][name] = layer_data.mean(axis=1).tolist()

        return evolution

    def prepare_tom_depth_visualization(self, belief_store) -> Dict[str, Any]:
        """Visualize ToM depth (beliefs about beliefs)."""
        depth_data = {}

        for order in range(belief_store.max_order + 1):
            order_beliefs = belief_store.beliefs[order]
            depth_data[f'order_{order}'] = {
                'count': len(order_beliefs),
                'avg_confidence': np.mean([b.conf for b in order_beliefs.values()]) if order_beliefs else 0,
                'targets': [list(k) for k in order_beliefs.keys()]
            }

        return {
            'max_order': belief_store.max_order,
            'depth_data': depth_data,
            'total_beliefs': sum(len(b) for b in belief_store.beliefs.values())
        }


class InterpretabilityAnalyzer:
    """
    Analyze and explain agent reasoning.

    Provides interpretable explanations for agent decisions.
    """

    def __init__(self, ontology=None):
        self.ontology = ontology

    def explain_belief_state(self, belief_vector: torch.Tensor,
                           threshold: float = 0.7) -> List[str]:
        """Generate natural language explanation of belief state."""
        explanations = []

        if torch.is_tensor(belief_vector):
            belief_vector = belief_vector.cpu().numpy()

        # Find salient beliefs (high or low values)
        high_indices = np.where(belief_vector > threshold)[0]
        low_indices = np.where(belief_vector < (1 - threshold))[0]

        for idx in high_indices[:5]:  # Top 5 high
            explanations.append(f"Strong belief in dimension {idx} (value: {belief_vector[idx]:.2f})")

        for idx in low_indices[:5]:  # Top 5 low
            explanations.append(f"Strong disbelief in dimension {idx} (value: {belief_vector[idx]:.2f})")

        return explanations

    def explain_action_choice(self, action_value: float, beliefs: torch.Tensor,
                            context: Dict) -> str:
        """Generate explanation for why an action was chosen."""
        if torch.is_tensor(beliefs):
            beliefs = beliefs.cpu().numpy()

        # Analyze which beliefs drove the action
        if action_value > 0.7:
            explanation = "Chose confident action because: "
            high_beliefs = np.argsort(beliefs)[-3:]
            explanation += f"high confidence in dims {high_beliefs.tolist()}"
        elif action_value < 0.3:
            explanation = "Chose cautious action because: "
            low_beliefs = np.argsort(beliefs)[:3]
            explanation += f"uncertainty in dims {low_beliefs.tolist()}"
        else:
            explanation = "Chose moderate action due to mixed beliefs"

        return explanation

    def analyze_tom_reasoning(self, model_output: Dict, belief_store) -> Dict[str, Any]:
        """Analyze Theory of Mind reasoning patterns."""
        analysis = {
            'used_recursive_reasoning': False,
            'max_depth_used': 0,
            'belief_types': [],
            'reasoning_chain': []
        }

        # Check recursion depth
        if 'recursion_depth' in model_output:
            analysis['used_recursive_reasoning'] = model_output['recursion_depth'] > 1
            analysis['max_depth_used'] = model_output['recursion_depth']

        # Analyze belief orders used
        for order in range(belief_store.max_order + 1):
            if belief_store.beliefs[order]:
                analysis['belief_types'].append(f'order_{order}')
                for chain in list(belief_store.beliefs[order].keys())[:3]:
                    analysis['reasoning_chain'].append({
                        'order': order,
                        'about': list(chain),
                        'confidence': belief_store.beliefs[order][chain].conf
                    })

        return analysis

    def generate_report(self, agent_id: int, model_output: Dict,
                       belief_store, context: Dict) -> str:
        """Generate comprehensive interpretability report."""
        report = []
        report.append(f"=== Agent {agent_id} Reasoning Report ===\n")

        # Belief state summary
        if 'beliefs' in model_output:
            beliefs = model_output['beliefs']
            if torch.is_tensor(beliefs):
                beliefs = beliefs.cpu().numpy()
            report.append(f"Belief Summary:")
            report.append(f"  Mean: {np.mean(beliefs):.3f}")
            report.append(f"  Std:  {np.std(beliefs):.3f}")
            report.append(f"  Max:  {np.max(beliefs):.3f}")
            report.append(f"  Min:  {np.min(beliefs):.3f}\n")

        # ToM analysis
        tom_analysis = self.analyze_tom_reasoning(model_output, belief_store)
        report.append(f"Theory of Mind Analysis:")
        report.append(f"  Recursive reasoning: {tom_analysis['used_recursive_reasoning']}")
        report.append(f"  Max depth: {tom_analysis['max_depth_used']}")
        report.append(f"  Belief types: {tom_analysis['belief_types']}\n")

        # Action explanation
        if 'actions' in model_output:
            action = model_output['actions']
            action_val = action.item() if torch.is_tensor(action) else action
            explanation = self.explain_action_choice(action_val, model_output.get('beliefs', torch.tensor([0.5])), context)
            report.append(f"Action Decision:")
            report.append(f"  Value: {action_val:.3f}")
            report.append(f"  Explanation: {explanation}\n")

        return '\n'.join(report)


class TransparencyDashboard:
    """
    Central dashboard for transparency tools.

    Aggregates all transparency data for monitoring and analysis.
    """

    def __init__(self, ontology_dim: int = 181):
        self.extractor = TraceExtractor()
        self.visualizer = BeliefVisualizer(ontology_dim)
        self.analyzer = InterpretabilityAnalyzer()

        # Data storage
        self.trace_history: List[List[ComputationStep]] = []
        self.belief_history: List[BeliefTrace] = []
        self.reports: List[str] = []

    def record_step(self, model: nn.Module, input_data: torch.Tensor,
                   model_type: str = 'TRN') -> List[ComputationStep]:
        """Record a computation step."""
        if model_type == 'TRN':
            trace = self.extractor.extract_trn_trace(model, input_data)
        elif model_type == 'RSAN':
            trace = self.extractor.extract_rsan_trace(model, input_data)
        else:
            trace = []

        self.trace_history.append(trace)
        return trace

    def record_belief_evolution(self, belief_store, order: int,
                               target_chain: List[int]):
        """Record belief evolution."""
        belief_trace = self.extractor.extract_belief_trace(
            belief_store, order, target_chain
        )
        self.belief_history.append(belief_trace)

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for dashboard display."""
        return {
            'num_traces': len(self.trace_history),
            'num_belief_traces': len(self.belief_history),
            'num_reports': len(self.reports),
            'latest_trace_summary': self._summarize_trace(self.trace_history[-1]) if self.trace_history else None,
            'latest_belief_summary': self._summarize_belief_trace(self.belief_history[-1]) if self.belief_history else None
        }

    def _summarize_trace(self, trace: List[ComputationStep]) -> Dict:
        """Summarize a computation trace."""
        return {
            'num_steps': len(trace),
            'layers_used': list(set(t.layer for t in trace)),
            'has_attention': any(t.attention_weights for t in trace),
            'has_gates': any(t.gate_values for t in trace)
        }

    def _summarize_belief_trace(self, trace: BeliefTrace) -> Dict:
        """Summarize a belief trace."""
        return {
            'agent': trace.agent_id,
            'order': trace.order,
            'target_chain': trace.target_chain,
            'num_updates': len(trace.timestamps),
            'final_confidence': trace.confidences[-1] if trace.confidences else 0
        }

    def export_data(self, filepath: str = None) -> str:
        """Export all transparency data to JSON."""
        data = {
            'traces': [[{
                'layer': t.layer,
                'timestamp': t.timestamp,
                'interpretation': t.interpretation
            } for t in trace] for trace in self.trace_history],
            'beliefs': [{
                'agent': bt.agent_id,
                'order': bt.order,
                'target': bt.target_chain,
                'timestamps': bt.timestamps,
                'confidences': bt.confidences
            } for bt in self.belief_history],
            'reports': self.reports
        }

        json_str = json.dumps(data, indent=2)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str
