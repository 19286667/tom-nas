"""
Metrics Tracking and Analysis for ToM-NAS
Comprehensive performance monitoring
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime


@dataclass
class TrainingMetrics:
    """Metrics collected during training"""

    epoch: int
    loss: float
    accuracy: float
    belief_accuracy: float
    action_accuracy: float
    learning_rate: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationMetrics:
    """Metrics from evaluation"""

    benchmark_scores: Dict[str, float]
    tom_order_scores: Dict[int, float]
    cooperation_score: float
    zombie_detection_score: float
    overall_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsTracker:
    """Track and analyze metrics over time"""

    def __init__(self):
        self.training_history = []
        self.evaluation_history = []
        self.custom_metrics = defaultdict(list)

    def log_training(
        self,
        epoch: int,
        loss: float,
        accuracy: float = 0.0,
        belief_accuracy: float = 0.0,
        action_accuracy: float = 0.0,
        learning_rate: float = 0.001,
    ):
        """Log training metrics"""
        metrics = TrainingMetrics(
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            belief_accuracy=belief_accuracy,
            action_accuracy=action_accuracy,
            learning_rate=learning_rate,
        )
        self.training_history.append(metrics)

    def log_evaluation(
        self,
        benchmark_scores: Dict,
        tom_order_scores: Dict,
        cooperation_score: float,
        zombie_detection_score: float,
        overall_score: float,
    ):
        """Log evaluation metrics"""
        metrics = EvaluationMetrics(
            benchmark_scores=benchmark_scores,
            tom_order_scores=tom_order_scores,
            cooperation_score=cooperation_score,
            zombie_detection_score=zombie_detection_score,
            overall_score=overall_score,
        )
        self.evaluation_history.append(metrics)

    def log_custom(self, metric_name: str, value: float):
        """Log custom metric"""
        self.custom_metrics[metric_name].append(value)

    def get_training_summary(self) -> Dict:
        """Get summary of training metrics"""
        if not self.training_history:
            return {}

        losses = [m.loss for m in self.training_history]
        accuracies = [m.accuracy for m in self.training_history]

        return {
            "num_epochs": len(self.training_history),
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "avg_loss": np.mean(losses),
            "final_accuracy": accuracies[-1],
            "best_accuracy": max(accuracies),
            "avg_accuracy": np.mean(accuracies),
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
        }

    def get_evaluation_summary(self) -> Dict:
        """Get summary of evaluation metrics"""
        if not self.evaluation_history:
            return {}

        overall_scores = [m.overall_score for m in self.evaluation_history]

        return {
            "num_evaluations": len(self.evaluation_history),
            "final_score": overall_scores[-1],
            "best_score": max(overall_scores),
            "avg_score": np.mean(overall_scores),
            "score_improvement": overall_scores[-1] - overall_scores[0] if len(overall_scores) > 1 else 0.0,
        }

    def save_to_file(self, filepath: str):
        """Save all metrics to JSON file"""
        data = {
            "training_history": [
                {
                    "epoch": m.epoch,
                    "loss": m.loss,
                    "accuracy": m.accuracy,
                    "belief_accuracy": m.belief_accuracy,
                    "action_accuracy": m.action_accuracy,
                    "learning_rate": m.learning_rate,
                    "timestamp": m.timestamp,
                }
                for m in self.training_history
            ],
            "evaluation_history": [
                {
                    "benchmark_scores": m.benchmark_scores,
                    "tom_order_scores": m.tom_order_scores,
                    "cooperation_score": m.cooperation_score,
                    "zombie_detection_score": m.zombie_detection_score,
                    "overall_score": m.overall_score,
                    "timestamp": m.timestamp,
                }
                for m in self.evaluation_history
            ],
            "custom_metrics": dict(self.custom_metrics),
            "summary": {"training": self.get_training_summary(), "evaluation": self.get_evaluation_summary()},
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct training history
        self.training_history = [TrainingMetrics(**m) for m in data.get("training_history", [])]

        # Reconstruct evaluation history
        self.evaluation_history = [EvaluationMetrics(**m) for m in data.get("evaluation_history", [])]

        # Load custom metrics
        self.custom_metrics = defaultdict(list, data.get("custom_metrics", {}))


class PerformanceAnalyzer:
    """Analyze performance trends and patterns"""

    @staticmethod
    def detect_overfitting(training_losses: List[float], validation_losses: List[float], window_size: int = 5) -> bool:
        """Detect if model is overfitting"""
        if len(training_losses) < window_size or len(validation_losses) < window_size:
            return False

        # Check if validation loss increasing while training loss decreasing
        recent_train = training_losses[-window_size:]
        recent_val = validation_losses[-window_size:]

        train_trend = recent_train[-1] - recent_train[0]
        val_trend = recent_val[-1] - recent_val[0]

        # Overfitting if train decreasing and val increasing
        return train_trend < -0.01 and val_trend > 0.01

    @staticmethod
    def detect_plateau(metric_history: List[float], window_size: int = 10, threshold: float = 0.01) -> bool:
        """Detect if metric has plateaued"""
        if len(metric_history) < window_size:
            return False

        recent = metric_history[-window_size:]
        variance = np.var(recent)

        return variance < threshold

    @staticmethod
    def calculate_learning_speed(metric_history: List[float]) -> float:
        """Calculate how fast the model is learning"""
        if len(metric_history) < 2:
            return 0.0

        # Fit linear trend
        x = np.arange(len(metric_history))
        y = np.array(metric_history)

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]

        return slope

    @staticmethod
    def compare_architectures(results1: Dict, results2: Dict) -> Dict:
        """Compare performance of two architectures"""
        comparison = {"architecture1_better": 0, "architecture2_better": 0, "tied": 0, "differences": {}}

        # Compare each metric
        for key in results1.keys():
            if key in results2:
                val1 = results1[key]
                val2 = results2[key]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val1 - val2
                    comparison["differences"][key] = diff

                    if abs(diff) < 0.01:
                        comparison["tied"] += 1
                    elif diff > 0:
                        comparison["architecture1_better"] += 1
                    else:
                        comparison["architecture2_better"] += 1

        return comparison

    @staticmethod
    def calculate_efficiency_score(accuracy: float, num_parameters: int, inference_time: float) -> float:
        """
        Calculate efficiency score combining accuracy, size, and speed
        Higher is better
        """
        # Normalize parameters (assume 1M is baseline)
        param_penalty = num_parameters / 1e6

        # Normalize inference time (assume 10ms is baseline)
        time_penalty = inference_time / 0.01

        # Combined score
        efficiency = accuracy / (1.0 + 0.1 * param_penalty + 0.1 * time_penalty)

        return efficiency


class ResultsAggregator:
    """Aggregate results across multiple runs"""

    def __init__(self):
        self.run_results = []

    def add_run(self, results: Dict, run_id: str = None):
        """Add results from a single run"""
        if run_id is None:
            run_id = f"run_{len(self.run_results) + 1}"

        self.run_results.append({"run_id": run_id, "results": results})

    def get_statistics(self, metric_name: str) -> Dict:
        """Get statistics for a specific metric across runs"""
        values = []

        for run in self.run_results:
            if metric_name in run["results"]:
                values.append(run["results"][metric_name])

        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "num_runs": len(values),
        }

    def get_best_run(self, metric_name: str, higher_is_better: bool = True) -> Optional[Dict]:
        """Get the best performing run for a metric"""
        if not self.run_results:
            return None

        valid_runs = [run for run in self.run_results if metric_name in run["results"]]

        if not valid_runs:
            return None

        if higher_is_better:
            best_run = max(valid_runs, key=lambda x: x["results"][metric_name])
        else:
            best_run = min(valid_runs, key=lambda x: x["results"][metric_name])

        return best_run

    def export_summary(self, filepath: str):
        """Export summary statistics to file"""
        summary = {"num_runs": len(self.run_results), "metrics": {}}

        # Find all metric names
        metric_names = set()
        for run in self.run_results:
            metric_names.update(run["results"].keys())

        # Calculate statistics for each metric
        for metric in metric_names:
            summary["metrics"][metric] = self.get_statistics(metric)

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for values"""
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)

    # t-score for 95% confidence (approximate)
    t_score = 1.96 if n > 30 else 2.0

    margin = t_score * std / np.sqrt(n)

    return (mean - margin, mean + margin)
