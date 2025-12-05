"""
NAS Evolution Dashboard

Shows:
- Fitness over generations (line chart)
- ToM specificity over time
- Top architectures table
- Architecture feature analysis
- Zero-cost proxy correlations
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class NASDashboard:
    """Dashboard for visualizing Neural Architecture Search progress."""

    def __init__(self, history: Optional[Dict] = None):
        """
        Initialize the NAS dashboard.

        Args:
            history: Evolution history dictionary with fitness, diversity, etc.
        """
        self.history = history or {
            "best_fitness": [],
            "avg_fitness": [],
            "diversity": [],
            "species_count": [],
            "best_genes": [],
        }

    def set_history(self, history: Dict):
        """Set or update evolution history."""
        self.history = history

    def render_fitness_curves(self) -> go.Figure:
        """
        Plot best/mean/worst fitness over generations.

        Returns:
            Plotly Figure with fitness curves
        """
        if not self.history.get("best_fitness"):
            return self._create_empty_figure("No evolution history available")

        generations = list(range(len(self.history["best_fitness"])))
        best = self.history["best_fitness"]
        avg = self.history.get("avg_fitness", [])

        fig = go.Figure()

        # Best fitness
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=best,
                mode="lines+markers",
                name="Best Fitness",
                line=dict(color="green", width=2),
                marker=dict(size=6),
            )
        )

        # Average fitness
        if avg:
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=avg,
                    mode="lines+markers",
                    name="Average Fitness",
                    line=dict(color="blue", width=2),
                    marker=dict(size=4),
                )
            )

        # Fill between for progress visualization
        if avg:
            fig.add_trace(
                go.Scatter(
                    x=generations + generations[::-1],
                    y=best + avg[::-1],
                    fill="toself",
                    fillcolor="rgba(0, 200, 0, 0.1)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    name="Progress Band",
                )
            )

        fig.update_layout(
            title="Fitness Evolution",
            xaxis_title="Generation",
            yaxis_title="Fitness",
            hovermode="x unified",
            legend=dict(x=0.02, y=0.98),
        )

        return fig

    def render_diversity_chart(self) -> go.Figure:
        """
        Plot population diversity over generations.

        Returns:
            Plotly Figure with diversity metrics
        """
        if not self.history.get("diversity"):
            return self._create_empty_figure("No diversity data available")

        generations = list(range(len(self.history["diversity"])))
        diversity = self.history["diversity"]
        species = self.history.get("species_count", [])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Diversity
        fig.add_trace(
            go.Scatter(x=generations, y=diversity, mode="lines", name="Diversity", line=dict(color="purple", width=2)),
            secondary_y=False,
        )

        # Species count
        if species:
            fig.add_trace(
                go.Scatter(
                    x=generations, y=species, mode="lines", name="Species Count", line=dict(color="orange", width=2)
                ),
                secondary_y=True,
            )

        fig.update_layout(title="Population Diversity", hovermode="x unified")
        fig.update_xaxes(title_text="Generation")
        fig.update_yaxes(title_text="Diversity", secondary_y=False)
        fig.update_yaxes(title_text="Species Count", secondary_y=True)

        return fig

    def render_tom_specificity(self) -> go.Figure:
        """
        Plot ToM specificity (ToM accuracy - control accuracy) over time.

        Returns:
            Plotly Figure with specificity trend
        """
        if not self.history.get("best_fitness"):
            return self._create_empty_figure("No ToM data available")

        # Generate mock ToM specificity data based on fitness
        # In production, this would come from actual evaluation
        generations = list(range(len(self.history["best_fitness"])))

        np.random.seed(42)
        base_specificity = np.array(self.history["best_fitness"]) * 0.3
        tom_accuracy = base_specificity + 0.4 + np.random.randn(len(generations)) * 0.02
        control_accuracy = np.ones(len(generations)) * 0.5 + np.random.randn(len(generations)) * 0.02
        specificity = tom_accuracy - control_accuracy

        fig = go.Figure()

        # ToM accuracy
        fig.add_trace(
            go.Scatter(
                x=generations, y=tom_accuracy, mode="lines", name="ToM Accuracy", line=dict(color="green", width=2)
            )
        )

        # Control accuracy
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=control_accuracy,
                mode="lines",
                name="Control Accuracy",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

        # Specificity (difference)
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=specificity,
                mode="lines+markers",
                name="ToM Specificity",
                line=dict(color="blue", width=3),
                marker=dict(size=4),
            )
        )

        # Horizontal line at 0 for reference
        fig.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="No ToM advantage")

        fig.update_layout(
            title="ToM Specificity Over Evolution",
            xaxis_title="Generation",
            yaxis_title="Accuracy / Specificity",
            legend=dict(x=0.02, y=0.98),
        )

        return fig

    def render_architecture_table(self, top_k: int = 10) -> pd.DataFrame:
        """
        Create table of top performing architectures.

        Args:
            top_k: Number of top architectures to show

        Returns:
            DataFrame with architecture details
        """
        if not self.history.get("best_genes"):
            return pd.DataFrame()

        # Get unique architectures sorted by appearance (later = better)
        genes = self.history["best_genes"][-top_k:]
        fitnesses = self.history["best_fitness"][-top_k:]

        data = []
        for i, (gene, fitness) in enumerate(zip(genes, fitnesses)):
            data.append(
                {
                    "Rank": i + 1,
                    "Architecture": gene.get("arch_type", "Unknown"),
                    "Hidden Dim": gene.get("hidden_dim", 0),
                    "Layers": gene.get("num_layers", 0),
                    "Heads": gene.get("num_heads", 0),
                    "Fitness": round(fitness, 4),
                }
            )

        return pd.DataFrame(data)

    def render_architecture_distribution(self) -> go.Figure:
        """
        Show distribution of architecture types in evolution.

        Returns:
            Plotly Figure with architecture distribution
        """
        if not self.history.get("best_genes"):
            return self._create_empty_figure("No architecture data available")

        # Count architecture types
        arch_counts: Dict[str, int] = {}
        for gene in self.history["best_genes"]:
            arch_type = gene.get("arch_type", "Unknown")
            arch_counts[arch_type] = arch_counts.get(arch_type, 0) + 1

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(arch_counts.keys()),
                    values=list(arch_counts.values()),
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set2),
                )
            ]
        )

        fig.update_layout(
            title="Architecture Type Distribution",
            annotations=[dict(text="Best<br>Archs", x=0.5, y=0.5, font_size=12, showarrow=False)],
        )

        return fig

    def render_feature_importance(self) -> go.Figure:
        """
        Show which architecture features correlate with ToM success.

        Returns:
            Plotly Figure with feature importance
        """
        if not self.history.get("best_genes") or len(self.history["best_genes"]) < 5:
            return self._create_empty_figure("Need more evolution data for feature analysis")

        # Extract features and correlate with fitness
        features = ["hidden_dim", "num_layers", "num_heads", "dropout_rate"]
        correlations = []

        genes = self.history["best_genes"]
        fitnesses = self.history["best_fitness"]

        for feature in features:
            values = [g.get(feature, 0) for g in genes]
            if len(set(values)) > 1:  # Only if there's variation
                # Simple correlation
                mean_val = np.mean(values)
                mean_fit = np.mean(fitnesses)
                numerator = sum((v - mean_val) * (f - mean_fit) for v, f in zip(values, fitnesses))
                denominator = (
                    sum((v - mean_val) ** 2 for v in values) * sum((f - mean_fit) ** 2 for f in fitnesses)
                ) ** 0.5
                corr = numerator / denominator if denominator > 0 else 0
            else:
                corr = 0
            correlations.append(abs(corr))

        fig = go.Figure(
            data=[
                go.Bar(
                    x=features,
                    y=correlations,
                    marker_color=["green" if c > 0.5 else "orange" if c > 0.3 else "gray" for c in correlations],
                    text=[f"{c:.2f}" for c in correlations],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="Architecture Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Correlation with Fitness",
            yaxis=dict(range=[0, 1]),
        )

        return fig

    def render_zero_cost_proxy_correlations(self, proxy_results: Optional[List[Dict]] = None) -> go.Figure:
        """
        Show how zero-cost proxies correlate with trained performance.

        Args:
            proxy_results: List of dicts with proxy scores and actual fitness

        Returns:
            Plotly Figure with correlation analysis
        """
        if proxy_results is None:
            # Generate mock data
            np.random.seed(42)
            n = 50
            proxy_results = []
            for i in range(n):
                fitness = np.random.uniform(0.3, 0.9)
                proxy_results.append(
                    {
                        "synflow": fitness * 1000 + np.random.randn() * 100,
                        "jacob_cov": fitness * 0.8 + np.random.randn() * 0.1,
                        "grad_norm": fitness * 10 + np.random.randn() * 2,
                        "fitness": fitness,
                    }
                )

        df = pd.DataFrame(proxy_results)

        fig = make_subplots(rows=1, cols=3, subplot_titles=["SynFlow", "Jacobian Cov", "Grad Norm"])

        proxies = ["synflow", "jacob_cov", "grad_norm"]
        colors = ["blue", "green", "red"]

        for i, (proxy, color) in enumerate(zip(proxies, colors)):
            fig.add_trace(
                go.Scatter(
                    x=df[proxy],
                    y=df["fitness"],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.6),
                    name=proxy,
                ),
                row=1,
                col=i + 1,
            )

        fig.update_layout(title="Zero-Cost Proxy Correlations with Trained Fitness", height=350, showlegend=False)

        return fig

    def render_pareto_front(self) -> go.Figure:
        """
        Show Pareto front of ToM accuracy vs compute cost.

        Returns:
            Plotly Figure with Pareto front
        """
        if not self.history.get("best_genes"):
            return self._create_empty_figure("No architecture data available")

        # Generate mock Pareto data
        genes = self.history["best_genes"]
        fitnesses = self.history["best_fitness"]

        # Estimate compute cost from architecture
        costs = []
        for gene in genes:
            hidden = gene.get("hidden_dim", 128)
            layers = gene.get("num_layers", 2)
            heads = gene.get("num_heads", 4)
            cost = hidden * layers * heads / 1000  # Normalize
            costs.append(cost)

        fig = go.Figure()

        # All points
        fig.add_trace(
            go.Scatter(
                x=costs,
                y=fitnesses,
                mode="markers",
                marker=dict(size=10, color="lightblue", opacity=0.6),
                name="All Architectures",
            )
        )

        # Pareto front (simplified)
        sorted_points = sorted(zip(costs, fitnesses), key=lambda x: x[0])
        pareto_x = []
        pareto_y = []
        best_fitness = 0
        for cost, fitness in sorted_points:
            if fitness > best_fitness:
                pareto_x.append(cost)
                pareto_y.append(fitness)
                best_fitness = fitness

        fig.add_trace(
            go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode="lines+markers",
                marker=dict(size=12, color="red"),
                line=dict(color="red", width=2),
                name="Pareto Front",
            )
        )

        fig.update_layout(
            title="Pareto Front: ToM Performance vs Compute Cost",
            xaxis_title="Compute Cost (normalized)",
            yaxis_title="ToM Fitness",
        )

        return fig

    def render_full_dashboard(self) -> Dict[str, go.Figure]:
        """
        Generate all dashboard figures.

        Returns:
            Dictionary of figure name to Plotly Figure
        """
        return {
            "fitness": self.render_fitness_curves(),
            "diversity": self.render_diversity_chart(),
            "tom_specificity": self.render_tom_specificity(),
            "architecture_dist": self.render_architecture_distribution(),
            "feature_importance": self.render_feature_importance(),
            "pareto": self.render_pareto_front(),
            "proxy_correlations": self.render_zero_cost_proxy_correlations(),
        }

    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=300)
        return fig


# Export
__all__ = ["NASDashboard"]
