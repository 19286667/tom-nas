"""
Soul Map Visualization System

Creates visual representations of the 181-dimensional psychological ontology.
Supports multiple visualization modes: radar, heatmap, timeline, comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch
from typing import Dict, List, Optional, Tuple
import json

# Import ontology
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.ontology import SoulMapOntology

class SoulMapVisualizer:
    """
    Visualizes Soul Maps in various formats.

    Visualization modes:
    - Radar: Multi-axis radar chart for key dimensions
    - Heatmap: Grid visualization of all 181 dimensions
    - Timeline: Evolution of soul map over time
    - Comparison: Side-by-side comparison of multiple soul maps
    - Aura: 3D-style aura effect based on emotional state
    """

    def __init__(self, ontology: SoulMapOntology):
        self.ontology = ontology

        # Define visual groupings for radar chart
        self.radar_dimensions = [
            ('Emotional', 'affect.valence'),
            ('Energy', 'affect.arousal'),
            ('Willpower', 'affect.dominance'),
            ('Social', 'social.openness'),
            ('Analytical', 'cognitive.analytical'),
            ('Creative', 'cognitive.creative'),
            ('Empathy', 'social.empathy'),
            ('Confidence', 'social.confidence')
        ]

        # Color schemes
        self.layer_colors = {
            0: '#FF6B6B',  # Biological - Red
            1: '#4ECDC4',  # Affective - Teal
            2: '#45B7D1',  # Cognitive - Blue
            3: '#96CEB4',  # Social - Green
            4: '#FFEAA7',  # Motivational - Yellow
            5: '#DDA15E',  # Moral - Orange
            6: '#BC6C25',  # Temporal - Brown
            7: '#8E44AD',  # Narrative - Purple
            8: '#E74C3C',  # Existential - Dark Red
        }

    def visualize_radar(
        self,
        soul_map: torch.Tensor,
        title: str = "Soul Map Radar",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a radar chart visualization.

        Shows 8 key psychological dimensions in a star pattern.
        """

        # Extract values for radar dimensions
        values = []
        labels = []

        soul_map_dict = self._tensor_to_dict(soul_map)

        for label, dim_name in self.radar_dimensions:
            value = soul_map_dict.get(dim_name, 0.5)
            values.append(value)
            labels.append(label)

        # Number of variables
        num_vars = len(labels)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label=title, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')

        # Fix axis to go in the right order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=10)

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Title
        plt.title(title, size=16, pad=20, weight='bold')

        # Legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def visualize_heatmap(
        self,
        soul_map: torch.Tensor,
        title: str = "Soul Map Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap visualization of all 181 dimensions.

        Organized by ontology layers.
        """

        # Organize dimensions by layer
        layer_data = {}
        for i, dim in enumerate(self.ontology.dimensions):
            if i >= len(soul_map):
                break

            layer = dim.layer
            if layer not in layer_data:
                layer_data[layer] = []

            layer_data[layer].append({
                'name': dim.name,
                'value': float(soul_map[i]),
                'index': i
            })

        # Create figure
        num_layers = len(layer_data)
        fig, axes = plt.subplots(1, num_layers, figsize=(20, 4))

        if num_layers == 1:
            axes = [axes]

        for layer_idx, (layer_num, dims) in enumerate(sorted(layer_data.items())):
            ax = axes[layer_idx]

            # Extract values
            values = [d['value'] for d in dims]
            names = [d['name'].split('.')[-1] for d in dims]

            # Create heatmap data
            data = np.array(values).reshape(-1, 1)

            # Plot
            im = ax.imshow(
                data.T,
                cmap='RdYlGn',
                aspect='auto',
                vmin=0,
                vmax=1
            )

            # Set ticks
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, ha='right', fontsize=6)
            ax.set_yticks([])

            # Layer title
            layer_names = [
                'Biological', 'Affective', 'Cognitive', 'Social',
                'Motivational', 'Moral', 'Temporal', 'Narrative', 'Existential'
            ]
            layer_name = layer_names[layer_num] if layer_num < len(layer_names) else f"Layer {layer_num}"
            ax.set_title(layer_name, fontsize=10, weight='bold')

        # Overall title
        fig.suptitle(title, fontsize=16, weight='bold', y=1.02)

        # Colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Activation Level', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def visualize_timeline(
        self,
        soul_map_history: List[torch.Tensor],
        title: str = "Soul Map Evolution",
        save_path: Optional[str] = None,
        key_dimensions: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Visualize how soul map changes over time.

        Shows timeline of key dimensions.
        """

        if not key_dimensions:
            # Default to radar dimensions
            key_dimensions = [dim for _, dim in self.radar_dimensions]

        # Extract time series
        time_series = {dim: [] for dim in key_dimensions}

        for soul_map in soul_map_history:
            soul_map_dict = self._tensor_to_dict(soul_map)
            for dim in key_dimensions:
                value = soul_map_dict.get(dim, 0.5)
                time_series[dim].append(value)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each dimension
        for dim in key_dimensions:
            label = dim.split('.')[-1].capitalize()
            ax.plot(
                time_series[dim],
                label=label,
                marker='o',
                linewidth=2,
                alpha=0.8
            )

        # Styling
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Activation Level', fontsize=12)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def visualize_comparison(
        self,
        soul_maps: Dict[str, torch.Tensor],
        title: str = "Soul Map Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare multiple soul maps side by side.

        Useful for comparing player vs NPC, or multiple NPCs.
        """

        # Number of soul maps
        num_maps = len(soul_maps)

        # Extract values for radar dimensions
        all_values = {}
        labels = []

        for name, soul_map in soul_maps.items():
            values = []
            soul_map_dict = self._tensor_to_dict(soul_map)

            for label, dim_name in self.radar_dimensions:
                value = soul_map_dict.get(dim_name, 0.5)
                values.append(value)

                if label not in labels:
                    labels.append(label)

            all_values[name] = values

        # Number of variables
        num_vars = len(labels)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Colors for different soul maps
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        # Plot each soul map
        for idx, (name, values) in enumerate(all_values.items()):
            values_plot = values + values[:1]  # Complete the circle
            angles_plot = angles + angles[:1]

            color = colors[idx % len(colors)]

            ax.plot(
                angles_plot,
                values_plot,
                'o-',
                linewidth=2,
                label=name,
                color=color
            )
            ax.fill(angles_plot, values_plot, alpha=0.15, color=color)

        # Fix axis
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, size=11)

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Title
        plt.title(title, size=18, pad=20, weight='bold')

        # Legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def visualize_aura(
        self,
        soul_map: torch.Tensor,
        title: str = "Psychological Aura",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create an 'aura' visualization based on emotional state.

        Uses color and intensity to represent psychological state.
        """

        soul_map_dict = self._tensor_to_dict(soul_map)

        # Extract key affective dimensions
        valence = soul_map_dict.get('affect.valence', 0.5)
        arousal = soul_map_dict.get('affect.arousal', 0.5)
        dominance = soul_map_dict.get('affect.dominance', 0.5)

        # Map to color
        # Valence: Red (negative) to Green (positive)
        # Arousal: Brightness
        # Dominance: Saturation

        # Convert to RGB
        hue = valence  # 0-1, maps to color
        saturation = dominance
        value = arousal

        # HSV to RGB (simplified)
        r = value * (1 + saturation * (np.cos(2 * np.pi * hue) - 1))
        g = value * (1 + saturation * (np.cos(2 * np.pi * (hue - 1/3)) - 1))
        b = value * (1 + saturation * (np.cos(2 * np.pi * (hue - 2/3)) - 1))

        # Clip
        r, g, b = np.clip([r, g, b], 0, 1)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Create aura effect (concentric circles)
        center = (0.5, 0.5)

        for i in range(10, 0, -1):
            radius = i * 0.05
            alpha = (11 - i) / 10 * 0.3  # Fade out towards edges

            circle = plt.Circle(
                center,
                radius,
                color=(r, g, b),
                alpha=alpha
            )
            ax.add_patch(circle)

        # Center core (brightest)
        core = plt.Circle(center, 0.05, color=(r, g, b), alpha=0.9)
        ax.add_patch(core)

        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title
        plt.title(title, size=18, weight='bold', pad=20)

        # Add legend with values
        info_text = (
            f"Valence: {valence:.2f}\n"
            f"Arousal: {arousal:.2f}\n"
            f"Dominance: {dominance:.2f}"
        )
        plt.text(
            0.05, 0.95,
            info_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def export_to_json(
        self,
        soul_map: torch.Tensor,
        filepath: str
    ):
        """
        Export soul map to JSON for game engine integration.
        """

        soul_map_dict = self._tensor_to_dict(soul_map)

        # Organize by layer
        organized = {
            'timestamp': None,  # To be filled by caller
            'total_dimensions': len(soul_map),
            'layers': {}
        }

        for i, dim in enumerate(self.ontology.dimensions):
            if i >= len(soul_map):
                break

            layer = dim.layer
            if layer not in organized['layers']:
                organized['layers'][layer] = {
                    'name': self._get_layer_name(layer),
                    'dimensions': {}
                }

            organized['layers'][layer]['dimensions'][dim.name] = float(soul_map[i])

        # Save
        with open(filepath, 'w') as f:
            json.dump(organized, f, indent=2)

        return organized

    def _tensor_to_dict(self, soul_map: torch.Tensor) -> Dict[str, float]:
        """Convert soul map tensor to dictionary"""
        result = {}
        for i, dim in enumerate(self.ontology.dimensions):
            if i < len(soul_map):
                result[dim.name] = float(soul_map[i])
        return result

    def _get_layer_name(self, layer: int) -> str:
        """Get human-readable layer name"""
        names = [
            'Biological', 'Affective', 'Cognitive', 'Social',
            'Motivational', 'Moral', 'Temporal', 'Narrative', 'Existential'
        ]
        return names[layer] if layer < len(names) else f"Layer {layer}"

# ============================================================================
# Interactive WebGL Visualizer (for browser integration)
# ============================================================================

class WebGLSoulMapExporter:
    """
    Exports soul map data in format suitable for WebGL/Three.js visualization.
    """

    def __init__(self, ontology: SoulMapOntology):
        self.ontology = ontology

    def export_for_threejs(
        self,
        soul_map: torch.Tensor,
        output_path: str
    ):
        """
        Export soul map data for Three.js visualization.

        Creates JSON with:
        - Radar chart data
        - Layer data for 3D visualization
        - Color mappings
        - Animation keyframes
        """

        visualizer = SoulMapVisualizer(self.ontology)
        soul_map_dict = visualizer._tensor_to_dict(soul_map)

        # Radar data
        radar_data = []
        for label, dim_name in visualizer.radar_dimensions:
            value = soul_map_dict.get(dim_name, 0.5)
            radar_data.append({
                'label': label,
                'dimension': dim_name,
                'value': value
            })

        # Layer data for 3D rings
        layer_data = []
        for layer in range(9):  # 9 layers
            if layer in visualizer.ontology.layer_ranges:
                start, end = visualizer.ontology.layer_ranges[layer]
                layer_values = [float(soul_map[i]) for i in range(start, min(end + 1, len(soul_map)))]
                avg_value = np.mean(layer_values) if layer_values else 0.5

                layer_data.append({
                    'layer': layer,
                    'name': visualizer._get_layer_name(layer),
                    'activation': avg_value,
                    'color': visualizer.layer_colors.get(layer, '#CCCCCC'),
                    'num_dimensions': len(layer_values)
                })

        # Complete data structure
        data = {
            'radar': radar_data,
            'layers': layer_data,
            'full_soul_map': soul_map_dict,
            'metadata': {
                'total_dimensions': len(soul_map),
                'num_layers': len(layer_data)
            }
        }

        # Save
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return data

# ============================================================================
# Demo / Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Soul Map Visualization System Demo")
    print("=" * 60)

    # Create ontology
    ontology = SoulMapOntology()

    # Create test soul map (simulating a character in distress)
    test_soul_map = ontology.get_default_state()
    test_soul_map[1] = 0.2  # Low valence (sad)
    test_soul_map[2] = 0.8  # High arousal (agitated)
    test_soul_map[3] = 0.3  # Low dominance (powerless)

    # Create visualizer
    viz = SoulMapVisualizer(ontology)

    print("\n1. Creating Radar Chart...")
    viz.visualize_radar(test_soul_map, title="Distressed Character", save_path="soul_map_radar.png")
    print("   Saved: soul_map_radar.png")

    print("\n2. Creating Heatmap...")
    viz.visualize_heatmap(test_soul_map, title="Full Soul Map", save_path="soul_map_heatmap.png")
    print("   Saved: soul_map_heatmap.png")

    print("\n3. Creating Aura Visualization...")
    viz.visualize_aura(test_soul_map, title="Psychological Aura", save_path="soul_map_aura.png")
    print("   Saved: soul_map_aura.png")

    print("\n4. Creating Comparison...")
    # Create second soul map (confident character)
    confident_soul_map = ontology.get_default_state()
    confident_soul_map[1] = 0.8  # High valence (happy)
    confident_soul_map[2] = 0.6  # Moderate arousal
    confident_soul_map[3] = 0.9  # High dominance (confident)

    soul_maps = {
        'Distressed': test_soul_map,
        'Confident': confident_soul_map
    }

    viz.visualize_comparison(soul_maps, title="Character Comparison", save_path="soul_map_comparison.png")
    print("   Saved: soul_map_comparison.png")

    print("\n5. Exporting to JSON...")
    viz.export_to_json(test_soul_map, "soul_map_data.json")
    print("   Saved: soul_map_data.json")

    print("\n6. Exporting for WebGL/Three.js...")
    webgl_exporter = WebGLSoulMapExporter(ontology)
    webgl_exporter.export_for_threejs(test_soul_map, "soul_map_threejs.json")
    print("   Saved: soul_map_threejs.json")

    print("\n" + "=" * 60)
    print("Demo complete! Visualization files created.")
    print("=" * 60)
