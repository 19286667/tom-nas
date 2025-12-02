"""
Visual renderer for the Liminal world using Plotly.

Shows:
- 2D map of current realm with locations
- NPC positions as colored dots (color = emotional state)
- Player position
- Soul Map radars on hover/click
- Belief bubbles showing what agents think
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..liminal import LiminalEnvironment, SoulMap
    from ..liminal.npcs.base_npc import BaseNPC


class WorldRenderer:
    """Renders the Liminal world state visually."""

    REALM_COLORS = {
        'peregrine': '#FFD700',      # Golden
        'spleen_towns': '#8B4513',    # Sepia
        'ministry': '#E0E0E0',        # Fluorescent grey
        'city': '#C0C0C0',            # Chrome
        'hollow': '#2F4F4F',          # Dark industrial
        'nothing': '#000000',         # Black/glitched
    }

    EMOTION_COLORS = {
        'calm': '#90EE90',
        'anxious': '#FFD700',
        'afraid': '#FF6347',
        'curious': '#87CEEB',
        'hostile': '#DC143C',
        'suspicious': '#FF8C00',
        'neutral': '#808080',
        'happy': '#32CD32',
        'sad': '#4169E1',
        'frightened': '#FF4500',
    }

    REALM_DESCRIPTIONS = {
        'peregrine': 'The Hub - Where complementary states coexist',
        'spleen_towns': 'The Loop - Where time moves strangely',
        'ministry': 'The Bureaucracy - Where identity is paperwork',
        'city': 'The Machine - Where constants rule',
        'hollow': 'The Shadow - Where corruption spreads',
        'nothing': 'The Void - Where reality breaks down',
    }

    def __init__(self, env: Optional['LiminalEnvironment'] = None):
        """
        Initialize the world renderer.

        Args:
            env: LiminalEnvironment instance to render
        """
        self.env = env

    def set_environment(self, env: 'LiminalEnvironment'):
        """Set or update the environment to render."""
        self.env = env

    def render_realm(self, realm_type: str = None) -> go.Figure:
        """
        Render a 2D view of a realm with NPCs.

        Args:
            realm_type: Type of realm to render (uses current if None)

        Returns:
            Plotly Figure with realm visualization
        """
        if self.env is None:
            return self._create_empty_figure("No environment set")

        if realm_type is None:
            realm_type = self.env.current_realm.value

        # Normalize realm type
        realm_key = realm_type.lower().replace(' ', '_')
        if '(' in realm_key:
            realm_key = realm_key.split('(')[0].strip().replace(' ', '_')

        # Map display names to realm keys
        realm_mapping = {
            'peregrine': 'peregrine',
            'spleen_towns': 'spleen_towns',
            'ministry_districts': 'ministry',
            'ministry': 'ministry',
            'city_of_constants': 'city',
            'city': 'city',
            'hollow_reaches': 'hollow',
            'hollow': 'hollow',
            'the_nothing': 'nothing',
            'nothing': 'nothing',
        }
        realm_key = realm_mapping.get(realm_key, realm_key)

        # Create figure with subplot for soul map radar
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "scatter"}, {"type": "polar"}]],
            subplot_titles=["Realm View", "Soul Map (hover over NPC)"]
        )

        # Get NPCs in this realm
        npcs = self._get_npcs_in_realm(realm_key)

        # Background color based on realm
        bg_color = self.REALM_COLORS.get(realm_key, '#FFFFFF')

        # Generate NPC positions (use actual positions if available)
        x_positions = []
        y_positions = []
        colors = []
        names = []
        hover_texts = []

        for npc in npcs[:50]:  # Limit for performance
            x, y = self._get_npc_position(npc)
            x_positions.append(x)
            y_positions.append(y)

            emotion = getattr(npc, 'emotional_state', 'neutral')
            colors.append(self.EMOTION_COLORS.get(emotion, '#808080'))
            names.append(getattr(npc, 'name', npc.npc_id))

            hover_text = self._create_hover_text(npc)
            hover_texts.append(hover_text)

        # Plot NPCs
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=colors,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=names,
                textposition='top center',
                textfont=dict(size=8),
                hovertext=hover_texts,
                hoverinfo='text',
                name='NPCs'
            ),
            row=1, col=1
        )

        # Add player position
        player_x, player_y = self.env.player_position
        fig.add_trace(
            go.Scatter(
                x=[player_x],
                y=[player_y],
                mode='markers',
                marker=dict(
                    size=20,
                    color='#FF0000',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name='Player',
                hovertext='Player'
            ),
            row=1, col=1
        )

        # Add placeholder soul map radar
        fig.add_trace(
            go.Scatterpolar(
                r=[0.5, 0.5, 0.5, 0.5, 0.5],
                theta=['Cognitive', 'Emotional', 'Motivational', 'Social', 'Self'],
                fill='toself',
                name='Soul Map',
                fillcolor='rgba(0, 100, 200, 0.3)',
                line=dict(color='rgb(0, 100, 200)')
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title=f"Realm: {realm_type} - {self.REALM_DESCRIPTIONS.get(realm_key, '')}",
            paper_bgcolor=bg_color,
            plot_bgcolor='rgba(255,255,255,0.8)',
            showlegend=True,
            height=600,
        )

        # Update axes
        fig.update_xaxes(range=[-100, 100], title='X', row=1, col=1)
        fig.update_yaxes(range=[-100, 100], title='Y', row=1, col=1)

        # Update polar layout
        fig.update_polars(radialaxis=dict(range=[0, 1], showticklabels=False))

        return fig

    def render_soul_map_radar(self, soul_map: 'SoulMap') -> go.Figure:
        """
        Render a Soul Map as a radar chart.

        Args:
            soul_map: SoulMap instance to render

        Returns:
            Plotly Figure with radar chart
        """
        categories = ['Cognitive', 'Emotional', 'Motivational', 'Social', 'Self']

        # Get cluster means
        values = []
        for cluster_name in ['cognitive', 'emotional', 'motivational', 'social', 'self']:
            cluster_data = getattr(soul_map, cluster_name, {})
            if cluster_data:
                mean_val = np.mean(list(cluster_data.values()))
            else:
                mean_val = 0.5
            values.append(mean_val)

        # Close the polygon
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig = go.Figure(data=go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Soul Map',
            fillcolor='rgba(65, 105, 225, 0.3)',
            line=dict(color='rgb(65, 105, 225)', width=2)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Soul Map Profile"
        )

        return fig

    def render_npc_details(self, npc: 'BaseNPC') -> go.Figure:
        """
        Render detailed view of a single NPC.

        Args:
            npc: NPC to render details for

        Returns:
            Plotly Figure with NPC details
        """
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "polar"}, {"type": "bar"}],
                [{"type": "scatter", "colspan": 2}, None]
            ],
            subplot_titles=["Soul Map", "Motivational Drives", "Emotional History"]
        )

        # Soul Map radar
        soul_map = npc.soul_map
        categories = ['Cognitive', 'Emotional', 'Motivational', 'Social', 'Self']
        values = []
        for cluster_name in ['cognitive', 'emotional', 'motivational', 'social', 'self']:
            cluster_data = getattr(soul_map, cluster_name, {})
            values.append(np.mean(list(cluster_data.values())) if cluster_data else 0.5)

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Profile'
            ),
            row=1, col=1
        )

        # Motivational drives bar chart
        if hasattr(soul_map, 'motivational'):
            drives = list(soul_map.motivational.keys())[:6]
            drive_values = [soul_map.motivational.get(d, 0.5) for d in drives]

            fig.add_trace(
                go.Bar(
                    x=drives,
                    y=drive_values,
                    name='Drives',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )

        # Emotional state indicator (placeholder)
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2, 3, 4],
                y=[0.5, 0.6, 0.4, 0.7, 0.5],
                mode='lines+markers',
                name='Emotional Trend'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f"NPC Details: {getattr(npc, 'name', npc.npc_id)}",
            height=600
        )

        return fig

    def render_realm_overview(self) -> go.Figure:
        """
        Render overview of all realms with NPC distribution.

        Returns:
            Plotly Figure with realm overview
        """
        if self.env is None:
            return self._create_empty_figure("No environment set")

        realms = ['peregrine', 'spleen_towns', 'ministry', 'city', 'hollow']
        counts = []
        colors = []

        for realm in realms:
            npcs = self._get_npcs_in_realm(realm)
            counts.append(len(npcs))
            colors.append(self.REALM_COLORS.get(realm, '#808080'))

        fig = go.Figure(data=[
            go.Bar(
                x=realms,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="NPC Distribution by Realm",
            xaxis_title="Realm",
            yaxis_title="Number of NPCs"
        )

        return fig

    def render_emotion_distribution(self, realm_type: str = None) -> go.Figure:
        """
        Render distribution of emotional states in a realm.

        Args:
            realm_type: Realm to analyze (all if None)

        Returns:
            Plotly Figure with emotion distribution
        """
        if self.env is None:
            return self._create_empty_figure("No environment set")

        if realm_type:
            npcs = self._get_npcs_in_realm(realm_type)
        else:
            npcs = list(self.env.npcs.values())

        # Count emotions
        emotion_counts: Dict[str, int] = {}
        for npc in npcs:
            emotion = getattr(npc, 'emotional_state', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        fig = go.Figure(data=[
            go.Pie(
                labels=list(emotion_counts.keys()),
                values=list(emotion_counts.values()),
                marker_colors=[self.EMOTION_COLORS.get(e, '#808080')
                              for e in emotion_counts.keys()]
            )
        ])

        fig.update_layout(
            title=f"Emotional State Distribution{f' - {realm_type}' if realm_type else ''}"
        )

        return fig

    def _get_npcs_in_realm(self, realm_type: str) -> List['BaseNPC']:
        """Get NPCs in a specific realm."""
        if self.env is None:
            return []

        from ..liminal.realms import RealmType

        # Map string to RealmType enum
        realm_mapping = {
            'peregrine': RealmType.PEREGRINE,
            'spleen_towns': RealmType.SPLEEN_TOWNS,
            'ministry': RealmType.MINISTRY,
            'city': RealmType.CITY_OF_CONSTANTS,
            'hollow': RealmType.HOLLOW_REACHES,
            'nothing': RealmType.THE_NOTHING,
        }

        target_realm = realm_mapping.get(realm_type.lower())
        if target_realm is None:
            return []

        return [npc for npc in self.env.npcs.values()
                if getattr(npc, 'current_realm', None) == target_realm]

    def _get_npc_position(self, npc: 'BaseNPC') -> tuple:
        """Get NPC position, generating random if not set."""
        pos = getattr(npc, 'position', None)
        if pos:
            return pos

        # Generate deterministic random position based on NPC ID
        np.random.seed(hash(npc.npc_id) % 2**32)
        x = np.random.uniform(-80, 80)
        y = np.random.uniform(-80, 80)
        return (x, y)

    def _create_hover_text(self, npc: 'BaseNPC') -> str:
        """Create hover text for NPC."""
        lines = [
            f"<b>{getattr(npc, 'name', npc.npc_id)}</b>",
            f"Archetype: {getattr(npc, 'archetype', 'Unknown')}",
            f"Emotional State: {getattr(npc, 'emotional_state', 'neutral')}",
            f"ToM Depth: {getattr(npc, 'tom_depth', 1)}",
        ]

        # Add soul map summary if available
        if hasattr(npc, 'soul_map'):
            stability = npc.soul_map.compute_stability()
            lines.append(f"Stability: {stability:.2f}")

        return "<br>".join(lines)

    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig


# Export
__all__ = ['WorldRenderer']
