"""
Belief Inspector: Visualize information asymmetry and false beliefs.

Shows:
- Event timeline with observer badges
- Belief state per agent (what they think is where)
- False belief highlighting (when belief != reality)
- Second-order beliefs (what A thinks B thinks)
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import streamlit as st

if TYPE_CHECKING:
    from ..core.events import Event, InformationAsymmetryTracker


class BeliefInspector:
    """Interactive belief visualization for Theory of Mind scenarios."""

    EVENT_COLORS = {
        "object_placed": "#4CAF50",
        "object_moved": "#2196F3",
        "agent_entered": "#9C27B0",
        "agent_exited": "#FF9800",
        "communication": "#E91E63",
        "observation": "#00BCD4",
    }

    AGENT_COLORS = {
        "Sally": "#FF6B6B",
        "Anne": "#4ECDC4",
        "Observer": "#95A5A6",
        "Mary": "#F39C12",
        "John": "#3498DB",
    }

    def __init__(self):
        """Initialize the belief inspector."""
        self.tracker = None

    def set_tracker(self, tracker: "InformationAsymmetryTracker"):
        """Set the information asymmetry tracker to inspect."""
        self.tracker = tracker

    def render_event_timeline(self, events: List["Event"], highlight_agent: str = None) -> go.Figure:
        """
        Render events as a vertical timeline with observer badges.

        Args:
            events: List of events to display
            highlight_agent: Agent to highlight (shows their perspective)

        Returns:
            Plotly Figure with timeline
        """
        if not events:
            return self._create_empty_figure("No events to display")

        fig = go.Figure()

        # Timeline line
        fig.add_trace(
            go.Scatter(x=[0, 0], y=[0, len(events)], mode="lines", line=dict(color="gray", width=2), showlegend=False)
        )

        # Add events
        for i, event in enumerate(events):
            y_pos = i + 0.5

            # Event color based on type
            color = self.EVENT_COLORS.get(event.event_type.value, "#808080")

            # Dim if highlight_agent didn't observe this event
            if highlight_agent and highlight_agent not in event.observed_by:
                color = "#D3D3D3"  # Light gray for unobserved

            # Event marker
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[y_pos],
                    mode="markers",
                    marker=dict(size=20, color=color, symbol="circle"),
                    name=event.event_type.value,
                    hovertext=self._create_event_hover(event),
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            # Event description
            desc = self._format_event_description(event)
            fig.add_annotation(x=0.5, y=y_pos, text=desc, showarrow=False, xanchor="left", font=dict(size=10))

            # Observer badges
            observers = list(event.observed_by)
            for j, observer in enumerate(observers[:5]):  # Max 5 badges
                badge_x = 3 + j * 0.8
                badge_color = self.AGENT_COLORS.get(observer, "#808080")

                fig.add_trace(
                    go.Scatter(
                        x=[badge_x],
                        y=[y_pos],
                        mode="markers+text",
                        marker=dict(size=15, color=badge_color),
                        text=observer[0],  # First letter
                        textposition="middle center",
                        textfont=dict(size=8, color="white"),
                        hovertext=f"Observed by: {observer}",
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

        # Layout
        fig.update_layout(
            title="Event Timeline",
            xaxis=dict(range=[-1, 8], showticklabels=False, showgrid=False),
            yaxis=dict(
                range=[-0.5, len(events) + 0.5],
                title="Time",
                tickmode="array",
                tickvals=list(range(len(events))),
                ticktext=[f"t={i}" for i in range(len(events))],
            ),
            height=100 + len(events) * 80,
            showlegend=False,
        )

        return fig

    def render_belief_comparison(self, agents: List[str], object_name: str = "marble") -> go.Figure:
        """
        Side-by-side comparison of what each agent believes.

        Args:
            agents: List of agent IDs to compare
            object_name: Object to check beliefs about

        Returns:
            Plotly Figure with belief comparison
        """
        if self.tracker is None:
            return self._create_empty_figure("No tracker set")

        # Get beliefs and reality
        beliefs = {}
        for agent in agents:
            beliefs[agent] = self.tracker.get_agent_belief(agent, object_name)

        reality = self.tracker.get_reality(object_name)

        # Create comparison chart
        fig = go.Figure()

        # Reality bar
        locations = list(set(list(beliefs.values()) + [reality]))
        locations = [loc for loc in locations if loc is not None]

        x_positions = list(range(len(agents) + 1))
        bar_colors = []
        bar_texts = []

        # Reality
        bar_texts.append(f"Reality<br>{reality}")
        bar_colors.append("#2ECC71")  # Green for reality

        # Agent beliefs
        for agent in agents:
            belief = beliefs.get(agent)
            bar_texts.append(f"{agent}<br>believes: {belief}")

            # Color based on correctness
            if belief == reality:
                bar_colors.append("#2ECC71")  # Green - correct
            else:
                bar_colors.append("#E74C3C")  # Red - false belief

        fig.add_trace(
            go.Bar(
                x=["Reality"] + agents,
                y=[1] * (len(agents) + 1),
                marker_color=bar_colors,
                text=bar_texts,
                textposition="inside",
                insidetextfont=dict(size=12, color="white"),
            )
        )

        fig.update_layout(
            title=f"Beliefs about {object_name} location", yaxis=dict(visible=False), showlegend=False, height=300
        )

        return fig

    def render_false_belief_highlight(self) -> go.Figure:
        """
        Highlight where beliefs diverge from reality.

        Returns:
            Plotly Figure showing false beliefs
        """
        if self.tracker is None:
            return self._create_empty_figure("No tracker set")

        # Find all false beliefs
        false_beliefs = []

        for agent_id, belief_state in self.tracker.agent_beliefs.items():
            for obj, loc in belief_state.object_locations.items():
                reality = self.tracker.get_reality(obj)
                if reality and loc != reality:
                    false_beliefs.append({"agent": agent_id, "object": obj, "belief": loc, "reality": reality})

        if not false_beliefs:
            fig = go.Figure()
            fig.add_annotation(
                text="No false beliefs detected - all agents have accurate beliefs",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="green"),
            )
            fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=200)
            return fig

        # Create Sankey diagram showing belief divergence
        agents = list(set(fb["agent"] for fb in false_beliefs))
        objects = list(set(fb["object"] for fb in false_beliefs))
        locations = list(set(fb["belief"] for fb in false_beliefs) | set(fb["reality"] for fb in false_beliefs))

        # Build node labels
        node_labels = (
            agents + objects + [f"{loc} (believed)" for loc in locations] + [f"{loc} (reality)" for loc in locations]
        )

        # Build links
        source = []
        target = []
        value = []
        colors = []

        for fb in false_beliefs:
            agent_idx = agents.index(fb["agent"])
            obj_idx = len(agents) + objects.index(fb["object"])
            belief_idx = len(agents) + len(objects) + locations.index(fb["belief"])
            reality_idx = len(agents) + len(objects) + len(locations) + locations.index(fb["reality"])

            # Agent -> Object
            source.append(agent_idx)
            target.append(obj_idx)
            value.append(1)
            colors.append("rgba(231, 76, 60, 0.5)")  # Red

            # Object -> Believed location
            source.append(obj_idx)
            target.append(belief_idx)
            value.append(1)
            colors.append("rgba(231, 76, 60, 0.5)")

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=node_labels),
                    link=dict(source=source, target=target, value=value, color=colors),
                )
            ]
        )

        fig.update_layout(title="False Belief Analysis", height=400)

        return fig

    def render_second_order_beliefs(self, agent1: str, agent2: str, object_name: str = "marble") -> go.Figure:
        """
        Visualize second-order beliefs: what agent1 thinks agent2 believes.

        Args:
            agent1: First agent (the thinker)
            agent2: Second agent (whose beliefs are being modeled)
            object_name: Object in question

        Returns:
            Plotly Figure showing nested beliefs
        """
        if self.tracker is None:
            return self._create_empty_figure("No tracker set")

        # Get beliefs
        agent1_belief = self.tracker.get_agent_belief(agent1, object_name)
        agent2_belief = self.tracker.get_agent_belief(agent2, object_name)
        reality = self.tracker.get_reality(object_name)

        # For second-order: what does agent1 think agent2 believes?
        # This requires knowing what events agent1 knows agent2 observed
        # For simplicity, we'll show the direct beliefs

        fig = go.Figure()

        # Create hierarchical visualization
        # Level 0: Reality
        # Level 1: First-order beliefs
        # Level 2: Second-order beliefs (what A thinks B thinks)

        levels = ["Reality", f"{agent1} believes", f"{agent2} believes", f"{agent1} thinks {agent2} believes"]
        values = [reality, agent1_belief, agent2_belief, agent1_belief]  # Simplified

        colors = ["#2ECC71"]  # Reality is green
        for i, val in enumerate(values[1:]):
            if val == reality:
                colors.append("#2ECC71")  # Correct
            else:
                colors.append("#E74C3C")  # False belief

        fig.add_trace(
            go.Sunburst(
                labels=levels,
                parents=["", "Reality", "Reality", f"{agent1} believes"],
                values=[4, 3, 3, 2],
                marker=dict(colors=colors),
                branchvalues="total",
                hovertext=[f"Location: {v}" for v in values],
                hoverinfo="label+text",
            )
        )

        fig.update_layout(title=f"Nested Beliefs about {object_name}", height=400)

        return fig

    def render_belief_network(self) -> go.Figure:
        """
        Render network visualization of agent beliefs about each other.

        Returns:
            Plotly Figure with belief network
        """
        if self.tracker is None:
            return self._create_empty_figure("No tracker set")

        agents = list(self.tracker.agent_beliefs.keys())

        if len(agents) < 2:
            return self._create_empty_figure("Need at least 2 agents for network")

        # Create network layout
        n = len(agents)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        fig = go.Figure()

        # Draw edges (beliefs about each other)
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    # Check if agent1 has any belief about agent2's location
                    agent1_beliefs = self.tracker.agent_beliefs[agent1]

                    # Draw connection
                    fig.add_trace(
                        go.Scatter(
                            x=[x_pos[i], x_pos[j]],
                            y=[y_pos[i], y_pos[j]],
                            mode="lines",
                            line=dict(color="lightgray", width=1),
                            showlegend=False,
                            hoverinfo="none",
                        )
                    )

        # Draw nodes (agents)
        colors = [self.AGENT_COLORS.get(agent, "#808080") for agent in agents]

        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode="markers+text",
                marker=dict(size=40, color=colors, line=dict(width=2, color="black")),
                text=agents,
                textposition="middle center",
                textfont=dict(size=12, color="white"),
                hoverinfo="text",
                hovertext=[
                    f"{a}: {len(self.tracker.agent_beliefs[a].observed_events)} events observed" for a in agents
                ],
                showlegend=False,
            )
        )

        fig.update_layout(
            title="Agent Belief Network",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400,
            showlegend=False,
        )

        return fig

    def _format_event_description(self, event: "Event") -> str:
        """Format event into readable description."""
        etype = event.event_type.value

        if etype == "object_placed":
            return f"{event.actor} placed {event.target} in {event.target_location}"
        elif etype == "object_moved":
            return f"{event.actor} moved {event.target} from {event.source_location} to {event.target_location}"
        elif etype == "agent_entered":
            return f"{event.actor} entered {event.target_location}"
        elif etype == "agent_exited":
            return f"{event.actor} left {event.source_location}"
        else:
            return f"{event.actor}: {etype}"

    def _create_event_hover(self, event: "Event") -> str:
        """Create hover text for event."""
        lines = [
            f"<b>{event.event_type.value}</b>",
            f"Time: {event.timestamp}",
            f"Actor: {event.actor}",
        ]
        if event.target:
            lines.append(f"Target: {event.target}")
        if event.target_location:
            lines.append(f"Location: {event.target_location}")
        lines.append(f"Observers: {', '.join(event.observed_by)}")

        return "<br>".join(lines)

    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=200)
        return fig


# Streamlit integration helpers
def render_belief_inspector_ui(tracker: "InformationAsymmetryTracker"):
    """
    Render the belief inspector UI in Streamlit.

    Args:
        tracker: Information asymmetry tracker to inspect
    """
    inspector = BeliefInspector()
    inspector.set_tracker(tracker)

    st.subheader("Event Timeline")
    events = tracker.world.events

    # Agent filter
    agents = list(tracker.agent_beliefs.keys())
    highlight_agent = st.selectbox("Highlight perspective of:", ["All"] + agents)

    fig = inspector.render_event_timeline(events, highlight_agent=highlight_agent if highlight_agent != "All" else None)
    st.plotly_chart(fig, use_container_width=True)

    # Belief comparison
    st.subheader("Belief Comparison")
    st.plotly_chart(inspector.render_belief_comparison(agents), use_container_width=True)

    # False belief analysis
    st.subheader("False Belief Analysis")
    st.plotly_chart(inspector.render_false_belief_highlight(), use_container_width=True)


# Export
__all__ = ["BeliefInspector", "render_belief_inspector_ui"]
