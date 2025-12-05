"""
Main Streamlit Application for ToM-NAS Liminal Architectures

Run with: streamlit run src/visualization/app.py

This application provides visualization for:
- Liminal game environment with NPCs and Soul Maps
- Theory of Mind belief inspection
- NAS evolution progress
- Experiment configuration and running
"""

import os
import sys

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional

import numpy as np
import torch


def main():
    """Main application entry point."""
    st.set_page_config(page_title="Liminal Architectures: ToM-NAS", page_icon="brain", layout="wide")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate", ["Liminal World", "Belief Inspector", "NAS Evolution", "Run Experiment", "About"]
    )

    if page == "Liminal World":
        render_liminal_world()
    elif page == "Belief Inspector":
        render_belief_inspector()
    elif page == "NAS Evolution":
        render_nas_dashboard()
    elif page == "Run Experiment":
        render_experiment_runner()
    else:
        render_about()


@st.cache_resource
def get_environment():
    """Get cached Liminal environment."""
    from src.liminal import LiminalEnvironment

    return LiminalEnvironment(population_size=200, include_heroes=True)


def render_liminal_world():
    """Render the Liminal world visualization page."""
    st.title("Liminal Architectures")
    st.markdown("*Grand Theft Ontology*")

    from src.liminal.realms import RealmType
    from src.visualization.world_renderer import WorldRenderer

    # Initialize environment
    try:
        env = get_environment()
        renderer = WorldRenderer(env)
    except Exception as e:
        st.error(f"Error initializing environment: {e}")
        return

    # Realm selector
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        realm_options = [
            "Peregrine (The Hub)",
            "Spleen Towns (The Loop)",
            "Ministry Districts (The Bureaucracy)",
            "City of Constants (The Machine)",
            "Hollow Reaches (The Shadow)",
        ]
        realm = st.selectbox("Select Realm", realm_options)

    with col2:
        # Map realm display name to type
        realm_mapping = {
            "Peregrine (The Hub)": RealmType.PEREGRINE,
            "Spleen Towns (The Loop)": RealmType.SPLEEN_TOWNS,
            "Ministry Districts (The Bureaucracy)": RealmType.MINISTRY,
            "City of Constants (The Machine)": RealmType.CITY_OF_CONSTANTS,
            "Hollow Reaches (The Shadow)": RealmType.HOLLOW_REACHES,
        }
        selected_realm = realm_mapping.get(realm, RealmType.PEREGRINE)
        npc_count = len(env.get_npcs_in_realm(selected_realm))
        st.metric("NPCs in Realm", npc_count)

    with col3:
        st.metric("Instability", f"{env.instability.instability:.1f}%")

    # Main world view
    try:
        fig = renderer.render_realm(realm)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering realm: {e}")

    # NPC details section
    st.subheader("NPCs")

    npcs = env.get_npcs_in_realm(selected_realm)[:20]

    if npcs:
        npc_names = [getattr(npc, "name", npc.npc_id) for npc in npcs]
        selected_npc_name = st.selectbox("Select NPC", npc_names)

        # Find selected NPC
        selected_npc = None
        for npc in npcs:
            if getattr(npc, "name", npc.npc_id) == selected_npc_name:
                selected_npc = npc
                break

        if selected_npc:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{getattr(selected_npc, 'name', selected_npc.npc_id)}**")
                st.markdown(f"- Archetype: {getattr(selected_npc, 'archetype', 'Unknown')}")
                st.markdown(f"- ToM Depth: {getattr(selected_npc, 'tom_depth', 1)}")
                st.markdown(f"- Emotional State: {getattr(selected_npc, 'emotional_state', 'neutral')}")

                # Active goal if available
                if hasattr(selected_npc, "active_goal") and selected_npc.active_goal:
                    st.markdown(f"- Active Goal: {selected_npc.active_goal}")

                # Stability score
                if hasattr(selected_npc, "soul_map"):
                    stability = selected_npc.soul_map.compute_stability()
                    st.markdown(f"- Psychological Stability: {stability:.2f}")

            with col2:
                # Soul Map radar
                if hasattr(selected_npc, "soul_map"):
                    soul_fig = renderer.render_soul_map_radar(selected_npc.soul_map)
                    st.plotly_chart(soul_fig, use_container_width=True)
    else:
        st.info("No NPCs found in this realm")

    # Realm overview
    st.subheader("Realm Overview")
    col1, col2 = st.columns(2)

    with col1:
        overview_fig = renderer.render_realm_overview()
        st.plotly_chart(overview_fig, use_container_width=True)

    with col2:
        emotion_fig = renderer.render_emotion_distribution(realm)
        st.plotly_chart(emotion_fig, use_container_width=True)


def render_belief_inspector():
    """Render the belief inspector page."""
    st.title("Belief Inspector")
    st.markdown("*Visualize information asymmetry and false beliefs*")

    from src.core.events import create_sally_anne_scenario, verify_information_asymmetry
    from src.visualization.belief_inspector import BeliefInspector

    inspector = BeliefInspector()

    scenario_type = st.selectbox("Scenario", ["Sally-Anne Classic", "Custom Scenario"])

    if scenario_type == "Sally-Anne Classic":
        events, questions = create_sally_anne_scenario()
        tracker = questions[0]["_tracker"]
        inspector.set_tracker(tracker)

        # Verification results
        results = verify_information_asymmetry()

        st.subheader("Scenario Verification")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sally believes marble in", results["sally_marble_belief"])
        with col2:
            st.metric("Anne believes marble in", results["anne_marble_belief"])
        with col3:
            st.metric("Reality", results["reality"])

        if results["sally_has_false_belief"]:
            st.success("Sally has FALSE BELIEF - Information asymmetry working!")
        else:
            st.error("Information asymmetry not working correctly")

        # Event timeline
        st.subheader("Event Timeline")
        agents = list(tracker.agent_beliefs.keys())
        highlight = st.selectbox("Show perspective of:", ["All"] + agents)

        timeline_fig = inspector.render_event_timeline(
            events, highlight_agent=highlight if highlight != "All" else None
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Belief comparison
        st.subheader("Agent Beliefs")
        comparison_fig = inspector.render_belief_comparison(agents, "marble")
        st.plotly_chart(comparison_fig, use_container_width=True)

        # False belief analysis
        st.subheader("False Belief Analysis")
        false_belief_fig = inspector.render_false_belief_highlight()
        st.plotly_chart(false_belief_fig, use_container_width=True)

        # Belief network
        st.subheader("Agent Belief Network")
        network_fig = inspector.render_belief_network()
        st.plotly_chart(network_fig, use_container_width=True)

    else:
        st.info("Custom scenario builder coming soon...")


def render_nas_dashboard():
    """Render the NAS evolution dashboard."""
    st.title("Neural Architecture Search")
    st.markdown("*Evolution progress and architecture analysis*")

    from src.visualization.nas_dashboard import NASDashboard

    # Try to load history or use mock data
    dashboard = NASDashboard()

    # Check for saved history
    history_file = st.sidebar.file_uploader("Upload evolution history (optional)", type=["pt", "json"])

    if history_file:
        try:
            import json

            history = json.load(history_file)
            dashboard.set_history(history)
            st.success("History loaded!")
        except Exception as e:
            st.error(f"Error loading history: {e}")
    else:
        # Generate mock history for demonstration
        np.random.seed(42)
        n_gens = 50
        mock_history = {
            "best_fitness": [0.3 + i * 0.01 + np.random.randn() * 0.02 for i in range(n_gens)],
            "avg_fitness": [0.25 + i * 0.008 + np.random.randn() * 0.03 for i in range(n_gens)],
            "diversity": [0.8 - i * 0.005 + np.random.randn() * 0.02 for i in range(n_gens)],
            "species_count": [max(1, 5 - i // 15 + np.random.randint(-1, 2)) for i in range(n_gens)],
            "best_genes": [
                {
                    "arch_type": np.random.choice(["TRN", "RSAN", "Transformer"]),
                    "hidden_dim": np.random.choice([64, 128, 256]),
                    "num_layers": np.random.randint(1, 5),
                    "num_heads": np.random.choice([2, 4, 8]),
                    "dropout_rate": np.random.uniform(0, 0.3),
                }
                for _ in range(n_gens)
            ],
        }
        dashboard.set_history(mock_history)
        st.info("Showing demonstration data. Upload evolution history to see real results.")

    # Dashboard layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fitness Over Generations")
        st.plotly_chart(dashboard.render_fitness_curves(), use_container_width=True)

    with col2:
        st.subheader("ToM Specificity")
        st.plotly_chart(dashboard.render_tom_specificity(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Population Diversity")
        st.plotly_chart(dashboard.render_diversity_chart(), use_container_width=True)

    with col2:
        st.subheader("Architecture Distribution")
        st.plotly_chart(dashboard.render_architecture_distribution(), use_container_width=True)

    # Architecture table
    st.subheader("Top Architectures")
    arch_table = dashboard.render_architecture_table(top_k=10)
    if not arch_table.empty:
        st.dataframe(arch_table)
    else:
        st.info("No architecture data available")

    # Feature importance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")
        st.plotly_chart(dashboard.render_feature_importance(), use_container_width=True)

    with col2:
        st.subheader("Pareto Front")
        st.plotly_chart(dashboard.render_pareto_front(), use_container_width=True)

    # Zero-cost proxy correlations
    st.subheader("Zero-Cost Proxy Correlations")
    st.plotly_chart(dashboard.render_zero_cost_proxy_correlations(), use_container_width=True)


def render_experiment_runner():
    """Render the experiment runner page."""
    st.title("Run Experiment")
    st.markdown("*Configure and run ToM-NAS evolution*")

    st.subheader("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        num_generations = st.slider("Generations", 10, 200, 50)
        population_size = st.slider("Population Size", 10, 100, 20)
        elite_size = st.slider("Elite Size", 1, 10, 2)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1)

    with col2:
        benchmarks = st.multiselect(
            "Benchmarks",
            ["ToMi (False Belief)", "SocialIQA (Social Reasoning)", "Social Games (Multi-Agent)", "Liminal NPCs"],
            default=["ToMi (False Belief)"],
        )

        arch_types = st.multiselect(
            "Architecture Types",
            ["TRN (Transparent RNN)", "RSAN (Recursive Self-Attention)", "Transformer"],
            default=["TRN (Transparent RNN)", "RSAN (Recursive Self-Attention)"],
        )

        device = st.selectbox("Device", ["cpu", "cuda"])

    # Advanced settings
    with st.expander("Advanced Settings"):
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.7)
        tournament_size = st.slider("Tournament Size", 2, 10, 3)
        use_speciation = st.checkbox("Use Speciation", value=True)
        use_coevolution = st.checkbox("Use Coevolution", value=True)

    # Run button
    if st.button("Start Evolution", type="primary"):
        st.warning("Full evolution not implemented in demo. Running quick simulation...")

        progress = st.progress(0)
        status = st.empty()
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        # Simulate evolution
        for gen in range(min(num_generations, 20)):  # Cap at 20 for demo
            # Simulate progress
            import time

            time.sleep(0.1)

            progress.progress((gen + 1) / min(num_generations, 20))
            status.text(f"Generation {gen + 1}/{min(num_generations, 20)}")

            # Update metrics periodically
            if gen % 2 == 0:
                with metrics_col1:
                    st.metric("Best Fitness", f"{0.3 + gen * 0.02:.3f}")
                with metrics_col2:
                    st.metric("Diversity", f"{0.8 - gen * 0.01:.3f}")
                with metrics_col3:
                    st.metric("Species", f"{max(1, 5 - gen // 5)}")

        st.success("Simulation complete!")
        st.balloons()


def render_about():
    """Render the about page."""
    st.title("About ToM-NAS")

    st.markdown("""
    ## Theory of Mind Neural Architecture Search

    This system uses evolutionary algorithms to discover neural network architectures
    that can perform **Theory of Mind reasoning** - understanding that others have
    beliefs, desires, and intentions that may differ from one's own.

    ### Key Components

    #### Liminal Architectures
    A psychological open-world game environment with 200+ NPCs, each characterized by
    a **60-dimensional Soul Map** representing their complete psychological profile.

    #### Soul Map Dimensions
    - **Cognitive** (12 dims): Processing, reasoning, metacognition
    - **Emotional** (12 dims): Affect, sensitivity, regulation
    - **Motivational** (12 dims): Drives, goals, risk orientation
    - **Social** (12 dims): Trust, cooperation, social intelligence
    - **Self** (12 dims): Identity, coherence, agency

    #### Information Asymmetry
    Events track which agents observed them, enabling realistic false belief scenarios
    like the classic Sally-Anne test.

    #### NAS Evolution
    Evolutionary search over three architecture families:
    - **TRN**: Transparent Recurrent Networks (interpretable)
    - **RSAN**: Recursive Self-Attention Networks (compositional)
    - **Transformer**: Standard attention-based architectures

    #### Benchmarks
    - **ToMi**: False belief scenarios (Sally-Anne style)
    - **SocialIQA**: Naturalistic social reasoning questions
    - **Social Games**: Interactive multi-agent scenarios

    ### Dissertation Hypothesis

    > Complex Theory of Mind tasks will independently discover skip connections
    > and attention mechanisms, while simple tasks will not.

    ---

    **Version**: 2.0.0
    **License**: Research/Educational Use
    """)

    # Show system info
    st.subheader("System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"PyTorch Version: {torch.__version__}")
        st.info(f"CUDA Available: {torch.cuda.is_available()}")

    with col2:
        st.info(f"NumPy Version: {np.__version__}")
        if torch.cuda.is_available():
            st.info(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
