#!/usr/bin/env python3
"""
ToM-NAS Interactive GUI Application
Theory of Mind Neural Architecture Search - Desktop Interface

A user-friendly interface for running and understanding ToM-NAS experiments.
Designed for both technical researchers and non-technical users.
"""
import sys
import os
import threading
import queue
import time
from typing import Dict, Any, Optional
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from dataclasses import dataclass

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import matplotlib for embedded plots
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import torch
import numpy as np


@dataclass
class ExplainerText:
    """Plain-English explanations for non-technical users."""

    @staticmethod
    def fitness() -> str:
        return """FITNESS SCORE (0.0 - 1.0)

Think of this like a "grade" for how well the AI understands
other minds. Higher is better!

- 0.0-0.3: Poor - The AI is just guessing
- 0.3-0.5: Learning - Starting to understand
- 0.5-0.7: Good - Can predict what others think
- 0.7-0.9: Excellent - Deep understanding
- 0.9-1.0: Expert - Human-like comprehension"""

    @staticmethod
    def species() -> str:
        return """SPECIES (Architecture Types)

We test 3 different "brain designs" to see which works best:

- TRN (Transparent RNN): Like short-term memory
  Good at: Sequential thinking, step-by-step reasoning

- RSAN (Recursive Self-Attention): Like focused attention
  Good at: Spotting important details, relationships

- Transformer: Like parallel processing
  Good at: Big picture understanding, complex patterns

They compete AND cooperate, like different species in nature."""

    @staticmethod
    def tom_order() -> str:
        return """THEORY OF MIND ORDERS (0-5)

How deeply can the AI think about what others think?

Order 0: "I see the ball" (just facts)
Order 1: "Sally thinks the ball is here" (beliefs)
Order 2: "Anne knows Sally thinks..." (beliefs about beliefs)
Order 3: "Sally knows Anne knows Sally thinks..."
Order 4: Even deeper nesting...
Order 5: Maximum depth - very hard!

Like nesting dolls - each level is inside the previous one.
Humans typically handle orders 2-4 comfortably."""

    @staticmethod
    def zombie() -> str:
        return """ZOMBIE DETECTION

A "philosophical zombie" looks human but has no real understanding.
These tests check if the AI TRULY understands or just pretends.

Tests include:
- Does it track beliefs correctly?
- Does it understand cause and effect?
- Does it know what it knows? (metacognition)
- Are its responses emotionally appropriate?

Passing these means the AI isn't just pattern-matching."""


class ToMNASApplication:
    """Main GUI Application for ToM-NAS."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ToM-NAS - Theory of Mind Neural Architecture Search")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Set icon and colors
        self.setup_style()

        # State
        self.evolution_running = False
        self.evolution_thread = None
        self.message_queue = queue.Queue()
        self.history = {'fitness': [], 'diversity': [], 'tom': [], 'zombie': []}
        self.current_generation = 0

        # Build UI
        self.create_menu()
        self.create_main_interface()

        # Start message processor
        self.process_messages()

    def setup_style(self):
        """Configure visual style."""
        style = ttk.Style()
        style.theme_use('clam')

        # Colors inspired by classic game UIs - warm, inviting
        self.colors = {
            'bg': '#2b2b3d',
            'fg': '#ffffff',
            'accent': '#4a90d9',
            'success': '#5cb85c',
            'warning': '#f0ad4e',
            'danger': '#d9534f',
            'panel': '#363648'
        }

        self.root.configure(bg=self.colors['bg'])

        style.configure('TFrame', background=self.colors['bg'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        style.configure('TButton', padding=10)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subheader.TLabel', font=('Arial', 12))
        style.configure('Big.TButton', font=('Arial', 12, 'bold'), padding=15)

    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Experiment", command=self.new_experiment)
        file_menu.add_command(label="Load Results...", command=self.load_results)
        file_menu.add_command(label="Save Results...", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="What is ToM-NAS?", command=self.show_about)
        help_menu.add_command(label="Understanding the Results", command=self.show_help_results)
        help_menu.add_command(label="Glossary", command=self.show_glossary)

    def create_main_interface(self):
        """Create the main application interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header, text="Theory of Mind", style='Header.TLabel').pack(side=tk.LEFT)
        ttk.Label(header, text=" Neural Architecture Search",
                  style='Subheader.TLabel').pack(side=tk.LEFT, pady=(5, 0))

        # Help button
        help_btn = ttk.Button(header, text="? Help", command=self.show_quick_help)
        help_btn.pack(side=tk.RIGHT)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Evolution Control
        self.create_evolution_tab()

        # Tab 2: Results & Analysis
        self.create_results_tab()

        # Tab 3: Agent Inspector
        self.create_inspector_tab()

        # Tab 4: Learn (for non-technical users)
        self.create_learn_tab()

    def create_evolution_tab(self):
        """Create the evolution control panel."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Run Evolution  ")

        # Left panel - Controls
        left_panel = ttk.LabelFrame(tab, text="Evolution Settings", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Population size
        ttk.Label(left_panel, text="Population Size:").pack(anchor=tk.W)
        self.pop_size_var = tk.IntVar(value=12)
        pop_spin = ttk.Spinbox(left_panel, from_=6, to=50, textvariable=self.pop_size_var, width=10)
        pop_spin.pack(anchor=tk.W, pady=(0, 10))
        self.add_tooltip(pop_spin, "Number of AI 'brains' competing each generation.\nMore = better exploration but slower.")

        # Generations
        ttk.Label(left_panel, text="Generations:").pack(anchor=tk.W)
        self.gen_var = tk.IntVar(value=20)
        gen_spin = ttk.Spinbox(left_panel, from_=5, to=200, textvariable=self.gen_var, width=10)
        gen_spin.pack(anchor=tk.W, pady=(0, 10))
        self.add_tooltip(gen_spin, "How many 'generations' to evolve.\nMore = better results but takes longer.")

        # Mutation rate
        ttk.Label(left_panel, text="Mutation Rate:").pack(anchor=tk.W)
        self.mutation_var = tk.DoubleVar(value=0.1)
        mutation_scale = ttk.Scale(left_panel, from_=0.01, to=0.5, variable=self.mutation_var, length=150)
        mutation_scale.pack(anchor=tk.W)
        self.mutation_label = ttk.Label(left_panel, text="0.10")
        self.mutation_label.pack(anchor=tk.W, pady=(0, 10))
        mutation_scale.configure(command=lambda v: self.mutation_label.configure(text=f"{float(v):.2f}"))

        # Architecture types
        ttk.Label(left_panel, text="Architectures:").pack(anchor=tk.W, pady=(10, 0))
        self.arch_trn = tk.BooleanVar(value=True)
        self.arch_rsan = tk.BooleanVar(value=True)
        self.arch_trans = tk.BooleanVar(value=True)

        ttk.Checkbutton(left_panel, text="TRN (Recurrent)", variable=self.arch_trn).pack(anchor=tk.W)
        ttk.Checkbutton(left_panel, text="RSAN (Attention)", variable=self.arch_rsan).pack(anchor=tk.W)
        ttk.Checkbutton(left_panel, text="Transformer", variable=self.arch_trans).pack(anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        self.start_btn = ttk.Button(button_frame, text="START", style='Big.TButton',
                                     command=self.start_evolution)
        self.start_btn.pack(fill=tk.X, pady=5)

        self.stop_btn = ttk.Button(button_frame, text="STOP", command=self.stop_evolution,
                                    state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=5)

        # Right panel - Live visualization
        right_panel = ttk.LabelFrame(tab, text="Live Progress", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Progress info
        info_frame = ttk.Frame(right_panel)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(info_frame, text="Generation:").pack(side=tk.LEFT)
        self.gen_label = ttk.Label(info_frame, text="0/0", font=('Arial', 12, 'bold'))
        self.gen_label.pack(side=tk.LEFT, padx=10)

        ttk.Label(info_frame, text="Best Fitness:").pack(side=tk.LEFT, padx=(20, 0))
        self.fitness_label = ttk.Label(info_frame, text="--", font=('Arial', 12, 'bold'))
        self.fitness_label.pack(side=tk.LEFT, padx=10)

        ttk.Label(info_frame, text="Species:").pack(side=tk.LEFT, padx=(20, 0))
        self.species_label = ttk.Label(info_frame, text="--", font=('Arial', 12, 'bold'))
        self.species_label.pack(side=tk.LEFT, padx=10)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(right_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status text
        self.status_text = scrolledtext.ScrolledText(right_panel, height=8, width=60,
                                                      bg='#1e1e2e', fg='#00ff00',
                                                      font=('Consolas', 10))
        self.status_text.pack(fill=tk.X, pady=(0, 10))
        self.status_text.insert(tk.END, "Ready to start evolution...\n")

        # Live plot
        if MATPLOTLIB_AVAILABLE:
            self.create_live_plot(right_panel)
        else:
            ttk.Label(right_panel, text="Install matplotlib for live plots").pack()

    def create_live_plot(self, parent):
        """Create embedded matplotlib plot for live updates."""
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor='#2b2b3d')
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#363648')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')

        self.ax1.set_title('Fitness Over Time')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')

        self.ax2.set_title('ToM Performance')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Score')

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_results_tab(self):
        """Create results and analysis tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Results  ")

        # Results will be populated after evolution
        self.results_frame = ttk.Frame(tab)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.results_frame, text="Run an evolution to see results here.",
                  font=('Arial', 14)).pack(pady=50)

    def create_inspector_tab(self):
        """Create agent inspector tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Agent Inspector  ")

        ttk.Label(tab, text="Agent Inspector", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Label(tab, text="Examine individual AI agents in detail").pack(anchor=tk.W, pady=(0, 20))

        # Agent selector
        select_frame = ttk.Frame(tab)
        select_frame.pack(fill=tk.X, pady=10)

        ttk.Label(select_frame, text="Select Agent:").pack(side=tk.LEFT)
        self.agent_selector = ttk.Combobox(select_frame, values=["No agents yet"], width=40)
        self.agent_selector.pack(side=tk.LEFT, padx=10)
        self.agent_selector.set("No agents yet")

        ttk.Button(select_frame, text="Inspect", command=self.inspect_agent).pack(side=tk.LEFT)

        # Inspector output
        self.inspector_text = scrolledtext.ScrolledText(tab, height=30, width=80,
                                                         font=('Consolas', 10))
        self.inspector_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_learn_tab(self):
        """Create educational tab for non-technical users."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Learn  ")

        # Scrollable content
        canvas = tk.Canvas(tab, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Content
        ttk.Label(scrollable_frame, text="Understanding ToM-NAS",
                  style='Header.TLabel').pack(anchor=tk.W, pady=(0, 20))

        sections = [
            ("What is Theory of Mind?", """
Theory of Mind (ToM) is the ability to understand that others have
their own thoughts, beliefs, and feelings that may differ from yours.

For example: If you hide a toy while a friend isn't looking,
you know they'll look where THEY last saw it, not where you
actually put it. That's Theory of Mind!

This project teaches AI to develop this crucial social ability."""),

            ("What This Program Does", """
ToM-NAS evolves AI "brains" that can understand what others think.

It works like natural evolution:
1. Create a population of different AI designs
2. Test how well each understands others' minds
3. The best designs "reproduce" with small changes
4. Repeat for many generations

Over time, the AIs get better at understanding minds!"""),

            ("Understanding the Metrics", ExplainerText.fitness() + "\n\n" +
             ExplainerText.species() + "\n\n" + ExplainerText.tom_order()),

            ("The Sally-Anne Test", """
A famous test for Theory of Mind:

1. Sally puts her marble in a BASKET
2. Sally leaves the room
3. Anne moves the marble to a BOX
4. Sally comes back

Question: Where will Sally look for her marble?

Correct answer: The BASKET (where Sally THINKS it is)
Wrong answer: The BOX (where it actually is)

If you said "basket," you have Theory of Mind!
You understood Sally's FALSE BELIEF - she believes
something that isn't actually true.

Our AI must pass this test at multiple levels of complexity."""),
        ]

        for title, content in sections:
            frame = ttk.LabelFrame(scrollable_frame, text=title, padding="10")
            frame.pack(fill=tk.X, pady=10, padx=5)

            text = tk.Text(frame, wrap=tk.WORD, height=content.count('\n') + 2,
                          bg='#363648', fg='white', font=('Arial', 11),
                          relief=tk.FLAT, padx=10, pady=10)
            text.insert(tk.END, content.strip())
            text.configure(state=tk.DISABLED)
            text.pack(fill=tk.X)

    def add_tooltip(self, widget, text):
        """Add hover tooltip to widget."""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            label = tk.Label(tooltip, text=text, justify=tk.LEFT,
                           background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                           font=("Arial", 10), padx=5, pady=5)
            label.pack()

            widget._tooltip = tooltip

        def hide_tooltip(event):
            if hasattr(widget, '_tooltip'):
                widget._tooltip.destroy()
                del widget._tooltip

        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)

    def start_evolution(self):
        """Start the evolution process."""
        self.evolution_running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        # Clear history
        self.history = {'fitness': [], 'diversity': [], 'tom': [], 'zombie': []}
        self.current_generation = 0

        # Clear status
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "Starting evolution...\n")

        # Start evolution in background thread
        self.evolution_thread = threading.Thread(target=self.run_evolution_thread)
        self.evolution_thread.daemon = True
        self.evolution_thread.start()

    def stop_evolution(self):
        """Stop the evolution process."""
        self.evolution_running = False
        self.message_queue.put(('status', 'Evolution stopped by user'))

    def run_evolution_thread(self):
        """Run evolution in background thread."""
        try:
            # Import here to avoid slow startup
            from src.core.ontology import SoulMapOntology
            from src.core.beliefs import BeliefNetwork
            from src.world.social_world import SocialWorld4
            from src.evolution.nas_engine import NASEngine, EvolutionConfig
            from src.evaluation.tom_benchmarks import ToMBenchmarkSuite
            from src.evaluation.zombie_detection import ZombieDetectionSuite

            self.message_queue.put(('status', 'Initializing components...'))

            # Initialize
            ontology = SoulMapOntology()
            world = SocialWorld4(num_agents=10, ontology_dim=181, num_zombies=2)
            belief_net = BeliefNetwork(num_agents=10, ontology_dim=181, max_order=5)

            config = EvolutionConfig(
                population_size=self.pop_size_var.get(),
                num_generations=self.gen_var.get(),
                mutation_rate=self.mutation_var.get()
            )

            engine = NASEngine(config, world, belief_net)
            tom_suite = ToMBenchmarkSuite(input_dim=191)
            zombie_suite = ZombieDetectionSuite()

            self.message_queue.put(('status', f'Population: {config.population_size}, Generations: {config.num_generations}'))

            engine.initialize_population()
            self.message_queue.put(('status', 'Population initialized'))

            num_gens = self.gen_var.get()

            for gen in range(num_gens):
                if not self.evolution_running:
                    break

                engine.evolve_generation()

                # Get stats
                best_fitness = engine.history['best_fitness'][-1] if engine.history['best_fitness'] else 0
                avg_fitness = engine.history['avg_fitness'][-1] if engine.history['avg_fitness'] else 0
                diversity = engine.history['diversity'][-1] if engine.history['diversity'] else 0
                species_count = engine.species_manager.get_species_count() if engine.species_manager else 1

                # Evaluate best individual
                if engine.best_individual:
                    tom_results = tom_suite.run_full_evaluation(engine.best_individual.model)
                    zombie_results = zombie_suite.run_full_evaluation(
                        engine.best_individual.model, {'input_dim': 191}
                    )
                    tom_score = tom_results['overall_score']
                    zombie_score = 1 - zombie_results.get('zombie_probability', 0.5)
                else:
                    tom_score = 0
                    zombie_score = 0.5

                # Update history
                self.history['fitness'].append(best_fitness)
                self.history['diversity'].append(diversity)
                self.history['tom'].append(tom_score)
                self.history['zombie'].append(zombie_score)

                # Get population breakdown for display
                population_info = []
                for ind in engine.population[:10]:  # Show top 10
                    arch = ind.gene.gene_dict.get('arch_type', 'Unknown')
                    fit = f"{ind.fitness:.4f}" if ind.fitness is not None else "Evaluating..."
                    population_info.append(f"{arch}: {fit}")

                # Send update
                self.message_queue.put(('update', {
                    'generation': gen + 1,
                    'total_gens': num_gens,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'species_count': species_count,
                    'tom_score': tom_score,
                    'best_arch': engine.best_individual.gene.gene_dict['arch_type'] if engine.best_individual else 'Initializing...',
                    'population_breakdown': population_info
                }))

            # Final results
            if engine.best_individual:
                final_tom = tom_suite.run_full_evaluation(engine.best_individual.model)
                final_zombie = zombie_suite.run_full_evaluation(
                    engine.best_individual.model, {'input_dim': 191}
                )

                self.message_queue.put(('complete', {
                    'best_fitness': engine.best_individual.fitness,
                    'best_arch': engine.best_individual.gene.gene_dict['arch_type'],
                    'tom_results': final_tom,
                    'zombie_results': final_zombie,
                    'gene_config': engine.best_individual.gene.gene_dict
                }))

        except Exception as e:
            self.message_queue.put(('error', str(e)))

    def process_messages(self):
        """Process messages from evolution thread."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()

                if msg_type == 'status':
                    self.status_text.insert(tk.END, f"{data}\n")
                    self.status_text.see(tk.END)

                elif msg_type == 'update':
                    self.update_display(data)

                elif msg_type == 'complete':
                    self.evolution_complete(data)

                elif msg_type == 'error':
                    messagebox.showerror("Error", f"Evolution failed: {data}")
                    self.start_btn.configure(state=tk.NORMAL)
                    self.stop_btn.configure(state=tk.DISABLED)

        except queue.Empty:
            pass

        self.root.after(100, self.process_messages)

    def update_display(self, data):
        """Update display with evolution progress."""
        gen = data['generation']
        total = data['total_gens']

        self.gen_label.configure(text=f"{gen}/{total}")
        self.fitness_label.configure(text=f"{data['best_fitness']:.3f}")
        self.species_label.configure(text=str(data['species_count']))

        self.progress_var.set((gen / total) * 100)

        # Show generation summary with architecture breakdown
        status_line = f"Gen {gen}: Best={data['best_fitness']:.4f}, Avg={data.get('avg_fitness', 0):.4f}, "
        status_line += f"ToM={data['tom_score']:.3f}, Arch={data['best_arch']}"
        self.status_text.insert(tk.END, status_line + "\n")

        # Show population breakdown if available
        if 'population_breakdown' in data and data['population_breakdown']:
            # Count by architecture
            arch_counts = {}
            for info in data['population_breakdown']:
                arch = info.split(':')[0].strip()
                arch_counts[arch] = arch_counts.get(arch, 0) + 1
            breakdown = ", ".join([f"{k}:{v}" for k, v in arch_counts.items()])
            self.status_text.insert(tk.END, f"  Population: {breakdown}\n")

        self.status_text.see(tk.END)

        # Update plot
        if MATPLOTLIB_AVAILABLE and len(self.history['fitness']) > 0:
            self.ax1.clear()
            self.ax1.plot(self.history['fitness'], 'b-', linewidth=2, label='Best')
            self.ax1.set_title('Fitness Over Time', color='white')
            self.ax1.set_xlabel('Generation', color='white')
            self.ax1.set_ylabel('Fitness', color='white')
            self.ax1.set_facecolor('#363648')
            self.ax1.tick_params(colors='white')

            self.ax2.clear()
            self.ax2.plot(self.history['tom'], 'purple', linewidth=2, label='ToM')
            self.ax2.plot(self.history['zombie'], 'orange', linewidth=2, label='Human-like')
            self.ax2.set_title('ToM & Zombie Detection', color='white')
            self.ax2.set_xlabel('Generation', color='white')
            self.ax2.set_ylabel('Score', color='white')
            self.ax2.legend(facecolor='#363648', labelcolor='white')
            self.ax2.set_facecolor('#363648')
            self.ax2.tick_params(colors='white')

            self.canvas.draw()

    def evolution_complete(self, data):
        """Handle evolution completion."""
        self.evolution_running = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

        self.status_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.status_text.insert(tk.END, "EVOLUTION COMPLETE!\n")
        self.status_text.insert(tk.END, f"Best Fitness: {data['best_fitness']:.4f}\n")
        self.status_text.insert(tk.END, f"Best Architecture: {data['best_arch']}\n")
        self.status_text.insert(tk.END, "="*50 + "\n")
        self.status_text.see(tk.END)

        # Update results tab
        self.populate_results(data)

        # Switch to results tab
        self.notebook.select(1)

    def populate_results(self, data):
        """Populate results tab with final data."""
        # Clear existing
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Header
        ttk.Label(self.results_frame, text="Evolution Results",
                  style='Header.TLabel').pack(anchor=tk.W, pady=(0, 20))

        # Summary card
        summary = ttk.LabelFrame(self.results_frame, text="Summary", padding="15")
        summary.pack(fill=tk.X, pady=10)

        summary_text = f"""
Best Fitness Achieved: {data['best_fitness']:.4f}
Best Architecture: {data['best_arch']}
Max ToM Order Passed: {data['tom_results'].get('max_tom_order', -1)}
Hierarchy Valid: {'Yes' if data['tom_results'].get('hierarchy_valid', False) else 'No'}
        """
        ttk.Label(summary, text=summary_text.strip(), font=('Consolas', 11)).pack(anchor=tk.W)

        # ToM Results
        tom_frame = ttk.LabelFrame(self.results_frame, text="Theory of Mind Tests", padding="15")
        tom_frame.pack(fill=tk.X, pady=10)

        progression = data['tom_results'].get('sally_anne_progression', [])
        tom_text = "Sally-Anne Test Results:\n"
        for i, score in enumerate(progression):
            status = "PASS" if score > 0.5 else "FAIL"
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            tom_text += f"  Order {i}: [{bar}] {score:.2f} {status}\n"

        ttk.Label(tom_frame, text=tom_text, font=('Consolas', 10)).pack(anchor=tk.W)

        # Plain English explanation
        explain_frame = ttk.LabelFrame(self.results_frame, text="What This Means (Plain English)", padding="15")
        explain_frame.pack(fill=tk.X, pady=10)

        max_order = data['tom_results'].get('max_tom_order', -1)
        if max_order >= 2:
            explanation = f"""
Great results! The AI can understand nested beliefs up to order {max_order}.

This means it can reason about thoughts like:
"I know that YOU know that I think..."

This is similar to how humans naturally think about social situations.
The AI shows genuine Theory of Mind capabilities, not just pattern matching.
            """
        elif max_order >= 0:
            explanation = """
The AI has basic Theory of Mind - it can track what others believe.

However, it struggles with deeper nested reasoning (thinking about
what others think about what you think...).

More evolution may help, or try different architecture settings.
            """
        else:
            explanation = """
The AI is still learning basic Theory of Mind.

Try running more generations or increasing population size.
            """

        ttk.Label(explain_frame, text=explanation.strip(),
                  font=('Arial', 11), wraplength=600).pack(anchor=tk.W)

    def inspect_agent(self):
        """Inspect selected agent."""
        self.inspector_text.delete(1.0, tk.END)
        self.inspector_text.insert(tk.END, "Agent inspection not yet implemented.\n")
        self.inspector_text.insert(tk.END, "Run an evolution first, then inspect the best agent here.\n")

    def new_experiment(self):
        """Start a new experiment."""
        if self.evolution_running:
            messagebox.showwarning("Warning", "Stop current evolution first")
            return
        self.history = {'fitness': [], 'diversity': [], 'tom': [], 'zombie': []}
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "Ready for new experiment...\n")

    def load_results(self):
        """Load saved results."""
        messagebox.showinfo("Info", "Load functionality coming soon")

    def save_results(self):
        """Save current results."""
        messagebox.showinfo("Info", "Save functionality coming soon")

    def show_about(self):
        """Show about dialog."""
        about_text = """
ToM-NAS: Theory of Mind Neural Architecture Search

This program evolves AI systems that can understand
what others think and believe.

Key Features:
- Evolves 3 different neural architectures
- Tests Theory of Mind at 6 levels of complexity
- Detects "philosophical zombies" (fake understanding)
- Produces scientifically valid, reproducible results

Created for research into machine consciousness
and social AI understanding.
        """
        messagebox.showinfo("About ToM-NAS", about_text.strip())

    def show_help_results(self):
        """Show help for understanding results."""
        help_text = ExplainerText.fitness() + "\n\n" + ExplainerText.tom_order()

        help_window = tk.Toplevel(self.root)
        help_window.title("Understanding Results")
        help_window.geometry("600x500")

        text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Arial', 11))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, help_text)
        text.configure(state=tk.DISABLED)

    def show_glossary(self):
        """Show glossary of terms."""
        glossary = """
GLOSSARY OF TERMS

Theory of Mind (ToM): The ability to understand that others have their own thoughts and beliefs.

False Belief: When someone believes something that isn't actually true.

Sally-Anne Test: A classic test where you predict where Sally will look for an object she didn't see move.

Fitness: A score (0-1) measuring how well an AI performs on all tests.

Generation: One cycle of evolution where AIs compete and reproduce.

Population: The group of different AI "brains" being tested.

Species: In our system, different architecture types (TRN, RSAN, Transformer).

Mutation: Random changes to AI design that create variation.

Zombie (philosophical): Something that acts human but lacks genuine understanding.

Metacognition: Thinking about thinking - knowing what you know.
        """

        help_window = tk.Toplevel(self.root)
        help_window.title("Glossary")
        help_window.geometry("500x400")

        text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Arial', 11))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, glossary.strip())
        text.configure(state=tk.DISABLED)

    def show_quick_help(self):
        """Show quick help popup."""
        help_text = """
QUICK START GUIDE

1. Set your parameters (or use defaults)
2. Click START to begin evolution
3. Watch the live progress
4. View detailed results when complete

TIPS:
- More generations = better results (but slower)
- Watch for species count = 3 (healthy diversity)
- ToM scores should decrease by order (0 > 1 > 2...)
- Check the "Learn" tab for explanations!
        """
        messagebox.showinfo("Quick Help", help_text.strip())


def main():
    """Launch the ToM-NAS GUI application."""
    root = tk.Tk()
    app = ToMNASApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
