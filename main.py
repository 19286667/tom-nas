
"""
ToM-NAS Main Execution Script
==============================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from ontology.soul_map import SoulMapOntology
from agents.belief_structures import RecursiveBeliefTracker
from agents.trn import TRNAgent
from agents.rsan import RSANAgent

def train_agents(agents, episodes=10):
    """Quick training loop"""
    print("\nTraining agents...")
    optimizers = {
        name: optim.Adam(agent.parameters(), lr=0.001)
        for name, agent in agents.items()
    }
    
    for ep in range(episodes):
        # Generate dummy data
        batch_size = 8
        seq_len = 10
        input_dim = 20
        ontology_dim = 55
        
        x = torch.randn(batch_size, seq_len, input_dim)
        target_belief = torch.rand(batch_size, ontology_dim)
        target_action = torch.rand(batch_size) > 0.5
        
        total_loss = 0
        for name, agent in agents.items():
            output = agent(x)
            
            loss = nn.MSELoss()(output["belief"], target_belief)
            loss += nn.BCELoss()(output["action"], target_action.float())
            
            optimizers[name].zero_grad()
            loss.backward()
            optimizers[name].step()
            
            total_loss += loss.item()
        
        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}: Loss = {total_loss/len(agents):.4f}")
    
    print("✓ Training complete")

def evaluate_agents(agents):
    """Basic evaluation"""
    print("\nEvaluating agents...")
    
    # Sally-Anne test (simplified)
    x = torch.zeros(1, 5, 20)
    x[0, 0, 0] = 1  # Sally present
    x[0, 0, 1] = 1  # Ball in basket
    x[0, 2, 2] = 1  # Anne moves ball
    x[0, 4, 0] = 1  # Sally returns
    
    for name, agent in agents.items():
        with torch.no_grad():
            output = agent(x)
            belief = output["belief"][0].numpy()
            print(f"  {name}: Belief strength = {belief[1]:.3f}")
    
    print("✓ Evaluation complete")

def main():
    parser = argparse.ArgumentParser(description="ToM-NAS System")
    parser.add_argument("--mode", default="test", choices=["test", "train", "evaluate", "full"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ToM-NAS: Theory of Mind Neural Architecture Search")
    print("="*80)
    
    # Initialize components
    print("\nInitializing components...")
    ontology = SoulMapOntology()
    belief_tracker = RecursiveBeliefTracker()
    print(f"✓ Belief tracker: {belief_tracker.max_order}th order")
    
    # Create agents
    agents = {
        "TRN": TRNAgent(),
        "RSAN": RSANAgent()
    }
    print(f"✓ Agents created: {list(agents.keys())}")
    
    # Run based on mode
    if args.mode in ["train", "full"]:
        train_agents(agents, episodes=args.epochs)
    
    if args.mode in ["evaluate", "full", "test"]:
        evaluate_agents(agents)
    
    print("\n✅ Complete!")
    print("\nThis system demonstrates:")
    print("  • Novel RSAN architecture (recursive self-attention)")
    print("  • Transparent TRN (interpretable reasoning)")
    print("  • 55-dimensional psychological ontology")
    print("  • 5th-order nested beliefs")
    
    return 0

if __name__ == "__main__":
    exit(main())
