
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RecursiveAttention(nn.Module):
    """Self-attention that can recurse"""
    def __init__(self, d_model, n_heads=4, max_recursion=5):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.recurse_gate = nn.Linear(d_model, 1)
        self.max_recursion = max_recursion
        
    def forward(self, x, depth=0):
        attn_out, _ = self.attention(x, x, x)
        
        if depth < self.max_recursion:
            should_recurse = torch.sigmoid(self.recurse_gate(attn_out.mean(dim=1)))
            if should_recurse.mean() > 0.5:
                return self.forward(attn_out, depth + 1)
        
        return attn_out

class RSANAgent(nn.Module):
    """Recursive Self-Attention Network Agent"""
    
    def __init__(self, input_dim=20, d_model=128, n_heads=4, n_layers=3, ontology_dim=55, max_agents=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            RecursiveAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.belief_head = nn.Linear(d_model, ontology_dim)
        self.action_head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.mean(dim=1) if x.dim() == 3 else x
        
        belief = torch.sigmoid(self.belief_head(x))
        action = torch.sigmoid(self.action_head(x)).squeeze(-1)
        
        return {"belief": belief, "action": action}
