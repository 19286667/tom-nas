
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class TransparentGate(nn.Module):
    """Transparent gating mechanism"""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        gate = torch.sigmoid((x * self.w).sum(dim=-1, keepdim=True) + self.b)
        return x * gate, gate

class TRNCell(nn.Module):
    """Transparent recurrent cell"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = TransparentGate(hidden_dim)
        
    def forward(self, x, h):
        combined = torch.tanh(self.W_in(x) + self.W_h(h))
        output, gate_value = self.gate(combined)
        return output, {"gate": gate_value.detach()}

class TRNAgent(nn.Module):
    """Transparent Recurrent Network Agent"""
    
    def __init__(self, input_dim=20, hidden_dim=64, ontology_dim=55, max_agents=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ontology_dim = ontology_dim
        
        # Cells for each layer
        self.bio_cell = TRNCell(input_dim, hidden_dim)
        self.aff_cell = TRNCell(hidden_dim, hidden_dim)
        self.cog_cell = TRNCell(hidden_dim, hidden_dim)
        self.soc_cell = TRNCell(hidden_dim, hidden_dim)
        
        # Output heads
        self.belief_head = nn.Linear(hidden_dim * 4, ontology_dim)
        self.action_head = nn.Linear(hidden_dim * 4, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1) if x.dim() > 2 else 1
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Initialize hidden states
        h_bio = torch.zeros(batch_size, self.hidden_dim)
        h_aff = torch.zeros(batch_size, self.hidden_dim)
        h_cog = torch.zeros(batch_size, self.hidden_dim)
        h_soc = torch.zeros(batch_size, self.hidden_dim)
        
        # Process through time
        for t in range(seq_len):
            h_bio, _ = self.bio_cell(x[:, t], h_bio)
            h_aff, _ = self.aff_cell(h_bio, h_aff)
            h_cog, _ = self.cog_cell(h_aff, h_cog)
            h_soc, _ = self.soc_cell(h_cog, h_soc)
        
        # Combine states
        h_all = torch.cat([h_bio, h_aff, h_cog, h_soc], dim=-1)
        
        belief = torch.sigmoid(self.belief_head(h_all))
        action = torch.sigmoid(self.action_head(h_all)).squeeze(-1)
        
        return {"belief": belief, "action": action}
