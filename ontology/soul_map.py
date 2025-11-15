
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class OntologyLayer(Enum):
    """Enumeration of ontology layers"""
    BIOLOGICAL = 0
    AFFECTIVE = 1
    MOTIVATIONAL = 2
    COGNITIVE = 3
    SELF = 4
    SOCIAL = 5
    VALUES = 6
    CONTEXTUAL = 7
    EXISTENTIAL = 8

@dataclass
class LayerDimension:
    """Single dimension in a layer"""
    name: str
    description: str
    low_pole: str
    high_pole: str
    default_value: float = 0.5

class SoulMapOntology:
    """Complete psychological ontology with 55+ dimensions"""
    
    def __init__(self):
        self.layers = self._build_layers()
        self.total_dim = sum(len(layer) for layer in self.layers.values())
        print(f"  Ontology initialized: {self.total_dim} dimensions")
    
    def _build_layers(self):
        """Build all ontology layers"""
        layers = {}
        
        # Biological layer
        layers["biological"] = [
            LayerDimension("energy", "Physical energy", "exhausted", "energized", 0.7),
            LayerDimension("pain", "Pain level", "comfortable", "suffering", 0.2),
            LayerDimension("arousal", "Arousal", "calm", "activated", 0.5),
            LayerDimension("hunger", "Hunger", "satiated", "starving", 0.3),
            LayerDimension("interoception", "Body awareness", "disconnected", "aware", 0.5)
        ]
        
        # Affective layer
        layers["affective"] = [
            LayerDimension("emotional_stability", "Stability", "volatile", "stable", 0.6),
            LayerDimension("affect_intensity", "Intensity", "numb", "intense", 0.5),
            LayerDimension("positive_affect", "Positive", "depressed", "joyful", 0.5),
            LayerDimension("anxiety", "Anxiety", "calm", "anxious", 0.4),
            LayerDimension("anger", "Anger", "peaceful", "furious", 0.3),
            LayerDimension("shame", "Shame", "shameless", "ashamed", 0.3)
        ]
        
        # Add more layers (simplified for now)
        layers["motivational"] = [LayerDimension(f"mot_{i}", f"Mot {i}", "low", "high") for i in range(7)]
        layers["cognitive"] = [LayerDimension(f"cog_{i}", f"Cog {i}", "low", "high") for i in range(6)]
        layers["self"] = [LayerDimension(f"self_{i}", f"Self {i}", "low", "high") for i in range(5)]
        layers["social"] = [LayerDimension(f"soc_{i}", f"Soc {i}", "low", "high") for i in range(7)]
        layers["values"] = [LayerDimension(f"val_{i}", f"Val {i}", "low", "high") for i in range(9)]
        layers["contextual"] = [LayerDimension(f"ctx_{i}", f"Ctx {i}", "low", "high") for i in range(5)]
        layers["existential"] = [LayerDimension(f"exist_{i}", f"Exist {i}", "low", "high") for i in range(5)]
        
        return layers
    
    def encode_state(self, state_dict: Dict) -> torch.Tensor:
        """Encode state dictionary to vector"""
        vector = []
        for layer_name, dimensions in self.layers.items():
            for dim in dimensions:
                key = f"{layer_name}_{dim.name}"
                vector.append(state_dict.get(key, dim.default_value))
        return torch.tensor(vector, dtype=torch.float32)
    
    def decode_state(self, vector: torch.Tensor) -> Dict:
        """Decode vector to state dictionary"""
        state = {}
        idx = 0
        for layer_name, dimensions in self.layers.items():
            for dim in dimensions:
                key = f"{layer_name}_{dim.name}"
                state[key] = float(vector[idx])
                idx += 1
        return state
