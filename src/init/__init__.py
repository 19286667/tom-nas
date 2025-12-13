"""
Neural Network Initialization

Provides sensible initializations for neural components so
the system produces meaningful outputs before any training.

Key principle: random weights = random behavior.
We need structured initialization that encodes inductive biases.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def init_xavier_uniform(module: nn.Module):
    """Xavier uniform initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_orthogonal(module: nn.Module):
    """Orthogonal initialization (good for RNNs)."""
    if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def init_embedding_uniform(module: nn.Module, scale: float = 0.1):
    """Small uniform initialization for embeddings."""
    if isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -scale, scale)


class SmartInitializer:
    """
    Smart initialization that encodes domain knowledge.

    Instead of random weights, we initialize networks with
    structured patterns that encode useful inductive biases
    for Theory of Mind reasoning.
    """

    @staticmethod
    def init_belief_encoder(model: nn.Module):
        """
        Initialize a belief encoder.

        Belief encoders should:
        - Preserve information (not collapse)
        - Have smooth gradients
        - Start near identity-like behavior
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier for hidden layers
                nn.init.xavier_uniform_(module.weight)

                # Initialize bias to create slight asymmetry
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.01, 0.01)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        logger.debug(f"Initialized belief encoder: {model.__class__.__name__}")

    @staticmethod
    def init_reasoning_rnn(model: nn.Module):
        """
        Initialize a reasoning RNN.

        RNNs for reasoning should:
        - Preserve long-term dependencies (orthogonal recurrence)
        - Have stable gradients
        - Not forget too quickly
        """
        for name, param in model.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # For LSTM, initialize forget gate bias to 1 (remember by default)
                if 'lstm' in name.lower():
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

        logger.debug(f"Initialized reasoning RNN: {model.__class__.__name__}")

    @staticmethod
    def init_code_generator(model: nn.Module):
        """
        Initialize a code generation model.

        Code generators should:
        - Produce valid syntax early
        - Have smooth token distributions
        - Not collapse to single tokens
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.TransformerDecoderLayer):
                # Initialize attention to be roughly uniform initially
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        if submodule.bias is not None:
                            nn.init.zeros_(submodule.bias)

        logger.debug(f"Initialized code generator: {model.__class__.__name__}")

    @staticmethod
    def init_tom_predictor(model: nn.Module, max_order: int = 5):
        """
        Initialize Theory of Mind prediction heads.

        ToM predictors should:
        - Handle recursive structure (order-0 to order-N beliefs)
        - Not collapse different orders to same output
        - Start with sensible priors (uncertainty)
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Scale initialization by depth
                # Higher-order beliefs should have more uncertainty
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                std = math.sqrt(2.0 / (fan_in + fan_out))

                # Extract order from name if present
                order = 1
                for i in range(max_order, 0, -1):
                    if f'order_{i}' in name or f'order{i}' in name:
                        order = i
                        break

                # Higher orders get smaller initial weights (more uncertainty)
                std = std / (1 + 0.2 * order)
                nn.init.normal_(module.weight, mean=0, std=std)

                if module.bias is not None:
                    # Start with slight positive bias (mild belief)
                    nn.init.uniform_(module.bias, 0, 0.1)

        logger.debug(f"Initialized ToM predictor: {model.__class__.__name__}")


def initialize_model(model: nn.Module, model_type: str = "generic"):
    """
    Initialize a model based on its type.

    Args:
        model: The PyTorch model to initialize
        model_type: One of 'belief_encoder', 'reasoning_rnn',
                   'code_generator', 'tom_predictor', 'generic'
    """
    initializers = {
        "belief_encoder": SmartInitializer.init_belief_encoder,
        "reasoning_rnn": SmartInitializer.init_reasoning_rnn,
        "code_generator": SmartInitializer.init_code_generator,
        "tom_predictor": SmartInitializer.init_tom_predictor,
    }

    if model_type in initializers:
        initializers[model_type](model)
    else:
        # Generic initialization
        model.apply(init_xavier_uniform)

    return model


class PretrainedWeights:
    """
    Manages pretrained weight loading and saving.

    Since we can't train during deployment, we provide
    pretrained weights that encode useful behaviors.
    """

    WEIGHT_REGISTRY = {
        "belief_encoder_v1": "weights/belief_encoder_v1.pt",
        "reasoning_rnn_v1": "weights/reasoning_rnn_v1.pt",
        "code_generator_v1": "weights/code_generator_v1.pt",
    }

    @classmethod
    def load_if_available(cls, model: nn.Module, weight_name: str) -> bool:
        """
        Load pretrained weights if available.

        Returns True if weights were loaded, False otherwise.
        """
        import os

        if weight_name not in cls.WEIGHT_REGISTRY:
            return False

        weight_path = cls.WEIGHT_REGISTRY[weight_name]

        if not os.path.exists(weight_path):
            logger.warning(f"Pretrained weights not found: {weight_path}")
            return False

        try:
            state_dict = torch.load(weight_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights: {weight_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False

    @classmethod
    def save_weights(cls, model: nn.Module, weight_name: str):
        """Save model weights for future use."""
        import os

        weight_path = cls.WEIGHT_REGISTRY.get(weight_name, f"weights/{weight_name}.pt")
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)

        torch.save(model.state_dict(), weight_path)
        logger.info(f"Saved weights to: {weight_path}")


def create_initialized_model(
    model_class,
    model_type: str,
    *args,
    pretrained: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Create and initialize a model in one call.

    Args:
        model_class: The model class to instantiate
        model_type: Type for initialization strategy
        *args, **kwargs: Arguments for model constructor
        pretrained: Optional pretrained weight name to load

    Returns:
        Initialized model
    """
    model = model_class(*args, **kwargs)

    # Try loading pretrained weights first
    if pretrained and PretrainedWeights.load_if_available(model, pretrained):
        return model

    # Otherwise initialize
    initialize_model(model, model_type)
    return model
