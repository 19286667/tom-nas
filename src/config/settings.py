"""
ToM-NAS Settings - Environment-aware Configuration

Runtime settings that can be configured via environment variables.
Supports local development, testing, and Google Cloud deployment.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from functools import lru_cache

from .constants import (
    DEFAULT_HIDDEN_DIM := 128,
    SOUL_MAP_DIMS,
    INPUT_DIMS,
    OUTPUT_DIMS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DEVICE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_GODOT_HOST,
    DEFAULT_GODOT_PORT,
)


@dataclass
class Settings:
    """
    Application settings loaded from environment variables.

    Usage:
        settings = get_settings()
        print(settings.device)
    """

    # Environment
    environment: str = field(default_factory=lambda: os.getenv('TOM_NAS_ENV', 'development'))
    debug: bool = field(default_factory=lambda: os.getenv('TOM_NAS_DEBUG', 'false').lower() == 'true')

    # Google Cloud
    gcp_project_id: Optional[str] = field(default_factory=lambda: os.getenv('GCP_PROJECT_ID'))
    gcp_region: str = field(default_factory=lambda: os.getenv('GCP_REGION', 'us-central1'))
    gcp_credentials_path: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

    # Device
    device: str = field(default_factory=lambda: os.getenv('TOM_NAS_DEVICE', DEFAULT_DEVICE))

    # Dimensions (can override constants for experimentation)
    input_dims: int = field(default_factory=lambda: int(os.getenv('TOM_NAS_INPUT_DIMS', INPUT_DIMS)))
    output_dims: int = field(default_factory=lambda: int(os.getenv('TOM_NAS_OUTPUT_DIMS', OUTPUT_DIMS)))
    hidden_dims: int = field(default_factory=lambda: int(os.getenv('TOM_NAS_HIDDEN_DIMS', 128)))

    # Evolution
    population_size: int = field(default_factory=lambda: int(os.getenv('TOM_NAS_POPULATION_SIZE', DEFAULT_POPULATION_SIZE)))
    generations: int = field(default_factory=lambda: int(os.getenv('TOM_NAS_GENERATIONS', DEFAULT_GENERATIONS)))

    # Training
    batch_size: int = field(default_factory=lambda: int(os.getenv('TOM_NAS_BATCH_SIZE', DEFAULT_BATCH_SIZE)))
    learning_rate: float = field(default_factory=lambda: float(os.getenv('TOM_NAS_LEARNING_RATE', DEFAULT_LEARNING_RATE)))

    # Paths
    checkpoint_dir: str = field(default_factory=lambda: os.getenv('TOM_NAS_CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR))
    results_dir: str = field(default_factory=lambda: os.getenv('TOM_NAS_RESULTS_DIR', DEFAULT_RESULTS_DIR))
    logs_dir: str = field(default_factory=lambda: os.getenv('TOM_NAS_LOGS_DIR', DEFAULT_LOGS_DIR))

    # Godot Bridge
    godot_host: str = field(default_factory=lambda: os.getenv('GODOT_HOST', DEFAULT_GODOT_HOST))
    godot_port: int = field(default_factory=lambda: int(os.getenv('GODOT_PORT', DEFAULT_GODOT_PORT)))

    # Streamlit / Web UI
    streamlit_port: int = field(default_factory=lambda: int(os.getenv('STREAMLIT_PORT', 8501)))
    api_port: int = field(default_factory=lambda: int(os.getenv('API_PORT', 8080)))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv('TOM_NAS_LOG_LEVEL', 'INFO'))
    enable_cloud_logging: bool = field(default_factory=lambda: os.getenv('ENABLE_CLOUD_LOGGING', 'false').lower() == 'true')

    # Monitoring
    enable_metrics: bool = field(default_factory=lambda: os.getenv('ENABLE_METRICS', 'true').lower() == 'true')
    enable_tracing: bool = field(default_factory=lambda: os.getenv('ENABLE_TRACING', 'false').lower() == 'true')

    def __post_init__(self):
        """Validate and adjust settings after initialization."""
        import torch

        # Auto-detect device if CUDA requested but unavailable
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

        # Ensure directories exist
        for dir_path in [self.checkpoint_dir, self.results_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'

    @property
    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self.environment == 'testing'

    @property
    def godot_websocket_url(self) -> str:
        """Get the Godot WebSocket URL."""
        return f'ws://{self.godot_host}:{self.godot_port}'

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'device': self.device,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'hidden_dims': self.hidden_dims,
            'population_size': self.population_size,
            'generations': self.generations,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'log_level': self.log_level,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the application settings singleton.

    Settings are loaded once and cached for performance.

    Returns:
        Settings: The application settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.device)
        'cuda'
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Force reload settings (clears cache).

    Useful for testing or when environment variables change.

    Returns:
        Settings: Fresh settings instance
    """
    get_settings.cache_clear()
    return get_settings()
