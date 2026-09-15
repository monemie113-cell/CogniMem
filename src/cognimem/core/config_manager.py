import os
import yaml
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = None):
        self.config: Dict[str, Any] = {}
        if config_path and os.path.exists(config_path):
            self.load(config_path)
        else:
            self.config = self._default_config()

    def load(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def _default_config(self) -> Dict:
        return {
            'pipeline': {
                'input_gate': {'enabled': True, 'score_threshold': 0.5},
                'engram': {'enabled': True, 'embed_dim': 768, 'table_size': 10000},
                'hybrid_encoder': {'enabled': True, 'full_attn_layers': 5, 'sparse_ratio': 0.3},
                'liquid_engine': {'enabled': True, 'hidden_dim': 512, 'time_constants': [0.1, 1.0, 10.0]}
            },
            'memory': {
                'episodic_db': 'episodic.db',
                'semantic_db': 'semantic.db',
                'working_memory_capacity': 10,
                'routing': {'cache_ttl': 60, 'bm25_alpha': 0.8}
            }
        }