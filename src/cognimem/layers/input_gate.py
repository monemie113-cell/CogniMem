import numpy as np
from typing import Any, Dict
from ..models.small_gate import SmallGateModel
from ..utils.logger import get_logger

logger = get_logger(__name__)

class InputGate:
    def __init__(self, config: Dict):
        self.config = config
        # 默认阈值从 0.5 调整为 0.3
        self.threshold = config.get('score_threshold', 0.3)
        self.model = SmallGateModel(config)
        self.stats = {'total': 0, 'passed': 0, 'blocked': 0}

    def process(self, inputs: Any, context: Dict = None):
        self.stats['total'] += 1
        scores = self.model.compute_scores(inputs, context)
        overall = np.mean(list(scores.values()))
        if overall >= self.threshold:
            self.stats['passed'] += 1
            return inputs, {'passed': True, 'scores': scores}
        else:
            self.stats['blocked'] += 1
            return None, {'passed': False, 'scores': scores}

    def get_stats(self):
        return self.stats