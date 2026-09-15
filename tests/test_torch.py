import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import torch
import numpy as np
from cognimem.layers.hybrid_encoder_torch import HybridEncoderTorch

config = {'dim': 384, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encoder = HybridEncoderTorch(config)
seq = np.random.randn(100, 384).astype(np.float32)
out, meta = encoder.process(seq)
print(out.shape, meta)