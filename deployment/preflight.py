import importlib.metadata
import json
from pathlib import Path

import torch

assert torch.cuda.is_available(), 'CUDA is unavailable'
device = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
x = torch.ones((64, 64), dtype=torch.bfloat16, device='cuda')
y = x @ x
torch.cuda.synchronize()
assert torch.all(y == 64).item(), 'CUDA BF16 matrix multiplication failed'

from robometer.evals.eval_server import MultiGPUEvalServer  # noqa: E402,F401

result = {
    'gpu': device,
    'compute_capability': list(capability),
    'torch_cuda': torch.version.cuda,
    'bf16_cuda_matmul': 'passed',
    'server_import': 'passed',
    'packages': {
        name: importlib.metadata.version(name)
        for name in ['torch', 'torchvision', 'transformers', 'peft', 'huggingface-hub', 'fastapi']
    },
}
Path('/home/ubuntu/robometer/deployment/preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
