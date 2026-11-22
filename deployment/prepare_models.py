import json
from pathlib import Path

import yaml

root = Path('/home/ubuntu/robometer')
manifest = json.loads((root / 'deployment/models.json').read_text())
model_root = root / 'models'
model_root.mkdir(exist_ok=True)
qwen_source = Path(manifest['Qwen/Qwen3-VL-4B-Instruct']['snapshot'])
qwen_target = model_root / 'Qwen3-VL-4B-Instruct'
if not qwen_target.exists():
    qwen_target.symlink_to(qwen_source, target_is_directory=True)
rbm_source = Path(manifest['robometer/Robometer-4B']['snapshot'])
rbm_target = model_root / 'Robometer-4B'
rbm_target.mkdir(exist_ok=True)
for source in rbm_source.iterdir():
    target = rbm_target / source.name
    if source.name != 'config.yaml' and not target.exists():
        target.symlink_to(source)
config = yaml.safe_load((rbm_source / 'config.yaml').read_text())
# Use the repository's standard Transformers inference path. Training-only
# Unsloth optimizations are unnecessary for this BF16 / SDPA deployment.
config['model']['use_unsloth'] = False
config['model']['base_model_id'] = str(qwen_target)
(rbm_target / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
print(f'Prepared offline inference checkpoint: {rbm_target}')
print('Original downloaded checkpoint and config remain unchanged in HF cache.')
