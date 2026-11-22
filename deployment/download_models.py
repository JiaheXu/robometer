import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path('/home/ubuntu/robometer')
MODELS = (
    ('robometer/Robometer-4B', 'beef63bc914c5c189329d49c6d712d96d632aa34'),
    ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17'),
)
manifest = {}
for repo, revision in MODELS:
    print(f'Downloading {repo}@{revision}', flush=True)
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=['*.safetensors', '*.json', '*.yaml', '*.txt', '*.model', '*.jinja'],
        max_workers=4,
    )
    manifest[repo] = {'revision': revision, 'snapshot': path}
    (ROOT / 'deployment' / 'models.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Cached {repo}: {path}', flush=True)

print('All model files downloaded.', flush=True)
