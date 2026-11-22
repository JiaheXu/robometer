# Robometer deployment

Deployed and verified on November 18, 2026 (Asia/Shanghai).

## Access

- SSH: `ssh ubuntu@192.168.200.102`
- Repository on Ubuntu: `/home/ubuntu/robometer`
- Inference API: `http://192.168.200.102:8000`
- Interactive API documentation: `http://192.168.200.102:8000/docs`
- Health check: `http://192.168.200.102:8000/health`
- Model: Robometer-4B, running on the NVIDIA GeForce RTX 5090.

The service starts automatically at boot through the `ubuntu` user's systemd service. User lingering is enabled, so it does not require an SSH session or desktop login. Models are loaded from local disk, with Hugging Face offline mode enabled. The temporary Mac proxy has been stopped.

## Manage the service

Run these commands on Ubuntu as `ubuntu`:

```bash
systemctl --user status robometer
systemctl --user restart robometer
systemctl --user stop robometer
systemctl --user start robometer
journalctl --user -u robometer -f
```

To disable automatic startup and stop the server:

```bash
systemctl --user disable --now robometer
```

To enable it again:

```bash
systemctl --user enable --now robometer
```

Stopping the service releases its GPU allocation for other work. The service unit is `/home/ubuntu/.config/systemd/user/robometer.service`; the launcher is `/home/ubuntu/robometer/deployment/serve.sh`.

## Run inference

On Ubuntu:

```bash
cd /home/ubuntu/robometer
.venv/bin/python scripts/example_inference.py \
  --eval-server-url http://127.0.0.1:8000 \
  --video scripts/example_videos/soar_put_green_stick_in_brown_bowl.mp4 \
  --task "Put green stick in brown bowl" \
  --fps 3 \
  --out deployment/smoke/soar_rewards.npy
```

For your own video, replace `--video`, `--task`, and `--out`. The client writes a progress `.npy`, a success-probability `.npy`, and a plot. The same client can run on another computer with NumPy, requests, matplotlib, and Decord installed; use `--eval-server-url http://192.168.200.102:8000`.

The API supports `/evaluate_batch`, `/evaluate_batch_npy`, `/health`, `/gpu_status`, and `/model_info`. Sampling more frames increases inference memory use. The deployment runs one GPU worker.

## Verified results

- PyTorch detected the RTX 5090, compute capability 12.0, and CUDA 12.8.
- A BF16 matrix multiplication ran successfully on the GPU.
- The server and all required inference modules imported successfully.
- The model loaded with the Mac proxy stopped and Hugging Face offline mode enabled.
- Health, GPU status, OpenAPI schema, and API documentation were reachable from the Mac over the LAN.
- The bundled SOAR clip was sampled at 3 fps into 10 frames. Inference returned 10 finite progress scores and 10 finite success probabilities, all in `[0, 1]`.
- The example client completed in approximately 0.81 seconds in this smoke test, including frame extraction, the HTTP request, and output generation. This is a single test, not a general performance benchmark.
- Progress ranged from 0.0652 to 0.9522 for this clip.
- Startup is enabled and `Linger=yes` is configured. The machine was not rebooted during validation.

Test files on Ubuntu are under `/home/ubuntu/robometer/deployment/smoke/`. GPU and package checks are recorded in `deployment/preflight.json`.

## Versions and local compatibility change

Source: `https://github.com/JiaheXu/robometer`, commit `d6e78a755ef202ba31cef5fc005df61506bfc2bd`.

| Component | Installed version |
| --- | --- |
| Ubuntu | 22.04.5 LTS |
| Python | 3.10.12 |
| NVIDIA driver | 595.80 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| Transformers | 4.57.6 |
| TorchAO | 0.13.0 |
| PEFT | 0.21.3 |
| Hugging Face Hub | 0.36.2 |
| FastAPI | 0.162.5 |

The original unbounded TorchAO dependency resolved to 0.18.0, which failed to import with PyTorch 2.8 (`ScalingType` was unavailable). A local `constraint-dependencies = ["torchao==0.13.0"]` entry was added to `pyproject.toml`, and a `uv.lock` was generated. Full installed versions are saved in `deployment/requirements-frozen.txt`. The compatibility diff is saved in `deployment/compatibility.patch`. No changes were pushed to GitHub.

Reinstall the locked environment from the repository with:

```bash
~/.local/bin/uv sync --frozen --extra robometer
```

`uv pip check` reports one upstream Decord 0.6.0 wheel-tag mismatch: the package's internal metadata says CPython 3.6 although its distributed wheel is named `py3-none`. Actual MP4 decoding on Python 3.10 and end-to-end inference both passed. This warning is recorded in `deployment/dependency-check.txt`; no other dependency incompatibilities were reported.

## Model files and runtime configuration

| Model | Pinned revision |
| --- | --- |
| `robometer/Robometer-4B` | `beef63bc914c5c189329d49c6d712d96d632aa34` |
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

Original snapshots are in `/home/ubuntu/.cache/huggingface/hub/`. The `models/` directory links to these files, avoiding a second copy of the weights. The manifest is `deployment/models.json`.

The deployment copy of the checkpoint's `config.yaml` changes only the base-model path to a local directory and `use_unsloth` to `false`. It uses the repository's standard Transformers inference path with BF16 and PyTorch SDPA. The original downloaded configuration is preserved. The service sets offline mode and uses the virtual environment directly, avoiding dependency updates at startup.

Training datasets were not downloaded, and training/fine-tuning was not tested. The installed environment includes the repository's main dependencies and `robometer` extra.

## Network setup used during installation

GitHub, PyPI, and the model CDN were reachable directly from Ubuntu. Hugging Face metadata needed the existing proxy script on the Mac. A temporary provider used `192.168.200.30:17892`, restricted to `192.168.200.102/32`. Model downloading used that proxy with `NO_PROXY=localhost,127.0.0.1,.hf.co`, allowing the large CDN transfers to connect directly. The original files under `~/Desktop/internet-sharing` were not modified.

The proxy is not required for the deployed service. To download additional models later, start a provider using the Mac's current LAN address, and apply proxy variables only to the download command.
