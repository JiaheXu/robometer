# Independent GPT vs RoboMeter comparison

Run from `/home/jiahe/robometer` using `.venv/bin/python scripts/compare_sub2api.py`.
The default model is `gpt-6-astra`, served at `https://sub2api.cocomatrix.cn/v1/responses`.
Authentication uses the existing `SUB2API_KEY` environment variable. Never put the key in code or command-line arguments. `SUB2API_BASE_URL` can override the base URL.

The script independently scores **both** head-only and three-camera inputs. GPT receives the exact same selected time-step IDs, instruction, and up-to-eight-step causal histories used in the saved RoboMeter experiments. It receives no RoboMeter scores or source review labels. For head input, the RGB images are resized to 427 × 240; three-camera input reuses the lossless 1281 × 240 tiles in head / left-wrist / right-wrist order. These are sizes before each model's own image processor. GPT receives PNG data URLs with `detail: high`, labeled timestamps, and `reasoning.effort: medium`.

Both RoboMeter result directories are read-only inputs:

- `outputs/system012_head_20261114`
- `outputs/system012_three_camera_20261114`

New results go to `outputs/system012_gpt6_astra`.

## Commands

Validate all image hashes and histories without API calls or output writes:

```bash
.venv/bin/python scripts/compare_sub2api.py --dry-run --limit 0
```

Run a bounded live check: the first selected time step, once per camera setup (two requests):

```bash
.venv/bin/python scripts/compare_sub2api.py --limit 1
```

Score all 457 time steps in both modes (914 total requests, less any already saved):

```bash
.venv/bin/python scripts/compare_sub2api.py --limit 0 --workers 8
```

This sends episode images to Sub2API and consumes API quota. Calls run sequentially by default; `--workers 8` enables eight concurrent requests (maximum 16). The per-request timeout is 180 seconds; override with `--timeout 300` if needed. Failed requests are recorded while other judgments continue; there are no automatic retries. Reports refresh after each episode. A failed or interrupted request might have been billed even if no result was saved.

Resume by rerunning the same command. Completed judgments are flushed to disk immediately and skipped on subsequent runs. Changing the model, reasoning effort, prompt, baseline files, or endpoint requires a new `--output` directory. Do not run two writers against the same output directory simultaneously.

Regenerate reports from saved judgments without API calls:

```bash
.venv/bin/python scripts/compare_sub2api.py --report-only
```

## Outputs and interpretation

- `index.html` and per-episode reports: RoboMeter-style subtask panels: progress/success overlays directly above a three-camera image grid; A/B/C markers match the image columns. Each episode also includes final subtask scores, GPT visual evidence, and source review notes. Only evaluated time steps are plotted; coverage explicitly distinguishes a partial run from all 914 judgments.
- `results.jsonl`: one record per camera/time-step judgment, including scores, status, evidence, image hashes, request hash, response ID, returned model, latency, and provider token usage when available. No API keys or image data URLs are saved.
- `summary.json`: per-camera score differences and success-threshold agreement with RoboMeter, excluding null GPT scores.
- `config.json`: exact prompt, model settings, and baseline file hashes for resume checks.
- `last_error.json`: error class and HTTP status after a failed call; provider response bodies are omitted to avoid logging echoed credentials or images. It records the last failure and may remain after a successful resume.

GPT returns `progress`, `success_probability`, `status`, and `evidence` for the last time step. Missing visual evidence is represented by null scores; unknown success has status `uncertain`. The success threshold is strictly greater than 0.5, matching the existing experiments. The endpoint receives `store: false`; this parameter is not a guarantee about the gateway's retention policy.

**This measures agreement between models, not accuracy.** GPT is an additional evaluator, not human ground truth; its estimated probabilities are not calibrated. Three-camera tiling also changes image area/layout. Selecting a better camera setup still requires trusted success labels.

Run local checks:

```bash
MPLCONFIGDIR=/tmp/robometer-mpl .venv/bin/python scripts/test_compare_sub2api.py
```
