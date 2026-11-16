import hashlib
import html
import json
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

SOURCE = Path('/home/jiahe/system012/output/important_frames_all_20261114_complete')
OUT = Path(__file__).resolve().parent
CHECKPOINT = '/home/jiahe/robometer/outputs/models/Robometer-4B'


def prepare():
    annotations = json.loads((SOURCE / 'codex_subtask_instructions.json').read_text())
    episodes = {}
    for frame in annotations['frames']:
        episodes.setdefault(frame['episode'], []).append(frame)
    jobs = []
    for episode, frames in episodes.items():
        frames.sort(key=lambda f: f['source_timestamp_ns'])
        events = json.loads((SOURCE / episode / 'events.json').read_text())
        raw_path = Path(events['episode']) / 'observation.images.cam_head.parquet'
        raw = pq.read_table(raw_path, columns=['source_timestamp_ns', 'observation.images.cam_head'])
        history = []
        instruction = None
        segment = -1
        for frame in frames:
            index = frame['head_frame_index']
            assert raw['source_timestamp_ns'][index].as_py() == frame['source_timestamp_ns']
            image_path = SOURCE / frame['images']['cam_head']
            expected = frame['head_image_sha256']
            assert hashlib.sha256(image_path.read_bytes()).hexdigest() == expected
            assert hashlib.sha256(raw['observation.images.cam_head'][index].as_py()).hexdigest() == expected
            with Image.open(image_path) as image:
                image.verify()
            if frame['next_subtask_instruction'] != instruction:
                instruction = frame['next_subtask_instruction']
                history = []
                segment += 1
            history.append(frame)
            indices = np.rint(np.linspace(0, len(history) - 1, min(8, len(history)))).astype(int)
            context = [history[i] for i in indices]
            assert context[-1]['event_id'] == frame['event_id']
            assert all(f['source_timestamp_ns'] <= frame['source_timestamp_ns'] for f in context)
            jobs.append({**frame, 'segment': segment, 'raw_head_parquet': str(raw_path),
                         'context_event_ids': [f['event_id'] for f in context],
                         'context_head_frame_indices': [f['head_frame_index'] for f in context],
                         'context_images': [f['images']['cam_head'] for f in context]})
        del raw
    manifest = {'source': str(SOURCE), 'checkpoint': CHECKPOINT, 'camera': 'cam_head',
                'annotation_provenance': annotations['provenance'],
                'method': 'For each selected frame, sample up to 8 selected frames uniformly from the start of its contiguous instruction segment through the current frame. Return the current frame score.',
                'resize': 'RGB, shortest side 240 pixels, aspect ratio preserved, then Qwen processor resizing',
                'success_threshold': 0.5, 'frames': len(jobs), 'episodes': len(episodes),
                'raw_timestamp_and_jpeg_sha256_verified': True, 'jobs': jobs}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def infer(manifest):
    result_path = OUT / 'results.jsonl'
    results = []
    if result_path.exists():
        results = [json.loads(line) for line in result_path.read_text().splitlines()]
    for saved, job in zip(results, manifest['jobs']):
        assert (saved['episode'], saved['event_id'], saved['next_subtask_instruction']) == (job['episode'], job['event_id'], job['next_subtask_instruction'])
    assert len(results) <= len(manifest['jobs'])
    if len(results) == len(manifest['jobs']):
        return results
    assert torch.cuda.is_available(), 'GPU required for this run'
    torch.set_num_threads(4)
    cfg, tokenizer, processor, model = load_model_from_hf(CHECKPOINT, torch.device('cuda'))
    collator = setup_batch_collator(processor, tokenizer, cfg, is_eval=True)
    cache = {}
    with result_path.open('a') as stream:
        for job in manifest['jobs'][len(results):]:
            frames = []
            for relative in job['context_images']:
                if relative not in cache:
                    with Image.open(SOURCE / relative) as image:
                        image = image.convert('RGB')
                        scale = 240 / min(image.size)
                        image = image.resize(tuple(round(n * scale) for n in image.size), Image.Resampling.BICUBIC)
                        cache[relative] = np.asarray(image)
                frames.append(cache[relative])
            frames = np.stack(frames)
            trajectory = Trajectory(frames=frames, frames_shape=frames.shape,
                                    task=job['next_subtask_instruction'], id=job['event_id'],
                                    metadata={'subsequence_length': len(frames)}, video_embeddings=None)
            inputs = collator([ProgressSample(trajectory=trajectory, sample_type='progress')])['progress_inputs']
            inputs = {key: value.to('cuda') if hasattr(value, 'to') else value for key, value in inputs.items()}
            with torch.inference_mode():
                prediction = compute_batch_outputs(model, tokenizer, inputs, sample_type='progress',
                                                   is_discrete_mode=True, num_bins=cfg.model.progress_discrete_bins)
            progress = np.asarray(prediction['progress_pred'][0], dtype=float)
            success = np.asarray(prediction['outputs_success']['success_probs'][0], dtype=float)
            assert progress.shape == success.shape == (len(frames),)
            assert np.isfinite(progress).all() and np.isfinite(success).all()
            assert ((progress >= 0) & (progress <= 1)).all()
            assert ((success >= 0) & (success <= 1)).all()
            row = {**job, 'progress': float(progress[-1]), 'success_probability': float(success[-1]),
                   'success_at_0_5': bool(success[-1] > 0.5), 'input_frames_shape': list(frames.shape),
                   'context_progress': progress.tolist(), 'context_success_probability': success.tolist()}
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            results.append(row)
            print(f"RESULT {len(results)}/{len(manifest['jobs'])} {job['episode']} {job['event_id']} progress={progress[-1]:.3f} success={success[-1]:.3f}", flush=True)
    return results


def report(manifest, results):
    document = ['<!doctype html><html><meta charset="utf-8"><title>Robometer head-camera results</title>',
                '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left}img{max-width:100%}summary{cursor:pointer;padding:12px;background:#eee}.flag{color:#a44}.frames{display:flex;flex-wrap:wrap;gap:14px}.frames figure{margin:0;width:240px}.frames img{width:240px;border-radius:6px}.frames figcaption{font-size:13px;line-height:1.5}.subtask{border-top:1px solid #ddd;padding:14px 0}.subtask h3{font-size:16px;margin:0 0 12px}</style>',
                '<h1>Robometer head-camera results</h1>',
                '<p>Each selected frame is scored against its supplied next-subtask instruction using up to eight selected frames from the current instruction segment. Images are resized to a 240-pixel short side. A new instruction resets the context.</p>',
                '<p>Instructions are draft visual annotations. November 11 recordings have annotated destination conflicts. Scores are model predictions. No human success labels were supplied for accuracy measurement.</p>']
    report_header = document[:]
    summaries = []
    for episode in dict.fromkeys(r['episode'] for r in results):
        rows = [r for r in results if r['episode'] == episode]
        section_start = len(document)
        segments = list(dict.fromkeys(r['segment'] for r in rows))
        thumbnail_rows = (len(segments) + 3) // 4
        fig = plt.figure(figsize=(14, 7 + 2.5 * thumbnail_rows))
        grid = fig.add_gridspec(2 + thumbnail_rows, 4, height_ratios=[2, 2] + [1.5] * thumbnail_rows)
        axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :])]
        storyboard = ['<h2>Subtask images and scores</h2><p>Each panel shows one instruction segment. Lettered markers on the plot match the selected head-camera images below it; the lines include every scored frame in that segment.</p>']
        for position, segment in enumerate(segments):
            group = [r for r in rows if r['segment'] == segment]
            label = f"{segment + 1}. {group[0]['target_object']}"
            axes[0].plot([r['seconds'] for r in group], [r['progress'] for r in group], '.-', label=label)
            axes[1].plot([r['seconds'] for r in group], [r['success_probability'] for r in group], '.-')
            summaries.append({'episode': episode, 'segment': segment, 'instruction': group[0]['next_subtask_instruction'],
                              'frames': len(group), 'final_progress': group[-1]['progress'],
                              'final_success_probability': group[-1]['success_probability'],
                              'max_success_probability': max(r['success_probability'] for r in group),
                              'review_flagged_frames': sum(r['review_required'] for r in group)})
            storyboard.append(f'<section class="subtask"><h3>{segment + 1}. {html.escape(group[0]["next_subtask_instruction"])}</h3>')
            selected_indices = sorted({0, len(group) // 2, len(group) - 1})
            panel = plt.figure(figsize=(10, 5.6), layout='constrained')
            panel_grid = panel.add_gridspec(2, 3, height_ratios=[2, 1.25])
            score_ax = panel.add_subplot(panel_grid[0, :])
            times = [r['seconds'] for r in group]
            score_ax.plot(times, [r['progress'] for r in group], '.-', label='Progress', color='#2563eb')
            score_ax.plot(times, [r['success_probability'] for r in group], '.-', label='Success probability', color='#d97706')
            score_ax.axhline(0.5, color='#9ca3af', linestyle='--', linewidth=1, label='Success threshold')
            score_ax.set(xlabel='Episode time (seconds)', ylabel='Model score', ylim=(-0.03, 1.15))
            score_ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            score_ax.grid(alpha=0.15)
            score_ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=9)
            image_descriptions = []
            for image_index, selected in enumerate(selected_indices):
                row = group[selected]
                marker = chr(65 + image_index)
                score_ax.axvline(row['seconds'], color='#64748b', linestyle=':', linewidth=1)
                score_ax.text(row['seconds'], 1.07, marker, ha='center', va='center', fontsize=10,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='#e2e8f0', edgecolor='none'))
                image_ax = panel.add_subplot(panel_grid[1, image_index])
                with Image.open(SOURCE / row['images']['cam_head']) as source_image:
                    thumbnail = source_image.convert('RGB')
                    thumbnail.thumbnail((320, 180))
                    image_ax.imshow(thumbnail)
                image_ax.axis('off')
                stage = 'Start' if selected == 0 else ('End' if selected == len(group) - 1 else 'Middle')
                caption = f'{marker} · {stage} · {row["seconds"]:.2f}s · Frame {row["head_frame_index"]}'
                image_ax.set_title(caption, fontsize=9, loc='left')
                scores = f'Progress {row["progress"]:.3f} · Success {row["success_probability"]:.3f}'
                image_ax.text(0, -0.06, scores, transform=image_ax.transAxes, fontsize=9, va='top')
                image_descriptions.append(f'{caption}; {scores}')
            panel.suptitle(textwrap.fill(f'{segment + 1}. {group[0]["next_subtask_instruction"]}', 95), fontsize=12, fontweight='bold')
            panel_name = f'{episode}_subtask_{segment + 1:02d}.png'
            panel.savefig(OUT / panel_name, dpi=120)
            plt.close(panel)
            storyboard.append(f'<img width="960" src="{panel_name}" alt="{html.escape(". ".join(image_descriptions))}"></section>')
            last = group[-1]
            thumbnail_ax = fig.add_subplot(grid[2 + position // 4, position % 4])
            with Image.open(SOURCE / last['images']['cam_head']) as source_image:
                thumbnail = source_image.convert('RGB')
                thumbnail.thumbnail((320, 180))
                thumbnail_ax.imshow(thumbnail)
            thumbnail_ax.axis('off')
            caption = textwrap.fill(f'{segment + 1}. {last["next_subtask_instruction"]}', width=43)
            thumbnail_ax.set_title(caption, fontsize=8, color=f'C{position % 10}', loc='left')
            thumbnail_ax.text(0, -0.08, f'{last["seconds"]:.2f}s · Progress {last["progress"]:.2f} · Success {last["success_probability"]:.2f}', transform=thumbnail_ax.transAxes, fontsize=8, va='top')
        axes[0].set_ylabel('Progress')
        axes[1].set_ylabel('Success probability')
        axes[1].set_xlabel('Episode time (seconds)')
        axes[1].axhline(0.5, color='gray', linestyle='--')
        for ax in axes:
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.suptitle(episode)
        fig.tight_layout()
        plot = f'{episode}.png'
        fig.savefig(OUT / plot, dpi=140, bbox_inches='tight')
        plt.close(fig)
        document.extend([f'<h2>{html.escape(episode)}</h2><img src="{plot}">',
                         '<details><summary>Selected head frames and individual scores</summary><table><tr><th>Frame / time</th><th>Head camera</th><th>Instruction and review notes</th><th>Progress / success</th></tr>'])
        for row in rows:
            relative = os.path.relpath(SOURCE / row['images']['cam_head'], OUT)
            notes = html.escape('; '.join(row['review_notes']))
            document.append(f'<tr><td>{row["head_frame_index"]}<br>{row["seconds"]:.2f}s</td><td><img loading="lazy" width="320" src="{html.escape(relative)}"></td><td>{html.escape(row["next_subtask_instruction"])}<p class="flag">{notes}</p></td><td>{row["progress"]:.3f} / {row["success_probability"]:.3f}</td></tr>')
        document.append('</table></details>')
        episode_summaries = [s for s in summaries if s['episode'] == episode]
        flagged = sum(r['review_required'] for r in rows)
        above_threshold = sum(r['success_at_0_5'] for r in rows)
        notes = list(dict.fromkeys(note for row in rows for note in row['review_notes']))
        details = [
            '<p><a href="index.html">All episodes</a></p>',
            f'<h2>{html.escape(episode)}</h2>',
            f'<p>Evaluated {len(rows)} selected head-camera frames across {len(episode_summaries)} instruction segments, spanning {rows[0]["seconds"]:.2f}–{rows[-1]["seconds"]:.2f} seconds of episode time. {flagged} frames carry source annotation review flags. The model assigned success probability above 0.5 to {above_threshold} frames.</p>',
            '<p>The 0.5 threshold is the example script’s decision threshold. Frame counts above this threshold describe model outputs. Ground-truth success rates and accuracy require human labels.</p>',
            '<h2>Subtask results</h2>',
            '<p>Start and end scores refer to the first and last selected frame carrying that instruction. These boundaries come from next-subtask annotations. They may omit the previous action’s final release or verification frame. Each score uses up to eight frames available by that point within its instruction segment.</p>',
            '<div style="overflow-x:auto"><table><tr><th>Instruction</th><th>Time range / frames</th><th>Progress start → end</th><th>Success at end / peak</th><th>Frames above 0.5</th><th>Review flags</th></tr>'
        ]
        for summary in episode_summaries:
            group = [r for r in rows if r['segment'] == summary['segment']]
            count_above = sum(r['success_at_0_5'] for r in group)
            details.append(
                f'<tr><td>{html.escape(summary["instruction"])}</td>'
                f'<td>{group[0]["seconds"]:.2f}–{group[-1]["seconds"]:.2f}s<br>{len(group)} frames</td>'
                f'<td>{group[0]["progress"]:.3f} → {group[-1]["progress"]:.3f}</td>'
                f'<td>{summary["final_success_probability"]:.3f} / {summary["max_success_probability"]:.3f}</td>'
                f'<td>{count_above}/{len(group)}</td><td>{summary["review_flagged_frames"]}</td></tr>'
            )
        details.extend(['</table></div>', '<h2>Points to review</h2><ul>'])
        for summary in episode_summaries:
            group = [r for r in rows if r['segment'] == summary['segment']]
            first_crossing = next((r for r in group if r['success_at_0_5']), None)
            statement = 'Success probability stays at or below 0.5 throughout this segment.'
            if first_crossing is not None:
                statement = f'Success probability first exceeds 0.5 at {first_crossing["seconds"]:.2f}s (head frame {first_crossing["head_frame_index"]}).'
                if not group[-1]['success_at_0_5']:
                    statement += ' It falls back to 0.5 or below by the final selected frame.'
            details.append(f'<li><strong>{html.escape(summary["instruction"])}</strong> {statement}</li>')
        details.extend(['</ul><h2>Source annotation review notes</h2>',
                        '<p>These notes were supplied with the selected frames. They have not been independently relabeled for this evaluation.</p><ul>'])
        details.extend(f'<li>{html.escape(note)}</li>' for note in notes)
        details.extend(['</ul><h2>Input provenance and interpretation</h2>',
                        f'<p>Raw head stream <code>{html.escape(rows[0]["raw_head_parquet"])}</code>. Every selected JPEG was verified against the raw frame’s timestamp and SHA-256. The inference input uses RGB images with the short side resized to 240 pixels before Qwen processing.</p>',
                        '<p>The gallery lists the original selected frames and their exact instructions. Changing the selected history can change a prediction. Instruction transitions reset the context, so progress values across different segments describe different tasks.</p>',
                        '<p><a href="manifest.json">Frame and context manifest</a> · <a href="results.jsonl">All frame scores</a> · <a href="summary.json">All subtask summaries</a></p>'])
        report_name = f'{episode}_report.html'
        header = report_header[:3]
        header[0] = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(episode)} — Robometer report</title>'
        (OUT / report_name).write_text('\n'.join(header + details[:4] + storyboard + details[4:] + document[section_start:]) + '</html>')
        document[section_start + 1:section_start + 1] = storyboard
        document.insert(section_start, f'<p><a href="{report_name}">Open detailed episode report</a></p>')
    (OUT / 'summary.json').write_text(json.dumps({'frames':len(results), 'segments':summaries, 'method':manifest['method']}, indent=2))
    (OUT / 'index.html').write_text('\n'.join(document) + '</html>')


if __name__ == '__main__':
    manifest = prepare()
    print(f"PREFLIGHT {manifest['frames']} frames in {manifest['episodes']} episodes verified", flush=True)
    results = infer(manifest)
    assert len(results) == manifest['frames']
    report(manifest, results)
    print('COMPLETE', OUT / 'index.html', flush=True)
