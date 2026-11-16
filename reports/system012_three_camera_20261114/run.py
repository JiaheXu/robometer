"""Three-camera tiled-input experiment; head-only artifacts are read-only."""
import hashlib
import html
import io
import json
import os
import runpy
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / 'system012_head_20261114'
SOURCE = Path('/home/jiahe/system012/output/important_frames_all_20261114_complete')
CAMERAS = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
LABELS = ['Head', 'Left wrist', 'Right wrist']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_snapshot():
    return {str(p.relative_to(BASE)): digest(p) for p in BASE.rglob('*') if p.is_file()}


def prepare():
    baseline = json.loads((BASE / 'manifest.json').read_text())
    jobs = []
    for episode in dict.fromkeys(j['episode'] for j in baseline['jobs']):
        events = json.loads((SOURCE / episode / 'events.json').read_text())
        event_map = {e['id']: e for e in events['events']}
        raw = {}
        for camera in CAMERAS:
            path = Path(events['episode']) / f'observation.images.{camera}.parquet'
            table = pq.read_table(path, columns=['source_timestamp_ns', f'observation.images.{camera}'])
            timestamps = table['source_timestamp_ns'].to_numpy()
            assert np.all(np.diff(timestamps) > 0), camera
            raw[camera] = (path, table, timestamps)
        for job in (j for j in baseline['jobs'] if j['episode'] == episode):
            event = event_map[job['event_id']]
            provenance = {}
            tiles = []
            for camera in CAMERAS:
                info = event['images'][camera]
                path, table, timestamps = raw[camera]
                index = info['frame_index']
                timestamp = int(timestamps[index])
                assert timestamp == info['source_timestamp_ns'] <= job['source_timestamp_ns']
                assert index == int(np.searchsorted(timestamps, job['source_timestamp_ns'], side='right')) - 1
                relative = str(Path(episode) / info['file'])
                assert relative == job['images'][camera]
                content = (SOURCE / relative).read_bytes()
                image_hash = hashlib.sha256(content).hexdigest()
                assert image_hash == hashlib.sha256(table[f'observation.images.{camera}'][index].as_py()).hexdigest()
                if camera == 'cam_head':
                    assert image_hash == job['head_image_sha256']
                with Image.open(io.BytesIO(content)) as im:
                    original_size = list(im.size)
                    im = im.convert('RGB')
                    scale = 240 / min(im.size)
                    im = im.resize(tuple(round(n * scale) for n in im.size), Image.Resampling.BICUBIC)
                    assert im.size == (427, 240), (camera, im.size)
                    tiles.append(np.asarray(im))
                provenance[camera] = dict(raw_parquet=str(path), frame_index=index,
                    source_timestamp_ns=timestamp, age_ms=(job['source_timestamp_ns'] - timestamp) / 1e6,
                    jpeg_sha256=image_hash, original_size=original_size, image=relative)
            combined = np.concatenate(tiles, axis=1)
            assert combined.shape == (240, 1281, 3)
            image_relative = f'inputs/{episode}/{job["event_id"]}.png'
            target = OUT / image_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(combined).save(target)
            jobs.append({**job, 'camera_provenance': provenance, 'composite_image': image_relative,
                         'composite_sha256': digest(target),
                         'context_images': [f'inputs/{episode}/{event_id}.png' for event_id in job['context_event_ids']]})
        print('VERIFIED', episode, flush=True)
    for old, new in zip(baseline['jobs'], jobs):
        for key in ('episode', 'event_id', 'source_timestamp_ns', 'next_subtask_instruction', 'segment', 'context_event_ids', 'context_head_frame_indices'):
            assert old[key] == new[key], key
        assert all((OUT / image).is_file() for image in new['context_images'])
    assert len(jobs) == baseline['frames'] == 457
    manifest = {**baseline, 'camera': CAMERAS, 'jobs': jobs,
        'baseline_directory': str(BASE), 'layout': 'Horizontal: head | left wrist | right wrist, no text or padding',
        'resize': 'Each view RGB 427x240, aspect ratio preserved; concatenate to 1281x240; then the unchanged Qwen processor',
        'method': baseline['method'] + ' Each time step is a horizontal three-camera composite; context IDs and instructions match the saved head-only run exactly.',
        'interpretation': 'Tiled-input sensitivity experiment, not a dedicated multi-camera architecture or an accuracy benchmark. More cameras also increase image area and change visual layout.',
        'all_camera_timestamp_and_jpeg_sha256_verified': True,
        'max_camera_age_ms': {camera: max(j['camera_provenance'][camera]['age_ms'] for j in jobs) for camera in CAMERAS}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def compare(results):
    old = [json.loads(line) for line in (BASE / 'results.jsonl').read_text().splitlines()]
    assert len(old) == len(results) == 457
    paired = []
    for a, b in zip(old, results):
        for key in ('episode', 'event_id', 'next_subtask_instruction', 'context_event_ids', 'segment'):
            assert a[key] == b[key], key
        assert b['input_frames_shape'] == [len(b['context_event_ids']), 240, 1281, 3]
        assert all(np.isfinite(b[k]) and 0 <= b[k] <= 1 for k in ('progress', 'success_probability'))
        paired.append(dict(episode=b['episode'], event_id=b['event_id'], segment=b['segment'], seconds=b['seconds'],
            instruction=b['next_subtask_instruction'], head_progress=a['progress'], three_progress=b['progress'],
            head_success=a['success_probability'], three_success=b['success_probability'],
            delta_progress=b['progress']-a['progress'], delta_success=b['success_probability']-a['success_probability'],
            threshold_flip=a['success_at_0_5'] != b['success_at_0_5']))
    (OUT / 'comparison.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in paired))
    return paired


def metrics(pairs):
    return dict(frames=len(pairs), mean_delta_progress=float(np.mean([p['delta_progress'] for p in pairs])),
        mean_absolute_delta_progress=float(np.mean([abs(p['delta_progress']) for p in pairs])),
        mean_delta_success=float(np.mean([p['delta_success'] for p in pairs])),
        mean_absolute_delta_success=float(np.mean([abs(p['delta_success']) for p in pairs])),
        success_threshold_flips=sum(p['threshold_flip'] for p in pairs),
        head_frames_above_threshold=sum(p['head_success'] > .5 for p in pairs),
        three_frames_above_threshold=sum(p['three_success'] > .5 for p in pairs))


def report(manifest, results, pairs):
    header = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Three-camera comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px;color:#18202c}img{max-width:100%;height:auto}section{border-top:1px solid #ddd;margin:24px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse}p{line-height:1.5}.note{background:#f1f5f9;padding:14px}</style>'
    baseline_link = os.path.relpath(BASE / 'index.html', OUT)
    method = f'<p><a href="{baseline_link}">Preserved head-only reports</a> · <a href="manifest.json">Input provenance</a> · <a href="comparison.jsonl">Paired scores</a></p><p class="note">Same 457 selected times, 33 instruction segments, checkpoint, prompts, and up-to-eight-step histories. Three-camera input tiles head | left wrist | right wrist horizontally, each at 427×240 pixels (combined 1281×240 before Qwen processing). Camera frames are the latest available at or before the head timestamp, verified against raw data. This changes both visual information and input area/layout. Score differences do not establish better accuracy; human success labels are unavailable.</p>'
    overall = metrics(pairs)
    index = [header, '<h1>Head-only vs three-camera evaluation</h1>',
        f'<p><strong>{overall["frames"]} matched frames</strong> · Mean absolute change: progress {overall["mean_absolute_delta_progress"]:.3f}, success {overall["mean_absolute_delta_success"]:.3f} · {overall["success_threshold_flips"]} success-threshold flips. <a href="summary.json">Full summary</a></p>', method,
        '<p>Dashed blue curves: head only. Solid orange curves: three cameras. A/B/C markers match the image columns. Images beneath each plot show head, left wrist, and right wrist at those selected times.</p>',
        '<table><tr><th>Episode</th><th>Frames</th><th>Mean |Δ progress|</th><th>Mean |Δ success|</th><th>Threshold flips</th></tr>']
    summaries = dict(overall=overall, episodes=[], segments=[])
    for episode in dict.fromkeys(r['episode'] for r in results):
        rows = [r for r in results if r['episode'] == episode]
        ep_pairs = [p for p in pairs if p['episode'] == episode]
        stat = dict(episode=episode, **metrics(ep_pairs))
        summaries['episodes'].append(stat)
        name = f'{episode}_report.html'
        index.append(f'<tr><td><a href="{name}">{html.escape(episode)}</a></td><td>{len(rows)}</td><td>{stat["mean_absolute_delta_progress"]:.3f}</td><td>{stat["mean_absolute_delta_success"]:.3f}</td><td>{stat["success_threshold_flips"]}</td></tr>')
        page = [header, '<p><a href="index.html">All episodes</a></p>', f'<h1>{html.escape(episode)}</h1>', method,
                f'<p>{len(rows)} frames; mean absolute change: progress {stat["mean_absolute_delta_progress"]:.3f}, success {stat["mean_absolute_delta_success"]:.3f}; {stat["success_threshold_flips"]} frames switch sides of the 0.5 success threshold.</p>']
        for segment in dict.fromkeys(r['segment'] for r in rows):
            group = [r for r in rows if r['segment'] == segment]
            ps = [p for p in ep_pairs if p['segment'] == segment]
            selected = sorted({0, len(group)//2, len(group)-1})
            fig = plt.figure(figsize=(11, 10), layout='constrained')
            grid = fig.add_gridspec(5, 3, height_ratios=[1.25, 1.25, 1, 1, 1])
            for row_index, metric in enumerate(('progress', 'success')):
                ax = fig.add_subplot(grid[row_index, :])
                for prefix, label, color, style in [('head', 'Head only', '#2563eb', '--'), ('three', 'Three cameras', '#ea580c', '-')]:
                    ax.plot([p['seconds'] for p in ps], [p[f'{prefix}_{metric}'] for p in ps], marker='.', linestyle=style, color=color, label=label)
                ax.set(ylabel='Progress' if metric == 'progress' else 'Success probability', ylim=(-.03, 1.15))
                ax.set_yticks([0, .5, 1]); ax.grid(alpha=.15)
                if metric == 'success':
                    ax.axhline(.5, color='gray', linestyle=':', linewidth=1)
                    ax.set_xlabel('Episode time (seconds)')
                else:
                    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, frameon=False)
                for col, selected_index in enumerate(selected):
                    time = group[selected_index]['seconds']
                    ax.axvline(time, color='#64748b', linestyle=':', linewidth=1)
                    ax.text(time, 1.07, chr(65+col), ha='center', va='center', fontsize=9)
            for col, selected_index in enumerate(selected):
                row = group[selected_index]
                for camera_index, camera in enumerate(CAMERAS):
                    ax = fig.add_subplot(grid[2+camera_index, col])
                    with Image.open(SOURCE / row['images'][camera]) as im:
                        im = im.convert('RGB'); im.thumbnail((320, 180)); ax.imshow(im)
                    ax.axis('off')
                    title = f'{LABELS[camera_index]} · age {row["camera_provenance"][camera]["age_ms"]:.1f} ms'
                    if camera_index == 0:
                        title = f'{chr(65+col)} · {row["seconds"]:.2f}s · Head'
                    ax.set_title(title, fontsize=9, loc='left')
            instruction = group[0]['next_subtask_instruction']
            fig.suptitle(textwrap.fill(f'{segment+1}. {instruction}', 95), fontsize=12, fontweight='bold')
            plot = f'{episode}_subtask_{segment+1:02d}.png'
            fig.savefig(OUT / plot, dpi=120); plt.close(fig)
            stat = dict(episode=episode, segment=segment, instruction=instruction, **metrics(ps),
                final_head_progress=ps[-1]['head_progress'], final_three_progress=ps[-1]['three_progress'],
                final_head_success=ps[-1]['head_success'], final_three_success=ps[-1]['three_success'])
            summaries['segments'].append(stat)
            page.extend([f'<section><h2>{segment+1}. {html.escape(instruction)}</h2>',
                f'<p>{len(group)} frames · Final progress: head {ps[-1]["head_progress"]:.3f} → three {ps[-1]["three_progress"]:.3f}; final success: head {ps[-1]["head_success"]:.3f} → three {ps[-1]["three_success"]:.3f}. Threshold flips: {stat["success_threshold_flips"]}.</p>',
                f'<img src="{plot}" alt="Head-only and three-camera progress and success curves with labeled views at matching timestamps">'])
            notes = list(dict.fromkeys(note for r in group for note in r['review_notes']))
            if notes:
                page.append('<p>Source annotation review notes: ' + html.escape('; '.join(notes)) + '</p>')
            page.append('</section>')
        (OUT / name).write_text('\n'.join(page) + '</html>')
    index.extend(['</table>', '<p>All changes are three-camera minus head-only. Threshold flips count model decisions, not ground-truth errors.</p></html>'])
    (OUT / 'index.html').write_text('\n'.join(index))
    (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('SUMMARY', json.dumps(summaries['overall']), flush=True)


if __name__ == '__main__':
    before = baseline_snapshot()
    if '--report-only' in sys.argv:
        manifest = json.loads((OUT / 'manifest.json').read_text())
        results = [json.loads(line) for line in (OUT / 'results.jsonl').read_text().splitlines()]
    else:
        manifest = prepare()
        if '--prepare-only' in sys.argv:
            assert before == baseline_snapshot()
            print('PREFLIGHT COMPLETE', manifest['max_camera_age_ms'], flush=True)
            sys.exit(0)
        namespace = runpy.run_path(str(BASE / 'run.py'))
        infer = namespace['infer']
        infer.__globals__.update(OUT=OUT, SOURCE=OUT)
        results = infer(manifest)
    pairs = compare(results)
    report(manifest, results, pairs)
    assert before == baseline_snapshot(), 'Head-only artifacts changed'
    (OUT / 'baseline_preservation.json').write_text(json.dumps(dict(verified_unchanged=True, sha256=before), indent=2))
    print('COMPLETE', OUT / 'index.html', flush=True)
