"""Inference-only neutral-candidate -> apex extraction; never modifies source data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipeline import ROOT, frame_number, read_csv, save_json, uniform_indices, write_csv

PROTOCOL = dict(version=2, smooth_seconds=0.10, baseline_seconds=0.15,
                sustain_seconds=0.10, min_amplitude=0.08, max_gap_seconds=0.15,
                low_fraction=0.20, peak_fraction=0.95, neutral_return_seconds=0.40)


def detect(t, scores, params=None):
    """Indices refer to observed frames, never interpolated or time-warped frames."""
    p = {**PROTOCOL, **(params or {})}
    t, s = np.asarray(t, float), np.asarray(scores, float)
    result = dict(status="review", reasons=[], start_index=None, onset_index=None,
                  peak_index=None, maximum_index=None, neutral_confirmed=False)
    if len(t) < 5 or not np.all(np.isfinite(s)) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        return {**result, "reasons": ["invalid_or_insufficient_samples"]}, s
    y = np.array([np.median(s[np.abs(t - time) <= p['smooth_seconds'] / 2]) for time in t])
    low = float(np.quantile(y, .20))
    maximum = int(np.argmax(y))
    amplitude = float(y[maximum] - low)
    noise = float(1.4826 * np.median(np.abs(s - y)))
    result.update(amplitude=amplitude, noise_mad=noise, maximum_index=maximum,
                  low_reference=low, peak_score=float(y[maximum]))
    if amplitude < max(p['min_amplitude'], 6 * noise):
        return {**result, "reasons": ["weak_or_flat_signal"]}, y
    # Validate apex support, but export through the actual smoothed maximum.
    threshold = low + p['peak_fraction'] * amplitude
    apex_entry = maximum
    sustained_peak = False
    for i in range(maximum + 1):
        j = int(np.searchsorted(t, t[i] + p['sustain_seconds']))
        if j < len(t) and np.all(y[i:j + 1] >= threshold):
            apex_entry, sustained_peak = i, True
            break
    peak = maximum
    result['peak_index'] = peak
    result['apex_entry_index'] = apex_entry
    if not sustained_peak:
        result['reasons'].append('brief_or_unconfirmed_peak')
    if t[-1] - t[maximum] < p['sustain_seconds']:
        result['reasons'].append('peak_near_end_possible_truncation')
    low_limit = low + p['low_fraction'] * amplitude
    # Find the FIRST supported rise. Never reset onset to a later dip in the same smile.
    windows = []
    for end in range(peak):
        start = int(np.searchsorted(t, t[end] - p['baseline_seconds'], side='right') - 1)
        if start < 0 or t[end] - t[start] < p['baseline_seconds']:
            continue
        segment = y[start:end + 1]
        if np.max(segment) <= low_limit and np.ptp(segment) <= max(.02, amplitude * .1):
            windows.append((start, end))
    if not windows:
        return {**result, "reasons": result['reasons'] + ['no_stable_low_baseline']}, y
    start, base_end = windows[0]
    onset = None
    for i in range(base_end + 1, peak + 1):
        j = int(np.searchsorted(t, t[i] + p['sustain_seconds'] - 1e-6))
        if j < len(t) and np.all(y[i:j+1] > low_limit):
            onset = i
            break
    if onset is None or onset >= peak:
        return {**result, "reasons": result['reasons'] + ['no_sustained_rise_before_peak']}, y
    start, base_end = [w for w in windows if w[1] < onset][-1]
    result.update(start_index=start, onset_index=onset,
                  rise_seconds=float(t[peak]-t[onset]),
                  clip_seconds=float(t[peak]-t[start]))
    if np.any(np.diff(t[start:peak+1]) > p['max_gap_seconds']):
        result['reasons'].append('tracking_gap_in_selected_interval')
    if y[0] > low_limit + .1 * amplitude:
        result['reasons'].append('initial_expression_above_baseline')
    # Local reversals are valid within-smile dynamics, not separate smiles or QC failures.
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y, prominence=max(p['min_amplitude'], .15 * amplitude))
    proposals = sorted(set([int(i) for i in peaks if i >= onset and y[i] >= low + .3*amplitude] + [maximum]))
    peaks = []
    for i in proposals:
        if peaks and min(y[peaks[-1]],y[i])-float(np.min(y[peaks[-1]:i+1])) < max(p['min_amplitude'], .15*amplitude):
            # argmax and find_peaks may choose different frames on the SAME plateau.
            if y[i] > y[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)
    last_peak = peaks[-1]
    result.update(local_peak_count=len(peaks), has_rebound=len(peaks)>1, last_peak_index=last_peak)
    low_start = None
    for i in range(onset, last_peak+1):
        if y[i] <= low_limit:
            if low_start is None:
                low_start = i
            if t[i]-t[low_start] >= p['neutral_return_seconds']:
                result['reasons'].append('sustained_low_return_review_event_boundary')
                break
        else:
            low_start = None
    if not result['reasons']:
        result['status'] = 'candidate'
    return result, y


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(path):
    p = Path(path); stat = p.stat()
    return [str(p.resolve()), stat.st_size, stat.st_mtime_ns]


def timestamps(source):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
               '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(source)]
    payload = json.loads(subprocess.check_output(command, text=True, encoding='utf8'))
    times = np.array([float(f['best_effort_timestamp_time']) for f in payload['frames']])
    if len(times) == 0 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Missing/non-increasing source timestamps')
    return times - times[0]


class Scorer:
    def __init__(self, cfg, weights, anchor_paths, batch_size):
        import torch
        from PIL import Image
        sys.path.insert(0, cfg['smilecomp_dir'])
        sys.path.insert(0, str(ROOT / 'annotation'))
        sys.path.insert(0, str(ROOT / 'Analysis/smile_ranking_gradcam'))
        import face_comp_torch as fc
        from build_true_smile_scale_from_model import prepare_transform
        from roi_pattern_common import score_feature_batch
        self.torch, self.Image, self.score_batch = torch, Image, score_feature_batch
        self.device = torch.device(cfg['device'])
        self.model = fc.face_comp_siamese_model_vgg16based_for_predict(str(weights), on_device=self.device).to(self.device).eval()
        self.transform = prepare_transform(fc)
        self.batch_size = batch_size
        self.anchors = self.features(anchor_paths)

    def features(self, paths):
        torch = self.torch
        batches = []
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                tensors = []
                for path in paths[start:start+self.batch_size]:
                    with self.Image.open(path) as im:
                        tensors.append(self.transform(im.convert('RGB')))
                batches.append(self.model.extractor(torch.stack(tensors).to(self.device)).detach())
        return torch.cat(batches)

    def score(self, paths):
        scores, consistency = [], []
        with self.torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                result = self.score_batch(self.model, self.features(paths[start:start+self.batch_size]), self.anchors)
                scores.extend(map(float, result['mean_probability']))
                consistency.extend(map(float, result['mean_anchor_consistency']))
        return scores, consistency


def export_video(source, dest, start_frame, peak_frame):
    # trim by decoded source frame ordinal. Preserve variable timestamps; no stream-copy keyframe drift.
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.stem + '.partial.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-map', '0:v:0',
                    '-vf', f'trim=start_frame={start_frame}:end_frame={peak_frame+1},setpts=PTS-STARTPTS',
                    '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-fps_mode', 'vfr', '-movflags', '+faststart', str(temp)], check=True)
    temp.replace(dest)


def render_evidence(folder, paths, times, scores, smooth, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, scores, alpha=.4, label='raw anchor score')
    ax.plot(times, smooth, label='time-window median')
    for key, color in [('start_index', 'gray'), ('onset_index', 'green'), ('peak_index', 'red')]:
        if result.get(key) is not None:
            exact_time = result.get(key.replace('_index', '_seconds'))
            ax.axvline(times[result[key]] if exact_time is None else exact_time, color=color, label=key.removesuffix('_index'))
    ax.set(xlabel='Source time (seconds)', ylabel='Mean comparison probability', ylim=(-.02, 1.02))
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/'curve.png', dpi=110); plt.close(fig)
    selected = sorted(set(uniform_indices(len(paths), 8) +
                          [result[k] for k in ['start_index','onset_index','peak_index'] if result.get(k) is not None]))
    montage = Image.new('RGB', (160*len(selected), 185), 'white')
    for j, i in enumerate(selected):
        with Image.open(paths[i]) as im:
            montage.paste(im.convert('RGB').resize((160,160)), (160*j,25))
        ImageDraw.Draw(montage).text((160*j+2, 3), f'{times[i]:.3f}s  {scores[i]:.3f}', fill='black')
    montage.save(folder/'frames.jpg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--person', required=True, choices=['JIANG','XU','YAMAMOTO','TAKEDA','REN'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit-per-label', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--anchors', type=int, default=32)
    parser.add_argument('--export', action='store_true', help='Export all structurally valid candidates, including flagged ones, into separate folders.')
    parser.add_argument('--score-cache', type=Path, help='Reuse matching raw inference from an earlier run; boundary detection is always rerun.')
    parser.add_argument('--overrides', type=Path, help='JSON keyed by sequence_id: start_frame, peak_frame, neutral_confirmed, optional exclude=true.')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    person = Path(cfg['output_root']) / args.person
    out = args.output.resolve() / args.person
    if out == person.resolve() or person.resolve() in out.parents:
        raise ValueError('Use a separate output directory outside the prepared experiment.')
    if args.anchors < 2 or args.batch_size < 1:
        raise ValueError('At least 2 anchors and a positive batch size are required')
    out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(args.overrides.read_text(encoding='utf-8-sig')) if args.overrides else {}
    records = read_csv(person/'sequences.csv')
    unknown = set(overrides) - {r['sequence_id'] for r in records}
    if unknown:
        raise ValueError(f'Unknown override IDs: {sorted(unknown)}')
    summary, skipped, errors = [], [], []
    protocol = {**PROTOCOL, 'person': args.person, 'config': cfg, 'anchor_count': args.anchors,
                'score_definition': 'Mean symmetric P(frame stronger than anchor); not absolute smile/neutral probability',
                'anchor_selection': 'Uniform over sorted train-only candidate filenames, existing source videos only',
                'neutral_policy': 'Low stable score is only a neutral candidate; manual confirmation required',
                'selection_policy': 'Native category model; first supported onset -> global smoothed maximum; preserve within-smile dips and last-peak alternative'}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text(encoding='utf8')) != protocol:
        raise ValueError('Protocol changed: use a new output directory')
    save_json(out/'protocol.json', protocol)
    for label in cfg.get('labels', ['polite','bitter']):
        active = {r['sequence_id'] for r in records if Path(r['source_video']).is_file()}
        rank = person/'ranking'/label
        items = sorted([r for r in read_csv(rank/'ranking_items.csv') if r['split']=='train' and r['sequence_id'] in active], key=lambda r:r['image_file'])
        if len(items) < args.anchors:
            raise ValueError(f'{label}: too few train anchors')
        anchors = [rank/'images'/items[i]['image_file'] for i in uniform_indices(len(items), args.anchors)]
        anchor_manifest = [{'path':str(p), 'sha256':sha(p)} for p in anchors]
        save_json(out/f'anchors_{label}.json', anchor_manifest)
        weights = rank/'model/model.pth'
        model_hash = sha(weights)
        scorer = None
        selected = [r for r in records if r['label']==label]
        missing = [r for r in selected if not Path(r['source_video']).is_file()]
        skipped.extend({**r,'reason':'source_video_missing_or_deleted'} for r in missing)
        selected = [r for r in selected if Path(r['source_video']).is_file()]
        if args.limit_per_label is not None:
            selected = selected[:args.limit_per_label]
        for number, seq in enumerate(selected, 1):
            sid = seq['sequence_id']; folder = out/label/sid; folder.mkdir(parents=True, exist_ok=True)
            print(f'{args.person} {label} {number}/{len(selected)} {sid}', flush=True)
            try:
                paths = sorted(Path(seq['sequence_dir']).glob('*.png'), key=frame_number)
                indices = [frame_number(p) for p in paths]
                if not paths or len(set(indices)) != len(indices):
                    raise ValueError('Empty/duplicate frame indices')
                fingerprint = dict(protocol=1, model=model_hash, anchors=anchor_manifest,
                                   source=signature(seq['source_video']), crops=[signature(p) for p in paths])
                cache = folder/'scores.json'
                old_cache = args.score_cache/args.person/label/sid/'scores.json' if args.score_cache else cache
                cache_input = cache if cache.exists() else old_cache
                if cache_input.exists():
                    saved = json.loads(cache_input.read_text(encoding='utf8'))
                    if saved['fingerprint'] != fingerprint:
                        raise ValueError('Input changed: use a new output directory')
                    times, scores, consistency = saved['times'], saved['scores'], saved['consistency']
                    if cache_input != cache:
                        save_json(cache, saved)
                else:
                    source_times = timestamps(seq['source_video'])
                    if min(indices) < 0 or max(indices) >= len(source_times):
                        raise ValueError('Crop frame indices do not map to decoded video frames')
                    times = source_times[indices].tolist()
                    if scorer is None:
                        scorer = Scorer(cfg, weights, anchors, args.batch_size)
                    scores, consistency = scorer.score(paths)
                    save_json(cache, dict(fingerprint=fingerprint, times=times, scores=scores, consistency=consistency))
                result, smooth = detect(times, scores)
                if float(np.mean(consistency)) < .8:
                    result['reasons'].append('low_pair_swap_consistency')
                    result['status'] = 'review'
                if result['start_index'] is not None:
                    a, b = result['start_index'], result['peak_index']
                    coverage = (b-a+1) / (indices[b]-indices[a]+1)
                    result['interval_tracking_coverage'] = coverage
                    if coverage < .85:
                        result['reasons'].append('low_interval_tracking_coverage')
                        result['status'] = 'review'
                override = overrides.get(sid)
                if override:
                    if override.get('exclude'):
                        result.update(status='excluded', reasons=['manual_exclusion'], start_index=None, peak_index=None)
                    else:
                        start, peak = indices.index(int(override['start_frame'])), indices.index(int(override['peak_frame']))
                        if start >= peak:
                            raise ValueError('Manual start must precede peak, both must be observed frames')
                        result.update(start_index=start, peak_index=peak, onset_index=None,
                                      status='manual', neutral_confirmed=bool(override.get('neutral_confirmed',False)),
                                      reasons=['manual_boundaries'], rise_seconds=None,
                                      clip_seconds=float(times[peak]-times[start]))
                result.update(sequence_id=sid, label=label, person=args.person,
                              source_video=seq['source_video'], split=seq['split'],
                              source_group=seq['source_group'], mean_swap_consistency=float(np.mean(consistency)),
                              model_sha256=model_hash)
                for key in ['start','onset','peak','maximum','last_peak','apex_entry']:
                    i = result.get(key+'_index')
                    result[key+'_frame'] = indices[i] if i is not None else None
                    result[key+'_seconds'] = times[i] if i is not None else None
                export = None
                if args.export and result['start_index'] is not None and result['peak_index'] is not None:
                    export = out/'clips'/result['status']/label/(sid+'.mp4')
                    old_selection = json.loads((folder/'selection.json').read_text(encoding='utf8')) if (folder/'selection.json').exists() else {}
                    if not (export.exists() and old_selection.get('export_video')==str(export)
                            and old_selection.get('start_frame')==result['start_frame']
                            and old_selection.get('peak_frame')==result['peak_frame']):
                        export_video(seq['source_video'], export, result['start_frame'], result['peak_frame'])
                result['export_video'] = str(export) if export else None
                previous = folder/'selection.json'
                if previous.exists():
                    old_export = json.loads(previous.read_text(encoding='utf8')).get('export_video')
                    if old_export and old_export != result['export_video']:
                        old_path = Path(old_export).resolve()
                        if (out/'clips').resolve() not in old_path.parents:
                            raise ValueError('Previous export outside managed clips folder')
                        old_path.unlink(missing_ok=True)
                save_json(folder/'selection.json', result)
                write_csv(folder/'curve.csv', [dict(source_frame=f, seconds=t, raw_score=s, smoothed_score=float(y), swap_consistency=c)
                          for f,t,s,y,c in zip(indices,times,scores,smooth,consistency)])
                render_evidence(folder, paths, times, scores, smooth, result)
                summary.append(result)
            except Exception as exc:
                errors.append(dict(sequence_id=sid, source_video=seq['source_video'], error=str(exc)))
                print(f'ERROR {sid}: {exc}', flush=True)
        if scorer is not None:
            del scorer
            import torch
            torch.cuda.empty_cache()
    known_sources = {Path(r['source_video']).resolve() for r in records}
    for label, dirname in cfg['label_folders'].items():
        for source in sorted((Path(cfg['input_root'])/args.person/dirname).glob('*.mp4')):
            if source.resolve() not in known_sources:
                skipped.append(dict(source_video=str(source), label=label, reason='no_prepared_sequence'))
    save_json(out/'summary.json', summary)
    save_json(out/'skipped.json', skipped)
    save_json(out/'errors.json', errors)
    if summary:
        fields = ['person','label','sequence_id','source_video','split','source_group','status','reasons','neutral_confirmed',
                  'start_frame','onset_frame','peak_frame','maximum_frame','start_seconds','onset_seconds','peak_seconds',
                  'maximum_seconds','last_peak_frame','last_peak_seconds','has_rebound','local_peak_count','rise_seconds','clip_seconds','amplitude','mean_swap_consistency','interval_tracking_coverage','export_video']
        write_csv(out/'segments.csv', [{k:(';'.join(r[k]) if k=='reasons' else r.get(k)) for k in fields} for r in summary])
    parts = ['<!doctype html><meta charset="utf-8"><title>Neutral → apex review</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto}article{border-top:1px solid #ccc;padding:24px 0}img{max-width:100%}video{width:480px}code{overflow-wrap:anywhere}</style>',
             f'<h1>{args.person}: neutral候选 → 模型峰值</h1><p>这是模型建议边界，不是人工确认的neutral。分数是相对参考图的比较概率，不可跨人物/模型直接比较。候选和需复核片段分开存放。</p><p>{len(summary)} processed; {len(skipped)} missing/deleted; {len(errors)} errors.</p>']
    for r in summary:
        prefix = f'{r["label"]}/{r["sequence_id"]}'
        parts.append(f'<article><h2>{html.escape(Path(r["source_video"]).name)}</h2><p>{r["label"]} / {r["status"]} / {html.escape(", ".join(r["reasons"]))}</p><p>start={r["start_seconds"]}; onset={r["onset_seconds"]}; peak={r["peak_seconds"]} seconds</p><a href="{Path(r["source_video"]).as_uri()}">原视频</a><br><img src="{prefix}/curve.png"><img src="{prefix}/frames.jpg">')
        if r['export_video']:
            parts.append(f'<video controls preload="none" src="{Path(r["export_video"]).relative_to(out).as_posix()}"></video>')
        parts.append('</article>')
    (out/'review.html').write_text('\n'.join(parts), encoding='utf8')
    save_json(out/'needs_review.json', [r for r in summary if r['status'] != 'candidate'])
    print(f'DONE {out}: {len(summary)} processed, {len(skipped)} skipped, {len(errors)} errors', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
