"""Clip-level RBF comparators for native and fixed, PTS-resampled ranking sequences.

Training entry points take train and validation ONLY. ``predict`` and
``predict_proba`` require only x/times/duration/breaks, not labels or identities.
The fitted model includes its fitted scaler. Probabilities are reported in the
explicit order LABELS, while predictions come from the SVC decision rule.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/tensor_sequence_classification_20261109'
LABELS = ('polite', 'bitter', 'true')
SEED = 20261109


def _arrays(seq):
    times = np.asarray(seq['times'], dtype=np.float64)
    x = np.asarray(seq['x'], dtype=np.float64)
    duration = float(seq['duration'])
    starts = np.asarray(seq['breaks'], dtype=bool)
    if (times.ndim != 1 or len(times) < 2 or x.shape != (len(times), 18)
            or starts.shape != times.shape or not starts[0]):
        raise ValueError('Expected N>=2 observations, N×18 scores and a segment-start mask')
    if (not np.isfinite(times).all() or not np.isfinite(x).all() or not np.isfinite(duration)
            or duration <= 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Expected finite observations and strictly increasing PTS')
    if abs(times[0]) > 2e-6 or abs(times[-1] - duration) > 2e-6:
        raise ValueError('Original onset and peak must be present at times 0 and duration')
    return times, x, duration, starts


def clip_embedding(seq, degree=3, raw73=False):
    """Return one 73D (control), 127D (degree 3) or 199D (degree 7) vector.

    For the cosine model, let u=t/original_duration and assign half of each
    contiguous edge's Δu to either endpoint. Normalize the resulting trapezoid
    weights by their sum (total observed contiguous span). A break contributes
    zero edge weight. For channel j, μ_j = Σw_i x_ij and coefficient k is
    2 Σ w_i (x_ij−μ_j) cos(kπu_i). DC is represented separately by μ_j.
    No feature fits or labels are used by this transformation.
    """
    t, x, duration, starts = _arrays(seq)
    start, end = x[0], x[-1]
    if raw73:
        # Match the old 73D control's ordinary trapezoidal mean over all PTS,
        # including its treatment of occasional gaps. Only raw vs smooth differs
        # in the aggregate feature semantics; SVC settings below are fixed.
        mean = np.trapezoid(x, t, axis=0) / duration
        return np.concatenate(([duration], np.stack((start, end, end-start, mean), axis=1).ravel()))
    if degree not in (3, 7):
        raise ValueError('Predeclared cosine orders are 3 and 7')
    u = t / duration
    edges = np.where(starts[1:], 0.0, np.diff(u))
    observed_span = edges.sum()
    if observed_span <= 0:
        raise ValueError('No positive-duration contiguous observed edge')
    w = np.zeros_like(u)
    w[:-1] += 0.5 * edges
    w[1:] += 0.5 * edges
    w /= observed_span
    mean = w @ x
    cosines = np.cos(np.pi * u[:, None] * np.arange(1, degree + 1)[None, :])
    coefficients = 2.0 * ((x - mean).T @ (w[:, None] * cosines))
    return np.concatenate(([np.log(duration)], np.stack((start, end, end-start, mean), axis=1).ravel(),
                           coefficients.ravel()))


class TrajectoryKernelModel:
    """A frozen, label-free sequence-to-prediction model (for joblib)."""

    def __init__(self, pipeline, degree, raw73):
        self.pipeline = pipeline
        self.degree = degree
        self.raw73 = raw73
        self.classes_ = np.asarray(LABELS)

    def transform(self, sequences):
        items = [sequences] if isinstance(sequences, dict) else sequences
        return np.stack([clip_embedding(seq, self.degree, self.raw73) for seq in items])

    def predict(self, sequences):
        return self.pipeline.predict(self.transform(sequences))

    def predict_proba(self, sequences):
        order = [list(self.pipeline.classes_).index(label) for label in LABELS]
        return self.pipeline.predict_proba(self.transform(sequences))[:, order]


def _validate_partition(train, val):
    for name, rows in [('train', train), ('val', val)]:
        if not rows or any(r['split'] != name for r in rows):
            raise ValueError(f'{name} argument must contain only the named split')
        if set(r['label'] for r in rows) != set(LABELS):
            raise ValueError(f'{name} lacks three-class support')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate clips in {name}')
    for key in ('id', 'component'):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f'Train/validation overlap by {key}')


def _fingerprint(train, val, arm):
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    h.update(arm.encode())
    for rows in (train, val):
        for r in rows:
            h.update(json.dumps([r['split'], r['id'], r['component'], r['label']], ensure_ascii=True).encode())
            t, x, d, b = _arrays(r)
            for a in (np.array([len(t)], dtype='<i8'), t, x, np.array([d]), b):
                h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _metrics(y, predicted):
    return dict(n=len(y), correct=int(np.sum(y == predicted)), accuracy=float(accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, labels=LABELS, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                confusion_matrix=confusion_matrix(y, predicted, labels=LABELS).tolist())


def _run(train, val, arm, run_root, raw73):
    train, val = list(train), list(val)
    _validate_partition(train, val)
    destination = Path(run_root) / 'raw73_control' if raw73 else Path(run_root) / arm / 'kernel'
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / 'classifier.joblib'
    record = destination / 'selection_before_test.json'
    digest = _fingerprint(train, val, arm)
    if record.is_file():
        choice = json.loads(record.read_text(encoding='utf8'))
        if digest != choice['input_protocol_sha256']:
            raise ValueError(f'Frozen training inputs or source changed: {destination}')
        model = joblib.load(model_path)
    else:
        choices, best, model = [], None, None
        y_train = np.asarray([r['label'] for r in train]); y_val = np.asarray([r['label'] for r in val])
        for degree in ([0] if raw73 else [3, 7]):
            x_train = np.stack([clip_embedding(r, degree, raw73) for r in train])
            x_val = np.stack([clip_embedding(r, degree, raw73) for r in val])
            for C in ([10.] if raw73 else [.1, 1., 10.]):
                # Internal SVC probability estimates use only TRAIN examples;
                # no candidate fits preprocessing, model or calibration on val.
                estimator = make_pipeline(StandardScaler(), SVC(C=C, kernel='rbf', gamma='scale',
                                          class_weight='balanced', probability=True, random_state=SEED))
                estimator.fit(x_train, y_train)
                metrics = _metrics(y_val, estimator.predict(x_val))
                candidate = dict(selected_model=('raw73' if raw73 else f'cosine{degree}') + f'_rbf_C{C:g}',
                                 C=C, degree=degree, n_features=x_train.shape[1], validation=metrics)
                choices.append(candidate)
                rank = (metrics['accuracy'], metrics['macro_f1'])
                if best is None or rank > best:
                    best, chosen = rank, candidate
                    model = TrajectoryKernelModel(estimator, degree, raw73)
        choice = {**chosen, 'arm':arm, 'input_protocol_sha256':digest, 'seed':SEED,
                  'selection_rule':('single prespecified C=10 control' if raw73 else
                      'validation accuracy, then validation macro-F1, then first declared order'),
                  'trials':choices, 'probability_order':list(LABELS),
                  'training_classes':dict(Counter(r['label'] for r in train)),
                  'validation_classes':dict(Counter(r['label'] for r in val)),
                  'protocol':('raw duration seconds, endpoints, delta, trapezoidal mean per raw channel (all native PTS)' if raw73 else
                      'log(duration), start/end/delta/observed-span trapezoidal mean per raw channel, '
                      'then centered nonzero cosine coefficients of orders 1..K on normalized actual timestamps; '
                      'edges crossing breaks contribute zero integration weight'),
                  'probability_note':'Unchanged SVC decisions for ACC; libsvm probability estimates are fitted internally on the training set.'}
        temporary = destination / 'classifier.tmp'
        joblib.dump(model, temporary)
        temporary.replace(model_path)
        temporary = destination / 'selection_before_test.tmp'
        temporary.write_text(json.dumps(choice, ensure_ascii=False, indent=2), encoding='utf8')
        temporary.replace(record)
    return dict(model=model, model_path=str(model_path), selection=choice,
                val_pred=model.predict(val), val_proba=model.predict_proba(val))


def run_arm(train_sequences, val_sequences, arm, run_root=OUT):
    """Fit/select six RBF candidates on either 'fixed' or 'native' representation."""
    if arm not in ('fixed', 'native'):
        raise ValueError('arm must be fixed or native')
    return _run(train_sequences, val_sequences, arm, run_root, raw73=False)


def run_raw73_control(train_native, val_native, run_root=OUT):
    """Prespecified raw-sequence analog of the previous smooth 73D baseline."""
    return _run(train_native, val_native, 'raw73_control', run_root, raw73=True)
