"""CPU-only, source-aligned D/E/F forward masking calculations.

The distribution and span rules are translated from the verified experiment
scripts. NumPy random samples are NOT bit-identical to PyTorch RNG samples.
No trained model weights are loaded by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEATURES = ("ACT_SITE", "BINDING", "MOTIF", "DOMAIN", "REGION")
PRETTY = {
    "ACT_SITE": "Active site",
    "BINDING": "Binding site",
    "MOTIF": "Motif",
    "DOMAIN": "Domain",
    "REGION": "Region",
    "NORMAL": "Ordinary",
}
# Source: facebook/esm2_t30_150M_UR50D/vocab.txt (vocabulary index = token id)
ESM_VOCAB = (
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T",
    "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O",
    ".", "-", "<null_1>", "<mask>",
)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(ESM_VOCAB)}
assert TOKEN_TO_ID['<cls>'] == 0 and TOKEN_TO_ID['<eos>'] == 2 and TOKEN_TO_ID['<mask>'] == 32

# Higher annotation categories replace lower categories when overlapping.
PRIORITY = ("REGION", "DOMAIN", "MOTIF", "BINDING", "ACT_SITE")


@dataclass
class Result:
    experiment: str
    time: float
    seed: int
    sequence: str
    frame: pd.DataFrame
    token_ids: list[int]
    corrupted_ids: list[int]
    span_features: np.ndarray

    @property
    def masked_count(self) -> int:
        return int(self.frame['Masked'].sum())

    @property
    def expected_count(self) -> float:
        return float(self.frame['q(t)'].sum())

    @property
    def observed_fraction(self) -> float:
        return self.masked_count / len(self.sequence)


def feature_matrix(sequence: str, features: list[dict[str, Any]]) -> np.ndarray:
    """Return Boolean [L,5] feature channels; feature locations are 1-based inclusive."""
    length = len(sequence)
    matrix = np.zeros((length, len(FEATURES)), dtype=bool)
    for f in features:
        kind = f.get('type')
        if kind not in FEATURES:
            continue
        start, end = int(f['start']), int(f['end'])
        if 1 <= start <= end <= length:
            matrix[start - 1:end, FEATURES.index(kind)] = True
    return matrix


def categories_from_features(features: np.ndarray) -> list[str]:
    categories = ['NORMAL'] * len(features)
    for category in PRIORITY:
        column = features[:, FEATURES.index(category)]
        for pos in np.flatnonzero(column):
            categories[int(pos)] = category
    return categories


def exponents(experiment: str, features: np.ndarray) -> np.ndarray:
    if experiment not in ('D', 'E', 'F'):
        raise ValueError('Experiment must be D, E, or F')
    length = len(features)
    if experiment == 'D':
        return np.ones(length, dtype=float)
    values = np.ones(length, dtype=float) if experiment == 'E' else np.full(length, 1.40)
    if experiment == 'E':
        values[features[:, 4]] = 1.25  # REGION
        values[features[:, 3]] = 1.50  # DOMAIN
        values[features[:, :3].any(axis=1)] = 2.0  # functional
    else:
        values[features[:, 4]] = 0.95
        values[features[:, 3]] = 0.75
        values[features[:, :3].any(axis=1)] = 0.55
    return values


def span_anchors(categories: list[str], max_span: int = 8) -> np.ndarray:
    """Exact F rule: separate runs of equal category into blocks of max_span."""
    if max_span < 1:
        raise ValueError('max_span must be >=1')
    anchors = np.zeros(len(categories), dtype=int)
    run_start = 0
    for pos, category in enumerate(categories):
        if pos == 0 or category != categories[pos - 1]:
            run_start = pos
        anchors[pos] = run_start + ((pos - run_start) // max_span) * max_span
    return anchors


def calculate(protein: dict[str, Any], experiment: str, time: float, seed: int = 2026) -> Result:
    """Reproduce corruption law and span grouping; use NumPy as a CPU random source.

    For a single protein, all residue positions are eligible. Special tokens are
    appended in tokenization and intentionally excluded from corruption.
    """
    seq = protein['sequence']
    if not seq or len(seq) > 510:
        raise ValueError('Protein must contain 1..510 residues for this demo')
    if not 0.0 < time < 1.0:
        raise ValueError('Continuous diffusion time must lie strictly between 0 and 1')
    features = feature_matrix(seq, protein.get('features', []))
    categories = categories_from_features(features)
    alpha = exponents(experiment, features)
    probability = np.power(time, alpha)
    derivative = alpha * np.power(time, alpha - 1)

    # D/E sample one draw per residue. F also draws one vector [L], but
    # every position in a span reuses the draw at the span's anchor.
    draws = np.random.default_rng(seed).random(len(seq))
    if experiment == 'F':
        anchors = span_anchors(categories, max_span=8)
        shared_draws = draws[anchors]
    else:
        anchors = np.arange(len(seq))
        shared_draws = draws

    masked = shared_draws < probability
    token_ids = [TOKEN_TO_ID['<cls>']] + [TOKEN_TO_ID.get(aa, 3) for aa in seq] + [TOKEN_TO_ID['<eos>']]
    corrupted_ids = token_ids.copy()
    for index in np.flatnonzero(masked):
        corrupted_ids[int(index) + 1] = TOKEN_TO_ID['<mask>']

    labels = [', '.join(FEATURES[i] for i in range(5) if row[i]) or '—' for row in features]
    span_lengths = np.bincount(anchors, minlength=len(seq))[anchors]
    relative_position = (np.arange(len(seq)) - anchors) / np.maximum(span_lengths - 1, 1)
    span_features = np.column_stack([span_lengths / 8, relative_position, (span_lengths > 1).astype(float)])

    frame = pd.DataFrame({
        'Position': np.arange(1, len(seq) + 1),
        'AA': list(seq),
        'Annotations': labels,
        'Priority': categories,
        'α': alpha,
        'q(t)': probability,
        "q'(t)": derivative,
        'Loss weight': derivative / (probability + 1e-3),
        'Span': anchors + 1 if experiment == 'F' else np.full(len(seq), -1),
        'Random u': shared_draws,
        'Masked': masked,
        'Corrupted': np.where(masked, '[MASK]', list(seq)),
    })
    return Result(experiment, float(time), int(seed), seq, frame, token_ids, corrupted_ids, span_features)


def category_probabilities(experiment: str, time: float) -> pd.DataFrame:
    mapping = {
        'D': [('Ordinary / all positions', 1.0)],
        'E': [('Ordinary', 1.0), ('REGION', 1.25), ('DOMAIN', 1.50), ('Functional', 2.0)],
        'F': [('Ordinary', 1.40), ('REGION', 0.95), ('DOMAIN', 0.75), ('Functional', 0.55)],
    }[experiment]
    return pd.DataFrame([{'Category': cat, 'α': alpha, 'Probability': time ** alpha,
                          'Derivative': alpha * time ** (alpha - 1)} for cat, alpha in mapping])


def all_experiments(protein: dict[str, Any], time: float, seed: int) -> pd.DataFrame:
    rows = []
    for letter in 'DEF':
        result = calculate(protein, letter, time, seed)
        rows.append({
            'Experiment': letter,
            'Expected masked': round(result.expected_count, 2),
            'Actually masked': result.masked_count,
            'Masked %': round(100 * result.observed_fraction, 2),
        })
    return pd.DataFrame(rows)


def forward_record(protein: dict[str, Any], result: Result) -> dict[str, Any]:
    """Small export of *calculated* forward process, never denoiser output."""
    df = result.frame
    return {
        'accession': protein['id'],
        'experiment': result.experiment,
        'time': result.time,
        'seed_numpy': result.seed,
        'provenance': 'NumPy CPU translation of original D/E/F forward-corruption equations',
        'annotation_source': protein.get('annotation_note', ''),
        'sequence_length': len(result.sequence),
        'expected_masked_count': result.expected_count,
        'actual_masked_count': result.masked_count,
        'original_token_ids': result.token_ids,
        'corrupted_token_ids': result.corrupted_ids,
        'masked_positions_1based': df.loc[df['Masked'], 'Position'].astype(int).tolist(),
        'probabilities': [round(v, 8) for v in df['q(t)'].tolist()],
        'derivatives': [round(v, 8) for v in df["q'(t)"].tolist()],
        'alpha': df['α'].tolist(),
        'span_anchors_1based': df['Span'].astype(int).tolist() if result.experiment == 'F' else None,
        'not_present': ['model logits', 'trained attention matrices', 'reverse diffusion trajectories'],
    }
