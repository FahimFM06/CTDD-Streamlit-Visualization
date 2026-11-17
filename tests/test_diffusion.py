import json
from pathlib import Path

import numpy as np
import pytest

from modules.diffusion import (
    FEATURES, TOKEN_TO_ID, all_experiments, calculate,
    categories_from_features, exponents, feature_matrix, span_anchors,
)

ROOT = Path(__file__).resolve().parents[1]
PROTEINS = json.loads((ROOT / 'data/proteins.json').read_text())


def test_protein_examples_have_valid_length_and_coordinates():
    assert [len(p['sequence']) for p in PROTEINS] == [148, 260, 497]
    for protein in PROTEINS:
        assert len(protein['sequence']) <= 510
        for f in protein['features']:
            assert 1 <= f['start'] <= f['end'] <= len(protein['sequence'])
            assert f['type'] in FEATURES


def test_esm_special_tokens():
    assert TOKEN_TO_ID['<cls>'] == 0
    assert TOKEN_TO_ID['<pad>'] == 1
    assert TOKEN_TO_ID['<eos>'] == 2
    assert TOKEN_TO_ID['<mask>'] == 32


def test_all_nine_modes_are_deterministic_and_shape_consistent():
    for protein in PROTEINS:
        for exp in 'DEF':
            first = calculate(protein, exp, 0.5, 2026)
            second = calculate(protein, exp, 0.5, 2026)
            assert len(first.frame) == len(protein['sequence'])
            assert len(first.token_ids) == len(protein['sequence']) + 2
            assert first.token_ids[0] == first.corrupted_ids[0] == 0
            assert first.token_ids[-1] == first.corrupted_ids[-1] == 2
            assert np.array_equal(first.frame['Masked'], second.frame['Masked'])
            assert first.masked_count == sum(v == 32 for v in first.corrupted_ids)
            assert (first.frame['q(t)'] > 0).all()
            assert (first.frame['q(t)'] < 1).all()


def test_d_is_uniform_and_independent():
    result = calculate(PROTEINS[0], 'D', 0.37, 3)
    np.testing.assert_allclose(result.frame['q(t)'], 0.37)
    np.testing.assert_allclose(result.frame["q'(t)"], 1.0)
    assert np.array_equal(result.frame['Masked'], result.frame['Random u'] < result.frame['q(t)'])


def test_e_f_priority_and_derivatives():
    proto = {'sequence': 'AAAAA', 'features': [
        {'type': 'REGION', 'start': 2, 'end': 5},
        {'type': 'DOMAIN', 'start': 3, 'end': 5},
        {'type': 'ACT_SITE', 'start': 4, 'end': 4},
    ]}
    f = feature_matrix(proto['sequence'], proto['features'])
    assert categories_from_features(f) == ['NORMAL','REGION','DOMAIN','ACT_SITE','DOMAIN']
    np.testing.assert_allclose(exponents('E', f), [1.,1.25,1.5,2.,1.5])
    np.testing.assert_allclose(exponents('F', f), [1.4, .95,.75,.55,.75])
    for letter in 'EF':
        res = calculate(proto, letter, 0.5, 2)
        alpha = res.frame['α'].to_numpy()
        np.testing.assert_allclose(res.frame['q(t)'], np.power(0.5, alpha))
        np.testing.assert_allclose(res.frame["q'(t)"], alpha * np.power(0.5, alpha-1))


def test_f_category_runs_and_span_lengths():
    cats = ['NORMAL'] * 12 + ['DOMAIN'] * 4 + ['NORMAL'] * 3
    anchors = span_anchors(cats, max_span=8)
    assert anchors.tolist() == [0]*8 + [8]*4 + [12]*4 + [16]*3
    res = calculate({'sequence': 'A'*19, 'features': [
        {'type':'DOMAIN', 'start':13, 'end':16}]}, 'F', .5, 42)
    for anchor in sorted(set(res.frame['Span'])):
        segment = res.frame[res.frame['Span'] == anchor]
        assert len(segment) <= 8
        assert segment['Random u'].nunique() == 1
        assert segment['Masked'].nunique() == 1


def test_invalid_time():
    with pytest.raises(ValueError):
        calculate(PROTEINS[0], 'D', 0, 1)
    with pytest.raises(ValueError):
        calculate(PROTEINS[0], 'D', 1, 1)
