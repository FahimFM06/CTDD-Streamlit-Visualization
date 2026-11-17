"""CTDD Protein Diffusion Explorer v2.1 — single-file deployment repair.
All verified visualization modules and three example proteins are bundled in this
one file to prevent GitHub uploads from breaking Python package imports.
This app demonstrates CPU forward masking; trained-model predictions are not run.
"""
from __future__ import annotations

# DATA: three pinned example proteins (fallback when data/ is absent)
EMBEDDED_PROTEIN_EXAMPLES = '[\n  {\n    "id": "P61626",\n    "name": "Human Lysozyme C",\n    "gene": "LYZ",\n    "organism": "Homo sapiens",\n    "sequence": "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",\n    "ec": [\n      "3.2.1.17"\n    ],\n    "features": [\n      {\n        "type": "DOMAIN",\n        "start": 19,\n        "end": 148,\n        "description": "C-type lysozyme",\n        "source": "UniProt P61626"\n      },\n      {\n        "type": "ACT_SITE",\n        "start": 53,\n        "end": 53,\n        "description": "Catalytic residue",\n        "source": "UniProt P61626"\n      },\n      {\n        "type": "ACT_SITE",\n        "start": 71,\n        "end": 71,\n        "description": "Catalytic residue",\n        "source": "UniProt P61626"\n      }\n    ],\n    "sequence_source": "https://www.uniprot.org/uniprotkb/P61626/entry",\n    "annotation_note": "Selected UniProt annotations, not a complete export."\n  },\n  {\n    "id": "P00918",\n    "name": "Human Carbonic Anhydrase II",\n    "gene": "CA2",\n    "organism": "Homo sapiens",\n    "sequence": "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",\n    "ec": [\n      "4.2.1.1"\n    ],\n    "features": [\n      {\n        "type": "DOMAIN",\n        "start": 3,\n        "end": 259,\n        "description": "Alpha-carbonic anhydrase",\n        "source": "UniProt P00918"\n      },\n      {\n        "type": "ACT_SITE",\n        "start": 64,\n        "end": 64,\n        "description": "Proton donor/acceptor",\n        "source": "UniProt P00918"\n      },\n      {\n        "type": "BINDING",\n        "start": 94,\n        "end": 94,\n        "description": "Zinc coordination (His94)",\n        "source": "Carbonic anhydrase structural literature"\n      },\n      {\n        "type": "BINDING",\n        "start": 96,\n        "end": 96,\n        "description": "Zinc coordination (His96)",\n        "source": "Carbonic anhydrase structural literature"\n      },\n      {\n        "type": "BINDING",\n        "start": 119,\n        "end": 119,\n        "description": "Zinc coordination (His119)",\n        "source": "Carbonic anhydrase structural literature"\n      }\n    ],\n    "sequence_source": "https://www.uniprot.org/uniprotkb/P00918/entry",\n    "annotation_note": "Selected UniProt/domain annotations supplemented with documented zinc-binding residues; not a complete export."\n  },\n  {\n    "id": "P10635",\n    "name": "Human Cytochrome P450 2D6",\n    "gene": "CYP2D6",\n    "organism": "Homo sapiens",\n    "sequence": "MGLEALVPLAVIVAIFLLLVDLMHRRQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQLRRRFGDVFSLQLAWTPVVVLNGLAAVREALVTHGEDTADRPPVPITQILGFGPRSQGVFLARYGPAWREQRRFSVSTLRNLGLGKKSLEQWVTEEAACLCAAFANHSGRPFRPNGLLDKAVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGLKEESGFLREVLNAVPVLLHIPALAGKVLRFQKAFLTQLDELLTEHRMTWDPAQPPRDLTEAFLAEMEKAKGNPESSFNDENLRIVVADLFSAGMVTTSTTLAWGLLLMILHPDVQRRVQQEIDDVIGQVRRPEMGDQAHMPYTTAVIHEVQRFGDIVPLGVTHMTSRDIEVQGFRIPKGTTLITNLSSVLKDEAVWEKPFRFHPEHFLDAQGHFVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVPTGQPRPSHHGVFAFLVSPSPYELCAVPR",\n    "ec": [],\n    "features": [\n      {\n        "type": "BINDING",\n        "start": 443,\n        "end": 443,\n        "description": "Heme iron coordinating Cys443",\n        "source": "CYP2D6 structural literature"\n      }\n    ],\n    "sequence_source": "https://www.uniprot.org/uniprotkb/P10635/entry",\n    "annotation_note": "Selected literature-supported Cys443 heme-binding residue only; use live UniProt to view more feature annotations."\n  }\n]\n'


# ----- BUILT-IN modules/diffusion.py -----


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


# ----- BUILT-IN modules/tokens.py -----

import pandas as pd


def token_table(result: Result) -> pd.DataFrame:
    """Include every residue, plus the CLS and EOS token, in model order."""
    n = len(result.sequence)
    if len(result.token_ids) != n + 2 or len(result.corrupted_ids) != n + 2:
        raise ValueError('Incomplete token sequence')
    rows = [{
        'Model position (0-based)': 0,
        'Protein residue (1-based)': '—',
        'Token': '<cls>', 'Original token ID': TOKEN_TO_ID['<cls>'],
        'Corrupted token': '<cls>', 'Corrupted token ID': TOKEN_TO_ID['<cls>'],
        'Biological annotation': 'Special token (not maskable)',
        'Masked?': False,
    }]
    for i, rec in result.frame.iterrows():
        masked = bool(rec['Masked'])
        rows.append({
            'Model position (0-based)': i + 1,
            'Protein residue (1-based)': i + 1,
            'Token': rec['AA'],
            'Original token ID': result.token_ids[i + 1],
            'Corrupted token': '<mask>' if masked else rec['AA'],
            'Corrupted token ID': result.corrupted_ids[i + 1],
            'Biological annotation': rec['Annotations'],
            'Masked?': masked,
        })
    rows.append({
        'Model position (0-based)': n + 1,
        'Protein residue (1-based)': '—',
        'Token': '<eos>', 'Original token ID': TOKEN_TO_ID['<eos>'],
        'Corrupted token': '<eos>', 'Corrupted token ID': TOKEN_TO_ID['<eos>'],
        'Biological annotation': 'Special token (not maskable)',
        'Masked?': False,
    })
    return pd.DataFrame(rows)


def vocabulary_table() -> pd.DataFrame:
    return pd.DataFrame({'Token ID': range(len(ESM_VOCAB)), 'ESM2 token': list(ESM_VOCAB)})


# ----- BUILT-IN modules/uniprot.py -----
import copy
import requests

FEATURE_TYPES = {
    'Active site': 'ACT_SITE',
    'Binding site': 'BINDING',
    'Motif': 'MOTIF',
    'Domain': 'DOMAIN',
    'Region': 'REGION',
}


def fetch_protein(local: dict, timeout: int = 10) -> dict:
    """Get sequence and compatible residue features directly from UniProtKB.

    Raises on a network/schema problem; callers should fall back to local data.
    """
    accession = local['id']
    response = requests.get(
        f'https://rest.uniprot.org/uniprotkb/{accession}.json',
        headers={'Accept': 'application/json', 'User-Agent': 'CTDD-Streamlit-Explorer/1.0'},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    seq = payload['sequence']['value']
    if not seq or len(seq) > 510:
        raise ValueError('UniProt sequence is missing or too long for this demo')
    if seq != local['sequence']:
        raise ValueError('Online UniProt sequence differs from pinned sequence; refusing to mix coordinates')
    result = copy.deepcopy(local)
    features = []
    for item in payload.get('features', []):
        kind = FEATURE_TYPES.get(item.get('type'))
        if kind is None:
            continue
        start = item.get('location', {}).get('start', {}).get('value')
        end = item.get('location', {}).get('end', {}).get('value')
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not (1 <= start <= end <= len(seq)):
            continue
        features.append({
            'type': kind,
            'start': start,
            'end': end,
            'description': item.get('description', ''),
            'source': f'Live UniProtKB {accession}',
        })
    result['features'] = features
    result['annotation_note'] = 'Live UniProtKB feature extraction (ACT_SITE, BINDING, MOTIF, DOMAIN, REGION only)'
    crossrefs = payload.get('uniProtKBCrossReferences', [])
    result['go_terms'] = sorted({x['id'] for x in crossrefs if x.get('database') == 'GO' and x.get('id')})
    return result


# ----- BUILT-IN modules/visuals.py -----

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

INK = '#1E4264'
BLUE = '#168FD6'
PALE = '#E9F6FF'
COLORS = {
    'NORMAL': '#AAC5D7',
    'REGION': '#3CBDB1',
    'DOMAIN': '#619BEE',
    'MOTIF': '#B79BF2',
    'BINDING': '#F2AE70',
    'ACT_SITE': '#F48FAA',
    'MASK': '#2478D0',
}


def finish(fig: go.Figure, height: int = 320, ytitle: str = '', xtitle: str = '') -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=14, r=16, t=32, b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FFFFFF',
        font=dict(family='Inter, Arial, sans-serif', color=INK, size=12),
        legend=dict(orientation='h', y=1.19, x=0),
        hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='#EBF3FA', zeroline=False, title=xtitle),
        yaxis=dict(showgrid=True, gridcolor='#EBF3FA', zeroline=False, title=ytitle),
    )
    return fig


def time_curves(exp: str, t: float) -> go.Figure:
    xx = np.linspace(.001, .999, 240)
    rules = {'D': [('All residues', 1., BLUE)],
             'E': [('Ordinary', 1., COLORS['NORMAL']), ('REGION', 1.25, COLORS['REGION']),
                   ('DOMAIN', 1.50, COLORS['DOMAIN']), ('Functional', 2.0, COLORS['ACT_SITE'])],
             'F': [('Ordinary', 1.40, COLORS['NORMAL']), ('REGION', .95, COLORS['REGION']),
                   ('DOMAIN', .75, COLORS['DOMAIN']), ('Functional', .55, COLORS['ACT_SITE'])]}[exp]
    fig = go.Figure()
    for name, alpha, color in rules:
        fig.add_trace(go.Scatter(x=xx, y=xx**alpha, mode='lines', name=f'{name} (α={alpha:g})', line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(x=[t], y=[t**alpha], mode='markers', marker=dict(size=11, color=color, line=dict(color='white', width=2)), showlegend=False,
                                 hovertemplate=f'{name}<br>t={t:.2f}<br>q={t**alpha:.4f}<extra></extra>'))
    fig.add_vline(x=t, line_dash='dash', line_color='#517DAD', line_width=1)
    fig.update_xaxes(range=[0,1]); fig.update_yaxes(range=[0,1])
    return finish(fig, 345, 'Probability of masking q(t)', 'Continuous time t')


def residue_probabilities(result: Result) -> go.Figure:
    df = result.frame
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Position'], y=df['q(t)'], mode='lines', line=dict(color='#B2D6F6', width=2), name='Masking probability', hoverinfo='skip'))
    for cat in ('NORMAL','REGION','DOMAIN','MOTIF','BINDING','ACT_SITE'):
        subset = df[df['Priority'] == cat]
        if subset.empty: continue
        fig.add_trace(go.Scattergl(x=subset['Position'], y=subset['q(t)'], mode='markers', name=cat.replace('_',' ').title(),
                                   marker=dict(color=COLORS[cat], size=7), customdata=np.stack([subset['AA'].values,subset['α'].values,subset['Annotations'].values],axis=1),
                                   hovertemplate='Residue %{x}: %{customdata[0]}<br>q(t)=%{y:.4f}<br>α=%{customdata[1]}<br>%{customdata[2]}<extra></extra>'))
    fig.update_yaxes(range=[0,1.04])
    return finish(fig, 345, 'qᵢ(t)', 'Residue position (all positions included)')


def random_vs_probability(result: Result, focus: int, radius: int = 10) -> go.Figure:
    df = result.frame.iloc[max(0,focus-1-radius):min(len(result.frame),focus+radius)].copy()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['Position'], y=df['q(t)'], name='Masking threshold q', mode='lines+markers',
                             line=dict(color='#089EC2',width=3),marker=dict(size=7),hovertemplate='Position %{x}<br>q=%{y:.4f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=df['Position'], y=df['Random u'],name='Random value u',mode='markers',
                             marker=dict(size=13,color=[COLORS['MASK'] if b else '#E9A26A' for b in df['Masked']],line=dict(width=1,color='white')),
                             hovertemplate='Position %{x}<br>u=%{y:.4f}<extra></extra>'))
    if result.experiment == 'F':
        for anchor, g in df.groupby('Span',sort=True):
            a,b=int(g['Position'].min()),int(g['Position'].max())
            fig.add_vrect(x0=a-0.45, x1=b+.45, fillcolor='#B8D9F9', opacity=.14, line_width=0)
    fig.add_vline(x=focus,line_dash='dash',line_color='#A586DB')
    fig.update_yaxes(range=[0,1.05])
    fig.update_xaxes(dtick=max(1, int(len(df)//16)))
    return finish(fig, 330, 'q threshold vs random draw', 'Residue positions near the selected residue')


def comparison_figure(stats) -> go.Figure:
    fig=go.Figure()
    fig.add_trace(go.Bar(x=stats['Experiment'],y=stats['Expected masked'],name='Expected count',marker_color='#99CDEA',text=stats['Expected masked'],textposition='outside'))
    fig.add_trace(go.Bar(x=stats['Experiment'],y=stats['Actually masked'],name='Randomly sampled count',marker_color='#368ED1',text=stats['Actually masked'],textposition='outside'))
    fig.update_layout(barmode='group')
    return finish(fig,360,'Number of residues','Experiment')


def toy_attention(exp: str) -> go.Figure:
    # These are small, handmade pedagogical matrices—not checkpoint outputs.
    a=np.array([[.70,.20,.10],[.15,.70,.15],[.25,.30,.45]])
    b=np.array([[.15,.60,.25],[.50,.20,.30],[.30,.40,.30]])
    m=a-0.5*b if exp=='F' else a
    title='A₁ − 0.5 × A₂ (toy example)' if exp=='F' else 'Ordinary attention (toy example)'
    zmax=1. if exp!='F' else 1.
    fig=go.Figure(go.Heatmap(z=m, x=['A','G','L'], y=['A','G','L'], zmin=-.3 if exp=='F' else 0,zmax=zmax,
                             colorscale='Blues' if exp!='F' else 'RdBu', reversescale=(exp=='F'),
                             text=[[f'{v:.2f}' for v in row] for row in m],texttemplate='%{text}',
                             hovertemplate='Query %{y} → Key %{x}<br>Value %{z:.2f}<extra></extra>'))
    fig.update_layout(title=dict(text=f'{title} — illustrative only',font=dict(size=13)))
    fig.update_yaxes(autorange='reversed')
    return finish(fig,300,'Query token','Key token')


def toy_logits() -> tuple[go.Figure,dict[str,float]]:
    tokens=['A','G','L','V']
    logits=np.array([2.,1.,0.,-0.5])
    scores=30*np.tanh(logits/30)
    probs=np.exp(scores-np.max(scores));probs=probs/probs.sum()
    fig=go.Figure(go.Bar(x=tokens,y=probs,marker_color=['#258CD3','#7BB8D9','#AEC9E5','#CBDCEC'],
                         text=[f'{p:.1%}' for p in probs],textposition='outside',hovertemplate='%{x}: %{y:.4f}<extra></extra>'))
    fig.update_yaxes(range=[0,max(probs)*1.25])
    return finish(fig,280,'Example probability (not trained)','Toy candidate tokens'),dict(zip(tokens,probs.tolist()))

# ----- ORIGINAL STREAMLIT INTERFACE -----

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CTDD | Visual Protein Diffusion', page_icon='🧬', layout='wide')

st.markdown('''
<style>
:root {--blue:#1588CE; --ink:#244762; --line:#D9EAF6; --soft:#EFF8FF;}
.stApp {background:linear-gradient(165deg,#F5FBFF 5%,#FFFFFF 65%);color:#244762;}
.block-container {max-width:1430px;padding-top:1.4rem;padding-bottom:5rem;}
section[data-testid="stSidebar"] {background:#EAF6FF;border-right:1px solid #DBECF8;}
h1,h2,h3,h4 {color:#1E4769!important; letter-spacing:-.025em;}
h1 {font-size:2.2rem!important;}
[data-testid="stMetric"] {background:white;padding:1rem;border:1px solid #DDECF5;border-radius:14px;box-shadow:0 2px 12px #78C3EF13;}
[data-testid="stMetricLabel"] {color:#537894;}
[data-testid="stTabs"] button {font-weight:700;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:7px;flex-wrap:wrap;}
[data-testid="stTabs"] [data-baseweb="tab"] {background:#EAF5FD;border-radius:11px 11px 0 0;padding:10px 16px;}
[data-testid="stTabs"] [aria-selected="true"] {background:#D8EEFC;}
div[data-testid="stPlotlyChart"] {border:1px solid #E1EDF5;border-radius:16px;background:white;padding:4px;}
.hero {background:linear-gradient(112deg,#D9F2FF 0%,#EBF8FE 47%,#F7F0FF 100%);border:1px solid #C9E7F7;border-radius:20px;padding:25px 29px;margin-bottom:18px;}
.hero-tag {color:#287FAC;font-size:.76rem;font-weight:850;letter-spacing:.16em;text-transform:uppercase;}
.hero h1 {margin:.45rem 0 .55rem!important;}
.hero p {font-size:1.03rem;color:#486B87;max-width:920px;line-height:1.65;margin:.2rem 0;}
.section-title {font-size:1.36rem;font-weight:800;margin-top:2.3rem;margin-bottom:.4rem;color:#1B4969;}
.section-lead {color:#52748C;margin-bottom:1.2rem;line-height:1.65;}
.step-top {display:flex;align-items:center;gap:13px;background:#F0F8FF;border:1px solid #D7EDF9;border-radius:14px;padding:14px 17px;margin:24px 0 13px;}
.step-num {display:flex;align-items:center;justify-content:center;flex:0 0 auto;width:39px;height:39px;background:#168DD2;border-radius:12px;font-weight:850;color:white;font-size:1.1rem;}
.step-top h3 {margin:0!important;font-size:1.14rem!important;}
.step-top p {margin:2px 0 0;color:#58778F;font-size:.93rem;line-height:1.4;}
.explain {background:#FFFFFF;border:1px solid #DDEAF4;border-radius:13px;padding:15px 17px;min-height:156px;box-shadow:0 3px 12px #77BCE90B;margin:10px 0;}
.explain .tag {display:block;font-size:.73rem;font-weight:850;text-transform:uppercase;letter-spacing:.08em;color:#1487B6;margin-bottom:9px;}
.explain .body {font-size:.94rem;line-height:1.6;color:#31536E;}
.explain.example {background:#F1F9FF;border-color:#C9E7FA;}
.explain.method {background:#F9F7FF;border-color:#E2D9FB;}
.flow {display:flex;align-items:stretch;gap:6px;flex-wrap:wrap;margin:12px 0 18px;}
.flowitem {flex:1 1 114px;background:#F0F8FE;border:1px solid #CEE5F5;border-radius:12px;text-align:center;padding:14px 9px;font-weight:750;color:#27628E;font-size:.87rem;}
.flowarrow {align-self:center;color:#6CB0D6;font-size:1.2rem;font-weight:850;}
.diagram-caption {font-size:.84rem;color:#5A7E96;margin-top:6px;}
.seq-wrap {background:white;border:1px solid #DAEAF5;border-radius:14px;padding:12px;overflow-x:auto;}
.seq-row {display:flex;align-items:center;gap:2px;margin:4px 0;min-width:0;}
.seq-index {font-size:.69rem;font-family:ui-monospace,monospace;min-width:67px;color:#5C7890;}
.residue {font-family:ui-monospace,SFMono-Regular,Consolas,monospace;display:inline-flex;align-items:center;justify-content:center;width:20px;height:27px;border:1px solid #D4E5EF;border-radius:5px;font-size:.79rem;font-weight:800;color:#365B75;}
.residue.masked {background:#2E86D4;border-color:#2E86D4;color:white;}
.residue.normal {background:#E7F0F6;}
.residue.region {background:#C5F2EA;}
.residue.domain {background:#CCE1FC;}
.residue.motif {background:#E4DAFA;}
.residue.binding {background:#FCE2CD;}
.residue.act_site {background:#FBD6E3;}
.residue.spotlight {outline:2px solid #E09E28;outline-offset:1px;}
.pill {display:inline-block;padding:5px 10px;border-radius:24px;background:#E7F4FF;color:#24658E;border:1px solid #C8E7F7;font-weight:700;font-size:.82rem;margin:2px;}
.note {border-left:4px solid #50A9DB;background:#EAF6FE;padding:12px 16px;border-radius:8px;font-size:.92rem;color:#3B6380;margin:13px 0;line-height:1.7;}
.note-purple {border-left-color:#A78CD9;background:#F5F1FF;}
.emphasis {color:#1475B8;font-weight:800;}
.layer-grid {display:grid;grid-template-columns:repeat(10,minmax(28px,1fr));gap:5px;margin:11px 0;}
.layer {background:#D5E9F8;border:1px solid #BCDAED;text-align:center;padding:9px 2px;border-radius:7px;font-size:.76rem;font-weight:750;color:#2E678C;}
.layer.diff {background:#DCCDF6;border-color:#C1AFF2;color:#6542A4;}
.small-caption {font-size:.82rem;color:#607F94;line-height:1.55;}
@media(max-width:780px) {.residue{width:17px;font-size:.69rem}.seq-index{min-width:53px;font-size:.6rem}.hero{padding:18px}.flowitem{flex-basis:100px}}
</style>
''', unsafe_allow_html=True)


def safe(s: object) -> str:
    return html.escape(str(s), quote=True)


@st.cache_data
def local_proteins() -> list[dict]:
    protein_path = ROOT / 'data' / 'proteins.json'
    if protein_path.is_file():
        return json.loads(protein_path.read_text(encoding='utf-8'))
    return json.loads(EMBEDDED_PROTEIN_EXAMPLES)


@st.cache_data(ttl=86400, show_spinner=False)
def online_protein(protein: dict) -> dict:
    return fetch_protein(protein)


def step(i: int, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="step-top"><div class="step-num">{i:02d}</div><div><h3>{safe(title)}</h3><p>{safe(subtitle)}</p></div></div>',unsafe_allow_html=True)


def teaching(what: str, method: str, example: str, source: str) -> None:
    specs=[('In simple words',what,''),('Method used',method,'method'),('Small example',example,'example')]
    cs=st.columns(3, gap='small')
    for c,(label,body,cl) in zip(cs,specs):
        with c:
            st.markdown(f'<div class="explain {cl}"><span class="tag">{safe(label)}</span><div class="body">{safe(body)}</div></div>',unsafe_allow_html=True)
    st.caption(f'📁 Original implementation: `{source}`' if source else '📁 Educational explanation; numerical examples are not model predictions.')


def flow(items: list[str]) -> None:
    markup=[]
    for idx,item in enumerate(items):
        if idx:markup.append('<div class="flowarrow">→</div>')
        markup.append(f'<div class="flowitem">{safe(item)}</div>')
    st.markdown('<div class="flow">'+''.join(markup)+'</div>',unsafe_allow_html=True)


def sequence_grid(sequence: str, frame: pd.DataFrame, highlight: int, masked: bool=False, line: int=50) -> None:
    """Show *all* residues. No 5-, 14-, 70-, or 120-residue preview limits."""
    rows=[]
    for start in range(0,len(sequence),line):
        parts=[f'<span class="seq-index">{start+1}–{min(start+line,len(sequence))}</span>']
        for i in range(start,min(start+line,len(sequence))):
            rec=frame.iloc[i]
            is_mask=bool(rec['Masked']) and masked
            category=str(rec['Priority']).lower()
            cl='masked' if is_mask else ('normal' if category=='normal' else category)
            selected=' spotlight' if i+1==highlight else ''
            letter='■' if is_mask else sequence[i]
            tooltip=f'Position {i+1}, original {sequence[i]}, category {rec["Priority"]}, q={rec["q(t)"]:.4f}, masked={bool(rec["Masked"])}'
            parts.append(f'<span class="residue {cl}{selected}" title="{safe(tooltip)}">{safe(letter)}</span>')
        rows.append('<div class="seq-row">'+''.join(parts)+'</div>')
    st.markdown('<div class="seq-wrap">'+''.join(rows)+'</div>',unsafe_allow_html=True)
    st.caption(f'All {len(sequence)} amino acids are displayed, 50 per row. Blue ■ means MASK token (ID 32). Gold outline = selected position {highlight}.')


def labeled_note(text: str, purple: bool=False) -> None:
    st.markdown(f'<div class="note{" note-purple" if purple else ""}">{safe(text)}</div>',unsafe_allow_html=True)


PROTEINS=local_proteins()
with st.sidebar:
    st.markdown('### 🧬 Choose your example')
    accession=st.selectbox('Protein', [x['id'] for x in PROTEINS], format_func=lambda k:next(f'{x["name"]} · {k}' for x in PROTEINS if x['id']==k))
    experiment=st.radio('Experiment',('D','E','F'),horizontal=True,help='D: uniform; E: delayed functional masking; F: function-first span masking.')
    time=st.slider('Continuous diffusion time t',0.01,0.99,0.50,0.01)
    seed=st.number_input('Random seed (reproducible)',min_value=0,max_value=2**31-1,value=2026,step=1)
    load_online=st.checkbox('Refresh full compatible annotations from UniProt',value=False,
                            help='Uses UniProtKB REST when accessible. Offline examples include selected annotations only.')
    st.divider()
    st.markdown('**Guide**')
    st.caption('Start with Forward diffusion. Change t and follow one amino acid through all six stages. Then open Denoiser and Compare.')
    st.caption('🟦 Actual CPU masking calculation  ·  🟪 Architecture or toy illustration')
    st.caption('No checkpoint, training, or GPU needed to run this app.')

protein=next(x for x in PROTEINS if x['id']==accession)
annotation_source='Selected bundled annotations (not the complete UniProt feature set)'
if load_online:
    try:
        protein=online_protein(protein)
        annotation_source='Live UniProtKB compatible feature annotations'
    except Exception as exc:
        st.warning(f'UniProt request unavailable. Using bundled annotation subset. Detail: {exc}')

result=calculate(protein,experiment,float(time),int(seed))
frame=result.frame
length=len(frame)
functional_positions=frame.index[frame['Priority'].isin(('ACT_SITE','BINDING','MOTIF'))].tolist()
initial_focus=(functional_positions[0]+1) if functional_positions else max(1,length//3)

st.markdown('''<div class="hero"><div class="hero-tag">CTDD · INTERACTIVE PROTEIN DIFFUSION</div>
<h1>See every step. Understand every choice.</h1>
<p>Pick a real protein, change the noise level, and see how Experiments D, E and F differ. Each drawing is followed by a plain-English explanation, the method used in the source code, and a worked example.</p></div>''',unsafe_allow_html=True)
cols=st.columns(4)
cols[0].metric('Protein length',f'{length} amino acids')
cols[1].metric('Current time t',f'{time:.2f}')
cols[2].metric('Expected masks',f'{result.expected_count:.1f}')
cols[3].metric('Actually masked',f'{result.masked_count} / {length}')
st.caption(f'**{protein["name"]}** ({accession})  ·  Experiment **{experiment}**  ·  {annotation_source}')

st.markdown('<div class="note"><b>What is real here?</b> Every residue, its token ID, masking probability, random mask decision, and corrupted token are calculated using the D/E/F source equations. <b>What is explanatory?</b> The 150M-parameter Transformer is not loaded on Streamlit Cloud; attention maps, example logits, and reverse-generation drawings are explicitly illustrative.</div>',unsafe_allow_html=True)

focus=st.slider('🔎 Follow one amino acid position through the calculations',1,length,initial_focus,key=f'focus_{accession}',help='The gold-outline cell and the worked examples will use this residue.')
r=frame.iloc[focus-1]
st.markdown(f'<span class="pill">Residue {focus}: {safe(r["AA"])}</span><span class="pill">Annotation priority: {safe(r["Priority"])}</span><span class="pill">Masking q(t): {r["q(t)"]:.3f}</span><span class="pill">Random u: {r["Random u"]:.3f}</span><span class="pill">{"MASKED" if bool(r["Masked"]) else "KEPT"}</span>',unsafe_allow_html=True)

intro,forward,denoiser,comparison,methods=st.tabs(['🏠 Start here','🔵 Forward diffusion · 6 steps','🟣 Denoising · 6 steps','📊 Compare D / E / F','📚 All methods & downloads'])

with intro:
    st.markdown('<div class="section-title">The story in one picture</div>',unsafe_allow_html=True)
    st.write('A protein is a chain of amino acids (letters). Diffusion temporarily hides some letters. The learned Transformer is trained to guess the original letters from the masked input and context.')
    flow(['1. Real protein','2. Token IDs','3. Choose time t','4. Sample masks','5. Add context','6. Transformer predicts'])
    st.markdown('#### Before masking — full protein')
    sequence_grid(protein['sequence'],frame,focus,masked=False)
    st.markdown('#### After masking — full protein')
    sequence_grid(protein['sequence'],frame,focus,masked=True)
    teaching('Each letter is an amino acid. The blue squares are letters hidden from the model.',
             f'Experiment {experiment}: '+('independent mask probability q=t.' if experiment=='D' else ('independent q=t^α; annotated functional residues use α=2.' if experiment=='E' else 'q=t^α plus shared random numbers within spans of at most 8 residues.')),
             f'At position {focus}, {r["AA"]} has q={r["q(t)"]:.3f}. The random draw u={r["Random u"]:.3f}; it is {"hidden" if r["Masked"] else "kept"}.',
             {'D':'Experiment_D/scripts/04_bio_corruptor.py; 37_experiment_d_ddp.py',
              'E':'Experiment_E/scripts/43_experiment_e_delayed_corruptor.py',
              'F':'Experiment_F/scripts/52_experiment_f_corruptor.py'}[experiment])
    st.markdown('#### Why three experiments?')
    st.dataframe(pd.DataFrame([
        ['D','Uniform masking','Every eligible residue has the same chance t.','Biological features are provided to the denoiser.'],
        ['E','Function-protecting','Functional residues are less likely to be masked at the same t.','GO/EC, time and priority context are added.'],
        ['F','Function-first spans','Functional positions have a higher mask chance; up to 8 same-category positions share a draw.','Span features and Differential Attention are added.'],
    ],columns=['Experiment','Idea','What changes in forward diffusion','What changes in denoiser']),hide_index=True,use_container_width=True)
    labeled_note('Read Forward diffusion from Step 1 to Step 6 to see the actual calculations. Denoising then explains which learned operations occur; it does not pretend to run the trained checkpoint.')

with forward:
    st.markdown('<div class="section-title">Forward diffusion: start from a clean protein, then hide residues</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-lead">All six steps are visible below. No residue is omitted from the sequence or token tables.</div>',unsafe_allow_html=True)
    flow(['Clean sequence','Tokenize','Sample time','Calculate q(t)','Draw random masks','Replace by MASK'])

    step(1,'Tokenization — convert every amino acid into a token ID','The model reads numbers, not letters. We show the entire protein, including CLS and EOS.')
    sequence_grid(protein['sequence'],frame,focus,masked=False)
    st.markdown('**Complete tokenization table — all residues, plus two special tokens**')
    tokens=token_table(result)
    st.dataframe(tokens[['Model position (0-based)','Protein residue (1-based)','Token','Original token ID','Biological annotation']],
                 hide_index=True,use_container_width=True,height=335)
    st.caption(f'Total token positions: {length+2} = 1 CLS + {length} amino acids + 1 EOS. No PAD tokens are needed for a single unpadded example. Scroll or download to view any row.')
    teaching('Each amino acid gets an ESM2 vocabulary number. CLS marks the beginning, and EOS marks the end. Both stay unmasked.',
             'ESM2 token vocabulary with 33 token IDs; CLS=0, PAD=1, EOS=2 and MASK=32. The residue indices start at 1, while token positions start at 0.',
             f'Position {focus} has letter {r["AA"]}, token ID {result.token_ids[focus]}. Its model position is {focus} (after CLS).',
             'Experiment_A/dsm_dataset.py and Experiment_A/esm2_embeddings.py')
    st.download_button('⬇ Download ALL tokens as CSV',tokens.to_csv(index=False).encode(),file_name=f'{accession}_{experiment}_all_tokens.csv',mime='text/csv',key='tokens_full_download')

    step(2,'Continuous-time sampling — choose the noise level','Time t is a real number between 0 and 1, not an integer step.')
    st.plotly_chart(time_curves(experiment,time),use_container_width=True)
    st.latex(r't\in(0,1)\qquad\text{and}\qquad q_i(t)=t^{\alpha_i}')
    training_rule='D: t is drawn from Uniform(0,1) then clamped to [10⁻⁵, 1−10⁻⁵].' if experiment=='D' else 'E/F: t = 0.001 + 0.998 × U(0,1), independently for each protein in a training batch.'
    teaching('The time slider tells the forward process how much of the protein to hide. Small t = little noise. Large t = more noise.',
             'Continuous-time discrete-state absorbing diffusion. '+training_rule+' The dashboard uses a fixed slider value instead of drawing a fresh t on every interaction.',
             f'At t={time:.2f}, an ordinary residue in D/E would have mask probability {time:.2f}; the selected position in {experiment} has probability {r["q(t)"]:.3f}.',
             'D: 04_bio_corruptor.py; E: 43_experiment_e_delayed_corruptor.py; F: 52_experiment_f_corruptor.py')

    step(3,'Masking probabilities — decide how likely each residue is to disappear','E and F use biological annotations to change the probability at each position.')
    st.plotly_chart(residue_probabilities(result),use_container_width=True)
    rules=category_probabilities(experiment,time)
    st.markdown('**Exact category exponent table at the selected time**')
    st.dataframe(rules.rename(columns={'Probability':'q(t)','Derivative':"q′(t)"}).style.format({'q(t)':'{:.4f}',"q′(t)":'{:.4f}'}),hide_index=True,use_container_width=True)
    st.latex(r'q_i(t)=t^{\alpha_i},\qquad q_i\!\,\prime(t)=\alpha_i t^{\alpha_i-1}')
    teaching('Each letter gets its own masking chance. In D every position has the same chance. In E, functional sites are protected. In F, they are hidden earlier.',
             'Annotation priority: ACT_SITE > BINDING > MOTIF > DOMAIN > REGION > ordinary. The top-priority applicable feature selects exponent α. The 5-channel condition map is still preserved.',
             f'Position {focus} is {r["Priority"]}: α={r["α"]:.2f}. Hence q={time:.2f}^{r["α"]:.2f}={r["q(t)"]:.4f}. Its derivative is {r["q\u0027(t)"]:.4f}.',
             'E: 43_experiment_e_delayed_corruptor.py; F: 52_experiment_f_corruptor.py; D: 04_bio_corruptor.py')

    step(4,'Random masking — compare a random value with the probability','A residue is hidden if its random draw is smaller than q(t). Experiment F shares draws inside spans.')
    st.plotly_chart(random_vs_probability(result,focus),use_container_width=True)
    st.latex(r'M_i=\mathbf{1}[u_i<q_i(t)]\quad\text{(D/E)}')
    if experiment=='F':
        st.latex(r'M_i=\mathbf{1}[u_{\mathrm{span}(i)}<q_i(t)]\quad\text{(F)}')
        st.write(f'Span groups in this protein: **{int(frame["Span"].nunique())}**. Each group has at most eight consecutive residues of the same priority category.')
    decision='MASKED' if bool(r['Masked']) else 'KEPT'
    shared=f' It shares its draw with other residues in the same span starting at {int(r["Span"])}.' if experiment=='F' else ' Its random draw is independent of other positions.'
    teaching('Imagine flipping a different biased coin for each letter. In F, a small group of neighbouring letters shares one coin flip.',
             'Bernoulli masking: u < q. D/E generate one independent uniform random value per eligible residue; F groups equal-priority contiguous residues into spans of at most 8 and uses the anchor draw for the whole group.',
             f'Position {focus}: u={r["Random u"]:.4f}, q={r["q(t)"]:.4f}; since u {"<" if r["Masked"] else "≥"} q, the answer is {decision}.{shared}',
             'D: 04_bio_corruptor.py; E: 43_experiment_e_delayed_corruptor.py; F: 52_experiment_f_corruptor.py')
    st.markdown('**Every masking decision — entire protein**')
    st.dataframe(frame[['Position','AA','Annotations','Priority','α','q(t)','Random u','Span','Masked']],hide_index=True,use_container_width=True,height=290)
    st.caption('For D/E, Span = -1 (no correlated spans). NumPy random numbers follow the original sampling distribution but are not bit-for-bit identical to PyTorch random draws.')

    step(5,'Mask replacement — turn chosen letters into MASK tokens','Only eligible amino acid tokens can be replaced. CLS and EOS remain unchanged.')
    st.markdown('**Before: full original sequence**')
    sequence_grid(protein['sequence'],frame,focus,masked=False)
    st.markdown('**After: full corrupted sequence**')
    sequence_grid(protein['sequence'],frame,focus,masked=True)
    st.latex(r'\widetilde{x}_i=\begin{cases}\texttt{MASK},&M_i=1\\x_i,&M_i=0\end{cases}')
    teaching('A masked residue no longer tells the model which amino acid was originally there. The original answer is still stored as the training target.',
             'Absorbing [MASK] corruption: replace selected residue token IDs with ID 32. Non-masked IDs stay unchanged; CLS=0 and EOS=2 are protected.',
             f'Position {focus}: {r["AA"]} (ID {result.token_ids[focus]}) → '+('[MASK] (ID 32).' if r['Masked'] else f'{r["AA"]} (same ID, kept).'),
             'D: 04_bio_corruptor.py; E: 43_experiment_e_delayed_corruptor.py; F: 52_experiment_f_corruptor.py')

    step(6,'Final tensors — what goes into the denoising Transformer','These are the actual computed token IDs and the expected interface shapes.')
    flow(['Corrupted IDs [1,L+2]','Attention mask','5 biological channels','Time / extra context','30-layer DSM150'])
    tensor_rows=[
        ('input_ids',f'[1,{length+2}]','Full corrupted token IDs; CLS and EOS included'),
        ('attention_mask',f'[1,{length+2}]','All ones for this single unpadded sequence'),
        ('conditions',f'[1,{length+2},5]','ACT_SITE, BINDING, MOTIF, DOMAIN, REGION'),
        ('residue_mask',f'[1,{length+2}]','True for amino acids; false for CLS and EOS'),
        ('condition_available','[1]','1 when local annotations are supplied (illustrated as on)'),
    ]
    if experiment in ('E','F'):
        tensor_rows += [('diffusion_time','[1]',f't={time:.2f}'),
                        ('masking_exponents',f'[1,{length+2}]','α at each residue; special positions handled by training collator'),
                        ('go_ids / ec_ids','Variable-length','Requires original training label vocabularies for checkpoint-compatible indices')]
    if experiment=='F':
        tensor_rows += [('span_features',f'[1,{length+2},3]','Length/8, relative position, multiple-residue flag'),
                        ('observed fraction','[1,1]',f'{result.observed_fraction:.4f}, internally computed from input IDs')]
    st.dataframe(pd.DataFrame(tensor_rows,columns=['Model input','Shape','Meaning']),hide_index=True,use_container_width=True)
    st.markdown('**Full original and corrupted token IDs (including both special tokens)**')
    st.dataframe(tokens,hide_index=True,use_container_width=True,height=330)
    teaching('The denoiser receives the masked IDs plus helpful information about the protein. During training it tries to predict the original letters.',
             'ESM2 embedding → additive biological/context projections → 30 Transformer layers → language-model head. E adds GO/EC and time/priority context; F adds span and observed-mask features.',
             f'This protein has {length} residues and {length+2} model positions. {result.masked_count} are MASK=32. Its expected masked count is {result.expected_count:.2f}.',
             'D: 33_experiment_d_model.py; E: 44_experiment_e_context_model.py; F: 54_experiment_f_context_model.py')
    st.download_button('⬇ Full masking table (CSV)',frame.to_csv(index=False).encode(),file_name=f'{accession}_{experiment}_masking_all.csv',mime='text/csv',key='mask_full_csv')
    st.download_button('⬇ Full forward record (JSON)',json.dumps(forward_record(protein,result),indent=2).encode(),file_name=f'{accession}_{experiment}_t{time:.2f}_seed{seed}.json',mime='application/json',key='mask_full_json')

with denoiser:
    st.markdown('<div class="section-title">Denoising: how the Transformer would predict missing letters</div>',unsafe_allow_html=True)
    labeled_note('Important distinction: the diagrams and small calculations below teach the model architecture. They are not outputs from your 150M-parameter trained checkpoints. We never invent learned attention, logits or reverse trajectories.',purple=True)
    flow(['Masked protein','Token + biological embeddings','30 Transformer layers','33 scores per position','Loss during training','Sampler only if implemented'])

    step(1,'Input — give the network the masked sequence','The Transformer sees masked letters, not the original answer at masked positions.')
    sequence_grid(protein['sequence'],frame,focus,masked=True)
    st.markdown(f'<span class="pill">Protein tokens: {length+2}</span><span class="pill">Hidden letters: {result.masked_count}</span><span class="pill">5 biological annotation channels</span>',unsafe_allow_html=True)
    teaching('The network tries to fill in missing letters using surrounding amino acids and biological context.',
             'Denoiser input: corrupted input_ids [B,L], attention_mask [B,L], conditions [B,L,5], and residue_mask [B,L]; E/F accept additional inputs.',
             f'If position {focus} is hidden, the network sees ID 32 instead of the true {r["AA"]}. The target remains {r["AA"]}. If it is kept, it sees the original ID.',
             'D: 33_experiment_d_model.py; E: 44_experiment_e_context_model.py; F: 54_experiment_f_context_model.py')

    step(2,'Embeddings — turn input IDs and context into useful vectors','Each token becomes a 640-number vector. Learned projections combine different sources of information.')
    paths=['ESM2 token embedding (640 dimensions)','Five local annotation bits → learned projection','Condition-availability gate']
    if experiment in ('E','F'):
        paths += ['GO / EC labels → learned projections','t, t², sin(πt), cos(πt) → projection','Masking exponent → projection']
    if experiment=='F':
        paths += ['Observed mask fraction → projection','Span features (length, position, multi) → projection']
    st.markdown('**Where the information comes from**')
    flow(paths)
    st.latex(r'h_i=\mathrm{ESM2Embedding}(\widetilde{x}_i)+g_i(W_{\mathrm{bio}}c_i+p_{\mathrm{presence}})+\mathrm{extra\ context}')
    st.caption('The expression summarizes additive inputs. Each extra term has its own gating, masking, and learned parameters in the actual implementation.')
    extra=f'For t={time:.2f}, the four exact time features are [{time:.3f}, {time**2:.3f}, {math.sin(math.pi*time):.3f}, {math.cos(math.pi*time):.3f}].' if experiment in ('E','F') else 'Experiment D has local five-channel biological conditioning but no added time-projection tensor.'
    if experiment=='F':
        fvec=result.span_features[focus-1]
        st.dataframe(pd.DataFrame([{'Span length / 8':fvec[0],'Relative position':fvec[1],'Multi-residue indicator':fvec[2]}]).style.format('{:.3f}'),hide_index=True,use_container_width=True)
    teaching('An embedding is like a numerical description of a letter. Biological labels and time provide additional clues.',
             'Learned linear projection + additive fusion, gated by valid residues and conditioning availability. Training uses 15% condition dropout. Actual 640-dimensional values depend on checkpoint weights.',
             extra,
             'D: 33_experiment_d_model.py; E: 41/42/44_experiment_e_*.py; F: 54_experiment_f_context_model.py, 55_experiment_f_span_features.py')

    step(3,'Transformer attention — compare one position with the others','30 layers update each position using information from the sequence.')
    layer_html=''.join(f'<div class="layer{" diff" if experiment=="F" and i>20 else ""}">{i}</div>' for i in range(1,31))
    st.markdown('<div class="layer-grid">'+layer_html+'</div>',unsafe_allow_html=True)
    st.caption('Each numbered box is one Transformer layer. Lavender boxes in F = layers 21–30, where custom Differential Attention is installed. D and E use the original attention in all 30 layers.')
    st.plotly_chart(toy_attention(experiment),use_container_width=True)
    st.caption('The 3×3 heatmap uses handmade numbers solely to explain attention. It is NOT the trained model’s actual attention matrix.')
    st.latex(r'Q=HW_Q,\quad K=HW_K,\quad V=HW_V')
    if experiment=='F':
        st.latex(r'A_{\mathrm{diff}}=A_1-\sigma(\lambda_{\mathrm{logit}})A_2')
        ex='Two toy attention maps can be subtracted: 0.70 − 0.5 × 0.15 = 0.625. The trained λ values are NOT loaded. Ten head pairs are used in layers 21–30.'
        meth='Custom paired-head DifferentialESMAttention: 20 original heads become 10 pairs in the last 10 layers. The difference is applied to paired values, RMS-normalized, and mixed with standard attention. RoPE is retained.'
        src='Experiment_F/scripts/53_experiment_f_differential_attention.py; 56_experiment_f_install_attention.py'
    else:
        st.latex(r'A=\mathrm{softmax}(QK^{\top}/\sqrt{d_k}+\mathrm{mask}),\qquad O=AV')
        ex='In a toy row, 0.70 of the attention weight may go to A, 0.20 to G, and 0.10 to L. These are sample weights, not measured values.'
        meth='ESM2-derived multi-head self-attention with 20 heads of 32 dimensions each and rotary position embeddings (RoPE). All 30 layers use inherited attention.'
        src='Experiment_A/esm2_attention.py; Experiment_A/esm2_layer.py'
    teaching('Attention helps each residue use information from other residues. It is a learned weighting mechanism, not a biological annotation itself.',meth,ex,src)

    step(4,'Logits — turn the final vectors into amino-acid scores','The model produces 33 scores per token position and converts them into probabilities.')
    fig,example_probs=toy_logits()
    st.plotly_chart(fig,use_container_width=True)
    st.latex(r'z_{\mathrm{cap}}=30\tanh(z/30),\qquad p(v)=\mathrm{softmax}(z_{\mathrm{cap}})_v')
    st.caption(f'The example shows only four of the 33 possible tokens for readability. Actual model output shape would be [1, {length+2}, 33]. No real checkpoint logits are computed.')
    teaching('Logits are scores; softmax turns them into probabilities. A higher score means a token receives a higher probability, after normalization.',
             'Final layer normalization → DSM language-model head with tied input/output embeddings → soft logit cap 30×tanh(z/30) → softmax. The real head covers all 33 tokens.',
             f'With toy scores 2, 1, 0, −0.5 for A/G/L/V, A gets ≈ {example_probs["A"]:.1%} probability. This is NOT a prediction for residue {focus}.',
             'Experiment_A/dsm_lm_head.py; Experiment_A/dsm150_model.py')

    step(5,'Training loss — how the model learns from masked positions','This is a TRAINING calculation, not a reverse-generation step.')
    q=float(r['q(t)']); derivative=float(r["q'(t)"]); weight=float(r['Loss weight'])
    illustrative_target_probability=0.4
    cross_entropy=-math.log(illustrative_target_probability)
    illustrative_contribution=weight*cross_entropy if bool(r['Masked']) else 0.0
    st.latex(r'w_i=\frac{q_i\,\prime(t)}{q_i(t)+10^{-3}},\qquad \mathrm{CE}_i=-\log p_\theta(x_i\mid\widetilde{x})')
    st.latex(r'\mathcal{L}=\frac{1}{B L}\sum_{b,i}M_{b,i}\,w_{b,i}\,\mathrm{CE}_{b,i}')
    table=pd.DataFrame([
        ['Actual from masking calculation','q(t)',f'{q:.5f}'],
        ['Actual from masking calculation',"q′(t)",f'{derivative:.5f}'],
        ['Actual from masking calculation','Diffusion weight',f'{weight:.5f}'],
        ['Illustrative assumption','P(true amino acid)',f'{illustrative_target_probability:.2f}'],
        ['Illustrative result','Cross-entropy',f'{cross_entropy:.5f}'],
        ['Illustrative result','Single-position numerator contribution',f'{illustrative_contribution:.5f}'],
    ],columns=['Source of number','Quantity','Value'])
    st.dataframe(table,hide_index=True,use_container_width=True)
    teaching('The training loss measures how surprised the model is by the correct missing amino acid. Positions that were not masked contribute zero.',
             'Masked-token cross-entropy weighted by q′(t)/(q(t)+0.001), summed over masked positions and divided by batch size × full token length. Training then uses backpropagation and an optimizer.',
             f'At position {focus}, the real diffusion weight is {weight:.3f}. IF a hypothetical model assigned 0.40 to the correct token, CE=−ln(0.40)={cross_entropy:.3f}. '+('This masked position would contribute to the sum.' if r['Masked'] else 'This unmasked position contributes zero to the loss.'),
             'Experiment_D/scripts/05_bio_loss.py; Experiment_E/F reuse this masked diffusion loss')

    step(6,'Reverse generation — what is and is not implemented here','Predicting token probabilities is different from implementing a complete reverse-time sampler.')
    flow(['Start from x(t) with MASK tokens','Denoiser gives logits','Choose a reverse transition rule','Update tokens at an earlier time','Repeat toward clean sequence'])
    st.markdown('<div class="note note-purple"><b>Scientific status:</b> This panel is a conceptual sequence of operations. No trained weights or reverse-time transition sampler run in this web app. Therefore no reconstructed protein, sampler trajectory, or claimed recovery accuracy is displayed.</div>',unsafe_allow_html=True)
    teaching('A denoiser suggests likely letters for masked positions. A full generative process needs another rule for which positions to reveal and when.',
             'Reverse-time sampling requires a verified transition kernel/schedule and iterative model evaluation. The source files inspected for this lightweight demo establish training corruption and the denoiser, but this app does not ship a validated full sampler.',
             'A masked position might receive probabilities A=0.6, G=0.3, L=0.1 (illustration only). Those numbers alone do not tell us when to unmask it or whether to sample A.',
             'Inspect and validate a separate reverse sampler before presenting actual generated trajectories')

with comparison:
    st.markdown('<div class="section-title">The same protein and time in all three experiments</div>',unsafe_allow_html=True)
    st.write('We run the three CPU masking rules on the same sequence, annotations, time and NumPy seed. Compare expected mask counts with one random realization.')
    stats=all_experiments(protein,time,int(seed))
    st.plotly_chart(comparison_figure(stats),use_container_width=True)
    st.dataframe(stats,hide_index=True,use_container_width=True)
    st.caption('Expected count is the sum of per-residue probabilities. Actually masked is a random realization, so it may differ from expectation. F introduces correlated draws and can have larger variation.')
    teaching('D masks all residue positions uniformly. E protects annotated functional residues. F starts masking functional positions sooner and groups neighbours into spans.',
             'D: α=1, independent; E: α=(1,1.25,1.50,2.0), independent; F: α=(1.40,0.95,0.75,0.55), shared draws within category-homogeneous spans of maximum 8.',
             f'The current time is t={time:.2f}. The same residue may have different q(t) across D/E/F, even though the protein sequence is identical.',
             'Experiment_D/scripts/04_bio_corruptor.py; Experiment_E/scripts/43_experiment_e_delayed_corruptor.py; Experiment_F/scripts/52_experiment_f_corruptor.py')
    st.markdown('#### Every residue of the corrupted sequence, for each experiment')
    for letter in 'DEF':
        other=calculate(protein,letter,float(time),int(seed))
        st.markdown(f'**Experiment {letter}: {other.masked_count} of {length} masked**')
        sequence_grid(protein['sequence'],other.frame,focus,masked=True)
    st.markdown('#### Compare the selected residue numerically')
    def row_of(letter):
        f=calculate(protein,letter,float(time),int(seed)).frame.iloc[focus-1]
        return {'Experiment':letter,'Residue':f['AA'],'Category':f['Priority'],'Exponent α':f['α'],
                'q(t)':f['q(t)'],'Random u':f['Random u'],'Masked':bool(f['Masked'])}
    st.dataframe(pd.DataFrame([row_of(x) for x in 'DEF']),hide_index=True,use_container_width=True)
    labeled_note('These are comparisons of the forward masking rules, not a ranking of denoising accuracy. You would need a common, checkpoint-based evaluation to compare reconstruction performance.')

with methods:
    st.markdown('<div class="section-title">All methods, assumptions and downloads</div>',unsafe_allow_html=True)
    st.write('This page records the exact methods used in the teaching app so you can answer questions during a supervisor demonstration.')
    st.markdown('#### Diffusion mathematics')
    st.latex(r'q_i(t)=t^{\alpha_i},\qquad q_i\,\prime(t)=\alpha_i t^{\alpha_i-1},\qquad w_i(t)=\frac{q_i\,\prime(t)}{q_i(t)+0.001}')
    st.dataframe(pd.DataFrame([
        ['D','Ordinary and annotated',1.00,'Independent per residue','Biology-condition embedding before Transformer'],
        ['E','Ordinary',1.00,'Independent per residue','GO/EC, time and priority context'],
        ['E','REGION',1.25,'Independent per residue','Same E model'],
        ['E','DOMAIN',1.50,'Independent per residue','Same E model'],
        ['E','ACT_SITE/BINDING/MOTIF',2.00,'Independent per residue','Same E model'],
        ['F','Ordinary',1.40,'Shared within spans ≤8','Span context + Differential Attention'],
        ['F','REGION',0.95,'Shared within spans ≤8','Same F model'],
        ['F','DOMAIN',0.75,'Shared within spans ≤8','Same F model'],
        ['F','ACT_SITE/BINDING/MOTIF',0.55,'Shared within spans ≤8','Same F model'],
    ],columns=['Experiment','Category','α','Random masking','Denoising architecture']),hide_index=True,use_container_width=True)
    st.markdown('#### Full 33-token ESM2 vocabulary')
    st.dataframe(vocabulary_table(),hide_index=True,use_container_width=True,height=360)
    st.markdown('#### Protein details and feature annotations')
    st.write(f'UniProt accession **{accession}** — [{protein["name"]}]({protein["sequence_source"]}) · {length} residues.')
    st.markdown(f'**Annotation source:** {annotation_source}. {protein.get("annotation_note", "")}')
    features=pd.DataFrame(protein.get('features',[]))
    if features.empty:
        st.info('No selected compatible feature annotations are available for this protein.')
    else:
        st.dataframe(features,hide_index=True,use_container_width=True)
    st.caption('Only these feature types are used: ACT_SITE, BINDING, MOTIF, DOMAIN, REGION. They are separate input channels; overlapping labels remain available even though one priority selects α.')
    st.markdown('#### Exact source files used for the explanation')
    st.code('''Experiment_A/dsm150_model.py
Experiment_A/esm2_embeddings.py
Experiment_A/esm2_attention.py
Experiment_A/esm2_layer.py
Experiment_A/dsm_lm_head.py
Experiment_D/scripts/04_bio_corruptor.py
Experiment_D/scripts/05_bio_loss.py
Experiment_D/scripts/32_experiment_d_conditions.py
Experiment_D/scripts/33_experiment_d_model.py
Experiment_E/scripts/43_experiment_e_delayed_corruptor.py
Experiment_E/scripts/44_experiment_e_context_model.py
Experiment_F/scripts/52_experiment_f_corruptor.py
Experiment_F/scripts/53_experiment_f_differential_attention.py
Experiment_F/scripts/54_experiment_f_context_model.py
Experiment_F/scripts/55_experiment_f_span_features.py
Experiment_F/scripts/56_experiment_f_install_attention.py''',language='text')
    st.markdown('#### What is computed vs explained?')
    st.dataframe(pd.DataFrame([
        ['Full token IDs for every amino acid','Computed from pinned ESM2 vocabulary','Yes'],
        ['Annotation channels and priority','Computed from bundled or live UniProt-compatible features','Yes'],
        ['Continuous-time masking probabilities and derivatives','Computed for the selected t','Yes'],
        ['Random masking decisions and F spans','Computed with seeded NumPy (not PyTorch-identical draws)','Yes'],
        ['Corrupted token IDs','Computed','Yes'],
        ['640-dimensional learned embeddings','Architecture explained; no weights loaded','No'],
        ['Trained attention matrices','Illustrative 3×3 toy only','No'],
        ['Learned 33-token logits and amino-acid predictions','Toy bar plot only','No'],
        ['Training loss at real model logits','Equation and hypothetical target probability','No'],
        ['Validated reverse sampler / generated protein','Conceptual workflow only','No'],
    ],columns=['Quantity','Provenance','Real computation in app?']),hide_index=True,use_container_width=True)
    st.markdown('#### Export the complete results')
    full_table=token_table(result)
    dl1,dl2,dl3=st.columns(3)
    with dl1:
        st.download_button('⬇ Full token table CSV',full_table.to_csv(index=False).encode(),f'{accession}_{experiment}_tokens.csv','text/csv',key='methods_tokens')
    with dl2:
        st.download_button('⬇ Complete masking CSV',frame.to_csv(index=False).encode(),f'{accession}_{experiment}_masking.csv','text/csv',key='methods_masking')
    with dl3:
        st.download_button('⬇ Forward trace JSON',json.dumps(forward_record(protein,result),indent=2).encode(),f'{accession}_{experiment}_trace.json','application/json',key='methods_record')
    st.download_button('⬇ FASTA (complete original sequence)',f'>{accession} {protein["name"]}\n'+ '\n'.join(protein['sequence'][j:j+60] for j in range(0,length,60))+'\n',
                       f'{accession}.fasta','text/plain',key='methods_fasta')
    st.markdown('#### Common questions')
    with st.expander('Is the diffusion time a discrete integer?'):
        st.write('No. The time variable t is continuous (for example, 0.37), while the states of the protein are discrete token IDs. This is continuous-time discrete-state diffusion.')
    with st.expander('Does a lower training loss prove F is worse?'):
        st.write('No. D/E/F have different masking distributions and loss weights, so their saved validation objectives are not directly comparable without a matched evaluation.')
    with st.expander('Are CLS, EOS or PAD ever masked?'):
        st.write('No. All three masking implementations use a valid-residue mask. This unpadded single-protein example has CLS and EOS but no PAD. PAD is relevant when batching sequences of different lengths.')
    with st.expander('Why is the same random seed not identical to PyTorch training?'):
        st.write('The app uses NumPy’s default_rng to make stable browser results. Original training uses torch.rand. Both produce uniform random values with the right dependence structure, but their generated numbers are not bit-identical.')
    with st.expander('Does the chart show real Transformer attention?'):
        st.write('No. The attention heatmap is a labelled, small numerical example. A trained attention map requires loading the checkpoint and running the full model.')

st.divider()
st.caption('CTDD Protein Diffusion Explorer · Interactive forward diffusion (real CPU calculations) + denoising architecture (clearly marked teaching examples). Source-aligned; no checkpoint or GPU required for website use.')
