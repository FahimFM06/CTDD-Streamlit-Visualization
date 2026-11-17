"""CTDD Protein Diffusion Explorer v3.1 — readable light-theme guided interface.
All core CPU masking calculations and protein examples remain in one file
for straightforward GitHub / Streamlit Community Cloud deployment.
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

# ============================================================================
# Version 3.1 UI. Exactly one stage is rendered; enforced high-contrast light theme.
# ============================================================================
import html
import json
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="CTDD | Protein Diffusion — Step by Step",
    page_icon="🧬", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --navy:#163d59; --blue:#2086bc; --muted:#58758b; --border:#d8e9f5; }
/* Reliable light mode for Streamlit Cloud, even when .streamlit/config.toml
   was not uploaded or the browser/host prefers a dark theme. */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    color-scheme: light !important;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #F7FBFF !important;
    color: #17324D !important;
}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #ECF5FC !important;
    color: #17324D !important;
}
/* Markdown, navigation, help text and widget labels were inheriting a pale
   color from Streamlit's dark theme on top of our pale page background. */
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label {
    color: #17324D !important;
}
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] > p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
    color: #17324D !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
[data-testid="stAppViewContainer"] small,
[data-testid="stSidebar"] small {
    color: #426179 !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #D0E1EC !important;
    border-radius: 13px !important;
    padding: .65rem .85rem !important;
    min-height: 112px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #142F47 !important;
    font-weight: 740 !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #365B74 !important;
}
/* Native form controls are white with dark text, rather than black/gray. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #17324D !important;
    border-color: #AECADA !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stNumberInput"] input::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #17324D !important;
}
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
    background-color: #FFFFFF !important;
    color: #18547B !important;
    border: 1px solid #B5D0E0 !important;
    border-radius: 9px !important;
}
[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    background-color: #E4F3FD !important;
    border-color: #2789BF !important;
    color: #0F486E !important;
}
[data-testid="stButton"] button:disabled,
[data-testid="stButton"] button:disabled * {
    color: #687C8B !important;
    background-color: #F3F6F9 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #147FAF !important;
}
a, [data-testid="stMarkdownContainer"] a {
    color: #126EA4 !important;
}
/* On narrow screens the four metric cards stack with readable text. */
@media(max-width:700px) {
    [data-testid="stMetric"] {min-height: 0; padding: .65rem .8rem !important;}
}
[data-testid="stAppViewContainer"] {background:#f7fbff;}
[data-testid="stSidebar"] {background:#eff7fd; border-right:1px solid #d9eaf7;}
.block-container {max-width:1160px; padding-top:1.25rem; padding-bottom:3.5rem;}
h1,h2,h3 {color:#163d59;letter-spacing:-.02em}
h1 {font-size:2.45rem !important;}
.stMarkdown p,.stMarkdown li {line-height:1.64;}
.hero {background:linear-gradient(115deg,#eaf7ff 0%,#f7fcff 65%,#e7f1ff 100%);
       border:1px solid #cfe7f6;border-radius:20px;padding:1.55rem 1.75rem 1.25rem;
       margin:.4rem 0 1.25rem;}
.eyebrow {text-transform:uppercase;letter-spacing:.14em;font-size:.77rem;color:#2384ae;font-weight:760;}
.hero-title {color:#163d59;font-size:2.23rem;font-weight:810;letter-spacing:-.045em;line-height:1.15;margin:.34rem 0 .55rem;}
.hero-lead {color:#527189;max-width:740px;font-size:1.02rem;line-height:1.6;}
.mini-context {color:#58758b;font-size:.85rem;margin:0 0 .9rem;}
.stage-head {background:white;border:1px solid var(--border);border-left:5px solid #69bce6;
             padding:1.1rem 1.3rem;border-radius:13px;margin:.75rem 0 1rem;}
.stage-no {font-size:.77rem;color:#2086bc;font-weight:780;letter-spacing:.1em;text-transform:uppercase;}
.stage-title {font-size:1.5rem;font-weight:770;color:#163d59;margin:.22rem 0;}
.stage-summary {color:#5c778b;font-size:.96rem;line-height:1.5;}
.figure-caption {color:#567287;font-size:.84rem;margin:.2rem 0 1rem;}
.lesson-panel {border:1px solid #d6e8f5;background:white;padding:1.15rem 1.3rem;
               border-radius:14px;margin:.8rem 0 .85rem;}
.lesson-blue {border-left:4px solid #48abdb;}
.lesson-mint {border-left:4px solid #30b5a6;background:#f4fdfb;}
.lesson-sand {border-left:4px solid #e6b26b;background:#fffaf2;}
.lesson-h {color:#244c64;font-size:1.02rem;font-weight:750;margin-bottom:.38rem;}
.lesson-body {color:#31566d;line-height:1.72;font-size:.96rem;}
.source-line {font-size:.84rem;color:#668198;overflow-wrap:anywhere;line-height:1.55;}
.tiny-tag {display:inline-block;font-size:.75rem;font-weight:700;color:#236d9a;background:#eaf5fc;
           padding:.23rem .55rem;border:1px solid #d5eaf7;border-radius:30px;margin-bottom:.5rem;}
.diagram {display:flex;gap:.62rem;align-items:stretch;flex-wrap:wrap;margin:1.15rem 0 1rem;}
.diagram-step {background:#eff8ff;border:1px solid #cde6f7;border-radius:12px;
               color:#265678;padding:.9rem .75rem;text-align:center;font-size:.9rem;font-weight:675;
               flex:1 1 140px;min-width:130px;}
.diagram-arrow {align-self:center;color:#65a9cb;font-size:1.35rem;font-weight:900;}
.diagram-step.purple {background:#f1efff;border-color:#dad6fa;color:#595090;}
.diagram-step.mint {background:#eafbf6;border-color:#c5ebdf;color:#237663;}
.seq-card {background:white;border:1px solid #d7e9f5;border-radius:14px;padding:.8rem 1rem; margin:.45rem 0 1rem;}
.seq-row {display:flex;align-items:flex-start;gap:.5rem;margin:.35rem 0;}
.seq-row-label {font:600 .73rem ui-monospace,monospace;color:#7893a8;min-width:68px;padding-top:.35rem;}
.seq-items {display:flex;flex-wrap:wrap;gap:3px;min-width:0;}
.aa-tile {font:720 .80rem ui-monospace,SFMono-Regular,Consolas,monospace;
          width:25px;height:28px;display:inline-flex;align-items:center;justify-content:center;
          border:1px solid #d9e7ee;border-radius:5px;color:#2e4e65;background:#f3f7fa;}
.aa-tile.NORMAL {background:#e9f0f5;}
.aa-tile.REGION {background:#d6f7ef;border-color:#b2e8da;}
.aa-tile.DOMAIN {background:#d9ebff;border-color:#bbd7f7;}
.aa-tile.MOTIF {background:#ede3ff;border-color:#d8c8f3;}
.aa-tile.BINDING {background:#ffecd6;border-color:#f3d8b7;}
.aa-tile.ACT_SITE {background:#ffdfeb;border-color:#f3c4d2;}
.aa-tile.masked {background:#487ba9;color:white;border-color:#487ba9;}
.aa-tile.focused {outline:2px solid #e6a745;outline-offset:1px;}
.legend-line {font-size:.8rem;color:#70869a;margin:.2rem 0 .55rem;}
.layer-strip {display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:7px;margin:1.1rem 0;}
.layer-chip {text-align:center;padding:.54rem .1rem;background:#e5f5ff;border:1px solid #cde5f6;border-radius:8px;color:#2b6c93;font-size:.87rem;font-weight:740;}
.layer-chip.diff {background:#edeaff;border-color:#d4cef8;color:#6256a0;}
.status-note {background:#fff9e9;border:1px solid #f3dfaa;color:#705b32;border-radius:11px;
              padding:.8rem .95rem;font-size:.91rem;line-height:1.5;margin:.8rem 0;}
@media(max-width:700px){
 .block-container {padding-top:.5rem;}
 .hero {padding:1.1rem 1rem;border-radius:13px;}
 .hero-title {font-size:1.68rem;}
 .stage-title {font-size:1.25rem;}
 .seq-card {padding:.65rem .5rem;}
 .seq-row {display:block;margin:.75rem 0;}
 .seq-row-label {display:block;padding:0;margin-bottom:.2rem;}
 .aa-tile {width:24px;height:27px;}
 .layer-strip {grid-template-columns:repeat(6,minmax(0,1fr));}
 .diagram-arrow {display:none;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def proteins_catalog():
    # Embedded pinned full-length protein examples; no external files required.
    return json.loads(EMBEDDED_PROTEIN_EXAMPLES)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_cached(protein):
    return fetch_protein(protein)


def diagram(items):
    parts = []
    for i, item in enumerate(items):
        if i:
            parts.append('<span class="diagram-arrow">→</span>')
        c = ['','mint','purple'][i % 3]
        parts.append(f'<div class="diagram-step {c}">{html.escape(str(item))}</div>')
    st.markdown('<div class="diagram">' + ''.join(parts) + '</div>', unsafe_allow_html=True)


def protein_map(frame, focus, mode='original'):
    """Draw ALL amino-acid positions with no missing pages or preview cutoff."""
    tiles = []
    for start in range(0, len(frame), 40):
        grp = frame.iloc[start:start + 40]
        items = []
        for _, rr in grp.iterrows():
            pos = int(rr['Position'])
            is_masked = bool(rr['Masked']) and mode == 'corrupted'
            character = '×' if is_masked else str(rr['AA'])
            category = 'masked' if is_masked else str(rr['Priority'])
            classes = f'aa-tile {category}' + (' focused' if pos == focus else '')
            tip = (f"Position {pos} · original {rr['AA']} · {rr['Priority']} · "
                   f"q={rr['q(t)']:.4f}" + (' · MASK' if is_masked else ''))
            items.append(f'<span class="{classes}" title="{html.escape(tip,quote=True)}">{html.escape(character)}</span>')
        tiles.append('<div class="seq-row"><div class="seq-row-label">'
                     f'{start+1}–{min(start+40,len(frame))}'
                     '</div><div class="seq-items">' + ''.join(items) + '</div></div>')
    st.markdown('<div class="seq-card">' + ''.join(tiles) + '</div>',unsafe_allow_html=True)
    st.markdown('<div class="legend-line">Every residue is shown · yellow outline = selected position · '
                'pink = active site · orange = binding · blue = domain · mint = region · '
                'purple = motif · dark blue × = MASK</div>', unsafe_allow_html=True)


def step_header(index, title, short, section='Forward diffusion'):
    st.markdown(f'<div class="stage-head"><div class="stage-no">{html.escape(section)} · '
                f'Step {index} of 6</div><div class="stage-title">{html.escape(title)}</div>'
                f'<div class="stage-summary">{html.escape(short)}</div></div>',unsafe_allow_html=True)


def lesson(plain, example, method, source, formula=None, caveat=None):
    """Only the chosen stage gets its explanation and method after its figure."""
    st.markdown(f'<div class="lesson-panel lesson-blue"><div class="lesson-h">What is happening?</div>'
                f'<div class="lesson-body">{plain}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="lesson-panel lesson-mint"><div class="lesson-h">Example from this protein</div>'
                f'<div class="lesson-body">{example}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="lesson-panel lesson-sand"><div class="lesson-h">Method used in the original code</div>'
                f'<div class="lesson-body">{method}</div></div>',unsafe_allow_html=True)
    if formula is not None:
        st.latex(formula)
    st.markdown(f'<div class="source-line"><b>Source module(s):</b> {html.escape(source)}</div>',unsafe_allow_html=True)
    if caveat:
        st.caption(caveat)


def select_step(prefix, options):
    """Single navigation control; do not render the other five stage bodies."""
    key = prefix + '_step'
    st.session_state.setdefault(key, 0)
    def shift(delta):
        st.session_state[key] = min(5, max(0, st.session_state[key] + delta))
    a, b, c = st.columns([1.15, 4, 1.15])
    with a:
        st.button('← Previous', key=prefix + '_prev', disabled=st.session_state[key] == 0,
                  on_click=shift, args=(-1,), use_container_width=True)
    with b:
        chosen = st.selectbox('Choose one step to explore', options=list(range(6)), key=key,
                              format_func=lambda i: f'{i+1:02d}  ·  {options[i]}')
    with c:
        st.button('Next →', key=prefix + '_next', disabled=st.session_state[key] == 5,
                  on_click=shift, args=(1,), use_container_width=True)
    st.progress((chosen+1)/6)
    return chosen


def layer_strip(experiment):
    items = []
    for i in range(1,31):
        cl = 'layer-chip diff' if (experiment == 'F' and i > 20) else 'layer-chip'
        items.append(f'<span class="{cl}">{i:02d}</span>')
    st.markdown('<div class="layer-strip">' + ''.join(items) + '</div>',unsafe_allow_html=True)
    st.caption('Each square is one Transformer layer. Purple layers 21–30 are the custom Differential Attention layers in F.')


CATALOG = proteins_catalog()
EXPERIMENTS = {
    'D':'Uniform independent masking + residue conditioning',
    'E':'Function-protecting masking + GO/EC context',
    'F':'Function-first spans + Differential Attention',
}

st.markdown('''<div class="hero"><div class="eyebrow">CTDD · interactive protein diffusion</div>
<div class="hero-title">Explore one step at a time</div>
<div class="hero-lead">Choose a real protein and an experiment. Open one stage to see
what happens to <b>every amino acid</b>, followed by a small example, the formula,
and the exact method used in the project.</div></div>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### Your protein and model')
    acc = st.selectbox('1 · Choose protein', [p['id'] for p in CATALOG],
                      format_func=lambda x: next(f"{p['name']} · {x}" for p in CATALOG if p['id'] == x))
    expt = st.radio('2 · Choose experiment', list('DEF'), horizontal=True,
                    format_func=lambda x: f'Experiment {x}')
    t = st.slider('3 · Continuous diffusion time  t', min_value=0.01, max_value=0.99,
                  value=0.50, step=0.01, help='Fixed t for teaching. Original training samples a fresh t.')
    seed = int(st.number_input('4 · Reproducible masking seed', min_value=0,
                                max_value=2**31-1, value=2026, step=1))
    live = st.checkbox('Use current UniProt feature annotations', value=False,
                       help='Online optional; bundled annotated examples work offline.')
    st.divider()
    st.caption('CPU-only teaching dashboard. Original forward masking laws; trained 150M-parameter neural networks are not executed here.')

base_protein = next(p for p in CATALOG if p['id']==acc)
protein = base_protein
if live:
    try:
        protein = fetch_cached(base_protein)
    except Exception:
        st.warning('UniProt could not be reached or the sequence changed. Showing pinned, selected annotations instead.')

res = calculate(protein, expt, float(t), seed)
frame = res.frame
n = len(frame)
focus_default = next((int(f['start']) for k in ('ACT_SITE','BINDING','MOTIF')
                      for f in protein.get('features',[]) if f.get('type')==k), 1)
with st.sidebar:
    focus = st.slider('5 · Follow amino-acid position', 1, n,
                      value=max(1, min(n,focus_default)), key=f'focus_{acc}')
    st.caption('The yellow-outlined letter is the one used in the worked example.')

record = frame.iloc[focus-1]
letter = str(record['AA'])
category = str(record['Priority'])
alpha = float(record['α'])
q = float(record['q(t)'])
u = float(record['Random u'])
mask = bool(record['Masked'])
dq = float(record["q'(t)"])
weight = float(record['Loss weight'])

# Minimal context stays visible without exposing all stage explanations.
st.markdown(f'<div class="mini-context"><b>{html.escape(protein["name"])}</b> · '
            f'<a href="{html.escape(protein["sequence_source"],quote=True)}">UniProt {html.escape(acc)}</a>'
            f' · {n} amino acids · {html.escape(EXPERIMENTS[expt])}</div>',unsafe_allow_html=True)
metric_cols = st.columns(4)
metric_cols[0].metric('Full protein length', f'{n} aa')
metric_cols[1].metric('Current t', f'{t:.2f}')
metric_cols[2].metric('Expected masked', f'{res.expected_count:.1f}')
metric_cols[3].metric('Actually masked', f'{res.masked_count} / {n}')

st.caption('The forward-process numbers are calculated with NumPy using the verified D/E/F masking laws. '
           'Random samples are reproducible but not bit-for-bit identical to the original PyTorch training draws. '
           'Protein features are the bundled selected annotations unless you enable live UniProt refresh.')

section = st.radio('Choose what you want to understand',
                   ['Forward diffusion','Denoising / model','Compare D · E · F','Data and sources'],
                   horizontal=True)
st.divider()

if section == 'Forward diffusion':
    st.markdown('### Forward diffusion · clean protein → masked protein')
    st.write('Pick one of the six stages. **Only the selected stage** is displayed; the protein, t, and focused residue stay the same.')
    stage = select_step('forward', [
        'Tokenization', 'Continuous-time sampling', 'Masking probabilities',
        'Random masking decisions', 'Replace residues with MASK', 'Final model inputs',
    ])

    if stage == 0:
        step_header(1,'Tokenization','Every amino acid becomes an ESM2 vocabulary token. Nothing is cut off.')
        diagram(['<cls> · ID 0', f'{n} original amino-acid tokens', '<eos> · ID 2'])
        protein_map(frame,focus)
        tok = token_table(res)
        st.markdown(f'**Full tokenization: all {n} residues + CLS + EOS = {n+2} tokens**')
        st.dataframe(tok[['Model position (0-based)','Protein residue (1-based)',
                          'Token','Original token ID','Biological annotation']],
                     use_container_width=True,hide_index=True,height=440)
        st.download_button('Download all token IDs (CSV)',tok.to_csv(index=False),
                           file_name=f'{acc}_{expt}_all_tokens.csv',mime='text/csv',key='forward_tokens_download')
        lesson('A protein is written with amino-acid letters. The Transformer cannot directly read the letters, '
               'so the ESM2 tokenizer gives each letter an integer ID. It also adds a start token (CLS) and an end token (EOS).',
               f'At protein position <b>{focus}</b>, the amino acid is <b>{html.escape(letter)}</b>. '
               f'Its ESM2 ID is <b>{res.token_ids[focus]}</b>. The start token uses ID 0 and the end token uses ID 2.',
               'ESM2 discrete amino-acid vocabulary with 33 IDs; model position 0 is CLS and residue 1 follows it. '
               'PAD (ID 1) is needed for batched padding, not for this single unpadded protein.',
               'Experiment_A/dsm_dataset.py; Experiment_A/esm2_embeddings.py',
               formula=r'\text{Input length}=L+2\quad (\mathrm{CLS}+L\;\mathrm{residues}+\mathrm{EOS})')

    elif stage == 1:
        step_header(2,'Continuous-time sampling','A single real number t controls the noise level for this protein.')
        diagram(['Choose a protein', 'Sample a real time t', f't = {t:.2f} for this demonstration'])
        st.plotly_chart(time_curves(expt,float(t)),use_container_width=True)
        st.markdown('<div class="figure-caption">The dashed line shows your selected time. Curves show the masking law at that time.</div>',unsafe_allow_html=True)
        train_rule = ('The D corruptor samples t from Uniform(0,1), then clamps it to [10⁻⁵,1−10⁻⁵].'
                      if expt == 'D' else 'The E/F corruptor samples t = 0.001 + 0.998 × U(0,1), separately for each protein in a training batch.')
        lesson('Time is continuous, so it can be 0.15, 0.37, 0.83, and so on. Near zero the sequence is mostly '
               'visible; near one most residues become hidden. The protein characters remain discrete token IDs.',
               f'You have selected <b>t={t:.2f}</b>. For the highlighted <b>{html.escape(letter)}{focus}</b>, '
               f'the next stage will calculate a masking probability of <b>{q:.4f}</b>.',
               train_rule + ' The Streamlit slider fixes t to help you study one precise noise level.',
               'D: Experiment_D/scripts/04_bio_corruptor.py; E: Experiment_E/scripts/43_experiment_e_delayed_corruptor.py; F: Experiment_F/scripts/52_experiment_f_corruptor.py',
               formula=r't\in(0,1)\qquad \text{(continuous time, discrete token states)}')

    elif stage == 2:
        step_header(3,'Masking probabilities','Every residue gets qᵢ(t), its chance of becoming MASK at the selected time.')
        st.plotly_chart(residue_probabilities(res),use_container_width=True)
        protein_map(frame,focus)
        st.markdown('**Probability rules at this time**')
        st.dataframe(category_probabilities(expt,t),use_container_width=True,hide_index=True)
        lesson('The model calculates a chance of hiding each residue. Experiment D uses the same chance '
               'everywhere. E protects functional residues. F masks functional residues sooner.',
               f'The selected <b>{html.escape(letter)}{focus}</b> is labelled <b>{html.escape(category)}</b>. '
               f'Its exponent is <b>α={alpha:.2f}</b>, so <b>q={t:.2f}<sup>{alpha:.2f}</sup>={q:.4f}</b>. '
               'This is a probability, not a guarantee of masking.',
               'D: exponent 1 for all residues. E: ordinary 1.00, REGION 1.25, DOMAIN 1.50, '
               'functional 2.00. F: ordinary 1.40, REGION 0.95, DOMAIN 0.75, functional 0.55. '
               'Feature priority: ACT_SITE > BINDING > MOTIF > DOMAIN > REGION > ordinary.',
               'D: 04_bio_corruptor.py (beta=0 during D training); E: 43_experiment_e_delayed_corruptor.py; F: 52_experiment_f_corruptor.py',
               formula=r'q_i(t)=t^{\alpha_i},\qquad q_i\,\prime(t)=\alpha_i t^{\alpha_i-1}')

    elif stage == 3:
        step_header(4,'Random masking decisions','Compare qᵢ(t) with a random draw; F shares a draw within a span.')
        st.plotly_chart(random_vs_probability(res,focus,radius=12),use_container_width=True)
        st.caption('Blue/orange dots are the random draws around your chosen residue; the line is the masking threshold.')
        protein_map(frame,focus,mode='corrupted')
        st.markdown(f'**All {n} masking decisions, not just the chart window**')
        cols=['Position','AA','Annotations','Priority','α','q(t)','Random u','Masked']
        if expt=='F': cols.insert(-1,'Span')
        st.dataframe(frame[cols],use_container_width=True,hide_index=True,height=390)
        f_extra=(f' In F, this position belongs to a category-consistent span anchored at '
                 f'<b>{int(record["Span"])}</b>; all members of that span reuse the same draw.'
                 if expt=='F' else ' In D/E, different residues have independent random draws.')
        lesson('We compare a uniform random number u between 0 and 1 with each probability q. '
               'If u is smaller, that residue is hidden. In F, nearby residues of the same category '
               'form short spans and share the same random number.',
               f'For <b>{html.escape(letter)}{focus}</b>, q={q:.4f} and random u={u:.4f}. '
               f'Because u is <b>{"smaller" if mask else "not smaller"}</b> than q, this residue is '
               f'<b>{"MASKED" if mask else "KEPT"}</b>.{f_extra}',
               'D/E: independent Bernoulli draws for each eligible residue. F: contiguous '
               'same-priority-category runs, split into spans of at most eight, with the anchor random '
               'draw shared across each span. This dashboard uses a fixed NumPy seed.',
               'D: 04_bio_corruptor.py; E: 43_experiment_e_delayed_corruptor.py; F: 52_experiment_f_corruptor.py',
               formula=(r'M_i=\mathbf1[u_{\mathrm{span}(i)}<q_i(t)]\quad \text{(F)}' if expt=='F'
                        else r'M_i=\mathbf1[u_i<q_i(t)]\quad \text{(D/E)}'),
               caveat='NumPy and PyTorch use different random-number streams; the probability law and span dependence are reproduced, not the exact original training random values.')

    elif stage == 4:
        step_header(5,'Replace selected residues with MASK','The selected letters become token ID 32. The rest of the protein stays as it was.')
        st.markdown('**Before — complete original protein**')
        protein_map(frame,focus,'original')
        st.markdown('**After — complete corrupted protein**')
        protein_map(frame,focus,'corrupted')
        st.metric('Residues replaced by MASK',f'{res.masked_count} of {n}')
        lesson('A masked amino acid no longer shows its identity to the denoising model. '
               'The original amino-acid token is still stored as the correct answer for training.',
               f'At position <b>{focus}</b>, the original letter <b>{html.escape(letter)}</b> '
               f'has token ID <b>{res.token_ids[focus]}</b>. The final input is '
               f'<b>{"MASK (ID 32)" if mask else html.escape(letter)+" (unchanged)"}</b>.',
               'Absorbing-mask corruption: replace selected amino-acid token IDs with ESM2 '
               'MASK (ID 32). CLS and EOS are never masked. Positions not selected are unchanged.',
               'D: 04_bio_corruptor.py; E: 43_experiment_e_delayed_corruptor.py; F: 52_experiment_f_corruptor.py',
               formula=r'\widetilde x_i=\begin{cases}\text{MASK}, & M_i=1\\x_i, & M_i=0\end{cases}')

    else:
        step_header(6,'Final model inputs','Collect the exact corrupted token IDs and conditioning inputs expected by the denoiser.')
        diagram(['Corrupted IDs', 'Attention and residue masks', 'Biological annotations', 'Denoiser input'])
        protein_map(frame,focus,'corrupted')
        rows=[('input_ids',f'[1,{n+2}]','Corrupted sequence with CLS/EOS'),
              ('attention_mask',f'[1,{n+2}]','Ones for the entire unpadded sequence'),
              ('residue_mask',f'[1,{n+2}]','True for original amino-acid positions'),
              ('conditions',f'[1,{n+2},5]','Five residue-level biological feature channels')]
        if expt in ('E','F'):
            rows += [('diffusion_time','[1]',f'Continuous time t={t:.2f}'),
                     ('masking_exponents',f'[1,{n+2}]','Per-position exponents'),
                     ('go_ids / ec_ids','Variable','Protein-function labels; trained vocab mapping not bundled')]
        if expt=='F':
            rows += [('span_features',f'[1,{n+2},3]','Span-length/position features'),
                     ('observed mask fraction','Derived','Fraction of residues that are masked')]
        st.dataframe(pd.DataFrame(rows,columns=['Input','Shape','Meaning']),use_container_width=True,hide_index=True)
        st.markdown(f'**Complete original and corrupted IDs ({n+2} rows)**')
        tok=token_table(res)
        st.dataframe(tok,use_container_width=True,hide_index=True,height=400)
        st.download_button('Download complete model token inputs (CSV)',tok.to_csv(index=False),
                           file_name=f'{acc}_{expt}_model_tokens.csv',mime='text/csv',key='model_inputs_download')
        lesson('All of the information is now arranged as arrays (tensors). The trained denoiser '
               'would use these tensors to predict amino-acid identities, but this web app does not load the neural weights.',
               f'For <b>{n}</b> residues, we have <b>{n+2}</b> token positions (including CLS/EOS). '
               f'The selected residue <b>{focus}</b> has final token ID <b>{res.corrupted_ids[focus]}</b>.',
               'The D model takes corrupted token IDs, attention masks and five-channel annotations. '
               'E adds time, masking-exponent and GO/EC context. F also adds span context and observed mask information.',
               'D: 33_experiment_d_model.py; E: 44_experiment_e_context_model.py; F: 54_experiment_f_context_model.py',
               formula=r'\mathrm{input\_ids}\in\mathbb{N}^{1\times(L+2)}',
               caveat='GO/EC trained vocabulary indices and learned embeddings are not reconstructed by this lightweight demo.')

elif section == 'Denoising / model':
    st.markdown('### Denoising · how the trained architecture is designed to process the masked protein')
    st.markdown('<div class="status-note"><b>Important:</b> This section teaches the verified network architecture and equations. '
                'It does not run the 150M-parameter checkpoint. Any numerical prediction below is explicitly hypothetical.</div>',
                unsafe_allow_html=True)
    stage=select_step('denoise',[
        'What the denoiser receives','Token embeddings and conditioning',
        'Transformer and attention','Logits and amino-acid probabilities',
        'Training loss (not generation)','Reverse sampling: what is and is not available',
    ])
    if stage == 0:
        step_header(1,'The denoiser receives masked protein tokens','The model receives the corrupted sequence plus biological information.','Denoising')
        protein_map(frame,focus,'corrupted')
        diagram(['MASK token IDs','Attention mask','Biological annotations','DSM150 denoiser'])
        st.markdown(f'**Full corrupted model input: {n+2} tokens**')
        st.dataframe(token_table(res)[['Model position (0-based)','Token','Corrupted token','Corrupted token ID','Masked?']],
                     use_container_width=True,hide_index=True,height=400)
        lesson('The denoiser sees the protein with some letters hidden and tries to infer which '
               'amino acid could belong at each hidden position.',
               f'At position <b>{focus}</b>, the input is '
               f'<b>{"MASK (32)" if mask else html.escape(letter)+" (still visible)"}</b>. '
               'If it is masked, the original target for training is still the amino acid '
               f'<b>{html.escape(letter)}</b>.',
               'Continuous-time absorbing-mask denoising with the DSM150 model. '
               'The forward corruption creates input_ids; model-specific conditioning accompanies them.',
               'Experiment_A/dsm150_model.py; D: 33_experiment_d_model.py; E: 44_experiment_e_context_model.py; F: 54_experiment_f_context_model.py')

    elif stage == 1:
        step_header(2,'Embeddings and biological conditioning','Numerical vectors combine token identity with available biological context.','Denoising')
        extras = {'D':'Five local biological feature channels',
                  'E':'Five local channels + GO/EC + time + masking priority',
                  'F':'E context + span features + observed mask fraction'}[expt]
        diagram(['Input token ID', 'ESM2 embedding · 640 values', extras, 'Conditioned vector · 640 values'])
        protein_map(frame,focus,'corrupted')
        st.markdown('**Biological feature channels for the selected position**')
        present=[x.strip() for x in str(record['Annotations']).split(',') if x.strip() and x.strip()!='—']
        bits=[{'Channel':nm,'Present at this residue':nm in present} for nm in FEATURES]
        st.dataframe(pd.DataFrame(bits),use_container_width=True,hide_index=True)
        example=f'At <b>{html.escape(letter)}{focus}</b>, the five annotations are <b>{html.escape(", ".join(present) if present else "none")}</b>. '
        example+='In D the model projects those five 0/1 values into a 640-dimensional vector and adds it to the token embedding.'
        if expt!='D': example+=' E also uses protein-level function labels and time/priority context.'
        if expt=='F': example+=' F additionally has span features and observed masking fraction.'
        lesson('An embedding is a learned numerical description of a token. Biological annotations '
               'are projected into the same 640-dimensional space so they can influence the model.',
               example,
               f'{extras}. The original code uses learned linear projections and additive fusion '
               'before the Transformer layers. Those learned values are not calculated in this CPU demo.',
               'D: 33_experiment_d_model.py; E: 44_experiment_e_context_model.py; F: 54_experiment_f_context_model.py',
               formula=r'h_i^{\mathrm{conditioned}}=h_i^{\mathrm{token}}+h_i^{\mathrm{context}}\quad\text{(schematic)}',
               caveat='The example describes verified projection paths; no numerical 640-dimensional trained vector is shown.')

    elif stage == 2:
        step_header(3,'Transformer layers and attention','Thirty layers communicate information along the protein. F changes attention in layers 21–30.','Denoising')
        diagram(['Query Q · what am I looking for?', 'Key K · what can I match?',
                 'Value V · what information is shared?', 'Updated residue representation'])
        layer_strip(expt)
        method='The DSM150 backbone contains 30 Transformer layers with 20 standard attention heads, RoPE positional information, and feed-forward blocks.'
        if expt=='F':
            method+=' In F, layers 21–30 replace their attention with a paired-head ESM-compatible Differential Attention variant; each head pair forms A₁ − λ A₂.'
        lesson('Attention lets a residue use information from other positions in the protein. '
               'The network repeats this information exchange across 30 layers.',
               f'The model could use neighbouring visible amino acids to infer what belongs near '
               f'<b>position {focus}</b>. The diagram shows the mechanism, not an observed attention map.',
               method,
               'Experiment_A/esm2_attention.py; Experiment_A/esm2_layer.py; F: 53_experiment_f_differential_attention.py; 56_experiment_f_install_attention.py',
               formula=(r'A_{\mathrm{diff}}=A_1-\lambda A_2\quad\text{(F, paired heads)}' if expt=='F'
                        else r'\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V'),
               caveat='No trained attention matrices are recorded or visualized in this app.')

    elif stage == 3:
        step_header(4,'Logits and amino-acid probabilities','The prediction head would output scores for all 33 vocabulary tokens at each position.','Denoising')
        diagram(['Final hidden vector · 640', 'ESM2-tied LM head', '33 token scores', 'Softmax → probabilities'])
        st.markdown('**Small teaching-only example for three possible letters**')
        toy=pd.DataFrame({'Hypothetical amino acid':['A','G','L'],'Illustrative probability':[0.60,0.30,0.10]})
        fig=go.Figure(go.Bar(x=toy['Hypothetical amino acid'],y=toy['Illustrative probability'],
                             marker_color=['#53add8','#8dc8e6','#bbdced'],
                             text=['0.60','0.30','0.10'],textposition='outside'))
        fig.update_yaxes(range=[0,1]); st.plotly_chart(finish(fig,270,'Example probability','Amino acid'),use_container_width=True)
        st.caption('The three bars above are made-up numbers for teaching. They are NOT measured checkpoint predictions. The real output has 33 scores per position.')
        lesson('The language-model head turns each residue’s hidden vector into 33 scores called '
               'logits. Softmax converts scores into probabilities that add up to one.',
               f'For a hypothetical masked site at position <b>{focus}</b>, pretend A has probability '
               '0.60, G has 0.30 and L has 0.10. This shows how probabilities work; '
               'we do <b>not</b> claim the trained model actually predicted them.',
               'A tied ESM2 language-model head maps hidden vectors to the 33-token vocabulary. '
               'A softmax normalizes the resulting logits; the model checkpoints would be required for real scores.',
               'Experiment_A/dsm_lm_head.py; Experiment_A/dsm150_model.py',
               formula=r'p(x_i=a\mid\widetilde x,t,c)=\frac{\exp(z_{i,a})}{\sum_{v=1}^{33}\exp(z_{i,v})}')

    elif stage == 4:
        step_header(5,'Training loss — how the denoiser learns','The original amino acid is used as a target; this is training, not a reverse sampling step.','Denoising')
        diagram(['Masked position', 'Predict probabilities', 'Compare with true amino acid', 'Weighted cross-entropy loss'])
        st.markdown('**Real time-dependent weight for the selected residue**')
        st.dataframe(pd.DataFrame([{'Position':focus,'Amino acid':letter,'q(t)':q,
                                   'q′(t)':dq,'Weight q′/(q+0.001)':weight,
                                   'Masked in this sample?':mask}]),use_container_width=True,hide_index=True)
        hyp_p=.60
        hyp_ce=-np.log(hyp_p)
        lesson('During training, the model is told the original correct amino acid. '
               'Cross-entropy penalizes a low probability for that correct letter. '
               'The loss is weighted according to the diffusion-time probability and derivative.',
               f'Imagine the model assigned the true amino acid <b>{html.escape(letter)}</b> '
               f'a probability of <b>0.60</b> (hypothetical). Then CE = −ln(0.60) = '
               f'<b>{hyp_ce:.4f}</b>. At t={t:.2f} this residue’s actual diffusion '
               f'weight is <b>{weight:.4f}</b>; its hypothetical weighted CE would be '
               f'<b>{weight*hyp_ce:.4f}</b> <b>only if the position is masked</b>. '
               f'In this sampled corruption it is <b>{"masked" if mask else "unmasked"}</b>.',
               'Masked-position cross-entropy with time-dependent weight '
               'w(t)=q′(t)/(q(t)+0.001), aggregated over batch and sequence positions. '
               'Training uses gradients and an optimizer; the website does not update the model.',
               'D: 05_bio_loss.py; E: 05_bio_loss.py and 43_experiment_e_delayed_corruptor.py; F: training loss in 58_experiment_f_ddp.py',
               formula=r'\ell_i=\mathbf1[M_i=1]\;\frac{q_i\,\prime(t)}{q_i(t)+10^{-3}}\big[-\log p_\theta(x_i\mid\widetilde x,t,c)\big]',
               caveat='The q, derivative and weight here are calculated. The example cross-entropy uses a hypothetical model probability, not a trained-model output.')

    else:
        step_header(6,'Reverse-time sampling: a separate operation','Predicting one missing letter is not yet a complete generation trajectory.','Denoising')
        diagram(['Masked protein at high t', 'Evaluate trained denoiser',
                 'Apply a validated reverse transition', 'Repeat toward lower t'])
        st.markdown('<div class="status-note"><b>Not implemented in this dashboard:</b> '
                    'No verified reverse-time sampler, checkpoint-based predictions, or generated reconstruction trace '
                    'is included. We will not invent them.</div>', unsafe_allow_html=True)
        st.markdown('**The selected real protein is the known training target, not a generated sequence.**')
        protein_map(frame,focus,'original')
        lesson('Forward diffusion hides letters. A denoiser may estimate which letters are missing. '
               'A full reverse diffusion process also needs a mathematically defined rule to update '
               'the masked state repeatedly over decreasing time.',
               f'At position <b>{focus}</b>, the known original amino acid is '
               f'<b>{html.escape(letter)}</b>. Showing this target does not mean the model '
               'has reconstructed it; no model reconstruction has been run here.',
               'A full reverse-time process needs a tested transition kernel/sampling schedule '
               'and repeated calls to the trained denoiser. The original checkpoint would be loaded '
               'separately if numerical generation is added later.',
               'Experiment_A/dsm_diffusion.py and trained D/E/F scripts (sampler capability still requires independent verification)')

elif section == 'Compare D · E · F':
    st.markdown('### Compare the same complete protein across D, E and F')
    st.write('All three cases use the identical amino-acid sequence, annotation maps, selected time, '
             'and reproducible NumPy seed. The corruption rules differ.')
    stats=all_experiments(protein,t,seed)
    st.plotly_chart(comparison_figure(stats),use_container_width=True)
    st.dataframe(stats,use_container_width=True,hide_index=True)
    for letter_expt in 'DEF':
        one=calculate(protein,letter_expt,float(t),seed)
        st.markdown(f'#### Experiment {letter_expt} · {one.masked_count} of {n} residues masked')
        protein_map(one.frame,focus,'corrupted')
    st.markdown('#### The selected amino acid in all three experiments')
    rows=[]
    for letter_expt in 'DEF':
        rr=calculate(protein,letter_expt,float(t),seed).frame.iloc[focus-1]
        rows.append({'Experiment':letter_expt,'Original AA':letter,'Priority':rr['Priority'],
                     'Exponent α':rr['α'],'Mask probability':rr['q(t)'],
                     'Random u':rr['Random u'],'Masked?':rr['Masked']})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.caption('Differences here compare *forward masking*, not trained denoising quality. '
               'Because E/F have different objectives and F has correlated masks, do not rank '
               'model accuracy from these mask counts alone.')

else:
    st.markdown('### Full protein data, methods and downloads')
    st.markdown(f'**Protein:** {protein["name"]} · {acc} · {n} amino acids')
    st.caption(protein.get('annotation_note','Selected pinned annotations.'))
    st.markdown('#### Entire original protein (no truncation)')
    protein_map(frame,focus,'original')
    st.markdown('#### All biological features supplied to this demonstration')
    fdf=pd.DataFrame(protein.get('features',[]))
    if fdf.empty: st.info('No mapped compatible annotations available.')
    else: st.dataframe(fdf,use_container_width=True,hide_index=True)
    st.markdown('#### All 33 ESM2 token IDs')
    st.dataframe(vocabulary_table(),use_container_width=True,hide_index=True,height=440)
    st.markdown('#### Complete download files')
    tdf=token_table(res)
    ca,cb,cc=st.columns(3)
    with ca:
        st.download_button('Full token table (CSV)',tdf.to_csv(index=False),
                           f'{acc}_{expt}_tokens.csv','text/csv',key='download_tokens')
    with cb:
        st.download_button('Full masking data (CSV)',frame.to_csv(index=False),
                           f'{acc}_{expt}_masking.csv','text/csv',key='download_masks')
    with cc:
        st.download_button('Forward record (JSON)',json.dumps(forward_record(protein,res),indent=2),
                           f'{acc}_{expt}_forward.json','application/json',key='download_forward')
    st.download_button('Full FASTA protein sequence',
                       f'>{acc} {protein["name"]}\n'+ '\n'.join(protein['sequence'][k:k+60] for k in range(0,n,60))+'\n',
                       f'{acc}.fasta','text/plain',key='download_fasta')
    st.markdown('#### What is actually computed in this dashboard?')
    st.dataframe(pd.DataFrame([
        ('Full amino-acid token IDs', 'Calculated from ESM2 vocabulary', 'Yes'),
        ('Residue annotations and priority', 'Bundled features or live UniProtKB', 'Yes'),
        ('Masking q(t), derivatives and time weights', 'Original D/E/F formulas, computed on CPU', 'Yes'),
        ('Random masks and F spans', 'Seeded NumPy draws with original dependency structure', 'Yes'),
        ('Corrupted sequence and model input token IDs', 'Calculated', 'Yes'),
        ('Trained token embeddings', 'Architecture explanation only', 'No'),
        ('Trained attention matrices', 'Architecture explanation only', 'No'),
        ('Trained 33-token logits', 'Hypothetical numbers only in teaching example', 'No'),
        ('Measured cross-entropy/gradients', 'Formula and hypothetical example', 'No'),
        ('Reverse sampler and reconstructed protein', 'Not integrated or claimed', 'No'),
    ],columns=['Item','What is shown','Real numerical computation?']),
                 use_container_width=True,hide_index=True)
    st.markdown('#### Verified source files underlying this explanation')
    st.code('''Experiment_A/dsm150_model.py
Experiment_A/esm2_embeddings.py
Experiment_A/esm2_attention.py
Experiment_A/esm2_layer.py
Experiment_A/dsm_lm_head.py
Experiment_D/scripts/04_bio_corruptor.py
Experiment_D/scripts/05_bio_loss.py
Experiment_D/scripts/33_experiment_d_model.py
Experiment_E/scripts/43_experiment_e_delayed_corruptor.py
Experiment_E/scripts/44_experiment_e_context_model.py
Experiment_F/scripts/52_experiment_f_corruptor.py
Experiment_F/scripts/53_experiment_f_differential_attention.py
Experiment_F/scripts/54_experiment_f_context_model.py
Experiment_F/scripts/55_experiment_f_span_features.py
Experiment_F/scripts/56_experiment_f_install_attention.py''',language='text')

st.divider()
st.caption('CTDD · Guided teaching dashboard · CPU forward calculations · Model predictions shown only when verified (none bundled).')
