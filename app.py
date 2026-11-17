"""CTDD Streamlit Explorer — CPU-only educational forward diffusion dashboard.

NO neural model inference: denoising pages are source-aligned structural and
mathematical explanations, with optional explicitly labeled teaching examples.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from modules.diffusion import (
    ESM_VOCAB,
    FEATURES,
    PRETTY,
    all_experiments,
    calculate,
    category_probabilities,
    feature_matrix,
    forward_record,
)
from modules.uniprot import fetch_protein

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title='CTDD | Protein Diffusion Explorer', page_icon='🧬', layout='wide')
st.markdown('''
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px;}
h1, h2, h3 {letter-spacing:-.025em;}
.kicker {font-size:.78rem; font-weight:750; letter-spacing:.15em; color:#008b86; margin-bottom:.25rem;}
.subhead {color:#657384; font-size:1.05rem; margin-top:-.6rem; margin-bottom:1.2rem;}
.mode-note {background:rgba(70,128,142,.07);padding:.75rem 1rem;border:1px solid rgba(80,130,150,.2);border-radius:11px;margin:1rem 0;}
.seq {line-height:2.65; margin-top:.6rem; margin-bottom:1rem; overflow-wrap:anywhere;}
.res {font-family:ui-monospace,SFMono-Regular,Consolas,monospace;display:inline-block;min-width:20px; padding:0 3px; text-align:center; margin:2px 1px;border-radius:5px; border:1px solid #9aa4b12e; font-size:.86rem; font-weight:650;}
.res.normal {background:#94a3b81c;}
.res.region {background:#14b8a626;}
.res.domain {background:#2563eb25;}
.res.motif {background:#a855f72f;}
.res.binding {background:#f973162f;}
.res.act_site {background:#e11d482b;}
.res.masked {background:#17212e;color:white;border-color:#17212e;}
.marker {font-size:.77rem;color:#64748b;}
.legend {font-size:.85rem;color:#64748b;}
.small {font-size:.85rem;color:#637381;}
</style>
''', unsafe_allow_html=True)

@st.cache_data

def local_proteins() -> list[dict]:
    return json.loads((ROOT / 'data' / 'proteins.json').read_text(encoding='utf-8'))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def online_protein(local: dict) -> dict:
    return fetch_protein(local)


def color_residues(frame: pd.DataFrame, start: int, count: int = 80,
                   reveal_mask: np.ndarray | None = None) -> str:
    result = []
    for _, row in frame.iloc[start:start + count].iterrows():
        pos = int(row['Position'])
        hidden = bool(row['Masked']) if reveal_mask is None else bool(reveal_mask[pos - 1])
        category = str(row['Priority']).lower()
        letter = '·' if hidden else str(row['AA'])
        cl = 'masked' if hidden else category
        descr = f"#{pos} {row['AA']} | {row['Priority']} | q={row['q(t)']:.4f}"
        result.append(f'<span class="res {cl}" title="{html.escape(descr, quote=True)}">{html.escape(letter)}</span>')
        if (pos - start) % 10 == 0:
            result.append('<span class="marker"> </span>')
    return '<div class="seq">' + ''.join(result) + '</div>'


def show_sequence(frame: pd.DataFrame, key: str, label: str = 'Amino-acid positions',
                  reveal_mask: np.ndarray | None = None) -> None:
    max_start = max(0, len(frame) - 80)
    start = st.slider(f'{label} — window begins at residue', min_value=1,
                      max_value=max_start + 1, value=1, step=1, key=key) - 1
    st.markdown(f'**Residues {start + 1}–{min(start + 80, len(frame))}**')
    st.markdown(color_residues(frame, start, 80, reveal_mask), unsafe_allow_html=True)
    st.caption('Dark cells = [MASK]; colored cells indicate the highest-priority annotation. Hover over a residue for position, annotation and q(t).')


def formula_head(title: str, explanation: str) -> None:
    st.markdown(f'#### {title}')
    st.write(explanation)


PROTEINS = local_proteins()

st.markdown('<div class="kicker">CTDD · PROTEIN SEQUENCE DIFFUSION</div>', unsafe_allow_html=True)
st.title('Diffusion Model Explorer')
st.markdown('<div class="subhead">Understand how Experiments D, E, and F corrupt protein sequences — and how their trained denoisers are designed to process them.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Your experiment')
    chosen = st.selectbox('Protein', options=[p['id'] for p in PROTEINS],
                          format_func=lambda x: next(f"{p['name']} · {p['id']}" for p in PROTEINS if p['id'] == x))
    experiment = st.radio('Experiment', ['D', 'E', 'F'], horizontal=True)
    time = st.slider('Continuous diffusion time (t)', min_value=0.01, max_value=0.99,
                     value=0.50, step=0.01,
                     help='Training samples a new continuous time. The dashboard lets you fix t to inspect a particular noise level.')
    seed = st.number_input('Reproducible NumPy seed', min_value=0, max_value=2**31 - 1, value=2026, step=1)
    online = st.checkbox('Fetch current UniProt annotations', value=False,
                         help='Requests biological feature coordinates from UniProt; the bundled annotation subset works offline.')
    st.divider()
    st.caption('CPU-only Streamlit app · no checkpoints · no model training')
    st.caption('Forward masking: calculated. Learned predictions and attention: not supplied.')

protein = next(p for p in PROTEINS if p['id'] == chosen)
annotation_status = 'Bundled selected annotations'
if online:
    try:
        protein = online_protein(protein)
        annotation_status = 'Live UniProtKB annotations'
    except Exception as exc:
        st.warning(f'UniProt refresh unavailable; using bundled annotations. Details: {exc}')

result = calculate(protein, experiment, time=float(time), seed=int(seed))
frame = result.frame
n = len(protein['sequence'])
labels = {
    'D': 'Uniform residue masking + biological conditioning',
    'E': 'Function-protecting delayed masking + GO/EC context',
    'F': 'Function-first correlated spans + Differential Attention',
}

c1, c2, c3, c4 = st.columns(4)
c1.metric('Protein length', f'{n} aa')
c2.metric('Diffusion time', f'{time:.2f}')
c3.metric('Expected masked', f'{result.expected_count:.1f} aa')
c4.metric('Sampled masked', f'{result.masked_count} aa', f'{100 * result.observed_fraction:.1f}%')
st.caption(f"**{protein['name']}** ({chosen}; {protein['gene']}) · {labels[experiment]} · {annotation_status}")
st.markdown('<div class="mode-note">🔬 <b>Scientific scope:</b> Forward corruption is calculated from the verified D/E/F equations using NumPy. Random draws follow the same distributions, not PyTorch’s exact random bitstream. The full trained 150M-parameter models are <b>not</b> executed here. Denoiser pages describe the verified architecture and clearly label teaching examples.</div>', unsafe_allow_html=True)

page1, page2, page3, page4 = st.tabs(['Forward diffusion · six steps', 'Denoiser · six steps', 'Compare D / E / F', 'Sources & methods'])

with page1:
    st.markdown('### Forward process: clean protein → masked protein')
    st.write('Each stage uses the same selected protein, time and random seed. Changing the slider updates all six steps.')
    stage = st.radio('Inspect a stage', options=[1, 2, 3, 4, 5, 6], horizontal=True, key='forward_stage',
                     format_func=lambda x: {
                         1: '① Tokens', 2: '② Time', 3: '③ Probabilities',
                         4: '④ Random masking', 5: '⑤ Replace tokens', 6: '⑥ Model input',
                     }[x])
    st.divider()
    if stage == 1:
        formula_head('1. Original sequence and tokenization',
                     'The protein sequence is tokenized with ESM2 vocabulary (33 tokens). The model receives CLS and EOS tokens in addition to its amino-acid tokens. Special tokens are not masked.')
        st.code(f'<cls> {protein["sequence"][:75]}{"…" if n > 75 else ""} <eos>', language=None)
        st.write(f'Token count: **{n + 2}** = {n} amino acids + 2 special tokens.')
        preview = pd.DataFrame({'Token position': [0] + list(range(1, min(n, 14) + 1)) + [n + 1],
                                'Token': ['<cls>'] + list(protein['sequence'][:14]) + ['<eos>'],
                                'Token ID': [0] + result.token_ids[1:15] + [2]})
        st.dataframe(preview, hide_index=True, use_container_width=True)
        with st.expander('Bundled biological annotations (1-based inclusive)'):
            fdf = pd.DataFrame(protein.get('features', []))
            if fdf.empty:
                st.info('No selected annotations for this protein.')
            else:
                st.dataframe(fdf, hide_index=True, use_container_width=True)
            st.caption(protein.get('annotation_note', ''))

    if stage == 2:
        formula_head('2. Continuous diffusion time',
                     'At training time each protein receives a sampled continuous value t in (0,1). Here the sidebar slider fixes t for reproducible inspection.')
        st.latex(r't\in(0,1),\qquad t_{\mathrm{selected}}=' + f'{time:.2f}')
        if experiment == 'D':
            st.write('The D corruptor samples a uniform time and clamps it to [10⁻⁵, 1−10⁻⁵].')
        else:
            st.write('The E/F training corruptors sample time using `0.001 + 0.998 × U(0,1)` by default.')
        st.write('Small t → fewer masked residues. Large t → more masked residues. The same t is applied to every amino-acid position, but E/F use position-dependent masking exponents.')
        st.line_chart(pd.DataFrame({'t': np.linspace(0.01, 0.99, 99),
                                    'ordinary q(t)': np.linspace(0.01, 0.99, 99) ** (1 if experiment != 'F' else 1.40),
                                    'functional q(t)': np.linspace(0.01, 0.99, 99) ** ({'D': 1, 'E': 2.0, 'F': 0.55}[experiment])}).set_index('t'))

    if stage == 3:
        formula_head('3. Calculate masking probabilities and derivatives',
                     'D applies the same probability to all eligible residues. E protects functional positions by masking them later. F prioritizes functional positions for masking earlier.')
        st.latex(r'q_i(t)=t^{\alpha_i},\qquad q_i^{\prime}(t)=\alpha_i t^{\alpha_i-1}')
        st.dataframe(category_probabilities(experiment, float(time)).style.format({'α': '{:.2f}', 'Probability': '{:.4f}', 'Derivative': '{:.4f}'}),
                     hide_index=True, use_container_width=True)
        st.caption('Priority order at overlapping annotations: ACT_SITE > BINDING > MOTIF > DOMAIN > REGION > ordinary.')
        st.line_chart(frame.set_index('Position')[['q(t)']])
        st.write('Actual distribution over residue priorities:')
        st.bar_chart(frame['Priority'].value_counts())

    if stage == 4:
        formula_head('4. Draw random values and decide which residues to mask',
                     'D/E compare an independent random number against the probability at each residue. F divides runs of matching annotation category into groups of at most eight residues, then shares one random number inside each group.')
        if experiment == 'F':
            st.latex(r'M_i = \mathbf{1}[u_{\mathrm{span}(i)} < q_i(t)]')
            st.write(f'**F span groups:** {frame["Span"].nunique()} · maximum 8 residues per group. The “Span” column is the 1-based anchor position.')
            st.caption('In the original implementation each span reuses the random draw at its first position. Groups are category-homogeneous, so masking is correlated within a group.')
        else:
            st.latex(r'M_i = \mathbf{1}[u_i < q_i(t)],\quad u_i\sim \mathcal{U}(0,1)')
        st.dataframe(frame[['Position', 'AA', 'Priority', 'α', 'q(t)', 'Random u', 'Span', 'Masked']].head(120),
                     hide_index=True, use_container_width=True)
        st.caption('Only the first 120 positions are shown. Download the complete table from Step 6.')

    if stage == 5:
        formula_head('5. Replace selected amino acids with the absorbing mask token',
                     'For every masked residue the input ID is replaced by MASK=32. The target remains the original amino-acid token. CLS and EOS stay intact.')
        st.latex(r'\widetilde{x}_i=\begin{cases}\mathrm{[MASK]}, & M_i=1\\ x_i, & M_i=0\end{cases}')
        show_sequence(frame, key=f'forward_window_{chosen}_{experiment}')
        st.write(f'**Original residues:** `{protein["sequence"][:70]}{"…" if n>70 else ""}`')
        st.write('**Corrupted (first 70):**')
        st.code(' '.join('[MASK]' if bool(row['Masked']) else row['AA'] for _, row in frame.head(70).iterrows()), language=None)

    if stage == 6:
        formula_head('6. Prepare exactly shaped input tensors for the denoiser',
                     'In the original training pipeline the corrupted tokens and biological context are passed to the Transformer. We can calculate their shapes and input values without executing trained weights.')
        rows = [
            ('input_ids', f'[1, {n+2}]', 'Corrupted token IDs'),
            ('attention_mask', f'[1, {n+2}]', 'One for CLS/residues/EOS; no padding here'),
            ('conditions', f'[1, {n+2}, 5]', 'ACT_SITE, BINDING, MOTIF, DOMAIN, REGION'),
            ('condition_available', '[1]', 'Evaluation gate = 1 (conditioning enabled)'),
            ('residue_mask', f'[1, {n+2}]', 'True for amino-acid positions only'),
        ]
        if experiment in ('E', 'F'):
            rows += [('diffusion_time', '[1]', f't = {time:.3f}'),
                     ('masking_exponents', f'[1, {n+2}]', 'α for residues; special token slots defined by collator'),
                     ('go_ids, ec_ids', '[1, N_go], [1, N_ec]', 'Requires training GO/EC vocabularies for real model indexing')]
        if experiment == 'F':
            rows += [('span_features', f'[1, {n+2}, 3]', 'Span length / 8, relative position, multi-residue indicator'),
                     ('observed_mask_fraction', '[1, 1]', f'{result.observed_fraction:.4f}')]
        st.dataframe(pd.DataFrame(rows, columns=['Model argument', 'Shape', 'Description']), hide_index=True, use_container_width=True)
        st.caption('No trained model forward pass has been made. Shape descriptions follow the original model interface; some GO/EC inputs require training vocabulary assets not included in this web application.')
        st.download_button('Download full forward calculations (CSV)', data=frame.to_csv(index=False).encode('utf-8'),
                           file_name=f'{chosen}_experiment_{experiment}_t{time:.2f}_seed{seed}.csv', mime='text/csv')
        st.download_button('Download forward trace (JSON)',
                           data=json.dumps(forward_record(protein, result), indent=2).encode('utf-8'),
                           file_name=f'{chosen}_experiment_{experiment}_trace.json', mime='application/json')

with page2:
    st.markdown('### Denoising architecture: what happens after masking?')
    st.info('This tab explains operations defined in the trained-model source code. It does **not** generate actual model logits, attention heatmaps, or a true reverse diffusion trajectory. Those require trained weights and a verified sampler.')
    reverse = st.radio('Inspect a stage', [1, 2, 3, 4, 5, 6], horizontal=True, key='denoiser_stage',
                       format_func=lambda x: {1: '① Inputs', 2: '② Embeddings', 3: '③ Attention',
                                              4: '④ Logits', 5: '⑤ Loss example', 6: '⑥ Reverse process'}[x])
    st.divider()

    if reverse == 1:
        formula_head('1. Inputs to the learned denoiser',
                     'The corrupted protein, attention mask and conditioning context are model inputs. The original clean protein is the training target, not an input to the prediction head.')
        st.write(f'**Current input:** {result.masked_count} masked amino acids / {n} total residues.')
        show_sequence(frame, key=f'reverse_window_{chosen}_{experiment}')
        st.code(f'input_ids.shape = [1, {n+2}]\nattention_mask.shape = [1, {n+2}]\nresidue_mask.shape = [1, {n+2}]', language='text')

    elif reverse == 2:
        formula_head('2. Embed tokens and add available biological context',
                     'The original ESM2 token embedding is enriched by experiment-specific learned projections before the 30 Transformer layers.')
        st.latex(r'h_i^{(0)} = \operatorname{ESM2Embedding}(\widetilde{x}_i)')
        st.latex(r'h_i \leftarrow h_i + g_i (W_{\mathrm{bio}} c_i + p_{\mathrm{presence}})')
        if experiment in ('E', 'F'):
            st.write('E/F additionally add protein-level mean-pooled GO/EC embeddings (where recognized) plus diffusion-time and exponent context.')
            st.latex(r'\phi(t)=[t,\ t^2,\ \sin(\pi t),\ \cos(\pi t)]')
            st.write(f'At **t={time:.2f}**, the time features are:')
            st.code(str([round(float(v), 5) for v in [time, time**2, math.sin(math.pi * time), math.cos(math.pi * time)]]), language=None)
            st.latex(r'h_i\leftarrow h_i + W_t\phi(t) + g_i W_{\alpha}(\alpha_i-1) + \mathrm{GO/EC}_i')
            st.caption('Terms are gated and applied only at residue positions as in the source implementation; this equation summarizes the additive terms.')
        if experiment == 'F':
            st.write('F also adds a learned projection of the observed masked fraction and a projection of three span features.')
            st.latex(r'\mathrm{span}_i=[\ell_i/8,\ (i-a_i)/\max(\ell_i-1,1),\ \mathbf{1}(\ell_i>1)]')
            st.write('Example of computed F span feature vectors (first 12 residues):')
            st.dataframe(pd.DataFrame(result.span_features[:12], columns=['Span length / 8', 'Relative position', 'Multi-residue']), hide_index=True, use_container_width=True)
        st.caption('All 640-dimensional numerical embeddings depend on checkpoint weights and are not calculated in this lightweight app.')

    elif reverse == 3:
        formula_head('3. Process representations through Transformer layers',
                     'The backbone contains 30 ESM2-derived Transformer layers, 20 attention heads per layer and RoPE positional encoding. D and E use the original attention implementation.')
        st.latex(r'Q=HW_Q,\quad K=HW_K,\quad V=HW_V')
        st.latex(r'A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+\mathrm{attention\ mask}\right),\qquad O=AV')
        st.write('For this model: hidden dimension **640**, 20 attention heads, and head size **32**.')
        if experiment == 'F':
            st.write('**Experiment F:** layers 1–20 retain regular attention; layers 21–30 use paired-head Differential Attention (10 pairs).')
            st.latex(r'A_{\mathrm{diff}}=A_1-\sigma(\lambda_{\mathrm{logit}})A_2')
            st.write('The combined values are RMS-normalized, scaled per paired head, then mixed with ordinary attention before the output projection. The saved training schedule reaches a mixture coefficient of 1 after 1,000 optimizer updates.')
            st.caption('This is the custom ESM2-compatible implementation, not an exact reproduction of the original Differential Transformer.')
        else:
            st.write('All 30 layers use the inherited ESM2 attention mechanism.')
        st.caption('Attention heatmaps require actual numerical Q/K/V and trained weights; they are intentionally not fabricated.')

    elif reverse == 4:
        formula_head('4. Predict amino-acid token logits',
                     'After the last Transformer layer, hidden states pass through final normalization and the DSM language-model head. Output scores cover 33 ESM2 vocabulary tokens.')
        st.latex(r'Z=\operatorname{LMHead}(\operatorname{LayerNorm}(H^{(30)}))')
        st.latex(r'Z_{\mathrm{capped}}=30\tanh(Z/30),\qquad P(v\mid\widetilde{x})=\operatorname{softmax}(Z_{\mathrm{capped}})_v')
        st.code(f'Expected logits: [1, {n+2}, 33]', language='text')
        st.warning('No numerical logits or amino-acid predictions are shown because no checkpoint is executed here. A predicted residue must not be confused with the known original target.')

    elif reverse == 5:
        formula_head('5. Training loss — a separate calculation from generation',
                     'The training objective evaluates the predicted probability of the original token at masked positions and weights it with the continuous-time diffusion factor.')
        st.latex(r'\mathcal{L}=\frac{1}{B L}\sum_{b,i}\mathbf{1}[M_{b,i}=1]\,\frac{q_{b,i}^{\prime}(t_b)}{q_{b,i}(t_b)+10^{-3}}\,\operatorname{CE}_{b,i}')
        pos_masked = frame.loc[frame['Masked'], 'Position'].astype(int).tolist()
        if not pos_masked:
            st.info('No residues were masked with this seed/time. Increase t to see a worked loss example.')
        else:
            chosen_pos = st.selectbox('Choose a masked residue', pos_masked,
                                      format_func=lambda p: f'Position {p} · true residue {protein["sequence"][p-1]}')
            row = frame.iloc[chosen_pos - 1]
            toy_prob = st.slider('Illustrative probability assigned to the correct residue',
                                 min_value=0.01, max_value=0.99, value=0.60, step=0.01)
            ce = -math.log(toy_prob)
            weight = float(row['Loss weight'])
            contrib = ce * weight / (n + 2)
            st.write(f'At position **{chosen_pos}**: q = **{row["q(t)"]:.4f}**, q′ = **{row["q\'(t)"]:.4f}**, diffusion weight = **{weight:.4f}**.')
            st.write(f'Teaching-only CE = −ln({toy_prob:.2f}) = **{ce:.4f}**; normalized contribution = **{contrib:.6f}** for batch size 1 and token length {n+2}.')
            st.caption('The chosen probability is a teaching input, not a prediction produced by the trained checkpoint. The actual loss requires genuine model logits for every masked residue.')

    elif reverse == 6:
        formula_head('6. Reverse diffusion / reconstruction',
                     'The trained denoiser estimates token probabilities. An actual multi-step reverse process additionally requires a defined sampling algorithm; we have not verified and packaged one.')
        st.write('**Optional teaching illustration:** reveal known original residues one by one. This demonstrates how a masked sequence could become visible again, but it is **not** model inference, not sampled reverse diffusion, and not evidence of generation quality.')
        positions = np.flatnonzero(frame['Masked'].to_numpy())
        if len(positions):
            reveal_step = st.slider('Number of masked positions revealed using the known target',
                                    min_value=0, max_value=len(positions), value=0, step=1)
        else:
            reveal_step = 0
            st.info('No positions were masked at this time and seed; increase t to explore the reveal demonstration.')
        order = np.random.default_rng(int(seed) + 12345).permutation(positions)
        remain = frame['Masked'].to_numpy().copy()
        remain[order[:reveal_step]] = False
        show_sequence(frame, key=f'oracle_{chosen}_{experiment}', label='Teaching illustration window', reveal_mask=remain)
        st.caption(f'{reveal_step}/{len(positions)} originally masked residues revealed from ground truth. This is not a trained model prediction.')

with page3:
    st.markdown('### Same protein and time, three different corruption rules')
    st.write('All three calculations use the same NumPy seed. The shared random vector enables reproducible comparisons, while Experiment F correlates draws inside spans.')
    comparison = all_experiments(protein, float(time), int(seed))
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.bar_chart(comparison.set_index('Experiment')[['Masked %']])
    weights = pd.DataFrame({'Position': np.arange(1, n + 1)})
    for letter in 'DEF':
        r = calculate(protein, letter, float(time), int(seed))
        weights[f'{letter} probability'] = r.frame['q(t)'].to_numpy()
    st.write('#### Per-residue masking probability comparison')
    st.line_chart(weights.set_index('Position'))
    st.caption('Differences in sampled counts are stochastic. The expected counts are sums of the respective position-specific masking probabilities, not validation metrics.')
    st.download_button('Download D/E/F comparison (CSV)', comparison.to_csv(index=False),
                       file_name=f'{chosen}_comparison_t{time:.2f}.csv', mime='text/csv')

with page4:
    st.markdown('### Scientific provenance and limitations')
    st.markdown('''
**Exactly calculated here**

- ESM2-style token IDs for three selected full-length proteins, with CLS, EOS and MASK special tokens.
- All five supported residue-level annotation channels; annotation priority and overlaps.
- D: uniform independent Bernoulli masking, `q(t)=t` and `q′(t)=1`.
- E: independent Bernoulli masking with exponents 1, 1.25, 1.5, 2.0.
- F: functional-first exponents 1.40, 0.95, 0.75, 0.55, category-based spans up to eight residues, shared random draw per span.
- Continuous-time weights `q′/(q+0.001)` and the F span-feature vectors.

**Not computed in this repository**

- Learned 640-dimensional embeddings, actual model attention weights or token logits.
- Actual trained-model cross-entropy, optimizer gradients or reverse-time sampling trajectories.
- GO/EC label indices from the original training vocabularies (which must not be guessed from public ontology labels).

The dashboard uses NumPy RNG for lightweight deployment, so sampled masks are **distribution-equivalent** but not bit-identical to PyTorch `torch.rand` with the same numeric seed. This is intentional and is noted in every JSON export.
''')
    st.markdown('#### Original implementation files')
    st.code('''Experiment D: 04_bio_corruptor.py, 05_bio_loss.py, 32_experiment_d_conditions.py, 33_experiment_d_model.py
Experiment E: 43_experiment_e_delayed_corruptor.py, 44_experiment_e_context_model.py
Experiment F: 52_experiment_f_corruptor.py, 53_experiment_f_differential_attention.py,
              54_experiment_f_context_model.py, 55_experiment_f_span_features.py
Shared DSM150: dsm150_model.py and ESM2 backbone/attention/LM-head modules''', language='text')
    st.markdown('#### Protein references')
    for p in PROTEINS:
        st.markdown(f'- [{p["id"]} · {p["name"]}]({p["sequence_source"]}) ({len(p["sequence"])} amino acids)')
    st.markdown('- [ESM2 vocabulary](https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/vocab.txt)')
    st.markdown('- [UniProt REST API](https://www.uniprot.org/help/api_queries)')
    st.caption('Bundled coordinates use 1-based inclusive UniProt positions. The online refresh returns only exact-location features of the five supported types.')

st.divider()
st.caption('CTDD · Lightweight didactic dashboard · D / E / F · No GPU, checkpoints, training, or GitHub connection required')
