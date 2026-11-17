# CTDD Protein Diffusion Explorer — Experiments D, E, F

A lightweight, interactive **CPU-only Streamlit dashboard** for explaining how three DSM150-based protein-sequence diffusion experiments work. Designed for teaching, supervision meetings, and thesis demonstrations.

## Launch online (GitHub → Streamlit Community Cloud)

1. **Download and extract** `CTDD_Streamlit_GitHub_Ready.zip` from ChatGPT.
2. Open your repository: <https://github.com/FahimFM06/CTDD-Streamlit-Visualization>.
3. Select **Add file → Upload files** and upload the **contents** of the extracted `CTDD-Streamlit-Visualization` folder — not the ZIP itself. The repository root must contain `app.py` and `requirements.txt`, together with the `data`, `modules`, and `tests` folders. Commit the files.
4. Open <https://share.streamlit.io>, sign in with your GitHub account, select that repository, and set the main file path to `app.py`.
5. Select **Deploy**. Share the resulting `*.streamlit.app` URL with your supervisor.

**No HPC session, GPU, model checkpoint, GitHub connector, or retraining is needed for the app.** A public repository is usually simplest for Streamlit Community Cloud.

## Run locally (optional)

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Run unit tests with:

```bash
python -m pip install pytest
python -m pytest -q
```

## What it does

Choose among **three real human UniProt proteins**:

| UniProt accession | Protein | Sequence length |
|---|---|---:|
| P61626 | Lysozyme C | 148 aa |
| P00918 | Carbonic anhydrase II | 260 aa |
| P10635 | Cytochrome P450 2D6 | 497 aa |

Then choose **Experiment D, E, or F**, a diffusion time `t`, and a NumPy random seed. The app displays the same six explanatory forward stages for all nine combinations: tokenization, diffusion time, masking probabilities, random masking, MASK substitution, and model input shapes. It also contains six *denoising explanation* pages and a D/E/F comparison.

**Optional:** Tick *Fetch current UniProt annotations* to query UniProtKB's JSON API for a full set of compatible residue-level annotations. Local selected reference annotations are included to make the dashboard usable without network connectivity.

## Original masking calculations, faithfully translated to NumPy

| Variant | Masking probability `q_i(t)` | Random sampling | Other conditioning |
|---|---|---|---|
| D | `t` for every residue | Independent per residue | Five residue-level biological channels |
| E | `t ** alpha_i`, with 1.0/1.25/1.5/2.0 | Independent per residue | Biological channels, GO/EC, time, exponent |
| F | `t ** alpha_i`, with 1.40/0.95/0.75/0.55 | Correlated per annotation-category span (max 8 residues) | E features plus span and observed-mask features; Differential Attention in layers 21–30 |

The exponent derivative is `q'_i(t) = alpha_i * t**(alpha_i - 1)`. Position-specific diffusion weight: `q'_i(t) / (q_i(t) + 0.001)`.

Priority for overlapping annotations: `ACT_SITE > BINDING > MOTIF > DOMAIN > REGION > ordinary`.

These details were checked against the user's original Experiment D/E/F source archive (source script names listed inside the dashboard). `modules/diffusion.py` uses NumPy rather than PyTorch to keep deployment small. The **probabilities and span grouping match the original equations**, but random masks with the same seed are **not bit-for-bit identical** to PyTorch's `torch.rand` results. The time slider is a chosen value, whereas training samples time.

## Scientific limits (important)

- This application **does not load or execute** the ~150M-parameter DSM150 neural networks.
- It does **not** show trained token predictions, learned attention heatmaps, actual cross-entropy or sampled reverse diffusion trajectories.
- Denoiser pages explain the verified network operations and formulas. The optional step-by-step residue reveal uses the **known original sequence**; it is a clearly labeled **teaching illustration, not generative inference**.
- GO/EC descriptions are not the same thing as the integer labels from the original training vocabularies. The app does not invent those vocabulary indices.
- The bundled features are **selected residue annotations**, not exhaustive. P61626 and P00918 are based on UniProt; selected documented metal/heme coordination residues for P00918 and P10635 are additionally based on structural literature. Live UniProt extraction provides an optional more complete source-based annotation set.
- Do not compare the three published training validation losses as if they measured an identical masking distribution.

For actual trained model logits and a verified reverse sampler, a separate offline capture script could eventually produce small `.npz`/JSON traces to attach to this dashboard. That capture step is **not required** to use the current teaching app.

## Project files

```text
CTDD-Streamlit-Visualization/
├── app.py                      # The complete Streamlit interface
├── requirements.txt           # Four lightweight runtime packages
├── data/
│   └── proteins.json          # Three sequences + selected annotations
├── modules/
│   ├── __init__.py
│   ├── diffusion.py            # D/E/F masking, derivatives, spans, exports
│   └── uniprot.py              # Optional UniProtKB annotation refresh
├── tests/
│   └── test_diffusion.py       # Reproducibility and equation tests
└── README.md
```

## Sources

- UniProtKB [P61626](https://www.uniprot.org/uniprotkb/P61626/entry), [P00918](https://www.uniprot.org/uniprotkb/P00918/entry), [P10635](https://www.uniprot.org/uniprotkb/P10635/entry).
- [ESM2 tokenizer vocabulary](https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/vocab.txt).
- [CYP2D6 crystal structure overview](https://www.rcsb.org/structure/2F9Q).
- Original project source: `Experiment_D/scripts/04_bio_corruptor.py`, `Experiment_E/scripts/43_experiment_e_delayed_corruptor.py`, `Experiment_F/scripts/52_experiment_f_corruptor.py`, `Experiment_F/scripts/55_experiment_f_span_features.py`, `Experiment_F/scripts/53_experiment_f_differential_attention.py`, plus shared DSM150 architecture and training loss implementations.
