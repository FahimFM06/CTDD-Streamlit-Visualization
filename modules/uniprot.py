"""Optional UniProtKB annotation refresh; local validated examples always work offline."""
from __future__ import annotations
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
