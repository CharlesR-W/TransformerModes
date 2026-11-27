"""Freeze the nine synthetic inputs from the 32 Stage A texts (SPEC_DEPTH §2)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probe import (CONTEXT_TOKENS, FORBIDDEN_IDS, MODELS, POOL_SIZE, PROBE_SCHEMA_VERSION,
                   model_input_ids, write_json_atomic)

PROBE_PATH = 'probe/probe.json'
# Probe identity recorded by both Stage A extraction receipts, not a hash of a regenerated probe.
PROBE_SHA256 = 'e6beef6cf07ad0a0492072dd2fc46d785696341fe666bd58b6de9953a2d74d4d'
LOCAL_PROBE = Path('/data/tm_acts') / PROBE_PATH
LOCAL_INPUTS = Path('/data/tm_acts/depth_acts/inputs.json')
OUTPUT_ROOT = 'acts_depth'
INPUTS_PATH = f'{OUTPUT_ROOT}/inputs.json'
SEED_RANDOM, SEED_SHUFFLE = 20261126, 20261127
RANDOM_KEYS = tuple(f'r{n}' for n in range(5))
SHUFFLE_SOURCES = (0, 1, 2, 4)
KEYS = RANDOM_KEYS + tuple(f's{n}' for n in range(len(SHUFFLE_SOURCES)))


def ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i4').tobytes()).hexdigest()


def check_probe_sha(raw, expected=PROBE_SHA256):
    if raw is None or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('missing probe or SHA-256 differs from the Stage A extraction receipts')


def probe_documents(probe):
    documents = probe['contexts'][:POOL_SIZE]  # Do not include the four stitched controls.
    if (probe['schema_version'] != PROBE_SCHEMA_VERSION or probe['models'] != MODELS
            or [doc['id'] for doc in documents] != list(range(POOL_SIZE))):
        raise ValueError('expected the pinned Stage A probe with documents 0..31')
    for doc in documents:
        ids = doc['token_ids']
        if (len(ids) != CONTEXT_TOKENS or any(type(i) is not int or not 0 <= i < 2**31 for i in ids)
                or FORBIDDEN_IDS.intersection(ids) or doc['model_input_ids'] != model_input_ids(ids)):
            raise ValueError(f"invalid Stage A token ids or separator for context {doc['id']}")
    allowed = sorted({i for doc in documents for i in doc['token_ids']})
    if len(allowed) != 9490:
        raise ValueError(f'expected 9490 distinct document ids, found {len(allowed)}')
    return documents, allowed


def build_inputs(documents, allowed):
    draws = np.random.default_rng(SEED_RANDOM).choice(allowed, size=(5, CONTEXT_TOKENS), replace=True)
    contexts = {key: {'ids': row.tolist()} for key, row in zip(RANDOM_KEYS, draws, strict=True)}
    for n, source in enumerate(SHUFFLE_SOURCES):
        ids = documents[source]['token_ids']
        perm = np.random.default_rng(SEED_SHUFFLE + n).permutation(CONTEXT_TOKENS)
        contexts[f's{n}'] = dict(ids=[ids[j] for j in perm], source_context=source, perm=(perm + 1).tolist())
    return dict(seed_random=SEED_RANDOM, seed_shuffle=SEED_SHUFFLE, allowed_ids_sha256=ids_sha256(allowed),
                contexts=contexts, sha256={key: ids_sha256(context['ids']) for key, context in contexts.items()})


def validate_frozen(frozen, documents, allowed):
    expected = build_inputs(documents, allowed)
    # JSON comparison also distinguishes ints from floats and booleans.
    if json.dumps(frozen, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('frozen inputs do not match SPEC_DEPTH §2 and the Stage A probe')


def write_inputs(probe_path, output, *, expected_sha256=PROBE_SHA256):
    raw = Path(probe_path).read_bytes()
    check_probe_sha(raw, expected_sha256)
    documents, allowed = probe_documents(json.loads(raw))
    frozen = build_inputs(documents, allowed)
    write_json_atomic(output, frozen)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, default=LOCAL_PROBE)
    parser.add_argument('--output', type=Path, default=LOCAL_INPUTS)
    args = parser.parse_args()
    frozen = write_inputs(args.probe, args.output)
    print(json.dumps(dict(allowed_ids_sha256=frozen['allowed_ids_sha256'], **frozen['sha256']), indent=2))
    print(f'{args.output}: sha256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
