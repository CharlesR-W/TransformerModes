"""Extract depth-only residuals using the pinned Stage A forward path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from depth_inputs import INPUTS_PATH, KEYS, OUTPUT_ROOT, PROBE_PATH, PROBE_SHA256, check_probe_sha, probe_documents, validate_frozen
from probe import CONTEXT_TOKENS, MODELS, TokenIdentityError, model_input_ids, tokenizer_provenance, write_json_atomic

ACT_SHAPE = (17, 1 + CONTEXT_TOKENS, 2048)
PARITY_ATOL = PARITY_RTOL = 1e-6  # float32-scale slack; no layer-average or vector norm can hide a bad entry


def output_dir(mount, output_root, model_key):
    if Path(output_root) != Path(OUTPUT_ROOT) or model_key not in MODELS:
        raise ValueError('depth extraction may write only to acts_depth/<model>/')
    parent = Path(mount) / OUTPUT_ROOT
    dest = parent / model_key
    if parent.is_symlink() or dest.is_symlink():
        raise ValueError('symlinks are not allowed in the depth output directory')
    return dest


def assert_model_inputs(frozen_ids, inputs, allowed, tokenizers):
    """Validate every context and both variants before any forward pass."""
    if not frozen_ids or inputs.keys() != frozen_ids.keys() or tokenizers.keys() != MODELS.keys():
        raise TokenIdentityError('incomplete inputs or missing a pinned tokenizer')
    allowed_set = set(allowed)
    for key, ids in frozen_ids.items():
        if len(ids) != CONTEXT_TOKENS or any(type(i) is not int or i not in allowed_set for i in ids):
            raise TokenIdentityError(f'{key}: token id missing from the allowed vocabulary, or wrong length/type')
        if inputs[key] != model_input_ids(ids):
            raise TokenIdentityError(f'{key}: model input does not match separator plus frozen ids')
    for name, tok in tokenizers.items():
        if tok.eos_token_id != MODELS[name]['eot']:
            raise TokenIdentityError(f'{name}: incorrect pinned separator')
    # Stronger than checking only sampled ids, and still just two vocabulary lookups.
    first, second = (tok.convert_ids_to_tokens(allowed) for tok in tokenizers.values())
    if (not isinstance(first, list) or len(first) != len(allowed) or first != second
            or any(not isinstance(s, str) for s in first)):
        raise TokenIdentityError('pinned tokenizer vocabulary strings differ or are absent')


def check_c00(actual, reference, model_key):
    if any(a.shape != ACT_SHAPE or a.dtype != np.float32 for a in (actual, reference)):
        raise ValueError('c00 parity requires two float32 activation arrays of the expected shape')
    maxima, failed = [], []
    for layer, (a, b) in enumerate(zip(actual, reference, strict=True)):
        diff = np.abs(a - b)
        maxima.append(float(diff.max()))
        if not np.all(diff <= PARITY_ATOL + PARITY_RTOL * np.abs(b)) or not np.isfinite(b).all():
            failed.append(layer)
    print(f'{model_key} c00 max_abs_by_layer={maxima}', flush=True)
    if failed:
        raise ValueError(f'{model_key}: c00 differs from Stage A above float32 tolerance at layers {failed}')
    return dict(passed=True, exact=not any(maxima), atol=PARITY_ATOL, rtol=PARITY_RTOL, max_abs_by_layer=maxima)


def extract_model(model_key, mount, inputs_sha256, *, output_root=OUTPUT_ROOT, commit):
    dest = output_dir(mount, output_root, model_key)  # no torch import, let alone GPU work, for a refused root
    import torch
    import extract
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    if transformers_version != extract.TRANSFORMERS_VERSION or torch.__version__.split('+')[0] != extract.TORCH_VERSION:
        raise RuntimeError('depth extraction requires the pinned Stage A library versions')
    if not torch.cuda.is_available():
        raise RuntimeError('depth extraction requires CUDA')
    mount = Path(mount)
    probe_bytes = (mount / PROBE_PATH).read_bytes()
    check_probe_sha(probe_bytes, PROBE_SHA256)
    documents, allowed = probe_documents(json.loads(probe_bytes))
    frozen_bytes = (mount / INPUTS_PATH).read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != inputs_sha256:
        raise ValueError('Volume inputs.json differs from the launch reservation')
    frozen = json.loads(frozen_bytes)
    validate_frozen(frozen, documents, allowed)
    ids = {key: frozen['contexts'][key]['ids'] for key in KEYS}
    ids['c00'] = documents[0]['token_ids']
    inputs = {key: model_input_ids(row) for key, row in ids.items()}

    def pinned(meta):
        return dict(revision=meta['revision'], cache_dir=mount / 'hf', trust_remote_code=False, token=False,
                    local_files_only=True)

    tokenizers = {key: AutoTokenizer.from_pretrained(meta['id'], use_fast=True, **pinned(meta))
                  for key, meta in MODELS.items()}
    assert_model_inputs(ids, inputs, allowed, tokenizers)
    if inputs['c00'] != documents[0]['model_input_ids']:
        raise TokenIdentityError('c00 input differs from the Stage A probe')
    ref_path = Path('acts') / model_key / 'c00.npy'
    ref = np.load(mount / ref_path, mmap_mode='r', allow_pickle=False)
    if ref.shape != ACT_SHAPE or ref.dtype != np.float32:
        raise ValueError(f'{ref_path} is not a Stage A float32 activation array')

    meta = MODELS[model_key]
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(meta['id'], dtype=torch.float32, attn_implementation='eager',
                                                 use_safetensors=True, **pinned(meta)).to('cuda')
    if (model.config.num_hidden_layers, model.config.hidden_size, model.config.eos_token_id) != (16, 2048, meta['eot']):
        raise ValueError('loaded model does not match the pinned architecture or separator')
    extract.set_precision()
    model.eval()
    weight = model.get_input_embeddings().weight
    print(f'{model_key} loaded on {weight.device} in {time.monotonic() - start:.2f}s', flush=True)

    def forward(key):
        # Same Stage A residual hooks and base-model settings as its causal check; no unused readouts.
        acts, _, _ = extract.forward(model, inputs[key][model_key], verify_hooks=True, readouts=False)
        if acts.shape != ACT_SHAPE:
            raise ValueError(f'{key}: unexpected activation shape {acts.shape}')
        return acts

    start = time.monotonic()
    parity = check_c00(forward('c00'), ref, model_key)
    del ref
    parity.update(reference=str(ref_path), seconds=time.monotonic() - start)
    print(f"{model_key} c00 self-check passed in {parity['seconds']:.2f}s", flush=True)

    timings = {}
    for key in KEYS:
        start = time.monotonic()
        acts = forward(key)
        forwarded = time.monotonic()
        extract._save_atomic(dest / f'{key}.npy', np.save, acts, allow_pickle=False)
        del acts
        saved = time.monotonic()
        print(f'{model_key} {key} forward={forwarded - start:.2f}s save={saved - forwarded:.2f}s; committing',
              flush=True)
        commit()
        timings[key] = dict(forward_seconds=forwarded - start, save_seconds=saved - forwarded,
                            commit_seconds=time.monotonic() - saved)
        print(f"{model_key} {key} committed in {timings[key]['commit_seconds']:.2f}s", flush=True)

    report = dict(
        model=dict(name_or_path=str(model.name_or_path), class_name=type(model).__name__, revision=meta['revision'],
                   commit_metadata=getattr(model.config, '_commit_hash', None), eos_token_id=meta['eot']),
        tokenizer=tokenizer_provenance(tokenizers[model_key], meta),
        token_identity={key: tokenizer_provenance(tok, MODELS[key]) for key, tok in tokenizers.items()},
        libraries=dict(transformers=transformers_version, torch=torch.__version__, numpy=np.__version__,
                       cuda=torch.version.cuda),
        forward_pass=dict(eval=not model.training, dtype=str(weight.dtype), device=str(weight.device), batch_size=1,
                          attention_implementation=model.config._attn_implementation,
                          matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                          readouts=False),
        contexts=len(KEYS), hook_equality=dict(passed=True, exact=True, contexts=['c00', *KEYS], layers=list(range(1, 16))),
        c00_parity=parity, timings=timings, probe_sha256=hashlib.sha256(probe_bytes).hexdigest(),
        inputs_sha256=inputs_sha256, allowed_ids_sha256=frozen['allowed_ids_sha256'], context_sha256=frozen['sha256'],
    )
    write_json_atomic(dest / 'provenance.json', report)
    commit()
    return dict(contexts=len(KEYS), provenance_path=f'/{OUTPUT_ROOT}/{model_key}/provenance.json', c00_parity=parity)


def run_unit(model_key, mount, inputs_sha256, volume):
    """Modal worker. Catch ordinary failures to retain measured execution seconds."""
    start = time.monotonic()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'  # before importing Torch/Transformers in extract_model
        volume.reload()
        result = extract_model(model_key, mount, inputs_sha256, commit=volume.commit)
    except Exception as error:
        traceback.print_exc()
        return dict(status='failed', seconds=time.monotonic() - start, error=f'{type(error).__name__}: {error}')
    return dict(status='done', seconds=time.monotonic() - start, **result)
