"""U1: extract raw residual streams, next-token predictions and attention for one model.

Array row u always corresponds to input position u, including the separator at
0 (stored, not in the probe's node sets or attention query ranges). Predictions
at u predict u+1, even for the final position. State k>0 is the raw stream just
after decoder block k, before the final model norm at the last block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from probe import (CONTEXT_TOKENS, FORBIDDEN_IDS, MODELS, POOL_SIZE, PROBE_SCHEMA_VERSION, QUERY_START, SUFFIX_START,
                   tokenizer_provenance, write_json_atomic)

TRANSFORMERS_VERSION = "4.57.6"
TORCH_VERSION = "2.5.1"
CAUSAL_SPLIT = 1024


def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


def set_precision():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def attention_sums(weights, query_start):
    """Reduce a [1, heads, query, key] eager matrix in bounded-memory query blocks.

    Distance is a mean *of ratios*, one ratio per query and head. Exact zero
    non-sink mass (possible in finite precision) is counted separately rather
    than silently treated as zero distance.
    """
    if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[2] != weights.shape[3]:
        raise ValueError("expected square eager attention weights for one context")
    a = weights[0]
    heads, seq, _ = a.shape
    sink = a[:, query_start:, 0].sum(dim=-1, dtype=torch.float64)
    distance = torch.zeros_like(sink)
    valid_count = torch.zeros(heads, dtype=torch.int64, device=a.device)
    keys = torch.arange(1, seq, device=a.device)
    for start in range(query_start, seq, 128):
        stop = min(start + 128, seq)
        non_sink = a[:, start:stop, 1:]
        offsets = (torch.arange(start, stop, device=a.device)[:, None] - keys).to(a.dtype)
        mass = non_sink.sum(dim=-1)
        valid = mass > 0
        numerator = (non_sink * offsets).sum(dim=-1)
        ratio = numerator / torch.where(valid, mass, 1.0)
        distance += torch.where(valid, ratio, 0.0).sum(dim=-1, dtype=torch.float64)
        valid_count += valid.sum(dim=-1)
    return {
        "sink_mass_sum": sink,
        "distance_sum": distance,
        "query_count": torch.full_like(valid_count, max(0, seq - query_start)),
        "distance_count": valid_count,
    }


def predictions(logits):
    """Top 20 mass from the full vocabulary softmax, without renormalising."""
    prob = logits[0].float().softmax(dim=-1)
    values, indices = prob.topk(20, dim=-1, largest=True, sorted=True)
    ids = indices.to(torch.int32).cpu().numpy()
    result = dict(top20_ids=ids, top20_probs=values.cpu().numpy(), top1_id=ids[:, 0].copy(),
                  entropy=torch.special.entr(prob).sum(dim=-1).cpu().numpy())
    if not np.isfinite(result["entropy"]).all():
        raise ValueError("non-finite next-token entropy")
    return result


@torch.inference_mode()
def forward(model, input_ids, *, verify_hooks=False, readouts=True, query_start=QUERY_START):
    """One batch of one context; return residuals and optional per-position/head readouts.

    Eager attention modules return their weights even with output_attentions
    disabled in the pinned implementations. Reduce each matrix in its hook
    so we never hold a whole model's attention matrices simultaneously.
    """
    if model.config._attn_implementation != "eager" or model.training:
        raise ValueError("extraction requires an eval-mode eager model")
    embedding = model.get_input_embeddings()
    if embedding.weight.dtype != torch.float32:
        raise ValueError("extraction requires float32 weights")
    blocks = model.base_model.layers  # GPTNeoXModel and OlmoModel, in the pinned Transformers
    ids = torch.as_tensor(input_ids, device=embedding.weight.device, dtype=torch.long).reshape(1, -1)
    residuals = {}
    attention = {}

    def capture_residual(layer):
        def hook(_module, _args, output):
            residuals[layer] = _tensor(output)
        return hook

    def capture_attention(layer):
        def hook(_module, _args, output):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise RuntimeError("eager attention hook did not return a weight matrix")
            attention[layer] = attention_sums(output[1], query_start)
        return hook

    handles = [embedding.register_forward_hook(capture_residual(0))]
    try:
        for layer, block in enumerate(blocks, 1):
            handles.append(block.register_forward_hook(capture_residual(layer)))
            if readouts:
                handles.append(block.attention.register_forward_hook(capture_attention(layer))
                               if model.config.model_type == "gpt_neox"
                               else block.self_attn.register_forward_hook(capture_attention(layer)))
        out = (model if readouts else model.base_model)(ids, use_cache=False, return_dict=True,
                                                       output_hidden_states=verify_hooks, output_attentions=False)
    finally:
        for handle in handles:
            handle.remove()
    if verify_hooks:
        if out.hidden_states is None or len(out.hidden_states) != len(blocks) + 1:
            raise AssertionError("model did not return the expected hidden-state oracle")
        for layer in range(1, len(blocks)):
            if not torch.equal(residuals[layer], out.hidden_states[layer]):
                raise AssertionError(f"raw hook and output_hidden_states disagree at layer {layer}")
    acts = torch.stack([residuals[k][0] for k in range(len(blocks) + 1)]).cpu().numpy()
    if acts.dtype != np.float32 or not np.isfinite(acts).all():
        raise ValueError("non-float32 or non-finite residual stream")
    if not readouts:
        return acts, None, None
    stats = {name: torch.stack([attention[k][name] for k in range(1, len(blocks) + 1)]).cpu().numpy()
             for name in attention[1]}
    if any(not np.isfinite(x).all() for x in stats.values()):
        raise ValueError("non-finite attention statistics")
    return acts, predictions(out.logits), stats


def _relative_l2(reference, other):
    """Vector L2 change at each (state, position), relative to the original."""
    return np.linalg.norm(reference - other, axis=-1) / np.maximum(np.linalg.norm(reference, axis=-1), 1e-30)


def causal_check(model, inputs, reference, split):
    """§2 equal-shape suffix swap (hard); separate truncated-prefix diagnostic."""
    original = inputs[0]
    if not 0 < split < len(original):
        raise ValueError("causal split must be inside context 0")
    donor = next((i for i in sorted(inputs) if i != 0 and len(inputs[i]) == len(original)
                  and inputs[i][split] != original[split]), None)
    if donor is None:
        raise AssertionError("no same-length suffix with a different first token was found")
    changed = original[:split] + inputs[donor][split:]
    if len(changed) != len(original) or changed[split] == original[split]:
        raise AssertionError("invalid equal-shape causal test input")
    swap, _, _ = forward(model, changed, readouts=False)
    prefix = _relative_l2(reference[:, :split], swap[:, :split]).max(axis=1)
    boundary = _relative_l2(reference[:, split:split + 1], swap[:, split:split + 1])[:, 0]
    if not (np.all(prefix <= 1e-5) and np.all(boundary > 1e-3)):
        raise AssertionError(f"equal-shape causal check failed: prefix {prefix}, boundary {boundary}")
    del swap
    trunc, _, _ = forward(model, original[:split], readouts=False)
    return dict(split=split, donor_context_id=donor, prefix_max_relative_l2=prefix.tolist(),
                boundary_relative_l2=boundary.tolist(),
                truncated_prefix_max_relative_l2=_relative_l2(reference[:, :split], trunc).max(axis=1).tolist())


def output_paths(root, model_key, context_ids):
    root = Path(root)
    acts = {c: root / "acts" / model_key / f"c{c:02d}.npy" for c in context_ids}
    pred = {c: root / "pred" / model_key / f"c{c:02d}.npz" for c in context_ids}
    return acts, pred, root / "attn" / f"{model_key}.npz"


def _save_atomic(path, save, *args, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        tmp = Path(output.name)
        try:
            save(output, *args, **kwargs)
            output.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def extract_contexts(model, contexts, eot, root, model_key, *, query_start=QUERY_START, causal_split=CAUSAL_SPLIT):
    """Extract using a loaded model, including on CPU for tests."""
    contexts = list(contexts)
    inputs = {c["id"]: [eot, *c["token_ids"]] for c in contexts}
    if 0 not in inputs or len(inputs) != len(contexts) or query_start < 1:
        raise ValueError("expected unique contexts including 0 and a non-separator query start")
    set_precision()
    model.eval()
    context_ids = sorted(inputs)
    act_files, pred_files, attn_file = output_paths(root, model_key, context_ids)
    attention = []
    for c in context_ids:
        acts, pred, sums = forward(model, inputs[c], verify_hooks=(c == 0), query_start=query_start)
        if c == 0:
            diagnostic = causal_check(model, inputs, acts, causal_split)
        _save_atomic(act_files[c], np.save, acts, allow_pickle=False)
        _save_atomic(pred_files[c], np.savez, **pred)
        attention.append(sums)
        del acts, pred
    by_context = {name: np.stack([a[name] for a in attention]) for name in attention[0]}
    _save_atomic(attn_file, np.savez, context_ids=np.asarray(context_ids, dtype=np.int32),
                 query_start=np.int32(query_start), layer_ids=np.arange(1, len(model.base_model.layers) + 1),
                 **{name: value.sum(axis=0) for name, value in by_context.items()},
                 **{f"context_{name}": value for name, value in by_context.items()})
    return dict(contexts=len(contexts), hook_oracle_layers=list(range(1, len(model.base_model.layers))),
                causal=diagnostic)


def main(model_key, root, cache_dir, *, device="cuda"):
    from transformers import AutoModelForCausalLM, __version__ as transformers_version

    if transformers_version != TRANSFORMERS_VERSION or torch.__version__.split("+")[0] != TORCH_VERSION:
        raise RuntimeError(f"U1 requires Transformers {TRANSFORMERS_VERSION} and torch {TORCH_VERSION}")
    root = Path(root)
    model_meta = MODELS[model_key]
    probe_data = json.loads((root / "probe" / "probe.json").read_text(encoding="utf-8"))
    contexts = probe_data["contexts"]
    if (probe_data["models"] != MODELS or [c["id"] for c in contexts] != list(range(POOL_SIZE + 4))
            or any(len(c["token_ids"]) != CONTEXT_TOKENS or FORBIDDEN_IDS.intersection(c["token_ids"])
                   for c in contexts)):
        raise ValueError("probe does not match the pinned extraction inputs")
    model = AutoModelForCausalLM.from_pretrained(model_meta["id"], revision=model_meta["revision"], cache_dir=cache_dir,
                                                 dtype=torch.float32, attn_implementation="eager",
                                                 use_safetensors=True, trust_remote_code=False).to(device)
    if (model.config.num_hidden_layers, model.config.hidden_size, model.config.eos_token_id) != (16, 2048, model_meta["eot"]):
        raise ValueError("loaded model does not match the pinned architecture or separator")
    report = extract_contexts(model, contexts, model_meta["eot"], root, model_key)
    return dict(model=model_meta, transformers=transformers_version, torch=torch.__version__, **report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("root", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(main(args.model, args.root, args.cache_dir, device=args.device))
