"""U0: select the Pile probe and build stitched controls without re-tokenising joins.

All token lists exclude the separator; model positions therefore start at 1 in
these lists. Node and segment ranges are half-open. Pile line numbers and
source token offsets are zero-based. Network access is confined to main().
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import tempfile

MODELS = {
    "pythia": dict(id="EleutherAI/pythia-1b", revision="f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2", eot=0),
    "olmo": dict(id="allenai/OLMo-1B-0724-hf", revision="d7cbab742d80589e714b1a2d7f838dcd21cbe143", eot=50279),
}
CORPUS = dict(id="monology/pile-uncopyrighted", revision="3be90335b66f24456a5d6659d9c8d208c0357119",
              filename="test.jsonl.zst", size_bytes=337_771_483)
FORBIDDEN_TEXT = ("<|endoftext|>", "<|padding|>", "|||IP_ADDRESS|||", "|||EMAIL_ADDRESS|||", "|||PHONE_NUMBER|||", "\0")
FORBIDDEN_IDS = frozenset((0, 1, 50277, 50278, 50279))
INVALID_DISPLAY = "�"  # ByteLevel's standard Unicode replacement character.
POOL_SIZE, CONTEXT_TOKENS, SUBSET_CAP = 32, 2047, 4
PROBE_SCHEMA_VERSION = 2
QUERY_START, SUFFIX_START = 256, 1536
STITCH_LENGTHS = {"long": (384, 640), "short": (48, 80)}
REASONS = ("forbidden_text", "ascii", "too_short", "whitespace", "internal_32gram", "shared_32gram",
           "subset_cap", "stitch_same_subset")


@dataclass(frozen=True)
class Document:
    line: int
    subset: str
    text: str


class TokenIdentityError(ValueError):
    """A required tokenizer invariant failed; the whole unit must fail."""


def _encode(tokenizer, text):
    # A Pile document can legitimately exceed the model context; do not truncate.
    return tokenizer.encode(text, add_special_tokens=False, verbose=False)


def assert_token_identity(text, pythia, olmo, *, pythia_ids=None):
    """Check all §2 invariants on the full source text; return the shared ids."""
    if any(s in text for s in FORBIDDEN_TEXT):
        raise TokenIdentityError("forbidden source text literal")
    ids = _encode(pythia, text) if pythia_ids is None else pythia_ids
    if ids != _encode(olmo, text):
        raise TokenIdentityError("tokenizer id sequences differ")
    unique = sorted(set(ids))
    if FORBIDDEN_IDS.intersection(unique):
        raise TokenIdentityError("forbidden id in ordinary text")
    p_strings = pythia.convert_ids_to_tokens(unique)
    o_strings = olmo.convert_ids_to_tokens(unique)
    if p_strings != o_strings or len(p_strings) != len(unique) or any(not isinstance(s, str) for s in p_strings):
        raise TokenIdentityError("token vocabulary strings differ or are absent")
    return ids


def model_input_ids(ids):
    return {key: [model["eot"], *ids] for key, model in MODELS.items()}


def tokenizer_provenance(key, tokenizer):
    return dict(MODELS[key], name_or_path=str(tokenizer.name_or_path), class_name=type(tokenizer).__name__,
                is_fast=tokenizer.is_fast, eos_token_id=tokenizer.eos_token_id,
                commit_metadata=tokenizer.init_kwargs.get("_commit_hash"),
                transformers=version("transformers"), tokenizers=version("tokenizers"))


def token_display(tokenizer, token):
    """Decode a single id, including leading whitespace; mark invalid byte fragments."""
    text = tokenizer.decode([token], clean_up_tokenization_spaces=False)
    return INVALID_DISPLAY if "�" in text else text


def grams32(ids):
    """One gram per inclusive model position u >= 32, in position order."""
    return (tuple(ids[i:i + 32]) for i in range(len(ids) - 31))


def repetition32(ids):
    """Return distinct grams, number of repeat positions, and number of gram positions."""
    seen, repeat, total = set(), 0, 0
    for gram in grams32(ids):
        total += 1
        repeat += gram in seen
        seen.add(gram)
    return seen, repeat, total


def select_documents(documents: Iterable[Document], pythia, olmo, *, pool_size=POOL_SIZE,
                     text_tokens=CONTEXT_TOKENS, subset_cap=SUBSET_CAP):
    """Screen for eligibility, then walk the shuffled eligible indices once.

    Eligibility counts cover the whole file. Among eligible documents, only
    candidates examined before the last stitch source contribute walk counts.
    Each rejected document is assigned its first applicable reason. The
    cross-document bank indexes the 32-grams of the retained prefixes.
    """
    import numpy as np

    docs = documents if isinstance(documents, Sequence) else list(documents)
    eligible = []
    by_subset = defaultdict(Counter)
    for i, doc in enumerate(docs):
        counts = by_subset[doc.subset]
        counts["documents"] += 1
        text = doc.text
        if any(s in text for s in FORBIDDEN_TEXT):
            counts["forbidden_text"] += 1
        elif not text.isascii() and 20 * sum(c.isascii() for c in text) < 19 * len(text):
            counts["ascii"] += 1
        elif len(_encode(pythia, text)) < text_tokens + 1:
            counts["too_short"] += 1
        else:
            eligible.append(i)
            counts["eligible"] += 1

    accepted, bank, accepted_subsets, displays = [], set(), Counter(), {}
    for rank, index in enumerate(np.random.default_rng(0).permutation(eligible)):
        doc = docs[int(index)]
        counts = by_subset[doc.subset]
        counts["walked"] += 1
        full_ids = _encode(pythia, doc.text)
        ids = full_ids[:text_tokens]
        for token in set(ids).difference(displays):
            displays[token] = token_display(pythia, token)
        token_strings = [displays[token] for token in ids]
        if 10 * sum(s.isspace() for s in token_strings) > 3 * len(ids):
            counts["whitespace"] += 1
            continue
        grams, repeats, total = repetition32(ids)
        if 10 * repeats > total:
            counts["internal_32gram"] += 1
            continue
        if not bank.isdisjoint(grams):
            counts["shared_32gram"] += 1
            continue
        if accepted_subsets[doc.subset] >= subset_cap:
            counts["subset_cap"] += 1
            continue
        if len(accepted) in (pool_size + 1, pool_size + 3) and doc.subset == accepted[-1]["subset"]:
            counts["stitch_same_subset"] += 1
            continue
        try:
            assert_token_identity(doc.text, pythia, olmo, pythia_ids=full_ids)
        except TokenIdentityError as error:
            raise TokenIdentityError(f"Pile line {doc.line}: {error}") from error
        bank.update(grams)
        accepted_subsets[doc.subset] += 1
        counts["accepted_pool" if len(accepted) < pool_size else "accepted_stitch_source"] += 1
        accepted.append(dict(file_index=doc.line, text_sha256=hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                             subset=doc.subset, eligible_shuffle_rank=rank, token_ids=ids,
                             model_input_ids=model_input_ids(ids), token_strings=token_strings,
                             text=pythia.decode(ids, clean_up_tokenization_spaces=False)))
        if len(accepted) == pool_size + 4:
            break
    if len(accepted) != pool_size + 4:
        raise ValueError(f"eligible documents exhausted: {len(accepted)}/{pool_size + 4} accepted")

    fields = ("documents", "eligible", "walked", "accepted_pool", "accepted_stitch_source", *REASONS)
    totals = sum(by_subset.values(), Counter())
    counts = dict(total={k: totals[k] for k in fields},
                  by_subset={subset: {k: c[k] for k in fields} for subset, c in sorted(by_subset.items())})
    return accepted[:pool_size], accepted[pool_size:], counts


def stitch(sources, context_id, schedule, rng, tokenizer, *, size=CONTEXT_TOKENS):
    """Alternate two distinct-subset source prefixes, advancing each source independently."""
    lo, hi = STITCH_LENGTHS[schedule]
    if len(sources) != 2 or sources[0]["subset"] == sources[1]["subset"]:
        raise ValueError("a stitch needs two sources from different subsets")
    ids, strings, segments, source_positions = [], [], [], [None]  # position 0 has no source
    consumed = [0, 0]
    while len(ids) < size:
        i = len(segments) % 2
        source = sources[i]
        start = consumed[i]
        drawn = int(rng.integers(lo, hi + 1))
        count = min(drawn, size - len(ids))
        stop = start + count
        if stop > len(source["token_ids"]):
            raise ValueError("stitch source exhausted")
        file_index = source["file_index"]
        segments.append(dict(source=source["source"], source_file_index=file_index, drawn_length=drawn,
                             source_start=start, source_stop=stop,
                             position_start=len(ids) + 1, position_stop=len(ids) + 1 + count))
        ids.extend(source["token_ids"][start:stop])
        strings.extend(source["token_strings"][start:stop])
        source_positions.extend([file_index, pos] for pos in range(start + 1, stop + 1))
        consumed[i] = stop
    return dict(id=context_id, kind="stitched", schedule=schedule, token_ids=ids, model_input_ids=model_input_ids(ids),
                token_strings=strings, text=tokenizer.decode(ids, clean_up_tokenization_spaces=False),
                segments=segments, source_positions=source_positions)


def node_sets():
    """Store ordered node sets compactly, as [context id, start position, stop position]."""
    import numpy as np

    stop = CONTEXT_TOKENS + 1
    suffix = [[i, SUFFIX_START, stop] for i in range(POOL_SIZE)]
    q_rng, audit_rng = np.random.default_rng(21), np.random.default_rng(22)
    q, audit = [], []
    for c in range(POOL_SIZE):
        late = q_rng.choice(stop - SUFFIX_START, size=64, replace=False) + SUFFIX_START
        early = q_rng.choice(SUFFIX_START - QUERY_START, size=32, replace=False) + QUERY_START
        q.extend([c, int(p)] for p in (*late, *early))
        audit.append([c, int(audit_rng.integers(SUFFIX_START, stop))])
    companions = {}
    for focal in (0, 1):
        draw = np.random.default_rng(100 + focal).permutation([i for i in range(POOL_SIZE) if i != focal]).tolist()
        companions[str(focal)] = {
            key: [[i, SUFFIX_START, stop] for i in [focal, *others]]
            for key, others in (("4", draw[:3]), ("H1", draw[:15]), ("H2", draw[15:30]))
        }
    return dict(census_pool=[[i, QUERY_START, stop] for i in range(POOL_SIZE)],
                suffix=suffix, query_sample=q, audit_queries=audit,
                one_text={str(i): [[i, QUERY_START, stop]] for i in range(POOL_SIZE + 4)},
                matched_one_text={str(i): [suffix[i]] for i in range(POOL_SIZE)},
                all_texts=suffix, companions=companions)


def build_probe(documents: Iterable[Document], pythia, olmo):
    import numpy as np

    pool, sources, counts = select_documents(documents, pythia, olmo)
    for i, item in enumerate(pool):
        item.update(id=i, kind="pool")
    for source, label in zip(sources, "ABCD"):
        source["source"] = label
    rng = np.random.default_rng(0)
    contexts = pool[:]
    for i in (0, 2):
        for schedule in STITCH_LENGTHS:
            contexts.append(stitch(sources[i:i + 2], len(contexts), schedule, rng, pythia))
    return dict(schema_version=PROBE_SCHEMA_VERSION, corpus=dict(CORPUS), models={k: dict(v) for k, v in MODELS.items()},
                description="a capped sample of the beginnings of long, mostly ASCII Pile test documents",
                context_tokens=CONTEXT_TOKENS, pool_size=POOL_SIZE, contexts=contexts, stitch_sources=sources,
                counts=counts, node_sets=node_sets(), invalid_token_display=INVALID_DISPLAY,
                token_identity=["text", "position"],
                seeds=dict(eligible_shuffle=0, stitches=0, query_sample=21, audit_queries=22,
                           companions={"0": 100, "1": 101}))


def read_pile(path):
    """Stream pinned test.jsonl.zst in original file order."""
    import zstandard

    with open(path, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as lines:
            for line, row in enumerate(lines):
                data = json.loads(row)
                yield Document(line, data["meta"]["pile_set_name"], data["text"])


class _DocumentSpool(Sequence):
    """Index a one-shot corpus stream on scratch disk instead of retaining it in RAM."""

    def __init__(self, file, documents):
        self.file, self.offsets = file, []
        for doc in documents:
            self.offsets.append(file.tell())
            file.write(json.dumps((doc.line, doc.subset, doc.text)).encode("utf-8") + b"\n")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        self.file.seek(self.offsets[index])
        return Document(*json.loads(self.file.readline()))


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        tmp = Path(output.name)
        try:
            json.dump(value, output, ensure_ascii=False, allow_nan=False)
            output.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def main(out_path, cache_dir):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    pythia, olmo = (AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], cache_dir=cache_dir,
                                                  use_fast=True, trust_remote_code=False)
                    for m in MODELS.values())
    for tok, model in zip((pythia, olmo), MODELS.values()):
        if tok.eos_token_id != model["eot"]:
            raise TokenIdentityError(f"{model['id']}: unexpected end-of-text id")
    pile = Path(hf_hub_download(CORPUS["id"], CORPUS["filename"], revision=CORPUS["revision"],
                                repo_type="dataset", cache_dir=cache_dir))
    if pile.stat().st_size != CORPUS["size_bytes"]:
        raise ValueError("unexpected compressed corpus size")
    with tempfile.TemporaryFile() as spool:
        data = build_probe(_DocumentSpool(spool, read_pile(pile)), pythia, olmo)
    data["tokenizer_provenance"] = {key: tokenizer_provenance(key, tok) for key, tok in zip(MODELS, (pythia, olmo))}
    write_json_atomic(out_path, data)
    return dict(contexts=len(data["contexts"]), counts=data["counts"]["total"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    print(main(args.out_path, args.cache_dir))
