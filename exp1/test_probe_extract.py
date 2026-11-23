"""CPU tests for U0/U1. TM_EXP1_HF_FIXTURES may point at three pinned local snapshots
named pythia, olmo and pythia70m; otherwise normal Hugging Face loading is used.
Only the 70M checkpoint (safetensors) is needed; the 1B checks load no weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, OlmoConfig

import extract
import probe


class ToyTokenizer:
    def __init__(self, mapping, *, vocabulary=None, displays=None):
        self.mapping = mapping
        self.vocabulary = vocabulary or {}
        self.displays = displays or {}

    def encode(self, text, *, add_special_tokens, verbose):
        assert not add_special_tokens and not verbose
        return self.mapping[text]

    def convert_ids_to_tokens(self, ids):
        return [self.vocabulary.get(i, str(i)) for i in ids]

    def decode(self, ids, *, clean_up_tokenization_spaces):
        assert not clean_up_tokenization_spaces
        return "".join(self.displays.get(i, "x") for i in ids)


def distinct(i, n):
    return list(range(60_000 + i * 4096, 60_000 + i * 4096 + n))


def test_shuffled_walk_exhaustive_reasons():
    # Construct a file order whose eligible shuffle produces this exact trace.
    specifications = [
        ("alpha", "first", distinct(0, 81)),
        ("shared", "one shared gram", distinct(0, 32) + distinct(1, 49)),
        ("space", "space test", [777] * 25 + distinct(2, 56)),
        ("repeat", "repeat test", [778] * 81),
        ("alpha", "second", distinct(3, 81)),
        ("alpha", "over quota", distinct(4, 81)),
        ("beta", "stitch A", distinct(5, 81)),
        ("beta", "stitch A again", distinct(6, 81)),
        ("gamma", "stitch B", distinct(7, 81)),
        ("gamma", "stitch C", distinct(8, 81)),
        ("delta", "stitch D", distinct(9, 81)),
        ("tail", "never walked", distinct(10, 81)),
    ]
    shuffled = np.random.default_rng(0).permutation(len(specifications))
    docs = [None] * len(specifications)
    mapping = {}
    for index, (subset, text, ids) in zip(shuffled, specifications, strict=True):
        docs[index] = probe.Document(int(index), subset, text)
        mapping[text] = ids
    docs.insert(2, probe.Document(100, "pre", "short"))
    docs.insert(5, probe.Document(101, "pre", "ééé"))
    docs.append(probe.Document(102, "pre", "hidden <|padding|>"))
    mapping["short"] = distinct(11, 80)  # needs 81 with only 80 stored
    tok = ToyTokenizer(mapping, displays={777: " \t"})
    pool, sources, counts = probe.select_documents(iter(docs), tok, tok, pool_size=2, text_tokens=80, subset_cap=2)
    assert [p["file_index"] for p in pool + sources] == shuffled[[0, 4, 6, 8, 9, 10]].tolist()
    assert [p["eligible_shuffle_rank"] for p in pool + sources] == [0, 4, 6, 8, 9, 10]
    assert pool[0]["token_ids"] == distinct(0, 80)
    assert {p["file_index"] for p in pool + sources}.isdisjoint((100, 101, 102))
    assert {reason: counts["total"][reason] for reason in probe.REASONS} == dict.fromkeys(probe.REASONS, 1)
    assert counts["by_subset"]["pre"]["walked"] == counts["by_subset"]["tail"]["walked"] == 0
    assert counts["by_subset"]["beta"]["stitch_same_subset"] == counts["by_subset"]["alpha"]["subset_cap"] == 1
    assert counts["total"]["walked"] == 11
    assert counts["total"]["eligible"] == 12


def test_32gram_position_boundaries_and_threshold():
    assert probe.repetition32([5] * 31) == (set(), 0, 0)
    assert probe.repetition32([5] * 32) == ({(5,) * 32}, 0, 1)
    assert probe.repetition32([5] * 34) == ({(5,) * 32}, 2, 3)
    assert probe.repetition32(list(range(32)) + list(range(32)))[1:] == (1, 33)
    # At 41 tokens there are 10 eligible endpoint positions, including u=32.
    docs = [probe.Document(i, str(i), f"test {i}") for i in range(7)]
    mapping = {d.text: distinct(d.line, 42) for d in docs}
    passes, fails = np.random.default_rng(0).permutation(7)[:2].tolist()
    mapping[f"test {passes}"][:33] = [5] * 33  # 1 of 10 repeat positions: pass
    mapping[f"test {fails}"][:34] = [6] * 34  # 2 of 10: reject
    tok = ToyTokenizer(mapping)
    pool, sources, counts = probe.select_documents(docs, tok, tok, pool_size=1, text_tokens=41)
    assert pool[0]["file_index"] == passes
    assert fails not in [d["file_index"] for d in sources]
    assert counts["by_subset"][str(fails)]["internal_32gram"] == 1


@pytest.mark.parametrize("literal", probe.FORBIDDEN_TEXT)
def test_literal_guard(literal):
    tok = ToyTokenizer({})
    with pytest.raises(probe.TokenIdentityError, match="forbidden source text"):
        probe.assert_token_identity(f"before {literal} after", tok, tok)


def test_token_identity_fails_closed():
    p = ToyTokenizer({"plain": [2]})
    assert probe.assert_token_identity("plain", p, p) == [2]
    with pytest.raises(probe.TokenIdentityError, match="sequences differ"):
        probe.assert_token_identity("plain", p, ToyTokenizer({"plain": [3]}))
    for bad in probe.FORBIDDEN_IDS:
        forbidden = ToyTokenizer({"plain": [2, bad]})
        with pytest.raises(probe.TokenIdentityError, match="forbidden id"):
            probe.assert_token_identity("plain", forbidden, forbidden)
    with pytest.raises(probe.TokenIdentityError, match="vocabulary strings"):
        probe.assert_token_identity("plain", p, ToyTokenizer({"plain": [2]}, vocabulary={2: "other"}))
    with pytest.raises(probe.TokenIdentityError, match="vocabulary strings"):
        missing = ToyTokenizer({"plain": [2]}, vocabulary={2: None})
        probe.assert_token_identity("plain", missing, missing)
    full = distinct(0, 80) + [50279]
    doc = probe.Document(0, "a", "tokens")
    tok = ToyTokenizer({"tokens": full})
    with pytest.raises(probe.TokenIdentityError, match="Pile line 0.*forbidden id"):
        probe.select_documents([doc], tok, tok, pool_size=1, text_tokens=80)
    assert probe.token_display(ToyTokenizer({}, displays={5: " �"}), 5) == probe.INVALID_DISPLAY


def test_probe_and_stitched_segment_tables():
    docs = [probe.Document(i, f"subset {i}", f"source {i}") for i in range(40)]
    tok = ToyTokenizer({d.text: distinct(d.line, 2048) for d in docs})
    out = probe.build_probe(iter(docs), tok, tok)
    assert json.loads(json.dumps(out)) == out
    assert [c["id"] for c in out["contexts"]] == list(range(36))
    assert [c["kind"] for c in out["contexts"]] == ["pool"] * 32 + ["stitched"] * 4
    sources = {s["source"]: s for s in out["stitch_sources"]}
    assert set(sources) == set("ABCD")
    assert out["counts"]["total"]["accepted_stitch_source"] == 4
    assert out["token_identity"] == ["text", "position"]
    assert len({d["file_index"] for d in out["contexts"][:32] + out["stitch_sources"]}) == 36
    for doc in out["contexts"][:32] + out["stitch_sources"]:
        assert doc["text_sha256"] == hashlib.sha256(docs[doc["file_index"]].text.encode()).hexdigest()
    for item in out["contexts"] + out["stitch_sources"]:
        assert item["model_input_ids"]["pythia"] == [0, *item["token_ids"]]
        assert item["model_input_ids"]["olmo"] == [50279, *item["token_ids"]]
    for c, want in zip(out["contexts"][32:], ("AB", "AB", "CD", "CD"), strict=True):
        lo, hi = probe.STITCH_LENGTHS[c["schedule"]]
        cursor, rebuilt, consumed = 1, [], {source: 0 for source in want}
        assert c["source_positions"][0] is None and len(c["source_positions"]) == 2048
        for i, s in enumerate(c["segments"]):
            label = want[i % 2]
            assert s["source"] == label
            assert s["source_start"] == consumed[label]
            assert s["position_start"] == cursor
            n = s["position_stop"] - cursor
            assert n == s["source_stop"] - s["source_start"]
            assert lo <= s["drawn_length"] <= hi
            assert n == min(s["drawn_length"], 2048 - cursor)
            assert s["source_file_index"] == sources[label]["file_index"]
            assert c["source_positions"][cursor:cursor + n] == [
                [s["source_file_index"], p] for p in range(s["source_start"] + 1, s["source_stop"] + 1)
            ]
            rebuilt.extend(sources[label]["token_ids"][s["source_start"]:s["source_stop"]])
            cursor += n
            consumed[label] += n
        assert cursor == 2048 and rebuilt == c["token_ids"]
        assert len(c["token_strings"]) == 2047
    # The toy tokenizer can encode only original source documents, not joins.
    assert [c["schedule"] for c in out["contexts"][32:]] == ["long", "short", "long", "short"]


def test_node_sets():
    nodes = probe.node_sets()

    def count(ranges):
        return sum(stop - start for _, start, stop in ranges)

    assert nodes["census_pool"] == [[i, 256, 2048] for i in range(32)]
    assert count(nodes["census_pool"]) == 57_344
    assert nodes["one_text"] == {str(i): [[i, 256, 2048]] for i in range(36)}
    assert all(count(r) == 1792 for r in nodes["one_text"].values())
    assert nodes["matched_one_text"] == {str(i): [[i, 1536, 2048]] for i in range(32)}
    assert nodes["all_texts"] == [[i, 1536, 2048] for i in range(32)]
    assert count(nodes["all_texts"]) == 16_384
    assert nodes["suffix"] == nodes["all_texts"]
    for c in (0, 1):
        g = nodes["companions"][str(c)]
        assert [count(g[k]) for k in ("4", "H1", "H2")] == [2048, 8192, 8192]
        assert all(rows[0] == [c, 1536, 2048] for rows in g.values())
        h1, h2 = ([row[0] for row in g[k][1:]] for k in ("H1", "H2"))
        assert set(h1).isdisjoint(h2)
        assert [row[0] for row in g["4"][1:]] == h1[:3]
        assert h1 + h2 == np.random.default_rng(100 + c).permutation([i for i in range(32) if i != c])[:30].tolist()


def test_query_sample_and_audit_drawn_from_suffix_of_q():
    nodes = probe.node_sets()
    q, audit = nodes["query_sample"], nodes["audit_queries"]
    assert len(q) == len(set(map(tuple, q))) == 3072
    assert [pair[0] for pair in audit] == list(range(32))
    q_rng, audit_rng = np.random.default_rng(21), np.random.default_rng(22)
    for c, audit_pos in audit:
        suffix = [p for text, p in q if text == c and 1536 <= p < 2048]
        early = [p for text, p in q if text == c and 256 <= p < 1536]
        assert suffix == q_rng.choice(np.arange(1536, 2048), 64, replace=False).tolist()
        assert early == q_rng.choice(np.arange(256, 1536), 32, replace=False).tolist()
        assert audit_pos in suffix and audit_pos == audit_rng.choice(suffix)


def test_pile_stream(tmp_path):
    import zstandard

    docs = [dict(text="abc", meta=dict(pile_set_name="a")), dict(text="é", meta=dict(pile_set_name="b"))]
    path = tmp_path / "test.jsonl.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress("\n".join(map(json.dumps, docs)).encode()))
    assert list(probe.read_pile(path)) == [probe.Document(0, "a", "abc"), probe.Document(1, "b", "é")]


def fixture_args(key, repo, revision):
    root = os.getenv("TM_EXP1_HF_FIXTURES")
    if root:
        return (Path(root) / key,), {"local_files_only": True, "trust_remote_code": False}
    return (repo,), {"revision": revision, "trust_remote_code": False, "token": False}


def test_real_tokenizer_identity_and_pinned_configs():
    tokens = []
    for key, meta in probe.MODELS.items():
        args, kwargs = fixture_args(key, meta["id"], meta["revision"])
        tok = AutoTokenizer.from_pretrained(*args, use_fast=True, **kwargs)
        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert tok.eos_token_id == config.eos_token_id == meta["eot"]
        assert (config.num_hidden_layers, config.hidden_size) == (16, 2048)
        tokens.append(tok)
    for text in ("The quiet engine started.\nThen it stopped.", "def f(x):\n    return x + 1\n",
                 "Guten Morgen! Café and a smile 🙂."):
        ids = probe.assert_token_identity(text, *tokens)
        assert ids and set(ids).isdisjoint(probe.FORBIDDEN_IDS)
    assert any(probe.token_display(tokens[0], i) == probe.INVALID_DISPLAY
               for i in probe.assert_token_identity("🙂", *tokens))


def test_cpu_pythia_end_to_end(tmp_path):
    torch.set_num_threads(2)
    repo, revision = "EleutherAI/pythia-70m", "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
    args, kwargs = fixture_args("pythia70m", repo, revision)
    model = AutoModelForCausalLM.from_pretrained(*args, dtype=torch.float32, use_safetensors=True,
                                               attn_implementation="eager", **kwargs)
    tok = AutoTokenizer.from_pretrained(*args, **kwargs)
    texts = ("The house stood at a crossroads where the forest opened out toward the lake. " * 3,
             "Flowers appeared between stones and the old tree leaned against the wall. " * 3)
    n, split, query_start, query_split = 20, 9, 4, 13
    contexts = [dict(id=i, token_ids=tok.encode(text, add_special_tokens=False)[:n - 1]) for i, text in enumerate(texts)]
    assert all(len(c["token_ids"]) == n - 1 for c in contexts)
    assert contexts[0]["token_ids"][split - 1] != contexts[1]["token_ids"][split - 1]
    meta = extract.extract_contexts(
        model, contexts, tok.eos_token_id, tmp_path, "70m", causal_split=split, query_start=query_start,
        query_split=query_split, model_revision=revision,
        tokenizer_info=probe.tokenizer_provenance(tok, dict(id=repo, revision=revision, eot=tok.eos_token_id)),
    )
    layers, width, heads = model.config.num_hidden_layers, model.config.hidden_size, model.config.num_attention_heads
    assert meta == json.loads((tmp_path / "acts" / "70m" / "provenance.json").read_text())
    assert meta["model"]["revision"] == meta["tokenizer"]["revision"] == revision
    assert meta["tokenizer"]["class_name"] == type(tok).__name__
    assert meta["libraries"]["torch"] == torch.__version__
    assert meta["forward_pass"] == dict(eval=True, dtype="torch.float32", device="cpu", batch_size=1,
                                        attention_implementation="eager", matmul_tf32=False, cudnn_tf32=False)
    assert meta["hook_equality"] == dict(passed=True, exact=True, context_id=0, layers=list(range(1, layers)))
    assert meta["causal"]["passed"] and meta["causal"]["donor_context_id"] == 1
    assert meta["causal"]["original_cut_token_id"] == contexts[0]["token_ids"][split - 1]
    assert meta["causal"]["replacement_cut_token_id"] == contexts[1]["token_ids"][split - 1]
    assert meta["causal"]["replacement_suffix_ids"] == contexts[1]["token_ids"][split - 1:]
    assert max(meta["causal"]["prefix_max_relative_l2"]) <= 1e-5
    assert min(meta["causal"]["boundary_relative_l2"]) > 1e-3
    assert len(meta["causal"]["truncated_prefix_max_relative_l2"]) == layers + 1
    acts_files, pred_files, attention_file = extract.output_paths(tmp_path, "70m", [0, 1])
    attn = np.load(attention_file)
    ranges = [(query_start, query_split), (query_split, n)]
    np.testing.assert_array_equal(attn["query_ranges"], ranges)
    np.testing.assert_array_equal(attn["context_ids"], [0, 1])
    np.testing.assert_array_equal(attn["layer_ids"], range(1, layers + 1))
    for name in ("query_count", "non_sink_mass_sum", "sink_mass_sum", "distance_numerator_sum"):
        assert attn[name].shape == (2, 2, layers, heads)
    for band, (lo, hi) in enumerate(ranges):
        np.testing.assert_array_equal(attn["query_count"][:, band], hi - lo)
        np.testing.assert_allclose(attn["sink_mass_sum"][:, band] + attn["non_sink_mass_sum"][:, band], hi - lo,
                                   rtol=1e-7)

    for c in contexts:
        i = c["id"]
        ids = torch.tensor([[tok.eos_token_id, *c["token_ids"]]])
        with torch.inference_mode():
            oracle = model(ids, output_hidden_states=True, output_attentions=True, use_cache=False)
            log_prob = oracle.logits[0].log_softmax(-1)
            p = log_prob.exp()
        act = np.load(acts_files[i])
        assert act.shape == (layers + 1, n, width) and act.dtype == np.float32
        np.testing.assert_array_equal(act[:-1], torch.stack(oracle.hidden_states[:-1])[:, 0].numpy())
        assert not np.array_equal(act[-1], oracle.hidden_states[-1][0].numpy())  # pre-final norm
        with torch.inference_mode():
            restored_logits = model.embed_out(model.gpt_neox.final_layer_norm(torch.from_numpy(act[-1][None])))
        torch.testing.assert_close(restored_logits, oracle.logits)
        pred = np.load(pred_files[i])
        values, top_ids = pred["top20_probs"], pred["top20_ids"]
        assert top_ids.dtype == pred["top1_id"].dtype == np.int32
        assert values.dtype == pred["entropy"].dtype == pred["top20_mass"].dtype == np.float32
        assert top_ids.shape == values.shape == (n, 20) and pred["entropy"].shape == pred["top20_mass"].shape == (n,)
        assert "logits" not in pred.files
        np.testing.assert_allclose(pred["top20_mass"], values.sum(axis=1, dtype=np.float64), rtol=5e-7)
        np.testing.assert_array_equal(pred["top1_id"], top_ids[:, 0])
        assert np.all(values[:, :-1] >= values[:, 1:])
        assert np.all((values.sum(axis=1) > 0) & (values.sum(axis=1) < 1))
        np.testing.assert_allclose(values, p.gather(-1, torch.from_numpy(top_ids).long()).numpy(), rtol=4e-6)
        np.testing.assert_allclose(values, torch.sort(p, descending=True).values[:, :20].numpy(), rtol=4e-6)
        np.testing.assert_allclose(pred["entropy"], -(p * log_prob).sum(dim=-1).numpy(), rtol=4e-6)

        a = oracle.attentions[0][0].double().numpy()
        for band, (lo, hi) in enumerate(ranges):
            non_sink = a[:, lo:hi, 1:]
            expected_sink = a[:, lo:hi, 0].sum(-1)
            expected_mass = non_sink.sum(axis=(1, 2))
            expected_num = (non_sink * (np.arange(lo, hi)[:, None] - np.arange(1, n))).sum(axis=(1, 2))
            np.testing.assert_allclose(attn["sink_mass_sum"][i, band, 0], expected_sink, rtol=1e-7)
            np.testing.assert_allclose(attn["non_sink_mass_sum"][i, band, 0], expected_mass, rtol=1e-7)
            np.testing.assert_allclose(attn["distance_numerator_sum"][i, band, 0], expected_num, rtol=2e-7)


def test_tiny_olmo_hook_and_attention_compatibility():
    model = AutoModelForCausalLM.from_config(OlmoConfig(vocab_size=40, hidden_size=32, intermediate_size=64,
                                                         num_hidden_layers=3, num_attention_heads=4),
                                            attn_implementation="eager", dtype=torch.float32).eval()
    acts, predictions, attn = extract.forward(model, [3, 4, 5, 6], verify_hooks=True, query_start=1, query_split=2)
    assert acts.shape == (4, 4, 32)
    assert predictions["top20_ids"].shape == (4, 20)
    assert attn["query_count"].shape == (2, 3, 4)
    np.testing.assert_array_equal(attn["query_count"][0], 1)
    np.testing.assert_array_equal(attn["query_count"][1], 2)


def test_causal_check_requires_changed_boundary_token():
    with pytest.raises(AssertionError, match="no same-length suffix"):
        extract.causal_check(None, {0: [0, 2, 3], 1: [0, 4, 3]}, None, 2)


def test_coordinator_ledger_and_cap_without_rpc(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    import modal_app as app

    class LocalVolume:
        def reload(self):
            pass

        def commit(self):
            pass

    class LocalExtract:
        def map(self, models, *, return_exceptions):
            assert return_exceptions and models == ["pythia", "olmo"]
            for key in models:
                attempt = app._reservation("run", "U1", key)
                durable = json.loads(app._ledger_path(attempt).read_text())
                assert durable["status"] == "reserved"
                assert durable["reserved_usd"] == durable["timeout_seconds"] * app.rate("U1")
            return iter((dict(cached=False, seconds=300), RuntimeError("fixture failure")))

    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "volume", LocalVolume())
    monkeypatch.setattr(app, "u1_extract", LocalExtract())
    monkeypatch.setattr(app, "_probe_identity", lambda: "test-sha")
    monkeypatch.setattr(app, "_complete", lambda *args: None)
    with pytest.warns(UserWarning, match="executing locally"):
        result = app.coordinator.local("run", "extract")
    assert result["status"] == "failed"
    passed, failed = result["attempts"]
    assert app._charge(passed) == pytest.approx((300 + app.STARTUP_SECONDS) * app.rate("U1"))
    assert failed["status"] == "failed" and app._charge(failed) > 0
    assert app._ledger_total() > app._charge(passed) + app._charge(failed)
    with pytest.raises(RuntimeError, match="exceeds"):
        app._require_budget(7.99, [app._reservation("next", "U1", "pythia")])
    with pytest.raises(ValueError, match="no implemented worker"):
        app._stage_units("A-pilot", False)
    monkeypatch.setitem(app.STAGE_B_WORKERS, "test-stage-b", (("U1", "pythia"),))
    with pytest.raises(ValueError, match="requires a separate"):
        app._stage_units("test-stage-b", False)
    assert app._stage_units("test-stage-b", True) == (("U1", "pythia"),)
    assert app.FETCH_PREFIXES == ("analysis", "audit_a", "ledger", "site")
