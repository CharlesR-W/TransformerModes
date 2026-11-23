"""Read packaged bytes independently; compare with artifacts from the census fixture."""
import gzip
import json
from math import prod
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import census as c
import pack_site
import readout
import select_b
from synthetic import generate
from test_census import _write_preds_and_acts, _write_probe


def _load(path):
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("viewer-stage-a")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    data = generate(5, "planted", texts=3, positions=24, width=10, sites=2, vocab=7)
    pool = _write_probe(root, data.tokens, 2, 10)
    path = root / "probe" / "probe.json"
    probe = _load(path)
    probe["models"] = {"pythia": {"id": "toy-pythia"}, "olmo": {"id": "toy-olmo"}}
    for item in probe["contexts"]:
        item["token_strings"] = [f" T{v}" for v in item["token_ids"]]
        item["text"] = "".join(item["token_strings"])
    tokens = data.tokens[0].tolist()
    segments = [dict(source="A", position_start=1, position_stop=10, source_start=0, source_stop=9),
                dict(source="B", position_start=10, position_stop=25, source_start=0, source_stop=15)]
    probe["contexts"].append(dict(id=3, kind="stitched", token_ids=tokens, token_strings=[f" T{v}" for v in tokens],
                                   schedule="short", segments=segments,
                                   source_positions=[None] + [[1, p] for p in range(1, 10)] + [[9, p] for p in range(1, 16)]))
    probe["node_sets"]["one_text"] = {"3": [[3, 2, 25]]}
    c.write_json(path, probe)
    stitch_pool = c.pool_from_probe(probe, stitch=3)
    for model, variant in (("pythia", data), ("olmo", generate(5, "strong", texts=3, positions=24, width=10, sites=2, vocab=7))):
        _write_preds_and_acts(root, model, pool, variant.states[1], variant.predictors)
        _write_preds_and_acts(root, model, stitch_pool, variant.states[1, :1], variant.predictors[:1])
        # Reuse the existing fixture's exact IDs and exercise three prediction
        # slots (the census test only needed two), plus float32 input precision.
        for i in (0, 1, 2, 3):
            f = root / "pred" / model / f"c{i:02d}.npz"
            with np.load(f) as saved:
                p = dict(saved)
            p["top20_ids"] = np.column_stack((p["top20_ids"], np.full(len(p["top1_id"]), 50023)))
            p["top20_probs"] = np.column_stack((p["top20_probs"], np.full(len(p["top1_id"]), .125))).astype("<f4")
            p["top20_mass"] = p["top20_probs"].sum(1)
            c.atomic(f, np.savez, **p)
        inputs = {"site04": variant.states[1], "site08": variant.states[0],
                  "embedding": variant.embeddings[variant.tokens], "random": c.random_vectors(pool.tokens, 10)}
        for reading, centring in c.PRIORITY:
            c.run_group(root, model, pool, inputs, reading, centring, k=8, target=8, window=2, row_block=12, pair_count=0)
        for reading in ("cosine", "dot"):
            c.run_group(root, f"{model}_stitch03", stitch_pool,
                        {"site04": variant.states[1, :1], "site08": variant.states[0, :1]},
                        reading, "C", masks=("none", "window"), k=8, window=2, row_block=12, screen=False, pair_count=0)
        readout.run_readout(root, model, sites=(4, 8), min_matched=1)
        c.atomic(root / "attn" / f"{model}.npz", np.savez, context_ids=np.arange(4), layer_ids=[1],
                 query_count=np.full((4, 2, 1, 2), 4), sink_mass_sum=np.ones((4, 2, 1, 2)),
                 non_sink_mass_sum=np.full((4, 2, 1, 2), 3.), distance_numerator_sum=np.full((4, 2, 1, 2), 6.))
        readout.run_timescales(root, model, sites=(4,))
    # With an unmeasured site the fixture cannot accidentally select a dot
    # construction; all measured site numbers still come from select_b.
    select_b.select(root, sites=(4, 8, 12), readout_sites=(4, 8), required_sites=2)
    return root


def _asset(out, manifest, path):
    """Only the public manifest contract; never call helpers from pack_site."""
    spec = manifest["files"][path]
    buf = (out / manifest["data_directory"] / path).read_bytes()
    assert len(buf) == spec["bytes"] and spec["compression"] == "gzip"
    raw = gzip.decompress(buf)
    assert len(raw) == spec["decoded_bytes"]
    if spec["encoding"] == "json":
        obj = json.loads(raw)
        assert spec["dtype"] == "json"
        assert spec["shape"] == ([len(obj)] if isinstance(obj, list) else [])
        return obj
    assert spec["encoding"] == "typed_arrays" and manifest["byte_order"] == "little"
    dtype_to_unpacked_char = {"<u2": "H", "|u1": "B", "<f4": "f"}
    result, offset = {}, 0
    for field in manifest["layouts"][spec["dtype"]]:
        char = dtype_to_unpacked_char[field["dtype"]]
        size = struct.calcsize("<" + char)
        gap = (-offset) % size
        assert raw[offset:offset + gap] == b"\0" * gap
        offset += gap
        shape = spec["shape"] + ([field["columns"]] if "columns" in field else [])
        result[field["name"]] = np.array(struct.unpack_from("<" + str(prod(shape)) + char, raw, offset)).reshape(shape)
        offset += prod(shape) * size
    assert offset == len(raw)
    return result


def _manifest(out):
    manifest = _load(out / "manifest.json")
    assert manifest["manifest_file"]["path"] == "manifest.json"
    paths = {p.relative_to(out / manifest["data_directory"]).as_posix()
             for p in (out / manifest["data_directory"]).rglob("*") if p.is_file()}
    assert paths == set(manifest["files"])
    assert sum(r["bytes"] for r in manifest["files"].values()) == manifest["asset_bytes"]
    assert sum(r["decoded_bytes"] for r in manifest["files"].values()) == manifest["decoded_asset_bytes"]
    for spec in manifest["files"].values():
        assert all(spec.get(k) is not None for k in ("shape", "dtype", "units", "encoding"))
    return manifest


def _direct_cosine_distance(data, setting, ranges, stitched):
    x = data.states[1 if setting["site"] == 4 else 0]
    if stitched:
        x = x[:1]
    x = x - x.mean((0, 1) if setting["centring"] == "C" else 1, keepdims=True)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    t, w = x.shape[1:]
    pairs = [(i, q - 1) for i, (_, start, stop) in enumerate(ranges) for q in range(start, stop)]
    history = np.zeros((len(pairs), t, w))
    for i, (doc, p) in enumerate(pairs):
        history[i, :p + 1] = x[doc, :p + 1][::-1] * setting["alpha"] ** (np.arange(p + 1)[:, None] * .5)
    feat = history.reshape(len(pairs), -1)
    feat /= np.linalg.norm(feat, axis=-1, keepdims=True)
    return 1. - feat @ feat.T


def test_all_assets_decode_and_equal_upstream(staged, tmp_path):
    out = tmp_path / "site"
    assert pack_site.run(staged, out) == _manifest(out)
    manifest = _manifest(out)
    sources = {c["id"]: c for c in _load(staged / "probe" / "probe.json")["contexts"]}
    seen = set()

    def asset(name):
        seen.add(name)
        return _asset(out, manifest, name)

    ids_used = set()
    for text in manifest["texts"]:
        origin = sources[text["id"]]
        data = asset(text["text_file"])
        assert data == {k: origin[k] if k != "position_start" else 1 for k in data}
        if origin["kind"] == "stitched":
            assert data["segments"] == origin["segments"]
        for name, path in text["predictions"].items():
            packed = asset(path)
            with np.load(staged / "pred" / name / f"c{text['id']:02d}.npz") as f:
                np.testing.assert_array_equal(packed["token_id"], f["top20_ids"][1:, :3])
                np.testing.assert_array_equal(packed["probability"], f["top20_probs"][1:, :3])
            ids_used.update(np.unique(packed["token_id"]).tolist())
    info = manifest["token_card"]["prediction_display"]
    vocab = {k: s for r in sources.values() for k, s in zip(r["token_ids"], r["token_strings"])}
    known = {str(k): vocab[k] for k in ids_used.intersection(vocab)}
    assert asset(info["file"]) == known
    assert info["missing_ids"] == len(ids_used) - len(known) and "50023" not in known
    assert info["tokenizer"]["status"] == "no_pinned_tokenizer_in_probe"
    distance_meta = manifest["token_card"]["neighbour_distance"]
    assert distance_meta["available"] and distance_meta["complete"] and "reason" not in distance_meta
    assert distance_meta["by_reading"]["cosine"]["similarity"] == "1 - distance"
    assert "mean K(t,t)" in distance_meta["by_reading"]["dot"]["units"]

    latent = {m: generate(5, v, texts=3, positions=24, width=10, sites=2, vocab=7)
              for m, v in (("pythia", "planted"), ("olmo", "strong"))}
    controls_real, controls_stitch = set(), set()
    for setting in manifest["settings"]:
        assert setting["reading"] == "cosine"
        group = (setting["centring"], setting["mask"])
        ranges = manifest["candidate_pools"][setting["candidate_pool"]]
        stitched = setting["candidate_pool"] != "corpus"
        (controls_stitch if stitched else controls_real).add(group)
        identities = [(text, position) for text, start, stop in ranges for position in range(start, stop)]
        explicit = _direct_cosine_distance(latent[setting["model"]], setting, ranges, stitched)
        assert setting["distance"]["available"] and setting["distance"]["units"] == "dimensionless"
        with np.load(staged / setting["source"]) as original:
            for text, _, _ in ranges:
                packed = asset(setting["files"][str(text)])
                rows = [i for i, (t, _) in enumerate(identities) if text == t]
                np.testing.assert_array_equal(packed["origin_counts"], original["origins"][rows])
                top = original["indices"][rows]
                np.testing.assert_array_equal(packed["index"], np.where(top[:, :8] < 0, 65535, top[:, :8]))
                np.testing.assert_array_equal(packed["distance"], original["display_distances"][rows])
                expected_dist = np.where(top[:, :8] >= 0, explicit[np.asarray(rows)[:, None], top[:, :8]], np.nan).astype("<f4")
                np.testing.assert_array_equal(packed["distance"], expected_dist)
                for i, row in enumerate(rows):
                    full = [j for j in top[i] if j >= 0]
                    t_in, t_all = original["t_in"][row], original["t_all"][row]
                    cutoff_start = len(full) - t_in
                    tied_bits = sum(j >= cutoff_start for j in range(len(full[:8]))) if t_all > 1 else 0
                    expected = tied_bits | ((t_all > 1) << 4) | ((t_all > 1 and t_all > t_in) << 5)
                    assert packed["tie_flags"][i] == expected
                    assert [identities[j] for j in packed["index"][i] if j != 65535] == [identities[j] for j in full[:8]]
    assert controls_real == {("C", "none"), ("C", "window"), ("O", "window")}
    assert controls_stitch == {("C", "none"), ("C", "window")}
    e1, e2, e3 = (asset(manifest["figures"][key]) for key in ("e1", "e2", "e3"))
    for part in e1["geometry"]:
        assert part["rows"] == _load(staged / part["source"])
    for part in e1["timescales"]:
        assert part["data"] == _load(staged / part["source"])
    whiskers = e1["other_text_share_document_bootstrap"]
    assert whiskers
    for item in whiskers:
        source = _load(staged / item["source"])[item["source_row"]]
        assert source["input"] == "site08" and source["mask"] == item["mask"]
        assert source["memory"] == item["memory"]
        interval = source["origin_document_bootstrap"]
        assert item["p5"] == interval["p5"][0] and item["p95"] == interval["p95"][0]
        assert item["seed"] == interval["seed"] and item["draws"] == 200
        doc = source["origin_by_document"]
        shares = np.asarray(doc["share_sums"])
        n_docs = len(doc["document_ids"])
        draws = np.random.default_rng(item["seed"]).integers(0, n_docs, size=(200, n_docs))
        bootstrap = shares[draws, 0].sum(1) / np.asarray(doc["queries_with_neighbours"])[draws].sum(1)
        np.testing.assert_allclose(np.percentile(bootstrap, [5, 95]), [item["p5"], item["p95"]], atol=1e-15)
    for part in e2["prediction"]:
        source = _load(staged / part["source"])["settings"]
        assert part["rows"] == [source[i] for i in part["source_rows"]]
    assert e3["selection"] == _load(staged / e3["source"])
    assert e3["source_rows"] == list(range(len(e3["selection"]["table"])))
    assert e3["suffix_lead"] and all(row["role"] == "R" for row in e3["suffix_lead"])
    for row in e3["suffix_lead"]:
        assert row["row"] == _load(staged / row["source"])[row["source_row"]]
        assert "screen_origin" in row["row"]
    assert seen == set(manifest["files"])


def test_cap_omits_exactly_dropped_groups_and_replaces_manifest(staged, tmp_path):
    out = tmp_path / "site"
    plan_path = staged / "analysis" / "stage_a_plan.json"
    original_selection = _load(staged / "analysis" / "stage_b_selection.json")
    try:
        full = pack_site.run(staged, out)
        before_e1 = _asset(out, full, full["figures"]["e1"])
        before_e2 = _asset(out, full, full["figures"]["e2"])
        for approved in (c.PRIORITY[:3], c.PRIORITY[:2]):
            c.write_json(plan_path, dict(approved_groups=approved))
            newer = pack_site.run(staged, out)
            assert newer == _manifest(out)
            assert newer["settings"] == full["settings"]  # dot is never a Stage A Viewer option
            assert newer["approved_groups"] == [list(g) for g in approved]
            assert newer["dropped_groups"] == [list(g) for g in c.PRIORITY[len(approved):]]
            e1, e2, e3 = (_asset(out, newer, newer["figures"][key]) for key in ("e1", "e2", "e3"))
            assert e1["geometry"] == [r for r in before_e1["geometry"] if (r["reading"], r["centring"]) in approved]
            for a, b in zip(e2["prediction"], before_e2["prediction"]):
                expected = [r for r in b["rows"] if (r["reading"], r["centring"]) in approved]
                assert a["rows"] == expected
            assert e3["selection"]["table"] == [original_selection["table"][i] for i in e3["source_rows"]]
            for row in e3["selection"]["table"]:
                if any(v is not None for m in row["models"].values() for v in m["sites"].values()):
                    assert (row["construction"]["reading"], row["construction"]["centring"]) in approved
            assert not (out / full["data_directory"]).exists()
            assert len(list(out.glob("stage-a-data-*"))) == 1

        # No remaking an inconsistent stored decision; the prior generation
        # should stay valid if assembly raises.
        broken = dict(original_selection, M=dict(reading="dot", centring="O", memory="2", mask="none"))
        c.write_json(staged / "analysis" / "stage_b_selection.json", broken)
        with pytest.raises(ValueError, match="stored selection M"):
            pack_site.run(staged, out)
        assert _manifest(out) == newer
        assert len(list(out.glob("stage-a-data-*"))) == 1
    finally:
        plan_path.unlink(missing_ok=True)
        c.write_json(staged / "analysis" / "stage_b_selection.json", original_selection)


def test_legacy_rank_files_preserve_unavailable_layout(staged, tmp_path):
    source = c.setting_path(staged, "pythia", "cosine", "C", "site04", "none", "none")
    before = source.read_bytes()
    out = tmp_path / "site"
    try:
        with np.load(source) as stored:
            old = {k: stored[k] for k in stored if k != "display_distances"}
        c.atomic(source, np.savez, **old)
        manifest = pack_site.run(staged, out)
        choice = next(s for s in manifest["settings"] if s["source"] == source.relative_to(staged).as_posix())
        assert not choice["distance"]["available"] and "reason" in choice["distance"]
        assert not manifest["token_card"]["neighbour_distance"]["complete"]
        tile = _asset(out, manifest, choice["files"]["0"])
        assert set(tile) == {"index", "origin_counts", "tie_flags"}
        assert any(spec["dtype"] == "neighbours_with_distance" for spec in manifest["files"].values())
    finally:
        source.write_bytes(before)


def test_prediction_display_uses_pinned_cache_and_original_whitespace(tmp_path, monkeypatch):
    seen = []

    class Tokenizer:
        def convert_ids_to_tokens(self, token):
            return None if token == 50299 else "unused raw spelling"

        def decode(self, ids, *, clean_up_tokenization_spaces):
            assert len(ids) == 1 and clean_up_tokenization_spaces is False
            return {2: " a", 3: "left�right"}[ids[0]]

    def from_pretrained(name, **kw):
        seen.append((name, kw))
        return Tokenizer()

    pythia = {"id": "fixture-tokenizer", "revision": "pinned-rev"}
    probe = {"models": {"pythia": pythia}, "invalid_token_display": "�"}
    vocab, predicted = {1: " stored "}, {1, 2, 3, 50299}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)))
    decoded, info = pack_site._prediction_display(tmp_path, probe, predicted, vocab)
    assert decoded == {"1": " stored ", "2": " a", "3": "�"}
    assert info == dict(status="pinned_pythia_tokenizer", model_id=pythia["id"], revision=pythia["revision"], extra_ids=2)
    assert seen == [(pythia["id"], dict(revision=pythia["revision"], cache_dir=str(tmp_path / "hf"), local_files_only=True,
                                       trust_remote_code=False, use_fast=True, token=False))]
    monkeypatch.setitem(sys.modules, "transformers", None)
    fallback, info = pack_site._prediction_display(tmp_path, probe, predicted, vocab)
    assert fallback == {"1": " stored "} and info["status"] == "transformers_unavailable"


def test_cutoff_flags_for_more_than_displayed_ties(tmp_path):
    # Per-row scenarios: no tie; all 8 displayed tied; tie entirely after
    # the 8 displayed; unselected ties with three tied members in a short row.
    n = 12
    path = tmp_path / "cutoffs.npz"
    indices = np.tile(np.arange(10), (n, 1))
    indices[3, 5:] = -1
    counts = np.tile([4, 5, 1], (n, 1))
    counts[3] = [2, 2, 1]
    t_in, t_all = np.ones(n, int), np.ones(n, int)
    t_in[:4], t_all[:4] = [1, 10, 2, 3], [1, 10, 12, 7]
    np.savez(path, indices=indices, origins=counts, t_in=t_in, t_all=t_all)
    f = pack_site._neighbour_planes(path, n)
    assert f["tie_flags"][:4].tolist() == [0, 16 | 8, 16 | 32, 16 | 32 | 3]
    assert f["index"][3].tolist() == [0, 1, 2, 3, 4, 65535, 65535, 65535]
