"""Local depth extraction tests; no models, CUDA, or Modal API calls."""
import hashlib
import json
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import depth_inputs as di
import extract_depth as ed
import modal_depth as md
from probe import CONTEXT_TOKENS, MODELS, PROBE_SCHEMA_VERSION, TokenIdentityError, model_input_ids


@pytest.fixture
def probe_file(tmp_path):
    docs = []
    for context in range(32):
        ids = [2 + (context * CONTEXT_TOKENS + i) % 9490 for i in range(CONTEXT_TOKENS)]
        docs.append(dict(id=context, token_ids=ids, model_input_ids=model_input_ids(ids)))
    # Stitched contexts should not contribute to the allowed vocabulary.
    docs.extend(dict(id=i, token_ids=[50000] * CONTEXT_TOKENS) for i in range(32, 36))
    path = tmp_path / 'probe' / 'probe.json'
    path.parent.mkdir()
    path.write_text(json.dumps(dict(schema_version=PROBE_SCHEMA_VERSION, models=MODELS, contexts=docs)), encoding='utf-8')
    return path


def sha_le32(values):
    return hashlib.sha256(struct.pack(f'<{len(values)}i', *values)).hexdigest()


def test_determinism_hashes_and_permutations(probe_file, tmp_path):
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    probe_sha = hashlib.sha256(probe_file.read_bytes()).hexdigest()
    di.write_inputs(probe_file, a, expected_sha256=probe_sha)
    di.write_inputs(probe_file, b, expected_sha256=probe_sha)
    assert a.read_bytes() == b.read_bytes()
    frozen = json.loads(a.read_bytes())
    assert set(frozen) == {'seed_random', 'seed_shuffle', 'allowed_ids_sha256', 'contexts', 'sha256'}
    assert (frozen['seed_random'], frozen['seed_shuffle']) == (20261126, 20261127)
    allowed = list(range(2, 9492))
    assert frozen['allowed_ids_sha256'] == sha_le32(allowed)
    assert list(frozen['contexts']) == ['r0', 'r1', 'r2', 'r3', 'r4', 's0', 's1', 's2', 's3']
    random = np.random.default_rng(20261126).choice(allowed, size=(5, 2047), replace=True)
    for i in range(5):
        assert frozen['contexts'][f'r{i}'] == {'ids': random[i].tolist()}
    docs = json.loads(probe_file.read_bytes())['contexts']
    for n, source in enumerate((0, 1, 2, 4)):
        record = frozen['contexts'][f's{n}']
        perm = record['perm']
        assert sorted(perm) == list(range(1, 2048))
        assert perm == (np.random.default_rng(20261127 + n).permutation(2047) + 1).tolist()
        assert record['source_context'] == source
        assert record['ids'] == [docs[source]['token_ids'][i - 1] for i in perm]
    assert frozen['sha256'] == {key: sha_le32(context['ids']) for key, context in frozen['contexts'].items()}
    # Even a tamper with internally valid hashes is not the frozen draw.
    frozen['contexts']['r0']['ids'][0] = 2 + (frozen['contexts']['r0']['ids'][0] - 1) % 9490
    frozen['sha256']['r0'] = sha_le32(frozen['contexts']['r0']['ids'])
    with pytest.raises(ValueError, match='frozen inputs do not match'):
        di.validate_frozen(frozen, docs[:32], allowed)


@pytest.mark.parametrize('root', ['acts', 'acts_depth_backup', '../acts_depth', '/vol/acts_depth', 'acts_depth/../acts'])
def test_refuse_uncanonical_output_root(root, tmp_path):
    with pytest.raises(ValueError, match='acts_depth/<model>'):
        ed.extract_model('pythia', tmp_path, 'unused', output_root=root, commit=lambda: None)


def test_paths_and_symlinks(tmp_path):
    assert ed.output_dir(tmp_path, 'acts_depth/', 'pythia') == tmp_path / 'acts_depth' / 'pythia'
    with pytest.raises(ValueError):
        ed.output_dir(tmp_path, 'acts_depth', '../pythia')
    (tmp_path / 'acts_depth').mkdir()
    (tmp_path / 'acts_depth' / 'pythia').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        ed.output_dir(tmp_path, 'acts_depth', 'pythia')


class StubTokenizer:
    def __init__(self, key, substitute=None):
        self.eos_token_id = MODELS[key]['eot']
        self.substitute = substitute or {}

    def convert_ids_to_tokens(self, ids):
        return [self.substitute.get(i, str(i)) for i in ids]


@pytest.mark.parametrize('problem', [None, 'disallowed', 'tokens_differ', 'missing', 'wrong_order', 'wrong_separator',
                                     'wrong_eos', 'non_integer'])
def test_assert_ids(problem):
    allowed = [2, 3]
    ids = [2, 3] + [2] * (CONTEXT_TOKENS - 2)
    if problem == 'disallowed':
        ids[-1] = 4
    if problem == 'non_integer':
        ids[-1] = 2.0
    variants = model_input_ids(ids)
    if problem == 'wrong_order':
        variants['olmo'][1:3] = [3, 2]
    if problem == 'wrong_separator':
        variants['olmo'][0] = MODELS['pythia']['eot']
    tokenizers = {key: StubTokenizer(key) for key in MODELS}
    if problem == 'wrong_eos':
        tokenizers['pythia'].eos_token_id = 1
    if problem == 'tokens_differ':
        tokenizers['olmo'].substitute[2] = 'different'
    if problem == 'missing':
        for tok in tokenizers.values():
            tok.substitute[2] = None
    if problem is None:
        ed.assert_model_inputs({'r0': ids}, {'r0': variants}, allowed, tokenizers)
    else:
        with pytest.raises(TokenIdentityError):
            ed.assert_model_inputs({'r0': ids}, {'r0': variants}, allowed, tokenizers)


def test_budget_and_manual_retry():
    quote = md.reservation()
    assert quote['rate_usd_per_second'] == 0.00025192
    assert quote['seconds_per_attempt'] == 720
    assert quote['attempts_reserved'] == 4
    assert quote['reserved_usd'] == 0.7255296
    with pytest.raises(ValueError, match=r'above \$1\.00'):
        md.reservation(rate_override=0.00035)  # $1.008
    ledger = dict(reservation=quote, attempts=[])
    assert md.next_attempt(ledger, 'pythia', False) == 1
    with pytest.raises(ValueError, match='previous failed'):
        md.next_attempt(ledger, 'pythia', True)
    row = dict(model='pythia', status='pending')
    ledger['attempts'].append(row)
    with pytest.raises(ValueError, match='unresolved'):
        md.next_attempt(ledger, 'pythia', True)
    row['status'] = 'done'
    with pytest.raises(ValueError, match='confirmed failure'):
        md.next_attempt(ledger, 'pythia', True)
    row['status'] = 'failed'
    with pytest.raises(ValueError, match='only with --retry'):
        md.next_attempt(ledger, 'pythia', False)
    assert md.next_attempt(ledger, 'pythia', True) == 2
    ledger['attempts'].append(dict(row))
    with pytest.raises(ValueError, match='budget exhausted'):
        md.next_attempt(ledger, 'pythia', True)
    assert md.next_attempt(ledger, 'olmo', False) == 1


@pytest.mark.parametrize('probe_bytes', [None, b'not the Stage A probe'])
def test_launch_refuses_wrong_volume_before_writing(monkeypatch, probe_bytes):
    reads = []

    def read(path):
        reads.append(path)
        return probe_bytes

    def must_not_run(*args, **kwargs):
        pytest.fail('wrong Volume must be rejected before opening the run ledger or starting an app')

    monkeypatch.setattr(md, 'read_volume', read)
    monkeypatch.setattr(md, 'load_ledger', must_not_run)
    monkeypatch.setattr(md.app, 'run', must_not_run)
    with pytest.raises(ValueError, match='Stage A extraction receipts'):
        md.launch('pythia', False, md.reservation(), b'', 'unused')
    assert reads == [di.PROBE_PATH]


def test_receipts_do_not_treat_a_transport_timeout_as_a_failure(monkeypatch):
    row = dict(model='pythia', status='pending')
    ledger = dict(reservation=md.reservation(), attempts=[row])
    saved = []
    monkeypatch.setattr(md, 'save_ledger', lambda ledger: saved.append(ledger))

    def get():
        raise TimeoutError('client timed out, worker may still be running')

    with pytest.raises(TimeoutError):
        md.settle(ledger, row, SimpleNamespace(get=get))
    assert row['status'] == 'pending'
    assert not saved
    md.settle(ledger, row, SimpleNamespace(get=lambda: dict(status='done', seconds=19)))
    assert row['seconds'] == 19
    assert row['billed_seconds_estimate'] == 139
    assert row['estimated_usd'] == 139 * 0.00025192
    assert saved == [ledger]


@pytest.mark.parametrize('parity_passes', [True, False])
def test_c00_gate_and_context_commit_order(monkeypatch, capsys, probe_file, tmp_path, parity_passes):
    # Exercise the wrapper's sequencing with small residual arrays and no Transformers/Torch imports.
    shape = (17, 2, 3)
    monkeypatch.setattr(ed, 'ACT_SHAPE', shape)
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=16, hidden_size=2048, eos_token_id=0, _attn_implementation='eager'),
        to=lambda _: model, name_or_path='stub', training=False, eval=lambda: None,
        get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(dtype='float32', device='cuda:0')),
    )
    fake_torch = SimpleNamespace(__version__='2.5.1', float32='float32',
                                 cuda=SimpleNamespace(is_available=lambda: True), version=SimpleNamespace(cuda='stub'),
                                 backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                                          cudnn=SimpleNamespace(allow_tf32=False)))
    fake_hf = SimpleNamespace(__version__='4.57.6',
                              AutoTokenizer=SimpleNamespace(from_pretrained=lambda name, **kwargs: StubTokenizer(
                                  next(key for key, meta in MODELS.items() if meta['id'] == name))),
                              AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_hf)
    monkeypatch.setattr(ed, 'tokenizer_provenance', lambda *_: {})
    calls = []
    ref = np.zeros(shape, dtype=np.float32)

    def forward(model_arg, ids, **kwargs):
        assert model_arg is model
        assert kwargs == dict(verify_hooks=True, readouts=False)
        calls.append(ids)
        result = ref.copy()
        if len(calls) == 1 and not parity_passes:
            result[16, 1, 2] = 0.01
        return result, None, None

    def save(path, fn, acts, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        fn(path, acts, **kwargs)

    monkeypatch.setitem(sys.modules, 'extract', SimpleNamespace(TRANSFORMERS_VERSION='4.57.6', TORCH_VERSION='2.5.1',
                                                                set_precision=lambda: None, forward=forward, _save_atomic=save))
    frozen_path = tmp_path / di.INPUTS_PATH
    probe_sha = hashlib.sha256(probe_file.read_bytes()).hexdigest()
    monkeypatch.setattr(ed, 'PROBE_SHA256', probe_sha)
    frozen = di.write_inputs(probe_file, frozen_path, expected_sha256=probe_sha)
    c00 = tmp_path / 'acts' / 'pythia' / 'c00.npy'
    c00.parent.mkdir(parents=True)
    np.save(c00, ref, allow_pickle=False)
    dest = frozen_path.parent / 'pythia'
    commits = []

    def commit():
        commits.append({path.name for path in dest.iterdir()})

    args = ('pythia', tmp_path, hashlib.sha256(frozen_path.read_bytes()).hexdigest())
    if parity_passes:
        receipt = ed.extract_model(*args, commit=commit)
        assert len(calls) == 10
        for n, key in enumerate(di.KEYS, start=1):
            assert calls[n] == [0, *frozen['contexts'][key]['ids']]
            assert commits[n - 1] == {f'{k}.npy' for k in di.KEYS[:n]}
        assert commits[-1] == commits[-2] | {'provenance.json'}
        assert receipt['c00_parity']['max_abs_by_layer'] == [0.0] * 17
    else:
        with pytest.raises(ValueError, match=r'float32 tolerance at layers \[16\]'):
            ed.extract_model(*args, commit=commit)
        assert len(calls) == 1
        assert not dest.exists()
        assert not commits
    assert calls[0] == json.loads(probe_file.read_bytes())['contexts'][0]['model_input_ids']['pythia']
    assert 'c00 max_abs_by_layer=' in capsys.readouterr().out
