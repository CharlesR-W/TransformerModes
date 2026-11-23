"""Free local diagnostic (CPU, Pythia-160m, 8 texts): who are a token's nearest neighbours under the memory kernel?

Not part of the experiment. It answers one design question before anything is spent: in a graph over the tokens of
several texts, do a token's nearest neighbours stay inside its own text, and does removing each text's mean vector
change that?  Run:

    uv run --with torch --with transformers --with datasets --with numpy python local_pilot.py
"""
from __future__ import annotations

import glob
import json
import sysconfig
from pathlib import Path

import numpy as np
import torch

from similarity import memory_neighbours

MODEL = 'EleutherAI/pythia-160m'
T = 1024            # positions per text, position 0 = end-of-text token
BURN = 128          # nodes are positions BURN..T-1
LAYERS = (3, 6, 9)  # hidden_states indices (block outputs)
MEMS = {'none': 0.0, '8': 1 - 1 / 8, '64': 1 - 1 / 64, 'whole': 1.0}
K = 64
K_NB = 1024      # kept per row, so that excluded neighbours can be skipped
WINDOW = 64      # exclusion window in positions (the 'near' rule)


def texts(tok):
    out = []
    from datasets import load_dataset
    wiki = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
    doc, docs = [], []
    for line in wiki['text']:
        if line.startswith(' = ') and not line.startswith(' = = '):
            if doc:
                docs.append(''.join(doc))
            doc = []
        doc.append(line)
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(docs)):
        if len(tok(docs[i])['input_ids']) >= T and len(out) < 4:
            out.append(('wiki', docs[i]))
    std = sorted(glob.glob(sysconfig.get_paths()['stdlib'] + '/*.py'))
    for i in rng.permutation(len(std)):
        s = Path(std[i]).read_text(errors='ignore')
        if len(tok(s)['input_ids']) >= T and sum(k == 'code' for k, _ in out) < 2:
            out.append(('code', s))
    for p in ('/usr/share/common-licenses/GPL-3', '/usr/share/common-licenses/Apache-2.0'):
        out.append(('legal', Path(p).read_text()))
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(3)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    docs = texts(tok)
    ids, acts, preds = [], [], []
    for kind, s in docs:
        x = [tok.eos_token_id] + tok(s)['input_ids'][:T - 1]
        ids.append(x)
        with torch.no_grad():
            o = model(torch.tensor([x]), output_hidden_states=True)
        h = o.hidden_states
        preds.append(o.logits[0, 1:].argmax(-1).numpy())
        acts.append(torch.stack([h[l][0, 1:] for l in LAYERS]).double())   # (sites, T-1, d), position 0 dropped
    acts = torch.stack(acts)                                              # (M, sites, T-1, d)
    ids = np.array(ids)[:, 1:]
    m, kept = len(docs), T - 1 - (BURN - 1)
    text_of = np.repeat(np.arange(m), kept)
    pos_of = np.tile(np.arange(BURN, T), m)
    tok_of = ids[:, BURN - 1:].reshape(-1)
    pred_of = np.array(preds)[:, BURN - 1:].reshape(-1)
    rng = np.random.default_rng(1)
    rand = rng.integers(0, len(pred_of), (len(pred_of), 64))
    print(json.dumps(dict(pred_match_random_pairs=float((pred_of[rand] == pred_of[:, None]).mean()))), flush=True)
    order = np.argsort(tok_of, kind='stable')          # random pairs sharing the current token
    st = tok_of[order]
    same_tok_rate = float(np.mean([(pred_of[order[i]] == pred_of[order[i + 1]])
                                   for i in range(len(st) - 1) if st[i] == st[i + 1]]))
    print(json.dumps(dict(pred_match_same_current_token_pairs=same_tok_rate)), flush=True)
    rows = []
    for si, layer in enumerate(LAYERS):
        x = acts[:, si]
        for centring in ('corpus mean', 'own-text mean'):
            mu = x.mean((0, 1), keepdim=True) if centring == 'corpus mean' else x.mean(1, keepdim=True)
            y = x - mu
            y = y / y.norm(dim=-1, keepdim=True)
            nb = memory_neighbours(y, list(MEMS.values()), K_NB, burn_in=BURN, reading='cosine')
            for ai, name in enumerate(MEMS):
                jall = nb.indices[ai].numpy()
                same_all = text_of[jall] == text_of[:, None]
                gap_all = np.abs(pos_of[jall] - pos_of[:, None])
                rules = {'all tokens': np.ones_like(same_all),
                         'not within %d positions' % WINDOW: ~(same_all & (gap_all <= WINDOW)),
                         'other texts only': ~same_all}
                for rule, ok in rules.items():
                    # first K allowed neighbours of each row (rows are sorted by distance)
                    rank = np.cumsum(ok, 1)
                    pick = ok & (rank <= K)
                    short = float((pick.sum(1) < K).mean())
                    same = same_all & pick
                    n = pick.sum()
                    rows.append(dict(layer=layer, centring=centring, memory=name, rule=rule,
                                     same_text=float(same.sum() / n),
                                     chance=float((kept - 1) / (m * kept - 1)),
                                     same_token=float(((tok_of[jall] == tok_of[:, None]) & pick).sum() / n),
                                     median_gap_same_text=float(np.median(gap_all[same])) if same.any() else None,
                                     pred_match=float(((pred_of[jall] == pred_of[:, None]) & pick).sum() / n),
                                     rows_short_of_k=short))
                    print(json.dumps(rows[-1]), flush=True)
    Path('local_pilot.json').write_text(json.dumps(dict(model=MODEL, kinds=[k for k, _ in docs], T=T, burn=BURN,
                                                        k=K, k_nb=K_NB, window=WINDOW, rows=rows), indent=1))


if __name__ == '__main__':
    main()
