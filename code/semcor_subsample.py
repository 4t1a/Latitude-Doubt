"""semcor_subsample.py -- how the plug-in ledger of SemCor moves with sample size (no convergence for C and E)."""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
from ledgerlib import ledger_from_pair_counts, h_plugin
rng = np.random.default_rng(5)
toks = [t for t in pickle.load(open(RAW / "semcor_tokens.pkl", "rb")) if not t[0].startswith("brownv")]
pairs = [(t[2], t[1]) for t in toks]; out = []
for frac in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
    acc = []
    for _ in range(12 if frac < 1 else 1):
        idx = rng.choice(len(pairs), int(frac * len(pairs)), replace=False)
        L = ledger_from_pair_counts([pairs[i] for i in idx], est=h_plugin); acc.append([L[k] for k in ("n", "C", "E", "S", "D")])
    m = np.mean(acc, 0); out.append(dict(frac=frac, n=int(m[0]), C=float(m[1]), E=float(m[2]), S=float(m[3]), D=float(m[4])))
    print(out[-1])
json.dump(out, open(OUT / "semcor_subsample.json", "w"), indent=1)
