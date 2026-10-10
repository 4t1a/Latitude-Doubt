"""estimator_validation.py -- bias of the ledger estimators on a Zipf-type law with known truth,
and the empirical coverage of the certified interval for a small alphabet."""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
from ledgerlib import *

rng = np.random.default_rng(99)

def zipf_pool(n, s, q): 
    w = (np.arange(1, n + 1) + q) ** (-s); return w / w.sum()

def make_truth(Kw=3000, Nx=2500):
    pw = zipf_pool(Kw, 1.1, 2.7); px = zipf_pool(Nx, 1.1, 2.7); rows, cols, vals = [], [], []
    sel = px ** 0.5; sel /= sel.sum()
    for w in range(Kw):
        m = min(4, 1 + rng.poisson(0.4)); xs = rng.choice(Nx, size=m, replace=False, p=sel)
        a = rng.dirichlet(np.full(m, 1.5)); rows += [w] * m; cols += list(xs); vals += list(pw[w] * a)
    vals = np.array(vals); vals /= vals.sum(); return np.array(rows), np.array(cols), vals

def truth_ledger(r, c, v):
    Pw = np.bincount(r, weights=v); Px = np.bincount(c, weights=v)
    C, E, J = ent(Pw), ent(Px), ent(v); return dict(C=C, E=E, S=J - C, D=J - E)

def est_ledger(r, c, kind):
    """r,c: sampled (w,x) index arrays."""
    N = len(r); code = r * 100000 + c; u, cnt = np.unique(code, return_counts=True); wu, xu = u // 100000, u % 100000
    cw = np.bincount(r); cx = np.bincount(c); cw, cx = cw[cw > 0], cx[cx > 0]
    if kind == "cs":
        C, E, J = h_chao_shen(cw), h_chao_shen(cx), h_chao_shen(cnt); return dict(C=C, E=E, S=J - C, D=J - E)
    S = D = 0.0
    for (keyu, other) in ((wu, "S"), (xu, "D")):
        order = np.argsort(keyu, kind="stable"); ks, cs = keyu[order], cnt[order]
        bounds = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]; acc = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            n = cs[a:b].sum(); p = cs[a:b] / n; h = float(-(p * np.log2(p)).sum())
            if kind == "mm": h += (b - a - 1) / (2 * n * LN2)
            acc += n * h
        if other == "S": S = acc / N
        else: D = acc / N
    C, E = ent(cw / N), ent(cx / N)
    if kind == "mm": C += (len(cw) - 1) / (2 * N * LN2); E += (len(cx) - 1) / (2 * N * LN2)
    return dict(C=C, E=E, S=S, D=D)

r, c, v = make_truth(); T = truth_ledger(r, c, v)
print("truth:", {k: round(x, 4) for k, x in T.items()}, "pairs:", len(v))
NS = [2000, 5000, 10000, 30000, 100000, 300000]; REP = 60; out = dict(truth=T, ns=NS, bias={}, sd={})
for kind in ("plugin", "mm", "cs"):
    out["bias"][kind] = {q: [] for q in "CESD"}; out["sd"][kind] = {q: [] for q in "CESD"}
for n in NS:
    acc = {k: {q: [] for q in "CESD"} for k in ("plugin", "mm", "cs")}
    for _ in range(REP):
        idx = rng.choice(len(v), size=n, p=v); rr, cc = r[idx], c[idx]
        for kind in acc:
            e = est_ledger(rr, cc, kind)
            for q in "CESD": acc[kind][q].append(e[q])
    for kind in acc:
        for q in "CESD":
            out["bias"][kind][q].append(float(np.mean(acc[kind][q]) - T[q])); out["sd"][kind][q].append(float(np.std(acc[kind][q])))
    print(n, {k: (round(out["bias"][k]["S"][-1], 3), round(out["bias"][k]["D"][-1], 3)) for k in acc}, flush=True)

# ---- certified interval for a small alphabet
K = 25; p = 1.0 / np.arange(1, K + 1); p /= p.sum(); Ht = ent(p); delta = 0.05; cov = []
for n in [500, 1000, 2000, 5000, 10000, 30000, 100000]:
    hw = entropy_halfwidth(n, K, delta); errs = []
    for _ in range(2000):
        cnt = np.bincount(rng.choice(K, size=n, p=p), minlength=K); errs.append(abs(h_plugin(cnt) - Ht))
    errs = np.array(errs)
    cov.append(dict(n=n, halfwidth=None if math.isnan(hw) else float(hw), eps_l1=float(eps_l1(n, K, delta)), q95_abs_error=float(np.percentile(errs, 95)),
                    max_abs_error=float(errs.max()), coverage=None if math.isnan(hw) else float(np.mean(errs <= hw))))
    print(cov[-1])
out["certified"] = dict(K=K, delta=delta, H=Ht, rows=cov)
json.dump(out, open(OUT / "estimator_validation.json", "w"), indent=1)
