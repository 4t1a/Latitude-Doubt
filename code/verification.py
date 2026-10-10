"""verification.py -- randomized checks of every identity and theorem used in the paper, and validation of the estimators.
Nothing here is evidence for the theorems (they are proved); it is a guard against algebra and coding errors."""
import os, sys, json, math, heapq, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
from ledgerlib import *

rng = np.random.default_rng(424242)
R = {}

def huffman_len(p):
    p = [x for x in p if x > 0]
    if len(p) == 1: return 0.0
    h = [(x, i) for i, x in enumerate(p)]; heapq.heapify(h); depth = [0] * len(p); grp = {i: [i] for i in range(len(p))}; nid = len(p)
    while len(h) > 1:
        a, i = heapq.heappop(h); b, j = heapq.heappop(h)
        for k in grp[i] + grp[j]: depth[k] += 1
        grp[nid] = grp.pop(i) + grp.pop(j); heapq.heappush(h, (a + b, nid)); nid += 1
    return float(sum(x * d for x, d in zip(p, depth)))

# ---------- (1) ledger identities on random joint laws ----------
err_a = err_b = err_c = 0.0; min_terms = 1e9; viol_vocab = 0; n1 = 100000
err_hw = 0.0; huff_lo = 1e9; huff_hi = -1e9; n_h = 0
for t in range(n1):
    K, N = rng.integers(2, 13), rng.integers(2, 13); alpha = rng.choice([0.05, 0.3, 1.0, 3.0])
    P = rng.dirichlet(np.full(K * N, alpha)).reshape(K, N)
    if rng.random() < 0.3: P = P * (rng.random((K, N)) < 0.5) + 1e-300; P /= P.sum()
    L = ledger_from_joint(P)
    err_a = max(err_a, abs(L["E"] - (L["I"] + L["S"])), abs(L["C"] - (L["I"] + L["D"])), abs(L["J"] - (L["I"] + L["S"] + L["D"])))
    err_b = max(err_b, abs((L["E"] - L["C"]) - (L["S"] - L["D"])))
    min_terms = min(min_terms, L["S"], L["D"], L["I"], min(L["C"], L["E"]) - L["I"])
    if L["D"] < L["C"] - math.log2(N) + L["S"] - 1e-12: viol_vocab += 1
    if t < 3000:      # Huffman checks: conditional code for W given X, and code for W
        pw = P.sum(1); n_h += 1
        lw = huffman_len(pw); huff_lo = min(huff_lo, lw - ent(pw)); huff_hi = max(huff_hi, lw - ent(pw))
        px = P.sum(0); lc = sum(px[x] * huffman_len(P[:, x] / px[x]) for x in range(N) if px[x] > 0)
        huff_lo = min(huff_lo, lc - L["D"]); huff_hi = max(huff_hi, lc - L["D"])
R["ledger"] = dict(instances=n1, max_abs_err_partition=err_a, max_abs_err_balance=err_b, min_term=min_terms, vocab_bound_violations=viol_vocab,
                   huffman_instances=n_h, huffman_excess_min=huff_lo, huffman_excess_max=huff_hi)

# ---------- (2) derived meaning: sufficiency, minimality (exhaustive on small alphabets), monotonicity in purpose ----------
def group_rows(Ryx, tol=10):
    key = [tuple(np.round(r, tol)) for r in Ryx]; ids = {}
    return np.array([ids.setdefault(k, len(ids)) for k in key])

def set_partitions(n):
    def rec(i, cur):
        if i == n: yield list(cur); return
        for b in range(max(cur, default=-1) + 2):
            cur.append(b); yield from rec(i + 1, cur); cur.pop()
    yield from rec(0, [])

def info_of_partition(Pxy, lab):
    Pw = np.zeros((lab.max() + 1, Pxy.shape[1]))
    for x, l in enumerate(lab): Pw[l] += Pxy[x]
    return mi(Pw)

suff_err = 0.0; n2 = 600; minimal_ok = 0; sufficient_seen = 0; refine_ok = 0; mono_ok = 0; mono_n = 0; strict_loss_min = 1e9
for t in range(n2):
    nx, ny, m = rng.integers(3, 7), rng.integers(2, 5), rng.integers(2, 5); m = min(m, nx)
    rows = rng.dirichlet(np.ones(ny), size=m)
    cls = np.r_[np.arange(m), rng.integers(0, m, nx - m)]; rng.shuffle(cls)
    px = rng.dirichlet(np.ones(nx) * 2); Pxy = px[:, None] * rows[cls]
    Ryx = Pxy / Pxy.sum(1, keepdims=True); lab = group_rows(Ryx)          # meaning map M (planted classes)
    IXY = mi(Pxy); suff_err = max(suff_err, abs(IXY - info_of_partition(Pxy, lab)))
    for T in set_partitions(nx):
        T = np.array(T); suff = abs(info_of_partition(Pxy, T) - IXY) < 1e-10
        if suff:
            sufficient_seen += 1
            # minimality: M must be a function of T  (x,x' in the same T-block => same M-label)
            ok = all(lab[a] == lab[b] for a in range(nx) for b in range(nx) if T[a] == T[b]); minimal_ok += ok
    # strict information loss when two distinct meanings are merged
    if lab.max() >= 1:
        merged = np.where(lab == 1, 0, lab); strict_loss_min = min(strict_loss_min, IXY - info_of_partition(Pxy, merged))
    # monotonicity in purpose: Y' = h(Y)
    if ny >= 3:
        h = np.r_[np.arange(ny - 1), ny - 2]; Pxy2 = np.zeros((nx, ny - 1))
        for y in range(ny): Pxy2[:, h[y]] += Pxy[:, y]
        lab2 = group_rows(Pxy2 / Pxy2.sum(1, keepdims=True)); mono_n += 1
        coarser = all(lab2[a] == lab2[b] for a in range(nx) for b in range(nx) if lab[a] == lab[b])
        mono_ok += coarser
R["derived_meaning"] = dict(instances=n2, sufficiency_max_err=suff_err, sufficient_partitions_seen=sufficient_seen, minimality_holds=minimal_ok,
                            min_strict_loss_on_merging_distinct_meanings=strict_loss_min, monotonicity_cases=mono_n, monotonicity_holds=mono_ok)

# ---------- (3) leakage identities ----------
err_L1 = err_L2 = 0.0; n3 = 5000; err_adeq = 0.0
for t in range(n3):
    nx, ny = rng.integers(3, 9), rng.integers(2, 5); Pxy = rng.dirichlet(np.ones(nx * ny) * 0.7).reshape(nx, ny)
    k1 = rng.integers(2, nx + 1); w1 = np.r_[np.arange(k1), rng.integers(0, k1, nx - k1)]; rng.shuffle(w1)
    k2 = rng.integers(1, k1 + 1); g = np.r_[np.arange(k2), rng.integers(0, k2, k1 - k2)]; rng.shuffle(g); w2 = g[w1]
    def Lk(w):      # leakage I(X;Y|W) = I(X;Y) - I(W;Y) for W a function of X, computed by the general 3-way formula
        P3 = np.zeros((w.max() + 1, nx, ny))
        for x in range(nx): P3[w[x], x] = Pxy[x]
        return mi_xy_given_w(P3)
    IXY = mi(Pxy); err_L1 = max(err_L1, abs(Lk(w1) - (IXY - info_of_partition(Pxy, w1))))
    P3 = np.zeros((w1.max() + 1, w2.max() + 1, ny))
    for x in range(nx): P3[w1[x], w2[w1[x]]] += Pxy[x]
    I_w1_y_given_w2 = mi_xy_given_w(np.transpose(np.array([[P3[a, b] for b in range(P3.shape[1])] for a in range(P3.shape[0])]), (1, 0, 2))) if False else None
    # I(W1;Y|W2) = I(W1;Y) - I(W2;Y) because W2 is a function of W1
    i_w1_y = info_of_partition(Pxy, w1); i_w2_y = info_of_partition(Pxy, w2)
    err_L2 = max(err_L2, abs(Lk(w2) - (Lk(w1) + (i_w1_y - i_w2_y))))
    # only the identity is adequate for every purpose:  purpose Y = X gives leakage H(X|W)
    Pxx = np.diag(rng.dirichlet(np.ones(nx))); P3 = np.zeros((w1.max() + 1, nx, nx))
    for x in range(nx): P3[w1[x], x] = Pxx[x]
    px = Pxx.diagonal(); hxw = -sum(px[x] * math.log2(px[x] / px[w1 == w1[x]].sum()) for x in range(nx))
    err_adeq = max(err_adeq, abs(mi_xy_given_w(P3) - hxw))
R["leakage"] = dict(instances=n3, err_function_case=err_L1, err_depth_identity=err_L2, err_purpose_identity=err_adeq)

json.dump(R, open(OUT / "verification_identities.json", "w"), indent=1)
print(json.dumps(R, indent=1))
