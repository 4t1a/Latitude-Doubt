"""ledgerlib.py -- exact information ledger of a joint law P(W, X), plus estimators.
All entropies in bits. Nothing here depends on the data set."""
import math
import numpy as np
from collections import Counter

LN2 = math.log(2.0)

def ent(p):
    """Shannon entropy (bits) of a probability vector (zeros allowed)."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def ledger_from_joint(P):
    """P[w, x] >= 0.  Returns the five ledger quantities."""
    P = np.asarray(P, dtype=float)
    P = P / P.sum()
    C, E, J = ent(P.sum(1)), ent(P.sum(0)), ent(P)
    return dict(C=C, E=E, J=J, S=J - C, D=J - E, I=C + E - J)

def mi_xy_given_w(Pwxy):
    """I(X;Y|W) for a 3-way array P[w,x,y]."""
    P = np.asarray(Pwxy, dtype=float); P = P / P.sum()
    Hwx, Hwy, Hw, Hwxy = ent(P.sum(2)), ent(P.sum(1)), ent(P.sum((1, 2))), ent(P)
    return Hwx + Hwy - Hw - Hwxy

def mi(P):
    """Mutual information (bits) of a 2-way joint array."""
    P = np.asarray(P, dtype=float); P = P / P.sum()
    return ent(P.sum(0)) + ent(P.sum(1)) - ent(P)

# ---------- estimators from counts ----------
def h_plugin(counts):
    c = np.asarray(list(counts), dtype=float); c = c[c > 0]
    return ent(c / c.sum())

def h_mm(counts):
    """Miller-Madow: plug-in + (K_obs - 1)/(2 n ln 2)."""
    c = np.asarray(list(counts), dtype=float); c = c[c > 0]
    n = c.sum()
    return ent(c / n) + (len(c) - 1) / (2.0 * n * LN2)

def h_chao_shen(counts):
    """Chao-Shen coverage-adjusted entropy estimator (bits)."""
    c = np.asarray(list(counts), dtype=float); c = c[c > 0]
    n = c.sum(); f1 = float((c == 1).sum())
    if f1 == n: f1 = n - 1.0
    cov = 1.0 - f1 / n
    pa = cov * c / n
    return float(-(pa * np.log2(pa) / (1.0 - (1.0 - pa) ** n)).sum())

def ledger_from_pair_counts(pairs, est=h_plugin):
    """pairs: iterable of (w, x).  Ledger with a chosen entropy estimator."""
    cwx = Counter(pairs)
    cw, cx = Counter(), Counter()
    for (w, x), k in cwx.items():
        cw[w] += k; cx[x] += k
    C, E, J = est(cw.values()), est(cx.values()), est(cwx.values())
    return dict(C=C, E=E, J=J, S=J - C, D=J - E, I=C + E - J,
                n=sum(cwx.values()), Kw=len(cw), Kx=len(cx), Kj=len(cwx))

# ---------- certified interval for the entropy of a small alphabet ----------
def eps_l1(n, K, delta):
    """With prob >= 1-delta, ||p_hat - p||_1 <= eps (2^K exponential-inequality form)."""
    return math.sqrt((2.0 / n) * (K * LN2 + math.log(1.0 / delta)))

def entropy_halfwidth(n, K, delta):
    """|H_hat - H| <= eps*log2(K/eps) whenever eps<=1/2 (Cover-Thomas Thm 17.3.3)."""
    e = eps_l1(n, K, delta)
    return (e * math.log2(K / e)) if e <= 0.5 else float("nan")
