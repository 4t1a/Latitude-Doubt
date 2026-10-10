"""Unit tests for ledgerlib.py information-theoretic functions and estimators."""
import math
import numpy as np
import pytest

from ledgerlib import (
    ent,
    ledger_from_joint,
    mi_xy_given_w,
    mi,
    h_plugin,
    h_mm,
    h_chao_shen,
    ledger_from_pair_counts,
    eps_l1,
    entropy_halfwidth,
)


def test_entropy_fair_coin():
    """Entropy of fair Bernoulli(0.5) is exactly 1.0 bit."""
    assert math.isclose(ent([0.5, 0.5]), 1.0, abs_tol=1e-12)


def test_entropy_deterministic():
    """Entropy of deterministic vector [1.0, 0.0] is 0.0 bits."""
    assert math.isclose(ent([1.0, 0.0]), 0.0, abs_tol=1e-12)


def test_ledger_identities_independent():
    """For independent W and X: I(W;X) = 0, S = H(X), D = H(W)."""
    # 2x2 uniform independent distribution
    P = np.array([[0.25, 0.25], [0.25, 0.25]])
    L = ledger_from_joint(P)
    assert math.isclose(L["I"], 0.0, abs_tol=1e-12)
    assert math.isclose(L["C"], 1.0, abs_tol=1e-12)
    assert math.isclose(L["E"], 1.0, abs_tol=1e-12)
    assert math.isclose(L["J"], 2.0, abs_tol=1e-12)
    assert math.isclose(L["S"], 1.0, abs_tol=1e-12)
    assert math.isclose(L["D"], 1.0, abs_tol=1e-12)
    # Conservation of joint entropy: J = I + S + D
    assert math.isclose(L["J"], L["I"] + L["S"] + L["D"], abs_tol=1e-12)


def test_ledger_identities_identity_channel():
    """For identity channel W = X: I(W;X) = H(W) = H(X), S = 0, D = 0."""
    P = np.array([[0.5, 0.0], [0.0, 0.5]])
    L = ledger_from_joint(P)
    assert math.isclose(L["I"], 1.0, abs_tol=1e-12)
    assert math.isclose(L["S"], 0.0, abs_tol=1e-12)
    assert math.isclose(L["D"], 0.0, abs_tol=1e-12)
    assert math.isclose(L["J"], 1.0, abs_tol=1e-12)
    assert math.isclose(L["J"], L["I"] + L["S"] + L["D"], abs_tol=1e-12)


def test_mi_consistency():
    """Mutual information from mi(P) matches ledger I."""
    rng = np.random.default_rng(42)
    P = rng.uniform(0.1, 1.0, size=(4, 5))
    P = P / P.sum()
    L = ledger_from_joint(P)
    assert math.isclose(mi(P), L["I"], abs_tol=1e-12)


def test_conditional_mi_independence():
    """When X and Y are conditionally independent given W, I(X;Y|W) = 0."""
    # P(w, x, y) = P(w) P(x|w) P(y|w)
    Pw = np.array([0.4, 0.6])
    Px_w = np.array([[0.8, 0.2], [0.3, 0.7]])
    Py_w = np.array([[0.5, 0.5], [0.9, 0.1]])

    Pwxy = np.zeros((2, 2, 2))
    for w in range(2):
        for x in range(2):
            for y in range(2):
                Pwxy[w, x, y] = Pw[w] * Px_w[w, x] * Py_w[w, y]

    cmi = mi_xy_given_w(Pwxy)
    assert math.isclose(cmi, 0.0, abs_tol=1e-12)


def test_estimators_uniform():
    """Check plug-in, Miller-Madow, and Chao-Shen on uniform counts."""
    counts = [1000, 1000]
    hp = h_plugin(counts)
    hmm = h_mm(counts)
    hcs = h_chao_shen(counts)
    assert math.isclose(hp, 1.0, abs_tol=1e-10)
    assert hmm >= hp  # Miller-Madow adds non-negative correction
    assert math.isclose(hcs, 1.0, abs_tol=1e-3)


def test_ledger_from_pair_counts():
    """Check pair count processing reproduces empirical ledger."""
    pairs = [("w1", "x1"), ("w1", "x2"), ("w2", "x3"), ("w2", "x3")]
    res = ledger_from_pair_counts(pairs)
    assert res["n"] == 4
    assert res["Kw"] == 2
    assert res["Kx"] == 3
    assert math.isclose(res["J"], res["I"] + res["S"] + res["D"], abs_tol=1e-12)


def test_certified_intervals():
    """Check eps_l1 and entropy_halfwidth bounds."""
    e = eps_l1(n=100000, K=4, delta=1e-3)
    assert e > 0
    hw = entropy_halfwidth(n=100000, K=4, delta=1e-3)
    assert hw > 0
    assert not math.isnan(hw)
