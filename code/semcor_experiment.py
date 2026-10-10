"""semcor_experiment.py -- the ledger of English content words on the sense-tagged SemCor corpus.
Meaning W = WordNet synset.  Form X = lemma (main analysis), (lemma, POS) or surface word (sensitivity analyses).
Neither map is a function of the other: synonymy (many lemmas per synset) and ambiguity (many synsets per lemma) are both present.
Intervals: bootstrap standard error from B = 1000 resamplings of documents (the independent unit), normal interval centred on the estimate.
A percentile interval is NOT used: a resample contains fewer distinct types than the sample, which shifts diversity-type statistics downward.
The standard errors are cross-checked by half-sampling and by the delete-one-document jackknife."""
import os, sys, json, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
from ledgerlib import ent, LN2

B = 1000
rng = np.random.default_rng(11)
toks = [t for t in pickle.load(open(RAW / "semcor_tokens.pkl", "rb")) if not t[0].startswith("brownv")]
docs = sorted({t[0] for t in toks}); did = {d: i for i, d in enumerate(docs)}; ND = len(docs)
def ids(vals):
    u = {v: i for i, v in enumerate(sorted(set(vals)))}; return np.array([u[v] for v in vals]), len(u)
posn = ["a" if t[3] in ("a", "s") else t[3] for t in toks]
d_arr = np.array([did[t[0]] for t in toks]); pos_arr = np.array(posn)
W, KW = ids([t[2] for t in toks])
FORMS = {"lemma": ids([t[1] for t in toks]), "lemma+pos": ids([(t[1], p) for t, p in zip(toks, posn)]), "surface": ids([t[4] for t in toks])}
coll = np.array([0 if docs[d].startswith("brown1") else 1 for d in range(ND)])        # collection of each document

def prep(form, mask=None):
    x, KX = FORMS[form]; w, d = W, d_arr
    if mask is not None: x, w, d = x[mask], w[mask], d[mask]
    code = x.astype(np.int64) * KW + w; cu, inv = np.unique(code, return_inverse=True)
    return dict(xu=cu // KW, wu=cu % KW, inv=inv, d=d, KX=KX)

def ledger(P, mult, tau):
    tw = mult[P["d"]]; total = tw.sum()
    cnt = np.bincount(P["inv"], weights=tw, minlength=len(P["xu"])); keep = cnt > 0; xu, wu = P["xu"], P["wu"]
    if tau > 1:
        while True:
            nx = np.bincount(xu[keep], weights=cnt[keep], minlength=P["KX"]); nw = np.bincount(wu[keep], weights=cnt[keep], minlength=KW)
            new = keep & (nx[xu] >= tau) & (nw[wu] >= tau)
            if new.sum() == keep.sum(): break
            keep = new
    c, xk, wk = cnt[keep], xu[keep], wu[keep]; N = c.sum()
    nx = np.bincount(xk, weights=c, minlength=P["KX"]); nw = np.bincount(wk, weights=c, minlength=KW)
    S = -(c * np.log2(c / nw[wk])).sum() / N; D = -(c * np.log2(c / nx[xk])).sum() / N
    px, pw = nx[nx > 0] / N, nw[nw > 0] / N; C = -(pw * np.log2(pw)).sum(); E = -(px * np.log2(px)).sum()
    npairs, kx, kw = len(c), len(px), len(pw)
    Smm = S + (npairs - kw) / (2 * N * LN2); Dmm = D + (npairs - kx) / (2 * N * LN2)
    return dict(n=float(N), kept=float(N / total), Kx=int(kx), Kw=int(kw), C=C, E=E, S=S, D=D, Cmm=C + (kw - 1) / (2 * N * LN2), Emm=E + (kx - 1) / (2 * N * LN2), Smm=Smm, Dmm=Dmm)

def boot_mult(doc_ids):
    return np.bincount(rng.choice(doc_ids, size=len(doc_ids)), minlength=ND).astype(float)

def run(label, form, tau, mask=None, doc_ids=None):
    P = prep(form, mask); doc_ids = np.arange(ND) if doc_ids is None else doc_ids
    base = np.zeros(ND); base[doc_ids] = 1.0
    L = ledger(P, base, tau); L.update(label=label, form=form, tau=tau)
    ks = ("S", "D", "Smm", "Dmm", "C", "E", "Cmm", "Emm"); bs = {q: [] for q in ks}; bs["DmS"] = []; bs["kept"] = []
    for _ in range(B):
        Lb = ledger(P, boot_mult(doc_ids), tau)
        for q in ks: bs[q].append(Lb[q])
        bs["DmS"].append(Lb["Dmm"] - Lb["Smm"]); bs["kept"].append(Lb["kept"])
    L["DmS"] = L["Dmm"] - L["Smm"]
    est = dict(S=L["S"], D=L["D"], Smm=L["Smm"], Dmm=L["Dmm"], C=L["C"], E=L["E"], Cmm=L["Cmm"], Emm=L["Emm"], DmS=L["DmS"], kept=L["kept"])
    L["se"] = {q: float(np.std(v, ddof=1)) for q, v in bs.items()}
    L["ci"] = {q: [est[q] - 1.96 * L["se"][q], est[q] + 1.96 * L["se"][q]] for q in bs}
    L["boot_mean_shift"] = {q: float(np.mean(v) - est[q]) for q, v in bs.items() if q in ("Smm", "Dmm", "DmS")}
    print(f"{label:28s} tau={tau:3d} kept={L['kept']:.3f} n={int(L['n']):6d} Kx={L['Kx']:6d} Kw={L['Kw']:6d} Smm={L['Smm']:.3f} Dmm={L['Dmm']:.3f} D-S={L['DmS']:.3f}+-{1.96 * L['se']['DmS']:.3f}", flush=True)
    return L

res = dict(B=B, docs=ND, tokens=len(toks))
res["taus"] = [run("lemma", "lemma", t) for t in (1, 2, 3, 5, 10, 20, 50)]
res["variants"] = [run(f"{f}", f, t) for f in ("lemma+pos", "surface") for t in (10, 20, 50)]
res["pos"] = [run(f"POS {p}", "lemma", t, mask=(pos_arr == p)) for p in ("n", "v", "a", "r") for t in (10, 20)]
res["collections"] = [run(f"brown{c + 1}", "lemma", t, mask=np.isin(d_arr, np.where(coll == c)[0]), doc_ids=np.where(coll == c)[0]) for c in (0, 1) for t in (10, 20)]
res["collection_sizes"] = [int((coll == c).sum()) for c in (0, 1)]

# ---- cross-check of the standard errors: half-sampling and delete-one-document jackknife
def se_check(tau, form="lemma"):
    P = prep(form); out = {}
    hs = []
    for _ in range(500):
        m = np.zeros(ND); m[rng.choice(ND, ND // 2, replace=False)] = 1.0; L = ledger(P, m, tau); hs.append((L["Smm"], L["Dmm"], L["Dmm"] - L["Smm"]))
    jk = []
    for i in range(ND):
        m = np.ones(ND); m[i] = 0.0; L = ledger(P, m, tau); jk.append((L["Smm"], L["Dmm"], L["Dmm"] - L["Smm"]))
    hs, jk = np.array(hs), np.array(jk)
    out["half"] = [float(x) for x in hs.std(0, ddof=1)]; out["jackknife"] = [float(x) for x in np.sqrt((ND - 1) / ND * ((jk - jk.mean(0)) ** 2).sum(0))]
    return out
res["se_check"] = {str(t): se_check(t) for t in (1, 20)}
for t, v in res["se_check"].items():
    b = [r for r in res["taus"] if r["tau"] == int(t)][0]
    print("tau", t, "SE (Smm, Dmm, D-S): bootstrap", [round(b["se"][q], 4) for q in ("Smm", "Dmm", "DmS")], "half", [round(x, 4) for x in v["half"]], "jackknife", [round(x, 4) for x in v["jackknife"]], flush=True)
# ---- profiles: doubt per lemma, latitude per synset, on well-attested items (n >= 20); lemma strings are POS-blind
def strat_entropy(s, o):
    df = pd.DataFrame({"s": s, "o": o}); c = df.groupby(["s", "o"]).size().rename("c").reset_index()
    n = c.groupby("s")["c"].transform("sum"); p = c["c"] / n; c["h"] = -p * np.log2(p)
    g = c.groupby("s").agg(n=("c", "sum"), H=("h", "sum"), K=("c", "size")); g["Hmm"] = g["H"] + (g["K"] - 1) / (2 * g["n"] * LN2); return g
xx = FORMS["lemma"][0]
gD = strat_entropy(xx, W); gS = strat_entropy(W, xx)
pd_ = gD[gD["n"] >= 20].copy(); ps_ = gS[gS["n"] >= 20].copy()
rho_D = spearmanr(np.log2(pd_["n"]), pd_["Hmm"]); rho_S = spearmanr(np.log2(ps_["n"]), ps_["Hmm"])
lemma_of = {i: l for l, i in {t[1]: None for t in toks}.items()} if False else None
names = sorted({t[1] for t in toks}); name_of = {i: n for i, n in enumerate(names)}
top = pd_.sort_values("Hmm", ascending=False).head(12)
res["profiles"] = dict(
    doubt=dict(n_items=int(len(pd_)), spearman=float(rho_D.statistic), p=float(rho_D.pvalue), mean_Hmm=float(pd_["Hmm"].mean()), frac_zero_plugin=float((pd_["H"] == 0).mean()),
               bins=[dict(lo=float(a), hi=float(b), n=int(len(g)), mean=float(g["Hmm"].mean()), eff=float(2 ** g["Hmm"].mean())) for (a, b), g in
                     [((lo, hi), pd_[(pd_["n"] >= lo) & (pd_["n"] < hi)]) for lo, hi in [(20, 40), (40, 80), (80, 160), (160, 320), (320, 10 ** 7)]] if len(g)],
               top=[dict(lemma=name_of[i], n=int(r["n"]), Hmm=float(r["Hmm"]), senses=int(r["K"])) for i, r in top.iterrows()]),
    latitude=dict(n_items=int(len(ps_)), spearman=float(rho_S.statistic), p=float(rho_S.pvalue), mean_Hmm=float(ps_["Hmm"].mean()), frac_zero_plugin=float((ps_["H"] == 0).mean()),
                  bins=[dict(lo=float(a), hi=float(b), n=int(len(g)), mean=float(g["Hmm"].mean())) for (a, b), g in
                        [((lo, hi), ps_[(ps_["n"] >= lo) & (ps_["n"] < hi)]) for lo, hi in [(20, 40), (40, 80), (80, 160), (160, 10 ** 7)]] if len(g)]))
json.dump(res, open(OUT / "semcor_results.json", "w"), indent=1)
print("doubt profile spearman", res["profiles"]["doubt"]["spearman"], "items", res["profiles"]["doubt"]["n_items"])
print("top:", [(t["lemma"], t["n"], round(t["Hmm"], 2)) for t in res["profiles"]["doubt"]["top"][:6]])
