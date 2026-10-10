"""dna_identification.py -- held-out test of the relevance ledger.
Split genes 50/50 per genome; fit class-conditional codon models on the training half; on the test half
(i) identify the genome of each gene from its codons, its amino acids only, or its silent choices only;
(ii) measure the held-out information gain per codon, to be compared with I(X;Y), I(W;Y), L of the training law."""
import os, sys, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
import dna_experiment as dx
from ledgerlib import mi

rng = np.random.default_rng(7)
seqs = {}
for name, (acc, kind, path) in dx.GENOMES.items():
    s = dx.read_fasta(path) if kind == "fa" else dx.cds_from_gb(path)
    M, st = dx.gene_counts(s); seqs[name] = M
names = list(seqs); K = len(names)
W = dx.W_OF; A = 0.5                                   # KT smoothing
res_gene = {m: [] for m in ("codon", "aa", "silent")}; rows = []; lens = []; truth = []
tr_tot, info = [], {m: [] for m in ("codon", "aa", "silent")}
split = {}
for k, n in enumerate(names):
    idx = rng.permutation(len(seqs[n])); h = len(idx) // 2; split[n] = (idx[:h], idx[h:])
trainN = np.array([seqs[n][split[n][0]].sum(0) for n in names])          # [y, x] training counts
def models(trainN):
    Px = (trainN + A) / (trainN + A).sum(1, keepdims=True)                 # P_y(x)
    Pw = np.zeros((K, len(dx.AAS)))
    for x in range(64): Pw[:, W[x]] += trainN[:, x]
    Pw = (Pw + A) / (Pw + A).sum(1, keepdims=True)                          # P_y(w)
    Ps = np.zeros((K, 64))                                                  # P_y(x | w(x))
    for x in range(64):
        same = [i for i in range(64) if W[i] == W[x]]
        Ps[:, x] = (trainN[:, x] + A) / (trainN[:, same] + A).sum(1)
    return np.log2(Px), np.log2(Pw[:, W]), np.log2(Ps)
LX, LW, LS = models(trainN)                                                # each [y, x]
conf = {m: np.zeros((K, K)) for m in ("codon", "aa", "silent")}
gain = {m: [] for m in ("codon", "aa", "silent")}                           # per-codon held-out gain, macro over genomes
for k, n in enumerate(names):
    test = seqs[n][split[n][1]]
    g = {m: [] for m in gain}; 
    for v in test:
        L = v.sum()
        for m, LL in (("codon", LX), ("aa", LW), ("silent", LS)):
            score = LL @ v                                                  # [y] log-likelihood of the gene under each genome
            conf[m][k, np.argmax(score)] += 1
            res_gene[m].append((k, int(L), int(np.argmax(score))))
    # held-out information gain per codon (pooled over the test codons of genome k):
    #   codon: log2 P_k(x) - log2 mean_y P_y(x);   aa: same for the amino-acid law;
    #   silent = codon - aa, which equals log2 P_k(x|w) - log2 P(x|w) with P(x|w) = sum_y P(y|w) P_y(x|w)  (chain rule)
    Nt = test.sum(0)
    mixX = np.log2(np.mean(2.0 ** LX, axis=0)); mixW = np.log2(np.mean(2.0 ** LW, axis=0))
    gx = float((Nt * (LX[k] - mixX)).sum() / Nt.sum()); gw = float((Nt * (LW[k] - mixW)).sum() / Nt.sum())
    gain["codon"].append(gx); gain["aa"].append(gw); gain["silent"].append(gx - gw)
out = dict(names=names, n_test_genes=[int(len(split[n][1])) for n in names], n_train_genes=[int(len(split[n][0])) for n in names])
out["train_panel_info"] = dx.panel_info(list(trainN))
out["heldout_gain_macro"] = {m: float(np.mean(v)) for m, v in gain.items()}
out["heldout_gain_by_genome"] = {m: [float(x) for x in v] for m, v in gain.items()}
bins = [(0, 100), (100, 200), (200, 400), (400, 10 ** 9)]
out["accuracy"] = {}
for m in res_gene:
    arr = np.array(res_gene[m])
    tot = float(np.mean([np.mean(arr[arr[:, 0] == k, 2] == k) for k in range(K)]))       # macro-average recall
    per_bin = []
    for lo, hi in bins:
        sel = (arr[:, 1] >= lo) & (arr[:, 1] < hi)
        rec = [np.mean(arr[sel & (arr[:, 0] == k), 2] == k) for k in range(K) if (sel & (arr[:, 0] == k)).sum() > 0]
        per_bin.append(dict(lo=lo, hi=hi if hi < 10 ** 9 else None, n=int(sel.sum()), macro_recall=float(np.mean(rec))))
    out["accuracy"][m] = dict(macro_recall=tot, by_length=per_bin, confusion=conf[m].tolist())
json.dump(out, open(OUT / "dna_identification.json", "w"), indent=1)
print("train-panel information (bits/codon):", {k: round(v, 4) for k, v in out["train_panel_info"].items()})
print("held-out gain (bits/codon):", {k: round(v, 4) for k, v in out["heldout_gain_macro"].items()})
for m in out["accuracy"]:
    a = out["accuracy"][m]; print(f"{m:7s} macro recall {a['macro_recall']:.3f} | by length:", [(b['lo'], b['hi'], b['n'], round(b['macro_recall'], 3)) for b in a["by_length"]])
print("codon-model confusion (rows=true):"); print(np.array(out["accuracy"]["codon"]["confusion"]).astype(int))
