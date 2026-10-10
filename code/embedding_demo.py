"""embedding_demo.py -- Latitude as a channel, on real genes (Theorem 6).
Cover: the amino-acid sequence of real E. coli genes (the meaning, known to both ends).  Form: codons, drawn from the
genome-wide synonymous-usage law q_a(x).  The embedder is an exact arithmetic decoder driven by a key-masked payload."""
import os, sys, json, math, random, time
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
import dna_experiment as dx
from ledgerlib import ent

rng = random.Random(20260929)
name, (acc, kind, path) = "E. coli K-12", dx.GENOMES["E. coli K-12"]
seqs = dx.read_fasta(path)
ok = [(h, s) for h, s in seqs if len(s) % 3 == 0 and not set(s) - set("ACGT") and not any(s[i:i+3] in dx.STOP for i in range(0, len(s) - 3, 3))]
M, _ = dx.gene_counts(seqs); Ngenome = M.sum(0)
CODONS, AA = dx.CODONS, dx.AA
syn = {}
for c in CODONS: syn.setdefault(AA[c], []).append(c)
# exact conditional law and cumulative intervals per amino acid
Q, CUM = {}, {}
for a, cs in syn.items():
    tot = int(sum(Ngenome[dx.CIDX[c]] for c in cs)); cum = Fraction(0); rows = []
    for c in cs:
        p = Fraction(int(Ngenome[dx.CIDX[c]]), tot); rows.append((c, cum, p)); cum += p
    CUM[a] = rows; Q[a] = {c: p for c, _, p in rows}
Hcond = {a: ent([float(p) for p in Q[a].values()]) for a in Q}                       # H(X | W = a)
Vcond = {a: float(sum(float(p) * (-math.log2(float(p)) - Hcond[a]) ** 2 for p in Q[a].values())) for a in Q}
Lmax = {a: max(-math.log2(float(p)) for p in Q[a].values()) for a in Q}                 # worst-case codeword length per amino acid
S_BITS = 64                                                                             # security parameter s of the finite-precision proposition
def key_bits(aa): return int(math.ceil(sum(Lmax[a] for a in aa))) + S_BITS + 1          # P >= log2(1/q_min) + s + 1

def gene_meaning(s): return [AA[s[i:i+3]] for i in range(0, len(s), 3)]

def embed(aa, m, k, R, V, P):
    z = (Fraction(m) + V) / (1 << k); U = (z + R) % 1
    lo, wd, out = Fraction(0), Fraction(1), []
    for a in aa:
        for c, cum, p in CUM[a]:
            if U < lo + wd * (cum + p):
                out.append(c); lo = lo + wd * cum; wd = wd * p; break
    return out

def decode(aa, cods, k, R):
    lo, wd = Fraction(0), Fraction(1)
    for a, c in zip(aa, cods):
        for cc, cum, p in CUM[a]:
            if cc == c: lo = lo + wd * cum; wd = wd * p; break
    s = (lo - R) % 1; j = int(s * (1 << k))
    if s + wd <= Fraction(j + 1, 1 << k): return j            # arc inside a single cell -> payload determined
    return None                                                # arc straddles a cell boundary -> detected failure

def translate(cods): return [AA[c] for c in cods]

# ---- experiment: many genes, random payloads of capacity-derived length
cand = [(h, s) for h, s in ok if 60 <= len(s) // 3 <= 700]
rng.shuffle(cand); genes = cand[:240]
rows = []; t0 = time.time(); stego_counts = np.zeros(64); nat_counts = np.zeros(64); per_gene_stat = []
def chi_gene(cnts, aa):
    """Pearson chi-square of a gene's codon counts against the conditional model, per degree of freedom."""
    x2, dof = 0.0, 0
    for a in sorted(set(aa)):
        idx = [dx.CIDX[c] for c in syn[a]]; n_a = sum(cnts[i] for i in idx)
        if len(idx) < 2 or n_a == 0: continue
        for c in syn[a]:
            e = n_a * float(Q[a][c]); x2 += (cnts[dx.CIDX[c]] - e) ** 2 / e
        dof += len(idx) - 1
    return x2 / max(dof, 1)
for gi, (h, s) in enumerate(genes):
    aa = gene_meaning(s); n = len(aa)
    mu = sum(Hcond[a] for a in aa); sig = math.sqrt(sum(Vcond[a] for a in aa))
    k = max(0, int(math.floor(mu - 3 * sig - 7))); P = key_bits(aa)
    R = Fraction(rng.getrandbits(P), 1 << P); V = Fraction(rng.getrandbits(P), 1 << P); m = rng.getrandbits(k) if k > 0 else 0
    cods = embed(aa, m, k, R, V, P)
    same_protein = translate(cods) == aa; back = decode(aa, cods, k, R)
    nat = [s[i:i+3] for i in range(0, len(s), 3)]
    cs, cn = np.zeros(64), np.zeros(64)
    for c in cods: cs[dx.CIDX[c]] += 1
    for c in nat: cn[dx.CIDX[c]] += 1
    stego_counts += cs; nat_counts += cn
    rows.append(dict(gene=h[:40], n=n, mu=mu, sigma=sig, k=k, P=P, ok=bool(back == m), detected_fail=bool(back is None), protein_preserved=bool(same_protein),
                     changed=float(np.mean([a != b for a, b in zip(cods, nat)])), chi_stego=chi_gene(cs, aa), chi_nat=chi_gene(cn, aa)))
print("elapsed", round(time.time() - t0, 1), "s for", len(genes), "genes")
R_ = rows
res = dict(n_genes=len(rows), all_protein_preserved=all(r["protein_preserved"] for r in rows), success=float(np.mean([r["ok"] for r in rows])),
           detected_failures=int(sum(r["detected_fail"] for r in rows)), silent_errors=int(sum((not r["ok"]) and (not r["detected_fail"]) for r in rows)),
           total_bits=int(sum(r["k"] for r in rows)), total_codons=int(sum(r["n"] for r in rows)),
           mean_capacity_mu_per_codon=float(sum(r["mu"] for r in rows) / sum(r["n"] for r in rows)),
           mean_rate_per_codon=float(sum(r["k"] for r in rows) / sum(r["n"] for r in rows)),
           mean_changed_fraction=float(np.mean([r["changed"] for r in rows])), mean_key_bits=float(np.mean([r["P"] for r in rows])), mean_key_over_payload=float(sum(r["P"] for r in rows) / max(sum(r["k"] for r in rows), 1)), security_log2_tv=-(S_BITS + 2),
           chi_per_dof_stego=float(np.mean([r["chi_stego"] for r in rows])), chi_per_dof_natural=float(np.mean([r["chi_nat"] for r in rows])))
# rate vs length bins
bins = [(60, 120), (120, 200), (200, 350), (350, 701)]; res["bins"] = []
for lo, hi in bins:
    sel = [r for r in rows if lo <= r["n"] < hi]
    if sel: res["bins"].append(dict(lo=lo, hi=hi, n_genes=len(sel), capacity=float(sum(r["mu"] for r in sel) / sum(r["n"] for r in sel)),
                                    rate=float(sum(r["k"] for r in sel) / sum(r["n"] for r in sel)), success=float(np.mean([r["ok"] for r in sel]))))
# pooled goodness of fit of stego and natural codon usage to the model, aa-conditional
def pooled_chi(counts):
    x2, dof = 0.0, 0
    for a, cs in syn.items():
        idx = [dx.CIDX[c] for c in cs]; n_a = counts[idx].sum()
        if len(cs) < 2 or n_a == 0: continue
        for c in cs: e = n_a * float(Q[a][c]); x2 += (counts[dx.CIDX[c]] - e) ** 2 / e
        dof += len(cs) - 1
    return x2, dof
from scipy.stats import chi2
for nm, cnt in (("stego", stego_counts), ("natural", nat_counts)):
    x2, dof = pooled_chi(cnt); res[f"pooled_chi_{nm}"] = dict(x2=x2, dof=dof, p=float(chi2.sf(x2, dof)))
# a showcase: hide a sentence in one gene
MSG = b"Expression is free exactly where meaning is silent."
best = None
for h, s in cand:
    aa = gene_meaning(s); mu = sum(Hcond[a] for a in aa); sig = math.sqrt(sum(Vcond[a] for a in aa)); kmax = int(math.floor(mu - 3 * sig - 7))
    if kmax >= 8 * len(MSG) + 16 and (best is None or len(aa) < len(best[1])): best = (h, aa, s)
h, aa, s = best; k = 8 * len(MSG) + 16; m = int.from_bytes(MSG, "big") << 16 | 0xA5A5
mu = sum(Hcond[a] for a in aa); sig = math.sqrt(sum(Vcond[a] for a in aa)); P = key_bits(aa)
for attempt in range(50):                                        # a detected failure is simply retried with fresh randomness
    R = Fraction(rng.getrandbits(P), 1 << P); V = Fraction(rng.getrandbits(P), 1 << P)
    cods = embed(aa, m, k, R, V, P); back = decode(aa, cods, k, R)
    if back == m: break
nat = [s[i:i+3] for i in range(0, len(s), 3)]
res["showcase"] = dict(gene=h[:60], n_codons=len(aa), message_bytes=len(MSG), payload_bits=k, capacity_bits=mu, attempts=attempt + 1, recovered=(back.to_bytes((k + 7) // 8, "big")[:len(MSG)] == MSG),
                       protein_identical=translate(cods) == aa, codons_changed=int(sum(a != b for a, b in zip(cods, nat))),
                       natural_head=nat[:24], stego_head=cods[:24])
json.dump(dict(summary=res, genes=rows), open(OUT / "embedding_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "showcase"}, indent=1)); print(json.dumps(res["showcase"], indent=1)[:900])
