"""dna_experiment.py -- the ledger of the genetic code on six real genomes.
Meaning W = amino acid (or stop), form X = codon, meaning map = genetic code (pi is a function: D = 0).
Purposes Y: which genome a codon came from; whether a codon came from a highly expressed E. coli gene."""
import os, sys, json, gzip, itertools, math
import numpy as np
from Bio import SeqIO
from Bio.Data import CodonTable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
from ledgerlib import ent, mi

RNG = np.random.default_rng(20260929)
STOP = {"TAA", "TAG", "TGA"}
TBL = CodonTable.standard_dna_table
CODONS = ["".join(c) for c in itertools.product("TCAG", repeat=3)]
CIDX = {c: i for i, c in enumerate(CODONS)}
AA = {c: (TBL.forward_table[c] if c in TBL.forward_table else "*") for c in CODONS}
AAS = sorted(set(AA.values()))
DAYHOFF = {"C": "Cys", **{a: "Small" for a in "AGPST"}, **{a: "AcidAmide" for a in "DENQ"},
           **{a: "Basic" for a in "HKR"}, **{a: "Aliphatic" for a in "ILMV"}, **{a: "Aromatic" for a in "FWY"}, "*": "Stop"}
CLS = sorted(set(DAYHOFF.values()))
W_OF = np.array([AAS.index(AA[c]) for c in CODONS])            # codon -> aa index
G_OF = np.array([CLS.index(DAYHOFF[AA[c]]) for c in CODONS])   # codon -> class index

def read_fasta(path, opener=open):
    recs, h, buf = [], None, []
    with opener(path, "rt") as f:
        for l in f:
            if l.startswith(">"):
                if h is not None: recs.append((h, "".join(buf)))
                h, buf = l[1:].strip(), []
            else: buf.append(l.strip().upper())
    if h is not None: recs.append((h, "".join(buf)))
    return recs

def cds_from_gb(path):
    out = []
    for rec in SeqIO.parse(path, "genbank"):
        for f in rec.features:
            if f.type == "CDS":
                q = f.qualifiers
                out.append((rec.id + ":" + q.get("locus_tag", ["?"])[0], str(f.extract(rec.seq)).upper()))
    return out

def gene_counts(seqs):
    """Return (n_genes x 64 count matrix, dict of filter statistics)."""
    rows, stat = [], dict(input=len(seqs), bad_length=0, bad_alphabet=0, internal_stop=0, kept=0)
    for h, s in seqs:
        if len(s) % 3: stat["bad_length"] += 1; continue
        if set(s) - set("ACGT"): stat["bad_alphabet"] += 1; continue
        cod = [s[i:i + 3] for i in range(0, len(s), 3)]
        if any(c in STOP for c in cod[:-1]): stat["internal_stop"] += 1; continue
        v = np.zeros(64)
        for c in cod: v[CIDX[c]] += 1
        rows.append(v); stat["kept"] += 1
    return np.array(rows), stat

GENOMES = {
 "E. coli K-12":        ("U00096.3",    "fa", str(RAW / "codonu" / "e_coli.fasta")),
 "S. aureus NCTC 8325": ("CP000253.1",  "fa", str(RAW / "codonu" / "Staphylococcus_aureus_nucleotide.fasta")),
 "S. argenteus":        ("FR821777.2",  "fa", str(RAW / "codonu" / "Staphylococcus_argenteus_nucleotide.fasta")),
 "S. agnetis":          ("CP045927.1",  "fa", str(RAW / "codonu" / "Staphylococcus_agnetis_nucleotide.fasta")),
 "C. simulans PES1":    ("NZ_CP014634.1","gb", str(RAW / "codoff" / "Coryne_simulans_PES1.gbk")),
 "A. thaliana plastid": ("NC_000932.1", "gb", str(RAW / "biopython" / "NC_000932.gb")),
}

def funcs(N):
    """N: length-64 count vector -> dict of ledger terms for meaning maps aa and class."""
    p = N / N.sum()
    pw = np.bincount(W_OF, weights=p, minlength=len(AAS)); pg = np.bincount(G_OF, weights=p, minlength=len(CLS))
    Hx, Hw, Hg = ent(p), ent(pw), ent(pg)
    return dict(E=Hx, C=Hw, S=Hx - Hw, Hg=Hg, Hw_g=Hw - Hg)

def panel_info(Ns):
    """Y uniform over the columns of Ns (list of 64-vectors).  Returns relevance ledger."""
    K = len(Ns); Pyx = np.array([N / N.sum() for N in Ns]) / K          # [y, x]
    Pyw = np.zeros((K, len(AAS))); Pyg = np.zeros((K, len(CLS)))
    for x in range(64):
        Pyw[:, W_OF[x]] += Pyx[:, x]; Pyg[:, G_OF[x]] += Pyx[:, x]
    Ix, Iw, Ig = mi(Pyx), mi(Pyw), mi(Pyg)
    return dict(I_X=Ix, I_W=Iw, I_G=Ig, I_class=Ig, I_aa_given_class=Iw - Ig, L=Ix - Iw, share=(Ix - Iw) / Ix)

def main():
    out = dict(genomes={}, filters={}, panel={}, pairs=[], heg={}, per_aa={})
    per_gene, tot = {}, {}
    for name, (acc, kind, path) in GENOMES.items():
        seqs = read_fasta(path) if kind == "fa" else cds_from_gb(path)
        M, st = gene_counts(seqs)
        per_gene[name] = M; tot[name] = M.sum(0); out["filters"][name] = dict(accession=acc, **st)
    # ---- per genome ledger with gene-bootstrap intervals
    B = 400
    for name, M in per_gene.items():
        N = M.sum(0); f = funcs(N)
        gc = sum(N[i] * sum(ch in "GC" for ch in CODONS[i]) for i in range(64)) / (3 * N.sum())
        sense = np.array([c not in STOP for c in CODONS]); gc3 = sum(N[i] for i in range(64) if sense[i] and CODONS[i][2] in "GC") / N[sense].sum()
        bs = {k: [] for k in ("E", "C", "S")}
        for _ in range(B):
            idx = RNG.integers(0, len(M), len(M)); g = funcs(M[idx].sum(0))
            for k in bs: bs[k].append(g[k])
        ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in bs.items()}
        out["genomes"][name] = dict(n_genes=int(len(M)), n_codons=int(N.sum()), GC=float(gc), GC3=float(gc3), **{k: float(v) for k, v in f.items()},
                                    rho=float(f["S"] / f["E"]), perp_codon=2 ** f["E"], perp_aa=2 ** f["C"], perp_syn=2 ** f["S"], ci=ci)
    # ---- per amino acid latitude (effective number of codons), all genomes
    for name, M in per_gene.items():
        N = M.sum(0); rows = []
        for ai, a in enumerate(AAS):
            cs = [i for i in range(64) if W_OF[i] == ai]; v = N[cs]
            rows.append(dict(aa=a, deg=len(cs), p=float(v.sum() / N.sum()), eff=float(2 ** ent(v / v.sum())) if v.sum() > 0 else None))
        out["per_aa"][name] = rows
    # ---- panel: genome identity
    names = list(per_gene)
    Ns = [tot[n] for n in names]
    out["panel"]["point"] = panel_info(Ns)
    bsp = {k: [] for k in out["panel"]["point"]}
    for _ in range(B):
        Nb = [per_gene[n][RNG.integers(0, len(per_gene[n]), len(per_gene[n]))].sum(0) for n in names]
        r = panel_info(Nb)
        for k in bsp: bsp[k].append(r[k])
    out["panel"]["ci"] = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in bsp.items()}
    out["panel"]["names"] = names
    out["panel"]["log2K"] = math.log2(len(names))
    # ---- pairs
    for a, b in itertools.combinations(range(len(names)), 2):
        r = panel_info([Ns[a], Ns[b]]); ga, gb = out["genomes"][names[a]], out["genomes"][names[b]]
        out["pairs"].append(dict(a=names[a], b=names[b], dGC3=abs(ga["GC3"] - gb["GC3"]), **r, frac=r["L"] / r["I_X"]))
    # ---- purpose 2: highly expressed genes in E. coli
    heg = {s for _, s in read_fasta(str(RAW / "freqgen" / "ecoli.heg.fna"))}
    ec = read_fasta(GENOMES["E. coli K-12"][2]); ec_ok = [(h, s) for h, s in ec if len(s) % 3 == 0 and not set(s) - set("ACGT") and not any(s[i:i+3] in STOP for i in range(0, len(s) - 3, 3))]
    is_heg = np.array([s in heg for _, s in ec_ok]); M = per_gene["E. coli K-12"]
    assert len(M) == len(ec_ok)
    def heg_info(M, mask):
        return panel_info([M[~mask].sum(0), M[mask].sum(0)]) | dict(pY1=float(M[mask].sum() / M.sum()))
    # weights: use the natural class frequency, not 1/2
    def heg_info_nat(M, mask):
        Nn, Nh = M[~mask].sum(0), M[mask].sum(0); w = np.array([Nn.sum(), Nh.sum()]) / (Nn.sum() + Nh.sum())
        Pyx = np.array([Nn / Nn.sum() * w[0], Nh / Nh.sum() * w[1]])
        Pyw = np.zeros((2, len(AAS)))
        for x in range(64): Pyw[:, W_OF[x]] += Pyx[:, x]
        return dict(I_X=mi(Pyx), I_W=mi(Pyw), L=mi(Pyx) - mi(Pyw), share=(mi(Pyx) - mi(Pyw)) / mi(Pyx), pY1=float(w[1]), HY=ent(w))
    r0 = heg_info_nat(M, is_heg); bs = {k: [] for k in ("I_X", "I_W", "L", "share")}
    for _ in range(B):
        i0 = RNG.choice(np.where(~is_heg)[0], (~is_heg).sum()); i1 = RNG.choice(np.where(is_heg)[0], is_heg.sum())
        mask = np.r_[np.zeros(len(i0), bool), np.ones(len(i1), bool)]; Mb = np.vstack([M[i0], M[i1]])
        r = heg_info_nat(Mb, mask)
        for k in bs: bs[k].append(r[k])
    out["heg"] = dict(n_heg_matched=int(is_heg.sum()), n_heg_file=len(heg), n_genes=int(len(M)), point=r0,
                      ci={k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in bs.items()})
    json.dump(out, open(OUT / "dna_results.json", "w"), indent=1)
    np.savez(OUT / "codon_counts.npz", codons=np.array(CODONS), **{n: tot[n] for n in names})
    return out

if __name__ == "__main__":
    o = main()
    print("== filters"); [print(f"  {k:22s}", v) for k, v in o["filters"].items()]
    print("== genomes"); 
    for k, g in o["genomes"].items():
        print(f"  {k:22s} genes={g['n_genes']:5d} codons={g['n_codons']:8d} GC3={g['GC3']:.3f}  E={g['E']:.3f} C={g['C']:.3f} S={g['S']:.3f} [{g['ci']['S'][0]:.3f},{g['ci']['S'][1]:.3f}] rho={g['rho']:.3f} perp_syn={g['perp_syn']:.3f}  H(class)={g['Hg']:.3f} H(aa|class)={g['Hw_g']:.3f}")
    print("== panel", {k: round(v, 4) for k, v in o["panel"]["point"].items()}, " log2K=", round(o["panel"]["log2K"], 3))
    print("   ci", {k: [round(x, 4) for x in v] for k, v in o["panel"]["ci"].items()})
    print("== pairs (frac of species information carried by silent choice):")
    for p in sorted(o["pairs"], key=lambda z: -z["dGC3"]): print(f"  {p['a'][:14]:14s} {p['b'][:14]:14s} dGC3={p['dGC3']:.3f} I_X={p['I_X']:.4f} I_W={p['I_W']:.4f} L={p['L']:.4f} frac={p['frac']:.3f}")
    print("== HEG", o["heg"])
