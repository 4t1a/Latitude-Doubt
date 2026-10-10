"""make_assets.py -- turn the JSON results into CSV files for the plots, LaTeX table bodies and number macros.
The paper never contains a transcribed number: every figure in the text is a macro defined here."""
import json, math, csv
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT
R = str(ROOT) + "/"
os.makedirs(R + "data", exist_ok=True); os.makedirs(R + "tables", exist_ok=True)
import numpy as np
from ledgerlib import ledger_from_joint
# the three-meaning lexicon of Figure 1 (rows: river edge, tilt, financial institution; columns: shore, bank, lender)
P_EX = np.array([[0.25, 0.25, 0.0], [0.0, 0.125, 0.0], [0.0, 0.25, 0.125]])
_L = ledger_from_joint(P_EX)
open(R + "tables/example_numbers.tex", "w").write("".join("\\newcommand{\\Ex%s}{%.3f}\n" % (k, v) for k, v in _L.items()))
dna = json.load(open(R + "data_out/dna_results.json")); ide = json.load(open(R + "data_out/dna_identification.json"))
sem = json.load(open(R + "data_out/semcor_results.json")); ver = json.load(open(R + "data_out/verification_identities.json"))
est = json.load(open(R + "data_out/estimator_validation.json")); emb = json.load(open(R + "data_out/embedding_results.json"))["summary"]
SHORT = {"E. coli K-12": "Eco", "S. aureus NCTC 8325": "Sau", "S. argenteus": "Sar", "S. agnetis": "Sag", "C. simulans PES1": "Csi", "A. thaliana plastid": "Ath"}
G = dna["genomes"]; names = list(G)
M = {}   # macros
def mac(k, v): M[k] = v
def sgn(x):
    return x if not x.startswith('-') else (x[1:] if float(x) == 0 else '\\ensuremath{-}' + x[1:])
def th(n): return f"{n:,}".replace(",", "{,}")
def f(x, d=3): return f"{x:.{d}f}"
def pct(x, d=1): return f"{100 * x:.{d}f}"
def wcsv(path, header, rows):
    with open(R + path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
# ---------------- genomes
wcsv("data/genomes_depth.csv", ["idx", "short", "name", "Hclass", "Haa_class", "S", "E", "rho", "GC3"],
     [[i, SHORT[n], n, G[n]["Hg"], G[n]["Hw_g"], G[n]["S"], G[n]["E"], G[n]["rho"], G[n]["GC3"]] for i, n in enumerate(names)])
rows = []
for n in names:
    for r in dna["per_aa"][n]:
        if r["eff"] is not None: rows.append([SHORT[n], r["aa"], r["deg"], r["p"], r["eff"]])
wcsv("data/genomes_bars.csv", ["name", "Hclass", "Haa_class", "S", "rhopct"], [[n, G[n]["Hg"], G[n]["Hw_g"], G[n]["S"], 100 * G[n]["rho"]] for n in names])
wcsv("data/per_aa.csv", ["genome", "aa", "deg", "p", "eff"], rows)
pp = dna["pairs"]
wcsv("data/pairs_bact.csv", ["I_X", "share"], [[p["I_X"], 100 * p["frac"]] for p in pp if "A. thaliana plastid" not in (p["a"], p["b"])])
wcsv("data/pairs_plastid.csv", ["I_X", "share"], [[p["I_X"], 100 * p["frac"]] for p in pp if "A. thaliana plastid" in (p["a"], p["b"])])
wcsv("data/pairs.csv", ["a", "b", "dGC3", "I_X", "I_W", "L", "frac", "plastid"],
     [[SHORT[p["a"]], SHORT[p["b"]], p["dGC3"], p["I_X"], max(p["I_W"], 1e-5), max(p["L"], 1e-5), p["frac"], int("Ath" in (SHORT[p["a"]], SHORT[p["b"]]))] for p in dna["pairs"]])
acc = ide["accuracy"]; wcsv("data/heldout_accuracy.csv", ["bin", "mid", "n", "codon", "aa", "silent"],
     [[i, [70, 150, 300, 550][i], acc["codon"]["by_length"][i]["n"], acc["codon"]["by_length"][i]["macro_recall"], acc["aa"]["by_length"][i]["macro_recall"], acc["silent"]["by_length"][i]["macro_recall"]] for i in range(4)])
T = sem["taus"]
wcsv("data/semcor_tau.csv", ["tau", "kept", "n", "Smm", "Slo", "Shi", "Dmm", "Dlo", "Dhi", "S", "D"],
     [[t["tau"], t["kept"], t["n"], t["Smm"], t["ci"]["Smm"][0], t["ci"]["Smm"][1], t["Dmm"], t["ci"]["Dmm"][0], t["ci"]["Dmm"][1], t["S"], t["D"]] for t in T])
db = sem["profiles"]["doubt"]["bins"]; lb = sem["profiles"]["latitude"]["bins"]
wcsv("data/semcor_doubt.csv", ["mid", "n_items", "mean", "eff"], [[math.sqrt(b["lo"] * min(b["hi"], 640)), b["n"], b["mean"], b["eff"]] for b in db])
wcsv("data/semcor_lat.csv", ["mid", "n_items", "mean"], [[math.sqrt(b["lo"] * min(b["hi"], 320)), b["n"], b["mean"]] for b in lb])
wcsv("data/estimator_bias.csv", ["n", "S_plugin", "S_mm", "S_cs", "D_plugin", "D_mm", "D_cs"],
     [[n, est["bias"]["plugin"]["S"][i], est["bias"]["mm"]["S"][i], est["bias"]["cs"]["S"][i], est["bias"]["plugin"]["D"][i], est["bias"]["mm"]["D"][i], est["bias"]["cs"]["D"][i]] for i, n in enumerate(est["ns"])])
wcsv("data/embedding_bins.csv", ["mid", "n_genes", "capacity", "rate"], [[math.sqrt(b["lo"] * b["hi"]), b["n_genes"], b["capacity"], b["rate"]] for b in emb["bins"]])
wcsv("data/certified.csv", ["n", "halfwidth", "q95"], [[r["n"], r["halfwidth"], r["q95_abs_error"]] for r in est["certified"]["rows"]])
# ---------------- tables
def tab(name, body): open(R + f"tables/{name}.tex", "w").write(body)
# T1: per-genome ledger
L = ["\\begin{tabular}{@{}l r r r r r r r r@{}}", "\\toprule", "Genome & Genes & Codons & GC3 & $E$ & $2^{E}$ & $C$ & $\\Lat$ & $\\Lat/E$ \\\\", "\\midrule"]
for n in names:
    g = G[n]; L.append(f"{n} & {th(g['n_genes'])} & {th(g['n_codons'])} & {f(g['GC3'],2)} & {f(g['E'])} & {f(2**g['E'],1)} & {f(g['C'])} & {f(g['S'])} & {pct(g['rho'])}\\,\\% \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_genomes", "\n".join(L))
# T2: provenance / filters
L = ["\\begin{tabular}{@{}l l r r r@{}}", "\\toprule", "Genome & Accession & CDS in file & Excluded & Used \\\\", "\\midrule"]
for n in names:
    s = dna["filters"][n]; ex = s["input"] - s["kept"]
    L.append(f"{n} & \\texttt{{{s['accession'].replace('_', chr(92) + '_')}}} & {th(s['input'])} & {ex} & {th(s['kept'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_filters", "\n".join(L))
# T3: relevance ledger for the genome-identity purpose
P = dna["panel"]["point"]; C_ = dna["panel"]["ci"]
def ci(k, d=4): return f"[{C_[k][0]:.{d}f},\\,{C_[k][1]:.{d}f}]"
L = ["\\begin{tabular}{@{}p{4.5cm} l r r r@{}}", "\\toprule", "Level of meaning & Term & Bits per codon & 95\\% interval & Share \\\\", "\\midrule",
     f"Chemical class ($G$) & $I(G;Y)$ & {f(P['I_G'],4)} & {ci('I_G')} & {pct(P['I_G']/P['I_X'])}\\,\\% \\\\",
     f"Amino acid within its class & $I(W;Y\\mid G)$ & {f(P['I_aa_given_class'],4)} & {ci('I_aa_given_class')} & {pct(P['I_aa_given_class']/P['I_X'])}\\,\\% \\\\",
     f"Synonymous choice: leakage & $I(X;Y\\mid W)$ & {f(P['L'],4)} & {ci('L')} & {pct(P['L']/P['I_X'])}\\,\\% \\\\",
     "\\midrule", f"Codon (total) & $I(X;Y)$ & {f(P['I_X'],4)} & {ci('I_X')} & 100.0\\,\\% \\\\", "\\bottomrule", "\\end{tabular}"]
tab("tab_panel", "\n".join(L))
# T4: held-out
hg = ide["heldout_gain_macro"]; tp = ide["train_panel_info"]
L = ["\\begin{tabular}{@{}p{4.2cm} r r r@{}}", "\\toprule", " & \\multicolumn{2}{c}{Information, bits per codon} & Identification \\\\", "\\cmidrule(lr){2-3}",
     "Evidence used & Training law & Held-out gain & macro recall \\\\", "\\midrule"]
def rec(m):
    b = acc[m]["by_length"]; lo = (b[0]["n"] * b[0]["macro_recall"] + b[1]["n"] * b[1]["macro_recall"]) / (b[0]["n"] + b[1]["n"]); hi = (b[2]["n"] * b[2]["macro_recall"] + b[3]["n"] * b[3]["macro_recall"]) / (b[2]["n"] + b[3]["n"])
    return acc[m]["macro_recall"], lo, hi
for lab, m, tr, hgk in (("Codons", "codon", tp["I_X"], hg["codon"]), ("Amino acids only", "aa", tp["I_W"], hg["aa"]), ("Synonymous choice only", "silent", tp["L"], hg["silent"])):
    a, lo, hi = rec(m); L.append(f"{lab} & {f(tr,4)} & {f(hgk,4)} & {f(a,3)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_heldout", "\n".join(L))
# T5: SemCor tau  (Miller-Madow with 95% bootstrap normal intervals; plug-in shown for comparison)
def pm(v, se, d=3): return sgn(f"{v:.{d}f}") + f"\\,$\\pm$\\,{1.96 * se:.{d}f}"
L = ["\\begin{tabular}{@{}r r r r r r@{}}", "\\toprule", "$\\tau$ & Tokens kept & Fraction kept & $\\Lat$ & $\\Dbt$ & $\\Dbt-\\Lat$ \\\\", "\\midrule"]
for t in T:
    L.append(f"{t['tau']} & {th(int(t['n']))} & {pct(t['kept'],0)}\\,\\% & {pm(t['Smm'], t['se']['Smm'])} & {pm(t['Dmm'], t['se']['Dmm'])} & {pm(t['DmS'], t['se']['DmS'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_semcor", "\n".join(L))
# T5b: sensitivity table at tau = 20 (and tau = 10 for the difference)
def get(lst, label, tau): return [x for x in lst if x["label"] == label and x["tau"] == tau][0]
rows = [("Form of the word", [("Lemma (main analysis)", T, "lemma"), ("Lemma and part of speech", sem["variants"], "lemma+pos"), ("Inflected word as written", sem["variants"], "surface")]),
        ("Part of speech (form: lemma)", [("Nouns", sem["pos"], "POS n"), ("Verbs", sem["pos"], "POS v"), ("Adjectives", sem["pos"], "POS a"), ("Adverbs", sem["pos"], "POS r")]),
        ("Collection (form: lemma)", [("Brown 1 (103 documents)", sem["collections"], "brown1"), ("Brown 2 (83 documents)", sem["collections"], "brown2")])]
L = ["\\begin{tabular}{@{}p{4.1cm} r r r r r@{}}", "\\toprule", " & \\multicolumn{4}{c}{$\\tau=20$} & $\\tau=10$ \\\\", "\\cmidrule(lr){2-5}\\cmidrule(l){6-6}", "Analysis & Kept & $\\Lat$ & $\\Dbt$ & $\\Dbt-\\Lat$ & $\\Dbt-\\Lat$ \\\\"]
forest = []
for gi, (gname, items) in enumerate(rows):
    L.append("\\midrule"); L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{gname}}}}} \\\\")
    for nm, lst, lab in items:
        if lst is T: r20 = [x for x in T if x["tau"] == 20][0]; r10 = [x for x in T if x["tau"] == 10][0]
        else: r20, r10 = get(lst, lab, 20), get(lst, lab, 10)
        L.append(f"{nm} & {pct(r20['kept'],0)}\\,\\% & {pm(r20['Smm'], r20['se']['Smm'], 2)} & {pm(r20['Dmm'], r20['se']['Dmm'], 2)} & {pm(r20['DmS'], r20['se']['DmS'], 2)} & {pm(r10['DmS'], r10['se']['DmS'], 2)} \\\\")
        forest.append([nm, r20["DmS"], 1.96 * r20["se"]["DmS"]])
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_semcor_sens", "\n".join(L))
wcsv("data/semcor_forest.csv", ["label", "est", "err"], forest)
r20 = [x for x in T if x["tau"] == 20][0]; r1 = T[0]
mac("SemDmSTwenty", sgn(f(r20["DmS"], 2))); mac("SemDmSTwentyHW", f(1.96 * r20["se"]["DmS"], 2)); mac("SemDmSOne", sgn(f(r1["DmS"], 2))); mac("SemDmSOneHW", f(1.96 * r1["se"]["DmS"], 2))
mac("SemB", th(sem["B"]))
lp = get(sem["variants"], "lemma+pos", 20); su = get(sem["variants"], "surface", 20); mac("SemDmSLemmaPos", sgn(f(lp["DmS"], 2))); mac("SemDmSSurface", sgn(f(su["DmS"], 2))); mac("SemDmSSurfaceHW", f(1.96 * su["se"]["DmS"], 2)); mac("SemDmSLemmaPosHW", f(1.96 * lp["se"]["DmS"], 2))
for nm, lab in (("N", "POS n"), ("V", "POS v"), ("A", "POS a"), ("R", "POS r")):
    r = get(sem["pos"], lab, 20); mac("SemPos" + nm, sgn(f(r["DmS"], 2))); mac("SemPos" + nm + "HW", f(1.96 * r["se"]["DmS"], 2))
b1, b2 = get(sem["collections"], "brown1", 20), get(sem["collections"], "brown2", 20); mac("SemBrOne", sgn(f(b1["DmS"], 2))); mac("SemBrTwo", sgn(f(b2["DmS"], 2)))
mac("SemPlugGap", f(max(abs(t["S"] - t["Smm"]) for t in T if t["tau"] >= 20) , 3)); mac("SemPlugGapD", f(max(abs(t["D"] - t["Dmm"]) for t in T if t["tau"] >= 20), 3))
sc = sem["se_check"]; mac("SemSeBootOne", f(T[0]["se"]["DmS"], 3)); mac("SemSeJackOne", f(sc["1"]["jackknife"][2], 3)); mac("SemSeHalfOne", f(sc["1"]["half"][2], 3))
mac("SemSeBootTwenty", f(r20["se"]["DmS"], 3)); mac("SemSeJackTwenty", f(sc["20"]["jackknife"][2], 3)); mac("SemSeHalfTwenty", f(sc["20"]["half"][2], 3)); mac("SemJackRatio", f(sc["20"]["jackknife"][2] / r20["se"]["DmS"], 1))
# ---------- inventory test
inv = json.load(open(R + "data_out/inventory_results.json"))
NAMES = dict(als="Albanian", arb="Arabic", bul="Bulgarian", cat="Catalan", cmn="Chinese (Mandarin)", dan="Danish", ell="Greek", eng="English", eus="Basque", fin="Finnish", fra="French", glg="Galician", heb="Hebrew", hrv="Croatian",
             ind="Indonesian", isl="Icelandic", ita="Italian", **{"ita-iwn": "Italian (second wordnet)"}, jpn="Japanese", lit="Lithuanian", nld="Dutch", nno="Norwegian (Nynorsk)", nob="Norwegian (Bokm\\aa l)", pol="Polish", por="Portuguese", ron="Romanian",
             slk="Slovak", slv="Slovene", spa="Spanish", swe="Swedish", tha="Thai", zsm="Malay")
big = {k: v for k, v in inv.items() if v["synsets"] >= 10000}
wcsv("data/inventory_big.csv", ["code", "Lat", "Dbt"], [[k, v["Lat"], v["Dbt"]] for k, v in big.items() if k != "eng"])
wcsv("data/inventory_eng.csv", ["code", "Lat", "Dbt"], [["eng", inv["eng"]["Lat"], inv["eng"]["Dbt"]]])
wcsv("data/inventory_semcor.csv", ["code", "Lat", "Dbt"], [["lemma (all)", r1["Smm"], r1["Dmm"]], ["lemma (core)", r20["Smm"], r20["Dmm"]], ["word (core)", su["Smm"], su["Dmm"]]])
mac("InvN", str(len(big))); mac("InvLatGreater", str(sum(v["Lat"] > v["Dbt"] for v in big.values()))); mac("InvDbtGreater", str(sum(v["Lat"] < v["Dbt"] for v in big.values()))); mac("InvAll", str(len(inv)))
mac("InvEngLat", f(inv["eng"]["Lat"], 2)); mac("InvEngDbt", f(inv["eng"]["Dbt"], 2)); mac("InvEngPairs", th(inv["eng"]["pairs"])); mac("InvEngSynsets", th(inv["eng"]["synsets"]))
mac("InvDiffMin", sgn(f(min(v["Lat"] - v["Dbt"] for v in big.values()), 2))); mac("InvDiffMax", sgn(f(max(v["Lat"] - v["Dbt"] for v in big.values()), 2)))
L = ["\\begin{tabular}{@{}l l r r r r r@{}}", "\\toprule", "Code & Language & Synsets & Pairs & $\\Lat$ & $\\Dbt$ & $\\Lat-\\Dbt$ \\\\", "\\midrule"]
for k in sorted(inv, key=lambda z: -inv[z]["synsets"]):
    v = inv[k]; star = "" if v["synsets"] >= 10000 else "$^{\\dagger}$"
    L.append(f"{k}{star} & {NAMES.get(k, k)} & {th(v['synsets'])} & {th(v['pairs'])} & {f(v['Lat'])} & {f(v['Dbt'])} & {f(v['Lat'] - v['Dbt'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_inventory", "\n".join(L))
# ---------- estimator: where is Miller--Madow the closest?
ns = est["ns"]; mm_from = None
for i in range(len(ns)):
    if all(abs(est["bias"]["mm"][q][j]) <= min(abs(est["bias"]["plugin"][q][j]), abs(est["bias"]["cs"][q][j])) for q in "SD" for j in range(i, len(ns))): mm_from = ns[i]; break
mac("EstMMFrom", th(mm_from) if mm_from else "never")
subs = json.load(open(R + "data_out/semcor_subsample.json")); mac("SubCInc", f(subs[-1]["C"] - subs[-2]["C"], 2)); mac("SubEInc", f(subs[-1]["E"] - subs[-2]["E"], 2)); mac("SubHalfN", th(subs[-2]["n"]))
# ---------- embedding additions
mac("EmbOK", str(int(round(emb["success"] * emb["n_genes"])))); mac("EmbFail", str(emb["detected_failures"])); mac("EmbSilent", str(emb["silent_errors"]))
mac("EmbKeyBits", f"{emb['mean_key_bits']:.0f}"); mac("EmbKeyRatio", f(emb["mean_key_over_payload"], 1)); mac("EmbSecBits", str(-emb["security_log2_tv"]))
# T6: verification
v1, v2, v3 = ver["ledger"], ver["derived_meaning"], ver["leakage"]
def sci(x): 
    if x == 0: return "0"
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e; return f"${m:.1f}\\times10^{{{e}}}$"
L = ["\\begin{tabular}{@{}p{7.9cm} r p{3.6cm}@{}}", "\\toprule", "Check & Instances & Result \\\\", "\\midrule",
     f"Ledger partitions $E=I+\\Lat$, $C=I+\\Dbt$, $H(W,X)=I+\\Lat+\\Dbt$ & {th(v1['instances'])} & max error {sci(v1['max_abs_err_partition'])} \\\\",
     f"Balance $E-C=\\Lat-\\Dbt$ & {th(v1['instances'])} & max error {sci(v1['max_abs_err_balance'])} \\\\",
     f"Vocabulary bound $\\Dbt\\ge C-\\log_2 N+\\Lat$ & {th(v1['instances'])} & {v1['vocab_bound_violations']} violations \\\\",
     f"Huffman length minus entropy, for $W$ and for $W\\mid X$ & {th(v1['huffman_instances'])} & in $[0,1)$ to within $3\\times10^{{-16}}$ \\\\",
     f"Sufficiency $I(M;Y)=I(X;Y)$ & {v2['instances']} & max error {sci(v2['sufficiency_max_err'])} \\\\",
     f"Minimality; all set partitions of alphabets of 3 to 6 messages & {th(v2['sufficient_partitions_seen'])} & each sufficient partition refines $M$ \\\\",
     f"Monotonicity in purpose & {v2['monotonicity_cases']} & {v2['monotonicity_holds']} of {v2['monotonicity_cases']} hold \\\\",
     f"Leakage equals $I(X;Y)-I(W;Y)$ & {th(v3['instances'])} & max error {sci(v3['err_function_case'])} \\\\",
     f"Depth identity $\\Lk_Y(W_2)=\\Lk_Y(W_1)+I(W_1;Y\\mid W_2)$ & {th(v3['instances'])} & max error {sci(v3['err_depth_identity'])} \\\\",
     f"Purpose $Y=X$ gives leakage $H(X\\mid W)=\\Lat$ & {th(v3['instances'])} & max error {sci(v3['err_purpose_identity'])} \\\\",
     "\\bottomrule", "\\end{tabular}"]
tab("tab_verif", "\n".join(L))
# T7: certified interval
L = ["\\begin{tabular}{@{}r r r r r@{}}", "\\toprule", "$n$ & $\\varepsilon$ & Certified half-width (bits) & 95th percentile of actual error & Coverage \\\\", "\\midrule"]
for r in est["certified"]["rows"]:
    L.append(f"{th(r['n'])} & {f(r['eps_l1'])} & {f(r['halfwidth'])} & {f(r['q95_abs_error'])} & {f(r['coverage'],3)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_certified", "\n".join(L))
# T8: embedding
L = ["\\begin{tabular}{@{}r r r r r@{}}", "\\toprule", "Length (codons) & Genes & Capacity, bits/codon & Rate $k/n$, bits/codon & Decoded correctly \\\\", "\\midrule"]
for b in emb["bins"]:
    hi = "" if b["hi"] > 700 else f"--{b['hi']-1}"; lo = f"{b['lo']}"
    lab = f"{lo}--{b['hi']-1}" if b["hi"] <= 700 else f"$\\geq$ {b['lo']}"
    L.append(f"{lab} & {b['n_genes']} & {f(b['capacity'])} & {f(b['rate'])} & {pct(b['success'],0)}\\,\\% \\\\")
L.append("\\midrule"); L.append(f"All & {emb['n_genes']} & {f(emb['mean_capacity_mu_per_codon'])} & {f(emb['mean_rate_per_codon'])} & {pct(emb['success'],0)}\\,\\% \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_embed", "\n".join(L))
# ---------------- number macros
E_ = G["E. coli K-12"]; heg = dna["heg"]; hp = heg["point"]; hc = heg["ci"]
mac("EcoS", f(E_["S"])); mac("EcoE", f(E_["E"])); mac("EcoC", f(E_["C"])); mac("EcoRho", pct(E_["rho"])); mac("EcoPerp", f(E_["perp_syn"], 2))
mac("SminGenome", f(min(G[n]["S"] for n in names))); mac("SmaxGenome", f(max(G[n]["S"] for n in names)))
mac("rhoMin", pct(min(G[n]["rho"] for n in names))); mac("rhoMax", pct(max(G[n]["rho"] for n in names)))
mac("PanelIX", f(P["I_X"], 3)); mac("PanelIW", f(P["I_W"], 4)); mac("PanelL", f(P["L"], 3)); mac("PanelFrac", pct(P["L"] / P["I_X"])); mac("PanelLogK", f(dna["panel"]["log2K"], 3))
mac("PanelFracLo", pct(C_["share"][0])); mac("PanelFracHi", pct(C_["share"][1]))
mac("PanelIG", f(P["I_G"], 4)); mac("PanelIAA", f(P["I_aa_given_class"], 4))
fr = [p["frac"] for p in dna["pairs"]]; mac("PairFracMin", pct(min(fr), 0)); mac("PairFracMax", pct(max(fr), 0))
frn = [p["frac"] for p in dna["pairs"] if "A. thaliana plastid" not in (p["a"], p["b"])]; mac("PairFracNoPlastidMin", pct(min(frn), 0))
frp = [p["frac"] for p in dna["pairs"] if "A. thaliana plastid" in (p["a"], p["b"])]; mac("PairFracPlastidMin", pct(min(frp), 0)); mac("PairFracPlastidMax", pct(max(frp), 0))
mac("HegIX", f(hp["I_X"], 4)); mac("HegIW", f(hp["I_W"], 4)); mac("HegL", f(hp["L"], 4)); mac("HegFrac", pct(hp["L"] / hp["I_X"])); mac("HegHY", f(hp["HY"], 3)); mac("HegPone", pct(hp["pY1"]))
mac("HegMatched", str(heg["n_heg_matched"])); mac("HegFile", str(heg["n_heg_file"])); mac("HegGenes", f"{th(heg['n_genes'])}")
mac("HegLo", pct(hc["share"][0])); mac("HegHi", pct(hc["share"][1]))
mac("HoldCodon", f(hg["codon"], 4)); mac("HoldAA", f(hg["aa"], 4)); mac("HoldSilent", f(hg["silent"], 4)); mac("TrainCodon", f(tp["I_X"], 4)); mac("TrainAA", f(tp["I_W"], 4)); mac("TrainSilent", f(tp["L"], 4))
mac("AccCodon", f(acc["codon"]["macro_recall"], 3)); mac("AccAA", f(acc["aa"]["macro_recall"], 3)); mac("AccSilent", f(acc["silent"]["macro_recall"], 3))
t20 = [t for t in T if t["tau"] == 20][0]; t1 = T[0]
mac("SemTokens", f"{th(sem['tokens'])}"); mac("SemDocs", str(sem["docs"])); mac("SemKx", f"{th(t1['Kx'])}"); mac("SemKw", f"{th(t1['Kw'])}")
mac("SemSTwenty", f(t20["Smm"], 2)); mac("SemDTwenty", f(t20["Dmm"], 2)); mac("SemKeptTwenty", pct(t20["kept"], 0)); mac("SemSOne", f(t1["Smm"], 2)); mac("SemDOne", f(t1["Dmm"], 2))
mac("SemSPlainOne", f(t1["S"], 3)); mac("SemDPlainOne", f(t1["D"], 3)); mac("SemCOne", f(t1["C"], 2)); mac("SemEOne", f(t1["E"], 2))
pdp = sem["profiles"]["doubt"]; mac("SemSpearman", f(pdp["spearman"], 2)); mac("SemNItems", f"{th(pdp['n_items'])}")
mac("SemDoubtPeak", f(max(b["mean"] for b in pdp["bins"]), 2)); pk = max(pdp["bins"], key=lambda b: b["mean"]); mac("SemEffPeak", f(pk["eff"], 1)); mac("SemPeakLo", str(int(pk["lo"]))); mac("SemPeakHi", str(int(pk["hi"]) - 1)); mac("SemDoubtTopBand", f(pdp["bins"][-1]["mean"], 2)); mac("SemTopBandN", str(pdp["bins"][-1]["n"])); lat_ = sem["profiles"]["latitude"]; mac("SemLatLow", f(lat_["bins"][0]["mean"], 2)); mac("SemLatHigh", f(lat_["bins"][-1]["mean"], 2)); mac("SemLatSpearman", f(lat_["spearman"], 2)); mac("SemDoubtLow", f(pdp["bins"][0]["mean"], 2)); mac("SemDoubtHigh", f(max(b["mean"] for b in pdp["bins"]), 2)); mac("SemEffLow", f(pdp["bins"][0]["eff"], 1)); mac("SemEffHigh", f(max(b["eff"] for b in pdp["bins"]), 1))
mac("EmbGenes", str(emb["n_genes"])); mac("EmbCodons", f"{th(emb['total_codons'])}"); mac("EmbBits", f"{th(emb['total_bits'])}")
mac("EmbCap", f(emb["mean_capacity_mu_per_codon"], 3)); mac("EmbRate", f(emb["mean_rate_per_codon"], 3)); mac("EmbEff", pct(emb["mean_rate_per_codon"] / emb["mean_capacity_mu_per_codon"]))
mac("EmbChanged", pct(emb["mean_changed_fraction"], 0)); mac("EmbChiStego", f(emb["chi_per_dof_stego"], 2)); mac("EmbChiNat", f(emb["chi_per_dof_natural"], 2))
mac("EmbPstego", f(emb["pooled_chi_stego"]["p"], 2)); mac("EmbPnat", f(emb["pooled_chi_natural"]["p"], 3)); mac("EmbDof", str(emb["pooled_chi_stego"]["dof"]))
sc = emb["showcase"]; mac("ShowBits", str(sc["payload_bits"])); mac("ShowCap", f(sc["capacity_bits"], 0)); mac("ShowCodons", str(sc["n_codons"])); mac("ShowChanged", str(sc["codons_changed"])); mac("ShowBytes", str(sc["message_bytes"]))
mac("HoldGenes", th(sum(ide["n_test_genes"]))); mac("ShowAttempts", str(sc["attempts"]))
mac("VerN", f"{th(v1['instances'])}")
mac("EstTruthS", f(est["truth"]["S"], 3)); mac("EstTruthD", f(est["truth"]["D"], 3))
i2k = est["ns"].index(2000); i1e5 = est["ns"].index(100000)
mac("EstBiasSPlain", f(est["bias"]["plugin"]["S"][i2k], 3)); mac("EstBiasDPlain", f(est["bias"]["plugin"]["D"][i2k], 3)); mac("EstBiasSMM", f(est["bias"]["mm"]["S"][i1e5], 3)); mac("EstBiasDMM", f(est["bias"]["mm"]["D"][i1e5], 3))
cr = est["certified"]["rows"]; mac("CertRatioLo", f(min(r["halfwidth"] / r["q95_abs_error"] for r in cr), 0)); mac("CertRatioHi", f(max(r["halfwidth"] / r["q95_abs_error"] for r in cr), 0))
effs = [2 ** G[n]["E"] for n in names]; mac("EffMin", f(min(effs), 1)); mac("EffMax", f(max(effs), 1)); mac("EcoEff", f(2 ** E_["E"], 1)); mac("EffShareMin", pct(min(effs) / 64, 0)); mac("EffShareMax", pct(max(effs) / 64, 0))
sc = emb["showcase"]; nat, ste = sc["natural_head"], sc["stego_head"]
def cell(c, changed): return ("\\textcolor{leak}{\\textbf{%s}}" % c) if changed else c
NC = 16; row1 = " ".join(cell(a, a != b) for a, b in list(zip(nat, ste))[:NC]); row2 = " ".join(cell(b, a != b) for a, b in list(zip(nat, ste))[:NC])
open(R + "tables/showcase_codons.tex", "w").write("\\ttfamily natural\\ \\ " + row1 + "\\\\\n\\ttfamily embedded\\ " + row2 + "\\\\\n")
mac("ShowDiffCount", str(sum(a != b for a, b in list(zip(nat, ste))[:NC])))
sub = json.load(open(R + "data_out/semcor_subsample.json"))
L = ["\\begin{tabular}{@{}r r r r r r@{}}", "\\toprule", "Fraction of corpus & Tokens & $C$ & $E$ & $\\Lat$ & $\\Dbt$ \\\\", "\\midrule"]
for r in sub: L.append(f"{pct(r['frac'],0)}\\,\\% & {th(r['n'])} & {f(r['C'],2)} & {f(r['E'],2)} & {f(r['S'],3)} & {f(r['D'],3)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; tab("tab_subsample", "\n".join(L))
mac("SubCLow", f(sub[0]["C"], 2)); mac("SubCHigh", f(sub[-1]["C"], 2)); mac("SubELow", f(sub[0]["E"], 2)); mac("SubEHigh", f(sub[-1]["E"], 2))
top = sem["profiles"]["doubt"]["top"]; mac("SemTopA", top[0]["lemma"]); mac("SemTopAH", f(top[0]["Hmm"], 1)); mac("SemTopB", top[1]["lemma"]); mac("SemTopC", top[2]["lemma"]); mac("SemTopD", top[3]["lemma"])
lab = []
for n in names:
    g = G[n]; lab.append("\\node[anchor=west,font=\\scriptsize\\sffamily,text=form!70!black] at (axis cs:%.3f,{%s}) {%.2f\\,bits\\enspace(%.0f\\,\\%%)};" % (g["E"] + 0.07, n, g["S"], 100 * g["rho"]))
open(R + "tables/genome_labels.tex", "w").write("\n".join(lab) + "\n")
open(R + "tables/numbers.tex", "w").write("% generated by code/make_assets.py -- do not edit\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print(len(M), "macros;", "tables:", sorted(__import__("os").listdir(R + "tables")))
