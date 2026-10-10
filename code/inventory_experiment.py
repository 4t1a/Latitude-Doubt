"""inventory_experiment.py -- the ledger of a sense inventory, with no corpus.
Pairs (synset, lemma) are the observations, each distinct pair counted once: P(w, x) is proportional to 1 on the pairs of a wordnet.
Then Latitude = mean over pairs of log2(lemmas in the pair's synset), Doubt = mean over pairs of log2(synsets of the pair's lemma).
English comes from Princeton WordNet 3.0; the other languages from the Open Multilingual Wordnet, all attached to the same synsets.
The result describes the resource as much as the language: wordnets are built in different ways."""
import os, sys, json, zipfile, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RAW, OUT
from ledgerlib import ent
os.environ["NLTK_DATA"] = str(RAW / "nltk_data")
import nltk
nltk.data.path.insert(0, str(RAW / "nltk_data"))
from nltk.corpus import wordnet as wn

def ledger_of_pairs(pairs):
    pairs = sorted(pairs); N = len(pairs)                  # sorted: summation order, hence the last bits, do not depend on hash randomisation
    nw = collections.Counter(w for w, x in pairs); sx = collections.Counter(x for w, x in pairs)
    Lat = sum(np.log2(nw[w]) for w, x in pairs) / N; Dbt = sum(np.log2(sx[x]) for w, x in pairs) / N
    C = ent(np.array(list(nw.values())) / N); E = ent(np.array(list(sx.values())) / N)
    assert abs((E - C) - (Lat - Dbt)) < 1e-9
    return dict(pairs=N, synsets=len(nw), lemmas=len(sx), C=C, E=E, Lat=float(Lat), Dbt=float(Dbt), mean_syn_per_lemma=N / len(sx), mean_lemmas_per_synset=N / len(nw))

res = {}
eng = set()
for ss in wn.all_synsets():
    key = f"{ss.offset():08d}-{'a' if ss.pos() == 's' else ss.pos()}"
    for l in ss.lemmas(): eng.add((key, l.name().lower()))
res["eng"] = ledger_of_pairs(eng); print("eng", {k: round(v, 3) for k, v in res["eng"].items()}, flush=True)
z = zipfile.ZipFile(RAW / "nltk_data" / "corpora" / "omw-1.4.zip")
for name in sorted(n for n in z.namelist() if n.endswith(".tab")):
    lang = name.split("wn-data-")[1][:-4]; folder = name.split("/")[1]; pairs = set()
    for line in z.read(name).decode("utf8").splitlines():
        if line.startswith("#"): continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[1].endswith("lemma"): pairs.add((parts[0], parts[2].strip().lower()))
    key = lang if lang not in res else f"{lang}-{folder}"
    if len(pairs) >= 1000:
        res[key] = ledger_of_pairs(pairs); res[key]["folder"] = folder; print(key, {k: round(v, 3) for k, v in res[key].items() if k != "folder"}, flush=True)
json.dump(res, open(OUT / "inventory_results.json", "w"), indent=1)
