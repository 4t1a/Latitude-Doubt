"""extract_semcor.py -- read SemCor through NLTK and store (document, lemma, synset, pos, surface form) for every sense-tagged chunk."""
import os, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW
os.environ["NLTK_DATA"] = str(RAW / "nltk_data")
import nltk
nltk.data.path.insert(0, str(RAW / "nltk_data"))
from nltk.corpus import semcor
from nltk.tree import Tree
toks = []
for fid in semcor.fileids():
    for sent in semcor.tagged_sents(fid, tag="sem"):
        for ch in sent:
            if isinstance(ch, Tree) and hasattr(ch.label(), "synset"):
                lem = ch.label(); toks.append((fid, lem.name().lower(), lem.synset().name(), lem.synset().pos(), " ".join(ch.leaves()).lower()))
pickle.dump(toks, open(RAW / "semcor_tokens.pkl", "wb"))
print(len(toks), "tokens from", len({t[0] for t in toks}), "documents")
