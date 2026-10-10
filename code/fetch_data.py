"""fetch_data.py -- download the public raw data used in the paper into ./raw and record SHA-256 sums.
Sources: example files of the CodonU, codoff and freqgen repositories, the Biopython test suite, and the NLTK data packages semcor, wordnet and omw-1.4."""
import hashlib, io, sys, tarfile, urllib.request, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, RAW

def get(url):
    print("fetching", url, flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ledger-of-meaning-fetch"}), timeout=300) as r:
        return r.read()

def zip_member(zbytes, suffix):
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    names = [n for n in z.namelist() if n.endswith(suffix)]
    assert len(names) == 1, (suffix, names)
    return z.read(names[0])

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)

def main():
    cu = get("https://codeload.github.com/SouradiptoC/CodonU/zip/refs/heads/master")
    for f in ("e_coli.fasta", "Staphylococcus_aureus_nucleotide.fasta", "Staphylococcus_argenteus_nucleotide.fasta", "Staphylococcus_agnetis_nucleotide.fasta"):
        save(RAW / "codonu" / f, zip_member(cu, "Examples/Nucleotide/" + f))
    cd = get("https://codeload.github.com/Kalan-Lab/codoff/zip/refs/heads/main")
    tar = tarfile.open(fileobj=io.BytesIO(zip_member(cd, "Csimulans_Data.tar.gz")), mode="r:gz")
    m = [x for x in tar.getmembers() if x.name.endswith("Coryne_simulans_PES1.gbk")][0]
    save(RAW / "codoff" / "Coryne_simulans_PES1.gbk", tar.extractfile(m).read())
    save(RAW / "biopython" / "NC_000932.gb", get("https://raw.githubusercontent.com/biopython/biopython/master/Tests/GenBank/NC_000932.gb"))
    fg = get("https://codeload.github.com/IQTLabs/freqgen/zip/refs/heads/master")
    save(RAW / "freqgen" / "ecoli.heg.fna", zip_member(fg, "examples/ecoli.heg.fna"))
    import nltk
    for pkg in ("semcor", "wordnet", "omw-1.4"):
        nltk.download(pkg, download_dir=str(RAW / "nltk_data"), quiet=True)
    sums = []
    for p in sorted(list((RAW / "codonu").glob("*")) + list((RAW / "codoff").glob("*")) + list((RAW / "biopython").glob("*")) + list((RAW / "freqgen").glob("*")) + list((RAW / "nltk_data" / "corpora").glob("*.zip"))):
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}")
    (ROOT / "data").mkdir(exist_ok=True); (ROOT / "data" / "CHECKSUMS.txt").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))

if __name__ == "__main__":
    main()
