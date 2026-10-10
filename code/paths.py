"""paths.py -- every script finds raw data in ROOT/raw and writes results to ROOT/data_out."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = ROOT / "data_out"
OUT.mkdir(exist_ok=True)
