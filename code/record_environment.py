"""record_environment.py -- write the versions of the software used to data/ENVIRONMENT.txt."""
import sys, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT
import numpy, scipy, pandas, Bio, nltk
lines = [f"python {platform.python_version()}", f"numpy {numpy.__version__}", f"scipy {scipy.__version__}", f"pandas {pandas.__version__}", f"biopython {Bio.__version__}", f"nltk {nltk.__version__}"]
(ROOT / "data").mkdir(exist_ok=True); (ROOT / "data" / "ENVIRONMENT.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
