import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MOULD_DATA", ROOT / "data"))
RAW = DATA / "raw"
META = DATA / "docs_extracted" / "metadata"
SURVEYS = DATA / "docs_extracted" / "surveys"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "docs" / "figures"
