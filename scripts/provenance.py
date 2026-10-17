"""Provenance manifest: commit, environment, and SHA-256 of every data and result file."""
import glob, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def git(*a):
    try: return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None

files = sorted(set(glob.glob("data/*") + glob.glob("results/*.json") + glob.glob("results/*.png") + glob.glob("results/*.npz") + ["requirements.txt"]))
files = [f for f in files if Path(f).is_file() and not f.endswith("PROVENANCE.json")]
print(json.dumps({
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
    "python": sys.version.split()[0], "platform": platform.platform(),
    "seeds": {"permutation_null": 0, "tiny_random_model": 0},
    "files": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
}, indent=2))
