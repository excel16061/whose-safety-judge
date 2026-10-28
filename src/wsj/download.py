"""Download DICES-350/990 from GitHub, pinned to one commit and checked by SHA-256."""

import hashlib
import urllib.request
from pathlib import Path

COMMIT = "b5596d9edd585361967991a591abc7b11fb1f7a3"
BASE = f"https://raw.githubusercontent.com/google-research-datasets/dices-dataset/{COMMIT}"

# file name on disk -> (URL, SHA-256)
FILES = {
    "diverse_safety_adversarial_dialog_350.csv": (
        f"{BASE}/350/diverse_safety_adversarial_dialog_350.csv",
        "63cb8620eb04e64957f9e45b1f7cb3d079c4eed8e8b28909f46e0b7038ce4301",
    ),
    "diverse_safety_adversarial_dialog_990.csv": (
        f"{BASE}/990/diverse_safety_adversarial_dialog_990.csv",
        "14332b6391cdee80383b765c734de3bebf6d8553141ee7d2f47ef0910a12fcf7",
    ),
    "README_350.md": (
        f"{BASE}/350/README.md",
        "0e9d5b8fa1ae3d102a77ca4f773a2551b9c0a56f88f5cf825e5db2d405ebd5f3",
    ),
    "README_990.md": (
        f"{BASE}/990/README.md",
        "f4d7431ecca4e96bb503bb4a1ba84cc11a99d83f9a90a3af04f4fbd2546ea19e",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manual_instructions(raw_dir: Path) -> str:
    lines = [
        "Automatic download failed. Download these files by hand and save them",
        f"under {raw_dir} with exactly these names:",
        "",
    ]
    for name, (url, digest) in FILES.items():
        lines += [f"  {name}", f"    from   {url}", f"    sha256 {digest}", ""]
    lines.append("Then run `wsj download` again; it will only verify the checksums.")
    return "\n".join(lines)


def download_all(raw_dir: Path, opener=urllib.request.urlopen) -> list[str]:
    """Download missing files and verify every checksum. Returns a list of problems (empty = OK)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, (url, digest) in FILES.items():
        target = raw_dir / name
        if not target.exists():
            try:
                with opener(url, timeout=120) as r:
                    data = r.read()
                tmp = target.with_suffix(target.suffix + ".part")
                tmp.write_bytes(data)
                tmp.rename(target)
            except Exception as e:  # network errors, HTTP errors
                problems.append(f"{name}: download failed ({e})")
                continue
        got = sha256(target)
        if got != digest:
            problems.append(f"{name}: checksum mismatch (expected {digest}, got {got})")
    return problems
