"""Download and verify the DICES data. Same as `wsj download`."""

import sys

from wsj.download import download_all, manual_instructions
from wsj.paths import default_paths

if __name__ == "__main__":
    raw = default_paths().raw
    problems = download_all(raw)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(manual_instructions(raw), file=sys.stderr)
        sys.exit(1)
    print(f"All files present and verified in {raw}")
