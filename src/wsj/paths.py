"""Project directory layout. The root is the current directory unless WSJ_ROOT is set."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.cache, self.results, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    return Paths(Path(os.environ.get("WSJ_ROOT", os.getcwd())).resolve())
