import shutil
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class FakeBackend:
    """Stands in for a transformers model. Says "unsafe" when the text contains 'stupid'."""

    def __init__(self, name="fake/model", revision="rev0"):
        self.name, self.revision = name, revision
        self.vocab = {"Yes": 0, "No": 1, "Maybe": 2}
        self.calls = 0

    def render(self, messages):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "\n<assistant>"

    def first_token_id(self, word):
        return self.vocab[word], 1

    def token_text(self, token_id):
        return {v: k for k, v in self.vocab.items()}[token_id]

    def next_token_logits(self, text):
        self.calls += 1
        response = text.split("final response:\n")[-1]
        yes = 2.0 if "stupid" in response else -2.0
        return np.array([yes, 0.0, -1.0]), len(text.split())

    def describe(self):
        return {"fake": True}


@pytest.fixture
def fake_backend():
    return FakeBackend()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project root with the synthetic fixture CSVs in data/raw."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for f in FIXTURES.glob("*.csv"):
        shutil.copy(f, raw / f.name)
    monkeypatch.setenv("WSJ_ROOT", str(tmp_path))
    return tmp_path
