import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wsj import judges as J
from conftest import FakeBackend
from wsj.prompts import PROMPTS

ITEMS = pd.DataFrame({
    "item_id": [1, 2],
    "context": ["USER: hi\nLAMDA: hello", "USER: hi"],
    "response": ["you are stupid", "have a nice day"],
})


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_scores_from_answer_logits(tmp_path, fake_backend):
    cache = J.ScoreCache(tmp_path / "c.jsonl")
    rows, s = J.score_items(fake_backend, PROMPTS["P0"], ITEMS, "350", cache)
    # Yes logit 2 vs No logit 0 -> sigmoid(2); -2 vs 0 -> sigmoid(-2)
    assert rows[0]["p_unsafe"] == pytest.approx(sigmoid(2.0))
    assert rows[1]["p_unsafe"] == pytest.approx(sigmoid(-2.0))
    assert 0 < rows[0]["mass_on_answers"] < 1
    assert s.n_scored == 2 and s.n_cached == 0


def test_reversed_prompt_reads_the_no_token(tmp_path, fake_backend):
    cache = J.ScoreCache(tmp_path / "c.jsonl")
    rows, _ = J.score_items(fake_backend, PROMPTS["P2"], ITEMS, "350", cache)
    # P2 asks "is it safe?", so P(unsafe) = P("No") = sigmoid(0 - 2)
    assert rows[0]["p_unsafe"] == pytest.approx(sigmoid(-2.0))


def test_cache_prevents_recomputation_and_persists(tmp_path, fake_backend):
    path = tmp_path / "c.jsonl"
    J.score_items(fake_backend, PROMPTS["P0"], ITEMS, "350", J.ScoreCache(path))
    calls = fake_backend.calls
    rows, s = J.score_items(fake_backend, PROMPTS["P0"], ITEMS, "350", J.ScoreCache(path))
    assert fake_backend.calls == calls and s.n_cached == 2
    # a different prompt is a different cache entry
    J.score_items(fake_backend, PROMPTS["P1"], ITEMS, "350", J.ScoreCache(path))
    assert fake_backend.calls == calls + 2


def test_run_log_records_prompt_and_tokens(tmp_path, fake_backend):
    log = tmp_path / "run_log.jsonl"
    J.score_items(fake_backend, PROMPTS["P0"], ITEMS, "350", J.ScoreCache(tmp_path / "c.jsonl"), run_log=log)
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["prompt_id"] == "P0"
    assert entry["prompt_template"] == PROMPTS["P0"].user_template
    assert entry["unsafe_token"][1] == "Yes" and entry["safe_token"][1] == "No"


def test_load_scores_filters_on_registered_revision(tmp_path):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    path = tmp_path / "c.jsonl"
    J.score_items(FakeBackend(model, "some-other-revision"), PROMPTS["P0"], ITEMS, "350", J.ScoreCache(path))
    assert J.load_scores(path, model, "P0", "350") == {}
    J.score_items(FakeBackend(model, J.MODELS[model]), PROMPTS["P0"], ITEMS, "350", J.ScoreCache(path))
    assert set(J.load_scores(path, model, "P0", "350")) == {1, 2}


def _hf_cached(name):
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (root / ("models--" + name.replace("/", "--"))).exists()


@pytest.mark.slow
@pytest.mark.skipif(not _hf_cached("Qwen/Qwen2.5-0.5B-Instruct"), reason="model not downloaded")
def test_real_model_gives_valid_probability(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    b = J.HFBackend(name, J.MODELS[name], threads=4)
    assert b.first_token_id("Yes")[1] == 1 and b.first_token_id("No")[1] == 1
    rows, _ = J.score_items(b, PROMPTS["P0"], ITEMS.iloc[1:], "350", J.ScoreCache(tmp_path / "c.jsonl"))
    assert 0.0 < rows[0]["p_unsafe"] < 1.0
    # deterministic: a second forward pass gives the same number
    rows2, _ = J.score_items(b, PROMPTS["P0"], ITEMS.iloc[1:], "350", J.ScoreCache(tmp_path / "d.jsonl"))
    assert rows2[0]["p_unsafe"] == rows[0]["p_unsafe"]


def test_same_text_in_two_datasets_is_cached_separately(tmp_path, fake_backend):
    """Regression: the cache key once ignored dataset/item, so identical texts collided."""
    path = tmp_path / "c.jsonl"
    J.score_items(fake_backend, PROMPTS["P0"], ITEMS, "350", J.ScoreCache(path))
    J.score_items(FakeBackend(), PROMPTS["P0"], ITEMS, "990", J.ScoreCache(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted({r["dataset"] for r in rows}) == ["350", "990"] and len(rows) == 4
