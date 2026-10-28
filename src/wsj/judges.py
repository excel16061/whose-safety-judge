"""LLM judges: P(unsafe) from next-token logits, with a disk cache and a run log.

No text is generated: each conversation gets one forward pass and only the
logits of the answer words "Yes" and "No" are read.
"""

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from .prompts import PROMPTS, Prompt, build_messages

# Pre-registered judges (docs/methodology.md, section 4).
MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
}
SENSITIVITY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class Backend(Protocol):
    name: str
    revision: str

    def render(self, messages: list[dict]) -> str: ...
    def first_token_id(self, word: str) -> tuple[int, int]: ...  # (id, number of tokens in word)
    def token_text(self, token_id: int) -> str: ...
    def next_token_logits(self, text: str) -> tuple[np.ndarray, int]: ...  # (logits, n_prompt_tokens)
    def describe(self) -> dict: ...


class HFBackend:
    """transformers causal LM on CPU, float32, deterministic."""

    def __init__(self, name: str, revision: str, threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.name, self.revision = name, revision
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32)
        self.model.eval()

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def first_token_id(self, word):
        ids = self.tok.encode(word, add_special_tokens=False)
        return ids[0], len(ids)

    def token_text(self, token_id):
        return self.tok.decode([token_id])

    def next_token_logits(self, text):
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        with self._torch.inference_mode():
            logits = self.model(**enc).logits[0, -1].float().numpy()
        return logits, int(enc["input_ids"].shape[1])

    def describe(self):
        import torch
        import transformers
        return {"torch": torch.__version__, "transformers": transformers.__version__,
                "dtype": "float32", "device": "cpu", "threads": torch.get_num_threads(),
                "python": platform.python_version()}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_key(model: str, revision: str, prompt_id: str, dataset: str, item_id: int, rendered: str) -> str:
    # The rendered text already contains the prompt and the conversation.
    return _sha(json.dumps([model, revision, prompt_id, dataset, int(item_id), rendered]))


class ScoreCache:
    """Append-only JSONL cache: one line per scored (model, revision, prompt, rendered text)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._rows[row["key"]] = row

    def get(self, key):
        return self._rows.get(key)

    def put(self, row: dict):
        self._rows[row["key"]] = row
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


def softmax_pair(l_unsafe: float, l_safe: float) -> float:
    m = max(l_unsafe, l_safe)
    a, b = np.exp(l_unsafe - m), np.exp(l_safe - m)
    return float(a / (a + b))


def mass_on(logits: np.ndarray, ids: list[int]) -> float:
    z = logits - logits.max()
    e = np.exp(z)
    return float(e[ids].sum() / e.sum())


@dataclass
class RunSummary:
    model: str
    prompt_id: str
    dataset: str
    n_items: int
    n_scored: int
    n_cached: int
    seconds: float


def score_items(backend: Backend, prompt: Prompt, items, dataset: str, cache: ScoreCache,
                run_log: Path | None = None, progress=None) -> tuple[list[dict], RunSummary]:
    """Score every item; return cache rows in item order. `items` is a DataFrame from data.items_table."""
    uid, n_uid = backend.first_token_id(prompt.unsafe_answer)
    sid, n_sid = backend.first_token_id(prompt.safe_answer)
    rows, n_cached, t0 = [], 0, time.time()
    for k, it in enumerate(items.itertuples(index=False)):
        rendered = backend.render(build_messages(prompt, it.context, it.response))
        key = cache_key(backend.name, backend.revision, prompt.prompt_id, dataset, it.item_id, rendered)
        row = cache.get(key)
        if row is None:
            t = time.time()
            logits, n_tok = backend.next_token_logits(rendered)
            row = {
                "key": key, "dataset": dataset, "item_id": int(it.item_id),
                "model": backend.name, "revision": backend.revision, "prompt_id": prompt.prompt_id,
                "p_unsafe": softmax_pair(float(logits[uid]), float(logits[sid])),
                "logit_unsafe_answer": float(logits[uid]), "logit_safe_answer": float(logits[sid]),
                "mass_on_answers": mass_on(logits, [uid, sid]),
                "top_token": backend.token_text(int(np.argmax(logits))),
                "n_prompt_tokens": n_tok, "seconds": time.time() - t,
            }
            cache.put(row)
        else:
            n_cached += 1
        rows.append(row)
        if progress:
            progress(k + 1, len(items))
    summary = RunSummary(backend.name, prompt.prompt_id, dataset, len(items),
                         len(items) - n_cached, n_cached, time.time() - t0)
    if run_log is not None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **summary.__dict__,
            "revision": backend.revision,
            "prompt_template": prompt.user_template,
            "unsafe_answer": prompt.unsafe_answer, "safe_answer": prompt.safe_answer,
            "unsafe_token": [uid, backend.token_text(uid), n_uid],
            "safe_token": [sid, backend.token_text(sid), n_sid],
            "example_rendered_prompt": backend.render(build_messages(prompt, "<context>", "<response>")),
            "environment": backend.describe(),
        }
        with open(run_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
    return rows, summary


def load_scores(cache_path: Path, model: str, prompt_id: str, dataset: str) -> dict[int, float]:
    """item_id -> p_unsafe for the pre-registered revision of `model`."""
    out = {}
    cache = ScoreCache(cache_path)
    for row in cache._rows.values():
        if (row["model"] == model and row["revision"] == MODELS[model]
                and row["prompt_id"] == prompt_id and row["dataset"] == dataset):
            out[row["item_id"]] = row["p_unsafe"]
    return out


def get_prompt(prompt_id: str) -> Prompt:
    return PROMPTS[prompt_id]
