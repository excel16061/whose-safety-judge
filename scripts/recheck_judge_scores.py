# Re-score a few cached Qwen2.5-1.5B items from scratch and compare with the cache.
# Read-only: nothing is written to the cache. Run from the repo root:
#   HF_HUB_OFFLINE=1 .venv/bin/python scripts/recheck_judge_scores.py
import json, sys, torch
sys.path.insert(0, "src")
from wsj.judges import HFBackend, MODELS, softmax_pair
from wsj.prompts import PROMPTS, build_messages
from wsj.cli import _load
from wsj.data import items_table
from wsj.paths import default_paths
m = "Qwen/Qwen2.5-1.5B-Instruct"
cached = {}
for l in open("cache/judge_scores.jsonl"):
    r = json.loads(l)
    if r["model"] == m and r["dataset"] == "350":
        cached[r["item_id"]] = r
items = items_table(_load(default_paths(), "350")).set_index("item_id")
b = HFBackend(m, MODELS[m], threads=8)
P = PROMPTS["P0"]
for th in (8, 12):
    torch.set_num_threads(th)
    for iid in (1, 8, 349):  # 1, 8 = pilot rows (12 threads); 349 = resumed row (8 threads)
        it = items.loc[iid]
        txt = b.render(build_messages(P, it.context, it.response))
        lg, _ = b.next_token_logits(txt)
        print(f"threads={th} item={iid} fresh p={softmax_pair(lg[9454], lg[2753]):.6f} cached p={cached[iid]['p_unsafe']:.6f}")
it = items.loc[8]
enc = b.tok(b.render(build_messages(P, it.context, it.response)), return_tensors="pt", add_special_tokens=False)
out = b.model.generate(**enc, max_new_tokens=5, do_sample=False)
print("greedy answer item 8:", repr(b.tok.decode(out[0, enc["input_ids"].shape[1]:])))
