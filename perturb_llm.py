"""Adds LLM-generated variants (paraphrase, Hindi) for every original in the file.
Used: paraphrase with ollama/qwen2.5:7b (temperature 0.7),
      Hindi with ollama/aya-expanse:8b (--only hindi --temperature 0).
Every generated row was then reviewed by hand (apply_review.py): refusals dropped,
mistranslations corrected.
"""

import argparse
import asyncio
import json

from inspect_ai.model import GenerateConfig, get_model

TEMPLATES = {
    "paraphrase": ("Rewrite the following message so it asks for exactly the same thing using "
                   "different wording. Output only the rewritten message.\n\nMessage: {text}"),
    "hindi": ("Translate the following message into natural, fluent Hindi written in Devanagari "
              "script. Keep the meaning exactly the same. Output only the translation.\n\nMessage: {text}"),
}


async def run(path, model_name,  only, temperature):
    rows = [json.loads(l) for l in open(path)]
    originals = [r for r in rows if r["perturbation"] == "original"]
    model = get_model(model_name, config=GenerateConfig(temperature=temperature, max_tokens=300))
    sem = asyncio.Semaphore(8)

    async def one(r, pert, tpl):
        async with sem:
            out = await model.generate(tpl.format(text=r["input"]))
        return {**r, "id": f"{r['source_id']}-{pert}", "input": out.completion.strip(), "perturbation": pert}

    todo = {k: v for k, v in TEMPLATES.items() if k in only}
    existing = {r["perturbation"] for r in rows}
    clash = existing & set(todo)
    if clash:
        raise SystemExit(f"{path} already has {sorted(clash)} rows; restore a backup before regenerating")
    new = await asyncio.gather(*[one(r, p, t) for r in originals for p, t in todo.items()])
    with open(path, "a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(new)} LLM variants -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", nargs="+", default=list(TEMPLATES), choices=list(TEMPLATES))
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    asyncio.run(run(a.jsonl, a.model, a.only, a.temperature))
