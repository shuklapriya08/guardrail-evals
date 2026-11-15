"""Apply hand review of LLM rewrites: keep YES, replace FIX with your corrected text, drop NO.

  python apply_review.py llm_review.csv data/perturbed.jsonl data/perturbed_validated.jsonl
"""
import sys
import pandas as pd

review_csv, in_jsonl, out_jsonl = sys.argv[1:4]
rev = pd.read_csv(review_csv, encoding="utf-8-sig", dtype=str).fillna("")
rev["label"] = rev["label"].str.strip().str.upper()

bad = rev[~rev.label.isin(["YES", "FIX", "NO"])]
if len(bad):
    sys.exit(f"{len(bad)} rows not labelled YES/FIX/NO, e.g. ids: {bad.id.head(5).tolist()}")
missing = rev[(rev.label == "FIX") & (rev.fixed_text.str.strip() == "")]
if len(missing):
    sys.exit(f"{len(missing)} FIX rows have no fixed_text, e.g. ids: {missing.id.head(5).tolist()}")

print("Review counts:")
print(pd.crosstab(rev.perturbation, rev.label, margins=True), "\n")

p = pd.read_json(in_jsonl, lines=True, dtype={"id": str, "source_id": str})
fixes = rev[rev.label == "FIX"].set_index("id")["fixed_text"].str.strip()
drops = set(rev[rev.label == "NO"].id)

p["input"] = [fixes.get(i, t) for i, t in zip(p["id"], p["input"])]
p["human_fixed"] = p["id"].isin(fixes.index)
p = p[~p["id"].isin(drops)]

p.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
print(f"applied {len(fixes)} fixes, dropped {len(drops)} rows -> {len(p)} rows in {out_jsonl}")
print(p.perturbation.value_counts().to_string())