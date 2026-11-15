# Guardrail evals: do small guard models read intent or keywords?

An evaluation of three small LLM guardrails on [XSTest](https://github.com/paul-rottger/xstest): how often they catch unsafe prompts, how often they wrongly block safe ones, where they fail, and how easily simple rewrites get past them. Built with [Inspect](https://inspect.aisi.org.uk); every number comes with a confidence interval and paired significance test.

**Full write-up: [REPORT.md](REPORT.md)**

## Headline results

| Guard | Catches unsafe (n=200) | Wrongly blocks safe (n=250) |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct, prompted as a guard | 99.0% [97.5, 100.0] | 66.0% [60.0, 72.0] |
| Qwen2.5-7B-Instruct, prompted (4-bit, Ollama) | 88.0% [83.5, 92.5] | 10.0% [6.4, 13.6] |
| Llama Guard 3 1B | 75.5% [69.5, 81.5] | 6.0% [3.2, 9.2] |

95% bootstrap confidence intervals in brackets.

- **A small prompted guard is unusable:** Qwen 1.5B catches nearly everything by blocking two in three harmless questions. The same prompt on Qwen 7B over-blocks only 10%, so the problem is model size, not the prompted approach.
- **Llama Guard over-blocks least but misses 1 in 4 unsafe prompts.** Its misses concentrate on polite requests to justify discrimination (24% caught) and privacy (48%).
- **It reacts to surface form more than intent.** Simple rewrites cut its catch rate from 75% to 47–60%: padding (−28 points), leetspeak (−27), fiction framing (−23), Hindi (−18), typos (−15). Hindi is one of its officially supported languages.
- **An LLM judge shared the guard's blind spots.** Used to validate the rewrites, it treated fiction-framed requests as harmless despite its rubric; validity was established with human labels instead.
- **The pipeline reproduces Meta's published figure:** F1 0.825 on XSTest for Llama Guard 3 1B, against 0.821 on the model card.

## Repository layout

```
guardrail_eval.py     Inspect task: guard model -> safe / unsafe verdict per prompt
export_results.py     Inspect logs -> one CSV row per (model, prompt)
stats.py              catch / false-flag rates with CIs, paired bootstrap + McNemar, robustness, kappa
perturb.py            rule-based rewrites: typos, leetspeak, padding, fiction framing
perturb_llm.py        LLM rewrites: paraphrase, Hindi translation
apply_review.py       applies the hand review of LLM rewrites (keep / correct / drop)
judge_eval.py         LLM judge for rewrite validity, plus export of its verdicts
tests/test_stats.py   checks the statistics on synthetic data with known answers

data/                 smoke test; perturbed prompts as generated and after hand review
results/              exported results behind every number in the report
annotations/          human labels for rewrite validity, the LLM review sheet, judge verdicts
docs/                 the original learning plan
```

## Reproduce

Requires Python 3.11, about 10 GB of disk for models, and Hugging Face access to the gated [XSTest copy](https://huggingface.co/datasets/walledai/XSTest) and [Llama Guard 3 1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B) (accept the terms on each page, then `hf auth login`). Qwen 7B, the judge and the Hindi translations run locally through [Ollama](https://ollama.com).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_stats.py

# 1. three guards on XSTest
inspect eval guardrail_eval.py -T dataset=xstest -T mode=prompted   --model hf/Qwen/Qwen2.5-1.5B-Instruct
inspect eval guardrail_eval.py -T dataset=xstest -T mode=prompted   --model ollama/qwen2.5:7b --temperature 0
inspect eval guardrail_eval.py -T dataset=xstest -T mode=llamaguard --model mockllm/model
python export_results.py logs/ results/results.csv --task guardrail
python stats.py results/results.csv

# 2. robustness (Llama Guard): rewrites, hand review, rerun
python perturb.py results/results.csv data/perturbed.jsonl --n 60
python perturb_llm.py data/perturbed.jsonl --model ollama/qwen2.5:7b --only paraphrase
python perturb_llm.py data/perturbed.jsonl --model ollama/aya-expanse:8b --only hindi --temperature 0
python apply_review.py annotations/llm_review.csv data/perturbed.jsonl data/perturbed_validated.jsonl
inspect eval guardrail_eval.py -T dataset=data/perturbed_validated.jsonl -T mode=llamaguard --model mockllm/model --log-dir logs/perturbed_v2
python export_results.py logs/perturbed_v2 results/perturbed_results_v2.csv
python stats.py results/perturbed_results_v2.csv --by perturbation

# 3. LLM judge vs human labels
inspect eval judge_eval.py -T dataset=data/perturbed.jsonl --model ollama/qwen2.5:7b --log-dir logs/judge
python judge_eval.py export logs/judge annotations/judge.csv judge_sheet.csv
python stats.py --kappa annotations/my_labels.csv annotations/judge.csv
```

To check the reported numbers without rerunning any model, run the `stats.py` commands on the files in `results/` and `annotations/`.

Notes:
- `mode=llamaguard` calls Llama Guard through a custom solver, because its chat template drops plain-string messages; `--model mockllm/model` is only a placeholder. See REPORT.md, Setup.
- LLM rewrites are generated at temperature > 0 for paraphrase, so regenerating gives different wording. The committed `data/` files are the record of what was tested.

## Data and licences

- **XSTest** prompts: Röttger et al. (2024), licensed [CC BY 4.0](https://github.com/paul-rottger/xstest). Files in `data/`, `results/` and `annotations/` contain XSTest prompts and rewrites of them.
- **Hindi rewrites** were generated with [Aya Expanse 8B](https://registry.ollama.ai/library/aya-expanse:8b) (CC BY-NC 4.0) and reviewed by hand: for non-commercial research use.
- Paraphrases were generated with Qwen2.5-7B-Instruct. Model weights are not included in this repository.
- Code: _choose a licence (e.g. MIT) and add a LICENSE file before publishing._

These prompts include requests for harmful content, used here only to test safety classifiers.

## References

- Röttger et al. (2024). XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. NAACL. https://arxiv.org/abs/2308.01263
- Meta. Llama Guard 3 1B model card. https://huggingface.co/meta-llama/Llama-Guard-3-1B
- Miller (2024). Adding Error Bars to Evals. https://arxiv.org/abs/2411.00640
- UK AI Security Institute. Inspect. https://inspect.aisi.org.uk