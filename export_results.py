"""Flatten Inspect logs into one CSV: one row per (model, sample).

  python export_results.py logs/ results.csv
  python export_results.py logs/ results.csv --task guardrail   # only guardrail runs

Reads only the top level of log_dir (not subfolders). If several successful runs share the
same model + mode + dataset, only the newest is kept and the older ones are reported.

Columns: model, task, dataset, mode, id, input, target, pred, raw, + every metadata field.
"""

import argparse

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--task", default=None, help="keep only logs from this task name")
    args = ap.parse_args()

    # Inspect log filenames start with the run's start time, so sorting by name,
    # newest first, means the first run we meet for each key is the newest one.
    logs = sorted(list_eval_logs(args.log_dir, recursive=False),
                  key=lambda i: i.name, reverse=True)

    rows, seen = [], set()
    for info in logs:
        log = read_eval_log(info)
        #only successful 
        if log.status != "success" or not log.samples:
            continue
        if args.task and log.eval.task.split("/")[-1] != args.task:
            continue
        targs = log.eval.task_args or {}
        # newest log
        key = (log.eval.model, targs.get("mode"), targs.get("dataset"))
        if key in seen:
            print(f"skipping older duplicate run: {info.name.split('/')[-1]}")
            continue
        seen.add(key)

        for s in log.samples:
            if not s.scores:
                continue
            sc = next(iter(s.scores.values()))
            target = s.target if isinstance(s.target, str) else s.target[0]
            rows.append({
                "model": (targs.get("guard_model", "meta-llama/Llama-Guard-3-1B")
                          if targs.get("mode") == "llamaguard" else log.eval.model),
                "task": log.eval.task,
                "dataset": targs.get("dataset", "xstest"),
                "mode": targs.get("mode", ""),
                "id": s.id,
                "input": s.input if isinstance(s.input, str) else s.input[-1].text,
                "target": target,
                "pred": sc.answer,
                "raw": sc.explanation,
                **(s.metadata or {}) #perturbation column gets into the CSV,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {len(df)} rows from {df['model'].nunique() if len(df) else 0} model(s) -> {args.out_csv}")


if __name__ == "__main__":
    main()