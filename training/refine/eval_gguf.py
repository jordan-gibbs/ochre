"""Eval hook: run a fine-tuned GGUF on the held-out set through tools/eval/eval_refine.py.

Uses the harness unchanged: production prompt path (checked byte-identical to prompts.rs),
llama-server b11398 with the docs/refinement.md §3.2 flags, clean_output + normalize, the guard,
and the deterministic scoring. **No LLM judge by default**: the final judgement is done blind by
the lead's workflow from the per-row output file. ``--quick-judge`` adds the old gpt-4.1-mini
judge for fast iteration only.

Run with the repo's main venv (it has the app package + httpx), from the repo root:

    .venv/Scripts/python training/refine/eval_gguf.py training/refine/output/<run>/gguf/<run>-Q4_K_M.gguf
    .venv/Scripts/python training/refine/eval_gguf.py <gguf> --quick-judge     # + gpt-4.1-mini judge (~$0.1)

Writes:
- ``tools/eval/out/refine-<system>.jsonl``: one row per eval example, same shape as
  ``refine-baseline-*.jsonl`` (system, id, bucket, tags, style, context, raw, clean, content, output,
  prod_output, typed, guard, wall_ms, timings, em_*, wer_*, fwer_clean, answered, added_words,
  dropped_words, digits_added, tag_checks; ``judge`` only with --quick-judge).
- ``<gguf dir>/eval-<system>.md``: the baseline-format report, with Quill 2B's saved outputs
  re-scored (not re-run) alongside. docs/refine-baseline.md is never touched.
- stdout: the deterministic table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/eval"))

import eval_refine as ev  # noqa: E402

DET_ROWS = [  # (label, summary key, format) — the deterministic rows of the baseline table
    ("Exact match, byte-exact", "em_exact", "pct"), ("Exact match, light norm", "em_light", "pct"),
    ("Exact match, light + normalize()", "em_norm", "pct"), ("WER vs clean", "wer_clean", "f3"),
    ("Formatted WER vs clean", "fwer_clean", "f3"), ("WER vs raw", "wer_raw", "f3"),
    ("Answered / acted (guard)", "answered", "pct"), ("Added content", "added", "pct"),
    ("Dropped content", "dropped", "pct"), ("Guard rejection", "guard", "pct"),
    ("Wrote digits where reference keeps words", "digits", "pct"),
    ("Typed: exact match (norm)", "typed_em", "pct"), ("Typed: WER vs clean", "typed_wer", "f3"),
    ("Latency median (ms)", "lat_med", "f0"), ("Latency p95 (ms)", "lat_p95", "f0"),
]


def fmt(v: float, kind: str) -> str:
    return {"pct": lambda x: f"{100 * x:.0f}%", "f3": lambda x: f"{x:.3f}", "f0": lambda x: f"{x:.0f}"}[kind](v)


def det_table(recs_by_sys: dict[str, list[dict]]) -> str:
    names = list(recs_by_sys)
    S = {n: ev.summary(recs_by_sys[n]) for n in names}
    L = ["| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for label, key, kind in DET_ROWS:
        L.append(f"| {label} | " + " | ".join(fmt(S[n][key], kind) for n in names) + " |")
    tags = ["self_correction", "already_clean", "list", "trailing_off", "voice_command", "discourse_filler", "question/request"]
    first = recs_by_sys[names[0]]
    for t in tags:
        n_t = sum(1 for r in first if t in r["tag_checks"])
        if n_t:
            L.append(f"| tag check: {t} (n={n_t}) | " + " | ".join(
                fmt(ev.mean([r["tag_checks"][t] for r in recs_by_sys[n] if t in r["tag_checks"]]), "pct") for n in names) + " |")
    for b in "ABCDE":
        L.append(f"| bucket {b}: EM(norm) / WER | " + " | ".join(
            (lambda s: f"{100 * s['em_norm']:.0f}% / {s['wer_clean']:.3f}")(ev.summary([r for r in recs_by_sys[n] if r["bucket"] == b]))
            if any(r["bucket"] == b for r in recs_by_sys[n]) else "–" for n in names) + " |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--name", help="system name (default: gguf stem)")
    ap.add_argument("--baselines", default="quill-2b-shared", help="saved baseline systems to re-score alongside")
    ap.add_argument("--eval", type=Path, default=ev.EVAL)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--quick-judge", action="store_true", help="also run the gpt-4.1-mini judge (iteration only)")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--budget", type=float, default=0.80)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", action="store_true",
                    help="chunked refinement for long input, as the app's refine.chunk_long = \"on\" (chunk.rs)")
    a = ap.parse_args()

    gguf = a.gguf.resolve()
    if not gguf.is_file():
        raise SystemExit(f"{gguf} not found")
    name = a.name or gguf.stem
    ev.SYSTEMS[name] = ev.System(name, "local", gguf.name, "shared", f"{gguf.stem} (fine-tune), shared prompt"
                                 + (", chunked long input" if a.chunk else ""), chunk=a.chunk)
    ev.MODEL_DIRS.insert(0, gguf.parent)
    # eval_refine re-saves every --skip-run system truncated to the eval rows, so with --limit the
    # saved baselines would be cut short: never load them on a partial run.
    baselines = [] if a.limit else [b for b in a.baselines.split(",") if b and ev.load_saved(b) is not None]
    report = a.report or gguf.parent / f"eval-{name}.md"
    argv = ["eval_refine.py", "--systems", ",".join(baselines + [name]), "--skip-run", "--rerun", name,
            "--eval", str(a.eval), "--report", str(report), "--best", name,
            # judge validation references every baseline system; it was done for the baseline report
            "--labels", str(HERE / "data/no-judge-validation.json")]
    if a.quick_judge:
        argv += ["--judge-model", a.judge_model, "--budget", str(a.budget)]
    else:
        argv.append("--no-judge")
    if a.limit:
        argv += ["--limit", str(a.limit)]
    sys.argv = argv
    ev.main()

    recs_by_sys = {}
    for n in baselines + [name]:
        recs = ev.load_saved(n)[: a.limit or None]
        recs_by_sys[n] = recs
    rows = recs_by_sys[name]
    if not a.quick_judge:
        rows = [{k: v for k, v in r.items() if k != "judge"} for r in rows]
    out = ev.OUT / f"refine-{name}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("\n" + det_table(recs_by_sys))
    print(f"\nper-row outputs: {out}\nreport: {report}")


if __name__ == "__main__":
    main()
