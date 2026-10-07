"""Pipeline report: recognizer WER by voice backend / voice / condition / engine, audit-flag
rates, throughput and a projection for the full script set. Prints Markdown and writes
``training/refine/datasets/real-v2/REPORT.md``.

    .venv-data/Scripts/python tools/refine-data/report.py [--per-file N] [--wall-s SECONDS]
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from augment import summary  # noqa: E402
from common import DATA, DRY, OUT, load_scripts, read_jsonl  # noqa: E402


def wer_table(rows: list[dict], key, title: str, min_n: int = 1, top: int | None = None) -> list[str]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    lines = [f"| {title} | n | WER (micro) | WER (mean) | sub / del / ins | suspect | recoverability flag |",
             "|---|---:|---:|---:|---|---:|---:|"]
    items = sorted(g.items(), key=lambda kv: -len(kv[1]))
    if top:
        items = items[:top]
    for k, rs in items:
        if len(rs) < min_n:
            continue
        a = [r["audit"]["auto"] for r in rs]
        ref = sum(x["wer_ops"]["ref_words"] for x in a)
        err = sum(x["wer_ops"]["sub"] + x["wer_ops"]["del"] + x["wer_ops"]["ins"] for x in a)
        ops = [sum(x["wer_ops"][o] for x in a) for o in ("sub", "del", "ins")]
        sus = sum(x["severity"] == "suspect" for x in a)
        rec = sum(bool(x["missing"]) for x in a)
        lines.append(f"| {k} | {len(rs)} | {err / max(1, ref):.3f} | {sum(x['wer'] for x in a) / len(a):.3f} | "
                     f"{ops[0]} / {ops[1]} / {ops[2]} | {sus / len(rs):.0%} | {rec / len(rs):.0%} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--per-file", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--wall-s", type=float, default=0.0, help="wall time of TTS+recognition for this selection")
    a = ap.parse_args()
    sel = {r["id"] for r in load_scripts(a.files, a.limit, a.per_file)}
    rows = [r for f in sorted(list(OUT.glob("W*.jsonl")) + list(OUT.glob("E*.jsonl"))) for r in read_jsonl(f) if r["id"] in sel]
    all_scripts = load_scripts()
    if not rows:
        print("no rows")
        return
    L: list[str] = [f"# real-v2 pipeline report ({len(rows)} rows of {len(sel)} selected scripts)", ""]
    L += wer_table(rows, lambda r: r["voice_detail"]["backend"], "TTS backend") + [""]
    L += wer_table(rows, lambda r: r["raw_engine"], "recognizer") + [""]
    L += wer_table(rows, lambda r: "clean" if r["audio_conditions"].get("clean") else "augmented", "audio") + [""]
    L += wer_table(rows, lambda r: (lambda c: "none" if "noise" not in c else
                                    ("snr 5-12" if c["noise"]["snr_db"] < 12 else "snr 12-20" if c["noise"]["snr_db"] < 20 else "snr 20-30"))(r["audio_conditions"]),
                   "background noise") + [""]
    L += wer_table(rows, lambda r: "room IR" if "rir" in r["audio_conditions"] else "no room IR", "reverb") + [""]
    L += wer_table(rows, lambda r: r["audio_conditions"].get("mic", {}).get("type", "clean"), "mic chain") + [""]
    L += wer_table(rows, lambda r: summary(r["audio_conditions"]), "condition (combined)", top=15) + [""]
    L += wer_table(rows, lambda r: r["voice"], "voice (most frequent 15)", top=15) + [""]
    a_ = [r["audit"]["auto"] for r in rows]
    n = len(rows)
    sev = Counter(x["severity"] for x in a_)
    flags = Counter(f for x in a_ for f in x["flags"])
    kinds = Counter(m["kind"] for x in a_ for m in x["missing"])
    L += ["## Audit pass 1", "",
          f"- severity: " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in sev.most_common()),
          f"- recoverability suspects (a key token of the target was said but is absent from raw): "
          f"{sum(bool(x['missing']) for x in a_)} rows ({sum(bool(x['missing']) for x in a_) / n:.0%}); "
          f"by kind: {dict(kinds)}",
          f"- target tokens not found in spoken either (script/format): {sum(bool(x['not_in_spoken']) for x in a_)} rows",
          f"- flags: {dict(flags.most_common())}", ""]
    audio_s = sum(r["audio_s"] for r in rows)
    words = sum(len(r["spoken"].split()) for r in rows)
    render = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        f = DRY / f"{r['id']}.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8-sig"))
            render[r["voice_detail"]["backend"]][0] += m.get("render_s", 0)
            render[r["voice_detail"]["backend"]][1] += m.get("seconds", 0)
    L += ["## Throughput", "",
          f"- audio rendered and recognized: {audio_s / 60:.1f} min for {n} clips ({words} spoken words, "
          f"{audio_s / max(1, words):.2f} s per word)"]
    L += [f"- TTS {b}: {v[1] / max(v[0], 1e-6):.1f}x realtime per worker" for b, v in sorted(render.items())]
    asr_ms = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        f = DATA / "asr-v2" / f"{r['id']}.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8-sig"))
            asr_ms[r["raw_engine"]][0] += m.get("asr_ms", 0) / 1000
            asr_ms[r["raw_engine"]][1] += r["audio_s"]
    L += [f"- recognition {e}: {v[1] / max(v[0], 1e-6):.1f}x realtime per worker" for e, v in sorted(asr_ms.items())]
    if a.wall_s:
        cpm = n / (a.wall_s / 60)
        tot_words = sum(len(r["spoken"].split()) for r in all_scripts)
        proj = a.wall_s * tot_words / max(1, words)
        L += [f"- wall time (all jobs in parallel): {a.wall_s / 60:.1f} min -> **{cpm:.1f} clips/min**, "
              f"{audio_s / a.wall_s:.1f}x realtime overall",
              f"- projection for all {len(all_scripts)} scripts ({tot_words} spoken words, ~{tot_words * audio_s / max(1, words) / 3600:.1f} h audio), "
              f"scaled by spoken words: **{proj / 3600:.1f} h** (model loading is a fixed ~2 min, so this is slightly pessimistic)"]
    text = "\n".join(L) + "\n"
    print(text)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
