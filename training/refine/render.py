"""Render refinement JSONL into training examples, byte-identical to the production local prompt.

Input rows (training/refine/README.md, datasets/GUIDE.md): ``raw``, ``clean``, ``mode``, ``style``,
``dictionary`` (other fields are carried through as metadata). Each row becomes

    prompt     = prompts.chatml(prompts.system_prompt(ctx), prompts.user_message(raw))
                 (ends with the pre-seeded empty think block "<think>\\n\\n</think>\\n\\n")
    completion = clean + "<|im_end|>"                     (loss is on this span only)

which is exactly what ``ochre_refine::local::build_prompt`` sends to llama-server ``/completion``.

Checks (``--check``, also run by train.py on every dataset before training):

1. **Rust parity.** Every rendered prompt is compared byte-for-byte with the Rust production
   renderer (``crates/ochre-refine/examples/render.rs`` -> ``ochre_refine::local::build_prompt``). The
   example binary is built on demand with ``CARGO_TARGET_DIR=target/finetune``.
2. **Chat-template parity.** The base model's own HF chat template with ``enable_thinking=False``
   renders the same prompt string, so the fine-tune starts in the instruct model's distribution.
3. **Leakage.** No training row shares an ``id`` or ``raw`` text with the held-out eval set.

    .venv-train-refine/Scripts/python render.py datasets/synth-v1/train.jsonl --check
    .venv-train-refine/Scripts/python render.py IN.jsonl [IN2.jsonl ...] --out data/rendered.jsonl --check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openwhisprflow.refine import prompts  # noqa: E402
from openwhisprflow.refine.base import RefineContext  # noqa: E402

IM_END = "<|im_end|>"
EVAL_FILE = HERE / "datasets/synth-v1/eval.jsonl"
RUST_TARGET = ROOT / "target/finetune"
RUST_BIN = RUST_TARGET / "debug/examples" / ("render.exe" if os.name == "nt" else "render")


# ---------------------------------------------------------------- loading


def is_dropped(row: dict) -> bool:
    """Audited data may mark rows as dropped; never train on them."""
    audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
    for d in (row.get("decision"), audit.get("decision"), (audit.get("final") or {}).get("decision")
              if isinstance(audit.get("final"), dict) else None):
        if d == "drop":
            return True
    return bool(row.get("drop"))


def load_rows(paths: list[Path], *, keep_dropped: bool = False) -> tuple[list[dict], int]:
    """Rows to train on, plus the number skipped (dropped by the audit, or empty raw/clean, e.g. the
    recognizer heard nothing)."""
    rows, dropped = [], 0
    for p in paths:
        for n, line in enumerate(Path(p).read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("raw", "clean")):
                print(f"render: skipping {p.name}:{n} ({row.get('id')}): empty raw or clean")
                dropped += 1
                continue
            if not keep_dropped and is_dropped(row):
                dropped += 1
                continue
            row.setdefault("id", f"{Path(p).stem}-{n}")
            rows.append(row)
    return rows, dropped


# ---------------------------------------------------------------- rendering


def ctx_of(row: dict) -> RefineContext:
    return RefineContext(mode=row.get("mode") or "clean", style=row.get("style") or "",
                         dictionary=list(row.get("dictionary") or []))


def render_prompt(row: dict) -> str:
    system = prompts.system_prompt(ctx_of(row))
    return prompts.chatml(system, prompts.user_message(row["raw"]))


def render_completion(row: dict) -> str:
    # Production output is tidied (stripped) before use, so the target never has edge whitespace.
    return row["clean"].strip() + IM_END


def render(row: dict) -> dict:
    system = prompts.system_prompt(ctx_of(row))
    user = prompts.user_message(row["raw"])
    prompt = render_prompt(row)
    completion = render_completion(row)
    return {"id": row["id"], "prompt": prompt, "completion": completion, "text": prompt + completion,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                         {"role": "assistant", "content": row["clean"].strip()}],
            "tags": row.get("tags", []), "source": row.get("source", "")}


# ---------------------------------------------------------------- checks


def rust_render(rows: list[dict]) -> list[str]:
    if not RUST_BIN.exists():
        print(f"building {RUST_BIN.name} (cargo, CARGO_TARGET_DIR={RUST_TARGET}) ...", flush=True)
        env = dict(os.environ, CARGO_TARGET_DIR=str(RUST_TARGET))
        subprocess.run(["cargo", "build", "-p", "ochre-refine", "--example", "render"], cwd=ROOT, env=env, check=True)
    payload = "".join(json.dumps({k: r.get(k) for k in ("raw", "mode", "style", "dictionary")}, ensure_ascii=False) + "\n"
                      for r in rows)
    for attempt in range(3):   # 0xC0000142 (DLL init failed) happens transiently when the box is starved
        proc = subprocess.run([str(RUST_BIN)], input=payload.encode("utf-8"), capture_output=True)
        if proc.returncode == 0:
            break
        print(f"render: Rust renderer exited 0x{proc.returncode & 0xFFFFFFFF:08X}, retrying", flush=True)
        time.sleep(2)
    else:
        raise SystemExit(f"Rust renderer failed: {proc.stderr.decode(errors='replace')[-500:]}")
    out = proc.stdout
    return [json.loads(line)["prompt"] for line in out.decode("utf-8").splitlines() if line.strip()]


def check_rust(rows: list[dict]) -> None:
    rs = rust_render(rows)
    if len(rs) != len(rows):
        raise SystemExit(f"rust renderer returned {len(rs)} prompts for {len(rows)} rows")
    for r, want in zip(rows, rs):
        got = render_prompt(r)
        if got != want:
            i = next(k for k in range(min(len(got), len(want))) if got[k] != want[k]) if got[:len(want)] != want[:len(got)] else min(len(got), len(want))
            raise SystemExit(f"prompt for {r['id']} differs from Rust build_prompt at byte {i}:\n"
                             f"py:   {got[max(0, i - 40):i + 40]!r}\nrust: {want[max(0, i - 40):i + 40]!r}")
    print(f"render: Rust parity OK ({len(rows)} prompts byte-identical to ochre_refine::local::build_prompt)")


def check_chat_template(rows: list[dict], tokenizer) -> None:
    seen = set()
    for r in rows:
        key = json.dumps([r.get("mode"), r.get("style"), r.get("dictionary")])
        if key in seen:
            continue
        seen.add(key)
        ex = render(r)
        hf = tokenizer.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        if hf != ex["prompt"]:
            raise SystemExit(f"HF chat template differs from production prompt for {r['id']}:\n{hf!r}\n{ex['prompt']!r}")
    print(f"render: base chat template (enable_thinking=False) matches production ({len(seen)} contexts)")


def check_leakage(rows: list[dict], eval_file: Path = EVAL_FILE) -> None:
    if not eval_file.exists():
        return
    ev = [json.loads(x) for x in eval_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    ids = {e["id"] for e in ev}
    raws = {" ".join(e["raw"].lower().split()) for e in ev}
    bad = [r["id"] for r in rows if r["id"] in ids or " ".join(r["raw"].lower().split()) in raws]
    if bad:
        raise SystemExit(f"{len(bad)} training rows overlap the held-out eval set {eval_file.name}: {bad[:10]}")
    print(f"render: no overlap with held-out {eval_file.relative_to(ROOT)} ({len(ev)} rows)")


# ---------------------------------------------------------------- tokenization (used by train.py)


def tokenize(ex: dict, tokenizer) -> dict:
    """Prompt and completion are tokenized separately (the prompt exactly as llama-server tokenizes
    it, special tokens parsed; the completion as the model would emit it), loss on completion only."""
    p = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
    c = tokenizer(ex["completion"][: -len(IM_END)], add_special_tokens=False)["input_ids"]
    c = c + [tokenizer.convert_tokens_to_ids(IM_END)]
    return {"input_ids": p + c, "labels": [-100] * len(p) + c, "n_prompt": len(p), "n_completion": len(c)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="write rendered JSONL (id, prompt, completion, text, messages)")
    ap.add_argument("--check", action="store_true", help="Rust parity + chat-template parity + leakage checks")
    ap.add_argument("--base", type=Path, default=HERE / "data/base/Qwen3.5-2B", help="tokenizer for the template check")
    ap.add_argument("--keep-dropped", action="store_true")
    a = ap.parse_args()
    rows, dropped = load_rows(a.inputs, keep_dropped=a.keep_dropped)
    print(f"render: {len(rows)} rows ({dropped} skipped: audit drop or empty) from {len(a.inputs)} file(s)")
    if a.check:
        check_rust(rows)
        check_leakage(rows)
        if a.base.exists():
            from transformers import AutoTokenizer
            check_chat_template(rows, AutoTokenizer.from_pretrained(a.base))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(json.dumps(render(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
