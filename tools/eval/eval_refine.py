"""Evaluation harness for the dictation-cleanup ("refinement") model. SPEC §9, training/refine/README.md.

Runs every example of a held-out set through the *production* prompt path and scores it:

- exact match with ``clean`` (byte-exact, after light normalization, and after ``normalize()`` on both
  sides, which production applies to local output anyway),
- word error rate vs ``clean`` and vs ``raw`` (how far the system moved the text),
- faithfulness flags (answered/acted, added content, dropped content, guard rejection),
- per-tag checks (self_correction, already_clean, list, trailing_off, voice_command, discourse_filler,
  question/request), and
- an LLM judge (pass/fail per GUIDE rule plus a one-line reason, strict JSON),
- latency (median / p95 wall time per example).

Prompts come from ``src/openwhisprflow/refine/prompts.py``; on start the harness checks they are
byte-identical to the Rust production prompts in ``crates/ochre-refine/src/prompts.rs`` and aborts if not.
The guard is ``refine/guard.py``, a line-for-line port of ``guard.rs`` (same regexes, thresholds, stems).

    .venv/Scripts/python tools/eval/eval_refine.py                       # run all systems, judge, report
    .venv/Scripts/python tools/eval/eval_refine.py --systems quill-0.8b-shared --limit 10
    .venv/Scripts/python tools/eval/eval_refine.py --skip-run            # re-score / re-report saved outputs

The OpenAI key (cloud system + judge) is read from ``--env-file`` into this process only.
Raw outputs: ``tools/eval/out/refine-baseline-<system>.jsonl`` (gitignored).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from openwhisprflow.refine import chunk, guard, prompts  # noqa: E402
from openwhisprflow.refine import local_server as ls  # noqa: E402
from openwhisprflow.refine.base import RefineContext  # noqa: E402
from openwhisprflow.refine.local import clean_output  # noqa: E402
from openwhisprflow.text.normalize import normalize  # noqa: E402

EVAL = ROOT / "training/refine/datasets/synth-v1/eval.jsonl"
GUIDE = ROOT / "training/refine/datasets/GUIDE.md"
OUT = Path(__file__).resolve().parent / "out"
LAPP = Path(os.environ.get("LOCALAPPDATA", "")) / "openwhisprflow"
MODEL_DIRS = [LAPP / "models/refine", LAPP / "data/models/refine"]
LLAMA_DIRS = [LAPP / "data/models/llama.cpp/b11398/win-cuda-13.4-x64", LAPP / "models/llama.cpp/b11398/win-cuda-13.4-x64"]
# OWF_LLAMA_BIN_DIR: a b11398 build elsewhere (the Linux cloud job, tools/cloud/, uses the ubuntu-cuda release).
if os.environ.get("OWF_LLAMA_BIN_DIR"):
    LLAMA_DIRS.insert(0, Path(os.environ["OWF_LLAMA_BIN_DIR"]))
LLAMA_SERVER = "llama-server.exe" if os.name == "nt" else "llama-server"

# USD per 1M tokens (input, cached input, output), Oct 2026 list prices.
PRICES = {"gpt-4.1-nano": (0.10, 0.025, 0.40), "gpt-4.1-mini": (0.40, 0.10, 1.60),
          "gpt-5.4-nano": (0.20, 0.02, 1.25),
          "gpt-5.4-mini": (0.75, 0.075, 4.50)}


@dataclass(frozen=True)
class System:
    name: str
    kind: str          # local | openai
    model: str         # gguf file name or OpenAI model id
    prompt: str        # shared | quill
    label: str
    chunk: bool = False  # chunked refinement for long input (crates/ochre-refine/src/chunk.rs via chunk.py)


SYSTEMS = {s.name: s for s in [
    System("quill-0.8b-shared", "local", "quill-0.8b-Q4_K_M.gguf", "shared", "Quill 0.8B Q4_K_M, shared prompt (prod default)"),
    System("quill-2b-shared", "local", "quill-2b-Q4_K_M.gguf", "shared", "Quill 2B Q4_K_M, shared prompt"),
    System("quill-0.8b-card", "local", "quill-0.8b-Q4_K_M.gguf", "quill", "Quill 0.8B Q4_K_M, Quill card prompt (reference)"),
    System("gpt-4.1-nano", "openai", "gpt-4.1-nano", "shared", "gpt-4.1-nano (OpenAI), shared prompt"),
]}


# ---------------------------------------------------------------- data, key


def load_eval(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["bucket"] = r["id"].split("-")[2]
    return rows[:limit] if limit else rows


def load_key(env_file: Path) -> str:
    """OPENAI_API_KEY from a dotenv file into this process only (never written anywhere)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for line in env_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit(f"OPENAI_API_KEY not found in {env_file}")


def ctx_of(ex: dict) -> RefineContext:
    return RefineContext(mode=ex.get("mode") or "clean", style=ex.get("style") or "",
                         dictionary=list(ex.get("dictionary") or []))


# ---------------------------------------------------------------- prompt parity with Rust


def _rust_literal(src: str, i: int) -> tuple[str, int]:
    """Parse the Rust string literal whose opening quote is at src[i]. Returns (value, end index)."""
    assert src[i] == '"'
    out, i = [], i + 1
    while src[i] != '"':
        c = src[i]
        if c == "\\":
            n = src[i + 1]
            if n == "\n" or n == "\r":           # line continuation: skip newline + leading whitespace
                i += 1
                while src[i] in " \t\r\n":
                    i += 1
                continue
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\", "'": "'", "0": "\0"}[n])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out), i + 1


def _rust_after(src: str, marker: str) -> str:
    i = src.index(marker)
    return _rust_literal(src, src.index('"', i + len(marker)))[0]


def rust_system_prompt(src: str, ctx: RefineContext) -> str:
    rules, clean_tail, polish_tail = (_rust_after(src, f"const {k}: &str =") for k in ("RULES", "CLEAN_TAIL", "POLISH_TAIL"))
    out = (f"You clean up and lightly polish dictated text.\n\n{rules}{polish_tail}" if ctx.mode == "polish"
           else f"You clean up dictated text.\n\n{rules}{clean_tail}")
    if ctx.style in ("casual", "formal", "literal"):
        out += "\n\n" + _rust_after(src, f'"{ctx.style}" => Some(')
    seen, words = set(), []
    for w in ctx.dictionary:
        w = w.strip()
        if w and w not in seen:
            seen.add(w)
            words.append(w)
    if words:
        out += "\n\nPreferred spellings (use these exact forms when the speaker says them): " + ", ".join(words)
    return out


def check_prompt_parity(rows: list[dict]) -> None:
    src = (ROOT / "crates/ochre-refine/src/prompts.rs").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert _rust_after(src, "pub const QUILL_SYSTEM: &str =") == prompts.QUILL_SYSTEM, "QUILL_SYSTEM differs"
    assert _rust_after(src, "pub const THINK_SEED: &str =") == prompts.THINK_SEED, "THINK_SEED differs"
    tmpl = _rust_after(src, "pub fn chatml(")
    rust_chatml = tmpl.replace("{THINK_SEED}", prompts.THINK_SEED)
    assert rust_chatml.replace("{system}", "S").replace("{user}", "U") == prompts.chatml("S", "U"), "chatml differs"
    assert _rust_after(src, "pub fn user_message(").replace("{}", "x") == prompts.user_message(" x "), "user_message differs"
    ctxs = {json.dumps([c.mode, c.style, c.dictionary]): c for c in map(ctx_of, rows)}
    ctxs["polish"] = RefineContext(mode="polish", style="casual", dictionary=["A", " B ", "A"])
    for c in ctxs.values():
        py, rs = prompts.system_prompt(c), rust_system_prompt(src, c)
        if py != rs:
            raise SystemExit(f"system prompt for {c} differs between prompts.py and prompts.rs:\n{py!r}\n{rs!r}")
    print(f"prompt parity with crates/ochre-refine/src/prompts.rs: OK ({len(ctxs)} contexts)")


# ---------------------------------------------------------------- systems


def find_file(dirs: list[Path], name: str) -> Path:
    for d in dirs:
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"{name} not found in {[str(d) for d in dirs]}")


def local_prompt(sysm: System, ex: dict) -> tuple[str, str]:
    """(full prompt, priming prefix) exactly as production / bench_refine render them."""
    if sysm.prompt == "quill":
        return prompts.chatml(prompts.QUILL_SYSTEM, ex["raw"].strip()), prompts.chatml_prefix(prompts.QUILL_SYSTEM)
    system = prompts.system_prompt(ctx_of(ex))
    return prompts.chatml(system, prompts.user_message(ex["raw"])), prompts.chatml_prefix(system)


def n_predict(text: str) -> int:   # local/mod.rs n_predict
    return max(32, int(len(text) / 3.2 * 2) + 16)


def gpu_snapshot() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:   # noqa: BLE001
        return f"n/a ({e})"


def run_local(sysm: System, rows: list[dict]) -> tuple[list[dict], dict]:
    exe = find_file(LLAMA_DIRS, LLAMA_SERVER)
    model = find_file(MODEL_DIRS, sysm.model)
    OUT.mkdir(parents=True, exist_ok=True)
    server = ls.LlamaServer(exe, model, accel="cuda", options=ls.ServerOptions(),
                            log_path=OUT / f"llama-server-{sysm.name}.log")
    meta: dict[str, Any] = {"args": " ".join(server.args()[3:]).replace(str(model), model.name),
                            "gpu_before": gpu_snapshot()}
    t0 = time.perf_counter()
    server.start()
    meta["cold_start_ms"] = round((time.perf_counter() - t0) * 1000)
    out: list[dict] = []
    gpu_samples: list[str] = []
    try:
        # Warm-up exactly like LocalRefiner::load: one tiny refinement, then prime.
        warm = {"raw": "okay so this is a warm up", "mode": "clean", "style": "", "dictionary": []}
        p, _ = local_prompt(sysm, warm)
        server.completion(p, n_predict=n_predict(warm["raw"]), timeout=60, stop=list(prompts.STOP))
        for k, ex in enumerate(rows):
            pieces = chunk.split(ex["raw"]) if sysm.chunk else []
            if len(pieces) >= 2:
                out.append(run_chunked(server, sysm, ex, pieces))
                print(f"  [{sysm.name}] {k + 1}/{len(rows)} {out[-1]['wall_ms']:.0f} ms ({len(pieces)} chunks)",
                      end="\r", flush=True)
                continue
            prompt, prefix = local_prompt(sysm, ex)
            # Production primes the slot with the *current* context's prefix after every request, so
            # consecutive dictations in one app hit the cache. Prime for this example (off the clock),
            # which measures that steady state rather than an app switch.
            server.completion(prefix, n_predict=0, timeout=60)
            t = time.perf_counter()
            data = server.completion(prompt, n_predict=n_predict(ex["raw"]), timeout=120, stop=list(prompts.STOP))
            wall = (time.perf_counter() - t) * 1000
            content = data.get("content", "")
            model_out = clean_output(content)
            tm = data.get("timings", {})
            out.append(base_record(sysm, ex, content, model_out, normalize(model_out), wall, {
                k2: tm.get(k2) for k2 in ("prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms")}))
            if k % 25 == 0:
                gpu_samples.append(gpu_snapshot())
            print(f"  [{sysm.name}] {k + 1}/{len(rows)} {wall:.0f} ms", end="\r", flush=True)
    finally:
        server.stop()
    print()
    meta["gpu_during"] = gpu_samples
    return out, meta


def run_chunked(server: Any, sysm: System, ex: dict, pieces: list) -> dict:
    """Chunked refinement exactly as app.rs does it: each piece refined with the dictation's context,
    clean_output + normalize, guarded on its own (a rejected piece falls back to its raw text), and
    joined with the pieces' separators. The first prime is off the clock like the unchunked path; the
    primes between pieces (the app's background primer) and every request are on it."""
    _, prefix = local_prompt(sysm, ex)
    server.completion(prefix, n_predict=0, timeout=60)
    mode = ex.get("mode") or "clean"
    parts, tm_sum = [], {k: 0 for k in ("prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms")}
    t = time.perf_counter()
    for i, pc in enumerate(pieces):
        if i:
            server.completion(prefix, n_predict=0, timeout=60)
        prompt, _ = local_prompt(sysm, {**ex, "raw": pc.text})
        data = server.completion(prompt, n_predict=n_predict(pc.text), timeout=120, stop=list(prompts.STOP))
        content = data.get("content", "")
        model_out = clean_output(content)
        prod = normalize(model_out)
        reject = guard.check(pc.text, prod, mode=mode)
        for k2 in tm_sum:
            tm_sum[k2] += (data.get("timings") or {}).get(k2) or 0
        parts.append({"sep": pc.sep, "raw": pc.text, "content": content, "output": model_out, "prod_output": prod,
                      "guard": reject or "ok", "typed": pc.text if reject else prod})
    wall = (time.perf_counter() - t) * 1000
    join = lambda f: chunk.join((p["sep"], p[f]) for p in parts)  # noqa: E731
    rejected = [p["guard"] for p in parts if p["guard"] != "ok"]
    return {"system": sysm.name, "id": ex["id"], "bucket": ex["bucket"], "tags": ex["tags"],
            "style": ex.get("style", ""), "context": ex.get("context", ""), "raw": ex["raw"], "clean": ex["clean"],
            "content": join("content"), "output": join("output"), "prod_output": join("prod_output"),
            "guard": "chunk:" + ",".join(rejected) if rejected else "ok", "typed": join("typed"),
            "wall_ms": round(wall, 1), "timings": {k2: round(v, 2) for k2, v in tm_sum.items()},
            "chunks": parts}


class OpenAI:
    def __init__(self, key: str) -> None:
        self.client = httpx.Client(base_url="https://api.openai.com/v1", timeout=60.0,
                                   headers={"Authorization": f"Bearer {key}"})
        self.lock = threading.Lock()
        self.cost = 0.0
        self.usage: dict[str, list[int]] = {}

    def chat(self, model: str, messages: list[dict], max_tokens: int, json_mode: bool = False) -> tuple[str, float]:
        body: dict[str, Any] = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
                                "stream": False}
        if model.startswith("gpt-5"):
            body["reasoning_effort"] = "none"
        else:
            body["temperature"] = 0
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(5):
            t = time.perf_counter()
            r = self.client.post("/chat/completions", json=body)
            wall = (time.perf_counter() - t) * 1000
            if r.status_code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
            data = r.json()
            u = data.get("usage", {})
            pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            price = PRICES.get(model, (1.0, 1.0, 4.0))
            with self.lock:
                self.cost += ((pin - cached) * price[0] + cached * price[1] + pout * price[2]) / 1e6
                acc = self.usage.setdefault(model, [0, 0, 0, 0])
                for i, v in enumerate((1, pin, cached, pout)):
                    acc[i] += v
            return data["choices"][0]["message"].get("content") or "", wall
        raise RuntimeError("unreachable")


def run_openai(sysm: System, rows: list[dict], api: OpenAI) -> tuple[list[dict], dict]:
    out = []
    # Warm the TLS connection first (production keeps one pooled client per engine).
    api.chat(sysm.model, [{"role": "user", "content": "hi"}], 16)
    for k, ex in enumerate(rows):
        msgs = [{"role": "system", "content": prompts.system_prompt(ctx_of(ex))},
                {"role": "user", "content": prompts.user_message(ex["raw"])}]
        cap = max(64, min(4096, len(ex["raw"]) // 4 * 2 + 64))     # http.rs max_tokens_for
        content, wall = api.chat(sysm.model, msgs, cap)
        model_out = guard.tidy(content.strip())
        # Production does not run normalize() on cloud output; scoring applies it to both sides anyway.
        out.append(base_record(sysm, ex, content, model_out, model_out, wall, {}))
        print(f"  [{sysm.name}] {k + 1}/{len(rows)} {wall:.0f} ms", end="\r", flush=True)
    print()
    return out, {"note": "sequential requests, warm pooled HTTPS connection"}


def base_record(sysm: System, ex: dict, content: str, model_out: str, prod_out: str, wall: float,
                timings: dict) -> dict:
    reject = guard.check(ex["raw"], prod_out, mode=ex.get("mode") or "clean")
    return {"system": sysm.name, "id": ex["id"], "bucket": ex["bucket"], "tags": ex["tags"],
            "style": ex.get("style", ""), "context": ex.get("context", ""), "raw": ex["raw"], "clean": ex["clean"],
            "content": content,           # the model's raw completion
            "output": model_out,          # after clean_output/tidy (what the model wrote)
            "prod_output": prod_out,      # local: + normalize(); this is what the guard sees
            "guard": reject or "ok",
            "typed": ex["raw"] if reject else prod_out,   # what production would insert
            "wall_ms": round(wall, 1), "timings": timings}


# ---------------------------------------------------------------- deterministic scoring

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def light(s: str) -> str:
    """Whitespace + curly-vs-straight quotes only (newlines kept: list layout matters)."""
    s = s.translate(_QUOTES)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.strip().split("\n")]
    return "\n".join(lines)


def words(s: str) -> list[str]:
    """WER tokens: lower-case, punctuation removed (apostrophes inside words kept)."""
    s = s.translate(_QUOTES).lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return [w.strip("'") for w in s.split() if w.strip("'")]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r = words(ref)
    return edit_distance(r, words(hyp)) / max(1, len(r))


def fwer(ref: str, hyp: str) -> float:
    """Formatted WER: whitespace tokens with case and punctuation attached."""
    r = light(ref).split()
    return edit_distance(r, light(hyp).split()) / max(1, len(r))


_ORDINALS = set("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
                "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth st nd rd th".split())


def cw_set(s: str) -> set[str]:
    """Guard content words, minus ordinals/number suffixes ("thirteenth" vs "13th" is the normalizer's business)."""
    return {w for w in guard.content_words(s.translate(_QUOTES)) if w not in _ORDINALS and len(w) > 1}


def missing(src: set[str], target: set[str]) -> list[str]:
    return sorted(w for w in src if w not in target and not guard._near(w, target))


_ALLWORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_NUMWORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred "
                "thousand million billion first second third fourth fifth eighth ninth thirteenth".split())
_CUES = set("no wait actually mean sorry scratch rather make let rephrase that or i yeah nope oops".split())
_FUNC = guard._STOPWORDS - _NUMWORDS


def retracted_words(raw: str, clean: str) -> list[str]:
    """Words the speaker took back: in raw, not in clean, minus correction cues and function words."""
    r = set(_ALLWORD.findall(raw.lower().translate(_QUOTES)))
    c = set(_ALLWORD.findall(clean.lower().translate(_QUOTES)))
    c_stems = {guard._stem(w) for w in c}
    return sorted(w for w in r - c if w not in _CUES and w not in _FUNC and guard._stem(w) not in c_stems)


_CUE_PHRASES = ["no wait", "no", "wait", "sorry", "actually", "i mean", "scratch that", "or rather", "make that",
                "let me rephrase", "rather"]
_LIST_LINE = re.compile(r"^\s*(?:(-|\*|•)|(\d+)[.)])\s+", re.M)


def list_shape(s: str) -> tuple[int, str]:
    items = _LIST_LINE.findall(s)
    if not items:
        return 0, ""
    return len(items), "bullet" if items[0][0] else "numbered"


_FILLERS = ["you know", "i mean", "sort of", "kind of", "okay so", "like", "basically", "literally", "actually", "so"]
_VOICE = ["new line", "new paragraph", "question mark", "comma", "period"]


def count_phrases(s: str, phrases: list[str]) -> dict[str, int]:
    low = s.lower().translate(_QUOTES)
    return {p: len(re.findall(rf"(?<![\w']){re.escape(p)}(?![\w'])", low)) for p in phrases}


def tail_words(s: str, n: int = 3) -> list[str]:
    return words(s)[-n:]


def tag_checks(rec: dict) -> dict[str, bool]:
    """Deterministic per-tag success. Only for tags whose success is mechanically checkable."""
    raw, clean, out = rec["raw"], rec["clean"], rec["output"]
    tags, res = set(rec["tags"]), {}
    if "self_correction" in tags:
        out_w = set(_ALLWORD.findall(out.lower().translate(_QUOTES)))
        kept_retracted = [w for w in retracted_words(raw, clean) if w in out_w]
        cues_kept = [c for c, n in count_phrases(out, _CUE_PHRASES).items() if n > count_phrases(clean, [c])[c]]
        res["self_correction"] = not kept_retracted and not cues_kept and not rec["dropped_words"]
    if "already_clean" in tags:
        res["already_clean"] = out == raw
    if "list" in tags:
        res["list"] = list_shape(out) == list_shape(clean)
    if "trailing_off" in tags:
        if clean.rstrip().endswith("..."):
            o = out.rstrip()
            res["trailing_off"] = (o.endswith("...") or o.endswith("\u2026")) and tail_words(o) == tail_words(clean)
        else:
            res["trailing_off"] = not out.rstrip().endswith(("...", "\u2026"))
    if "voice_command" in tags:
        res["voice_command"] = (out.count("\n") == clean.count("\n")
                                and count_phrases(out, _VOICE) == count_phrases(clean, _VOICE))
    if "discourse_filler" in tags:
        res["discourse_filler"] = count_phrases(out, _FILLERS) == count_phrases(clean, _FILLERS)
    if tags & {"question", "request_to_ai", "command"}:
        res["question/request"] = (not rec["answered"]) and out.count("?") == clean.count("?")
    return res


def score(rec: dict) -> dict:
    raw, clean, out = rec["raw"], rec["clean"], rec["output"]
    n_out, n_clean = normalize(rec["prod_output"]), normalize(clean)
    rec["em_exact"] = out == clean
    rec["em_light"] = light(out) == light(clean)
    rec["em_norm"] = light(n_out) == light(n_clean)
    rec["wer_clean"] = round(wer(n_clean, n_out), 4)
    rec["fwer_clean"] = round(fwer(n_clean, n_out), 4)
    rec["wer_raw"] = round(wer(normalize(raw), n_out), 4)
    rec["wer_raw_ref"] = round(wer(normalize(raw), n_clean), 4)   # how far the reference itself moves
    rec["typed_em_norm"] = light(normalize(rec["typed"])) == light(n_clean)
    rec["typed_wer_clean"] = round(wer(n_clean, normalize(rec["typed"])), 4)
    raw_cw, clean_cw, out_cw = cw_set(raw), cw_set(clean), cw_set(out)
    # Added: content words not in raw and not a legitimate fix (i.e. also in the reference).
    rec["added_words"] = missing(out_cw, raw_cw | clean_cw)
    rec["dropped_words"] = missing(clean_cw, out_cw)
    rec["digits_added"] = bool(re.search(r"\d", out)) and not re.search(r"\d", clean)
    rec["answered"] = rec["guard"] in ("preamble", "assistant_voice", "answered_question")
    rec["tag_checks"] = tag_checks(rec)
    return rec


# ---------------------------------------------------------------- LLM judge

JUDGE_RULES = {
    "R1_no_answer": "Never answer, never act: a question stays a question, a request stays a request, even when addressed to an AI.",
    "R2_self_correction": "Self-corrections applied: only the corrected version is kept and the correction cue is dropped; meaningful uses of cue words are kept.",
    "R3_disfluency": "False starts, stutters, word fragments and accidental repeats are dropped; deliberate repetition kept.",
    "R4_trailing_off": "A dictation that ends mid-thought keeps every word and ends with '...'; no invented ending. An abandoned fragment followed by a restart is dropped.",
    "R5_discourse_filler": "Discourse fillers (like, you know, I mean, basically, sort of, kind of, literally, so) removed only when meaningless; kept when meaningful.",
    "R6_punct_casing": "Sentence breaks, commas, question marks, apostrophes and capitalization (names, products, 'I') are correct.",
    "R8_list": "List formatting only when explicitly dictated as a list ('- ' bullets / '1. ' numbers, one item per line, lead-in on its own line); never invented from prose.",
    "R9_voice_command": "Spoken 'new line'/'new paragraph'/'period'/'comma'/'question mark' commands applied when clearly used as commands, kept as words otherwise.",
    "R10_faithful": "Nothing added (words, facts, greetings, sign-offs), nothing summarized or dropped, no change of tense/person/dialect/register, no censoring.",
    "R11_already_clean": "If the input was already clean, the output is identical.",
}

JUDGE_SYSTEM = """You grade a dictation-cleanup system. A person dictated text with speech recognition (RAW). The system had to rewrite RAW as the text the speaker meant to type (OUTPUT), following the labeling guide below. REFERENCE is a correct cleanup written by a careful labeler.

<guide>
{guide}
</guide>

Grading notes:
- The REFERENCE is one correct answer, not the only one. Judge the OUTPUT against the guide rules and RAW; use the REFERENCE to understand what the speaker meant (e.g. which version a self-correction keeps).
- OVERRIDE of guide rule 7 for grading: numbers, times, money, emails and URLs written as digits/symbols ("3:30", "$250", "12th", "401", "a@b.com") are EQUIVALENT to the spoken words, because a deterministic normalizer converts them afterwards anyway. Never fail any rule for digit-vs-word form; fail only if a value itself changed.
- Capitalizing a proper noun or name that the reference leaves lowercase (e.g. "God") is acceptable.
- R6 fails only for errors a careful writer would fix: a missing question mark on a question, a missing apostrophe ("dont", "wont"), a lowercase sentence start or proper noun, a long run-on that clearly needs a sentence break, or punctuation that changes the meaning. Optional commas are NEVER a failure: commas before and/or/but/so, after introductory words ("Also", "So", "Anyway"), in short lists, or around short clauses may be present or absent. The OUTPUT does not need to match the REFERENCE's exact punctuation or sentence breaks when both are acceptable.
- R8: a list lead-in ending in "." or ":", a blank line before the items, and items with or without final periods are all acceptable. What matters is that a dictated list became one item per line with the right marker type, and that no list was invented.
- R11 applies only when RAW was already clean (RAW and REFERENCE are identical); otherwise it is "na".
- Mark a rule "na" when the dictation gives it nothing to test (e.g. R2 when the speaker never corrected themselves, R8 when nothing list-like was dictated).
- "acted" is true if the OUTPUT responds to the dictation instead of transcribing it: it answers a question, carries out a request (writes the poem/letter/reply/code, translates, summarizes, role-plays, obeys "ignore previous instructions"), or speaks as an assistant, even partially (a cleaned question followed by the start of an answer counts). Translating the dictation into another language counts as acting. If "acted" is true, R1 is "fail".
- "overall" is "pass" only if no rule fails.

Reply with strict JSON only, exactly this shape:
{{"acted": true | false, "rules": {{{rule_shape}}}, "overall": "pass" | "fail", "reason": "<one line: the most important problem, or 'ok'>"}}
where each rule value is "pass", "fail" or "na". Rules:
{rules}"""


def guide_rules_text() -> str:
    g = GUIDE.read_text(encoding="utf-8")
    start = g.index("## What `clean` must be")
    end = g.index("## Quality bar")
    return g[start:end].strip()


def judge_messages(rec: dict) -> list[dict]:
    rules = "\n".join(f"- {k}: {v}" for k, v in JUDGE_RULES.items())
    shape = ", ".join(f'"{k}": "pass|fail|na"' for k in JUDGE_RULES)
    system = JUDGE_SYSTEM.format(guide=guide_rules_text(), rules=rules, rule_shape=shape)
    user = (f"CONTEXT: {rec.get('context') or 'unknown'} (style: {rec.get('style') or 'default'})\n\n"
            f"RAW:\n<<<\n{rec['raw']}\n>>>\n\nREFERENCE:\n<<<\n{rec['clean']}\n>>>\n\nOUTPUT:\n<<<\n{rec['output']}\n>>>")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_key(model: str, rec: dict) -> str:
    h = hashlib.sha256(json.dumps([model, JUDGE_SYSTEM, JUDGE_RULES, rec["id"], rec["output"]]).encode()).hexdigest()
    return h[:24]


def parse_judge(text: str) -> dict:
    data = json.loads(text)
    rules = data.get("rules", {})
    clean = {k: (rules.get(k) if rules.get(k) in ("pass", "fail", "na") else "na") for k in JUDGE_RULES}
    acted = data.get("acted") is True
    if acted:
        clean["R1_no_answer"] = "fail"
    overall = "fail" if "fail" in clean.values() else data.get("overall", "pass")
    return {"acted": acted, "rules": clean, "overall": overall if overall in ("pass", "fail") else "fail",
            "reason": str(data.get("reason", ""))[:300]}


JUDGE_HISTORY = (
    "How the rubric got here. With the first rubric the judge agreed on 7/10 tuning cases; all three misses were "
    "false fails on optional commas or a misapplied R11. After the rubric spelled out which punctuation is optional, "
    "it agreed on 9/10. On 10 fresh holdout cases it then agreed on 8/10; both misses were false fails, one for digits "
    "vs spoken numbers and one for capitalising \"God\". The rubric now overrides both explicitly (guide rule 7 is "
    "moot because the normalizer runs afterwards). `gpt-5.4-mini` did no better on the same 20 cases (17/20) and cost "
    "about twice as much. Under R1 the judge also missed obvious acting (a written cover letter, a translated "
    "dictation), so it now answers a separate yes/no \"acted\" question, and a yes forces R1 to fail. Its remaining "
    "errors go both ways: it is too strict about commas in long run-on inputs, and it once accepted an unformatted "
    "dictated list. With the final rubric it agrees on all 20 cases, but those 20 cases shaped the rubric, so read the earlier 8/10 on "
    "fresh cases as the more honest estimate: the judge errs toward failing. The deterministic tag checks do not "
    "depend on the judge.")


AUTO_PASS = {"acted": False, "rules": {k: "pass" for k in JUDGE_RULES}, "overall": "pass", "reason": "exact match (not sent to judge)"}


class Judge:
    def __init__(self, api: OpenAI, model: str, budget: float) -> None:
        self.api, self.model, self.budget = api, model, budget
        self.cache_path = OUT / "judge-cache.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["key"]] = d["verdict"]
        self.lock = threading.Lock()

    def one(self, rec: dict, force: bool = False) -> dict:
        if rec["em_light"] and not force:
            return dict(AUTO_PASS)
        key = judge_key(self.model, rec)
        if key in self.cache:
            return self.cache[key]
        if self.api.cost > self.budget:
            return {"rules": {}, "overall": "skipped", "reason": "judge budget exhausted"}
        for attempt in range(3):
            text, _ = self.api.chat(self.model, judge_messages(rec), 400, json_mode=True)
            try:
                verdict = parse_judge(text)
                break
            except (json.JSONDecodeError, AttributeError) as e:
                if attempt == 2:
                    verdict = {"rules": {}, "overall": "error", "reason": f"unparseable judge output: {e}"}
        with self.lock:
            self.cache[key] = verdict
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "id": rec["id"], "verdict": verdict}) + "\n")
        return verdict

    def all(self, recs: list[dict], workers: int = 8) -> None:
        with cf.ThreadPoolExecutor(workers) as pool:
            for rec, v in zip(recs, pool.map(self.one, recs)):
                rec["judge"] = v


# ---------------------------------------------------------------- judge validation

def validate_judge(judge: Judge, recs_by_sys: dict[str, list[dict]], labels_path: Path) -> dict:
    """Agreement between the judge and hand labels (tools/eval/judge_validation.json).
    Each label: {"system", "id", "overall": "pass"|"fail", "note"}. Judged with force=True so exact
    matches are also sent to the judge (the auto-pass shortcut is not what is being validated)."""
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    rows = []
    for lab in labels:
        rec = next(r for r in recs_by_sys[lab["system"]] if r["id"] == lab["id"])
        v = judge.one(rec, force=True)
        rows.append({**lab, "judge": v["overall"], "judge_reason": v["reason"], "agree": v["overall"] == lab["overall"],
                     "output": rec["output"], "raw": rec["raw"], "clean": rec["clean"]})
    by_set = {}
    for r in rows:
        st = by_set.setdefault(r.get("set", "all"), [0, 0])
        st[0] += r["agree"]
        st[1] += 1
    return {"n": len(rows), "agree": sum(r["agree"] for r in rows), "rows": rows, "by_set": by_set}


# ---------------------------------------------------------------- report


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def p95(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] if s else 0.0


def judged(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("judge", {}).get("overall") in ("pass", "fail")]


def answered(r: dict) -> bool:
    j = r.get("judge", {})
    return bool(r["answered"] or j.get("acted") or j.get("rules", {}).get("R1_no_answer") == "fail")


def summary(recs: list[dict]) -> dict:
    j = judged(recs)
    return {
        "n": len(recs),
        "em_exact": mean([r["em_exact"] for r in recs]), "em_light": mean([r["em_light"] for r in recs]),
        "em_norm": mean([r["em_norm"] for r in recs]),
        "wer_clean": mean([r["wer_clean"] for r in recs]), "fwer_clean": mean([r["fwer_clean"] for r in recs]),
        "wer_raw": mean([r["wer_raw"] for r in recs]),
        "answered": mean([answered(r) for r in recs]),
        "added": mean([bool(r["added_words"]) for r in recs]), "dropped": mean([bool(r["dropped_words"]) for r in recs]),
        "guard": mean([r["guard"] != "ok" for r in recs]),
        "judge_pass": mean([r["judge"]["overall"] == "pass" for r in j]) if j else float("nan"),
        "typed_em": mean([r["typed_em_norm"] for r in recs]), "typed_wer": mean([r["typed_wer_clean"] for r in recs]),
        "lat_med": statistics.median([r["wall_ms"] for r in recs]), "lat_p95": p95([r["wall_ms"] for r in recs]),
        "digits": mean([r["digits_added"] for r in recs]),
    }


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", "⏎")


def write_report(path: Path, recs_by_sys: dict[str, list[dict]], meta: dict, validation: dict | None,
                 best: str, api_cost: float | None) -> None:
    systems = [s for s in SYSTEMS if s in recs_by_sys]
    S = {s: summary(recs_by_sys[s]) for s in systems}
    L: list[str] = []
    w = L.append
    w("# Refinement baseline (synth-v1 eval)\n")
    w(f"Generated by `tools/eval/eval_refine.py` on {time.strftime('%Y-%m-%d')}. Eval set: "
      f"`training/refine/datasets/synth-v1/eval.jsonl` ({S[systems[0]]['n']} examples, 20 per bucket A–E). "
      "Raw outputs: `tools/eval/out/refine-baseline-<system>.jsonl` (gitignored).\n")
    w("Every system runs the production prompt path: `prompts.system_prompt(ctx)` (mode, style, dictionary from "
      "each example; checked byte-identical to `crates/ochre-refine/src/prompts.rs` at start-up) and the "
      "`<dictation>` user message; local models get raw ChatML with the empty think block on llama-server "
      "`/completion`, greedy, with the `docs/refinement.md` §3.2 flags; output goes through `clean_output` + "
      "`normalize` (local) or `tidy` (cloud), then the guard (`refine/guard.py`, a line-for-line port of "
      "`guard.rs`). The Quill card-prompt row uses `You clean up dictated text.` plus the bare transcript.\n")
    w("## Overall\n")
    w("| metric | " + " | ".join(SYSTEMS[s].label for s in systems) + " |")
    w("|---|" + "---:|" * len(systems))
    rows = [
        ("Exact match, byte-exact", "em_exact", pct), ("Exact match, light norm (whitespace, quotes)", "em_light", pct),
        ("Exact match, light + `normalize()` both sides", "em_norm", pct),
        ("**LLM judge pass**", "judge_pass", pct),
        ("WER vs clean (words)", "wer_clean", lambda x: f"{x:.3f}"),
        ("Formatted WER vs clean (case + punct)", "fwer_clean", lambda x: f"{x:.3f}"),
        ("WER vs raw (how far it moved; reference itself: {ref})", "wer_raw", lambda x: f"{x:.3f}"),
        ("Answered / acted", "answered", pct), ("Added content (≥1 word)", "added", pct),
        ("Dropped content (≥1 word)", "dropped", pct), ("Guard rejection", "guard", pct),
        ("Wrote digits where reference keeps words", "digits", pct),
        ("What gets typed (guard applied): exact match (norm)", "typed_em", pct),
        ("What gets typed: WER vs clean", "typed_wer", lambda x: f"{x:.3f}"),
        ("Latency median (ms)", "lat_med", lambda x: f"{x:.0f}"), ("Latency p95 (ms)", "lat_p95", lambda x: f"{x:.0f}"),
    ]
    ref_move = mean([r["wer_raw_ref"] for r in recs_by_sys[systems[0]]])
    for name, key, f in rows:
        w(f"| {name.format(ref=f'{ref_move:.3f}')} | " + " | ".join(f(S[s][key]) for s in systems) + " |")
    w("")
    w("Definitions. *Answered/acted*: the guard flagged `preamble`, `assistant_voice` or `answered_question`, or "
      "the judge said the output acted on the dictation (answered, carried out or translated it). *Added content*: a guard content word (stemmed) in the output that is in neither "
      "`raw` nor `clean` (so homophone fixes and list lead-ins the reference makes are not counted). *Dropped "
      "content*: a content word of `clean` missing from the output. Spoken numbers are guard stopwords, so a wrong "
      "number value is caught by the judge, not by these two flags. *WER* lower-cases and strips punctuation; both "
      "sides go through `normalize()` first. *Judge*: exact matches (light norm) are auto-passed; everything else "
      "is graded by the judge model per GUIDE rule.\n")
    # Latency + environment
    w("### Latency and environment\n")
    for s in systems:
        m = meta.get(s, {})
        recs = recs_by_sys[s]
        line = f"- **{SYSTEMS[s].label}**: median {S[s]['lat_med']:.0f} ms, p95 {S[s]['lat_p95']:.0f} ms, max {max(r['wall_ms'] for r in recs):.0f} ms"
        tm = [r["timings"] for r in recs if r.get("timings")]
        if tm:
            cache = mean([t.get("cache_n") or 0 for t in tm])
            gen = sum(t.get("predicted_n") or 0 for t in tm) / max(1e-9, sum(t.get("predicted_ms") or 0 for t in tm)) * 1000
            line += f"; mean cached prompt tokens {cache:.0f}, generation {gen:.0f} tok/s"
            if m.get("cold_start_ms"):
                line += f"; cold start {m['cold_start_ms']} ms"
        if m.get("gpu_before"):
            line += f". GPU before run (util, mem used, total): `{m['gpu_before']}`"
        if m.get("gpu_during"):
            line += f"; during: `{' / '.join(m['gpu_during'])}`"
        w(line)
    if meta.get("_env"):
        w(f"\n{meta['_env']}")
    w("")
    # Per bucket
    buckets = {"A": "self-corrections", "B": "false starts / stutters / trailing off", "C": "punctuation / casing / proper nouns",
               "D": "never answer + already clean", "E": "lists / voice commands / discourse fillers"}
    w("## Per bucket\n")
    w("Cells: judge pass / exact match (light + normalize) / WER vs clean.\n")
    w("| bucket | " + " | ".join(s for s in systems) + " |")
    w("|---|" + "---|" * len(systems))
    for b, desc in buckets.items():
        cells = []
        for s in systems:
            rb = [r for r in recs_by_sys[s] if r["bucket"] == b]
            if not rb:
                cells.append("–")
                continue
            sb = summary(rb)
            cells.append(f"{pct(sb['judge_pass'])} / {pct(sb['em_norm'])} / {sb['wer_clean']:.3f}")
        w(f"| {b}: {desc} | " + " | ".join(cells) + " |")
    w("")
    # Per tag checks
    w("## Per tag\n")
    w("Deterministic tag checks (n = examples carrying the tag):\n")
    w("- `self_correction`: no retracted word (in raw, not in clean, not a cue/function word) survives, no correction cue (no, wait, sorry, actually, I mean, scratch that, or rather, make that) appears more often than in clean, and no clean content word is dropped.")
    w("- `already_clean`: output is byte-identical to raw.")
    w("- `list`: same number of list lines and the same marker type (bullet vs numbered) as clean; no list when clean has none.")
    w("- `trailing_off`: ends with `...` and the last three words match clean (nothing invented); no `...` when the reference restarts.")
    w("- `voice_command`: same newline count as clean and the same count of spoken command words (`new line`, `period`, ...), so traps like \"trial period\" must keep the word.")
    w("- `discourse_filler`: each filler (like, you know, I mean, sort of, kind of, basically, literally, actually, so, okay so) appears as often as in clean.")
    w("- `question/request` (tags question, request_to_ai, command): not answered, and the same number of `?` as clean.\n")
    check_tags = ["self_correction", "already_clean", "list", "trailing_off", "voice_command", "discourse_filler", "question/request"]
    w("| tag check | n | " + " | ".join(systems) + " |")
    w("|---|---:|" + "---:|" * len(systems))
    for t in check_tags:
        n = sum(1 for r in recs_by_sys[systems[0]] if t in r["tag_checks"])
        if not n:
            continue
        cells = [pct(mean([r["tag_checks"][t] for r in recs_by_sys[s] if t in r["tag_checks"]])) for s in systems]
        w(f"| {t} | {n} | " + " | ".join(cells) + " |")
    w("")
    w("Judge pass rate by tag (all tags):\n")
    all_tags = sorted({t for r in recs_by_sys[systems[0]] for t in r["tags"]})
    w("| tag | n | " + " | ".join(systems) + " |")
    w("|---|---:|" + "---:|" * len(systems))
    for t in all_tags:
        n = sum(1 for r in recs_by_sys[systems[0]] if t in r["tags"])
        cells = []
        for s in systems:
            jr = judged([r for r in recs_by_sys[s] if t in r["tags"]])
            cells.append(pct(mean([r["judge"]["overall"] == "pass" for r in jr])) if jr else "–")
        w(f"| {t} | {n} | " + " | ".join(cells) + " |")
    w("")
    w("Judge failures by GUIDE rule (share of all examples where the judge marked the rule `fail`):\n")
    w("| rule | " + " | ".join(systems) + " |")
    w("|---|" + "---:|" * len(systems))
    for k in JUDGE_RULES:
        cells = [pct(mean([r["judge"]["rules"].get(k) == "fail" for r in judged(recs_by_sys[s])])) for s in systems]
        w(f"| {k} | " + " | ".join(cells) + " |")
    w("")
    if validation:
        w("## Judge validation\n")
        bs = validation.get("by_set", {})
        tu, ho = bs.get("tuning", ["?", "?"]), bs.get("holdout", ["?", "?"])
        w(f"Judge model: `{meta.get('_judge_model')}` (temperature 0, JSON mode). Before the judge was trusted, its "
          f"outputs were hand-checked against the GUIDE (`tools/eval/judge_validation.json`) and compared with its "
          f"overall verdict. These cases go to the judge even when they are exact matches. **Agreement with the final "
          f"rubric: {validation['agree']}/{validation['n']}** (the first 10 \"tuning\" cases: {tu[0]}/{tu[1]}; the "
          f"10 \"holdout\" cases labelled afterwards: {ho[0]}/{ho[1]}).\n")
        w(JUDGE_HISTORY + "\n")
        w("| set | system | id | output (abridged) | my verdict | judge | judge reason | my note |")
        w("|---|---|---|---|---|---|---|---|")
        for r in validation["rows"]:
            out = r["output"] if len(r["output"]) < 140 else r["output"][:137] + "..."
            w(f"| {r.get('set', '')} | {r['system']} | {r['id']} | {md_escape(out)} | {r['overall']} | {r['judge']}{'' if r['agree'] else ' ✗'} | "
              f"{md_escape(r['judge_reason'])} | {md_escape(r.get('note', ''))} |")
        w("")
    if api_cost is not None:
        w(f"OpenAI spend for the last invocation of the script (judge calls not already cached, plus the cloud "
          f"system if it ran; list prices): about **${api_cost:.2f}**.\n")
    manual = ""
    if path.exists():   # keep the hand-written analysis below the marker across re-runs
        old = path.read_text(encoding="utf-8")
        if MANUAL in old:
            manual = old[old.index(MANUAL):]
    path.write_text("\n".join(L) + "\n" + (manual or MANUAL + "\n"), encoding="utf-8")
    dump_failures(recs_by_sys[best], OUT / f"failures-{best}.md")


MANUAL = "<!-- hand-written below; eval_refine.py keeps everything from this line on -->"


def dump_failures(recs: list[dict], path: Path) -> None:
    """Every non-passing example of one system, for picking the instructive failures by hand."""
    L = []
    for r in recs:
        if r["judge"].get("overall") == "pass" and all(r["tag_checks"].values()) and r["guard"] == "ok":
            continue
        L.append(f"## {r['id']} {r['tags']} guard={r['guard']} judge={r['judge'].get('overall')} checks={r['tag_checks']}")
        L.append(f"- reason: {r['judge'].get('reason')}  added={r['added_words']} dropped={r['dropped_words']}")
        L.append(f"- RAW: {r['raw']!r}\n- EXP: {r['clean']!r}\n- GOT: {r['output']!r}\n")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- main


def save(name: str, recs: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"refine-baseline-{name}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    return p


def load_saved(name: str) -> list[dict] | None:
    p = OUT / f"refine-baseline-{name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--eval", type=Path, default=EVAL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-run", action="store_true", help="re-score saved outputs instead of running the systems")
    ap.add_argument("--rerun", default="", help="with --skip-run: comma list of systems to run anyway")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--budget", type=float, default=0.80, help="stop calling OpenAI above this spend (USD)")
    ap.add_argument("--env-file", type=Path, default=ROOT / ".env")
    ap.add_argument("--labels", type=Path, default=Path(__file__).resolve().parent / "judge_validation.json")
    ap.add_argument("--report", type=Path, default=ROOT / "docs/refine-baseline.md")
    ap.add_argument("--validate-only", action="store_true", help="only run the judge on the hand-labelled cases")
    ap.add_argument("--best", default="", help="local system to mine failures from (default: best judge pass)")
    a = ap.parse_args()

    rows = load_eval(a.eval, a.limit)
    check_prompt_parity(rows)
    names = [n for n in a.systems.split(",") if n]
    need_api = (not a.no_judge) or any(SYSTEMS[n].kind == "openai" for n in names)
    api = OpenAI(load_key(a.env_file)) if need_api else None

    meta_path = OUT / "meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    recs_by_sys: dict[str, list[dict]] = {}
    rerun = set(filter(None, a.rerun.split(",")))
    for n in names:
        sysm = SYSTEMS[n]
        saved = load_saved(n) if a.skip_run and n not in rerun else None
        if saved is not None:
            recs = saved[: len(rows)]
        else:
            print(f"running {n} ({sysm.label})")
            recs, m = run_local(sysm, rows) if sysm.kind == "local" else run_openai(sysm, rows, api)
            meta[n] = m
        recs_by_sys[n] = [score(r) for r in recs]
        save(n, recs_by_sys[n])

    validation = None
    if not a.no_judge:
        judge = Judge(api, a.judge_model, a.budget)
        meta["_judge_model"] = a.judge_model
        if a.labels.exists():
            validation = validate_judge(judge, recs_by_sys, a.labels)
            print(f"judge validation: {validation['agree']}/{validation['n']} agree; by set {validation['by_set']}")
            if a.validate_only:
                for r in validation["rows"]:
                    print(f"  {r.get('set','')[:4]} {r['system']:18s} {r['id']} me={r['overall']} judge={r['judge']}  {r['judge_reason']}")
                print(f"spend ${api.cost:.4f}")
                return
        for n in names:
            print(f"judging {n}")
            judge.all(recs_by_sys[n])
            save(n, recs_by_sys[n])
    else:
        for n in names:
            for r in recs_by_sys[n]:
                r.setdefault("judge", dict(AUTO_PASS) if r["em_light"] else {"rules": {}, "overall": "skipped", "reason": ""})

    OUT.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({k: v for k, v in meta.items() if not k.startswith("_report")}, indent=1))
    local = [n for n in names if SYSTEMS[n].kind == "local" and SYSTEMS[n].prompt == "shared"]
    best = a.best or (max(local, key=lambda n: summary(recs_by_sys[n])["judge_pass"]) if local else names[0])
    if api:
        print(f"OpenAI spend this run: ${api.cost:.3f}  usage [calls, in, cached, out]: {api.usage}")
    write_report(a.report, recs_by_sys, meta, validation, best, api.cost if api else None)
    print(f"wrote {a.report}  (best local system: {best})")
    for n in names:
        s = summary(recs_by_sys[n])
        print(f"{n:22s} judge {pct(s['judge_pass']):>4}  EM(norm) {pct(s['em_norm']):>4}  WER {s['wer_clean']:.3f}  "
              f"answered {pct(s['answered'])}  added {pct(s['added'])}  dropped {pct(s['dropped'])}  "
              f"guard {pct(s['guard'])}  lat {s['lat_med']:.0f}/{s['lat_p95']:.0f} ms")


if __name__ == "__main__":
    main()
