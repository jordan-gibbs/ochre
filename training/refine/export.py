"""Merge a LoRA adapter into Qwen3.5, convert to GGUF, quantize, and smoke-test in llama-server.

    .venv-train-refine/Scripts/python export.py output/<run>/adapter            # -> output/<run>/gguf/
    .venv-train-refine/Scripts/python export.py output/<run>/adapter --quants Q4_K_M,Q8_0 --skip-verify-merge

Stages (each timed, results in ``<out>/export_summary.json``):

1. **merge**: ``W += (alpha/r) * B @ A`` (fp32 math, stored bf16) directly into a copy of the base
   checkpoint's safetensors, keeping its original layout (``Qwen3_5ForConditionalGeneration`` keys,
   vision tower, MTP head) so llama.cpp's converter sees exactly the format Quill was built from.
2. **verify-merge**: logits of the merged checkpoint vs the base+adapter (PEFT) model on a few
   training prompts must match (bf16 tolerance).
3. **convert**: ``.llama.cpp/convert_hf_to_gguf.py --outtype bf16 --no-mtp`` (llama.cpp pinned to
   b11398 = a7b94df2c, the build the app ships; Quill GGUFs have no MTP/nextn tensors either).
4. **quantize**: ``llama-quantize`` from the app's b11398 CUDA build -> Q4_K_M, Q8_0.
5. **serve**: each quant is started in that llama-server with the production flags
   (docs/refinement.md §3.2); checks /health, that /tokenize of rendered prompts + targets equals
   the HF token ids used in training, and prints a few greedy /completion outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Sequential weight loading: the threaded loader spikes host commit, which crashed on this box
# while the TTS/ASR data jobs were running.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import render  # noqa: E402

LLAMA_TAG = "b11398"
LLAMA_COMMIT = "a7b94df2c616bc1f62a73b964b4a71cb0dcc488e"
LLAMA_SRC = HERE / ".llama.cpp"
LAPP = Path(os.environ.get("LOCALAPPDATA", "")) / "openwhisprflow"
LLAMA_BIN_DIRS = [LAPP / f"data/models/llama.cpp/{LLAMA_TAG}/win-cuda-13.4-x64", LAPP / f"models/llama.cpp/{LLAMA_TAG}/win-cuda-13.4-x64"]
# OWF_LLAMA_BIN_DIR: a b11398 build elsewhere (the Linux cloud job, tools/cloud/, uses the ubuntu-cuda release).
if os.environ.get("OWF_LLAMA_BIN_DIR"):
    LLAMA_BIN_DIRS.insert(0, Path(os.environ["OWF_LLAMA_BIN_DIR"]))
SAMPLE = HERE / "datasets/synth-v1/train.jsonl"


def find_bin(name: str) -> Path:
    if os.name != "nt":
        name = name.removesuffix(".exe")
    for d in LLAMA_BIN_DIRS:
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"{name} not found in {LLAMA_BIN_DIRS}")


def ensure_llama_src() -> None:
    if not (LLAMA_SRC / "convert_hf_to_gguf.py").exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/ggml-org/llama.cpp.git",
                        str(LLAMA_SRC)], check=True)
        subprocess.run(["git", "checkout", LLAMA_COMMIT], cwd=LLAMA_SRC, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=LLAMA_SRC, capture_output=True, text=True).stdout.strip()
    if head != LLAMA_COMMIT:
        raise SystemExit(f"{LLAMA_SRC} is at {head}, expected {LLAMA_COMMIT} ({LLAMA_TAG})")


# ---------------------------------------------------------------- 1. merge


def merge(adapter: Path, base: Path, out: Path) -> dict:
    acfg = json.loads((adapter / "adapter_config.json").read_text())
    r, alpha = acfg["r"], acfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if acfg.get("use_rslora") else alpha / r
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        m = re.match(r"^base_model\.model\.model\.(layers\.\d+\..+)\.lora_([AB])\.weight$", k)
        if not m:
            raise SystemExit(f"unexpected adapter tensor {k}")
        pairs.setdefault(f"model.language_model.{m.group(1)}.weight", {})[m.group(2)] = v
    idx = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(idx.values()))
    out.mkdir(parents=True, exist_ok=True)
    merged = 0
    for f in files:
        tensors = {}
        with safe_open(str(base / f), framework="pt") as sf:
            meta = sf.metadata()
            for k in sf.keys():
                t = sf.get_tensor(k)
                if k in pairs:
                    A, B = pairs[k]["A"].float(), pairs[k]["B"].float()
                    t = (t.float() + scale * (B @ A)).to(t.dtype)
                    merged += 1
                tensors[k] = t
        save_file(tensors, str(out / f), metadata=meta or {"format": "pt"})
    if merged != len(pairs):
        raise SystemExit(f"merged {merged} of {len(pairs)} LoRA pairs; missing: "
                         f"{[k for k in pairs if k not in idx][:5]}")
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (".gitattributes",):
            shutil.copy2(p, out / p.name)
    print(f"merge: {merged} LoRA deltas (r={r}, alpha={alpha}, scale={scale:g}) -> {out}")
    return {"merged_modules": merged, "r": r, "alpha": alpha}


# ---------------------------------------------------------------- 2. verify merge


def verify_merge(adapter: Path, base: Path, merged: Path, n: int = 3) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    rows, _ = render.load_rows([SAMPLE])
    batch = [render.tokenize(render.render(r), tok)["input_ids"] for r in rows[:n]]
    dev = "cuda"

    def logits(model) -> list[torch.Tensor]:
        model.to(dev).eval()
        with torch.no_grad():
            res = [model(input_ids=torch.tensor([ids], device=dev)).logits[0, -64:].float().cpu() for ids in batch]
        model.to("cpu")
        return res

    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    ref = logits(PeftModel.from_pretrained(m, adapter))
    del m
    got = logits(AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16))
    base_m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    orig = logits(base_m)
    del base_m
    torch.cuda.empty_cache()
    diff = max((a - b).abs().max().item() for a, b in zip(ref, got))
    moved = max((a - b).abs().max().item() for a, b in zip(ref, orig))
    agree = sum((a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(ref, got)) / len(ref)
    print(f"verify-merge: max |logit diff| merged vs PEFT = {diff:.3f} (vs base model: {moved:.3f}); "
          f"argmax agreement {agree:.1%}")
    # Argmax runs over every position, prompt included, where near-ties flip on bf16 rounding of
    # the merged weights (r32/alpha64 scales the delta 2x: 96.9% measured with a 0.53 max diff).
    # Q4_K_M quantization moves logits far more than this, so 95% still catches a broken merge.
    if agree < 0.95 or diff > max(1.0, moved / 4):
        raise SystemExit("merged checkpoint does not reproduce the adapter's logits")
    return {"max_logit_diff": round(diff, 4), "max_logit_shift_vs_base": round(moved, 4), "argmax_agree": round(agree, 4)}


# ---------------------------------------------------------------- 3-4. convert, quantize


def convert(merged: Path, out_file: Path) -> None:
    ensure_llama_src()
    subprocess.run([sys.executable, str(LLAMA_SRC / "convert_hf_to_gguf.py"), str(merged), "--outtype", "bf16",
                    "--no-mtp", "--outfile", str(out_file)], check=True,
                   env=dict(os.environ, PYTHONPATH=str(LLAMA_SRC / "gguf-py"), NO_LOCAL_GGUF=""))


def quantize(src: Path, dst: Path, qtype: str) -> None:
    subprocess.run([str(find_bin("llama-quantize.exe")), str(src), str(dst), qtype], check=True,
                   stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------- 5. serve


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def serve_check(gguf: Path, n: int = 6, base: Path = HERE / "data/base/Qwen3.5-2B") -> dict:
    from transformers import AutoTokenizer
    port = free_port()
    # docs/refinement.md §3.2 (GPU build) flags.
    args = [str(find_bin("llama-server.exe")), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
            "-c", "2048", "-np", "1", "-ngl", "999", "-fa", "auto", "-b", "1024", "--no-webui", "--no-jinja",
            "--cache-prompt", "--no-context-shift", "--fit", "off", "--cache-ram", "0"]
    log = gguf.with_suffix(".server.log")
    t0 = time.perf_counter()
    proc = subprocess.Popen(args, stdout=log.open("w"), stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({proc.returncode}); see {log}")
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.perf_counter() - t0 > 120:
                raise SystemExit(f"llama-server did not become healthy; see {log}")
            time.sleep(0.25)
        cold = time.perf_counter() - t0
        tok = AutoTokenizer.from_pretrained(base)
        rows, _ = render.load_rows([SAMPLE])
        mism, samples = 0, []
        for r in rows[:n]:
            ex = render.render(r)
            t = render.tokenize(ex, tok)
            ids = post(base_url + "/tokenize", {"content": ex["prompt"] + ex["completion"], "add_special": False,
                                                "parse_special": True})["tokens"]
            mism += ids != t["input_ids"]
            res = post(base_url + "/completion", {"prompt": ex["prompt"], "n_predict": max(32, int(len(r["raw"]) / 3.2 * 2) + 16),
                                                  "temperature": 0.0, "top_k": 1, "cache_prompt": True,
                                                  "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"], "stream": False})
            samples.append({"id": r["id"], "raw": r["raw"], "clean": r["clean"], "out": res["content"],
                            "tok_s": round(res["timings"].get("predicted_per_second", 0))})
        print(f"serve[{gguf.name}]: healthy in {cold:.1f}s; tokenizer parity HF==llama.cpp on {n - mism}/{n}")
        for s in samples:
            print(f"  {s['id']}: {s['out']!r}  (ref {s['clean']!r}, {s['tok_s']} tok/s)")
        if mism:
            raise SystemExit("llama.cpp tokenization differs from the HF tokenization used in training")
        return {"cold_start_s": round(cold, 2), "tokenizer_parity": f"{n - mism}/{n}", "samples": samples}
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", type=Path)
    ap.add_argument("--base", type=Path, default=HERE / "data/base/Qwen3.5-2B")
    ap.add_argument("--out", type=Path, help="default: <adapter>/../gguf")
    ap.add_argument("--name", help="GGUF file stem (default: <run_name>)")
    ap.add_argument("--quants", default="Q4_K_M,Q8_0")
    ap.add_argument("--skip-verify-merge", action="store_true")
    ap.add_argument("--keep-merged", action="store_true", help="keep the merged HF checkpoint (4.5 GB)")
    a = ap.parse_args()
    adapter = a.adapter.resolve()
    run_dir = adapter.parent
    out = (a.out or run_dir / "gguf").resolve()
    name = a.name or run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    # Resumable: every stage writes atomically (tmp + rename) and the summary is saved after each
    # stage, so a crash (this box has thrown native 0xC000070A faults under load) costs only a re-run.
    sum_path = out / "export_summary.json"
    summary: dict = json.loads(sum_path.read_text()) if sum_path.exists() else {}
    if summary.get("adapter") != str(adapter) or summary.get("adapter_mtime") != (adapter / "adapter_model.safetensors").stat().st_mtime:
        summary = {"adapter": str(adapter), "adapter_mtime": (adapter / "adapter_model.safetensors").stat().st_mtime,
                   "llama_cpp": f"{LLAMA_TAG} ({LLAMA_COMMIT[:9]})", "seconds": {}, "files": {}}
        for f in out.glob(f"{name}-*.gguf"):
            f.unlink()
        shutil.rmtree(run_dir / "merged-hf", ignore_errors=True)
    T = summary["seconds"]

    def save() -> None:
        sum_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def stage(key: str, fn) -> None:
        if key in T:
            print(f"{key}: done earlier ({T[key]} s), skipping")
            return
        t = time.perf_counter()
        fn()
        T[key] = round(time.perf_counter() - t, 1)
        save()

    merged = run_dir / "merged-hf"
    bf16 = out / f"{name}-bf16.gguf"
    quants = [q for q in a.quants.split(",") if q]
    need_merged = "convert_bf16" not in T or (not a.skip_verify_merge and "verify_merge" not in T)
    if need_merged and not (merged / ".complete").exists():
        T.pop("merge", None)
    if need_merged:
        def do_merge():
            tmp = run_dir / "merged-hf.partial"
            shutil.rmtree(tmp, ignore_errors=True)
            summary["merge"] = merge(adapter, a.base, tmp)
            shutil.rmtree(merged, ignore_errors=True)
            tmp.rename(merged)
            (merged / ".complete").write_text("ok")
        stage("merge", do_merge)
    if not a.skip_verify_merge:
        stage("verify_merge", lambda: summary.__setitem__("verify_merge", verify_merge(adapter, a.base, merged)))

    def do_convert():
        tmp = bf16.with_suffix(".partial")
        convert(merged, tmp)
        tmp.replace(bf16)
        summary["files"]["bf16"] = {"path": str(bf16), "bytes": bf16.stat().st_size}
    stage("convert_bf16", do_convert)
    for q in quants:
        dst = out / f"{name}-{q}.gguf"

        def do_quant(q=q, dst=dst):
            tmp = dst.with_suffix(".partial")
            quantize(bf16, tmp, q)
            tmp.replace(dst)
            summary["files"][q] = {"path": str(dst), "bytes": dst.stat().st_size}
            print(f"quantize: {dst.name} {dst.stat().st_size / 2**20:.0f} MiB")
        stage(f"quantize_{q}", do_quant)
    for q in quants:
        stage(f"serve_{q}", lambda q=q: summary["files"][q].__setitem__("serve", serve_check(out / f"{name}-{q}.gguf", base=a.base)))
    if not a.keep_merged:
        shutil.rmtree(merged, ignore_errors=True)
    save()
    print(json.dumps(T, indent=1))
    print(f"ggufs in {out}: " + ", ".join(f"{k}={v['bytes'] / 2**20:.0f} MiB" for k, v in summary["files"].items()))

if __name__ == "__main__":
    main()
