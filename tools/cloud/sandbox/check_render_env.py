"""Runs inside the render sandbox before any audio is made (tools/cloud/owf_daytona.py render).

Fails fast unless the sandbox has exactly the inputs the local pipeline uses:
* noise + room-IR files: the same relative paths and byte sizes as the local copy
  (tools/cloud/lkw_manifest.tsv), and the same sorted order augment.py draws from on Windows
  (case-insensitive path order there, case-sensitive here);
* Piper / Kokoro / Parakeet model files: the local SHA-256s;
* CUDA in the data venv (Piper) and the CUDA execution provider in the Kokoro venv.
Then prints the voice plan for the selected scripts (SAPI is redistributed off Windows).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "refine-data"))

import common  # noqa: E402

SHA = {
    common.PIPER_CKPT: "75a7e47205896630aff91c483d20d9b331350854c5836a24f44d8cdddcd7a568",
    common.PIPER_CKPT.with_suffix(".json"): "e0d23282eb18ec7a496ebfbc347e3a458b21ab6c77e99dcbea858184e3514bf6",
    common.TTS_MODELS / "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    common.TTS_MODELS / "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
PARAKEET = {  # src/openwhisprflow/stt/parakeet.py, revision 8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce
    "config.json": "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
    "vocab.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    "decoder_joint-model.int8.onnx": "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
    "encoder-model.int8.onnx": "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    bad: list[str] = []
    want = {}
    for line in (ROOT / "tools/cloud/lkw_manifest.tsv").read_text(encoding="utf-8").splitlines():
        rel, size = line.split("\t")
        want[rel] = int(size)
    lkw = common.LKW_DIR
    have = {str(p.relative_to(lkw)).replace("\\", "/"): p.stat().st_size
            for d in ("rirs", "backgrounds") for p in (lkw / d).rglob("*.wav")}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    sized = sorted(k for k in set(want) & set(have) if want[k] != have[k])
    print(f"lkw assets: {len(have)} wav files here, {len(want)} in the local manifest; "
          f"missing {len(missing)}, extra {len(extra)}, size mismatch {len(sized)}")
    if missing or extra or sized:
        bad.append(f"asset set differs: missing {missing[:5]} extra {extra[:5]} size {sized[:5]}")
    import augment  # noqa: E402

    for name, files in (("rir", augment.rir_files()), ("noise", augment.noise_files())):
        win = sorted(files, key=lambda p: str(p).lower())
        if list(files) != win:
            bad.append(f"{name} file order differs from the Windows order")
        print(f"{name}: {len(files)} files, order identical to Windows: {list(files) == win}")
    for p, h in SHA.items():
        got = sha256(p) if p.exists() else "MISSING"
        print(f"{p.name}: {'OK' if got == h else 'MISMATCH ' + got}")
        if got != h:
            bad.append(f"{p} sha256")
    import asr

    for n, h in PARAKEET.items():
        p = asr.PARAKEET_DIR / n
        got = sha256(p) if p.exists() else "MISSING"
        print(f"parakeet {n}: {'OK' if got == h else 'MISMATCH ' + got}")
        if got != h:
            bad.append(f"{p} sha256")
    import torch

    print(f"data venv: torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    if not torch.cuda.is_available():
        bad.append("no CUDA in the data venv")
    kgpu = ROOT / ".venv-kgpu" / "bin" / "python"
    probe = ("import torch, onnxruntime as o; o.set_default_logger_severity(3); "
             f"s = o.InferenceSession({str(common.TTS_MODELS / 'kokoro-v1.0.onnx')!r}, "
             "providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider']); "
             "print(o.__version__, s.get_providers())")
    r = subprocess.run([str(kgpu), "-c", probe], capture_output=True, text=True)
    print(f"kokoro venv session: {r.stdout.strip() or r.stderr[-300:]}")
    if "CUDAExecutionProvider" not in r.stdout:
        bad.append("no CUDAExecutionProvider in the Kokoro venv")
    print(subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True).stdout.strip())
    print(f"NO_SAPI={common.NO_SAPI}")
    files = sys.argv[1:]
    rows = common.load_scripts(files or None)
    c = Counter(common.plan_for(r)["voice"]["backend"] for r in rows)
    print(f"voice plan for {len(rows)} scripts ({' '.join(files) or 'all'}): {dict(c)}")
    if "sapi" in c:
        bad.append("SAPI clips planned off Windows")
    if bad:
        print("ENV CHECK FAILED:\n  " + "\n  ".join(bad))
        raise SystemExit(3)
    print("ENV CHECK OK")


if __name__ == "__main__":
    main()
