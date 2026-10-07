"""Check asr.phrases_like_app (Python PhraseCutter port) against the Rust PhraseCutter.

Runs `cargo run -p owf-audio --example cutter_parity` over cached "heard" clips and compares the
phrase lengths (in samples) clip by clip.

    .venv-data/Scripts/python tools/refine-data/cutter_parity.py [--n 200]
"""

from __future__ import annotations

import nowmi  # noqa: F401

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, HEARD, ROOT  # noqa: E402


def main() -> None:
    import numpy as np
    import soundfile as sf

    from asr import phrases_like_app

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    files = sorted(HEARD.glob("*.flac"))[: a.n]
    tmp = DATA / "scratch" / "cutter"
    tmp.mkdir(parents=True, exist_ok=True)
    py, raws = {}, []
    for f in files:
        x, _ = sf.read(str(f), dtype="float32")
        p = tmp / (f.stem + ".f32")
        x.astype("<f4").tofile(p)
        raws.append(str(p))
        py[p.as_posix()] = [len(c) for c in phrases_like_app(x)]
    env = {**os.environ, "CARGO_TARGET_DIR": str(ROOT / "target" / "refine-data")}
    out = subprocess.run(["cargo", "run", "-q", "-p", "owf-audio", "--example", "cutter_parity", "--", *raws],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    if out.returncode:
        sys.stderr.write(out.stderr)
        raise SystemExit("cargo failed")
    same = bad = 0
    for line in out.stdout.splitlines():
        r = json.loads(line)
        mine = py[Path(r["file"]).as_posix()]
        if mine == r["phrases"]:
            same += 1
        else:
            bad += 1
            if bad <= 10:
                print("MISMATCH", Path(r["file"]).stem, "rust", r["phrases"], "python", mine)
    for p in raws:
        Path(p).unlink(missing_ok=True)
    print(f"cutter parity: {same}/{same + bad} clips cut identically")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
