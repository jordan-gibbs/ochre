"""Shared paths, script loading and the deterministic per-clip plan (voice, conditions, engine).

Every random choice for a script id comes from ``rng_for(id, salt)``, so re-running any stage
reproduces the same clip, and finished work is skipped (all stages resume).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "training" / "refine" / "datasets" / "scripts-v2"
OUT = ROOT / "training" / "refine" / "datasets" / "real-v2"
AUDIT_DIR = OUT / "audit"
# gitignored (training/**/data/): audio cache, models, plans, logs. Never committed.
DATA = ROOT / "training" / "refine" / "data"
AUDIO = DATA / "audio-v2"
DRY = AUDIO / "tts"        # TTS output, 16 kHz mono FLAC, before augmentation
HEARD = AUDIO / "heard"    # what the recognizer heard (augmented), 16 kHz mono FLAC
PLAN = DATA / "plan-v2"
# Re-render set (r2.py): hesitation-free TTS input for rows the audit dropped as
# hesitation_misheard_as_word. Only loaded when asked for by name ("R2").
SCRIPTS_R2 = DATA / "scripts-r2"
TTS_MODELS = DATA / "tts-models"
LOGS = DATA / "logs-v2"
SR = 16000

# Read in place, never copied (livekit-wakeword data; the same files the wake word training uses).
# OWF_LKW_DIR points elsewhere (the cloud render job, tools/cloud/: the same livekit-wakeword
# downloads, checked file-for-file against this directory's manifest).
LKW_DIR = Path(os.environ.get("OWF_LKW_DIR") or DATA / "lkw")
PIPER_CKPT = LKW_DIR / "piper" / "en-us-libritts-high.pt"
RIR_DIRS = [LKW_DIR / "rirs"]
NOISE_DIRS = [LKW_DIR / "backgrounds"]
# Qwen3-TTS Base clone reference voice ("clone-a"; a synthetic assistant voice, NOT the user's).
CLONE_REF = Path(os.environ.get("OWF_CLONE_REF") or DATA / "voice" / "reference.wav")
CLONE_REF_TXT = CLONE_REF.with_suffix(".txt")

sys.path.insert(0, str(ROOT / "src"))  # openwhisprflow.audio.segmenter (split_at_pauses)

# ----------------------------------------------------------------------------- voices

KOKORO_EN = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
             "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
             "am_liam", "am_michael", "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella",
             "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
# Non-English Kokoro voices reading English: accented speakers (used sparingly).
KOKORO_ACCENT = ["hf_alpha", "hf_beta", "hm_omega", "hm_psi", "ef_dora", "em_alex", "ff_siwis",
                 "if_sara", "im_nicola", "pf_dora", "pm_alex", "jf_alpha", "zf_xiaoxiao", "zm_yunxi"]
QWEN_PRESETS = ["aiden", "ryan", "vivian", "serena", "uncle_fu", "dylan", "eric", "ono_anna", "sohee"]
SAPI_VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
PIPER_SPEAKERS = 904


@lru_cache(maxsize=1)
def piper_pool() -> list[int]:
    """LibriTTS speakers that passed piper_screen.py (intelligible at default prosody)."""
    f = HERE / "piper_speakers.json"
    if f.exists():
        kept = json.loads(f.read_text(encoding="utf-8-sig"))["kept"]
        if kept:
            return kept
    return list(range(PIPER_SPEAKERS))

# Share of clips per backend (sums to 1).
BACKEND_WEIGHTS = {"piper": 0.42, "kokoro": 0.28, "qwen": 0.21, "qwen_clone": 0.03, "sapi": 0.06}
# Whisper dropped for the full run (2026-10-04, lead): Parakeet is the shipped default. Rows
# recognized by Whisper before that keep raw_engine = whisper-large-v3-turbo.
WHISPER_SHARE = 0.0
CLEAN_SHARE = 0.20


# Since 2026-10-04 (audit of batches 001-014): TTS voices turned um/uh/er/hmm into real words
# ("Number", "her") far more often than real speakers' hesitations get misrecognized, and the app
# strips hesitations before the model anyway. New renders speak `spoken` without them; the
# exact TTS input is stored as `tts_text` on the row. (The first 4,000 renders included them.)
TTS_STRIP_HESITATIONS = True
TTS_HES = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "hm", "mm", "mhm"}


def strip_spoken_hesitations(spoken: str) -> str:
    """Drop hesitation tokens; a pause/trail marker on a dropped token moves to the word before it.
    Repeats, cut-offs, false starts, corrections, pauses and "..." are kept."""
    out: list[str] = []
    for tok in spoken.split():
        core = tok.rstrip(",.;:")
        marker = tok[len(core):]
        if core.lower() in TTS_HES:
            if marker and out and not out[-1].endswith((",", "...", "-")):
                out[-1] += "..." if "..." in marker else ","
            continue
        out.append(tok)
    return " ".join(out)


def rng_for(key: str, salt: str = "") -> random.Random:
    h = hashlib.sha256(f"real-v2|{key}|{salt}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def read_jsonl(path: Path) -> list[dict]:
    """Tolerates a partially written last line (script files grow while agents write them)."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def is_eval_file(stem: str) -> bool:
    """E1, E2, ...: eval-only script files (round 4, eval-v4). Never part of the training export
    and never in the main batch order; they are rendered and audited only when named."""
    return stem[:1] == "E" and stem[1:].isdigit()


def script_files(only: list[str] | None = None) -> list[Path]:
    files = sorted(SCRIPTS.glob("W*.jsonl"), key=lambda p: (len(p.stem), p.stem))
    extra = (sorted(SCRIPTS.glob("E*.jsonl"), key=lambda p: (len(p.stem), p.stem))
             + sorted(SCRIPTS_R2.glob("*.jsonl"))) if only else []
    return [f for f in files + extra if not only or f.stem in only]


def load_scripts(only: list[str] | None = None, limit: int | None = None,
                 per_file: int | None = None) -> list[dict]:
    """Scripts in file order. ``limit`` caps the total, ``per_file`` the first N per file."""
    rows: list[dict] = []
    for f in script_files(only):
        rs = [r for r in read_jsonl(f) if r.get("id") and isinstance(r.get("spoken"), str)
              and r["spoken"].strip()]
        for r in rs[:per_file] if per_file else rs:
            r["_file"] = f.stem
            rows.append(r)
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out[:limit] if limit else out


def _pick(rng: random.Random, weights: dict[str, float]) -> str:
    x, acc = rng.random(), 0.0
    for k, w in weights.items():
        acc += w
        if x < acc:
            return k
    return k


# Qwen3-TTS ran at 0.2-1x realtime on the shared GPU (vs 2-10x for Kokoro and Piper). First 35%
# of its clips were kept (the rest went to Piper); then, mid-run, every Qwen clip not yet rendered
# was re-assigned to Kokoro (QWEN_TO_KOKORO). Rendered clips keep their voice (see plan_voice).
QWEN_KEEP = 0.35
QWEN_TO_KOKORO = True
# Windows SAPI exists only on Windows. Off Windows (the Linux cloud render job, tools/cloud/), or
# with OWF_NO_SAPI=1, a clip planned for SAPI goes to Piper or Kokoro in proportion to their
# effective shares above (piper 0.42 + 0.21 * 0.65, kokoro 0.28 + 0.21 * 0.35 + 0.03), drawn from
# its own stream so every other clip's plan is unchanged.
NO_SAPI = os.environ.get("OWF_NO_SAPI", "1" if sys.platform != "win32" else "0") == "1"
SAPI_TO_PIPER = 0.5565 / 0.94


def plan_voice(sid: str) -> dict[str, Any]:
    done = DRY / f"{sid}.json"
    if done.exists():  # already rendered: the plan is whatever voice actually spoke it
        try:
            return json.loads(done.read_text(encoding="utf-8-sig"))["voice"]
        except (OSError, ValueError, KeyError):
            pass
    rng = rng_for(sid, "voice")
    backend = _pick(rng, BACKEND_WEIGHTS)
    if backend == "qwen" and rng_for(sid, "qwen-keep").random() >= QWEN_KEEP:
        backend = "piper"
    if QWEN_TO_KOKORO and backend in ("qwen", "qwen_clone"):
        backend = "kokoro"
    if NO_SAPI and backend == "sapi":
        backend = "piper" if rng_for(sid, "no-sapi").random() < SAPI_TO_PIPER else "kokoro"
    if backend == "piper":
        pool = piper_pool()
        a = rng.choice(pool)
        v: dict[str, Any] = {"backend": "piper", "speaker": a,
                             "length_scale": round(rng.uniform(0.92, 1.15), 3),
                             "noise": round(rng.uniform(0.55, 0.72), 3), "noise_w": round(rng.uniform(0.65, 0.9), 3)}
        if rng.random() < 0.3:  # blend two LibriTTS speakers (spherical interpolation of embeddings)
            v["speaker2"], v["mix"] = rng.choice(pool), round(rng.uniform(0.2, 0.8), 2)
            v["id"] = f"piper:libritts:{a}+{v['speaker2']}@{v['mix']}"
        else:
            v["id"] = f"piper:libritts:{a}"
    elif backend == "kokoro":
        pool = KOKORO_ACCENT if rng.random() < 0.15 else KOKORO_EN
        a = rng.choice(pool)
        v = {"backend": "kokoro", "voice": a, "speed": round(rng.uniform(0.9, 1.15), 3),
             "lang": "en-gb" if a[0] == "b" else "en-us"}
        if pool is KOKORO_EN and rng.random() < 0.35:
            b = rng.choice([x for x in KOKORO_EN if x[:2] == a[:2] and x != a] or KOKORO_EN)
            v["voice2"], v["mix"] = b, round(rng.uniform(0.25, 0.75), 2)
            v["id"] = f"kokoro:{a}+{b}@{v['mix']}"
        else:
            v["id"] = f"kokoro:{a}"
    elif backend == "qwen":
        a = rng.choice(QWEN_PRESETS)
        v = {"backend": "qwen", "speaker": a, "id": f"qwen3tts:{a}"}
    elif backend == "qwen_clone":
        v = {"backend": "qwen_clone", "id": "qwen3tts-clone:clone-a"}
    else:
        a = rng.choice(SAPI_VOICES)
        v = {"backend": "sapi", "voice": a, "rate": rng.choice([-2, -1, 0, 0, 1, 2]),
             "id": "sapi:" + a.split()[1].lower()}
    v["pause_scale"] = round(rng.uniform(0.7, 1.5), 2)  # this speaker's pause habits
    return v


def plan_engine(sid: str) -> str:
    return "whisper-large-v3-turbo" if rng_for(sid, "engine").random() < WHISPER_SHARE else "parakeet-tdt-0.6b-v3-int8"


# Round 4: long dictations under the default room/noise mix lost so many words that the audit
# dropped most of them (eval E7: 30/50). These files re-render long scripts (new ids) with clean
# audio only, the condition of a decent headset mic, so long-text behaviour gets measured/trained.
CLEAN_AUDIO_FILES = {"E14", "W52"}


def plan_for(row: dict) -> dict[str, Any]:
    sid = row["id"]
    return {"id": sid, "file": row.get("_file", ""), "voice": plan_voice(sid), "engine": plan_engine(sid),
            "clean_audio": row.get("_file") in CLEAN_AUDIO_FILES or rng_for(sid, "clean").random() < CLEAN_SHARE}


def iter_plans(rows: list[dict]) -> Iterator[dict]:
    for r in rows:
        yield plan_for(r)
