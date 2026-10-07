"""Wake word training pipeline (any word): synthetic data -> features -> livekit conv-attention head -> ONNX.

Grew out of an earlier wake word training pipeline. Run it with the
**training venv** (never the app venv) from the repo root:

    training/wakeword/.venv-train/Scripts/python -m training.wakeword.pipeline all --word transcribe
    training/wakeword/.venv-train/Scripts/python -m training.wakeword.pipeline retrain --word transcribe

Per-word settings live in ``training/wakeword/configs/<word>.yaml``. Word-independent data (Piper
LibriTTS checkpoint, room impulse responses, MUSAN backgrounds, ACAV100M + validation features) is
read from ``--data DIR``, else ``$OWF_WAKE_DATA``, else ``training/wakeword/data/lkw`` (gitignored).
Everything generated goes to ``training/wakeword/output/<word>/`` (gitignored); ``install`` copies the
head to ``assets/wake/<word>.onnx`` + ``<word>.json``.

Stages (each is resumable / skips finished work):
  generate  Piper VITS (LibriTTS, 904 speakers, random SLERP speaker pairs) -> raw clips.
            Test split uses held-out speakers. Positives are "<word>" alone or "<word>, <speech>";
            phoneme durations give the exact end of the wake word. With `wake.prefix` set, extra
            splits add "Hey/Okay/So/<a sentence>[,.] <word>[, <speech>]" positives and
            "<prefix> <near-miss>" negatives (*_prefix_{train,test}).
  features  Augment (EQ/distortion, room IR, MUSAN noise, gain) and extract openWakeWord embeddings
            **at int16 scale**, exactly what openwhisprflow/wake/frontend.py feeds the frozen mel /
            embedding ONNX models (livekit itself uses float [-1,1]). Also folds in
            data/real_positive_<word>/*.wav and data/real_room_<word>/*.wav (record_samples.py).
  train     livekit-wakeword 3-phase trainer (focal loss, mixup, checkpoint averaging).
  export    ONNX head, input (batch,16,96) "embeddings" -> (batch,1) "score".
  eval      Held-out recall / near-miss false-accept rate on the test split, and streaming false
            positives per hour on the openWakeWord validation set (~11 h) through the runtime's gate
            (threshold + 2 consecutive frames + cooldown). Picks a threshold (<= wake.max_fpph).
  install   Copy to assets/wake/<word>.onnx + <word>.json (previous kept in output/<word>/*.prev.*).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SR = 16000
WIN = 32000  # 2.0 s window -> 16 embedding frames
ESPEAK_DIRS = [Path(r"C:\Program Files\eSpeak NG"), Path(r"C:\Program Files (x86)\eSpeak NG")]
ACAV_FILES = ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", "acav100m_partial_16bit.npy")
INSTALL_DIR = REPO / "assets" / "wake"


# --------------------------------------------------------------------------- config


def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root(cli: str | None) -> Path:
    """Word-independent data: --data, else $OWF_WAKE_DATA, else training/wakeword/data/lkw."""
    p = Path(cli or os.environ.get("OWF_WAKE_DATA") or HERE / "data" / "lkw").expanduser().resolve()
    need = [p / "piper" / "en-us-libritts-high.pt", p / "features" / "validation_set_features.npy"]
    missing = [str(x) for x in need if not x.exists()]
    if not any((p / "features" / a).exists() for a in ACAV_FILES):
        missing.append(str(p / "features" / ACAV_FILES[0]) + " (or the partial slice)")
    if missing:
        raise SystemExit("wake word data not found; set --data or OWF_WAKE_DATA (see training/wakeword/README.md). "
                         "Missing:\n  " + "\n  ".join(missing))
    return p


def lk_config(raw: dict, P: Paths):
    from livekit.wakeword.config import WakeWordConfig

    kw = {k: v for k, v in raw.items() if k != "wake"}
    kw["data_dir"] = str(P.data)
    kw["output_dir"] = str(P.out.parent)
    aug = dict(kw.get("augmentation") or {})
    aug.setdefault("background_paths", [str(p) for p in P.backgrounds])
    aug.setdefault("rir_paths", [str(p) for p in P.rirs])
    kw["augmentation"] = aug
    return WakeWordConfig(**kw)


def wcfg(raw: dict) -> dict:
    """Our own generation/eval section."""
    return raw["wake"]


def word_of(raw: dict) -> str:
    return str(raw["model_name"])


class Paths:
    def __init__(self, raw: dict, data: Path) -> None:
        self.word = word = str(raw["model_name"])
        self.data = data
        self.out = HERE / "output" / word
        self.clips = self.out / "clips"
        self.feats = self.out / "features"
        self.piper = data / "piper" / "en-us-libritts-high.pt"
        self.acav = next((data / "features" / a for a in ACAV_FILES if (data / "features" / a).exists()),
                         data / "features" / ACAV_FILES[0])
        self.val = data / "features" / "validation_set_features.npy"
        self.backgrounds = [data / "backgrounds"]
        self.rirs = [data / "rirs"]
        self.real_pos = HERE / "data" / f"real_positive_{word}"
        self.real_room = HERE / "data" / f"real_room_{word}"
        self.models = INSTALL_DIR


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- TTS


def espeak_exe() -> str:
    exe = shutil.which("espeak-ng")
    if exe:
        return exe
    for d in ESPEAK_DIRS:
        if (d / "espeak-ng.exe").exists():
            return str(d / "espeak-ng.exe")
    raise FileNotFoundError("espeak-ng not found (winget install eSpeak-NG.eSpeak-NG)")


class Phonemizer:
    def __init__(self) -> None:
        self.exe = espeak_exe()
        self.cache: dict[str, str] = {}

    def __call__(self, text: str) -> str:
        if text not in self.cache:
            r = subprocess.run([self.exe, "--ipa", "-q", "-v", "en-us", text], capture_output=True,
                               encoding="utf-8", check=True)
            self.cache[text] = " ".join(line.strip() for line in r.stdout.splitlines() if line.strip())
        return self.cache[text]

    def warm(self, texts: list[str]) -> None:
        from concurrent.futures import ThreadPoolExecutor

        todo = sorted({t for t in texts if t not in self.cache})
        with ThreadPoolExecutor(12) as ex:
            for t, p in zip(todo, ex.map(lambda t: Phonemizer.__call__(self, t), todo), strict=True):
                self.cache[t] = p


class Synth:
    """Piper VITS (livekit's vendored copy) with per-token durations exposed."""

    def __init__(self, ckpt: Path, gpu_mem_fraction: float = 0.3) -> None:
        import torch
        import torchaudio
        from livekit.wakeword.data.piper.synthesis import _load_vits_model

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dev.type == "cuda":  # leave VRAM for the desktop / other apps
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
        self.model = _load_vits_model(ckpt, self.dev)
        cfg = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
        self.idmap: dict[str, list[int]] = cfg["phoneme_id_map"]
        self.n_speakers = int(cfg["num_speakers"])
        self.resampler = torchaudio.transforms.Resample(
            22050, 16000, lowpass_filter_width=64, rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
        self.hop = 256

    def ids(self, ipa: str) -> list[int]:
        out: list[int] = []
        for ch in unicodedata.normalize("NFD", ipa):
            if ch in self.idmap:
                out.extend(self.idmap[ch])
                out.extend(self.idmap["_"])
        return out

    def run(self, seqs: list[tuple[list[int], int]], spk1: list[int], spk2: list[int], slerp_w: float,
            length_scale: float, noise: float, noise_w: float) -> list[tuple[np.ndarray, int]]:
        """seqs: (inner ids, n inner ids belonging to the wake part or -1). Returns (audio16k, wake_end16k)."""
        torch = self.torch
        from livekit.wakeword.data.piper.vits_utils import generate_path, sequence_mask, slerp

        full = [self.idmap["^"] + s + self.idmap["$"] for s, _ in seqs]
        lens = [len(s) for s in full]
        mx = max(lens)
        full = [s + [1] * (mx - len(s)) for s in full]
        m = self.model
        with torch.no_grad():
            x = torch.LongTensor(full).to(self.dev)
            xl = torch.LongTensor(lens).to(self.dev)
            x_enc, m_p0, logs_p0, x_mask = m.enc_p(x, xl)
            g = slerp(m.emb_g(torch.LongTensor(spk1).to(self.dev)), m.emb_g(torch.LongTensor(spk2).to(self.dev)),
                      slerp_w).unsqueeze(-1)
            logw = m.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_w) if m.use_sdp else m.dp(x_enc, x_mask, g=g)
            # clamp per-token duration (<= ~0.35 s) so a rare SDP outlier can't make a huge batch
            w_ceil = torch.clamp(torch.ceil(torch.exp(logw) * x_mask * length_scale), max=30)
            y_len = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(sequence_mask(y_len, int(y_len.max().item())), 1).type_as(x_mask)
            attn = generate_path(w_ceil, torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1))
            m_p = torch.matmul(attn.squeeze(1), m_p0.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p0.transpose(1, 2)).transpose(1, 2)
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise
            z = m.flow(z_p, y_mask, g=g, reverse=True)
            audio = m.dec(z * y_mask, g=g).float().cpu()
            audio16 = self.resampler(audio).numpy()[:, 0, :]
            cum = torch.cumsum(w_ceil[:, 0, :], dim=1).cpu().numpy()
            y_len = y_len.cpu().numpy()
        res = []
        for i, (_, n_wake) in enumerate(seqs):
            n16 = int(y_len[i] * self.hop * 16000 / 22050)
            a = audio16[i, :n16]
            # tokens: ^ + inner; wake part covers inner[:n_wake] -> cumulative frames through index n_wake
            wake_frames = cum[i, n_wake] if n_wake >= 0 else cum[i, lens[i] - 1]
            res.append((a, int(wake_frames * self.hop * 16000 / 22050)))
        return res


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio))) or 1.0
    pcm = np.clip(audio / peak * 0.9 * 32767, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a[:, 0]
    if sr != SR:
        import librosa

        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a


def trim_lead(a: np.ndarray, wake_end: int, keep: int = 1600) -> tuple[np.ndarray, int]:
    """Drop leading near-silence (keeping `keep` samples), shifting wake_end accordingly."""
    env = np.abs(a)
    thr = 0.02 * float(env.max() or 1.0)
    hit = np.flatnonzero(env > thr)
    start = max(0, int(hit[0]) - keep) if len(hit) else 0
    return a[start:], max(1, wake_end - start)


def _ph_key(ipa: str) -> str:
    """Phoneme string without stress marks / spaces, for 'is this the wake word?' checks."""
    return "".join(c for c in ipa if c not in "ˈˌ ,.-")


def cmu_adversarial(n: int, rng: random.Random, word: str, ph: Phonemizer | None = None,
                    wake_spellings: list[str] | tuple[str, ...] = ()) -> list[str]:
    from livekit.wakeword.data.generate import generate_adversarial_phrases

    random.seed(rng.random())
    words = [w for w in generate_adversarial_phrases([word]) if word not in w.lower().replace(" ", "")
             and "'" not in w and w.isascii()]
    rng.shuffle(words)
    words = words[: n * 2]
    if ph is not None and wake_spellings:  # drop exact homophones of the wake word (e.g. "done more" for a two-word name)
        ph.warm(words + list(wake_spellings))
        wake = {_ph_key(ph(s)) for s in wake_spellings}
        words = [w for w in words if _ph_key(ph(w)) not in wake]
    return words[:n]


def stage_generate(raw: dict, P: Paths) -> None:
    o = wcfg(raw)
    rng = random.Random(o["seed"])
    ph = Phonemizer()
    synth = Synth(P.piper, float(o.get("gpu_mem_fraction", 0.3)))
    spell, weights = zip(*o["wake_spellings"].items(), strict=True)
    cmds = o["commands"]
    near = o["near_misses"]
    cmu = cmu_adversarial(o["cmu_adversarial_words"], rng, word_of(raw), ph, spell)
    log(f"{len(cmu)} CMU phonetic-neighbour words, {len(near)} near-misses, {len(cmds)} commands")
    ph.warm(list(spell) + cmds + near + cmu)
    for s in spell:
        log(f"  wake spelling {s!r:12} -> {ph(s)}")
    wake_keys = {_ph_key(ph(s)) for s in spell}
    clash = [t for t in near if "," not in t and _ph_key(ph(t)) in wake_keys]
    if clash:  # a "negative" that sounds exactly like the wake word would teach the model to reject it
        log(f"  dropping near-misses that are homophones of the wake word: {clash}")
        near = [t for t in near if t not in clash]

    seps = {"comma": ",", "space": "", "period": "."}
    frags = o.get("fragments") or [word_of(raw)[: max(2, len(word_of(raw)) // 2)]]
    ph.warm([p.strip() for f in frags for p in f.split(",")])

    def pos_item() -> tuple[list[int], int, str]:
        w = rng.choices(spell, weights)[0]
        wake = synth.ids(ph(w))
        if rng.random() < o["command_fraction"]:
            c = rng.choice(cmds)
            sep = rng.choice(["comma", "comma", "space", "period"])
            mid = synth.ids(seps[sep]) + synth.ids(" ")
            return wake + mid + synth.ids(ph(c)), len(wake), f"{w}{seps[sep]} {c}"
        end = rng.choice(["", "", ".", "!", "?", ","])
        return wake + synth.ids(end), len(wake), w + end

    def neg_item() -> tuple[list[int], int, str]:
        r = rng.random()
        if r < 0.40:
            t = rng.choice(near)
        elif r < 0.55:
            t = f"{rng.choice(near)}, {rng.choice(cmds)}"
        elif r < 0.68:
            t = rng.choice(cmds)
        elif r < 0.73:  # the command with the wake word removed / only a fragment of it
            t = f"{rng.choice(frags)}, {rng.choice(cmds)}"
        else:
            k = rng.choice([1, 1, 2, 3])
            t = " ".join(rng.choice(cmu) for _ in range(k))
        ids = synth.ids(", ".join(ph(p.strip()) for p in t.split(",")) if "," in t else ph(t))
        return ids, -1, t

    # "Hey <word>" / "Okay, <word>" / "<a sentence>. <word> send": the prefix sits inside the 2 s window.
    # Earlier models never saw that (every positive started after silence) -> ~10-30% detection.
    pf = o.get("prefix") or {}
    pre_words, pre_w = zip(*pf["words"].items(), strict=True) if pf else ((), ())
    pre_seps = {"comma": ",", "space": "", "period": "."}
    if pf:
        ph.warm(list(pre_words))

    def prefix_ids() -> tuple[list[int], str]:
        p = rng.choices(pre_words, pre_w)[0]
        sep = rng.choices(list(pf["pause"]), list(pf["pause"].values()))[0]  # comma/period = audible pause
        return synth.ids(ph(p)) + synth.ids(pre_seps[sep]) + synth.ids(" "), f"{p}{pre_seps[sep]} "

    def pos_prefix_item() -> tuple[list[int], int, str]:
        ids, n_wake, text = pos_item()
        pre, pt = prefix_ids()
        return pre + ids, len(pre) + n_wake, pt + text

    def neg_prefix_item() -> tuple[list[int], int, str]:
        r = rng.random()
        if r < 0.55:
            t = rng.choice(near)
        elif r < 0.75:
            t = rng.choice(cmds)
        else:
            t = " ".join(rng.choice(cmu) for _ in range(rng.choice([1, 1, 2])))
        pre, pt = prefix_ids()
        ids = synth.ids(", ".join(ph(p.strip()) for p in t.split(",")) if "," in t else ph(t))
        return pre + ids, -1, pt + t

    train_spk = [s for s in range(synth.n_speakers) if s % o["test_speaker_mod"]]
    test_spk = [s for s in range(synth.n_speakers) if not s % o["test_speaker_mod"]]
    splits = [
        ("positive_train", o["n_positive_train"], pos_item, train_spk),
        ("positive_test", o["n_positive_test"], pos_item, test_spk),
        ("negative_train", o["n_negative_train"], neg_item, train_spk),
        ("negative_test", o["n_negative_test"], neg_item, test_spk),
    ]
    if pf:
        splits += [
            ("positive_prefix_train", pf["n_positive_train"], pos_prefix_item, train_spk),
            ("positive_prefix_test", pf["n_positive_test"], pos_prefix_item, test_spk),
            ("negative_prefix_train", pf["n_negative_train"], neg_prefix_item, train_spk),
            ("negative_prefix_test", pf["n_negative_test"], neg_prefix_item, test_spk),
        ]
    for name, n, make, spk in splits:
        d = P.clips / name
        d.mkdir(parents=True, exist_ok=True)
        man_path = d / "manifest.jsonl"
        done = sum(1 for _ in open(man_path, encoding="utf-8")) if man_path.exists() else 0
        if done >= n:
            log(f"{name}: {done}/{n} clips already present")
            continue
        log(f"{name}: generating {n - done} clips (speakers {len(spk)})")
        t0 = time.time()
        with open(man_path, "a", encoding="utf-8") as man:
            i = done
            while i < n:
                b = min(o["tts_batch"], n - i)
                items = [make() for _ in range(b)]
                # the text for neg items needs the comma-split phonemes; items already hold ids
                s1 = [rng.choice(spk) for _ in range(b)]
                s2 = [rng.choice(spk) for _ in range(b)]
                out = synth.run([(ids, nw) for ids, nw, _ in items], s1, s2, rng.uniform(0.0, 1.0),
                                rng.uniform(*o["length_scales"]), rng.uniform(*o["noise_scales"]),
                                rng.uniform(*o["noise_scale_ws"]))
                for (_ids, nw, text), (audio, wake_end), a, bb in zip(items, out, s1, s2, strict=True):
                    if len(audio) < 1600 or not np.isfinite(audio).all() or np.abs(audio).max() < 1e-4:
                        continue
                    audio, wake_end = trim_lead(audio, wake_end)
                    fn = f"clip_{i:06d}.wav"
                    write_wav(d / fn, audio)
                    man.write(json.dumps({"file": fn, "text": text, "wake_end": wake_end if nw >= 0 else None,
                                          "len": len(audio), "spk": [a, bb]}) + "\n")
                    i += 1
                if (i // o["tts_batch"]) % 25 == 0:
                    man.flush()
                    rate = (i - done) / (time.time() - t0)
                    log(f"  {name}: {i}/{n} ({rate:.0f} clips/s)")
        log(f"{name}: done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- augmentation + features

_W: dict = {}


def _worker_init(bg_dirs: list[str], rir_dirs: list[str], seed: int) -> None:
    import onnxruntime as ort
    import soundfile as sf
    from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    _W["mel"] = ort.InferenceSession(str(get_mel_model_path()), so, providers=["CPUExecutionProvider"])
    _W["emb"] = ort.InferenceSession(str(get_embedding_model_path()), so, providers=["CPUExecutionProvider"])
    bgs = []
    for d in bg_dirs:
        for p in sorted(Path(d).glob("**/*.wav")):
            try:
                info = sf.info(str(p))
                if info.frames > WIN // 4:
                    bgs.append((str(p), info.frames, info.samplerate))
            except Exception:
                pass
    _W["bgs"] = bgs
    rirs = []
    for d in rir_dirs:
        for p in sorted(Path(d).glob("**/*.wav")):
            r, _ = sf.read(str(p), dtype="float32", always_2d=True)
            r = r[:, 0]
            rirs.append(r / (np.max(np.abs(r)) + 1e-8))
    _W["rirs"] = rirs
    from audiomentations import Compose, SevenBandParametricEQ, TanhDistortion

    _W["aug"] = Compose([SevenBandParametricEQ(p=0.25), TanhDistortion(min_distortion=0.01, max_distortion=0.4, p=0.2)])
    _W["rng"] = np.random.default_rng(seed + os.getpid())


def embed_int16_scale(x: np.ndarray) -> np.ndarray:
    """(WIN,) float audio in int16 units -> (16, 96) embeddings (last 16 frames)."""
    mel = _W["mel"].run(None, {"input": x.astype(np.float32)[None, :]})[0]
    mel = np.squeeze(mel).reshape(-1, 32) / 10.0 + 2.0
    wins = [mel[i:i + 76] for i in range(0, mel.shape[0] - 75, 8)]
    e = _W["emb"].run(None, {"input_1": np.stack(wins)[..., None].astype(np.float32)})[0].reshape(-1, 96)
    e = e[-16:]
    if e.shape[0] < 16:
        e = np.concatenate([np.zeros((16 - e.shape[0], 96), np.float32), e])
    return e


def _bg_segment(n: int, rng) -> np.ndarray:
    import soundfile as sf

    path, frames, sr = _W["bgs"][rng.integers(len(_W["bgs"]))]
    need = int(n * sr / SR) + 1
    start = int(rng.integers(0, max(1, frames - need)))
    a, _ = sf.read(path, start=start, frames=need, dtype="float32", always_2d=True)
    a = a[:, 0]
    if sr != SR:
        import librosa

        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    if len(a) < n:
        a = np.tile(a, n // max(1, len(a)) + 1)
    return a[:n]


def _finish(win: np.ndarray, rng, speech: bool, snr: tuple[float, float], p_noise: float) -> np.ndarray:
    if speech and _W["rirs"] and rng.random() < 0.5:
        from scipy.signal import fftconvolve

        r = _W["rirs"][rng.integers(len(_W["rirs"]))]
        win = fftconvolve(win, r, mode="full")[: len(win)].astype(np.float32)
    rms_s = float(np.sqrt(np.mean(win ** 2))) + 1e-9
    if _W["bgs"] and rng.random() < p_noise:
        bg = _bg_segment(len(win), rng)
        rms_b = float(np.sqrt(np.mean(bg ** 2))) + 1e-9
        db = rng.uniform(*snr)
        win = win + bg * (rms_s / rms_b) / (10 ** (db / 20))
    # mic self-noise floor
    win = win + rng.normal(0, 1, len(win)).astype(np.float32) * rms_s * 10 ** (-rng.uniform(40, 65) / 20)
    target_db = rng.uniform(-42, -14)
    rms = float(np.sqrt(np.mean(win ** 2))) + 1e-9
    win = win * (10 ** (target_db / 20) / rms)
    return np.clip(win, -1.0, 1.0) * 32767.0


def _task(args: tuple) -> np.ndarray:
    kind, path, wake_end, n_aug = args
    rng = _W["rng"]
    out = np.zeros((n_aug, 16, 96), np.float32)
    if kind == "bg":
        for k in range(n_aug):
            if rng.random() < 0.15:  # quiet room tone
                win = rng.normal(0, 1, WIN).astype(np.float32) * 1e-3
                win = _finish(win, rng, False, (0, 0), 0.0)
            else:
                win = _bg_segment(WIN, rng)
                win = _finish(win, rng, False, (0, 0), 0.0)
            out[k] = embed_int16_scale(win)
        return out
    if kind == "room":  # real room recording: path=(file), wake_end=start sample
        a = read_wav(Path(path))
        seg = a[wake_end:wake_end + WIN]
        seg = np.pad(seg, (0, WIN - len(seg)))
        for k in range(n_aug):
            g = 10 ** (rng.uniform(-6, 6) / 20)
            out[k] = embed_int16_scale(np.clip(seg * g, -1, 1) * 32767.0)
        return out
    audio = read_wav(Path(path))
    for k in range(n_aug):
        a = audio.copy()
        if rng.random() < 0.45:
            a = _W["aug"](samples=a, sample_rate=SR).astype(np.float32)
        a = a / (np.max(np.abs(a)) + 1e-9) * 0.5
        win = np.zeros(WIN, np.float32)
        if kind in ("pos", "realpos"):
            # end of window = end of the wake word + 0..0.45 s (trailing command speech or silence)
            end = wake_end + int(rng.uniform(0.0, 0.45) * SR)
            seg = a[:min(end, len(a))]
            seg = np.pad(seg, (0, max(0, end - len(a))))[-WIN:]
            win[WIN - len(seg):] = seg
            snr = (3, 25) if kind == "pos" else (5, 30)
            win = _finish(win, rng, True, snr, 0.75)
        else:
            if len(a) > WIN:
                s = int(rng.integers(0, len(a) - WIN + 1))
                a = a[s:s + WIN]
            end = int(rng.uniform(0.55, 1.12) * WIN)  # sometimes truncated at the window edge
            end = max(end, min(len(a), WIN))
            start = end - len(a)
            lo, hi = max(0, start), min(WIN, end)
            win[lo:hi] = a[lo - start:hi - start]
            win = _finish(win, rng, True, (0, 25), 0.8)
        out[k] = embed_int16_scale(win)
    return out


def _run_tasks(tasks: list[tuple], P: Paths, raw: dict, seed: int, desc: str) -> np.ndarray:
    from multiprocessing import Pool

    from tqdm import tqdm

    if not tasks:
        return np.zeros((0, 16, 96), np.float32)
    n_proc = max(1, min(14, (os.cpu_count() or 4) - 2))
    pool = Pool(n_proc, initializer=_worker_init,
                initargs=([str(p) for p in P.backgrounds], [str(p) for p in P.rirs], seed))
    try:
        res = list(tqdm(pool.imap(_task, tasks, chunksize=16), total=len(tasks), desc=desc, mininterval=10))
    finally:
        pool.close()  # close/join, not terminate: terminate() hits "Access is denied" on Windows
        pool.join()
    return np.concatenate(res, axis=0)


def _manifest(d: Path) -> list[dict]:
    with open(d / "manifest.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(path: Path, x: np.ndarray, shuffle_seed: int | None = None) -> None:
    if shuffle_seed is not None and len(x):
        x = x[np.random.default_rng(shuffle_seed).permutation(len(x))]
    np.save(path, x.astype(np.float32))
    log(f"  saved {path.name} {x.shape}")


def stage_features(raw: dict, P: Paths, only_real: bool = False) -> None:
    o = wcfg(raw)
    P.feats.mkdir(parents=True, exist_ok=True)
    seed = o["seed"]
    if not only_real:
        splits = [("positive_train", "pos", o["aug_pos_train"]), ("positive_test", "pos", o["aug_test"]),
                  ("negative_train", "neg", o["aug_neg_train"]), ("negative_test", "neg", o["aug_test"])]
        if o.get("prefix"):
            splits += [("positive_prefix_train", "pos", o["aug_pos_train"]),
                       ("positive_prefix_test", "pos", o["aug_test"]),
                       ("negative_prefix_train", "neg", o["aug_neg_train"]),
                       ("negative_prefix_test", "neg", o["aug_test"])]
        for split, kind, n_aug in splits:
            out = P.feats / f"synth_{split}.npy"
            if out.exists():
                log(f"{out.name} exists, skipping")
                continue
            d = P.clips / split
            drop = tuple(o.get("drop_negative_texts") or ())  # labels that became ambiguous ("sober on")
            tasks = [(kind, str(d / m["file"]), m["wake_end"] if m["wake_end"] is not None else m["len"], n_aug)
                     for m in _manifest(d)
                     if not (kind == "neg" and drop and m["text"].lower().startswith(drop))]
            _save(out, _run_tasks(tasks, P, raw, seed, split), seed)
        for split, n in [("background_train", o["n_background_train"]), ("background_test", o["n_background_test"])]:
            out = P.feats / f"synth_{split}.npy"
            if out.exists():
                log(f"{out.name} exists, skipping")
                continue
            shared = os.environ.get("OWF_WAKE_SHARED_BG")  # word-independent: reuse another word's windows
            if shared and (Path(shared) / out.name).exists():
                shutil.copyfile(Path(shared) / out.name, out)
                log(f"{out.name}: copied word-independent background features from {shared}")
                continue
            _save(out, _run_tasks([("bg", "", 0, 1)] * n, P, raw, seed + 7, split), seed)

    # real recordings (record_samples.py) -> always recomputed (cheap)
    real_pos = sorted(P.real_pos.glob("*.wav")) if P.real_pos.exists() else []
    tr, te = [], []
    for i, p in enumerate(real_pos):
        a = read_wav(p)
        side = p.with_suffix(".json")  # "<word>, <command>" takes: record_samples.py stores where the word ends
        if side.exists():
            wake_end = int(json.loads(side.read_text(encoding="utf-8"))["wake_end"])
        else:
            env = np.abs(a)
            hit = np.flatnonzero(env > 0.1 * float(env.max() or 1))
            wake_end = int(hit[-1]) + int(0.03 * SR) if len(hit) else len(a)
        (te if (i % o["real_test_every"] == o["real_test_every"] - 1) else tr).append((p, wake_end))
    _save(P.feats / "real_positive_train.npy",
          _run_tasks([("realpos", str(p), w, o["real_pos_aug"]) for p, w in tr], P, raw, seed + 11, "real_pos_train"),
          seed)
    _save(P.feats / "real_positive_test.npy",
          _run_tasks([("realpos", str(p), w, 10) for p, w in te], P, raw, seed + 12, "real_pos_test"))
    room = sorted(P.real_room.glob("*.wav")) if P.real_room.exists() else []
    rtr, rte = [], []
    for p in room:
        n = len(read_wav(p))
        starts = list(range(0, max(1, n - WIN), SR // 4))
        cut = int(len(starts) * 0.8)
        rtr += [("room", str(p), s, 2) for s in starts[:cut]]
        rte += [("room", str(p), s, 1) for s in starts[cut:]]
    _save(P.feats / "real_room_train.npy", _run_tasks(rtr, P, raw, seed + 13, "room_train"), seed)
    _save(P.feats / "real_room_test.npy", _run_tasks(rte, P, raw, seed + 14, "room_test"))


def _cat(paths: list[Path]) -> np.ndarray:
    xs = [np.load(p) for p in paths if p.exists()]
    xs = [x for x in xs if len(x)]
    return np.concatenate(xs) if xs else np.zeros((0, 16, 96), np.float32)


def assemble(raw: dict, P: Paths) -> None:
    """Write livekit trainer inputs into output/<word>/ (names the trainer expects)."""
    f = P.feats
    real_tr = np.load(f / "real_positive_train.npy") if (f / "real_positive_train.npy").exists() else None
    pos = [np.load(f / "synth_positive_train.npy")]
    if (f / "synth_positive_prefix_train.npy").exists():  # "Hey/Okay/So <word>"
        pos.append(np.load(f / "synth_positive_prefix_train.npy"))
    if real_tr is not None and len(real_tr):
        pos += [real_tr, real_tr]  # real voice weighted x2 (x60 augmentations each already)
    _save(P.out / "positive_features_train.npy", np.concatenate(pos), 1)
    _save(P.out / "positive_features_test.npy", _cat([f / "synth_positive_test.npy", f / "real_positive_test.npy"]))
    _save(P.out / "negative_features_train.npy",
          _cat([f / "synth_negative_train.npy", f / "synth_negative_prefix_train.npy"]), 3)
    _save(P.out / "negative_features_test.npy", _cat([f / "synth_negative_test.npy"]))
    _save(P.out / "background_noise_features_train.npy",
          _cat([f / "synth_background_train.npy", f / "real_room_train.npy"]), 2)
    _save(P.out / "background_noise_features_test.npy",
          _cat([f / "synth_background_test.npy", f / "real_room_test.npy"]))


# --------------------------------------------------------------------------- train / export


def stage_train(raw: dict, P: Paths) -> None:
    import torch
    from livekit.wakeword.training.trainer import WakeWordTrainer

    assemble(raw, P)
    cfg = lk_config(raw, P)
    torch.manual_seed(wcfg(raw)["seed"])

    class Trainer(WakeWordTrainer):
        """Our feature files (incl. the partial ACAV100M slice) instead of livekit's default names."""

        def _build_dataloader(self):
            from livekit.wakeword.data.dataset import create_dataloader

            files = {
                "positive": P.out / "positive_features_train.npy",
                "adversarial_negative": P.out / "negative_features_train.npy",
                "ACAV100M_sample": P.acav,
                "background_noise": P.out / "background_noise_features_train.npy",
            }
            labels = {"positive": lambda _: 1, "adversarial_negative": lambda _: 0,
                      "ACAV100M_sample": lambda _: 0, "background_noise": lambda _: 0}
            return create_dataloader(files, self.config.batch_n_per_class, labels)

    t0 = time.time()
    trainer = Trainer(cfg)
    trainer.train()
    trainer.save(P.out / f"{P.word}.pt")
    log(f"training done in {time.time() - t0:.0f}s")


def stage_export(raw: dict, P: Paths) -> Path:
    from livekit.wakeword.export.onnx import export_onnx

    cfg = lk_config(raw, P)
    return export_onnx(cfg, P.out / f"{P.word}.pt", P.out / f"{P.word}.onnx")


# --------------------------------------------------------------------------- eval


def _scores(sess, x: np.ndarray, bs: int = 4096) -> np.ndarray:
    name = sess.get_inputs()[0].name
    out = [sess.run(None, {name: x[i:i + bs].astype(np.float32)})[0].reshape(-1) for i in range(0, len(x), bs)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def gate_count(scores: np.ndarray, thr: float, consecutive: int = 2, cooldown: int = 19) -> int:
    """Mirror of openwhisprflow.wake.detector.WakeGate over a frame-score stream."""
    run, last, n = 0, -10 ** 9, 0
    for i, s in enumerate(scores):
        if s >= thr:
            run += 1
        else:
            run = 0
            continue
        if run >= consecutive and i - last >= cooldown:
            n += 1
            last = i
            run = 0
    return n


def stage_eval(raw: dict, P: Paths, model: Path | None = None) -> dict:
    import onnxruntime as ort

    model = model or (P.out / f"{P.word}.onnx")
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    f = P.feats
    pos = _scores(sess, np.load(f / "synth_positive_test.npy"))
    neg = _scores(sess, np.load(f / "synth_negative_test.npy"))
    bg = _scores(sess, np.load(f / "synth_background_test.npy"))
    real_pos = _scores(sess, _cat([f / "real_positive_test.npy"]))
    room = _scores(sess, _cat([f / "real_room_test.npy"]))
    pre = _scores(sess, _cat([f / "synth_positive_prefix_test.npy"]))  # "Hey/Okay/So <word>"
    pre_neg = _scores(sess, _cat([f / "synth_negative_prefix_test.npy"]))  # "hey, over on", ...
    # streaming over 10.7 h of held-out general audio (never used in training)
    v = np.load(P.val, mmap_mode="r")
    frames = np.lib.stride_tricks.sliding_window_view(np.asarray(v, np.float32), (16, 96))[:, 0]
    stream = np.concatenate([_scores(sess, frames[i:i + 50000]) for i in range(0, len(frames), 50000)])
    hours = len(v) * 0.08 / 3600
    # near-miss detail: per-text score (first 1000 test negatives)
    man = _manifest(P.clips / "negative_test")
    worst = sorted(zip(neg[: len(man)], [m["text"] for m in man], strict=False), reverse=True)[:15]

    rows = []
    for thr in [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]:
        rows.append({
            "threshold": thr,
            "recall_synth_heldout": float(np.mean(pos >= thr)),
            "recall_real": float(np.mean(real_pos >= thr)) if len(real_pos) else None,
            "recall_prefix_heldout": float(np.mean(pre >= thr)) if len(pre) else None,
            "prefix_near_miss_false_accept": float(np.mean(pre_neg >= thr)) if len(pre_neg) else None,
            "near_miss_false_accept": float(np.mean(neg >= thr)),
            "noise_false_accept": float(np.mean(bg >= thr)),
            "room_false_accept": float(np.mean(room >= thr)) if len(room) else None,
            "fpph_stream_gate": gate_count(stream, thr) / hours,
            "fpph_stream_single_frame": gate_count(stream, thr, consecutive=1) / hours,
        })
    # <= max_fpph (default 0.3) FP/h on the 10.7 h stream: the app's transcript check (wakeword.verify) drops the rest
    max_fpph = float(wcfg(raw).get("max_fpph", 0.3))
    ok = [r for r in rows if r["fpph_stream_gate"] <= max_fpph and r["near_miss_false_accept"] <= 0.05]

    def _rec(r: dict) -> float:  # plain and prefixed wake phrases count equally
        if r["recall_prefix_heldout"] is None:
            return r["recall_synth_heldout"]
        return (r["recall_synth_heldout"] + r["recall_prefix_heldout"]) / 2

    best = max(ok, key=lambda r: (_rec(r), -r["threshold"])) if ok else rows[-1]
    res = {"model": str(model), "validation_hours": round(hours, 2), "n_pos_test": int(len(pos)),
           "n_neg_test": int(len(neg)), "n_prefix_pos_test": int(len(pre)), "n_prefix_neg_test": int(len(pre_neg)), "n_bg_test": int(len(bg)), "n_real_pos_test": int(len(real_pos)),
           "recommended_threshold": best["threshold"], "at_recommended": best, "table": rows,
           "top_near_miss_scores": [[round(float(s), 3), t] for s, t in worst]}
    (P.out / f"{P.word}_eval.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    log(f"eval ({hours:.1f} h stream, {len(pos)} held-out positives, {len(neg)} held-out near-miss negatives)")
    log(f"{'thr':>5} {'recall':>7} {'prefix':>7} {'realrec':>7} {'nearFA':>7} {'preFA':>7} {'noiseFA':>7} "
        f"{'FPPH(gate)':>10} {'FPPH(1f)':>9}")
    for r in rows:
        rr = f"{r['recall_real']:.3f}" if r["recall_real"] is not None else "   -"
        pr = f"{r['recall_prefix_heldout']:.3f}" if r["recall_prefix_heldout"] is not None else "   -"
        pn = f"{r['prefix_near_miss_false_accept']:.3f}" if r["prefix_near_miss_false_accept"] is not None else "   -"
        log(f"{r['threshold']:5.2f} {r['recall_synth_heldout']:7.3f} {pr:>7} {rr:>7} {r['near_miss_false_accept']:7.3f} {pn:>7} "
            f"{r['noise_false_accept']:7.4f} {r['fpph_stream_gate']:10.2f} {r['fpph_stream_single_frame']:9.2f}")
    log(f"recommended threshold {best['threshold']}; top near-miss scores: {res['top_near_miss_scores'][:6]}")
    return res


def stage_install(raw: dict, P: Paths, res: dict, train_s: float | None = None) -> None:
    P.models.mkdir(parents=True, exist_ok=True)
    w = P.word
    o = wcfg(raw)
    dst = P.models / f"{w}.onnx"
    if dst.exists():  # keep the previous model next to the training output (not in the package)
        shutil.copyfile(dst, P.out / f"{w}.prev.onnx")
        if (P.models / f"{w}.json").exists():
            shutil.copyfile(P.models / f"{w}.json", P.out / f"{w}.prev.json")
    shutil.copyfile(P.out / f"{w}.onnx", dst)
    best = res["at_recommended"]
    n_real = len(list(P.real_pos.glob("*.wav"))) if P.real_pos.exists() else 0
    meta = {
        "name": w,
        "phrase": o.get("phrase", w.capitalize()),
        "pronunciation": o.get("pronunciation"),
        "positive_variants": o.get("positive_variants"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "onnx classifier head",
        "input": {"name": "embeddings", "shape": ["batch", 16, 96], "dtype": "float32"},
        "output": {"name": "score", "shape": ["batch", 1], "range": "sigmoid probability [0,1]"},
        "frontend": {
            "type": "openWakeWord melspectrogram.onnx + embedding_model.onnx (frozen)",
            "audio": "16 kHz mono, int16-scale samples (as openwhisprflow/wake/frontend.py feeds them)",
            "frame_samples": 1280, "embedding_frames": 16, "embedding_dim": 96,
            "mel_transform": "x/10 + 2",
        },
        "architecture": f"livekit-wakeword {raw['model']['model_type']} ({raw['model']['model_size']})",
        "recommended": {"threshold": res["recommended_threshold"], "consecutive_frames": 2, "cooldown_s": 1.5},
        "stats": {
            "heldout_synthetic_recall": round(best["recall_synth_heldout"], 4),
            "real_recording_recall": best["recall_real"],
            "near_miss_false_accept_rate": round(best["near_miss_false_accept"], 4),
            "fpph_validation_stream_gate": round(best["fpph_stream_gate"], 3),
            "validation_hours": res["validation_hours"],
            "n_heldout_positive": res["n_pos_test"], "n_heldout_near_miss": res["n_neg_test"],
            "real_positive_recordings": n_real,
            "heldout_prefix_recall": best.get("recall_prefix_heldout"),
            "prefix_near_miss_false_accept_rate": best.get("prefix_near_miss_false_accept"),
        },
        "training": {
            "tts": "Piper VITS en-us-libritts-high, 904 speakers, SLERP pairs; test split = held-out speakers",
            "negatives": f"near-miss + CMUdict phonetic neighbours + commands, MUSAN noise, ACAV100M features ({P.acav.name})",
            "prefixes": sorted((o.get("prefix") or {}).get("words", {})),
            "train_seconds": train_s,
            "pipeline": "training/wakeword/pipeline.py",
        },
        "threshold_table": res["table"],
    }
    (P.models / f"{w}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"installed {dst} (+ {w}.json, threshold {res['recommended_threshold']})")


# --------------------------------------------------------------------------- main


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "features", "train", "export", "eval", "install", "all", "retrain"])
    ap.add_argument("--word", default="transcribe", help="wake word; loads configs/<word>.yaml (default transcribe)")
    ap.add_argument("--config", default=None, help="explicit config path (overrides --word)")
    ap.add_argument("--data", default=None, help="word-independent data dir (default $OWF_WAKE_DATA or data/lkw)")
    ap.add_argument("--no-install", action="store_true", help="all/retrain: don't copy into wake/models")
    a = ap.parse_args()
    raw = load_cfg(Path(a.config) if a.config else HERE / "configs" / f"{a.word.lower()}.yaml")
    P = Paths(raw, data_root(a.data))
    random.seed(wcfg(raw)["seed"])
    np.random.seed(wcfg(raw)["seed"])
    t0 = time.time()
    if a.stage in ("generate", "all"):
        stage_generate(raw, P)
    if a.stage in ("features", "all"):
        stage_features(raw, P)
    if a.stage == "retrain":
        if not P.feats.joinpath("synth_positive_train.npy").exists():
            stage_generate(raw, P)
            stage_features(raw, P)
        else:
            stage_features(raw, P, only_real=True)
    train_s = None
    if a.stage in ("train", "all", "retrain"):
        t1 = time.time()
        stage_train(raw, P)
        train_s = round(time.time() - t1)
    if a.stage in ("export", "all", "retrain", "train"):
        stage_export(raw, P)
    res = None
    if a.stage in ("eval", "all", "retrain", "install"):
        res = stage_eval(raw, P)
    if a.stage == "install" or (a.stage in ("all", "retrain") and not a.no_install):
        stage_install(raw, P, res, train_s)
    log(f"{a.stage} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
