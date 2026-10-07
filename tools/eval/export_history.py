"""Export your own Ochre dictation history as an eval set (history eval). Run it yourself only.

Nothing here runs automatically and nothing is uploaded: the script reads the local history
database read-only and writes JSON Lines next to it (or where you say). See docs/history-eval.md.

    # 1. export (read-only; default DB: %LOCALAPPDATA%\\ochre\\data\\history.sqlite3 or $OCHRE_DATA_DIR)
    python tools/eval/export_history.py export --out my-voice/history.jsonl [--since 2026-10-01]
           [--min-words 3] [--max-rows 600] [--exclude-app "1Password" --exclude-app "KeePass"]
           [--exclude-regex "(?i)password|ssn"] [--redact]
    # 2. review my-voice/review.md, list ids to remove (one per line) in my-voice/remove.txt, then
    python tools/eval/export_history.py strip --in my-voice/history.jsonl --remove my-voice/remove.txt
           --out my-voice/eval.jsonl
    # 3. (optional) label: my-voice/eval.jsonl rows have "clean": "" until labeled (see the doc)

Row format = the eval format used by training/refine/eval_gguf.py and tools/eval/build_judge_v4.py:
    {"id": "own-h-000123", "source": "owner-history", "mode": "clean", "style": "", "dictionary": [...],
     "context": "slack", "raw": <model input as the app saw it>, "clean": "", "shipped_output": <what was typed>,
     "refiner": "...", "stt": "...", "app": "...", "created": "2026-10-05T12:34:56", "tags": []}

`raw` in the history table is already the model input (recognizer text after the app's
strip_hesitations + dictionary corrections, crates/ochre/src/app.rs), so it is used as-is.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

# crates/ochre-core/src/paths.rs: ProjectDirs("", "", "ochre").data_local_dir(), or $OCHRE_DATA_DIR
def default_db() -> Path:
    if os.environ.get("OCHRE_DATA_DIR"):
        return Path(os.environ["OCHRE_DATA_DIR"]) / "history.sqlite3"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "ochre" / "data" / "history.sqlite3"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ochre" / "history.sqlite3"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "ochre" / "history.sqlite3"


def default_config() -> Path:
    if os.environ.get("OCHRE_CONFIG_DIR"):
        return Path(os.environ["OCHRE_CONFIG_DIR"]) / "config.toml"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "ochre" / "config" / "config.toml"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ochre" / "config.toml"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "ochre" / "config.toml"


# Focused app -> GUIDE context (best effort; the app name is kept on the row too).
APP_CONTEXT = [
    (r"slack|discord|teams|mattermost", "slack"), (r"outlook|thunderbird|mail|gmail|superhuman", "email"),
    (r"chatgpt|claude|gemini|copilot|perplexity", "ai_chat"),
    (r"windowsterminal|terminal|powershell|cmd\.exe|iterm|wezterm|alacritty|kitty|warp", "terminal"),
    (r"code|cursor|pycharm|idea|rider|webstorm|zed|sublime|vim|nvim|visual studio", "code_comment"),
    (r"word|docs|notion|obsidian|onenote|pages|writer", "docs"), (r"whatsapp|messages|signal|telegram", "sms"),
    (r"jira|linear|github|gitlab", "issue_tracker"), (r"chrome|firefox|edge|safari|brave|arc", "notes"),
]


def context_for(app: str) -> str:
    a = (app or "").lower()
    for pat, ctx in APP_CONTEXT:
        if re.search(pat, a):
            return ctx
    return "notes"


def dictionary_from_config(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        import tomllib
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    d = cfg.get("dictionary")
    if isinstance(d, dict):
        d = d.get("words") or d.get("entries") or []
    return [str(x) for x in d] if isinstance(d, list) else []


REDACT = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
          (re.compile(r"\+?\d[\d ()-]{7,}\d"), "<phone>"),
          (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<card>")]


def redact(s: str) -> str:
    for rx, rep in REDACT:
        s = rx.sub(rep, s)
    return s


def open_ro(db: Path) -> tuple[sqlite3.Connection, Path | None]:
    """Read-only. With WAL the DB may need its -shm/-wal: copy all three to a temp dir and read the copy,
    so the live database is never touched (even if the app is running)."""
    tmp = Path(tempfile.mkdtemp(prefix="ochre-hist-"))
    for suf in ("", "-wal", "-shm"):
        src = Path(str(db) + suf)
        if src.exists():
            shutil.copy2(src, tmp / (db.name + suf))
    con = sqlite3.connect(f"file:{(tmp / db.name).as_posix()}?mode=ro", uri=True)
    return con, tmp


def cmd_export(a: argparse.Namespace) -> None:
    db = a.db or default_db()
    if not db.exists():
        raise SystemExit(f"no history database at {db} (pass --db)")
    con, tmp = open_ro(db)
    try:
        q = "SELECT id, created, raw, text, app, stt, refiner, duration_ms, inserted FROM dictations"
        args: list = []
        if a.since:
            q += " WHERE created >= ?"
            args.append(dt.datetime.fromisoformat(a.since).timestamp())
        q += " ORDER BY created"
        rows = con.execute(q, args).fetchall()
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)
    dictionary = dictionary_from_config(a.config or default_config())
    ex_app = [x.lower() for x in a.exclude_app or []]
    ex_rx = [re.compile(x) for x in a.exclude_regex or []]
    out, skipped = [], {"short": 0, "app": 0, "regex": 0, "unrefined": 0, "dupe": 0}
    seen = set()
    for (rid, created, raw, text, app, stt, refiner, dur, inserted) in rows:
        raw = (raw or "").strip()
        if len(raw.split()) < a.min_words:
            skipped["short"] += 1
            continue
        if any(x in (app or "").lower() for x in ex_app):
            skipped["app"] += 1
            continue
        if any(rx.search(raw) or rx.search(text or "") for rx in ex_rx):
            skipped["regex"] += 1
            continue
        if a.only_refined and not refiner:
            skipped["unrefined"] += 1
            continue
        if raw.lower() in seen:
            skipped["dupe"] += 1
            continue
        seen.add(raw.lower())
        r = {"id": f"own-h-{rid:06d}", "source": "owner-history", "mode": "clean", "style": "",
             "dictionary": dictionary, "context": context_for(app), "raw": redact(raw) if a.redact else raw,
             "clean": "", "shipped_output": redact(text or "") if a.redact else (text or ""),
             "refiner": refiner or "", "stt": stt or "", "app": app or "",
             "created": dt.datetime.fromtimestamp(created).isoformat(timespec="seconds"),
             "duration_ms": dur, "inserted": bool(inserted), "tags": []}
        if r["context"] == "terminal":
            r["style"] = "literal"
        out.append(r)
    if a.max_rows and len(out) > a.max_rows:   # keep a spread over time, not just the newest
        step = len(out) / a.max_rows
        out = [out[int(i * step)] for i in range(a.max_rows)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    review = a.out.parent / "review.md"
    L = ["# History export: review before anything leaves this machine", "",
         "Put the id of every row you do not want used (private, wrong, or not a dictation) in remove.txt,",
         "one per line, then run the `strip` step. Nothing has been uploaded.", ""]
    for r in out:
        L.append(f"- `{r['id']}` [{r['app'] or '?'} -> {r['context']}] {r['raw'][:300]}")
    review.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(out)} rows -> {a.out} (of {len(rows)} in the DB; skipped {skipped}); review list: {review}")


def cmd_strip(a: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in a.inp.read_text(encoding="utf-8").splitlines() if x.strip()]
    remove = set()
    if a.remove and a.remove.exists():
        remove = {x.strip() for x in a.remove.read_text(encoding="utf-8").splitlines() if x.strip()}
    keep = [r for r in rows if r["id"] not in remove]
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    print(f"{len(keep)} rows kept, {len(rows) - len(keep)} removed -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--db", type=Path)
    e.add_argument("--config", type=Path, help="config.toml for the dictionary (default: the app's)")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--since", help="ISO date/time")
    e.add_argument("--min-words", type=int, default=3)
    e.add_argument("--max-rows", type=int, default=600)
    e.add_argument("--only-refined", action="store_true", help="only dictations a refiner processed")
    e.add_argument("--exclude-app", action="append")
    e.add_argument("--exclude-regex", action="append")
    e.add_argument("--redact", action="store_true", help="mask emails, phone and card numbers")
    s = sub.add_parser("strip")
    s.add_argument("--in", dest="inp", type=Path, required=True)
    s.add_argument("--remove", type=Path)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"export": cmd_export, "strip": cmd_strip}[a.cmd](a)


if __name__ == "__main__":
    main()
