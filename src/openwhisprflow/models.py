"""Model and binary downloads: resumable, atomic, optionally SHA-256 verified.

The invariant: a partially downloaded
file never sits at its final path, so a loader never sees a truncated model. Bytes stream
into ``<dest>.partial``; an interrupted download resumes from there with an HTTP Range request
and the finished file is renamed into place only after the size/hash checks pass.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import httpx

log = logging.getLogger("openwhisprflow.models")

CHUNK = 1024 * 1024
USER_AGENT = "openwhisprflow/0.1 (+https://github.com/jordan-gibbs/openwhisprflow)"

Progress = Callable[[str, int, int], None]   # (item name, bytes done, bytes total; 0 = unknown)
Cancel = Callable[[], bool]


class DownloadCancelled(Exception):
    """Raised when ``cancel()`` returns True mid-download. The partial file is kept for resume."""


class DownloadError(RuntimeError):
    """Size or checksum mismatch, or an HTTP failure the caller should surface to the user."""


def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    """Direct-download URL for a file in a Hugging Face model repo."""
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote(filename)}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _complete(path: Path, sha256: str | None, size: int | None) -> bool:
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    if sha256 is not None and sha256_file(path) != sha256.lower():
        return False
    return True


def ensure_file(url: str, dest: Path, *, sha256: str | None = None, size: int | None = None,
                progress: Progress | None = None, cancel: Cancel | None = None,
                timeout: float = 45.0) -> Path:
    """Make sure ``dest`` exists and is complete; download it from ``url`` if not.

    Without ``sha256``/``size`` an existing ``dest`` is trusted as-is (it only ever got there
    through the atomic rename below). Returns ``dest``.
    """
    dest = Path(dest)
    name = dest.name
    if _complete(dest, sha256, size):
        if progress:
            n = dest.stat().st_size
            progress(name, n, n)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if size is not None and offset >= size:
        # Either complete-but-unrenamed or garbage; verify rather than refetch blindly.
        if offset == size and (sha256 is None or sha256_file(partial) == sha256.lower()):
            partial.replace(dest)
            return dest
        partial.unlink()
        offset = 0

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 416 and offset:
                    # Server says our partial already covers the file; restart cleanly next time.
                    partial.unlink(missing_ok=True)
                    raise DownloadError(f"{name}: resume rejected (416); retry the download")
                if resp.status_code not in (200, 206):
                    raise DownloadError(f"{name}: HTTP {resp.status_code} from {resp.url.host}")
                if resp.status_code == 206:
                    if not resp.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                        raise DownloadError(f"{name}: invalid Content-Range on resume")
                else:
                    offset = 0  # server ignored Range: start over
                length = int(resp.headers.get("content-length") or 0)
                total = size or (offset + length if length else 0)
                received = offset
                with partial.open("ab" if offset else "wb") as f:
                    for block in resp.iter_bytes(CHUNK):
                        if cancel and cancel():
                            raise DownloadCancelled(name)
                        received += len(block)
                        if size is not None and received > size:
                            raise DownloadError(f"{name}: download exceeds expected size")
                        f.write(block)
                        if progress:
                            progress(name, received, total)
                    f.flush()
                    os.fsync(f.fileno())
    except httpx.HTTPError as e:
        raise DownloadError(f"{name}: {type(e).__name__}: {e}") from e

    got = partial.stat().st_size
    if size is None and total and got != total:
        raise DownloadError(f"{name}: incomplete download ({got} of {total} bytes); retry to resume")
    if size is not None and got != size:
        raise DownloadError(f"{name}: incomplete download ({got} of {size} bytes); retry to resume")
    if sha256 is not None and sha256_file(partial) != sha256.lower():
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{name}: checksum mismatch; retry the download")
    partial.replace(dest)
    log.info("downloaded %s (%d bytes)", name, got)
    return dest
