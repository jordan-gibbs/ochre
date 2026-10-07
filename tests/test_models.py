"""models.ensure_file against a local HTTP server with Range support."""

from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path

import pytest

from openwhisprflow.models import DownloadCancelled, DownloadError, ensure_file, hf_url

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Handler(http.server.BaseHTTPRequestHandler):
    honor_range = True
    hits: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        rng = self.headers.get("Range")
        type(self).hits.append(rng)
        if self.path == "/missing":
            self.send_error(404)
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/file")
            self.end_headers()
            return
        start = 0
        if rng and self.honor_range:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server():
    Handler.hits = []
    Handler.honor_range = True
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_full_download_with_hash_and_progress(server: str, tmp_path: Path) -> None:
    seen = []
    dest = ensure_file(f"{server}/file", tmp_path / "m.bin", sha256=SHA, size=len(PAYLOAD),
                       progress=lambda n, d, t: seen.append((n, d, t)))
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "m.bin.partial").exists()
    assert seen[-1] == ("m.bin", len(PAYLOAD), len(PAYLOAD))


def test_existing_complete_file_skips_network(server: str, tmp_path: Path) -> None:
    dest = tmp_path / "m.bin"
    dest.write_bytes(PAYLOAD)
    ensure_file(f"{server}/file", dest, sha256=SHA, size=len(PAYLOAD))
    assert Handler.hits == []


def test_resume_uses_range(server: str, tmp_path: Path) -> None:
    (tmp_path / "m.bin.partial").write_bytes(PAYLOAD[:300_000])
    ensure_file(f"{server}/file", tmp_path / "m.bin", sha256=SHA, size=len(PAYLOAD))
    assert Handler.hits == ["bytes=300000-"]
    assert (tmp_path / "m.bin").read_bytes() == PAYLOAD


def test_complete_partial_is_promoted_without_network(server: str, tmp_path: Path) -> None:
    (tmp_path / "m.bin.partial").write_bytes(PAYLOAD)
    ensure_file(f"{server}/file", tmp_path / "m.bin", sha256=SHA, size=len(PAYLOAD))
    assert Handler.hits == [] and (tmp_path / "m.bin").read_bytes() == PAYLOAD


def test_server_ignoring_range_restarts(server: str, tmp_path: Path) -> None:
    Handler.honor_range = False
    (tmp_path / "m.bin.partial").write_bytes(b"x" * 1000)
    ensure_file(f"{server}/file", tmp_path / "m.bin", sha256=SHA)
    assert (tmp_path / "m.bin").read_bytes() == PAYLOAD


def test_follows_redirects(server: str, tmp_path: Path) -> None:
    ensure_file(f"{server}/redirect", tmp_path / "m.bin", size=len(PAYLOAD))
    assert (tmp_path / "m.bin").stat().st_size == len(PAYLOAD)


def test_checksum_mismatch_never_lands(server: str, tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="checksum"):
        ensure_file(f"{server}/file", tmp_path / "m.bin", sha256="0" * 64)
    assert not (tmp_path / "m.bin").exists()
    assert not (tmp_path / "m.bin.partial").exists()


def test_cancel_keeps_nothing_at_final_path(server: str, tmp_path: Path) -> None:
    with pytest.raises(DownloadCancelled):
        ensure_file(f"{server}/file", tmp_path / "m.bin", cancel=lambda: True)
    assert not (tmp_path / "m.bin").exists()


def test_http_error(server: str, tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="404"):
        ensure_file(f"{server}/missing", tmp_path / "m.bin")


def test_oversize_download_rejected(server: str, tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        ensure_file(f"{server}/file", tmp_path / "m.bin", size=len(PAYLOAD) - 1)
    assert not (tmp_path / "m.bin").exists()


def test_hf_url() -> None:
    assert hf_url("org/repo", "dir/a b.onnx", "abc") == "https://huggingface.co/org/repo/resolve/abc/dir/a%20b.onnx"
