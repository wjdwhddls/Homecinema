"""save_server.py — MUSHRA 결과 저장 서버.

기존 `python -m http.server`와 동등한 정적 파일 서빙에 더해,
`POST /save_results` 엔드포인트로 평가 JSON을 `results/` 디렉토리에 저장.

사용:
    cd evaluation/webmushra
    python save_server.py               # 기본 포트 8765
    python save_server.py --port 8000

기능:
  - 정적 파일 서빙 (html/js/css/wav 모두 MIME 자동 설정)
  - POST /save_results  → results/<timestamp>_<type>_<evaluator>.json 저장
  - GET  /health         → {"ok": true} (서버 생존 체크)
  - CORS 허용 (같은 LAN 다른 기기에서 접근 가능)
  - Ctrl+C 그레이스풀 종료

저장 파일명 패턴:
  <ISO8601 timestamp>_<page_type>_<evaluator_id>_<short_uuid>.json
  예: 2026-04-18T23-01-15_full_trailer_yun_a1b2c3.json

JSON 저장 성공 시 응답:
  {"ok": true, "path": "results/...", "bytes": 1234}

실패 시:
  HTTP 400/500 + {"ok": false, "error": "..."}
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import uuid
from datetime import datetime
from pathlib import Path


SERVER_DIR   = Path(__file__).resolve().parent
RESULTS_DIR  = SERVER_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_filename_component(s: str, default: str = "anon") -> str:
    """파일명에 안전한 문자만 허용. 비어있거나 위험문자만 있으면 default."""
    s = str(s or "").strip()
    if not s:
        return default
    cleaned = SAFE_CHARS.sub("_", s)
    cleaned = cleaned.strip("._")[:32]
    return cleaned or default


class MushraSaveHandler(http.server.SimpleHTTPRequestHandler):
    """static serving + POST /save_results + GET /health + CORS headers."""

    # 서브디렉토리 컨텍스트 고정: SERVER_DIR를 root로 서빙
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVER_DIR), **kwargs)

    # ── CORS helpers ─────────────────────────────────────
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")

    def end_headers(self):
        # SimpleHTTPRequestHandler의 GET도 CORS 헤더 받도록
        self._send_cors_headers()
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers_raw()

    def end_headers_raw(self):
        # end_headers 오버라이드 우회용 (preflight response body 없을 때)
        http.server.BaseHTTPRequestHandler.end_headers(self)

    # ── POST /save_results ──────────────────────────────
    def do_POST(self):
        if self.path in ("/save_results", "/save_results/"):
            self._handle_save()
            return
        self._json_response(404, {"ok": False, "error": "unknown endpoint"})

    def _handle_save(self):
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 10_000_000:  # 10MB 상한 (대부분의 결과 ~10KB)
            self._json_response(400, {"ok": False, "error": f"bad content-length: {length}"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON: {e}"})
            return

        # 메타 추출 — 클라이언트에서 page_type / evaluator 필드 보내길 권장
        metadata = payload.get("metadata") or {}
        page_type = payload.get("page_type") or metadata.get("mode", "unknown")
        page_type = safe_filename_component(page_type, "unknown")
        participant = payload.get("participant") or {}
        evaluator   = (
            payload.get("evaluator")
            or participant.get("name")
            or metadata.get("evaluator_id")
            or "anon"
        )
        evaluator   = safe_filename_component(evaluator, "anon")

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        short_uuid = uuid.uuid4().hex[:6]
        fname = f"{ts}_{page_type}_{evaluator}_{short_uuid}.json"
        fpath = RESULTS_DIR / fname

        try:
            fpath.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            self._json_response(500, {"ok": False, "error": f"write failed: {e}"})
            return

        rel = fpath.relative_to(SERVER_DIR).as_posix()
        self._json_response(200, {
            "ok": True,
            "path": rel,
            "filename": fname,
            "bytes": len(raw),
            "timestamp": ts,
        })
        self._log_save(fname, len(raw), evaluator, page_type)

    # ── GET /health ─────────────────────────────────────
    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"ok": True, "service": "mushra-save-server"})
            return
        # fallback → static file serving
        return super().do_GET()

    # ── helpers ────────────────────────────────────────
    def _json_response(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_save(self, fname: str, size: int, evaluator: str, page_type: str):
        sys.stderr.write(
            f"[SAVE] {fname} ({size} B)  evaluator={evaluator}  page_type={page_type}\n"
        )
        sys.stderr.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="bind address (기본 0.0.0.0 — LAN 접근 허용)")
    args = ap.parse_args()

    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.ThreadingTCPServer(
        (args.bind, args.port), MushraSaveHandler
    ) as httpd:
        print(f"MUSHRA save server on http://{args.bind}:{args.port}")
        print(f"  static root: {SERVER_DIR}")
        print(f"  results dir: {RESULTS_DIR}")
        print(f"  endpoints:")
        print(f"    GET  /              → static files")
        print(f"    POST /save_results  → save JSON to results/<timestamp>_<type>_<evaluator>.json")
        print(f"    GET  /health        → {{ok: true}}")
        print(f"")
        print(f"Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
            httpd.server_close()


if __name__ == "__main__":
    main()
