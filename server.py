#!/usr/bin/env python3
"""Hyperion local HTTP server. Requires Python 3.11+. Run: python server.py --port 8000"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from core import HyperionEngine

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
ENGINE = HyperionEngine()
ENGINE_LOCK = threading.RLock()
MAX_BODY = 524_288


class Handler(BaseHTTPRequestHandler):
    server_version = "HyperionLocal/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json({
                "online": True, "name": "HYPERION", "version": ENGINE.VERSION,
                "mode": "100% LOCAL · XL", "documents": len(ENGINE.documents),
                "learned": ENGINE.learned_count, "network": "disabled", "engine": "neuro-symbolic-xl",
            })
            return
        self._static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/api/learn", "/api/reset"}:
            self._json({"error": "Ruta no encontrada."}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_BODY or (length == 0 and path != "/api/reset"):
                raise ValueError("Tamaño de solicitud no válido.")
            payload = json.loads(self.rfile.read(length)) if length else {}
            with ENGINE_LOCK:
                if path == "/api/chat":
                    message = payload.get("message")
                    style = payload.get("style", "balanced")
                    if not isinstance(message, str) or style not in {"concise", "balanced", "technical"}:
                        raise ValueError("Parámetros no válidos.")
                    self._json(ENGINE.respond(message, style))
                elif path == "/api/learn":
                    title, content = payload.get("title"), payload.get("content")
                    if not isinstance(title, str) or not isinstance(content, str):
                        raise ValueError("Documento no válido.")
                    chunks = ENGINE.learn(title, content)
                    self._json({"learned": True, "chunks": chunks, "documents": len(ENGINE.documents)})
                else:
                    ENGINE.reset_learned()
                    self._json({"reset": True, "documents": len(ENGINE.documents)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self._json({"error": "Fallo interno controlado."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else unquote(request_path.lstrip("/"))
        candidate = (PUBLIC / relative).resolve()
        if PUBLIC.resolve() not in candidate.parents and candidate != PUBLIC.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            candidate = PUBLIC / "index.html"
        content = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[local] {self.address_string()} — {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor local de Hyperion")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"HYPERION LOCAL activo en http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
