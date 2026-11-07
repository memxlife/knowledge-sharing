"""HTTP(S) between nodes. Spec section 7.

Plain JSON over HTTP. Run with --tls-cert and --tls-key to serve HTTPS.
A real peer-to-peer library can replace this file later; the data formats
do not depend on it.

Endpoints
  GET  /info                     who this node is
  POST /sync      {have}         pull: entries newer than what the caller has,
                                 plus this node's own summary so the caller can
                                 push back what this node lacks
  POST /entries   {entries}      push: new entries from a peer
  GET  /peers                    known peer announcements
  POST /peers     {announcement} tell this node where you are
  POST /inbox     {message}      a message addressed to this node
  POST /mailbox   {message}      relays only: hold a message for an offline node
  GET  /mailbox?to=ID&after=N    relays only: collect held messages
"""

from __future__ import annotations

import json
import ssl
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MAX_BODY = 20 * 1024 * 1024


class NetError(Exception):
    pass


def _opener(url: str):
    host = urlparse(url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def call(base_url: str, path: str, payload=None, timeout: float = 5.0):
    """GET when payload is None, else POST JSON. Raises NetError on any failure."""
    url = base_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="GET" if data is None else "POST", headers={"Content-Type": "application/json"}
    )
    try:
        with _opener(base_url).open(req, timeout=timeout) as resp:
            raw = resp.read(MAX_BODY + 1)
    except urllib.error.HTTPError as e:
        detail = e.read(2000).decode("utf-8", "replace")
        raise NetError(f"{url}: HTTP {e.code} {detail}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise NetError(f"{url}: {e}") from None
    try:
        return json.loads(raw or b"null")
    except json.JSONDecodeError:
        raise NetError(f"{url}: reply is not JSON") from None


class _Handler(BaseHTTPRequestHandler):
    server_version = "rvn/0.1"
    node = None  # set per server
    relay = False

    def log_message(self, fmt, *args):  # keep test output quiet
        pass

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/info":
                return self._send(200, self.node.info(relay=self.relay))
            if url.path == "/peers":
                return self._send(200, {"peers": self.node.peer_announcements()})
            if url.path == "/mailbox" and self.relay:
                return self._send(200, self.node.relay_fetch(q.get("to", ""), int(q.get("after", 0))))
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001 - report, never crash the server
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            if path == "/sync":
                ledger = self.node.ledger
                return self._send(200, {"entries": ledger.entries_since(body.get("have") or {}),
                                        "have": ledger.summary()})
            if path == "/entries":
                entries = body.get("entries")
                if not isinstance(entries, list):
                    raise ValueError("entries must be a list")
                res = self.node.ledger.ingest(entries)
                return self._send(200, {k: v for k, v in res.items() if k != "errors"})
            if path == "/peers":
                return self._send(200, {"ok": self.node.add_peer(body.get("announcement"))})
            if path == "/inbox":
                return self._send(200, self.node.receive_message(body.get("message")))
            if path == "/mailbox" and self.relay:
                return self._send(200, self.node.relay_store(body.get("message")))
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": str(e)})


class Server:
    def __init__(self, node, host: str = "127.0.0.1", port: int = 0, relay: bool = False,
                 tls_cert: str | None = None, tls_key: str | None = None):
        handler = type("Handler", (_Handler,), {"node": node, "relay": relay})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        scheme = "http"
        if tls_cert:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(tls_cert, tls_key)
            self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
            scheme = "https"
        h, p = self.httpd.server_address[:2]
        self.url = f"{scheme}://{'127.0.0.1' if h in ('0.0.0.0', '') else h}:{p}"
        self._thread: threading.Thread | None = None

    def start(self) -> "Server":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
