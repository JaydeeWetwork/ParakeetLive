"""Log 55: the WSL server's request checks, against the REAL live_server.py code with the model stubbed
(tests/server_stub_runner.py: no NeMo, no CUDA, 0 VRAM). On Windows the stub runs inside WSL from
/tmp/plive-test, started through core.ServerProcess (so the WSLENV hand-over of the token is what's tested),
on spare ports, and is stopped only via its own /shutdown or its own pid - never with a pkill pattern that
could match Jaydee's running server. Elsewhere it runs as a local subprocess.
  A1 token required: none / wrong -> 401, right -> 200
  A2 browser requests (Origin header) -> 403 even with the right token; the server keeps running
  A3 foreign Host header (DNS rebinding) -> 403
  A4 bad Content-Length: negative / non-numeric -> 400, huge -> 413 at once (no read, no hang)
  A5 core.Client with the token: 3 transcriptions over one keep-alive connection
  A6 core.Client without the token: transcribe raises "unauthorized"
  A7 /shutdown without the token -> 401 and still alive; with it -> exits
  A8 the token is not on the server's command line
  A9 no PLIVE_TOKEN (old widget + new server): works without a token, browser requests still 403"""
import json
import os
import secrets
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "widget"))
if os.name != "nt":
    import ctypes
    from unittest import mock
    ctypes.WinDLL = mock.MagicMock()
    ctypes.WINFUNCTYPE = mock.MagicMock()
import plive_core as core  # noqa: E402

PORTS = (51790, 51791)
RESULTS = []
TMPDIR = "/tmp/plive-test"


def check(name, ok, **info):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(info, default=str) if info else ''}", flush=True)


def to_wsl(p):
    p = os.path.abspath(p)
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


def start(port, token):
    log = os.path.join(os.environ.get("TEMP", "/tmp"), f"plive-auth-{port}.log")
    if os.name == "nt":
        core._wsl(["bash", "-c", f"mkdir -p {TMPDIR} && cp '{to_wsl(os.path.join(REPO, 'server', 'live_server.py'))}' "
                                 f"'{to_wsl(os.path.join(REPO, 'tests', 'server_stub_runner.py'))}' {TMPDIR}/"])
        orig = core.SERVER_PY
        core.SERVER_PY = f"{TMPDIR}/server_stub_runner.py"
        try:
            sp = core.ServerProcess(port, log, idle_exit=60, mode="gpu", token=token)
            sp.start()
        finally:
            core.SERVER_PY = orig
        return sp
    env = dict(os.environ)
    env.pop("PLIVE_TOKEN", None)
    if token:
        env["PLIVE_TOKEN"] = token
    sp = core.ServerProcess(port, log, token=None)
    sp.logf = open(log, "a")
    sp.proc = subprocess.Popen([sys.executable, os.path.join(REPO, "tests", "server_stub_runner.py"), "--port",
                                str(port), "--idle-exit", "60", "--gpu"], env=env, stdout=sp.logf,
                               stderr=subprocess.STDOUT)
    return sp


def raw(port, head, body=b"", read=True):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        s.sendall(head.encode() + body)
        if not read:
            return None
        data = b""
        t0 = time.time()
        while b"\r\n\r\n" not in data and time.time() - t0 < 5:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        return int(data.split(b" ", 2)[1]) if data.startswith(b"HTTP/") else None
    finally:
        s.close()


def req(port, method="GET", path="/health", token=None, extra="", body=b"", clen=None):
    h = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n"
    if token is not None:
        h += f"X-Plive-Token: {token}\r\n"
    if method == "POST":
        h += f"Content-Length: {len(body) if clen is None else clen}\r\n"
    return raw(port, h + extra + "\r\n", body)


def wait_up(port, token, t=60):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            if req(port, token=token) == 200:
                return True
        except OSError:
            pass
        time.sleep(0.3)
    return False


def alive(port, token):
    try:
        return req(port, token=token) == 200
    except OSError:
        return False


def gone(sp, port, t=10):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
        except OSError:
            if sp.proc.poll() is not None or os.name != "nt":
                return True
        if sp.proc.poll() is not None:
            return True
        time.sleep(0.3)
    return False


def finish(sp, port, token):
    try:
        core.Client(port, token=token).shutdown()
    except Exception:
        pass
    try:
        sp.proc.wait(10)
    except Exception:
        sp.proc.kill()
    if sp.logf:
        sp.logf.close()


T = secrets.token_hex(16)
port = PORTS[0]
for p in PORTS:
    if not core.port_free(p):
        sys.exit(f"port {p} busy - pick other test ports")
sp = start(port, T)
try:
    up = wait_up(port, T)
    check("A0_stub_server_up", up)
    if up:
        a1 = (req(port), req(port, token="0" * 32), req(port, token=T))
        check("A1_token_required", a1 == (401, 401, 200), got=a1)
        a2 = (req(port, "POST", "/shutdown", token=T, extra="Origin: http://evil.example\r\n"),
              req(port, "POST", "/activate", token=T, extra="Origin: null\r\n"),
              req(port, token=T, extra="Origin: http://127.0.0.1\r\n"))
        check("A2_browser_origin_refused", a2 == (403, 403, 403) and alive(port, T), got=a2)
        a3 = raw(port, f"GET /health HTTP/1.1\r\nHost: evil.example:{port}\r\nX-Plive-Token: {T}\r\n"
                       "Connection: close\r\n\r\n")
        a3b = raw(port, f"GET /health HTTP/1.1\r\nHost: localhost:{port}\r\nX-Plive-Token: {T}\r\n"
                        "Connection: close\r\n\r\n")
        check("A3_foreign_host_refused", a3 == 403 and a3b == 200, foreign=a3, localhost=a3b)
        t0 = time.perf_counter()
        a4 = (req(port, "POST", "/transcribe", token=T, clen=-5), req(port, "POST", "/transcribe", token=T, clen="abc"),
              req(port, "POST", "/transcribe", token=T, clen=10 ** 10))
        dt = time.perf_counter() - t0
        check("A4_content_length_checked", a4 == (400, 400, 413) and dt < 3 and alive(port, T), got=a4,
              seconds=round(dt, 2))
        c = core.Client(port, token=T)
        outs = [c.transcribe(bytes(32000))["text"] for _ in range(3)]
        sock_reused = c.conn is not None
        check("A5_client_with_token_keepalive", outs == ["stub 16000 samples"] * 3 and sock_reused, texts=outs)
        c.close()
        try:
            core.Client(port).transcribe(bytes(3200))
            a6 = "no error"
        except RuntimeError as e:
            a6 = str(e)
        check("A6_client_without_token_refused", a6 == "unauthorized", got=a6)
        if os.name == "nt":
            cmd = core._wsl(["bash", "-c", "pgrep -af '[s]erver_stub_runner' || true"]).stdout.decode("utf-8", "replace")
        else:
            cmd = " ".join(open(f"/proc/{sp.proc.pid}/cmdline").read().split("\0"))
        check("A8_token_not_on_command_line", "server_stub_runner" in cmd and T not in cmd)
        a7 = req(port, "POST", "/shutdown")
        still = alive(port, T)
        a7b = req(port, "POST", "/shutdown", token=T)
        check("A7_shutdown_needs_token", a7 == 401 and still and a7b == 200 and gone(sp, port),
              without=a7, alive_after=still, with_token=a7b)
finally:
    finish(sp, port, T)
port = PORTS[1]
sp = start(port, None)
try:
    up = wait_up(port, None)
    a9 = (req(port) if up else None, req(port, "POST", "/shutdown", extra="Origin: http://evil.example\r\n") if up else None)
    still = alive(port, None)
    check("A9_no_token_backward_compatible", a9 == (200, 403) and still, got=a9)
finally:
    finish(sp, port, None)
if os.name == "nt":
    left = core._wsl(["bash", "-c", "pgrep -f '[s]erver_stub_runner' || true"]).stdout.strip()
    check("A10_no_stub_left_running", not left)
bad = [n for n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
