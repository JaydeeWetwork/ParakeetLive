"""Parakeet Live - non-GUI core: WSL server lifecycle, keep-alive client, mic capture,
streaming resampler, energy VAD, Win32 clipboard and global hotkey.

Nothing here touches Tk; the GUI (parakeet_live.pyw) talks to these pieces through queues,
so audio capture and network work never run on the UI thread.
"""
import ctypes
import ctypes.wintypes as wt
import http.client
import io
import itertools
import json
import os
import queue
import socket
import subprocess
import threading
import time
import wave
from collections import deque

import numpy as np

SR = 16000
FRAME = 320                      # 20 ms VAD frames at 16 kHz
DISTRO = "Ubuntu-24.04"
SERVER_PY = "/opt/parakeet/scripts/live_server.py"
VENV_PY = "/opt/parakeet/venv/bin/python"
CREATE_NO_WINDOW = 0x08000000


# --------------------------------------------------------------------------- server
class ServerError(RuntimeError):
    """The server answered with an HTTP error; .status tells a refusal (4xx) from a failure (5xx)."""

    def __init__(self, msg, status):
        super().__init__(msg)
        self.status = status


def retryable(exc):
    """Worth another try later: connection problems and server-side failures, not refusals."""
    st = getattr(exc, "status", None)
    return st is None or st >= 500


class Client:
    """One persistent HTTP/1.1 keep-alive connection to the WSL server (TCP_NODELAY)."""

    def __init__(self, port, host="127.0.0.1", token=None):
        self.host, self.port = host, port
        self.token = token            # the server's shared secret (X-Plive-Token), if it has one
        self.conn = None
        self.lock = threading.Lock()

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = None

    def _request(self, method, path, body=None, timeout=30.0):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.conn is None:
                        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
                        c.connect()
                        c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        self.conn = c
                    elif self.conn.sock is not None:
                        self.conn.sock.settimeout(timeout)
                    hdrs = {"Content-Type": "application/octet-stream"} if body is not None else {}
                    if self.token:
                        hdrs["X-Plive-Token"] = self.token
                    self.conn.request(method, path, body=body, headers=hdrs)
                    r = self.conn.getresponse()
                    data = r.read()
                    try:
                        obj = json.loads(data or b"{}")
                    except ValueError:
                        obj = {"error": data[:200].decode("utf-8", "replace")}
                    return r.status, obj
                except (http.client.HTTPException, OSError) as e:
                    self.close()          # stale keep-alive socket: reconnect once
                    if attempt or isinstance(e, TimeoutError):
                        raise
            raise OSError("unreachable")

    def health(self, timeout=2.0):
        return self._request("GET", "/health", timeout=timeout)[1]

    def transcribe(self, pcm16: bytes, timeout=60.0):
        status, obj = self._request("POST", "/transcribe", body=pcm16, timeout=timeout)
        if status != 200:
            raise ServerError(obj.get("error") or f"HTTP {status}", status)
        return obj

    def activate(self, timeout=5.0):
        """Ask a standby (RAM) server to move to the GPU; returns its status."""
        return self._request("POST", "/activate", body=b"", timeout=timeout)[1]

    def shutdown(self, timeout=2.0):
        try:
            return self._request("POST", "/shutdown", body=b"", timeout=timeout)
        except Exception:
            return None


def _wsl(args, timeout=20):
    return subprocess.run(["wsl.exe", "-d", DISTRO, "-u", "root", "-e"] + args,
                          capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW)


def kill_stale_servers():
    """Only one widget can run (mutex), so any live_server.py already in WSL is a leftover."""
    try:
        if _wsl(["pgrep", "-f", SERVER_PY]).returncode == 0:
            _wsl(["pkill", "-f", SERVER_PY])
            time.sleep(1.0)
            return True
    except Exception:
        pass
    return False


def port_free(port):
    """True if nothing (including a stale wslrelay listener) holds 127.0.0.1:port on Windows."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def pick_port(preferred, tries=10):
    for p in range(preferred, preferred + tries):
        if port_free(p):
            return p
    raise RuntimeError(f"ports {preferred}-{preferred + tries - 1} are all busy")


class ServerProcess:
    """Starts live_server.py inside WSL with no console window; stops it and frees VRAM."""

    def __init__(self, port, log_path, idle_exit=60, mode="gpu", low_priority=False, token=None):
        """mode "gpu": load and go straight to the GPU; "standby": load into RAM only (0 VRAM).
        low_priority: run under nice 10 / ionice best-effort 7 (pre-warm; renices itself on activate).
        token: shared secret the server will require (handed over in the environment via WSLENV)."""
        self.port, self.log_path, self.idle_exit = port, log_path, idle_exit
        self.token = token
        self.mode, self.low_priority = mode, low_priority
        self.proc = None
        self.logf = None
        self.adopted = False      # started by an earlier widget run and reconnected to (keep-server, 2026-10-09)

    @classmethod
    def adopt(cls, port, log_path, token, mode="gpu"):
        """A server an earlier widget left running: no process handle here; /health decides if it lives."""
        s = cls(port, log_path, mode=mode, token=token)
        s.adopted = True
        return s

    def start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        try:                                          # keep earlier sessions, but cap the file
            if os.path.getsize(self.log_path) > 2_000_000:
                os.replace(self.log_path, self.log_path + ".1")
        except OSError:
            pass
        self.logf = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self.logf.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} start mode={self.mode} "
                        f"low_priority={self.low_priority}\n")
        self.logf.flush()
        prio = ["nice", "-n", "10", "ionice", "-c2", "-n7"] if self.low_priority else []
        env = None
        if self.token:                                # environment, not argv: not visible to ps / tasklist
            env = dict(os.environ, PLIVE_TOKEN=self.token)
            names = [v for v in env.get("WSLENV", "").split(":") if v and v.split("/")[0] != "PLIVE_TOKEN"]
            env["WSLENV"] = ":".join(names + ["PLIVE_TOKEN"])
        try:
            self.proc = subprocess.Popen(
                ["wsl.exe", "-d", DISTRO, "-u", "root", "-e"] + prio + [VENV_PY, SERVER_PY,
                 "--port", str(self.port), "--idle-exit", str(self.idle_exit),
                 "--standby" if self.mode == "standby" else "--gpu"],
                stdin=subprocess.DEVNULL, stdout=self.logf, stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW, env=env)
        except Exception:
            self.logf.close()
            self.logf = None
            raise

    def alive(self):
        if self.adopted:
            return True           # no handle to poll: the health checks (3 misses = gone) decide
        return self.proc is not None and self.proc.poll() is None

    def stop(self, client=None, wait=8.0):
        """Ask the server to exit; fall back to pkill inside WSL; returns True when gone."""
        if client is not None:
            client.shutdown()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                pass
        gone = True
        try:
            r = _wsl(["pgrep", "-f", SERVER_PY])
            if r.returncode == 0 and r.stdout.strip():
                _wsl(["pkill", "-f", SERVER_PY])
                time.sleep(1.0)
                gone = _wsl(["pgrep", "-f", SERVER_PY]).returncode != 0
        except Exception:
            pass
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        if self.logf:
            try:
                self.logf.close()
            except Exception:
                pass
        return gone


def boot_time():
    """Wall-clock time (epoch s) of the current Windows boot, or None (not Windows / unknown)."""
    try:
        import ctypes
        f = ctypes.windll.kernel32.GetTickCount64
        f.restype = ctypes.c_ulonglong
        return time.time() - f() / 1000.0
    except Exception:
        return None


class ServerRecord:
    """keep-server (2026-10-09): what the next widget run needs to reconnect to the model server this one
    leaves running - port, per-server secret, Linux pid and start time. Lives in the locked state folder
    (never in the repo, never logged). One use: the reader removes it; a new one is written once the
    (re)connected or newly started server answers."""

    def __init__(self, state_dir):
        self.path = os.path.join(os.path.abspath(state_dir), "server.json")

    def save(self, port, token, health):
        d = {"v": 1, "port": int(port), "token": token, "pid": health.get("pid"),
             "started": health.get("started"), "saved": time.time()}
        return SessionStore._atomic(self.path, json.dumps(d).encode("utf-8"))

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            return d if d.get("v") == 1 and d.get("token") and d.get("port") else None
        except Exception:
            return None

    def clear(self):
        SessionStore._rm(self.path)

    @staticmethod
    def saved_before_boot(rec, boot=None):
        """staleskip (2026-10-09): a record written before this Windows boot points at a server that died
        with the old session; skip the 2 s health timeout and cold-start at once. Unknown -> False (try)."""
        try:
            if boot is None:
                boot = boot_time()
            return boot is not None and float(rec.get("saved")) < boot
        except (TypeError, ValueError):
            return False

    @staticmethod
    def matches(rec, health):
        """Same server process (pid + start time), and in a state we can take over."""
        try:
            return (health.get("pid") == rec.get("pid")
                    and abs(float(health.get("started")) - float(rec.get("started"))) < 1.0
                    and health.get("status") in ("ready", "standby"))
        except (TypeError, ValueError):
            return False


# --------------------------------------------------------------------------- audio
_UTT_IDS = itertools.count(1)   # utterance ids are unique for the whole process, across recordings (log 55)


def new_utterance_id():
    return next(_UTT_IDS)


class SessionStore:
    """What a restart or crash must not lose (log 57), under <data>/state (the data folder is locked to the
    user's account; never in the repo; never logged - only counts and lengths are):
      draft.json            the unsent message: box text, utterance spans, clear-on-return flags
      pending/<id>.wav      audio of a failed transcription waiting for a retry (16 kHz mono 16-bit)
      pending/<id>.json     its metadata, written after the .wav (a .wav without .json is unfinished)
    Every file is written to a temp file, fsync'ed, then renamed over the old one (atomic)."""
    DRAFT_MAX_AGE_S = 7 * 86400

    def __init__(self, root):
        self.dir = os.path.abspath(root)
        self.pending_dir = os.path.join(self.dir, "pending")
        self.draft_path = os.path.join(self.dir, "draft.json")

    @staticmethod
    def _atomic(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for _ in range(20):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:               # another program has it open this instant
                time.sleep(0.05)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False

    @staticmethod
    def _rm(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def save_draft(self, d):
        """d = {"text", "spans", "cor"}; an empty box removes the file."""
        if not (d.get("text") or "").strip():
            self._rm(self.draft_path)
            return True
        return self._atomic(self.draft_path, json.dumps(dict(d, v=1, saved=time.time()), ensure_ascii=False)
                            .encode("utf-8"))

    def load_draft(self):
        try:
            with open(self.draft_path, encoding="utf-8") as f:
                d = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return None
        if not isinstance(d, dict) or not isinstance(d.get("text"), str) or not d["text"].strip():
            return None
        if time.time() - float(d.get("saved") or 0) > self.DRAFT_MAX_AGE_S:
            self._rm(self.draft_path)            # a week old: not "unsent" any more
            return None
        return d

    def save_pending(self, kid, pcm, meta):
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        base = os.path.join(self.pending_dir, kid)
        return (self._atomic(base + ".wav", b.getvalue())
                and self._atomic(base + ".json", json.dumps(dict(meta, v=1)).encode("utf-8")))

    def drop_pending(self, kid):
        base = os.path.join(self.pending_dir, kid)
        self._rm(base + ".json")                 # metadata first: a lone .wav is cleaned up later
        self._rm(base + ".wav")

    def load_pending(self, max_age_s):
        """-> [(kid, pcm, meta)] oldest first; expired or broken entries are removed."""
        out = []
        try:
            names = os.listdir(self.pending_dir)
        except FileNotFoundError:
            return out
        now = time.time()
        for n in names:
            p = os.path.join(self.pending_dir, n)
            if n.endswith(".tmp") or (n.endswith(".wav") and n[:-4] + ".json" not in names):
                try:
                    if now - os.path.getmtime(p) > 3600:
                        self._rm(p)              # an unfinished write from a crash
                except OSError:
                    pass
                continue
            if not n.endswith(".json"):
                continue
            kid = n[:-5]
            try:
                with open(p, encoding="utf-8") as f:
                    meta = json.load(f)
                if now - float(meta["failed_at"]) > max_age_s:
                    self.drop_pending(kid)
                    continue
                with wave.open(os.path.join(self.pending_dir, kid + ".wav"), "rb") as w:
                    pcm = w.readframes(w.getnframes())
                out.append((kid, pcm, meta))
            except Exception:
                self.drop_pending(kid)           # unreadable: nothing to retry
        out.sort(key=lambda x: float(x[2]["failed_at"]))
        return out


class SpecCache:
    """The last speculative transcription, reusable only by the final of the very same audio
    (same utterance id and version). Any final consumes or drops it, so nothing stale survives."""

    def __init__(self):
        self.item = None              # (uid, ver, result, rtt_ms)

    def put(self, uid, ver, result, rtt):
        self.item = (uid, ver, result, rtt)

    def take_for_final(self, uid, ver):
        it, self.item = self.item, None
        if it is not None and it[0] == uid and it[1] == ver:
            return it[2], it[3]
        return None


class StreamResampler:
    """Streaming low-pass FIR + fractional (linear) resampler, any rate -> 16 kHz.
    Keeps filter history between blocks so there are no seams; ~0.1 ms per 20 ms block."""

    def __init__(self, sr_in, sr_out=SR, taps=63):
        self.sr_in, self.sr_out = float(sr_in), float(sr_out)
        self.passthrough = int(sr_in) == int(sr_out)
        self.ratio = self.sr_in / self.sr_out
        if not self.passthrough:
            fc = 0.45 * self.sr_out / self.sr_in            # cutoff (cycles/sample), ~7.2 kHz
            n = np.arange(taps) - (taps - 1) / 2
            h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
            self.h = (h / h.sum()).astype(np.float32)
            self.hist = np.zeros(taps - 1, np.float32)
            self.prev = np.float32(0.0)
            self.t = 0.0

    def process(self, x):
        x = np.asarray(x, np.float32)
        if self.passthrough or len(x) == 0:
            return x
        buf = np.concatenate([self.hist, x])
        y = np.convolve(buf, self.h, mode="valid").astype(np.float32)   # len == len(x)
        self.hist = buf[-(len(self.h) - 1):]
        yy = np.concatenate([[self.prev], y])
        last = len(yy) - 1
        if self.t > last:
            self.t -= last
            self.prev = yy[-1]
            return np.zeros(0, np.float32)
        pos = np.arange(self.t, last, self.ratio)
        out = np.interp(pos, np.arange(len(yy)), yy).astype(np.float32)
        self.t = (pos[-1] + self.ratio - last) if len(pos) else self.t - last
        self.prev = yy[-1]
        return out


def list_input_devices():
    """WASAPI input devices (full names), default first. Returns (devices, default_key)."""
    import sounddevice as sd
    apis = sd.query_hostapis()
    devs = sd.query_devices()
    wasapi = next((i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None)
    api = wasapi if wasapi is not None else sd.default.hostapi
    out = []
    for i, d in enumerate(devs):
        if d["hostapi"] == api and d["max_input_channels"] > 0:
            out.append({"index": i, "name": d["name"], "hostapi": apis[api]["name"],
                        "channels": d["max_input_channels"], "rate": d["default_samplerate"]})
    dflt = apis[api].get("default_input_device", -1)
    default_name = next((d["name"] for d in out if d["index"] == dflt), out[0]["name"] if out else None)
    return out, default_name


def refresh_portaudio():
    import sounddevice as sd
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


class Segmenter:
    """Energy VAD on 20 ms frames with an adaptive noise floor.

    Calls on_utterance(pcm16_bytes, t_end_of_speech, audio_s, kind, uid, ver):
      kind="spec"  - speculative send after spec_s of silence (inference runs while we keep waiting)
      kind="final" - silence reached silence_s (or max_s / Stop); if ver equals the last spec's ver the
                     audio is identical and the speculative result can be shown immediately.
    """

    def __init__(self, on_utterance, silence_s=0.5, max_s=25.0, preroll_s=0.3,
                 margin_db=9.0, floor_db=-62.0, min_voiced_s=0.2, tail_s=0.15, speculate=True):
        self.on_utterance = on_utterance
        self.silence_s, self.max_s = silence_s, max_s
        self.margin_db, self.floor_db = margin_db, floor_db
        self.min_voiced_s, self.tail_frames = min_voiced_s, int(round(tail_s * SR / FRAME))
        self.speculate = speculate
        self.pre = deque(maxlen=int(round(preroll_s * SR / FRAME)))
        self.noise = -60.0
        self.level_db = -100.0
        self.uid = 0
        self.reset()
        self.left = np.zeros(0, np.float32)

    @property
    def spec_s(self):
        return max(0.16, self.silence_s - 0.3)

    def reset(self):
        self.active = False
        self.frames, self.frame_db = [], []
        self.silence = 0
        self.voiced = 0
        self.run = 0
        self.last_voiced_idx = -1
        self.spec_ver = None
        self.t_last_voiced = 0.0

    def feed(self, x, now=None):
        now = time.perf_counter() if now is None else now
        buf = np.concatenate([self.left, x]) if len(self.left) else x
        n = len(buf) // FRAME
        self.left = buf[n * FRAME:].copy()
        for k in range(n):
            self._frame(buf[k * FRAME:(k + 1) * FRAME], now)

    def _frame(self, f, now):
        db = 20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-9)
        self.level_db = db
        # adaptive noise floor: falls fast, rises slowly (speech doesn't drag it up)
        self.noise += (0.2 if db < self.noise else 0.003) * (db - self.noise)
        thr = max(self.noise + self.margin_db, self.floor_db)
        voiced = db > (thr - 3.0 if self.active else thr)
        if not self.active:
            self.pre.append(f)
            self.run = self.run + 1 if voiced else 0
            if self.run >= 2:                       # 40 ms of sound starts an utterance
                self.active = True
                self.uid = next(_UTT_IDS)
                self.frames = list(self.pre)
                self.frame_db = [db] * len(self.frames)
                self.voiced = self.run
                self.last_voiced_idx = len(self.frames) - 1
                self.t_last_voiced = now
                self.silence = 0
                self.spec_ver = None
                self.pre.clear()
            return
        self.frames.append(f)
        self.frame_db.append(db)
        if voiced:
            self.voiced += 1
            self.silence = 0
            self.last_voiced_idx = len(self.frames) - 1
            self.t_last_voiced = now
        else:
            self.silence += 1
        sil = self.silence * FRAME / SR
        if sil >= self.silence_s:
            self._emit("final")
        elif len(self.frames) * FRAME / SR >= self.max_s:
            self._cut_long(now)
        elif self.speculate and sil >= self.spec_s and self.spec_ver != self.last_voiced_idx:
            self.spec_ver = self.last_voiced_idx
            self._emit("spec")

    def _pcm(self, end):
        a = np.concatenate(self.frames[:max(1, end)])
        return (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes(), len(a) / SR

    def _emit(self, kind):
        if self.voiced * FRAME / SR < self.min_voiced_s or not self.frames:
            if kind == "final":
                self.reset()
            return
        pcm, dur = self._pcm(self.last_voiced_idx + 1 + self.tail_frames)
        self.on_utterance(pcm, self.t_last_voiced, dur, kind, self.uid, self.last_voiced_idx)
        if kind == "final":
            self.reset()

    def _cut_long(self, now):
        # cut at the quietest frame of the last 2 s; carry the rest into a new utterance
        look = int(2.0 * SR / FRAME)
        dbs = self.frame_db[-look:]
        cut = len(self.frames) - look + int(np.argmin(dbs)) + 1
        rest_f, rest_db = self.frames[cut:], self.frame_db[cut:]
        if self.voiced * FRAME / SR >= self.min_voiced_s:
            pcm, dur = self._pcm(cut)
            self.on_utterance(pcm, now, dur, "final", self.uid, -cut)
        self.uid = next(_UTT_IDS)
        self.frames, self.frame_db = rest_f, rest_db
        self.voiced = len(rest_f)
        self.silence = 0
        self.spec_ver = None
        self.last_voiced_idx = len(self.frames) - 1
        self.t_last_voiced = now

    def flush(self):
        """Stop pressed: send whatever speech is buffered right away."""
        if self.active:
            self._emit("final")
        self.reset()
        self.pre.clear()
        self.left = np.zeros(0, np.float32)


class MicCapture:
    """sounddevice InputStream (WASAPI shared, device rate) -> worker thread -> 16 kHz mono.
    The PortAudio callback only copies the block into a queue."""

    def __init__(self, device_name, on_audio, on_error):
        self.device_name, self.on_audio, self.on_error = device_name, on_audio, on_error
        self.q = queue.Queue()
        self.stream = None
        self.stop_evt = threading.Event()
        self.info = ""
        self.ch_energy = None
        self.stall_s = 3.0              # no audio block for this long = the device is gone (log 55)
        self.stalled = False

    def _open(self):
        import sounddevice as sd
        devs, default_name = list_input_devices()
        name = self.device_name or default_name
        d = next((x for x in devs if x["name"] == name), None) or next(
            (x for x in devs if x["name"] == default_name), None)
        errors = []
        if d is not None:
            ch = min(2, d["channels"])
            try:
                return self._start(d["index"], int(d["rate"]), ch, f"{d['name']} ({d['hostapi']}, {int(d['rate'])} Hz)")
            except Exception as e:
                errors.append(f"WASAPI: {e}")
        # fallback: MME lets Windows resample to 16 kHz for us
        apis = sd.query_hostapis()
        mme = next((i for i, a in enumerate(apis) if a["name"] == "MME"), None)
        if mme is not None:
            cands = [i for i, x in enumerate(sd.query_devices())
                     if x["hostapi"] == mme and x["max_input_channels"] > 0]
            pick = None
            if d is not None:
                pick = next((i for i in cands if d["name"].startswith(sd.query_devices(i)["name"][:28])), None)
            if pick is None:
                pick = apis[mme]["default_input_device"]
            try:
                info = sd.query_devices(pick)
                return self._start(pick, SR, min(2, info["max_input_channels"]), f"{info['name']} (MME, 16000 Hz)")
            except Exception as e:
                errors.append(f"MME: {e}")
        raise RuntimeError("; ".join(errors) or "no input device found")

    def _start(self, index, rate, ch, label):
        import sounddevice as sd
        self.rs = StreamResampler(rate)
        self.ch = ch
        self.ch_energy = np.zeros(ch)
        self.stream = sd.InputStream(device=index, samplerate=rate, channels=ch, dtype="float32",
                                     blocksize=int(rate * 0.02), latency="low", callback=self._cb)
        self.stream.start()
        self.info = label

    def _cb(self, indata, frames, t, status):
        self.q.put(indata.copy())

    def start(self):
        self._open()
        self.stop_evt.clear()
        self.worker = threading.Thread(target=self._run, name="mic-worker", daemon=True)
        self.worker.start()

    def _run(self):
        last_check = last_block = time.perf_counter()
        while not self.stop_evt.is_set():
            try:
                blk = self.q.get(timeout=0.25)
            except queue.Empty:
                blk = None
            if blk is not None:
                last_block = time.perf_counter()
                if self.ch > 1:
                    # follow the louder channel (AudioBox-style interfaces: mic on input 1 or 2)
                    e = (blk * blk).mean(axis=0)
                    self.ch_energy = 0.95 * self.ch_energy + 0.05 * e
                    x = blk[:, int(np.argmax(self.ch_energy))]
                else:
                    x = blk[:, 0]
                y = self.rs.process(x)
                if len(y):
                    self.on_audio(y)
            now = time.perf_counter()
            if now - last_check > 0.5:
                last_check = now
                s = self.stream
                if s is not None and not s.active and not self.stop_evt.is_set():
                    self.on_error("Microphone stopped (unplugged or in use?)")
                    return
                if now - last_block > self.stall_s and not self.stop_evt.is_set():
                    self.stalled = True
                    self.on_error("Microphone stopped delivering audio (unplugged or driver problem?)")
                    return

    def stop(self):
        self.stop_evt.set()
        w = getattr(self, "worker", None)
        if w is not None and w is not threading.current_thread():
            w.join(timeout=1.0)
        s, self.stream = self.stream, None
        if s is not None:
            try:
                if self.stalled:
                    s.abort()                   # a dead device may never drain
                else:
                    s.stop()
                s.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- Win32
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32.OpenClipboard.argtypes = [wt.HWND]
user32.OpenClipboard.restype = wt.BOOL
user32.SetClipboardData.argtypes = [wt.UINT, wt.HANDLE]
user32.SetClipboardData.restype = wt.HANDLE
user32.GetClipboardData.argtypes = [wt.UINT]
user32.GetClipboardData.restype = wt.HANDLE
kernel32.GlobalAlloc.argtypes = [wt.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wt.HGLOBAL
kernel32.GlobalLock.argtypes = [wt.HGLOBAL]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [wt.HGLOBAL]
kernel32.GlobalFree.argtypes = [wt.HGLOBAL]
CF_UNICODETEXT = 13


def _open_clipboard():
    for _ in range(25):
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.02)
    return False


def set_clipboard(text: str) -> bool:
    """Real Win32 clipboard write (persists after the widget closes)."""
    data = text.encode("utf-16-le") + b"\x00\x00"
    if not _open_clipboard():
        return False
    try:
        user32.EmptyClipboard()
        h = kernel32.GlobalAlloc(0x0002, len(data))          # GMEM_MOVEABLE
        if not h:
            return False
        p = kernel32.GlobalLock(h)
        if not p:
            kernel32.GlobalFree(h)
            return False
        ctypes.memmove(p, data, len(data))
        kernel32.GlobalUnlock(h)
        if not user32.SetClipboardData(CF_UNICODETEXT, h):
            kernel32.GlobalFree(h)
            return False
        return True
    finally:
        user32.CloseClipboard()


def get_clipboard() -> str:
    if not _open_clipboard():
        return ""
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return ""
        p = kernel32.GlobalLock(h)
        if not p:
            return ""
        try:
            return ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


MODS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
KEYS = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "pause": 0x13, "insert": 0x2D,
        "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22}


def parse_hotkey(s):
    mods, vk = 0, None
    for part in s.lower().replace(" ", "").split("+"):
        if part in MODS:
            mods |= MODS[part]
        elif part in KEYS:
            vk = KEYS[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            vk = 0x6F + int(part[1:])
    if vk is None:
        raise ValueError(f"bad hotkey {s!r}")
    return mods, vk

