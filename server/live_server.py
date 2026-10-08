#!/opt/parakeet/venv/bin/python
"""Parakeet Live server: keeps NVIDIA Parakeet TDT 0.6B v2 ready for the Windows "Parakeet Live" widget.

Listens on 127.0.0.1 only (WSL2 localhost forwarding makes it reachable from Windows).

States (GET /health -> "status"):
  loading     reading the model into CPU RAM (fast checkpoint, ~3 s warm / ~10 s cold + imports)
  standby     model in CPU RAM, NO CUDA context (0 VRAM); POST /activate moves it to the GPU
  activating  CUDA init + copy to GPU + warm-up (~1-3 s)
  ready       on the GPU, transcribing
  error
  GET  /health      -> {"status": ..., "load_s", "activate_s", "rss_mib", vram (only when on GPU), ...}
  POST /activate    -> start moving to the GPU (if still loading: as soon as the load finishes)
  POST /transcribe  body = raw 16 kHz mono int16 little-endian PCM (no header, no temp files)
                    -> {"text": "...", "audio_s": 3.1, "infer_ms": 120}
  POST /shutdown    -> server exits (process exit frees VRAM and RAM)
HTTP/1.1 keep-alive + TCP_NODELAY, so the widget reuses one connection for every utterance.
Requests with an Origin header (browsers) or a Host other than 127.0.0.1/localhost get 403; bodies are
capped at 60 s of audio. If PLIVE_TOKEN is set (the widget passes a fresh one via WSLENV), every request
must carry it in X-Plive-Token, else 401.

--standby: load into RAM and wait (pre-warm / idle tier; usually started under nice).
--gpu:     activate as soon as loaded (default when neither flag is given).
Exits on its own if no request arrives for --idle-exit seconds (the widget pings /health),
so a crashed widget can never leave the model sitting in VRAM or RAM.
"""
import os
import sys

sys.path.insert(0, "/opt/parakeet/scripts")
import transcribe as tx  # sets the /opt/parakeet cache env vars; reuses load_model() as fallback

import argparse
import hmac
import json
import socket
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")

STATE = {"status": "loading", "started": time.time(), "load_s": None, "activate_s": None, "error": None,
         "device": None, "dtype": None, "requests": 0, "loader": None, "mode": None, "timings": {}}
MODEL = {"model": None, "dtype": None, "device": None}
LOCK = threading.Lock()          # one GPU job at a time
ACT_LOCK = threading.Lock()      # one activation at a time
WANT_GPU = threading.Event()
LAST_SEEN = [time.time()]
DEBUG = [False]
SR = 16000
MAX_AUDIO_S = 60.0
MAX_BODY = int(MAX_AUDIO_S * SR * 2) + 65536    # largest accepted request body (bytes)
TOKEN = os.environ.pop("PLIVE_TOKEN", "")      # shared secret from the widget (via WSLENV); "" = not required


def log(msg):
    print(f"[live {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def rss_mib():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    return round(int(ln.split()[1]) / 1024)
    except Exception:
        pass
    return None


def decode_pcm(data):
    """Raw little-endian int16 mono PCM at 16 kHz -> float32 numpy (no temp files)."""
    import numpy as np
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0


def _infer_fast(audio):
    """Direct tensor path (no dataloader, no temp files): preprocessor -> bf16 encoder -> TDT greedy."""
    import torch
    model, dtype, device = MODEL["model"], MODEL["dtype"], MODEL["device"]
    sig = torch.from_numpy(audio).unsqueeze(0).to(device, non_blocking=True)
    ln = torch.tensor([sig.shape[1]], device=device)
    with torch.inference_mode():
        feats, flen = model.preprocessor(input_signal=sig, length=ln)          # fp32 mel features
        ctx = torch.autocast("cuda", dtype=dtype) if device == "cuda" else torch.autocast("cpu", enabled=False)
        with ctx:
            enc, elen = model.encoder(audio_signal=feats, length=flen)
        hyps = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=enc.float(), encoded_lengths=elen, return_hypotheses=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    h = hyps[0]
    text = getattr(h, "text", h)
    if not isinstance(text, str):
        text = model.tokenizer.ids_to_text([int(t) for t in text])
    return text.strip()


def _infer_slow(audio):
    """NeMo's high-level transcribe() (same as transcribe.py); used as fallback."""
    import torch
    model, dtype, device = MODEL["model"], MODEL["dtype"], MODEL["device"]
    ctx = torch.autocast("cuda", dtype=dtype) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), ctx:
        hyps = model.transcribe([audio], batch_size=1, verbose=False)
    h = hyps[0]
    if isinstance(h, list):        # some NeMo versions return (best, all)
        h = h[0]
    return (getattr(h, "text", h) or "").strip()


def infer(audio, path="auto"):
    import torch
    with LOCK:                     # one GPU job at a time
        try:
            if path == "slow" or not MODEL.get("fast_ok", True):
                text = _infer_slow(audio)
            else:
                try:
                    text = _infer_fast(audio)
                except Exception as e:
                    log(f"fast path failed ({e!r}); using transcribe() from now on")
                    MODEL["fast_ok"] = False
                    text = _infer_slow(audio)
        finally:
            if MODEL["device"] == "cuda":
                # give cached blocks back only when a long utterance grew the pool (keeps idle VRAM
                # low without paying cudaMalloc on every short utterance)
                if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 192 * 2**20:
                    torch.cuda.empty_cache()
    return text


def vram():
    """Only when the model is on the GPU: querying CUDA would otherwise create a context (VRAM)."""
    if MODEL.get("device") != "cuda":
        return {}
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return {"torch_alloc_mib": round(torch.cuda.memory_allocated() / 2**20),
                "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20),
                "gpu_used_mib": round((total - free) / 2**20), "gpu_total_mib": round(total / 2**20)}
    except Exception:
        return {}


def _fadvise(path, advice):
    """Page-cache hints: WILLNEED = start reading the weights now (overlaps the ~12 s of imports);
    DONTNEED = drop them from the VM page cache once copied into the model (saves ~1.2 GB of VM RAM)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, advice)
        finally:
            os.close(fd)
        return True
    except Exception:
        return False


def _weights_files():
    sys.path.insert(0, "/opt/parakeet/scripts")
    import plive_fastload as fl
    return [os.path.join(fl.FAST_DIR, fl.WEIGHTS), os.path.realpath(fl.NEMO_FILE)]


def _cached_mib():
    try:
        with open("/proc/meminfo") as f:
            m = dict(l.split(":", 1) for l in f)
        return int(m["Cached"].split()[0]) // 1024
    except Exception:
        return None


def trim_cache():
    """Standby only: the imports pulled ~1.7 GB of library files into the VM page cache, but the process keeps
    only ~0.45 GB of them mapped. Drop the unmapped rest (DONTNEED never evicts mapped pages) so the WSL VM
    hands that RAM back to Windows. Runs at the pre-warm's low priority, after the model is in RAM."""
    import site
    t = time.perf_counter()
    before = _cached_mib()
    roots = site.getsitepackages() + [os.path.dirname(os.__file__)]
    n = 0
    for root in roots:
        for dp, _dn, fn in os.walk(root):
            if STATE["status"] != "standby":
                break                                  # a user clicked: stop trimming, CUDA may need these
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) >= 65536 and not os.path.islink(p):
                        n += _fadvise(p, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
    tm = STATE["timings"]
    tm["trim_cache_s"] = round(time.perf_counter() - t, 2)
    after = _cached_mib()
    if before is not None and after is not None:
        tm["trim_cache_freed_mib"] = before - after
    log(f"page cache trimmed: {n} files, cached {before} -> {after} MiB in {tm['trim_cache_s']} s")


def load(args):
    """Model into CPU RAM (no CUDA). Fast checkpoint if built, else the stock .nemo path."""
    tm = STATE["timings"]
    try:
        t0 = time.perf_counter()
        import numpy  # noqa: F401
        import torch
        tm["import_torch_s"] = round(time.perf_counter() - t0, 2)
        t = time.perf_counter()
        import nemo.collections.asr  # noqa: F401
        import logging
        logging.getLogger("nemo_logger").setLevel(logging.ERROR)
        try:
            from nemo.utils import logging as nemo_logging
            nemo_logging.setLevel(logging.ERROR)
        except Exception:
            pass
        tm["import_nemo_s"] = round(time.perf_counter() - t, 2)
        t = time.perf_counter()
        import plive_fastload as fl
        if fl.available() and not args.stock_load:
            model = fl.load_cpu()
            STATE["loader"] = "fast"
        else:
            model, _ = tx.load_model("cpu", no_cuda_graphs=True)
            model.encoder.to(torch.bfloat16)
            STATE["loader"] = "stock"
        tm["restore_s"] = round(time.perf_counter() - t, 2)
        t = time.perf_counter()
        try:
            model.change_subsampling_conv_chunking_factor(1)
        except Exception:
            pass
        tx.disable_cuda_graph_decoder(model)       # CUDA graphs crash under WSL
        try:   # deterministic features, like transcribe() sets during inference
            model.preprocessor.featurizer.dither = 0.0
            model.preprocessor.featurizer.pad_to = 0
        except Exception:
            pass
        tm["configure_s"] = round(time.perf_counter() - t, 2)
        for f in _weights_files():
            _fadvise(f, os.POSIX_FADV_DONTNEED)
        MODEL.update(model=model, dtype=torch.bfloat16, device="cpu", fast_ok=not args.slow)
        STATE.update(status="standby", load_s=round(time.perf_counter() - t0, 1), device="cpu", dtype="bfloat16")
        log(f"in RAM (standby) after {STATE['load_s']} s via {STATE['loader']} loader; rss={rss_mib()} MiB; {tm}")
    except Exception as e:  # report through /health instead of dying silently
        STATE.update(status="error", error=f"{e.__class__.__name__}: {e}")
        log(f"model load failed: {STATE['error']}")
        return
    if WANT_GPU.is_set():
        activate()
    elif not args.no_trim:
        trim_cache()


def _unnice():
    """Linux nice is per thread: renice every thread (main = HTTP accept loop, whose request threads
    inherit it) back to 0, plus I/O priority back to the best-effort default."""
    for tid in os.listdir("/proc/self/task"):
        try:
            os.setpriority(os.PRIO_PROCESS, int(tid), 0)
        except Exception:
            pass
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        for tid in os.listdir("/proc/self/task"):   # ioprio_set(IOPRIO_WHO_PROCESS, tid, BE class 2, level 4)
            libc.syscall(251, 1, int(tid), (2 << 13) | 4)
    except Exception:
        pass


def activate():
    """CPU RAM -> GPU + warm-up. Called from /activate or right after loading (--gpu)."""
    import numpy as np
    import torch
    with ACT_LOCK:
        if STATE["status"] != "standby":
            return
        STATE["status"] = "activating"
        tm = STATE["timings"]
        try:
            _unnice()                                    # undo the pre-warm 'nice' now that a user waits
            t0 = time.perf_counter()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            torch.cuda.init()
            torch.zeros(1, device="cuda")
            tm["cuda_init_s"] = round(time.perf_counter() - t0, 2)
            t = time.perf_counter()
            model = MODEL["model"].to("cuda")
            torch.cuda.synchronize()
            tm["to_gpu_s"] = round(time.perf_counter() - t, 2)
            MODEL.update(model=model, device="cuda")
            t = time.perf_counter()
            warm = (np.random.default_rng(0).standard_normal(SR * 2) * 0.01).astype("float32")
            infer(np.zeros(SR, dtype="float32"))       # loads cuDNN/cuBLAS kernels
            infer(warm)
            tm["warmup_s"] = round(time.perf_counter() - t, 2)
            STATE.update(status="ready", device="cuda", activate_s=round(time.perf_counter() - t0, 2))
            log(f"ready on cuda after activate {STATE['activate_s']} s ({tm}); vram={vram()}")
        except Exception as e:
            STATE.update(status="error", error=f"activate failed: {e.__class__.__name__}: {e}")
            log(STATE["error"])


class Handler(BaseHTTPRequestHandler):
    server_version = "ParakeetLive/1.1"
    protocol_version = "HTTP/1.1"      # keep-alive: one persistent connection per client
    timeout = 300

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt, *a):
        if DEBUG[0]:
            log("http " + (fmt % a))

    def _send(self, code, obj, close=False):
        body = json.dumps(obj).encode()
        head = (f"HTTP/1.1 {code} {'OK' if code == 200 else 'ERR'}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                f"Connection: {'close' if close else 'keep-alive'}\r\n\r\n").encode()
        if close:
            self.close_connection = True
        self.wfile.write(head + body)   # one send: no Nagle/delayed-ACK stall

    def _allowed(self):
        """Only our own widget: no browser (Origin / foreign Host), the right token when one is set."""
        host = (self.headers.get("Host") or "").strip().lower()
        if self.headers.get("Origin") is not None or (
                host and host.rsplit(":", 1)[0] not in ("127.0.0.1", "localhost", "[::1]")):
            self._send(403, {"error": "forbidden"}, close=True)
            return False
        if TOKEN and not hmac.compare_digest((self.headers.get("X-Plive-Token") or "").encode(),
                                             TOKEN.encode()):
            self._send(401, {"error": "unauthorized"}, close=True)
            return False
        return True

    def do_GET(self):
        if not self._allowed():
            return
        LAST_SEEN[0] = time.time()
        if self.path.startswith("/health"):
            d = dict(STATE)
            d["timings"] = dict(STATE["timings"])
            d["uptime_s"] = round(time.time() - STATE["started"], 1)
            d["pid"] = os.getpid()
            d["fast_path"] = MODEL.get("fast_ok")
            d["rss_mib"] = rss_mib()
            d.update(vram())
            return self._send(200, d)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._allowed():
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0:
            return self._send(400, {"error": "bad Content-Length"}, close=True)
        if n > MAX_BODY:
            return self._send(413, {"error": f"request too large (max {MAX_AUDIO_S:.0f}s of audio)"}, close=True)
        LAST_SEEN[0] = time.time()
        data = self.rfile.read(n) if n else b""
        if len(data) != n:
            return self._send(400, {"error": "truncated body"}, close=True)
        if self.path.startswith("/shutdown"):
            self._send(200, {"ok": True})
            log("shutdown requested")
            threading.Thread(target=_exit_soon, daemon=True).start()
            return
        if self.path.startswith("/activate"):
            WANT_GPU.set()
            if STATE["status"] == "standby":
                threading.Thread(target=activate, name="activate", daemon=True).start()
            return self._send(200, {"ok": True, "status": STATE["status"]})
        if self.path.startswith("/transcribe"):
            if STATE["status"] != "ready":
                return self._send(503, {"error": f"model {STATE['status']}", "detail": STATE["error"]})
            try:
                t0 = time.perf_counter()
                audio = decode_pcm(data)
                dur = len(audio) / SR
                if dur > MAX_AUDIO_S:
                    return self._send(413, {"error": f"utterance too long ({dur:.0f}s > {MAX_AUDIO_S:.0f}s)"})
                path = "slow" if "path=slow" in self.path else "auto"
                text = infer(audio, path) if dur >= 0.1 else ""
                ms = round((time.perf_counter() - t0) * 1000)
                STATE["requests"] += 1
                log(f"utt {dur:.2f}s -> {ms} ms: {len(text)} chars")   # privacy (log 55): never the text itself
                return self._send(200, {"text": text, "audio_s": round(dur, 2), "infer_ms": ms,
                                        "path": "slow" if path == "slow" or not MODEL.get("fast_ok") else "fast"})
            except Exception as e:
                log(f"transcribe error: {e!r}")
                return self._send(500, {"error": f"{e.__class__.__name__}: {e}"})
        self._send(404, {"error": "not found"})


def _exit_soon():
    time.sleep(0.2)
    os._exit(0)   # process exit releases the CUDA context / VRAM and the RAM copy


def watchdog(idle_s):
    while True:
        time.sleep(2)
        if time.time() - LAST_SEEN[0] > idle_s:
            log(f"no client for {idle_s}s, exiting")
            os._exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=51761)
    ap.add_argument("--idle-exit", type=float, default=90, help="exit after N s without requests (0 = never)")
    ap.add_argument("--standby", action="store_true", help="load into CPU RAM only and wait for /activate")
    ap.add_argument("--gpu", action="store_true", help="move to the GPU as soon as loaded (default)")
    ap.add_argument("--cpu", action="store_true", help="(kept for compatibility; CPU inference is not supported)")
    ap.add_argument("--slow", action="store_true", help="always use NeMo transcribe() instead of the direct fast path")
    ap.add_argument("--stock-load", action="store_true", help="load from the .nemo instead of the fast checkpoint")
    ap.add_argument("--no-trim", action="store_true", help="standby: keep the library files in the VM page cache")
    ap.add_argument("--debug", action="store_true", help="log every HTTP request")
    args = ap.parse_args()
    DEBUG[0] = args.debug
    STATE["mode"] = "standby" if args.standby else "gpu"
    if not args.standby:
        WANT_GPU.set()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)   # localhost only
    srv.daemon_threads = True
    log(f"listening on 127.0.0.1:{args.port} pid={os.getpid()} mode={STATE['mode']} nice={os.nice(0)} "
        f"token={'required' if TOKEN else 'off'}")
    if not args.stock_load:
        _fadvise(_weights_files()[0], os.POSIX_FADV_WILLNEED)     # async readahead while Python imports
    threading.Thread(target=load, args=(args,), daemon=True).start()
    if args.idle_exit > 0:
        threading.Thread(target=watchdog, args=(args.idle_exit,), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
