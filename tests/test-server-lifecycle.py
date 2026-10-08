"""Start/stop live_server.py exactly the way the widget does (plive_core.ServerProcess + Client)
and log health, keep-alive transcription timings and VRAM before/during/after."""
import os, sys, time, subprocess, json, wave, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "widget"))
import plive_core as core

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 51761
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else None
def ts(): return time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
def P(*a): print(ts(), *a, flush=True)

P("nvidia-smi before:", smi())
srv = core.ServerProcess(PORT, os.path.join((os.environ.get("PARAKEET_LIVE_DATA") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ParakeetLive")), "logs", f"server-lifecycle-{PORT}.txt"), idle_exit=60)
t0 = time.perf_counter(); srv.start(); P("server process started (hidden wsl.exe)")
hc = core.Client(PORT)
last = None
while time.perf_counter() - t0 < 200:
    try:
        h = hc.health(timeout=2.0); r = h.get("status")
    except Exception as e:
        r = type(e).__name__
    if r != last:
        P(f"health -> {r} at {time.perf_counter()-t0:.1f}s"); last = r
    if r in ("ready", "error"):
        break
    time.sleep(0.5)
P("health:", json.dumps(h))
P("nvidia-smi warm idle:", smi())
fails = 0
for i in range(20):
    try: hc.health(timeout=2.0)
    except Exception: fails += 1
    time.sleep(0.1)
P(f"20 health pings on keep-alive connection, failures: {fails}")
if SAMPLE:
    with wave.open(SAMPLE, "rb") as w: pcm = w.readframes(w.getnframes())
    tc = core.Client(PORT)
    rtt, inf = [], []
    for i in range(10):
        a = time.perf_counter(); r = tc.transcribe(pcm); rtt.append((time.perf_counter()-a)*1000); inf.append(r["infer_ms"])
    P("text:", r["text"], "| path:", r.get("path"))
    P(f"7.4 s clip x10: server infer median {statistics.median(inf):.0f} ms worst {max(inf)}; round trip median {statistics.median(rtt):.0f} worst {max(rtt):.0f}")
    pcm2 = pcm[:16000*2*2]; rtt, inf = [], []
    for i in range(10):
        a = time.perf_counter(); r = tc.transcribe(pcm2); rtt.append((time.perf_counter()-a)*1000); inf.append(r["infer_ms"])
    P(f"2 s clip x10: server infer median {statistics.median(inf):.0f} ms worst {max(inf)}; round trip median {statistics.median(rtt):.0f} worst {max(rtt):.0f}; text={r['text']!r}")
    r = tc._request("POST", "/transcribe?path=slow", body=pcm)[1]
    P("slow path (NeMo transcribe) text:", r["text"], f"({r['infer_ms']} ms)")
    P("nvidia-smi after transcriptions:", smi(), "| health:", json.dumps(hc.health()))
a = time.perf_counter(); gone = srv.stop(core.Client(PORT)); P(f"stop() -> gone={gone} in {time.perf_counter()-a:.1f}s; wsl.exe exit code {srv.proc.poll()}")
time.sleep(1.5)
P("nvidia-smi after stop:", smi())
r = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "-e", "pgrep", "-af", "live_server.py"], capture_output=True, text=True)
P("pgrep live_server.py inside WSL:", repr(r.stdout.strip()), "rc", r.returncode)
