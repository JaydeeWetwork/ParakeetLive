"""Log 48 (page-cache trim in standby); helpers from log 45: load-time tests through the widget's own client code (plive_core).
  A  cold: wsl --shutdown -> server --gpu -> ready (what a click does with nothing pre-loaded)
  B  pre-warm: wsl --shutdown -> server --standby under nice (what login does) -> standby; RAM + VRAM cost
  C  standby -> /activate -> ready (what a click does after pre-warm / idle standby)
  each: transcripts of both samples vs the reference texts; VRAM after stop."""
import json, os, subprocess, sys, time, wave
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "widget"))
import plive_core as core
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = (os.environ.get("PARAKEET_LIVE_DATA") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ParakeetLive"))
LOG = os.path.join(DATA, "logs", "server-test.log")
REF = {"libri sample 16k mono.wav": "Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.",
       "espeak sample 16k mono.wav": "Hello JD. This is a local transcription test running on the workstation graphics card. The bundle was a win."}
def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def vram(): return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
def vm_mem():
    r = core._wsl(["bash", "-c", "free -m | awk '/Mem:/{print $3, $6}'"])
    used, cache = r.stdout.decode().split()
    return {"vm_used_mib": int(used), "vm_buffcache_mib": int(cache)}
def vmmem_ws():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        "(Get-Process -Name vmmem,vmmemWSL -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum"], capture_output=True, text=True)
    try: return round(int(r.stdout.strip()) / 2**20)
    except Exception: return None
def wait_status(c, want, timeout):
    t0 = time.perf_counter(); last = None
    while time.perf_counter() - t0 < timeout:
        try:
            h = c.health(timeout=2)
            if h.get("status") != last: p("  status", h.get("status"), f"+{time.perf_counter()-t0:.1f}s"); last = h.get("status")
            if h.get("status") in want: return h, time.perf_counter() - t0
            if h.get("status") == "error": raise RuntimeError(h.get("error"))
        except (OSError, ConnectionError):
            pass
        time.sleep(0.1)
    raise TimeoutError(want)
def samples(c):
    out = {}
    for f in REF:
        with wave.open(os.path.join(DATA, "test-samples", f)) as w:
            pcm = w.readframes(w.getnframes())
        r = c.transcribe(pcm)
        out[f] = (r["text"], r["infer_ms"], r["text"] == REF[f])
    return out
def shutdown_wsl():
    subprocess.run(["wsl.exe", "--shutdown"]); time.sleep(4)

res = {}
port = 51761
def wait_trim(c, timeout=120):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        h = c.health()
        if "trim_cache_s" in h["timings"]: return h
        time.sleep(0.5)
    raise TimeoutError("trim")
for rnd, cold in (("cold", True), ("warm", False)):
    p(f"{rnd}: pre-warm --standby (nice) -> standby -> trim")
    if cold: shutdown_wsl()
    t0 = time.perf_counter(); core.kill_stale_servers()
    srv = core.ServerProcess(port, LOG, idle_exit=90, mode="standby", low_priority=True); srv.start(); c = core.Client(port)
    h, _ = wait_status(c, ("standby",), 240); tB = time.perf_counter() - t0
    mem_pre = {**vm_mem(), "vmmem_ws_mib": vmmem_ws()}
    h = wait_trim(c); tT = time.perf_counter() - t0
    time.sleep(8)                       # let free-page reporting hand the pages back to Windows
    r = {"to_standby_s": round(tB, 1), "to_trimmed_s": round(tT, 1), "timings": h["timings"], "server_rss_mib": h["rss_mib"],
         "before_trim": mem_pre, "after_trim": {**vm_mem(), "vmmem_ws_mib": vmmem_ws()}, "vram_standby": vram()}
    t0 = time.perf_counter(); c.activate(); h2, _ = wait_status(c, ("ready",), 60); tC = time.perf_counter() - t0
    r.update({"activate_click_to_ready_s": round(tC, 2), "server_activate_s": h2["activate_s"], "timings_ready": h2["timings"],
              "vram_ready": vram(), "ready_mem": {**vm_mem(), "vmmem_ws_mib": vmmem_ws()}, "samples": samples(c)})
    srv.stop(c); time.sleep(1); r["vram_after_stop"] = vram()
    res[rnd] = r; p(rnd, json.dumps(r))
res["server_left"] = core._wsl(["bash", "-c", "pgrep -af live_server.py | grep -v pgrep || true"]).stdout.decode().strip()
print("RESULT " + json.dumps(res, indent=1))
