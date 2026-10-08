"""Log 45: load-time tests through the widget's own client code (plive_core).
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
# ---------- A: cold GPU
p("A: cold (wsl --shutdown) -> --gpu -> ready"); shutdown_wsl(); base = vram()
t0 = time.perf_counter()
core.kill_stale_servers()                         # the widget does this first; it boots the VM
t_vm = time.perf_counter() - t0
srv = core.ServerProcess(port, LOG, idle_exit=90, mode="gpu"); srv.start(); c = core.Client(port)
h, _ = wait_status(c, ("ready",), 240); tA = time.perf_counter() - t0
res["A_cold_gpu"] = {"total_s": round(tA, 1), "vm_boot_via_pgrep_s": round(t_vm, 1), "server_load_s": h["load_s"],
                     "activate_s": h["activate_s"], "timings": h["timings"], "loader": h["loader"], "vram_ready": vram(), "baseline_vram": base,
                     "rss_mib": h["rss_mib"], "samples": samples(c)}
p("A", json.dumps(res["A_cold_gpu"]))
srv.stop(c); time.sleep(1); res["A_cold_gpu"]["vram_after_stop"] = vram()
# ---------- B: pre-warm (cold VM, nice) -> standby
p("B: cold pre-warm --standby (nice) -> standby"); shutdown_wsl()
t0 = time.perf_counter(); core.kill_stale_servers()
srv = core.ServerProcess(port, LOG, idle_exit=90, mode="standby", low_priority=True); srv.start(); c = core.Client(port)
h, _ = wait_status(c, ("standby",), 240); tB = time.perf_counter() - t0
time.sleep(2)
h = c.health()
res["B_prewarm"] = {"total_s": round(tB, 1), "server_load_s": h["load_s"], "timings": h["timings"], "loader": h["loader"],
                    "vram_standby": vram(), "baseline_vram": base, "server_rss_mib": h["rss_mib"], **vm_mem(), "vmmem_ws_mib": vmmem_ws()}
p("B", json.dumps(res["B_prewarm"]))
# ---------- C: standby -> GPU
p("C: /activate from standby")
ts = []
for i in range(1):
    t0 = time.perf_counter(); st = c.activate(); h, _ = wait_status(c, ("ready",), 60); tC = time.perf_counter() - t0
    res["C_activate"] = {"click_to_ready_s": round(tC, 2), "server_activate_s": h["activate_s"], "timings": h["timings"],
                         "vram_ready": vram(), "server_rss_mib": h["rss_mib"], **vm_mem(), "vmmem_ws_mib": vmmem_ws(), "samples": samples(c)}
p("C", json.dumps(res["C_activate"]))
srv.stop(c); time.sleep(1); res["C_activate"]["vram_after_stop"] = vram()
# ---------- D: warm standby restart (VM up, page cache warm): what idle->standby costs in the background, then activate again
p("D: warm standby restart + activate")
t0 = time.perf_counter(); srv = core.ServerProcess(port, LOG, idle_exit=90, mode="standby", low_priority=True); srv.start(); c = core.Client(port)
h, _ = wait_status(c, ("standby",), 240); tD = time.perf_counter() - t0
t0 = time.perf_counter(); c.activate(); h2, _ = wait_status(c, ("ready",), 60); tD2 = time.perf_counter() - t0
res["D_warm"] = {"standby_reload_s": round(tD, 1), "timings_load": h["timings"], "click_to_ready_s": round(tD2, 2), "samples": samples(c)}
p("D", json.dumps(res["D_warm"]))
srv.stop(c); time.sleep(1)
res["final_vram"] = vram()
res["server_left"] = core._wsl(["pgrep", "-af", core.SERVER_PY]).stdout.decode().strip()
print("RESULT " + json.dumps(res, indent=1))
