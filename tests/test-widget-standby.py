"""Log 46: end-to-end tests of the RAM-standby / pre-warm widget (the real parakeet_live.pyw, driven by its IPC).
  W0 cold click: wsl --shutdown -> tray (no pre-warm) -> Load -> ready
  W1 simulated login: wsl --shutdown -> --tray (pre-warm delay 5 s) -> RAM standby; RAM + VRAM cost
  W2 click from standby -> ready; transcripts; thread priorities back to nice 0
  W3 idle (18 s here, 15 min default) -> RAM standby: VRAM back to baseline
  W4 standby (72 s here, 4 h default) -> full unload: RAM freed
  W5 click from unloaded with the VM still up -> ready
  W6 click while the login pre-warm is still loading (cold VM) -> ready
  W7 widget self-test replay of both samples through the live path (VAD -> HTTP -> UI)"""
import json, os, re, subprocess, sys, tempfile, time, wave
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = (os.environ.get("PARAKEET_LIVE_DATA") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ParakeetLive"))
W = os.path.join(REPO, "widget")
sys.path.insert(0, W)
import plive_core as core
PY, PYW = os.path.join(W, ".venv", "Scripts", "python.exe"), os.path.join(W, ".venv", "Scripts", "pythonw.exe")
APP = os.path.join(W, "parakeet_live.pyw")
TOKEN = __import__("secrets").token_hex(16)   # the widget's server requires its token since log 55
WLOG = os.path.join(DATA, "logs", "widget.log")
STATE = os.path.join(DATA, "logs", "state.json")
TRAIN = os.path.join(tempfile.gettempdir(), "plive-test-training")
REF = {"libri sample 16k mono.wav": "Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.",
       "espeak sample 16k mono.wav": "Hello JD. This is a local transcription test running on the workstation graphics card. The bundle was a win."}
def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def vram(): return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
def vm_mem():
    try:
        r = core._wsl(["bash", "-c", "free -m | awk '/Mem:/{print $3, $6}'"])
        used, cache = r.stdout.decode().split()
        return {"vm_used_mib": int(used), "vm_buffcache_mib": int(cache)}
    except Exception as e:
        return {"vm_mem_err": str(e)}
def vmmem_ws():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        "(Get-Process -Name vmmem,vmmemWSL -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum"], capture_output=True, text=True)
    try: return round(int(r.stdout.strip()) / 2**20)
    except Exception: return 0
def widget_ws(pid):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).WorkingSet64"], capture_output=True, text=True)
    try: return round(int(r.stdout.strip()) / 2**20)
    except Exception: return None
def server_nice():
    r = core._wsl(["bash", "-c", "for p in $(pgrep -f live_server.py); do ps -L -o ni= -p $p; done | sort | uniq -c | tr -s ' ' | tr '\\n' ';'"])
    return r.stdout.decode().strip()
def servers(): return core._wsl(["bash", "-c", "pgrep -af live_server.py | grep -v pgrep || true"]).stdout.decode().strip()
def shutdown_wsl(): subprocess.run(["wsl.exe", "--shutdown"]); time.sleep(4)
class Log:
    def __init__(self): self.off = os.path.getsize(WLOG) if os.path.exists(WLOG) else 0
    def wait(self, pat, timeout):
        t0 = time.perf_counter(); rx = re.compile(pat)
        while time.perf_counter() - t0 < timeout:
            with open(WLOG, encoding="utf-8", errors="replace") as f:
                f.seek(self.off); chunk = f.read()
            for line in chunk.splitlines(True):
                if not line.endswith("\n"): break
                self.off += len(line.encode("utf-8"))
                if rx.search(line): return line.strip(), time.perf_counter() - t0
            time.sleep(0.1)
        raise TimeoutError(pat)
def cmd(c): subprocess.run([PY, APP, "--cmd", c], timeout=30)
def state():
    cmd("dump"); time.sleep(0.6)
    with open(STATE, encoding="utf-8") as f: return json.load(f)
def launch(*extra):
    return subprocess.Popen([PYW, APP, "--tray", "--no-save", "--training-dir", TRAIN, "--server-token", TOKEN, *extra])
def quit_widget(proc):
    cmd("quit"); proc.wait(30); time.sleep(2)
    return {"servers_left": servers(), "vram_after_quit": vram()}
def samples(port):
    c = core.Client(port, token=TOKEN); out = {}
    for f in REF:
        with wave.open(os.path.join(DATA, "test-samples", f)) as w: pcm = w.readframes(w.getnframes())
        r = c.transcribe(pcm); out[f] = {"text": r["text"], "infer_ms": r["infer_ms"], "exact": r["text"] == REF[f]}
    c.close(); return out
def short(st): return {k: st.get(k) for k in ("engine", "server_status", "server_rss_mib", "load_s", "activate_s", "server_timings", "status", "port")}

res = {}
try:
    # ---------------- W0
    p("W0 cold click (no pre-warm)"); shutdown_wsl(); base = vram(); res["baseline_vram"] = base
    L = Log(); w = launch("--prewarm-delay", "99999"); L.wait(r"pre-load into RAM scheduled", 30); time.sleep(2)
    t0 = time.perf_counter(); cmd("load"); line, _ = L.wait(r"engine ready", 240); t = time.perf_counter() - t0
    st = state()
    res["W0_cold_click"] = {"click_to_ready_s": round(t, 1), "widget_line": line[:160], **short(st), "vram_ready": vram(),
                            "nice": server_nice(), "samples": samples(st["port"])}
    res["W0_cold_click"].update(quit_widget(w)); p("W0", json.dumps(res["W0_cold_click"]))
    # ---------------- W1
    p("W1 simulated login: pre-warm into RAM"); shutdown_wsl(); res["vmmem_after_shutdown"] = vmmem_ws()
    L = Log(); t_login = time.perf_counter()
    w = launch("--prewarm-delay", "5", "--idle-min", "0.3", "--standby-h", "0.02")
    L.wait(r"pre-loading into RAM \(login", 30); t_pw = time.perf_counter()
    line, _ = L.wait(r"model in RAM \(standby", 240); t_sb = time.perf_counter()
    time.sleep(3); st = state()
    res["W1_prewarm"] = {"login_to_standby_s": round(t_sb - t_login, 1), "prewarm_start_to_standby_s": round(t_sb - t_pw, 1),
                         "widget_line": line[:200], **short(st), "vram_standby": vram(), **vm_mem(), "vmmem_ws_mib": vmmem_ws(),
                         "widget_ws_mib": widget_ws(st.get("pid")), "nice": server_nice(), "tray_tip_engine": st.get("engine")}
    p("W1", json.dumps(res["W1_prewarm"]))
    # ---------------- W2
    p("W2 click from standby")
    t0 = time.perf_counter(); cmd("load"); line, _ = L.wait(r"engine ready", 60); t = time.perf_counter() - t0
    st = state()
    res["W2_standby_click"] = {"click_to_ready_s": round(t, 2), "widget_line": line[:120], **short(st), "vram_ready": vram(),
                               "nice_after_activate": server_nice(), **vm_mem(), "vmmem_ws_mib": vmmem_ws(), "samples": samples(st["port"])}
    p("W2", json.dumps(res["W2_standby_click"]))
    # ---------------- W3
    p("W3 idle -> RAM standby")
    line1, t1 = L.wait(r"moving model to RAM \(idle", 90)
    line2, t2 = L.wait(r"model in RAM \(standby", 120); time.sleep(3); st = state()
    res["W3_idle_to_standby"] = {"trigger": line1[20:], "reload_to_ram_s": round(t2, 1), "widget_line": line2[:200], **short(st),
                                 "vram_standby": vram(), **vm_mem(), "vmmem_ws_mib": vmmem_ws(), "nice": server_nice()}
    p("W3", json.dumps(res["W3_idle_to_standby"]))
    # ---------------- W4
    p("W4 standby -> full unload")
    line, t = L.wait(r"model unloaded \(in RAM standby", 150); time.sleep(4); st = state()
    res["W4_full_unload"] = {"line": line[20:], **short(st), "vram": vram(), "servers": servers(), **vm_mem(), "vmmem_ws_mib": vmmem_ws()}
    p("W4", json.dumps(res["W4_full_unload"]))
    # ---------------- W5
    p("W5 click from unloaded, VM still up")
    t0 = time.perf_counter(); cmd("load"); line, _ = L.wait(r"engine ready", 240); t = time.perf_counter() - t0
    st = state()
    res["W5_warm_vm_click"] = {"click_to_ready_s": round(t, 1), **short(st), "vram_ready": vram(), "samples": samples(st["port"])}
    res["W5_warm_vm_click"].update(quit_widget(w)); p("W5", json.dumps(res["W5_warm_vm_click"]))
    # ---------------- W6
    p("W6 click during the login pre-warm (cold VM)"); shutdown_wsl()
    L = Log(); w = launch("--prewarm-delay", "1"); L.wait(r"pre-loading into RAM \(login", 30); time.sleep(4)
    t0 = time.perf_counter(); cmd("load"); line, _ = L.wait(r"engine ready", 240); t = time.perf_counter() - t0
    st = state()
    res["W6_click_during_prewarm"] = {"click_to_ready_s": round(t, 1), "click_at_s_after_prewarm_start": 4, **short(st), "vram_ready": vram(),
                                      "nice": server_nice(), "samples": samples(st["port"])}
    res["W6_click_during_prewarm"].update(quit_widget(w)); p("W6", json.dumps(res["W6_click_during_prewarm"]))
    # ---------------- W7
    p("W7 widget self-test replay (warm VM)")
    out = os.path.join(tempfile.gettempdir(), "plive-selftest-46.json")
    r = subprocess.run([PY, APP, "--selftest", out, "--no-save", "--training-dir", TRAIN, "--record-seconds", "1",
                        "--replay"] + [os.path.join(DATA, "test-samples", f) for f in REF], timeout=400)
    with open(out, encoding="utf-8") as f: stt = json.load(f)
    res["W7_selftest"] = {"rc": r.returncode, "engine_ready_s": stt.get("engine_ready_s"),
                          **{k: v for k, v in stt.items() if any(x in k for x in ("replay", "error", "text", "ok"))}}
    time.sleep(2); res["W7_selftest"].update({"servers_left": servers(), "vram_after": vram()})
    p("W7", json.dumps(res["W7_selftest"])[:3000])
except Exception as e:
    import traceback; res["error"] = traceback.format_exc(); p("ERROR", res["error"])
    subprocess.run([PY, APP, "--cmd", "quit"])
res["final_vram"] = vram(); res["final_servers"] = servers()
print("RESULT " + json.dumps(res, indent=1))
