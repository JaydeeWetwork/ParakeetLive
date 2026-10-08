"""Log 51: end-to-end tests of the GPU policy + auto-park (real parakeet_live.pyw driven by its IPC; warm WSL VM).
  M  config migration v1 -> v2 on a temp copy (backup written, other settings kept)
  A1 simulated login with the GPU policy: RAM pre-load (low priority) -> GPU; transcripts exact
  A2 allocator 600 MB while the model is on the GPU -> parks; allocator exits -> unparks ~60 s later; transcripts exact
  A3 allocator while recording -> park waits until the recording stops
  A4 game list (dummy javaw.exe under a "minecraft" folder, like the launcher's bundled Java) -> parks; Record
     while parked -> temporary GPU, re-park after the hold; exit -> unpark
  A4b (log 57) a javaw.exe that is not Minecraft (no "minecraft" in its command line or window titles) -> no park
  A5 low free VRAM (5 x 385 MB, each under the 400 MB trigger) -> parks; Record refused with "GPU busy"; release -> unpark
  A6 login while an allocator already runs -> stays in RAM standby; allocator exits -> GPU
  A7 watcher CPU: widget process CPU over 60 s with the watcher on vs paused; per-poll thread CPU
No microphone is opened (silent test source); the widget window is only shown for the GPU-busy message (hidden right after)."""
import importlib.machinery, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time, wave
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = (os.environ.get("PARAKEET_LIVE_DATA") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ParakeetLive"))
W = os.path.join(REPO, "widget")
sys.path.insert(0, W)
import plive_core as core
PY, PYW = os.path.join(W, ".venv", "Scripts", "python.exe"), os.path.join(W, ".venv", "Scripts", "pythonw.exe")
APP = os.path.join(W, "parakeet_live.pyw")
TOKEN = __import__("secrets").token_hex(16)   # the widget's server requires its token since log 55
ALLOC = os.path.join(REPO, "tests", "gpu_alloc_test.py")
WLOG = os.path.join(DATA, "logs", "widget.log")
STATE = os.path.join(DATA, "logs", "state.json")
TMP = os.path.join(tempfile.gettempdir(), "plive-autopark-test"); os.makedirs(TMP, exist_ok=True)
TRAIN = os.path.join(TMP, "training")
REF = {"libri sample 16k mono.wav": "Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.",
       "espeak sample 16k mono.wav": "Hello JD. This is a local transcription test running on the workstation graphics card. The bundle was a win."}
NOWIN = 0x08000000
def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def vram(): return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, creationflags=NOWIN).stdout.strip())
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
    def absent(self, pat, seconds):
        try: line, _ = self.wait(pat, seconds); return line
        except TimeoutError: return None
def cmd(c): subprocess.run([PY, APP, "--cmd", c], timeout=30, creationflags=NOWIN)
def state():
    cmd("dump"); time.sleep(0.6)
    with open(STATE, encoding="utf-8") as f: return json.load(f)
def launch(*extra): return subprocess.Popen([PYW, APP, "--tray", "--no-save", "--training-dir", TRAIN, "--server-token", TOKEN, *extra])
def quit_widget(proc): cmd("quit"); proc.wait(40); time.sleep(2)
def samples(port):
    c = core.Client(port, token=TOKEN); out = {}
    for f in REF:
        with wave.open(os.path.join(DATA, "test-samples", f)) as w: pcm = w.readframes(w.getnframes())
        r = c.transcribe(pcm); out[f] = {"exact": r["text"] == REF[f], "infer_ms": r["infer_ms"]}
    c.close(); return out
class Alloc:
    def __init__(self, mb, chunk=100, tag="a"):
        self.stop = os.path.join(TMP, f"stop-{tag}-{time.time():.0f}")
        self.p = subprocess.Popen([PY, ALLOC, "--mb", str(mb), "--chunk-mb", str(chunk), "--seconds", "900", "--stop-file", self.stop],
                                  stdout=subprocess.PIPE, text=True, creationflags=NOWIN)
        self.first = self.p.stdout.readline().strip(); self.t = time.perf_counter()
    def close(self):
        open(self.stop, "w").close(); self.p.wait(30); self.t_exit = time.perf_counter()
def short(st): return {k: st.get(k) for k in ("engine", "parked", "park_pending", "status", "recording")}
def park_times(line):
    m = re.search(r"\{.*\}", line); return eval(m.group(0)) if m else line   # our own log dict repr
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
def run(name): return ONLY is None or name in ONLY
res = {}
w = None
try:
    if ONLY:   # a partial re-run still needs a widget hot on the GPU
        L = Log(); w = launch("--prewarm-delay", "3"); L.wait(r"engine ready", 240); time.sleep(5)
    # ---------------- M: migration on a temp copy
    if run("M"):
        p("M config migration")
        ld = importlib.machinery.SourceFileLoader("plw", APP); mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("plw", ld)); ld.exec_module(mod)
        tmpc = os.path.join(TMP, "config.json")
        with open(tmpc, "w", encoding="utf-8") as f: json.dump({"idle_unload_min": 15, "silence_s": 0.6, "device": "Test Mic", "standby_unload_h": 4}, f)
        mod.CONFIG_PATH = tmpc
        msg = mod.migrate_config(); new = json.load(open(tmpc, encoding="utf-8"))
        baks = [x for x in os.listdir(TMP) if x.startswith("config.json.bak-")]
        msg2 = mod.migrate_config()
        res["M_migration"] = {"msg": msg, "after": new, "backup_written": bool(baks), "second_run": msg2,
                              "ok": new.get("idle_unload_min") == 0 and new.get("gpu_always") is True and new.get("silence_s") == 0.6
                                    and new.get("device") == "Test Mic" and new.get("config_version") == 2 and bool(baks) and msg2 is None}
        p("M", json.dumps(res["M_migration"]))
    # ---------------- A1 login with GPU policy
    if run("A1"):
        p("A1 login: RAM pre-load then GPU"); v0 = vram()
        L = Log(); t0 = time.perf_counter(); w = launch("--prewarm-delay", "3")
        L.wait(r"pre-load into RAM scheduled in 3 s, then GPU", 30)
        L.wait(r"pre-loading into RAM \(login", 30)
        l_ram, _ = L.wait(r"model in RAM \(standby", 240); t_ram = time.perf_counter() - t0
        l_mv, _ = L.wait(r"login: model in RAM, moving it to the GPU", 10)
        l_rdy, _ = L.wait(r"engine ready", 60); t_rdy = time.perf_counter() - t0
        time.sleep(2); st = state()
        res["A1_login"] = {"launch_to_ram_s": round(t_ram, 1), "launch_to_gpu_s": round(t_rdy, 1), "ram_line": l_ram[20:140], "ready_line": l_rdy[20:80],
                           "vram_before": v0, "vram_ready": vram(), **short(st), "watch": st.get("watch", {}).get("adapter"),
                           "samples": samples(st["port"])}
        res["A1_login"]["ok"] = st["engine"] == "ready" and all(x["exact"] for x in res["A1_login"]["samples"].values())
        p("A1", json.dumps(res["A1_login"]))
    # ---------------- A2 allocator on GPU -> park -> exit -> unpark
    if run("A2"):
        p("A2 allocator 600 MB"); time.sleep(5)
        a = Alloc(600, tag="a2"); l_trig, t_trig = L.wait(r"auto-park: vram \d+ MB \(python\.exe\)", 40)
        l_park, _ = L.wait(r"auto-park: parked ", 90); time.sleep(1); st = state(); v_park = vram()
        hold = 15; time.sleep(hold); st2 = state()
        a.close(); v_exit = vram()
        l_un, t_un = L.wait(r"auto-park: unparking", 120); l_unp, _ = L.wait(r"auto-park: unparked", 60); st3 = state()
        res["A2_alloc"] = {"alloc": a.first, "alloc_start_to_park_decision_s": round(t_trig, 1), "trigger": l_trig[20:],
                           "park_times": park_times(l_park), "vram_parked_with_alloc": v_park, "state_parked": short(st),
                           "still_parked_after_15s": st2.get("parked") is not None, "vram_after_alloc_exit": v_exit,
                           "unpark_line": l_un[20:150], "unpark_times": park_times(l_unp), "after": short(st3), "vram_back": vram(),
                           "samples": samples(st3["port"])}
        res["A2_alloc"]["alloc_exit_to_unpark_decision_s"] = round(t_un, 1)   # the Log.wait timer starts right after the exit
        res["A2_alloc"]["ok"] = (st3["engine"] == "ready" and st3["parked"] is None and 55 <= t_un <= 75
                                 and all(x["exact"] for x in res["A2_alloc"]["samples"].values()))
        p("A2", json.dumps(res["A2_alloc"]))
    # ---------------- A3 allocator while recording
    if run("A3"):
        p("A3 allocator while recording"); time.sleep(3)
        cmd("silentrec"); time.sleep(1); st = state(); assert st["recording"], "silent recording did not start"
        a = Alloc(600, tag="a3"); l_trig, _ = L.wait(r"auto-park: vram \d+ MB", 40); l_def, _ = L.wait(r"auto-park: deferred", 5)
        early = L.absent(r"moving model to RAM", 15); st_rec = state()
        t_stop = time.perf_counter(); cmd("silentrec")
        l_mv, t_mv = L.wait(r"moving model to RAM \(auto-park", 20)
        l_park, _ = L.wait(r"auto-park: parked ", 90)
        a.close(); l_unp, _ = L.wait(r"auto-park: unparked", 150); st3 = state()
        res["A3_while_recording"] = {"deferred_line": l_def[20:], "parked_while_recording": early, "state_during": short(st_rec),
                                     "stop_to_park_start_s": round(t_mv, 2), "park_times": park_times(l_park), "unpark_times": park_times(l_unp),
                                     "after": short(st3)}
        res["A3_while_recording"]["ok"] = early is None and st_rec["engine"] == "ready" and t_mv < 6 and st3["engine"] == "ready"
        p("A3", json.dumps(res["A3_while_recording"]))
    # ---------------- A4 game list + record while parked (temporary GPU) + re-park
    if run("A4"):
        p("A4b plain javaw.exe (not Minecraft) must not park"); time.sleep(3)
        plain = os.path.join(TMP, "javaw.exe"); shutil.copy2(os.path.join(os.environ["SystemRoot"], "System32", "PING.EXE"), plain)
        g0 = subprocess.Popen([plain, "-n", "600", "127.0.0.1"], stdout=subprocess.DEVNULL, creationflags=NOWIN)
        no_park = L.absent(r"auto-park: game \(javaw\.exe\)", 20); st_b = state(); g0.kill(); g0.wait()
        res["A4b_plain_java"] = {"park_line": no_park, "state": short(st_b),
                                 "ok": no_park is None and not st_b["parked"] and st_b["engine"] == "ready"}
        p("A4b", json.dumps(res["A4b_plain_java"]))
        p("A4 game list dummy javaw.exe (Minecraft)"); time.sleep(3)
        os.makedirs(os.path.join(TMP, "minecraft-sim"), exist_ok=True)
        dummy = os.path.join(TMP, "minecraft-sim", "javaw.exe"); shutil.copy2(os.path.join(os.environ["SystemRoot"], "System32", "PING.EXE"), dummy)
        g = subprocess.Popen([dummy, "-n", "600", "127.0.0.1"], stdout=subprocess.DEVNULL, creationflags=NOWIN); tg = time.perf_counter()
        l_trig, t_trig = L.wait(r"auto-park: game \(javaw\.exe\)", 30); l_park, _ = L.wait(r"auto-park: parked ", 90); st_p = state()
        time.sleep(3)
        t_r = time.perf_counter(); cmd("silentrec"); l_tmp, _ = L.wait(r"record while parked for javaw\.exe: temporary GPU", 10)
        l_rdy, t_rdy = L.wait(r"engine ready", 30); st_t = state(); v_t = vram(); cmd("silentrec")
        l_rep, t_rep = L.wait(r"temporary GPU use over, re-parking", 90); l_park2, _ = L.wait(r"auto-park: parked ", 90); st_p2 = state()
        g.kill(); g.wait(); l_unp, t_unp = L.wait(r"auto-park: unparked", 150); st3 = state()
        res["A4_game"] = {"start_to_park_decision_s": round(t_trig, 1), "trigger": l_trig[20:], "park_times": park_times(l_park),
                          "tray_tip_state": short(st_p), "record_while_parked_line": l_tmp[20:], "record_to_gpu_ready_s": round(t_rdy, 2),
                          "state_temp": short(st_t), "vram_temp": v_t, "stop_to_repark_s": round(t_rep, 1), "repark_times": park_times(l_park2),
                          "state_reparked": short(st_p2), "exit_to_unparked_s": round(t_unp, 1), "unpark_times": park_times(l_unp), "after": short(st3)}
        res["A4_game"]["ok"] = (st_p["parked"] and st_p["parked"]["proc"] == "javaw.exe" and st_t["engine"] == "ready"
                                and st_p2["parked"] and not st_p2["parked"]["temp"] and st3["engine"] == "ready" and st3["parked"] is None)
        p("A4", json.dumps(res["A4_game"]))
    # ---------------- A5 low free VRAM + GPU busy
    if run("A5"):
        p("A5 low free VRAM"); time.sleep(3); v_before = vram()
        al = [Alloc(385, chunk=77, tag=f"a5{i}") for i in range(5)]   # each < 400 MB, together ~1.9 GB
        l_trig, _ = L.wait(r"auto-park: low VRAM", 60); l_park, _ = L.wait(r"auto-park: parked ", 90); time.sleep(5); st_p = state()
        cmd("silentrec"); l_busy, _ = L.wait(r"record while parked refused", 10); time.sleep(0.8); st_b = state(); cmd("hide")
        stay = L.absent(r"auto-park: unparking", 70)          # free while parked (~1.5 GB) < 2200: must stay parked
        for x in al: x.close()
        l_unp, t_unp = L.wait(r"auto-park: unparked", 150); st3 = state()
        res["A5_low_vram"] = {"vram_before": v_before, "allocs": [x.first for x in al], "trigger": l_trig[20:], "park_times": park_times(l_park),
                              "parked_state": {**short(st_p), "gpu_last": st_p.get("gpu_last")}, "busy_line": l_busy[20:],
                              "state_busy": short(st_b), "unparked_early": stay, "exit_to_unparked_s": round(t_unp, 1),
                              "unpark_times": park_times(l_unp), "after": short(st3)}
        res["A5_low_vram"]["ok"] = (st_b["recording"] is False and st_b["status"].startswith("GPU busy") and stay is None
                                    and 55 <= t_unp <= 80 and st3["engine"] == "ready")
        p("A5", json.dumps(res["A5_low_vram"]))
    # ---------------- A7 watcher CPU (before quitting this instance)
    if run("A7"):
        p("A7 watcher CPU"); pid = state()["pid"]
        def cpu(): return float(subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).TotalProcessorTime.TotalMilliseconds"],
                                               capture_output=True, text=True, creationflags=NOWIN).stdout.strip())
        c0 = cpu(); time.sleep(60); c1 = cpu(); st_on = state()
        cmd("pausepark"); time.sleep(2); c2 = cpu(); time.sleep(60); c3 = cpu(); cmd("pausepark"); st_off = state()
        res["A7_cpu"] = {"widget_cpu_ms_per_min_watcher_on": round(c1 - c0), "widget_cpu_ms_per_min_paused": round(c3 - c2),
                         "watch": st_on.get("watch"), "pct_one_core_on": round((c1 - c0) / 600, 3), "pct_one_core_paused": round((c3 - c2) / 600, 3),
                         "paused_flag_after": st_off.get("park_paused")}
        p("A7", json.dumps(res["A7_cpu"]))
        quit_widget(w); w = None
    # ---------------- A6 login while a heavy GPU app runs
    if run("A6"):
        p("A6 login with allocator already running"); a = Alloc(600, tag="a6"); time.sleep(2)
        L = Log(); t0 = time.perf_counter(); w = launch("--prewarm-delay", "3")
        l_stay, t_stay = L.wait(r"login: GPU busy \(parked for python\.exe\)", 240); time.sleep(10); st_s = state(); v_s = vram()
        a.close(); l_un, t_un = L.wait(r"auto-park: unparked", 150); st3 = state()
        res["A6_login_busy"] = {"launch_to_standby_s": round(t_stay, 1), "line": l_stay[20:], "state": short(st_s), "vram_standby_with_alloc": v_s,
                                "exit_to_unparked_s": round(t_un, 1), "unpark_times": park_times(l_un), "after": short(st3), "samples": samples(st3["port"])}
        res["A6_login_busy"]["ok"] = st_s["engine"] == "standby" and st3["engine"] == "ready" and all(x["exact"] for x in res["A6_login_busy"]["samples"].values())
        p("A6", json.dumps(res["A6_login_busy"]))
        quit_widget(w); w = None
    if w is not None:
        quit_widget(w); w = None
except Exception:
    import traceback; res["error"] = traceback.format_exc(); p("ERROR", res["error"])
    if w is not None:
        try: quit_widget(w)
        except Exception: pass
res["final_vram"] = vram()
res["summary"] = {k: v.get("ok") for k, v in res.items() if isinstance(v, dict) and "ok" in v}
print("RESULT " + json.dumps(res, indent=1, default=str))
