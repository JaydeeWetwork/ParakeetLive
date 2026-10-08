"""Log 52: batch transcriber <-> live widget GPU hold (real batch wrapper, real batch, real widget via IPC; warm VM).
  B1 real batch (libri sample) with the model hot on the GPU: hold -> park, job, release -> back on the GPU;
     tray tip; transcript vs reference; VRAM peak; WSL RAM
  B2 failing job (missing file): finally still releases, model returns
  B3 launcher killed mid-job (child wsl.exe keeps running): hold moves to the child, ends with it, model returns
  B4 launcher + children killed (taskkill /T): hold ends by itself, no transcribe.py left in WSL, model returns
  B6 WSL-orphan safety net: a detached dummy whose command line is /opt/parakeet/scripts/transcribe.py
     runs in WSL while a hold's process exits; the widget must keep the model parked until it is gone
  B5 no widget running: batch runs exactly as before, same transcript
No microphone is opened; no widget window is shown."""
import atexit, json, os, re, subprocess, sys, tempfile, threading, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = (os.environ.get("PARAKEET_LIVE_DATA") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ParakeetLive"))
W = os.path.join(REPO, "widget")
PY, PYW = os.path.join(W, ".venv", "Scripts", "python.exe"), os.path.join(W, ".venv", "Scripts", "pythonw.exe")
APP = os.path.join(W, "parakeet_live.pyw")
WRAP = os.path.join(REPO, "batch", "Transcribe-WithGpuHold.ps1")
WLOG = os.path.join(DATA, "logs", "widget.log")
STATE = os.path.join(DATA, "logs", "state.json")
TMP = os.path.join(tempfile.gettempdir(), "plive-batch-test"); os.makedirs(TMP, exist_ok=True)
SAMPLE = os.path.join(DATA, "test-samples", "libri sample 16k mono.wav")
REF = "Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait."
NOWIN = 0x08000000
def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
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
                if rx.search(line): return line.strip(), time.perf_counter()
            time.sleep(0.05)
        raise TimeoutError(pat)
def cmd(c): subprocess.run([PY, APP, "--cmd", c], timeout=30, creationflags=NOWIN)
def state():
    cmd("dump"); time.sleep(0.6)
    with open(STATE, encoding="utf-8") as f: return json.load(f)
def launch(*extra): return subprocess.Popen([PYW, APP, "--tray", "--no-save", "--training-dir", os.path.join(TMP, "training"), *extra])
def quit_widget(proc): cmd("quit"); proc.wait(40); time.sleep(2)
def short(st): return {k: st.get(k) for k in ("engine", "parked", "status", "tray_tip", "batch_holds")}
class Sampler:
    """nvidia-smi every 250 ms (one process) + WSL MemAvailable every 1 s (one wsl process)."""
    def __init__(self):
        self.vram, self.avail = [], []
        self.p1 = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "250"],
                                   stdout=subprocess.PIPE, text=True, creationflags=NOWIN)
        self.p2 = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04", "-e", "bash", "-c",
                                    "while true; do awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; sleep 1; done"],
                                   stdout=subprocess.PIPE, text=True, creationflags=NOWIN)
        atexit.register(lambda: (self.p1.kill(), self.p2.kill()))   # a failed section must not leak samplers
        threading.Thread(target=self._rd, args=(self.p1, self.vram), daemon=True).start()
        threading.Thread(target=self._rd, args=(self.p2, self.avail), daemon=True).start()
    def _rd(self, pr, out):
        for line in pr.stdout:
            try: out.append((time.perf_counter(), int(line.strip())))
            except ValueError: pass
    def stop(self):
        self.p1.kill(); self.p2.kill()
        return {"vram_peak_mib": max((v for _, v in self.vram), default=None), "vram_samples": len(self.vram),
                "wsl_mem_available_min_mib": min((v for _, v in self.avail), default=None)}
def batch(path, outdir):
    return subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", WRAP, path, "-OutDir", outdir],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            creationflags=NOWIN)
def transcript(outdir):
    f = os.path.join(outdir, "libri sample 16k mono.txt")
    return open(f, encoding="utf-8").read().strip() if os.path.exists(f) else None
def wsl_transcribe_running():
    # no "bash -c" here: a shell whose command line contains the pattern matches itself (that fooled run 1's B4)
    r = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "-e", "pgrep", "-f", "/opt/parakeet/scripts/transcribe.py"],
                       capture_output=True, text=True, creationflags=NOWIN)
    return r.stdout.split()
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
def run(name): return ONLY is None or name in ONLY
import shutil
for _d in ("out-B1", "out-B2", "out-B3", "out-B4", "out-B5"):
    shutil.rmtree(os.path.join(TMP, _d), ignore_errors=True)
res = {}
w = None
try:
    p("start test widget (login path, delay 3 s)"); L = Log(); w = launch("--prewarm-delay", "3"); L.wait(r"engine ready", 240); time.sleep(5)
    # ---------------- B1
    if run("B1"):
        p("B1 real batch with the model on the GPU"); out = os.path.join(TMP, "out-B1"); L = Log()
        S = Sampler(); time.sleep(1); t0 = time.perf_counter(); b = batch(SAMPLE, out)
        l_hold, t_hold = L.wait(r"batch hold from pid \d+ \(launcher\)", 60)
        l_park, t_park = L.wait(r"auto-park: parked ", 90); time.sleep(0.5); st_mid = state()
        l_rel, t_rel = L.wait(r"batch hold released: pid \d+ \(released by the launcher\)", 400)
        outp = b.communicate(timeout=60)[0]; t_end = time.perf_counter()
        l_unp, t_unp = L.wait(r"auto-park: unparked", 120); st_after = state(); s = S.stop()
        off = re.search(r"off the GPU for the batch job \(([\d.]+) s\)", outp)
        res["B1_batch"] = {"rc": b.returncode, "launcher_says_off_gpu_after_s": float(off.group(1)) if off else None,
                           "park_event": l_park[20:], "start_to_hold_s": round(t_hold - t0, 2),
                           "hold_to_model_parked_in_ram_s": round(t_park - t_hold, 2), "job_total_s": round(t_end - t0, 1),
                           "release_to_back_on_gpu_s": round(t_unp - t_rel, 2), "unpark_event": l_unp[20:], "during": short(st_mid),
                           "after": short(st_after), "transcript": transcript(out), "transcript_exact": transcript(out) == REF,
                           "device_not_ready": "device not ready" in outp, **s,
                           "batch_lines": [x for x in outp.splitlines() if x.startswith(("Parakeet Live", "[parakeet]"))]}
        r = res["B1_batch"]
        r["ok"] = (r["rc"] == 0 and r["transcript_exact"] and not r["device_not_ready"] and r["vram_peak_mib"] < 4096
                   and st_mid["parked"] and st_mid["parked"].get("hold") and "batch transcription" in (st_mid.get("tray_tip") or "")
                   and st_after["engine"] == "ready" and st_after["parked"] is None and not st_after["batch_holds"])
        p("B1", json.dumps(r))
    # ---------------- B2
    if run("B2"):
        p("B2 failing job (missing file)"); time.sleep(3); L = Log(); t0 = time.perf_counter()
        b = batch(os.path.join(TMP, "does-not-exist.wav"), os.path.join(TMP, "out-B2")); outp = b.communicate(timeout=120)[0]
        l_rel, t_rel = L.wait(r"batch hold released: pid \d+ \(released by the launcher\)", 30)
        l_unp, t_unp = L.wait(r"auto-park: unparked", 120); st_after = state()
        res["B2_failure"] = {"rc": b.returncode, "error_shown": "File not found" in outp, "release_line": l_rel[20:],
                             "release_to_back_on_gpu_s": round(t_unp - t_rel, 2), "after": short(st_after)}
        res["B2_failure"]["ok"] = b.returncode != 0 and st_after["engine"] == "ready" and not st_after["batch_holds"]
        p("B2", json.dumps(res["B2_failure"]))
    # ---------------- B3
    if run("B3"):
        p("B3 launcher killed mid-job (child keeps running)"); time.sleep(3); out = os.path.join(TMP, "out-B3"); L = Log()
        S = Sampler(); b = batch(SAMPLE, out)
        L.wait(r"batch hold from pid \d+ \(launcher\)", 60); L.wait(r"auto-park: parked ", 90); time.sleep(3)
        t_kill = time.perf_counter(); b.kill()       # TerminateProcess on powershell.exe only
        l_move, t_move = L.wait(r"batch hold from pid \d+ \(wsl\.exe, child of exited pid", 15)
        l_r1, _ = L.wait(r"batch hold released: pid \d+ \(process exited without releasing; still running: wsl\.exe", 15)
        st_mid = state()
        l_r2, t_r2 = L.wait(r"batch hold released: pid \d+ \(process exited without releasing\)", 400)
        l_unp, t_unp = L.wait(r"auto-park: unparked", 120); st_after = state(); s = S.stop()
        res["B3_killed_launcher"] = {"kill_to_hold_moved_s": round(t_move - t_kill, 2), "moved": l_move[20:], "launcher_release": l_r1[20:],
                                     "during": short(st_mid), "child_end_release": l_r2[20:], "kill_to_child_end_s": round(t_r2 - t_kill, 1),
                                     "child_end_to_back_on_gpu_s": round(t_unp - t_r2, 2), "after": short(st_after),
                                     "transcript_exact": transcript(out) == REF, **s}
        r = res["B3_killed_launcher"]
        r["ok"] = st_mid["parked"] is not None and st_after["engine"] == "ready" and not st_after["batch_holds"] and r["vram_peak_mib"] < 4096
        p("B3", json.dumps(r))
    # ---------------- B4
    if run("B4"):
        p("B4 launcher and children killed (taskkill /T)"); time.sleep(3); L = Log(); S = Sampler(); b = batch(SAMPLE, os.path.join(TMP, "out-B4"))
        L.wait(r"batch hold from pid \d+ \(launcher\)", 60); L.wait(r"auto-park: parked ", 90); time.sleep(3)
        t_kill = time.perf_counter(); subprocess.run(["taskkill", "/PID", str(b.pid), "/T", "/F"], capture_output=True, creationflags=NOWIN)
        l_r, t_r = L.wait(r"batch hold released: pid \d+ \(process exited without releasing\)", 15)
        left = wsl_transcribe_running(); st_mid = state()
        l_unp, t_unp = L.wait(r"auto-park: unparked", 400); left_at_unpark = wsl_transcribe_running(); time.sleep(5)
        st_after = state(); s = S.stop()
        res["B4_killed_tree"] = {"kill_to_release_s": round(t_r - t_kill, 2), "release": l_r[20:], "transcribe_py_left_in_wsl": left,
                                 "release_to_back_on_gpu_s": round(t_unp - t_r, 2), "unpark": l_unp[20:],
                                 "transcribe_py_running_at_unpark": left_at_unpark, "after": short(st_after), **s}
        r = res["B4_killed_tree"]
        r["ok"] = (not left_at_unpark and st_after["engine"] == "ready" and not st_after["batch_holds"]
                   and r["vram_peak_mib"] < 4096)
        p("B4", json.dumps(r))
    # ---------------- B6
    if run("B6"):
        p("B6 WSL orphan safety net (dummy transcribe.py, 30 s)"); time.sleep(3); L = Log()
        fake = "exec -a '/opt/parakeet/venv/bin/python /opt/parakeet/scripts/transcribe.py --plive-test-dummy' sleep 30"
        subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "-e", "setsid", "-f", "bash", "-c", fake],
                       capture_output=True, timeout=30, creationflags=NOWIN)
        t_orphan = time.perf_counter(); dummy = wsl_transcribe_running()
        holder = subprocess.Popen(["ping", "-n", "8", "127.0.0.1"], stdout=subprocess.DEVNULL, creationflags=NOWIN)
        hr = subprocess.run([PY, APP, "--cmd", "batchhold", "--pid", str(holder.pid), "--wait", "120"],
                            capture_output=True, text=True, timeout=150, creationflags=NOWIN)
        L.wait(r"batch hold from pid %d " % holder.pid, 30)
        l_r, t_r = L.wait(r"batch hold released: pid %d \(process exited without releasing\)" % holder.pid, 60)
        l_w, t_w = L.wait(r"transcribe\.py still running inside WSL", 20); st_mid = state()
        l_unp, t_unp = L.wait(r"auto-park: unparked", 300); left_at_unpark = wsl_transcribe_running(); time.sleep(3)
        st_after = state()
        res["B6_wsl_orphan"] = {"dummy_pids": dummy, "hold_cli_rc": hr.returncode, "release": l_r[20:], "wait_line": l_w[20:],
                                "during_wait": short(st_mid), "wsl_hold_during": st_mid.get("wsl_hold"),
                                "dummy_start_to_unpark_s": round(t_unp - t_orphan, 1), "release_to_unpark_s": round(t_unp - t_r, 1),
                                "transcribe_py_running_at_unpark": left_at_unpark, "after": short(st_after)}
        r = res["B6_wsl_orphan"]
        r["ok"] = (bool(dummy) and hr.returncode == 0 and st_mid["parked"] is not None and st_mid.get("wsl_hold") is True
                   and not left_at_unpark and st_after["engine"] == "ready" and not st_after.get("wsl_hold"))
        p("B6", json.dumps(r))
    if w is not None and run("B5"):
        quit_widget(w); w = None
    # ---------------- B5
    if run("B5"):
        p("B5 no widget running"); out = os.path.join(TMP, "out-B5"); S = Sampler(); t0 = time.perf_counter(); b = batch(SAMPLE, out)
        outp = b.communicate(timeout=400)[0]; s = S.stop()
        res["B5_no_widget"] = {"rc": b.returncode, "job_total_s": round(time.perf_counter() - t0, 1), "says_not_running": "is not running" in outp,
                               "transcript_exact": transcript(out) == REF, **s}
        res["B5_no_widget"]["ok"] = b.returncode == 0 and res["B5_no_widget"]["transcript_exact"]
        p("B5", json.dumps(res["B5_no_widget"]))
    if w is not None:
        quit_widget(w); w = None
except Exception:
    import traceback; res["error"] = traceback.format_exc(); p("ERROR", res["error"])
    if w is not None:
        try: quit_widget(w)
        except Exception: pass
res["summary"] = {k: v.get("ok") for k, v in res.items() if isinstance(v, dict) and "ok" in v}
print("RESULT " + json.dumps(res, indent=1, default=str))
