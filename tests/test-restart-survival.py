"""Log 57: the unsent message and kept retries survive a restart and a crash.

Each phase is a separate process running a sandboxed App (same sandbox as test-code-review.py: own data
dir, temp config, temp training folder, fake clipboard - the real one is never touched -, fake microphone,
no tray icon, no global hotkey, no model; the speech server is a fake client). Phases share a temp state
folder (--state-dir), which is what <data folder>\\state is for the real widget.
  S1 quit with an unsent message that was copied and taken elsewhere, plus a kept failed utterance ->
     next start: the box text is back, nothing is copied at start, the next recording still starts fresh
     (Ctrl+Z brings it back), the kept utterance is transcribed once the engine is ready and its files go
  S2 crash right after the draft was saved (process killed, no quit) -> the draft is back
  S3 a retry older than 1 h and a week-old draft are not restored, and their files are removed
  S4 only draft.json and pending/ in the state folder; no temp files left
  S5 the self-test and --no-save runs never use the state folder
  S6 the self-test window is click-through (layered + transparent, alpha kept); a normal window is not.
     Checked on the App's own _apply_style with the window mapped at -20000,-20000 (on no monitor)
"""
import importlib.machinery, importlib.util, json, os, shutil, subprocess, sys, tempfile, threading, time, traceback  # noqa: E401
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(REPO, "widget")
TMP = os.path.join(tempfile.gettempdir(), "plive-restart-test")
STATE, STATE2, STATE3 = (os.path.join(TMP, n) for n in ("state", "state-crash", "state-old"))
PHASE = sys.argv[1] if len(sys.argv) > 1 else None

if PHASE is None:                       # ---- driver: run the phases one after another
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(os.path.join(TMP, "data", "logs"))
    sys.path.insert(0, W)
    import plive_core as core          # noqa: E402
    clip0 = core.get_clipboard()
    out = {}
    for ph in ("p1", "p2", "p3", "p4", "p5", "p6", "p7"):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), ph], capture_output=True, text=True, timeout=120)
        line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
        out[ph] = json.loads(line[-1][7:]) if line else {"rc": r.returncode, "stderr": r.stderr[-1500:]}
        if ph == "p3":
            out[ph]["killed_rc"] = r.returncode
        print(ph, json.dumps(out[ph])[:900], flush=True)
    checks = {
        "S1_restored_after_quit": out["p2"].get("S1", False),
        "S2_restored_after_crash": out["p3"].get("killed_rc") == 3 and out["p4"].get("S2", False),
        "S3_expired_not_restored": out["p5"].get("S3", False),
        "S4_state_folder_clean": out["p2"].get("S4", False),
        "S5_selftest_and_no_save_never_use_state": out["p6"].get("S5", False),
        "S6_selftest_window_click_through": out["p7"].get("S6", False) and out["p6"].get("S6_normal", False),
        "no_phase_errors": not any("error" in out[ph] or "stderr" in out[ph] for ph in out if ph != "p3"),
        "sandbox_never_clickable": all(out[ph].get("guard", {"ok": True}).get("ok", False) for ph in out),
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    real_ok = core.get_clipboard() == clip0
    print("real_clipboard_untouched", real_ok)
    ok = all(checks.values()) and real_ok
    print("ALL PASS" if ok else "SOME FAILED", sum(checks.values()), "/", len(checks))
    sys.exit(0 if ok else 1)

# ---- a phase: one sandboxed App
os.environ["PARAKEET_LIVE_DATA"] = os.path.join(TMP, "data")
sys.path.insert(0, W)
import plive_core as core   # noqa: E402
import plive_tray as trayw  # noqa: E402
fake = {"clip": None, "mics": 0, "sets": 0}
core.set_clipboard = lambda t: (fake.__setitem__("clip", t), fake.__setitem__("sets", fake["sets"] + 1), True)[2]
core.get_clipboard = lambda: fake["clip"]
core.kill_stale_servers = lambda: False


class FakeMic:
    def __init__(self, device, on_audio, on_error):
        self.info = "fake mic (test, no audio device)"

    def start(self):
        fake["mics"] += 1

    def stop(self):
        pass


class FakeClient:
    def transcribe(self, pcm):
        return {"text": "Retried words.", "infer_ms": 1}


core.MicCapture = FakeMic
core.parse_hotkey = lambda s: (_ for _ in ()).throw(ValueError("test: no global hotkey"))
trayw.CLASS_NAME = "ParakeetLiveTrayRestartTest"
trayw.Tray._add = lambda self: None
loader = importlib.machinery.SourceFileLoader("plw", os.path.join(W, "parakeet_live.pyw"))
spec = importlib.util.spec_from_loader("plw", loader)
plw = importlib.util.module_from_spec(spec)
loader.exec_module(plw)
import sandbox_guard  # noqa: E402  (log 57: sandbox windows click-through, no menus, checked after each show)
sandbox_guard.install(plw)
assert plw.STATE_PATH.startswith(TMP), plw.STATE_PATH
cfg = dict(plw.DEFAULTS)
cfg.update({"prewarm_login": False, "gpu_always": False, "auto_park": False, "autocopy": True,
            "clear_on_return": True, "config_version": plw.CONFIG_VERSION, "opacity": 0.02})
plw.CONFIG_PATH = os.path.join(TMP, "config.json")
json.dump(cfg, open(plw.CONFIG_PATH, "w"), indent=1)
plw.App.load_model = lambda self, *a, **k: None
plw.App.prewarm = lambda self, *a, **k: None
plw.App._gpu_watch_loop = lambda self: None
TRAIN = os.path.join(TMP, "training")
PCM, PCM2 = bytes(32000), bytes(range(256)) * 125
OTHER = (0x7F0000A1, False)

if PHASE == "p5":                       # old files, written before the App starts
    st = core.SessionStore(STATE3)
    st.save_pending("old-1", PCM2, {"dur": 1.0, "tries": 1, "failed_at": time.time() - 7200})
    st.save_draft({"text": "Very old draft.", "spans": [], "cor": {}})
    d = json.load(open(st.draft_path))
    d["saved"] = time.time() - 8 * 86400
    json.dump(d, open(st.draft_path, "w"))

if PHASE == "p7":
    plw.App._selftest = lambda self: None            # only the window style is checked here

sd = {"p1": STATE, "p2": STATE, "p3": STATE2, "p4": STATE2, "p5": STATE3}.get(PHASE)
argv = ["--tray", "--no-save", "--training-dir", TRAIN] + (["--state-dir", sd] if sd else []) \
    + (["--selftest", os.path.join(TMP, "selftest-unused.json")] if PHASE == "p7" else [])
app = plw.App(plw.build_parser().parse_args(argv))
res = {}


def ui(fn): return app._call(fn)
def box(): return ui(lambda: app.text.get("1.0", "end-1c"))
def wait(c, t=3): return app._wait(c, t)
def text(t, pcm=b""): ui(lambda: app._handle(("text", t, time.perf_counter(), 0, 0, 1.0, pcm)))
def offscreen_style():
    """Map the window on no monitor (-20000,-20000; tool window: no taskbar button, not in Alt+Tab), run the
    App's own _apply_style (not the guard's), read the ex-style and layered alpha, withdraw again."""
    import ctypes
    u = ctypes.windll.user32

    def run():
        app.root.geometry("+-20000+-20000")
        app.root.attributes("-alpha", 1.0)             # not layered, like a widget at full opacity
        app.root.deiconify()
        app.root.update_idletasks()
        h = app._hwnd()
        u.SetWindowLongW(h, -20, u.GetWindowLongW(h, -20) & ~0x20)
        sandbox_guard.ORIGINALS["_apply_style"](app)
        st = u.GetWindowLongW(h, -20)
        a, fl = ctypes.c_ubyte(), ctypes.c_ulong()
        u.GetLayeredWindowAttributes(h, None, ctypes.byref(a), ctypes.byref(fl))
        x, y = app.root.winfo_x(), app.root.winfo_y()
        app.root.withdraw()
        return {"transparent": bool(st & 0x20), "layered": bool(st & 0x80000), "alpha": a.value,
                "toolwindow": bool(st & 0x80), "at": (x, y)}
    return ui(run)


def pending_files(d): return sorted(os.listdir(os.path.join(d, "pending"))) if os.path.isdir(os.path.join(d, "pending")) else []


def body():
    wait(lambda: app.tray.ready.is_set(), 5)
    time.sleep(0.5)
    if PHASE == "p1":
        app._fg_override = OTHER
        text("Unsent words here.", PCM)
        c = wait(lambda: fake["clip"] == "Unsent words here." and app.cor_armed, 3)
        app._fg_override = (0x7F0000B2, False)
        left = wait(lambda: app.cor_left, 3)
        uid = core.new_utterance_id()
        ui(lambda: app._handle(("job_error", "test: server hiccup", (PCM2, 1.0, uid))))
        res.update(copied=c, left=left, kept=len(app.failed), files=pending_files(STATE))
    elif PHASE == "p2":
        time.sleep(1.0)                 # > the 400 ms auto-copy debounce: nothing may be copied at start
        b0, sets0, left0 = box(), fake["sets"], app.cor_left
        kept0, files0 = len(app.failed), pending_files(STATE)
        app._fg_override = OTHER
        ui(lambda: app._tray_event("hotkey"))            # next recording: starts fresh
        fresh = box() == "" and app.recording
        ui(lambda: app._tray_event("hotkey"))            # stop
        z = ui(lambda: app._on_ctrl_z())
        back = box() == "Unsent words here."
        app.tclient = FakeClient()
        app.engine = "ready"
        app.ready_evt.set()
        ui(app._requeue_failed)
        got = wait(lambda: "Retried words." in app.text.get("1.0", "end-1c"), 5)
        gone = wait(lambda: pending_files(STATE) == [], 3)
        res["S1"] = (b0 == "Unsent words here." and sets0 == 0 and left0 and kept0 == 1 and len(files0) == 2
                     and fresh and z == "break" and back and got and gone)
        res.update(box_at_start=b0, copies_at_start=sets0, left_restored=left0, kept_restored=kept0,
                   files_at_start=len(files0), fresh=fresh, ctrl_z=z, restored=back, retried=got, files_gone=gone)
        ui(app._save_draft)
        names = sorted(os.listdir(STATE))
        res["S4"] = names == ["draft.json", "pending"] and not any(n.endswith(".tmp") for n in names)
        res["state_folder"] = names
    elif PHASE == "p3":
        app._fg_override = OTHER
        text("Crash words.")
        wait(lambda: fake["clip"] == "Crash words.", 3)
        saved = wait(lambda: os.path.exists(os.path.join(STATE2, "draft.json")), 4)   # 1.5 s debounce
        print("RESULT " + json.dumps({"draft_saved_before_kill": saved}), flush=True)
        os._exit(3)                     # no quit, no cleanup: like a crash or a killed process
    elif PHASE == "p4":
        time.sleep(1.0)
        res["S2"] = box() == "Crash words." and fake["sets"] == 0
        res.update(box_at_start=box(), copies_at_start=fake["sets"])
    elif PHASE == "p5":
        time.sleep(0.5)
        res["S3"] = box() == "" and len(app.failed) == 0 and pending_files(STATE3) == [] \
            and not os.path.exists(os.path.join(STATE3, "draft.json"))
        res.update(box=box(), kept=len(app.failed), files=pending_files(STATE3))
    elif PHASE == "p6":
        P = plw.build_parser().parse_args
        sel = plw.session_dir(P(["--selftest", "x.json"]))
        nos = plw.session_dir(P(["--no-save"]))
        normal = plw.session_dir(P([]))
        res["S5"] = app.store is None and sel is None and nos is None \
            and normal == os.path.join(plw.DATA_DIR, "state")
        b = offscreen_style()
        res["S6_normal"] = not b["transparent"] and b["toolwindow"] and b["at"][0] < -10000
        res["style_normal"] = b
        res.update(no_save_store=str(app.store), selftest=sel, no_save=nos, normal=normal)
    elif PHASE == "p7":
        b = offscreen_style()
        want = max(1, min(255, int(255 * float(app.cfg["opacity"]))))
        res["S6"] = b["transparent"] and b["layered"] and b["alpha"] == want and b["toolwindow"] and b["at"][0] < -10000
        res.update(style_selftest=b, alpha_expected=want)


def runner():
    try:
        body()
    except Exception:
        res["error"] = traceback.format_exc()
    finally:
        app._call(app.quit, wait=False)


threading.Thread(target=runner, daemon=True).start()
app.run()
res["guard"] = sandbox_guard.summary()
print("RESULT " + json.dumps(res, default=str), flush=True)
