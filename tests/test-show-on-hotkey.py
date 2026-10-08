"""Log 54: show_on_hotkey, tested next to the running widget without disturbing it.

Sandboxed second App in this process (same sandbox as test-clear-on-return.py: own data dir, temp
config, temp training dir, no tray icon, own IPC class, model loading disabled, fake clipboard, fake
microphone), drawn at 2 % opacity at Jaydee's saved position/size so nothing visibly pops up.
The hotkey is REAL: the sandbox registers F24 (a key no keyboard has) and the test presses it with
SendInput, so the sandbox gets exactly the foreground rights a real Ctrl+Alt+Space press gives it.
Pass = the widget never becomes the foreground window (focus stays in Jaydee's app).
  S1 tray-only start (never mapped) + hotkey -> shown at saved geometry, topmost, tool window
     (no taskbar button), foreground not ours, recording (fake mic)
  S2 hotkey again -> stops, stays visible
  S3 hidden via hide() + hotkey -> shown again, same checks
  S4 minimized-to-tray + hotkey -> shown at the saved geometry (not the bogus minimized one)
  S5 already visible: hotkey only toggles (show() not called)
  S6 show_on_hotkey off: stays hidden, still records
  S7 with clear_on_return: copied + switched + hidden + hotkey -> shown, cleared, recording
  S8 config migration v1/v3/v4 -> current (v5 since log 57; backup, other keys kept, show_on_hotkey off
     stays off); a current config is untouched
"""
import ctypes, ctypes.wintypes as wt
import importlib.machinery, importlib.util, json, os, shutil, sys, tempfile, threading, time, traceback
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(REPO, "widget")
TMP = os.path.join(tempfile.gettempdir(), "plive-shk-test")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(os.path.join(TMP, "data", "logs"))
os.environ["PARAKEET_LIVE_DATA"] = os.path.join(TMP, "data")
sys.path.insert(0, W)
import plive_core as core   # noqa: E402
import plive_tray as trayw  # noqa: E402

real_get_clip = core.get_clipboard
CLIP_BEFORE = real_get_clip()
fake = {"clip": None, "mics": 0}
core.set_clipboard = lambda t: (fake.__setitem__("clip", t), True)[1]
core.get_clipboard = lambda: fake["clip"]
core.kill_stale_servers = lambda: False


class FakeMic:
    def __init__(self, device, on_audio, on_error):
        self.info = "fake mic (test, no audio device)"

    def start(self):
        fake["mics"] += 1

    def stop(self):
        pass


core.MicCapture = FakeMic
core.parse_hotkey = lambda s: (0, 0x87)        # F24 (no physical key): a real global hotkey
trayw.CLASS_NAME = "ParakeetLiveTrayShkTest"
trayw.Tray._add = lambda self: None          # no tray icon

loader = importlib.machinery.SourceFileLoader("plw", os.path.join(W, "parakeet_live.pyw"))
spec = importlib.util.spec_from_loader("plw", loader)
plw = importlib.util.module_from_spec(spec)
loader.exec_module(plw)
import sandbox_guard  # noqa: E402  (log 57: sandbox windows click-through, no menus, checked after each show)
sandbox_guard.install(plw)
assert plw.STATE_PATH.startswith(TMP), plw.STATE_PATH
cfg = dict(plw.DEFAULTS)
cfg.update({"prewarm_login": False, "gpu_always": False, "auto_park": False, "autocopy": True,
            "clear_on_return": True, "show_on_hotkey": True, "config_version": plw.CONFIG_VERSION,
            "opacity": 0.02})                       # practically invisible: no pop-up for Jaydee
try:                                         # his saved position and size (read-only)
    _real = json.load(open(os.path.join(W, "config.json"), encoding="utf-8"))
    cfg.update({k: _real[k] for k in ("x", "y", "w", "h") if _real.get(k) is not None})
except Exception:
    pass
plw.CONFIG_PATH = os.path.join(TMP, "config.json")
json.dump(cfg, open(plw.CONFIG_PATH, "w"), indent=1)
plw.App.load_model = lambda self, *a, **k: None
plw.App.prewarm = lambda self, *a, **k: None
plw.App._gpu_watch_loop = lambda self: None
TRAIN = os.path.join(TMP, "training")
args = plw.build_parser().parse_args(["--tray", "--no-save", "--training-dir", TRAIN])
app = plw.App(args)


u32 = ctypes.WinDLL("user32", use_last_error=True)
u32.GetForegroundWindow.restype = wt.HWND
u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _IU(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _IU)]


u32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
HOTKEYS = []                                  # perf_counter of every real WM_HOTKEY delivered
_orig_tray_event = app._tray_event


def _te(ev):
    if ev == "hotkey":
        HOTKEYS.append(time.perf_counter())
    return _orig_tray_event(ev)


app._tray_event = _te
SHOWS = []
_orig_show = app.show
app.show = lambda load=True: (SHOWS.append(time.perf_counter()), _orig_show(load))[1]
res, results = {}, {}
SAVED = (cfg["x"], cfg["y"], cfg["w"], cfg["h"])


def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def ui(fn): return app._call(fn)
def wait(c, t=3): return app._wait(c, t)


def fg_owner():
    h = u32.GetForegroundWindow()
    pid = wt.DWORD(0)
    if h:
        u32.GetWindowThreadProcessId(h, ctypes.byref(pid))
    return h, pid.value == os.getpid()


def press_hotkey():
    """F24 down/up through SendInput; returns True if the sandbox got a real WM_HOTKEY."""
    n = len(HOTKEYS)
    ins = (INPUT * 2)()
    for i, flags in enumerate((0, 2)):          # 2 = KEYEVENTF_KEYUP
        ins[i].type = 1
        ins[i].u.ki = KEYBDINPUT(0x87, 0, flags, 0, 0)
    sent = u32.SendInput(2, ins, ctypes.sizeof(INPUT))
    ok = wait(lambda: len(HOTKEYS) > n, 2)
    if not ok:                                  # e.g. an elevated app in front blocks injected input
        p(f"  (no WM_HOTKEY after SendInput={sent}; using the tray-event path directly)")
        ui(lambda: _te("hotkey"))
    return ok


def geom(): return ui(lambda: (app.root.winfo_x(), app.root.winfo_y(), app.root.winfo_width(), app.root.winfo_height()))


def shown_checks(fg0):
    """Sample the foreground for 1 s after the hotkey; the widget must never be in front."""
    samples = []
    for _ in range(10):
        samples.append(fg_owner())
        time.sleep(0.1)
    st = ui(app.exstyle)
    return {"visible": app.visible, "iswindowvisible": st.get("iswindowvisible"), "toolwindow": st.get("toolwindow"),
            "appwindow": st.get("appwindow"), "topmost": st.get("topmost"), "geometry": geom(), "saved": SAVED,
            "recording": app.recording, "foreground_ever_ours": any(o for _, o in samples),
            "foreground_unchanged": all(h == fg0 for h, _ in samples)}


def shown_ok(c):
    return (c["visible"] and c["iswindowvisible"] and c["toolwindow"] and not c["appwindow"] and c["topmost"]
            and tuple(c["geometry"]) == tuple(SAVED) and c["recording"] and not c["foreground_ever_ours"])


def check(name, ok, **info):
    results[name] = bool(ok)
    res[name] = {"ok": bool(ok), **info}
    p(name, "PASS" if ok else "FAIL", json.dumps(info, default=str)[:900])


def stop():
    if app.recording:
        press_hotkey()
        wait(lambda: not app.recording, 2)


def body():
    wait(lambda: app.tray.ready.is_set(), 5)
    time.sleep(0.6)
    res["hotkey_registered"] = app.tray.hotkey_ok
    # S1 tray-only start, never mapped
    never = not app.visible and not ui(app.exstyle).get("iswindowvisible")
    fg0, own0 = fg_owner()
    real = press_hotkey()
    wait(lambda: app.recording, 2)
    c = shown_checks(fg0)
    check("S1_tray_only_hotkey_shows_without_focus", never and not own0 and shown_ok(c), real_hotkey=real,
          started_hidden=never, **c)
    # S2
    press_hotkey()
    wait(lambda: not app.recording, 2)
    time.sleep(0.3)
    check("S2_stop_keeps_visible", not app.recording and app.visible and ui(app.exstyle).get("iswindowvisible"),
          visible=app.visible)
    # S3
    ui(app.hide)
    time.sleep(0.4)
    hidden = not ui(app.exstyle).get("iswindowvisible")
    fg0, _ = fg_owner()
    real = press_hotkey()
    wait(lambda: app.recording, 2)
    c = shown_checks(fg0)
    check("S3_hidden_hotkey_shows_without_focus", hidden and shown_ok(c), real_hotkey=real, **c)
    stop()
    # S4 minimized -> caught -> hidden to tray
    # what Win+D / Win+Down do (Tk cannot iconify an override-redirect window): SW_SHOWMINNOACTIVE,
    # posted async from this thread - a synchronous ShowWindow inside a Tk callback re-enters Tk's
    # window proc (and Python bindings) while ctypes has dropped the GIL -> fatal error
    hw = ui(app._hwnd)
    ctypes.windll.user32.ShowWindowAsync(wt.HWND(hw), 7)
    caught = wait(lambda: not app.visible, 3)
    time.sleep(0.6)
    fg0, _ = fg_owner()
    real = press_hotkey()
    wait(lambda: app.recording, 2)
    c = shown_checks(fg0)
    check("S4_minimized_to_tray_hotkey_saved_geometry", caught and shown_ok(c), real_hotkey=real, caught=caught, **c)
    stop()
    # S5 already visible
    n = len(SHOWS)
    fg0, _ = fg_owner()
    press_hotkey()
    wait(lambda: app.recording, 2)
    c = shown_checks(fg0)
    rec = app.recording
    stop()
    check("S5_visible_hotkey_only_toggles", rec and len(SHOWS) == n and c["visible"] and not c["foreground_ever_ours"],
          show_calls=len(SHOWS) - n)
    # S6 setting off
    ui(lambda: app._set_show_on_hotkey(False))
    ui(app.hide)
    time.sleep(0.4)
    press_hotkey()
    wait(lambda: app.recording, 2)
    off = {"visible": app.visible, "iswindowvisible": ui(app.exstyle).get("iswindowvisible"), "recording": app.recording}
    stop()
    ui(lambda: app._set_show_on_hotkey(True))
    check("S6_setting_off_stays_hidden", not off["visible"] and not off["iswindowvisible"] and off["recording"]
          and app.cfg["show_on_hotkey"] and app.shk_var.get(), **off)
    # S7 with clear_on_return
    ui(app.clear_text)
    time.sleep(0.6)
    app._fg_override = (0x7F0000A1, False)
    msg = "Message to paste somewhere else."
    ui(lambda: app._handle(("text", msg, time.perf_counter(), 0, 0, 1.0, b"")))
    wait(lambda: app._copied_text == msg and app.cor_armed, 3)
    app._fg_override = (0x7F0000B2, False)
    left = wait(lambda: app.cor_left, 2)
    app._fg_override = None                   # the real foreground decides "widget not focused" now
    ui(app.hide)
    time.sleep(0.4)
    fg0, _ = fg_owner()
    real = press_hotkey()
    wait(lambda: app.recording, 2)
    c = shown_checks(fg0)
    box = ui(lambda: app.text.get("1.0", "end-1c"))
    stop()
    check("S7_with_clear_on_return", left and shown_ok(c) and box == "" and bool(app.cor_restore), real_hotkey=real,
          left=left, box=box, foreground_ever_ours=c["foreground_ever_ours"], visible=c["visible"])
    ui(app.hide)
    # S8 migration
    real_cp = plw.CONFIG_PATH
    out = {}
    for name, raw in (("v1", {"idle_unload_min": 10, "autocopy": False, "x": 5}),
                      ("v3", {"config_version": 3, "clear_on_return": False, "autocopy": False, "x": 5}),
                      ("v4_off", {"config_version": 4, "show_on_hotkey": False, "x": 5}),
                      ("cur_off", {"config_version": plw.CONFIG_VERSION, "show_on_hotkey": False, "x": 5})):
        plw.CONFIG_PATH = os.path.join(TMP, f"mig-{name}.json")
        json.dump(raw, open(plw.CONFIG_PATH, "w"))
        msg = plw.migrate_config()
        got = json.load(open(plw.CONFIG_PATH))
        baks = [f for f in os.listdir(TMP) if f.startswith(f"mig-{name}.json.bak-")]
        out[name] = {"msg": msg, "got": got, "backups": len(baks)}
    plw.CONFIG_PATH = real_cp
    g1, g3, g4, gc = out["v1"]["got"], out["v3"]["got"], out["v4_off"]["got"], out["cur_off"]["got"]
    V = plw.CONFIG_VERSION
    check("S8_config_migration_current",
          g1["config_version"] == V and g1["show_on_hotkey"] is True and g1["clear_on_return"] is True
          and g1["gpu_always"] and g1["autocopy"] is False and out["v1"]["backups"] == 1
          and g3["config_version"] == V and g3["show_on_hotkey"] is True and g3["clear_on_return"] is False
          and g3["autocopy"] is False and out["v3"]["backups"] == 1
          and g4["config_version"] == V and g4["show_on_hotkey"] is False and out["v4_off"]["backups"] == 1
          and out["cur_off"]["msg"] is None and gc["show_on_hotkey"] is False and out["cur_off"]["backups"] == 0,
          **{k: {"msg": v["msg"], "backups": v["backups"]} for k, v in out.items()})


def runner():
    try:
        body()
    except Exception:
        res["error"] = traceback.format_exc()
        p("ERROR", res["error"])
    finally:
        try:
            stop()
            ui(app.hide)
        except Exception:
            pass
        app._call(app.quit, wait=False)


threading.Thread(target=runner, daemon=True).start()
app.run()
res["real_clipboard_untouched"] = real_get_clip() == CLIP_BEFORE
res["real_hotkey_presses"] = len(HOTKEYS)
res["summary"] = results
res["sandbox_guard"] = sandbox_guard.summary()
res["all_ok"] = (bool(results) and all(results.values()) and res["real_clipboard_untouched"] and "error" not in res
                 and sandbox_guard.ok())
print("RESULT " + json.dumps(res, indent=1, default=str))
sys.exit(0 if res["all_ok"] else 1)
