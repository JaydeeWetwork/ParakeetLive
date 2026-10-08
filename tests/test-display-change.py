"""Log 57 (e): monitor unplugged -> the widget is pulled back on screen, and returns when the monitor is back.
Sandbox as in test-code-review.py (a second App in this process with its own data dir and temp config,
fake clipboard, fake microphone, no tray icon, no global hotkey, no model; 2% opacity, shown without focus).
WM_DISPLAYCHANGE is sent only to the sandbox's own hidden tray window (never broadcast); which spots are "on
a monitor" is faked by patching on_a_monitor, so no real display setting changes.
  D1 the tray window passes WM_DISPLAYCHANGE on; bursts are coalesced (two re-checks, 1 s and 3 s)
  D2 visible widget whose monitor goes away -> moved to the default spot (bottom centre of the main screen)
  D3 hiding it then keeps the saved spot in the config (size still saved)
  D4 the hotkey shows it at the default spot while the saved spot is on no monitor
  D5 the monitor comes back -> the widget goes back to its saved spot
  D6 dragged while displaced -> the new spot is the saved one; no move back later
  D7 a widget on a monitor is never moved; the foreground window is never one of the test's own
"""
import ctypes, importlib.machinery, importlib.util, json, os, shutil, sys, tempfile, threading, time, traceback  # noqa: E401
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(REPO, "widget")
TMP = os.path.join(tempfile.gettempdir(), "plive-display-test")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(os.path.join(TMP, "data", "logs"))
os.environ["PARAKEET_LIVE_DATA"] = os.path.join(TMP, "data")
sys.path.insert(0, W)
import plive_core as core   # noqa: E402
import plive_tray as trayw  # noqa: E402
real_get_clip = core.get_clipboard
CLIP_BEFORE = real_get_clip()
fake = {"clip": None}
core.set_clipboard = lambda t: (fake.__setitem__("clip", t), True)[1]
core.get_clipboard = lambda: fake["clip"]
core.kill_stale_servers = lambda: False


class FakeMic:
    def __init__(self, device, on_audio, on_error):
        self.info = "fake mic (test, no audio device)"

    def start(self):
        pass

    def stop(self):
        pass


core.MicCapture = FakeMic
core.parse_hotkey = lambda s: (_ for _ in ()).throw(ValueError("test: no global hotkey"))
trayw.CLASS_NAME = "ParakeetLiveTrayDisplayTest"
trayw.Tray._add = lambda self: None
loader = importlib.machinery.SourceFileLoader("plw", os.path.join(W, "parakeet_live.pyw"))
spec = importlib.util.spec_from_loader("plw", loader)
plw = importlib.util.module_from_spec(spec)
loader.exec_module(plw)
import sandbox_guard  # noqa: E402  (log 57: sandbox windows click-through, no menus, checked after each show)
sandbox_guard.install(plw)
assert plw.STATE_PATH.startswith(TMP), plw.STATE_PATH
u32 = ctypes.windll.user32
u32.GetForegroundWindow.restype = ctypes.c_void_p
VX, VY = u32.GetSystemMetrics(76), u32.GetSystemMetrics(77)
SPOT = (VX + 120, VY + 140)               # "his" saved spot, on the left-/top-most monitor
SPOT2 = (VX + 400, VY + 300)              # where he drags it while displaced (also on the missing monitor)
missing = {"on": False}
real_on = plw.on_a_monitor


def fake_on(x, y, margin=40):
    if missing["on"] and VX <= x < VX + 700 and VY <= y < VY + 500:
        return False                       # that monitor is "unplugged"
    return real_on(x, y, margin)


plw.on_a_monitor = fake_on
cfg = dict(plw.DEFAULTS)
cfg.update({"prewarm_login": False, "gpu_always": False, "auto_park": False, "autocopy": True,
            "config_version": plw.CONFIG_VERSION, "opacity": 0.02, "x": SPOT[0], "y": SPOT[1], "w": 640, "h": 132})
plw.CONFIG_PATH = os.path.join(TMP, "config.json")
json.dump(cfg, open(plw.CONFIG_PATH, "w"), indent=1)
plw.App.load_model = lambda self, *a, **k: None
plw.App.prewarm = lambda self, *a, **k: None
plw.App._gpu_watch_loop = lambda self: None
app = plw.App(plw.build_parser().parse_args(["--tray", "--no-save", "--training-dir", os.path.join(TMP, "training")]))
LOG = os.path.join(TMP, "data", "logs", "widget.log")
results = {}


def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def ui(fn): return app._call(fn)
def wait(c, t=3): return app._wait(c, t)
def pos(): return ui(lambda: (app.root.update_idletasks(), (app.root.winfo_x(), app.root.winfo_y()))[1])


def check(name, ok, **info):
    results[name] = bool(ok)
    p(("PASS " if ok else "FAIL ") + name, json.dumps(info, default=str))


def display_change():
    u32.SendMessageW(ctypes.c_void_p(app.tray.hwnd), 0x007E, 32, 0)   # only the sandbox's own tray window


fg_watch = {"n": 0, "ours": 0, "stop": False}


def fg_sampler():
    """Every 50 ms: does the foreground window belong to this test process? (only the process id is read)"""
    pid = ctypes.c_ulong()
    while not fg_watch["stop"]:
        h = u32.GetForegroundWindow()
        if h:
            u32.GetWindowThreadProcessId(ctypes.c_void_p(h), ctypes.byref(pid))
            fg_watch["ours"] += pid.value == os.getpid()
        fg_watch["n"] += 1
        time.sleep(0.05)


def body():
    wait(lambda: app.tray.ready.is_set() and app.tray.hwnd, 5)
    threading.Thread(target=fg_sampler, daemon=True).start()
    default = ui(lambda: app._default_xy(640, 132))
    ui(app._show_for_hotkey)
    at_spot = pos() == SPOT and app._displaced_from is None
    # D7a: a display change while the monitor is there -> nothing moves
    display_change()
    time.sleep(1.3)
    stayed = pos() == SPOT and app._displaced_from is None
    # D1 + D2: the monitor goes away
    missing["on"] = True
    jobs0 = len(app._disp_jobs)
    for _ in range(3):
        display_change()
    time.sleep(0.1)
    coalesced = len(app._disp_jobs) == 2
    moved = wait(lambda: pos() == default, 3)
    check("D1_display_change_passed_on_and_coalesced", coalesced and moved, jobs_before=jobs0,
          jobs_after_burst=len(app._disp_jobs))
    check("D2_pulled_back_to_default_spot", at_spot and moved and app._displaced_from == SPOT,
          shown_at_spot=at_spot, now=pos(), default=default, displaced_from=app._displaced_from)
    # D3: hide -> the saved spot stays in the config
    ui(app.hide)
    saved = ui(lambda: dict(app.cfg))          # what is written to config.json (--no-save: memory only)
    check("D3_saved_spot_kept_on_hide", (saved["x"], saved["y"]) == SPOT and saved["w"] == 640 and saved["h"] == 132,
          saved=(saved["x"], saved["y"], saved["w"], saved["h"]))
    # D4: the hotkey shows it while the saved spot is still on no monitor
    ui(app._show_for_hotkey)
    check("D4_shown_at_default_while_monitor_missing", pos() == default and app._displaced_from == SPOT,
          now=pos(), displaced_from=app._displaced_from)
    # D5: the monitor is back
    missing["on"] = False
    display_change()
    back = wait(lambda: pos() == SPOT and app._displaced_from is None, 3)
    log_ok = "is on a monitor again" in open(LOG, encoding="utf-8", errors="replace").read()
    check("D5_back_to_saved_spot_when_monitor_returns", back and log_ok, now=pos(), log=log_ok)
    # D6: displaced again, then he drags it somewhere -> that is the saved spot; no move back later
    missing["on"] = True
    display_change()
    wait(lambda: pos() == default, 3)
    ui(lambda: (app.root.geometry(f"+{SPOT2[0]}+{SPOT2[1]}"), app.root.update_idletasks()))
    ui(lambda: app._remember_geometry(moved=True))
    saved2 = ui(lambda: dict(app.cfg))
    missing["on"] = False
    display_change()
    time.sleep(3.4)
    check("D6_drag_makes_new_saved_spot", (saved2["x"], saved2["y"]) == SPOT2 and pos() == SPOT2
          and app._displaced_from is None, saved=(saved2["x"], saved2["y"]), now=pos())
    fg_watch["stop"] = True
    check("D7_on_screen_never_moved_and_never_took_focus", stayed and fg_watch["ours"] == 0 and fg_watch["n"] > 50
          and app.visible, stayed=stayed, samples=fg_watch["n"], ours=fg_watch["ours"])
    ui(lambda: app.hide(remember=False))


def runner():
    try:
        body()
    except Exception:
        p("ERROR", traceback.format_exc())
    finally:
        app._call(app.quit, wait=False)


threading.Thread(target=runner, daemon=True).start()
app.run()
clip_ok = real_get_clip() == CLIP_BEFORE
print("real_clipboard_untouched", clip_ok)
print("sandbox_guard", json.dumps(sandbox_guard.summary()))
ok = len(results) == 7 and all(results.values()) and clip_ok and sandbox_guard.ok() \
    and sandbox_guard.STATE["shows_checked"] >= 2
print("ALL PASS" if ok else "SOME FAILED", sum(results.values()), "/ 7", flush=True)
sys.exit(0 if ok else 1)
