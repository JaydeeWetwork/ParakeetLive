"""Parakeet Live - floating local live-transcription widget that lives in the notification area.

Speech -> local NVIDIA Parakeet TDT 0.6B v2 (GPU, inside WSL) -> editable text box,
auto-copied to the clipboard. Nothing leaves the machine.

  pythonw parakeet_live.pyw            show the widget (loads the model)
  pythonw parakeet_live.pyw --tray     login mode: tray icon only; after a delay the model pre-loads into RAM at low
                                       priority, then (GPU policy on) moves to the GPU unless a game/heavy GPU app runs
  pythonw parakeet_live.pyw --cmd X    talk to the running instance: show|hide|toggle|load|unload|record|quit|dump|...
  python  parakeet_live.pyw --cmd batchhold --pid N --wait 120   batch job N needs the GPU: move the model to RAM and
                                       hold it there until "--cmd batchrelease --pid N" or until process N (and its
                                       children) exit. Exit code 0 = off the GPU, 3 = widget not running, 4 = timeout.

GPU policy ("Keep model on GPU", default on): the model stays hot on the NVIDIA card. A watcher thread (plive_gpu,
PDH via ctypes, every ~4 s) parks it in RAM while another app needs the GPU and brings it back when that app is gone.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import queue
import secrets
import statistics
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tkinter as tk  # noqa: E402

import plive_art as art  # noqa: E402
import plive_core as core  # noqa: E402
import plive_train as train  # noqa: E402
import plive_tray as trayw  # noqa: E402

CONFIG_PATH = os.path.join(HERE, "config.json")       # machine-specific, gitignored
DEFAULT_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ParakeetLive")  # outside the repo: training data + runtime logs


def _data_dir():
    """Data folder: env PARAKEET_LIVE_DATA, else "data_dir" in config.json, else DEFAULT_DATA_DIR."""
    d = os.environ.get("PARAKEET_LIVE_DATA")
    if not d:
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                d = json.load(f).get("data_dir")
        except Exception:
            d = None
    return os.path.abspath(os.path.expandvars(d or DEFAULT_DATA_DIR))


DATA_DIR = _data_dir()
LOG_DIR = os.path.join(DATA_DIR, "logs")
STATE_PATH = os.path.join(LOG_DIR, "state.json")
ICONS = {"unloaded": os.path.join(HERE, "icons", "parakeet-unloaded.ico"),   # Jaydee's parakeet v3, dimmed
         "ready": os.path.join(HERE, "icons", "parakeet-ready.ico"),
         "recording": os.path.join(HERE, "icons", "parakeet-recording.ico"),
         "standby": os.path.join(HERE, "icons", "parakeet-standby.ico")}  # + red dot badge
APP_ICON = os.path.join(HERE, "icons", "parakeet-v3.ico")
README = os.path.abspath(os.path.join(HERE, "..", "README.md"))
TRAINING_DIR = os.path.join(DATA_DIR, "training-data")   # never deleted by the app; never committed
TITLE = "Parakeet Live"

# palette: soft dark
BORDER = "#2a2f3c"
BG = "#171a21"
TEXT_BG = "#1f232d"
FG = "#e6e8ee"
MUTED = "#8a90a0"
DIM = "#5d6375"
HOVER_BG = "#262b37"
ACCENT = "#7aa2f7"
GREEN = "#8bd5a0"
AMBER = "#f2c46d"
RED = "#f7606c"
RED_HOVER = "#ff7883"
RED_REC = "#ef4b5a"
GRAY_DISC = "#454c5e"
MENU_BG = "#20242e"
SEL_BG = "#34405e"

DEFAULTS = {"data_dir": DEFAULT_DATA_DIR, "x": None, "y": None, "w": None, "h": None, "device": None, "silence_s": 0.5,
            "opacity": 0.97, "autocopy": True, "topmost": True, "hotkey": "ctrl+alt+space",
            "port": 51761, "max_utterance_s": 25.0, "font_size": 11,
            "idle_unload_min": 0,       # minutes idle on the GPU before moving to RAM; 0 = never (GPU policy default)
            "save_training": True,
            "ram_standby": True,        # idle: move the model to RAM (VRAM freed) instead of unloading it
            "standby_unload_h": 4,      # ... and free the RAM after this many hours in standby (0 = never)
            "prewarm_login": True,      # --tray (login): load the model at once (GPU with gpu_always; RAM only while the GPU is busy)
            "prewarm_delay_s": 0,
            # ---- GPU policy + auto-park (config_version 2, 2026-10-07)
            "config_version": 4,
            "gpu_always": True,         # keep the model on the GPU; login pre-load goes RAM -> GPU; no 4 h RAM unload
            "auto_park": True,          # park in RAM while another app needs the GPU (only with gpu_always)
            "park_poll_s": 4,           # watcher poll interval
            "park_mem_mb": 400,         # another process holding >= this much dedicated VRAM ...
            "park_confirm_polls": 2,    # ... for this many polls in a row
            "park_3d_pct": 25,          # another process at >= this 3D utilization ...
            "park_3d_s": 10,            # ... for this long
            "park_free_min_mb": 600,    # free VRAM below this (model on GPU) -> park
            "unpark_free_mb": 1800,     # unpark needs max(this, model_vram_mb + park_free_min_mb + 200) free
            "unpark_after_s": 60,       # ... and no trigger for this long
            "model_vram_mb": 1400,      # what the model costs on the GPU (measured 1395 MB)
            "park_temp_hold_s": 30,     # after a temporary GPU load while parked, re-park after this much idle
            # park at once while any of these runs; java/javaw only when it is Minecraft Java (log 57)
            "park_games": ["javaw.exe", "java.exe", "Minecraft.Windows.exe", "Minecraft.exe", "MinecraftLauncher.exe"],
            "park_ignore": [],          # extra process names never to park for (case-insensitive)
            "batch_hold_max_h": 6,      # safety cap for a batch hold whose launcher stays alive but never releases
            # ---- config_version 3 (log 53)
            "clear_on_return": True,    # next recording after you pasted elsewhere starts a fresh message (restorable)
            # ---- config_version 4 (log 54)
            "show_on_hotkey": True,     # hotkey that starts a recording shows the hidden widget (no focus steal)
            # ---- config_version 5 (log 57)
            "paste_detect": True}       # clear-on-return: Ctrl+V after the copy counts as "pasted" (same window)
CONFIG_VERSION = 6
OLD_PARK_GAMES = {"javaw.exe", "minecraft.windows.exe"}   # the v4 default (log 51)

SILENCE_CHOICES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]
OPACITY_CHOICES = [1.0, 0.97, 0.9, 0.8, 0.7, 0.6]
IDLE_CHOICES = [5, 10, 15, 30, 60, 0]          # minutes, 0 = never
STANDBY_CHOICES = [1, 2, 4, 8, 24, 0]          # hours in RAM standby before freeing the RAM, 0 = never


def session_dir(args):
    """Where the unsent message and kept retries live (log 57); None = not saved (--no-save, the self-test)."""
    return args.state_dir or (None if (args.no_save or args.selftest) else os.path.join(DATA_DIR, "state"))


RETRY_MAX_AGE_S = 3600                         # a failed utterance is retried for up to 1 h after it failed,
                                               # across restarts too (log 57; was 15 min in memory only)

# tray menu command ids
T_SHOWHIDE, T_RECORD, T_MODEL, T_SAVE, T_SETTINGS, T_AUTOCOPY, T_QUIT = 1, 2, 3, 4, 5, 6, 9
T_QUITALL = 8
T_IDLE0 = 20
T_UNLOAD, T_RAMSTBY, T_PREWARM = 10, 11, 12
T_STBY0 = 40
T_GPUALWAYS, T_PAUSEPARK, T_PARKINFO = 13, 14, 15
T_RESTORE, T_CLEARRET = 16, 17
T_SHOWHK = 18
T_PASTEDET = 19


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.OpenProcess.restype = wt.HANDLE
_k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
_k32.WaitForSingleObject.restype = wt.DWORD
_k32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
_k32.CloseHandle.argtypes = [wt.HANDLE]
SYNCHRONIZE, PROCESS_QUERY_LIMITED_INFORMATION = 0x00100000, 0x1000
_u32 = ctypes.WinDLL("user32", use_last_error=True)
_u32.GetForegroundWindow.restype = wt.HWND
_u32.GetAsyncKeyState.restype = ctypes.c_short
_u32.GetAsyncKeyState.argtypes = [ctypes.c_int]
VK_CONTROL, VK_MENU, VK_V = 0x11, 0x12, 0x56          # the only keys the paste watch ever looks at (log 57)
_u32.GetWindowThreadProcessId.restype = wt.DWORD
_u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
_u32.MonitorFromPoint.restype = wt.HANDLE
_u32.MonitorFromPoint.argtypes = [wt.POINT, wt.DWORD]


def log(msg):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, "widget.log")
        try:                                       # keep one older generation, cap the file (log 55)
            if os.path.getsize(path) > 2_000_000:
                os.replace(path, path + ".1")
        except OSError:
            pass
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def _valid(key, v):
    """Is v an acceptable value for config key `key` (same kind as its default)?"""
    d = DEFAULTS[key]
    if key in ("x", "y", "w", "h"):
        return v is None or (isinstance(v, int) and not isinstance(v, bool))
    if key == "device":
        return v is None or isinstance(v, (int, str)) and not isinstance(v, bool)
    if key == "port":
        return isinstance(v, int) and not isinstance(v, bool) and 1024 <= v <= 65000
    if isinstance(d, bool):
        return isinstance(v, bool)
    if isinstance(d, (int, float)):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
    if isinstance(d, str):
        return isinstance(v, str) and v.strip() != ""
    if isinstance(d, list):
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    return True


def _set_aside(reason):
    """Keep an unusable config.json for inspection instead of silently overwriting it later."""
    dst = CONFIG_PATH + time.strftime(".corrupt-%Y%m%d-%H%M%S")
    try:
        import shutil
        shutil.copy2(CONFIG_PATH, dst)
        log(f"config.json unusable ({reason}); copied to {os.path.basename(dst)}, using defaults")
    except Exception as e:
        log(f"config.json unusable ({reason}); could not copy it aside: {e}")


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return cfg
    except Exception as e:
        _set_aside(f"{e.__class__.__name__}")
        return cfg
    if not isinstance(raw, dict):
        _set_aside("not a JSON object")
        return cfg
    bad = [k for k, v in raw.items() if k in DEFAULTS and not _valid(k, v)]
    cfg.update({k: v for k, v in raw.items() if k in DEFAULTS and k not in bad})
    if bad:
        log(f"config.json: invalid value(s) for {', '.join(sorted(bad))} - using the defaults for those")
    return cfg


def migrate_config():
    """config_version < 2 -> GPU policy defaults: idle-to-RAM timer Never, gpu_always + auto_park on.
    config_version < 3 -> clear_on_return on (unless already set).
    config_version < 4 -> show_on_hotkey on (unless already set).
    config_version < 5 -> park_games: the v4 default list gets the Minecraft launchers + java.exe (a list you
                          changed is kept); paste_detect on (unless already set).
    Keeps every other user setting; writes a timestamped backup first. Returns a log line or None."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        return f"config migration skipped (unreadable config.json: {e})"
    if not isinstance(raw, dict):
        return "config migration skipped (config.json is not a JSON object)"
    try:
        old = int(raw.get("config_version", 1) or 1)
    except (TypeError, ValueError):
        return "config migration skipped (config_version is not a number)"
    if old >= CONFIG_VERSION:
        return None
    bak = CONFIG_PATH + time.strftime(".bak-%Y%m%d-%H%M%S")
    import shutil
    shutil.copy2(CONFIG_PATH, bak)
    before = {k: raw.get(k) for k in ("idle_unload_min", "gpu_always", "auto_park", "clear_on_return", "show_on_hotkey",
                                      "park_games", "paste_detect", "prewarm_delay_s")}
    if old < 2:
        raw.update({"idle_unload_min": 0, "gpu_always": True, "auto_park": True})
    if old < 3:
        raw.setdefault("clear_on_return", True)
    if old < 4:
        raw.setdefault("show_on_hotkey", True)
    if old < 5:
        pg = raw.get("park_games")
        if pg is None or (isinstance(pg, list) and {str(x).lower() for x in pg} == OLD_PARK_GAMES):
            raw["park_games"] = list(DEFAULTS["park_games"])
        raw.setdefault("paste_detect", True)
    if old < 6:                                  # 2026-10-09: load at login without the old 60 s delay
        if raw.get("prewarm_delay_s") in (None, 60, 60.0):
            raw["prewarm_delay_s"] = 0
    raw["config_version"] = CONFIG_VERSION
    save_config(raw)
    return f"config migrated v{old} -> v{CONFIG_VERSION} (backup {os.path.basename(bak)}): was {before}"


def save_config(cfg):
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.flush()
            os.fsync(f.fileno())             # on disk before the rename (log 55)
        os.replace(tmp, CONFIG_PATH)
    except Exception as e:
        log(f"config save failed: {e}")


def on_a_monitor(x, y, margin=40):
    """Is the point just inside the top-left corner of a window at (x, y) on a real monitor? (log 55)"""
    try:
        return bool(_u32.MonitorFromPoint(wt.POINT(int(x) + margin, int(y) + margin // 2), 0))  # DEFAULTTONULL
    except Exception:
        return True                    # never let a failed check move the widget


def icon_font():
    import tkinter.font as tkfont
    fams = set(tkfont.families())
    for f in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if f in fams:
            return f
    return "Segoe UI Symbol"


GLYPH = {"copy": "\uE8C8", "clear": "\uE74D", "gear": "\uE713", "close": "\uE8BB", "minimize": "\uE921", "save": "\uE74E"}


def idle_label(m):
    return "Never" if not m else f"{m} min"


def standby_label(h):
    return "Never" if not h else f"{h:g} h"


class App:
    def __init__(self, args):
        self.args = args
        mig = None if args.no_save else migrate_config()
        self.cfg = load_config()
        if args.idle_min is not None:
            self.cfg["idle_unload_min"] = args.idle_min
        if args.standby_h is not None:
            self.cfg["standby_unload_h"] = args.standby_h
        self.uiq = queue.Queue()        # worker threads -> UI thread
        self.jobq = queue.Queue()       # utterances -> transcription thread
        self.ready_evt = threading.Event()
        self.wake_evt = threading.Event()
        self.engine_lock = threading.Lock()
        self.engine = "unloaded"        # unloaded | prewarming | standby | loading | ready | unloading | error
        self.want_gpu = False           # the user is waiting for the GPU (vs. RAM-only pre-load)
        self.standby_since = None
        self.load_t0 = None
        self.gen = 0
        self.server = None
        self.port = self.cfg["port"]
        # shared secret for our WSL server, new every widget run (log 55); handed over via WSLENV only
        self.server_token = getattr(args, "server_token", None) or secrets.token_hex(16)
        self.tclient = core.Client(self.port, token=self.server_token)
        self.hclient = None
        self.closing = False
        self.visible = False
        self.styled = False
        self.recording = False
        self.mic = None
        self.seg = None
        self.level_db = -100.0
        self.pending = 0
        self.state = "unloaded"
        self.last_activity = time.monotonic()
        self.latencies, self.infer_ms, self.rtt_ms = [], [], []
        self.health = {}
        self._last_copied = None
        self._deb = None
        self._tdeb = None
        self._anim_k = 0
        self._press = None
        self.copy_count = 0
        self.latest = {}                # utterance id -> latest audio version (speculation bookkeeping)
        self.spec_hits = self.spec_misses = 0
        self.ui_gaps = []
        self._hb_last = time.perf_counter()
        self.spans = {}                 # training record id -> last text written for that span
        self.devices, self.default_dev = [], None
        self.training = train.TrainingLog(args.training_dir or TRAINING_DIR)
        # GPU policy / auto-park
        self.parked = None              # {"reason", "proc", "since", "temp"} while parked for another GPU app
        self.park_pending = False       # park requested, waiting for the utterance / load to finish
        self.park_paused = False        # tray "Pause auto-park" (this session only)
        self.goal_gpu = False           # login pre-load: continue RAM -> GPU once in standby
        self.gpu_last = None            # last watcher sample summary
        self.watch_stats = {"polls": 0, "cpu_s": 0.0, "errors": 0, "adapter": None, "last_error": None}
        self.watch_evt = threading.Event()
        self.park_log = []              # recent park/unpark events with timings
        self._park_t0 = self._unpark_t0 = None
        self._park_info = {}
        self.holds = {}                 # batch holds: pid -> {"h": process handle, "since", "t0", "via"}
        self.wsl_hold = False           # a hold ended but transcribe.py still runs inside WSL: keep waiting
        self._hold_prev_gpu = False
        self._hold_loop = False
        # clear-on-return (log 53): after the box text was copied and you switched to another window,
        # the next hotkey press starts a fresh message; one-step restore via Ctrl+Z / menu
        self._copied_text = None        # last text the clipboard thread actually wrote
        self.cor_armed = False          # box text copied; watching the foreground window
        self.cor_left = False           # ... and you have switched to another (not our own) window since
        self._cor_fg = None             # last foreground window seen by the watcher
        self._cor_fg_req = None         # foreground window when the last copy was requested
        self._cor_job = None
        self.cor_restore = None         # {"text", "spans": [(rid, start, end, last)], "why", "t"}
        self._cor_prog_text = None      # box text right after the clear / later dictation (no hand edits)
        self._cor_dictated = None       # box text as dictation left it (anything else = edited by hand)
        self._cor_left_session = None   # last recording started before the departure was seen
        self._cor_clear_session = None  # the recording a clear-on-return clear was made for
        self._paste_job = None          # Ctrl+V watch (log 57): runs only while copied text waits, widget not in front
        self.paste_polls = 0            # how many times the two keys were looked at (for the CPU check)
        self.rec_session = 0            # +1 per recording start
        self._uid_session = {}          # utterance id -> recording it came from (until its text arrives)
        self._fg_override = None        # tests only: (hwnd, is_own) instead of the real foreground window

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title(TITLE)
        self.s = max(1.0, self.root.winfo_fpixels("1i") / 96.0)
        self.ifont = icon_font()
        self.root.report_callback_exception = self._tk_error
        if os.path.exists(APP_ICON):
            try:
                self.root.iconbitmap(default=APP_ICON)
            except Exception:
                pass
        self._build_art()
        self._build_ui()
        self._displaced_from = None    # saved spot while the widget sits elsewhere because it is on no monitor
        self._disp_jobs = []
        self._place_window()
        self.root.overrideredirect(True)

        threading.Thread(target=self._job_loop, name="transcriber", daemon=True).start()
        self.clip_evt = threading.Event()
        self.clip_text = None
        self.clip_lock = threading.Lock()   # guards clip_text between the UI and clipboard threads
        self.copy_failures = 0
        self.failed = []                # utterances whose transcription failed: kept for a retry (log 55)
        self._retry_tries = {}          # utterance id of a re-queued utterance -> tries so far
        self._retry_kid = {}            # utterance id of a re-queued kept utterance -> (file id, failed_at) (log 57)
        sd = session_dir(args)
        self.store = core.SessionStore(sd) if sd else None     # unsent message + kept retries (log 57)
        self.srv_rec = core.ServerRecord(sd) if sd else None   # keep-server: reconnect after a widget restart
        self.server_adopted = False
        self._quit_stop_server = True
        self._draft_job = None
        self._recover_times = []        # automatic engine restarts (GPU policy), for the rate limit
        self._quitting = False
        threading.Thread(target=self._clip_loop, name="clipboard", daemon=True).start()
        try:
            hk = core.parse_hotkey(self.cfg["hotkey"])
        except Exception:
            hk = None
        self.tray = trayw.Tray(ICONS, "Parakeet Live - model not loaded", lambda ev: self.uiq.put(("tray", ev)),
                               self._tray_menu, hotkey=hk)
        self.tray.start()
        self.tray.ready.wait(5)
        log(f"started pid={os.getpid()} tray_added={self.tray.added} hotkey_ok={self.tray.hotkey_ok} "
            f"mode={'tray' if args.tray else 'show'} {self.tray.error}")
        if mig:
            log(mig)
        log(f"GPU policy: gpu_always={self.cfg['gpu_always']} auto_park={self.cfg['auto_park']} "
            f"idle_unload_min={self.cfg['idle_unload_min']} unpark_needs={self._unpark_needed()} MB free")
        threading.Thread(target=self._gpu_watch_loop, name="gpu-watch", daemon=True).start()
        self.root.after(10, self._pump)
        self.root.after(40, self._animate)
        self.root.after(16, self._heartbeat)
        self.root.after(5000, self._idle_check)
        self.set_state("unloaded", "Model not loaded - click the mic to start")
        self._restore_session()
        self._try_adopt()                    # keep-server: a widget restart reuses the running model server
        if not args.tray:
            self.root.after(50, self.show)
        elif self.cfg["prewarm_login"]:
            d = float(self.cfg["prewarm_delay_s"] if args.prewarm_delay is None else args.prewarm_delay)
            if self.cfg["gpu_always"]:       # 2026-10-09: used all day - straight onto the GPU at login
                log(f"login: loading the model onto the GPU in {d:g} s")
                self.root.after(int(d * 1000), self._login_load)
            else:
                log(f"pre-load into RAM scheduled in {d:g} s")
                self.root.after(int(d * 1000), lambda: self.prewarm("login pre-warm"))
        self.root.after(500, self.dump_state)
        if args.selftest:
            threading.Thread(target=self._selftest, name="selftest", daemon=True).start()

    # ------------------------------------------------------------------ art + layout
    def _build_art(self):
        D = self.D = int(round(92 * self.s))
        T = lambda im: art.to_tk(im, tk)  # noqa: E731
        self.img = {
            "base": T(art.base_ring(D, "#1d212b", "#2b3140")),
            "idle": T(art.disc(D, RED, "mic")),
            "idle_hover": T(art.disc(D, RED_HOVER, "mic")),
            "rec": T(art.disc(D, RED_REC, "stop")),
            "rec_hover": T(art.disc(D, RED_HOVER, "stop")),
            "loading": T(art.disc(D, GRAY_DISC, "mic", icon_alpha=170)),
            "unloaded": T(art.disc(D, "#8a4a52", "mic", icon_alpha=220)),
            "unloaded_hover": T(art.disc(D, "#a8545e", "mic", icon_alpha=235)),
            "error": T(art.disc(D, "#6b5960", "mic", icon_alpha=170)),
        }
        self.pulse = [T(art.pulse_ring(D, RED, i / 24)) for i in range(24)]
        self.halos = [T(art.halo(D, RED, i / 9)) for i in range(10)]
        self.spin = [T(art.spinner(D, ACCENT, k)) for k in range(12)]

    def _build_ui(self):
        s = self.s
        r = self.root
        pad = int(12 * s)
        r.configure(bg=BORDER)
        self.outer = tk.Frame(r, bg=BG)
        self.outer.pack(fill="both", expand=True, padx=1, pady=1)

        D = self.D
        self.btn = tk.Canvas(self.outer, width=D, height=D, bg=BG, highlightthickness=0, bd=0, cursor="hand2")
        self.btn.pack(side="left", padx=(pad, int(6 * s)), pady=pad)
        c = D // 2
        self.i_base = self.btn.create_image(c, c, image=self.img["base"])
        self.i_pulse = self.btn.create_image(c, c, image=self.pulse[0], state="hidden")
        self.i_halo = self.btn.create_image(c, c, image=self.halos[0], state="hidden")
        self.i_disc = self.btn.create_image(c, c, image=self.img["unloaded"])
        self.i_spin = self.btn.create_image(c, c, image=self.spin[0], state="hidden")
        self._hover = False
        self.btn.bind("<Enter>", lambda e: self._set_hover(True))
        self.btn.bind("<Leave>", lambda e: self._set_hover(False))
        self.btn.bind("<ButtonPress-1>", self._btn_press)
        self.btn.bind("<B1-Motion>", self._btn_motion)
        self.btn.bind("<ButtonRelease-1>", self._btn_release)

        right = tk.Frame(self.outer, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(int(4 * s), pad), pady=(int(7 * s), pad))
        top = tk.Frame(right, bg=BG)
        top.pack(fill="x")
        small = ("Segoe UI", 9)
        self.dot = tk.Canvas(top, width=int(8 * s), height=int(8 * s), bg=BG, highlightthickness=0)
        self.dot_id = self.dot.create_oval(1, 1, int(8 * s) - 1, int(8 * s) - 1, fill=MUTED, outline="")
        self.dot.pack(side="left", padx=(int(2 * s), int(6 * s)))
        self.status = tk.Label(top, text="", bg=BG, fg=MUTED, font=small, anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        for name, cmd in (("close", self.hide), ("minimize", self.hide), ("gear", self._show_menu),
                          ("clear", lambda: self.clear_by_user("Clear button")),
                          ("copy", lambda: self.copy_all(force=True))):
            self._icon_button(top, name, cmd).pack(side="right")
        # small indicator shown while "Save training data" is on
        self.td_ind = tk.Label(top, text=GLYPH["save"], font=(self.ifont, 8), bg=BG, fg=DIM, padx=int(4 * s))
        self.td_ind.bind("<Button-1>", self._show_menu)
        self.flash = tk.Label(top, text="", bg=BG, fg=BG, font=small)
        self.td_ind.pack(side="right")
        self.flash.pack(side="right", padx=(0, int(6 * s)))
        self._update_td_indicator()

        self.tframe = tk.Frame(right, bg=TEXT_BG, highlightthickness=1, highlightbackground=BORDER)
        self.tframe.pack(fill="both", expand=True, pady=(int(6 * s), 0))
        self.text = tk.Text(self.tframe, bg=TEXT_BG, fg=FG, insertbackground=FG, selectbackground=SEL_BG,
                            selectforeground=FG, relief="flat", bd=0, wrap="word", undo=True, maxundo=200,
                            font=("Segoe UI", self.cfg["font_size"]), padx=int(10 * s), pady=int(7 * s),
                            highlightthickness=0, spacing1=1, spacing3=2, insertwidth=max(1, int(1.5 * s)))
        self.text.pack(fill="both", expand=True)
        self.placeholder = tk.Label(self.tframe, text=f"Click the mic or press {self._hotkey_label()} and talk.",
                                    bg=TEXT_BG, fg=DIM, font=("Segoe UI", self.cfg["font_size"]))
        self.placeholder.bind("<Button-1>", lambda e: self.text.focus_set())
        self._update_placeholder()
        self.text.bind("<<Modified>>", self._on_modified)
        self.text.bind("<FocusIn>", lambda e: self.tframe.configure(highlightbackground="#3b4762"))
        self.text.bind("<FocusOut>", lambda e: self.tframe.configure(highlightbackground=BORDER))
        self.text.bind("<Control-a>", lambda e: (self.text.tag_add("sel", "1.0", "end-1c"), "break")[1])
        self.text.bind("<Control-MouseWheel>", self._zoom)
        self.text.bind("<Control-z>", self._on_ctrl_z)
        self.text.bind("<Control-Z>", self._on_ctrl_z)
        self.text.bind("<Button-3>", self._text_menu)

        for w in (self.outer, right, top, self.status, self.dot, self.flash):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)
            w.bind("<ButtonRelease-1>", lambda e: self._remember_geometry(moved=True))
        g = int(12 * s)
        self.grip = tk.Canvas(self.outer, width=g, height=g, bg=BG, highlightthickness=0, cursor="size_nw_se")
        for i in range(3):
            for j in range(i + 1):
                x, y = g - 3 - (i - j) * 4 * s / 1.3, g - 3 - j * 4 * s / 1.3
                self.grip.create_oval(x - 1, y - 1, x + 1, y + 1, fill=DIM, outline="")
        self.grip.place(relx=1.0, rely=1.0, anchor="se", x=-2, y=-2)
        self.grip.bind("<ButtonPress-1>", self._resize_start)
        self.grip.bind("<B1-Motion>", self._resize_move)
        self.grip.bind("<ButtonRelease-1>", lambda e: self._remember_geometry())

        self._build_menu()
        self.root.bind("<Escape>", lambda e: self.hide())
        # a real minimize (Win+D, Win+Down, ...) must never leave a minimized taskbar item: route it to hide()
        self.root.bind("<Unmap>", self._on_unmap)
        self.root.after(400, self._iconic_watch)

    def _icon_button(self, parent, name, cmd):
        s = self.s
        lb = tk.Label(parent, text=GLYPH[name], font=(self.ifont, 9), bg=BG, fg=MUTED,
                      padx=int(6 * s), pady=int(3 * s), cursor="hand2")
        hover = "#5a2a31" if name == "close" else HOVER_BG
        lb.bind("<Enter>", lambda e: lb.configure(bg=hover, fg=FG))
        lb.bind("<Leave>", lambda e: lb.configure(bg=BG, fg=MUTED))
        lb.bind("<ButtonRelease-1>", lambda e: cmd(e) if name == "gear" else cmd())
        return lb

    def _menu(self, parent):
        return tk.Menu(parent, tearoff=0, bg=MENU_BG, fg=FG, activebackground=SEL_BG, activeforeground=FG,
                       selectcolor=ACCENT, bd=0, relief="flat", font=("Segoe UI", 9), disabledforeground=DIM)

    def _build_menu(self):
        m = self.menu = self._menu(self.root)
        self.mic_var = tk.StringVar(value=self.cfg["device"] or "")
        self.mic_menu = self._menu(m)
        self.mic_menu.configure(postcommand=lambda: self._fill_mic_menu(refresh=not self.recording))
        m.add_cascade(label="Microphone", menu=self.mic_menu)
        self.sil_var = tk.StringVar(value=f"{float(self.cfg['silence_s']):.1f}")
        sm = self._menu(m)
        for v in SILENCE_CHOICES:
            sm.add_radiobutton(label=f"{v:.1f} s" + ("  (default)" if v == 0.5 else ""), value=f"{v:.1f}",
                               variable=self.sil_var, command=self._set_silence)
        m.add_cascade(label="Pause before sending", menu=sm)
        self.ga_var = tk.BooleanVar(value=bool(self.cfg["gpu_always"]))
        m.add_checkbutton(label="Keep model on GPU (auto-park for games)", variable=self.ga_var,
                          command=lambda: self._set_gpu_always(self.ga_var.get()))
        self.pp_var = tk.BooleanVar(value=False)
        m.add_checkbutton(label="Pause auto-park", variable=self.pp_var,
                          command=lambda: self._set_park_paused(self.pp_var.get()))
        self.idle_var = tk.StringVar(value=str(self.cfg["idle_unload_min"]))
        im = self._menu(m)
        for v in IDLE_CHOICES:
            im.add_radiobutton(label=idle_label(v) + ("  (default)" if v == 0 else ""), value=str(v),
                               variable=self.idle_var, command=lambda: self._set_idle(int(self.idle_var.get())))
        m.add_cascade(label="Free VRAM when idle", menu=im)
        self.stby_var = tk.StringVar(value=f"{float(self.cfg['standby_unload_h']):g}")
        sbm = self._menu(m)
        for v in STANDBY_CHOICES:
            sbm.add_radiobutton(label=standby_label(v) + ("  (default)" if v == 4 else ""), value=f"{v:g}",
                                variable=self.stby_var, command=lambda: self._set_standby_h(float(self.stby_var.get())))
        m.add_cascade(label="Free RAM after (in standby; off while on GPU policy)", menu=sbm)
        self.rs_var = tk.BooleanVar(value=bool(self.cfg["ram_standby"]))
        m.add_checkbutton(label="Keep model in RAM when idle (fast restart)", variable=self.rs_var,
                          command=lambda: self._set_ram_standby(self.rs_var.get()))
        self.pw_var = tk.BooleanVar(value=bool(self.cfg["prewarm_login"]))
        m.add_checkbutton(label="Pre-load at login (RAM, then GPU)", variable=self.pw_var,
                          command=lambda: self._set_prewarm(self.pw_var.get()))
        self.op_var = tk.StringVar(value=f"{float(self.cfg['opacity']):.2f}")
        om = self._menu(m)
        for v in OPACITY_CHOICES:
            om.add_radiobutton(label=f"{int(round(v * 100))}%", value=f"{v:.2f}", variable=self.op_var,
                               command=self._set_opacity)
        m.add_cascade(label="Opacity", menu=om)
        self.ac_var = tk.BooleanVar(value=bool(self.cfg["autocopy"]))
        m.add_checkbutton(label="Auto-copy to clipboard", variable=self.ac_var,
                          command=lambda: self._set_autocopy(self.ac_var.get()))
        self.cor_var = tk.BooleanVar(value=bool(self.cfg["clear_on_return"]))
        m.add_checkbutton(label="Fresh message after pasting elsewhere (next recording clears)", variable=self.cor_var,
                          command=lambda: self._set_clear_on_return(self.cor_var.get()))
        self.paste_var = tk.BooleanVar(value=bool(self.cfg["paste_detect"]))
        m.add_checkbutton(label="... Ctrl+V counts as pasting (same window too)", variable=self.paste_var,
                          command=lambda: self._set_paste_detect(self.paste_var.get()))
        self.shk_var = tk.BooleanVar(value=bool(self.cfg["show_on_hotkey"]))
        m.add_checkbutton(label="Show widget when the hotkey starts recording", variable=self.shk_var,
                          command=lambda: self._set_show_on_hotkey(self.shk_var.get()))
        self.td_var = tk.BooleanVar(value=bool(self.cfg["save_training"]))
        m.add_checkbutton(label="Save training data (local only)", variable=self.td_var,
                          command=lambda: self._set_training(self.td_var.get()))
        self.top_var = tk.BooleanVar(value=bool(self.cfg["topmost"]))
        m.add_checkbutton(label="Always on top", variable=self.top_var, command=self._set_topmost)
        m.add_separator()
        self.model_index = m.index("end") + 1
        m.add_command(label="Load model", command=self.toggle_model)
        m.add_command(label="Unload model completely (free RAM)", command=lambda: self.unload_model("manual"))
        m.add_command(label="Copy all", command=lambda: self.copy_all(force=True))
        m.add_command(label="Clear text", command=lambda: self.clear_by_user("menu Clear text"))
        self.restore_index = m.index("end") + 1
        m.add_command(label="Restore last message", command=self.restore_last, state="disabled")
        m.add_separator()
        m.add_command(label="Open training data folder", command=lambda: self._open(self.training.dir, folder=True))
        m.add_command(label="Open README", command=lambda: self._open(README))
        m.add_command(label="Open logs folder", command=lambda: self._open(LOG_DIR, folder=True))
        m.add_command(label=f"Hotkey: {self._hotkey_label()}", state="disabled")
        m.add_separator()
        m.add_command(label="Hide to tray", command=self.hide)
        m.add_command(label="Quit Parakeet Live", command=self.quit)
        m.add_command(label="Quit (stop model too)", command=lambda: self.quit(stop_server=True))

    def _open(self, path, folder=False):
        try:
            if folder:
                os.makedirs(path, exist_ok=True)
            os.startfile(path)
        except Exception as e:
            log(f"open {path}: {e}")

    def _fill_mic_menu(self, refresh=False):
        mm = self.mic_menu
        mm.delete(0, "end")
        try:
            if refresh:
                core.refresh_portaudio()      # picks up USB mics plugged in since the last look
            self.devices, self.default_dev = core.list_input_devices()
        except Exception as e:
            self.devices, self.default_dev = [], None
            log(f"device list failed: {e}")
        mm.add_radiobutton(label=f"System default ({self.default_dev or 'none'})", value="",
                           variable=self.mic_var, command=self._set_mic)
        mm.add_separator()
        for d in self.devices:
            mm.add_radiobutton(label=d["name"], value=d["name"], variable=self.mic_var, command=self._set_mic)

    def _text_menu(self, e):
        m = self._menu(self.root)
        m.add_command(label="Cut", command=lambda: self.text.event_generate("<<Cut>>"))
        m.add_command(label="Copy", command=lambda: self.text.event_generate("<<Copy>>"))
        m.add_command(label="Paste", command=lambda: self.text.event_generate("<<Paste>>"))
        m.add_command(label="Select all", command=lambda: self.text.tag_add("sel", "1.0", "end-1c"))
        m.add_separator()
        m.add_command(label="Copy all", command=lambda: self.copy_all(force=True))
        m.add_command(label="Clear text", command=lambda: self.clear_by_user("menu Clear text"))
        if self.cor_restore:
            m.add_command(label="Restore last message", command=self.restore_last)
        m.tk_popup(e.x_root, e.y_root)

    def _show_menu(self, e=None):
        self.menu.entryconfigure(self.restore_index, state="normal" if self.cor_restore else "disabled")
        self.cor_var.set(bool(self.cfg["clear_on_return"]))
        self.paste_var.set(bool(self.cfg["paste_detect"]))
        self.shk_var.set(bool(self.cfg["show_on_hotkey"]))
        self.menu.entryconfigure(self.model_index, label=self._model_label(),
                                 state="disabled" if self.engine in ("loading", "unloading") else "normal")
        self.menu.entryconfigure(self.model_index + 1, state="normal" if self.engine in (
            "ready", "standby", "prewarming", "loading") else "disabled")
        x = e.x_root if e else self.root.winfo_rootx() + self.root.winfo_width() - int(40 * self.s)
        y = e.y_root if e else self.root.winfo_rooty() + int(24 * self.s)
        self.menu.tk_popup(x, y)

    def _model_label(self):
        return {"ready": "Move model to RAM (free VRAM)" if self.cfg["ram_standby"] else "Unload model (free VRAM)",
                "loading": "Loading model...", "unloading": "Unloading model...",
                "standby": "Load model to GPU now (~3 s)" if self.parked else "Load model to GPU (~2 s)",
                "prewarming": "Load model (pre-loading into RAM...)"}.get(self.engine, "Load model")

    # ------------------------------------------------------------------ tray
    def _tray_menu(self):                 # runs on the tray thread; only reads state
        F = trayw
        chk = lambda on: F.MF_CHECKED if on else 0  # noqa: E731
        busy = self.engine in ("loading", "unloading")
        idle = [(T_IDLE0 + i, idle_label(v), chk(self.cfg["idle_unload_min"] == v)) for i, v in enumerate(IDLE_CHOICES)]
        stby = [(T_STBY0 + i, standby_label(v), chk(float(self.cfg["standby_unload_h"]) == v))
                for i, v in enumerate(STANDBY_CHOICES)]
        loaded = self.engine in ("ready", "standby", "prewarming", "loading")
        pk = self.parked
        label = (f"Parked for {pk.get('proc') or 'another app'}" + ("" if pk.get("hold") or pk.get("after_batch")
                 else f" ({pk.get('reason')})")) if pk else ""
        head = [(T_PARKINFO, label, F.MF_GRAYED), None] if pk and not pk.get("temp") else []
        return head + [
            (T_SHOWHIDE, "Hide widget" if self.visible else "Show widget", 0),
            (T_RECORD, ("Stop recording" if self.recording else "Start recording") + f"\t{self._hotkey_label()}", 0),
            (T_MODEL, self._model_label(), F.MF_GRAYED if busy else 0),
            (T_UNLOAD, "Unload model completely (free RAM)", 0 if loaded else F.MF_GRAYED),
            (T_PAUSEPARK, "Pause auto-park", chk(self.park_paused) | (0 if self.cfg["gpu_always"] else F.MF_GRAYED)),
            (T_RESTORE, "Restore last message", 0 if self.cor_restore else F.MF_GRAYED),
            None,
            (T_SAVE, "Save training data", chk(self.cfg["save_training"])),
            ("Settings", [
                (T_SETTINGS, "Open settings...", 0),
                (T_AUTOCOPY, "Auto-copy to clipboard", chk(self.cfg["autocopy"])),
                (T_CLEARRET, "Fresh message after pasting elsewhere", chk(self.cfg["clear_on_return"])),
                (T_PASTEDET, "... Ctrl+V counts as pasting (same window too)",
                 chk(self.cfg["paste_detect"]) | (0 if self.cfg["clear_on_return"] else F.MF_GRAYED)),
                (T_SHOWHK, "Show widget when the hotkey starts recording", chk(self.cfg["show_on_hotkey"])),
                (T_GPUALWAYS, "Keep model on GPU (auto-park for games)", chk(self.cfg["gpu_always"])),
                (T_RAMSTBY, "Keep model in RAM when idle (fast restart)", chk(self.cfg["ram_standby"])),
                (T_PREWARM, "Pre-load at login (RAM, then GPU)", chk(self.cfg["prewarm_login"])),
                ("Free VRAM when idle", idle),
                ("Free RAM after (in standby)", stby),
            ]),
            None,
            (T_QUIT, "Quit Parakeet Live", 0),
            (T_QUITALL, "Quit (stop model too)", 0),
        ]

    def _tray_event(self, ev):
        if ev == "toggle_show":
            self.hide() if self.visible else self.show()
        elif ev == "hotkey":
            self._on_hotkey()
        elif ev == "display":
            self._on_display_change()
        elif ev.startswith("ipc:"):
            cmd, _, arg = ev[4:].partition(":")
            if cmd == "batchhold":
                self._batch_hold(int(arg or 0))
            elif cmd == "batchrelease":
                self._batch_release(int(arg or 0), "released by the launcher")
            else:
                self._ipc(cmd)
        elif ev.startswith("menu:"):
            self._menu_cmd(int(ev[5:]))
        if not self.closing:
            self.dump_state()

    def _ipc(self, cmd):
        {"show": self.show, "hide": self.hide, "toggle": lambda: self._tray_event("toggle_show"),
         "load": self.load_model, "unload": lambda: self.unload_model("manual"),
         "prewarm": lambda: self.prewarm("manual"), "standby": lambda: self.to_standby("manual"),
         "record": self.toggle, "quit": self.quit, "dump": self.dump_state,
         "quitall": lambda: self.quit(stop_server=True),
         "pausepark": lambda: self._set_park_paused(not self.park_paused),
         "silentrec": self._silent_record}.get(cmd, self.show)()

    def _menu_cmd(self, cid):
        if cid == T_SHOWHIDE:
            self.hide() if self.visible else self.show()
        elif cid == T_RECORD:
            self.toggle()
        elif cid == T_MODEL:
            self.toggle_model()
        elif cid == T_SAVE:
            self._set_training(not self.cfg["save_training"])
        elif cid == T_SETTINGS:
            self.show(load=False)
            self.root.after(150, self._show_menu)
        elif cid == T_AUTOCOPY:
            self._set_autocopy(not self.cfg["autocopy"])
        elif cid == T_RESTORE:
            self.restore_last()
        elif cid == T_CLEARRET:
            self._set_clear_on_return(not self.cfg["clear_on_return"])
        elif cid == T_PASTEDET:
            self._set_paste_detect(not self.cfg["paste_detect"])
        elif cid == T_SHOWHK:
            self._set_show_on_hotkey(not self.cfg["show_on_hotkey"])
        elif cid == T_UNLOAD:
            self.unload_model("manual")
        elif cid == T_RAMSTBY:
            self._set_ram_standby(not self.cfg["ram_standby"])
        elif cid == T_PREWARM:
            self._set_prewarm(not self.cfg["prewarm_login"])
        elif cid == T_GPUALWAYS:
            self._set_gpu_always(not self.cfg["gpu_always"])
        elif cid == T_PAUSEPARK:
            self._set_park_paused(not self.park_paused)
        elif T_STBY0 <= cid < T_STBY0 + len(STANDBY_CHOICES):
            self._set_standby_h(STANDBY_CHOICES[cid - T_STBY0])
        elif T_IDLE0 <= cid < T_IDLE0 + len(IDLE_CHOICES):
            self._set_idle(IDLE_CHOICES[cid - T_IDLE0])
        elif cid == T_QUIT:
            self.quit()
        elif cid == T_QUITALL:
            self.quit(stop_server=True)

    def _update_tray(self):
        pk = self.parked if (self.parked and not self.parked.get("temp")) else None
        if self.recording:
            icon, tip = "recording", "Parakeet Live - recording"
        elif self.state == "error" and self.status.cget("text").startswith("GPU busy"):
            icon, tip = "standby", "Parakeet Live - " + self.status.cget("text")
        elif pk and self.engine in ("standby", "prewarming", "unloading", "unloaded"):
            icon = "standby"
            tip = (f"Parakeet Live - parked for {pk.get('proc') or 'another app'} (model in RAM, GPU free; "
                   + ("back when the job ends)" if pk.get("hold") else "returns when it closes)"))
        elif self.engine == "ready":
            icon, tip = "ready", "Parakeet Live - ready (model loaded)"
        elif self.engine == "loading":
            icon, tip = "unloaded", "Parakeet Live - loading model..."
        elif self.engine == "standby":
            icon, tip = "standby", "Parakeet Live - model in RAM, GPU free (ready ~2 s after a click)"
        elif self.engine == "prewarming":
            icon, tip = "unloaded", "Parakeet Live - pre-loading model into RAM..."
        else:
            icon, tip = "unloaded", "Parakeet Live - model not loaded (0 VRAM)"
        if hasattr(self, "tray"):
            self.tray.set_state(icon, tip)

    # ------------------------------------------------------------------ show / hide
    def _hwnd(self):
        return ctypes.windll.user32.GetParent(self.root.winfo_id())

    def _apply_style(self):
        """Tool window: no taskbar button, no Alt-Tab entry; rounded corners on Windows 11."""
        try:
            u = ctypes.windll.user32
            hwnd = self._hwnd()
            if not hwnd:                    # Tk creates the wrapper window lazily
                return
            GWL_EXSTYLE, WS_EX_APPWINDOW, WS_EX_TOOLWINDOW = -20, 0x40000, 0x80
            st = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
            u.SetWindowLongW(hwnd, GWL_EXSTYLE, (st | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW)
            if self.args.selftest:          # automated run: clicks pass through to what is under it (log 57)
                st = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
                if not st & 0x80000:        # WS_EX_LAYERED needs its alpha set, or the window is not drawn
                    u.SetWindowLongW(hwnd, GWL_EXSTYLE, st | 0x80000)
                    u.SetLayeredWindowAttributes(hwnd, 0, max(1, min(255, int(255 * float(self.cfg["opacity"])))), 2)
                u.SetWindowLongW(hwnd, GWL_EXSTYLE, u.GetWindowLongW(hwnd, GWL_EXSTYLE) | 0x20)   # WS_EX_TRANSPARENT
            pref = ctypes.c_int(2)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(pref), 4)
            self.styled = True
        except Exception as e:
            log(f"style: {e}")

    def exstyle(self):
        try:
            hwnd = self._hwnd()
            st = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            return {"toolwindow": bool(st & 0x80), "appwindow": bool(st & 0x40000), "topmost": bool(st & 0x8),
                    "clickthrough": bool(st & 0x20) and bool(st & 0x80000),
                    "iswindowvisible": bool(ctypes.windll.user32.IsWindowVisible(hwnd))}
        except Exception:
            return {}

    def show(self, load=True):
        if self.closing:
            return
        # Order matters: Tk's first "-alpha" rewrites the ex-style and drops WS_EX_TOOLWINDOW
        # (which would add a taskbar button), so set alpha/topmost first, then force the
        # tool-window style, then map the window.
        self.root.update_idletasks()
        self.root.attributes("-topmost", bool(self.cfg["topmost"]))
        self.root.attributes("-alpha", float(self.cfg["opacity"]))
        self._apply_style()
        self.root.deiconify()
        self.root.update_idletasks()
        if not self.styled:
            self._apply_style()
        self.root.lift()
        self._assert_topmost()
        self.visible = True
        if load and self.engine in ("unloaded", "error"):
            self.load_model()
        self.dump_state()

    def hide(self, remember=True):
        self.visible = False                # first, so the <Unmap> from withdraw() is ignored
        if remember:
            self._remember_geometry()
        self.root.withdraw()
        self.dump_state()

    def _is_iconic(self):
        try:
            h = self._hwnd()
            return bool(h and ctypes.windll.user32.IsIconic(h)) or self.root.state() == "iconic"
        except Exception:
            return False

    def _minimized_to_tray(self, why):
        if self.visible and not self.closing:
            log(f"minimize caught ({why}) -> hide to tray")
            self.hide(remember=False)       # geometry of a minimized window is bogus (-32000)

    def _on_unmap(self, e):
        if e.widget is self.root and self.visible and not self.closing:
            self.root.after(0, lambda: self._minimized_to_tray("unmap"))

    def _iconic_watch(self):
        if self.closing:
            return
        if self.visible and self._is_iconic():
            self._minimized_to_tray("iconic")
        self.root.after(400, self._iconic_watch)

    def _assert_topmost(self):
        on = bool(self.cfg["topmost"])
        self.root.attributes("-topmost", on)
        try:
            h = self._hwnd()
            u = ctypes.windll.user32
            if h and on and not (u.GetWindowLongW(h, -20) & 0x8):      # WS_EX_TOPMOST missing
                u.SetWindowPos(h, -1, 0, 0, 0, 0, 0x13)                   # HWND_TOPMOST, NOMOVE|NOSIZE|NOACTIVATE
                log("topmost re-asserted with SetWindowPos")
        except Exception as e:
            log(f"topmost: {e}")

    def _place_window(self):
        s = self.s
        w = int(self.cfg["w"] or 640 * s)
        h = int(self.cfg["h"] or 132 * s)
        u = ctypes.windll.user32
        vx, vy, vw, vh = (u.GetSystemMetrics(i) for i in (76, 77, 78, 79))
        x, y = self.cfg["x"], self.cfg["y"]
        self._displaced_from = None
        if not self._spot_visible(x, y):
            if x is not None and y is not None:
                self._displaced_from = (x, y)  # kept as the saved spot; back there when its monitor is
            x, y = self._default_xy(w, h)
        self.root.minsize(int(380 * s), int(110 * s))
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _spot_visible(self, x, y):
        if x is None or y is None:
            return False
        u = ctypes.windll.user32
        vx, vy, vw, vh = (u.GetSystemMetrics(i) for i in (76, 77, 78, 79))
        return vx <= x <= vx + vw - 60 and vy <= y <= vy + vh - 40 and on_a_monitor(x, y)

    def _default_xy(self, w, h):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        return (sw - w) // 2, sh - h - int(90 * self.s)

    def _on_display_change(self):
        """WM_DISPLAYCHANGE (log 57): re-check where the widget is once the monitors have settled."""
        for j in self._disp_jobs:
            try:
                self.root.after_cancel(j)
            except Exception:
                pass
        self._disp_jobs = [self.root.after(1000, self._check_on_screen),
                           self.root.after(3000, self._check_on_screen)]
        log("display change: re-checking the widget position")

    def _check_on_screen(self):
        if self.closing:
            return
        if self._displaced_from is not None:
            ox, oy = self._displaced_from
            if self._spot_visible(ox, oy):
                self._displaced_from = None
                if self.visible:
                    self.root.geometry(f"+{ox}+{oy}")
                log(f"display change: the saved spot {ox},{oy} is on a monitor again"
                    + (": widget moved back" if self.visible else ""))
                self.dump_state()
            return
        if not self.visible:
            return                         # hidden: the next show places it (and falls back if needed)
        x, y = self.root.winfo_x(), self.root.winfo_y()
        if self._spot_visible(x, y):
            return
        nx, ny = self._default_xy(self.root.winfo_width(), self.root.winfo_height())
        self._displaced_from = (x, y)
        self.root.geometry(f"+{nx}+{ny}")
        log(f"display change: widget at {x},{y} was on no monitor: moved to {nx},{ny}; "
            "it goes back when that monitor is back")
        self.dump_state()

    def _drag_start(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag_move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _resize_start(self, e):
        self._rs = (e.x_root, e.y_root, self.root.winfo_width(), self.root.winfo_height())

    def _resize_move(self, e):
        x0, y0, w0, h0 = self._rs
        self.root.geometry(f"{max(int(380 * self.s), w0 + e.x_root - x0)}x{max(int(110 * self.s), h0 + e.y_root - y0)}")

    def _remember_geometry(self, moved=False):
        if moved:
            self._displaced_from = None    # he put it there: that is the saved spot now
        if self.visible:
            self.cfg.update(w=self.root.winfo_width(), h=self.root.winfo_height())
            if self._displaced_from is None:   # pulled back from a missing monitor: keep the saved spot
                self.cfg.update(x=self.root.winfo_x(), y=self.root.winfo_y())
        self._persist()

    def _zoom(self, e):
        fs = max(8, min(28, self.cfg["font_size"] + (1 if e.delta > 0 else -1)))
        self.cfg["font_size"] = fs
        self.text.configure(font=("Segoe UI", fs))
        self.placeholder.configure(font=("Segoe UI", fs))
        return "break"

    # ------------------------------------------------------------------ record button
    def _set_hover(self, on):
        self._hover = on
        self._refresh_disc()

    def _btn_press(self, e):
        self._press = (e.x_root, e.y_root, e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y(), False)

    def _btn_motion(self, e):
        if not self._press:
            return
        x0, y0, dx, dy, moved = self._press
        if moved or abs(e.x_root - x0) + abs(e.y_root - y0) > 5:
            self._press = (x0, y0, dx, dy, True)
            self.root.geometry(f"+{e.x_root - dx}+{e.y_root - dy}")

    def _btn_release(self, e):
        p, self._press = self._press, None
        if p and p[4]:
            self._remember_geometry(moved=True)
            return
        if 0 <= e.x <= self.D and 0 <= e.y <= self.D:
            self.toggle()

    def _refresh_disc(self):
        h = "_hover" if self._hover else ""
        if self.recording:
            key = "rec" + h
        elif self.engine == "error":
            key = "error"
        elif self.engine in ("loading", "unloading"):
            key = "loading"
        elif self.engine in ("unloaded", "prewarming"):
            key = "unloaded" + h
        else:
            key = "idle" + h
        self.btn.itemconfigure(self.i_disc, image=self.img[key])

    def _canvas_set(self, item, **kw):
        """itemconfigure only when something changed (an idle widget no longer redraws 25x a second)."""
        cache = self.__dict__.setdefault("_canvas_state", {})
        val = tuple(sorted((k, id(v) if k == "image" else v) for k, v in kw.items()))
        if cache.get(item) != val:
            cache[item] = val
            self.btn.itemconfigure(item, **kw)

    def _kick_anim(self):
        """A state changed: run one animation tick now instead of up to 250 ms later."""
        job = getattr(self, "_anim_job", None)
        if job is not None and not self.closing:
            self.root.after_cancel(job)
            self._animate()

    def _animate(self):
        self._anim_job = None
        if self.closing:
            return
        self._anim_k += 1
        spin = self.engine in ("loading", "unloading") or (self.pending > 0 and not self.recording)
        if self.visible:
            if self.recording:
                self._canvas_set(self.i_pulse, image=self.pulse[self._anim_k % 24], state="normal")
                lvl = min(1.0, max(0.0, (self.level_db + 60.0) / 40.0))   # -60..-20 dBFS -> 0..1
                self._canvas_set(self.i_halo, image=self.halos[int(round(lvl * 9))], state="normal")
            else:
                self._canvas_set(self.i_pulse, state="hidden")
                self._canvas_set(self.i_halo, state="hidden")
            if spin:
                self._canvas_set(self.i_spin, image=self.spin[self._anim_k % 12], state="normal")
            else:
                self._canvas_set(self.i_spin, state="hidden")
        moving = self.visible and (self.recording or spin)
        self._anim_job = self.root.after(40 if moving else 250, self._animate)

    def _heartbeat(self):
        now = time.perf_counter()
        fast = self.recording or self.pending > 0
        if getattr(self, "_hb_fast", False):  # UI responsiveness is only measured at the fast cadence
            self.ui_gaps.append((now - self._hb_last) * 1000)
            if len(self.ui_gaps) > 5000:
                self.ui_gaps = self.ui_gaps[-2500:]
        self._hb_last = now
        self._hb_fast = fast
        if not self.closing:
            self.root.after(16 if fast else 250, self._heartbeat)

    # ------------------------------------------------------------------ status
    def set_state(self, state, msg=None):
        self.state = state
        colors = {"loading": ACCENT, "ready": GREEN, "listening": RED, "transcribing": AMBER,
                  "error": RED, "unloaded": MUTED}
        self.dot.itemconfigure(self.dot_id, fill=colors.get(state, MUTED))
        self.status.configure(text=msg or state.capitalize(), fg=RED if state == "error" else MUTED)
        self._kick_anim()
        self._refresh_disc()
        self._update_tray()

    def _update_status(self):
        if self.engine in ("loading", "unloading"):
            msg = "Loading model..." if self.engine == "loading" else "Unloading model..."
            if self.recording:
                msg += "  \u00b7  listening (text appears once loaded)"
            self.set_state("loading", msg)
            return
        if self.engine == "error":
            return
        if self.recording:
            dev = (self.mic.info.split(" (")[0] if self.mic and getattr(self.mic, "info", "") else "mic")
            dev = dev if len(dev) <= 34 else dev[:33] + "..."
            msg = f"Listening  \u00b7  {dev}" + (f"  \u00b7  transcribing {self.pending}" if self.pending else "")
            self.set_state("listening", msg)
        elif self.pending:
            self.set_state("transcribing", "Transcribing...")
        elif self.parked and not self.parked.get("temp") and self.engine in ("standby", "prewarming"):
            self.set_state("unloaded", f"Parked for {self.parked.get('proc') or 'another app'}  \u00b7  "
                                       f"model in RAM, GPU free  \u00b7  back on GPU when "
                                       + ("the job ends" if self.parked.get("hold") else "it closes"))
        elif self.engine == "prewarming":
            self.set_state("unloaded", "Pre-loading into RAM (0 VRAM)  \u00b7  click the mic to start")
        elif self.engine == "standby":
            self.set_state("unloaded", f"In RAM, GPU free  \u00b7  click the mic or {self._hotkey_label()} (~2 s)")
        elif self.engine == "unloaded":
            self.set_state("unloaded", "Model not loaded (0 VRAM)  \u00b7  click the mic to start")
        else:
            self.set_state("ready", f"Ready  \u00b7  {self._hotkey_label()} to talk")

    def _hotkey_label(self):
        return "+".join(p.capitalize() for p in self.cfg["hotkey"].split("+"))

    def _flash_copied(self, ok=True):
        steps = [GREEN, GREEN, GREEN, GREEN, "#6fae84", "#557f66", "#3c5249", "#262d2f", BG]
        if not ok:
            steps = [RED] * 14 + ["#a85a5a", "#7a4646", "#4c3436", "#262d2f", BG]
        self.flash.configure(text="\u2713 Copied" if ok else "\u2717 Not copied")

        def step(i=0):
            if i < len(steps):
                self.flash.configure(fg=steps[i])
                self._flash_job = self.root.after(140, step, i + 1)
        if getattr(self, "_flash_job", None):
            self.root.after_cancel(self._flash_job)
        step()

    def _update_td_indicator(self):
        if self.cfg["save_training"]:
            if not self.td_ind.winfo_manager():
                self.td_ind.pack(side="right", before=self.flash)
        else:
            self.td_ind.pack_forget()

    def dump_state(self):
        try:
            d = {"time": time.strftime("%H:%M:%S"), "pid": os.getpid(), "engine": self.engine,
                 "visible": self.visible, "recording": self.recording, "pending": self.pending,
                 "port": self.port, "server_alive": bool(self.server and self.server.alive()),
                 "server_adopted": self.server_adopted,
                 "tray_added": self.tray.added, "tray_rect": self.tray.icon_rect() if self.tray.hwnd else None,
                 "hotkey_ok": self.tray.hotkey_ok, "failed_kept": len(self.failed), "copy_count": self.copy_count, "copy_failures": self.copy_failures, "save_training": self.cfg["save_training"],
                 "idle_unload_min": self.cfg["idle_unload_min"], "status": self.status.cget("text"),
                 "training_records": len(self.training.records), "training_errors": self.training.errors[-3:],
                 "ram_standby": self.cfg["ram_standby"], "prewarm_login": self.cfg["prewarm_login"],
                 "standby_unload_h": self.cfg["standby_unload_h"], "server_status": self.health.get("status"),
                 "server_rss_mib": self.health.get("rss_mib"), "server_timings": self.health.get("timings"),
                 "activate_s": self.health.get("activate_s"), "load_s": self.health.get("load_s"),
                 "data_dir": DATA_DIR, "training_dir": self.training.dir, "app_dir": HERE,
                 "state_dir": self.store.dir if self.store else None,
                 "displaced_from": self._displaced_from,
                 "gpu_always": self.cfg["gpu_always"], "auto_park": self.cfg["auto_park"],
                 "park_paused": self.park_paused, "parked": self.parked, "park_pending": self.park_pending,
                 "goal_gpu": self.goal_gpu, "gpu_last": self.gpu_last, "park_log": self.park_log[-10:],
                 "watch": {**self.watch_stats, "cpu_ms_per_poll": round(1000 * self.watch_stats["cpu_s"] /
                                                                         max(1, self.watch_stats["polls"]), 2)},
                 "unpark_needs_free_mb": self._unpark_needed(),
                 "batch_holds": [{"pid": p, "since": h["since"], "via": h["via"]} for p, h in self.holds.items()],
                 "wsl_hold": self.wsl_hold,
                 "clear_on_return": self.cfg["clear_on_return"], "show_on_hotkey": self.cfg["show_on_hotkey"],
                 "geometry": [self.root.winfo_x(), self.root.winfo_y(), self.root.winfo_width(), self.root.winfo_height()],
                 "cor": {"armed": self.cor_armed, "left": self.cor_left, "restore_pending": bool(self.cor_restore),
                         "box_chars": len(self.text.get("1.0", "end-1c")),
                         "box_is_copied": self.text.get("1.0", "end-1c") == self._copied_text,
                         "paste_detect": self.cfg["paste_detect"], "paste_watch": self._paste_job is not None},
                 "tray_tip": getattr(self.tray, "tip", None)}
            d.update(self.exstyle())
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(STATE_PATH + ".tmp", "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            os.replace(STATE_PATH + ".tmp", STATE_PATH)
        except Exception as e:
            log(f"dump_state: {e}")

    # ------------------------------------------------------------------ text, clipboard, training spans
    def _update_placeholder(self):
        if self.text.get("1.0", "end-1c"):
            self.placeholder.place_forget()
        else:
            self.placeholder.configure(text=(
                "New message. Ctrl+Z brings back the previous one." if self.cor_restore else
                f"Click the mic or press {self._hotkey_label()} and talk."))
            self.placeholder.place(x=int(11 * self.s), y=int(7 * self.s))

    def _on_modified(self, e=None):
        if not self.text.edit_modified():
            return
        self.text.edit_modified(False)
        self._draft_dirty()
        self._update_placeholder()
        if self.cfg["autocopy"]:
            if self._deb:
                self.root.after_cancel(self._deb)
            self._deb = self.root.after(400, self._debounced_copy)
        if self.spans:
            if self._tdeb:
                self.root.after_cancel(self._tdeb)
            self._tdeb = self.root.after(1000, self.sync_training_spans)

    def _debounced_copy(self):
        self._deb = None
        if self.cfg["autocopy"]:
            self.copy_all()

    def append_text(self, t, rid=None, session=None):
        t = t.strip()
        if not t:
            return False
        txt = self.text
        cur = txt.get("1.0", "end-1c")
        sep = "" if (not cur or cur[-1].isspace()) else " "
        end_before = txt.index("end-1c")
        txt.insert("end", sep)
        start = txt.index("end-1c")
        txt.insert("end", t)
        end = txt.index("end-1c")
        # end marks have right gravity: put back the ones this insert dragged along
        for other in self.spans:
            m = f"ue_{other}"
            if txt.compare(m, ">", end_before):
                txt.mark_set(m, end_before)
        if rid:
            txt.mark_set(f"us_{rid}", start)
            txt.mark_gravity(f"us_{rid}", "left")       # typing at the very start stays in this span
            txt.mark_set(f"ue_{rid}", end)
            txt.mark_gravity(f"ue_{rid}", "right")      # typing at the very end stays in this span
            self.spans[rid] = t
        txt.see("end")
        self._update_placeholder()
        # new text: "since the last text arrived" starts again - unless it is the tail of a recording that
        # began before you left (it arrived late; you took that message away already)
        keep = (self.cor_left and session is not None and self._cor_left_session is not None
                and session <= self._cor_left_session)
        self.cor_armed, self.cor_left = False, keep
        self._cor_dictated = txt.get("1.0", "end-1c")
        if self.cor_restore is not None:
            self._cor_prog_text = txt.get("1.0", "end-1c")
        if self.cfg["autocopy"]:
            self.copy_all()
        return True

    def sync_training_spans(self):
        """Map the box text back to each utterance span and record corrections (debounced)."""
        self._tdeb = None
        for rid, last in list(self.spans.items()):
            try:
                cur = self.text.get(f"us_{rid}", f"ue_{rid}")
            except tk.TclError:
                continue
            if cur != last:
                self.spans[rid] = cur
                self.training.update(rid, cur)

    def copy_all(self, force=False):
        txt = self.text.get("1.0", "end-1c")
        if not txt.strip():
            return            # never wipe the clipboard with an empty box
        if not force and txt == self._last_copied:
            return
        self._last_copied = txt
        self._cor_fg_req = self._fg_info()[0]
        with self.clip_lock:
            self.clip_text = txt
        self.clip_evt.set()

    def _clip_loop(self):
        while True:
            self.clip_evt.wait()
            self.clip_evt.clear()
            with self.clip_lock:
                txt, self.clip_text = self.clip_text, None
            if txt is None:
                if self.closing:
                    return
                continue
            ok = core.set_clipboard(txt)
            if not ok:                      # another app is holding the clipboard: one more try (log 55)
                time.sleep(0.3)
                with self.clip_lock:
                    newer = self.clip_text is not None
                if newer:
                    continue                # a newer copy is queued; it supersedes this one
                ok = core.set_clipboard(txt)
            self.uiq.put(("copied", ok, txt))

    def clear_text(self):
        self.sync_training_spans()           # save final corrections before the spans disappear
        for rid in self.spans:
            for m in (f"us_{rid}", f"ue_{rid}"):
                try:
                    self.text.mark_unset(m)
                except tk.TclError:
                    pass
        self.spans.clear()
        self.text.delete("1.0", "end")
        self._last_copied = None
        self.cor_armed = self.cor_left = False
        self._cor_dictated = ""
        self._update_placeholder()

    # ------------------------------------------------------------------ survive restarts (log 57)
    def _draft_dirty(self):
        """The box or its clear-on-return state changed: save the draft 1.5 s after the last change."""
        return    # 2026-10-09: a new launch starts empty, so the box text is no longer written to disk
        if self.store is None or self.closing:
            return
        if self._draft_job is not None:
            self.root.after_cancel(self._draft_job)
        self._draft_job = self.root.after(1500, self._save_draft)

    def _save_draft(self):
        self._draft_job = None
        return    # 2026-10-09: no cross-launch restore (see _restore_session)
        if self.store is None:
            return
        box = self.text.get("1.0", "end-1c")
        d = {"text": box, "spans": self._span_offsets(),
             "cor": {"armed": self.cor_armed, "left": self.cor_left, "copied": box == self._copied_text}}
        try:
            self.store.save_draft(d)
        except Exception as e:
            log(f"draft not saved: {e.__class__.__name__}")

    def _restore_session(self):
        """At start: put the unsent message back (without touching the clipboard) and re-queue kept retries."""
        if self.store is None:
            return
        try:
            d = self.store.load_draft()
        except Exception as e:
            d = None
            log(f"draft not restored: {e.__class__.__name__}")
        if d:     # 2026-10-09: a new launch starts with an empty box; the old message is discarded, not restored
            try:
                self.store.save_draft({"text": ""})
            except Exception:
                pass
            log(f"previous session's unsent message discarded ({len(d['text'])} chars); the box starts empty")
            d = None
        if d and not self.text.get("1.0", "end-1c"):
            t = d["text"]
            self._last_copied = t              # never re-copy it at start: the clipboard may hold newer things
            self.text.insert("1.0", t)
            for sp in d.get("spans") or []:
                try:
                    rid, a, b, last = sp
                    self.text.mark_set(f"us_{rid}", f"1.0+{int(a)}c")
                    self.text.mark_gravity(f"us_{rid}", "left")
                    self.text.mark_set(f"ue_{rid}", f"1.0+{int(b)}c")
                    self.text.mark_gravity(f"ue_{rid}", "right")
                    self.spans[rid] = last
                except (ValueError, TypeError, tk.TclError):
                    pass
            cor = d.get("cor") or {}
            if cor.get("copied"):
                self._copied_text = t
                self.cor_left = bool(cor.get("left"))
                self.cor_armed = bool(cor.get("armed")) or self.cor_left
                if self.cor_left:
                    self._cor_left_session = self.rec_session
                elif self.cor_armed:
                    self._cor_fg = self._fg_info()[0]
                    if self._cor_job is None:
                        self._cor_job = self.root.after(250, self._cor_watch)
            self._cor_dictated = t
            self.text.edit_reset()
            self.text.see("end")
            self._update_placeholder()
            age = (time.time() - float(d.get("saved") or time.time())) / 60
            log(f"restored the unsent message: {len(t)} chars, saved {age:.0f} min ago"
                + (" (already pasted elsewhere: the next recording starts fresh)" if self.cor_left else ""))
        try:
            items = self.store.load_pending(RETRY_MAX_AGE_S)
        except Exception as e:
            items = []
            log(f"kept utterances not restored: {e.__class__.__name__}")
        now = time.monotonic()
        for kid, pcm, meta in items:
            fa = float(meta["failed_at"])
            self.failed.append({"t": now - max(0.0, time.time() - fa), "pcm": pcm, "dur": float(meta.get("dur", 0)),
                                "tries": int(meta.get("tries", 1)), "kid": kid, "failed_at": fa})
        if items:
            log(f"restored {len(items)} kept utterance(s); retried when the engine is ready")

    # ------------------------------------------------------------------ clear-on-return (log 53)
    def _fg_info(self):
        """(foreground hwnd, is it one of our own windows?) - our widget, its menus, the tray menu."""
        if self._fg_override is not None:
            return self._fg_override
        h = _u32.GetForegroundWindow()
        if not h:
            return None, False
        pid = wt.DWORD(0)
        _u32.GetWindowThreadProcessId(h, ctypes.byref(pid))
        return h, pid.value == os.getpid()

    def _cor_arm(self, copied):
        """The clipboard now holds `copied`. If that is the box text, watch for you switching away."""
        self._copied_text = copied
        if copied != self.text.get("1.0", "end-1c"):
            return
        if self.cor_left and copied == self._cor_dictated:
            self.cor_armed = True              # late dictation re-copied: you had already left with it
            self._paste_kick()                 # recording: a Ctrl+V still stops it (2026-10-09)
            return
        self.cor_armed, self.cor_left = True, False
        # start from the window that was in front when the copy was requested (text arrived / edit /
        # Copy button), not when the clipboard thread confirmed it: a switch in between still counts
        self._cor_fg = self._cor_fg_req if self._cor_fg_req is not None else self._fg_info()[0]
        if self._cor_job is None:
            self._cor_job = self.root.after(250, self._cor_watch)
        self._paste_kick()
        self._draft_dirty()

    def _cor_watch(self):
        """4x a second while armed: any switch of the foreground to a window that is not ours = you left
        (e.g. to paste). Our own windows (widget, menus) don't count. Stops once left or disarmed."""
        self._cor_job = None
        if self.closing or not self.cor_armed or self.cor_left:
            return
        h, own = self._fg_info()
        if h and self._cor_fg is None:         # nothing was in front when armed (mid-switch): start from here
            self._cor_fg = h
        elif h and h != self._cor_fg:
            if not own:
                self.cor_left = True
                self._cor_left_session = self.rec_session
                log("clear-on-return: switched to another window after the text was copied")
                self._draft_dirty()
                self.dump_state()
                self._paste_kick()             # recording: a Ctrl+V there still stops it (2026-10-09)
                return
            self._cor_fg = h
        self._paste_kick()                     # (re)start the Ctrl+V watch once the widget is not in front
        self._cor_job = self.root.after(250, self._cor_watch)

    # ---- same-window paste (log 57): Ctrl+V after the copy counts as "pasted", like switching away
    def _paste_watching(self):
        """Is there anything to watch? Copied text waiting, not left yet, both settings on, widget not in front."""
        # 2026-10-09: while recording, keep looking after you switched away too (copy in the widget,
        # click into the chat, Ctrl+V): that Ctrl+V means "stop" (_paste_stop). Still only while copied
        # text is waiting; when not recording, leaving ends the watch as before.
        if (self.closing or not self.cor_armed or (self.cor_left and not self.recording)
                or not self.cfg["clear_on_return"] or not self.cfg["paste_detect"]):
            return False
        h, own = self._fg_info()
        return bool(h) and not own

    def _paste_kick(self):
        if self._paste_job is None and self._paste_watching():
            self._ctrl_v()                     # discard a stale "pressed since last look" bit from before
            self._paste_job = self.root.after(40, self._paste_watch)

    def _ctrl_v(self):
        """Ctrl (without Alt, so AltGr letters don't count) and V down, or V pressed since the last look.
        Reads only these keys; returns a bool and keeps nothing."""
        self.paste_polls += 1
        v = _u32.GetAsyncKeyState(VK_V) & 0x8001
        return bool(v and _u32.GetAsyncKeyState(VK_CONTROL) & 0x8000 and not _u32.GetAsyncKeyState(VK_MENU) & 0x8000)

    def _paste_watch(self):
        """25x a second, only while _paste_watching(); stops by itself otherwise (the foreground watch
        restarts it when the widget is no longer in front)."""
        self._paste_job = None
        if not self._paste_watching():
            return
        if self._ctrl_v():
            if self.recording:
                self._paste_stop()
                return
            self.cor_left = True
            self._cor_left_session = self.rec_session
            log("clear-on-return: Ctrl+V after the text was copied (pasted in the same window)")
            self._draft_dirty()
            self.dump_state()
            return
        self._paste_job = self.root.after(40, self._paste_watch)

    def _paste_stop(self):
        """Ctrl+V while recording (2026-10-09): you meant to stop and forgot. Same as pressing stop (what was
        captured is still transcribed and stays in the box), and the pasted message leaves the box like a
        clear-on-return clear (Ctrl+Z / 'Restore last message' brings it back). Nothing new is watched:
        this rides on the Ctrl+V watch above, which only runs while copied text is waiting."""
        log("paste detected while recording -> stopped")
        pasted = self._copied_text or ""
        box = self.text.get("1.0", "end-1c")
        if pasted.strip() and box == pasted:
            self._cor_clear("pasted while recording")
        elif pasted.strip() and box.startswith(pasted):
            self._cor_clear_prefix(pasted, "pasted while recording")   # text that came after the copy stays
        self.cor_armed = self.cor_left = False   # that paste is used up; the tail gets copied on its own
        self.stop_recording()
        self.dump_state()

    def _cor_clear_prefix(self, pasted, why):
        """Like _cor_clear, but only the pasted start of the box goes; text that arrived after the copy
        stays (restorable the same way, put back in front of it)."""
        self.sync_training_spans()
        box = self.text.get("1.0", "end-1c")
        rest = box[len(pasted):]
        n = len(pasted) + len(rest) - len(rest.lstrip())
        gone = [s for s in self._span_offsets() if s[2] <= len(pasted)]
        self.cor_restore = {"text": pasted, "spans": gone, "why": why, "t": time.strftime("%H:%M:%S")}
        self.text.edit_separator()
        for rid, *_ in gone:
            for m in (f"us_{rid}", f"ue_{rid}"):
                try:
                    self.text.mark_unset(m)
                except tk.TclError:
                    pass
            self.spans.pop(rid, None)
        self.text.delete("1.0", f"1.0+{n}c")
        self.text.edit_separator()
        self._cor_prog_text = self.text.get("1.0", "end-1c")
        self._cor_dictated = self._cor_prog_text
        self._cor_clear_session = None
        self._last_copied = None
        self._update_placeholder()
        log(f"clear-on-return: cleared {len(pasted)} pasted chars ({why}), kept {len(self._cor_prog_text)} "
            "chars that came after the copy; Ctrl+Z or 'Restore last message' brings the cleared part back")
        if self.cfg["autocopy"]:
            self.copy_all()
        self.dump_state()

    def _cor_should_clear(self):
        """Copied, then another window had the focus since, and the box was not edited after the copy.
        Where you start from does not matter (hotkey, record button, widget clicked first: log 55)."""
        if not self.cfg["clear_on_return"] or self.recording or not self.cor_left:
            return False
        box = self.text.get("1.0", "end-1c")
        return bool(box.strip()) and box in (self._copied_text, self._last_copied)

    def _start_recording_cor(self, source=None):
        """Every way of starting a recording: clear a message you already took elsewhere first."""
        fresh = self._cor_should_clear()
        if fresh:
            self._cor_clear("new recording after pasting elsewhere")
            self._cor_clear_session = self.rec_session + 1    # = the recording started next
        self.start_recording(source=source)
        if fresh and not self.recording:       # recording refused (GPU busy, mic error): give the text back
            self.restore_last(quiet=True)

    def _cor_add_late(self, t, rid=None):
        """Text of a recording that began before the clear: it belongs to the cleared message."""
        t = t.strip()
        r = self.cor_restore
        if not t or r is None:
            return False
        sep = "" if (not r["text"] or r["text"][-1].isspace()) else " "
        a = len(r["text"]) + len(sep)
        r["text"] += sep + t
        if rid:
            r["spans"].append((rid, a, a + len(t), t))
        log(f"clear-on-return: {len(t)} late chars added to the cleared message (Ctrl+Z brings it back)")
        return True

    def _on_hotkey(self, source=None):
        """Global hotkey: toggle recording. Starting a recording also
        - shows the widget if it is hidden (show_on_hotkey; saved position/size, topmost, no focus taken),
        - clears a message you already copied and took elsewhere (clear_on_return, restorable).
        Stopping leaves the widget as it is (visible stays visible)."""
        if self.recording:
            self.stop_recording()
            return
        if self.cfg["show_on_hotkey"] and not self.visible:
            self._show_for_hotkey()
        self._start_recording_cor(source)

    def _show_for_hotkey(self):
        """Show the hidden widget at its saved position and size without taking keyboard focus: Tk's
        deiconify maps a withdrawn toplevel with SW_SHOWNOACTIVATE and raises it with SWP_NOACTIVATE, and
        it only forces focus for windows that are not override-redirect (ours is); show() keeps
        WS_EX_TOOLWINDOW (no taskbar button) and topmost. Nothing here calls SetForegroundWindow."""
        self._place_window()                   # saved geometry (a minimized-to-tray window's own is bogus)
        self.show(load=False)                  # start_recording loads the model itself if needed
        log(f"hotkey: widget shown at {self.root.winfo_x()},{self.root.winfo_y()} "
            f"{self.root.winfo_width()}x{self.root.winfo_height()} (no focus change)")

    def _set_show_on_hotkey(self, on):
        self.cfg["show_on_hotkey"] = bool(on)
        if hasattr(self, "shk_var"):
            self.shk_var.set(bool(on))
        self._persist()
        log(f"show_on_hotkey = {bool(on)}")
        self.dump_state()

    def _span_offsets(self):
        out = []
        for rid, last in self.spans.items():
            try:
                a = len(self.text.get("1.0", f"us_{rid}"))
                b = len(self.text.get("1.0", f"ue_{rid}"))
            except tk.TclError:
                continue
            out.append((rid, a, b, last))
        return out

    def _cor_clear(self, why):
        self.sync_training_spans()
        text = self.text.get("1.0", "end-1c")
        self.cor_restore = {"text": text, "spans": self._span_offsets(), "why": why, "t": time.strftime("%H:%M:%S")}
        self.text.edit_separator()
        self.clear_text()                      # training records stay; only the box and its span marks go
        self.text.edit_separator()
        self._cor_prog_text = ""
        self._cor_clear_session = None
        self._update_placeholder()
        log(f"clear-on-return: cleared {len(text)} chars ({why}); Ctrl+Z or 'Restore last message' brings them back")
        self.dump_state()

    def clear_by_user(self, why):
        """Clear button / menu: same as before, but restorable with Ctrl+Z / 'Restore last message'."""
        if self.text.get("1.0", "end-1c").strip():
            self._cor_clear(why)
        else:
            self.clear_text()

    def restore_last(self, quiet=False):
        """One step: put the last cleared message back (in front of anything dictated since), with its
        training spans, so later edits still correct the saved training records."""
        r, self.cor_restore = self.cor_restore, None
        self._cor_prog_text = None
        self._cor_clear_session = None
        if not r:
            return False
        self.sync_training_spans()
        cur = self.text.get("1.0", "end-1c")
        sep = "" if (not cur or r["text"][-1:].isspace()) else " "
        shift = len(r["text"]) + len(sep)
        spans = r["spans"] + [(rid, a + shift, b + shift, last) for rid, a, b, last in self._span_offsets()]
        for rid in list(self.spans):
            for m in (f"us_{rid}", f"ue_{rid}"):
                try:
                    self.text.mark_unset(m)
                except tk.TclError:
                    pass
        self.spans.clear()
        self.text.edit_separator()
        self.text.delete("1.0", "end")
        self.text.insert("1.0", r["text"] + sep + cur)
        self.text.edit_separator()
        for rid, a, b, last in spans:
            self.text.mark_set(f"us_{rid}", f"1.0+{a}c")
            self.text.mark_gravity(f"us_{rid}", "left")
            self.text.mark_set(f"ue_{rid}", f"1.0+{b}c")
            self.text.mark_gravity(f"ue_{rid}", "right")
            self.spans[rid] = last
        self.text.see("end")
        self._cor_dictated = self.text.get("1.0", "end-1c")
        self._update_placeholder()
        log(f"clear-on-return: restored {len(r['text'])} chars (cleared {r['t']}, {r['why']})"
            + (f"; {len(cur)} chars dictated since kept after it" if cur else "") + (" [auto]" if quiet else ""))
        self.dump_state()
        return True

    def _on_ctrl_z(self, e=None):
        """Ctrl+Z in the box: restore the cleared message while nothing was typed since the clear;
        otherwise normal undo."""
        if self.cor_restore and self.text.get("1.0", "end-1c") == self._cor_prog_text:
            self.restore_last()
            return "break"
        return None

    def _set_clear_on_return(self, on):
        self.cfg["clear_on_return"] = bool(on)
        if hasattr(self, "cor_var"):
            self.cor_var.set(bool(on))
        self._persist()
        log(f"clear_on_return = {bool(on)}")
        self.dump_state()

    def _set_paste_detect(self, on):
        self.cfg["paste_detect"] = bool(on)
        if hasattr(self, "paste_var"):
            self.paste_var.set(bool(on))
        if not on and self._paste_job is not None:
            self.root.after_cancel(self._paste_job)
            self._paste_job = None
        self._persist()
        log(f"paste_detect = {bool(on)}")
        self.dump_state()

    # ------------------------------------------------------------------ settings
    def _persist(self):
        if not self.args.no_save:
            save_config(self.cfg)

    def _set_mic(self):
        self.cfg["device"] = self.mic_var.get() or None
        self._persist()
        if self.recording:
            self.stop_recording()
            self.start_recording()

    def _set_silence(self):
        self.cfg["silence_s"] = float(self.sil_var.get())
        if self.seg is not None:
            self.seg.silence_s = self.cfg["silence_s"]
        self._persist()

    def _set_idle(self, m):
        self.cfg["idle_unload_min"] = m
        self.idle_var.set(str(m))
        self._persist()

    def _set_standby_h(self, h):
        self.cfg["standby_unload_h"] = h
        self.stby_var.set(f"{float(h):g}")
        self._persist()

    def _set_ram_standby(self, on):
        self.cfg["ram_standby"] = bool(on)
        self.rs_var.set(bool(on))
        self._persist()
        log(f"keep model in RAM when idle: {on}")

    def _set_prewarm(self, on):
        self.cfg["prewarm_login"] = bool(on)
        self.pw_var.set(bool(on))
        self._persist()
        log(f"pre-load into RAM at login: {on}")

    def _set_gpu_always(self, on):
        self.cfg["gpu_always"] = bool(on)
        self.ga_var.set(bool(on))
        self._persist()
        log(f"GPU policy (keep model on GPU, auto-park): {on}")
        if not on and not self._held():
            self.parked, self.park_pending = None, False
        elif on and not self.parked and self.engine in ("standby", "prewarming", "unloaded", "error"):
            self.load_model()
        self.watch_evt.set()
        self._update_status()

    def _set_park_paused(self, on):
        self.park_paused = bool(on)
        self.pp_var.set(bool(on))
        log(f"auto-park paused: {on}")
        if on and self.parked:
            self._unpark("auto-park paused")
        self.park_pending = False if on else self.park_pending
        self.watch_evt.set()
        self._update_status()

    def _set_opacity(self):
        self.cfg["opacity"] = float(self.op_var.get())
        self.root.attributes("-alpha", self.cfg["opacity"])
        self._apply_style()
        self._persist()

    def _set_autocopy(self, on):
        self.cfg["autocopy"] = bool(on)
        self.ac_var.set(bool(on))
        self._persist()

    def _set_training(self, on):
        if not on:
            self.sync_training_spans()
        self.cfg["save_training"] = bool(on)
        self.td_var.set(bool(on))
        self._update_td_indicator()
        self._persist()
        log(f"save training data: {on}")

    def _set_topmost(self):
        self.cfg["topmost"] = bool(self.top_var.get())
        self._assert_topmost()
        self._persist()

    # ------------------------------------------------------------------ model (WSL server) lifecycle
    # engine: unloaded -> loading (a user waits; GPU) -> ready
    #         unloaded -> prewarming (RAM only, nice) -> standby (0 VRAM) -> loading (/activate, ~2 s) -> ready
    #         ready --idle--> standby (server restarted RAM-only: VRAM back to baseline at once)
    #         standby --hours--> unloaded (frees the RAM; the WSL VM then idles out by itself)
    def toggle_model(self):
        if self.engine == "ready":
            if self.cfg["ram_standby"]:
                self.to_standby("manual")
            else:
                self.unload_model("manual")
        elif self.engine in ("unloaded", "error", "standby", "prewarming"):
            self.load_model()

    def load_model(self):
        """The user wants the model on the GPU now."""
        if self.closing or self.engine in ("loading", "ready"):
            return
        if self._held():                # a batch job owns the GPU right now
            log("load refused: a batch transcription holds the GPU")
            self.set_state("error", "GPU busy - batch transcription running. Try again when it finishes.")
            self.root.after(6000, self._recover_status)
            return
        self.ready_evt.clear()
        self.last_activity = time.monotonic()
        self.load_t0 = time.perf_counter()
        self.want_gpu = True
        if self.engine in ("standby", "prewarming") and self.server is not None and self.server.alive():
            log(f"activating from {self.engine}")
            self.engine = "loading"
            self.wake_evt.set()                 # the server loop posts /activate right away
            self._update_status()
            return
        self.gen += 1
        gen = self.gen
        self.engine = "loading"
        self._update_status()
        log("loading model (cold start)")
        threading.Thread(target=self._engine_thread, args=(gen, 0, "gpu"), name="engine", daemon=True).start()

    def _try_adopt(self):
        """keep-server (2026-10-09): reconnect to the model server the previous widget run left running, so a
        widget restart is ready in ~1-2 s instead of a ~20 s cold load. It must answer /health with the saved
        per-server secret (a 401 or no answer = not ours), be the same process (pid + start time) and be ready
        (GPU) or in standby (RAM). Anything else: the usual cold start, which first kills any leftover server."""
        if self.srv_rec is None:
            return False
        rec = self.srv_rec.load()
        if not rec:
            return False
        self.srv_rec.clear()                 # one use; rewritten once the server answers again
        why, h = "", None
        try:
            h = core.Client(int(rec["port"]), token=rec["token"]).health(timeout=2.0)
        except Exception as e:
            why = type(e).__name__
        if not (h and core.ServerRecord.matches(rec, h)):
            log(f"keep-server: previous model server not reused ({why or (h or {}).get('status') or 'mismatch'}); "
                "cold start as usual")
            return False
        self.port = int(rec["port"])
        self.server_token = rec["token"]
        self.tclient = core.Client(self.port, token=self.server_token)
        st = h.get("status")
        self.server = core.ServerProcess.adopt(self.port, os.path.join(LOG_DIR, "server.log"), self.server_token,
                                               mode="gpu" if st == "ready" else "standby")
        self.server_adopted = True
        self.health = h
        self.gen += 1
        gen = self.gen
        self.ready_evt.clear()
        self.load_t0 = time.perf_counter()
        if st == "ready":
            self.want_gpu, self.engine = True, "loading"
        else:                                # in RAM: the standby handler moves it on (GPU policy, not parked)
            self.want_gpu, self.engine = False, "prewarming"
            self.goal_gpu = bool(self.cfg["gpu_always"])
        log(f"keep-server: reconnected to the running model server ({st}) on port {self.port}")
        self._update_status()
        threading.Thread(target=self._server_loop, args=(gen, 0, self.server.mode), name="engine",
                         daemon=True).start()
        return True

    def _login_load(self):
        """Login (2026-10-09): model straight onto the GPU at normal priority. Only while a game/batch job
        holds the GPU does it fall back to the RAM pre-load (which moves on to the GPU when it is free)."""
        if self.closing or self.engine not in ("unloaded", "error"):
            return
        if self.parked or self._held():
            self.prewarm("login pre-warm (GPU busy)", then_gpu=True)
        else:
            self.load_model()

    def prewarm(self, reason="", then_gpu=False):
        """Load into CPU RAM only (0 VRAM), at low priority, so the first click only needs ~2 s.
        then_gpu: login with the GPU policy - continue to the GPU once in RAM, unless parked."""
        if self.closing or self.engine not in ("unloaded", "error"):
            return
        self.goal_gpu = bool(then_gpu)
        self.gen += 1
        gen = self.gen
        self.want_gpu = False
        self.ready_evt.clear()
        self.engine = "prewarming"
        self._update_status()
        log(f"pre-loading into RAM ({reason})")
        threading.Thread(target=self._engine_thread, args=(gen, 0, "standby"), name="engine", daemon=True).start()

    def to_standby(self, reason=""):
        """Free the VRAM but keep the model in RAM: the GPU server exits, a RAM-only one takes over."""
        if self.engine == "ready":
            self.unload_model(reason, then_standby=True)

    def _engine_thread(self, gen, attempt=0, mode="gpu"):
        with self.engine_lock:
            if gen != self.gen:
                return
            try:
                if core.kill_stale_servers():
                    log("killed a stale live_server.py left in WSL")
                if self.srv_rec is not None:
                    self.srv_rec.clear()
                if not getattr(self.args, "server_token", None):
                    self.server_token = secrets.token_hex(16)   # new server, new secret (keep-server)
                self.server_adopted = False
                port = core.pick_port(self.cfg["port"] + attempt)
                if port != self.cfg["port"]:
                    log(f"port {self.cfg['port']} busy, using {port} this session")
                self.port = port
                self.tclient = core.Client(port, token=self.server_token)
                self.server = core.ServerProcess(port, os.path.join(LOG_DIR, "server.log"), idle_exit=90,
                                                 mode=mode, low_priority=(mode == "standby"),
                                                 token=self.server_token)
                self.server.start()
            except Exception as e:
                self.uiq.put(("server_error", gen, f"Could not start speech engine: {e}"))
                return
        self._server_loop(gen, attempt, mode)

    def _server_loop(self, gen, attempt=0, mode="gpu"):
        hc = core.Client(self.port, token=self.server_token)
        self.hclient = hc
        srv = self.server
        t_start = time.perf_counter()
        fails = 0
        established = False             # the server reached standby/activating/ready at least once
        activated = False               # /activate already posted in this generation
        standby_sent = False
        while not self.closing and gen == self.gen:
            try:
                h = hc.health(timeout=3.0)
                fails = 0
            except Exception:
                h = None
                fails += 1
            if gen != self.gen:
                return
            if h:
                self.health = h
                st = h.get("status")
                if st in ("standby", "activating", "ready"):
                    if not established and self.srv_rec is not None and gen == self.gen:
                        try:                    # keep-server: the next widget run can reconnect to this one
                            self.srv_rec.save(self.port, self.server_token, h)
                        except Exception as e:
                            log(f"keep-server: could not save the server record: {type(e).__name__}")
                    established = True
                if self.want_gpu and not activated and st in ("loading", "standby"):
                    try:
                        hc.activate(timeout=3.0)
                        activated = True
                    except Exception as e:
                        log(f"activate request failed: {e}")
                if st == "ready" and not self.ready_evt.is_set():
                    self.ready_evt.set()
                    self.uiq.put(("ready", gen, time.perf_counter() - (self.load_t0 or t_start)))
                elif st == "standby" and not self.want_gpu and not standby_sent:
                    standby_sent = True
                    self.uiq.put(("standby", gen, time.perf_counter() - t_start, h.get("rss_mib")))
                elif st == "error":
                    self.uiq.put(("server_error", gen, f"Engine error: {h.get('error')}"))
                    return
            else:
                if srv is not None and not srv.alive():
                    tail = ""
                    try:
                        with open(srv.log_path, encoding="utf-8", errors="replace") as f:
                            tail = f.read()[-2000:]
                    except Exception:
                        pass
                    if "Address already in use" in tail and attempt < 3 and not established:
                        log("port taken inside WSL, retrying on the next port")
                        threading.Thread(target=self._engine_thread, args=(gen, attempt + 1, mode), daemon=True).start()
                        return
                    self.ready_evt.clear()
                    if not self.want_gpu:       # a RAM-only server went away: fall back to unloaded quietly
                        self.uiq.put(("unloaded", gen, "standby server exited", False))
                    else:
                        self.uiq.put(("server_error", gen, "Speech engine stopped - tray > Load model to retry"))
                    return
                if established and fails >= 3:
                    self.ready_evt.clear()
                    self.uiq.put(("server_error", gen, "Speech engine not answering - tray > Load model to retry"))
                    return
                if not established and time.perf_counter() - t_start > 240:
                    self.uiq.put(("server_error", gen, "Speech engine did not start - see logs"))
                    return
            busy = not established or (self.want_gpu and not self.ready_evt.is_set())
            self.wake_evt.wait(0.25 if busy else 10.0)
            self.wake_evt.clear()

    def unload_model(self, reason="manual", then_standby=False):
        if self.engine in ("unloaded", "unloading"):
            return
        if self.recording:
            self.stop_recording()
        self.gen += 1
        gen = self.gen
        self.engine = "unloading"
        self.want_gpu = False
        self.ready_evt.clear()
        self.wake_evt.set()
        self._update_status()
        srv, port = self.server, self.port
        log(f"{'moving model to RAM' if then_standby else 'unloading model'} ({reason})")

        def work():
            with self.engine_lock:
                try:
                    if self.srv_rec is not None:
                        self.srv_rec.clear()
                    if srv is not None:
                        srv.stop(core.Client(port, token=self.server_token))
                    else:
                        core.kill_stale_servers()
                except Exception as e:
                    log(f"unload error: {e}")
                self.tclient.close()
            self.uiq.put(("unloaded", gen, reason, then_standby))
        threading.Thread(target=work, name="unload", daemon=True).start()

    def _idle_check(self):
        if self.closing:
            return
        m = float(self.cfg["idle_unload_min"] or 0)
        if (m > 0 and self.engine == "ready" and not self.recording and self.pending == 0
                and time.monotonic() - self.last_activity > m * 60):
            if self.cfg["ram_standby"]:
                self.to_standby(f"idle {m:g} min")
            else:
                self.unload_model(f"idle {m:g} min")
        hrs = float(self.cfg["standby_unload_h"] or 0)
        if (hrs > 0 and not self.cfg["gpu_always"] and self.engine == "standby" and self.standby_since is not None
                and time.monotonic() - self.standby_since > hrs * 3600):
            self.unload_model(f"in RAM standby for {hrs:g} h")
        self.root.after(5000 if (m < 1 or (0 < hrs < 0.1)) else 20000, self._idle_check)

    # ------------------------------------------------------------------ recording
    def toggle(self):                  # record button, tray > Record, --cmd record
        if self.recording:
            self.stop_recording()
        else:
            self._start_recording_cor()

    def start_recording(self, source=None):
        if self.recording or self.closing or self._quitting:
            return
        self.last_activity = time.monotonic()
        if self.engine in ("unloaded", "error", "unloading", "standby", "prewarming"):
            if self.parked and not self.parked.get("temp"):
                if self._held() or not self._gpu_room_ok():   # no CPU fallback: say so clearly and don't record
                    free = (self.gpu_last or {}).get("free_mb")
                    msg = ("GPU busy - batch transcription running. Try again when it finishes." if self._held() else
                           f"GPU busy - {self.parked.get('proc') or 'another app'} is using it "
                           f"({free} MB free). Try again when it closes.")
                    log(f"record while parked refused: {msg}")
                    if not self.visible:
                        self.show(load=False)
                    self.set_state("error", msg)
                    self.root.after(6000, self._recover_status)
                    return
                self.parked["temp"] = True
                self.park_pending = False
                log(f"record while parked for {self.parked.get('proc')}: temporary GPU load "
                    f"({(self.gpu_last or {}).get('free_mb')} MB free)")
            self.load_model()           # first Record loads the model; audio is queued meanwhile
        self.rec_session += 1
        sess = self.rec_session
        self.seg = core.Segmenter(lambda *a: self._on_utterance(*a, session=sess),
                                  silence_s=float(self.cfg["silence_s"]),
                                  max_s=float(self.cfg["max_utterance_s"]))
        if source is None:
            mic = None
            for attempt in (0, 1):
                mic = core.MicCapture(self.cfg["device"], self._on_audio, lambda m: self.uiq.put(("mic_error", m)))
                try:
                    mic.start()
                    break
                except Exception as e:
                    log(f"mic start failed (attempt {attempt + 1}): {e}")
                    mic = None
                    if attempt == 0:
                        core.refresh_portaudio()        # device list may be stale (USB re-plugged)
                    else:
                        self.set_state("error", f"Mic error: {str(e)[:90]}")
                        self.root.after(4000, self._recover_status)
                        return
            self.mic = mic
            log(f"recording from {self.mic.info}")
        else:
            self.mic = source
        self.recording = True
        self._refresh_disc()
        self._update_status()
        self.dump_state()

    def _recover_status(self):
        if self.state == "error" and self.engine != "error":
            self._update_status()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.last_activity = time.monotonic()
        mic, self.mic = self.mic, None
        if mic is not None:
            mic.stop()                 # joins the mic worker, so flushing below is race-free
        if self.seg is not None:
            self.seg.flush()
        self.level_db = -100.0
        self._refresh_disc()
        self._update_status()
        self.dump_state()
        self._try_park()

    def _silent_record(self):
        """Test hook (IPC "silentrec"): toggle a recording fed by nothing - no microphone is opened."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording(source=type("Silent", (), {"info": "silent test source", "stop": lambda s: None})())

    def _on_audio(self, y):            # mic worker thread
        seg = self.seg
        if seg is not None and self.recording:
            seg.feed(y)
            self.level_db = seg.level_db

    def _on_utterance(self, pcm, t_end, dur, kind, uid, ver, session=None):   # mic worker thread
        if kind == "final":
            if session is not None:
                self._uid_session[uid] = session
            self.latest.pop(uid, None)  # a still-queued spec of this utterance is skipped; the final runs anyway
            self.uiq.put(("pending", 1))
        else:
            self.latest[uid] = ver
        self.jobq.put((kind, uid, ver, pcm, t_end, dur))

    def _job_loop(self):               # transcription thread (persistent keep-alive connection)
        spec = core.SpecCache()        # the last speculative run; only its own final may reuse it (log 55)
        while True:
            item = self.jobq.get()
            if item is None:
                return
            kind, uid, ver, pcm, t_end, dur = item
            if kind == "spec" and self.latest.get(uid) != ver:
                continue               # speech resumed: this speculative job is already stale
            while not self.ready_evt.wait(0.5):
                if self.closing:
                    return
                if self.engine in ("unloaded", "error"):
                    break
            if not self.ready_evt.is_set():
                if kind == "final":
                    self.uiq.put(("job_error", "model not loaded", (pcm, dur, uid)))
                continue
            hit = spec.take_for_final(uid, ver) if kind == "final" else None
            if hit is not None:
                r, rtt = hit                       # identical audio already transcribed during the pause
                self.spec_hits += 1
                self.uiq.put(("text", r.get("text", ""), t_end, r.get("infer_ms", 0), rtt, dur, pcm, uid))
                continue
            t0 = time.perf_counter()
            try:
                r = self._transcribe(pcm, kind)
                rtt = (time.perf_counter() - t0) * 1000
                if kind == "spec":
                    spec.put(uid, ver, r, rtt)
                else:
                    self.spec_misses += 1
                    self.uiq.put(("text", r.get("text", ""), t_end, r.get("infer_ms", 0), rtt, dur, pcm, uid))
            except Exception as e:
                log(f"transcribe failed: {e!r}")
                if kind == "final":
                    self.uiq.put(("job_error", str(e), (pcm, dur, uid) if core.retryable(e) else None))

    def _transcribe(self, pcm, kind):  # transcription thread
        try:
            return self.tclient.transcribe(pcm)
        except Exception as e:
            if kind != "final" or not core.retryable(e) or self.closing or not self.ready_evt.is_set():
                raise
            log(f"transcribe failed ({e!r}), one more try")
            time.sleep(0.3)
            return self.tclient.transcribe(pcm)

    def _keep_failed(self, item):
        """UI thread: remember the audio of a failed final (on disk too, log 57); returns how many are waiting."""
        pcm, dur, uid = item
        tries = self._retry_tries.pop(uid, 0) + 1
        kid, failed_at = self._retry_kid.pop(uid, (None, None))
        if tries > 3 or not pcm or self.closing:
            log(f"utterance dropped after {tries - 1} retries ({dur:.1f}s audio)" if tries > 3 else
                "utterance dropped (closing)")
            if kid and self.store and tries > 3:
                self.store.drop_pending(kid)
            return len(self.failed)
        if failed_at is None:
            failed_at = time.time()
            kid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uid}"
        age = max(0.0, time.time() - failed_at)
        self.failed.append({"t": time.monotonic() - age, "pcm": pcm, "dur": dur, "tries": tries, "kid": kid,
                            "failed_at": failed_at})
        for old in self.failed[:-20]:
            self._drop_kept(old)
        del self.failed[:-20]
        if self.store:
            try:
                self.store.save_pending(kid, pcm, {"dur": dur, "tries": tries, "failed_at": failed_at})
            except Exception as e:
                log(f"kept utterance not saved to disk: {e.__class__.__name__}")
        log(f"utterance kept for a retry ({dur:.1f}s audio, try {tries}, {len(self.failed)} waiting)")
        return len(self.failed)

    def _drop_kept(self, it):
        if self.store and it.get("kid"):
            self.store.drop_pending(it["kid"])

    def _requeue_failed(self):
        """UI thread: transcribe the kept utterances now that the engine is ready."""
        if self.closing or self.engine != "ready" or not self.failed:
            return
        now = time.monotonic()
        items = [it for it in self.failed if now - it["t"] < RETRY_MAX_AGE_S]
        if len(items) < len(self.failed):
            log(f"{len(self.failed) - len(items)} kept utterance(s) older than {RETRY_MAX_AGE_S // 60} min dropped")
            for it in self.failed:
                if now - it["t"] >= RETRY_MAX_AGE_S:
                    self._drop_kept(it)
        self.failed = []
        if len(self._retry_tries) > 200:
            self._retry_tries.clear()
        for it in items:
            uid = core.new_utterance_id()
            self._retry_tries[uid] = it["tries"]
            if it.get("kid"):
                self._retry_kid[uid] = (it["kid"], it.get("failed_at") or time.time())
            self.pending += 1
            self.jobq.put(("final", uid, 0, it["pcm"], time.perf_counter(), it["dur"]))
        if items:
            log(f"re-queued {len(items)} kept utterance(s)")
            self._update_status()

    def _maybe_auto_recover(self):
        """GPU policy: an engine that stopped on its own comes back by itself (max 2 tries per 10 min)."""
        if self.closing or self._quitting or not self.cfg["gpu_always"] or self.parked or self._held():
            return
        now = time.monotonic()
        self._recover_times = [t for t in self._recover_times if now - t < 600]
        if len(self._recover_times) >= 2:
            log("engine stopped again - not restarting it automatically (tray > Load model)")
            return
        self._recover_times.append(now)
        gen = self.gen

        def go():
            if gen == self.gen and self.engine == "error" and not self.closing and not self._quitting \
                    and not self.parked and not self._held():
                log("restarting the speech engine after an unexpected stop (GPU policy)")
                self.load_model()
        self.root.after(3000, go)

    # ------------------------------------------------------------------ UI pump
    def _pump(self):
        try:
            while True:
                self._handle(self.uiq.get_nowait())
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(10, self._pump)

    def _handle(self, ev):
        kind = ev[0]
        if kind == "text":
            _, txt, t_end, infer_ms, rtt, dur, pcm = ev[:7]
            session = self._uid_session.pop(ev[7], None) if len(ev) > 7 else None
            kept = self._retry_kid.pop(ev[7], None) if len(ev) > 7 else None
            if kept and self.store:
                self.store.drop_pending(kept[0])     # a kept utterance came through: its files go
            self.pending = max(0, self.pending - 1)
            self.last_activity = time.monotonic()
            rid = None
            if self.cfg["save_training"] and txt.strip() and pcm:
                dev = getattr(self.mic, "info", "") if self.mic else ""
                rid = self.training.add(pcm, txt, {"device": dev})
            if (session is not None and self.cor_restore is not None and self._cor_clear_session is not None
                    and session < self._cor_clear_session):
                shown = self._cor_add_late(txt, rid)   # the end of the message that was already cleared
            else:
                shown = self.append_text(txt, rid, session)
            lat = (time.perf_counter() - t_end) * 1000
            if shown:
                self.latencies.append(lat)
                self.infer_ms.append(infer_ms)
                self.rtt_ms.append(rtt)
                if len(self.latencies) > 2000:  # bounded on long sessions (log 55)
                    del self.latencies[:1000], self.infer_ms[:1000], self.rtt_ms[:1000]
            # privacy (log 55): never write dictated text to the log, only its length
            log(f"utt {dur:.2f}s infer={infer_ms}ms rtt={rtt:.0f}ms end-of-speech->text={lat:.0f}ms "
                f"{len(txt.strip())} chars")
            self._update_status()
            self._try_park()
        elif kind == "pending":
            self.pending += ev[1]
            self._update_status()
        elif kind == "ready":
            if ev[1] == self.gen:
                self.engine = "ready"
                self.standby_since = None
                self.last_activity = time.monotonic()
                log(f"engine ready {ev[2]:.1f}s after the request on port {self.port}: {self.health}")
                self.goal_gpu = False
                if self._unpark_t0 is not None:
                    dt = time.perf_counter() - self._unpark_t0
                    self._unpark_t0 = None
                    self._park_event("unparked", gpu_s=round(dt, 2), **self._park_info)
                self._update_status()
                self.dump_state()
                self._requeue_failed()
                self._try_park()
        elif kind == "unloaded":
            if ev[1] == self.gen:
                self.engine = "unloaded"
                self.health = {}
                self.standby_since = None
                log(f"model unloaded ({ev[2]})")
                if self._park_t0 is not None:
                    self._park_info = {"vram_freed_s": round(time.perf_counter() - self._park_t0, 2)}
                self._update_status()
                self.dump_state()
                if len(ev) > 3 and (ev[3] == "force" or (ev[3] and self.cfg["ram_standby"])) and not self.closing:
                    self.prewarm(f"RAM standby after {ev[2]}")
        elif kind == "standby":
            if ev[1] == self.gen and self.engine == "prewarming":
                self.engine = "standby"
                self.standby_since = time.monotonic()
                log(f"model in RAM (standby, 0 VRAM) after {ev[2]:.1f}s; server rss {ev[3]} MiB; {self.health.get('timings')}")
                if self._park_t0 is not None:
                    self._park_info["in_ram_s"] = round(time.perf_counter() - self._park_t0, 2)
                    self._park_t0 = None
                    self._park_event("parked", **self._park_info)
                if self.goal_gpu:
                    self.goal_gpu = False
                    if not self.cfg["gpu_always"]:
                        pass
                    elif self.parked:
                        log(f"login: GPU busy (parked for {self.parked.get('proc')}) - staying in RAM standby until it exits")
                    else:
                        log("login: model in RAM, moving it to the GPU (GPU policy)")
                        self.load_model()
                self._update_status()
                self.dump_state()
        elif kind == "server_error":
            if ev[1] == self.gen:
                self.engine = "error"
                self.set_state("error", ev[2])
                log(ev[2])
                self.dump_state()
                self._maybe_auto_recover()
        elif kind == "job_error":
            self.pending = max(0, self.pending - 1)
            if len(ev) > 2 and ev[2] is not None:
                self._uid_session.pop(ev[2][2], None)
            n = self._keep_failed(ev[2]) if len(ev) > 2 and ev[2] is not None else 0
            self.set_state("error", f"Transcription failed: {ev[1][:60]}" +
                           (f"  \u00b7  kept, retrying when the engine is back ({n})" if n else ""))
            self.root.after(4000, self._recover_status)
            if n and self.engine == "ready":
                self.root.after(3000, self._requeue_failed)     # engine looks fine: a hiccup
            self._try_park()
        elif kind == "gpu":
            self._on_gpu_sample(ev[1], ev[2])
        elif kind == "mic_error":
            self.stop_recording()
            self.set_state("error", ev[1])
            self.root.after(5000, self._recover_status)
        elif kind == "tray":
            self._tray_event(ev[1])
        elif kind == "copied":
            if ev[1]:
                self.copy_count += 1
                if len(ev) > 2:
                    self._cor_arm(ev[2])
                if self.visible:
                    self._flash_copied()
            else:                           # log 55: say so, and let the next copy of this text through
                self.copy_failures += 1
                log("clipboard write failed (clipboard busy)")
                if len(ev) > 2 and self._last_copied == ev[2]:
                    self._last_copied = None
                if self.visible:
                    self._flash_copied(ok=False)
        elif kind == "call":
            ev[1]()

    def _tk_error(self, exc, val, tb):
        log("UI error: " + "".join(traceback.format_exception(exc, val, tb)))
        try:
            self.set_state("error", f"Error: {val}"[:90])
        except Exception:
            pass

    # ------------------------------------------------------------------ GPU policy: watcher + auto-park
    # The watcher thread only samples (PDH + Toolhelp32, no subprocesses) and runs the pure ParkPolicy;
    # every state change happens here on the UI thread. Park = the same path as "Move model to RAM"
    # (GPU server exits -> VRAM freed at once -> RAM-only server reloads from the page cache).
    # Unpark = /activate from RAM standby (~2-3 s). Never parks mid-recording or mid-transcription.
    def _unpark_needed(self):
        c = self.cfg
        return max(int(c["unpark_free_mb"]), int(c["model_vram_mb"]) + int(c["park_free_min_mb"]) + 200)

    def _watch_enabled(self):
        return bool(self.cfg["gpu_always"] and self.cfg["auto_park"] and not self.park_paused)

    def _gpu_watch_loop(self):
        try:
            import plive_gpu as gpu
            ad = gpu.find_adapter()
            if not ad:
                log("auto-park: no NVIDIA adapter found - watcher off")
                return
            sampler = gpu.GpuSampler(ad)
        except Exception as e:
            log(f"auto-park: watcher could not start: {e!r}")
            self.watch_stats["last_error"] = repr(e)
            return
        self.watch_stats["adapter"] = ad
        log(f"auto-park watcher on {ad['name']} ({ad['luid']}, {ad['dedicated_mb']} MB)")
        policy = gpu.ParkPolicy(self.cfg, own_pids=[os.getpid()])
        was_on = False
        while not self.closing:
            poll = max(1.0, float(self.cfg["park_poll_s"]))
            if not self._watch_enabled():
                if was_on:
                    policy.reset()
                    was_on = False
                self.watch_evt.wait(poll)
                self.watch_evt.clear()
                continue
            was_on = True
            c0 = time.thread_time()
            try:
                smp = sampler.sample()
                on_gpu = self.engine in ("ready", "loading") and self.want_gpu
                d = policy.update(smp, self.parked is not None, on_gpu)
                pr = smp["procs"]
                others = sorted(((pr.get(q, ("pid %d" % q,))[0], round(mb)) for q, mb in smp["mem_mb"].items() if mb >= 50),
                                key=lambda kv: -kv[1])[:6]
                summ = {"free_mb": None if smp["free_mb"] is None else round(smp["free_mb"]),
                        "used_mb": None if smp["used_mb"] is None else round(smp["used_mb"]),
                        "top_vram": others, "trigger": policy.last.get("trigger"), "t": time.strftime("%H:%M:%S")}
                self.uiq.put(("gpu", d, summ))
            except Exception as e:
                self.watch_stats["errors"] += 1
                if self.watch_stats["errors"] <= 3:
                    log(f"auto-park: sample failed: {e!r}")
                self.watch_stats["last_error"] = repr(e)
            self.watch_stats["polls"] += 1
            self.watch_stats["cpu_s"] += time.thread_time() - c0
            self.watch_evt.wait(poll)
            self.watch_evt.clear()
        try:
            sampler.close()
        except Exception:
            pass

    def _gpu_room_ok(self):
        free = (self.gpu_last or {}).get("free_mb")
        return free is None or free >= int(self.cfg["model_vram_mb"]) + 200

    def _busy(self):
        return self.recording or self.pending > 0 or not self.jobq.empty() or self.engine in ("loading", "unloading")

    def _park_event(self, what, **kw):
        e = {"t": time.strftime("%H:%M:%S"), "event": what, **kw}
        if self.parked:
            e.update(reason=self.parked.get("reason"), proc=self.parked.get("proc"))
        self.park_log = (self.park_log + [e])[-20:]
        log(f"auto-park: {what} {kw}")

    def _on_gpu_sample(self, d, summ):
        self.gpu_last = summ
        if not self._watch_enabled():
            return
        pk = self.parked
        if pk and pk.get("after_batch"):          # batch hold just ended: decide on this fresh sample
            trig, free = summ.get("trigger"), summ.get("free_mb")
            if trig:
                self.parked = {"reason": trig[0], "proc": trig[1], "since": time.strftime("%H:%M:%S"), "temp": False}
                log(f"batch finished; staying parked for {trig[1]} ({trig[0]})")
                self._update_status()
                self.dump_state()
            elif free is None or free >= self._unpark_needed():
                self._unpark("batch finished")
            elif not pk.get("waiting_logged"):
                pk["waiting_logged"] = True
                log(f"batch finished but only {free} MB VRAM free - waiting for {self._unpark_needed()} MB")
            return
        if d and d[0] == "park":
            self._request_park(d[1], d[2])
        elif d and d[0] == "unpark":
            self._unpark("trigger gone")
        pk = self.parked
        if (pk and pk.get("temp") and summ.get("trigger") and self.engine == "ready" and not self._busy()
                and time.monotonic() - self.last_activity > float(self.cfg["park_temp_hold_s"])):
            pk["temp"] = False
            log(f"auto-park: temporary GPU use over, re-parking for {pk.get('proc')}")
            self.park_pending = True
        self._try_park()

    def _request_park(self, reason, proc):
        if self.parked:
            return
        self.parked = {"reason": reason, "proc": proc, "since": time.strftime("%H:%M:%S"), "temp": False}
        on_gpu = self.engine in ("ready", "loading")
        log(f"auto-park: {reason} ({proc}); engine={self.engine}" + ("" if on_gpu else " - nothing on the GPU to move"))
        self.park_pending = on_gpu
        if on_gpu and self._busy():
            log("auto-park: deferred until the current utterance/load finishes")
        self._try_park()
        self._update_status()
        self.dump_state()

    def _try_park(self):
        if not self.park_pending or not self.parked or self.parked.get("temp") or self._busy():
            return
        if self.engine != "ready":
            self.park_pending = self.engine in ("loading",)
            return
        self.park_pending = False
        self._park_t0 = time.perf_counter()
        self._park_info = {}
        self.unload_model(f"auto-park for {self.parked.get('proc')}", then_standby="force")

    def _unpark(self, why):
        if self._held():                # a batch job owns the GPU; its release decides
            return
        pk, self.parked = self.parked, None
        self.park_pending = False
        if not pk:
            return
        log(f"auto-park: unparking ({why}; was parked for {pk.get('proc')} since {pk.get('since')}); engine={self.engine}")
        if self.engine in ("standby", "prewarming", "unloaded", "error") and self.cfg["gpu_always"]:
            self._unpark_t0 = time.perf_counter()
            self._park_info = {"from": self.engine, "why": why}
            self.load_model()
        self._update_status()
        self.dump_state()

    # ------------------------------------------------------------------ batch hold (log 52)
    # batch\Transcribe-WithGpuHold.ps1 sends "batchhold <its pid>" before a batch job and "batchrelease <pid>" in
    # its finally block. The hold parks the model like a game would (reason "batch transcription"), blocks
    # unparking/loading/recording, and works whether or not the GPU policy or auto-park is on. Lease: we keep
    # a handle to the launcher process (so its pid cannot be reused) and poll it every 2 s; if it dies without
    # releasing, the hold moves to its live child processes (the wsl.exe running transcribe.py), so the model
    # never returns while the batch still uses the GPU, and it ends when they end. Cap: batch_hold_max_h.
    def _batch_hold(self, pid, via="launcher"):
        if pid <= 0 or self.closing:
            return
        if pid in self.holds:
            self.dump_state()
            return
        h = _k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            log(f"batch hold: pid {pid} not found (err {ctypes.get_last_error()}) - ignored")
            return
        first = not self.holds
        self.holds[pid] = {"h": h, "since": time.strftime("%H:%M:%S"), "t0": time.monotonic(), "via": via}
        log(f"batch hold from pid {pid} ({via}); engine={self.engine}")
        if first:
            on_gpu = self.engine in ("ready", "loading")
            self._hold_prev_gpu = on_gpu or bool(self.cfg["gpu_always"])
            self.parked = {"reason": "batch transcription", "proc": "batch transcription",
                           "since": time.strftime("%H:%M:%S"), "temp": False, "hold": True}
            self.park_pending = on_gpu
            if on_gpu and self._busy():
                log("batch hold: park deferred until the current utterance/load finishes")
            self._try_park()
            if not self._hold_loop:
                self._hold_loop = True
                self.root.after(2000, self._hold_check)
        self._update_status()
        self.dump_state()

    def _batch_release(self, pid, why):
        hd = self.holds.pop(pid, None)
        if not hd:
            return
        _k32.CloseHandle(hd["h"])
        log(f"batch hold released: pid {pid} ({why}) after {time.monotonic() - hd['t0']:.0f} s")
        if self.holds:
            self.dump_state()
            return
        self.park_pending = False
        self._wsl_batch_check()

    def _held(self):
        return bool(self.holds or self.wsl_hold)

    def _wsl_batch_check(self):
        """A hold ended. If transcribe.py is still running inside WSL (its Windows launcher and wsl.exe were
        killed, which does not stop the Linux process), keep the model parked until it exits, checking
        every 5 s in a background thread; then resume. A normal release costs one quick wsl call."""
        if self.wsl_hold:
            return                      # a waiter is already running
        self.wsl_hold = True
        self.dump_state()

        def wait():
            first = True
            while not self.closing:
                try:
                    r = core._wsl(["pgrep", "-f", "/opt/parakeet/scripts/transcribe.py"], timeout=15)
                    pids = r.stdout.decode(errors="replace").split() if r.returncode == 0 else []
                except Exception as e:
                    log(f"batch hold: WSL check failed ({e!r}) - resuming")
                    pids = []
                if not pids:
                    break
                if first:
                    first = False
                    self.uiq.put(("call", lambda p=pids: log(
                        f"batch hold: transcribe.py still running inside WSL (pid {' '.join(p)}) - waiting for it")))
                time.sleep(5)
            self.uiq.put(("call", self._resume_after_batch))
        threading.Thread(target=wait, name="wsl-batch-wait", daemon=True).start()

    def _resume_after_batch(self):
        self.wsl_hold = False
        if self.holds or self.closing:  # a new batch took a hold meanwhile: it decides at its release
            return
        if not self._hold_prev_gpu:
            self.parked = None
        else:
            self.parked = {"reason": "batch finished", "proc": "batch transcription",
                           "since": time.strftime("%H:%M:%S"), "temp": False, "after_batch": True}
            if self._watch_enabled():
                self.watch_evt.set()    # decide on a fresh sample: game running? enough free VRAM?
            else:
                self.parked = None
                if self.engine in ("standby", "prewarming", "unloaded", "error"):
                    self._unpark_t0 = time.perf_counter()
                    self._park_info = {"from": self.engine, "why": "batch finished"}
                    self.load_model()
        self._update_status()
        self.dump_state()

    def _children(self, pid):
        try:
            import plive_gpu as gpu
            procs = gpu.processes()
        except Exception as e:
            log(f"batch hold: process list failed: {e!r}")
            return []
        return [(p, exe) for p, (exe, pp) in procs.items()
                if pp == pid and p != os.getpid() and exe.lower() not in ("conhost.exe", "openconsole.exe")]

    def _hold_check(self):
        if self.closing or not self.holds:
            self._hold_loop = False
            return
        cap = float(self.cfg["batch_hold_max_h"] or 0) * 3600
        for pid, hd in list(self.holds.items()):
            if _k32.WaitForSingleObject(hd["h"], 0) == 0:          # WAIT_OBJECT_0: the process has exited
                kids = self._children(pid)
                for k, exe in kids:
                    self._batch_hold(k, via=f"{exe}, child of exited pid {pid}")
                self._batch_release(pid, "process exited without releasing" +
                                    (f"; still running: {', '.join(e for _, e in kids)}" if kids else ""))
            elif cap > 0 and time.monotonic() - hd["t0"] > cap:
                self._batch_release(pid, f"hold expired after {cap / 3600:g} h")
        if self.holds:
            self.root.after(2000, self._hold_check)
        else:
            self._hold_loop = False

    # ------------------------------------------------------------------ quit
    def quit(self, stop_server=False):
        """keep-server (2026-10-09): a plain quit leaves a loaded model server running for the next widget run
        (it exits by itself after 90 s without a widget); stop_server (tray 'Quit (stop model too)',
        --cmd quitall) stops it as before."""
        if self.closing or self._quitting:
            return
        self._quit_stop_server = bool(stop_server)
        log("quitting" + (" (stopping the model server too)" if stop_server else ""))
        self._quitting = True
        try:
            self.stop_recording()      # flushes the last utterance into the queue
        except Exception:
            pass
        self._remember_geometry()
        self.root.withdraw()
        self.visible = False
        t0 = time.monotonic()
        st = {"copies": None}

        def drain():                   # log 55: let the last words reach the box, training data and clipboard
            waited = time.monotonic() - t0
            busy = self.pending > 0 or not self.jobq.empty() or not self.uiq.empty()
            if busy and self.engine == "ready" and waited < 6:
                return self.root.after(50, drain)
            if st["copies"] is None:
                if self._deb:          # an edit's auto-copy was still waiting for its debounce
                    self.root.after_cancel(self._deb)
                    self._debounced_copy()
                st["copies"] = self.copy_count + self.copy_failures
            with self.clip_lock:
                copying = self.clip_text is not None
            if (copying or (self._last_copied is not None and self._last_copied != self._copied_text
                            and self.copy_count + self.copy_failures == st["copies"])) and waited < 7.5:
                return self.root.after(50, drain)
            if self.pending or self.failed:
                log(f"quit: {self.pending} transcription(s) unfinished, {len(self.failed)} kept utterance(s) "
                    + ("saved for the next start" if self.store else "dropped"))
            self._quit_now()
        drain()

    def _quit_now(self):
        try:
            self.sync_training_spans()
        except Exception:
            pass
        if self._draft_job is not None:
            self.root.after_cancel(self._draft_job)
        self._save_draft()                     # the unsent message, right now (log 57)
        self.closing = True
        self.ready_evt.set()
        self.jobq.put(None)
        self.wake_evt.set()
        self.clip_evt.set()
        self.watch_evt.set()
        for hd in self.holds.values():
            _k32.CloseHandle(hd["h"])
        self.holds.clear()
        self.stop_result = None

        def fin():
            try:
                self.training.flush(5)
                srv, port = self.server, self.port
                keep = (not self._quit_stop_server and srv is not None and self.srv_rec is not None
                        and self.engine in ("ready", "standby") and not self.holds
                        and self.srv_rec.load() is not None)
                if keep:
                    self.stop_result = "kept"
                    log("quit: model server left running for the next widget run (it exits by itself after "
                        "90 s without one; tray 'Quit (stop model too)' stops it right away)")
                    if srv.logf:
                        try:
                            srv.logf.close()
                        except Exception:
                            pass
                    return
                if self.srv_rec is not None:
                    self.srv_rec.clear()
                with self.engine_lock:
                    # never loaded this session: don't wake WSL just to quit
                    self.stop_result = (srv.stop(core.Client(port, token=self.server_token))
                                        if srv is not None else True)
            except Exception as e:
                log(f"quit: server stop error: {e}")
                self.stop_result = False
            log(f"server stopped: {self.stop_result}")
        th = threading.Thread(target=fin, daemon=True)
        th.start()
        t0 = time.perf_counter()

        def wait_done():
            if th.is_alive() and time.perf_counter() - t0 < 20:
                self.root.after(100, wait_done)
            else:
                self.tray.stop()
                self.tray.join(2)
                self.root.destroy()
        wait_done()

    def run(self):
        self.root.mainloop()
        log(f"exited; stop_result={getattr(self, 'stop_result', None)}")

    # ------------------------------------------------------------------ self test
    def _call(self, fn, wait=True):
        box, done = {}, threading.Event()

        def run():
            try:
                box["v"] = fn()
            except Exception as e:
                box["e"] = e
            done.set()
        self.uiq.put(("call", run))
        if wait:
            done.wait(30)
            if "e" in box:
                raise box["e"]
        return box.get("v")

    def _shot(self, path):
        from PIL import ImageGrab
        time.sleep(0.6)
        x, y, w, h = self._call(lambda: (self.root.winfo_rootx(), self.root.winfo_rooty(),
                                         self.root.winfo_width(), self.root.winfo_height()))
        ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(path)
        return path

    @staticmethod
    def _wait(cond, timeout):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if cond():
                return True
            time.sleep(0.05)
        return False

    def _replay(self, x):
        src = type("Replay", (), {"info": "replay", "stop": lambda s: None})()
        self._call(lambda: self.start_recording(source=src))
        tstart = time.perf_counter()
        for i in range(0, len(x), 320):
            d = tstart + i / 16000 - time.perf_counter()
            if d > 0:
                time.sleep(d)
            self._on_audio(x[i:i + 320])
        self._call(self.stop_recording)
        self._wait(lambda: not self.pending and self.jobq.empty(), 30)
        time.sleep(0.2)

    def _selftest(self):
        import wave
        import numpy as np
        res = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "dpi_scale": self.s}
        try:
            res["tray_added"], res["hotkey_ok"] = self.tray.added, self.tray.hotkey_ok
            res["tray_rect"] = self.tray.icon_rect()
            t0 = time.perf_counter()
            while not self.ready_evt.wait(0.5):
                if self.engine == "error" or time.perf_counter() - t0 > 240:
                    raise RuntimeError(f"engine not ready: {self.status.cget('text')}")
            time.sleep(0.3)
            res["engine_ready_s"] = round(time.perf_counter() - t0, 1)
            res["health_ready"] = dict(self.health)
            res["window_style_visible"] = self._call(self.exstyle)
            self._call(lambda: self._fill_mic_menu(refresh=True))
            res["devices"] = [d["name"] for d in self.devices]
            res["default_device"] = self.default_dev

            # 1) real mic: a few seconds from the default input (skipped with --no-mic)
            self._call(lambda: self.start_recording(
                source=None if not self.args.no_mic else type("Silent", (), {"info": "no-mic", "stop": lambda s: None})()))
            time.sleep(0.3)
            res["mic_recording"] = self.recording
            res["mic_info"] = getattr(self.mic, "info", None)
            peak = -200.0
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < self.args.record_seconds:
                peak = max(peak, self.level_db)
                time.sleep(0.05)
            res["mic_peak_dbfs"] = round(float(peak), 1)
            res["mic_noise_floor_dbfs"] = round(float(self.seg.noise), 1) if self.seg else None
            self._shot(os.path.join(LOG_DIR, "screenshot-recording.png"))
            self._call(self.stop_recording)
            self._wait(lambda: not self.pending, 10)
            res["mic_text"] = self._call(lambda: self.text.get("1.0", "end-1c"))
            self._call(self.clear_text)
            self.latencies.clear(); self.infer_ms.clear(); self.rtt_ms.clear()
            self.spec_hits = self.spec_misses = 0

            # 2) replay clips through the exact live path (VAD -> keep-alive HTTP -> UI -> clipboard)
            clips = []
            for f in self.args.replay:
                with wave.open(f, "rb") as w:
                    assert w.getframerate() == 16000 and w.getnchannels() == 1
                    clips.append(np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768)
            gaps_start = len(self.ui_gaps)
            for rep in range(self.args.replay_rounds):
                for x in clips:
                    rng = np.random.default_rng(rep)
                    noise = lambda n: (rng.standard_normal(n) * 10 ** (-68 / 20)).astype(np.float32)  # noqa
                    self._replay(np.concatenate([noise(8000), x + noise(len(x)), noise(int(16000 * 1.2))]))
            gaps = self.ui_gaps[gaps_start:]
            res["ui_max_gap_ms_during_replay"] = round(max(gaps), 1) if gaps else None
            res["replay_text"] = self._call(lambda: self.text.get("1.0", "end-1c"))
            L = sorted(self.latencies)
            if L:
                sil = float(self.cfg["silence_s"]) * 1000
                res["latency_end_of_speech_to_text_ms"] = {
                    "n": len(L), "median": round(statistics.median(L)), "worst": round(L[-1]),
                    "best": round(L[0]), "silence_cut_ms": int(sil)}
                res["latency_beyond_silence_cut_ms"] = {"median": round(statistics.median(L) - sil),
                                                        "worst": round(L[-1] - sil)}
                res["server_infer_ms"] = {"median": round(statistics.median(self.infer_ms)), "worst": max(self.infer_ms)}
                res["http_roundtrip_ms"] = {"median": round(statistics.median(self.rtt_ms)), "worst": round(max(self.rtt_ms))}
                res["speculative_hits"], res["speculative_misses"] = self.spec_hits, self.spec_misses
            time.sleep(0.6)
            res["clipboard_matches_box_after_replay"] = core.get_clipboard() == res["replay_text"]
            self._shot(os.path.join(LOG_DIR, "screenshot.png"))

            # 3) auto-copy: simulated append, then simulated user edit with debounce
            self._call(self.clear_text)
            base = "Hello from the Parakeet Live self test."
            t_app = time.perf_counter()
            self._call(lambda: self._handle(("text", base, time.perf_counter(), 0, 0, 1.0, b"")))
            self._wait(lambda: core.get_clipboard() == base, 2)
            res["autocopy_append_ok"] = core.get_clipboard() == base
            res["autocopy_append_ms"] = round((time.perf_counter() - t_app) * 1000)
            edit = " Edited by hand."
            t_ed = time.perf_counter()
            self._call(lambda: self.text.insert("end", edit))
            time.sleep(0.2)
            res["clipboard_200ms_after_edit_is_old"] = core.get_clipboard() == base
            self._wait(lambda: core.get_clipboard() == base + edit, 3)
            res["autocopy_edit_ok"] = core.get_clipboard() == base + edit
            res["autocopy_edit_ms"] = round((time.perf_counter() - t_ed) * 1000)
            self._call(self.clear_text)

            # 4) training data: two utterances (real clip audio) + one edit inside the 2nd
            n0 = len(self.training.records)
            pcms = [(np.clip(c, -1, 1) * 32767).astype("<i2").tobytes() for c in clips[:2]]
            raw1 = "Well, I don't wish to see it any more, observed Phebe."
            raw2 = "This is a local transcription test running on the workstation graphics card."
            self._call(lambda: self._handle(("text", raw1, time.perf_counter(), 0, 0, 1.0, pcms[0])))
            self._call(lambda: self._handle(("text", raw2, time.perf_counter(), 0, 0, 1.0, pcms[1])))

            def user_edit():   # what typing does: replace words inside utterance 2, type at its end
                i = self.text.search("graphics card", "1.0", "end")
                self.text.delete(i, f"{i}+13c")
                self.text.insert(i, "RTX 3050 Ti")
                self.text.insert("end", " Done")
            self._call(user_edit)
            time.sleep(1.6)                               # > 1 s training debounce
            self.training.flush(5)
            with open(self.training.manifest, encoding="utf-8") as fh:
                recs = [json.loads(ln) for ln in fh if ln.strip()][-2:]
            exp2 = "This is a local transcription test running on the workstation RTX 3050 Ti. Done"
            keys = ("audio_filepath", "duration", "text", "raw_text", "corrected_text", "edited")
            res["training"] = {
                "records_added": len(self.training.records) - n0, "manifest": self.training.manifest,
                "rec1": {k: recs[0].get(k) for k in keys}, "rec2": {k: recs[1].get(k) for k in keys},
                "rec1_ok": recs[0]["raw_text"] == raw1 and recs[0]["corrected_text"] == raw1 and not recs[0]["edited"],
                "rec2_ok": (recs[1]["raw_text"] == raw2 and recs[1]["corrected_text"] == exp2
                            and recs[1]["edited"] and recs[1]["text"] == exp2),
                "wavs_exist": all(os.path.exists(r["wav_path"]) for r in recs),
                "wav_seconds": [round((os.path.getsize(r["wav_path"]) - 44) / 32000, 2) for r in recs],
                "errors": list(self.training.errors)}
            self._call(self.clear_text)                   # Clear must keep the records
            self.training.flush(5)
            res["training"]["records_after_clear"] = len(self.training.records) - n0

            # 5) tray behaviour: hide/show through the same handler as a left click; tool-window style
            self._call(lambda: self._tray_event("toggle_show"))
            time.sleep(0.4)
            res["after_hide"] = {"visible": self.visible, **self._call(self.exstyle)}
            self._call(lambda: self._tray_event("toggle_show"))
            time.sleep(0.4)
            res["after_show"] = {"visible": self.visible, **self._call(self.exstyle)}

            # 7) clear-on-return through the real hotkey handler (silent source, simulated window switch)
            silent = type("Silent", (), {"info": "no-mic", "stop": lambda s: None})()
            self._call(self.clear_text)
            msg = "First message for clear on return."
            self._call(lambda: self._handle(("text", msg, time.perf_counter(), 0, 0, 1.0, b"")))
            self._wait(lambda: self._copied_text == msg and self.cor_armed, 3)
            self._fg_override = (0x7FFF0001, False)              # "another app" is now in front
            self._wait(lambda: self.cor_left, 3)
            self._call(lambda: self._on_hotkey(source=silent))
            cor = {"cleared": self._call(lambda: self.text.get("1.0", "end-1c")) == "", "recording": self.recording}
            self._call(lambda: self._on_hotkey(source=silent))   # stop
            cor["ctrl_z_restored"] = self._call(lambda: self._on_ctrl_z()) == "break" and \
                self._call(lambda: self.text.get("1.0", "end-1c")) == msg
            self._fg_override = None
            cor["ok"] = cor["cleared"] and cor["recording"] and cor["ctrl_z_restored"]
            res["clear_on_return"] = cor
            self._call(self.clear_text)

            # 8) show_on_hotkey: hidden -> hotkey shows it (tool window, not ours in front), stop keeps it shown
            self._call(self.hide)
            time.sleep(0.3)
            fg0 = _u32.GetForegroundWindow()
            self._call(lambda: self._on_hotkey(source=silent))
            time.sleep(0.4)
            shk = {"visible": self.visible, "recording": self.recording, **self._call(self.exstyle)}
            self._call(lambda: self._on_hotkey(source=silent))   # stop
            time.sleep(0.2)
            shk["visible_after_stop"] = self.visible
            shk["foreground_unchanged"] = _u32.GetForegroundWindow() == fg0
            shk["foreground_not_ours"] = not self._fg_info()[1]
            shk["ok"] = (shk["visible"] and shk["recording"] and shk["toolwindow"] and not shk["appwindow"]
                         and shk["visible_after_stop"] and shk["foreground_not_ours"])
            res["show_on_hotkey"] = shk

            # 6) idle VRAM with the model warm
            time.sleep(1.0)
            res["health_idle"] = self.hclient.health(timeout=3)
            res["ok"] = True
        except Exception as e:
            res["ok"] = False
            res["error"] = f"{e.__class__.__name__}: {e}"
            res["trace"] = traceback.format_exc()
        res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.args.selftest, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        if not self.args.keep_open:
            self._call(self.quit, wait=False)


def batch_cmd(args):
    """--cmd batchhold/batchrelease for the batch launcher. Returns the process exit code."""
    if args.pid <= 0:
        print("--pid is required (the launcher's process id)")
        return 2
    sent = False
    for _ in range(20):
        if trayw.send_ipc(args.cmd, args.pid):
            sent = True
            break
        time.sleep(0.25)
    if not sent:
        print("Parakeet Live is not reachable - continuing without it")
        return 3
    if args.cmd == "batchrelease":
        print("Parakeet Live: batch hold released (the model goes back to the GPU unless something else needs it)")
        return 0
    t0 = time.perf_counter()
    last_dump = t0
    while True:
        time.sleep(0.2)
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            st = {}
        held = any(h.get("pid") == args.pid for h in (st.get("batch_holds") or []))
        if held and st.get("engine") in ("unloaded", "prewarming", "standby", "error"):
            print(f"Parakeet Live: model is off the GPU for the batch job ({time.perf_counter() - t0:.1f} s)")
            return 0
        if time.perf_counter() - t0 > args.wait:
            print(f"Parakeet Live: model still on the GPU after {args.wait:g} s (busy dictating?) - continuing; "
                  f"it moves as soon as the current utterance ends")
            return 4
        if time.perf_counter() - last_dump > 2:
            trayw.send_ipc("dump")
            last_dump = time.perf_counter()


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tray", action="store_true", help="start in the tray only; don't load the model")
    ap.add_argument("--cmd", choices=list(trayw.IPC_CODES), help="send a command to the running instance")
    ap.add_argument("--idle-min", type=float, default=None, help="override idle-unload minutes (not saved)")
    ap.add_argument("--training-dir", default=None, help="override the training-data folder (tests)")
    ap.add_argument("--prewarm-delay", type=float, default=None, help="override the login pre-load delay in s (tests)")
    ap.add_argument("--standby-h", type=float, default=None, help="override hours in RAM standby before freeing RAM (tests)")
    ap.add_argument("--selftest", help="run the automated test and write results JSON here")
    ap.add_argument("--record-seconds", type=float, default=4.0)
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--replay-rounds", type=int, default=1)
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--no-mic", action="store_true", help="self-test: don't open the microphone")
    ap.add_argument("--pid", type=int, default=0, help="--cmd batchhold/batchrelease: the batch launcher's process id")
    ap.add_argument("--wait", type=float, default=0, help="--cmd batchhold: wait up to this many s for the model to leave the GPU")
    ap.add_argument("--no-save", action="store_true", help="don't write config.json")
    ap.add_argument("--server-token", default=None, help=argparse.SUPPRESS)   # tests only: pin the server secret
    ap.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)      # tests only: where the draft/retries live
    return ap


def main():
    args = build_parser().parse_args()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("JaydeeWetwork.ParakeetLive")
    except Exception:
        pass
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wt.HANDLE
    mutex = k32.CreateMutexW(None, False, "Local\\ParakeetLive.Widget")
    already = ctypes.get_last_error() == 183
    if already and args.cmd in ("batchhold", "batchrelease"):
        raise SystemExit(batch_cmd(args))
    if already:                                        # already running: forward the request and exit
        cmd = args.cmd or (None if args.tray else "show")
        if cmd:
            for _ in range(20):                        # the first instance may still be creating its tray
                if trayw.send_ipc(cmd):
                    break
                time.sleep(0.25)
        return
    if args.cmd in ("batchhold", "batchrelease"):
        print("Parakeet Live is not running - nothing to move off the GPU")
        raise SystemExit(3)
    if args.cmd and args.cmd != "show":
        return                                         # nothing running: nothing to hide/quit/unload
    try:
        App(args).run()
    except Exception:
        log("fatal: " + traceback.format_exc())
        raise
    finally:
        k32.CloseHandle(mutex)


if __name__ == "__main__":
    main()
