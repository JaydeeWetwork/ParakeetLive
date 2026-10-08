"""Log 55: widget-side fixes of the code review, tested next to the running widget without disturbing it.
Same sandbox as test-clear-on-return.py: a second App in this process with its own data dir, temp config,
temp training folder, fake clipboard (the real one is never touched), fake microphone (no audio device),
no tray icon, no global hotkey, model loading disabled; the speech server is a fake client object.
  R1 clipboard busy: one retry; still busy -> counted, "Not copied" flash, next copy goes through
  R2 failed transcription: one immediate retry; still failing -> audio kept and transcribed when the engine
     is back (text + training record); a refusal (4xx) is not kept
  R3 GPU policy: an engine that stopped on its own restarts after ~3 s, at most 2 times in 10 min, never
     while parked
  R4 dictated text never goes to widget.log (only its length)
  R5 config: unparsable / non-object config.json is copied aside (never deleted) and defaults are used;
     wrong-typed values fall back per key; migrate_config survives junk; Jaydee's real config.json (a copy)
     loads unchanged
  R6 window placement: the saved position (Jaydee's, from a copy of his config) is used as is; a position
     on no monitor falls back to the default spot (the saved size is checked shown, in test-show-on-hotkey)
  R7 idle CPU: idle = slow animation cadence and no canvas redraws; recording = fast cadence
  R8 the server token: random per run, used by the client, pinned by --server-token
  R9 quit waits for the last transcription: text reaches the box, the training log and the clipboard
"""
import importlib.machinery, importlib.util, json, os, shutil, sys, tempfile, threading, time, traceback  # noqa: E401
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(REPO, "widget")
TMP = os.path.join(tempfile.gettempdir(), "plive-review-test")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(os.path.join(TMP, "data", "logs"))
REAL_CFG = os.path.join(W, "config.json")
REAL_CFG_COPY = os.path.join(TMP, "real-config-copy.json")
if os.path.exists(REAL_CFG):
    shutil.copy2(REAL_CFG, REAL_CFG_COPY)          # read-only use of Jaydee's settings
os.environ["PARAKEET_LIVE_DATA"] = os.path.join(TMP, "data")
sys.path.insert(0, W)
import plive_core as core   # noqa: E402
import plive_tray as trayw  # noqa: E402

real_get_clip = core.get_clipboard
CLIP_BEFORE = real_get_clip()
fake = {"clip": None, "mics": 0, "fail": 0}


def fake_set(t):
    if fake["fail"] > 0:
        fake["fail"] -= 1
        return False
    fake["clip"] = t
    return True


core.set_clipboard = fake_set
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
core.parse_hotkey = lambda s: (_ for _ in ()).throw(ValueError("test: no global hotkey"))
trayw.CLASS_NAME = "ParakeetLiveTrayReviewTest"
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
LOADS = []
plw.App.load_model = lambda self, *a, **k: LOADS.append(time.perf_counter())
plw.App.prewarm = lambda self, *a, **k: None
plw.App._gpu_watch_loop = lambda self: None
TRAIN = os.path.join(TMP, "training")
args = plw.build_parser().parse_args(["--tray", "--no-save", "--training-dir", TRAIN])
app = plw.App(args)
LOG = os.path.join(TMP, "data", "logs", "widget.log")
PCM = bytes(32000)
res, results = {}, {}


def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def ui(fn): return app._call(fn)
def box(): return ui(lambda: app.text.get("1.0", "end-1c"))
def wait(c, t=3): return app._wait(c, t)


def check(name, ok, **info):
    results[name] = bool(ok)
    res[name] = {"ok": bool(ok), **info}
    p(name, "PASS" if ok else "FAIL", json.dumps(info, default=str)[:1500])


class FakeClient:
    """Stands in for core.Client: scripted failures, then a fixed text."""
    def __init__(self):
        self.script, self.calls, self.delay, self.token = [], 0, 0.0, None

    def transcribe(self, pcm, timeout=60.0):
        self.calls += 1
        time.sleep(self.delay)
        if self.script:
            e = self.script.pop(0)
            if e is not None:
                raise e
        return {"text": self.text, "infer_ms": 5}

    def close(self):
        pass


def engine_ready(fc):
    def f():
        app.tclient = fc
        app.engine = "ready"
        app.ready_evt.set()
    ui(f)


def push_final(pcm=PCM):
    uid = core.new_utterance_id()
    app._on_utterance(pcm, time.perf_counter(), len(pcm) / 32000, "final", uid, 0)


def body():
    wait(lambda: app.tray.ready.is_set(), 5)
    time.sleep(0.5)
    # R8 token
    t2 = plw.build_parser().parse_args(["--tray", "--server-token", "pinned123"]).server_token
    check("R8_server_token", len(app.server_token) == 32 and int(app.server_token, 16) >= 0
          and app.tclient.token == app.server_token and t2 == "pinned123", token_len=len(app.server_token))
    # R1 clipboard busy (visible flag on so the flash label is updated; the window itself stays unmapped)
    ui(app.clear_text)
    ui(lambda: setattr(app, "visible", True))
    fake["fail"] = 1                                  # busy once -> the retry succeeds
    c0, f0 = app.copy_count, app.copy_failures
    ui(lambda: app.append_text("Clipboard retry test."))
    r1a = wait(lambda: fake["clip"] == "Clipboard retry test." and app.copy_count == c0 + 1, 3)
    fake["fail"] = 2                                  # busy for both tries
    ui(lambda: app.append_text("More."))
    r1b = wait(lambda: app.copy_failures == f0 + 1, 3)
    flash = ui(lambda: app.flash.cget("text"))
    last_reset = app._last_copied is None
    ui(lambda: app.copy_all())                        # the same text again: must not be suppressed now
    r1c = wait(lambda: fake["clip"] == "Clipboard retry test. More.", 3)
    ui(lambda: setattr(app, "visible", False))
    check("R1_clipboard_busy_retry_and_report", r1a and r1b and "Not copied" in flash and last_reset and r1c,
          retry_ok=r1a, failure_counted=r1b, flash=flash, last_copied_reset=last_reset, recopied=r1c)
    # R2 failed transcription kept and retried
    ui(app.clear_text)
    fc = FakeClient()
    fc.text = "Kept through a server restart."
    fc.script = [ConnectionResetError("server restarted"), ConnectionResetError("still down")]
    engine_ready(fc)
    n_rec0 = len(app.training.records)
    push_final()
    kept = wait(lambda: len(app.failed) == 1, 3)
    status = ui(lambda: app.status.cget("text"))
    calls_first = fc.calls
    got = wait(lambda: box() == "Kept through a server restart.", 6)   # requeued 3 s later (engine ready)
    app.training.flush(5)
    rec_ok = len(app.training.records) == n_rec0 + 1
    fc.script = [core.ServerError("utterance too long", 413)]
    push_final()
    time.sleep(1.0)
    not_kept = len(app.failed) == 0 and fc.calls == calls_first + 2
    check("R2_failed_transcription_kept_and_retried", kept and "kept" in status and calls_first == 2 and got
          and rec_ok and not_kept and app.pending == 0, kept=kept, status=status, first_calls=calls_first,
          text_arrived=got, training_record=rec_ok, refusal_not_kept=not_kept, pending=app.pending)
    # R2b engine down while the utterance is queued: kept, then re-queued on "ready"
    ui(app.clear_text)
    def down():
        app.ready_evt.clear()
        app.engine = "error"
    ui(down)
    fc.script, fc.text = [], "Said while the engine was down."
    push_final()
    kept2 = wait(lambda: len(app.failed) == 1, 4)
    ui(lambda: (setattr(app, "engine", "loading"), app.ready_evt.set()))
    ui(lambda: app._handle(("ready", app.gen, 1.0)))
    got2 = wait(lambda: box() == "Said while the engine was down.", 4)
    check("R2b_engine_down_then_ready_requeues", kept2 and got2, kept=kept2, delivered=got2)
    # R3 auto-recover (GPU policy)
    ui(app.clear_text)
    app.cfg["gpu_always"] = True
    LOADS.clear()
    for i in range(3):
        ui(lambda: app._handle(("server_error", app.gen, "Speech engine stopped - test")))
        wait(lambda: len(LOADS) > i, 4.5)
        ui(lambda: setattr(app, "engine", "error"))
    n_three = len(LOADS)
    app._recover_times.clear()
    app.parked = {"reason": "test", "proc": "game.exe", "temp": False}
    ui(lambda: app._handle(("server_error", app.gen, "Speech engine stopped - parked test")))
    time.sleep(3.6)
    parked_loads = len(LOADS) - n_three
    app.parked = None
    app.cfg["gpu_always"] = False
    check("R3_auto_recover_bounded", n_three == 2 and parked_loads == 0, restarts_for_3_crashes=n_three,
          restarts_while_parked=parked_loads)
    ui(lambda: setattr(app, "engine", "ready"))
    # R4 privacy
    secret = "Zebra quartz lantern"
    ui(lambda: app._handle(("text", secret, time.perf_counter(), 3, 4.0, 1.0, b"")))
    time.sleep(0.2)
    logtxt = open(LOG, encoding="utf-8", errors="replace").read()
    check("R4_no_dictated_text_in_log", secret not in logtxt and f"{len(secret)} chars" in logtxt)
    # R5 config
    real_cp = plw.CONFIG_PATH
    out = {}
    for name, content in (("broken", "{\"x\": 12, "), ("list", "[1, 2]"),
                          ("types", json.dumps({"x": "abc", "y": 5, "opacity": "high", "hotkey": "ctrl+alt+space",
                                                "autocopy": 1, "port": 80, "park_games": ["a.exe"]}))):
        plw.CONFIG_PATH = os.path.join(TMP, f"cfg-{name}.json")
        with open(plw.CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            mig = plw.migrate_config()
            c = plw.load_config()
            err = None
        except Exception as e:
            mig, c, err = None, None, repr(e)
        aside = [f for f in os.listdir(TMP) if f.startswith(f"cfg-{name}.json.corrupt-")]
        out[name] = {"err": err, "aside": len(aside), "still_there": os.path.exists(plw.CONFIG_PATH), "cfg": c}
    plw.CONFIG_PATH = os.path.join(TMP, "cfg-badver.json")
    json.dump({"config_version": "abc"}, open(plw.CONFIG_PATH, "w"))
    try:
        badver = plw.migrate_config()
    except Exception as e:
        badver = f"CRASH {e!r}"
    tc = out["types"]["cfg"] or {}
    real_ok, real_info = None, "no config.json"
    if os.path.exists(REAL_CFG_COPY):
        plw.CONFIG_PATH = REAL_CFG_COPY
        raw = json.load(open(REAL_CFG_COPY, encoding="utf-8"))
        lc = plw.load_config()
        diff = [k for k in raw if k in plw.DEFAULTS and lc.get(k) != raw[k]]
        real_ok, real_info = not diff and not [f for f in os.listdir(TMP) if "real-config-copy.json.corrupt" in f], diff
    plw.CONFIG_PATH = real_cp
    check("R5_config_corrupt_kept_types_checked_real_config_unchanged",
          all(out[k]["err"] is None and out[k]["aside"] == 1 and out[k]["still_there"]
              and out[k]["cfg"] == plw.DEFAULTS for k in ("broken", "list"))
          and out["types"]["aside"] == 0 and tc.get("x") is None and tc.get("y") == 5
          and tc.get("opacity") == plw.DEFAULTS["opacity"] and tc.get("autocopy") is True
          and tc.get("port") == plw.DEFAULTS["port"] and tc.get("park_games") == ["a.exe"]
          and isinstance(badver, str) and "skipped" in badver and real_ok,
          broken={k: v for k, v in out["broken"].items() if k != "cfg"},
          lst={k: v for k, v in out["list"].items() if k != "cfg"}, bad_version=badver,
          real_config_keys_changed=real_info)
    # R6 placement
    jc = json.load(open(REAL_CFG_COPY, encoding="utf-8")) if os.path.exists(REAL_CFG_COPY) else {}
    geo = {}
    saved = {k: app.cfg[k] for k in ("x", "y", "w", "h")}
    for name, vals in (("jaydee", {k: jc.get(k) for k in ("x", "y", "w", "h")}),
                       ("nowhere", {"x": -30000, "y": -30000, "w": 900, "h": 140})):
        app.cfg.update(vals)
        ui(app._place_window)
        ui(app.root.update_idletasks)
        geo[name] = ui(lambda: app.root.geometry())
    app.cfg.update(saved)
    jx, jy = jc.get("x"), jc.get("y")
    on_j = plw.on_a_monitor(jx, jy) if jx is not None else None

    def parse(g):                     # "1005x143+-2760+-405" / "900x140+10-20" -> (w, h, x, y)
        import re
        wh, rest = g.split("+", 1)[0].split("-", 1)[0], g[len(g.split("+", 1)[0].split("-", 1)[0]):]
        xy = [int(v) if sg == "+" else -int(v) for sg, v in re.findall(r"([+-])(-?\d+)", rest)]
        w, h = (int(v) for v in wh.split("x"))
        return (w, h, *xy)
    gj, gn = parse(geo["jaydee"]), parse(geo["nowhere"])
    check("R6_saved_position_kept_offscreen_falls_back",
          (jx is None or (on_j and gj[2:] == (jx, jy)))   # size of a never-mapped window reads back stale (S1/S4 check it)
          and not plw.on_a_monitor(-30000, -30000) and gn[2] != -30000 and gn[3] != -30000,
          jaydee_saved=[jx, jy, jc.get("w"), jc.get("h")], jaydee_geometry=geo["jaydee"], nowhere=geo["nowhere"])
    # R7 idle cadence / redraws (visible flag only; the sandbox window is never mapped)
    ticks, cfgs = {"a": 0, "h": 0}, {"n": 0}
    orig_anim, orig_hb, orig_ic = app._animate, app._heartbeat, app.btn.itemconfigure
    def anim():
        ticks["a"] += 1
        orig_anim()
    def hb():
        ticks["h"] += 1
        orig_hb()
    def ic(*a, **k):
        cfgs["n"] += 1
        return orig_ic(*a, **k)
    app._animate, app._heartbeat, app.btn.itemconfigure = anim, hb, ic
    ui(lambda: setattr(app, "visible", True))
    time.sleep(0.6)
    ticks.update(a=0, h=0)
    cfgs["n"] = 0
    time.sleep(2.0)
    idle = dict(ticks, redraws=cfgs["n"])
    ui(lambda: app.start_recording())
    time.sleep(0.4)
    ticks.update(a=0, h=0)
    time.sleep(2.0)
    rec = dict(ticks)
    ui(app.stop_recording)
    ui(lambda: setattr(app, "visible", False))
    app._animate, app._heartbeat, app.btn.itemconfigure = orig_anim, orig_hb, orig_ic
    check("R7_idle_slow_recording_fast", idle["a"] <= 10 and idle["h"] <= 10 and idle["redraws"] <= 2
          and rec["a"] >= 30 and rec["h"] >= 50, idle_2s=idle, recording_2s=rec)
    # R9 quit drains (last step: the app ends here)
    ui(app.clear_text)
    fc2 = FakeClient()
    fc2.text, fc2.delay = "The last words before quitting.", 1.0
    engine_ready(fc2)
    n_rec1 = len(app.training.records)
    seen = {}
    orig_qn = app._quit_now
    def qn():
        seen["box"] = app.text.get("1.0", "end-1c")
        seen["clip"] = fake["clip"]
        seen["pending"] = app.pending
        app.training.flush(5)
        seen["records"] = len(app.training.records) - n_rec1
        orig_qn()
    app._quit_now = qn
    push_final()
    time.sleep(0.05)
    t0 = time.perf_counter()
    res["_r9_t0"] = t0
    app._call(app.quit, wait=False)
    res["_r9_seen"] = seen


def runner():
    try:
        body()
    except Exception:
        res["error"] = traceback.format_exc()
        p("ERROR", res["error"])
        app._call(app.quit, wait=False)


threading.Thread(target=runner, daemon=True).start()
app.run()
seen = res.pop("_r9_seen", {})
t0 = res.pop("_r9_t0", None)
want = "The last words before quitting."
check("R9_quit_waits_for_last_transcription", seen.get("box") == want and seen.get("clip") == want
      and seen.get("pending") == 0 and seen.get("records") == 1, seen=seen)
res["real_clipboard_untouched"] = real_get_clip() == CLIP_BEFORE
res["fake_mic_starts"] = fake["mics"]
res["summary"] = results
res["sandbox_guard"] = sandbox_guard.summary()
res["all_ok"] = (bool(results) and all(results.values()) and res["real_clipboard_untouched"] and "error" not in res
                 and sandbox_guard.ok())
print("RESULT " + json.dumps({k: v for k, v in res.items() if k in ("summary", "all_ok", "real_clipboard_untouched", "error", "sandbox_guard")}, indent=1, default=str))
sys.exit(0 if res["all_ok"] else 1)
