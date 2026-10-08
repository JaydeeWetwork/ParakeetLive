"""Log 53: clear-on-return, tested next to the running widget without disturbing it.

Runs a second, sandboxed App in this process: own data dir (PARAKEET_LIVE_DATA), temp config copy,
temp training folder, no tray icon, no global hotkey, a different IPC window class, model loading
disabled, a fake clipboard (Jaydee's real clipboard is never touched) and a fake microphone (no audio
device is opened). Transcripts are injected through the real UI path (_handle "text"); window switches
are simulated through App._fg_override; the hotkey goes through the real tray-event path.
No window is shown.
  C1  copied + switched away + hotkey -> cleared, recording starts (fake mic), training records kept
  C2  came back by clicking the widget, then the hotkey -> still clears (log 55); never left + widget
      focused -> appends
  C3  never switched away since the text arrived -> appends
  C4  our own windows (menus) don't count as leaving; app A -> widget -> app A does
  C5  recording in progress: hotkey only stops it, never clears
  C6  not copied (auto-copy off) -> no clear; Copy button then counts
  C7  clear_on_return off -> no clear (gear toggle path)
  C8  Ctrl+Z in the box restores (with training spans: a later edit still corrects the record)
  C9  restore after new dictation: old message first, new one kept; tray/gear item state
  C10 refused recording (GPU busy) gives the text back automatically
  C11 Clear button is restorable too; Ctrl+Z after typing = normal undo
  C12 config migration v1/v2/v3 -> current (backup, other keys kept); current untouched
  C13 the real GetForegroundWindow path works and does not count a foreign window as ours
  log 55 (Jaydee: "sometimes the old message doesn't delete"):
  C14 record BUTTON after leaving and clicking back into the widget -> clears (real press/release handlers)
  C15 tray > Record and --cmd record clear too
  C16 the last words arrive (and are auto-copied) after you already left -> the departure still counts
  C17 start while the previous recording's text is still pending -> clears; that late text joins the
      cleared message (Ctrl+Z brings back all of it), the new dictation stays in the box
  C18 typing in the box after leaving cancels the clear (right away, and after its auto-copy); leaving
      again with the edited text clears it
  log 57 (same-window paste; keys are faked here, so a real Ctrl+V during the test changes nothing):
  C19 copied, stayed in the same window, Ctrl+V there -> counts as pasted; the next recording clears
  C20 setting off (gear) -> no key watch, Ctrl+V ignored, the next recording appends
  C21 widget in front -> the key watch stops; Ctrl+V into the widget does not count
  C22 nothing to watch (box cleared / already left) -> no key reads at all
  C23 CPU cost of the watch: real reads of the two keys for 10 s; each poll is timed (wall time per call
      = upper bound of its CPU) -> share of one core; the process CPU A/B (10 s off vs on) is shown too
      but is dominated by noise at this size
  C24 config v4 -> v5: unchanged game list gets the launchers + java.exe, a changed list is kept,
      paste_detect on
"""
import importlib.machinery, importlib.util, json, os, shutil, sys, tempfile, threading, time, traceback
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(REPO, "widget")
TMP = os.path.join(tempfile.gettempdir(), "plive-cor-test")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(os.path.join(TMP, "data", "logs"))
os.environ["PARAKEET_LIVE_DATA"] = os.path.join(TMP, "data")
sys.path.insert(0, W)
import plive_core as core   # noqa: E402
import plive_tray as trayw  # noqa: E402

real_get_clip = core.get_clipboard
CLIP_BEFORE = real_get_clip()
fake = {"clip": None, "mics": 0, "ctrlv": False}
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
core.parse_hotkey = lambda s: (_ for _ in ()).throw(ValueError("test: no global hotkey"))
trayw.CLASS_NAME = "ParakeetLiveTrayCorTest"
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
            "clear_on_return": True, "config_version": plw.CONFIG_VERSION,
            "opacity": 0.02})                       # the hotkey may show the sandbox now (log 54): keep it invisible
plw.CONFIG_PATH = os.path.join(TMP, "config.json")
json.dump(cfg, open(plw.CONFIG_PATH, "w"), indent=1)
plw.App.load_model = lambda self, *a, **k: None
plw.App.prewarm = lambda self, *a, **k: None
plw.App._gpu_watch_loop = lambda self: None
TRACE = []
_orig_arm, _orig_watch, _orig_append = plw.App._cor_arm, plw.App._cor_watch, plw.App.append_text


def _tr(tag, self, extra=""):
    TRACE.append(f"{time.perf_counter():.3f} {tag} armed={self.cor_armed} left={self.cor_left} fg={self._cor_fg} "
                 f"job={self._cor_job} ov={self._fg_override} {extra}")
    del TRACE[:-60]


def _arm(self, copied):
    _tr("arm>", self, repr(copied[:20]))
    _orig_arm(self, copied)
    _tr("arm<", self)


def _watch(self):
    _orig_watch(self)
    _tr("watch", self)


def _append(self, t, rid=None, session=None):
    r = _orig_append(self, t, rid, session)
    _tr("append", self, repr(t[:20]))
    return r


plw.App._cor_arm, plw.App._cor_watch, plw.App.append_text = _arm, _watch, _append
_orig_ctrl_v = plw.App._ctrl_v
plw.App._ctrl_v = lambda self: (setattr(self, "paste_polls", self.paste_polls + 1), fake["ctrlv"])[1]  # fake keys
TRAIN = os.path.join(TMP, "training")
args = plw.build_parser().parse_args(["--tray", "--no-save", "--training-dir", TRAIN])
app = plw.App(args)

OTHER_A, OTHER_B, OWN = (0x7F0000A1, False), (0x7F0000B2, False), (0x7F0000C3, True)
PCM = bytes(32000)                            # 1 s of 16 kHz int16 silence (training WAV in TMP only)
res, results = {}, {}


def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def ui(fn): return app._call(fn)
def box(): return ui(lambda: app.text.get("1.0", "end-1c"))
def wait(c, t=3): return app._wait(c, t)
def fg(v): app._fg_override = v


def text(t, pcm=b""):
    ui(lambda: app._handle(("text", t, time.perf_counter(), 0, 0, 1.0, pcm)))


def copied(t): return wait(lambda: app._copied_text == t and fake["clip"] == t, 3)


def hotkey(): ui(lambda: app._tray_event("hotkey"))


def stop():
    if app.recording:
        hotkey()


def reset(front=OTHER_A):
    stop()
    ui(app.clear_text)
    app.cor_restore = None
    app._cor_prog_text = None
    app.pending = 0
    fg(front)
    time.sleep(0.6)             # > the 400 ms auto-copy debounce of the clear


def leave(to=OTHER_B):
    fg(to)
    return wait(lambda: app.cor_left, 2)


def cor():
    return ui(lambda: {"armed": app.cor_armed, "left": app.cor_left, "fg": app._cor_fg, "job": app._cor_job,
                       "copied_is_box": app._copied_text == app.text.get("1.0", "end-1c"),
                       "fake_clip_is_box": fake["clip"] == app.text.get("1.0", "end-1c"), "override": app._fg_override})


def check(name, ok, **info):
    results[name] = bool(ok)
    if not ok:
        info["trace"] = TRACE[-25:]
    res[name] = {"ok": bool(ok), **info}
    p(name, "PASS" if ok else "FAIL", json.dumps(info)[:600 if ok else 6000])


def body():
    wait(lambda: app.tray.ready.is_set(), 5)
    time.sleep(0.5)
    # C1
    reset()
    m1 = "Hey, can you send me the report by Friday?"
    text(m1, PCM)
    ok_copy = copied(m1)
    app.training.flush(5)
    n_rec = len(app.training.records)
    left = leave()
    mics0 = fake["mics"]
    hotkey()
    app.training.flush(5)
    wavs_ok = all(os.path.exists(r["wav_path"]) for r in app.training.records.values())
    check("C1_cleared_after_paste_elsewhere", ok_copy and left and box() == "" and app.recording
          and fake["mics"] == mics0 + 1 and bool(app.cor_restore) and n_rec >= 1
          and len(app.training.records) == n_rec and wavs_ok,
          copied=ok_copy, left=left, box=box(), recording=app.recording, fake_mic_starts=fake["mics"] - mics0,
          training_records=len(app.training.records), training_wavs_exist=wavs_ok)
    stop()
    # C2
    reset()
    text("First part.")
    copied("First part.")
    leave()
    fg(OWN)                     # came back by clicking into the widget, then pressed the hotkey
    time.sleep(0.4)
    hotkey()
    b_rec, b_box = app.recording, box()
    stop()
    text("Second part.")
    after = box()
    reset(front=OWN)            # never left: dictating with the widget focused keeps adding
    text("Same message.")
    copied("Same message.")
    time.sleep(0.6)
    hotkey()
    stop()
    text("Continued.")
    check("C2_widget_clicked_first_still_clears", b_rec and b_box == "" and after == "Second part."
          and box() == "Same message. Continued.", cleared_box=b_box, after=after, never_left=box())
    # C3
    reset(front=OTHER_A)
    text("Dictating into the app in front.")
    copied("Dictating into the app in front.")
    time.sleep(0.8)             # app A stays in front the whole time
    hotkey()
    stop()
    text("More of the same message.")
    check("C3_no_switch_keeps_adding", box() == "Dictating into the app in front. More of the same message."
          and not app.cor_restore, box=box())
    # C4
    reset(front=OTHER_A)
    text("Own windows test.")
    copied("Own windows test.")
    fg(OWN)
    time.sleep(0.6)
    own_left = app.cor_left
    fg(OTHER_A)                 # back to the app it was in: that is a switch away from the widget
    back_left = wait(lambda: app.cor_left, 2)
    check("C4_own_windows_ignored_A_widget_A_counts", not own_left and back_left, own_window_counted=own_left,
          back_to_A_counted=back_left)
    # C5
    reset()
    text("Busy test.")
    copied("Busy test.")
    hotkey()                    # starts recording (no switch yet -> no clear)
    leave()
    hotkey()                    # stops recording: must not clear
    after_stop = box()
    check("C5_stop_never_clears", after_stop == "Busy test." and not app.recording, after_stop=after_stop)
    # C6
    reset()
    ui(lambda: app._set_autocopy(False))
    text("Not copied yet.")
    time.sleep(0.6)
    leave()
    time.sleep(0.4)
    hotkey()
    nc = box()
    stop()
    fg(OWN)
    ui(lambda: app.copy_all(force=True))          # Copy button
    cb = copied("Not copied yet.")
    left = leave()
    hotkey()
    check("C6_only_after_copy", nc == "Not copied yet." and cb and left and box() == "", not_copied_box=nc,
          after_copy_button_box=box())
    stop()
    ui(lambda: app._set_autocopy(True))
    # C7
    reset()
    ui(lambda: app._set_clear_on_return(False))
    text("Toggle off.")
    copied("Toggle off.")
    leave()
    hotkey()
    off_box = box()
    stop()
    ui(lambda: app._set_clear_on_return(True))
    check("C7_toggle_off", off_box == "Toggle off." and app.cfg["clear_on_return"] and app.cor_var.get(),
          box=off_box)
    # C8
    reset()
    m8 = "This is a local transcription test running on the workstation graphics card."
    text(m8, PCM)
    rid = list(app.spans)[-1]
    c8_copied = copied(m8)
    c8_before = cor()
    c8_left = leave()
    c8_after = cor()
    hotkey()
    cleared = box() == ""
    stop()
    r = ui(lambda: app._on_ctrl_z())
    restored = box() == m8
    span_back = rid in app.spans
    ui(lambda: (lambda i: (app.text.delete(i, f"{i}+13c"), app.text.insert(i, "RTX 3050 Ti")))(
        app.text.search("graphics card", "1.0", "end")))
    time.sleep(1.5)
    app.training.flush(5)
    with open(app.training.manifest, encoding="utf-8") as fh:
        recs = [json.loads(ln) for ln in fh if ln.strip()]
    rec = [x for x in recs if x.get("id") == rid]
    corrected = rec[-1].get("corrected_text") if rec else None
    bound = bool(app.text.bind("<Control-z>"))
    check("C8_ctrl_z_restores_with_training_spans", cleared and r == "break" and restored and span_back and bound
          and corrected == m8.replace("graphics card", "RTX 3050 Ti"), copied=c8_copied, before_leave=c8_before,
          left=c8_left, after_leave=c8_after, cleared=cleared, ctrl_z=r, span_back=span_back,
          ctrl_z_bound=bound, restored=restored,
          corrected_after_restore=corrected)
    # C9
    reset()
    text("Old message.")
    copied("Old message.")
    c9_left = leave()
    hotkey()
    c9_cleared = box() == ""
    text("New message.")
    stop()
    tray_item = [it for it in app._tray_menu() if it and it[0] == plw.T_RESTORE]
    ui(lambda: app.menu.entryconfigure(app.restore_index, state="normal" if app.cor_restore else "disabled"))
    gear_state = ui(lambda: str(app.menu.entrycget(app.restore_index, "state")))
    gear_label = ui(lambda: app.menu.entrycget(app.restore_index, "label"))
    ui(lambda: app._menu_cmd(plw.T_RESTORE))
    after = box()
    tray_after = [it for it in app._tray_menu() if it and it[0] == plw.T_RESTORE]
    check("C9_restore_keeps_new_text_menu_items", after == "Old message. New message." and tray_item
          and tray_item[0][2] == 0 and gear_state == "normal" and gear_label == "Restore last message"
          and tray_after[0][2] == trayw.MF_GRAYED and c9_left and c9_cleared, left=c9_left, cleared=c9_cleared,
          box=after, gear_state_before=gear_state, gear_label=gear_label)
    # C10
    reset()
    text("Keep me if recording is refused.")
    copied("Keep me if recording is refused.")
    leave()
    orig = app.start_recording
    app.start_recording = lambda source=None: None     # e.g. "GPU busy" refusal
    c10_left = app.cor_left
    log0 = os.path.getsize(os.path.join(TMP, "data", "logs", "widget.log"))
    hotkey()
    app.start_recording = orig
    with open(os.path.join(TMP, "data", "logs", "widget.log"), encoding="utf-8") as fh:
        fh.seek(log0); tail = fh.read()
    check("C10_refused_recording_restores", c10_left and "cleared" in tail and "[auto]" in tail
          and box() == "Keep me if recording is refused." and not app.cor_restore, left=c10_left, box=box())
    # C11
    reset()
    text("Cleared with the button.")
    ui(lambda: app.clear_by_user("Clear button"))
    btn_clear = box() == ""
    ui(lambda: app.text.insert("end", "typed"))
    r2 = ui(lambda: app._on_ctrl_z())
    ui(lambda: app.text.delete("1.0", "end"))
    r3 = ui(lambda: app._on_ctrl_z())
    check("C11_clear_button_restorable_typing_normal_undo", btn_clear and r2 is None and r3 == "break"
          and box() == "Cleared with the button.", typed_then_ctrl_z=r2, box=box())
    # C13 the real foreground API (no override): returns a window, and it is not ours (we never take focus)
    app._fg_override = None
    real = ui(app._fg_info)
    check("C13_real_foreground_api", real[0] is not None and real[1] is False, own=real[1], has_window=real[0] is not None)
    # state dump fields
    ui(app.dump_state)
    st = json.load(open(plw.STATE_PATH, encoding="utf-8"))
    res["state_fields"] = {"clear_on_return": st.get("clear_on_return"), "cor": st.get("cor")}
    # C14 record button (real press/release handlers) after clicking back into the widget
    class Ev:
        def __init__(self, x, y):
            self.x, self.y, self.x_root, self.y_root = x, y, x, y
    def button():
        d = app.D // 2
        ui(lambda: (app._btn_press(Ev(d, d)), app._btn_release(Ev(d, d))))
    reset()
    text("Message for the button test.")
    copied("Message for the button test.")
    l14 = leave()
    fg(OWN)
    time.sleep(0.4)
    button()
    b14, r14 = box(), app.recording
    button()                    # stop with the button: nothing else changes
    text("New one.")
    check("C14_record_button_clears", l14 and b14 == "" and r14 and not app.recording and box() == "New one."
          and app.cor_restore and app.cor_restore["text"] == "Message for the button test.", left=l14,
          box_at_start=b14, recording=r14, box=box())
    # C15 tray > Record, --cmd record
    out15 = {}
    for name, act in (("tray_menu", lambda: app._menu_cmd(plw.T_RECORD)), ("ipc", lambda: app._ipc("record"))):
        reset()
        text(f"Message for {name}.")
        copied(f"Message for {name}.")
        leave()
        ui(act)
        out15[name] = (box(), app.recording)
        stop()
    check("C15_tray_and_ipc_record_clear", all(v == ("", True) for v in out15.values()), **out15)
    # C16 the last words arrive after you already left (real session bookkeeping)
    reset()
    hotkey()                    # recording S (fake mic)
    s16 = app.rec_session
    stop()
    text("First words.")
    copied("First words.")
    l16 = leave()
    uid = core.new_utterance_id()
    app._uid_session[uid] = s16
    ui(lambda: app._handle(("text", "Last words.", time.perf_counter(), 0, 0, 1.0, b"", uid)))
    c16 = copied("First words. Last words.")
    still = app.cor_left
    fg(OWN)                     # clicks back into the widget
    time.sleep(0.4)
    hotkey()
    b16 = box()
    stop()
    check("C16_late_text_keeps_departure", l16 and c16 and still and b16 == ""
          and app.cor_restore["text"] == "First words. Last words.", left=l16, recopied=c16,
          still_left=still, box_at_start=b16)
    # C17 start while the previous recording's text is still pending
    reset()
    hotkey()
    s17 = app.rec_session
    stop()
    text("Part one.")
    copied("Part one.")
    leave()
    ui(lambda: setattr(app, "pending", 1))
    hotkey()                    # new recording S+1 while S's last text is pending -> clears
    b17 = box()
    u_old, u_new = core.new_utterance_id(), core.new_utterance_id()
    app._uid_session[u_old], app._uid_session[u_new] = s17, app.rec_session
    ui(lambda: app._handle(("text", "Late tail.", time.perf_counter(), 0, 0, 1.0, PCM, u_old)))
    after_late = box()
    ui(lambda: app._handle(("text", "Brand new.", time.perf_counter(), 0, 0, 1.0, b"", u_new)))
    stop()
    new_box = box()
    r17 = ui(lambda: app._on_ctrl_z())   # nothing typed since: restores in front
    check("C17_pending_start_clears_late_text_joins_old", b17 == "" and after_late == ""
          and new_box == "Brand new." and r17 == "break" and box() == "Part one. Late tail. Brand new."
          and app.pending == 0, box_at_start=b17, after_late=after_late, new_box=new_box, restored=box())
    # C18 typing after leaving cancels; leaving again with the edited text clears
    reset()
    text("Draft.")
    copied("Draft.")
    leave()
    fg(OWN)
    ui(lambda: app.text.insert("end", " More"))
    hotkey()                    # within the auto-copy debounce: box != copied -> no clear
    quick = box()
    stop()
    ui(lambda: app.text.insert("end", " typed"))
    c18 = copied("Draft. More typed")
    time.sleep(0.3)
    hotkey()                    # after the edit's auto-copy, not left again -> no clear
    later = box()
    stop()
    l18 = leave()
    hotkey()                    # took the edited text elsewhere -> clears
    final = box()
    stop()
    check("C18_typing_cancels_until_you_leave_again", quick == "Draft. More" and c18
          and later == "Draft. More typed" and l18 and final == "", quick=quick, edit_copied=c18, later=later,
          left_again=l18, final=final)
    # C19 same-window paste: no window switch, Ctrl+V -> pasted -> next recording clears
    reset()
    text("Paste me.")
    c19 = copied("Paste me.")
    w19 = wait(lambda: app._paste_job is not None, 2)
    time.sleep(0.4)
    not_yet = not app.cor_left
    fake["ctrlv"] = True
    l19 = wait(lambda: app.cor_left, 2)
    fake["ctrlv"] = False
    stopped19 = wait(lambda: app._paste_job is None, 1)
    hotkey()
    b19 = box()
    stop()
    check("C19_ctrl_v_same_window_clears", c19 and w19 and not_yet and l19 and stopped19 and b19 == ""
          and bool(app.cor_restore), copied=c19, watching=w19, not_left_before=not_yet, left=l19,
          watch_stopped=stopped19, box_at_start=b19)
    # C20 setting off: no watch, Ctrl+V ignored, appends
    reset()
    ui(lambda: app._set_paste_detect(False))
    text("Keep me.")
    copied("Keep me.")
    time.sleep(0.5)
    w20 = app._paste_job is None
    fake["ctrlv"] = True
    time.sleep(0.5)
    l20 = app.cor_left
    fake["ctrlv"] = False
    hotkey()
    b20 = box()
    stop()
    ui(lambda: app._set_paste_detect(True))
    saved20 = json.load(open(plw.CONFIG_PATH)).get("paste_detect")
    check("C20_setting_off_ignores_ctrl_v", w20 and not l20 and b20 == "Keep me." and saved20 is True,
          no_watch=w20, left=l20, box_at_start=b20, saved_back_on=saved20)
    # C21 widget in front: watch stops, Ctrl+V into the widget doesn't count
    reset()
    text("Mine.")
    copied("Mine.")
    wait(lambda: app._paste_job is not None, 2)
    fg(OWN)
    s21 = wait(lambda: app._paste_job is None, 1)
    polls = app.paste_polls
    fake["ctrlv"] = True
    time.sleep(0.6)
    l21 = app.cor_left
    no_reads = app.paste_polls == polls
    fake["ctrlv"] = False
    check("C21_widget_in_front_no_watch", s21 and not l21 and no_reads, stopped=s21, left=l21,
          no_key_reads_while_widget_in_front=no_reads)
    # C22 nothing to watch: cleared box / already left -> no reads
    reset()
    p0 = app.paste_polls
    time.sleep(1.0)
    idle_reads = app.paste_polls - p0
    text("Gone.")
    copied("Gone.")
    leave()
    time.sleep(0.3)
    p1 = app.paste_polls
    time.sleep(1.0)
    left_reads = app.paste_polls - p1
    check("C22_no_key_reads_when_nothing_to_watch", idle_reads == 0 and left_reads == 0,
          reads_with_empty_box=idle_reads, reads_after_leaving=left_reads)
    # C23 CPU: real reads of Ctrl and V (result ignored) for 10 s vs 10 s without
    reset()
    text("Cpu check.")
    copied("Cpu check.")
    ui(lambda: app._set_paste_detect(False))
    time.sleep(0.5)
    c0, t0 = time.process_time(), time.perf_counter()
    time.sleep(10)
    base = (time.process_time() - c0) / (time.perf_counter() - t0)
    plw.App._ctrl_v = lambda self: (_orig_ctrl_v(self), False)[1]
    timing = []
    _orig_watch = plw.App._paste_watch

    def timed_watch(self):
        t = time.perf_counter_ns()
        _orig_watch(self)
        timing.append(time.perf_counter_ns() - t)
    plw.App._paste_watch = timed_watch
    ui(lambda: app._set_paste_detect(True))
    ui(app._paste_kick)
    p2 = app.paste_polls
    c0, t0 = time.process_time(), time.perf_counter()
    time.sleep(10)
    el = time.perf_counter() - t0
    watch = (time.process_time() - c0) / el
    reads = app.paste_polls - p2
    plw.App._paste_watch = _orig_watch
    plw.App._ctrl_v = lambda self: (setattr(self, "paste_polls", self.paste_polls + 1), fake["ctrlv"])[1]
    ts = sorted(timing) or [0]
    per_poll_us = sum(ts) / len(ts) / 1000
    direct_pct = 100 * sum(ts) / 1e9 / el
    check("C23_cpu_cost_of_key_watch", reads >= 150 and len(timing) >= 150 and direct_pct < 0.5,
          polls_per_s=round(reads / el, 1), per_poll_us_mean=round(per_poll_us, 1),
          per_poll_us_median=round(ts[len(ts) // 2] / 1000, 1), per_poll_us_max=round(ts[-1] / 1000, 1),
          watch_cost_pct_of_one_core=round(direct_pct, 4),
          process_cpu_ab_pct_info=[round(100 * base, 2), round(100 * watch, 2)])
    # C24 config v4 -> v5
    real_cp = plw.CONFIG_PATH
    m24 = {}
    for name, raw in (("v4_default", {"config_version": 4, "park_games": ["javaw.exe", "Minecraft.Windows.exe"], "x": 5}),
                      ("v4_custom", {"config_version": 4, "park_games": ["javaw.exe", "eldenring.exe"],
                                     "paste_detect": False, "x": 5})):
        plw.CONFIG_PATH = os.path.join(TMP, f"mig5-{name}.json")
        json.dump(raw, open(plw.CONFIG_PATH, "w"))
        plw.migrate_config()
        m24[name] = json.load(open(plw.CONFIG_PATH))
    plw.CONFIG_PATH = real_cp
    d, c = m24["v4_default"], m24["v4_custom"]
    check("C24_config_v5_migration", d["config_version"] == 5 and d["park_games"] == plw.DEFAULTS["park_games"]
          and d["paste_detect"] is True and d["x"] == 5 and c["park_games"] == ["javaw.exe", "eldenring.exe"]
          and c["paste_detect"] is False and c["config_version"] == 5, v4_default=d, v4_custom=c)
    # C12 migration (functions only, temp files)
    real_cp = plw.CONFIG_PATH
    out = {}
    for name, raw in (("v1", {"idle_unload_min": 10, "autocopy": False, "x": 5}),
                      ("v2", {"config_version": 2, "idle_unload_min": 0, "gpu_always": True, "auto_park": True,
                              "autocopy": False, "x": 5}),
                      ("v3_off", {"config_version": 3, "clear_on_return": False, "x": 5}),
                      ("cur_off", {"config_version": plw.CONFIG_VERSION, "clear_on_return": False, "x": 5})):
        plw.CONFIG_PATH = os.path.join(TMP, f"mig-{name}.json")
        json.dump(raw, open(plw.CONFIG_PATH, "w"))
        msg = plw.migrate_config()
        got = json.load(open(plw.CONFIG_PATH))
        baks = [f for f in os.listdir(TMP) if f.startswith(f"mig-{name}.json.bak-")]
        out[name] = {"msg": msg, "got": got, "backups": len(baks)}
    plw.CONFIG_PATH = real_cp
    g1, g2, g3 = out["v1"]["got"], out["v2"]["got"], out["v3_off"]["got"]
    gc = out["cur_off"]["got"]
    cv = plw.CONFIG_VERSION   # 3 when written (log 53), 4 since show_on_hotkey (log 54)
    check("C12_config_migration",
          g1["config_version"] == cv and g1["clear_on_return"] is True and g1["idle_unload_min"] == 0
          and g1["gpu_always"] and g1["autocopy"] is False and g1["x"] == 5 and out["v1"]["backups"] == 1
          and g2["config_version"] == cv and g2["clear_on_return"] is True and g2["autocopy"] is False
          and out["v2"]["backups"] == 1 and g3["clear_on_return"] is False and g3["config_version"] == cv
          and out["cur_off"]["msg"] is None and out["cur_off"]["backups"] == 0 and gc["clear_on_return"] is False, **{k: {"msg": v["msg"], "backups": v["backups"]} for k, v in out.items()})


def runner():
    try:
        body()
    except Exception:
        res["error"] = traceback.format_exc()
        p("ERROR", res["error"])
    finally:
        try:
            stop()
        except Exception:
            pass
        app._call(app.quit, wait=False)


threading.Thread(target=runner, daemon=True).start()
app.run()
res["real_clipboard_untouched"] = real_get_clip() == CLIP_BEFORE
res["fake_mic_starts"] = fake["mics"]
res["summary"] = results
res["sandbox_guard"] = sandbox_guard.summary()
res["all_ok"] = (bool(results) and all(results.values()) and res["real_clipboard_untouched"] and "error" not in res
                 and sandbox_guard.ok())
print("RESULT " + json.dumps(res, indent=1, default=str))
sys.exit(0 if res["all_ok"] else 1)
