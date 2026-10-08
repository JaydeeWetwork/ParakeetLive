"""Log 55: unit tests for plive_core / plive_train fixes (pure logic; runs on Windows and, with the Win32 DLLs
stubbed, anywhere). No microphone, no network beyond 127.0.0.1, no GUI.
  U1 utterance ids unique across recordings; a stale speculative result can no longer be shown for new audio
  U2 training manifest: a line with invalid UTF-8 is kept byte-for-byte (start-up used to crash), the error
     list is capped, a failed write leaves the old manifest intact and no .tmp behind
  U3 Client sends the server token; ServerError carries the HTTP status; retryable() tells 5xx from 4xx
  U4 microphone stall watchdog: a stream that stays "active" but stops delivering audio -> error, aborted
  U5 (log 57) session store: the unsent message and kept retry audio round-trip; an empty box removes the
     draft; a week-old draft / an hour-old retry is not restored and is removed; broken or unfinished files
     are cleaned up; no temp files left; retries come back oldest first
  U6 (log 57) training manifest append-only: adding records to a 10,000-record manifest appends lines and
     never rewrites it; corrections go to the journal and survive a crash (replayed on start, then folded
     in); a crash between the fold and the journal removal replays to the same result; a torn last line
     is closed off, a failed append is cut back (no half lines, no duplicates); journal lines for unknown
     records are kept aside; idle / count / quit compaction; foreign lines kept"""
import ctypes
import os
import sys
import tempfile
import time
from unittest import mock

if not hasattr(ctypes, "WinDLL"):
    ctypes.WinDLL = mock.MagicMock()
    ctypes.WINFUNCTYPE = mock.MagicMock()
    import ctypes.wintypes  # noqa: F401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "widget"))
import itertools  # noqa: E402

import numpy as np  # noqa: E402

import plive_core as core  # noqa: E402
import plive_train as train  # noqa: E402

RESULTS = []


def check(name, ok, **info):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {info if info else ''}")


SR, F = core.SR, core.FRAME


def tone(s, amp=0.2):
    t = np.arange(int(s * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(s):
    return (np.random.default_rng(1).standard_normal(int(s * SR)) * 1e-4).astype(np.float32)


def record(chunks, stop_early=True):
    """Feed one 'recording' through a fresh Segmenter (like start_recording does); returns the emitted jobs."""
    jobs = []
    seg = core.Segmenter(lambda pcm, t, dur, kind, uid, ver: jobs.append((kind, uid, ver, pcm)), silence_s=0.5)
    seg.feed(quiet(0.5))
    for c in chunks:
        seg.feed(c)
    seg.flush()                       # hotkey stop
    return jobs


def run_jobs(jobs, cache, old_style=False, label=""):
    """What _job_loop does with speculative results. Returns [(uid, ver, text)] shown for finals."""
    shown = []
    for kind, uid, ver, pcm in jobs:
        fake_result = {"text": f"{label}:{uid}:{ver}"}
        if kind == "spec":
            if old_style:
                cache["item"] = (uid, ver, fake_result)
            else:
                cache.put(uid, ver, fake_result, 1.0)
            continue
        if old_style:
            it = cache.get("item")
            hit = it[2] if it and it[0] == uid and it[1] == ver else None
            if hit is not None:
                cache["item"] = None
        else:
            h = cache.take_for_final(uid, ver)
            hit = h[0] if h else None
        shown.append((uid, ver, (hit or {"text": f"{label}:{uid}:{ver}"})["text"]))
    return shown


# A: speech, a 0.25 s pause (speculative send), more speech, hotkey stop mid-speech (final without its own spec)
rec_a = [tone(1.0), quiet(0.25), tone(0.6)]
# B: the same 1 s of speech, hotkey stop 0.1 s later (before any spec) -> same frame index as A's spec
rec_b = [tone(1.0), quiet(0.1)]

# old behaviour: ids restart with every recording, cache survives a non-matching final
orig = core._UTT_IDS
core._UTT_IDS = itertools.count(1)
ja = record(rec_a)
core._UTT_IDS = itertools.count(1)
jb = record(rec_b)
old_cache = {}
run_jobs(ja, old_cache, old_style=True, label="A")
old_b = run_jobs(jb, old_cache, old_style=True, label="B")
old_collides = any(t.startswith("A:") for _, _, t in old_b)
core._UTT_IDS = orig

# new behaviour
ja2, jb2 = record(rec_a), record(rec_b)
cache = core.SpecCache()
run_jobs(ja2, cache, label="A")
new_b = run_jobs(jb2, cache, label="B")
uids_a = {u for _, u, _, _ in ja2}
uids_b = {u for _, u, _, _ in jb2}
check("U1_spec_from_earlier_recording_never_reused",
      old_collides and not any(t.startswith("A:") for _, _, t in new_b) and not (uids_a & uids_b),
      old_behaviour_showed=[t for _, _, t in old_b], new_shows=[t for _, _, t in new_b],
      uids_a=sorted(uids_a), uids_b=sorted(uids_b))
c = core.SpecCache()
c.put(5, 40, {"text": "x"}, 2.0)
same = c.take_for_final(5, 40)
c.put(5, 40, {"text": "x"}, 2.0)
other = c.take_for_final(6, 40)
check("U1b_speccache_matches_only_its_final_and_any_final_clears",
      same == ({"text": "x"}, 2.0) and other is None and c.take_for_final(5, 40) is None)

# U2 training manifest durability
d = tempfile.mkdtemp(prefix="plive-u2-")
good = '{"id": "20260101-000000-000-0001", "text": "ok", "audio_filepath": "audio/x.wav"}'
damaged = b'{"note": "foreign line with a bad byte \xff here"}'
with open(os.path.join(d, "manifest.jsonl"), "wb") as f:
    f.write(good.encode() + b"\n" + damaged + b"\n")
try:
    st = train.TrainingLog(d)
    started = True
except Exception as e:
    started, st = repr(e), None
if st:
    rid = st.add(bytes(16000), "hello there")
    st.flush(5)
    raw = open(os.path.join(d, "manifest.jsonl"), "rb").read()
    kept = damaged in raw and good.encode()[:30] in raw and rid.encode() in raw
    for i in range(120):
        st._error(f"e{i}")
    capped = len(st.errors) == 50 and st.errors[-1] == "e119"
    before = open(os.path.join(d, "manifest.jsonl"), "rb").read()
    real_dumps = train.json.dumps
    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    train.json.dumps = boom
    try:
        st._write_manifest()
        failed = False
    except OSError:
        failed = True
    finally:
        train.json.dumps = real_dumps
    intact = open(os.path.join(d, "manifest.jsonl"), "rb").read() == before
    no_tmp = not os.path.exists(os.path.join(d, "manifest.jsonl.tmp"))
    check("U2_manifest_damaged_bytes_kept_errors_capped_failed_write_safe",
          started is True and kept and capped and failed and intact and no_tmp,
          started=started, kept=kept, capped=capped, failed=failed, intact=intact, no_tmp=no_tmp)
else:
    check("U2_manifest_damaged_bytes_kept_errors_capped_failed_write_safe", False, started=started)

# U3 token header + error classification (no network: the connection is faked)
class FakeResp:
    status = 401
    def read(self):
        return b'{"error": "unauthorized"}'
class FakeConn:
    sock = None
    def __init__(self):
        self.sent = None
    def request(self, method, path, body=None, headers=None):
        self.sent = (method, path, dict(headers or {}))
    def getresponse(self):
        return FakeResp()
    def close(self):
        pass
c = core.Client(1, token="abc123")
c.conn = FakeConn()
try:
    c.transcribe(b"\0\0")
    err = None
except core.ServerError as e:
    err = e
sent = c.conn.sent if c.conn else None
c2 = core.Client(1)
c2.conn = FakeConn()
try:
    c2.transcribe(b"\0\0")
except core.ServerError:
    pass
check("U3_token_header_and_error_status",
      sent and sent[2].get("X-Plive-Token") == "abc123" and "X-Plive-Token" not in c2.conn.sent[2]
      and err is not None and err.status == 401 and str(err) == "unauthorized" and not core.retryable(err)
      and core.retryable(core.ServerError("x", 500)) and core.retryable(ConnectionResetError()),
      sent_headers=sorted(sent[2]) if sent else None)

# U4 mic stall watchdog (fake stream, no audio device)
import threading  # noqa: E402
class FakeStream:
    active = True
    def __init__(self):
        self.calls = []
    def stop(self):
        self.calls.append("stop")
    def abort(self):
        self.calls.append("abort")
    def close(self):
        self.calls.append("close")
errs, got = [], []
m = core.MicCapture(None, lambda y: got.append(len(y)), lambda msg: errs.append((time.perf_counter(), msg)))
m.stream, m.ch, m.rs, m.stall_s = FakeStream(), 1, core.StreamResampler(16000), 1.0
fs = m.stream
m.worker = threading.Thread(target=m._run, daemon=True)
m.worker.start()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.5:          # healthy: blocks every 20 ms -> no error
    m.q.put(np.zeros((320, 1), np.float32))
    time.sleep(0.02)
healthy = not errs
t_stop = time.perf_counter()                   # the device goes silent (stream still "active")
m.worker.join(4)
m.stop()
fired = bool(errs) and 0.9 < errs[0][0] - t_stop < 2.5
check("U4_mic_stall_detected_and_aborted", healthy and fired and m.stalled and fs.calls[:1] == ["abort"] and sum(got) > 0,
      healthy=healthy, error_after_s=round(errs[0][0] - t_stop, 2) if errs else None, calls=fs.calls)

# ---- U5 session store (log 57)
import json  # noqa: E402
import shutil  # noqa: E402
SD = os.path.join(tempfile.gettempdir(), "plive-u5-state")
shutil.rmtree(SD, ignore_errors=True)
st = core.SessionStore(SD)
d0 = {"text": "Draft words, with \u00e9 and \u2014 dash.", "spans": [["r1", 0, 6, "Draft "]],
      "cor": {"copied": True, "armed": True, "left": False}}
st.save_draft(d0)
d1 = st.load_draft()
round_trip = d1 is not None and all(d1[k] == d0[k] for k in d0)
st.save_draft({"text": "   ", "spans": [], "cor": {}})
empty_removes = not os.path.exists(st.draft_path) and st.load_draft() is None
st.save_draft(d0)
dd = json.load(open(st.draft_path, encoding="utf-8"))
dd["saved"] = time.time() - 8 * 86400
json.dump(dd, open(st.draft_path, "w", encoding="utf-8"))
old_draft = st.load_draft() is None and not os.path.exists(st.draft_path)
open(st.draft_path, "w").write("{not json")
broken_draft = st.load_draft() is None
pcm_a, pcm_b = bytes(range(256)) * 50, bytes(range(255, -1, -1)) * 30
now = time.time()
st.save_pending("b-newer", pcm_b, {"dur": 0.4, "tries": 2, "failed_at": now - 60})
st.save_pending("a-older", pcm_a, {"dur": 0.8, "tries": 1, "failed_at": now - 600})
st.save_pending("c-stale", pcm_a, {"dur": 0.8, "tries": 1, "failed_at": now - 7200})
open(os.path.join(st.pending_dir, "d-broken.json"), "w").write("{")
open(os.path.join(st.pending_dir, "d-broken.wav"), "wb").write(b"junk")
for n, age in (("e-lone.wav", 7200), ("f-fresh-lone.wav", 10), ("g.json.tmp", 7200)):
    q = os.path.join(st.pending_dir, n)
    open(q, "wb").write(b"x")
    os.utime(q, (now - age, now - age))
items = st.load_pending(3600)
left = sorted(os.listdir(st.pending_dir))
order_ok = [k for k, _, _ in items] == ["a-older", "b-newer"]
audio_ok = len(items) == 2 and items[0][1] == pcm_a and items[1][1] == pcm_b and items[1][2]["tries"] == 2
cleaned = left == ["a-older.json", "a-older.wav", "b-newer.json", "b-newer.wav", "f-fresh-lone.wav"]
st.drop_pending("a-older")
dropped = not any(n.startswith("a-older") for n in os.listdir(st.pending_dir))
no_tmp = not any(n.endswith(".tmp") for r, _, fs in os.walk(SD) for n in fs)
check("U5_session_store", round_trip and empty_removes and old_draft and broken_draft and order_ok and audio_ok
      and cleaned and dropped and no_tmp, round_trip=round_trip, empty_removes=empty_removes, old_draft=old_draft,
      broken_draft=broken_draft, order=order_ok, audio=audio_ok, left=left, dropped=dropped, no_tmp=no_tmp)
shutil.rmtree(SD, ignore_errors=True)

# ---- U6 training manifest: append-only adds, edits journal, compaction (log 57)
def until(cond, t=5.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < t:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def lines_of(path):
    if not os.path.exists(path):
        return []
    return [ln for ln in open(path, encoding="utf-8", errors="surrogateescape").read().split("\n") if ln]


TD = os.path.join(tempfile.gettempdir(), "plive-u6-train")
shutil.rmtree(TD, ignore_errors=True)
os.makedirs(TD)
MAN = os.path.join(TD, "manifest.jsonl")
FOREIGN = '{"note": "a line that is not a record"}'
with open(MAN, "w", encoding="utf-8", newline="\n") as f:
    f.write(FOREIGN + "\n")
    for i in range(10000):
        f.write(json.dumps({"id": f"syn-{i:05d}", "audio_filepath": f"audio/syn-{i:05d}.wav", "duration": 1.0,
                            "text": f"synthetic record {i}", "raw_text": f"synthetic record {i}",
                            "corrected_text": f"synthetic record {i}", "edited": False}) + "\n")
size0 = os.path.getsize(MAN)
tl = train.TrainingLog(TD)
rewrites = []
real_wm = tl._write_manifest
tl._write_manifest = lambda: (rewrites.append(1), real_wm())[1]
t0 = time.perf_counter()
rids = [tl.add(bytes(8000), f"new words {k}") for k in range(50)]
appended = until(lambda: tl.stats["appends"] == 50)
t_add = time.perf_counter() - t0
man_lines = lines_of(MAN)
grew_only = (os.path.getsize(MAN) > size0 and not rewrites and len(man_lines) == 10051
             and json.loads(man_lines[-1])["id"] == rids[-1] and man_lines[0] == FOREIGN)
t1 = time.perf_counter()
real_wm()
t_rewrite = time.perf_counter() - t1
# a correction -> journal only; then a "crash" (no flush, a second logger reads the same folder)
tl.update(rids[3], "new words three, corrected")
tl.update("syn-00007", "synthetic record seven, corrected")
journaled = until(lambda: tl.stats["journaled"] >= 2 and len(lines_of(tl.edits)) == 2)
man_untouched = not rewrites
before_crash = {k: dict(v) for k, v in tl.records.items()}
tl2 = train.TrainingLog(TD)
replayed = (tl2.records[rids[3]]["corrected_text"] == "new words three, corrected"
            and tl2.records["syn-00007"]["text"] == "synthetic record seven, corrected"
            and tl2.records["syn-00007"]["edited"] is True)
folded = until(lambda: not os.path.exists(tl2.edits))
on_disk = {json.loads(x)["id"]: json.loads(x) for x in lines_of(MAN) if x != FOREIGN}
folded_ok = (folded and on_disk[rids[3]]["corrected_text"] == "new words three, corrected"
             and on_disk["syn-00007"]["corrected_text"] == "synthetic record seven, corrected"
             and len(on_disk) == 10050 and lines_of(MAN)[0] == FOREIGN
             and {k: dict(v) for k, v in tl2.records.items()} == before_crash)
# a crash between the rewrite and the journal removal: the same journal is replayed over the folded manifest
with open(tl2.edits, "w", encoding="utf-8") as f:
    f.write(json.dumps({"edit": "syn-00007", "fields": {k: tl2.records["syn-00007"][k] for k in train.EDIT_FIELDS
                                                         if k in tl2.records["syn-00007"]}}) + "\n")
    f.write('{"edit": "no-such-record", "fields": {"text": "x"}}\n')
    f.write('{"edit": "syn-00009", "fie')                       # torn journal line
tl3 = train.TrainingLog(TD)
idem = ({k: dict(v) for k, v in tl3.records.items()} == before_crash and until(lambda: not os.path.exists(tl3.edits)))
orphans_kept = len(lines_of(tl3.orphans)) == 2
# a torn last manifest line (crash mid-append), then a failed append that must be cut back
with open(MAN, "ab") as f:
    f.write(b'{"id": "torn-rec", "text": "half')
real_fsync = os.fsync
fails = {"n": 1}
def flaky_fsync(fd):
    if fails["n"]:
        fails["n"] -= 1
        raise OSError(28, "No space left on device")
    return real_fsync(fd)
tl4 = train.TrainingLog(TD)
size_torn = os.path.getsize(MAN)
train.os.fsync = flaky_fsync
try:
    r_new = tl4.add(bytes(8000), "after the torn line")
    cut_back = until(lambda: tl4.errors and fails["n"] == 0, 3) and os.path.getsize(MAN) == size_torn
    retried = until(lambda: tl4.stats["appends"] == 1, 3)
finally:
    train.os.fsync = real_fsync
ml = lines_of(MAN)
torn_ok = (cut_back and retried and ml[-2] == '{"id": "torn-rec", "text": "half'
           and json.loads(ml[-1])["id"] == r_new and sum(1 for x in ml if r_new in x) == 1)
# compaction triggers: idle, count, flush
train.COMPACT_IDLE_S, train.COMPACT_EDITS = 0.6, 5
tl5 = train.TrainingLog(TD)
tl5.update(r_new, "after the torn line, edited")
idle_fold = until(lambda: tl5.stats["journaled"] == 1) and until(lambda: tl5.stats["compactions"] == 1, 3)
train.COMPACT_IDLE_S = 30.0
for k in range(5):
    tl5.update("syn-0001" + str(k), f"count edit {k}")
count_fold = until(lambda: tl5.stats["compactions"] == 2, 3)
tl5.update("syn-00020", "flush edit")
tl5.flush(5)
flush_fold = tl5.stats["compactions"] == 3 and not os.path.exists(tl5.edits) and \
    any('"flush edit"' in x for x in lines_of(MAN))
train.COMPACT_EDITS = 100
no_tmp6 = not os.path.exists(MAN + ".tmp")
check("U6_manifest_append_only_journal_crash_safe",
      appended and grew_only and journaled and man_untouched and replayed and folded_ok and idem and orphans_kept
      and torn_ok and idle_fold and count_fold and flush_fold and no_tmp6,
      add50_s=round(t_add, 3), full_rewrite_10k_s=round(t_rewrite, 3), grew_only=grew_only, journaled=journaled,
      replayed=replayed, folded=folded_ok, idempotent=idem, orphans=orphans_kept, torn=torn_ok,
      idle=idle_fold, count=count_fold, flush=flush_fold, no_tmp=no_tmp6)
shutil.rmtree(TD, ignore_errors=True)

if __name__ == "__main__":
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)
