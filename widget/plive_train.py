"""Private, local-only training-data logger for future fine-tuning on Jaydee's voice.

training-data/
  audio/<id>.wav     16 kHz mono 16-bit PCM (~32 KB per second of audio)
  manifest.jsonl     one JSON object per utterance, NeMo-friendly:
                     audio_filepath (relative to the manifest), duration, text (= corrected, else raw),
                     raw_text, corrected_text, edited, id, timestamp, wav_path, ...

  manifest-edits.jsonl  corrections not yet folded into manifest.jsonl (log 57): {"edit": id, "fields": {...}}

Records are never deleted. Disk work happens on a background thread, so the UI never waits on disk.
A new record is one line appended to manifest.jsonl (O(1), log 57). A correction appends the record's new
values to manifest-edits.jsonl; the journal is folded into manifest.jsonl by an atomic rewrite (temp file +
fsync + replace) after 30 s without edits, every 100 edits, at quit, and at start if one was left behind.
Nothing here uploads or syncs anything.
"""
import json
import os
import queue
import threading
import time
import wave
from collections import OrderedDict

SR = 16000
EDIT_FIELDS = ("text", "corrected_text", "edited", "removed_in_box", "updated")
COMPACT_IDLE_S = 30.0                     # fold the edits journal in after this long without edits
COMPACT_EDITS = 100                       # ... or after this many journaled edits


class TrainingLog:
    def __init__(self, root_dir, model="nvidia/parakeet-tdt-0.6b-v2"):
        self.dir = os.path.abspath(root_dir)
        self.audio_dir = os.path.join(self.dir, "audio")
        self.manifest = os.path.join(self.dir, "manifest.jsonl")
        self.edits = os.path.join(self.dir, "manifest-edits.jsonl")
        self.orphans = os.path.join(self.dir, "manifest-edits-orphaned.jsonl")
        self.model = model
        self.records = OrderedDict()      # id -> record (existing + new)
        self.foreign = []                 # manifest lines we don't own (kept verbatim)
        self.q = queue.Queue()
        self.n = 0
        self.errors = []
        self._to_append = []              # new records not yet appended (retried after an error)
        self._to_journal = []             # ids with corrections not yet journaled
        self._journal_n = 0               # journaled edits not yet folded into the manifest
        self._orphan_lines = []           # journal lines for no known record: kept aside, never dropped
        self._compact_due = False
        self.stats = {"appends": 0, "journaled": 0, "compactions": 0}
        self._load()
        threading.Thread(target=self._run, name="training-log", daemon=True).start()

    def _load(self):
        if not os.path.exists(self.manifest):
            return
        # surrogateescape: a damaged line (invalid UTF-8) is kept byte-for-byte instead of crashing start-up
        with open(self.manifest, encoding="utf-8", errors="surrogateescape") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if isinstance(r, dict) and r.get("id"):
                        self.records[r["id"]] = r
                        continue
                except ValueError:
                    pass
                self.foreign.append(line)
        self._replay_journal()

    def _replay_journal(self):
        """A journal left by a crash or a kill: apply it (idempotent) and fold it in on the logger thread."""
        if not os.path.exists(self.edits):
            return
        with open(self.edits, encoding="utf-8", errors="surrogateescape") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    r = self.records.get(e.get("edit")) if isinstance(e, dict) else None
                    if r is not None and isinstance(e.get("fields"), dict):
                        r.update(e["fields"])
                        self._journal_n += 1
                        continue
                except (ValueError, AttributeError, TypeError):
                    pass
                self._orphan_lines.append(line)
        self._compact_due = True

    # ---- called from the UI thread (cheap; disk work happens on the logger thread)
    def add(self, pcm16: bytes, raw_text: str, extra=None):
        raw_text = (raw_text or "").strip()
        dur = len(pcm16) / 2 / SR
        if not raw_text or dur < 0.25:
            return None                   # skip empty / silent utterances
        self.n += 1
        t = time.time()
        rid = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t * 1000) % 1000:03d}-{self.n:04d}"
        rec = OrderedDict([
            ("audio_filepath", f"audio/{rid}.wav"), ("duration", round(dur, 3)), ("text", raw_text),
            ("raw_text", raw_text), ("corrected_text", raw_text), ("edited", False), ("id", rid),
            ("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))),
            ("wav_path", os.path.join(self.audio_dir, f"{rid}.wav")), ("sample_rate", SR), ("model", self.model)])
        if extra:
            rec.update(extra)
        self.q.put(("add", rid, pcm16, rec))
        return rid

    def update(self, rid, corrected_text):
        self.q.put(("upd", rid, corrected_text, None))

    def flush(self, timeout=5.0):
        ev = threading.Event()
        self.q.put(("flush", None, ev, None))
        return ev.wait(timeout)

    # ---- logger thread
    def _apply_update(self, rid, corrected):
        r = self.records.get(rid)
        if r is None:
            return False
        c = (corrected or "").strip()
        if not c:
            # utterance text was deleted from the box: keep the last correction, just note it
            if not r.get("removed_in_box"):
                r["removed_in_box"] = True
                return True
            return False
        if c == r.get("corrected_text") and not r.get("removed_in_box"):
            return False
        r["corrected_text"] = c
        r["text"] = c
        r["edited"] = c != r.get("raw_text")
        r["removed_in_box"] = False
        r["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return True

    def _write_manifest(self):
        os.makedirs(self.dir, exist_ok=True)
        tmp = self.manifest + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8", errors="surrogateescape", newline="\n") as f:
                for line in self.foreign:
                    f.write(line + "\n")
                for r in self.records.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())             # on disk before it replaces the old manifest (log 55)
        except Exception:
            try:
                os.remove(tmp)                   # e.g. disk full: the old manifest stays as it was
            except OSError:
                pass
            raise
        for i in range(20):                      # another app may briefly hold the file open
            try:
                os.replace(tmp, self.manifest)
                return True
            except PermissionError:
                time.sleep(0.1)
        return False

    def _append(self, path, data):
        """Append whole lines and fsync. A torn last line (crash mid-append) is closed with a newline first;
        a failed append is cut back to where it started, so a retry never leaves half a line behind."""
        os.makedirs(self.dir, exist_ok=True)
        with open(path, "a+b") as f:
            f.seek(0, 2)
            size = f.tell()
            if size:
                f.seek(size - 1)
                if f.read(1) != b"\n":
                    data = b"\n" + data
            try:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                try:
                    f.truncate(size)
                except OSError:
                    pass
                raise

    def _persist(self):
        if self._to_append:
            data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self._to_append)
            self._append(self.manifest, data.encode("utf-8", "surrogateescape"))
            self.stats["appends"] += len(self._to_append)
            self._to_append = []
        if self._to_journal:
            rows = []
            for rid in dict.fromkeys(self._to_journal):
                r = self.records.get(rid)
                if r is not None:
                    rows.append(json.dumps({"edit": rid, "fields": {k: r[k] for k in EDIT_FIELDS if k in r}},
                                           ensure_ascii=False) + "\n")
            if rows:
                self._append(self.edits, "".join(rows).encode("utf-8", "surrogateescape"))
            self._journal_n += len(rows)
            self.stats["journaled"] += len(rows)
            self._to_journal = []

    def _compact(self):
        """Fold the journal in: atomic full rewrite, then remove the journal (a crash in between only
        replays the same values on the next start)."""
        if not self._write_manifest():
            return False
        self._to_append, self._to_journal = [], []        # both are in the rewrite
        if self._orphan_lines:
            self._append(self.orphans, "".join(x + "\n" for x in self._orphan_lines)
                         .encode("utf-8", "surrogateescape"))
            self._orphan_lines = []
        try:
            os.remove(self.edits)
        except FileNotFoundError:
            pass
        self._journal_n = 0
        self._compact_due = False
        self.stats["compactions"] += 1
        return True

    def _error(self, msg):
        self.errors.append(msg)
        del self.errors[:-50]                    # bounded (a full disk retries every 0.5 s)

    def _run(self):
        waiters = []
        last_edit = time.monotonic()
        while True:
            retry = bool(self._to_append or self._to_journal or self._compact_due)
            if retry:
                timeout = 0.5
            elif self._journal_n:
                timeout = max(0.05, COMPACT_IDLE_S - (time.monotonic() - last_edit))
            else:
                timeout = None
            try:
                item = self.q.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is not None:
                kind, rid, a, b = item
                try:
                    if kind == "add":
                        os.makedirs(self.audio_dir, exist_ok=True)
                        with wave.open(b["wav_path"], "wb") as w:   # the audio first, then its record
                            w.setnchannels(1)
                            w.setsampwidth(2)
                            w.setframerate(SR)
                            w.writeframes(a)
                        self.records[rid] = b
                        self._to_append.append(b)
                    elif kind == "upd":
                        if self._apply_update(rid, a):
                            self._to_journal.append(rid)
                            last_edit = time.monotonic()
                    elif kind == "flush":
                        waiters.append(a)
                except Exception as e:
                    self._error(f"{kind} {rid}: {e!r}")
                if not self.q.empty():
                    continue                     # batch: drain the queue before touching the disk
            if self._to_append or self._to_journal:
                try:
                    self._persist()
                except Exception as e:
                    self._error(f"manifest: {e!r}")
            if (self._compact_due or waiters or self._journal_n >= COMPACT_EDITS
                    or (self._journal_n and time.monotonic() - last_edit >= COMPACT_IDLE_S)):
                if self._journal_n or self._compact_due or self._orphan_lines or self._to_append:
                    try:
                        if not self._compact():
                            self._compact_due = True
                    except Exception as e:
                        self._compact_due = True
                        self._error(f"compact: {e!r}")
            for ev in waiters:
                ev.set()
            waiters = []
