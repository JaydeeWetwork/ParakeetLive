"""Send test clips through the widget's own client path (raw 16 kHz PCM, keep-alive HTTP)."""
import os, sys, time, wave, statistics, json, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "widget"))
import plive_core as core

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()

c = core.Client(int(os.environ.get("PLIVE_PORT", "51761")))
print("health:", json.dumps(c.health()))
for wav, ref in [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]:
    with wave.open(wav, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())
    reftxt = open(ref, encoding="utf-8").read().strip()
    rtts, infs = [], []
    for i in range(8):
        t0 = time.perf_counter()
        r = c.transcribe(pcm)
        rtts.append((time.perf_counter() - t0) * 1000); infs.append(r["infer_ms"])
    st, ss = time.perf_counter(), None
    rs = c._request("POST", "/transcribe?path=slow", body=pcm)[1]
    print(f"  slow path (NeMo transcribe) text same as fast path: {rs['text'] == r['text']} (infer {rs['infer_ms']} ms)")
    print("  path used:", r.get("path"))
    a, b = norm(r["text"]), norm(reftxt)
    print(f"\n{os.path.basename(wav)}: {r['audio_s']} s audio")
    print("  text :", r["text"])
    print("  ref  :", reftxt)
    print("  words match reference:", a == b)
    print(f"  server infer ms: median {statistics.median(infs):.0f}, worst {max(infs)}; all {infs}")
    print(f"  round trip ms  : median {statistics.median(rtts):.0f}, worst {max(rtts):.0f} (first {rtts[0]:.0f})")
# short utterance like live speech (first 2 s)
r = c.transcribe(pcm[:16000*2*2]); print("\n2 s slice:", r)
print("health after:", json.dumps(c.health()))
