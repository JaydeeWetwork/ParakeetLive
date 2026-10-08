"""Log 55 test helper: runs the REAL server/live_server.py request handling with the model load and inference
stubbed (no NeMo, no CUDA, no VRAM). Expects live_server.py next to this file (the test copies both to
/tmp/plive-test in WSL, a path that never matches the widget's own server, so its pkill rules can't hit it).
Arguments are passed through to live_server.main() (--port N --idle-exit S --gpu|--standby)."""
import importlib.util
import os
import sys
import types

sys.modules["transcribe"] = types.ModuleType("transcribe")     # the real one imports the model stack
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "live_server.py")
if not os.path.exists(src):                                     # run from the repo: tests/../server/
    src = os.path.join(os.path.dirname(here), "server", "live_server.py")
spec = importlib.util.spec_from_file_location("live_server", src)
m = importlib.util.module_from_spec(spec)
sys.argv[0] = src
spec.loader.exec_module(m)


def fake_load(args):
    m.STATE.update(status="standby" if args.standby else "ready", device="cpu", load_s=0.0)


m.load = fake_load
m.infer = lambda audio, path="auto": f"stub {len(audio)} samples"
m._fadvise = lambda *a, **k: None
m._weights_files = lambda: ["/dev/null"]
m.main()
