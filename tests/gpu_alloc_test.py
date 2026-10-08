"""Simulated heavy GPU app for the auto-park tests (Windows, stdlib ctypes only, no admin).
Creates a D3D11 device on the NVIDIA adapter and allocates N MB of DEFAULT-usage buffers with initial data
(so they are really resident in dedicated VRAM), then waits until --seconds pass or the stop file appears.
  python gpu_alloc_test.py --mb 600 --seconds 120 [--stop-file path] [--chunk-mb 100]"""
import argparse, ctypes, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "widget"))
import plive_gpu as g  # noqa: E402


class BUFDESC(ctypes.Structure):
    _fields_ = [("ByteWidth", ctypes.c_uint), ("Usage", ctypes.c_uint), ("BindFlags", ctypes.c_uint),
                ("CPUAccessFlags", ctypes.c_uint), ("MiscFlags", ctypes.c_uint), ("StructureByteStride", ctypes.c_uint)]


class SUBDATA(ctypes.Structure):
    _fields_ = [("pSysMem", ctypes.c_void_p), ("SysMemPitch", ctypes.c_uint), ("SysMemSlicePitch", ctypes.c_uint)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=600)
    ap.add_argument("--chunk-mb", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--stop-file", default=None)
    a = ap.parse_args()
    adapter = None
    for ad, d in g.enum_adapters():
        if adapter is None and d.VendorId == 0x10DE:
            adapter = ad
        else:
            g.release(ad)
    if adapter is None:
        sys.exit("no NVIDIA adapter")
    d3d = ctypes.WinDLL("d3d11")
    dev, ctx, fl = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_uint()
    hr = d3d.D3D11CreateDevice(adapter, 0, None, 0, None, 0, 7, ctypes.byref(dev), ctypes.byref(fl), ctypes.byref(ctx))
    if hr != 0:
        sys.exit(f"D3D11CreateDevice 0x{hr & 0xFFFFFFFF:08x}")
    create = g._vcall(dev, 3, ctypes.c_long, ctypes.POINTER(BUFDESC), ctypes.POINTER(SUBDATA), ctypes.POINTER(ctypes.c_void_p))
    chunk = a.chunk_mb * 2**20
    init = ctypes.create_string_buffer(b"\x5a" * 16, chunk)      # non-zero pattern, uploaded at creation
    bufs, done = [], 0
    while done < a.mb:
        desc = BUFDESC(chunk, 0, 0x8, 0, 0, 0)                  # DEFAULT usage, SHADER_RESOURCE
        sub = SUBDATA(ctypes.cast(init, ctypes.c_void_p), 0, 0)
        b = ctypes.c_void_p()
        hr = create(dev, ctypes.byref(desc), ctypes.byref(sub), ctypes.byref(b))
        if hr != 0:
            print(f"CreateBuffer failed at {done} MB: 0x{hr & 0xFFFFFFFF:08x}", flush=True)
            break
        bufs.append(b); done += a.chunk_mb
    print(f"pid {os.getpid()} allocated {done} MB on NVIDIA (feature level 0x{fl.value:x}); holding", flush=True)
    t0 = time.time()
    while time.time() - t0 < a.seconds and not (a.stop_file and os.path.exists(a.stop_file)):
        time.sleep(0.5)
    for b in bufs:
        g.release(b)
    g.release(ctx); g.release(dev); g.release(adapter)
    print("released", flush=True)


if __name__ == "__main__":
    main()
