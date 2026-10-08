"""Test 41: minimize button, Esc, Win+D / Win+Down -> hide to tray; topmost + no taskbar button across cycles."""
import ctypes, subprocess, sys, time, os
from ctypes import wintypes as wt
W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "widget")
APP = os.path.join(W, "parakeet_live.pyw"); PY = os.path.join(W, r".venv\Scripts\python.exe")
ctypes.windll.shcore.SetProcessDpiAwareness(1)
u = ctypes.windll.user32
u.GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
EnumProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
def cmd(c): subprocess.run([PY, APP, "--cmd", c]); time.sleep(0.8)
def find():
    out = []
    def cb(h, l):
        n = ctypes.create_unicode_buffer(256); u.GetWindowTextW(h, n, 256)
        c = ctypes.create_unicode_buffer(256); u.GetClassNameW(h, c, 256)
        if n.value == "Parakeet Live" and c.value.startswith("Tk"): out.append(h)
        return True
    u.EnumWindows(EnumProc(cb), 0); return out
def st(tag):
    hs = find(); r = []
    for h in hs:
        ex = u.GetWindowLongW(h, -20); rc = wt.RECT(); u.GetWindowRect(h, ctypes.byref(rc))
        # taskbar button rule: visible, unowned, and (APPWINDOW or not TOOLWINDOW)
        tb = bool(u.IsWindowVisible(h)) and not u.GetWindow(h, 4) and (bool(ex & 0x40000) or not ex & 0x80)
        r.append(dict(hwnd=h, visible=bool(u.IsWindowVisible(h)), iconic=bool(u.IsIconic(h)), toolwindow=bool(ex & 0x80),
                      topmost=bool(ex & 0x8), taskbar_button=tb, rect=(rc.left, rc.top, rc.right, rc.bottom)))
    print(f"{time.strftime('%H:%M:%S')} {tag}: {r}", flush=True); return r[0] if r else {}
def key(*vks):
    for v in vks: u.keybd_event(v, 0, 0, 0); time.sleep(0.03)
    for v in reversed(vks): u.keybd_event(v, 0, 2, 0); time.sleep(0.03)
def click(x, y):
    u.SetCursorPos(x, y); time.sleep(0.15); u.mouse_event(2, 0, 0, 0, 0); time.sleep(0.05); u.mouse_event(4, 0, 0, 0, 0)
def show(tag):
    cmd("show"); time.sleep(0.8); return st("shown before " + tag)
res = {}
subprocess.Popen([PY, APP, "--tray", "--no-save"]); time.sleep(5)
st("tray-only start")
s = show("button"); x0, y0, x1, y1 = s["rect"]
sc = 1.0  # args are physical pixels (process is DPI aware)
pos = (x1 - int(int(sys.argv[1]) * sc) if len(sys.argv) > 1 else x1 - int(46 * sc), y0 + int(int(sys.argv[2]) * sc) if len(sys.argv) > 2 else y0 + int(21 * sc))
from PIL import ImageGrab
ImageGrab.grab(bbox=(x1 - 200, y0, x1, y0 + 60), all_screens=True).save(os.path.join(W, "screenshot-toolbar.png"))
print("click at", pos, "scale", sc)
click(*pos); time.sleep(0.8); res["button"] = st("after minimize button")
s = show("esc"); click(x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2); time.sleep(0.3); key(0x1B); time.sleep(0.8); res["esc"] = st("after Esc")
s = show("win+down"); click(x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2); time.sleep(0.3); key(0x5B, 0x28); time.sleep(1.2); res["win_down"] = st("after Win+Down")
s = show("win+d"); key(0x5B, 0x44); time.sleep(1.5); res["win_d"] = st("after Win+D"); key(0x5B, 0x44); time.sleep(1.0)
res["final_show"] = show("end"); cmd("hide"); res["final_hide"] = st("after --cmd hide")
print("RESULT", res, flush=True)

