"""Guard for the sandboxed in-process tests (test-clear-on-return, -show-on-hotkey, -code-review,
-display-change, -restart-survival). Added in log 57 after a sandbox window caught one of Jaydee's clicks:
on Oct 8, 2026 at ~06:06 the clear-on-return sandbox sat at 2 % opacity, always on top, at the default
spot (bottom centre of the laptop screen); his left click on YouTube's "Skip ad" landed on its gear icon
and the gear menu (a separate Tk window, full opacity) popped up.

With install(plw) before the App is created:
  - every sandbox window is click-through (WS_EX_LAYERED | WS_EX_TRANSPARENT, re-applied after each style
    change, since Tk's -alpha rewrites the ex-style): clicks go to whatever is under it
  - Tk menus never open in a sandbox (tk_popup / post are counted instead of shown)
  - after every show, WindowFromPoint at 9 points of the window must not hit this process; if it would,
    the window is withdrawn at once and ok() is False (the test fails)
"""
import ctypes
import ctypes.wintypes as wt
import os
import tkinter

WS_EX_LAYERED, WS_EX_TRANSPARENT, GWL_EXSTYLE = 0x80000, 0x20, -20
u32 = ctypes.windll.user32
u32.WindowFromPoint.argtypes = [wt.POINT]
u32.WindowFromPoint.restype = wt.HWND
u32.GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
u32.GetWindowLongW.restype = wt.LONG
u32.SetWindowLongW.argtypes = [wt.HWND, ctypes.c_int, wt.LONG]
u32.SetWindowLongW.restype = wt.LONG
u32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
u32.SetLayeredWindowAttributes.argtypes = [wt.HWND, wt.DWORD, ctypes.c_ubyte, wt.DWORD]
STATE = {"menus_blocked": 0, "shows_checked": 0, "violations": []}
ORIGINALS = {}                          # the App's own methods, for tests that check them without the guard


def click_through(hwnd):
    st = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if not st & WS_EX_LAYERED:          # (sandboxes run at 2 % opacity, so Tk has made it layered already)
        u32.SetWindowLongW(hwnd, GWL_EXSTYLE, st | WS_EX_LAYERED)
        u32.SetLayeredWindowAttributes(hwnd, 0, 5, 2)          # LWA_ALPHA, ~2 %
        st = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if not st & WS_EX_TRANSPARENT:
        u32.SetWindowLongW(hwnd, GWL_EXSTYLE, st | WS_EX_TRANSPARENT)


def clickable_points(hwnd):
    """Points of the window where a click would reach this process (should be none)."""
    r = wt.RECT()
    u32.GetWindowRect(hwnd, ctypes.byref(r))
    bad, pid = [], wt.DWORD()
    for fx in (0.05, 0.5, 0.95):
        for fy in (0.12, 0.5, 0.88):
            x, y = int(r.left + fx * (r.right - r.left)), int(r.top + fy * (r.bottom - r.top))
            h = u32.WindowFromPoint(wt.POINT(x, y))
            if h:
                u32.GetWindowThreadProcessId(h, ctypes.byref(pid))
                if pid.value == os.getpid():
                    bad.append((x, y))
    return bad


def _no_menu(self, *a, **k):
    STATE["menus_blocked"] += 1


def install(plw):
    tkinter.Menu.tk_popup = _no_menu
    tkinter.Menu.post = _no_menu
    orig_style, orig_show = plw.App._apply_style, plw.App.show
    ORIGINALS.update(_apply_style=orig_style, show=orig_show)

    def style(self):
        orig_style(self)
        h = self._hwnd()
        if h:
            click_through(h)

    def show(self, *a, **k):
        r = orig_show(self, *a, **k)
        h = self._hwnd()
        if h and self.visible:
            click_through(h)
            self.root.update_idletasks()
            STATE["shows_checked"] += 1
            bad = clickable_points(h)
            if (u32.GetWindowLongW(h, GWL_EXSTYLE) & (WS_EX_LAYERED | WS_EX_TRANSPARENT)) != \
                    (WS_EX_LAYERED | WS_EX_TRANSPARENT):
                bad.append("click-through style missing")
            if bad:
                self.root.withdraw()
                self.visible = False
                STATE["violations"].append(bad)
        return r

    plw.App._apply_style = style
    plw.App.show = show


def ok():
    return not STATE["violations"]


def summary():
    return dict(STATE, ok=ok())
