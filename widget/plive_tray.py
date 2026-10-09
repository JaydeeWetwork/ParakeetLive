"""Notification-area (tray) icon, global hotkey and single-instance IPC - pure ctypes/Win32, no packages.

Runs its own Win32 message loop on a background thread with a hidden top-level window
(class "ParakeetLiveTray"). A second launch finds that window and posts a command to it.
Callbacks run on the tray thread; the GUI hands them to Tk through a queue.
"""
import ctypes
import ctypes.wintypes as wt
import threading

user32 = ctypes.WinDLL("user32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
CLASS_NAME = "ParakeetLiveTray"

WM_DESTROY, WM_CLOSE, WM_NULL, WM_HOTKEY = 0x0002, 0x0010, 0x0000, 0x0312
WM_DISPLAYCHANGE = 0x007E
WM_LBUTTONUP, WM_RBUTTONUP, WM_LBUTTONDBLCLK, WM_CONTEXTMENU = 0x0202, 0x0205, 0x0203, 0x007B
WM_APP = 0x8000
WM_TRAY = WM_APP + 1          # tray icon callback
WM_IPC = WM_APP + 2           # command from a second instance (wParam = command code)
WM_SETTIP = WM_APP + 3        # internal: refresh icon/tooltip on the tray thread
NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP, NIF_SHOWTIP = 0x1, 0x2, 0x4, 0x80
MF_STRING, MF_GRAYED, MF_CHECKED, MF_POPUP, MF_SEPARATOR = 0x0, 0x1, 0x8, 0x10, 0x800
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_NONOTIFY = 0x2, 0x100, 0x80
IMAGE_ICON, LR_LOADFROMFILE = 1, 0x10

IPC_CODES = {"show": 1, "hide": 2, "toggle": 3, "load": 4, "unload": 5, "record": 6, "quit": 7, "dump": 8,
             "prewarm": 9, "standby": 10, "pausepark": 11, "silentrec": 12, "batchhold": 13, "batchrelease": 14,
             "quitall": 15}
IPC_NAMES = {v: k for k, v in IPC_CODES.items()}


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("hWnd", wt.HWND), ("uID", wt.UINT), ("uFlags", wt.UINT),
                ("uCallbackMessage", wt.UINT), ("hIcon", wt.HICON), ("szTip", wt.WCHAR * 128),
                ("dwState", wt.DWORD), ("dwStateMask", wt.DWORD), ("szInfo", wt.WCHAR * 256),
                ("uVersion", wt.UINT), ("szInfoTitle", wt.WCHAR * 64), ("dwInfoFlags", wt.DWORD),
                ("guidItem", GUID), ("hBalloonIcon", wt.HICON)]


class NOTIFYICONIDENTIFIER(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("hWnd", wt.HWND), ("uID", wt.UINT), ("guidItem", GUID)]


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("style", wt.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wt.HINSTANCE), ("hIcon", wt.HICON),
                ("hCursor", wt.HANDLE), ("hbrBackground", wt.HBRUSH), ("lpszMenuName", wt.LPCWSTR),
                ("lpszClassName", wt.LPCWSTR), ("hIconSm", wt.HICON)]


user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = wt.ATOM
user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID]
user32.CreateWindowExW.restype = wt.HWND
user32.FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
user32.FindWindowW.restype = wt.HWND
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.LoadImageW.argtypes = [wt.HINSTANCE, wt.LPCWSTR, wt.UINT, ctypes.c_int, ctypes.c_int, wt.UINT]
user32.LoadImageW.restype = wt.HANDLE
user32.CreatePopupMenu.restype = wt.HMENU
user32.AppendMenuW.argtypes = [wt.HMENU, wt.UINT, ctypes.c_size_t, wt.LPCWSTR]
user32.TrackPopupMenu.argtypes = [wt.HMENU, wt.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.HWND, wt.LPVOID]
user32.TrackPopupMenu.restype = wt.BOOL
user32.SetMenuDefaultItem.argtypes = [wt.HMENU, wt.UINT, wt.UINT]
user32.DestroyMenu.argtypes = [wt.HMENU]
user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.DestroyWindow.argtypes = [wt.HWND]
user32.RegisterHotKey.argtypes = [wt.HWND, ctypes.c_int, wt.UINT, wt.UINT]
user32.UnregisterHotKey.argtypes = [wt.HWND, ctypes.c_int]
user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.RegisterWindowMessageW.argtypes = [wt.LPCWSTR]
user32.RegisterWindowMessageW.restype = wt.UINT
shell32.Shell_NotifyIconW.argtypes = [wt.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
shell32.Shell_NotifyIconW.restype = wt.BOOL
shell32.Shell_NotifyIconGetRect.argtypes = [ctypes.POINTER(NOTIFYICONIDENTIFIER), ctypes.POINTER(wt.RECT)]
shell32.Shell_NotifyIconGetRect.restype = ctypes.c_long
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HMODULE


def send_ipc(cmd, arg=0):
    """Post a command to the running instance (arg rides in lParam, e.g. a process id).
    Returns True if one was found."""
    hwnd = user32.FindWindowW(CLASS_NAME, None)
    if not hwnd:
        return False
    return bool(user32.PostMessageW(hwnd, WM_IPC, IPC_CODES[cmd], int(arg)))


class Tray(threading.Thread):
    """icons: dict name -> .ico path. on_event(name) receives 'toggle_show', 'hotkey', 'ipc:<cmd>',
    'menu:<id>'. menu_builder() returns [(id, label, flags) | None for separator | (label, [items])]."""

    def __init__(self, icons, tip, on_event, menu_builder, hotkey=None):
        super().__init__(name="tray", daemon=True)
        self.icons_paths, self.tip, self.on_event, self.menu_builder = icons, tip, on_event, menu_builder
        self.hotkey = hotkey             # (mods, vk) or None
        self.hotkey_ok = False
        self.added = False
        self.hwnd = None
        self.icon_name = next(iter(icons))
        self.ready = threading.Event()
        self.error = ""

    # ---- called from any thread
    def set_state(self, icon_name=None, tip=None):
        if icon_name:
            self.icon_name = icon_name
        if tip:
            self.tip = tip
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_SETTIP, 0, 0)

    def stop(self):
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)

    def icon_rect(self):
        nii = NOTIFYICONIDENTIFIER()
        nii.cbSize = ctypes.sizeof(nii)
        nii.hWnd = self.hwnd
        nii.uID = 1
        r = wt.RECT()
        hr = shell32.Shell_NotifyIconGetRect(ctypes.byref(nii), ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom) if hr == 0 else None

    # ---- tray thread
    def _nid(self, flags):
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(nid)
        nid.hWnd = self.hwnd
        nid.uID = 1
        nid.uFlags = flags
        nid.uCallbackMessage = WM_TRAY
        nid.hIcon = self.hicons.get(self.icon_name) or 0
        nid.szTip = self.tip[:127]
        return nid

    def _add(self):
        self.added = bool(shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(self._nid(NIF_MESSAGE | NIF_ICON | NIF_TIP))))
        if not self.added:
            self.error = f"Shell_NotifyIconW(NIM_ADD) failed, err {ctypes.get_last_error()}"

    def _menu(self):
        def build(items):
            hm = user32.CreatePopupMenu()
            for it in items:
                if it is None:
                    user32.AppendMenuW(hm, MF_SEPARATOR, 0, None)
                elif isinstance(it[1], list):
                    sub = build(it[1])
                    user32.AppendMenuW(hm, MF_POPUP, sub, it[0])
                else:
                    cid, label, flags = it
                    user32.AppendMenuW(hm, MF_STRING | flags, cid, label)
            return hm
        hm = build(self.menu_builder())
        user32.SetMenuDefaultItem(hm, 1, 0)        # id 1 (Show/Hide) is bold = left-click action
        pt = wt.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        user32.SetForegroundWindow(self.hwnd)      # required so the menu closes when clicking away
        cmd = user32.TrackPopupMenu(hm, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY, pt.x, pt.y, 0, self.hwnd, None)
        user32.PostMessageW(self.hwnd, WM_NULL, 0, 0)
        user32.DestroyMenu(hm)
        if cmd:
            self.on_event(f"menu:{cmd}")

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_TRAY:
                ev = lparam & 0xFFFF
                if ev == WM_LBUTTONUP:
                    self.on_event("toggle_show")
                elif ev in (WM_RBUTTONUP, WM_CONTEXTMENU):
                    self._menu()
                return 0
            if msg == WM_HOTKEY:
                self.on_event("hotkey")
                return 0
            if msg == WM_DISPLAYCHANGE:                # a monitor came or went, or the resolution changed
                self.on_event("display")               # (log 57); DefWindowProc below as usual
            if msg == WM_IPC:
                arg = int(lparam or 0)
                self.on_event("ipc:" + IPC_NAMES.get(int(wparam or 0), "show") + (f":{arg}" if arg else ""))
                return 0
            if msg == WM_SETTIP:
                if self.added:
                    shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(self._nid(NIF_ICON | NIF_TIP)))
                return 0
            if msg == self.WM_TASKBARCREATED:          # Explorer restarted: put the icon back
                self._add()
                return 0
            if msg == WM_CLOSE:
                if self.added:
                    shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self._nid(0)))
                    self.added = False
                if self.hotkey_ok:
                    user32.UnregisterHotKey(hwnd, 1)
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
        except Exception as e:   # never let an exception escape into Win32
            self.error = repr(e)
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def run(self):
        try:
            hinst = kernel32.GetModuleHandleW(None)
            self._proc = WNDPROC(self._wndproc)       # keep a reference
            wc = WNDCLASSEXW()
            wc.cbSize = ctypes.sizeof(wc)
            wc.lpfnWndProc = self._proc
            wc.hInstance = hinst
            wc.lpszClassName = CLASS_NAME
            user32.RegisterClassExW(ctypes.byref(wc))
            self.WM_TASKBARCREATED = user32.RegisterWindowMessageW("TaskbarCreated")
            self.hwnd = user32.CreateWindowExW(0, CLASS_NAME, "Parakeet Live tray", 0, 0, 0, 0, 0,
                                               None, None, hinst, None)
            cx = user32.GetSystemMetrics(49)          # SM_CXSMICON
            self.hicons = {k: user32.LoadImageW(None, p, IMAGE_ICON, cx, cx, LR_LOADFROMFILE)
                           for k, p in self.icons_paths.items()}
            self._add()
            if self.hotkey:
                mods, vk = self.hotkey
                self.hotkey_ok = bool(user32.RegisterHotKey(self.hwnd, 1, mods | 0x4000, vk))  # MOD_NOREPEAT
        except Exception as e:
            self.error = repr(e)
        self.ready.set()
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
