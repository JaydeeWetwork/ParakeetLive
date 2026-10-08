"""GPU watcher for auto-park (Windows, stdlib ctypes only: DXGI + PDH + Toolhelp32; no subprocesses).

- find_adapter(): the NVIDIA adapter (vendor 0x10DE) via DXGI -> LUID string as used in the GPU perf counters,
  description, dedicated VRAM bytes. Nothing is hard-coded.
- GpuSampler.sample(): one PDH collection of
    \\GPU Process Memory(*)\\Dedicated Usage, \\GPU Engine(*engtype_3D)\\Utilization Percentage,
    \\GPU Adapter Memory(*)\\Dedicated Usage
  filtered to that LUID, plus a Toolhelp32 process snapshot (pid -> exe name).
- MinecraftJava.check(): which java/javaw processes are Minecraft Java Edition (read-only: command line
  via PROCESS_QUERY_LIMITED_INFORMATION, else a window titled 'Minecraft...'); log 57.
- ParkPolicy.update(): pure decision logic with hysteresis (unit-testable, no Windows calls).
"""
import ctypes
import ctypes.wintypes as wt
import re
import time

# ----------------------------------------------------------------------------- DXGI adapter lookup
class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]


def _guid(s):
    s = s.strip("{}").replace("-", "")
    b = bytes.fromhex(s)
    g = _GUID()
    g.Data1 = int.from_bytes(b[0:4], "big"); g.Data2 = int.from_bytes(b[4:6], "big"); g.Data3 = int.from_bytes(b[6:8], "big")
    for i in range(8):
        g.Data4[i] = b[8 + i]
    return g


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", _LUID), ("Flags", ctypes.c_uint)]


def _vcall(obj, index, restype, *argtypes):
    vtbl = ctypes.cast(ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


def luid_str(luid):
    return f"luid_0x{luid.HighPart & 0xFFFFFFFF:08x}_0x{luid.LowPart:08x}"


def enum_adapters():
    """[(adapter_ptr or None, desc)] via IDXGIFactory1::EnumAdapters1 (index 12) / IDXGIAdapter1::GetDesc1 (index 10)."""
    dxgi = ctypes.WinDLL("dxgi")
    fac = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(_guid("{770aae78-f26f-4dba-a829-253c83d1b387}")), ctypes.byref(fac))
    if hr != 0:
        raise OSError(f"CreateDXGIFactory1 failed 0x{hr & 0xFFFFFFFF:08x}")
    out = []
    try:
        i = 0
        while True:
            ad = ctypes.c_void_p()
            hr = _vcall(fac, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(fac, i, ctypes.byref(ad))
            if hr != 0:
                break
            d = _DXGI_ADAPTER_DESC1()
            _vcall(ad, 10, ctypes.c_long, ctypes.POINTER(_DXGI_ADAPTER_DESC1))(ad, ctypes.byref(d))
            out.append((ad, d))
            i += 1
    finally:
        _vcall(fac, 2, ctypes.c_ulong)(fac)
    return out


def release(ptr):
    if ptr:
        _vcall(ptr, 2, ctypes.c_ulong)(ptr)


def find_adapter(vendor=0x10DE):
    """-> dict(luid, name, dedicated_mb) for the first adapter of that vendor (NVIDIA by default), or None."""
    found = None
    for ad, d in enum_adapters():
        if found is None and d.VendorId == vendor:
            found = {"luid": luid_str(d.AdapterLuid), "name": d.Description,
                     "dedicated_mb": round(d.DedicatedVideoMemory / 2**20)}
        release(ad)
    return found


# ----------------------------------------------------------------------------- processes (Toolhelp32)
class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", wt.LONG), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
_k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
_k32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
_k32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
_k32.CloseHandle.argtypes = [wt.HANDLE]


def processes():
    """{pid: (exe_name, parent_pid)}"""
    snap = _k32.CreateToolhelp32Snapshot(0x2, 0)
    out = {}
    if not snap or snap == wt.HANDLE(-1).value:
        return out
    try:
        e = _PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = _k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out[e.th32ProcessID] = (e.szExeFile, e.th32ParentProcessID)
            ok = _k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        _k32.CloseHandle(snap)
    return out


# ----------------------------------------------------------------------------- Minecraft Java (log 57)
JAVA_NAMES = {"javaw.exe", "java.exe"}       # in park_games these only count when the process is Minecraft
_MC_RX = re.compile(r"minecraft", re.I)      # net.minecraft main class, .minecraft game dir, launcher brand


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", wt.USHORT), ("MaximumLength", wt.USHORT), ("Buffer", ctypes.c_void_p)]


_k32.OpenProcess.restype = wt.HANDLE
_k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
try:
    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    _ntdll.NtQueryInformationProcess.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.ULONG,
                                                 ctypes.POINTER(wt.ULONG)]
except Exception:                                # pragma: no cover
    _ntdll = None
_u32 = ctypes.WinDLL("user32", use_last_error=True)
_WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
_u32.EnumWindows.argtypes = [_WNDENUMPROC, wt.LPARAM]
_u32.GetWindowThreadProcessId.restype = wt.DWORD
_u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
_u32.GetWindowTextLengthW.argtypes = [wt.HWND]
_u32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
_u32.IsWindowVisible.argtypes = [wt.HWND]


def process_cmdline(pid):
    """Command line of another (same-user) process, read-only; None if it cannot be read.
    PROCESS_QUERY_LIMITED_INFORMATION + ProcessCommandLineInformation (60): no admin, no memory reads."""
    if _ntdll is None:
        return None
    h = _k32.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    try:
        need = wt.ULONG(0)
        _ntdll.NtQueryInformationProcess(h, 60, None, 0, ctypes.byref(need))
        if not need.value or need.value > 1 << 20:
            return None
        buf = ctypes.create_string_buffer(need.value)
        if _ntdll.NtQueryInformationProcess(h, 60, buf, need.value, ctypes.byref(need)) != 0:
            return None
        us = _UNICODE_STRING.from_buffer(buf)
        return ctypes.wstring_at(us.Buffer, us.Length // 2) if us.Buffer and us.Length else ""
    except Exception:
        return None
    finally:
        _k32.CloseHandle(h)


def window_titles(pids):
    """{pid: [titles of its visible top-level windows]} for the given pids."""
    out = {}

    def cb(h, _lp):
        try:
            if _u32.IsWindowVisible(h):
                pid = wt.DWORD(0)
                _u32.GetWindowThreadProcessId(h, ctypes.byref(pid))
                if pid.value in pids:
                    n = _u32.GetWindowTextLengthW(h)
                    if n:
                        b = ctypes.create_unicode_buffer(n + 1)
                        _u32.GetWindowTextW(h, b, n + 1)
                        out.setdefault(pid.value, []).append(b.value)
        except Exception:
            pass
        return True
    _u32.EnumWindows(_WNDENUMPROC(cb), 0)
    return out


class MinecraftJava:
    """Which running java.exe/javaw.exe are Minecraft Java Edition: the command line mentions minecraft
    (any launcher: official, Prism, MultiMC, CurseForge, Modrinth, Lunar ...), or one of its windows is
    titled "Minecraft...". The command-line verdict is cached for the life of the pid; a Java app that
    is not Minecraft (IDE, Gradle, a server without a window) never parks the model."""

    def __init__(self, cmdline=process_cmdline, titles=window_titles):
        self.cmdline, self.titles = cmdline, titles
        self.cache = {}                      # pid -> (exe, True | False | None = unreadable)

    def check(self, procs):
        found, undecided = set(), set()
        for pid, (exe, _pp) in procs.items():
            if exe.lower() not in JAVA_NAMES:
                continue
            c = self.cache.get(pid)
            if c is None or c[0] != exe:     # new pid (or a reused one)
                cl = self.cmdline(pid)
                c = self.cache[pid] = (exe, None if cl is None else bool(_MC_RX.search(cl)))
            (found if c[1] else undecided).add(pid)
        if undecided:
            try:
                for pid, ts in self.titles(undecided).items():
                    if any(t.lower().startswith("minecraft") for t in ts):
                        found.add(pid)
                        self.cache[pid] = (procs[pid][0], True)
            except Exception:
                pass
        self.cache = {p: c for p, c in self.cache.items() if p in procs}
        return found


# ----------------------------------------------------------------------------- PDH
class _FMT_VALUE(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("longValue", wt.LONG), ("doubleValue", ctypes.c_double), ("largeValue", ctypes.c_longlong)]
    _fields_ = [("CStatus", wt.DWORD), ("u", _U)]


class _FMT_ITEM(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _FMT_VALUE)]


PDH_FMT_DOUBLE, PDH_FMT_LARGE, PDH_FMT_NOCAP100 = 0x200, 0x400, 0x8000
PDH_MORE_DATA = 0x800007D2
_PID_RX = re.compile(r"pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)", re.I)


class Pdh:
    def __init__(self):
        self.dll = ctypes.WinDLL("pdh")
        self.q = ctypes.c_void_p()
        st = self.dll.PdhOpenQueryW(None, None, ctypes.byref(self.q))
        if st != 0:
            raise OSError(f"PdhOpenQueryW 0x{st & 0xFFFFFFFF:08x}")
        self.counters = {}

    def add(self, key, path):
        h = ctypes.c_void_p()
        st = self.dll.PdhAddEnglishCounterW(self.q, ctypes.c_wchar_p(path), None, ctypes.byref(h))
        if st != 0:
            raise OSError(f"PdhAddEnglishCounterW({path}) 0x{st & 0xFFFFFFFF:08x}")
        self.counters[key] = h

    def collect(self):
        return self.dll.PdhCollectQueryData(self.q) == 0

    def array(self, key, fmt):
        h = self.counters[key]
        size, count = wt.DWORD(0), wt.DWORD(0)
        st = self.dll.PdhGetFormattedCounterArrayW(h, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if (st & 0xFFFFFFFF) != PDH_MORE_DATA or size.value == 0:
            return []
        buf = (ctypes.c_byte * size.value)()
        st = self.dll.PdhGetFormattedCounterArrayW(h, fmt, ctypes.byref(size), ctypes.byref(count), buf)
        if st != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(_FMT_ITEM))
        out = []
        for i in range(count.value):
            it = items[i]
            if it.FmtValue.CStatus in (0, 1):
                v = it.FmtValue.u.doubleValue if fmt & PDH_FMT_DOUBLE else it.FmtValue.u.largeValue
                out.append((it.szName, v))
        return out

    def close(self):
        if self.q:
            self.dll.PdhCloseQuery(self.q)
            self.q = ctypes.c_void_p()


class GpuSampler:
    def __init__(self, adapter):
        self.ad = adapter
        self.luid = adapter["luid"].lower()
        self.pdh = Pdh()
        self.pdh.add("pmem", r"\GPU Process Memory(*)\Dedicated Usage")
        self.pdh.add("eng3d", r"\GPU Engine(*engtype_3D)\Utilization Percentage")
        self.pdh.add("amem", r"\GPU Adapter Memory(*)\Dedicated Usage")
        self.pdh.collect()                 # rate counters (3D %) need two collections
        self.mc = MinecraftJava()

    def sample(self):
        self.pdh.collect()
        pmem, p3d = {}, {}
        for name, v in self.pdh.array("pmem", PDH_FMT_LARGE):
            m = _PID_RX.search(name)
            if m and m.group(2).lower() == self.luid:
                pid = int(m.group(1)); pmem[pid] = pmem.get(pid, 0) + v / 2**20
        for name, v in self.pdh.array("eng3d", PDH_FMT_DOUBLE | PDH_FMT_NOCAP100):
            m = _PID_RX.search(name)
            if m and m.group(2).lower() == self.luid:
                pid = int(m.group(1)); p3d[pid] = p3d.get(pid, 0.0) + v
        used = None
        for name, v in self.pdh.array("amem", PDH_FMT_LARGE):
            if name.lower().startswith(self.luid):
                used = (used or 0) + v / 2**20
        total = self.ad["dedicated_mb"]
        procs = processes()
        return {"t": time.monotonic(), "mem_mb": pmem, "util3d": p3d, "used_mb": used,
                "total_mb": total, "free_mb": None if used is None else total - used, "procs": procs,
                "mc_java": self.mc.check(procs)}

    def close(self):
        self.pdh.close()


# ----------------------------------------------------------------------------- policy (pure)
# Our own GPU users. WSL's CUDA memory (the model, ~1.4 GB) is attributed to vmwp.exe (Hyper-V VM worker),
# not to vmmem/vmmemWSL; measured 2026-10-07. dwm.exe draws the 4K desktop on this card.
OWN_NAMES = {"pythonw.exe", "vmmem", "vmmem.exe", "vmmemwsl", "vmmemwsl.exe", "vmwp.exe", "wslhost.exe", "wsl.exe",
             "wslservice.exe", "dwm.exe"}


class ParkPolicy:
    """Decides park/unpark from samples. Hysteresis:
       - memory and low-free triggers must hold for `confirm` consecutive polls; 3D must hold for park_3d_s;
         a game-list match parks at once.
       - unpark only after no trigger for unpark_after_s AND free VRAM (while parked) >= the effective unpark
         threshold = max(unpark_free_mb, model_mb + park_free_min_mb + margin_mb), so loading the model back can
         never itself cause a low-free re-park."""

    def __init__(self, cfg, own_pids=()):
        self.cfg = cfg
        self.own_pids = set(own_pids)
        self.mem_hits = {}           # pid -> consecutive polls over the memory threshold
        self.hot_since = {}          # pid -> time 3D util went over the threshold
        self.low_hits = 0
        self.clear_since = None
        self.last = {}

    def reset(self):
        self.mem_hits, self.hot_since, self.low_hits, self.clear_since = {}, {}, 0, None

    def _names(self, key):
        return {n.lower() for n in self.cfg.get(key, [])}

    def unpark_free_needed(self):
        c = self.cfg
        return max(c["unpark_free_mb"], c["model_vram_mb"] + c["park_free_min_mb"] + 200)

    def trigger(self, s, model_on_gpu):
        """-> (reason, process_name or None) or None"""
        c = self.cfg
        ignore = OWN_NAMES | self._names("park_ignore")
        games = self._names("park_games")
        procs = s["procs"]
        name = lambda pid: procs.get(pid, ("pid %d" % pid, 0))[0]  # noqa: E731
        mine = lambda pid: pid in self.own_pids or name(pid).lower() in ignore  # noqa: E731
        # 1) game list: any running process (GPU use not required); java/javaw only if it is Minecraft (log 57)
        mc = s.get("mc_java", ())
        for pid, (exe, _pp) in procs.items():
            lo = exe.lower()
            if lo in games and pid not in self.own_pids and (lo not in JAVA_NAMES or pid in mc):
                return ("game", exe)
        now = s["t"]
        # 2) dedicated memory on the NVIDIA adapter
        hit = None
        seen = set()
        for pid, mb in sorted(s["mem_mb"].items(), key=lambda kv: -kv[1]):
            if mine(pid) or mb < c["park_mem_mb"]:
                continue
            seen.add(pid)
            self.mem_hits[pid] = self.mem_hits.get(pid, 0) + 1
            if self.mem_hits[pid] >= c["park_confirm_polls"] and hit is None:
                hit = ("vram %d MB" % mb, name(pid))
        self.mem_hits = {p: n for p, n in self.mem_hits.items() if p in seen}
        if hit:
            return hit
        # 3) sustained 3D load
        hot = set()
        for pid, u in s["util3d"].items():
            if mine(pid) or u < c["park_3d_pct"]:
                continue
            hot.add(pid)
            t0 = self.hot_since.setdefault(pid, now)
            if now - t0 >= c["park_3d_s"]:
                hit = hit or ("3D %d%%" % u, name(pid))
        self.hot_since = {p: t for p, t in self.hot_since.items() if p in hot}
        if hit:
            return hit
        # 4) low free VRAM while our model is on the GPU
        if model_on_gpu and s["free_mb"] is not None and s["free_mb"] < c["park_free_min_mb"]:
            self.low_hits += 1
            if self.low_hits >= c["park_confirm_polls"] + 1:
                top = max(((p, mb) for p, mb in s["mem_mb"].items() if not mine(p)), key=lambda kv: kv[1], default=None)
                return ("low VRAM %d MB free" % s["free_mb"], name(top[0]) if top else None)
        else:
            self.low_hits = 0
        return None

    def update(self, s, parked, model_on_gpu):
        """-> ("park", reason, proc) | ("unpark", None, None) | None"""
        trig = self.trigger(s, model_on_gpu)
        self.last = {"trigger": trig, "free_mb": s["free_mb"], "used_mb": s["used_mb"]}
        if not parked:
            self.clear_since = None
            return ("park",) + trig if trig else None
        # model already on the GPU (a temporary load while parked): it only has to keep the low-free margin
        need = self.cfg["park_free_min_mb"] + 200 if model_on_gpu else self.unpark_free_needed()
        # "clear" = no trigger AND enough free VRAM; the 60 s only count while both hold, so a low-VRAM
        # park (many small allocations) also waits 60 s after the memory comes back
        if trig or (s["free_mb"] or 0) < need:
            self.clear_since = None
            return None
        if self.clear_since is None:
            self.clear_since = s["t"]
        if s["t"] - self.clear_since >= self.cfg["unpark_after_s"]:
            return ("unpark", None, None)
        return None
