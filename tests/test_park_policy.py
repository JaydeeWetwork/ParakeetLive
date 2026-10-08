"""Unit tests for plive_gpu.ParkPolicy (pure logic; runs anywhere - Windows DLLs are stubbed off-Windows)."""
import ctypes, os, sys
from unittest import mock
if not hasattr(ctypes, "WinDLL"):
    ctypes.WinDLL = mock.MagicMock(); ctypes.WINFUNCTYPE = mock.MagicMock()
    import ctypes.wintypes  # noqa
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "widget"))
import plive_gpu as g

CFG = {"park_mem_mb": 400, "park_confirm_polls": 2, "park_3d_pct": 25, "park_3d_s": 10, "park_free_min_mb": 600,
       "unpark_free_mb": 1800, "unpark_after_s": 60, "model_vram_mb": 1400, "park_games": ["javaw.exe", "Minecraft.Windows.exe"],
       "park_ignore": ["OBS64.exe"]}
BASE = {10: ("dwm.exe", 1), 20: ("vmwp.exe", 1), 30: ("explorer.exe", 1), 40: ("pythonw.exe", 1)}
def S(t, mem=None, u3d=None, free=2070, procs=None, mc=()):
    pr = dict(BASE); pr.update(procs or {})
    return {"t": t, "mem_mb": {10: 634, 20: 1395, 30: 26, **(mem or {})}, "util3d": {10: 40.0, 20: 90.0, **(u3d or {})},
            "free_mb": free, "used_mb": 3962 - free, "total_mb": 3962, "procs": pr, "mc_java": set(mc)}
ok = []
def check(name, cond):
    ok.append(cond); print(("PASS " if cond else "FAIL ") + name)

p = g.ParkPolicy(CFG, own_pids=[40])
check("threshold = max(1800, 1400+600+200) = 2200", p.unpark_free_needed() == 2200)
check("own/dwm/vmwp at high use never trigger", all(p.update(S(t), False, True) is None for t in range(0, 40, 4)))
p = g.ParkPolicy(CFG)
check("game list parks at once (case-insensitive)", p.update(S(0, procs={99: ("JAVAW.EXE", 1)}, mc={99}), False, False) == ("park", "game", "JAVAW.EXE"))
p = g.ParkPolicy(CFG)
r1 = p.update(S(0, mem={50: 608}, procs={50: ("python.exe", 1)}), False, True)
r2 = p.update(S(4, mem={50: 608}, procs={50: ("python.exe", 1)}), False, True)
check("dedicated >= 400 MB needs 2 polls", r1 is None and r2 == ("park", "vram 608 MB", "python.exe"))
p = g.ParkPolicy(CFG)
p.update(S(0, mem={51: 900}, procs={51: ("obs64.exe", 1)}), False, True)
check("ignore list (case-insensitive)", p.update(S(4, mem={51: 900}, procs={51: ("obs64.exe", 1)}), False, True) is None)
p = g.ParkPolicy(CFG)
seq = [p.update(S(t, u3d={60: 30.0}, procs={60: ("game.exe", 1)}), False, True) for t in (0, 4, 8, 12)]
check("3D >= 25% parks only after 10 s", seq[:3] == [None] * 3 and seq[3] == ("park", "3D 30%", "game.exe"))
p = g.ParkPolicy(CFG)
seq = [p.update(S(t, u3d={60: 30.0 if t != 8 else 5.0}, procs={60: ("game.exe", 1)}), False, True) for t in (0, 4, 8, 12, 16)]
check("3D dip resets the 10 s timer", all(x is None for x in seq))
p = g.ParkPolicy(CFG)
seq = [p.update(S(t, mem={70: 380, 71: 380}, free=500, procs={70: ("a.exe", 1), 71: ("b.exe", 1)}), False, True) for t in (0, 4, 8)]
check("low free VRAM (< 600) parks after 3 polls", seq[:2] == [None, None] and seq[2][0] == "park" and seq[2][1].startswith("low VRAM"))
check("low free ignored when model not on GPU", g.ParkPolicy(CFG).update(S(0, free=300), False, False) is None)
p = g.ParkPolicy(CFG)
seq = [(t, p.update(S(t, free=f), True, False)) for t, f in [(0, 3400), (30, 3400), (59, 3400), (60, 3400)]]
check("unpark after 60 s clear", [x[1] for x in seq] == [None, None, None, ("unpark", None, None)])
p = g.ParkPolicy(CFG)
check("no unpark while free < 2200", all(p.update(S(t, free=2100), True, False) is None for t in range(0, 200, 4)))
p = g.ParkPolicy(CFG)
r = [p.update(S(t, procs={99: ("javaw.exe", 1)} if t < 20 else {}, free=3400, mc={99}), True, False) for t in range(0, 84, 4)]
check("game exits -> unpark 60 s later, not before", r.index(("unpark", None, None)) * 4 == 80 and r.count(("unpark", None, None)) == 1)
p = g.ParkPolicy(CFG)
check("temporary GPU load (model on GPU): unpark needs only 800 free",
      [p.update(S(t, free=1400), True, True) for t in (0, 60)][1] == ("unpark", None, None))
p = g.ParkPolicy(CFG)
r = [p.update(S(t, free=1500 if t < 100 else 3400), True, False) for t in range(0, 200, 4)]
i = r.index(("unpark", None, None)) * 4 if ("unpark", None, None) in r else None
check("low-VRAM park: 60 s counted only after free >= 2200 again (unpark at 160 s)", i == 160)
# ---- log 57: java/javaw parks only when it is Minecraft Java; Bedrock and the launchers by name
CFG57 = dict(CFG, park_games=["javaw.exe", "java.exe", "Minecraft.Windows.exe", "Minecraft.exe", "MinecraftLauncher.exe"])
check("P15 javaw.exe that is not Minecraft (IDE, Gradle) never parks",
      g.ParkPolicy(CFG57).update(S(0, procs={99: ("javaw.exe", 1), 98: ("java.exe", 1)}), False, False) is None)
check("P16 java.exe that is Minecraft parks at once",
      g.ParkPolicy(CFG57).update(S(0, procs={98: ("java.exe", 1)}, mc={98}), False, False) == ("park", "game", "java.exe"))
check("P17 Bedrock and the launchers park by name",
      all(g.ParkPolicy(CFG57).update(S(0, procs={97: (n, 1)}), False, False) == ("park", "game", n)
          for n in ("Minecraft.Windows.exe", "Minecraft.exe", "MinecraftLauncher.exe")))
pp = g.ParkPolicy(CFG57)
r18 = [pp.update(S(t, mem={99: 900}, procs={99: ("javaw.exe", 1)}), False, True) for t in (0, 4)]
check("P18 the VRAM rules still apply to a Java app that is not Minecraft", r18 == [None, ("park", "vram 900 MB", "javaw.exe")])
cmds = {1: "C:\\jdk\\bin\\javaw.exe -Xmx2G -cp x.jar net.minecraft.client.main.Main --gameDir C:\\Users\\x\\AppData\\Roaming\\.minecraft",
        2: "javaw.exe -jar idea-launcher.jar", 3: None, 4: "java.exe -jar server.jar nogui"}
mcj = g.MinecraftJava(cmdline=lambda pid: cmds.get(pid), titles=lambda pids: {3: ["Minecraft* 1.21.1 - Singleplayer"]} if 3 in pids else {})
procs = {1: ("javaw.exe", 0), 2: ("javaw.exe", 0), 3: ("javaw.exe", 0), 4: ("java.exe", 0), 5: ("notepad.exe", 0)}
check("P19 MinecraftJava: command line or window title decides; other Java apps and servers do not", mcj.check(procs) == {1, 3})
mcj.cmdline = lambda pid: (_ for _ in ()).throw(AssertionError("re-read"))      # cached per pid: no second read
check("P20 MinecraftJava caches per pid and forgets exited pids", mcj.check({1: ("javaw.exe", 0)}) == {1} and set(mcj.cache) == {1})
print("ALL PASS" if all(ok) else "SOME FAILED", sum(ok), "/", len(ok))
