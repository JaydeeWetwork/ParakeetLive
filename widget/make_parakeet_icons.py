"""Build tray/window icons from Jaydee's parakeet icon v3 (icons-new/parakeet-v3.ico + parakeet-v3-1024.png;
transparent, antialiased, exact frames 16..256).
ready     = parakeet-v3.ico as-is (exact frames)
recording = built from the 1024 px PNG: red dot badge (1 px dark outline at every target size) composited at
            1024 px, then downsampled per size with premultiplied alpha (Pillow mode "RGBa") so edges don't
            get dark/light fringes
unloaded  = v3 frames desaturated, lightened, 80% alpha (model not loaded, 0 VRAM)"""
import os, shutil
from PIL import Image, ImageDraw, ImageEnhance, ImageChops, ImageStat
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ICO = os.path.join(HERE, "icons-new", "parakeet-v3.ico")
SRC_PNG = os.path.join(HERE, "icons-new", "parakeet-v3-1024.png")
OUT = os.path.join(HERE, "icons")
BIG = 1024

def ico_frames():
    im = Image.open(SRC_ICO)
    out = {}
    for sz in sorted(im.info["sizes"]):
        im.size = sz
        im.load()
        out[sz[0]] = im.convert("RGBA").copy()
    return out

def down(img, n, flt):
    """Premultiplied-alpha downsample."""
    return img.convert("RGBa").resize((n, n), flt).convert("RGBA")

def badge_layer(n):
    """Badge for target size n, drawn at 4x of BIG and reduced to BIG (premultiplied) for smooth edges."""
    ss = 4
    f = BIG * ss / n                                 # canvas px per target px
    dia = max(8, round(n * 0.5))                     # target px: 8 at 16, 12 at 24, 128 at 256
    ol = max(1, round(n / 24))                       # target px: 1 at 16-32, 2 at 40-48 ...
    lay = Image.new("RGBA", (BIG * ss, BIG * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    x0 = (n - dia) * f
    x1 = BIG * ss - 1
    d.ellipse((x0, x0, x1, x1), fill=(24, 18, 20, 255))
    r = ol * f
    d.ellipse((x0 + r, x0 + r, x1 - r, x1 - r), fill=(236, 40, 52, 255))
    return down(lay, BIG, Image.BOX)

def dim(fr):
    g = ImageEnhance.Color(fr).enhance(0.0)
    r, gg, b, a = g.split()
    lift = lambda v: int(70 + v * 0.6)
    a = a.point(lambda v: int(v * 0.8))
    return Image.merge("RGBA", (r.point(lift), gg.point(lift), b.point(lift), a))

def standby(fr):
    """model in RAM, GPU free: half-saturated, full opacity (between ready and the gray unloaded icon)"""
    return ImageEnhance.Color(fr).enhance(0.45)

def save_ico(imgs, path):
    sizes = sorted(imgs)
    imgs[sizes[-1]].save(path, format="ICO", sizes=[(s, s) for s in sizes],
                         append_images=[imgs[s] for s in sizes[:-1]])

def diff(a, b):
    return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 4

fr = ico_frames()
png = Image.open(SRC_PNG).convert("RGBA")
assert png.size == (BIG, BIG)
# pick the filter whose plain downsample of the 1024 PNG best matches v3's own exact frames
scores = {}
for name, flt in (("box", Image.BOX), ("lanczos", Image.LANCZOS), ("bicubic", Image.BICUBIC)):
    scores[name] = round(sum(diff(down(png, n, flt), fr[n]) for n in fr) / len(fr), 3)
best = min(scores, key=scores.get)
flt = {"box": Image.BOX, "lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC}[best]
print("mean abs diff vs v3 frames by filter:", scores, "-> using", best)
rec = {}
for n in fr:
    comp = png.copy()
    comp.alpha_composite(badge_layer(n))
    rec[n] = down(comp, n, flt)
shutil.copyfile(SRC_ICO, os.path.join(OUT, "parakeet-ready.ico"))
shutil.copyfile(SRC_ICO, os.path.join(OUT, "parakeet-v3.ico"))        # window + shortcut icon (new name busts icon cache)
save_ico(rec, os.path.join(OUT, "parakeet-recording.ico"))
save_ico({s: dim(f) for s, f in fr.items()}, os.path.join(OUT, "parakeet-unloaded.ico"))
save_ico({s: standby(f) for s, f in fr.items()}, os.path.join(OUT, "parakeet-standby.ico"))
names = ["parakeet-ready.ico", "parakeet-recording.ico", "parakeet-standby.ico", "parakeet-unloaded.ico"]
sheet = Image.new("RGBA", (4 * 160, 120), (0, 0, 0, 0))
ImageDraw.Draw(sheet).rectangle((0, 0, 640, 59), fill=(32, 32, 32, 255))
ImageDraw.Draw(sheet).rectangle((0, 60, 640, 119), fill=(238, 238, 238, 255))
for i, nm in enumerate(names):
    im = Image.open(os.path.join(OUT, nm))
    print(nm, sorted(im.info["sizes"]))
    for row in (0, 60):
        x = i * 160 + 4
        for sz in (16, 20, 24, 32, 48):
            im.size = (sz, sz); im.load(); sheet.alpha_composite(im.convert("RGBA"), (x, row + 6)); x += sz + 2
sheet.resize((sheet.width * 3, sheet.height * 3), Image.NEAREST).save(os.path.join(OUT, "preview-parakeet-icons.png"))
print("done")