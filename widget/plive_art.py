"""Anti-aliased artwork for the round record button (drawn 4x size with PIL, then downscaled)."""
import base64
import io

from PIL import Image, ImageDraw, ImageFilter

SS = 4  # supersampling factor


def _hex(c, a=255):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a)


def _mix(c1, c2, t):
    a, b = _hex(c1), _hex(c2)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,)


def _blank(D):
    return Image.new("RGBA", (D * SS, D * SS), (0, 0, 0, 0))


def _down(im, D):
    return im.resize((D, D), Image.LANCZOS)


def _circle(dr, cx, cy, r, **kw):
    dr.ellipse((cx - r, cy - r, cx + r, cy + r), **kw)


def base_ring(D, fill, outline):
    im = _blank(D)
    S = D * SS
    dr = ImageDraw.Draw(im)
    _circle(dr, S / 2, S / 2, 0.46 * S, fill=_hex(fill), outline=_hex(outline), width=max(1, int(0.012 * S)))
    return _down(im, D)


def disc(D, color, icon="mic", icon_alpha=245, shadow=True):
    S = D * SS
    c, r = S / 2, 0.31 * S
    im = _blank(D)
    if shadow:
        sh = _blank(D)
        _circle(ImageDraw.Draw(sh), c, c + 0.025 * S, r, fill=(0, 0, 0, 120))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(0.03 * S)))
    # soft vertical gradient: a little lighter on top
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    top, bot = _mix(color, "#ffffff", 0.18), _mix(color, "#000000", 0.10)
    for y in range(S):
        t = y / (S - 1)
        gd.line([(0, y), (S, y)], fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)) + (255,))
    mask = Image.new("L", (S, S), 0)
    _circle(ImageDraw.Draw(mask), c, c, r, fill=255)
    im.paste(grad, (0, 0), mask)
    dr = ImageDraw.Draw(im)
    white = (255, 255, 255, icon_alpha)
    w = max(1, int(0.024 * S))
    if icon == "mic":
        cw, ch = 0.105 * S, 0.175 * S
        cy = c - 0.045 * S
        dr.rounded_rectangle((c - cw / 2, cy - ch / 2, c + cw / 2, cy + ch / 2), radius=cw / 2, fill=white)
        ar = 0.092 * S
        dr.arc((c - ar, cy - ar - 0.005 * S, c + ar, cy + ar + 0.03 * S), start=10, end=170, fill=white, width=w)
        y0 = cy + ar + 0.03 * S
        dr.line([(c, y0 - 0.01 * S), (c, y0 + 0.045 * S)], fill=white, width=w)
        dr.rounded_rectangle((c - 0.055 * S, y0 + 0.035 * S, c + 0.055 * S, y0 + 0.035 * S + w), radius=w / 2, fill=white)
    elif icon == "stop":
        q = 0.075 * S
        dr.rounded_rectangle((c - q, c - q, c + q, c + q), radius=0.028 * S, fill=white)
    return _down(im, D)


def pulse_ring(D, color, phase):
    """Expanding, fading ring (phase 0..1)."""
    S = D * SS
    im = _blank(D)
    r = (0.31 + 0.15 * phase) * S
    a = int(210 * (1 - phase) ** 1.6)
    w = max(1, int((0.040 - 0.018 * phase) * S))
    _circle(ImageDraw.Draw(im), S / 2, S / 2, r, outline=_hex(color, a), width=w)
    return _down(im.filter(ImageFilter.GaussianBlur(0.006 * S)), D)


def halo(D, color, level):
    """Soft glow behind the disc that grows with mic level (0..1)."""
    S = D * SS
    im = _blank(D)
    r = (0.31 + 0.085 * level) * S
    _circle(ImageDraw.Draw(im), S / 2, S / 2, r, fill=_hex(color, int(60 + 120 * level)))
    return _down(im.filter(ImageFilter.GaussianBlur(0.035 * S)), D)


def spinner(D, color, k, n=12):
    S = D * SS
    im = _blank(D)
    dr = ImageDraw.Draw(im)
    r = 0.405 * S
    w = max(1, int(0.028 * S))
    box = (S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r)
    dr.arc(box, 0, 360, fill=_hex(color, 45), width=w)
    st = k * 360 / n
    dr.arc(box, st, st + 95, fill=_hex(color, 235), width=w)
    return _down(im, D)


def to_tk(im, tk):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode("ascii"))


def make_icon(path, color="#f7606c", icon="mic", icon_alpha=245):
    base = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    ring = base_ring(256, "#20242f", "#2c3242")
    im = Image.alpha_composite(base, ring)
    im = Image.alpha_composite(im, disc(256, color, icon, icon_alpha=icon_alpha, shadow=False))
    # crop the ring margin a little so the icon reads well at 16-32 px
    im = im.crop((14, 14, 242, 242)).resize((256, 256), Image.LANCZOS)
    im.save(path, sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path
