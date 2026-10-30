#!/usr/bin/env python3
"""
Palantir Tlo - Ty wewnatrz kuli na animowanym obrazku, jako kamera do Teams.

- Tlo: Twoj animowany GIF (tlo.gif obok skryptu). Wszystko poza kula zostaje
  dokladnie takie, jak na obrazku.
- W kuli: Ty (wyciety z pokoju), a za Toba scena, ktora jest w kuli na obrazku.
- Na Ciebie nakladane sa smugi dymu i cien szkla, zebys wygladal "w srodku".

Instalacja i start (uv):
    uv sync
    uv run palantir_tlo.py

Wirtualna kamera (raz):
    Linux:   sudo apt install v4l2loopback-dkms
             sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=Palantir exclusive_caps=1
    macOS / Windows: OBS Studio 30+, raz "Start Virtual Camera", zamknij OBS.

Klawisze w oknie podgladu:
    q / Esc  - wyjscie
    + / -    - wieksza / mniejsza postac w kuli (albo suwak "Zoom %")
    u / j    - postac wyzej / nizej (albo suwak "Wysokosc")
    m        - lustrzane odbicie
    g        - tryb "duch" (bez wycinania, twarz rozmyta w mgle)
    i        - odwroc maske (gdyby zamiast Ciebie bylo widac Twoj pokoj)
"""

import argparse
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError:
    pyvirtualcam = None

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
MODEL_PATH = os.path.join(HERE, "selfie_segmenter.tflite")

# Polozenie kuli na obrazku (dla obrazka 800x450): srodek x, srodek y, promien.
# Podane jako ulamki szerokosci/wysokosci, wiec dzialaja tez dla wiekszej wersji.
ORB_CX, ORB_CY, ORB_R = 396 / 800, 241 / 450, 243 / 450
# Ponizej tej linii (ulamek wysokosci) zaczyna sie podstawka - tam postac zanika.
ORB_BOTTOM = 372 / 450


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# Wczytanie animowanego tla
# ----------------------------------------------------------------------------
def load_background(path, W, H):
    """Zwraca liste klatek BGR (W x H), czasy klatek [s] i skale/przesuniecie."""
    from PIL import Image, ImageSequence
    im = Image.open(path)
    sw, sh = im.size
    s = max(W / sw, H / sh)
    nw, nh = int(round(sw * s)), int(round(sh * s))
    ox, oy = (nw - W) // 2, (nh - H) // 2
    frames, durs = [], []
    for fr in ImageSequence.Iterator(im):
        rgb = np.array(fr.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)[oy:oy + H, ox:ox + W]
        frames.append(np.ascontiguousarray(bgr))
        durs.append(max(fr.info.get("duration", im.info.get("duration", 80)) or 80, 20) / 1000.0)
    geom = dict(sw=sw, sh=sh, s=s, ox=ox, oy=oy)
    return frames, durs, geom


# ----------------------------------------------------------------------------
# Wycinanie postaci (MediaPipe)
# ----------------------------------------------------------------------------
class PersonSegmenter:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        if not os.path.exists(MODEL_PATH):
            print("Pobieram model do wycinania postaci (jednorazowo, ~250 KB)...")
            download(MODEL_URL, MODEL_PATH)
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False)
        self.mp = mp
        self.seg = vision.ImageSegmenter.create_from_options(opts)
        self.last_ts = -1
        self.prev = None

    def __call__(self, bgr_sq):
        rgb = cv2.cvtColor(bgr_sq, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = int(time.perf_counter() * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        res = self.seg.segment_for_video(img, ts)
        masks = res.confidence_masks
        m = np.squeeze(masks[0].numpy_view()).astype(np.float32)
        if len(masks) > 1:          # model wieloklasowy: [0] = tlo
            m = 1.0 - m
        if self.prev is not None and self.prev.shape == m.shape:
            m = 0.55 * m + 0.45 * self.prev    # mniej migotania krawedzi
        self.prev = m
        return m

    def close(self):
        self.seg.close()


def download(url, path):
    try:
        urllib.request.urlretrieve(url, path)
    except Exception:
        # Python z python.org na macOS czesto nie ma certyfikatow SSL - probujemy certifi
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx) as r, open(path, "wb") as f:
            f.write(r.read())


def ghost_mask(n):
    """Maska bez wycinania: miekki owal na twarz i ramiona."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    face = np.exp(-(((xx - 0.5) / 0.22) ** 2 + ((yy - 0.42) / 0.30) ** 2))
    body = np.exp(-(((xx - 0.5) / 0.42) ** 2 + ((yy - 1.0) / 0.30) ** 2))
    return np.clip(np.maximum(face, body * 0.8) * 1.3, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------
# Skladanie obrazu
# ----------------------------------------------------------------------------
class OrbComposer:
    SQ = 512    # rozmiar kwadratu, do ktorego przycinamy kamere

    def __init__(self, frames, durs, geom, W, H, zoom=1.0, yoff=0.0, opacity=1.0, smoke=0.25):
        self.frames, self.durs = frames, durs
        self.total = sum(durs)
        self.starts = np.cumsum([0.0] + durs[:-1])
        self.W, self.H = W, H
        g = geom
        sx = g["s"]
        self.cx = ORB_CX * g["sw"] * sx - g["ox"]
        self.cy = ORB_CY * g["sh"] * sx - g["oy"]
        self.R = ORB_R * g["sh"] * sx
        bottom = ORB_BOTTOM * g["sh"] * sx - g["oy"]
        self.zoom, self.yoff, self.opacity, self.smoke = zoom, yoff, opacity, smoke

        # Obszar roboczy = kwadrat wokol kuli przyciety do kadru
        self.x0 = max(int(self.cx - self.R), 0)
        self.x1 = min(int(self.cx + self.R) + 1, W)
        self.y0 = max(int(self.cy - self.R), 0)
        self.y1 = min(int(self.cy + self.R) + 1, H)
        yy, xx = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        self.dx = (xx - self.cx) / self.R
        self.dy = (yy - self.cy) / self.R
        rn = np.sqrt(self.dx ** 2 + self.dy ** 2)
        self.rn = rn
        th = np.arctan2(self.dy, self.dx)
        self.rip_s = (np.sin(4 * th) * np.minimum(rn, 1) ** 2).astype(np.float32)
        self.rip_c = (np.cos(4 * th) * np.minimum(rn, 1) ** 2).astype(np.float32)

        # Gdzie postac moze sie pojawic: srodek kuli, zanik tuz przy szkle i przy podstawce
        region = (1.0 - smoothstep(0.86, 0.97, rn)) * (1.0 - smoothstep(bottom - 0.12 * self.R, bottom, yy))
        self.region = region.astype(np.float32)
        # Strefa przy brzegu, w ktorej postac rozmywa sie i rozplywa w dym (0 = srodek, 1 = brzeg)
        self.edge = smoothstep(0.60, 0.93, rn).astype(np.float32)[..., None]
        # Lagodny zanik przy krawedziach kadru z kamery (zamiast ostrego kwadratu / ucietej glowy)
        u = (np.arange(self.SQ, dtype=np.float32) + 0.5) / self.SQ
        b = smoothstep(0.0, 0.18, np.minimum(u, 1.0 - u))
        self.frame_fade = np.outer(b, b).astype(np.float32)
        # Cien szkla: postac ciemnieje ku brzegom
        self.shade = np.clip(1.0 - 0.55 * np.minimum(rn, 1) ** 2.2, 0, 1).astype(np.float32)[..., None]
        self._build_maps()

    def _build_maps(self):
        """Mapa: piksel w kuli -> piksel w kwadracie z kamery (z lekka soczewka)."""
        rn = self.rn
        r_lens = np.where(rn < 1, rn ** 1.15, rn)     # lekko powiekszony srodek
        safe = np.maximum(rn, 1e-4)
        k = r_lens / safe / self.zoom
        h = self.SQ / 2.0
        self.base_mx = (h + self.dx * k * h).astype(np.float32)
        self.base_my = (h * (1 - self.yoff) + self.dy * k * h).astype(np.float32)

    def set_zoom(self, z):
        self.zoom = float(np.clip(z, 0.5, 3.0))
        self._build_maps()

    def set_yoff(self, y):
        self.yoff = float(np.clip(y, -0.6, 0.6))
        self._build_maps()

    def change_zoom(self, f):
        self.set_zoom(self.zoom * f)

    def move(self, d):
        self.set_yoff(self.yoff + d)

    def frame_at(self, t):
        tt = t % self.total
        i = int(np.searchsorted(self.starts, tt, side="right") - 1)
        return self.frames[max(i, 0)]

    def render(self, cam_sq, mask_sq, t):
        """cam_sq: kwadrat BGR SQxSQ z kamery, mask_sq: maska postaci (dowolny kwadrat)."""
        bg = self.frame_at(t)
        out = bg.copy()
        roi_bg = bg[self.y0:self.y1, self.x0:self.x1]

        # Lekkie falowanie szkla
        a = 0.010 * self.SQ
        rip = a * (self.rip_s * math.cos(t * 0.8) + self.rip_c * math.sin(t * 0.8))
        mx = self.base_mx + rip * 0.7
        my = self.base_my + rip
        person = cv2.remap(cam_sq, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if mask_sq.shape[0] != self.SQ:
            mask_sq = cv2.resize(mask_sq, (self.SQ, self.SQ), interpolation=cv2.INTER_LINEAR)
        m = cv2.remap(mask_sq, mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fade = cv2.remap(self.frame_fade, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = smoothstep(0.25, 0.75, m) * fade * self.region * self.opacity
        m = cv2.GaussianBlur(m, (0, 0), 1.2)[..., None]

        # Smugi dymu ze szkla (tylko szare, jasne smugi, bez kolorowych konturow budowli z tla)
        bgl = cv2.cvtColor(roi_bg, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
        sat = cv2.cvtColor(roi_bg, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32) * (1.0 / 255.0)
        gray_w = np.clip(1.0 - sat * 2.5, 0, 1)
        wisps = np.clip((bgl - cv2.GaussianBlur(bgl, (0, 0), 6)) * 2.5, 0, 1) * gray_w
        wisps = cv2.GaussianBlur(wisps, (0, 0), 1.0)[..., None]

        # Przy brzegu postac rozpada sie na dym: zostaje tam, gdzie sa smugi
        e = self.edge
        m = m * (1.0 - e * (1.0 - np.clip(wisps * 3.0, 0, 1)) * np.float32(0.6))

        bgf = roi_bg.astype(np.float32) * (1.0 / 255.0)
        p = person.astype(np.float32) * (1.0 / 255.0)

        # Kolor postaci dopasowany do wnetrza kuli
        inside = self.region > 0.5
        mean_col = bgf[inside].mean(axis=0) + 1e-3
        tint = (mean_col / mean_col.mean()).astype(np.float32)
        gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] * (1.0 / 255.0)
        p = cv2.pow(p, 1.15)
        p = p * np.float32(0.55) + gray * tint * np.float32(0.45)
        p *= self.shade * np.float32(0.85)

        # Rozmycie przy brzegu
        p = p + (cv2.GaussianBlur(p, (0, 0), 4.0) - p) * e

        # Dym nakladany na postac (tryb "screen"): self.smoke w srodku, mocny przy brzegu
        w = np.float32(self.smoke) + (np.float32(0.9) - np.float32(self.smoke)) * e
        p = 1.0 - (1.0 - p) * (1.0 - wisps * w)
        np.clip(p, 0, 1, out=p)

        res = bgf * (1.0 - m) + p * m
        out[self.y0:self.y1, self.x0:self.x1] = (res * 255.0).astype(np.uint8)
        return out


# ----------------------------------------------------------------------------
# Kamera i petla
# ----------------------------------------------------------------------------
def square_crop(frame, n):
    h, w = frame.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return cv2.resize(frame[y0:y0 + s, x0:x0 + s], (n, n), interpolation=cv2.INTER_AREA)


def open_camera(index, w, h):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        msg = (f"Nie moge otworzyc kamery nr {index}. Sprobuj --camera 1. "
               "Upewnij sie, ze zaden inny program (Teams, Discord, przegladarka) "
               "nie uzywa teraz tej kamery.")
        if sys.platform == "darwin":
            msg += ("\nmacOS: Ustawienia systemowe -> Prywatnosc i ochrona -> Kamera -> "
                    "zezwol Terminalowi (albo aplikacji, z ktorej uruchamiasz skrypt).")
        elif sys.platform.startswith("linux"):
            msg += ("\nLinux: lista kamer: v4l2-ctl --list-devices "
                    "(nie wybieraj wirtualnej kamery 'Palantir', domyslnie /dev/video10).\n"
                    "Kto trzyma kamere: fuser -v /dev/video0")
        sys.exit(msg)
    return cap


def vcam_help():
    if sys.platform.startswith("linux"):
        return ("Linux: potrzebny modul v4l2loopback (sudo apt install v4l2loopback-dkms), potem:\n"
                "  sudo modprobe v4l2loopback devices=1 video_nr=10 "
                "card_label=Palantir exclusive_caps=1")
    if sys.platform == "darwin":
        return ("macOS: zainstaluj OBS Studio 30 lub nowszy (obsproject.com), raz kliknij "
                "'Start Virtual Camera' (zezwol na rozszerzenie systemowe), potem zamknij OBS.")
    return "Zainstaluj OBS Studio, raz kliknij w nim 'Start Virtual Camera', zamknij OBS."


def run(args):
    if not os.path.exists(args.tlo):
        sys.exit(f"Nie widze obrazka tla: {args.tlo}  (zapisz swoj GIF jako tlo.gif obok skryptu)")
    frames, durs, geom = load_background(args.tlo, args.width, args.height)
    comp = OrbComposer(frames, durs, geom, args.width, args.height, args.zoom, args.yoff,
                       args.opacity, args.dym)
    print(f"Tlo: {len(frames)} klatek animacji.")

    cap = open_camera(args.camera, 1280, 720)
    mirror = args.mirror

    vcam = None
    if not args.no_vcam:
        if pyvirtualcam is None:
            sys.exit("Brak pyvirtualcam. Zainstaluj: pip install pyvirtualcam (albo --no-vcam).")
        try:
            vcam = pyvirtualcam.Camera(width=args.width, height=args.height, fps=args.fps,
                                       fmt=PixelFormat.BGR, device=args.vcam_device)
        except Exception as e:
            cap.release()
            sys.exit(f"Nie udalo sie uruchomic wirtualnej kamery: {e}\n{vcam_help()}")
        print(f"Wirtualna kamera: {vcam.device}  -> wybierz ja w Teams.")

    seg = None
    if not args.ghost:
        try:
            seg = PersonSegmenter()
        except Exception as e:
            print(f"Wycinanie postaci niedostepne ({e}). Uzywam trybu 'duch'.")
    ghost = ghost_mask(OrbComposer.SQ)
    invert = False
    use_seg = seg is not None

    print("Klawisze: q=wyjscie, +/-=rozmiar, u/j=gora/dol, m=lustro, g=duch, i=odwroc maske")
    print("Suwaki w oknie podgladu: Zoom % i Wysokosc.")
    t0 = time.perf_counter()
    n, ft = 0, time.perf_counter()
    win = "Palantir - podglad"
    TB_ZOOM, TB_Y = "Zoom %", "Wysokosc"
    if not args.no_preview:
        cv2.namedWindow(win)
        cv2.createTrackbar(TB_ZOOM, win, int(round(comp.zoom * 100)), 300, lambda v: None)
        cv2.setTrackbarMin(TB_ZOOM, win, 50)
        cv2.createTrackbar(TB_Y, win, int(round(comp.yoff * 100)) + 60, 120, lambda v: None)

    def sync_trackbars():
        cv2.setTrackbarPos(TB_ZOOM, win, int(round(comp.zoom * 100)))
        cv2.setTrackbarPos(TB_Y, win, int(round(comp.yoff * 100)) + 60)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            sq = square_crop(frame, OrbComposer.SQ)
            if use_seg:
                mask = seg(cv2.resize(sq, (256, 256), interpolation=cv2.INTER_AREA))
                if invert:
                    mask = 1.0 - mask
            else:
                mask = ghost
            out = comp.render(sq, mask, time.perf_counter() - t0)

            if vcam is not None:
                vcam.send(out)
            n += 1
            if time.perf_counter() - ft >= 5:
                print(f"~{n / (time.perf_counter() - ft):.1f} kl/s")
                n, ft = 0, time.perf_counter()

            if not args.no_preview:
                cv2.imshow(win, cv2.resize(out, (args.width // 2, args.height // 2)))
                k = cv2.waitKey(1) & 0xFF
                z = cv2.getTrackbarPos(TB_ZOOM, win) / 100.0
                y = (cv2.getTrackbarPos(TB_Y, win) - 60) / 100.0
                if z > 0 and abs(z - comp.zoom) > 0.005:
                    comp.set_zoom(z)
                if abs(y - comp.yoff) > 0.005:
                    comp.set_yoff(y)
                if k in (ord("q"), 27):
                    break
                elif k in (ord("+"), ord("=")):
                    comp.change_zoom(1.1)
                    sync_trackbars()
                elif k in (ord("-"), ord("_")):
                    comp.change_zoom(1 / 1.1)
                    sync_trackbars()
                elif k == ord("u"):
                    comp.move(0.05)
                    sync_trackbars()
                elif k == ord("j"):
                    comp.move(-0.05)
                    sync_trackbars()
                elif k == ord("m"):
                    mirror = not mirror
                elif k == ord("i"):
                    invert = not invert
                elif k == ord("g"):
                    use_seg = (not use_seg) and seg is not None

            if vcam is not None:
                vcam.sleep_until_next_frame()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if vcam is not None:
            vcam.close()
        if seg is not None:
            seg.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Ty w kuli na animowanym tle - kamera do Teams.")
    ap.add_argument("--tlo", default=os.path.join(HERE, "tlo.gif"), help="animowany GIF (albo zwykly obrazek)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zoom", type=float, default=1.0, help="rozmiar postaci w kuli")
    ap.add_argument("--yoff", type=float, default=0.0, help="przesuniecie postaci w gore (+) / dol (-)")
    ap.add_argument("--opacity", type=float, default=1.0,
                    help="krycie postaci: 1 = pelna, mniej = przezroczysta (np. 0.9)")
    ap.add_argument("--dym", type=float, default=0.0,
                    help="dym na srodku postaci: 0 = czysto (przy brzegu dym jest zawsze), 0.55 = mocny")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--ghost", action="store_true", help="bez wycinania postaci (tryb duch)")
    ap.add_argument("--no-vcam", action="store_true", help="tylko podglad, bez wirtualnej kamery")
    ap.add_argument("--vcam-device", default=None,
                    help="urzadzenie wirtualnej kamery (Linux, np. /dev/video10); domyslnie automatycznie")
    ap.add_argument("--no-preview", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
