"""The ZDepth effect: manifest and render."""

import hashlib
import os
import sys
from collections import OrderedDict

import numpy as np

EFFECT_ID = "io.github.koryakaflow.zdepth"

# Appended only: projects store the index.
MODELS = ["Small", "Base", "Large", "Video Small", "Video Large"]
QUALITY = [("Draft (364)", 364), ("Fast (518)", 518), ("Detail (770)", 770), ("Max (1036)", 1036)]
NORMALIZE = ["Per Frame", "Per Clip"]
OUTPUTS = ["Depth", "Depth (Color)", "Normals", "Depth Matte"]

MANIFEST = {
    "api": 2,
    "id": EFFECT_ID,
    "name": "ZDepth",
    "category": "3D Channel",
    "version": "0.1.0",
    "author": "koryakaflow",
    "description": "Depth map from footage (Depth Anything V2; Video Depth Anything for flicker-free "
                   "depth over a whole clip). White is near. Small and Video Small are Apache-2.0; Base, "
                   "Large and Video Large are for non-commercial use only (CC-BY-NC-4.0).",
    "params": [
        {"id": "model", "name": "Model", "type": "popup", "options": MODELS, "default": 0},
        {"id": "quality", "name": "Quality", "type": "popup", "options": [q[0] for q in QUALITY], "default": 1},
        {"id": "normalize", "name": "Normalize", "type": "popup", "options": NORMALIZE, "default": 1},
        {"id": "invert", "name": "Invert (White Is Far)", "type": "checkbox", "default": False},
        {"id": "black", "name": "Black Point", "type": "slider", "default": 0, "min": 0, "max": 100, "decimals": 1},
        {"id": "white", "name": "White Point", "type": "slider", "default": 100, "min": 0, "max": 100, "decimals": 1},
        {"id": "keep_alpha", "name": "Preserve Alpha", "type": "checkbox", "default": True},
        # Appended after the first release's parameters: projects store values by position.
        {"id": "output", "name": "Output", "type": "popup", "options": OUTPUTS, "default": 0},
        {"id": "normal_strength", "name": "Normal Strength", "type": "slider", "default": 10, "min": 0,
         "max": 1000, "sliderMax": 100, "decimals": 1, "group": "Normals"},
        {"id": "matte_near", "name": "Matte Near", "type": "slider", "default": 100, "min": 0, "max": 100,
         "decimals": 1, "group": "Depth Matte"},
        {"id": "matte_far", "name": "Matte Far", "type": "slider", "default": 50, "min": 0, "max": 100,
         "decimals": 1, "group": "Depth Matte"},
        {"id": "matte_softness", "name": "Matte Softness", "type": "slider", "default": 5, "min": 0, "max": 100,
         "sliderMax": 25, "decimals": 1, "group": "Depth Matte"},
    ],
}

# Raw model output per input frame, so changing levels / invert doesn't run the model again.
CACHE_FRAMES = 24
# Video models: whole-clip depth kept for this many applied effects.
CLIP_SESSIONS = 2
# Video models: input frames EffectCraft may fail to pass on in a row before the pass gives up.
MAX_MISSING_FRAMES = 3
# Per Clip: frames sampled across the layer's source range, and the percentiles kept.
CLIP_SAMPLES = 48
CLIP_PERCENTILES = (1.0, 99.0)


def parse_params(flat):
    model, quality, normalize, invert, black, white, keep_alpha = flat[:7]
    output, normal_strength, matte_near, matte_far, matte_softness = (list(flat[7:12]) + [0, 10, 100, 50, 5][len(flat[7:12]):])
    return {
        "output": OUTPUTS[min(int(output), len(OUTPUTS) - 1)],
        "normal_strength": normal_strength,
        "matte_near": matte_near / 100.0,
        "matte_far": matte_far / 100.0,
        "matte_softness": matte_softness / 100.0,
        "model": MODELS[min(int(model), len(MODELS) - 1)],
        "short_side": QUALITY[min(int(quality), len(QUALITY) - 1)][1],
        "per_clip": int(normalize) == 1,
        "invert": bool(invert),
        "black": black / 100.0,
        "white": white / 100.0,
        "keep_alpha": bool(keep_alpha),
    }


def unpremultiply(px):
    a = px[..., 3:4]
    rgb = np.divide(px[..., :3], a, out=np.zeros_like(px[..., :3]), where=a > 0)
    return rgb, a[..., 0]


def clip_times(frame_range, frame_rate, samples=CLIP_SAMPLES):
    """Layer times of the frames to sample in [start, end), at most `samples`, evenly spread."""
    start, end = frame_range
    n = max(1, int(round((end - start) * frame_rate)))
    idx = np.unique(np.linspace(0, n - 1, min(n, samples)).round().astype(int))
    return [start + i / frame_rate for i in idx]


def is_video(model):
    return model.startswith("Video")


def resize(d, h, w):
    """Bilinear resize of a 2-D depth map."""
    if d.shape == (h, w):
        return d.astype(np.float32)
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(d.astype(np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


class Depth:
    """Raw depth (larger = nearer, arbitrary scale) with a per-frame cache, per-clip ranges, and
    whole-clip depth for the video models."""

    def __init__(self):
        self.frames = OrderedDict()
        self.ranges = {}  # (instance, model, short_side, frame_range) -> (lo, hi)
        self.clips = OrderedDict()  # (instance, model, short_side, frame_range, fps) -> [depth]

    def raw(self, rgb, model, short_side):
        key = hashlib.blake2b(rgb.tobytes(), digest_size=16, person=f"{model}{short_side}".encode()[:16]).digest()
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key]
        if os.environ.get("ZDEPTH_BACKEND") == "luma":  # tests, and the plumbing without a GPU
            d = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        else:
            from . import model as m

            d = m.predict(rgb, model, short_side)
        d = d.astype(np.float32)
        self.frames[key] = d
        while len(self.frames) > CACHE_FRAMES:
            self.frames.popitem(last=False)
        return d

    def clip_range(self, req, p, host, read):
        key = (req["instance"], p["model"], p["short_side"], tuple(req["frame_range"]))
        if key not in self.ranges:
            los, his = [], []
            for t in clip_times(req["frame_range"], req["frame_rate"]):
                ref = host.input_at(t)
                if ref is None:
                    continue
                rgb, alpha = unpremultiply(read(ref))
                d = self.raw(rgb, p["model"], p["short_side"])
                if (alpha > 0).any():
                    d = d[alpha > 0]
                lo, hi = np.percentile(d, CLIP_PERCENTILES)
                los.append(lo)
                his.append(hi)
            if not los:
                return None
            self.ranges[key] = (float(min(los)), float(max(his)))
            print(f"zdepth: clip range {self.ranges[key]} from {len(los)} frames", file=sys.stderr)
        return self.ranges[key]

    def clip(self, req, p, host, read, rgb):
        """Whole-clip depth from a video model, and the index of this frame in it."""
        fr, fps = req.get("frame_range"), req["frame_rate"]
        if not fr or fps <= 0:  # a still: a clip of one frame
            fr, fps = (req["time"], req["time"] + 1.0), 1.0
            frame = lambda i: rgb
        else:
            prev = [rgb]
            missing = [0]

            def frame(i):
                # EffectCraft answers with no image when it can't pass one on: every frame of
                # the pass stays in the exchange file (RAM, /dev/shm on Linux) until the render
                # ends, so a long clip at a large preview size can fill it. A black frame would
                # put a dark blot in the depth: hold the previous one, and give up when frames
                # keep missing rather than return wrong depth for the rest of the clip.
                ref = host.input_at(fr[0] + i / fps)
                if ref is None:
                    missing[0] += 1
                    if missing[0] > MAX_MISSING_FRAMES:
                        raise RuntimeError(
                            f"EffectCraft could not pass frames on from frame {i} (shared memory full?): "
                            "lower the preview resolution, or trim the layer")
                    print(f"zdepth: no input for frame {i}, holding the previous one", file=sys.stderr)
                    return prev[0]
                missing[0] = 0
                prev[0] = unpremultiply(read(ref))[0]
                return prev[0]
        key = (req["instance"], p["model"], p["short_side"], tuple(fr), fps)
        count = max(1, int(round((fr[1] - fr[0]) * fps)))
        if key not in self.clips:
            from . import video

            self.clips = OrderedDict((k, v) for k, v in self.clips.items() if k[0] != req["instance"])
            report = lambda done, total: print(f"zdepth: {p['model']}: {done}/{total} frames", file=sys.stderr)
            try:
                depth = video.clip_depth(p["model"], count, frame, p["short_side"], progress=report)
            except Exception as e:
                if type(e).__name__ != "OutOfMemoryError":
                    raise
                import torch

                torch.cuda.empty_cache()
                raise RuntimeError(f"not enough GPU memory for {p['model']} at {p['short_side']}: "
                                   "pick a lower Quality (Draft) or close other GPU apps") from None
            sample = np.concatenate([d[::4, ::4].ravel() for d in depth]).astype(np.float32)
            lo, hi = np.percentile(sample, CLIP_PERCENTILES)
            self.clips[key] = (depth, (float(lo), float(hi)))
            while len(self.clips) > CLIP_SESSIONS:
                self.clips.popitem(last=False)
        self.clips.move_to_end(key)
        depth, rng = self.clips[key]
        i = min(count - 1, max(0, int(round((req["time"] - fr[0]) * fps))))
        return depth[i], rng

    def forget(self, instance):
        self.ranges = {k: v for k, v in self.ranges.items() if k[0] != instance}
        self.clips = OrderedDict((k, v) for k, v in self.clips.items() if k[0] != instance)


def render(px, req, p, depth, host, read):
    rgb, alpha = unpremultiply(px)
    rng = None
    if is_video(p["model"]):
        d, clip_rng = depth.clip(req, p, host, read, rgb)
        d = resize(d, *alpha.shape)
        if p["per_clip"]:
            rng = clip_rng
    else:
        d = depth.raw(rgb, p["model"], p["short_side"])
        if p["per_clip"] and req.get("frame_range"):
            rng = depth.clip_range(req, p, host, read)
    lo, hi = rng or (float(d.min()), float(d.max()))
    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(d)
    if p["invert"]:
        d = 1.0 - d
    span = p["white"] - p["black"]
    if abs(span) < 1e-6:
        d = (d >= p["black"]).astype(np.float32)
    else:
        d = np.clip((d - p["black"]) / span, 0.0, 1.0)
    if p["output"] == "Depth Matte":
        out = px * depth_matte(d, p["matte_far"], p["matte_near"], p["matte_softness"])[..., None]
        return out.astype(np.float32)
    if p["output"] == "Depth (Color)":
        color = turbo(d)
    elif p["output"] == "Normals":
        color = normals(d, p["normal_strength"])
    else:
        color = np.repeat(d[..., None], 3, axis=-1)
    a = alpha if p["keep_alpha"] else np.ones_like(alpha)
    out = np.empty_like(px)
    out[..., :3] = color * a[..., None]
    out[..., 3] = a
    return out


def depth_matte(d, far, near, softness):
    """1 between `far` and `near` (depth 0..1, 1 = near), easing to 0 over `softness` outside."""
    lo, hi = min(far, near), max(far, near)
    if softness <= 0:
        return ((d >= lo) & (d <= hi)).astype(np.float32)
    t = np.minimum((d - (lo - softness)) / softness, ((hi + softness) - d) / softness)
    t = np.clip(t, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def turbo(x):
    """Google's Turbo colormap, polynomial approximation (Ruofei Du): x 0..1 -> RGB."""
    x = np.clip(x, 0.0, 1.0)[..., None]
    c = np.array([
        [0.13572138, 0.09140261, 0.10667330],
        [4.61539260, 2.19418839, 12.64194608],
        [-42.66032258, 4.84296658, -60.58204836],
        [132.13108234, -14.18503333, 110.36276771],
        [-152.94239396, 4.27729857, -89.90310912],
        [59.28637943, 2.82956604, 27.34824973],
    ], dtype=np.float32)
    rgb = c[5]
    for k in range(4, -1, -1):
        rgb = rgb * x + c[k]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def normals(d, strength):
    """Tangent-space normals from depth (1 = near) packed into RGB 0..1 (OpenGL: +Y up).
    Slopes are measured per image width, so the look doesn't change with preview resolution."""
    h, w = d.shape
    gy, gx = np.gradient(d.astype(np.float32))
    k = strength * w / 100.0
    n = np.stack([-gx * k, gy * k, np.ones_like(d)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)
