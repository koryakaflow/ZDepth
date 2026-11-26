"""Video Depth Anything: temporally consistent depth for a whole clip.

The network sees 32 frames at a time. Windows advance by 22 frames; the first 10 inputs of a
window are keyframes of the previous one, and each window's output is fitted (scale and shift)
to the previous one through them, with the 8 frames where windows meet cross-faded. This is
`VideoDepthAnything.infer_video_depth` from the original repository, rewritten to take frames
one at a time (EffectCraft hands them over through `input_at`) instead of a whole decoded clip
in memory. Depth is kept at the network's resolution and scaled up per frame on output.
"""

import sys
import threading

import numpy as np

from .model import MEAN, PATCH, STD, device, precision

INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8
ALIGN_LEN = OVERLAP - INTERP_LEN
STEP = INFER_LEN - OVERLAP
ENCODER_CHUNK = 8

# name -> (repository, revision, file, encoder config)
MODELS = {
    "Video Small": ("depth-anything/Video-Depth-Anything-Small", "256875362cff76724b920335dfb4b29dd611f66e",
                    "video_depth_anything_vits.pth",
                    {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}),
    "Video Large": ("depth-anything/Video-Depth-Anything-Large", "7aafbcb5c6af0bac741aad2b6471894fb4761afa",
                    "video_depth_anything_vitl.pth",
                    {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}),
}
NONCOMMERCIAL = {"Video Large"}

_lock = threading.Lock()
_loaded = {}


def download(name):
    from huggingface_hub import hf_hub_download

    repo, rev, file, _ = MODELS[name]
    return hf_hub_download(repo, file, revision=rev)


def _load(name):
    with _lock:
        if name in _loaded:
            return _loaded[name]
        import torch

        from . import model as still
        from .vendor.video_depth_anything.dinov2 import DINOv2
        from .vendor.video_depth_anything.dpt_temporal import DPTHeadTemporal

        class VideoDepthAnything(torch.nn.Module):
            def __init__(self, encoder, features, out_channels):
                super().__init__()
                self.encoder = encoder
                self.layers = {"vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}[encoder]
                self.pretrained = DINOv2(model_name=encoder)
                self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, False, out_channels=out_channels,
                                            use_clstoken=False, num_frames=INFER_LEN, pe="ape")

            def forward(self, x):
                b, t, c, h, w = x.shape
                # The encoder sees frames independently: run it a few at a time to bound memory.
                flat = x.flatten(0, 1)
                parts = [self.pretrained.get_intermediate_layers(flat[i:i + ENCODER_CHUNK], self.layers,
                                                                 return_class_token=True)
                         for i in range(0, flat.shape[0], ENCODER_CHUNK)]
                feats = [tuple(torch.cat([p[l][j] for p in parts]) for j in range(2))
                         for l in range(len(self.layers))]
                del parts, flat
                # a list the head empties as it goes, so the features don't outlive their use
                d = self.head(feats, h // PATCH, w // PATCH, t)[0]
                d = torch.nn.functional.interpolate(d, size=(h, w), mode="bilinear", align_corners=True)
                return torch.relu(d).squeeze(1).unflatten(0, (b, t))

        _loaded.clear()
        still.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        repo, rev, file, cfg = MODELS[name]
        dev = device()
        print(f"zdepth: loading {repo} on {dev}", file=sys.stderr)
        m = VideoDepthAnything(**cfg)
        m.load_state_dict(torch.load(download(name), map_location="cpu", weights_only=True), strict=True)
        m = m.to(dev).eval()
        _loaded[name] = m
        return m


def unload():
    with _lock:
        _loaded.clear()


def input_size(h, w, short_side):
    """The original's sizing: shorter side to `short_side` (less for frames wider than 16:9, to
    bound memory), aspect kept, multiples of the patch size."""
    ratio = max(h, w) / min(h, w)
    if ratio > 1.78:
        short_side = round(int(short_side * 1.777 / ratio) / PATCH) * PATCH
    s = short_side / min(h, w)
    return (max(PATCH, round(h * s / PATCH) * PATCH), max(PATCH, round(w * s / PATCH) * PATCH))


def _scale_shift(pred, target):
    """Least squares `scale * pred + shift ~= target` (compute_scale_and_shift_full)."""
    p = pred.astype(np.float64).ravel()
    t = target.astype(np.float64).ravel()
    a00, a01, a11 = (p * p).sum(), p.sum(), float(p.size)
    b0, b1 = (p * t).sum(), t.sum()
    det = a00 * a11 - a01 * a01
    if det == 0:
        return 1.0, 0.0
    return (a11 * b0 - a01 * b1) / det, (-a01 * b0 + a00 * b1) / det


def clip_depth(name, count, frame, short_side=518, progress=None):
    """Depth for `count` frames, `frame(i)` giving straight RGB (HxWx3, 0..1) of frame i in
    order. Returns a list of float16 arrays at the network's resolution, larger = nearer, on
    one scale for the whole clip."""
    import torch
    import torch.nn.functional as F

    model = _load(name)
    dev = next(model.parameters()).device
    mean = torch.tensor(MEAN, device=dev).view(3, 1, 1)
    std = torch.tensor(STD, device=dev).view(3, 1, 1)
    size = None
    last = None

    def prepared(i):
        nonlocal size, last
        if i >= count:  # the original pads the clip with copies of its last frame
            return last
        rgb = frame(i)
        if size is None:
            size = input_size(rgb.shape[0], rgb.shape[1], short_side)
        x = torch.from_numpy(np.clip(rgb, 0.0, 1.0)).to(dev).permute(2, 0, 1)[None]
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False)[0]
        last = (x - mean) / std
        return last

    aligned = []
    ref_align = []
    pre_input = None
    half = precision(dev)
    for start in range(0, count, STEP):
        if pre_input is None:
            cur = [prepared(start + i) for i in range(INFER_LEN)]
        else:
            cur = [pre_input[k] for k in KEYFRAMES] + [prepared(start + i) for i in range(OVERLAP, INFER_LEN)]
        x = torch.stack(cur)[None]
        with torch.inference_mode():
            with torch.autocast(device_type=dev.type, enabled=half):
                out = model(x)[0].float()
            if half and not torch.isfinite(out).all():
                print("zdepth: fp16 gave non-finite depth, redoing the window in fp32", file=sys.stderr)
                out = model(x)[0].float()
            out = out.cpu().numpy()
        pre_input = cur

        if not aligned:
            aligned = list(out)
            ref_align = [out[k] for k in KEYFRAMES[:ALIGN_LEN]]
        else:
            s, t = _scale_shift(np.concatenate(out[:ALIGN_LEN]), np.concatenate(ref_align))
            fit = lambda d: np.maximum(d * s + t, 0.0)
            post = [fit(out[i]) for i in range(ALIGN_LEN, OVERLAP)]
            for i, w in enumerate(np.linspace(0.0, 1.0, INTERP_LEN)):
                aligned[-INTERP_LEN + i] = aligned[-INTERP_LEN + i] * (1 - w) + post[i] * w
            aligned += [fit(out[i]) for i in range(OVERLAP, INFER_LEN)]
            ref_align = ref_align[:1] + [fit(out[k]) for k in KEYFRAMES[1:ALIGN_LEN]]
        # frames before the cross-fade zone are final: keep them compact
        for i in range(max(0, len(aligned) - INFER_LEN - INTERP_LEN), len(aligned) - INTERP_LEN):
            aligned[i] = aligned[i].astype(np.float16)
        if progress:
            progress(min(count, len(aligned)), count)
    return [d.astype(np.float16) for d in aligned[:count]]
