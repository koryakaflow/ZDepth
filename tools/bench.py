"""Times ZDepth's models on this machine: ms per frame and peak GPU memory for each model,
quality and precision. Run with the helper's environment:

    ~/.local/share/zdepth/venv/bin/python tools/bench.py [--models Small,Video-Small] [--size 1080x1920]

Models that aren't downloaded yet are skipped (`effectcraft-helper-zdepth --setup <model>`).
Close other GPU apps for numbers that match a quiet machine.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zdepth import effect, model, video  # noqa: E402


def downloaded(name):
    from huggingface_hub import try_to_load_from_cache

    if name in model.MODELS:
        repo, rev = model.MODELS[name]
        return isinstance(try_to_load_from_cache(repo, "model.safetensors", revision=rev), str)
    repo, rev, file, _ = video.MODELS[name]
    return isinstance(try_to_load_from_cache(repo, file, revision=rev), str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(effect.MODELS))
    ap.add_argument("--quality", default="364,518,770,1036")
    ap.add_argument("--precision", default="fp16,fp32")
    ap.add_argument("--size", default="1080x1920", help="frame WxH")
    ap.add_argument("--frames", type=int, default=5, help="timed frames (still models)")
    a = ap.parse_args()

    import torch

    w, h = map(int, a.size.split("x"))
    rng = np.random.default_rng(0)
    # a smooth image with some structure, so attention and convolutions see real-ish values
    yy, xx = np.mgrid[0:h, 0:w] / max(w, h)
    rgb = np.stack([np.sin(xx * 9) * 0.3 + 0.5, yy, np.cos(yy * 7) * 0.3 + 0.5], -1).astype(np.float32)
    rgb += rng.normal(0, 0.02, rgb.shape).astype(np.float32)

    dev = model.device()
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
    print(f"device: {name}, torch {torch.__version__}, frame {w}x{h}")
    print(f"{'model':<12} {'quality':>7} {'prec':>5} {'ms/frame':>9} {'peak GB':>8}  note")
    for m in a.models.split(","):
        m = m.replace("-", " ").title()
        if not downloaded(m):
            print(f"{m:<12} not downloaded, skipped")
            continue
        for q in map(int, a.quality.split(",")):
            for prec in a.precision.split(","):
                os.environ["ZDEPTH_PRECISION"] = prec
                note = ""
                try:
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    if m in model.MODELS:
                        model.predict(rgb, m, q)  # load + warm up
                        t = time.perf_counter()
                        for _ in range(a.frames):
                            d = model.predict(rgb, m, q)
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        ms = (time.perf_counter() - t) / a.frames * 1000
                    else:
                        n = 76  # 3 windows
                        video.clip_depth(m, 32, lambda i: rgb, q)  # load + warm up
                        t = time.perf_counter()
                        d = video.clip_depth(m, n, lambda i: rgb, q)[-1]
                        ms = (time.perf_counter() - t) / n * 1000
                    if not np.isfinite(np.asarray(d, np.float32)).all():
                        note = "NON-FINITE"
                    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
                    print(f"{m:<12} {q:>7} {prec:>5} {ms:>9.0f} {peak:>8.2f}  {note}", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{m:<12} {q:>7} {prec:>5} {'-':>9} {'-':>8}  out of GPU memory", flush=True)
                    model.unload()
                    video.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
