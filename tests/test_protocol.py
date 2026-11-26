"""Drives the helper the way EffectCraft does: hello, describe, render (answering `input_at`
callbacks), ping, shutdown. The depth model is swapped for luminance (ZDEPTH_BACKEND=luma).

Run with any Python that has numpy, e.g. the helper's: `~/.local/share/zdepth/venv/bin/python tests/test_protocol.py`
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The Python side of the helper, started the way the launcher starts it (the launcher itself
# only picks the environment). Any Python with numpy runs this test.
HELPER = [sys.executable, "-I", "-c",
          "import sys; sys.path.insert(0, sys.argv.pop(1)); from zdepth.helper import main; sys.exit(main())", ROOT]
MAGIC = "@ec-helper "
W, H = 64, 32
FPS = 24.0
# model, quality, normalize, invert, black, white, keep_alpha
PER_FRAME = [0, 1, 0, 0, 0, 100, 1]
PER_CLIP = [0, 1, 1, 0, 0, 100, 1]


def solid(v):
    px = np.full((H, W, 4), v, dtype=np.float32)
    px[..., 3] = 1.0
    return px


class Host:
    """A stand-in for EffectCraft: the frame exchange file and a clip of solid frames."""

    def __init__(self, f, clip):
        self.f = f
        self.clip = clip  # frame index -> pixels
        self.p = subprocess.Popen([*HELPER, "--effectcraft-helper"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  env=dict(os.environ, ZDEPTH_BACKEND="luma"))
        self.callbacks = []

    def put(self, offset, px):
        self.f.seek(offset)
        self.f.write(np.ascontiguousarray(px, dtype=np.float32).tobytes())
        self.f.flush()
        return {"offset": offset, "width": px.shape[1], "height": px.shape[0], "origin": [0, 0], "scale": 1}

    def get(self, ref):
        self.f.seek(ref["offset"])
        n = ref["width"] * ref["height"]
        return np.frombuffer(self.f.read(n * 16), dtype=np.float32).reshape(ref["height"], ref["width"], 4)

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def call(self, msg):
        self.send(msg)
        while True:
            line = self.p.stdout.readline()
            assert line, "helper exited"
            if not line.startswith(MAGIC):
                continue
            r = json.loads(line[len(MAGIC):])
            if r["op"] == "input_at":
                self.callbacks.append(r["time"])
                i = int(round(r["time"] * FPS))
                img = self.put(1 << 20, self.clip[i]) if i in self.clip else None
                self.send({"op": "image", "image": img})
                continue
            return r

    def render(self, frame_index, flat, instance=7, effect="io.github.koryakaflow.zdepth"):
        ref = self.put(128, self.clip[frame_index])  # EffectCraft picks offsets; honour one
        r = self.call({"op": "render", "effect": effect, "instance": instance, "time": frame_index / FPS,
                       "frame": ref, "layer_size": [W, H], "flat": flat, "texts": [], "frame_rate": FPS,
                       "frame_range": [0.0, len(self.clip) / FPS]})
        return r, self.get(ref)


def main():
    rng = np.random.default_rng(1)
    noisy = rng.random((H, W, 4), dtype=np.float32)
    noisy[..., :3] *= noisy[..., 3:4]  # premultiplied
    clip = {0: solid(0.2), 1: solid(0.5), 2: solid(0.8), 3: noisy}

    with tempfile.NamedTemporaryFile(dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as f:
        f.truncate(2 << 20)
        host = Host(f, clip)

        r = host.call({"op": "hello", "version": 1, "frames": f.name})
        assert r["op"] == "hello" and r["version"] == 1, r
        (fx,) = host.call({"op": "describe"})["effects"]
        assert fx["manifest"]["id"] == "io.github.koryakaflow.zdepth"

        # Per frame: a premultiplied, gray result with the input's alpha.
        r, out = host.render(3, PER_FRAME)
        assert r == {"op": "done"}, r
        assert not host.callbacks, "per frame needs no other frames"
        assert np.allclose(out[..., 3], noisy[..., 3]), "alpha changed"
        assert np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2]), "not gray"
        assert (out[..., 0] <= out[..., 3] + 1e-6).all(), "not premultiplied"

        # Per clip: the middle solid frame sits in the middle of the clip's range.
        clip.pop(3)
        r, out = host.render(1, PER_CLIP, instance=8)
        assert r == {"op": "done"}, r
        assert sorted(round(t * FPS) for t in host.callbacks) == [0, 1, 2], host.callbacks
        assert np.allclose(out[..., 0], 0.5, atol=1e-3), out[0, 0]
        # ...and the range is kept: no callbacks the second time.
        host.callbacks.clear()
        r, out = host.render(2, PER_CLIP, instance=8)
        assert not host.callbacks and np.allclose(out[..., 0], 1.0, atol=1e-3), (host.callbacks, out[0, 0])

        r, _ = host.render(1, PER_FRAME, effect="nope")
        assert r["op"] == "error", r
        assert host.call({"op": "ping"}) == {"op": "pong"}

        host.send({"op": "shutdown"})
        host.p.stdin.close()
        assert host.p.wait(timeout=10) == 0
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
