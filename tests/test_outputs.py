"""Output modes, without a model: `~/.local/share/zdepth/venv/bin/python tests/test_outputs.py`"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zdepth import effect  # noqa: E402


def main():
    # Depth Matte: 1 inside [far, near], 0 beyond the softness, smooth in between.
    d = np.array([0.0, 0.4, 0.45, 0.5, 0.7, 1.0], dtype=np.float32)
    m = effect.depth_matte(d, far=0.5, near=1.0, softness=0.1)
    assert m[0] == 0 and m[1] == 0 and 0 < m[2] < 1 and m[3] == 1 and m[4] == 1 and m[5] == 1, m
    assert (effect.depth_matte(d, 0.5, 1.0, 0) == (d >= 0.5)).all()

    # Turbo: near black at 0, blue low, green in the middle, red high.
    zero, lo, mid, hi = effect.turbo(np.array([0.0, 0.15, 0.5, 0.85]))
    assert zero.max() < 0.2 and lo.argmax() == 2 and mid.argmax() == 1 and hi.argmax() == 0, (zero, lo, mid, hi)

    # Normals: a flat map faces the camera; a ramp nearer to the right tilts the normal left.
    flat = effect.normals(np.full((8, 8), 0.5, np.float32), 10)
    assert np.allclose(flat, [0.5, 0.5, 1.0])
    ramp = effect.normals(np.tile(np.linspace(0, 1, 8, dtype=np.float32), (8, 1)), 10)
    assert (ramp[..., 0] < 0.5).all() and np.allclose(ramp[..., 1], 0.5)

    # Video models: a frame EffectCraft can't pass on is held; frames that keep missing fail
    # the render instead of returning wrong depth. (The model is swapped for one that only
    # pulls every frame.)
    from zdepth import video

    video.clip_depth = lambda name, count, frame, *a, **k: [frame(i)[..., 0].astype(np.float16) for i in range(count)]
    rgb = np.full((4, 4, 3), 0.5, np.float32)
    req = {"instance": 1, "time": 0.0, "frame_rate": 10.0, "frame_range": [0.0, 1.0]}
    p = effect.parse_params([3, 0, 1, 0, 0, 100, 1])

    class Host:
        def __init__(self, gaps):
            self.gaps = gaps

        def input_at(self, t):
            return None if round(t * 10) in self.gaps else {"t": t}

    read = lambda ref: np.concatenate([rgb, np.ones((4, 4, 1), np.float32)], -1)
    d, _ = effect.Depth().clip(req, p, Host({4}), read, rgb)  # one gap: held
    assert d.shape == (4, 4)
    try:
        effect.Depth().clip(req, p, Host(set(range(3, 10))), read, rgb)
        raise AssertionError("missing frames were not reported")
    except RuntimeError as e:
        assert "could not pass frames" in str(e), e

    # Parameters saved before the output parameters existed still parse.
    p = effect.parse_params([0, 1, 1, 0, 0, 100, 1])
    assert p["output"] == "Depth" and p["matte_far"] == 0.5
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
