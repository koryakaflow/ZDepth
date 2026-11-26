"""EffectCraft external helper protocol, version 1.

Requests arrive as JSON lines on stdin; replies go to stdout as JSON lines prefixed with
`@ec-helper `. Pixels travel through the frame exchange file named in `hello`: premultiplied
RGBA float32, 16 bytes per pixel, rows top to bottom. See docs/external-helpers.md in
storytold/effectcraft.
"""

import json
import os
import sys

import numpy as np

VERSION = 1
MAGIC = b"@ec-helper "


class Frames:
    """The frame exchange file. EffectCraft unlinks it once we have opened it, so the
    descriptor is the only handle; pread / pwrite keep working when the file grows."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR)

    def read(self, ref):
        w, h = ref["width"], ref["height"]
        n = w * h * 16
        buf = os.pread(self.fd, n, ref["offset"])
        if len(buf) != n:
            raise IOError(f"short read from the frame exchange file ({len(buf)} of {n} bytes)")
        return np.frombuffer(buf, dtype=np.float32).reshape(h, w, 4).copy()

    def write(self, ref, pixels):
        data = np.ascontiguousarray(pixels, dtype=np.float32).tobytes()
        view = memoryview(data)
        off = ref["offset"]
        while view:
            done = os.pwrite(self.fd, view, off)
            view, off = view[done:], off + done


class Connection:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer

    def send(self, msg):
        self.stdout.write(MAGIC + json.dumps(msg, separators=(",", ":")).encode() + b"\n")
        self.stdout.flush()

    def recv(self):
        """The next request, or None when stdin is closed."""
        while True:
            line = self.stdin.readline()
            if not line:
                return None
            line = line.strip()
            if line:
                return json.loads(line)


class Host:
    """Callbacks a render may make. EffectCraft can send a nested `render` / `padding` while
    we wait for the answer; those are served on the spot before we keep waiting."""

    def __init__(self, conn, dispatch):
        self.conn = conn
        self.dispatch = dispatch

    def _ask(self, msg, answer_op):
        self.conn.send(msg)
        while True:
            req = self.conn.recv()
            if req is None:
                raise EOFError("stdin closed while waiting for " + answer_op)
            if req.get("op") == answer_op:
                return req
            self.dispatch(req)

    def input_at(self, time):
        """The effect's input at another layer time, as an image ref (or None)."""
        return self._ask({"op": "input_at", "time": time}, "image")["image"]

    def layer_input(self, param, time):
        return self._ask({"op": "layer_input", "param": param, "time": time}, "image")["image"]

    def params_at(self, time):
        return self._ask({"op": "params_at", "time": time}, "params")["values"]
