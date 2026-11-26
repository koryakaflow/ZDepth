"""Entry point: `python -m zdepth.helper --effectcraft-helper`."""

import os
import sys
import threading
import time
import traceback

from . import __version__, effect
from .protocol import VERSION, Connection, Frames, Host

# Seconds without a render after which the models leave the GPU (0 keeps them loaded).
IDLE_UNLOAD = float(os.environ.get("ZDEPTH_IDLE_UNLOAD", "60"))
# One line per render in EffectCraft's log: ZDEPTH_DEBUG=1, or a file named `debug` in ZDepth's
# data folder (no need to change how EffectCraft is started).
DEBUG = os.environ.get("ZDEPTH_DEBUG") == "1" or os.path.exists(os.path.join(
    os.environ.get("ZDEPTH_HOME") or os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
                                                  "zdepth"), "debug"))


class Helper:
    def __init__(self, conn):
        self.conn = conn
        self.frames = None
        self.host = Host(conn, self.dispatch)
        self.depth = effect.Depth()
        # Held for a whole render (re-entrant: EffectCraft can nest renders in callbacks), so
        # the idle unloader never pulls a model out from under one.
        self.busy = threading.RLock()
        self.last_render = time.monotonic()
        if IDLE_UNLOAD > 0:
            threading.Thread(target=self.unload_when_idle, daemon=True).start()

    def unload_when_idle(self):
        from . import model

        while True:
            time.sleep(min(5.0, IDLE_UNLOAD))
            if time.monotonic() - self.last_render < IDLE_UNLOAD or not self.busy.acquire(blocking=False):
                continue
            try:
                if model.release():
                    print(f"zdepth: idle for {IDLE_UNLOAD:.0f} s, models unloaded", file=sys.stderr)
            finally:
                self.busy.release()

    def dispatch(self, req):
        op = req.get("op")
        try:
            if op == "hello":
                if req.get("version") != VERSION:
                    self.conn.send({"op": "error", "message": f"ZDepth speaks protocol {VERSION}"})
                    return
                self.frames = Frames(req["frames"])
                self.conn.send({"op": "hello", "version": VERSION, "name": "ZDepth",
                                "helper_version": __version__, "loads": []})
            elif op == "describe":
                self.conn.send({"op": "effects", "effects": [{"manifest": effect.MANIFEST, "padding": False}],
                                "errors": []})
            elif op == "load":
                self.conn.send({"op": "error", "message": "ZDepth loads no files"})
            elif op == "padding":
                self.conn.send({"op": "padding", "pixels": 0})
            elif op == "render":
                self.render(req)
            elif op == "ping":
                self.conn.send({"op": "pong"})
            elif op == "destroy":
                self.depth.forget(req["instance"])
            elif op == "shutdown":
                sys.exit(0)
            else:
                print(f"zdepth: ignoring unknown request {op!r}", file=sys.stderr)
        except SystemExit:
            raise
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            if op in ("render", "padding", "load", "describe"):
                self.conn.send({"op": "error", "message": f"{type(e).__name__}: {e}"})

    def render(self, req):
        if req["effect"] != effect.EFFECT_ID:
            raise ValueError(f"unknown effect {req['effect']}")
        with self.busy:
            try:
                ref = req["frame"]
                p = effect.parse_params(req["flat"])
                if DEBUG:
                    print(f"zdepth: render instance={req['instance']} time={req['time']:.4f} model={p['model']} "
                          f"quality={p['short_side']} per_clip={p['per_clip']} output={p['output']} "
                          f"size={ref['width']}x{ref['height']} range={req.get('frame_range')} "
                          f"fps={req['frame_rate']:.3f}", file=sys.stderr)
                px = self.frames.read(ref)
                out = effect.render(px, req, p, self.depth, self.host, self.frames.read)
                self.frames.write(ref, out)
            finally:
                self.last_render = time.monotonic()
        self.conn.send({"op": "done"})

    def serve(self):
        while (req := self.conn.recv()) is not None:
            self.dispatch(req)


def main():
    args = sys.argv[1:]
    if args[:1] == ["--setup"]:
        from . import model

        return model.setup(args[1:] or ["Small"])
    if "--effectcraft-helper" not in args:
        print("ZDepth is an EffectCraft effect helper: EffectCraft starts it "
              "(Effect > Scan for Effect Helpers).", file=sys.stderr)
        return 2
    # Anything a library prints must not end up in the protocol stream.
    conn = Connection(sys.stdin.buffer, sys.stdout.buffer)
    sys.stdout = sys.stderr
    Helper(conn).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
