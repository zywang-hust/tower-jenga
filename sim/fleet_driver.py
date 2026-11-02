"""Many closed-loop episodes on one GPU in one process: sim/rollout.py's episode loop in N threads, sharing one
renderer that batches their camera requests, against one policy server (scripts/serve_policy.py, unchanged wire
path). The envs run in lockstep groups (--groups, default 2): a group renders as one batch and its policy
requests reach the server together as one forward, which runs while the other group renders and steps physics.
Each env thread owns one persistent Isaac worker (sim/physics_client.py Physics(None), with the nosmi shim) that
boots once and serves its episodes one after another (a new episode reopens the stage, about 1 s, instead of an Isaac
boot of 10-90 s); physics is bit-identical to a worker per episode (docs/throughput.md 8.4). The first episodes of
all threads start in the same lockstep round, once every worker has set up its episode and the policy server answers,
so they also end together: no ramp of small batches while workers boot, none while staggered episodes drain.
--fresh-workers restores the old scheme (one worker per episode, booted ahead with --prefetch).

Jobs are lines `<demo.hdf5> <seed> [dx dy dpsi]` (the optional layout offset of sim/rollout.py --offset); each
writes <out>/<variant>_<episode>_s<seed>.json (.npz, .log, .physics.log) exactly as sim/rollout.py does, and jobs
whose .json exists are skipped. Prints FLEET_DRIVER_DONE when every job has ended; a failed job leaves no .json, so
a rerun retries it.

Run with the renderer's venv:
  CUDA_VISIBLE_DEVICES=2 runs/simdev/render_venv/bin/python sim/fleet_driver.py <jobs.txt> <out_dir>
      --url ws://127.0.0.1:9202 [--envs 48] [--groups 2] [--seconds 210] [--video-every N] [--fresh-workers]
The driver may start before the server: it boots the workers and loads the renderer while the server compiles, and
waits for the server (bounded at 900 s) before the first episode.
"""
import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import Future
from itertools import zip_longest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fk import ArmFK  # noqa: E402
from render import CAMERAS, ROTATED, STATION, Renderer  # noqa: E402
from rollout import REFINE, Physics, episode  # noqa: E402
from websockets.sync.client import connect  # noqa: E402


class Started:
    """An episode already set up on a persistent worker, in the shape rollout.episode() expects of a Physics:
    connect() returns the header and reset state, close() leaves the worker running for the next episode."""

    def __init__(self, physics: Physics, head: tuple):
        self.physics, self.head = physics, head

    def connect(self):
        return self.head

    def step(self, actions):
        return self.physics.step(actions)

    def close(self):
        pass


def wait_server(url: str, timeout: float = 900.0):
    """Block until the policy server accepts a connection (it listens only once its buckets are compiled)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with connect(url, max_size=64 << 20, compression=None, proxy=None, open_timeout=10):
                return
        except Exception:  # noqa: BLE001
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def probe_env() -> dict:
    """The JENGA_* probe variables of this driver, passed to every episode of a persistent worker, which a worker per
    episode would have inherited."""
    return {k: v for k, v in os.environ.items() if k.startswith("JENGA_") and k not in ("JENGA_PHYSICS_KEY", "JENGA_NOSMI")}


class Lockstep:
    """The envs of one group step together: sync() returns once every current member has arrived, so the group
    renders in one batch and its policy requests reach the server together (one forward of the group's size).
    Members leave when their episode ends and join when the next starts; a late joiner waits for the next round."""

    def __init__(self):
        self.cond, self.members, self.arrived, self.round = threading.Condition(), 0, 0, 0

    def join(self):
        with self.cond:
            self.members += 1

    def leave(self):
        with self.cond:
            self.members -= 1
            self._release_if_complete()

    def sync(self) -> int:
        """-> number of members in this round."""
        with self.cond:
            r, self.arrived = self.round, self.arrived + 1
            n = self.members
            self._release_if_complete()
            self.cond.wait_for(lambda: self.round != r)
            return n

    def _release_if_complete(self):
        if self.arrived >= self.members:
            self.arrived, self.round = 0, self.round + 1
            self.cond.notify_all()


class RenderService:
    """Batches the render requests of one lockstep group: a batch closes when the `expect` requests of its round
    are in (or `window_ms` after the first). The GPU call is serialised across groups by `lock` (one nvdiffrast
    context). Returns uint8 [4, H, W, 3] in wire orientation."""

    def __init__(self, renderer: Renderer, lock: threading.Lock, window_ms: float = 50.0):
        self.r, self.lock, self.window = renderer, lock, window_ms / 1e3
        self.work: queue.Queue = queue.Queue()
        self.flip = [c in ROTATED for c in CAMERAS]
        self.calls = self.envs = 0
        self.busy = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def __call__(self, poses: np.ndarray, expect: int) -> np.ndarray:
        fut = Future()
        self.work.put((poses, expect, fut))
        return fut.result()

    def _render(self, P: np.ndarray) -> np.ndarray:
        nvtx = self.r.torch.cuda.nvtx
        with nvtx.range("render_lock"):
            self.lock.acquire()
        try:
            with nvtx.range(f"render {len(P)}"):
                img = self.r.shade(self.r.torch.as_tensor(P.astype(np.float32), device=self.r.dev))  # dataset orientation
                img[:, self.flip] = img[:, self.flip].flip(2, 3)  # the external cameras go out raw, turned 180 degrees
            with nvtx.range("readback"):
                out = self.r.torch.empty(img.shape, dtype=img.dtype, pin_memory=True)  # cached: reused once all views die
                return out.copy_(img).numpy()
        finally:
            self.lock.release()

    def _loop(self):
        nvtx = self.r.torch.cuda.nvtx
        while True:
            batch = [self.work.get()]
            nvtx.range_push("render_close")
            deadline = time.perf_counter() + self.window
            while len(batch) < batch[0][1]:
                try:
                    batch.append(self.work.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            nvtx.range_pop()
            t0 = time.perf_counter()
            try:
                out = self._render(np.stack([p for p, _, _ in batch]))
                for (_, _, fut), img in zip(batch, out):
                    fut.set_result(img)
            except BaseException as exc:  # noqa: BLE001
                for _, _, fut in batch:
                    fut.set_exception(exc)
            self.busy += time.perf_counter() - t0
            self.calls, self.envs = self.calls + 1, self.envs + len(batch)


class Sessions:
    """Policy-server sessions reused across trials of the same variant (a session is bound to its variant header).
    The server keeps a detached session for 300 s and holds at most 64, so a fresh session per short episode would
    run into session_capacity. With the jobs interleaved over variants (main) a driver opens about envs + variants."""

    def __init__(self):
        self.free, self.lock = defaultdict(list), threading.Lock()

    def take(self, variant: str) -> str:
        with self.lock:
            return self.free[variant].pop() if self.free[variant] else str(uuid.uuid4())

    def give(self, variant: str, sid: str):
        with self.lock:
            self.free[variant].append(sid)


def name(demo: str, seed: int) -> str:
    """Output name, as scripts/fleet.sh documents it: <variant>_<episode number>_s<seed>."""
    return f"{Path(demo).parent.name}_{Path(demo).name.split('_')[1]}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path, help="lines: <demo.hdf5> <seed> [dx dy dpsi]")
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="ws://127.0.0.1:9101")
    ap.add_argument("--envs", type=int, default=48,
                    help="episodes in flight; with the variants they must stay within the server's 64 sessions")
    ap.add_argument("--groups", type=int, default=2, help="lockstep groups: the envs of a group render and request "
                    "together; one group's policy forward overlaps the other's render and physics")
    ap.add_argument("--fresh-workers", action="store_true",
                    help="one Isaac worker per episode (booted ahead with --prefetch) instead of one persistent worker per env")
    ap.add_argument("--prefetch", type=int, default=32, help="--fresh-workers: workers booted ahead of the episodes in flight")
    ap.add_argument("--seconds", type=float, default=210.0)
    ap.add_argument("--video-every", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(w[0], int(w[1]), [float(x) for x in w[2:5]] or None)
            for w in (line.split() for line in args.jobs.read_text().splitlines() if line.strip())]
    todo = defaultdict(list)
    for d, s, o in jobs:
        if not (args.out / f"{name(d, s)}.json").exists():
            todo[Path(d).parent.name].append((d, s, o))
    # round-robin over variants, so each variant has about envs / variants episodes in flight (Sessions)
    todo = [j for group in zip_longest(*todo.values()) for j in group if j]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run with {args.envs} in flight", flush=True)

    workers: list[Physics] = []
    ready: queue.Queue = queue.Queue(maxsize=args.prefetch)
    stop = threading.Event()
    n_threads = args.envs if args.fresh_workers else min(args.envs, len(todo))
    jobs_left: queue.Queue = queue.Queue()
    for j in todo:
        jobs_left.put(j)

    def boot():  # --fresh-workers. Popen returns at once; the queue bound keeps `prefetch` booted workers waiting
        for d, s, o in todo:
            p = Physics(d, o, args.out / f"{name(d, s)}.physics.log")
            workers.append(p)
            while not stop.is_set():
                try:
                    ready.put((d, s, o, p), timeout=1.0)
                    break
                except queue.Full:
                    pass
            if stop.is_set():
                return
        for _ in range(args.envs):
            ready.put(None)

    nosmi, env = os.environ.get("JENGA_NOSMI", "1") != "0", probe_env()
    (args.out / "servers").mkdir(exist_ok=True)

    slots: list[Physics | None] = [None] * n_threads  # persistent mode: each thread's worker

    def persistent(i: int) -> Physics:
        """(Re)start thread i's persistent worker (boots in the background)."""
        if slots[i] is not None:
            slots[i].close()
        log = args.out / "servers" / f"physics_{os.getpid()}_{i}.log"
        if log.exists():
            log.rename(log.with_suffix(f".{int(time.time())}.log"))
        slots[i] = Physics(None, log=log, nosmi=nosmi)
        workers.append(slots[i])
        return slots[i]

    loaded = threading.Event()  # the renderer is up
    t0 = None

    def go():  # persistent mode: every thread's first episode starts in the same round, after renderer and server
        nonlocal t0
        loaded.wait()
        wait_server(args.url)
        t0 = time.perf_counter()
        threading.Thread(target=monitor, daemon=True).start()
        print(f"{n_threads} episodes set up, start", flush=True)

    start = threading.Barrier(max(n_threads, 1), action=go)
    sessions, done, failed, lock = Sessions(), [], [], threading.Lock()
    groups = [Lockstep() for _ in range(args.groups)]

    def fail(n: str):
        (args.out / f"{n}.log").write_text(traceback.format_exc())
        with lock:
            failed.append(n)
            print(f"{n} FAILED, see {args.out / n}.log", flush=True)

    def next_job(i: int, first: bool):
        """-> (demo, seed, offset, physics for episode()) or None when no job is left."""
        if args.fresh_workers:
            return ready.get()
        while True:
            try:
                d, s, o = jobs_left.get_nowait()
            except queue.Empty:
                if first:
                    try:
                        start.wait(timeout=1200)
                    except threading.BrokenBarrierError:
                        pass
                return None
            p = slots[i]
            try:
                head = p.episode(d, o, env=env)
                err = None
            except Exception:  # noqa: BLE001  the episode fails; the worker is replaced below
                err = traceback.format_exc()
            if first:  # also on a failed set-up, so the others are not held to the timeout
                first = False
                try:
                    start.wait(timeout=1200)
                except threading.BrokenBarrierError:
                    pass
            if err is None:
                return d, s, o, Started(p, head)
            (args.out / f"{name(d, s)}.log").write_text(err)
            with lock:
                failed.append(name(d, s))
                print(f"{name(d, s)} FAILED (set-up), see {args.out / name(d, s)}.log", flush=True)
            persistent(i)

    def run(i: int):
        group = groups[i % args.groups]
        if not args.fresh_workers:
            persistent(i)
        first = True
        while (job := next_job(i, first)) is not None:
            first, render = False, renders[i % args.groups]
            d, s, o, ph = job
            n, variant, sid, joined = name(d, s), Path(d).parent.name, None, False

            def images(q: np.ndarray, blocks: dict, dims: dict | None = None) -> dict:
                nonlocal joined
                if not joined:  # join at the first frame, so the group does not wait for this worker's connect
                    group.join()
                    joined = True
                with nvtx.range("poses"):
                    poses = renderer.poses(fk.link_poses(q), blocks, dims)
                with nvtx.range("sync"):
                    expect = group.sync()
                with nvtx.range("render_wait"):
                    return dict(zip(CAMERAS, render(poses, expect)))

            try:
                sid = sessions.take(variant)
                w0 = time.perf_counter()
                r = episode(d, args.out / f"{n}.json", ph, images, url=args.url, seconds=args.seconds, seed=s,
                            offset=o, video_every=args.video_every, session_id=sid)
                sessions.give(variant, sid)
                (args.out / f"{n}.log").write_text(f"ROLLOUT_DONE {json.dumps(r)}\n")
                with lock:
                    done.append(r)
                    print(f"{n} ok {time.perf_counter() - w0:.0f} s (lhpr {r['lhpr']:.2f}) "
                          f"[{len(done)} done, {(time.perf_counter() - t0) / 60:.1f} min]", flush=True)
            except Exception:  # noqa: BLE001
                fail(n)
                if not args.fresh_workers:  # a worker in an unknown state: replace it
                    persistent(i)
            finally:
                if joined:
                    group.leave()
        if not args.fresh_workers:
            slots[i].close()

    def stats():
        return sum(r.envs for r in renders), sum(r.calls for r in renders), sum(r.busy for r in renders)

    def monitor():  # steady-state rate: every policy request renders one 4-camera set, so envs counts requests
        last = (time.perf_counter(), 0, 0, 0.0)
        while not stop.wait(60.0):
            now = (time.perf_counter(), *stats())
            dt, envs, calls, busy = (b - a for a, b in zip(last, now))
            print(f"t={now[0] - t0:.0f}s {envs / dt:.1f} req/s, render batch {envs / max(calls, 1):.1f}, "
                  f"render busy {busy / dt * 100:.0f}%", flush=True)
            last = now

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(n_threads)]
    booter = threading.Thread(target=boot, daemon=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # unwinds the main thread through the finally below
    t_start = time.perf_counter()
    try:
        renderer = None
        if args.fresh_workers:
            booter.start()  # the first workers boot while the renderer loads
        else:  # the threads boot their workers and set up their first episodes while the renderer loads
            for t in threads:
                t.start()
        t_load = time.perf_counter()
        renderer, fk = Renderer(STATION, refine=REFINE), ArmFK(STATION)
        nvtx = renderer.torch.cuda.nvtx
        gpu = threading.Lock()
        renders = [RenderService(renderer, gpu) for _ in groups]
        for e in {1, -(-n_threads // args.groups)}:  # compile the shading now (E=1 specialises, the rest is dynamic)
            renders[0]._render(np.repeat(renderer.poses()[None], e, 0))
        print(f"renderer ready in {time.perf_counter() - t_load:.0f} s", flush=True)
        loaded.set()
        if args.fresh_workers:
            wait_server(args.url)
            t0 = time.perf_counter()
            threading.Thread(target=monitor, daemon=True).start()
            for t in threads:
                t.start()
        for t in threads:
            while t.is_alive():
                t.join(1.0)
    finally:  # normally every worker has quit; on a signal, kill the rest
        stop.set()
        if booter.is_alive():
            booter.join(5.0)  # no worker starts after this
        for p in list(workers):
            if p.proc.poll() is None:
                p.proc.kill()
    wall = time.perf_counter() - (t0 or t_start)
    med = {k: float(np.median([r["timing_s"][k] for r in done])) for k in ("render", "policy", "physics")} if done else {}
    envs, calls, busy = stats()
    print("FLEET_DRIVER_DONE", json.dumps(dict(
        done=len(done), failed=failed, wall_s=round(wall, 1), per_min=round(len(done) / wall * 60, 2),
        total_s=round(time.perf_counter() - t_start, 1),
        episode_wall_s=float(np.median([r["wall_s"] for r in done])) if done else None, timing_s=med,
        render_calls=calls, render_mean_batch=round(envs / max(calls, 1), 2), render_busy_s=round(busy, 1))), flush=True)


if __name__ == "__main__":
    main()
