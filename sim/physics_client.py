"""Driver side of the Isaac physics worker (sim/physics_worker.py). Imports only numpy, so a driver that never renders
(sim/expert_rollout.py, sim/expert_pool.py) does not load torch: importing it through sim/rollout.py cost each expert
rollout 4-6.5 s of serial start-up before its worker was even launched (docs/throughput.md 8.4).

Physics(demo, ...) is one episode's worker, as before. Physics(None, log=...) starts a persistent worker that boots
Isaac once and then serves any number of episodes, one per episode() call (each on a freshly opened stage).
"""
import json
import os
import tempfile
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


NOSMI_SRC = HERE / "nosmi.c"
NOSMI_LIB = Path(os.environ.get("JENGA_CACHE") or Path.home() / ".cache/jenga").expanduser() / "libnosmi.so"  # jenga.paths.CACHE


_nosmi_lock = threading.Lock()  # sim/fleet_driver.py starts its workers from many threads at once


def nosmi_lib() -> str:
    """sim/nosmi.c built once (persistent cache): the worker's gpu.foundation probes skip /usr/bin/nvidia-smi."""
    with _nosmi_lock:
        if not NOSMI_LIB.exists() or NOSMI_LIB.stat().st_mtime < NOSMI_SRC.stat().st_mtime:
            import subprocess
            NOSMI_LIB.parent.mkdir(parents=True, exist_ok=True)
            tmp = NOSMI_LIB.with_suffix(f".{os.getpid()}.so")
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(tmp), str(NOSMI_SRC), "-ldl"], check=True)
            os.replace(tmp, NOSMI_LIB)
    return str(NOSMI_LIB)


def _msg(prefix: bytes, obj: dict) -> bytes:
    """A control message; its length is never a multiple of 128, unlike a block of 16-float64 action rows."""
    msg = prefix + json.dumps(obj).encode()
    return msg + b" " * (len(msg) % 128 == 0)


class Physics:
    """The Isaac worker of one episode (sim/physics_worker.py), or with demo=None a persistent worker for many
    (episode()). The constructor only starts it (boot about 10 s), so a driver can boot the next episode's worker while
    others run; connect() waits for it, bounded at 240 s. `init` (npz, row): start from that logged state instead of
    the demo's first row."""

    def __init__(self, demo: str | None, offset=None, log: Path = None, init=None, nosmi: bool = False):
        self.key, self.sock, self.log = os.urandom(16), os.path.join(tempfile.mkdtemp(), "physics.sock"), log
        self.conn = None
        cmd = [str(HERE / "isaac.sh"), str(HERE / "physics_worker.py"), demo or "-", self.sock] + \
              (["--offset", *(f"{x:.9f}" for x in offset)] if offset else []) + \
              (["--init", str(init[0]), str(init[1])] if init else [])
        # CPU PhysX needs no GPU. Hide them all: with a visible GPU Kit holds a 0.6 GB CUDA context on it, and its
        # Vulkan enumeration opens a graphics context on GPU 0 whatever CUDA_VISIBLE_DEVICES says.
        env = dict(os.environ, JENGA_PHYSICS_KEY=self.key.hex(), GPU="", CUDA_VISIBLE_DEVICES="",
                   VK_ICD_FILENAMES=os.devnull, VK_DRIVER_FILES=os.devnull)
        if nosmi:  # sim/nosmi.c: gpu.foundation's nvidia-smi probes return nothing instead of queueing in the driver
            env["LD_PRELOAD"] = ":".join(filter(None, [nosmi_lib(), env.get("LD_PRELOAD")]))
        import subprocess
        self.proc = subprocess.Popen(cmd, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)

    def _connect(self):
        deadline = time.monotonic() + 240
        while True:
            assert self.proc.poll() is None, f"physics worker exited {self.proc.returncode}, see {self.log}"
            assert time.monotonic() < deadline, f"physics worker not ready in 240 s, see {self.log}"
            try:
                self.conn = Client(self.sock, "AF_UNIX", authkey=self.key)
                return
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.5)

    def connect(self) -> tuple[dict, np.ndarray]:
        """-> header {block_names, block_fixed, initial_block_poses}, reset state."""
        self._connect()
        return json.loads(self.conn.recv_bytes()), np.frombuffer(self.conn.recv_bytes())

    def episode(self, demo: str, offset=None, init=None, env: dict | None = None) -> tuple[dict, np.ndarray]:
        """Persistent worker: start a new episode on a freshly opened stage, exactly as a worker started with these
        arguments would (`env`: the JENGA_* probe variables for this episode) -> header, reset state."""
        if self.conn is None:
            self._connect()
        self.conn.send_bytes(_msg(b"EPISODE:", dict(demo=str(demo), offset=[float(f"{x:.9f}") for x in offset] if offset else None, env=env or {},
                                                    init=[str(init[0]), int(init[1])] if init else None)))
        head = self.conn.recv_bytes()
        assert not head.startswith(b"ERROR:"), f"{head[6:].decode()} (see {self.log})"
        return json.loads(head), np.frombuffer(self.conn.recv_bytes())

    def step(self, actions: np.ndarray) -> np.ndarray:
        self.conn.send_bytes(actions.tobytes())
        return np.frombuffer(self.conn.recv_bytes()).reshape(len(actions), -1)

    def reinit(self, npz: str, row: int) -> np.ndarray:
        """Move the worker to another logged state of the same episode -> that state."""
        self.conn.send_bytes(_msg(b"INIT:", dict(npz=str(npz), row=int(row))))
        return np.frombuffer(self.conn.recv_bytes())

    def close(self):
        """Quit the worker; kill it if it is not done within 60 s or never connected."""
        try:
            self.conn.send_bytes(b"")
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
