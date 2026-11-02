"""Diagnostic record of what the policy server saw and returned, for failure analysis after an evaluation.

One file per trial, <root>/<task_variant>/<trial_id>.rec: length-prefixed msgpack records (a reset record,
then one per infer with the four camera images as JPEG q95, the joint state, step_index, the returned actions and
timing). Writing runs on background threads and never blocks or fails a request: when
a writer falls behind, records are dropped and counted (the next record of that trial carries `dropped`).

Evaluation observations are for diagnosis only: training or tuning on them would fit the frozen evaluation scenes.
"""
from __future__ import annotations

import logging
import queue
import struct
import threading
from pathlib import Path

import cv2
import numpy as np

from jenga.server import pack, unpack

log = logging.getLogger(__name__)


class Recorder:
    def __init__(self, root: Path, *, writers: int = 4, depth: int = 512, jpeg_quality: int = 95):
        self.root, self.quality = Path(root), jpeg_quality
        self.queues = [queue.Queue(depth) for _ in range(writers)]
        self.dropped: dict[str, int] = {}
        for q in self.queues:
            threading.Thread(target=self._write, args=(q,), daemon=True).start()

    def put(self, trial_id: str, task_variant: str, record: dict) -> None:
        """Queue a record for the trial's file; images (if any) are encoded on the writer thread."""
        q = self.queues[hash(trial_id) % len(self.queues)]  # one writer per trial keeps its records in order
        try:
            q.put_nowait((trial_id, task_variant, record))
        except queue.Full:
            self.dropped[trial_id] = self.dropped.get(trial_id, 0) + 1

    def _write(self, q: queue.Queue) -> None:
        while True:
            trial_id, variant, record = q.get()
            try:
                if "images" in record:
                    record = record | {"images": {c: cv2.imencode(".jpg", np.ascontiguousarray(im[..., ::-1]),
                                                                  [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
                                                  for c, im in record["images"].items()}}
                if trial_id in self.dropped:
                    record = record | {"dropped": self.dropped.pop(trial_id)}
                path = self.root / _safe(variant) / f"{_safe(trial_id)}.rec"
                path.parent.mkdir(parents=True, exist_ok=True)
                data = pack(record)
                with path.open("ab") as f:
                    f.write(struct.pack("<I", len(data)) + data)
            except Exception:  # noqa: BLE001
                log.exception("record write failed for trial %s", trial_id)
            finally:
                q.task_done()

    def flush(self) -> None:
        for q in self.queues:
            q.join()


def read_records(path: Path) -> list[dict]:
    """Records of one trial file; images come back as JPEG bytes (cv2.imdecode gives BGR)."""
    out, data = [], Path(path).read_bytes()
    i = 0
    while i + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, i)
        out.append(unpack(data[i + 4 : i + 4 + n]))
        i += 4 + n
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:128] or "_"
