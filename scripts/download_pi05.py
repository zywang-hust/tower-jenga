"""Resume an eight-stream ModelScope mirror download of pi05_base into $JENGA_MODELS/pi05_base and verify it against
the official GCS checksums. Needs <dest>/download-manifest.json: {"repo": <ModelScope repo>, "official": <GCS object
listing of gs://openpi-assets/checkpoints/pi05_base>, "mirror": <ModelScope file listing>}. With direct GCS access,
`gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base <dest>` is simpler."""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import google_crc32c

from jenga.paths import PI05_BASE

ROOT = PI05_BASE
META = json.loads((ROOT / "download-manifest.json").read_text())
FILES = META["official"]["items"]
MIRROR = {f["Path"]: f for f in META["mirror"]["Data"]["Files"] if f["Type"] == "blob"}
PARTS = ROOT / ".download-parts"
PARTS.mkdir(exist_ok=True)
CHUNK = 64 * 1024**2
lock = threading.Lock()
received = 0
started = time.monotonic()
stop = threading.Event()


def download(job):
    global received
    rel, offset, length = job
    path = ROOT / rel
    marker = PARTS / (hashlib.sha256(rel.encode()).hexdigest() + f".{offset}")
    if marker.exists() and path.stat().st_size == int(MIRROR[rel]["Size"]):
        return
    revision = MIRROR[rel]["Revision"]
    url = f"https://modelscope.cn/models/{META['repo']}/resolve/{revision}/{quote(rel)}"
    for attempt in range(6):
        done = 0
        try:
            req = Request(
                url, headers={"Range": f"bytes={offset}-{offset + length - 1}", "Accept-Encoding": "identity"}
            )
            with urlopen(req, timeout=45) as response, path.open("r+b", buffering=0) as out:
                if response.status == 206:
                    expected = f"bytes {offset}-{offset + length - 1}/{MIRROR[rel]['Size']}"
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("Unexpected Content-Range")
                elif not (response.status == 200 and offset == 0 and length == MIRROR[rel]["Size"]):
                    raise ValueError(f"Unexpected status {response.status}")
                out.seek(offset)
                while done < length:
                    buf = response.read(min(1024**2, length - done))
                    if not buf:
                        raise IOError("Truncated response")
                    out.write(buf)
                    done += len(buf)
                    with lock:
                        received += len(buf)
                os.fsync(out.fileno())
            marker.write_text(str(length))
            return
        except Exception as exc:
            print(
                json.dumps({"retry": attempt + 1, "file": rel, "offset": offset, "error": str(exc)}),
                flush=True,
            )
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 15))


def monitor():
    previous = 0
    last = started
    while not stop.wait(20):
        now = time.monotonic()
        with lock:
            current = received
        status = {
            "elapsed_s": round(now - started, 1),
            "received_GB": round(current / 1e9, 3),
            "recent_MB_s": round((current - previous) / (now - last) / 1e6, 2),
            "average_MB_s": round(current / (now - started) / 1e6, 2),
            "completed_parts": len(list(PARTS.iterdir())),
            "total_parts": len(jobs),
        }
        print(json.dumps(status), flush=True)
        tmp = ROOT / "download-progress.tmp"
        tmp.write_text(json.dumps(status))
        tmp.replace(ROOT / "download-progress.json")
        previous, last = current, now


jobs = []
for item in FILES:
    rel = item["name"].removeprefix("checkpoints/pi05_base/")
    size = int(item["size"])
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("wb") as out:
            out.truncate(size)
    elif path.stat().st_size != size:
        raise ValueError(f"Existing file has unexpected size: {path}")
    jobs.extend((rel, offset, min(CHUNK, size - offset)) for offset in range(0, size, CHUNK))

# Start with eight full-sized ranges for a representative aggregate throughput sample.
jobs.sort(key=lambda job: job[2], reverse=True)
threading.Thread(target=monitor, daemon=True).start()
print(
    json.dumps(
        {
            "streams": 8,
            "files": len(FILES),
            "parts": len(jobs),
            "total_bytes": sum(int(x["size"]) for x in FILES),
        }
    ),
    flush=True,
)
try:
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(download, job) for job in jobs]):
            future.result()
finally:
    stop.set()
elapsed = time.monotonic() - started
print("Download finished; verifying official CRC32C and mirror SHA256.", flush=True)
verified = []
for item in FILES:
    rel = item["name"].removeprefix("checkpoints/pi05_base/")
    crc, sha = google_crc32c.Checksum(), hashlib.sha256()
    with (ROOT / rel).open("rb") as inp:
        while buf := inp.read(8 * 1024**2):
            crc.update(buf)
            sha.update(buf)
    actual_crc = base64.b64encode(crc.digest()).decode()
    actual_sha = sha.hexdigest()
    if actual_crc != item["crc32c"] or actual_sha != MIRROR[rel]["Sha256"]:
        raise ValueError(f"Checksum mismatch: {rel}")
    verified.append({"path": rel, "bytes": int(item["size"]), "crc32c": actual_crc, "sha256": actual_sha})
report = {
    "status": "verified",
    "streams": 8,
    "download_seconds": elapsed,
    "transferred_bytes": received,
    "average_MB_s": received / elapsed / 1e6,
    "official_source": "gs://openpi-assets/checkpoints/pi05_base",
    "mirror_repo": META["repo"],
    "files": verified,
}
(ROOT / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "files"}), flush=True)
