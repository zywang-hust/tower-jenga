"""Policy gap of a renderer: on fixed val-split demo states, how far the served policy's action moves when the four
recorded JPEGs are replaced by our render of the same state (FK of observations/qpos, sim/block_poses), against the
policy's own sampling spread (recorded images, seed 0 vs seed 1). Each query is a fresh trial (reset + one infer).
Per camera the gap is also measured with only that camera rendered (ablation). Image metrics per camera: PSNR, PSNR of the wrist cameras'
gripper region, the sigma 2-16 px band of the holding wrist camera on contact frames, and a VGG16 feature distance.
Diagnosis only: nothing here feeds training (val split only).

States: per val episode (task5-3 excluded, its station is not local) 4 evenly spaced frames and, after the first
grasp closes, +0.5 s, +1.5 s and +3 s (contact frames).

Run with the renderer's venv against a serve_policy.py of the snapshot (records under runs/simdev):
  CUDA_VISIBLE_DEVICES=2 runs/simdev/render_venv/bin/python sim/policy_gap.py runs/simdev/renderdev/gap <name> \
      [--refine PATH | --images set.npz] [--jpeg] [--url ws://127.0.0.1:9302] [--threads 4] [--policy TAG]
  real images and their actions are cached in <out>/real[_TAG].npz the first time; <out>/<name>/{images.npz,
  actions[_TAG].npz, metrics[_TAG].json} are written per renderer (the renders are shared across policies);
  --metrics-only recomputes the metrics from the saved files.
Tables of the results (docs/research.md 4.4.1; default: every renderer with metrics, TAG picks another policy's):
  runs/simdev/render_venv/bin/python sim/policy_gap.py table runs/simdev/renderdev/gap [--policy TAG] [name ...]
"""
import argparse
import io
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from websockets.sync.client import connect

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "external/tower-starter")]
from fk import ArmFK  # noqa: E402
from render import CAMERAS, ROTATED, STATION, Renderer, episode_dims  # noqa: E402
from rollout import ACTION_HZ  # noqa: E402
from tower_protocol import pack, unpack  # noqa: E402
from train_refine import D, GRIP, SPLIT, block_names, gauss  # noqa: E402

CAMS = list(CAMERAS)
ARM = np.r_[0:7, 8:15]
VGG = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"  # torchvision vgg16 IMAGENET1K_V1 weights


def states() -> list[dict]:
    out = []
    for task, s in sorted(SPLIT.items()):
        if task.startswith("task5-3"):
            continue
        for ep in sorted(s["val"]):
            with h5py.File(f"{D}/{ep}") as f:
                q = f["observations/qpos"][:]
                n = len(q)
                closes = [(int(np.argmax(q[:, c] < 0.07)), a) for c, a in ((7, "left"), (15, "back")) if (q[:, c] < 0.07).any()]
                frames = [(n // 8 + k * n // 4, "even", None) for k in range(4)]
                if closes:
                    close, arm = min(closes)
                    frames += [(close + dt, "contact", arm) for dt in (ACTION_HZ // 2, 3 * ACTION_HZ // 2, 3 * ACTION_HZ) if close + dt < n]
            out += [dict(episode=ep, variant=task, t=int(t), kind=k, arm=a) for t, k, a in frames]
    return out


def load_real(st: list[dict]):
    """Recorded JPEGs (dataset orientation) [N, 4, H, W, 3] and the observation of each state."""
    imgs, obs = [], []
    for s in st:
        with h5py.File(f"{D}/{s['episode']}") as f:
            t = s["t"]
            imgs.append(np.stack([np.asarray(Image.open(io.BytesIO(f[f"observations/images/{c}"][t].tobytes())).convert("RGB")) for c in CAMS]))
            q = f["observations/qpos"][t]
            obs.append({"schema": "tower-challenge-observation-v1", "step_index": t, "sim_time_s": t / ACTION_HZ,
                        "task_variant": s["variant"], "instruction": f.attrs["instruction"],
                        "joint_positions": np.stack([q[:7], q[8:15]]).astype("<f4"), "gripper_widths_m": q[[7, 15]].astype("<f4")})
    return np.stack(imgs), obs


def render_all(st: list[dict], refine: str | None, batch: int = 8) -> np.ndarray:
    """sim/render.py images of the states, dataset orientation, uint8 [N, 4, H, W, 3]."""
    R, fk, out, files = Renderer(STATION, refine=refine), ArmFK(STATION), [], {}
    for i in range(0, len(st), batch):
        P = []
        for s in st[i:i + batch]:
            if s["episode"] not in files:
                f = h5py.File(f"{D}/{s['episode']}")
                files[s["episode"]] = f, block_names(f)
            f, names = files[s["episode"]]
            P.append(R.poses(fk.link_poses(f["observations/qpos"][s["t"]]), dict(zip(names, f["sim/block_poses"][s["t"]])),
                             episode_dims(f["sim"].attrs, names)))
        out.append(R.render_batch(np.stack(P)))
    return np.concatenate(out)


def jpeg(imgs: np.ndarray, quality: int = 95) -> np.ndarray:
    """The generator's encoding of the recorded frames: libjpeg quality 95, 4:2:0."""
    def one(im):
        buf = io.BytesIO()
        Image.fromarray(im).save(buf, format="JPEG", quality=quality)
        return np.asarray(Image.open(buf).convert("RGB"))
    return np.stack([np.stack([one(im) for im in row]) for row in imgs])


class Policy:
    """One connection and session per task variant; every query is a fresh trial: reset with `seed`, one infer on
    dataset-orientation images -> chunk [k, 16]."""

    def __init__(self, url: str):
        self.url, self.sessions = url, {}

    def _session(self, variant: str):
        if variant not in self.sessions:
            headers = {"X-Tower-Task-Variant": variant, "X-Tower-Track-Version": "local-sim", "X-Tower-Evaluation-Version": "local-sim"}
            ws = connect(self.url, additional_headers=headers, max_size=64 << 20, compression=None, proxy=None)
            self.sessions[variant] = ws, dict(protocol="tower-infer-v1", session_id=str(uuid.uuid4()), policy_id=unpack(ws.recv())["policy_id"])
        return self.sessions[variant]

    def __call__(self, obs: dict, images: np.ndarray, seed: int = 0) -> np.ndarray:
        variant = obs["task_variant"]
        while True:
            ws, env = self._session(variant)
            env = dict(env, trial_id=str(uuid.uuid4()))
            ws.send(pack({**env, "type": "reset", "request_id": str(uuid.uuid4()), "seed": seed}))
            ack = unpack(ws.recv())
            if ack["type"] == "reset_ack":
                break
            # session_capacity: detached sessions of an earlier run live 300 s on the server, and the server closes
            # the connection after an error; wait and reconnect
            assert ack.get("error", {}).get("code") == "session_capacity", ack
            ws.close()
            del self.sessions[variant]
            time.sleep(15)
        wire = {c: np.ascontiguousarray(im[::-1, ::-1]) if c in ROTATED else im for c, im in zip(CAMS, images)}
        ws.send(pack({**env, "type": "infer", "request_id": str(uuid.uuid4()), "observation": dict(obs, images=wire)}))
        res = unpack(ws.recv())
        assert res["type"] == "infer_result", res
        return np.asarray(res["actions"], np.float32)


def query(url: str, threads: int, jobs: list[tuple]) -> np.ndarray:
    """jobs: (obs, images [4, H, W, 3], seed) -> actions [len(jobs), k, 16]; one Policy (threads x variants
    sessions) per thread."""
    local, policies = threading.local(), []

    def run(job):
        if not hasattr(local, "policy"):
            local.policy = Policy(url)
            policies.append(local.policy)
        return local.policy(*job)
    with ThreadPoolExecutor(threads) as ex:
        out = np.stack(list(ex.map(run, jobs)))
    for p in policies:  # sessions of finished threads would otherwise stay open and exhaust the server's 64-session cap
        for ws, _ in p.sessions.values():
            ws.close()
    return out


class VGGDist:
    """Mean over 4 VGG16 layers (relu1_2, relu2_2, relu3_3, relu4_3) of the MSE between unit-normalised features
    (LPIPS without its learned layer weights), on 640x480 sRGB."""

    def __init__(self, device):
        sd = torch.load(VGG, map_location=device)
        self.w = [(sd[f"features.{i}.weight"], sd[f"features.{i}.bias"]) for i in (0, 2, 5, 7, 10, 12, 14, 17, 19, 21)]
        self.taps, self.pool = {1, 3, 6, 9}, {2, 4, 7}
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def features(self, x):
        x, out = (x - self.mean) / self.std, []
        for k, (w, b) in enumerate(self.w):
            if k in self.pool:
                x = F.max_pool2d(x, 2)
            x = F.relu(F.conv2d(x, w, b, padding=1))
            if k in self.taps:
                out.append(F.normalize(x, dim=1))
        return out

    def __call__(self, a, b):  # [n, 3, H, W] in [0, 1] -> [n]
        with torch.no_grad():
            return torch.stack([((fa - fb) ** 2).sum(1).mean((1, 2)) for fa, fb in zip(self.features(a), self.features(b))]).mean(0)


def metrics(st: list[dict], real: np.ndarray, a_real: np.ndarray, imgs: np.ndarray, a_full: np.ndarray, a_cam: np.ndarray) -> dict:
    """a_real [N, 2, k, 16] (seeds 0, 1), a_full [N, k, 16], a_cam [N, 4, k, 16] (camera c rendered, others real)."""
    arm = lambda d: np.abs(d)[..., ARM].mean((-1, -2)) * 1e3  # noqa: E731  mrad per state
    grip = lambda d: np.abs(d)[..., [7, 15]].mean((-1, -2)) * 1e3  # noqa: E731  mm per state
    noise, gap, cam = a_real[:, 0] - a_real[:, 1], a_full - a_real[:, 0], a_cam - a_real[:, 0][:, None]
    kinds = {"all": np.ones(len(st), bool), "even": np.array([s["kind"] == "even" for s in st]), "contact": np.array([s["kind"] == "contact" for s in st])}
    row = {}
    for name, m in kinds.items():
        row[name] = dict(n=int(m.sum()), noise_arm_mrad=float(np.median(arm(noise)[m])), noise_arm_mean=float(arm(noise)[m].mean()),
                         noise_arm_p90=float(np.percentile(arm(noise)[m], 90)), gap_arm_mrad=float(np.median(arm(gap)[m])),
                         gap_arm_mean=float(arm(gap)[m].mean()), gap_arm_p90=float(np.percentile(arm(gap)[m], 90)),
                         noise_grip_mm=float(np.median(grip(noise)[m])), gap_grip_mm=float(np.median(grip(gap)[m])),
                         gap_grip_mean=float(grip(gap)[m].mean()),
                         per_camera_arm_mrad={c: float(np.median(arm(cam[:, k])[m])) for k, c in enumerate(CAMS)},
                         per_camera_grip_mm={c: float(np.median(grip(cam[:, k])[m])) for k, c in enumerate(CAMS)})
    for v in sorted({s["variant"] for s in st}):
        m = np.array([s["variant"] == v for s in st])
        row[v] = dict(n=int(m.sum()), noise_arm_mrad=float(np.median(arm(noise)[m])), gap_arm_mrad=float(np.median(arm(gap)[m])),
                      gap_grip_mm=float(np.median(grip(gap)[m])))
    dev = torch.device("cuda")
    vgg, psnr = VGGDist(dev), lambda e: float(10 * np.log10(1 / e))  # noqa: E731
    err = np.zeros(4)
    gerr, band, vd, n_band = np.zeros(2), 0.0, np.zeros(4), 0
    wrist = [k for k, c in enumerate(CAMS) if c.endswith("_wrist_camera")]
    for i in range(0, len(st), 8):
        a = torch.as_tensor(imgs[i:i + 8], device=dev).float() / 255
        b = torch.as_tensor(real[i:i + 8], device=dev).float() / 255
        d = a - b
        err += (d ** 2).mean((2, 3, 4)).sum(0).cpu().numpy()
        gerr += (d[:, wrist][:, :, GRIP[0], GRIP[1]] ** 2).mean((2, 3, 4)).sum(0).cpu().numpy()
        for j, s in enumerate(st[i:i + 8]):
            if s["kind"] == "contact":
                k = CAMS.index(f"{s['arm']}_wrist_camera")
                x = d[j, k].permute(2, 0, 1)[None]
                band += float(((gauss(x, 2.0) - gauss(x, 16.0))[..., GRIP[0], GRIP[1]] ** 2).mean())
                n_band += 1
        vd += np.array([float(vgg(a[:, k].permute(0, 3, 1, 2), b[:, k].permute(0, 3, 1, 2)).sum()) for k in range(4)])
    N = len(st)
    row["image"] = dict(psnr={c: psnr(err[k] / N) for k, c in enumerate(CAMS)},
                        grip_psnr={CAMS[k]: psnr(gerr[j] / N) for j, k in enumerate(wrist)},
                        contact_band_psnr=psnr(band / max(n_band, 1)), vgg={c: float(vd[k] / N) for k, c in enumerate(CAMS)},
                        vgg_mean=float(vd.mean() / N))
    return row


def table(root: Path, names: list[str], tag: str = "") -> None:
    """Print the policy-gap and image-metric tables of the renderers `names` under `root`."""
    tag = f"_{tag}" if tag else ""
    names = names or sorted(p.parent.name for p in root.glob(f"*/metrics{tag}.json"))
    rows = {n: json.loads((root / n / f"metrics{tag}.json").read_text()) for n in names}
    print(f"{'renderer':12s} {'med':>5s} {'mean':>5s} {'p90':>5s} {'grip':>5s} | {'contact':>7s} {'grip':>5s} | "
          "per-camera arm gap (lb/rf/lw/bw)")
    for i, (n, m) in enumerate(rows.items()):
        a, c = m["all"], m["contact"]
        if i == 0:
            print(f"{'(seed noise)':12s} {a['noise_arm_mrad']:5.2f} {a.get('noise_arm_mean', float('nan')):5.2f} "
                  f"{a.get('noise_arm_p90', float('nan')):5.2f} {a['noise_grip_mm']:5.2f} | {c['noise_arm_mrad']:7.2f} "
                  f"{c['noise_grip_mm']:5.2f} |")
        cams = " ".join(f"{v:.2f}" for v in a["per_camera_arm_mrad"].values())
        print(f"{n:12s} {a['gap_arm_mrad']:5.2f} {a['gap_arm_mean']:5.2f} {a['gap_arm_p90']:5.2f} {a['gap_grip_mm']:5.2f} | "
              f"{c['gap_arm_mrad']:7.2f} {c['gap_grip_mm']:5.2f} | {cams}")
    print(f"\n{'renderer':12s} PSNR lb/rf/lw/bw          grip lw/bw     band   VGG lb/rf/lw/bw")
    for n, m in rows.items():
        i = m["image"]
        print(f"{n:12s} {' '.join(f'{v:5.1f}' for v in i['psnr'].values())}   "
              f"{' '.join(f'{v:5.1f}' for v in i['grip_psnr'].values())}   "
              f"{i['contact_band_psnr']:5.1f}  {' '.join(f'{v:.3f}' for v in i['vgg'].values())}")


def main():
    if sys.argv[1:2] == ["table"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("root", type=Path)
        ap.add_argument("names", nargs="*")
        ap.add_argument("--policy", default="")
        args = ap.parse_args(sys.argv[2:])
        return table(args.root, args.names, args.policy)
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("name")
    ap.add_argument("--refine", help="refine checkpoint for sim/render.py (default: none = raster only)")
    ap.add_argument("--images", help="npz with images [N, 4, H, W, 3] of states() in dataset orientation (another renderer)")
    ap.add_argument("--jpeg", action="store_true", help="encode the renders as the generator does (JPEG q95) before serving")
    ap.add_argument("--url", default="ws://127.0.0.1:9302")
    ap.add_argument("--threads", type=int, default=4, help="threads x 9 variants sessions, detached ones live 300 s; cap 64")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--policy", default="", help="tag of the served policy: files get a _<tag> suffix (default r2-40000, no suffix)")
    ap.add_argument("--batch", type=int, default=8, help="render batch (lower it when the GPU is shared)")
    args = ap.parse_args()
    st = states()
    print(f"{len(st)} states: {sum(s['kind'] == 'contact' for s in st)} contact", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "states.json").write_text(json.dumps(st, indent=0))
    tag = f"_{args.policy}" if args.policy else ""
    cache = args.out / f"real{tag}.npz"
    real, obs = load_real(st)
    if cache.exists():
        a_real = np.load(cache)["actions"]
    else:
        a_real = query(args.url, args.threads, [(o, im, s) for o, im in zip(obs, real) for s in (0, 1)]).reshape(len(st), 2, -1, 16)
        np.savez(cache, images=real, actions=a_real)
        print("real actions cached", flush=True)
    d = args.out / args.name
    d.mkdir(exist_ok=True)
    if args.metrics_only or (d / "images.npz").exists() and not (d / f"actions{tag}.npz").exists():
        imgs = np.load(d / "images.npz")["images"]  # the renders of another policy's run: reuse them
    else:
        imgs = np.load(args.images)["images"] if args.images else render_all(st, args.refine, args.batch)
        assert imgs.shape == real.shape, (imgs.shape, real.shape)
        if args.jpeg:
            imgs = jpeg(imgs)
        np.savez(d / "images.npz", images=imgs)
    if args.metrics_only:
        z = np.load(d / f"actions{tag}.npz")
        a_full, a_cam = z["full"], z["cam"]
    else:
        jobs = [(o, im, 0) for o, im in zip(obs, imgs)]
        for k in range(4):
            jobs += [(o, np.concatenate([r[:k], im[k:k + 1], r[k + 1:]]), 0) for o, r, im in zip(obs, real, imgs)]
        a = query(args.url, args.threads, jobs)
        a_full, a_cam = a[:len(st)], a[len(st):].reshape(4, len(st), -1, 16).transpose(1, 0, 2, 3)
        np.savez(d / f"actions{tag}.npz", full=a_full, cam=a_cam)
    m = metrics(st, real, a_real, imgs, a_full, a_cam)
    (d / f"metrics{tag}.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("all", "contact", "image")}, indent=1))
    print("POLICY_GAP_DONE", flush=True)


if __name__ == "__main__":
    main()
