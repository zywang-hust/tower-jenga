"""Train sim/refine.py on (raster image, recorded JPEG) pairs of the same dataset state: random frames of the
train split, all variants, four cameras each. Loss: L1 in sRGB, optionally plus an L1 on the Laplacian pyramid
bands (--lap) and extra weight on the wrist cameras' gripper region (--grip); random crops (--crop) or full frames.

Every --eval steps, on fixed val-split states, and a checkpoint:
  val64:   64 random states, full-frame PSNR per camera, and the gripper-region PSNR of the two wrist cameras;
  contact: each val episode's first grasp + 0.5 s and + 2 s, the holding
           arm's wrist camera: full-frame, gripper-region, and gripper-region sigma 2-16 px band PSNR (the
           lift cue band of docs/research.md §4.4).
--eval-only <refine.pt> evaluates a checkpoint (and the raster) without training.

CUDA_VISIBLE_DEVICES=4 runs/simdev/render_venv/bin/python sim/train_refine.py <out_dir> [--steps N]
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pxr import Usd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
from fk import ArmFK  # noqa: E402
from refine import Refine  # noqa: E402
from render import CAMERAS, STATION, Renderer, episode_blocks, episode_dims  # noqa: E402
from rollout import ACTION_HZ  # noqa: E402

from jenga.paths import DATA_ROOT  # noqa: E402

D = str(DATA_ROOT)  # env JENGA_DATA_ROOT
SPLIT = json.loads((HERE.parent / "runs/prep/split.json").read_text())["tasks"]
CAMS = list(CAMERAS)
WRIST = [k for k, c in enumerate(CAMS) if c.endswith("_wrist_camera")]
GRIP = (slice(240, 480), slice(160, 480))  # wrist-camera rows, cols between and around the fingertips
_STAGES = Usd.StageCache()


def block_names(f) -> list[str]:
    """Station block names of the sim/block_poses columns."""
    with Usd.StageCacheContext(_STAGES):  # episode_blocks' own stage expires before it walks the prims
        return episode_blocks(STATION, json.loads(f["sim"].attrs["block_fixed_on_table"]))


def state(fk, f, names, t):
    links = fk.link_poses(f["observations/qpos"][t])
    return (list(links), np.stack(list(links.values())), names, f["sim/block_poses"][t],
            np.stack([np.asarray(Image.open(io.BytesIO(f[f"observations/images/{c}"][t].tobytes())).convert("RGB")) for c in CAMERAS]),
            episode_dims(f["sim"].attrs, names))


class Frames(torch.utils.data.IterableDataset):
    """Random (or, with a seed list, fixed) dataset states: FK link poses, block names and poses, the four images."""

    def __init__(self, part: str, fixed: int = 0):
        self.episodes = [e for s in SPLIT.values() for e in s[part]]
        self.fixed = fixed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(None if not self.fixed else 0)
        fk, files = ArmFK(STATION), {}
        picks = [(self.episodes[i], rng.random()) for i in rng.integers(0, len(self.episodes), self.fixed)] if self.fixed else None
        k = 0
        while True:
            if picks is not None:
                if k >= len(picks):
                    return
                ep, u = picks[k]
                k += 1
                if info and (k - 1) % info.num_workers != info.id:
                    continue
            else:
                ep, u = self.episodes[rng.integers(len(self.episodes))], rng.random()
            if ep not in files:
                f = h5py.File(f"{D}/{ep}")
                files[ep] = f, block_names(f)
            f, names = files[ep]
            yield state(fk, f, names, int(u * len(f["observations/qpos"])))


def contact_frames():
    """(state, holding wrist camera index) at each val episode's first grasp close + 0.5 s and + 2 s."""
    fk, out = ArmFK(STATION), []
    for ep in (e for s in SPLIT.values() for e in s["val"]):
        with h5py.File(f"{D}/{ep}") as f:
            q = f["observations/qpos"][:]
            closes = [(int(np.argmax(q[:, c] < 0.07)), arm) for c, arm in ((7, "left"), (15, "back")) if (q[:, c] < 0.07).any()]
            if not closes:
                continue
            close, arm = min(closes)
            names = block_names(f)
            for dt in (ACTION_HZ // 2, 2 * ACTION_HZ):
                if close + dt < len(q):
                    out.append((state(fk, f, names, close + dt), CAMS.index(f"{arm}_wrist_camera")))
    return out


def batches(ds, n, workers):
    return iter(torch.utils.data.DataLoader(ds, batch_size=n, num_workers=workers, collate_fn=list, persistent_workers=workers > 0))


def gauss(x, s):
    """Separable Gaussian blur of [N, 3, H, W], reflect padding."""
    r = int(3 * s)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * s * s))
    k = (k / k.sum()).repeat(3, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(3, 1, 1, -1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(3, 1, -1, 1), groups=3)


def pyramid_l1(d, w, levels=5):
    """Mean over levels of the weighted L1 of the Laplacian pyramid bands of d [N, 3, H, W] (w [N, 1, H, W])."""
    acc = 0.0
    for _ in range(levels):
        low = F.avg_pool2d(d, 2)
        acc = acc + ((d - F.interpolate(low, size=d.shape[-2:], mode="bilinear")).abs() * w).mean()
        d, w = low, F.avg_pool2d(w, 2)
    return acc / levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256, help="square crop side; 0 = full frames")
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--widths", default="32,64,128,256")
    ap.add_argument("--lap", type=float, default=0.0, help="weight of the Laplacian pyramid L1")
    ap.add_argument("--grip", type=float, default=0.0, help="extra loss weight on the wrist cameras' gripper region")
    ap.add_argument("--init", metavar="CKPT", help="start from this checkpoint's weights (same widths)")
    ap.add_argument("--eval-only", metavar="CKPT")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    R = Renderer(STATION)
    net = Refine.load(args.eval_only or args.init, "cuda") if (args.eval_only or args.init) else \
        Refine(tuple(int(w) for w in args.widths.split(","))).cuda()
    if args.init:
        net.requires_grad_(True).train()

    def pairs(items):
        P = np.stack([R.poses(dict(zip(names, links)), dict(zip(bnames, blocks)), dims) for names, links, bnames, blocks, _, dims in items])
        with torch.no_grad():
            rgb, gbuf = R.raster(torch.as_tensor(P, dtype=torch.float32, device="cuda"), gbuffer=True)
        real = torch.as_tensor(np.stack([x[4] for x in items]), device="cuda").float() / 255
        return rgb, gbuf, real

    val = [x for b in batches(Frames("val", fixed=64), 8, 4) for x in b]
    contact = contact_frames()
    psnr = lambda e: (10 * torch.log10(1 / e)).tolist()  # noqa: E731
    band = lambda a: gauss(a, 2.0) - gauss(a, 16.0)  # noqa: E731

    def evaluate(step):
        net.eval()
        full, grip = torch.zeros(2, len(CAMERAS), device="cuda"), torch.zeros(2, len(WRIST), device="cuda")
        con = torch.zeros(2, 3, device="cuda")
        with torch.no_grad():
            for i in range(0, len(val), 8):
                rgb, gbuf, real = pairs(val[i:i + 8])
                for k, im in enumerate((rgb, net(rgb, gbuf))):
                    full[k] += ((im - real) ** 2).mean((2, 3, 4)).sum(0)
                    grip[k] += ((im - real)[:, WRIST][:, :, GRIP[0], GRIP[1]] ** 2).mean((2, 3, 4)).sum(0)
            for i in range(0, len(contact), 8):
                chunk = contact[i:i + 8]
                rgb, gbuf, real = pairs([s for s, _ in chunk])
                cam = torch.tensor([c for _, c in chunk], device="cuda")
                pick = lambda a: a[torch.arange(len(chunk)), cam].permute(0, 3, 1, 2)  # noqa: E731  [n, 3, H, W]
                for k, im in enumerate((rgb, net(rgb, gbuf))):
                    d = pick(im) - pick(real)
                    con[k, 0] += (d ** 2).mean((1, 2, 3)).sum()
                    con[k, 1] += (d[..., GRIP[0], GRIP[1]] ** 2).mean((1, 2, 3)).sum()
                    con[k, 2] += (band(d)[..., GRIP[0], GRIP[1]] ** 2).mean((1, 2, 3)).sum()
        full, grip, con = full / len(val), grip / len(val), con / len(contact)
        row = dict(step=step)
        for k, name in enumerate(("raster", "refined")):
            row[name] = dict(zip(CAMERAS, psnr(full[k])), **{f"{CAMS[c]}_grip": v for c, v in zip(WRIST, psnr(grip[k]))},
                             **dict(zip(("contact_wrist", "contact_grip", "contact_grip_band"), psnr(con[k]))))
        print(json.dumps(row), flush=True)
        if not args.eval_only:
            torch.save(dict(widths=net.widths, state=net.state_dict(), step=step, eval=row, args=vars(args)), out / "refine.pt")
        net.train()
        return row

    print(f"val64 {len(val)} states, contact {len(contact)} frames", flush=True)
    if args.eval_only:
        json.dump(evaluate(0), open(out / f"eval_{Path(args.eval_only).parent.name}.json", "w"), indent=1)
        print("REFINE_EVAL_DONE", flush=True)
        return

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.03)
    train = batches(Frames("train"), args.batch, 16)
    evaluate(0)
    t0, c = time.time(), args.crop
    wmap = torch.ones(len(CAMERAS), R.h, R.w, device="cuda")
    wmap[WRIST, GRIP[0], GRIP[1]] += args.grip
    for step in range(1, args.steps + 1):
        rgb, gbuf, real = pairs(next(train))
        E, C, H, W, _ = rgb.shape
        if c:
            y, x = np.random.randint(0, H - c + 1), np.random.randint(0, W - c + 1)
            crop = lambda a: a[:, :, y:y + c, x:x + c]  # noqa: E731
            rgb, gbuf, real, w = crop(rgb), crop(gbuf), crop(real), wmap[None, :, y:y + c, x:x + c]
        else:
            w = wmap[None]
        w = w.expand(E, -1, -1, -1).flatten(0, 1)[:, None]  # [E*C, 1, h, w]
        d = (net(rgb, gbuf) - real).flatten(0, 1).permute(0, 3, 1, 2)
        loss = (d.abs() * w).mean()
        if args.lap:
            loss = loss + args.lap * pyramid_l1(d, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t0) / step * 1e3:.0f} ms/step", flush=True)
        if step % args.eval == 0 or step == args.steps:
            evaluate(step)
    print("REFINE_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
