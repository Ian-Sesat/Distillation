import os, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm 
import math, argparse

from exp1_model import Exp1Model
from exp1_loss import exp1_loss

def load_pairs(pairs_dir):
    tsv = os.path.join(pairs_dir, "captions.tsv")
    img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    cands = [pairs_dir, os.path.join(pairs_dir, "images")]

    def resolve(ref):
        for b in cands:
            p = os.path.join(b, ref)
            if os.path.exists(p):
                return p
        return ref if (os.path.isabs(ref) and os.path.exists(ref)) else None

    paths, col = [], None
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if col is None:
                c0 = parts[0].lower().endswith(img_exts)
                c1 = parts[1].lower().endswith(img_exts)
                if c0 and not c1:
                    col = 0
                elif c1 and not c0:
                    col = 1
                else:
                    if parts[0].lower() in ("image", "filename", "img", "path"):
                        continue
                    col = 0
            paths.append(resolve(parts[col]))

    return paths

class Exp1Dataset(Dataset):
    def __init__(self, paths, valid, transform):
        self.transform = transform
        self.items = [(i, p) for i, p in enumerate(paths) if valid[i] and p is not None]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        idx, path = self.items[k]
        try:
            img = self.transform(Image.open(path).convert("RGB"))
            return img, idx
        except Exception:
            return None      

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, idxs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--teacher_dir", required=True, help="MCIP-SigLIP2 img+txt+valid cache")
    ap.add_argument("--text_feats", required=True, help="base-SigLIP2 text features .npy")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shared_dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)          
    ap.add_argument("--weight_decay", type=float, default=0.2) 
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--tau_student", type=float, default=1/60) 
    ap.add_argument("--tau_teacher", type=float, default=1/60) 
    ap.add_argument("--lam", type=float, default=1.0)          
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    model = Exp1Model(shared_dim=args.shared_dim, device=device).to(device).train()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model trainable params: {n:,}")

    teach_img = np.load(f"{args.teacher_dir}/img_feats_siglip2_base.npy", mmap_mode="r")   
    teach_txt = np.load(f"{args.teacher_dir}/text_feats_siglip2_base.npy", mmap_mode="r")  
    valid     = np.load(f"{args.teacher_dir}/valid_siglip2_base.npy").astype(bool)
    text_feats = np.load(args.text_feats, mmap_mode="r")                                
    print(f"teacher img {teach_img.shape} txt {teach_txt.shape} | student text {text_feats.shape} | valid {valid.sum():,}")

    paths = load_pairs(args.pairs_dir)
    ds = Exp1Dataset(paths, valid, model.image_transform(256))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate, drop_last=True)
    print(f"pairs {len(paths):,} | trainable {len(ds):,} | batch {args.batch} | {len(loader):,} steps/epoch")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)

    def lr_at(s):
        if s < args.warmup_steps: return args.lr * (s+1)/args.warmup_steps
        prog = (s-args.warmup_steps)/max(1, total_steps-args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi*prog))

    start_epoch, step = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0); step = ck.get("step", start_epoch*len(loader))
        print(f"RESUMED epoch={start_epoch} step={step}")

    pbar = tqdm(total=total_steps, desc="exp1", dynamic_ncols=True)
    pbar.update(step)

    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            if batch is None: continue
            imgs, idxs = batch
            imgs = imgs.to(device, non_blocking=True)
            for pg in opt.param_groups: pg["lr"] = lr_at(step)

            s_img = model.encode_image(imgs)
            idxs_np = idxs.numpy()
            txt_in = torch.from_numpy(text_feats[idxs_np].copy()).float().to(device)
            s_txt = model.encode_text(txt_in)

            with torch.no_grad():
                t_img = F.normalize(torch.from_numpy(teach_img[idxs_np].copy()).float().to(device), dim=-1)
                t_txt = F.normalize(torch.from_numpy(teach_txt[idxs_np].copy()).float().to(device), dim=-1)

            loss, parts = exp1_loss(s_img, s_txt, t_img, t_txt,
                                    tau_s=args.tau_student, tau_t=args.tau_teacher, lam=args.lam)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   
            opt.step()

            pbar.update(1); pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                      f"it={parts['L_it']:.2f} ti={parts['L_ti']:.2f} lr={lr_at(step):.2e}", flush=True)
            step += 1

        ep = epoch + 1
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step}
        torch.save(ck, f"{args.out_dir}/exp1_latest.pth")
        if ep % 5 == 0 or ep == args.epochs:
            torch.save(ck, f"{args.out_dir}/exp1_epoch{ep}.pth")
            print(f"[epoch {ep}] saved", flush=True)
    print("DONE.")


if __name__ == "__main__":
    main()