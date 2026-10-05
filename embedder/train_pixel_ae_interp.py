"""Train a per-category nonlinear feature->3 head (PixelAutoEncoder) on top of the
INTERP (WiSE-FT, single global alpha) ResNet-101 backbone.

This learns a NONLINEAR replacement for the PCA->3 reduction: a 1x1-conv MLP that
maps the ~1792-dim ResNet feature vector at each pixel to a 3-dim latent and back,
trained to reconstruct the features (cosine loss) on nominal `train/good` images.

It deliberately reuses the SAME embedder that evaluate.py uses
(ood.embeddings.ResNetPixelEmbedder) and the SAME PixelAutoEncoder class, so the
saved .pth loads directly via evaluate.py --autoencoder_path. The only change vs
the stock trainer is that the backbone weights are overwritten with the interp
state dict (same injection as evaluate_ddad_fe.py).

Usage (env `ood`, from repo root):
  DDAD_FE_WEIGHTS=<interp.pth> CUDA_VISIBLE_DEVICES=0 \
  python embedder/train_pixel_ae_interp.py \
      --weights <interp.pth> \
      --data_dir ./data/mvtec_ad/<cat>/train/good \
      --save_model embedder/models/pixel_ae_interp/<cat>.pth \
      --latent_dim 3 --num_epochs 30
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


def set_seed(seed):
    """Make AE training reproducible: fix python/numpy/torch RNGs and force
    deterministic cuDNN. Without this, the random train/val split + weight init
    produce a different 3-d latent orientation each run -> unstable downstream AUC."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def ae_loss_fn(xh, z, feats, kind):
    """AE reconstruction loss. 'mse_cos' = MSE + (1 - cosine); 'cos' = 1 - cosine;
    'mseW_cos' (e.g. mse0.3_cos) = W*MSE + (1 - cosine);
    'wmse_cos' (VarAE) = inverse-variance-weighted MSE + (1 - cosine): each feature
    dim's squared error is divided by its nominal batch variance, so high-variance
    (texture) dims are downweighted and stable dims dominate — a per-dimension,
    data-driven replacement for the per-category MSE weight lambda;
    'mse_var' = MSE + 0.1*ReLU(0.33 - var(z)) (train_embedder selection loss)."""
    cos = 1.0 - F.cosine_similarity(xh, feats, dim=1).mean()
    if kind == "cos":
        return cos
    if kind == "mse_cos":
        return F.mse_loss(xh, feats) + cos
    if kind == "wmse_cos":
        var_d = feats.var(dim=[0, 2, 3], unbiased=False).detach().clamp_min(1e-6)
        w = (1.0 / var_d)
        w = w / w.mean()  # mean weight 1 -> same overall scale as plain MSE
        wmse = ((xh - feats) ** 2 * w.view(1, -1, 1, 1)).mean()
        return wmse + cos
    if kind.startswith("mse") and kind.endswith("_cos"):
        return float(kind[3:-4]) * F.mse_loss(xh, feats) + cos
    # mse_var
    z_var = z.var(dim=[0, 2, 3])
    return F.mse_loss(xh, feats) + 0.1 * F.relu(0.33 - z_var).mean()

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)

from ood.embeddings import ResNetPixelEmbedder, patchify_context  # SAME embedder eval uses
from evaluate import PixelAutoEncoder                 # SAME AE class eval loads
from embedder.evaluate_ddad_fe import _unwrap


class ImageFolderFlat(Dataset):
    def __init__(self, root, tf):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.paths = sorted(str(p) for p in Path(root).rglob("*")
                            if p.is_file() and p.suffix.lower() in exts)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def build_transform(augment):
    """Train/val image transforms. augment='none' -> plain Resize+ToTensor (default,
    keeps existing runs bit-identical). 'geo' -> geometry-only augmentation for the
    training split: horizontal flip + small affine (patient-positioning variance),
    all LABEL-PRESERVING for CT axial slices. Deliberately NO intensity/brightness
    jitter: CT anomalies are intensity deviations, so jittering them would teach the
    AE to reconstruct the very signal we score on. 'geo_intensity' adds mild
    brightness/contrast for ablation."""
    base = [T.Resize((256, 256)), T.ToTensor()]
    if augment == "none":
        return T.Compose(base)
    aug = [T.Resize((256, 256)),
           T.RandomHorizontalFlip(0.5),
           T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))]
    if augment == "geo_intensity":
        aug.append(T.ColorJitter(brightness=0.1, contrast=0.1))
    aug.append(T.ToTensor())
    return T.Compose(aug)


def embed_with_grad(emb, x):
    """ResNetPixelEmbedder.forward minus its @torch.no_grad decorator — used only
    for --train_last_layer so AE-loss gradients reach the unfrozen backbone block.
    Must mirror ood.embeddings.ResNetPixelEmbedder.forward exactly."""
    if x.min() < 0:
        x = (x + 1) / 2.0
    if emb.use_imagenet_norm:
        x = (x - emb.mean) / emb.std
    feats = emb.extractor(x)
    H, W = x.shape[-2:]
    outH, outW = (emb.out_size, emb.out_size) if emb.out_size is not None else (H, W)
    ups = []
    for ln in emb.layers:
        f = feats[ln]
        if emb.use_patch_context:
            f = patchify_context(f, patchsize=emb.patchify_size, stride=1)
        if emb.proj_dim_per_layer is not None:
            f = emb.proj[ln](f)
        f = F.interpolate(f, size=(outH, outW), mode="bilinear", align_corners=False)
        f = F.normalize(f, dim=1)
        ups.append(f)
    return torch.cat(ups, dim=1)


def build_embedder(weights, device):
    emb = ResNetPixelEmbedder(resnet_name="resnet101",
                              layers=("layer1", "layer2", "layer3"),
                              use_patch_context=True, patchify_size=3,
                              proj_dim_per_layer=None).to(device).eval()
    if weights:
        sd = _unwrap(torch.load(weights, map_location="cpu"))
        res = emb.extractor.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"interp backbone not fully loaded; missing {res.missing_keys[:4]}")
        print(f"  [AE-train] backbone <- {weights} (unexpected={len(res.unexpected_keys)} layer4/fc, ok)")
    else:
        print("  [AE-train] WARNING: no --weights -> ImageNet backbone")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("DDAD_FE_WEIGHTS", ""),
                    help="interp backbone state dict")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_model", required=True)
    ap.add_argument("--latent_dim", type=int, default=3)
    ap.add_argument("--num_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0,
                    help="base RNG seed -> reproducible AE (deterministic split/init)")
    ap.add_argument("--n_restarts", type=int, default=1,
                    help="train this many AEs (seed, seed+1, ...) and keep the best-val one")
    ap.add_argument("--max_images", type=int, default=0,
                    help="cap dataset to this many images (0 = all); seeded subset for large sets")
    ap.add_argument("--ae_loss", default="cos",
                    help="AE TRAINING loss: cos | mse_cos | mse_var | mse<W>_cos (e.g. mse0.3_cos).")
    ap.add_argument("--select_loss", default="mse_var",
                    help="checkpoint-SELECTION loss (val): same choices as --ae_loss. "
                         "Default mse_var = train_embedder.py recipe.")
    ap.add_argument("--feat_noise", type=float, default=0.0,
                    help="VarAE denoising: train-time Gaussian noise on the AE input, "
                         "scaled per-dim to feat std (x + g*std_d*eps); reconstruct CLEAN "
                         "feats. Val/selection stays clean. 0 = off.")
    ap.add_argument("--augment", default="none", choices=["none", "geo", "geo_intensity"],
                    help="training-split image augmentation (val stays clean). "
                         "none=Resize+ToTensor (default); geo=hflip+small affine; "
                         "geo_intensity=geo+mild brightness/contrast.")
    ap.add_argument("--lr_schedule", default="none", choices=["none", "cosine"],
                    help="none=constant lr (default); cosine=CosineAnnealingLR over epochs.")
    ap.add_argument("--train_last_layer", action="store_true",
                    help="ABLATION: unfreeze the LAST block of layer3 and train it "
                         "jointly with the AE on the AE loss itself (backbone BN stays "
                         "in eval mode: running stats, only conv/BN weights adapt). "
                         "Saves the tuned backbone next to the AE as <name>_backbone.pth "
                         "(load at eval via DDAD_FE_WEIGHTS + evaluate_ddad_fe.py).")
    ap.add_argument("--backbone_lr", type=float, default=1e-4,
                    help="lr for the unfrozen layer3[-1] params (train_last_layer only)")
    args = ap.parse_args()
    device = "cuda"

    # val split uses the plain (clean) transform; train split gets --augment.
    tf_plain = build_transform("none")
    tf_train = build_transform(args.augment)
    ds = ImageFolderFlat(args.data_dir, tf_plain)
    ds_aug = ImageFolderFlat(args.data_dir, tf_train)
    assert len(ds) > 1, f"no images under {args.data_dir}"
    keep = list(range(len(ds)))
    if args.max_images and len(ds) > args.max_images:
        import torch as _t
        keep = sorted(_t.randperm(len(ds), generator=_t.Generator().manual_seed(args.seed))[:args.max_images].tolist())
        print(f"  [AE-train] capped dataset to {len(keep)} images (seed {args.seed})")
    vN = max(1, int(0.1 * len(keep)))
    # fixed train/val split (seeded) so every run/restart sees the same data
    split_g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(keep), generator=split_g).tolist()
    tr_idx = [keep[i] for i in perm[vN:]]
    va_idx = [keep[i] for i in perm[:vN]]
    tr = torch.utils.data.Subset(ds_aug, tr_idx)   # augmented (or plain if --augment none)
    va = torch.utils.data.Subset(ds, va_idx)       # always clean

    emb = build_embedder(args.weights, device)

    bb_params, bb_init_sd, bb_path = [], None, None
    if args.train_last_layer:
        names = [n for n, _ in emb.extractor.named_parameters() if n.startswith("layer3.")]
        last_idx = max(int(n.split(".")[1]) for n in names)
        prefix = f"layer3.{last_idx}."
        for n, p in emb.extractor.named_parameters():
            p.requires_grad = n.startswith(prefix)
        bb_params = [p for n, p in emb.extractor.named_parameters() if n.startswith(prefix)]
        bb_init_sd = {k: v.detach().cpu().clone() for k, v in emb.extractor.state_dict().items()}
        bb_path = os.path.splitext(args.save_model)[0] + "_backbone.pth"
        print(f"  [AE-train] train_last_layer: unfroze {prefix}* "
              f"({sum(p.numel() for p in bb_params)/1e6:.2f}M params, lr {args.backbone_lr}) "
              f"-> backbone will be saved to {bb_path}")

    with torch.no_grad():
        edim = emb(torch.zeros(1, 3, 256, 256, device=device)).shape[1]
    print(f"  [AE-train] {args.data_dir}: {len(ds)} imgs  embed_dim={edim}  latent={args.latent_dim}"
          f"  seed={args.seed}  n_restarts={args.n_restarts}"
          + (f"  feat_noise={args.feat_noise} (VarAE)" if args.feat_noise > 0 else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
    best_overall = float("inf")
    for r in range(args.n_restarts):
        set_seed(args.seed + r)
        loader_g = torch.Generator().manual_seed(args.seed + r)
        trL = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2,
                         generator=loader_g, worker_init_fn=_seed_worker)
        vaL = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2,
                         worker_init_fn=_seed_worker)
        ae = PixelAutoEncoder(input_dim=edim, latent_dim=args.latent_dim).to(device)
        if args.train_last_layer:
            emb.extractor.load_state_dict(bb_init_sd)   # fresh backbone per restart
            opt = torch.optim.Adam([{"params": ae.parameters(), "lr": args.lr},
                                    {"params": bb_params, "lr": args.backbone_lr}])
        else:
            opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
                 if args.lr_schedule == "cosine" else None)
        best = float("inf")
        for ep in range(args.num_epochs):
            ae.train(); tot = 0.0
            for imgs in trL:
                imgs = imgs.to(device)
                if args.train_last_layer:
                    feats = embed_with_grad(emb, imgs)
                else:
                    with torch.no_grad():
                        feats = emb(imgs)
                if args.feat_noise > 0:
                    std_d = feats.std(dim=[0, 2, 3], keepdim=False).view(1, -1, 1, 1)
                    ae_in = feats + args.feat_noise * std_d * torch.randn_like(feats)
                else:
                    ae_in = feats
                xh, z = ae(ae_in)
                loss = ae_loss_fn(xh, z, feats, args.ae_loss)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            if sched is not None:
                sched.step()
            # select the checkpoint by the SAME loss used for training
            ae.eval(); vtot = 0.0
            with torch.no_grad():
                for imgs in vaL:
                    imgs = imgs.to(device)
                    feats = emb(imgs)
                    xh, z = ae(feats)
                    vtot += ae_loss_fn(xh, z, feats, args.select_loss).item()
            vloss = vtot / max(1, len(vaL))
            print(f"  [r{r}] ep {ep+1:02d}/{args.num_epochs}  train {tot/len(trL):.4f}  val {vloss:.4f}")
            if vloss < best:
                best = vloss
                best_sd = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
                if args.train_last_layer:
                    best_bb_sd = {k: v.detach().cpu().clone()
                                  for k, v in emb.extractor.state_dict().items()}
        print(f"  [r{r}] best val {best:.4f}")
        if best < best_overall:
            best_overall = best
            torch.save(best_sd, args.save_model)
            if args.train_last_layer:
                torch.save(best_bb_sd, bb_path)
            print(f"  [AE-train] new best (restart {r}) val {best_overall:.4f} -> {args.save_model}")
    print(f"  [AE-train] FINAL best val {best_overall:.4f} -> {args.save_model}")


if __name__ == "__main__":
    main()
