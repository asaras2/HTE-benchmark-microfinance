
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_ihdp_transtee_mps_official.py

IHDP benchmark for TransTEE with Apple Silicon (MPS) support.
Optionally uses the "official" TransTEE transformer block if you cloned the repo.

Usage (standalone proxy block):
  python run_ihdp_transtee_mps_official.py \
    --train_npz data/ihdp_npci_1-100.train.npz \
    --test_npz  data/ihdp_npci_1-100.test.npz \
    --R 100 --out_csv outputs/ihdp_transtee_results.csv

Usage (try official block with repo):
  python run_ihdp_transtee_mps_official.py \
    --train_npz data/ihdp_npci_1-100.train.npz \
    --test_npz  data/ihdp_npci_1-100.test.npz \
    --R 100 --use_official_block \
    --repo_dir ~/Projects/TransTEE
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class IHDPRepDataset(Dataset):
    def __init__(self, X, t, yf, mu0, mu1, mean=None, std=None):
        X = X.astype(np.float32)
        t = t.astype(np.float32).reshape(-1, 1)
        yf = yf.astype(np.float32).reshape(-1, 1)
        mu0 = mu0.astype(np.float32).reshape(-1, 1)
        mu1 = mu1.astype(np.float32).reshape(-1, 1)

        if mean is not None and std is not None:
            X = (X - mean) / (std + 1e-8)
        self.mean = mean
        self.std = std

        self.X = torch.from_numpy(X)
        self.t = torch.from_numpy(t)
        self.yf = torch.from_numpy(yf)
        self.mu0 = torch.from_numpy(mu0)
        self.mu1 = torch.from_numpy(mu1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "t": self.t[idx],
            "y": self.yf[idx],
            "mu0": self.mu0[idx],
            "mu1": self.mu1[idx],
        }


def _infer_R_from_npz(train_npz_path, test_npz_path):
    """Infer the number of replications from the .npz file shapes."""
    tr = np.load(train_npz_path)
    te = np.load(test_npz_path)
    R_tr = tr["t"].shape[-1] if "t" in tr else tr["x"].shape[-1]
    R_te = te["t"].shape[-1] if "t" in te else te["x"].shape[-1]
    if R_tr != R_te:
        raise ValueError(f"Replication count mismatch between train (R={R_tr}) and test (R={R_te}).")
    return R_tr


def load_ihdp_npz(train_path, test_path, r):
    """
    Load a single replication r from the IHDP NPZ train/test files.

    The IHDP .npz format used here stores:
      - x:   shape (n, p, R)  -> individuals x features x replications
      - t:   shape (n, R)
      - yf:  shape (n, R)
      - ycf: shape (n, R)
      - mu0: shape (n, R)
      - mu1: shape (n, R)

    We slice replications along the LAST axis (axis=-1).

    Returns:
      trd, ted  (dicts for train/test) each with keys:
        'x'  -> (n_train/test, p)
        't'  -> (n_train/test,)
        'yf','ycf','mu0','mu1' -> (n_train/test,)
    """
    tr = np.load(train_path)
    te = np.load(test_path)

    required = ("x", "t", "yf", "ycf", "mu0", "mu1")
    for arrs, name in ((tr, "train"), (te, "test")):
        for k in required:
            if k not in arrs:
                raise KeyError(f"Missing key '{k}' in {name} npz")

    # Validate shapes and shared replication count
    x_tr, x_te = tr["x"], te["x"]
    if x_tr.ndim != 3 or x_te.ndim != 3:
        raise ValueError("Expected x arrays to have shape (n, p, R).")

    n_tr, p_tr, R_tr = x_tr.shape
    n_te, p_te, R_te = x_te.shape

    if p_tr != p_te:
        raise ValueError(f"Feature dimension mismatch: train p={p_tr}, test p={p_te}.")
    if R_tr != R_te:
        raise ValueError(f"Replication count mismatch: train R={R_tr}, test R={R_te}.")

    if not (0 <= r < R_tr):
        raise IndexError(f"Replication index r={r} out of range [0, {R_tr-1}].")

    def slice_rep(d):
        return {
            "x":   d["x"][:, :, r].astype(np.float32),
            "t":   d["t"][:,  r].astype(np.float32),
            "yf":  d["yf"][:, r].astype(np.float32),
            "ycf": d["ycf"][:, r].astype(np.float32),
            "mu0": d["mu0"][:, r].astype(np.float32),
            "mu1": d["mu1"][:, r].astype(np.float32),
        }

    trd = slice_rep(tr)
    ted = slice_rep(te)
    return trd, ted


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(128, 64), drop=0.1):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleTransTEE(nn.Module):
    def __init__(self, d_x, d_model=128, n_heads=4, n_layers=2, drop=0.1):
        super().__init__()
        self.t_embed = nn.Embedding(2, 8)
        self.x_in = nn.Linear(d_x, d_model - 8)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=drop, activation="relu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.readout = MLP(d_model, hidden=(128, 64), drop=drop)
        self.head0 = nn.Linear(64, 1)
        self.head1 = nn.Linear(64, 1)

    def forward(self, x, t):
        # Ensure treatment indices are a 1D vector of length batch
        # embedding accepts any integer-shaped input; normalize to (batch, embed_dim)
        te = self.t_embed(t.long().clip(0, 1).view(-1))
        if te.dim() == 3:
            # if shape (batch, 1, embed_dim) -> squeeze the middle dim
            te = te.squeeze(1)

        xe = self.x_in(x)

        # defensive check to make debugging easier when batch sizes mismatch
        if xe.shape[0] != te.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch between x and t embeddings: x={tuple(xe.shape)}, t_embed={tuple(te.shape)}"
            )

        h = torch.cat([xe, te], dim=1).unsqueeze(1)
        z = self.enc(h).squeeze(1)
        r = self.readout(z)
        return self.head0(r), self.head1(r)


def try_import_official_block(repo_dir: str | None):
    candidates = [
        "TransTEE.models.transtee",
        "TransTEE.model",
        "models.transtee",
        "Continuous.model",
        "Structured.model",
        "src.models.transtee",
    ]
    if repo_dir:
        repo_dir = os.path.expanduser(repo_dir)
        if os.path.isdir(repo_dir) and (repo_dir not in sys.path):
            sys.path.insert(0, repo_dir)

    last_err = None
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["*"])
            for attr in dir(m):
                if "Encoder" in attr or "Transformer" in attr:
                    return getattr(m, attr)
        except Exception as e:
            last_err = e
            continue
    return None


class OfficialWrapper(nn.Module):
    def __init__(self, d_x, encoder_ctor, d_model=128, drop=0.1):
        super().__init__()
        self.t_embed = nn.Embedding(2, 8)
        self.x_in = nn.Linear(d_x, d_model - 8)
        try:
            self.enc = encoder_ctor(d_model=d_model, nhead=4, batch_first=True, dropout=drop, activation="relu")
        except TypeError:
            self.enc = encoder_ctor
        self.readout = MLP(d_model, hidden=(128, 64), drop=drop)
        self.head0 = nn.Linear(64, 1)
        self.head1 = nn.Linear(64, 1)

    def forward(self, x, t):
        # Normalize treatment embedding shape as in the compact proxy above
        te = self.t_embed(t.long().clip(0, 1).view(-1))
        if te.dim() == 3:
            te = te.squeeze(1)

        xe = self.x_in(x)

        if xe.shape[0] != te.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch between x and t embeddings: x={tuple(xe.shape)}, t_embed={tuple(te.shape)}"
            )

        h = torch.cat([xe, te], dim=1).unsqueeze(1)
        z = self.enc(h) if isinstance(self.enc, nn.Module) else h
        if z.dim() == 3:
            z = z.squeeze(1)
        r = self.readout(z)
        return self.head0(r), self.head1(r)


def pehe(mu0, mu1, y0_hat, y1_hat):
    tau = mu1 - mu0
    tau_hat = y1_hat - y0_hat
    return torch.sqrt(torch.mean((tau_hat - tau) ** 2)).item()


def ate_err(mu0, mu1, y0_hat, y1_hat):
    return torch.abs(torch.mean((y1_hat - y0_hat) - (mu1 - mu0))).item()


def policy_risk(mu0, mu1, y0_hat, y1_hat):
    treat_hat = (y1_hat > y0_hat).float()
    y_policy = treat_hat * mu1 + (1 - treat_hat) * mu0
    opt_policy = (mu1 > mu0).float()
    y_opt = opt_policy * mu1 + (1 - opt_policy) * mu0
    return torch.mean(y_opt - y_policy).item()


def run_rep(
    train_npz,
    test_npz,
    r,
    device,
    use_official,
    repo_dir,
    epochs=200,
    lr=1e-3,
    batch_size=256,
    num_workers=2,
    d_model=128,
    n_heads=4,
    n_layers=2,
    drop=0.1,
    seed=42,
    checkpoint_dir: str | None = None,
    save_every: int = 10,
    resume_checkpoint: str | None = None,
):
    torch.manual_seed(seed + r)
    np.random.seed(seed + r)

    tr, te = load_ihdp_npz(train_npz, test_npz, r)
    d_x = tr["x"].shape[1]

    mean = tr["x"].mean(axis=0, keepdims=True).astype(np.float32)
    std = tr["x"].std(axis=0, keepdims=True).astype(np.float32)

    ds_tr = IHDPRepDataset(tr["x"], tr["t"], tr["yf"], tr["mu0"], tr["mu1"], mean, std)
    ds_te = IHDPRepDataset(te["x"], te["t"], te["yf"], te["mu0"], te["mu1"], mean, std)

    dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    if use_official:
        enc_ctor = try_import_official_block(repo_dir)
        if enc_ctor is not None:
            print("[info] Using OFFICIAL encoder block from repo.")
            model = OfficialWrapper(d_x=d_x, encoder_ctor=enc_ctor, d_model=d_model, drop=drop).to(device)
        else:
            print("[warn] Could not import official block. Falling back to compact proxy.")
            model = SimpleTransTEE(d_x=d_x, d_model=d_model, n_heads=n_heads, n_layers=n_layers, drop=drop).to(device)
    else:
        model = SimpleTransTEE(d_x=d_x, d_model=d_model, n_heads=n_heads, n_layers=n_layers, drop=drop).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    torch.set_float32_matmul_precision("medium")

    start_epoch = 0

    # Optionally resume from a given checkpoint
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        # restore RNG states if available
        if "torch_rng_state" in ckpt:
            try:
                torch.set_rng_state(ckpt["torch_rng_state"])
            except Exception:
                pass
        if "numpy_rng_state" in ckpt:
            try:
                np.random.set_state(ckpt["numpy_rng_state"])
            except Exception:
                pass
        print(f"[info] Resumed rep {r} from checkpoint '{resume_checkpoint}' starting at epoch {start_epoch}")

    # Ensure checkpoint dir exists if saving
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(start_epoch, epochs), desc=f"rep {r+1:03d}"):
        model.train()
        for batch in dl:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            y0_hat, y1_hat = model(x, t)
            y_hat = (1 - t) * y0_hat + t * y1_hat
            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # checkpointing after epoch (1-indexed)
        if checkpoint_dir and ((epoch + 1) % save_every == 0 or (epoch + 1) == epochs):
            ckpt_path = Path(checkpoint_dir) / f"rep{r:03d}_epoch{epoch+1}.pt"
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
            }
            torch.save(ckpt, str(ckpt_path))
            print(f"[info] Saved checkpoint to {ckpt_path}")

    model.eval()
    with torch.no_grad():
        x = ds_te.X.to(device)
        t = ds_te.t.to(device)
        mu0 = ds_te.mu0.to(device)
        mu1 = ds_te.mu1.to(device)
        y0_hat, y1_hat = model(x, t)
        return {
            "pehe": pehe(mu0, mu1, y0_hat, y1_hat),
            "ate_err": ate_err(mu0, mu1, y0_hat, y1_hat),
            "policy_risk": policy_risk(mu0, mu1, y0_hat, y1_hat),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True, type=str)
    ap.add_argument("--test_npz", required=True, type=str)
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to write per-rep metrics CSV")
    ap.add_argument("--use_official_block", action="store_true", help="Try to import the official TransTEE encoder")
    ap.add_argument("--repo_dir", type=str, default=None, help="Path to your cloned TransTEE repo (optional)")
    ap.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints (optional)")
    ap.add_argument("--save_every", type=int, default=10, help="Save a checkpoint every N epochs")
    ap.add_argument("--resume_checkpoint", type=str, default=None, help="Path to a checkpoint file to resume from")
    args = ap.parse_args()

    # Validate requested number of repetitions (R) against what's actually in the .npz files.
    # This avoids mid-run IndexError when the user requests more reps than the dataset contains.
    try:
        trz = np.load(args.train_npz, allow_pickle=True)
        tez = np.load(args.test_npz, allow_pickle=True)
        keys = ["x", "t", "yf", "ycf", "mu0", "mu1"]
        def safe_len(arr, key):
            if key not in arr:
                return 0
            v = arr[key]
            # If stored as an object array of reps, len(v) will work. Otherwise, try first axis.
            try:
                return len(v)
            except Exception:
                try:
                    return v.shape[0]
                except Exception:
                    return 0

        available_reps = _infer_R_from_npz(args.train_npz, args.test_npz)
        if available_reps == 0:
            raise RuntimeError(f"Could not determine number of reps from {args.train_npz} / {args.test_npz}")
        if args.R > available_reps:
            print(f"[warn] Requested R={args.R} but only {available_reps} reps available in the provided .npz files."
                  f" Clamping R -> {available_reps} to avoid out-of-bounds access.")
            args.R = available_reps
    except Exception as e:
        # If anything goes wrong, surface an informative message and continue — the loader will still raise the
        # original IndexError if we truly do request an out-of-bounds rep.
        print(f"[warn] Could not validate .npz contents ahead of time: {e}")

    device = pick_device()
    print(f"Using device: {device} (MPS available: {torch.backends.mps.is_available()})")

    results = []
    skipped = []
    ckpt_base = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    for r in range(args.R):
        # Auto-skip reps that already have a checkpoint with epoch >= requested epochs
        if ckpt_base and ckpt_base.exists():
            rep_glob = list(ckpt_base.glob(f"rep{r:03d}_epoch*.pt"))
            if rep_glob:
                # parse epochs from filenames and check if any >= requested epochs
                max_epoch = 0
                for p in rep_glob:
                    name = p.stem  # e.g. rep000_epoch150
                    try:
                        parts = name.split("_epoch")
                        if len(parts) == 2:
                            e = int(parts[1])
                            if e > max_epoch:
                                max_epoch = e
                    except Exception:
                        continue
                if max_epoch >= args.epochs:
                    print(f"[info] Skipping rep {r} because checkpoint exists with epoch={max_epoch} >= requested={args.epochs}")
                    skipped.append(r)
                    continue

        metrics = run_rep(
            args.train_npz,
            args.test_npz,
            r,
            device,
            use_official=args.use_official_block,
            repo_dir=args.repo_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            drop=args.drop,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            resume_checkpoint=args.resume_checkpoint,
        )
        results.append(metrics)

    if skipped:
        print(f"[info] Skipped {len(skipped)} reps: {skipped}")

    if len(results) == 0:
        print("[info] No reps were run (all were skipped because final checkpoints exist). Exiting.")
        return

    keys = results[0].keys()
    mean = {k: float(np.mean([d[k] for d in results])) for k in keys}
    std = {k: float(np.std([d[k] for k in keys]) if False else np.std([d[k] for d in results])) for k in keys}  # ensure correct listing

    if args.out_csv:
        import csv
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rep"] + list(keys))
            for i, d in enumerate(results, 1):
                w.writerow([i] + [d[k] for k in keys])
        print(f"[OK] Wrote per-rep metrics CSV to {args.out_csv}")

    print(json.dumps({"mean": mean, "std": std}, indent=2))


if __name__ == "__main__":
    main()
