#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_synth_transtee_mps.py

Synthetic-data benchmark for TransTEE with Apple Silicon (MPS) support.

Usage (standalone proxy block):
  python run_synth_transtee_mps.py \
    --R 20 --n 2000 --d 25 --dgp sine \
    --epochs 150 --lr 3e-4 --bs 256 \
    --out_csv outputs/synth_transtee_results.csv

If you have the TransTEE repo cloned and want to use the official block,
pass --use_official_block and ensure your PYTHONPATH includes the repo.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Device (MPS/CPU/CUDA)
# ---------------------------
def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# Small MLP helper
# ---------------------------
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), drop=0.1, act="relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU}
        A = acts.get(act, nn.ReLU)
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), A(), nn.Dropout(drop)]
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------
# Simple TransTEE-like backbone (proxy)
# ----------------------------------------
class SimpleTransTEE(nn.Module):
    """A tiny proxy of TransTEE-style encoder with treatment embedding."""
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
        # t: (B,) or (B,1), int in {0,1}
        te = self.t_embed(t.long().clip(0, 1).view(-1))
        if te.dim() == 3:
            te = te.squeeze(1)
        xe = self.x_in(x)
        if xe.shape[0] != te.shape[0]:
            raise RuntimeError(f"Batch mismatch: x={xe.shape} t={te.shape}")
        h = torch.cat([xe, te], dim=1).unsqueeze(1)   # (B, 1, d_model)
        z = self.enc(h).squeeze(1)                    # (B, d_model)
        z = self.readout(z)                           # (B, 64)
        y0 = self.head0(z)
        y1 = self.head1(z)
        # factual head choose:
        y = torch.where(t.view(-1, 1) > 0.5, y1, y0)
        return y, y0, y1


# ----------------------------------------
# Official TransTEE block (optional)
# ----------------------------------------
class OfficialTransTEEWrapper(nn.Module):
    """Use when --use_official_block is set and repo is available."""
    def __init__(self, d_x, d_model=128, n_heads=4, n_layers=2, drop=0.1):
        super().__init__()
        try:
            from transtee.models import TransTEE  # noqa
        except Exception as e:
            raise ImportError("Could not import official TransTEE. Check PYTHONPATH.") from e
        # Hypothetical constructor; adapt if your repo differs:
        self.model = TransTEE(input_dim=d_x, d_model=d_model, nhead=n_heads, num_layers=n_layers, dropout=drop)

    def forward(self, x, t):
        # Expected to return y, y0, y1 (shape (B,1) each); adapt if API differs.
        return self.model(x, t)


# ---------------------------
# Dataset
# ---------------------------
class SynthDataset(Dataset):
    def __init__(self, x, t, yf, mu0, mu1, tau):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.t = torch.from_numpy(t.astype(np.float32)).view(-1, 1)
        self.yf = torch.from_numpy(yf.astype(np.float32)).view(-1, 1)
        self.mu0 = torch.from_numpy(mu0.astype(np.float32)).view(-1, 1)
        self.mu1 = torch.from_numpy(mu1.astype(np.float32)).view(-1, 1)
        self.tau = torch.from_numpy(tau.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.t[i], self.yf[i], self.mu0[i], self.mu1[i], self.tau[i]


# ---------------------------
# Metrics
# ---------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def policy_risk_from_scores(mu0_hat, mu1_hat, mu0, mu1):
    """
    Simple plug-in policy: treat if mu1_hat > mu0_hat.
    Risk = E[optimal_outcome - chosen_outcome] (lower is better).
    """
    treat = (mu1_hat > mu0_hat).astype(np.float32)
    chosen = treat * mu1 + (1 - treat) * mu0
    optimal = np.maximum(mu0, mu1)
    return float(np.mean(optimal - chosen))


# ---------------------------
# Synthetic DGPs
# ---------------------------
@dataclass
class DGPConfig:
    name: str = "sine"
    d: int = 25
    nonlin: bool = True
    confounding: float = 2.0
    overlap: float = 0.1      # min propensity away from 0/1 (0.1 => e(x) in [0.1, 0.9])
    noise: float = 1.0
    heteroskedastic: bool = False


def _sigmoid(z): return 1. / (1. + np.exp(-z))


def make_tau(x: np.ndarray, kind: str = "sine") -> np.ndarray:
    """Ground-truth hetero effect τ(x)."""
    if kind == "linear":
        w = np.linspace(1.0, 0.2, x.shape[1])
        return (x @ w) / (np.sqrt((w**2).sum()) + 1e-6)
    if kind == "sparse":
        w = np.zeros(x.shape[1]); w[:5] = [2, -1.5, 1.2, -0.8, 0.6]
        return x @ w / 3.0
    # default: siney / interaction heavy
    return np.sin(x[:, :3].sum(axis=1)) + 0.5 * x[:, 3] * x[:, 4]


def make_dgp(n: int, cfg: DGPConfig, seed: int = 0) -> Tuple[dict, dict]:
    """
    Returns (train, test) dicts each with keys: x, t, yf, mu0, mu1, tau
    """
    rng = np.random.default_rng(seed)
    def gen_split(n):
        # covariates
        x = rng.normal(size=(n, cfg.d))
        if cfg.nonlin:
            x[:, :5] = np.tanh(x[:, :5])  # add mild nonlinearity

        # true CATE and potential outcomes
        tau = make_tau(x, cfg.name)
        # baseline outcome
        f = x[:, :5].sum(axis=1) + 0.5 * (x[:, 5:10] ** 2).sum(axis=1) if cfg.nonlin else x.sum(axis=1)
        mu0 = f
        mu1 = f + tau

        # propensity with confounding via selected features
        logit = (cfg.confounding * (0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 2])
                 + 0.5 * np.sin(x[:, 3]) - 0.5 * x[:, 4])
        e = _sigmoid(logit)
        # enforce overlap
        alpha = cfg.overlap
        e = alpha + (1 - 2 * alpha) * e  # squish into [alpha, 1-alpha]

        t = rng.binomial(1, e)

        # noise
        if cfg.heteroskedastic:
            sigma = cfg.noise * (0.5 + np.abs(x[:, 0]))
        else:
            sigma = cfg.noise
        eps = rng.normal(scale=sigma, size=n)

        yf = mu0 + t * tau + eps
        return dict(x=x, t=t, yf=yf, mu0=mu0, mu1=mu1, tau=tau)

    tr = gen_split(n)
    te = gen_split(max(n // 2, 1000))
    return tr, te


# ---------------------------
# Training loop
# ---------------------------
def train_one_rep(rep: int, args, device: torch.device):
    torch.manual_seed(args.seed + rep)
    np.random.seed(args.seed + rep)

    cfg = DGPConfig(
        name=args.dgp, d=args.d, nonlin=not args.linear,
        confounding=args.confounding, overlap=args.overlap,
        noise=args.noise, heteroskedastic=args.heteroskedastic
    )
    tr, te = make_dgp(args.n, cfg, seed=args.seed + rep)

    # standardize X
    mean = tr["x"].mean(axis=0, keepdims=True).astype(np.float32)
    std = tr["x"].std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    for d in (tr, te):
        d["x"] = (d["x"] - mean) / std

    ds_tr = SynthDataset(**tr)
    ds_te = SynthDataset(**te)
    dl = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=False)

    # model
    if args.use_official_block:
        model = OfficialTransTEEWrapper(d_x=cfg.d, d_model=args.width, n_heads=args.heads,
                                        n_layers=args.layers, drop=args.drop)
    else:
        model = SimpleTransTEE(d_x=cfg.d, d_model=args.width, n_heads=args.heads,
                               n_layers=args.layers, drop=args.drop)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_f = nn.MSELoss()

    # train
    model.train()
    for epoch in range(1, args.epochs + 1):
        tot = 0.0
        for x, t, yf, _, _, _ in dl:
            x, t, yf = x.to(device), t.to(device), yf.to(device)
            y, _, _ = model(x, t)
            loss = loss_f(y, yf)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach().cpu())
        if epoch % max(1, args.epochs // 5) == 0:
            print(f"[rep {rep:02d}] epoch {epoch:03d}/{args.epochs} loss={tot/len(dl):.4f}")

    # evaluate (ITE and policy metrics use mu0/mu1 available on both train/test)
    model.eval()
    with torch.no_grad():
        x = ds_te.x.to(device)
        t0 = torch.zeros((x.shape[0], 1), device=device)
        t1 = torch.ones((x.shape[0], 1), device=device)
        _, mu0_hat, mu1_hat = model(x, t0)
        _, mu0_hat1, mu1_hat1 = model(x, t1)  # some proxies output heads regardless of t; this is defensive
        mu0_hat = (mu0_hat + mu0_hat1) / 2.0
        mu1_hat = (mu1_hat + mu1_hat1) / 2.0

        mu0_hat = mu0_hat.cpu().numpy().reshape(-1)
        mu1_hat = mu1_hat.cpu().numpy().reshape(-1)
        tau_hat = (mu1_hat - mu0_hat).reshape(-1)

        mu0 = ds_te.mu0.numpy().reshape(-1)
        mu1 = ds_te.mu1.numpy().reshape(-1)
        tau = ds_te.tau.numpy().reshape(-1)

        pehe = rmse(tau_hat, tau)
        ate_err = abs(float(tau_hat.mean() - tau.mean()))
        prisk = policy_risk_from_scores(mu0_hat, mu1_hat, mu0, mu1)

    return {"pehe": pehe, "ate_err": ate_err, "policy_risk": prisk}


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Synthetic TransTEE benchmark")
    p.add_argument("--R", type=int, default=10, help="number of reps")
    p.add_argument("--n", type=int, default=2000, help="train examples per rep")
    p.add_argument("--d", type=int, default=25, help="feature dimension")
    p.add_argument("--dgp", type=str, default="sine",
                   choices=["sine", "linear", "sparse"], help="form of true CATE τ(x)")
    p.add_argument("--linear", action="store_true", help="disable nonlinearities in baseline outcome")
    p.add_argument("--confounding", type=float, default=2.0, help="strength of confounding in propensity")
    p.add_argument("--overlap", type=float, default=0.1, help="keeps e(x) in [alpha, 1-alpha]")
    p.add_argument("--noise", type=float, default=1.0, help="noise std")
    p.add_argument("--heteroskedastic", action="store_true", help="use feature-dependent noise scale")
    # model/training
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use_official_block", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefer_cpu", action="store_true", help="force CPU even if MPS is available")
    p.add_argument("--out_csv", type=str, default="outputs/synth_transtee_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu") if args.prefer_cpu else get_device(prefer_mps=True)
    print(f"[device] {device}")
    results = []
    for r in range(1, args.R + 1):
        metrics = train_one_rep(r, args, device)
        results.append(metrics)
        print({"rep": r, **metrics})

    # aggregate and save
    import csv
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    keys = ["pehe", "ate_err", "policy_risk"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep"] + keys)
        for i, d in enumerate(results, 1):
            w.writerow([i] + [d[k] for k in keys])
    means = {k: float(np.mean([d[k] for d in results])) for k in keys}
    stds  = {k: float(np.std([d[k] for d in results], ddof=0)) for k in keys}
    print(json.dumps({"mean": means, "std": stds}, indent=2))
    print(f"[OK] Wrote per-rep metrics CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
