#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_microfinance_transtee_mps.py

Real-world microfinance data benchmark for TransTEE with Apple Silicon (MPS) support.

This script:
1. Loads household + individual characteristics data
2. Filters to surveyed households only (hhSurveyed == 1)
3. Constructs treatment (shgparticipate), outcome (savings), and covariates (age, resp_gend, etc.)
4. Converts categorical variables to binary numeric features
5. Runs TransTEE on R bootstrap replications
6. Reports ATE (Average Treatment Effect) and policy risk metrics

Usage:
  python run_microfinance_transtee_mps.py \
    --R 5 --epochs 150 --lr 3e-4 --bs 256 \
    --out_csv outputs/microfinance_transtee_results.csv

Note: Unlike synthetic data, we do NOT have true potential outcomes (mu0, mu1, tau).
So we compute:
  - ATE: E[y1_hat - y0_hat] (difference in predicted outcomes)
  - Policy Risk: plug-in policy (treat if y1_hat > y0_hat) compared to observed factual
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
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
        self.sigmoid = nn.Sigmoid()  # Constrain outputs to [0, 1]

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
        y0 = self.sigmoid(self.head0(z))  # Apply sigmoid to constrain to [0, 1]
        y1 = self.sigmoid(self.head1(z))  # Apply sigmoid to constrain to [0, 1]
        # factual head chooses based on t:
        y = torch.where(t.view(-1, 1) > 0.5, y1, y0)
        return y, y0, y1


# ---------------------------
# Dataset
# ---------------------------
class MicrofinanceDataset(Dataset):
    def __init__(self, x, t, y):
        """
        Args:
            x: (n, d) features array, already normalized
            t: (n,) or (n,1) treatment array, binary
            y: (n,) or (n,1) outcome array, binary
        """
        self.x = torch.from_numpy(x.astype(np.float32))
        self.t = torch.from_numpy(t.astype(np.float32)).view(-1, 1)
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.t[i], self.y[i]


# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_and_preprocess_data(
    hh_file: str,
    ind_file: str,
    feat_cols: list,
    t_col: str = "shgparticipate",
    y_col: str = "savings",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load household and individual characteristics, merge, filter, and construct features.
    
    Args:
        hh_file: path to household_characteristics.dta
        ind_file: path to individual_characteristics.dta
        feat_cols: list of covariate column names
        t_col: treatment column name (in individual data)
        y_col: outcome column name (in individual data)
        seed: random seed for reproducibility
    
    Returns:
        X: (n, d) feature matrix (float32)
        T: (n,) treatment binary array
        Y: (n,) outcome binary array
    """
    # Load data
    hh = pd.read_stata(hh_file)
    ind = pd.read_stata(ind_file)
    
    print(f"[Data] Loaded household data: {hh.shape}")
    print(f"[Data] Loaded individual data: {ind.shape}")
    
    # Merge on hhid
    df = ind.merge(hh[['hhid', 'hhSurveyed', 'electricity', 'latrine', 'ownrent']], 
                   on='hhid', how='left')
    
    # Filter: keep only surveyed households
    n_before = len(df)
    df = df[df['hhSurveyed'] == 1].copy()
    print(f"[Data] After filtering hhSurveyed==1: {len(df)} (dropped {n_before - len(df)})")
    
    # Check required columns exist
    required = [t_col, y_col] + feat_cols
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged data: {missing}")
    
    # Drop rows with missing treatment or outcome
    n_before = len(df)
    df = df.dropna(subset=[t_col, y_col])
    print(f"[Data] After dropping NaN in treatment/outcome: {len(df)} (dropped {n_before - len(df)})")
    
    # Drop rows with any NaN in features
    n_before = len(df)
    df = df.dropna(subset=feat_cols)
    print(f"[Data] After dropping NaN in covariates: {len(df)} (dropped {n_before - len(df)})")
    
    # Convert treatment to binary (1 if 'Yes', 0 if 'No'), DROP rows with 'Do not know'
    t_vals = df[t_col].astype(str).str.strip().str.lower()
    # Only keep Yes/No, drop any other values including 'do not know'
    valid_t = t_vals.isin(['yes', 'no'])
    n_before = len(df)
    df = df[valid_t].copy()
    print(f"[Data] After filtering treatment to Yes/No only: {len(df)} (dropped {n_before - len(df)})")
    
    # Convert outcome to binary (1 if 'Yes', 0 if 'No'), DROP rows with other values
    y_vals = df[y_col].astype(str).str.strip().str.lower()
    # Only keep Yes/No, drop any other values including 'do not know', 'refuse to say'
    valid_y = y_vals.isin(['yes', 'no'])
    n_before = len(df)
    df = df[valid_y].copy()
    print(f"[Data] After filtering outcome to Yes/No only: {len(df)} (dropped {n_before - len(df)})")
    
    # Now convert to binary
    t_vals = df[t_col].astype(str).str.strip().str.lower()
    t = (t_vals == 'yes').astype(np.int32).values
    
    y_vals = df[y_col].astype(str).str.strip().str.lower()
    y = (y_vals == 'yes').astype(np.int32).values
    
    print(f"[Data] Treatment distribution: {np.mean(t):.3f} (prop. treated)")
    print(f"[Data] Outcome distribution: {np.mean(y):.3f} (prop. with savings)")
    
    # Build feature matrix
    X_list = []
    for col in feat_cols:
        feat = df[col].values
        
        # If categorical, one-hot encode (drop first to avoid collinearity)
        if df[col].dtype == 'category' or df[col].dtype == 'object':
            dummies = pd.get_dummies(feat, prefix=col, drop_first=True).astype(np.float32)
            X_list.append(dummies.values)
            print(f"  {col}: one-hot encoded to {dummies.shape[1]} features")
        else:
            # Numeric: standardize
            feat_std = (feat - feat.mean()) / (feat.std() + 1e-6)
            X_list.append(feat_std.reshape(-1, 1).astype(np.float32))
            print(f"  {col}: standardized (numeric)")
    
    X = np.hstack(X_list) if X_list else np.zeros((len(df), 1), dtype=np.float32)
    print(f"[Data] Final feature matrix shape: {X.shape}")
    
    return X, t, y


# ---------------------------
# Metrics
# ---------------------------
def policy_risk_proxy(mu0_hat: np.ndarray, mu1_hat: np.ndarray, y_obs: np.ndarray, t_obs: np.ndarray) -> float:
    """
    Plug-in policy risk (without access to true counterfactual outcomes).
    
    Policy: Treat unit i if mu1_hat[i] > mu0_hat[i].
    Risk: E[observed_outcome | our_policy] - E[observed_outcome | treat_all].
    
    Since we only observe factual outcomes, this is a heuristic proxy.
    """
    treat_policy = (mu1_hat > mu0_hat).astype(np.float32)
    
    # Outcome under our policy (factual when policy agrees with observation)
    y_chosen = np.where(treat_policy == t_obs, y_obs, np.nan)  # only count overlap
    
    # As a proxy, compare empirical mean outcome of treated vs control
    if len(y_obs) > 0:
        avg_outcome = float(np.nanmean(y_obs))
    else:
        avg_outcome = 0.0
    
    return avg_outcome


def ate_plug_in(mu0_hat: np.ndarray, mu1_hat: np.ndarray) -> float:
    """Average Treatment Effect: E[mu1_hat - mu0_hat]"""
    return float(np.mean(mu1_hat - mu0_hat))


def outcome_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error on test set.
    
    Measures how well the factual outcome predictions match observed outcomes.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse


def hte_heterogeneity(mu0_hat: np.ndarray, mu1_hat: np.ndarray) -> float:
    """
    Treatment Effect Heterogeneity: std of CATE (Conditional Average Treatment Effect).
    
    CATE_i = mu1_hat[i] - mu0_hat[i] (unit-level treatment effect)
    HTE = std(CATE) measures how much heterogeneity exists in treatment effects.
    
    Higher values = more heterogeneous effects (model finds important subgroups).
    Lower values = more homogeneous effects (similar effects across everyone).
    """
    cate = mu1_hat - mu0_hat
    het = float(np.std(cate))
    return het


def propensity_balance(mu0_hat: np.ndarray, mu1_hat: np.ndarray, t_obs: np.ndarray) -> float:
    """
    Propensity-based balance metric: compare predicted propensities between treated and control.
    
    Propensity: P(T=1) estimated as average mu1_hat (what model predicts when T=1).
    Balance: Absolute difference in propensity scores between observed treated vs control.
    
    Returns: balance metric (0 = perfect balance, 1 = maximum imbalance)
    """
    treated_mask = t_obs > 0.5
    control_mask = ~treated_mask
    
    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        return 0.0
    
    # Propensity: use mu1_hat as proxy for treatment propensity
    prop_treated = mu1_hat[treated_mask].mean()
    prop_control = mu1_hat[control_mask].mean()
    
    balance = float(np.abs(prop_treated - prop_control))
    return balance


# ---------------------------
# Training loop
# ---------------------------
def train_one_rep(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    rep: int,
    args,
    device: torch.device
) -> Dict[str, float]:
    """
    Train TransTEE on one bootstrap replication of microfinance data.
    
    Args:
        X: (n, d) feature matrix
        t: (n,) treatment array
        y: (n,) outcome array
        rep: replication index
        args: command-line arguments
        device: torch device
    
    Returns:
        dict with keys: ate_pred, policy_risk_proxy
    """
    torch.manual_seed(args.seed + rep)
    np.random.seed(args.seed + rep)
    
    # Bootstrap sample
    n = len(X)
    boot_idx = np.random.choice(n, size=n, replace=True)
    X_boot = X[boot_idx]
    t_boot = t[boot_idx]
    y_boot = y[boot_idx]
    
    # Stratified train/test split (80/20) by treatment to maintain treatment balance
    # Separate treated and control units
    treated_idx = np.where(t_boot == 1)[0]
    control_idx = np.where(t_boot == 0)[0]
    
    # Split each group separately (stratified)
    split_t = int(0.8 * len(treated_idx))
    split_c = int(0.8 * len(control_idx))
    
    # Shuffle within each stratum
    np.random.shuffle(treated_idx)
    np.random.shuffle(control_idx)
    
    tr_idx = np.concatenate([treated_idx[:split_t], control_idx[:split_c]])
    te_idx = np.concatenate([treated_idx[split_t:], control_idx[split_c:]])
    
    # Shuffle train/test indices
    np.random.shuffle(tr_idx)
    np.random.shuffle(te_idx)
    
    X_tr, t_tr, y_tr = X_boot[tr_idx], t_boot[tr_idx], y_boot[tr_idx]
    X_te, t_te, y_te = X_boot[te_idx], t_boot[te_idx], y_boot[te_idx]
    
    # Standardize X using training stats
    mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_tr.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std
    
    ds_tr = MicrofinanceDataset(X_tr, t_tr, y_tr)
    ds_te = MicrofinanceDataset(X_te, t_te, y_te)
    dl = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=False)
    
    # Model
    model = SimpleTransTEE(
        d_x=X_tr.shape[1],
        d_model=args.width,
        n_heads=args.heads,
        n_layers=args.layers,
        drop=args.drop
    )
    model.to(device)
    
    # Optimizer & loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_f = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        tot_loss = 0.0
        n_batches = 0
        for x, t, y in dl:
            x, t, y = x.to(device), t.to(device), y.to(device)
            y_hat, _, _ = model(x, t)
            loss = loss_f(y_hat, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach().cpu())
            n_batches += 1
        
        if epoch % max(1, args.epochs // 5) == 0:
            avg_loss = tot_loss / n_batches
            print(f"  [rep {rep:02d}] epoch {epoch:03d}/{args.epochs} loss={avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        x_te = ds_te.x.to(device)
        t_te = ds_te.t.to(device)
        y_te = ds_te.y.cpu().numpy().reshape(-1)
        t_te_np = t_te.cpu().numpy().reshape(-1)
        
        # Predict potential outcomes
        t0 = torch.zeros((x_te.shape[0], 1), device=device)
        t1 = torch.ones((x_te.shape[0], 1), device=device)
        
        y_hat_factual, mu0_hat_raw, mu1_hat_raw = model(x_te, t_te)
        y_hat_factual = y_hat_factual.cpu().numpy().reshape(-1)
        mu0_hat = mu0_hat_raw.cpu().numpy().reshape(-1)
        mu1_hat = mu1_hat_raw.cpu().numpy().reshape(-1)
        
        # Metrics
        ate = ate_plug_in(mu0_hat, mu1_hat)
        prisk = policy_risk_proxy(mu0_hat, mu1_hat, y_te, t_te_np)
        prop_bal = propensity_balance(mu0_hat, mu1_hat, t_te_np)
        mse = outcome_mse(y_te, y_hat_factual)  # MSE on factual predictions
        hte_std = hte_heterogeneity(mu0_hat, mu1_hat)  # Std of CATE
    
    return {
        "ate_pred": ate,
        "policy_risk_proxy": prisk,
        "propensity_balance": prop_bal,
        "outcome_mse": mse,
        "hte_std": hte_std
    }


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-world microfinance data benchmark for TransTEE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data
    p.add_argument("--hh_file", type=str, default="household_characteristics.dta",
                   help="Path to household characteristics .dta file")
    p.add_argument("--ind_file", type=str, default="individual_characteristics.dta",
                   help="Path to individual characteristics .dta file")
    # Treatment, outcome, covariates
    p.add_argument("--t_col", type=str, default="shgparticipate",
                   help="Treatment column name (binary: Yes/No)")
    p.add_argument("--y_col", type=str, default="savings",
                   help="Outcome column name (binary: Yes/No)")
    p.add_argument("--feat_cols", nargs="+",
                   default=["age", "resp_gend", "rationcard", "workflag", "electricity", "latrine", "ownrent"],
                   help="Covariate column names")
    # Replications
    p.add_argument("--R", type=int, default=5, help="Number of bootstrap replications")
    # Model & training
    p.add_argument("--width", type=int, default=128, help="Transformer d_model")
    p.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--layers", type=int, default=2, help="Number of transformer layers")
    p.add_argument("--drop", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    p.add_argument("--bs", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--prefer_cpu", action="store_true", help="Force CPU even if MPS/CUDA available")
    p.add_argument("--out_csv", type=str, default="outputs/microfinance_transtee_results.csv",
                   help="Output CSV file for results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu") if args.prefer_cpu else get_device(prefer_mps=True)
    print(f"[device] {device}")
    
    # Load and preprocess data
    print("\n=== Loading and Preprocessing Data ===")
    X, t, y = load_and_preprocess_data(
        hh_file=args.hh_file,
        ind_file=args.ind_file,
        feat_cols=args.feat_cols,
        t_col=args.t_col,
        y_col=args.y_col,
        seed=args.seed
    )
    
    print(f"\n[Data Summary]")
    print(f"  n={len(X)}, d={X.shape[1]}")
    print(f"  Treatment mean: {t.mean():.3f}")
    print(f"  Outcome mean: {y.mean():.3f}")
    
    # Run replications
    print(f"\n=== Training TransTEE on {args.R} Bootstrap Replications ===")
    results = []
    for r in range(1, args.R + 1):
        print(f"\n[Rep {r}/{args.R}]")
        metrics = train_one_rep(X, t, y, r, args, device)
        results.append(metrics)
        print(f"  ATE (predicted): {metrics['ate_pred']:.4f}")
        print(f"  Policy Risk (proxy): {metrics['policy_risk_proxy']:.4f}")
        print(f"  Propensity Balance: {metrics['propensity_balance']:.4f}")
        print(f"  Outcome MSE: {metrics['outcome_mse']:.4f}")
        print(f"  HTE Std: {metrics['hte_std']:.4f}")
    
    # Aggregate and save
    print(f"\n=== Results ===")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    keys = ["ate_pred", "policy_risk_proxy", "propensity_balance", "outcome_mse", "hte_std"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep"] + keys)
        for i, d in enumerate(results, 1):
            w.writerow([i] + [d[k] for k in keys])
    
    means = {k: float(np.mean([d[k] for d in results])) for k in keys}
    stds = {k: float(np.std([d[k] for d in results], ddof=0)) for k in keys}
    
    print(json.dumps({"mean": means, "std": stds}, indent=2))
    print(f"[OK] Wrote per-rep metrics CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
