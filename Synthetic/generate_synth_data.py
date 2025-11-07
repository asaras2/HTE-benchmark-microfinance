#!/usr/bin/env python3
"""
generate_synth_data.py

The script saves two .npz files (train/test) where each key maps to an
object array of length R; each element is a numpy array for that rep.

Usage:
  python3 Synthetic/generate_synth_data.py --R 75 --n 2000 --d 25 --dgp sine --seed 42 --out_dir Synthetic/data

They contain keys: x, t, yf, mu0, mu1, tau

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

# Import DGP machinery from the existing script to ensure identical DGP
from run_synth_transtee_mps import make_dgp, DGPConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=75)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--d", type=int, default=25)
    p.add_argument("--dgp", type=str, default="sine", choices=["sine", "linear", "sparse"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="Synthetic/data")
    p.add_argument("--nonlin", action="store_true", help="enable baseline nonlinearity (same as not --linear in train script)")
    p.add_argument("--confounding", type=float, default=2.0)
    p.add_argument("--overlap", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--heteroskedastic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_list = {k: [] for k in ['x','t','yf','mu0','mu1','tau']}
    test_list = {k: [] for k in ['x','t','yf','mu0','mu1','tau']}

    for r in range(args.R):
        seed = args.seed + r
        cfg = DGPConfig(name=args.dgp, d=args.d, nonlin=args.nonlin,
                        confounding=args.confounding, overlap=args.overlap,
                        noise=args.noise, heteroskedastic=args.heteroskedastic)
        tr, te = make_dgp(args.n, cfg, seed=seed)
        # append each array to lists
        for k in train_list.keys():
            train_list[k].append(tr[k])
            test_list[k].append(te[k])
        if (r+1) % max(1, args.R//10) == 0 or r < 5:
            print(f"generated rep {r+1}/{args.R}")

    # convert to object arrays so np.savez preserves per-rep arrays
    train_np = {k: np.array(v, dtype=object) for k,v in train_list.items()}
    test_np = {k: np.array(v, dtype=object) for k,v in test_list.items()}

    train_path = out_dir / f"synth_train_R{args.R}_n{args.n}.npz"
    test_path = out_dir / f"synth_test_R{args.R}_n{args.n}.npz"

    np.savez(train_path, **train_np)
    np.savez(test_path, **test_np)

    print(f"Wrote train .npz -> {train_path}")
    print(f"Wrote test  .npz -> {test_path}")


if __name__ == '__main__':
    main()
