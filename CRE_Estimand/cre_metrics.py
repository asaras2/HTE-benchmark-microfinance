import argparse
import numpy as np
import pandas as pd

def load_tau_mu0_from_npz(path, rep=0):
    """
    Load tau (and optionally mu0) for a given replication index from the synthetic .npz.
    Expects keys like: 'tau', 'mu0', 'mu1', etc.
    Handles shapes:
      - (R, n) or (R, n, p): selects [rep]
      - (n,) or (n, p)     : uses as-is
    """
    data = np.load(path, allow_pickle=True)
    keys = {k.lower(): k for k in data.files}

    def get_any(candidates, required=True):
        for name in candidates:
            k = name.lower()
            if k in keys:
                return data[keys[k]]
        if required:
            raise ValueError(
                f"Missing one of {candidates} in {path}. Found keys: {list(data.files)}"
            )
        return None

    tau_all = get_any(["tau", "ite", "cate", "tau_true", "tau_y1_y0"], required=True)
    mu0_all = get_any(["mu0", "y0", "m0"], required=False)

    def select_rep(arr, label):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            # Interpret as (R, n) if rep index is valid; otherwise assume already (n, something)
            if rep < arr.shape[0]:
                return arr[rep]
            return arr
        if arr.ndim == 3:
            # (R, n, p)
            if rep < arr.shape[0]:
                return arr[rep]
            raise ValueError(f"{label}: rep {rep} out of range for shape {arr.shape}")
        raise ValueError(f"{label}: unsupported shape {arr.shape}")

    tau = select_rep(tau_all, "tau").ravel()
    mu0 = select_rep(mu0_all, "mu0").ravel() if mu0_all is not None else None

    return tau, mu0

def compute_metrics(test_cate_path, test_npz_path, rep=0):
    # ----- Load predictions from CRE -----
    df = pd.read_csv(test_cate_path)
    if "tau_hat" not in df.columns:
        raise ValueError(f"{test_cate_path} must contain column 'tau_hat'.")

    tau_hat = df["tau_hat"].values

    # ----- Load true tau (and mu0) from test .npz -----
    tau_true, mu0 = load_tau_mu0_from_npz(test_npz_path, rep=rep)

    if len(tau_true) != len(tau_hat):
        raise ValueError(
            f"Length mismatch: tau_hat ({len(tau_hat)}) vs tau_true ({len(tau_true)}). "
            "Ensure you are using the same replication index and test split."
        )

    # ----- PEHE -----
    pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))

    # ----- ATE Error -----
    ate_true = float(np.mean(tau_true))
    ate_hat = float(np.mean(tau_hat))
    ate_error = abs(ate_hat - ate_true)

    # ----- Policy Risk -----
    # Optimal policy: treat if true tau > 0
    pi_star = (tau_true > 0).astype(int)
    # Model (CRE) policy: treat if predicted tau_hat > 0
    pi_hat = (tau_hat > 0).astype(int)

    # Using the tau-only regret formula:
    # PolicyRisk = mean( (pi_star - pi_hat) * tau_true )
    policy_risk = float(np.mean((pi_star - pi_hat) * tau_true))

    # Alternative (equivalent) if mu0 is available:
    # if mu0 is not None:
    #     V_star = np.mean(mu0 + pi_star * tau_true)
    #     V_hat  = np.mean(mu0 + pi_hat  * tau_true)
    #     policy_risk_mu = float(V_star - V_hat)

    return pehe, ate_error, policy_risk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-cate", required=True,
                    help="Path to test_cate.csv output from CRE.")
    ap.add_argument("--test-npz", required=True,
                    help="Path to synthetic test .npz with true tau/mu0.")
    ap.add_argument("--rep", type=int, default=0,
                    help="Replication index used when running CRE (default: 0).")
    args = ap.parse_args()

    pehe, ate_err, pol_risk = compute_metrics(
        test_cate_path=args.test_cate,
        test_npz_path=args.test_npz,
        rep=args.rep
    )

    print("\n=== CRE Metrics (rep = {}) ===".format(args.rep))
    print(f"PEHE       : {pehe:.6f}")
    print(f"ATE Error  : {ate_err:.6f}")
    print(f"Policy Risk: {pol_risk:.6f}")

if __name__ == "__main__":
    main()
