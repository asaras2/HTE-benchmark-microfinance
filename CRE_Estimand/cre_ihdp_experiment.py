import numpy as np
import pandas as pd
from cre_pipeline import run as cre_run

TRAIN_PATH = "/Users/akshunagnihotri/Documents/CS-520_Proj/ihdp_npci_1-100.train.npz"
TEST_PATH  = "/Users/akshunagnihotri/Documents/CS-520_Proj/ihdp_npci_1-100.test.npz"
N_REPS = 100

def load_tau_from_ihdp_test(test_path, rep):
    d = np.load(test_path, allow_pickle=True)
    x = d["x"]      # (n, d, R)
    mu0 = d["mu0"]  # (n, R)
    mu1 = d["mu1"]  # (n, R)

    n, d_feat, R = x.shape
    if rep >= R:
        raise ValueError(f"rep {rep} out of range for x shape {x.shape}")

    tau = (mu1[:, rep] - mu0[:, rep]).ravel()
    return tau

def compute_metrics_for_rep(test_cate_path, test_npz_path, rep):
    df = pd.read_csv(test_cate_path)
    if "tau_hat" not in df.columns:
        raise ValueError("test_cate.csv must contain 'tau_hat'.")

    tau_hat = df["tau_hat"].values
    tau_true = load_tau_from_ihdp_test(test_npz_path, rep=rep)

    if len(tau_true) != len(tau_hat):
        raise ValueError(
            f"[rep {rep}] Length mismatch: tau_hat ({len(tau_hat)}) vs tau_true ({len(tau_true)})."
        )

    pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))

    ate_true = float(np.mean(tau_true))
    ate_hat = float(np.mean(tau_hat))
    ate_error = abs(ate_hat - ate_true)

    pi_star = (tau_true > 0).astype(int)
    pi_hat = (tau_hat > 0).astype(int)
    policy_risk = float(np.mean((pi_star - pi_hat) * tau_true))

    return pehe, ate_error, policy_risk

def main():
    all_pehe, all_ate, all_pol = [], [], []

    for rep in range(N_REPS):
        print(f"\n===== Rep {rep} =====")
        cre_run(TRAIN_PATH, TEST_PATH, rep=rep)

        pehe, ate_err, pol_risk = compute_metrics_for_rep(
            test_cate_path="test_cate.csv",
            test_npz_path=TEST_PATH,
            rep=rep
        )

        print(f"PEHE       : {pehe:.6f}")
        print(f"ATE Error  : {ate_err:.6f}")
        print(f"Policy Risk: {pol_risk:.6f}")

        all_pehe.append(pehe)
        all_ate.append(ate_err)
        all_pol.append(pol_risk)

    pehe_mean, pehe_std = np.mean(all_pehe), np.std(all_pehe)
    ate_mean, ate_std = np.mean(all_ate), np.std(all_ate)
    pol_mean, pol_std = np.mean(all_pol), np.std(all_pol)

    print("\n===== IHDP CRE Summary over {} reps =====".format(N_REPS))
    print(f"PEHE       : {pehe_mean:.3f} ± {pehe_std:.3f}")
    print(f"ATE Error  : {ate_mean:.3f} ± {ate_std:.3f}")
    print(f"Policy Risk: {pol_mean:.3f} ± {pol_std:.3f}")

    pd.DataFrame({
        "rep": np.arange(N_REPS),
        "pehe": all_pehe,
        "ate_error": all_ate,
        "policy_risk": all_pol
    }).to_csv("cre_ihdp_metrics_per_rep.csv", index=False)

if __name__ == "__main__":
    main()