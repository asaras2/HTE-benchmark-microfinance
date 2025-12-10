import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import _tree

# ============================================================
#                NPZ LOADING HELPERS
# ============================================================

def _pick(d, names, required=True):
    keys = {k.lower(): k for k in d.files}
    for name in names:
        k = name.lower()
        if k in keys:
            return d[keys[k]]
    if required:
        raise ValueError(f"Missing one of {names} in {list(d.files)}")
    return None

def load_synth_rep(path, rep=0):
    """
    Synthetic:
      x:   (R, n, d) or (n, d)
      t:   (R, n) or (n,)
      yf:  (R, n) or (n,)
      mu0, mu1, tau similarly (R, n) or (n,)
    """
    d = np.load(path, allow_pickle=True)

    X_all   = _pick(d, ["x"])
    t_all   = _pick(d, ["t"])
    yf_all  = _pick(d, ["yf", "y", "outcome"], required=False)
    mu0_all = _pick(d, ["mu0", "y0", "m0"], required=False)
    mu1_all = _pick(d, ["mu1", "y1", "m1"], required=False)
    tau_all = _pick(d, ["tau", "ite", "cate", "tau_true", "tau_y1_y0"], required=False)

    X_all = np.asarray(X_all)
    if X_all.ndim == 3:
        if rep >= X_all.shape[0]:
            raise ValueError(f"{path}: rep {rep} out of range for X {X_all.shape}")
        X = X_all[rep]
    elif X_all.ndim == 2:
        X = X_all
    else:
        raise ValueError(f"{path}: bad X shape {X_all.shape}")

    t_all = np.asarray(t_all)
    if t_all.ndim == 2:
        t = t_all[rep]
    elif t_all.ndim == 1:
        t = t_all
    else:
        raise ValueError(f"{path}: bad t shape {t_all.shape}")
    t = t.astype(int).ravel()

    if yf_all is not None:
        yf_all = np.asarray(yf_all)
        if yf_all.ndim == 2:
            y = yf_all[rep]
        elif yf_all.ndim == 1:
            y = yf_all
        else:
            raise ValueError(f"{path}: bad yf shape {yf_all.shape}")
        y = y.ravel()
    elif (mu0_all is not None) and (mu1_all is not None):
        mu0_all = np.asarray(mu0_all)
        mu1_all = np.asarray(mu1_all)
        if mu0_all.ndim == 2 and mu1_all.ndim == 2:
            mu0 = mu0_all[rep].ravel()
            mu1 = mu1_all[rep].ravel()
        else:
            raise ValueError(f"{path}: bad mu0/mu1 shapes {mu0_all.shape}, {mu1_all.shape}")
        y = t * mu1 + (1 - t) * mu0
    else:
        raise ValueError(f"{path}: no yf or (mu0,mu1) for outcome")

    tau = None
    if tau_all is not None:
        tau_all = np.asarray(tau_all)
        if tau_all.ndim == 2:
            tau = tau_all[rep].ravel()
        elif tau_all.ndim == 1:
            tau = tau_all.ravel()
    elif (mu0_all is not None) and (mu1_all is not None):
        tau = (mu1 - mu0).ravel()

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{path}: X must be 2D after slice, got {X.shape}")
    n = X.shape[0]

    if len(t) != n or len(y) != n:
        raise ValueError(f"{path}: length mismatch X={n}, t={len(t)}, y={len(y)}")
    if tau is not None and len(tau) != n:
        tau = None  # drop inconsistent tau

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["t"] = t
    df["y"] = y
    if tau is not None:
        df["tau"] = tau
    return df

def load_ihdp_rep(path, rep=0):
    """
    IHDP (ihdp_npci_1-100.*.npz) format:
      x:   (n, d, R)
      t:   (n, R)
      yf:  (n, R)
      mu0: (n, R)
      mu1: (n, R)
    We build:
      y   = factual yf
      tau = mu1 - mu0
    """
    d = np.load(path, allow_pickle=True)

    X_all   = d["x"]
    t_all   = d["t"]
    yf_all  = d["yf"]
    mu0_all = d["mu0"]
    mu1_all = d["mu1"]

    X_all = np.asarray(X_all)
    if X_all.ndim != 3:
        raise ValueError(f"{path}: expected x as (n,d,R), got {X_all.shape}")
    n, d_feat, R = X_all.shape
    if rep >= R:
        raise ValueError(f"{path}: rep {rep} out of range for R={R}")
    X = X_all[:, :, rep]

    def col_rep(arr, label):
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(f"{path}: bad {label} shape {arr.shape}")
        if rep >= arr.shape[1]:
            raise ValueError(f"{path}: rep {rep} out of range for {label} shape {arr.shape}")
        return arr[:, rep]

    t   = col_rep(t_all,  "t").astype(int).ravel()
    yf  = col_rep(yf_all, "yf").ravel()
    mu0 = col_rep(mu0_all, "mu0").ravel()
    mu1 = col_rep(mu1_all, "mu1").ravel()
    tau = (mu1 - mu0).ravel()
    y   = yf

    if X.ndim != 2:
        raise ValueError(f"{path}: X slice must be 2D, got {X.shape}")
    if len(t) != n or len(y) != n or len(tau) != n:
        raise ValueError(f"{path}: length mismatch after slice")

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["t"] = t
    df["y"] = y
    df["tau"] = tau
    return df

def load_table(path, rep=0):
    pl = path.lower()
    if pl.endswith(".npz"):
        if "ihdp_npci" in pl:
            return load_ihdp_rep(path, rep=rep)
        else:
            return load_synth_rep(path, rep=rep)
    if pl.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

# ============================================================
#                       DR CORE
# ============================================================

def split_XTY(df, tcol="t", ycol="y", drop_cols=None):
    drop_cols = drop_cols or []
    use = [c for c in df.columns if c not in [tcol, ycol] + drop_cols]
    X = df[use].copy()
    t = df[tcol].astype(int).values
    y = df[ycol].values
    return X, t, y, use

def fit_dr_models(X, t, y):
    prop = LogisticRegression(max_iter=2000)
    prop.fit(X, t)
    e = prop.predict_proba(X)[:, 1]

    m0 = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=7,
        n_jobs=-1,
    )
    m1 = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=8,
        n_jobs=-1,
    )
    m0.fit(X[t == 0], y[t == 0])
    m1.fit(X[t == 1], y[t == 1])

    mu0 = m0.predict(X)
    mu1 = m1.predict(X)

    eps = 1e-6
    w1 = t / np.clip(e, eps, 1 - eps)
    w0 = (1 - t) / np.clip(1 - e, eps, 1 - eps)
    dr_tau = (mu1 - mu0) + w1 * (y - mu1) - w0 * (y - mu0)

    return {"prop": prop, "m0": m0, "m1": m1}, dr_tau

def predict_cate(core, X):
    return core["m1"].predict(X) - core["m0"].predict(X)

# ============================================================
#                       RULE MINING
# ============================================================

def extract_rules_from_tree(tree, feat_names):
    rules = []
    t = tree.tree_

    def rec(node, conds):
        if t.feature[node] != -2:
            f = feat_names[t.feature[node]]
            thr = t.threshold[node]
            rec(t.children_left[node],  conds + [(f, "<=", thr)])
            rec(t.children_right[node], conds + [(f, ">",  thr)])
        else:
            rules.append(conds)

    rec(0, [])
    return rules

def extract_rules_from_forest(forest, feat_names):
    seen = set()
    uniq = []
    for est in forest.estimators_:
        for r in extract_rules_from_tree(est, feat_names):
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                uniq.append(r)
    return uniq

def rules_to_matrix(rules, X):
    n = len(X)
    m = len(rules)
    M = np.zeros((n, m), dtype=np.int8)
    Xv = X.values
    col_idx = {c: i for i, c in enumerate(X.columns)}
    for j, rule in enumerate(rules):
        mask = np.ones(n, dtype=bool)
        for f, op, thr in rule:
            c = col_idx[f]
            if op == "<=":
                mask &= Xv[:, c] <= thr
            else:
                mask &= Xv[:, c] > thr
        M[mask, j] = 1
    return M

# ============================================================
#                       CRE FIT / PREDICT
# ============================================================

def fit_cre(X, t, y, feat_names,
            y_target=None,
            max_depth=5,
            min_support=0.05,
            max_rules=200):
    core, dr_tau = fit_dr_models(X, t, y)
    target = dr_tau if y_target is None else y_target

    min_leaf = max(5, int(min_support * len(X)))
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=11,
        n_jobs=-1,
    )
    rf.fit(X, target)

    rules_all = extract_rules_from_forest(rf, feat_names)

    def support(rule):
        mask = np.ones(len(X), dtype=bool)
        Xv = X.values
        col_idx = {c: i for i, c in enumerate(X.columns)}
        for f, op, thr in rule:
            c = col_idx[f]
            if op == "<=":
                mask &= Xv[:, c] <= thr
            else:
                mask &= Xv[:, c] > thr
        return float(mask.mean())

    rules = [r for r in rules_all if support(r) >= min_support]
    rules = rules[:max_rules]

    if not rules:
        print("[CRE] No rules passed support threshold; using DR core only.")
        return {"core": core, "rules": [], "coefs": np.array([]), "feat_names": feat_names}

    R = rules_to_matrix(rules, X)
    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(R, target)
    coefs = lasso.coef_
    sel = np.abs(coefs) > 1e-6

    if sel.sum() == 0:
        print("[CRE] Lasso shrank all rules; using DR core only.")
        return {"core": core, "rules": [], "coefs": np.array([]), "feat_names": feat_names}

    rules_sel = [rules[i] for i in np.where(sel)[0]]
    coefs_sel = coefs[sel]

    print(f"[CRE] Candidate rules: {len(rules)} | Selected: {len(rules_sel)}")
    return {"core": core, "rules": rules_sel, "coefs": coefs_sel, "feat_names": feat_names}

def predict_cre(model, X):
    base = predict_cate(model["core"], X)
    if not model["rules"]:
        return base
    R = rules_to_matrix(model["rules"], X)
    return base + R @ model["coefs"]

# ============================================================
#                       METRICS / RUNNER
# ============================================================

def pehe(y_true_tau, y_hat_tau):
    return float(np.sqrt(np.mean((y_true_tau - y_hat_tau) ** 2)))

def run(train_path, test_path, rep=0):
    train = load_table(train_path, rep=rep)
    test  = load_table(test_path,  rep=rep)

    drop_tr = ["tau"] if "tau" in train.columns else []
    drop_te = ["tau"] if "tau" in test.columns else []

    Xtr, ttr, ytr, feats = split_XTY(train, "t", "y", drop_cols=drop_tr)
    Xte, tte, yte, _     = split_XTY(test,  "t", "y", drop_cols=drop_te)

    print(f"[Info] Train: {train.shape} | Test: {test.shape} | Features: {len(feats)}")

    y_target = train["tau"].values if "tau" in train.columns else None
    model = fit_cre(Xtr, ttr, ytr, feats, y_target=y_target)

    tau_hat = predict_cre(model, Xte)
    out = pd.DataFrame({"tau_hat": tau_hat})
    if "tau" in test.columns:
        out["tau_true"] = test["tau"].values
    out.to_csv("test_cate.csv", index=False)

    rows = []
    if len(model["rules"]):
        for w, rule in sorted(zip(model["coefs"], model["rules"]), key=lambda z: -abs(z[0])):
            rows.append({
                "weight": float(w),
                "rule": " AND ".join(f"{f} {op} {thr:.5g}" for (f, op, thr) in rule),
                "length": len(rule),
            })
    pd.DataFrame(rows).to_csv("cre_rules.csv", index=False)

    print("\n=== CRE Results ===")
    if "tau" in test.columns:
        print(f"PEHE (test): {pehe(test['tau'].values, tau_hat):.6f}")
    print(f"Selected rules: {len(model['rules'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--rep", type=int, default=0)
    args = parser.parse_args()
    run(args.train, args.test, rep=args.rep)
