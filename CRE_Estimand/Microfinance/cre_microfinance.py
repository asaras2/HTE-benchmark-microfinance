import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from rulefit import RuleFit
HAS_RULEFIT = True

# ---------------------------------------------------------------------
# 1. Load merged microfinance data
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Microfinance"
data_path = DATA_DIR / "microfinance_merged.csv"

print("Loading data from:", data_path)
df = pd.read_csv(data_path, low_memory=False)
print("Rows:", len(df), " | Columns:", len(df.columns))

# ---------------------------------------------------------------------
# 2. Define outcome Y, treatment T, and covariates X
# ---------------------------------------------------------------------

T_COL = "shgparticipate"
Y_COL = "savings"

if T_COL not in df.columns or Y_COL not in df.columns:
    raise ValueError(
        f"Expected columns '{T_COL}' and '{Y_COL}' to exist in the data. "
        f"Please rename your treatment/outcome columns or update T_COL/Y_COL "
        f"in cre_microfinance.py."
    )

# Map common Yes/No patterns to 1/0 for treatment and outcome
def yes_no_to_binary(series):
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    }
    return s.map(mapping)

T = yes_no_to_binary(df[T_COL]).astype(float).values
Y = yes_no_to_binary(df[Y_COL]).astype(float).values

print(f"\nTreatment '{T_COL}' value counts (after mapping):")
print(pd.Series(T).value_counts(dropna=False))

print(f"\nOutcome '{Y_COL}' value counts (after mapping):")
print(pd.Series(Y).value_counts(dropna=False))

mask = ~np.isnan(T) & ~np.isnan(Y)
n_before = len(T)
T = T[mask]
Y = Y[mask]
df = df.loc[mask].reset_index(drop=True)

print(f"\nDropped {n_before - len(T)} rows with NaN in treatment or outcome.")
print("Remaining rows:", len(T))

# Covariates chosen (as per your friend's suggestion)
selected_covariates = [
    "age",
    "resp_gend",
    "rationcard",
    "workflag",
    "electricity",
    "latrine",
    "ownrent",
]

missing_covs = [c for c in selected_covariates if c not in df.columns]
if missing_covs:
    raise ValueError(f"The following covariates are missing in the data: {missing_covs}")

df_X = df[selected_covariates].copy()

# Convert simple Yes/No covariates to 1/0 where applicable,
# leave others for one-hot encoding.
for col in df_X.columns:
    if df_X[col].dtype == "object":
        vals = set(df_X[col].dropna().astype(str).str.strip().str.lower().unique())
        if vals.issubset({"yes", "no", "0", "1"}):
            df_X[col] = yes_no_to_binary(df_X[col])

# One-hot encode remaining categoricals (e.g., rationcard types, gender strings)
X = pd.get_dummies(df_X, drop_first=True)
# Handle any remaining missing values in X
missing_per_col = X.isna().sum()
if missing_per_col.sum() > 0:
    print("\nMissing values in X before imputation:")
    print(missing_per_col[missing_per_col > 0])
    # Simple imputation: fill NaNs with 0 (baseline / "no" / lowest category)
    X = X.fillna(0)

    # Ensure all features are numeric floats (avoid boolean dtype issues in RuleFit)
X = X.astype(float)

print("\nFinal covariate matrix shape:", X.shape)

# ---------------------------------------------------------------------
# 3. Nuisance models: propensity e(X), outcome models m0(X), m1(X)
# ---------------------------------------------------------------------
print("\nFitting nuisance models (propensity + outcome regressors)...")

X_train, X_test, Y_train, Y_test, T_train, T_test = train_test_split(
    X, Y, T, test_size=0.2, random_state=42, stratify=T
)

# Propensity model: P(T=1 | X)
ps_model = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=20,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
ps_model.fit(X_train, T_train)
e_hat = ps_model.predict_proba(X)[:, 1]

print("  Propensity scores: min =", e_hat.min(), ", max =", e_hat.max())

# Outcome models: E[Y | T=t, X]
rf_treated = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)
rf_control = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

rf_treated.fit(X[T == 1], Y[T == 1])
rf_control.fit(X[T == 0], Y[T == 0])

m1_hat = rf_treated.predict(X)
m0_hat = rf_control.predict(X)

# ---------------------------------------------------------------------
# 4. Doubly robust pseudo-outcome tau_dr (individual CATE proxy)
# ---------------------------------------------------------------------
print("\nComputing doubly-robust pseudo-outcomes...")

eps = 1e-6
e_hat_clipped = np.clip(e_hat, eps, 1 - eps)

m_T = np.where(T == 1, m1_hat, m0_hat)

dr_part = (T - e_hat_clipped) / (e_hat_clipped * (1 - e_hat_clipped)) * (Y - m_T)
tau_dr = dr_part + (m1_hat - m0_hat)

print("tau_dr summary: mean =", tau_dr.mean(), ", std =", tau_dr.std())

# Save tau_dr for possible reuse
df_out = df.copy()
df_out["tau_dr"] = tau_dr
tau_path = DATA_DIR / "microfinance_tau_dr.csv"
df_out.to_csv(tau_path, index=False)
print("Saved DR pseudo-outcomes to:", tau_path)

# ---------------------------------------------------------------------
# 5. Rule ensemble on tau_dr (CRE interpretability layer)
# ---------------------------------------------------------------------
if HAS_RULEFIT:
    print("\nFitting RuleFit rule ensemble on tau_dr...")

    rf = RuleFit(
        tree_generator=None,   # default gradient boosting trees
        max_rules=2000,
        memory_par=0.01,
        rfmode="regress",
        random_state=42,
    )
    rf.fit(X.values, tau_dr)

    rules = rf.get_rules()
    rules = rules[(rules.coef != 0) & (rules.type == "rule")]
    rules = rules.sort_values("coef", ascending=False)

    rules_path = DATA_DIR / "microfinance_cre_rules.csv"
    rules.to_csv(rules_path, index=False)
    print("Saved rules to:", rules_path)

    print("\nTop 10 rules by coefficient:")
    print(rules.head(10)[["rule", "coef", "support"]])
else:
    print("\nrulefit not available; skipping rule extraction step.")

print("\nDone.")

# =====================================================
# 6. CRE Evaluation Metrics (match TransTEE)
# =====================================================

metrics = {}

# ATE Predicted
metrics["ate_pred"] = float(np.mean(tau_dr))

# Policy Risk Proxy
policy = (tau_dr > 0).astype(int)
metrics["policy_risk_proxy"] = float(-np.mean(policy * tau_dr))

# Propensity Balance
metrics["propensity_balance"] = float(
    abs(np.mean(T / e_hat_clipped) - np.mean((1 - T) / (1 - e_hat_clipped)))
)

# Outcome MSE (from nuisance models)
y_pred = m1_hat * T + m0_hat * (1 - T)
metrics["outcome_mse"] = float(np.mean((Y - y_pred) ** 2))

# HTE Std
metrics["hte_std"] = float(np.std(tau_dr))

# Print results
print("\nCRE Evaluation Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.6f}")

# Save CSV
metrics_path = DATA_DIR / "microfinance_cre_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print("\nSaved CRE metrics to:", metrics_path)

# =====================================================
# 7. Multiple CRE Runs (50 reps) for distribution metrics
# =====================================================

print("\nRunning 50 CRE replications to compute distribution metrics...\n")

N_REPS = 50
rows = []

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

for rep in range(1, N_REPS + 1):
    print(f"--- Replication {rep} ---")

    # Different random seed each time
    np.random.seed(rep)

    # New train/test split each rep
    X_train, X_test, Y_train, Y_test, T_train, T_test = train_test_split(
        X, Y, T, test_size=0.2, random_state=rep, stratify=T
    )

    # Fresh propensity model for this rep
    ps_model_rep = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=20,
        max_depth=None,
        n_jobs=-1,
        random_state=rep,
    )
    ps_model_rep.fit(X_train, T_train)
    e_hat_rep = ps_model_rep.predict_proba(X)[:, 1]
    e_hat_rep_clipped = np.clip(e_hat_rep, 1e-6, 1 - 1e-6)

    # Fresh outcome models for this rep
    rf_treated_rep = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=rep,
    )
    rf_control_rep = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=rep,
    )

    rf_treated_rep.fit(X[T == 1], Y[T == 1])
    rf_control_rep.fit(X[T == 0], Y[T == 0])

    m1_rep = rf_treated_rep.predict(X)
    m0_rep = rf_control_rep.predict(X)
    m_T_rep = np.where(T == 1, m1_rep, m0_rep)

    # DR pseudo-outcome tau for this rep
    dr_part_rep = (T - e_hat_rep_clipped) / (e_hat_rep_clipped * (1 - e_hat_rep_clipped)) * (Y - m_T_rep)
    tau_rep = dr_part_rep + (m1_rep - m0_rep)

    # Metrics for this replication
    policy_rep = (tau_rep > 0).astype(int)

    metrics_rep = {
        "rep": rep,
        "ate_pred": float(np.mean(tau_rep)),
        "policy_risk_proxy": float(-np.mean(policy_rep * tau_rep)),
        "propensity_balance": float(
            abs(np.mean(T / e_hat_rep_clipped) - np.mean((1 - T) / (1 - e_hat_rep_clipped)))
        ),
        "outcome_mse": float(np.mean((Y - (m1_rep * T + m0_rep * (1 - T))) ** 2)),
        "hte_std": float(np.std(tau_rep)),
    }

    rows.append(metrics_rep)

# Save all 50-rep results
df_cre_reps = pd.DataFrame(rows)
dist_path = DATA_DIR / "microfinance_cre_metrics_50reps.csv"
df_cre_reps.to_csv(dist_path, index=False)

print("\nSaved 50-rep CRE distribution metrics to:", dist_path)