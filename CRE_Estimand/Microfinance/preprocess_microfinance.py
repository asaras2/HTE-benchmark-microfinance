import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Microfinance"

ind_path = DATA_DIR / "individual_characteristics.dta"
hh_path  = DATA_DIR / "household_characteristics.dta"

print("Loading individual data from:", ind_path)
print("Loading household data from:", hh_path)

ind = pd.read_stata(ind_path)
hh  = pd.read_stata(hh_path)

print("\n--- Basic sizes ---")
print("Individual rows:", len(ind))
print("Household rows:", len(hh))
print("Unique HH in ind:", ind['hhid'].nunique())
print("Unique HH in hh:", hh['hhid'].nunique())

df_merged = ind.merge(hh, on="hhid", how="left", suffixes=("_ind", "_hh"))

print("\nMerged rows:", len(df_merged))
missing_hh = df_merged[df_merged.isna().any(axis=1)]
print("Rows with missing values:", len(missing_hh))

out_dta  = DATA_DIR / "microfinance_merged.dta"
out_csv  = DATA_DIR / "microfinance_merged.csv"

df_merged.to_stata(out_dta, write_index=False)
df_merged.to_csv(out_csv, index=False)

print("\nSaved merged data:")
print("  -", out_dta)
print("  -", out_csv)
print("\nDone.")