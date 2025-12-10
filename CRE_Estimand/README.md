CRE_Estimand – Causal Rule Ensemble for Heterogeneous Treatment Effect Estimation

This repository implements a full Causal Rule Ensemble (CRE) pipeline for heterogeneous treatment effect (HTE) estimation across three datasets: synthetic data, the IHDP benchmark dataset, and a real microfinance dataset consisting of individual-level and household-level information. The goal of the project is to compute doubly robust (DR) pseudo-outcomes, fit interpretable RuleFit models, and evaluate estimator performance using standard HTE metrics comparable to TransTEE and EP-Learner, including ATE predictions, policy value proxies, propensity balance, outcome MSE, and the variability of heterogeneous treatment effects across individuals. The repository includes end-to-end scripts for preprocessing, estimation, rule extraction, and repeated-run evaluation.

The project structure is designed to separate core pipelines, dataset-specific experiments, data storage, and output results. A recommended structure is:

CRE_Estimand/
  README.md
  requirements.txt
  .gitignore

  src/
    cre_pipeline.py
    cre_ihdp_experiment.py
    cre_metrics.py

  experiments/
    Microfinance/
      preprocess_microfinance.py
      cre_microfinance.py

  data/                               (gitignored)
    microfinance/
      individual_characteristics.dta
      household_characteristics.dta
      microfinance_merged.csv
      microfinance_merged.dta
    ihdp/
    synthetic/

  results/
    ihdp/
      cre_ihdp_metrics_per_rep.csv
      cre_rules.csv
      test_cate.csv

    microfinance/
      microfinance_tau_dr.csv
      microfinance_cre_rules.csv
      microfinance_cre_metrics_50reps.csv

To install and run the repository, first clone it using:

git clone https://github.com/<your-username>/CRE_Estimand.git
cd CRE_Estimand


Create a virtual environment and install the dependencies:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


A minimal requirements file may include:

numpy
pandas
scikit-learn
rulefit


Additional packages can be added based on your environment.

For the microfinance dataset, the workflow consists of preprocessing followed by CRE estimation. The preprocessing script merges individual_characteristics.dta and household_characteristics.dta using hhid and saves both .csv and .dta versions of the merged file. Run:

python3 experiments/Microfinance/preprocess_microfinance.py


Then run CRE estimation:

python3 experiments/Microfinance/cre_microfinance.py


The microfinance CRE pipeline uses shgparticipate as the treatment variable, savings as the binary outcome, and the following covariates: age, resp_gend, rationcard, workflag, electricity, latrine, and ownrent. The script performs one-hot encoding for categoricals, fits propensity (RandomForestClassifier) and outcome regressors (RandomForestRegressor for treated and control groups), computes DR pseudo-outcomes, fits a RuleFit model for rule-based interpretability, and generates 50 repeated replications to compute metrics: ate_pred, policy_risk_proxy, propensity_balance, outcome_mse, and hte_std. Outputs include microfinance_tau_dr.csv, microfinance_cre_rules.csv, and microfinance_cre_metrics_50reps.csv.

For IHDP experiments, run:

python3 src/cre_ihdp_experiment.py


This script loads the IHDP dataset, computes DR pseudo-outcomes, extracts rules, and produces repeated-run metrics saved as cre_ihdp_metrics_per_rep.csv, along with associated rule files.

Synthetic data experiments use:

python3 src/cre_pipeline.py


The repository uses repeated evaluation to measure stability across random seeds, random forest fitting, and RuleFit tree generation. All scripts use fixed random_state values to improve reproducibility.

Data files (.dta, .csv) should be stored inside the data/ folder and are intentionally excluded from Git commits. You can add new datasets by creating new subfolders under data/. Generated results are stored inside the results/ directory, separated by dataset, and include DR pseudo-outcomes, CRE rule tables, and repeated-run metric summaries.

This repository is designed for research and benchmarking in CS-520 and can be extended to additional datasets or estimators. The codebase is intentionally readable and modular, making it easy to modify hyperparameters, add new evaluation metrics, or integrate additional HTE models.