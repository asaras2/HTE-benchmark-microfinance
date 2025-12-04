
library(ranger)
library(hte3)

# This script evaluates pretrained HTE models on test data
# It loads models from ./trained_models and evaluates on test data

################################################################################
# LOAD THE CALCULATE_METRICS FUNCTION
################################################################################

# Function to calculate PEHE, ATE risk, ATT risk, and Policy risk
calculate_metrics <- function(tau_hat, true_ite, T, Y, Y_cf, method_name = "Method", n_boot = 100) {
  cat("\n=== Calculating Metrics for", method_name, "===\n")
  
  # Helper function to compute metrics for a given set of indices
  compute_metrics_single <- function(idx) {
    tau_hat_sub <- tau_hat[idx]
    true_ite_sub <- true_ite[idx]
    T_sub <- T[idx]
    Y_sub <- Y[idx]
    Y_cf_sub <- Y_cf[idx]
    
    # 1. PEHE
    pehe <- sqrt(mean((tau_hat_sub - true_ite_sub)^2))
    
    # 2. ATE Risk
    ate_estimated <- mean(tau_hat_sub)
    ate_true <- mean(true_ite_sub)
    ate_risk <- abs(ate_estimated - ate_true)
    
    # 3. Policy Risk
    # First, construct Y(1) and Y(0) for all units
    # For treated units (T=1): Y(1) = Y (observed), Y(0) = Y_cf (counterfactual)
    # For control units (T=0): Y(0) = Y (observed), Y(1) = Y_cf (counterfactual)
    
    # Debug: Print vector lengths
    cat("Debug - Vector lengths: T_sub=", length(T_sub), 
        ", Y_sub=", length(Y_sub), 
        ", Y_cf_sub=", length(Y_cf_sub),
        ", tau_hat_sub=", length(tau_hat_sub),
        ", true_ite_sub=", length(true_ite_sub), "\n")
    
    # Validate inputs
    if (length(T_sub) == 0 || length(Y_sub) == 0 || length(Y_cf_sub) == 0) {
      cat("Error: Empty vectors detected!\n")
      return(c(pehe = NA, ate_risk = NA, policy_risk = NA, att_risk = NA))
    }
    
    # Ensure all vectors have the same length
    n_sub <- length(T_sub)
    if (length(Y_sub) != n_sub || length(Y_cf_sub) != n_sub || length(tau_hat_sub) != n_sub || length(true_ite_sub) != n_sub) {
      cat("Error: Vector length mismatch! n_sub=", n_sub, 
          ", Y_sub=", length(Y_sub),
          ", Y_cf_sub=", length(Y_cf_sub),
          ", tau_hat_sub=", length(tau_hat_sub),
          ", true_ite_sub=", length(true_ite_sub), "\n")
      return(c(pehe = NA, ate_risk = NA, policy_risk = NA, att_risk = NA))
    }
    
    # Y1: Potential outcome under treatment
    # If T=1, Y1 = Y (observed under treatment)
    # If T=0, Y1 = Y_cf (counterfactual - what would have happened under treatment)
    Y1 <- rep(NA, n_sub)
    Y1[T_sub == 1] <- Y_sub[T_sub == 1]
    Y1[T_sub == 0] <- Y_cf_sub[T_sub == 0]
    
    # Y0: Potential outcome under control
    # If T=0, Y0 = Y (observed under control)
    # If T=1, Y0 = Y_cf (counterfactual - what would have happened under control)
    Y0 <- rep(NA, n_sub)
    Y0[T_sub == 0] <- Y_sub[T_sub == 0]
    Y0[T_sub == 1] <- Y_cf_sub[T_sub == 1]
    
    # Policy assignments (1 = treat, 0 = control)
    policy_estimated <- as.numeric(tau_hat_sub > 0)
    policy_true <- as.numeric(true_ite_sub > 0)
    
    # Value of estimated policy: treat if policy=1, control if policy=0
    value_estimated_policy <- mean(
      policy_estimated * Y1 + (1 - policy_estimated) * Y0,
      na.rm = TRUE
    )
    
    # Value of true optimal policy
    value_true_policy <- mean(
      policy_true * Y1 + (1 - policy_true) * Y0,
      na.rm = TRUE
    )
    
    # Policy Risk = regret = value of optimal policy - value of estimated policy
    policy_risk <- value_true_policy - value_estimated_policy
    
    # ATT Risk
    att_estimated <- mean(tau_hat_sub[T_sub == 1], na.rm = TRUE)
    att_true <- mean(true_ite_sub[T_sub == 1], na.rm = TRUE)
    att_risk <- abs(att_estimated - att_true)
    
    c(pehe = pehe, ate_risk = ate_risk, policy_risk = policy_risk, att_risk = att_risk)
  }
  
  # Point estimates (on full data)
  n <- length(tau_hat)
  point_est <- compute_metrics_single(1:n)
  
  # Bootstrap for SD
  boot_results <- matrix(NA, nrow = n_boot, ncol = 4)
  colnames(boot_results) <- c("pehe", "ate_risk", "policy_risk", "att_risk")
  
  for(i in 1:n_boot) {
    boot_idx <- sample(1:n, n, replace = TRUE)
    
    # Check if bootstrap sample has both treated and control units
    # If not, resample until we get a valid sample
    max_attempts <- 10
    attempt <- 1
    while(length(unique(T[boot_idx])) < 2 && attempt < max_attempts) {
      boot_idx <- sample(1:n, n, replace = TRUE)
      attempt <- attempt + 1
    }
    
    # Try to compute metrics, skip if error occurs
    tryCatch({
      boot_results[i, ] <- compute_metrics_single(boot_idx)
    }, error = function(e) {
      # Skip this bootstrap iteration if error occurs
      boot_results[i, ] <- NA
    })
  }
  
  sd_est <- apply(boot_results, 2, sd, na.rm = TRUE)
  
  # Print results
  cat("PEHE:", round(point_est["pehe"], 4), "(SD:", round(sd_est["pehe"], 4), ")\n")
  cat("ATE Risk:", round(point_est["ate_risk"], 4), "(SD:", round(sd_est["ate_risk"], 4), ")\n")
  cat("ATT Risk:", round(point_est["att_risk"], 4), "(SD:", round(sd_est["att_risk"], 4), ")\n")
  cat("Policy Risk:", round(point_est["policy_risk"], 4), "(SD:", round(sd_est["policy_risk"], 4), ")\n")
  
  # Return metrics
  list(
    pehe = point_est["pehe"],
    pehe_sd = sd_est["pehe"],
    ate_risk = point_est["ate_risk"],
    ate_risk_sd = sd_est["ate_risk"],
    att_risk = point_est["att_risk"],
    att_risk_sd = sd_est["att_risk"],
    policy_risk = point_est["policy_risk"],
    policy_risk_sd = sd_est["policy_risk"]
  )
}

################################################################################
# LOAD TEST DATA AND EVALUATE
################################################################################

cat("\n=== Loading Test Data ===\n")

# Load test data from CSV
test_data <- read.csv("idhp_data_test.csv")

# Extract components
X_test <- test_data[, grep("^x", names(test_data))]
T_test <- test_data$t
Y_test <- test_data$yf

# Check if counterfactual and true ITE exist
if ("ycf" %in% names(test_data)) {
  Y_cf_test <- test_data$ycf
} else {
  cat("Warning: 'ycf' column not found in test data!\n")
  Y_cf_test <- rep(NA, nrow(test_data))
}

if ("ite" %in% names(test_data)) {
  true_ite_test <- test_data$ite
} else {
  cat("Warning: 'ite' column not found in test data!\n")
  true_ite_test <- rep(NA, nrow(test_data))
}

# Impute Y_cf if missing but true_ite is available
if (any(is.na(Y_cf_test)) && !any(is.na(true_ite_test))) {
  cat("Note: Imputing missing counterfactuals (Y_cf) using observed Y and True ITE.\n")
  # If T=1, Y_cf = Y(0) = Y(1) - ITE = Y - ITE
  # If T=0, Y_cf = Y(1) = Y(0) + ITE = Y + ITE
  Y_cf_test <- Y_test + (1 - 2 * T_test) * true_ite_test
}

cat("Test set loaded successfully!\n")
cat("Test size:", nrow(X_test), "\n")
cat("Number of covariates:", ncol(X_test), "\n")
cat("Treatment distribution:", table(T_test), "\n")
cat("Y_cf NAs:", sum(is.na(Y_cf_test)), "\n")
cat("true_ite NAs:", sum(is.na(true_ite_test)), "\n\n")

# Load saved models and make predictions
results <- list()
save_dir <- "./trained_models"

# 1. T-Learner predictions
cat("--- Loading T-Learner models and predicting ---\n")
t_model0 <- readRDS(file.path(save_dir, "t_learner_model0.rds"))
t_model1 <- readRDS(file.path(save_dir, "t_learner_model1.rds"))

mu0_t <- predict(t_model0, data = data.frame(X_test))$predictions
mu1_t <- predict(t_model1, data = data.frame(X_test))$predictions
tau_t <- mu1_t - mu0_t

results$t_learner <- calculate_metrics(
  tau_hat = tau_t,
  true_ite = true_ite_test,
  T = T_test,
  Y = Y_test,
  Y_cf = Y_cf_test,
  method_name = "T-Learner"
)

# 2. DR-Learner predictions
cat("\n--- Loading DR-Learner model and predicting ---\n")
dr_model <- readRDS(file.path(save_dir, "dr_learner_model.rds"))

tau_dr <- predict(dr_model, data = data.frame(X_test))$predictions

results$dr_learner <- calculate_metrics(
  tau_hat = tau_dr,
  true_ite = true_ite_test,
  T = T_test,
  Y = Y_test,
  Y_cf = Y_cf_test,
  method_name = "DR-Learner"
)

# 3. HTE3-style predictions
cat("\n--- Loading HTE3-style models and predicting ---\n")
hte3_model0 <- readRDS(file.path(save_dir, "hte3_model0.rds"))
hte3_model1 <- readRDS(file.path(save_dir, "hte3_model1.rds"))

mu0_hte3 <- predict(hte3_model0, data = data.frame(X_test))$predictions
mu1_hte3 <- predict(hte3_model1, data = data.frame(X_test))$predictions
tau_hte3 <- mu1_hte3 - mu0_hte3

results$hte3 <- calculate_metrics(
  tau_hat = tau_hte3,
  true_ite = true_ite_test,
  T = T_test,
  Y = Y_test,
  Y_cf = Y_cf_test,
  method_name = "HTE3-style"
)

# Create summary table
cat("\n=== FINAL EVALUATION RESULTS ON TEST DATA ===\n")
summary_df <- data.frame(
  Method = c("T-Learner", "DR-Learner", "HTE3-style"),
  PEHE = c(results$t_learner$pehe, results$dr_learner$pehe, results$hte3$pehe),
  PEHE_SD = c(results$t_learner$pehe_sd, results$dr_learner$pehe_sd, results$hte3$pehe_sd),
  ATE_Risk = c(results$t_learner$ate_risk, results$dr_learner$ate_risk, results$hte3$ate_risk),
  ATE_Risk_SD = c(results$t_learner$ate_risk_sd, results$dr_learner$ate_risk_sd, results$hte3$ate_risk_sd),
  Policy_Risk = c(results$t_learner$policy_risk, results$dr_learner$policy_risk, results$hte3$policy_risk),
  Policy_Risk_SD = c(results$t_learner$policy_risk_sd, results$dr_learner$policy_risk_sd, results$hte3$policy_risk_sd)
)

# Round numeric columns only
numeric_cols <- sapply(summary_df, is.numeric)
summary_df[numeric_cols] <- round(summary_df[numeric_cols], 4)
print(summary_df)

cat("\n✓ Evaluation complete!\n")
cat("✓ Best method by PEHE:", summary_df$Method[which.min(summary_df$PEHE)], "\n")
cat("✓ Best method by Policy Risk:", summary_df$Method[which.min(summary_df$Policy_Risk)], "\n")
