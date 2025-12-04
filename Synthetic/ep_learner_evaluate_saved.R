
# LOAD AND EVALUATE SAVED EP-LEARNER MODEL
# Use this script to evaluate a previously trained model

library(ranger)

MODEL_DIR <- "./ep_learner_models"

cat("\n=== Loading Saved EP-Learner Model ===\n")

# Load the saved model
ep_model <- readRDS(file.path(MODEL_DIR, "ep_cate_model.rds"))
config <- readRDS(file.path(MODEL_DIR, "config.rds"))

cat("Model loaded successfully!\n")
cat("Training configuration:\n")
cat("  - Num trees:", config$num_trees, "\n")
cat("  - Num folds:", config$num_folds, "\n")
cat("  - Training samples:", config$n_train, "\n")
cat("  - Features:", config$p_features, "\n")
cat("  - Trained on:", as.character(config$training_date), "\n\n")

################################################################################
# LOAD TEST DATA
################################################################################

cat("=== Loading Test Data ===\n")
test_data <- read.csv("synth_test.csv")

X_test <- test_data[, grep("^x[0-9]", names(test_data))]
T_test <- test_data$t
Y_test <- test_data$yf
Y_cf_test <- if("ycf" %in% names(test_data)) test_data$ycf else rep(NA, nrow(test_data))
true_ite_test <- if("ite" %in% names(test_data)) test_data$ite else if("tau" %in% names(test_data)) test_data$tau else rep(NA, nrow(test_data))

# If Y_cf is missing but true_ite is available, compute Y_cf
if (any(is.na(Y_cf_test)) && !any(is.na(true_ite_test))) {
  cat("Note: Y_cf is missing. Computing from true ITE...\n")
  # For treated (T=1): Y_cf = Y(0) = Y(1) - ITE = Y - ITE
  # For control (T=0): Y_cf = Y(1) = Y(0) + ITE = Y + ITE
  Y_cf_test <- Y_test - (2 * T_test - 1) * true_ite_test
  cat("Y_cf computed successfully!\n")
}

cat("Test data loaded!\n")
cat("N =", nrow(X_test), "\n")
cat("Y_cf NAs:", sum(is.na(Y_cf_test)), "\n")
cat("true_ite NAs:", sum(is.na(true_ite_test)), "\n\n")

################################################################################
# MAKE PREDICTIONS
################################################################################

cat("=== Making Predictions ===\n")
tau_ep <- predict(ep_model, data = data.frame(X_test))$predictions
cat("Predictions complete!\n\n")

################################################################################
# CALCULATE METRICS WITH BOOTSTRAP FOR STANDARD DEVIATIONS
################################################################################

cat("=== Calculating Metrics with Bootstrap ===\n")

# Helper function to compute all metrics for a given set of indices
compute_metrics_single <- function(idx) {
  tau_hat_sub <- tau_ep[idx]
  true_ite_sub <- true_ite_test[idx]
  T_sub <- T_test[idx]
  Y_sub <- Y_test[idx]
  Y_cf_sub <- Y_cf_test[idx]
  
  # 1. PEHE
  pehe <- sqrt(mean((tau_hat_sub - true_ite_sub)^2, na.rm = TRUE))
  
  # 2. ATE Error
  ate_estimated <- mean(tau_hat_sub, na.rm = TRUE)
  ate_true <- mean(true_ite_sub, na.rm = TRUE)
  ate_error <- abs(ate_estimated - ate_true)
  
  # 3. Policy Risk - with proper Y1/Y0 construction
  n_sub <- length(T_sub)
  
  # Check for NAs
  if (any(is.na(Y_cf_sub))) {
    cat("Warning: Y_cf contains", sum(is.na(Y_cf_sub)), "NAs\n")
    return(c(pehe = pehe, ate_error = ate_error, policy_risk = NA))
  }
  
  # Construct Y1 and Y0
  Y1 <- rep(NA, n_sub)
  Y1[T_sub == 1] <- Y_sub[T_sub == 1]      # Observed for treated
  Y1[T_sub == 0] <- Y_cf_sub[T_sub == 0]  # Counterfactual for control
  
  Y0 <- rep(NA, n_sub)
  Y0[T_sub == 0] <- Y_sub[T_sub == 0]      # Observed for control
  Y0[T_sub == 1] <- Y_cf_sub[T_sub == 1]  # Counterfactual for treated
  
  # Check if Y1 and Y0 are properly constructed
  if (any(is.na(Y1)) || any(is.na(Y0))) {
    cat("Warning: Y1 or Y0 contains NAs after construction\n")
    cat("Y1 NAs:", sum(is.na(Y1)), ", Y0 NAs:", sum(is.na(Y0)), "\n")
    return(c(pehe = pehe, ate_error = ate_error, policy_risk = NA))
  }
  
  # Policy assignments
  policy_estimated <- as.numeric(tau_hat_sub > 0)
  policy_true <- as.numeric(true_ite_sub > 0)
  
  # Policy values
  value_estimated_policy <- mean(
    policy_estimated * Y1 + (1 - policy_estimated) * Y0,
    na.rm = TRUE
  )
  
  value_true_policy <- mean(
    policy_true * Y1 + (1 - policy_true) * Y0,
    na.rm = TRUE
  )
  
  # Policy Risk
  policy_risk <- value_true_policy - value_estimated_policy
  
  c(pehe = pehe, ate_error = ate_error, policy_risk = policy_risk)
}

# Point estimates (on full test data)
n_test <- length(tau_ep)
point_est <- compute_metrics_single(1:n_test)

cat("\nPoint estimates calculated!\n")
cat("PEHE:", round(point_est["pehe"], 4), "\n")
cat("ATE Error:", round(point_est["ate_error"], 4), "\n")
cat("Policy Risk:", round(point_est["policy_risk"], 4), "\n")

# Bootstrap for standard deviations
cat("\nRunning bootstrap for standard deviations (100 iterations)...\n")
n_boot <- 100
boot_results <- matrix(NA, nrow = n_boot, ncol = 3)
colnames(boot_results) <- c("pehe", "ate_error", "policy_risk")

for (i in 1:n_boot) {
  if (i %% 20 == 0) cat("  Bootstrap iteration", i, "/", n_boot, "\n")
  
  boot_idx <- sample(1:n_test, n_test, replace = TRUE)
  
  # Check if bootstrap sample has both treatment groups
  if (length(unique(T_test[boot_idx])) < 2) {
    # Resample if only one treatment group
    boot_idx <- sample(1:n_test, n_test, replace = TRUE)
  }
  
  tryCatch({
    boot_results[i, ] <- compute_metrics_single(boot_idx)
  }, error = function(e) {
    boot_results[i, ] <- NA
  })
}

# Calculate standard deviations
sd_est <- apply(boot_results, 2, sd, na.rm = TRUE)

cat("\nBootstrap complete!\n")

################################################################################
# DISPLAY RESULTS
################################################################################

cat("\n=== EP-LEARNER EVALUATION RESULTS ===\n")
cat("PEHE:        ", round(point_est["pehe"], 4), " (SD:", round(sd_est["pehe"], 4), ")\n")
cat("ATE Error:   ", round(point_est["ate_error"], 4), " (SD:", round(sd_est["ate_error"], 4), ")\n")
cat("Policy Risk: ", round(point_est["policy_risk"], 4), " (SD:", round(sd_est["policy_risk"], 4), ")\n")

# Calculate ATE values for reference
ate_estimated <- mean(tau_ep, na.rm = TRUE)
ate_true <- mean(true_ite_test, na.rm = TRUE)
cat("\nATE Estimated:", round(ate_estimated, 4), "\n")
cat("ATE True:     ", round(ate_true, 4), "\n")

results_df <- data.frame(
  Method = "EP-Learner",
  PEHE = round(point_est["pehe"], 4),
  PEHE_SD = round(sd_est["pehe"], 4),
  ATE_Error = round(point_est["ate_error"], 4),
  ATE_Error_SD = round(sd_est["ate_error"], 4),
  Policy_Risk = round(point_est["policy_risk"], 4),
  Policy_Risk_SD = round(sd_est["policy_risk"], 4),
  ATE_Estimated = round(ate_estimated, 4),
  ATE_True = round(ate_true, 4)
)

print(results_df)
write.csv(results_df, "ep_learner_results.csv", row.names = FALSE)
cat("\n✓ Results saved to: ep_learner_results.csv\n")
