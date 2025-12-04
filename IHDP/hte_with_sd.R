
library(ranger)
library(hte3)

###############################################
# STEP 1 : LOAD THE DATASET
###############################################
data <- read.csv("idhp_data_train.csv")

# Separate the components needed for HTE estimation
# X: covariates (all columns starting with 'x')
X <- data[, grep("^x", names(data))]

# T: treatment assignment
T <- data$t

# Y: factual outcome (observed outcome)
Y <- data$yf

# Store counterfactual and ITE for validation later
Y_cf <- data$ycf
true_ite <- data$ite

# Create a clean dataset for hte3
hte_data <- list(
  X = X,           # Covariates
  T = T,           # Treatment (0 or 1)
  Y = Y,           # Observed outcome
  Y_cf = Y_cf,     # Counterfactual outcome (for validation)
  true_ite = true_ite  # True ITE (for validation)
)

# Test: Verify data structure
cat("Dataset loaded successfully!\n")
cat("Number of observations:", nrow(X), "\n")
cat("Number of covariates:", ncol(X), "\n")
cat("Treatment distribution:", table(T), "\n")
cat("Outcome summary:", summary(Y), "\n")

# # Quick verification test
# stopifnot(nrow(X) == length(T))
# stopifnot(length(T) == length(Y))
# stopifnot(all(T %in% c(0, 1)))
# cat("All data integrity checks passed!\n")

###############################################
# STEP 2: HELPER FUNCTIONS FOR BASELINES WITH ATE/ATT
###############################################
# T-Learner (for comparison)
t_learner_baseline <- function(X, W, Y, Y_cf = NULL) {
  cat("Running T-Learner baseline...\n")
  
  # Fit separate models for control and treated
  idx0 <- which(W == 0)
  idx1 <- which(W == 1)
  
  model0 <- ranger(y = Y[idx0], x = data.frame(X[idx0, ]), num.trees = 500)
  model1 <- ranger(y = Y[idx1], x = data.frame(X[idx1, ]), num.trees = 500)
  
  # Predict on full data
  mu0 <- predict(model0, data = data.frame(X))$predictions
  mu1 <- predict(model1, data = data.frame(X))$predictions
  
  tau <- mu1 - mu0
  
  # Calculate ATE
  ate <- mean(tau)
  
  # Calculate ATT using counterfactual outcomes if available
  if (!is.null(Y_cf)) {
    # For treated units: Y - Y_cf (factual - counterfactual)
    att <- mean(Y[W == 1] - Y_cf[W == 1])
  } else {
    # Fallback: use estimated CATE on treated
    att <- mean(tau[W == 1])
  }
  
  list(tau = tau, mu0 = mu0, mu1 = mu1, ate = ate, att = att)
}

# DR-Learner (for comparison)
dr_learner_baseline <- function(X, W, Y, Y_cf = NULL) {
  cat("Running DR-Learner baseline...\n")
  
  n <- nrow(X)
  folds <- cut(sample(1:n), breaks = 5, labels = FALSE)
  
  e_hat <- mu0_hat <- mu1_hat <- rep(NA, n)
  
  # Cross-fit nuisances
  for (k in 1:5) {
    test_idx <- which(folds == k)
    train_idx <- which(folds != k)
    
    X_train <- X[train_idx, ]
    X_test <- X[test_idx, ]
    W_train <- W[train_idx]
    Y_train <- Y[train_idx]
    
    # Propensity score
    ps_model <- ranger(y = as.factor(W_train), x = data.frame(X_train), 
                       probability = TRUE, num.trees = 500)
    e_hat[test_idx] <- predict(ps_model, data = data.frame(X_test))$predictions[, 2]
    
    # Outcome models
    idx0 <- which(W_train == 0)
    idx1 <- which(W_train == 1)
    
    mu0_model <- ranger(y = Y_train[idx0], x = data.frame(X_train[idx0, ]), num.trees = 500)
    mu0_hat[test_idx] <- predict(mu0_model, data = data.frame(X_test))$predictions
    
    mu1_model <- ranger(y = Y_train[idx1], x = data.frame(X_train[idx1, ]), num.trees = 500)
    mu1_hat[test_idx] <- predict(mu1_model, data = data.frame(X_test))$predictions
  }
  
  # Clip propensity scores
  e_hat <- pmax(pmin(e_hat, 0.95), 0.05)
  
  # AIPW pseudo-outcome
  phi <- (W * (Y - mu1_hat) / e_hat + mu1_hat) - 
    ((1 - W) * (Y - mu0_hat) / (1 - e_hat) + mu0_hat)
  
  # Regress on X
  dr_model <- ranger(y = phi, x = data.frame(X), num.trees = 500)
  tau <- predict(dr_model, data = data.frame(X))$predictions
  
  # Calculate ATE
  ate <- mean(tau)
  
  # Calculate ATT using counterfactual outcomes if available
  if (!is.null(Y_cf)) {
    # For treated units: Y - Y_cf (factual - counterfactual)
    att <- mean(Y[W == 1] - Y_cf[W == 1])
  } else {
    # Fallback: use estimated CATE on treated
    att <- mean(tau[W == 1])
  }
  
  list(tau = tau, phi = phi, ate = ate, att = att)
}

# # Test: Run both methods on sample data with counterfactuals
# cat("\n=== Testing Baseline Methods ===\n")
# set.seed(123)
# n_test <- 100
# X_test <- matrix(rnorm(n_test * 5), ncol = 5)
# W_test <- rbinom(n_test, 1, 0.5)
# true_tau <- rowSums(X_test[, 1:2]) + 2
# Y_test <- rowSums(X_test) + W_test * true_tau + rnorm(n_test, 0, 0.5)
# Y_cf_test <- rowSums(X_test) + (1 - W_test) * true_tau + rnorm(n_test, 0, 0.5)
# 
# t_result <- t_learner_baseline(X_test, W_test, Y_test, Y_cf_test)
# cat("T-Learner ATE:", round(t_result$ate, 3), "| ATT:", round(t_result$att, 3), "\n")
# 
# dr_result <- dr_learner_baseline(X_test, W_test, Y_test, Y_cf_test)
# cat("DR-Learner ATE:", round(dr_result$ate, 3), "| ATT:", round(dr_result$att, 3), "\n")
# 
# # True ATT for comparison
# true_att <- mean(true_tau[W_test == 1])
# cat("True ATT:", round(true_att, 3), "\n")
# cat("Baseline methods with proper ATE/ATT work \n")


################################################################################
# STEP 3: CALCULATE ATE AND ATT USING HTE3
################################################################################

# Function to calculate ATE and ATT using hte3 approach
calculate_ate_att_hte3 <- function(X, T, Y, Y_cf = NULL) {
  cat("Calculating ATE and ATT using hte3 approach...\n")
  
  # Combine into a dataframe
  data_df <- as.data.frame(X)
  data_df$Y <- Y
  data_df$T <- T
  
  # Manual cross-fitted estimation (hte3 style)
  n <- nrow(data_df)
  folds <- cut(sample(1:n), breaks = 5, labels = FALSE)
  
  pi_hat <- mu0_hat <- mu1_hat <- rep(NA, n)
  
  cat("Cross-fitting nuisance functions...\n")
  for (k in 1:5) {
    test_idx <- which(folds == k)
    train_idx <- which(folds != k)
    
    X_train <- data_df[train_idx, colnames(X), drop = FALSE]
    X_test <- data_df[test_idx, colnames(X), drop = FALSE]
    T_train <- data_df$T[train_idx]
    Y_train <- data_df$Y[train_idx]
    
    # Estimate propensity score
    pi_model <- ranger(y = as.factor(T_train), x = X_train, 
                       probability = TRUE, num.trees = 500)
    pi_hat[test_idx] <- predict(pi_model, data = X_test)$predictions[, 2]
    
    # Estimate mu0 (outcome under control)
    idx0 <- which(T_train == 0)
    mu0_model <- ranger(y = Y_train[idx0], x = X_train[idx0, , drop = FALSE], 
                        num.trees = 500)
    mu0_hat[test_idx] <- predict(mu0_model, data = X_test)$predictions
    
    # Estimate mu1 (outcome under treatment)
    idx1 <- which(T_train == 1)
    mu1_model <- ranger(y = Y_train[idx1], x = X_train[idx1, , drop = FALSE], 
                        num.trees = 500)
    mu1_hat[test_idx] <- predict(mu1_model, data = X_test)$predictions
  }
  
  # Calculate CATE estimates
  tau_hat <- mu1_hat - mu0_hat
  
  # Calculate ATE (average over all units)
  ate_estimate <- mean(tau_hat)
  
  # Calculate ATT using counterfactual if available (oracle ATT)
  if (!is.null(Y_cf)) {
    # Oracle ATT: For treated units, Y - Y_cf
    att_oracle <- mean(Y[T == 1] - Y_cf[T == 1])
  } else {
    att_oracle <- NULL
  }
  
  # Calculate ATT using estimated CATE (on treated units)
  att_estimate <- mean(tau_hat[T == 1])
  
  cat("ATE estimate:", round(ate_estimate, 4), "\n")
  cat("ATT estimate (CATE-based):", round(att_estimate, 4), "\n")
  if (!is.null(att_oracle)) {
    cat("ATT oracle (true):", round(att_oracle, 4), "\n")
  }
  
  list(
    ate = ate_estimate,
    att = att_estimate,
    att_oracle = att_oracle,
    tau_hat = tau_hat,
    mu0_hat = mu0_hat,
    mu1_hat = mu1_hat,
    pi_hat = pi_hat
  )
}

# # Test: Run on sample data
# cat("\n=== Testing HTE3-style ATE/ATT Calculation ===\n")
# set.seed(456)
# n_test <- 200
# X_test <- matrix(rnorm(n_test * 10), ncol = 10)
# colnames(X_test) <- paste0("x", 1:10)
# T_test <- rbinom(n_test, 1, 0.5)
# true_tau <- 2 + X_test[, 1] + 0.5 * X_test[, 2]
# Y_test <- 1 + rowMeans(X_test) + T_test * true_tau + rnorm(n_test, 0, 0.5)
# Y_cf_test <- 1 + rowMeans(X_test) + (1 - T_test) * true_tau + rnorm(n_test, 0, 0.5)
# 
# hte3_results <- calculate_ate_att_hte3(X_test, T_test, Y_test, Y_cf_test)
# 
# # Compare with true values
# true_ate <- mean(true_tau)
# true_att <- mean(true_tau[T_test == 1])
# cat("\nTrue ATE:", round(true_ate, 4), "\n")
# cat("True ATT:", round(true_att, 4), "\n")
# cat("ATE Error:", round(abs(hte3_results$ate - true_ate), 4), "\n")
# cat("ATT Error:", round(abs(hte3_results$att_oracle - true_att), 4), "\n")
# 
# cat("✓ HTE3-style ATE/ATT calculation ready!\n")

################################################################################
# STEP 4: CALCULATE EVALUATION METRICS
################################################################################

# Function to calculate PEHE, ATE risk, and Policy risk
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
    
    # Validate inputs
    if (length(T_sub) == 0 || length(Y_sub) == 0 || length(Y_cf_sub) == 0) {
      return(c(pehe = NA, ate_risk = NA, policy_risk = NA, att_risk = NA))
    }
    
    # Ensure all vectors have the same length
    n_sub <- length(T_sub)
    if (length(Y_sub) != n_sub || length(Y_cf_sub) != n_sub || length(tau_hat_sub) != n_sub || length(true_ite_sub) != n_sub) {
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



# # Test: Calculate metrics on sample data
# cat("\n=== Testing Metrics Calculation ===\n")
# set.seed(789)
# n_test <- 300
# X_test <- matrix(rnorm(n_test * 5), ncol = 5)
# T_test <- rbinom(n_test, 1, 0.5)
# true_tau_test <- 1 + 2 * X_test[, 1] - X_test[, 2]
# Y_test <- 5 + rowMeans(X_test) + T_test * true_tau_test + rnorm(n_test, 0, 0.5)
# Y_cf_test <- 5 + rowMeans(X_test) + (1 - T_test) * true_tau_test + rnorm(n_test, 0, 0.5)
# 
# # Create slightly noisy estimates
# tau_estimated <- true_tau_test + rnorm(n_test, 0, 1.5)
# 
# # Calculate all metrics
# metrics <- calculate_metrics(
#   tau_hat = tau_estimated,
#   true_ite = true_tau_test,
#   T = T_test,
#   Y = Y_test,
#   Y_cf = Y_cf_test,
#   method_name = "Test Method"
# )
# 
# cat("\n✓ All evaluation metrics calculated successfully!\n")
# cat("Lower values indicate better performance for all metrics.\n")

################################################################################
# STEP 5: TRAIN MODELS, SAVE, LOAD TEST DATA, PREDICT AND EVALUATE
################################################################################

# Function to train models and save them
train_and_save_models <- function(X_train, T_train, Y_train, save_dir = "./models") {
  cat("\n=== Training Models on Training Data ===\n")
  
  # Create directory for saving models
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  # Get indices for control and treated groups
  idx0 <- which(T_train == 0)
  idx1 <- which(T_train == 1)
  
  cat("Training set size:", nrow(X_train), "\n")
  cat("Control group size:", length(idx0), "\n")
  cat("Treated group size:", length(idx1), "\n")
  
  # 1. Train T-Learner models
  cat("\n--- Training T-Learner ---\n")
  t_model0 <- ranger(y = Y_train[idx0], x = data.frame(X_train[idx0, ]), num.trees = 500)
  t_model1 <- ranger(y = Y_train[idx1], x = data.frame(X_train[idx1, ]), num.trees = 500)
  saveRDS(t_model0, file.path(save_dir, "t_learner_model0.rds"))
  saveRDS(t_model1, file.path(save_dir, "t_learner_model1.rds"))
  cat("T-Learner models saved.\n")
  
  # 2. Train DR-Learner
  cat("\n--- Training DR-Learner ---\n")
  dr_result <- dr_learner_baseline(X_train, T_train, Y_train, Y_cf = NULL)
  dr_model <- ranger(y = dr_result$phi, x = data.frame(X_train), num.trees = 500)
  saveRDS(dr_model, file.path(save_dir, "dr_learner_model.rds"))
  cat("DR-Learner model saved.\n")
  
  # 3. Train HTE3-style models
  cat("\n--- Training HTE3-style models ---\n")
  hte3_model0 <- ranger(y = Y_train[idx0], x = data.frame(X_train[idx0, ]), num.trees = 500)
  hte3_model1 <- ranger(y = Y_train[idx1], x = data.frame(X_train[idx1, ]), num.trees = 500)
  saveRDS(hte3_model0, file.path(save_dir, "hte3_model0.rds"))
  saveRDS(hte3_model1, file.path(save_dir, "hte3_model1.rds"))
  cat("HTE3-style models saved.\n")
  
  cat("\n✓ All models trained and saved to:", save_dir, "\n")
}

# Function to load test data, predict and evaluate
load_test_predict_evaluate <- function(test_file_path, save_dir = "./models") {
  cat("\n=== Loading Test Data ===\n")
  
  # Load test data from CSV
  test_data <- read.csv(test_file_path)
  
  # Extract components
  X_test <- test_data[, grep("^x", names(test_data))]
  T_test <- test_data$t
  Y_test <- test_data$yf
  Y_cf_test <- test_data$ycf
  true_ite_test <- test_data$ite
  
  # Impute Y_cf if missing but true_ite is available
  if (any(is.na(Y_cf_test)) && !any(is.na(true_ite_test))) {
    cat("Note: Imputing missing counterfactuals (Y_cf) using observed Y and True ITE.\n")
    # If T=1, Y_cf = Y(0) = Y(1) - ITE = Y - ITE
    # If T=0, Y_cf = Y(1) = Y(0) + ITE = Y + ITE
    # Formula: Y + (1 - 2*T) * ITE
    # T=1: Y + (1-2)*ITE = Y - ITE
    # T=0: Y + (1-0)*ITE = Y + ITE
    Y_cf_test <- Y_test + (1 - 2 * T_test) * true_ite_test
  }
  
  cat("Test set loaded successfully!\n")
  cat("Test size:", nrow(X_test), "\n")
  cat("Number of covariates:", ncol(X_test), "\n")
  cat("Treatment distribution:", table(T_test), "\n\n")
  
  # Load saved models and make predictions
  results <- list()
  
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
  
  list(results = results, summary = summary_df)
}

# Execute: Train on training data from Step 1
train_and_save_models(X, T, Y, save_dir = "./trained_models")

# Execute: Load test data, predict and evaluate
# Using the actual test file
final_results <- load_test_predict_evaluate("idhp_data_test.csv", save_dir = "./trained_models")

cat("\n✓ Training, prediction, and evaluation complete!\n")
cat("✓ Best method by PEHE:", 
    final_results$summary$Method[which.min(final_results$summary$PEHE)], "\n")
cat("✓ Best method by Policy Risk:", 
    final_results$summary$Method[which.min(final_results$summary$Policy_Risk)], "\n")