
# EP-Learner Analysis for SHG Participation and Savings
# Calculates 5 metrics with bootstrap standard deviations

library(haven)
library(ranger)
library(dplyr)

################################################################################
# CONFIGURATION
################################################################################

NUM_TREES <- 100
NUM_FOLDS <- 3
BOOTSTRAP_ITERATIONS <- 100

################################################################################
# STEP 1: LOAD AND PREPARE DATA
################################################################################

cat("\n=== Loading Data ===\n")
data <- read_dta("individual_characteristics.dta")
cat("Total observations:", nrow(data), "\n")

# Select and clean data
# IMPORTANT: shgparticipate is coded as 1=Yes, 2=No in Stata
analysis_data <- data %>%
  select(shgparticipate, savings, age, resp_gend, rationcard, workflag) %>%
  filter(
    shgparticipate %in% c(1, 2),
    savings %in% c(1, 2)
  ) %>%
  mutate(
    shgparticipate = ifelse(shgparticipate == 1, 1, 0),
    savings = ifelse(savings == 1, 1, 0)
  ) %>%
  na.omit()

cat("After cleaning:", nrow(analysis_data), "\n")

# Extract components
T <- as.numeric(analysis_data$shgparticipate)
Y <- as.numeric(analysis_data$savings)
X <- analysis_data %>% select(age, resp_gend, rationcard, workflag) %>% as.data.frame()

cat("Treatment: 0=", sum(T==0), ", 1=", sum(T==1), "\n")

# Split train/test
set.seed(123)
n <- nrow(X)
train_idx <- sample(1:n, floor(0.8 * n))
test_idx <- setdiff(1:n, train_idx)

X_train <- X[train_idx, ]
T_train <- T[train_idx]
Y_train <- Y[train_idx]
X_test <- X[test_idx, ]
T_test <- T[test_idx]
Y_test <- Y[test_idx]

cat("Train:", length(train_idx), "| Test:", length(test_idx), "\n")

################################################################################
# STEP 2: TRAIN EP-LEARNER
################################################################################

cat("\n=== Training EP-Learner ===\n")

n_train <- nrow(X_train)
folds <- cut(sample(1:n_train), breaks = NUM_FOLDS, labels = FALSE)
e_hat <- mu0_hat <- mu1_hat <- rep(NA, n_train)

for (k in 1:NUM_FOLDS) {
  cat(sprintf("Fold %d/%d\n", k, NUM_FOLDS))
  
  test_idx_fold <- which(folds == k)
  train_idx_fold <- which(folds != k)
  
  X_tr <- X_train[train_idx_fold, ]
  X_te <- X_train[test_idx_fold, ]
  T_tr <- T_train[train_idx_fold]
  Y_tr <- Y_train[train_idx_fold]
  
  # Propensity score
  ps_model <- ranger(y = as.factor(T_tr), x = X_tr, probability = TRUE, 
                     num.trees = NUM_TREES, min.node.size = 5, verbose = FALSE)
  e_hat[test_idx_fold] <- predict(ps_model, data = X_te)$predictions[, 2]
  
  # Outcome models
  idx0 <- which(T_tr == 0)
  idx1 <- which(T_tr == 1)
  
  mu0_model <- ranger(y = Y_tr[idx0], x = X_tr[idx0, ], num.trees = NUM_TREES, 
                      min.node.size = 5, verbose = FALSE)
  mu0_hat[test_idx_fold] <- predict(mu0_model, data = X_te)$predictions
  
  mu1_model <- ranger(y = Y_tr[idx1], x = X_tr[idx1, ], num.trees = NUM_TREES, 
                      min.node.size = 5, verbose = FALSE)
  mu1_hat[test_idx_fold] <- predict(mu1_model, data = X_te)$predictions
}

e_hat <- pmax(pmin(e_hat, 0.95), 0.05)

################################################################################
# SIEVE ADJUSTMENT STEP
################################################################################

cat("\n=== Sieve Adjustment ===\n")

# Step 1: Create polynomial sieve basis φ(W) = (1, W, W²)
# We'll use all covariates in X_train
sieve_basis <- as.matrix(X_train)
sieve_basis_sq <- sieve_basis^2
colnames(sieve_basis_sq) <- paste0(colnames(sieve_basis), "_sq")

# Combine: φ(W) = (1, W, W²)
phi_W <- cbind(1, sieve_basis, sieve_basis_sq)
colnames(phi_W)[1] <- "intercept"

cat("Sieve basis dimensions:", nrow(phi_W), "x", ncol(phi_W), "\n")

# Step 2: Create treatment-sieve interactions
# A*φ(W) and (1-A)*φ(W)
phi_treated <- T_train * phi_W
phi_control <- (1 - T_train) * phi_W

# Combine all sieve features
sieve_features <- cbind(phi_treated, phi_control)
colnames(sieve_features) <- c(
  paste0("T1_", colnames(phi_W)),
  paste0("T0_", colnames(phi_W))
)

cat("Sieve features created:", ncol(sieve_features), "columns\n")

# Step 3: Weighted regression with offset
# For mu1 (treated outcome model)
cat("Refining mu1 with sieve adjustment...\n")
offset_mu1 <- mu1_hat
weights_mu1 <- T_train / e_hat

# Residual: Y - μ̂(A,W)
residual_mu1 <- Y_train - mu1_hat

# Fit weighted regression: residual ~ sieve_features with weights
# Only use observations where T=1 (treated)
idx_treated <- which(T_train == 1)
if (length(idx_treated) > ncol(sieve_features)) {
  sieve_model_mu1 <- lm(
    residual_mu1[idx_treated] ~ sieve_features[idx_treated, ] - 1,
    weights = weights_mu1[idx_treated]
  )
  beta_mu1 <- coef(sieve_model_mu1)
  beta_mu1[is.na(beta_mu1)] <- 0
} else {
  beta_mu1 <- rep(0, ncol(sieve_features))
}

# For mu0 (control outcome model)
cat("Refining mu0 with sieve adjustment...\n")
offset_mu0 <- mu0_hat
weights_mu0 <- (1 - T_train) / (1 - e_hat)

# Residual: Y - μ̂(A,W)
residual_mu0 <- Y_train - mu0_hat

# Fit weighted regression
idx_control <- which(T_train == 0)
if (length(idx_control) > ncol(sieve_features)) {
  sieve_model_mu0 <- lm(
    residual_mu0[idx_control] ~ sieve_features[idx_control, ] - 1,
    weights = weights_mu0[idx_control]
  )
  beta_mu0 <- coef(sieve_model_mu0)
  beta_mu0[is.na(beta_mu0)] <- 0
} else {
  beta_mu0 <- rep(0, ncol(sieve_features))
}

# Step 4: Construct refined μ*
# μ̂*(a,w) = μ̂(a,w) + β̂ᵀφ(a,w)
mu1_star <- mu1_hat + as.vector(sieve_features %*% beta_mu1)
mu0_star <- mu0_hat + as.vector(sieve_features %*% beta_mu0)

cat("✓ Sieve adjustment complete!\n")
cat("  mu1 adjustment range:", range(mu1_star - mu1_hat), "\n")
cat("  mu0 adjustment range:", range(mu0_star - mu0_hat), "\n")

# Construct pseudo-outcome using refined μ*
phi <- (mu1_star - mu0_star) + 
       (T_train / e_hat) * (Y_train - mu1_star) - 
       ((1 - T_train) / (1 - e_hat)) * (Y_train - mu0_star)

# Train final model
ep_model <- ranger(y = phi, x = X_train, num.trees = NUM_TREES, verbose = FALSE)
cat("✓ Training complete!\n")

################################################################################
# STEP 3: PREDICT AND CALCULATE METRICS
################################################################################

cat("\n=== Predictions and Metrics ===\n")
tau_hat <- predict(ep_model, data = X_test)$predictions

# Helper function for metrics
calculate_metrics <- function(idx) {
  tau_sub <- tau_hat[idx]
  T_sub <- T_test[idx]
  Y_sub <- Y_test[idx]
  
  # 1. ATE
  ate <- mean(tau_sub)
  
  # 2. HTE Std
  hte_std <- sd(tau_sub)
  
  # 3. Outcome MSE
  mu0_est <- mean(Y_sub[T_sub == 0])
  mu1_est <- mean(Y_sub[T_sub == 1])
  Y_pred <- ifelse(T_sub == 1, mu1_est, mu0_est)
  outcome_mse <- mean((Y_sub - Y_pred)^2)
  
  # 4. Propensity Balance
  tau_q <- cut(tau_sub, breaks = 5, labels = FALSE)
  balance <- sapply(1:5, function(q) mean(T_sub[tau_q == q]))
  prop_balance <- sd(balance, na.rm = TRUE)
  
  # 5. Policy Risk Proxy
  policy <- as.numeric(tau_sub > 0)
  treat_all <- mean(Y_sub[T_sub == 1])
  treat_none <- mean(Y_sub[T_sub == 0])
  n_treat <- sum(policy)
  n_control <- sum(1 - policy)
  
  if (n_treat > 0 && n_control > 0) {
    policy_val <- (n_treat * treat_all + n_control * treat_none) / length(policy)
  } else {
    policy_val <- ifelse(n_treat > 0, treat_all, treat_none)
  }
  
  policy_risk <- max(treat_all, treat_none) - policy_val
  
  c(ate = ate, hte_std = hte_std, outcome_mse = outcome_mse, 
    prop_balance = prop_balance, policy_risk = policy_risk)
}

# Point estimates
n_test <- length(tau_hat)
point_est <- calculate_metrics(1:n_test)

# Bootstrap
cat("Bootstrap (", BOOTSTRAP_ITERATIONS, " iterations)...\n")
boot_results <- matrix(NA, nrow = BOOTSTRAP_ITERATIONS, ncol = 5)
colnames(boot_results) <- c("ate", "hte_std", "outcome_mse", "prop_balance", "policy_risk")

for (i in 1:BOOTSTRAP_ITERATIONS) {
  if (i %% 20 == 0) cat("  ", i, "/", BOOTSTRAP_ITERATIONS, "\n")
  boot_idx <- sample(1:n_test, n_test, replace = TRUE)
  tryCatch({
    boot_results[i, ] <- calculate_metrics(boot_idx)
  }, error = function(e) {
    boot_results[i, ] <- NA
  })
}

sd_est <- apply(boot_results, 2, sd, na.rm = TRUE)

################################################################################
# STEP 4: DISPLAY RESULTS
################################################################################

cat("\n=== RESULTS ===\n\n")
cat("1. ATE Prediction:     ", sprintf("%.4f (SD: %.4f)", point_est["ate"], sd_est["ate"]), "\n")
cat("2. Policy Risk Proxy:  ", sprintf("%.4f (SD: %.4f)", point_est["policy_risk"], sd_est["policy_risk"]), "\n")
cat("3. Propensity Balance: ", sprintf("%.4f (SD: %.4f)", point_est["prop_balance"], sd_est["prop_balance"]), "\n")
cat("4. Outcome MSE:        ", sprintf("%.4f (SD: %.4f)", point_est["outcome_mse"], sd_est["outcome_mse"]), "\n")
cat("5. HTE Std:            ", sprintf("%.4f (SD: %.4f)", point_est["hte_std"], sd_est["hte_std"]), "\n")

results_df <- data.frame(
  Metric = c("ATE", "Policy_Risk_Proxy", "Propensity_Balance", "Outcome_MSE", "HTE_Std"),
  Value = round(point_est, 4),
  SD = round(sd_est, 4)
)

cat("\n")
print(results_df)

write.csv(results_df, "shg_ep_learner_results.csv", row.names = FALSE)
cat("\n✓ Results saved to: shg_ep_learner_results.csv\n")
