
# EP-Learner Analysis for SHG Participation and Savings
# Simplified version with better error handling

library(haven)      # For reading .dta files
library(ranger)     # For random forests
library(dplyr)      # For data manipulation

################################################################################
# CONFIGURATION
################################################################################

NUM_TREES <- 100    # Reduced for faster training
NUM_FOLDS <- 3      # Reduced to avoid small sample issues
BOOTSTRAP_ITERATIONS <- 50  # Reduced for speed

################################################################################
# STEP 1: LOAD AND PREPARE DATA
################################################################################

cat("\n=== Loading Data ===\n")

# Load Stata file
data <- read_dta("individual_characteristics.dta")

cat("Data loaded successfully!\n")
cat("Total observations:", nrow(data), "\n")

# Select and clean data
# IMPORTANT: shgparticipate is coded as 1=Yes, 2=No in Stata
# We need to recode: 1 (Yes) -> 1 (treated), 2 (No) -> 0 (control)
analysis_data <- data %>%
  select(
    shgparticipate,
    savings,
    age,
    resp_gend,
    rationcard,
    workflag
  ) %>%
  filter(
    shgparticipate %in% c(1, 2),  # Keep only valid responses
    savings %in% c(1, 2)           # Keep only valid responses
  ) %>%
  mutate(
    # Recode treatment: 1=Yes (participated) -> 1, 2=No -> 0
    shgparticipate = ifelse(shgparticipate == 1, 1, 0),
    # Recode outcome: 1=Yes (has savings) -> 1, 2=No -> 0
    savings = ifelse(savings == 1, 1, 0)
  ) %>%
  na.omit()  # Remove rows with missing values

cat("After cleaning and recoding:", nrow(analysis_data), "\n")

# Extract components
T <- as.numeric(analysis_data$shgparticipate)
Y <- as.numeric(analysis_data$savings)
X <- analysis_data %>% 
  select(age, resp_gend, rationcard, workflag) %>%
  as.data.frame()

cat("\nData summary:\n")
cat("Treatment (SHG participation): 0=", sum(T==0), ", 1=", sum(T==1), "\n")
cat("Outcome (Has savings): 0=", sum(Y==0), ", 1=", sum(Y==1), "\n")
cat("Number of covariates:", ncol(X), "\n")

# Check if we have enough data in each group (lowered threshold)
if (sum(T==0) < 10 || sum(T==1) < 10) {
  cat("\nERROR: Insufficient data!\n")
  cat("Control group (T=0):", sum(T==0), "observations\n")
  cat("Treated group (T=1):", sum(T==1), "observations\n")
  stop("Need at least 10 observations in each treatment group for EP-Learner.")
}

if (sum(T==0) < 30 || sum(T==1) < 30) {
  cat("\nWARNING: Small sample size detected. Results may be unstable.\n")
  cat("Consider using a simpler method or collecting more data.\n")
}

# Split into train/test (80/20)
set.seed(123)
n <- nrow(X)
train_idx <- sample(1:n, size = floor(0.8 * n))
test_idx <- setdiff(1:n, train_idx)

X_train <- X[train_idx, ]
T_train <- T[train_idx]
Y_train <- Y[train_idx]

X_test <- X[test_idx, ]
T_test <- T[test_idx]
Y_test <- Y[test_idx]

cat("\nTrain set:", length(train_idx), "observations\n")
cat("  Control:", sum(T_train==0), ", Treated:", sum(T_train==1), "\n")
cat("Test set:", length(test_idx), "observations\n")
cat("  Control:", sum(T_test==0), ", Treated:", sum(T_test==1), "\n")

################################################################################
# STEP 2: TRAIN EP-LEARNER WITH ERROR HANDLING
################################################################################

cat("\n=== Training EP-Learner ===\n")

n_train <- nrow(X_train)
folds <- cut(sample(1:n_train), breaks = NUM_FOLDS, labels = FALSE)

# Initialize storage
e_hat <- mu0_hat <- mu1_hat <- rep(NA, n_train)

cat("Cross-fitting nuisance functions (", NUM_FOLDS, " folds)...\n")

for (k in 1:NUM_FOLDS) {
  cat(sprintf("  Fold %d/%d\n", k, NUM_FOLDS))
  
  test_idx_fold <- which(folds == k)
  train_idx_fold <- which(folds != k)
  
  X_tr <- X_train[train_idx_fold, ]
  X_te <- X_train[test_idx_fold, ]
  T_tr <- T_train[train_idx_fold]
  Y_tr <- Y_train[train_idx_fold]
  
  # Check sample sizes
  n_control <- sum(T_tr == 0)
  n_treated <- sum(T_tr == 1)
  cat(sprintf("    Training fold: %d control, %d treated\n", n_control, n_treated))
  
  if (n_control < 10 || n_treated < 10) {
    cat("    Warning: Small sample size in this fold, using simpler model\n")
  }
  
  # 1. Propensity score
  tryCatch({
    ps_model <- ranger(
      y = as.factor(T_tr), 
      x = X_tr,
      probability = TRUE, 
      num.trees = NUM_TREES,
      min.node.size = 5,  # Prevent overfitting
      verbose = FALSE
    )
    e_hat[test_idx_fold] <- predict(ps_model, data = X_te)$predictions[, 2]
  }, error = function(e) {
    cat("    Error in propensity model, using marginal probability\n")
    e_hat[test_idx_fold] <<- mean(T_tr)
  })
  
  # 2. Outcome model for control
  tryCatch({
    idx0 <- which(T_tr == 0)
    if (length(idx0) >= 10) {
      mu0_model <- ranger(
        y = Y_tr[idx0], 
        x = X_tr[idx0, ],
        num.trees = NUM_TREES,
        min.node.size = 5,
        verbose = FALSE
      )
      mu0_hat[test_idx_fold] <- predict(mu0_model, data = X_te)$predictions
    } else {
      mu0_hat[test_idx_fold] <- mean(Y_tr[idx0])
    }
  }, error = function(e) {
    cat("    Error in mu0 model, using mean\n")
    mu0_hat[test_idx_fold] <<- mean(Y_tr[T_tr == 0])
  })
  
  # 3. Outcome model for treated
  tryCatch({
    idx1 <- which(T_tr == 1)
    if (length(idx1) >= 10) {
      mu1_model <- ranger(
        y = Y_tr[idx1], 
        x = X_tr[idx1, ],
        num.trees = NUM_TREES,
        min.node.size = 5,
        verbose = FALSE
      )
      mu1_hat[test_idx_fold] <- predict(mu1_model, data = X_te)$predictions
    } else {
      mu1_hat[test_idx_fold] <- mean(Y_tr[idx1])
    }
  }, error = function(e) {
    cat("    Error in mu1 model, using mean\n")
    mu1_hat[test_idx_fold] <<- mean(Y_tr[T_tr == 1])
  })
}

# Clip propensity scores
e_hat <- pmax(pmin(e_hat, 0.95), 0.05)

# Construct efficient pseudo-outcome
phi <- (mu1_hat - mu0_hat) + 
  (T_train / e_hat) * (Y_train - mu1_hat) - 
  ((1 - T_train) / (1 - e_hat)) * (Y_train - mu0_hat)

cat("\nPseudo-outcome constructed!\n")
cat("Pseudo-outcome summary:\n")
print(summary(phi))

# Train final CATE model
cat("\nTraining final CATE model...\n")
ep_model <- ranger(
  y = phi, 
  x = X_train,
  num.trees = NUM_TREES,
  verbose = FALSE
)

cat("✓ EP-Learner trained successfully!\n")

################################################################################
# STEP 3: PREDICT AND CALCULATE METRICS
################################################################################

cat("\n=== Making Predictions on Test Set ===\n")
tau_hat <- predict(ep_model, data = X_test)$predictions

cat("\n=== RESULTS ===\n")

# 1. ATE Prediction
ate_estimate <- mean(tau_hat)
ate_boot <- replicate(BOOTSTRAP_ITERATIONS, mean(sample(tau_hat, replace=TRUE)))
ate_se <- sd(ate_boot)

cat("\n1. ATE Prediction:", round(ate_estimate, 4), "(SE:", round(ate_se, 4), ")\n")

# 2. HTE Std
hte_std <- sd(tau_hat)
hte_std_boot <- replicate(BOOTSTRAP_ITERATIONS, sd(sample(tau_hat, replace=TRUE)))
hte_std_se <- sd(hte_std_boot)

cat("2. HTE Std:", round(hte_std, 4), "(SE:", round(hte_std_se, 4), ")\n")

# 3. Outcome MSE (simple version)
# Predict outcomes
mu0_simple <- mean(Y_test[T_test == 0])
mu1_simple <- mean(Y_test[T_test == 1])
Y_pred <- ifelse(T_test == 1, mu1_simple, mu0_simple)
outcome_mse <- mean((Y_test - Y_pred)^2)

cat("3. Outcome MSE:", round(outcome_mse, 4), "\n")

# 4. Propensity Balance
tau_quantiles <- cut(tau_hat, breaks = 5, labels = FALSE)
balance_test <- sapply(1:5, function(q) mean(T_test[tau_quantiles == q]))
propensity_balance <- sd(balance_test, na.rm = TRUE)

cat("4. Propensity Balance:", round(propensity_balance, 4), "\n")

# 5. Policy Risk Proxy
policy_estimated <- as.numeric(tau_hat > 0)
treat_all_outcome <- mean(Y_test[T_test == 1])
treat_none_outcome <- mean(Y_test[T_test == 0])
policy_outcome <- mean(ifelse(policy_estimated == 1, 
                              Y_test[T_test == 1][1:sum(policy_estimated)],
                              Y_test[T_test == 0][1:sum(1-policy_estimated)]))

cat("5. Policy Risk Proxy: (comparing policies)\n")
cat("   Treat all outcome:", round(treat_all_outcome, 4), "\n")
cat("   Treat none outcome:", round(treat_none_outcome, 4), "\n")

# Summary
results_df <- data.frame(
  Metric = c("ATE", "ATE_SE", "HTE_Std", "HTE_Std_SE", "Outcome_MSE", "Propensity_Balance"),
  Value = round(c(ate_estimate, ate_se, hte_std, hte_std_se, outcome_mse, propensity_balance), 4)
)

print(results_df)
write.csv(results_df, "shg_ep_learner_results.csv", row.names = FALSE)
cat("\n✓ Results saved!\n")
