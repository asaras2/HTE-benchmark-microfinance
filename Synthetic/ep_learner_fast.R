
# FAST EP-Learner with Model Saving
# Optimized for speed with fewer trees and model persistence

library(ranger)

################################################################################
# CONFIGURATION
################################################################################

# Reduce these for faster training (at cost of some accuracy)
NUM_TREES <- 100      # Reduced from 500 (5x faster)
NUM_FOLDS <- 3        # Reduced from 5 (fewer models to train)
MODEL_DIR <- "./ep_learner_models"

# Create directory for saving models
if (!dir.exists(MODEL_DIR)) {
  dir.create(MODEL_DIR, recursive = TRUE)
}

################################################################################
# STEP 1: LOAD DATA
################################################################################

cat("\n=== Loading Training Data ===\n")
train_data <- read.csv("synth_train.csv")

X_train <- train_data[, grep("^x[0-9]", names(train_data))]
T_train <- train_data$t
Y_train <- train_data$yf
Y_cf_train <- if("ycf" %in% names(train_data)) train_data$ycf else rep(NA, nrow(train_data))
true_ite_train <- if("ite" %in% names(train_data)) train_data$ite else if("tau" %in% names(train_data)) train_data$tau else rep(NA, nrow(train_data))

cat("Training data loaded!\n")
cat("N =", nrow(X_train), ", P =", ncol(X_train), "\n")
cat("Treatment distribution:", table(T_train), "\n\n")

################################################################################
# STEP 2: TRAIN EP-LEARNER WITH MODEL SAVING
################################################################################

cat("\n=== Training EP-Learner (Optimized) ===\n")
cat("Configuration: num_trees =", NUM_TREES, ", num_folds =", NUM_FOLDS, "\n\n")

n <- nrow(X_train)
folds <- cut(sample(1:n), breaks = NUM_FOLDS, labels = FALSE)

# Initialize storage
e_hat <- mu0_hat <- mu1_hat <- rep(NA, n)

# Store models for later use
ps_models <- list()
mu0_models <- list()
mu1_models <- list()

cat("Cross-fitting nuisance functions...\n")
for (k in 1:NUM_FOLDS) {
  cat(sprintf("\n--- Fold %d/%d ---\n", k, NUM_FOLDS))
  
  test_idx <- which(folds == k)
  train_idx <- which(folds != k)
  
  X_tr <- X_train[train_idx, ]
  X_te <- X_train[test_idx, ]
  T_tr <- T_train[train_idx]
  Y_tr <- Y_train[train_idx]
  
  # 1. Propensity score
  cat("Training propensity score model...\n")
  ps_model <- ranger(
    y = as.factor(T_tr), 
    x = data.frame(X_tr),
    probability = TRUE, 
    num.trees = NUM_TREES,
    verbose = FALSE  # Suppress progress for cleaner output
  )
  e_hat[test_idx] <- predict(ps_model, data = data.frame(X_te))$predictions[, 2]
  ps_models[[k]] <- ps_model
  
  # 2. Outcome model for control (T=0)
  cat("Training mu0 model...\n")
  idx0 <- which(T_tr == 0)
  mu0_model <- ranger(
    y = Y_tr[idx0], 
    x = data.frame(X_tr[idx0, ]),
    num.trees = NUM_TREES,
    verbose = FALSE
  )
  mu0_hat[test_idx] <- predict(mu0_model, data = data.frame(X_te))$predictions
  mu0_models[[k]] <- mu0_model
  
  # 3. Outcome model for treated (T=1)
  cat("Training mu1 model...\n")
  idx1 <- which(T_tr == 1)
  mu1_model <- ranger(
    y = Y_tr[idx1], 
    x = data.frame(X_tr[idx1, ]),
    num.trees = NUM_TREES,
    verbose = FALSE
  )
  mu1_hat[test_idx] <- predict(mu1_model, data = data.frame(X_te))$predictions
  mu1_models[[k]] <- mu1_model
  
  cat(sprintf("Fold %d complete!\n", k))
}

# Clip propensity scores
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

# Construct efficient pseudo-outcome using refined μ*
cat("\nConstructing pseudo-outcomes...\n")
phi <- (mu1_star - mu0_star) + 
       (T_train / e_hat) * (Y_train - mu1_star) - 
       ((1 - T_train) / (1 - e_hat)) * (Y_train - mu0_star)

cat("Pseudo-outcome summary:\n")
print(summary(phi))

# Train final CATE model
cat("\nTraining final CATE model...\n")
ep_model <- ranger(
  y = phi, 
  x = data.frame(X_train),
  num.trees = NUM_TREES,
  verbose = FALSE
)

cat("\n✓ EP-Learner trained successfully!\n")

################################################################################
# STEP 3: SAVE ALL MODELS
################################################################################

cat("\n=== Saving Models ===\n")

# Save nuisance models
saveRDS(list(
  ps_models = ps_models,
  mu0_models = mu0_models,
  mu1_models = mu1_models,
  folds = folds,
  e_hat = e_hat,
  mu0_hat = mu0_hat,
  mu1_hat = mu1_hat,
  phi = phi
), file.path(MODEL_DIR, "nuisance_models.rds"))

# Save final CATE model
saveRDS(ep_model, file.path(MODEL_DIR, "ep_cate_model.rds"))

# Save configuration
config <- list(
  num_trees = NUM_TREES,
  num_folds = NUM_FOLDS,
  n_train = n,
  p_features = ncol(X_train),
  training_date = Sys.time()
)
saveRDS(config, file.path(MODEL_DIR, "config.rds"))

cat("✓ Models saved to:", MODEL_DIR, "\n")
cat("  - nuisance_models.rds (propensity & outcome models)\n")
cat("  - ep_cate_model.rds (final CATE model)\n")
cat("  - config.rds (training configuration)\n")

################################################################################
# STEP 4: LOAD TEST DATA
################################################################################

cat("\n=== Loading Test Data ===\n")
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
cat("N =", nrow(X_test), ", P =", ncol(X_test), "\n")
cat("Y_cf NAs:", sum(is.na(Y_cf_test)), "\n")
cat("true_ite NAs:", sum(is.na(true_ite_test)), "\n\n")

################################################################################
# STEP 5: MAKE PREDICTIONS
################################################################################

cat("\n=== Making Predictions ===\n")
tau_ep <- predict(ep_model, data = data.frame(X_test))$predictions

cat("CATE predictions summary:\n")
print(summary(tau_ep))

################################################################################
# STEP 6: CALCULATE METRICS
################################################################################

cat("\n=== Calculating Metrics ===\n")

# 1. PEHE
pehe <- sqrt(mean((tau_ep - true_ite_test)^2, na.rm = TRUE))

# 2. ATE Error
ate_estimated <- mean(tau_ep, na.rm = TRUE)
ate_true <- mean(true_ite_test, na.rm = TRUE)
ate_error <- abs(ate_estimated - ate_true)

# 3. Policy Risk
n_test <- length(T_test)
Y1 <- rep(NA, n_test)
Y1[T_test == 1] <- Y_test[T_test == 1]
Y1[T_test == 0] <- Y_cf_test[T_test == 0]

Y0 <- rep(NA, n_test)
Y0[T_test == 0] <- Y_test[T_test == 0]
Y0[T_test == 1] <- Y_cf_test[T_test == 1]

policy_estimated <- as.numeric(tau_ep > 0)
policy_true <- as.numeric(true_ite_test > 0)

value_estimated_policy <- mean(policy_estimated * Y1 + (1 - policy_estimated) * Y0, na.rm = TRUE)
value_true_policy <- mean(policy_true * Y1 + (1 - policy_true) * Y0, na.rm = TRUE)
policy_risk <- value_true_policy - value_estimated_policy

################################################################################
# STEP 7: DISPLAY AND SAVE RESULTS
################################################################################

cat("\n=== EP-LEARNER EVALUATION RESULTS ===\n")
cat("PEHE:        ", round(pehe, 4), "\n")
cat("ATE Error:   ", round(ate_error, 4), "\n")
cat("Policy Risk: ", round(policy_risk, 4), "\n")
cat("\nATE Estimated:", round(ate_estimated, 4), "\n")
cat("ATE True:     ", round(ate_true, 4), "\n")

results_df <- data.frame(
  Method = "EP-Learner",
  PEHE = round(pehe, 4),
  ATE_Error = round(ate_error, 4),
  Policy_Risk = round(policy_risk, 4),
  ATE_Estimated = round(ate_estimated, 4),
  ATE_True = round(ate_true, 4),
  Num_Trees = NUM_TREES,
  Num_Folds = NUM_FOLDS
)

print(results_df)

write.csv(results_df, "ep_learner_results.csv", row.names = FALSE)
cat("\n✓ Results saved to: ep_learner_results.csv\n")

predictions_df <- data.frame(
  tau_estimated = tau_ep,
  tau_true = true_ite_test,
  treatment = T_test,
  outcome = Y_test
)
write.csv(predictions_df, "ep_learner_predictions.csv", row.names = FALSE)
cat("✓ Predictions saved to: ep_learner_predictions.csv\n")

cat("\n=== TRAINING COMPLETE ===\n")
cat("All models and results saved successfully!\n")
cat("To reload models later, use:\n")
cat("  ep_model <- readRDS('", file.path(MODEL_DIR, "ep_cate_model.rds"), "')\n", sep="")
