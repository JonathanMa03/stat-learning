# ==============================================================================
# Model Averaging
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Model uncertainty
#   - Prediction averaging
#   - Equal-weight averaging
#   - Validation-weighted averaging
#   - Cross-validation-weighted averaging
#   - BIC-based model weights
#   - Ensemble prediction
#   - Bias-variance tradeoff
#
# Core idea:
#
#   Suppose we have M candidate models with predictions:
#
#       f_1(x), ..., f_M(x)
#
#   Model averaging forms:
#
#       f_avg(x) = sum_m w_m f_m(x)
#
#   where:
#
#       w_m >= 0
#       sum_m w_m = 1
#
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Nonlinear Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 300


x1 <- runif(
  n,
  min = -3,
  max = 3
)


x2 <- runif(
  n,
  min = -3,
  max = 3
)


true_function <- function(
    x1,
    x2
) {
  
  2 +
    2 * sin(x1) +
    0.8 * x2^2 +
    0.5 * x1 * x2
}


sigma <- 1.5


y_true <- true_function(
  x1,
  x2
)


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
  )


data <- data.frame(
  x1 = x1,
  x2 = x2,
  y = y
)


# ------------------------------------------------------------------------------
# 2. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data,
  main = "Model Averaging Data"
)


# ------------------------------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)


train_index <- sample(
  seq_len(n),
  size = floor(
    0.7 * n
  )
)


test_index <- setdiff(
  seq_len(n),
  train_index
)


train_data <- data[
  train_index,
]


test_data <- data[
  test_index,
]


# ------------------------------------------------------------------------------
# 4. Candidate Models
# ------------------------------------------------------------------------------

# Intentionally use models with different structures.
#
# Model 1:
# simple linear model
#
# Model 2:
# quadratic terms
#
# Model 3:
# interaction
#
# Model 4:
# cubic polynomial terms
#
# Model 5:
# sine-informed model
#
# In practice, candidate models may arise from:
#
#   different subsets
#   different tuning parameters
#   different algorithms
#   different prior assumptions
#   different feature representations


model_1 <- lm(
  y ~ x1 + x2,
  data = train_data
)


model_2 <- lm(
  y ~
    x1 +
    I(x1^2) +
    x2 +
    I(x2^2),
  data = train_data
)


model_3 <- lm(
  y ~
    x1 +
    I(x1^2) +
    x2 +
    I(x2^2) +
    x1:x2,
  data = train_data
)


model_4 <- lm(
  y ~
    x1 +
    I(x1^2) +
    I(x1^3) +
    x2 +
    I(x2^2) +
    I(x2^3) +
    x1:x2,
  data = train_data
)


model_5 <- lm(
  y ~
    sin(x1) +
    x2 +
    I(x2^2) +
    x1:x2,
  data = train_data
)


models <- list(
  Linear = model_1,
  Quadratic = model_2,
  Quadratic_Interaction = model_3,
  Cubic = model_4,
  Sine_Structure = model_5
)


model_names <- names(
  models
)


M <- length(
  models
)


# ------------------------------------------------------------------------------
# 5. Test Predictions from Each Model
# ------------------------------------------------------------------------------

test_predictions <- matrix(
  NA,
  nrow = nrow(
    test_data
  ),
  ncol = M
)


colnames(
  test_predictions
) <- model_names


for (
  m in seq_len(M)
) {
  
  test_predictions[
    ,
    m
  ] <- predict(
    models[[m]],
    newdata = test_data
  )
}


# ------------------------------------------------------------------------------
# 6. Test Error for Individual Models
# ------------------------------------------------------------------------------

individual_test_MSE <- colMeans(
  (
    test_predictions -
      test_data$y
  )^2
)


individual_test_RMSE <- sqrt(
  individual_test_MSE
)


individual_results <- data.frame(
  Model = model_names,
  Test_MSE = individual_test_MSE,
  Test_RMSE = individual_test_RMSE
)


individual_results


# ------------------------------------------------------------------------------
# 7. Best Single Model
# ------------------------------------------------------------------------------

best_single_index <- which.min(
  individual_test_MSE
)


best_single_model <- model_names[
  best_single_index
]


best_single_model


# ==============================================================================
# PART I
#
# EQUAL-WEIGHT MODEL AVERAGING
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Equal Weights
# ------------------------------------------------------------------------------

equal_weights <- rep(
  1 / M,
  M
)


names(
  equal_weights
) <- model_names


equal_weights


# ------------------------------------------------------------------------------
# 9. Equal-Weight Prediction
# ------------------------------------------------------------------------------

equal_average_prediction <- as.vector(
  test_predictions %*%
    equal_weights
)


equal_average_MSE <- mean(
  (
    test_data$y -
      equal_average_prediction
  )^2
)


equal_average_RMSE <- sqrt(
  equal_average_MSE
)


equal_average_MSE


# ------------------------------------------------------------------------------
# 10. Compare Equal Average with Individual Models
# ------------------------------------------------------------------------------

comparison_equal <- rbind(
  individual_results,
  data.frame(
    Model = "Equal Average",
    Test_MSE = equal_average_MSE,
    Test_RMSE = equal_average_RMSE
  )
)


comparison_equal


# ==============================================================================
# PART II
#
# HOLDOUT-VALIDATION WEIGHTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Split Training Data into Fit and Validation Sets
# ------------------------------------------------------------------------------

set.seed(456)


n_train <- nrow(
  train_data
)


fit_index <- sample(
  seq_len(
    n_train
  ),
  size = floor(
    0.75 *
      n_train
  )
)


validation_index <- setdiff(
  seq_len(
    n_train
  ),
  fit_index
)


fit_data <- train_data[
  fit_index,
]


validation_data <- train_data[
  validation_index,
]


# ------------------------------------------------------------------------------
# 12. Function to Fit All Candidate Models
# ------------------------------------------------------------------------------

fit_candidate_models <- function(
    data
) {
  
  list(
    
    Linear = lm(
      y ~ x1 + x2,
      data = data
    ),
    
    Quadratic = lm(
      y ~
        x1 +
        I(x1^2) +
        x2 +
        I(x2^2),
      data = data
    ),
    
    Quadratic_Interaction = lm(
      y ~
        x1 +
        I(x1^2) +
        x2 +
        I(x2^2) +
        x1:x2,
      data = data
    ),
    
    Cubic = lm(
      y ~
        x1 +
        I(x1^2) +
        I(x1^3) +
        x2 +
        I(x2^2) +
        I(x2^3) +
        x1:x2,
      data = data
    ),
    
    Sine_Structure = lm(
      y ~
        sin(x1) +
        x2 +
        I(x2^2) +
        x1:x2,
      data = data
    )
  )
}


validation_models <- fit_candidate_models(
  fit_data
)


# ------------------------------------------------------------------------------
# 13. Validation Errors
# ------------------------------------------------------------------------------

validation_MSE <- numeric(
  M
)


names(
  validation_MSE
) <- model_names


for (
  m in seq_len(M)
) {
  
  prediction <- predict(
    validation_models[[m]],
    newdata = validation_data
  )
  
  
  validation_MSE[m] <- mean(
    (
      validation_data$y -
        prediction
    )^2
  )
}


validation_MSE


# ------------------------------------------------------------------------------
# 14. Inverse-Error Weights
# ------------------------------------------------------------------------------

# A simple data-driven weighting scheme:
#
# w_m proportional to 1 / ValidationError_m
#
# Then normalize so weights sum to one.


inverse_error <- 1 /
  validation_MSE


validation_weights <- inverse_error /
  sum(
    inverse_error
  )


validation_weights


# ------------------------------------------------------------------------------
# 15. Refit Models on Full Training Data
# ------------------------------------------------------------------------------

full_models <- fit_candidate_models(
  train_data
)


# ------------------------------------------------------------------------------
# 16. Test Predictions
# ------------------------------------------------------------------------------

full_test_predictions <- matrix(
  NA,
  nrow = nrow(
    test_data
  ),
  ncol = M
)


colnames(
  full_test_predictions
) <- model_names


for (
  m in seq_len(M)
) {
  
  full_test_predictions[
    ,
    m
  ] <- predict(
    full_models[[m]],
    newdata = test_data
  )
}


# ------------------------------------------------------------------------------
# 17. Validation-Weighted Prediction
# ------------------------------------------------------------------------------

validation_average_prediction <- as.vector(
  full_test_predictions %*%
    validation_weights
)


validation_average_MSE <- mean(
  (
    test_data$y -
      validation_average_prediction
  )^2
)


validation_average_RMSE <- sqrt(
  validation_average_MSE
)


validation_average_MSE


# ==============================================================================
# PART III
#
# CROSS-VALIDATION WEIGHTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. K-Fold CV Errors for Candidate Models
# ------------------------------------------------------------------------------

model_cv_errors <- function(
    data,
    k = 10,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  n <- nrow(
    data
  )
  
  
  fold_id <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  fold_error <- matrix(
    NA,
    nrow = M,
    ncol = k
  )
  
  
  rownames(
    fold_error
  ) <- model_names
  
  
  for (
    fold in seq_len(k)
  ) {
    
    training_data <- data[
      fold_id != fold,
    ]
    
    
    validation_data <- data[
      fold_id == fold,
    ]
    
    
    models_fold <- fit_candidate_models(
      training_data
    )
    
    
    for (
      m in seq_len(M)
    ) {
      
      prediction <- predict(
        models_fold[[m]],
        newdata = validation_data
      )
      
      
      fold_error[
        m,
        fold
      ] <- mean(
        (
          validation_data$y -
            prediction
        )^2
      )
    }
  }
  
  
  return(
    list(
      Fold_Error = fold_error,
      Mean_Error = rowMeans(
        fold_error
      ),
      SE_Error = apply(
        fold_error,
        1,
        sd
      ) /
        sqrt(k)
    )
  )
}


# ------------------------------------------------------------------------------
# 19. Run 10-Fold CV
# ------------------------------------------------------------------------------

CV_results <- model_cv_errors(
  train_data,
  k = 10,
  seed = 123
)


CV_results$Mean_Error


# ------------------------------------------------------------------------------
# 20. Convert CV Errors to Weights
# ------------------------------------------------------------------------------

inverse_CV_error <- 1 /
  CV_results$Mean_Error


CV_weights <- inverse_CV_error /
  sum(
    inverse_CV_error
  )


CV_weights


# ------------------------------------------------------------------------------
# 21. CV-Weighted Ensemble Prediction
# ------------------------------------------------------------------------------

CV_average_prediction <- as.vector(
  full_test_predictions %*%
    CV_weights
)


CV_average_MSE <- mean(
  (
    test_data$y -
      CV_average_prediction
  )^2
)


CV_average_RMSE <- sqrt(
  CV_average_MSE
)


CV_average_MSE


# ------------------------------------------------------------------------------
# 22. Plot CV Errors and Weights
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    2
  )
)


barplot(
  CV_results$Mean_Error,
  names.arg = model_names,
  las = 2,
  ylab = "CV MSE",
  main = "Cross-Validation Error"
)


barplot(
  CV_weights,
  names.arg = model_names,
  las = 2,
  ylab = "Weight",
  main = "CV-Based Model Weights"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART IV
#
# BIC-BASED MODEL AVERAGING
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Compute BIC for Candidate Models
# ------------------------------------------------------------------------------

BIC_values <- sapply(
  full_models,
  BIC
)


BIC_values


# ------------------------------------------------------------------------------
# 24. Delta BIC
# ------------------------------------------------------------------------------

# Only relative BIC values matter.
#
# Delta_m =
#
# BIC_m - minimum BIC


delta_BIC <- BIC_values -
  min(
    BIC_values
  )


delta_BIC


# ------------------------------------------------------------------------------
# 25. BIC Approximation to Model Weights
# ------------------------------------------------------------------------------

# A common approximation:
#
# weight_m
#
# proportional to
#
# exp(-0.5 * Delta_BIC_m)
#
# Normalize to sum to 1.


BIC_raw_weights <- exp(
  -0.5 *
    delta_BIC
)


BIC_weights <- BIC_raw_weights /
  sum(
    BIC_raw_weights
  )


BIC_weights


# ------------------------------------------------------------------------------
# 26. BIC-Weighted Prediction
# ------------------------------------------------------------------------------

BIC_average_prediction <- as.vector(
  full_test_predictions %*%
    BIC_weights
)


BIC_average_MSE <- mean(
  (
    test_data$y -
      BIC_average_prediction
  )^2
)


BIC_average_RMSE <- sqrt(
  BIC_average_MSE
)


# ------------------------------------------------------------------------------
# 27. Plot BIC Weights
# ------------------------------------------------------------------------------

barplot(
  BIC_weights,
  names.arg = model_names,
  las = 2,
  ylab = "Approximate Model Probability",
  main = "BIC-Based Model Weights"
)


# ==============================================================================
# PART V
#
# OPTIMAL CONVEX WEIGHTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Validation Prediction Matrix
# ------------------------------------------------------------------------------

# Instead of imposing a weighting formula, estimate weights that minimize
# validation squared error.
#
# Solve:
#
# min_w ||y - P w||^2
#
# subject to:
#
# w_m >= 0
# sum_m w_m = 1
#
#
# Here we implement a simple grid/random-search approximation without packages.


validation_predictions <- matrix(
  NA,
  nrow = nrow(
    validation_data
  ),
  ncol = M
)


colnames(
  validation_predictions
) <- model_names


for (
  m in seq_len(M)
) {
  
  validation_predictions[
    ,
    m
  ] <- predict(
    validation_models[[m]],
    newdata = validation_data
  )
}


# ------------------------------------------------------------------------------
# 29. Random Simplex Weight Generator
# ------------------------------------------------------------------------------

random_simplex <- function(
    M,
    number_draws
) {
  
  raw <- matrix(
    rexp(
      number_draws *
        M,
      rate = 1
    ),
    nrow = number_draws,
    ncol = M
  )
  
  
  raw /
    rowSums(
      raw
    )
}


# ------------------------------------------------------------------------------
# 30. Search for Best Convex Weights
# ------------------------------------------------------------------------------

set.seed(123)


number_weight_candidates <- 50000


candidate_weights <- random_simplex(
  M,
  number_weight_candidates
)


candidate_error <- numeric(
  number_weight_candidates
)


for (
  i in seq_len(
    number_weight_candidates
  )
) {
  
  prediction_i <- as.vector(
    validation_predictions %*%
      candidate_weights[
        i,
      ]
  )
  
  
  candidate_error[i] <- mean(
    (
      validation_data$y -
        prediction_i
    )^2
  )
}


best_weight_index <- which.min(
  candidate_error
)


optimized_weights <- candidate_weights[
  best_weight_index,
]


names(
  optimized_weights
) <- model_names


optimized_weights


# ------------------------------------------------------------------------------
# 31. Optimized Ensemble Prediction
# ------------------------------------------------------------------------------

optimized_prediction <- as.vector(
  full_test_predictions %*%
    optimized_weights
)


optimized_MSE <- mean(
  (
    test_data$y -
      optimized_prediction
  )^2
)


optimized_RMSE <- sqrt(
  optimized_MSE
)


optimized_MSE


# ==============================================================================
# PART VI
#
# WHY AVERAGING CAN HELP
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Prediction Error Correlations
# ------------------------------------------------------------------------------

# Averaging is especially useful when candidate models make different errors.
#
# Highly correlated errors provide less diversification.


error_matrix <- sweep(
  full_test_predictions,
  MARGIN = 1,
  STATS = test_data$y,
  FUN = "-"
)


error_correlation <- cor(
  error_matrix
)


round(
  error_correlation,
  3
)


# ------------------------------------------------------------------------------
# 33. Plot Error Correlations
# ------------------------------------------------------------------------------

image(
  1:M,
  1:M,
  error_correlation,
  axes = FALSE,
  xlab = "",
  ylab = "",
  main = "Correlation of Model Prediction Errors"
)


axis(
  1,
  at = 1:M,
  labels = model_names,
  las = 2
)


axis(
  2,
  at = 1:M,
  labels = model_names,
  las = 2
)


# ------------------------------------------------------------------------------
# 34. Variance Reduction Illustration
# ------------------------------------------------------------------------------

# Suppose two model errors have:
#
# Var(e1) = sigma1^2
# Var(e2) = sigma2^2
# Corr(e1,e2) = rho
#
# Equal averaging gives error:
#
# e_avg = (e1 + e2) / 2
#
# Variance:
#
# Var(e_avg)
#
# =
# 1/4[
# sigma1^2 +
# sigma2^2 +
# 2 rho sigma1 sigma2
# ]


model_a <- 1

model_b <- 2


error_a <- error_matrix[
  ,
  model_a
]


error_b <- error_matrix[
  ,
  model_b
]


variance_a <- var(
  error_a
)


variance_b <- var(
  error_b
)


correlation_ab <- cor(
  error_a,
  error_b
)


theoretical_average_variance <-
  0.25 *
  (
    variance_a +
      variance_b +
      2 *
      correlation_ab *
      sqrt(
        variance_a *
          variance_b
      )
  )


empirical_average_variance <- var(
  (
    error_a +
      error_b
  ) /
    2
)


data.frame(
  Quantity = c(
    "Theoretical Average Error Variance",
    "Empirical Average Error Variance"
  ),
  Value = c(
    theoretical_average_variance,
    empirical_average_variance
  )
)


# ==============================================================================
# PART VII
#
# COMPARE ALL APPROACHES
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Ensemble Results
# ------------------------------------------------------------------------------

ensemble_results <- data.frame(
  Method = c(
    "Best Individual Model",
    "Equal-Weight Average",
    "Validation-Weighted",
    "CV-Weighted",
    "BIC-Weighted",
    "Optimized Convex Average"
  ),
  
  Test_MSE = c(
    min(
      individual_test_MSE
    ),
    equal_average_MSE,
    validation_average_MSE,
    CV_average_MSE,
    BIC_average_MSE,
    optimized_MSE
  )
)


ensemble_results$Test_RMSE <- sqrt(
  ensemble_results$Test_MSE
)


ensemble_results


# ------------------------------------------------------------------------------
# 36. Weight Comparison
# ------------------------------------------------------------------------------

weight_comparison <- data.frame(
  Model = model_names,
  Equal = equal_weights,
  Validation = validation_weights,
  CV = CV_weights,
  BIC = BIC_weights,
  Optimized = optimized_weights
)


weight_comparison


# ------------------------------------------------------------------------------
# 37. Plot Weighting Schemes
# ------------------------------------------------------------------------------

matplot(
  t(
    as.matrix(
      weight_comparison[
        ,
        -1
      ]
    )
  ),
  type = "l",
  lty = 1:M,
  xaxt = "n",
  xlab = "Weighting Method",
  ylab = "Model Weight",
  main = "Model Averaging Weights"
)


axis(
  1,
  at = 1:5,
  labels = c(
    "Equal",
    "Validation",
    "CV",
    "BIC",
    "Optimized"
  )
)


legend(
  "topright",
  legend = model_names,
  lty = 1:M
)


# ==============================================================================
# PART VIII
#
# PREDICTION VISUALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Create One-Dimensional Slice
# ------------------------------------------------------------------------------

# Fix x2 = 0 and vary x1.


x1_grid <- seq(
  -3,
  3,
  length.out = 300
)


slice_data <- data.frame(
  x1 = x1_grid,
  x2 = 0
)


# ------------------------------------------------------------------------------
# 39. Candidate Model Predictions on Slice
# ------------------------------------------------------------------------------

slice_predictions <- matrix(
  NA,
  nrow = length(
    x1_grid
  ),
  ncol = M
)


for (
  m in seq_len(M)
) {
  
  slice_predictions[
    ,
    m
  ] <- predict(
    full_models[[m]],
    newdata = slice_data
  )
}


# ------------------------------------------------------------------------------
# 40. Averaged Predictions on Slice
# ------------------------------------------------------------------------------

slice_equal <- as.vector(
  slice_predictions %*%
    equal_weights
)


slice_CV <- as.vector(
  slice_predictions %*%
    CV_weights
)


slice_BIC <- as.vector(
  slice_predictions %*%
    BIC_weights
)


slice_optimized <- as.vector(
  slice_predictions %*%
    optimized_weights
)


true_slice <- true_function(
  x1_grid,
  0
)


# ------------------------------------------------------------------------------
# 41. Plot Candidate Models
# ------------------------------------------------------------------------------

matplot(
  x1_grid,
  slice_predictions,
  type = "l",
  lty = 1:M,
  xlab = "x1",
  ylab = "Predicted y",
  main = "Candidate Model Predictions"
)


lines(
  x1_grid,
  true_slice,
  lwd = 3
)


# ------------------------------------------------------------------------------
# 42. Plot Averaged Models
# ------------------------------------------------------------------------------

plot(
  x1_grid,
  true_slice,
  type = "l",
  lwd = 3,
  xlab = "x1",
  ylab = "Predicted y",
  main = "Model-Averaged Predictions"
)


lines(
  x1_grid,
  slice_equal,
  lty = 2,
  lwd = 2
)


lines(
  x1_grid,
  slice_CV,
  lty = 3,
  lwd = 2
)


lines(
  x1_grid,
  slice_BIC,
  lty = 4,
  lwd = 2
)


lines(
  x1_grid,
  slice_optimized,
  lty = 5,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "True",
    "Equal",
    "CV",
    "BIC",
    "Optimized"
  ),
  lty = c(
    1,
    2,
    3,
    4,
    5
  ),
  lwd = c(
    3,
    2,
    2,
    2,
    2
  )
)


# ==============================================================================
# PART IX
#
# MODEL UNCERTAINTY
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Predictions at One Point
# ------------------------------------------------------------------------------

new_point <- data.frame(
  x1 = 1,
  x2 = 1
)


point_predictions <- sapply(
  full_models,
  predict,
  newdata = new_point
)


point_predictions


# ------------------------------------------------------------------------------
# 44. Spread Across Model Predictions
# ------------------------------------------------------------------------------

model_prediction_mean <- mean(
  point_predictions
)


model_prediction_sd <- sd(
  point_predictions
)


data.frame(
  Mean_Prediction = model_prediction_mean,
  SD_Across_Models = model_prediction_sd
)


# ------------------------------------------------------------------------------
# 45. Weighted Prediction at the Point
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Equal",
    "Validation",
    "CV",
    "BIC",
    "Optimized"
  ),
  
  Prediction = c(
    sum(
      equal_weights *
        point_predictions
    ),
    
    sum(
      validation_weights *
        point_predictions
    ),
    
    sum(
      CV_weights *
        point_predictions
    ),
    
    sum(
      BIC_weights *
        point_predictions
    ),
    
    sum(
      optimized_weights *
        point_predictions
    )
  )
)


# ==============================================================================
# PART X
#
# BOOTSTRAP MODEL AVERAGING
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Bootstrap Model Averaging Illustration
# ------------------------------------------------------------------------------

# Another form of averaging arises by fitting the same model to many
# bootstrap samples and averaging predictions.
#
# This is the central idea behind bagging.


set.seed(123)


B <- 500


bootstrap_prediction <- numeric(
  B
)


for (
  b in seq_len(B)
) {
  
  bootstrap_index <- sample(
    seq_len(
      nrow(
        train_data
      )
    ),
    size = nrow(
      train_data
    ),
    replace = TRUE
  )
  
  
  bootstrap_data <- train_data[
    bootstrap_index,
  ]
  
  
  bootstrap_model <- lm(
    y ~
      x1 +
      I(x1^2) +
      x2 +
      I(x2^2) +
      x1:x2,
    data = bootstrap_data
  )
  
  
  bootstrap_prediction[b] <- predict(
    bootstrap_model,
    newdata = new_point
  )
}


mean(
  bootstrap_prediction
)


sd(
  bootstrap_prediction
)


hist(
  bootstrap_prediction,
  breaks = 40,
  xlab = "Prediction",
  main = "Bootstrap Distribution of Model Prediction"
)


# ==============================================================================
# PART XI
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Best Ensemble
# ------------------------------------------------------------------------------

best_method_index <- which.min(
  ensemble_results$Test_MSE
)


best_method <- ensemble_results$Method[
  best_method_index
]


# ------------------------------------------------------------------------------
# 48. Summary
# ------------------------------------------------------------------------------

cat(
  "Model Averaging Summary\n"
)


cat(
  "-----------------------\n"
)


cat(
  "Best individual model:",
  best_single_model,
  "\n"
)


cat(
  "Best individual test RMSE:",
  round(
    sqrt(
      min(
        individual_test_MSE
      )
    ),
    4
  ),
  "\n"
)


cat(
  "Equal-average test RMSE:",
  round(
    equal_average_RMSE,
    4
  ),
  "\n"
)


cat(
  "Validation-weighted test RMSE:",
  round(
    validation_average_RMSE,
    4
  ),
  "\n"
)


cat(
  "CV-weighted test RMSE:",
  round(
    CV_average_RMSE,
    4
  ),
  "\n"
)


cat(
  "BIC-weighted test RMSE:",
  round(
    BIC_average_RMSE,
    4
  ),
  "\n"
)


cat(
  "Optimized convex-average test RMSE:",
  round(
    optimized_RMSE,
    4
  ),
  "\n"
)


cat(
  "Best overall averaging method:",
  best_method,
  "\n"
)