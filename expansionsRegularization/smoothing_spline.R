# ==============================================================================
# Smoothing Splines
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Penalized least squares
#   - Smoothness penalty
#   - Smoothing parameter lambda
#   - Bias-variance tradeoff
#   - Effective degrees of freedom
#   - Smoother matrix
#   - Leave-one-out cross-validation
#   - Generalized cross-validation
#   - Comparison with smooth.spline()
#
# Note:
#   The manual model below uses a dense cubic B-spline basis with a
#   second-difference penalty on adjacent coefficients.
#
#   This is a P-spline-style approximation to the smoothing-spline idea.
#   Base R's smooth.spline() is then used to verify the exact classical
#   smoothing-spline behavior.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Nonlinear Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 250

x <- sort(
  runif(
    n,
    min = 0,
    max = 10
  )
)


true_function <- function(x) {
  
  2 +
    0.5 * x +
    2 * sin(x)
}


y_true <- true_function(
  x
)


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = 0.8
  )


data <- data.frame(
  x = x,
  y = y
)


head(data)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Nonlinear Regression Data"
)


lines(
  x,
  y_true,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Build a Dense Cubic B-Spline Basis
# ------------------------------------------------------------------------------

# We use many interior knots.
#
# The smoothing penalty, rather than the raw number of knots,
# will control effective complexity.


number_internal_knots <- 25


internal_knots <- as.numeric(
  quantile(
    x,
    probs = seq(
      0,
      1,
      length.out = number_internal_knots + 2
    )[
      -c(
        1,
        number_internal_knots + 2
      )
    ]
  )
)


boundary_knots <- range(
  x
)


# splines is part of the standard R installation.

B <- splines::bs(
  x,
  knots = internal_knots,
  degree = 3,
  Boundary.knots = boundary_knots,
  intercept = TRUE
)


B <- as.matrix(
  B
)


dim(
  B
)


# ------------------------------------------------------------------------------
# 4. Plot B-Spline Basis Functions
# ------------------------------------------------------------------------------

matplot(
  x,
  B,
  type = "l",
  lty = 1,
  xlab = "x",
  ylab = "Basis Value",
  main = "Dense Cubic B-Spline Basis"
)


# ------------------------------------------------------------------------------
# 5. Construct Roughness Penalty
# ------------------------------------------------------------------------------

# Let beta be the B-spline coefficient vector.
#
# We penalize second differences:
#
# beta_j - 2*beta_{j+1} + beta_{j+2}
#
# This discourages rapid changes in slope.


p <- ncol(
  B
)


D2 <- diff(
  diag(p),
  differences = 2
)


dim(
  D2
)


# Penalty matrix:

Omega <- crossprod(
  D2
)


dim(
  Omega
)


# ------------------------------------------------------------------------------
# 6. Penalized Least-Squares Objective
# ------------------------------------------------------------------------------

# Minimize:
#
# ||y - B beta||^2
#
# +
#
# lambda * beta' Omega beta


smoothing_objective <- function(
    beta,
    B,
    y,
    lambda,
    Omega
) {
  
  residual <- y -
    as.vector(
      B %*%
        beta
    )
  
  
  RSS <- sum(
    residual^2
  )
  
  
  penalty <- lambda *
    as.numeric(
      t(beta) %*%
        Omega %*%
        beta
    )
  
  
  return(
    RSS +
      penalty
  )
}


# ------------------------------------------------------------------------------
# 7. Closed-Form Penalized Solution
# ------------------------------------------------------------------------------

# The normal equations are:
#
# (B'B + lambda*Omega) beta = B'y
#
#
# Therefore:
#
# beta_hat =
# (B'B + lambda*Omega)^(-1) B'y


smoothing_fit <- function(
    B,
    y,
    lambda,
    Omega
) {
  
  B <- as.matrix(
    B
  )
  
  
  penalty_matrix <- lambda *
    Omega
  
  
  beta_hat <- solve(
    crossprod(B) +
      penalty_matrix,
    crossprod(
      B,
      y
    )
  )
  
  
  fitted <- as.vector(
    B %*%
      beta_hat
  )
  
  
  return(
    list(
      coefficients = as.vector(
        beta_hat
      ),
      fitted = fitted,
      lambda = lambda
    )
  )
}


# ------------------------------------------------------------------------------
# 8. Fit One Smoothing Parameter
# ------------------------------------------------------------------------------

lambda <- 10


model <- smoothing_fit(
  B = B,
  y = y,
  lambda = lambda,
  Omega = Omega
)


model$coefficients


# ------------------------------------------------------------------------------
# 9. Plot Smoothed Fit
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = paste(
    "Penalized Spline: lambda =",
    lambda
  )
)


lines(
  x,
  model$fitted,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 10. Training Error
# ------------------------------------------------------------------------------

MSE <- mean(
  (
    y -
      model$fitted
  )^2
)


RMSE <- sqrt(
  MSE
)


data.frame(
  MSE = MSE,
  RMSE = RMSE
)


# ------------------------------------------------------------------------------
# 11. Roughness of the Fit
# ------------------------------------------------------------------------------

roughness <- as.numeric(
  t(
    model$coefficients
  ) %*%
    Omega %*%
    model$coefficients
)


roughness


# ------------------------------------------------------------------------------
# 12. Compare Different Lambda Values
# ------------------------------------------------------------------------------

lambda_values <- c(
  0,
  0.01,
  0.1,
  1,
  10,
  100,
  1000,
  10000
)


lambda_models <- vector(
  "list",
  length(
    lambda_values
  )
)


lambda_results <- data.frame(
  Lambda = lambda_values,
  Training_MSE = NA,
  Roughness = NA
)


for (i in seq_along(
  lambda_values
)) {
  
  lambda_i <- lambda_values[i]
  
  
  model_i <- smoothing_fit(
    B = B,
    y = y,
    lambda = lambda_i,
    Omega = Omega
  )
  
  
  lambda_models[[i]] <- model_i
  
  
  lambda_results$Training_MSE[i] <- mean(
    (
      y -
        model_i$fitted
    )^2
  )
  
  
  lambda_results$Roughness[i] <-
    as.numeric(
      t(
        model_i$coefficients
      ) %*%
        Omega %*%
        model_i$coefficients
    )
}


lambda_results


# ------------------------------------------------------------------------------
# 13. Plot Fits for Different Lambda Values
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    2,
    4
  )
)


for (i in seq_along(
  lambda_values
)) {
  
  plot(
    x,
    y,
    pch = 19,
    xlab = "x",
    ylab = "y",
    main = paste(
      "lambda =",
      lambda_values[i]
    )
  )
  
  
  lines(
    x,
    lambda_models[[i]]$fitted,
    lwd = 2
  )
  
  
  lines(
    x,
    y_true,
    lty = 2
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 14. Training Error vs Lambda
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_results$Lambda +
      1e-8
  ),
  lambda_results$Training_MSE,
  type = "b",
  pch = 19,
  xlab = "log10(lambda)",
  ylab = "Training MSE",
  main = "Training Error vs Smoothing"
)


# ------------------------------------------------------------------------------
# 15. Roughness vs Lambda
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_results$Lambda +
      1e-8
  ),
  lambda_results$Roughness,
  type = "b",
  pch = 19,
  xlab = "log10(lambda)",
  ylab = "Roughness",
  main = "Roughness vs Smoothing Parameter"
)


# ------------------------------------------------------------------------------
# 16. Smoother Matrix
# ------------------------------------------------------------------------------

# The fitted values can be written:
#
# y_hat = S_lambda y
#
# where
#
# S_lambda =
#
# B (B'B + lambda Omega)^(-1) B'


smoother_matrix <- function(
    B,
    lambda,
    Omega
) {
  
  middle <- solve(
    crossprod(B) +
      lambda *
      Omega
  )
  
  
  S <- B %*%
    middle %*%
    t(B)
  
  
  return(
    S
  )
}


# ------------------------------------------------------------------------------
# 17. Verify Smoother Matrix
# ------------------------------------------------------------------------------

S <- smoother_matrix(
  B,
  lambda = 10,
  Omega = Omega
)


fitted_from_S <- as.vector(
  S %*%
    y
)


max(
  abs(
    fitted_from_S -
      model$fitted
  )
)


# ------------------------------------------------------------------------------
# 18. Effective Degrees of Freedom
# ------------------------------------------------------------------------------

# Effective degrees of freedom:
#
# df(lambda) = trace(S_lambda)
#
# Small lambda:
# high df
#
# Large lambda:
# low df


effective_df <- function(
    B,
    lambda,
    Omega
) {
  
  S <- smoother_matrix(
    B,
    lambda,
    Omega
  )
  
  
  sum(
    diag(
      S
    )
  )
}


effective_df(
  B,
  lambda = 10,
  Omega = Omega
)


# ------------------------------------------------------------------------------
# 19. Degrees of Freedom Across Lambda
# ------------------------------------------------------------------------------

lambda_grid <- 10^seq(
  -5,
  6,
  length.out = 200
)


df_path <- numeric(
  length(
    lambda_grid
  )
)


training_MSE_path <- numeric(
  length(
    lambda_grid
  )
)


roughness_path <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  lambda_i <- lambda_grid[i]
  
  
  model_i <- smoothing_fit(
    B,
    y,
    lambda_i,
    Omega
  )
  
  
  df_path[i] <- effective_df(
    B,
    lambda_i,
    Omega
  )
  
  
  training_MSE_path[i] <- mean(
    (
      y -
        model_i$fitted
    )^2
  )
  
  
  roughness_path[i] <-
    as.numeric(
      t(
        model_i$coefficients
      ) %*%
        Omega %*%
        model_i$coefficients
    )
}


# ------------------------------------------------------------------------------
# 20. Plot Effective Degrees of Freedom
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  df_path,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "Effective Degrees of Freedom",
  main = "Smoothing Spline Complexity"
)


# ------------------------------------------------------------------------------
# 21. Lambda vs Degrees of Freedom
# ------------------------------------------------------------------------------

complexity_table <- data.frame(
  Lambda = lambda_grid[
    seq(
      1,
      length(
        lambda_grid
      ),
      length.out = 10
    )
  ],
  Degrees_of_Freedom = df_path[
    seq(
      1,
      length(
        df_path
      ),
      length.out = 10
    )
  ]
)


complexity_table


# ------------------------------------------------------------------------------
# 22. Hat Values
# ------------------------------------------------------------------------------

# Like linear regression, the diagonal of S measures leverage.


hat_values <- diag(
  S
)


summary(
  hat_values
)


# ------------------------------------------------------------------------------
# 23. Leave-One-Out Cross-Validation
# ------------------------------------------------------------------------------

# For a linear smoother:
#
# LOOCV residual_i =
#
# (y_i - yhat_i) / (1 - S_ii)
#
#
# This avoids refitting n separate models.


loocv_error <- function(
    B,
    y,
    lambda,
    Omega
) {
  
  S <- smoother_matrix(
    B,
    lambda,
    Omega
  )
  
  
  fitted <- as.vector(
    S %*%
      y
  )
  
  
  residuals <- y -
    fitted
  
  
  leverage <- diag(
    S
  )
  
  
  adjusted_residuals <- residuals /
    (
      1 -
        leverage
    )
  
  
  mean(
    adjusted_residuals^2
  )
}


# ------------------------------------------------------------------------------
# 24. LOOCV Across Lambda
# ------------------------------------------------------------------------------

LOOCV_path <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  LOOCV_path[i] <- loocv_error(
    B,
    y,
    lambda_grid[i],
    Omega
  )
}


best_LOOCV_index <- which.min(
  LOOCV_path
)


best_LOOCV_lambda <- lambda_grid[
  best_LOOCV_index
]


best_LOOCV_lambda


# ------------------------------------------------------------------------------
# 25. Plot LOOCV Error
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  LOOCV_path,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "LOOCV MSE",
  main = "LOOCV Selection of Smoothing Parameter"
)


abline(
  v = log10(
    best_LOOCV_lambda
  ),
  lty = 2
)


# ------------------------------------------------------------------------------
# 26. Generalized Cross-Validation
# ------------------------------------------------------------------------------

# GCV(lambda) =
#
# [ RSS / n ]
# ---------------------------
# [1 - df(lambda)/n]^2


gcv_error <- function(
    B,
    y,
    lambda,
    Omega
) {
  
  S <- smoother_matrix(
    B,
    lambda,
    Omega
  )
  
  
  fitted <- as.vector(
    S %*%
      y
  )
  
  
  RSS <- sum(
    (
      y -
        fitted
    )^2
  )
  
  
  df <- sum(
    diag(
      S
    )
  )
  
  
  n <- length(
    y
  )
  
  
  GCV <- (
    RSS /
      n
  ) /
    (
      1 -
        df /
        n
    )^2
  
  
  return(
    GCV
  )
}


# ------------------------------------------------------------------------------
# 27. GCV Across Lambda
# ------------------------------------------------------------------------------

GCV_path <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  GCV_path[i] <- gcv_error(
    B,
    y,
    lambda_grid[i],
    Omega
  )
}


best_GCV_index <- which.min(
  GCV_path
)


best_GCV_lambda <- lambda_grid[
  best_GCV_index
]


best_GCV_lambda


# ------------------------------------------------------------------------------
# 28. Plot GCV Error
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  GCV_path,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "GCV",
  main = "Generalized Cross-Validation"
)


abline(
  v = log10(
    best_GCV_lambda
  ),
  lty = 2
)


# ------------------------------------------------------------------------------
# 29. Compare LOOCV and GCV Choices
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "LOOCV",
    "GCV"
  ),
  Lambda = c(
    best_LOOCV_lambda,
    best_GCV_lambda
  )
)


# ------------------------------------------------------------------------------
# 30. Fit Final Penalized Spline
# ------------------------------------------------------------------------------

final_lambda <- best_GCV_lambda


final_model <- smoothing_fit(
  B = B,
  y = y,
  lambda = final_lambda,
  Omega = Omega
)


final_df <- effective_df(
  B,
  final_lambda,
  Omega
)


final_MSE <- mean(
  (
    y -
      final_model$fitted
  )^2
)


# ------------------------------------------------------------------------------
# 31. Plot Final Penalized Spline
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Final Penalized Spline"
)


lines(
  x,
  final_model$fitted,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Penalized Spline",
    "True Function"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 32. Compare Underfitting and Overfitting
# ------------------------------------------------------------------------------

lambda_small <- 1e-5

lambda_medium <- final_lambda

lambda_large <- 1e6


model_small <- smoothing_fit(
  B,
  y,
  lambda_small,
  Omega
)


model_medium <- smoothing_fit(
  B,
  y,
  lambda_medium,
  Omega
)


model_large <- smoothing_fit(
  B,
  y,
  lambda_large,
  Omega
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Effect of Smoothing Parameter"
)


lines(
  x,
  model_small$fitted,
  lty = 3,
  lwd = 2
)


lines(
  x,
  model_medium$fitted,
  lty = 1,
  lwd = 2
)


lines(
  x,
  model_large$fitted,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Small lambda",
    "Selected lambda",
    "Large lambda"
  ),
  lty = c(
    3,
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 33. Effective Degrees of Freedom for Three Fits
# ------------------------------------------------------------------------------

data.frame(
  Model = c(
    "Small Lambda",
    "Selected Lambda",
    "Large Lambda"
  ),
  Lambda = c(
    lambda_small,
    lambda_medium,
    lambda_large
  ),
  Effective_DF = c(
    effective_df(
      B,
      lambda_small,
      Omega
    ),
    
    effective_df(
      B,
      lambda_medium,
      Omega
    ),
    
    effective_df(
      B,
      lambda_large,
      Omega
    )
  )
)


# ------------------------------------------------------------------------------
# 34. Compare with Unpenalized Regression Spline
# ------------------------------------------------------------------------------

unpenalized_model <- smoothing_fit(
  B,
  y,
  lambda = 0,
  Omega = Omega
)


unpenalized_MSE <- mean(
  (
    y -
      unpenalized_model$fitted
  )^2
)


data.frame(
  Model = c(
    "Unpenalized Dense Regression Spline",
    "Penalized Spline"
  ),
  Training_MSE = c(
    unpenalized_MSE,
    final_MSE
  ),
  Effective_DF = c(
    effective_df(
      B,
      0,
      Omega
    ),
    final_df
  )
)


# ------------------------------------------------------------------------------
# 35. Verify Against Base R smooth.spline()
# ------------------------------------------------------------------------------

# Base R provides the classical cubic smoothing spline.
#
# cv = TRUE asks smooth.spline() to choose its smoothing level
# using ordinary leave-one-out cross-validation.


builtin_spline <- smooth.spline(
  x = x,
  y = y,
  cv = TRUE
)


builtin_spline


# ------------------------------------------------------------------------------
# 36. Built-In Smoothing Spline Predictions
# ------------------------------------------------------------------------------

builtin_prediction <- predict(
  builtin_spline,
  x
)$y


builtin_MSE <- mean(
  (
    y -
      builtin_prediction
  )^2
)


builtin_MSE


# ------------------------------------------------------------------------------
# 37. Built-In Effective Degrees of Freedom
# ------------------------------------------------------------------------------

builtin_spline$df


# ------------------------------------------------------------------------------
# 38. Plot Manual and Classical Smoothing Splines
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Penalized B-Spline vs Classical Smoothing Spline"
)


lines(
  x,
  final_model$fitted,
  lwd = 2
)


lines(
  x,
  builtin_prediction,
  lty = 2,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Manual Penalized B-Spline",
    "smooth.spline()",
    "True Function"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 39. Compare Manual and Built-In Fits
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Manual Penalized B-Spline",
    "Classical smooth.spline()"
  ),
  Training_MSE = c(
    final_MSE,
    builtin_MSE
  ),
  Effective_DF = c(
    final_df,
    builtin_spline$df
  )
)


# ------------------------------------------------------------------------------
# 40. smooth.spline() with Specified Degrees of Freedom
# ------------------------------------------------------------------------------

# Instead of supplying lambda directly, smoothing splines are often
# tuned through their effective degrees of freedom.


df_values <- c(
  3,
  5,
  10,
  20
)


builtin_df_models <- vector(
  "list",
  length(
    df_values
  )
)


par(
  mfrow = c(
    2,
    2
  )
)


for (i in seq_along(
  df_values
)) {
  
  model_i <- smooth.spline(
    x,
    y,
    df = df_values[i]
  )
  
  
  builtin_df_models[[i]] <- model_i
  
  
  prediction_i <- predict(
    model_i,
    x
  )$y
  
  
  plot(
    x,
    y,
    pch = 19,
    xlab = "x",
    ylab = "y",
    main = paste(
      "df =",
      df_values[i]
    )
  )
  
  
  lines(
    x,
    prediction_i,
    lwd = 2
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 41. Bias-Variance Illustration
# ------------------------------------------------------------------------------

# Low df:
# smoother, higher bias
#
# High df:
# more flexible, higher variance


df_comparison <- data.frame(
  Degrees_of_Freedom = df_values,
  Training_MSE = NA
)


for (i in seq_along(
  df_values
)) {
  
  prediction_i <- predict(
    builtin_df_models[[i]],
    x
  )$y
  
  
  df_comparison$Training_MSE[i] <- mean(
    (
      y -
        prediction_i
    )^2
  )
}


df_comparison


# ------------------------------------------------------------------------------
# 42. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)


train_index <- sample(
  seq_len(n),
  size = floor(
    0.7 *
      n
  )
)


test_index <- setdiff(
  seq_len(n),
  train_index
)


x_train <- x[
  train_index
]


y_train <- y[
  train_index
]


x_test <- x[
  test_index
]


y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 43. Fit Classical Smoothing Spline on Training Data
# ------------------------------------------------------------------------------

train_spline <- smooth.spline(
  x = x_train,
  y = y_train,
  cv = TRUE
)


train_spline$df


# ------------------------------------------------------------------------------
# 44. Test Predictions
# ------------------------------------------------------------------------------

test_prediction <- predict(
  train_spline,
  x_test
)$y


test_MSE <- mean(
  (
    y_test -
      test_prediction
  )^2
)


test_RMSE <- sqrt(
  test_MSE
)


data.frame(
  Test_MSE = test_MSE,
  Test_RMSE = test_RMSE
)


# ------------------------------------------------------------------------------
# 45. Summary
# ------------------------------------------------------------------------------

cat(
  "Selected manual lambda:",
  final_lambda,
  "\n"
)


cat(
  "Manual effective degrees of freedom:",
  round(
    final_df,
    3
  ),
  "\n"
)


cat(
  "Manual training MSE:",
  round(
    final_MSE,
    4
  ),
  "\n"
)


cat(
  "smooth.spline() effective degrees of freedom:",
  round(
    builtin_spline$df,
    3
  ),
  "\n"
)


cat(
  "smooth.spline() training MSE:",
  round(
    builtin_MSE,
    4
  ),
  "\n"
)


cat(
  "smooth.spline() test RMSE:",
  round(
    test_RMSE,
    4
  ),
  "\n"
)