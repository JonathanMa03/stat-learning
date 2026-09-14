# ==============================================================================
# Quadratic Regularization
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Penalized least squares
#   - General quadratic penalties
#   - Ridge as a special case
#   - Structured coefficient penalties
#   - Difference penalties
#   - Smoother / hat matrix
#   - Effective degrees of freedom
#   - Bias-variance tradeoff
#   - Cross-validation for lambda
#
# General objective:
#
#   minimize_beta
#
#       ||y - X beta||^2
#       +
#       lambda * beta' Omega beta
#
#
# where Omega is a positive semidefinite penalty matrix.
#
# Special cases:
#
#   Omega = I
#
# gives ridge regression.
#
#   Omega = D'D
#
# gives structured difference penalties.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Structured Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 200

p <- 20


# Predictors

X <- matrix(
  rnorm(
    n * p
  ),
  nrow = n,
  ncol = p
)


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# True coefficients vary smoothly across predictor index.
#
# This creates a setting where structured regularization is useful.

beta_true <- 3 *
  sin(
    seq(
      0,
      pi,
      length.out = p
    )
  )


beta_true <- beta_true +
  0.5 *
  cos(
    seq(
      0,
      3 * pi,
      length.out = p
    )
  )


beta_0 <- 2

sigma <- 3


epsilon <- rnorm(
  n,
  mean = 0,
  sd = sigma
)


y <- beta_0 +
  as.vector(
    X %*%
      beta_true
  ) +
  epsilon


# ------------------------------------------------------------------------------
# 2. Plot True Coefficient Structure
# ------------------------------------------------------------------------------

plot(
  seq_len(p),
  beta_true,
  type = "b",
  pch = 19,
  xlab = "Coefficient Index",
  ylab = "Coefficient",
  main = "True Regression Coefficients"
)


# ------------------------------------------------------------------------------
# 3. Standardize Predictors
# ------------------------------------------------------------------------------

X_means <- colMeans(
  X
)


X_sds <- apply(
  X,
  2,
  sd
)


X_scaled <- scale(
  X,
  center = X_means,
  scale = X_sds
)


X_scaled <- as.matrix(
  X_scaled
)


y_mean <- mean(
  y
)


y_centered <- y -
  y_mean


# ------------------------------------------------------------------------------
# 4. Ordinary Least Squares
# ------------------------------------------------------------------------------

beta_OLS <- solve(
  crossprod(
    X_scaled
  ),
  crossprod(
    X_scaled,
    y_centered
  )
)


beta_OLS <- as.vector(
  beta_OLS
)


# ------------------------------------------------------------------------------
# 5. General Quadratic-Regularization Solver
# ------------------------------------------------------------------------------

# Objective:
#
# ||y - X beta||^2
# +
# lambda * beta' Omega beta
#
#
# First-order condition:
#
# (X'X + lambda Omega) beta = X'y
#
#
# Therefore:
#
# beta_hat =
#
# (X'X + lambda Omega)^(-1) X'y


quadratic_fit <- function(
    X,
    y,
    lambda,
    Omega
) {
  
  X <- as.matrix(
    X
  )
  
  
  Omega <- as.matrix(
    Omega
  )
  
  
  p <- ncol(
    X
  )
  
  
  if (
    nrow(
      Omega
    ) != p ||
    ncol(
      Omega
    ) != p
  ) {
    
    stop(
      "Omega must be a p x p penalty matrix."
    )
  }
  
  
  beta_hat <- solve(
    crossprod(
      X
    ) +
      lambda *
      Omega,
    crossprod(
      X,
      y
    )
  )
  
  
  fitted <- as.vector(
    X %*%
      beta_hat
  )
  
  
  residuals <- y -
    fitted
  
  
  return(
    list(
      coefficients = as.vector(
        beta_hat
      ),
      fitted = fitted,
      residuals = residuals,
      lambda = lambda,
      Omega = Omega
    )
  )
}


# ------------------------------------------------------------------------------
# 6. Ridge Penalty Matrix
# ------------------------------------------------------------------------------

# Ridge:
#
# beta' beta
#
# =
#
# beta' I beta


Omega_ridge <- diag(
  p
)


# ------------------------------------------------------------------------------
# 7. Fit Ridge Through General Quadratic Solver
# ------------------------------------------------------------------------------

lambda <- 20


ridge_model <- quadratic_fit(
  X = X_scaled,
  y = y_centered,
  lambda = lambda,
  Omega = Omega_ridge
)


ridge_model$coefficients


# ------------------------------------------------------------------------------
# 8. Verify Ridge Formula
# ------------------------------------------------------------------------------

ridge_direct <- solve(
  crossprod(
    X_scaled
  ) +
    lambda *
    diag(
      p
    ),
  crossprod(
    X_scaled,
    y_centered
  )
)


max(
  abs(
    ridge_direct -
      ridge_model$coefficients
  )
)


# ------------------------------------------------------------------------------
# 9. First-Difference Penalty
# ------------------------------------------------------------------------------

# Instead of penalizing coefficient magnitude, penalize changes between
# neighboring coefficients:
#
# sum_j (beta_{j+1} - beta_j)^2
#
#
# Define D1 beta:
#
# beta_2 - beta_1
# beta_3 - beta_2
# ...
#
# Then:
#
# ||D1 beta||^2
#
# =
#
# beta' D1'D1 beta


D1 <- diff(
  diag(
    p
  ),
  differences = 1
)


dim(
  D1
)


Omega_first <- crossprod(
  D1
)


# ------------------------------------------------------------------------------
# 10. Inspect First-Difference Penalty Matrix
# ------------------------------------------------------------------------------

Omega_first


# ------------------------------------------------------------------------------
# 11. Fit First-Difference Regularization
# ------------------------------------------------------------------------------

first_difference_model <- quadratic_fit(
  X = X_scaled,
  y = y_centered,
  lambda = lambda,
  Omega = Omega_first
)


# ------------------------------------------------------------------------------
# 12. Second-Difference Penalty
# ------------------------------------------------------------------------------

# Penalize curvature in the coefficient sequence:
#
# beta_j - 2 beta_{j+1} + beta_{j+2}
#
#
# This encourages the sequence of regression coefficients to change smoothly.


D2 <- diff(
  diag(
    p
  ),
  differences = 2
)


dim(
  D2
)


Omega_second <- crossprod(
  D2
)


# ------------------------------------------------------------------------------
# 13. Fit Second-Difference Regularization
# ------------------------------------------------------------------------------

second_difference_model <- quadratic_fit(
  X = X_scaled,
  y = y_centered,
  lambda = lambda,
  Omega = Omega_second
)


# ------------------------------------------------------------------------------
# 14. Compare Coefficient Estimates
# ------------------------------------------------------------------------------

plot(
  seq_len(p),
  beta_OLS,
  type = "b",
  pch = 1,
  xlab = "Coefficient Index",
  ylab = "Coefficient",
  main = "Quadratic Regularization"
)


lines(
  seq_len(p),
  ridge_model$coefficients,
  type = "b",
  pch = 2,
  lty = 2
)


lines(
  seq_len(p),
  first_difference_model$coefficients,
  type = "b",
  pch = 3,
  lty = 3
)


lines(
  seq_len(p),
  second_difference_model$coefficients,
  type = "b",
  pch = 4,
  lty = 4
)


lines(
  seq_len(p),
  beta_true,
  lwd = 3
)


legend(
  "bottomleft",
  legend = c(
    "True",
    "OLS",
    "Ridge",
    "1st Difference",
    "2nd Difference"
  ),
  lty = c(
    1,
    1,
    2,
    3,
    4
  ),
  pch = c(
    NA,
    1,
    2,
    3,
    4
  ),
  lwd = c(
    3,
    1,
    1,
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 15. Compare Penalty Values
# ------------------------------------------------------------------------------

quadratic_penalty <- function(
    beta,
    Omega
) {
  
  as.numeric(
    t(beta) %*%
      Omega %*%
      beta
  )
}


penalty_comparison <- data.frame(
  Model = c(
    "OLS",
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  
  Ridge_Penalty = c(
    quadratic_penalty(
      beta_OLS,
      Omega_ridge
    ),
    
    quadratic_penalty(
      ridge_model$coefficients,
      Omega_ridge
    ),
    
    quadratic_penalty(
      first_difference_model$coefficients,
      Omega_ridge
    ),
    
    quadratic_penalty(
      second_difference_model$coefficients,
      Omega_ridge
    )
  ),
  
  First_Difference_Penalty = c(
    quadratic_penalty(
      beta_OLS,
      Omega_first
    ),
    
    quadratic_penalty(
      ridge_model$coefficients,
      Omega_first
    ),
    
    quadratic_penalty(
      first_difference_model$coefficients,
      Omega_first
    ),
    
    quadratic_penalty(
      second_difference_model$coefficients,
      Omega_first
    )
  ),
  
  Second_Difference_Penalty = c(
    quadratic_penalty(
      beta_OLS,
      Omega_second
    ),
    
    quadratic_penalty(
      ridge_model$coefficients,
      Omega_second
    ),
    
    quadratic_penalty(
      first_difference_model$coefficients,
      Omega_second
    ),
    
    quadratic_penalty(
      second_difference_model$coefficients,
      Omega_second
    )
  )
)


penalty_comparison


# ------------------------------------------------------------------------------
# 16. Mean Squared Coefficient Error
# ------------------------------------------------------------------------------

# Since this is simulated data, we know the true coefficients.
#
# Because the predictors were standardized, convert the true coefficients
# to the standardized-X scale:
#
# beta_scaled_j = beta_original_j * sd(X_j)


beta_true_scaled <- beta_true *
  X_sds


coefficient_MSE <- data.frame(
  Model = c(
    "OLS",
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  
  MSE = c(
    mean(
      (
        beta_OLS -
          beta_true_scaled
      )^2
    ),
    
    mean(
      (
        ridge_model$coefficients -
          beta_true_scaled
      )^2
    ),
    
    mean(
      (
        first_difference_model$coefficients -
          beta_true_scaled
      )^2
    ),
    
    mean(
      (
        second_difference_model$coefficients -
          beta_true_scaled
      )^2
    )
  )
)


coefficient_MSE


# ------------------------------------------------------------------------------
# 17. Effect of Lambda
# ------------------------------------------------------------------------------

lambda_grid <- 10^seq(
  -4,
  5,
  length.out = 200
)


ridge_path <- matrix(
  NA,
  nrow = length(
    lambda_grid
  ),
  ncol = p
)


first_path <- matrix(
  NA,
  nrow = length(
    lambda_grid
  ),
  ncol = p
)


second_path <- matrix(
  NA,
  nrow = length(
    lambda_grid
  ),
  ncol = p
)


for (i in seq_along(
  lambda_grid
)) {
  
  ridge_path[i, ] <-
    quadratic_fit(
      X_scaled,
      y_centered,
      lambda_grid[i],
      Omega_ridge
    )$coefficients
  
  
  first_path[i, ] <-
    quadratic_fit(
      X_scaled,
      y_centered,
      lambda_grid[i],
      Omega_first
    )$coefficients
  
  
  second_path[i, ] <-
    quadratic_fit(
      X_scaled,
      y_centered,
      lambda_grid[i],
      Omega_second
    )$coefficients
}


# ------------------------------------------------------------------------------
# 18. Ridge Coefficient Path
# ------------------------------------------------------------------------------

matplot(
  log10(
    lambda_grid
  ),
  ridge_path,
  type = "l",
  lty = 1,
  xlab = "log10(lambda)",
  ylab = "Coefficient",
  main = "Ridge Regularization Path"
)


# ------------------------------------------------------------------------------
# 19. First-Difference Coefficient Path
# ------------------------------------------------------------------------------

matplot(
  log10(
    lambda_grid
  ),
  first_path,
  type = "l",
  lty = 1,
  xlab = "log10(lambda)",
  ylab = "Coefficient",
  main = "First-Difference Regularization Path"
)


# ------------------------------------------------------------------------------
# 20. Second-Difference Coefficient Path
# ------------------------------------------------------------------------------

matplot(
  log10(
    lambda_grid
  ),
  second_path,
  type = "l",
  lty = 1,
  xlab = "log10(lambda)",
  ylab = "Coefficient",
  main = "Second-Difference Regularization Path"
)


# ------------------------------------------------------------------------------
# 21. Understand Large-Lambda Limits
# ------------------------------------------------------------------------------

# Ridge:
#
# large lambda -> beta approaches zero.
#
#
# First differences:
#
# large lambda -> neighboring coefficients become approximately equal.
#
#
# Second differences:
#
# large lambda -> coefficient sequence becomes approximately linear.


large_lambda <- 1e8


ridge_large <- quadratic_fit(
  X_scaled,
  y_centered,
  large_lambda,
  Omega_ridge
)


first_large <- quadratic_fit(
  X_scaled,
  y_centered,
  large_lambda,
  Omega_first
)


second_large <- quadratic_fit(
  X_scaled,
  y_centered,
  large_lambda,
  Omega_second
)


# ------------------------------------------------------------------------------
# 22. Plot Large-Lambda Behavior
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    3
  )
)


plot(
  seq_len(p),
  ridge_large$coefficients,
  type = "b",
  pch = 19,
  xlab = "Coefficient Index",
  ylab = "Coefficient",
  main = "Large Lambda: Ridge"
)


plot(
  seq_len(p),
  first_large$coefficients,
  type = "b",
  pch = 19,
  xlab = "Coefficient Index",
  ylab = "Coefficient",
  main = "Large Lambda: 1st Difference"
)


plot(
  seq_len(p),
  second_large$coefficients,
  type = "b",
  pch = 19,
  xlab = "Coefficient Index",
  ylab = "Coefficient",
  main = "Large Lambda: 2nd Difference"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 23. Hat / Smoother Matrix
# ------------------------------------------------------------------------------

# The fitted values are:
#
# y_hat =
#
# X (X'X + lambda Omega)^(-1) X'y
#
#
# Therefore:
#
# H_lambda =
#
# X (X'X + lambda Omega)^(-1) X'


quadratic_hat_matrix <- function(
    X,
    lambda,
    Omega
) {
  
  X <- as.matrix(
    X
  )
  
  
  middle <- solve(
    crossprod(
      X
    ) +
      lambda *
      Omega
  )
  
  
  H <- X %*%
    middle %*%
    t(X)
  
  
  return(
    H
  )
}


# ------------------------------------------------------------------------------
# 24. Effective Degrees of Freedom
# ------------------------------------------------------------------------------

# df(lambda) = trace(H_lambda)


effective_df <- function(
    X,
    lambda,
    Omega
) {
  
  H <- quadratic_hat_matrix(
    X,
    lambda,
    Omega
  )
  
  
  sum(
    diag(
      H
    )
  )
}


# ------------------------------------------------------------------------------
# 25. Effective Degrees of Freedom at lambda = 20
# ------------------------------------------------------------------------------

data.frame(
  Model = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  
  Effective_DF = c(
    effective_df(
      X_scaled,
      lambda,
      Omega_ridge
    ),
    
    effective_df(
      X_scaled,
      lambda,
      Omega_first
    ),
    
    effective_df(
      X_scaled,
      lambda,
      Omega_second
    )
  )
)


# ------------------------------------------------------------------------------
# 26. Degrees-of-Freedom Paths
# ------------------------------------------------------------------------------

ridge_df <- numeric(
  length(
    lambda_grid
  )
)


first_df <- numeric(
  length(
    lambda_grid
  )
)


second_df <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  ridge_df[i] <- effective_df(
    X_scaled,
    lambda_grid[i],
    Omega_ridge
  )
  
  
  first_df[i] <- effective_df(
    X_scaled,
    lambda_grid[i],
    Omega_first
  )
  
  
  second_df[i] <- effective_df(
    X_scaled,
    lambda_grid[i],
    Omega_second
  )
}


# ------------------------------------------------------------------------------
# 27. Plot Effective Degrees of Freedom
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  ridge_df,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "Effective Degrees of Freedom",
  main = "Effective Model Complexity"
)


lines(
  log10(
    lambda_grid
  ),
  first_df,
  lty = 2
)


lines(
  log10(
    lambda_grid
  ),
  second_df,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  lty = c(
    1,
    2,
    3
  )
)


# ------------------------------------------------------------------------------
# 28. Generalized Cross-Validation
# ------------------------------------------------------------------------------

quadratic_gcv <- function(
    X,
    y,
    lambda,
    Omega
) {
  
  model <- quadratic_fit(
    X,
    y,
    lambda,
    Omega
  )
  
  
  RSS <- sum(
    model$residuals^2
  )
  
  
  df <- effective_df(
    X,
    lambda,
    Omega
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
# 29. GCV Paths
# ------------------------------------------------------------------------------

ridge_GCV <- numeric(
  length(
    lambda_grid
  )
)


first_GCV <- numeric(
  length(
    lambda_grid
  )
)


second_GCV <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  ridge_GCV[i] <- quadratic_gcv(
    X_scaled,
    y_centered,
    lambda_grid[i],
    Omega_ridge
  )
  
  
  first_GCV[i] <- quadratic_gcv(
    X_scaled,
    y_centered,
    lambda_grid[i],
    Omega_first
  )
  
  
  second_GCV[i] <- quadratic_gcv(
    X_scaled,
    y_centered,
    lambda_grid[i],
    Omega_second
  )
}


# ------------------------------------------------------------------------------
# 30. Best Lambda by GCV
# ------------------------------------------------------------------------------

best_ridge_lambda <- lambda_grid[
  which.min(
    ridge_GCV
  )
]


best_first_lambda <- lambda_grid[
  which.min(
    first_GCV
  )
]


best_second_lambda <- lambda_grid[
  which.min(
    second_GCV
  )
]


data.frame(
  Penalty = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  Lambda = c(
    best_ridge_lambda,
    best_first_lambda,
    best_second_lambda
  )
)


# ------------------------------------------------------------------------------
# 31. Plot GCV Paths
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  ridge_GCV,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "GCV",
  main = "Generalized Cross-Validation"
)


lines(
  log10(
    lambda_grid
  ),
  first_GCV,
  lty = 2
)


lines(
  log10(
    lambda_grid
  ),
  second_GCV,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  lty = c(
    1,
    2,
    3
  )
)


# ------------------------------------------------------------------------------
# 32. Train-Test Split
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


X_train_raw <- X[
  train_index,
  ,
  drop = FALSE
]


X_test_raw <- X[
  test_index,
  ,
  drop = FALSE
]


y_train <- y[
  train_index
]


y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 33. Standardize Using Training Data Only
# ------------------------------------------------------------------------------

train_means <- colMeans(
  X_train_raw
)


train_sds <- apply(
  X_train_raw,
  2,
  sd
)


X_train <- scale(
  X_train_raw,
  center = train_means,
  scale = train_sds
)


X_test <- scale(
  X_test_raw,
  center = train_means,
  scale = train_sds
)


X_train <- as.matrix(
  X_train
)


X_test <- as.matrix(
  X_test
)


y_train_mean <- mean(
  y_train
)


y_train_centered <- y_train -
  y_train_mean


# ------------------------------------------------------------------------------
# 34. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

quadratic_cv <- function(
    X,
    y,
    lambda_grid,
    Omega,
    k = 5
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  folds <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  CV_MSE <- matrix(
    NA,
    nrow = length(
      lambda_grid
    ),
    ncol = k
  )
  
  
  for (fold in seq_len(
    k
  )) {
    
    train_index <- which(
      folds != fold
    )
    
    
    validation_index <- which(
      folds == fold
    )
    
    
    X_train_raw <- X[
      train_index,
      ,
      drop = FALSE
    ]
    
    
    X_validation_raw <- X[
      validation_index,
      ,
      drop = FALSE
    ]
    
    
    y_train <- y[
      train_index
    ]
    
    
    y_validation <- y[
      validation_index
    ]
    
    
    means <- colMeans(
      X_train_raw
    )
    
    
    sds <- apply(
      X_train_raw,
      2,
      sd
    )
    
    
    X_train <- scale(
      X_train_raw,
      center = means,
      scale = sds
    )
    
    
    X_validation <- scale(
      X_validation_raw,
      center = means,
      scale = sds
    )
    
    
    X_train <- as.matrix(
      X_train
    )
    
    
    X_validation <- as.matrix(
      X_validation
    )
    
    
    y_mean <- mean(
      y_train
    )
    
    
    y_train_centered <- y_train -
      y_mean
    
    
    for (i in seq_along(
      lambda_grid
    )) {
      
      model <- quadratic_fit(
        X_train,
        y_train_centered,
        lambda_grid[i],
        Omega
      )
      
      
      prediction <- y_mean +
        as.vector(
          X_validation %*%
            model$coefficients
        )
      
      
      CV_MSE[
        i,
        fold
      ] <- mean(
        (
          y_validation -
            prediction
        )^2
      )
    }
  }
  
  
  return(
    list(
      Lambda = lambda_grid,
      Fold_MSE = CV_MSE,
      Mean_MSE = rowMeans(
        CV_MSE
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 35. Use Smaller Lambda Grid for CV
# ------------------------------------------------------------------------------

lambda_grid_cv <- 10^seq(
  -4,
  4,
  length.out = 80
)


set.seed(123)


CV_ridge <- quadratic_cv(
  X,
  y,
  lambda_grid_cv,
  Omega_ridge,
  k = 5
)


set.seed(123)


CV_first <- quadratic_cv(
  X,
  y,
  lambda_grid_cv,
  Omega_first,
  k = 5
)


set.seed(123)


CV_second <- quadratic_cv(
  X,
  y,
  lambda_grid_cv,
  Omega_second,
  k = 5
)


# ------------------------------------------------------------------------------
# 36. Select Lambda
# ------------------------------------------------------------------------------

best_ridge_lambda_CV <-
  CV_ridge$Lambda[
    which.min(
      CV_ridge$Mean_MSE
    )
  ]


best_first_lambda_CV <-
  CV_first$Lambda[
    which.min(
      CV_first$Mean_MSE
    )
  ]


best_second_lambda_CV <-
  CV_second$Lambda[
    which.min(
      CV_second$Mean_MSE
    )
  ]


data.frame(
  Penalty = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  Best_Lambda = c(
    best_ridge_lambda_CV,
    best_first_lambda_CV,
    best_second_lambda_CV
  )
)


# ------------------------------------------------------------------------------
# 37. Plot Cross-Validation Curves
# ------------------------------------------------------------------------------

plot(
  log10(
    CV_ridge$Lambda
  ),
  CV_ridge$Mean_MSE,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "Cross-Validated MSE",
  main = "Quadratic Regularization CV"
)


lines(
  log10(
    CV_first$Lambda
  ),
  CV_first$Mean_MSE,
  lty = 2
)


lines(
  log10(
    CV_second$Lambda
  ),
  CV_second$Mean_MSE,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  lty = c(
    1,
    2,
    3
  )
)


# ------------------------------------------------------------------------------
# 38. Fit Final Training Models
# ------------------------------------------------------------------------------

final_ridge <- quadratic_fit(
  X_train,
  y_train_centered,
  best_ridge_lambda_CV,
  Omega_ridge
)


final_first <- quadratic_fit(
  X_train,
  y_train_centered,
  best_first_lambda_CV,
  Omega_first
)


final_second <- quadratic_fit(
  X_train,
  y_train_centered,
  best_second_lambda_CV,
  Omega_second
)


# ------------------------------------------------------------------------------
# 39. Test Predictions
# ------------------------------------------------------------------------------

predict_quadratic <- function(
    X_new,
    model,
    y_mean
) {
  
  y_mean +
    as.vector(
      X_new %*%
        model$coefficients
    )
}


ridge_test_prediction <- predict_quadratic(
  X_test,
  final_ridge,
  y_train_mean
)


first_test_prediction <- predict_quadratic(
  X_test,
  final_first,
  y_train_mean
)


second_test_prediction <- predict_quadratic(
  X_test,
  final_second,
  y_train_mean
)


# ------------------------------------------------------------------------------
# 40. OLS Test Model
# ------------------------------------------------------------------------------

OLS_train <- solve(
  crossprod(
    X_train
  ),
  crossprod(
    X_train,
    y_train_centered
  )
)


OLS_test_prediction <- y_train_mean +
  as.vector(
    X_test %*%
      OLS_train
  )


# ------------------------------------------------------------------------------
# 41. Test Error Comparison
# ------------------------------------------------------------------------------

test_results <- data.frame(
  Model = c(
    "OLS",
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  
  Test_MSE = c(
    mean(
      (
        y_test -
          OLS_test_prediction
      )^2
    ),
    
    mean(
      (
        y_test -
          ridge_test_prediction
      )^2
    ),
    
    mean(
      (
        y_test -
          first_test_prediction
      )^2
    ),
    
    mean(
      (
        y_test -
          second_test_prediction
      )^2
    )
  )
)


test_results$Test_RMSE <- sqrt(
  test_results$Test_MSE
)


test_results


# ------------------------------------------------------------------------------
# 42. Back-Transform Final Coefficients
# ------------------------------------------------------------------------------

ridge_beta_original <-
  final_ridge$coefficients /
  train_sds


first_beta_original <-
  final_first$coefficients /
  train_sds


second_beta_original <-
  final_second$coefficients /
  train_sds


ridge_intercept <-
  y_train_mean -
  sum(
    ridge_beta_original *
      train_means
  )


first_intercept <-
  y_train_mean -
  sum(
    first_beta_original *
      train_means
  )


second_intercept <-
  y_train_mean -
  sum(
    second_beta_original *
      train_means
  )


# ------------------------------------------------------------------------------
# 43. Compare Final Coefficients with Truth
# ------------------------------------------------------------------------------

plot(
  seq_len(p),
  beta_true,
  type = "l",
  lwd = 3,
  xlab = "Coefficient Index",
  ylab = "Coefficient",
  main = "Final Quadratic-Regularized Estimates"
)


lines(
  seq_len(p),
  ridge_beta_original,
  type = "b",
  pch = 1,
  lty = 2
)


lines(
  seq_len(p),
  first_beta_original,
  type = "b",
  pch = 2,
  lty = 3
)


lines(
  seq_len(p),
  second_beta_original,
  type = "b",
  pch = 3,
  lty = 4
)


legend(
  "bottomleft",
  legend = c(
    "True",
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  lty = c(
    1,
    2,
    3,
    4
  ),
  pch = c(
    NA,
    1,
    2,
    3
  ),
  lwd = c(
    3,
    1,
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 44. Eigenvalue Interpretation
# ------------------------------------------------------------------------------

# Because Omega is symmetric positive semidefinite, it can be decomposed:
#
# Omega = V diag(d) V'
#
#
# Directions with large penalty eigenvalues are heavily shrunk.
# Directions in the null space of Omega are not penalized.


ridge_eigen <- eigen(
  Omega_ridge,
  symmetric = TRUE
)


first_eigen <- eigen(
  Omega_first,
  symmetric = TRUE
)


second_eigen <- eigen(
  Omega_second,
  symmetric = TRUE
)


data.frame(
  Penalty = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  Number_Near_Zero_Eigenvalues = c(
    sum(
      ridge_eigen$values <
        1e-10
    ),
    
    sum(
      first_eigen$values <
        1e-10
    ),
    
    sum(
      second_eigen$values <
        1e-10
    )
  )
)


# ------------------------------------------------------------------------------
# 45. Plot Penalty Eigenvalues
# ------------------------------------------------------------------------------

plot(
  ridge_eigen$values,
  type = "b",
  pch = 19,
  xlab = "Eigenvalue Index",
  ylab = "Penalty Eigenvalue",
  main = "Eigenvalues of Quadratic Penalties"
)


lines(
  first_eigen$values,
  type = "b",
  pch = 1,
  lty = 2
)


lines(
  second_eigen$values,
  type = "b",
  pch = 2,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "Ridge",
    "First Difference",
    "Second Difference"
  ),
  lty = c(
    1,
    2,
    3
  ),
  pch = c(
    19,
    1,
    2
  )
)


# ------------------------------------------------------------------------------
# 46. Null-Space Interpretation
# ------------------------------------------------------------------------------

# Ridge:
#
# Omega = I
#
# has no null space.
# Every coefficient direction is penalized.
#
#
# First-difference penalty:
#
# D1 beta = 0
#
# when every beta_j is equal.
#
# Therefore constant coefficient sequences are unpenalized.
#
#
# Second-difference penalty:
#
# D2 beta = 0
#
# when beta_j varies linearly with j.
#
# Therefore linear coefficient trends are unpenalized.


constant_beta <- rep(
  2,
  p
)


linear_beta <- seq(
  -2,
  2,
  length.out = p
)


data.frame(
  Coefficient_Structure = c(
    "Constant",
    "Linear"
  ),
  
  First_Difference_Penalty = c(
    quadratic_penalty(
      constant_beta,
      Omega_first
    ),
    quadratic_penalty(
      linear_beta,
      Omega_first
    )
  ),
  
  Second_Difference_Penalty = c(
    quadratic_penalty(
      constant_beta,
      Omega_second
    ),
    quadratic_penalty(
      linear_beta,
      Omega_second
    )
  )
)


# ------------------------------------------------------------------------------
# 47. Objective Decomposition
# ------------------------------------------------------------------------------

objective_components <- function(
    X,
    y,
    beta,
    lambda,
    Omega
) {
  
  residuals <- y -
    as.vector(
      X %*%
        beta
    )
  
  
  RSS <- sum(
    residuals^2
  )
  
  
  penalty <- lambda *
    quadratic_penalty(
      beta,
      Omega
    )
  
  
  data.frame(
    RSS = RSS,
    Penalty = penalty,
    Objective = RSS +
      penalty
  )
}


objective_components(
  X_scaled,
  y_centered,
  ridge_model$coefficients,
  lambda,
  Omega_ridge
)


objective_components(
  X_scaled,
  y_centered,
  first_difference_model$coefficients,
  lambda,
  Omega_first
)


objective_components(
  X_scaled,
  y_centered,
  second_difference_model$coefficients,
  lambda,
  Omega_second
)


# ------------------------------------------------------------------------------
# 48. Summary
# ------------------------------------------------------------------------------

cat(
  "Best ridge lambda:",
  best_ridge_lambda_CV,
  "\n"
)


cat(
  "Best first-difference lambda:",
  best_first_lambda_CV,
  "\n"
)


cat(
  "Best second-difference lambda:",
  best_second_lambda_CV,
  "\n"
)


cat(
  "\nTest RMSE:\n"
)


print(
  test_results[
    ,
    c(
      "Model",
      "Test_RMSE"
    )
  ]
)


cat(
  "\nPenalty null-space dimensions:\n"
)


cat(
  "Ridge:",
  sum(
    ridge_eigen$values <
      1e-10
  ),
  "\n"
)


cat(
  "First difference:",
  sum(
    first_eigen$values <
      1e-10
  ),
  "\n"
)


cat(
  "Second difference:",
  sum(
    second_eigen$values <
      1e-10
  ),
  "\n"
)