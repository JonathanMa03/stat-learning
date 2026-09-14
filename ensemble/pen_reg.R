# ==============================================================================
# Penalized Regression
# Jonathan Ma
#
# Main ideas:
#   - Ordinary least squares
#   - Regularization
#   - Penalized least squares
#   - Ridge regression
#   - Lasso regression
#   - Elastic Net
#   - General quadratic penalties
#   - Bias-variance tradeoff
#   - Coefficient shrinkage
#   - Sparsity
#   - Correlated predictors
#   - Regularization paths
#   - Cross-validation
#   - One-standard-error rule
#   - Effective degrees of freedom
#   - Prediction vs variable selection
#
#
# General penalized regression objective:
#
#       minimize_beta
#
#       (1 / (2n)) ||y - X beta||_2^2
#
#       +
#
#       lambda * P(beta)
#
#
# Examples:
#
# Ridge:
#
#       P(beta) = (1/2) ||beta||_2^2
#
#
# Lasso:
#
#       P(beta) = ||beta||_1
#
#
# Elastic Net:
#
#       P(beta)
#
#       =
#
#       alpha ||beta||_1
#
#       +
#
#       ((1-alpha)/2) ||beta||_2^2
#
#
# The intercept is NOT penalized.
#
# We therefore center y and standardize X before fitting.
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE CORRELATED REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulation Settings
# ------------------------------------------------------------------------------

set.seed(123)

n <- 300

p <- 20


# ------------------------------------------------------------------------------
# 2. Correlated Predictors
#
# AR(1)-style covariance:
#
#       Cov(X_j, X_k) = rho^|j-k|
# ------------------------------------------------------------------------------

rho <- 0.75


Sigma <- outer(
  seq_len(p),
  seq_len(p),
  function(j, k) {
    
    rho^abs(
      j - k
    )
  }
)


# ------------------------------------------------------------------------------
# 3. Simulate Multivariate Gaussian X
#
# If Z has iid N(0,1) entries and
#
#       Sigma = L L'
#
# then
#
#       X = Z L'
#
# has covariance Sigma.
# ------------------------------------------------------------------------------

L <- chol(
  Sigma
)


Z <- matrix(
  rnorm(
    n * p
  ),
  nrow = n,
  ncol = p
)


X <- Z %*%
  L


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 4. Sparse True Coefficients
#
# Only a subset of variables matter.
# ------------------------------------------------------------------------------

beta_true <- c(
  3.0,
  -2.5,
  2.0,
  0,
  0,
  1.5,
  0,
  0,
  -1.5,
  0,
  0,
  1.0,
  rep(
    0,
    8
  )
)


beta_0_true <- 2


sigma_error <- 2


# ------------------------------------------------------------------------------
# 5. Response
# ------------------------------------------------------------------------------

signal <- beta_0_true +
  as.numeric(
    X %*%
      beta_true
  )


y <- signal +
  rnorm(
    n,
    sd =
      sigma_error
  )


# ==============================================================================
# PART II
#
# EXPLORE CORRELATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Predictor Correlation Matrix
# ------------------------------------------------------------------------------

correlation_matrix <- cor(
  X
)


image(
  correlation_matrix,
  axes = FALSE,
  main = "Predictor Correlation Matrix"
)


# ------------------------------------------------------------------------------
# 7. Condition Number
#
# Large condition numbers indicate instability / multicollinearity.
# ------------------------------------------------------------------------------

eigenvalues_XtX <- eigen(
  t(X) %*%
    X,
  symmetric = TRUE,
  only.values = TRUE
)$values


condition_number <- max(
  eigenvalues_XtX
) /
  min(
    eigenvalues_XtX
  )


condition_number


# ==============================================================================
# PART III
#
# TRAIN / TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Split
# ------------------------------------------------------------------------------

set.seed(456)


train_indices <- sample(
  seq_len(n),
  size =
    floor(
      0.70 *
        n
    )
)


test_indices <- setdiff(
  seq_len(n),
  train_indices
)


X_train_raw <- X[
  train_indices,
  ,
  drop = FALSE
]


y_train_raw <- y[
  train_indices
]


X_test_raw <- X[
  test_indices,
  ,
  drop = FALSE
]


y_test <- y[
  test_indices
]


# ==============================================================================
# PART IV
#
# STANDARDIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Training Means and Standard Deviations
#
# IMPORTANT:
#
# Standardization is estimated using the TRAINING SET ONLY.
# ------------------------------------------------------------------------------

x_means <- colMeans(
  X_train_raw
)


x_sds <- apply(
  X_train_raw,
  2,
  sd
)


y_mean <- mean(
  y_train_raw
)


# ------------------------------------------------------------------------------
# 10. Standardize Predictors
# ------------------------------------------------------------------------------

X_train <- sweep(
  X_train_raw,
  2,
  x_means,
  "-"
)


X_train <- sweep(
  X_train,
  2,
  x_sds,
  "/"
)


X_test <- sweep(
  X_test_raw,
  2,
  x_means,
  "-"
)


X_test <- sweep(
  X_test,
  2,
  x_sds,
  "/"
)


# ------------------------------------------------------------------------------
# 11. Center Response
# ------------------------------------------------------------------------------

y_train <- y_train_raw -
  y_mean


n_train <- nrow(
  X_train
)


# ==============================================================================
# PART V
#
# ORDINARY LEAST SQUARES
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. OLS
#
# Because p < n:
#
#       beta_hat = (X'X)^(-1) X'y
#
# solve(A,b) is preferable to explicitly computing inverse(A).
# ------------------------------------------------------------------------------

ols_beta <- solve(
  t(X_train) %*%
    X_train,
  t(X_train) %*%
    y_train
)


# ------------------------------------------------------------------------------
# 13. Prediction
# ------------------------------------------------------------------------------

predict_standardized_linear <- function(
    X,
    beta,
    y_mean
) {
  
  y_mean +
    as.numeric(
      X %*%
        beta
    )
}


ols_test_prediction <- predict_standardized_linear(
  X_test,
  ols_beta,
  y_mean
)


# ==============================================================================
# PART VI
#
# METRICS
# ==============================================================================


regression_metrics <- function(
    y,
    prediction
) {
  
  mse <- mean(
    (
      y -
        prediction
    )^2
  )
  
  
  rmse <- sqrt(
    mse
  )
  
  
  mae <- mean(
    abs(
      y -
        prediction
    )
  )
  
  
  r_squared <- 1 -
    sum(
      (
        y -
          prediction
      )^2
    ) /
    sum(
      (
        y -
          mean(y)
      )^2
    )
  
  
  c(
    MSE =
      mse,
    RMSE =
      rmse,
    MAE =
      mae,
    R2 =
      r_squared
  )
}


regression_metrics(
  y_test,
  ols_test_prediction
)


# ==============================================================================
# PART VII
#
# GENERAL QUADRATIC PENALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Quadratically Penalized Regression
#
# Objective:
#
#       (1 / (2n)) ||y-X beta||^2
#
#       +
#
#       (lambda / 2) beta' Omega beta
#
#
# Solution:
#
#       beta_hat
#
#       =
#
#       (X'X + n lambda Omega)^(-1) X'y
# ------------------------------------------------------------------------------

quadratic_penalty_fit <- function(
    X,
    y,
    lambda,
    Omega = NULL
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  if (
    is.null(
      Omega
    )
  ) {
    
    Omega <- diag(
      p
    )
  }
  
  
  beta <- solve(
    t(X) %*%
      X +
      n *
      lambda *
      Omega,
    t(X) %*%
      y
  )
  
  
  as.numeric(
    beta
  )
}


# ==============================================================================
# PART VIII
#
# RIDGE REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Ridge
#
# Omega = I
# ------------------------------------------------------------------------------

ridge_fit <- function(
    X,
    y,
    lambda
) {
  
  quadratic_penalty_fit(
    X,
    y,
    lambda =
      lambda,
    Omega =
      diag(
        ncol(X)
      )
  )
}


# ------------------------------------------------------------------------------
# 16. Example
# ------------------------------------------------------------------------------

ridge_beta <- ridge_fit(
  X_train,
  y_train,
  lambda = 0.10
)


# ==============================================================================
# PART IX
#
# SOFT THRESHOLDING
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Lasso Building Block
#
#       S(z, gamma)
#
#       =
#
#       sign(z) max(|z|-gamma, 0)
# ------------------------------------------------------------------------------

soft_threshold <- function(
    z,
    gamma
) {
  
  sign(z) *
    pmax(
      abs(z) -
        gamma,
      0
    )
}


# ------------------------------------------------------------------------------
# 18. Plot Soft Thresholding
# ------------------------------------------------------------------------------

z_grid <- seq(
  -4,
  4,
  length.out = 400
)


plot(
  z_grid,
  soft_threshold(
    z_grid,
    1
  ),
  type = "l",
  lwd = 2,
  xlab = "z",
  ylab = "S(z, 1)",
  main = "Soft Thresholding"
)


abline(
  0,
  1,
  lty = 2
)


# ==============================================================================
# PART X
#
# ELASTIC NET COORDINATE DESCENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Objective
#
#       (1/(2n)) ||y-X beta||^2
#
#       +
#
#       lambda [
#
#           alpha ||beta||_1
#
#           +
#
#           (1-alpha)/2 ||beta||_2^2
#
#       ]
#
#
# alpha = 1:
#       Lasso
#
# alpha = 0:
#       Ridge
#
# 0 < alpha < 1:
#       Elastic Net
# ------------------------------------------------------------------------------

elastic_net_fit <- function(
    X,
    y,
    lambda,
    alpha = 1,
    beta_initial = NULL,
    tolerance = 1e-8,
    maximum_iterations = 10000
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  if (
    is.null(
      beta_initial
    )
  ) {
    
    beta <- numeric(
      p
    )
    
  } else {
    
    beta <- beta_initial
  }
  
  
  # --------------------------------------------------------------------------
  # Mean squared value for each standardized column.
  #
  # These will be near (n-1)/n because base R sd() uses n-1.
  # --------------------------------------------------------------------------
  
  column_squared_mean <- colSums(
    X^2
  ) /
    n
  
  
  for (
    iteration in seq_len(
      maximum_iterations
    )
  ) {
    
    beta_old <- beta
    
    
    for (
      j in seq_len(
        p
      )
    ) {
      
      # ----------------------------------------------------------------------
      # Partial residual excluding predictor j
      # ----------------------------------------------------------------------
      
      residual_j <- y -
        as.numeric(
          X %*%
            beta
        ) +
        X[
          ,
          j
        ] *
        beta[j]
      
      
      # ----------------------------------------------------------------------
      # Coordinate-wise correlation
      # ----------------------------------------------------------------------
      
      rho_j <- sum(
        X[
          ,
          j
        ] *
          residual_j
      ) /
        n
      
      
      # ----------------------------------------------------------------------
      # Elastic Net update
      # ----------------------------------------------------------------------
      
      beta[j] <- soft_threshold(
        rho_j,
        lambda *
          alpha
      ) /
        (
          column_squared_mean[j] +
            lambda *
            (
              1 -
                alpha
            )
        )
    }
    
    
    if (
      max(
        abs(
          beta -
          beta_old
        )
      ) <
      tolerance
    ) {
      
      break
    }
  }
  
  
  list(
    beta =
      beta,
    iterations =
      iteration,
    converged =
      iteration <
      maximum_iterations
  )
}


# ==============================================================================
# PART XI
#
# LASSO
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Lasso is alpha = 1
# ------------------------------------------------------------------------------

lasso_fit <- function(
    X,
    y,
    lambda,
    beta_initial = NULL
) {
  
  elastic_net_fit(
    X,
    y,
    lambda =
      lambda,
    alpha = 1,
    beta_initial =
      beta_initial
  )
}


# ------------------------------------------------------------------------------
# 21. Example
# ------------------------------------------------------------------------------

lasso_example <- lasso_fit(
  X_train,
  y_train,
  lambda = 0.10
)


lasso_example$beta


# ==============================================================================
# PART XII
#
# ELASTIC NET
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Example
# ------------------------------------------------------------------------------

elastic_example <- elastic_net_fit(
  X_train,
  y_train,
  lambda = 0.10,
  alpha = 0.50
)


elastic_example$beta


# ==============================================================================
# PART XIII
#
# LAMBDA MAX
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Largest Lambda Producing All-Zero Lasso Coefficients
#
# For standardized X:
#
#       lambda_max
#
#       =
#
#       max_j |X_j' y| / n
# ------------------------------------------------------------------------------

lambda_max <- max(
  abs(
    as.numeric(
      t(X_train) %*%
        y_train
    )
  )
) /
  n_train


lambda_max


# ==============================================================================
# PART XIV
#
# REGULARIZATION GRID
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Lambda Grid
#
# Use descending lambda values for warm starts.
# ------------------------------------------------------------------------------

lambda_grid <- exp(
  seq(
    log(
      lambda_max
    ),
    log(
      lambda_max *
        0.001
    ),
    length.out = 100
  )
)


# ==============================================================================
# PART XV
#
# LASSO REGULARIZATION PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Fit Lasso Path
# ------------------------------------------------------------------------------

lasso_path <- matrix(
  0,
  nrow =
    length(
      lambda_grid
    ),
  ncol = p
)


beta_previous <- numeric(
  p
)


for (
  i in seq_along(
    lambda_grid
  )
) {
  
  fit_i <- elastic_net_fit(
    X_train,
    y_train,
    lambda =
      lambda_grid[i],
    alpha = 1,
    beta_initial =
      beta_previous
  )
  
  
  beta_previous <- fit_i$beta
  
  
  lasso_path[
    i,
  ] <- beta_previous
}


colnames(
  lasso_path
) <- colnames(
  X_train
)


# ------------------------------------------------------------------------------
# 26. Plot Lasso Path
# ------------------------------------------------------------------------------

matplot(
  log(
    lambda_grid
  ),
  lasso_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Coefficient",
  main = "Lasso Regularization Path"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART XVI
#
# RIDGE REGULARIZATION PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Ridge Fits
# ------------------------------------------------------------------------------

ridge_path <- matrix(
  0,
  nrow =
    length(
      lambda_grid
    ),
  ncol = p
)


for (
  i in seq_along(
    lambda_grid
  )
) {
  
  ridge_path[
    i,
  ] <- ridge_fit(
    X_train,
    y_train,
    lambda =
      lambda_grid[i]
  )
}


colnames(
  ridge_path
) <- colnames(
  X_train
)


# ------------------------------------------------------------------------------
# 28. Plot
# ------------------------------------------------------------------------------

matplot(
  log(
    lambda_grid
  ),
  ridge_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Coefficient",
  main = "Ridge Regularization Path"
)


abline(
  h = 0,
  lty = 2
)


# Ridge coefficients shrink continuously toward zero but generally do not
# become exactly zero.


# ==============================================================================
# PART XVII
#
# ELASTIC NET PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. alpha = 0.5
# ------------------------------------------------------------------------------

elastic_path <- matrix(
  0,
  nrow =
    length(
      lambda_grid
    ),
  ncol = p
)


beta_previous <- numeric(
  p
)


for (
  i in seq_along(
    lambda_grid
  )
) {
  
  fit_i <- elastic_net_fit(
    X_train,
    y_train,
    lambda =
      lambda_grid[i],
    alpha = 0.5,
    beta_initial =
      beta_previous
  )
  
  
  beta_previous <- fit_i$beta
  
  
  elastic_path[
    i,
  ] <- beta_previous
}


# ------------------------------------------------------------------------------
# 30. Plot
# ------------------------------------------------------------------------------

matplot(
  log(
    lambda_grid
  ),
  elastic_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Coefficient",
  main = "Elastic Net Regularization Path"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART XVIII
#
# COEFFICIENT NORM PATHS
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Ridge L2 Norm
# ------------------------------------------------------------------------------

ridge_l2_norm <- apply(
  ridge_path,
  1,
  function(beta) {
    
    sqrt(
      sum(
        beta^2
      )
    )
  }
)


# ------------------------------------------------------------------------------
# 32. Lasso L1 Norm
# ------------------------------------------------------------------------------

lasso_l1_norm <- apply(
  lasso_path,
  1,
  function(beta) {
    
    sum(
      abs(beta)
    )
  }
)


# ------------------------------------------------------------------------------
# 33. Plot
# ------------------------------------------------------------------------------

plot(
  log(
    lambda_grid
  ),
  ridge_l2_norm,
  type = "l",
  lwd = 2,
  xlab = "log(lambda)",
  ylab = "L2 Norm",
  main = "Ridge Coefficient Norm"
)


plot(
  log(
    lambda_grid
  ),
  lasso_l1_norm,
  type = "l",
  lwd = 2,
  xlab = "log(lambda)",
  ylab = "L1 Norm",
  main = "Lasso Coefficient Norm"
)


# ==============================================================================
# PART XIX
#
# LASSO SPARSITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Number of Nonzero Coefficients
# ------------------------------------------------------------------------------

number_nonzero <- apply(
  lasso_path,
  1,
  function(beta) {
    
    sum(
      abs(beta) >
        1e-8
    )
  }
)


plot(
  log(
    lambda_grid
  ),
  number_nonzero,
  type = "s",
  xlab = "log(lambda)",
  ylab = "Number of Nonzero Coefficients",
  main = "Lasso Sparsity Path"
)


# ==============================================================================
# PART XX
#
# MANUAL K-FOLD CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Create Fold Assignments
# ------------------------------------------------------------------------------

make_folds <- function(
    n,
    number_folds = 10,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  sample(
    rep(
      seq_len(
        number_folds
      ),
      length.out = n
    )
  )
}


# ------------------------------------------------------------------------------
# 36. Penalized Regression CV
#
# alpha:
#
#       0     -> ridge
#       1     -> lasso
#       0-1   -> elastic net
# ------------------------------------------------------------------------------

cross_validate_penalty <- function(
    X_raw,
    y,
    lambda_grid,
    alpha,
    number_folds = 10,
    seed = 123
) {
  
  X_raw <- as.matrix(
    X_raw
  )
  
  
  n <- nrow(
    X_raw
  )
  
  
  folds <- make_folds(
    n,
    number_folds =
      number_folds,
    seed =
      seed
  )
  
  
  fold_mse <- matrix(
    NA_real_,
    nrow =
      number_folds,
    ncol =
      length(
        lambda_grid
      )
  )
  
  
  for (
    fold in seq_len(
      number_folds
    )
  ) {
    
    validation_indices <- which(
      folds ==
        fold
    )
    
    
    training_indices <- which(
      folds !=
        fold
    )
    
    
    X_fold_train_raw <- X_raw[
      training_indices,
      ,
      drop = FALSE
    ]
    
    
    X_fold_validation_raw <- X_raw[
      validation_indices,
      ,
      drop = FALSE
    ]
    
    
    y_fold_train_raw <- y[
      training_indices
    ]
    
    
    y_fold_validation <- y[
      validation_indices
    ]
    
    
    # ------------------------------------------------------------------------
    # Fold-specific preprocessing
    # ------------------------------------------------------------------------
    
    fold_means <- colMeans(
      X_fold_train_raw
    )
    
    
    fold_sds <- apply(
      X_fold_train_raw,
      2,
      sd
    )
    
    
    # Guard against constant columns.
    
    fold_sds[
      fold_sds <
        1e-12
    ] <- 1
    
    
    fold_y_mean <- mean(
      y_fold_train_raw
    )
    
    
    X_fold_train <- sweep(
      X_fold_train_raw,
      2,
      fold_means,
      "-"
    )
    
    
    X_fold_train <- sweep(
      X_fold_train,
      2,
      fold_sds,
      "/"
    )
    
    
    X_fold_validation <- sweep(
      X_fold_validation_raw,
      2,
      fold_means,
      "-"
    )
    
    
    X_fold_validation <- sweep(
      X_fold_validation,
      2,
      fold_sds,
      "/"
    )
    
    
    y_fold_train <- y_fold_train_raw -
      fold_y_mean
    
    
    # ------------------------------------------------------------------------
    # Warm-start path
    # ------------------------------------------------------------------------
    
    beta_previous <- numeric(
      ncol(
        X_raw
      )
    )
    
    
    for (
      lambda_index in seq_along(
        lambda_grid
      )
    ) {
      
      fit <- elastic_net_fit(
        X_fold_train,
        y_fold_train,
        lambda =
          lambda_grid[
            lambda_index
          ],
        alpha =
          alpha,
        beta_initial =
          beta_previous
      )
      
      
      beta_previous <- fit$beta
      
      
      prediction <- fold_y_mean +
        as.numeric(
          X_fold_validation %*%
            fit$beta
        )
      
      
      fold_mse[
        fold,
        lambda_index
      ] <- mean(
        (
          y_fold_validation -
            prediction
        )^2
      )
    }
  }
  
  
  mean_mse <- colMeans(
    fold_mse
  )
  
  
  standard_error <- apply(
    fold_mse,
    2,
    sd
  ) /
    sqrt(
      number_folds
    )
  
  
  list(
    fold_mse =
      fold_mse,
    mean_mse =
      mean_mse,
    standard_error =
      standard_error,
    lambda_grid =
      lambda_grid
  )
}


# ==============================================================================
# PART XXI
#
# CROSS-VALIDATE RIDGE, LASSO, AND ELASTIC NET
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Ridge
# ------------------------------------------------------------------------------

cv_ridge <- cross_validate_penalty(
  X_train_raw,
  y_train_raw,
  lambda_grid =
    lambda_grid,
  alpha = 0,
  number_folds = 10,
  seed = 123
)


# ------------------------------------------------------------------------------
# 38. Lasso
# ------------------------------------------------------------------------------

cv_lasso <- cross_validate_penalty(
  X_train_raw,
  y_train_raw,
  lambda_grid =
    lambda_grid,
  alpha = 1,
  number_folds = 10,
  seed = 123
)


# ------------------------------------------------------------------------------
# 39. Elastic Net
# ------------------------------------------------------------------------------

cv_elastic <- cross_validate_penalty(
  X_train_raw,
  y_train_raw,
  lambda_grid =
    lambda_grid,
  alpha = 0.5,
  number_folds = 10,
  seed = 123
)


# ==============================================================================
# PART XXII
#
# MINIMUM-CV LAMBDA
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Helper
# ------------------------------------------------------------------------------

select_lambda_min <- function(
    cv_result
) {
  
  index <- which.min(
    cv_result$mean_mse
  )
  
  
  list(
    index =
      index,
    lambda =
      cv_result$lambda_grid[
        index
      ],
    mse =
      cv_result$mean_mse[
        index
      ]
  )
}


ridge_min <- select_lambda_min(
  cv_ridge
)


lasso_min <- select_lambda_min(
  cv_lasso
)


elastic_min <- select_lambda_min(
  cv_elastic
)


ridge_min


lasso_min


elastic_min


# ==============================================================================
# PART XXIII
#
# ONE-STANDARD-ERROR RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. One-SE Rule
#
# Choose the most strongly regularized model whose CV error lies within
# one standard error of the minimum.
#
# lambda_grid is ordered from LARGE to SMALL.
# Therefore the first eligible lambda is the strongest regularization.
# ------------------------------------------------------------------------------

select_lambda_one_se <- function(
    cv_result
) {
  
  minimum_index <- which.min(
    cv_result$mean_mse
  )
  
  
  threshold <- cv_result$mean_mse[
    minimum_index
  ] +
    cv_result$standard_error[
      minimum_index
    ]
  
  
  eligible <- which(
    cv_result$mean_mse <=
      threshold
  )
  
  
  selected_index <- min(
    eligible
  )
  
  
  list(
    index =
      selected_index,
    lambda =
      cv_result$lambda_grid[
        selected_index
      ],
    mse =
      cv_result$mean_mse[
        selected_index
      ],
    threshold =
      threshold
  )
}


ridge_one_se <- select_lambda_one_se(
  cv_ridge
)


lasso_one_se <- select_lambda_one_se(
  cv_lasso
)


elastic_one_se <- select_lambda_one_se(
  cv_elastic
)


lasso_one_se


# ==============================================================================
# PART XXIV
#
# CV CURVES
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Plot Helper
# ------------------------------------------------------------------------------

plot_cv_curve <- function(
    cv_result,
    title
) {
  
  plot(
    log(
      cv_result$lambda_grid
    ),
    cv_result$mean_mse,
    type = "l",
    lwd = 2,
    xlab = "log(lambda)",
    ylab = "Cross-Validated MSE",
    main = title
  )
  
  
  upper <- cv_result$mean_mse +
    cv_result$standard_error
  
  
  lower <- cv_result$mean_mse -
    cv_result$standard_error
  
  
  lines(
    log(
      cv_result$lambda_grid
    ),
    upper,
    lty = 2
  )
  
  
  lines(
    log(
      cv_result$lambda_grid
    ),
    lower,
    lty = 2
  )
}


plot_cv_curve(
  cv_ridge,
  "Ridge Cross-Validation"
)


plot_cv_curve(
  cv_lasso,
  "Lasso Cross-Validation"
)


plot_cv_curve(
  cv_elastic,
  "Elastic Net Cross-Validation"
)


# ==============================================================================
# PART XXV
#
# FINAL MODELS
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Ridge
# ------------------------------------------------------------------------------

beta_ridge_final <- ridge_fit(
  X_train,
  y_train,
  lambda =
    ridge_min$lambda
)


# ------------------------------------------------------------------------------
# 44. Lasso
# ------------------------------------------------------------------------------

beta_lasso_final <- elastic_net_fit(
  X_train,
  y_train,
  lambda =
    lasso_min$lambda,
  alpha = 1
)$beta


# ------------------------------------------------------------------------------
# 45. Elastic Net
# ------------------------------------------------------------------------------

beta_elastic_final <- elastic_net_fit(
  X_train,
  y_train,
  lambda =
    elastic_min$lambda,
  alpha = 0.5
)$beta


# ==============================================================================
# PART XXVI
#
# TEST SET PREDICTIONS
# ==============================================================================


ridge_test_prediction <- predict_standardized_linear(
  X_test,
  beta_ridge_final,
  y_mean
)


lasso_test_prediction <- predict_standardized_linear(
  X_test,
  beta_lasso_final,
  y_mean
)


elastic_test_prediction <- predict_standardized_linear(
  X_test,
  beta_elastic_final,
  y_mean
)


# ==============================================================================
# PART XXVII
#
# PERFORMANCE COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Test Metrics
# ------------------------------------------------------------------------------

method_names <- c(
  "OLS",
  "Ridge",
  "Lasso",
  "Elastic Net"
)


predictions <- list(
  ols_test_prediction,
  ridge_test_prediction,
  lasso_test_prediction,
  elastic_test_prediction
)


performance <- data.frame(
  Method =
    method_names,
  MSE =
    NA_real_,
  RMSE =
    NA_real_,
  MAE =
    NA_real_,
  R2 =
    NA_real_
)


for (
  i in seq_along(
    predictions
  )
) {
  
  metrics <- regression_metrics(
    y_test,
    predictions[[i]]
  )
  
  
  performance$MSE[i] <-
    metrics["MSE"]
  
  
  performance$RMSE[i] <-
    metrics["RMSE"]
  
  
  performance$MAE[i] <-
    metrics["MAE"]
  
  
  performance$R2[i] <-
    metrics["R2"]
}


performance


# ==============================================================================
# PART XXVIII
#
# BACK-TRANSFORM COEFFICIENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Convert Standardized Coefficients Back to Original Units
#
# If:
#
#       x*_j = (x_j - mean_j) / sd_j
#
# and
#
#       y_hat = y_bar + sum beta*_j x*_j
#
# then:
#
#       beta_j = beta*_j / sd_j
#
# and:
#
#       beta_0 = y_bar - sum beta_j mean_j
# ------------------------------------------------------------------------------

back_transform_coefficients <- function(
    beta_standardized,
    x_means,
    x_sds,
    y_mean
) {
  
  beta_original <- beta_standardized /
    x_sds
  
  
  intercept <- y_mean -
    sum(
      beta_original *
        x_means
    )
  
  
  c(
    Intercept =
      intercept,
    beta_original
  )
}


ridge_original <- back_transform_coefficients(
  beta_ridge_final,
  x_means,
  x_sds,
  y_mean
)


lasso_original <- back_transform_coefficients(
  beta_lasso_final,
  x_means,
  x_sds,
  y_mean
)


elastic_original <- back_transform_coefficients(
  beta_elastic_final,
  x_means,
  x_sds,
  y_mean
)


# ==============================================================================
# PART XXIX
#
# COEFFICIENT COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. OLS Back-Transformation
# ------------------------------------------------------------------------------

ols_original <- back_transform_coefficients(
  ols_beta,
  x_means,
  x_sds,
  y_mean
)


# ------------------------------------------------------------------------------
# 49. Table
# ------------------------------------------------------------------------------

coefficient_comparison <- data.frame(
  Variable = c(
    "Intercept",
    colnames(X)
  ),
  True = c(
    beta_0_true,
    beta_true
  ),
  OLS =
    ols_original,
  Ridge =
    ridge_original,
  Lasso =
    lasso_original,
  Elastic_Net =
    elastic_original
)


round(
  coefficient_comparison,
  3
)


# ==============================================================================
# PART XXX
#
# VISUALIZE COEFFICIENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Exclude Intercept
# ------------------------------------------------------------------------------

coefficient_matrix <- rbind(
  True =
    beta_true,
  OLS =
    ols_original[-1],
  Ridge =
    ridge_original[-1],
  Lasso =
    lasso_original[-1],
  ElasticNet =
    elastic_original[-1]
)


matplot(
  t(
    coefficient_matrix
  ),
  type = "b",
  pch = 19,
  lty = 1,
  xlab = "Predictor",
  ylab = "Coefficient",
  xaxt = "n",
  main = "Coefficient Estimates"
)


axis(
  1,
  at = seq_len(p),
  labels =
    colnames(X),
  las = 2
)


legend(
  "topright",
  legend =
    rownames(
      coefficient_matrix
    ),
  lty = 1,
  pch = 19,
  cex = 0.7
)


# ==============================================================================
# PART XXXI
#
# VARIABLE SELECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Lasso Selected Variables
# ------------------------------------------------------------------------------

lasso_selected <- which(
  abs(
    beta_lasso_final
  ) >
    1e-8
)


colnames(
  X_train
)[
  lasso_selected
]


# ------------------------------------------------------------------------------
# 52. Elastic Net Selected Variables
# ------------------------------------------------------------------------------

elastic_selected <- which(
  abs(
    beta_elastic_final
  ) >
    1e-8
)


colnames(
  X_train
)[
  elastic_selected
]


# ------------------------------------------------------------------------------
# 53. True Variables
# ------------------------------------------------------------------------------

true_selected <- which(
  beta_true !=
    0
)


colnames(
  X
)[
  true_selected
]


# ==============================================================================
# PART XXXII
#
# SELECTION METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Selection Evaluation
# ------------------------------------------------------------------------------

selection_metrics <- function(
    selected,
    truth,
    p
) {
  
  selected_indicator <- seq_len(p) %in%
    selected
  
  
  truth_indicator <- seq_len(p) %in%
    truth
  
  
  TP <- sum(
    selected_indicator &
      truth_indicator
  )
  
  
  FP <- sum(
    selected_indicator &
      !truth_indicator
  )
  
  
  TN <- sum(
    !selected_indicator &
      !truth_indicator
  )
  
  
  FN <- sum(
    !selected_indicator &
      truth_indicator
  )
  
  
  precision <- if (
    TP + FP >
    0
  ) {
    
    TP /
      (
        TP +
          FP
      )
    
  } else {
    
    NA_real_
  }
  
  
  recall <- TP /
    (
      TP +
        FN
    )
  
  
  c(
    TP = TP,
    FP = FP,
    TN = TN,
    FN = FN,
    Precision =
      precision,
    Recall =
      recall
  )
}


selection_metrics(
  lasso_selected,
  true_selected,
  p
)


selection_metrics(
  elastic_selected,
  true_selected,
  p
)


# ==============================================================================
# PART XXXIII
#
# ONE-SE MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. More Regularized Lasso
# ------------------------------------------------------------------------------

beta_lasso_one_se <- elastic_net_fit(
  X_train,
  y_train,
  lambda =
    lasso_one_se$lambda,
  alpha = 1
)$beta


sum(
  abs(
    beta_lasso_final
  ) >
    1e-8
)


sum(
  abs(
    beta_lasso_one_se
  ) >
    1e-8
)


# ------------------------------------------------------------------------------
# 56. Test Performance
# ------------------------------------------------------------------------------

lasso_one_se_prediction <- predict_standardized_linear(
  X_test,
  beta_lasso_one_se,
  y_mean
)


data.frame(
  Model = c(
    "Minimum CV Error",
    "One-SE Rule"
  ),
  Lambda = c(
    lasso_min$lambda,
    lasso_one_se$lambda
  ),
  Nonzero = c(
    sum(
      abs(
        beta_lasso_final
      ) >
        1e-8
    ),
    sum(
      abs(
        beta_lasso_one_se
      ) >
        1e-8
    )
  ),
  Test_MSE = c(
    mean(
      (
        y_test -
          lasso_test_prediction
      )^2
    ),
    mean(
      (
        y_test -
          lasso_one_se_prediction
      )^2
    )
  )
)


# ==============================================================================
# PART XXXIV
#
# RIDGE EFFECTIVE DEGREES OF FREEDOM
# ==============================================================================


# ------------------------------------------------------------------------------
# 57. Ridge Smoother Matrix
#
#       H_lambda
#
#       =
#
#       X (X'X + n lambda I)^(-1) X'
#
#
# Effective degrees of freedom:
#
#       df(lambda) = trace(H_lambda)
# ------------------------------------------------------------------------------

ridge_effective_df <- function(
    X,
    lambda
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  H <- X %*%
    solve(
      t(X) %*%
        X +
        n *
        lambda *
        diag(
          p
        ),
      t(X)
    )
  
  
  sum(
    diag(
      H
    )
  )
}


# ------------------------------------------------------------------------------
# 58. Degrees of Freedom Path
# ------------------------------------------------------------------------------

ridge_df <- sapply(
  lambda_grid,
  function(lambda) {
    
    ridge_effective_df(
      X_train,
      lambda
    )
  }
)


plot(
  log(
    lambda_grid
  ),
  ridge_df,
  type = "l",
  lwd = 2,
  xlab = "log(lambda)",
  ylab = "Effective Degrees of Freedom",
  main = "Ridge Model Complexity"
)


# Larger lambda -> smaller effective model complexity.


# ==============================================================================
# PART XXXV
#
# SVD VIEW OF RIDGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 59. Singular Value Decomposition
#
#       X = U D V'
#
# Ridge:
#
#       beta_ridge
#
#       =
#
#       V diag(
#
#           d_j / (d_j^2 + n lambda)
#
#       ) U' y
#
#
# In fitted-value space, principal directions are shrunk by:
#
#       d_j^2 / (d_j^2 + n lambda)
# ------------------------------------------------------------------------------

svd_X <- svd(
  X_train
)


ridge_lambda_demo <- ridge_min$lambda


ridge_shrinkage_factors <- (
  svd_X$d^2
) /
  (
    svd_X$d^2 +
      n_train *
      ridge_lambda_demo
  )


data.frame(
  Direction =
    seq_along(
      svd_X$d
    ),
  Singular_Value =
    svd_X$d,
  Ridge_Shrinkage =
    ridge_shrinkage_factors
)


# ------------------------------------------------------------------------------
# 60. Plot
# ------------------------------------------------------------------------------

plot(
  svd_X$d,
  ridge_shrinkage_factors,
  pch = 19,
  xlab = "Singular Value",
  ylab = "Ridge Shrinkage Factor",
  main = "Ridge Shrinkage by Principal Direction"
)


# Weak / poorly identified directions receive more shrinkage.


# ==============================================================================
# PART XXXVI
#
# BIAS-VARIANCE SIMULATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 61. Fixed Design Repeated Responses
#
# Hold X fixed.
#
# Generate many response vectors with new noise.
#
# Compare variability of OLS and ridge coefficients.
# ------------------------------------------------------------------------------

set.seed(777)


number_simulations <- 300


ols_simulation <- matrix(
  NA_real_,
  nrow =
    number_simulations,
  ncol = p
)


ridge_simulation <- matrix(
  NA_real_,
  nrow =
    number_simulations,
  ncol = p
)


# Use standardized design and corresponding true standardized coefficients.
#
# Original model:
#
#       y = beta0 + X_raw beta
#
# If X* = (X_raw - mean) / sd:
#
# coefficient in standardized X is beta * sd.

beta_true_standardized <- beta_true *
  x_sds


true_training_signal_centered <- as.numeric(
  X_train %*%
    beta_true_standardized
)


for (
  simulation in seq_len(
    number_simulations
  )
) {
  
  y_sim <- true_training_signal_centered +
    rnorm(
      n_train,
      sd =
        sigma_error
    )
  
  
  ols_simulation[
    simulation,
  ] <- solve(
    t(X_train) %*%
      X_train,
    t(X_train) %*%
      y_sim
  )
  
  
  ridge_simulation[
    simulation,
  ] <- ridge_fit(
    X_train,
    y_sim,
    lambda =
      ridge_min$lambda
  )
}


# ------------------------------------------------------------------------------
# 62. Coefficient Bias
# ------------------------------------------------------------------------------

ols_mean_beta <- colMeans(
  ols_simulation
)


ridge_mean_beta <- colMeans(
  ridge_simulation
)


ols_bias <- ols_mean_beta -
  beta_true_standardized


ridge_bias <- ridge_mean_beta -
  beta_true_standardized


# ------------------------------------------------------------------------------
# 63. Coefficient Variance
# ------------------------------------------------------------------------------

ols_variance <- apply(
  ols_simulation,
  2,
  var
)


ridge_variance <- apply(
  ridge_simulation,
  2,
  var
)


# ------------------------------------------------------------------------------
# 64. Compare Average Squared Bias and Variance
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "OLS",
    "Ridge"
  ),
  Mean_Squared_Bias = c(
    mean(
      ols_bias^2
    ),
    mean(
      ridge_bias^2
    )
  ),
  Mean_Coefficient_Variance = c(
    mean(
      ols_variance
    ),
    mean(
      ridge_variance
    )
  )
)


# Ridge deliberately adds bias in exchange for reduced variance.


# ==============================================================================
# PART XXXVII
#
# COEFFICIENT STABILITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 65. Example Correlated Coefficients
# ------------------------------------------------------------------------------

plot(
  ols_simulation[
    ,
    1
  ],
  ols_simulation[
    ,
    2
  ],
  pch = 19,
  cex = 0.5,
  xlab = "OLS Beta1",
  ylab = "OLS Beta2",
  main = "OLS Coefficient Instability"
)


plot(
  ridge_simulation[
    ,
    1
  ],
  ridge_simulation[
    ,
    2
  ],
  pch = 19,
  cex = 0.5,
  xlab = "Ridge Beta1",
  ylab = "Ridge Beta2",
  main = "Ridge Coefficient Stability"
)


# ==============================================================================
# PART XXXVIII
#
# P > N EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 66. High-Dimensional Problem
#
# OLS is no longer uniquely identified when p > n.
# Penalization still produces useful solutions.
# ------------------------------------------------------------------------------

set.seed(888)


n_hd <- 60

p_hd <- 120


X_hd <- matrix(
  rnorm(
    n_hd *
      p_hd
  ),
  nrow = n_hd
)


beta_hd <- c(
  rep(
    2,
    5
  ),
  rep(
    -1.5,
    5
  ),
  rep(
    0,
    p_hd -
      10
  )
)


y_hd <- as.numeric(
  X_hd %*%
    beta_hd
) +
  rnorm(
    n_hd,
    sd = 2
  )


# ------------------------------------------------------------------------------
# 67. Standardize
# ------------------------------------------------------------------------------

X_hd <- scale(
  X_hd
)


y_hd_centered <- y_hd -
  mean(
    y_hd
  )


# ------------------------------------------------------------------------------
# 68. X'X is Singular / Rank Deficient
# ------------------------------------------------------------------------------

qr(
  X_hd
)$rank


p_hd


# ------------------------------------------------------------------------------
# 69. Ridge Still Works
# ------------------------------------------------------------------------------

beta_hd_ridge <- ridge_fit(
  X_hd,
  y_hd_centered,
  lambda = 0.1
)


length(
  beta_hd_ridge
)


# ------------------------------------------------------------------------------
# 70. Lasso Still Works
# ------------------------------------------------------------------------------

lambda_max_hd <- max(
  abs(
    as.numeric(
      t(X_hd) %*%
        y_hd_centered
    )
  )
) /
  n_hd


beta_hd_lasso <- elastic_net_fit(
  X_hd,
  y_hd_centered,
  lambda =
    0.1 *
    lambda_max_hd,
  alpha = 1
)$beta


sum(
  abs(
    beta_hd_lasso
  ) >
    1e-8
)


# ==============================================================================
# PART XXXIX
#
# CORRELATED FEATURES: LASSO VS ELASTIC NET
# ==============================================================================


# ------------------------------------------------------------------------------
# 71. Highly Correlated Two-Variable Example
# ------------------------------------------------------------------------------

set.seed(999)


n_corr <- 200


x_1 <- rnorm(
  n_corr
)


x_2 <- x_1 +
  rnorm(
    n_corr,
    sd = 0.08
  )


x_3 <- rnorm(
  n_corr
)


X_corr <- cbind(
  X1 = x_1,
  X2 = x_2,
  X3 = x_3
)


y_corr <- 3 *
  x_1 +
  3 *
  x_2 +
  rnorm(
    n_corr,
    sd = 1
  )


X_corr <- scale(
  X_corr
)


y_corr <- y_corr -
  mean(
    y_corr
  )


# ------------------------------------------------------------------------------
# 72. Compare
# ------------------------------------------------------------------------------

lambda_corr <- 0.2


lasso_corr <- elastic_net_fit(
  X_corr,
  y_corr,
  lambda =
    lambda_corr,
  alpha = 1
)$beta


elastic_corr <- elastic_net_fit(
  X_corr,
  y_corr,
  lambda =
    lambda_corr,
  alpha = 0.5
)$beta


ridge_corr <- elastic_net_fit(
  X_corr,
  y_corr,
  lambda =
    lambda_corr,
  alpha = 0
)$beta


data.frame(
  Variable =
    colnames(
      X_corr
    ),
  Lasso =
    lasso_corr,
  ElasticNet =
    elastic_corr,
  Ridge =
    ridge_corr
)


# Elastic Net often exhibits a stronger grouping effect for correlated
# predictors than pure Lasso.


# ==============================================================================
# PART XL
#
# GENERALIZED PENALTY MATRICES
# ==============================================================================


# ------------------------------------------------------------------------------
# 73. First-Difference Penalty
#
# Suppose coefficients have an ordered interpretation and we believe adjacent
# coefficients should be similar.
#
# Penalize:
#
#       sum_j (beta_{j+1} - beta_j)^2
#
# This can be written:
#
#       beta' Omega beta
#
# where:
#
#       Omega = D'D
# ------------------------------------------------------------------------------

first_difference_matrix <- function(
    p
) {
  
  D <- matrix(
    0,
    nrow = p - 1,
    ncol = p
  )
  
  
  for (
    j in seq_len(
      p - 1
    )
  ) {
    
    D[
      j,
      j
    ] <- -1
    
    
    D[
      j,
      j + 1
    ] <- 1
  }
  
  
  D
}


D_first <- first_difference_matrix(
  p
)


Omega_first <- t(
  D_first
) %*%
  D_first


# ------------------------------------------------------------------------------
# 74. Fit Smooth-Coefficient Penalized Regression
# ------------------------------------------------------------------------------

beta_smooth <- quadratic_penalty_fit(
  X_train,
  y_train,
  lambda = 0.10,
  Omega =
    Omega_first
)


# ------------------------------------------------------------------------------
# 75. Compare Coefficient Roughness
# ------------------------------------------------------------------------------

coefficient_roughness <- function(
    beta
) {
  
  sum(
    diff(beta)^2
  )
}


data.frame(
  Model = c(
    "OLS",
    "Ridge",
    "First-Difference Penalty"
  ),
  Roughness = c(
    coefficient_roughness(
      ols_beta
    ),
    coefficient_roughness(
      beta_ridge_final
    ),
    coefficient_roughness(
      beta_smooth
    )
  )
)


# This penalty is included to emphasize that penalized regression is broader
# than only Ridge and Lasso.


# ==============================================================================
# PART XLI
#
# OPTIONAL glmnet VERIFICATION
# ==============================================================================


if (
  requireNamespace(
    "glmnet",
    quietly = TRUE
  )
) {
  
  # --------------------------------------------------------------------------
  # glmnet scales its objective similarly, but implementation / standardization
  # conventions can still create small numerical differences.
  # --------------------------------------------------------------------------
  
  glmnet_lasso <- glmnet::glmnet(
    x =
      X_train_raw,
    y =
      y_train_raw,
    alpha = 1,
    lambda =
      lambda_grid,
    standardize = TRUE,
    intercept = TRUE
  )
  
  
  glmnet_ridge <- glmnet::glmnet(
    x =
      X_train_raw,
    y =
      y_train_raw,
    alpha = 0,
    lambda =
      lambda_grid,
    standardize = TRUE,
    intercept = TRUE
  )
  
  
  glmnet_elastic <- glmnet::glmnet(
    x =
      X_train_raw,
    y =
      y_train_raw,
    alpha = 0.5,
    lambda =
      lambda_grid,
    standardize = TRUE,
    intercept = TRUE
  )
  
  
  cat(
    "\nglmnet models successfully fit.\n"
  )
  
  
  # glmnet also provides highly optimized coordinate descent and CV:
  #
  # glmnet::cv.glmnet(
  #   X_train_raw,
  #   y_train_raw,
  #   alpha = 1
  # )
}


# ==============================================================================
# PART XLII
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nPenalized Regression Summary\n"
)


cat(
  "----------------------------\n"
)


cat(
  "Training observations:",
  nrow(
    X_train
  ),
  "\n"
)


cat(
  "Test observations:",
  nrow(
    X_test
  ),
  "\n"
)


cat(
  "Predictors:",
  p,
  "\n"
)


cat(
  "True nonzero coefficients:",
  sum(
    beta_true !=
      0
  ),
  "\n"
)


cat(
  "Ridge selected lambda:",
  round(
    ridge_min$lambda,
    6
  ),
  "\n"
)


cat(
  "Lasso selected lambda:",
  round(
    lasso_min$lambda,
    6
  ),
  "\n"
)


cat(
  "Elastic Net selected lambda:",
  round(
    elastic_min$lambda,
    6
  ),
  "\n"
)


cat(
  "Lasso selected variables:",
  sum(
    abs(
      beta_lasso_final
    ) >
      1e-8
  ),
  "\n"
)


cat(
  "Elastic Net selected variables:",
  sum(
    abs(
      beta_elastic_final
    ) >
      1e-8
  ),
  "\n"
)


cat(
  "\nTest performance:\n"
)


print(
  performance
)