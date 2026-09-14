# ==============================================================================
# Lasso Regression
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 200
p <- 10

X <- matrix(
  rnorm(n * p),
  nrow = n,
  ncol = p
)

colnames(X) <- paste0("x", 1:p)

# Sparse coefficient vector

beta_true <- c(
  3,
  -2.5,
  2,
  0,
  0,
  1.5,
  0,
  0,
  0.5,
  0
)

beta_0 <- 2
sigma <- 2

epsilon <- rnorm(
  n,
  mean = 0,
  sd = sigma
)

y <- beta_0 +
  X %*% beta_true +
  epsilon

y <- as.vector(y)

data <- data.frame(
  y = y,
  X
)

head(data)
summary(data)


# ------------------------------------------------------------------------------
# 2. Explore the Data
# ------------------------------------------------------------------------------

pairs(
  data,
  pch = 19,
  main = "Lasso Regression Data"
)

cor(data)


# ------------------------------------------------------------------------------
# 3. Standardize Predictors
# ------------------------------------------------------------------------------

# Lasso depends on predictor scale, so predictors should be standardized.

X_means <- colMeans(X)

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


# ------------------------------------------------------------------------------
# 4. Center the Response
# ------------------------------------------------------------------------------

y_mean <- mean(y)

y_centered <- y - y_mean


# ------------------------------------------------------------------------------
# 5. Soft-Thresholding Function
# ------------------------------------------------------------------------------

soft_threshold <- function(z, gamma) {
  
  sign(z) *
    pmax(
      abs(z) - gamma,
      0
    )
}


# Examples

soft_threshold(4, 1)
soft_threshold(-4, 1)
soft_threshold(0.5, 1)


# ------------------------------------------------------------------------------
# 6. Coordinate Descent for Lasso
# ------------------------------------------------------------------------------

lasso_fit <- function(
    X,
    y,
    lambda,
    beta_init = NULL,
    max_iter = 10000,
    tol = 1e-8
) {
  
  n <- nrow(X)
  p <- ncol(X)
  
  if (is.null(beta_init)) {
    
    beta <- rep(
      0,
      p
    )
    
  } else {
    
    beta <- beta_init
  }
  
  
  # Precompute column squared norms
  
  z <- colSums(
    X^2
  ) / n
  
  
  for (iteration in 1:max_iter) {
    
    beta_old <- beta
    
    
    for (j in 1:p) {
      
      # Partial residual:
      #
      # r_j = y - sum_{k != j} x_k beta_k
      
      residual_j <-
        y -
        X %*% beta +
        X[, j] * beta[j]
      
      
      # Partial correlation
      
      rho_j <-
        sum(
          X[, j] *
            residual_j
        ) / n
      
      
      # Coordinate update
      
      beta[j] <-
        soft_threshold(
          rho_j,
          lambda
        ) / z[j]
    }
    
    
    # Check convergence
    
    if (
      max(
        abs(beta - beta_old)
      ) < tol
    ) {
      
      break
    }
  }
  
  
  return(
    list(
      coefficients = beta,
      iterations = iteration
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Fit Lasso for a Single Lambda
# ------------------------------------------------------------------------------

lambda <- 0.5

lasso_model <- lasso_fit(
  X_scaled,
  y_centered,
  lambda
)

beta_lasso_scaled <-
  lasso_model$coefficients

beta_lasso_scaled

lasso_model$iterations


# ------------------------------------------------------------------------------
# 8. Convert Coefficients Back to Original Scale
# ------------------------------------------------------------------------------

beta_lasso <-
  beta_lasso_scaled /
  X_sds

intercept_lasso <-
  y_mean -
  sum(
    beta_lasso *
      X_means
  )

intercept_lasso
beta_lasso


# ------------------------------------------------------------------------------
# 9. Compare with Ordinary Least Squares
# ------------------------------------------------------------------------------

ols_model <- lm(
  y ~ .,
  data = data
)

beta_ols <- coef(
  ols_model
)[-1]

comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  OLS = beta_ols,
  Lasso = beta_lasso
)

comparison


# ------------------------------------------------------------------------------
# 10. Number of Selected Variables
# ------------------------------------------------------------------------------

sum(
  beta_lasso != 0
)

colnames(X)[
  beta_lasso != 0
]


# ------------------------------------------------------------------------------
# 11. Predictions
# ------------------------------------------------------------------------------

y_hat <-
  intercept_lasso +
  X %*% beta_lasso

y_hat <- as.vector(
  y_hat
)

head(y_hat)


# ------------------------------------------------------------------------------
# 12. Training Error
# ------------------------------------------------------------------------------

residuals <-
  y - y_hat

RSS <- sum(
  residuals^2
)

MSE <- mean(
  residuals^2
)

RMSE <- sqrt(
  MSE
)

RSS
MSE
RMSE


# ------------------------------------------------------------------------------
# 13. Maximum Lambda
# ------------------------------------------------------------------------------

# lambda_max is the smallest lambda for which
# all slope coefficients are zero.

lambda_max <- max(
  abs(
    as.vector(
      crossprod(
        X_scaled,
        y_centered
      )
    )
  )
) / n

lambda_max


# ------------------------------------------------------------------------------
# 14. Lambda Grid
# ------------------------------------------------------------------------------

lambda_min <-
  lambda_max * 0.001

lambda_grid <- exp(
  seq(
    log(lambda_max),
    log(lambda_min),
    length.out = 200
  )
)

head(lambda_grid)
tail(lambda_grid)


# ------------------------------------------------------------------------------
# 15. Compute the Lasso Regularization Path
# ------------------------------------------------------------------------------

lasso_path <- matrix(
  NA,
  nrow = length(lambda_grid),
  ncol = p
)

colnames(lasso_path) <-
  colnames(X)


# Warm start:
# use coefficients from previous lambda as starting values

beta_previous <- rep(
  0,
  p
)

for (i in seq_along(lambda_grid)) {
  
  fit <- lasso_fit(
    X_scaled,
    y_centered,
    lambda = lambda_grid[i],
    beta_init = beta_previous
  )
  
  lasso_path[i, ] <-
    fit$coefficients
  
  beta_previous <-
    fit$coefficients
}


# ------------------------------------------------------------------------------
# 16. Plot Lasso Coefficient Paths
# ------------------------------------------------------------------------------

matplot(
  log(lambda_grid),
  lasso_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Standardized Coefficient",
  main = "Lasso Coefficient Paths"
)

abline(
  h = 0,
  lty = 2
)

legend(
  "topright",
  legend = colnames(X),
  lty = 1,
  cex = 0.7
)


# ------------------------------------------------------------------------------
# 17. Plot Against L1 Norm
# ------------------------------------------------------------------------------

L1_norm <- apply(
  lasso_path,
  1,
  function(beta) {
    sum(
      abs(beta)
    )
  }
)

matplot(
  L1_norm,
  lasso_path,
  type = "l",
  lty = 1,
  xlab = "L1 Norm",
  ylab = "Standardized Coefficient",
  main = "Lasso Regularization Path"
)

abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 18. Number of Nonzero Coefficients
# ------------------------------------------------------------------------------

nonzero_count <- apply(
  lasso_path,
  1,
  function(beta) {
    sum(
      abs(beta) > 1e-8
    )
  }
)

plot(
  log(lambda_grid),
  nonzero_count,
  type = "s",
  xlab = "log(lambda)",
  ylab = "Number of Nonzero Coefficients",
  main = "Lasso Model Complexity"
)


# ------------------------------------------------------------------------------
# 19. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)

train_index <- sample(
  1:n,
  size = floor(
    0.7 * n
  )
)

test_index <- setdiff(
  1:n,
  train_index
)

X_train <- X[
  train_index,
  ,
  drop = FALSE
]

X_test <- X[
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
# 20. Standardize Using Training Data Only
# ------------------------------------------------------------------------------

train_means <-
  colMeans(X_train)

train_sds <- apply(
  X_train,
  2,
  sd
)

X_train_scaled <- scale(
  X_train,
  center = train_means,
  scale = train_sds
)

X_test_scaled <- scale(
  X_test,
  center = train_means,
  scale = train_sds
)

y_train_mean <-
  mean(y_train)

y_train_centered <-
  y_train -
  y_train_mean


# ------------------------------------------------------------------------------
# 21. Lambda Grid for Training Data
# ------------------------------------------------------------------------------

lambda_max_train <- max(
  abs(
    as.vector(
      crossprod(
        X_train_scaled,
        y_train_centered
      )
    )
  )
) / length(y_train)

lambda_grid_train <- exp(
  seq(
    log(lambda_max_train),
    log(lambda_max_train * 0.001),
    length.out = 200
  )
)


# ------------------------------------------------------------------------------
# 22. Evaluate Lambda Values on Test Data
# ------------------------------------------------------------------------------

test_mse <- numeric(
  length(lambda_grid_train)
)

beta_previous <- rep(
  0,
  p
)

for (i in seq_along(lambda_grid_train)) {
  
  fit <- lasso_fit(
    X_train_scaled,
    y_train_centered,
    lambda_grid_train[i],
    beta_init = beta_previous
  )
  
  beta_hat <-
    fit$coefficients
  
  predictions <-
    y_train_mean +
    X_test_scaled %*%
    beta_hat
  
  test_mse[i] <- mean(
    (
      y_test -
        predictions
    )^2
  )
  
  beta_previous <-
    beta_hat
}


# ------------------------------------------------------------------------------
# 23. Plot Test Error
# ------------------------------------------------------------------------------

plot(
  log(lambda_grid_train),
  test_mse,
  type = "l",
  xlab = "log(lambda)",
  ylab = "Test MSE",
  main = "Lasso Test Error"
)


# ------------------------------------------------------------------------------
# 24. Best Lambda from Test Set
# ------------------------------------------------------------------------------

best_index <-
  which.min(
    test_mse
  )

best_lambda <-
  lambda_grid_train[
    best_index
  ]

best_test_mse <-
  test_mse[
    best_index
  ]

best_lambda
best_test_mse


# ------------------------------------------------------------------------------
# 25. Fit Best Train-Test Model
# ------------------------------------------------------------------------------

best_fit <- lasso_fit(
  X_train_scaled,
  y_train_centered,
  best_lambda
)

best_beta_scaled <-
  best_fit$coefficients

best_beta_original <-
  best_beta_scaled /
  train_sds

best_intercept <-
  y_train_mean -
  sum(
    best_beta_original *
      train_means
  )

best_intercept
best_beta_original


# ------------------------------------------------------------------------------
# 26. Compare Estimated and True Coefficients
# ------------------------------------------------------------------------------

coefficient_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  Estimated = best_beta_original
)

coefficient_comparison


# ------------------------------------------------------------------------------
# 27. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

lasso_cv <- function(
    X,
    y,
    lambda_grid,
    k = 10
) {
  
  n <- nrow(X)
  
  folds <- sample(
    rep(
      1:k,
      length.out = n
    )
  )
  
  cv_error <- matrix(
    NA,
    nrow = length(lambda_grid),
    ncol = k
  )
  
  
  for (fold in 1:k) {
    
    train_index <- which(
      folds != fold
    )
    
    valid_index <- which(
      folds == fold
    )
    
    
    X_train <- X[
      train_index,
      ,
      drop = FALSE
    ]
    
    X_valid <- X[
      valid_index,
      ,
      drop = FALSE
    ]
    
    y_train <- y[
      train_index
    ]
    
    y_valid <- y[
      valid_index
    ]
    
    
    # Standardize using training fold only
    
    means <-
      colMeans(X_train)
    
    sds <- apply(
      X_train,
      2,
      sd
    )
    
    X_train_scaled <- scale(
      X_train,
      center = means,
      scale = sds
    )
    
    X_valid_scaled <- scale(
      X_valid,
      center = means,
      scale = sds
    )
    
    
    y_mean <-
      mean(y_train)
    
    y_centered <-
      y_train -
      y_mean
    
    
    beta_previous <- rep(
      0,
      ncol(X)
    )
    
    
    for (i in seq_along(lambda_grid)) {
      
      fit <- lasso_fit(
        X_train_scaled,
        y_centered,
        lambda_grid[i],
        beta_init = beta_previous
      )
      
      beta_hat <-
        fit$coefficients
      
      
      predictions <-
        y_mean +
        X_valid_scaled %*%
        beta_hat
      
      
      cv_error[i, fold] <-
        mean(
          (
            y_valid -
              predictions
          )^2
        )
      
      
      beta_previous <-
        beta_hat
    }
  }
  
  
  mean_cv_error <-
    rowMeans(
      cv_error
    )
  
  
  return(
    list(
      lambda = lambda_grid,
      cv_error = mean_cv_error,
      fold_errors = cv_error
    )
  )
}


# ------------------------------------------------------------------------------
# 28. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)

cv_results <- lasso_cv(
  X,
  y,
  lambda_grid,
  k = 10
)

head(
  cv_results$cv_error
)


# ------------------------------------------------------------------------------
# 29. Plot Cross-Validation Error
# ------------------------------------------------------------------------------

plot(
  log(cv_results$lambda),
  cv_results$cv_error,
  type = "l",
  xlab = "log(lambda)",
  ylab = "Cross-Validated MSE",
  main = "Lasso Cross-Validation"
)


# ------------------------------------------------------------------------------
# 30. Select Lambda by Cross-Validation
# ------------------------------------------------------------------------------

cv_best_index <-
  which.min(
    cv_results$cv_error
  )

cv_best_lambda <-
  cv_results$lambda[
    cv_best_index
  ]

cv_best_error <-
  cv_results$cv_error[
    cv_best_index
  ]

cv_best_lambda
cv_best_error


# ------------------------------------------------------------------------------
# 31. Standard Error of Cross-Validation Error
# ------------------------------------------------------------------------------

cv_se <- apply(
  cv_results$fold_errors,
  1,
  sd
) / sqrt(
  ncol(
    cv_results$fold_errors
  )
)

cv_min_error <-
  cv_results$cv_error[
    cv_best_index
  ]

cv_threshold <-
  cv_min_error +
  cv_se[
    cv_best_index
  ]


# ------------------------------------------------------------------------------
# 32. One-Standard-Error Rule
# ------------------------------------------------------------------------------

# Lambda values are ordered from largest to smallest.
# Choose the largest lambda whose error is within one
# standard error of the minimum.

eligible <- which(
  cv_results$cv_error <=
    cv_threshold
)

lambda_1se <-
  cv_results$lambda[
    min(eligible)
  ]

lambda_1se


# ------------------------------------------------------------------------------
# 33. Plot CV Error with Selected Lambdas
# ------------------------------------------------------------------------------

plot(
  log(cv_results$lambda),
  cv_results$cv_error,
  type = "l",
  xlab = "log(lambda)",
  ylab = "Cross-Validated MSE",
  main = "Lasso Cross-Validation"
)

abline(
  v = log(cv_best_lambda),
  lty = 2
)

abline(
  v = log(lambda_1se),
  lty = 3
)


# ------------------------------------------------------------------------------
# 34. Fit Final Lasso Model
# ------------------------------------------------------------------------------

X_means_final <-
  colMeans(X)

X_sds_final <- apply(
  X,
  2,
  sd
)

X_scaled_final <- scale(
  X,
  center = X_means_final,
  scale = X_sds_final
)

y_mean_final <-
  mean(y)

y_centered_final <-
  y - y_mean_final


final_fit <- lasso_fit(
  X_scaled_final,
  y_centered_final,
  cv_best_lambda
)

beta_final_scaled <-
  final_fit$coefficients

beta_final <-
  beta_final_scaled /
  X_sds_final

intercept_final <-
  y_mean_final -
  sum(
    beta_final *
      X_means_final
  )

intercept_final
beta_final


# ------------------------------------------------------------------------------
# 35. Selected Predictors
# ------------------------------------------------------------------------------

selected <- abs(
  beta_final
) > 1e-8

colnames(X)[
  selected
]

beta_final[
  selected
]


# ------------------------------------------------------------------------------
# 36. Final Coefficient Comparison
# ------------------------------------------------------------------------------

final_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  OLS = beta_ols,
  Lasso = beta_final
)

final_comparison


# ------------------------------------------------------------------------------
# 37. Compare Sparsity
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "True Model",
    "OLS",
    "Lasso"
  ),
  
  Nonzero = c(
    sum(beta_true != 0),
    sum(abs(beta_ols) > 1e-8),
    sum(abs(beta_final) > 1e-8)
  )
)


# ------------------------------------------------------------------------------
# 38. Visualize Coefficients
# ------------------------------------------------------------------------------

matplot(
  cbind(
    beta_true,
    beta_ols,
    beta_final
  ),
  type = "b",
  pch = 19,
  lty = 1,
  xaxt = "n",
  xlab = "Predictor",
  ylab = "Coefficient",
  main = "True vs OLS vs Lasso Coefficients"
)

axis(
  1,
  at = 1:p,
  labels = colnames(X)
)

abline(
  h = 0,
  lty = 2
)

legend(
  "topright",
  legend = c(
    "True",
    "OLS",
    "Lasso"
  ),
  pch = 19,
  lty = 1
)