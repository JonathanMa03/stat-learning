# ==============================================================================
# Ridge Regression
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
p <- 8

X <- matrix(
  rnorm(n * p),
  nrow = n,
  ncol = p
)

colnames(X) <- paste0("x", 1:p)

beta_true <- c(
  3,
  -2.5,
  2,
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
  main = "Ridge Regression Data"
)

cor(data)


# ------------------------------------------------------------------------------
# 3. Standardize Predictors
# ------------------------------------------------------------------------------

# Ridge regression depends on the scale of the predictors,
# so standardization is important.

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

head(X_scaled)


# ------------------------------------------------------------------------------
# 4. Center the Response
# ------------------------------------------------------------------------------

y_mean <- mean(y)

y_centered <- y - y_mean

head(y_centered)


# ------------------------------------------------------------------------------
# 5. Ridge Regression Function
# ------------------------------------------------------------------------------

ridge_fit <- function(X, y, lambda) {
  
  p <- ncol(X)
  
  XtX <- t(X) %*% X
  
  penalty <- lambda * diag(p)
  
  beta_hat <- solve(
    XtX + penalty,
    t(X) %*% y
  )
  
  return(
    as.vector(beta_hat)
  )
}


# ------------------------------------------------------------------------------
# 6. Fit Ridge Regression for a Single Lambda
# ------------------------------------------------------------------------------

lambda <- 10

beta_ridge <- ridge_fit(
  X_scaled,
  y_centered,
  lambda
)

beta_ridge


# ------------------------------------------------------------------------------
# 7. Compare with Ordinary Least Squares
# ------------------------------------------------------------------------------

beta_ols <- solve(
  t(X_scaled) %*% X_scaled,
  t(X_scaled) %*% y_centered
)

beta_ols <- as.vector(beta_ols)

comparison <- data.frame(
  Predictor = colnames(X),
  OLS = beta_ols,
  Ridge = beta_ridge
)

comparison


# ------------------------------------------------------------------------------
# 8. Convert Ridge Coefficients Back to Original Scale
# ------------------------------------------------------------------------------

beta_original <- beta_ridge / X_sds

intercept_original <- y_mean -
  sum(
    beta_original * X_means
  )

intercept_original
beta_original


# ------------------------------------------------------------------------------
# 9. Prediction
# ------------------------------------------------------------------------------

y_hat <- intercept_original +
  X %*% beta_original

y_hat <- as.vector(y_hat)

head(y_hat)


# ------------------------------------------------------------------------------
# 10. Training Error
# ------------------------------------------------------------------------------

residuals <- y - y_hat

RSS <- sum(
  residuals^2
)

MSE <- mean(
  residuals^2
)

RMSE <- sqrt(MSE)

RSS
MSE
RMSE


# ------------------------------------------------------------------------------
# 11. Ridge Coefficient Path
# ------------------------------------------------------------------------------

lambda_grid <- 10^seq(
  -3,
  4,
  length.out = 200
)

ridge_path <- matrix(
  NA,
  nrow = length(lambda_grid),
  ncol = p
)

colnames(ridge_path) <- colnames(X)

for (i in seq_along(lambda_grid)) {
  
  ridge_path[i, ] <- ridge_fit(
    X_scaled,
    y_centered,
    lambda_grid[i]
  )
}


# ------------------------------------------------------------------------------
# 12. Plot Ridge Coefficient Paths
# ------------------------------------------------------------------------------

matplot(
  log(lambda_grid),
  ridge_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Coefficient",
  main = "Ridge Coefficient Paths"
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
# 13. Coefficient Magnitudes
# ------------------------------------------------------------------------------

coefficient_norm <- apply(
  ridge_path,
  1,
  function(beta) {
    sqrt(sum(beta^2))
  }
)

plot(
  log(lambda_grid),
  coefficient_norm,
  type = "l",
  xlab = "log(lambda)",
  ylab = "L2 Norm of Coefficients",
  main = "Ridge Shrinkage"
)


# ------------------------------------------------------------------------------
# 14. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)

train_index <- sample(
  1:n,
  size = floor(0.7 * n)
)

test_index <- setdiff(
  1:n,
  train_index
)

X_train <- X[train_index, ]
X_test <- X[test_index, ]

y_train <- y[train_index]
y_test <- y[test_index]


# ------------------------------------------------------------------------------
# 15. Standardize Using Training Data Only
# ------------------------------------------------------------------------------

train_means <- colMeans(X_train)

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

y_train_mean <- mean(y_train)

y_train_centered <- y_train -
  y_train_mean


# ------------------------------------------------------------------------------
# 16. Evaluate Lambda Values on Test Data
# ------------------------------------------------------------------------------

test_mse <- numeric(
  length(lambda_grid)
)

for (i in seq_along(lambda_grid)) {
  
  beta_hat <- ridge_fit(
    X_train_scaled,
    y_train_centered,
    lambda_grid[i]
  )
  
  y_pred <- y_train_mean +
    X_test_scaled %*% beta_hat
  
  test_mse[i] <- mean(
    (y_test - y_pred)^2
  )
}


# ------------------------------------------------------------------------------
# 17. Plot Test MSE
# ------------------------------------------------------------------------------

plot(
  log(lambda_grid),
  test_mse,
  type = "l",
  xlab = "log(lambda)",
  ylab = "Test MSE",
  main = "Ridge Regression Test Error"
)


# ------------------------------------------------------------------------------
# 18. Best Lambda from Test Set
# ------------------------------------------------------------------------------

best_index <- which.min(
  test_mse
)

best_lambda <- lambda_grid[
  best_index
]

best_test_mse <- test_mse[
  best_index
]

best_lambda
best_test_mse


# ------------------------------------------------------------------------------
# 19. Best Ridge Model
# ------------------------------------------------------------------------------

best_beta_scaled <- ridge_fit(
  X_train_scaled,
  y_train_centered,
  best_lambda
)

best_beta_scaled


# ------------------------------------------------------------------------------
# 20. Convert Best Model Back to Original Scale
# ------------------------------------------------------------------------------

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
# 21. Compare Estimated and True Coefficients
# ------------------------------------------------------------------------------

coefficient_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  Estimated = best_beta_original
)

coefficient_comparison


# ------------------------------------------------------------------------------
# 22. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

ridge_cv <- function(
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
    
    
    # Standardization must be based only on training fold
    
    means <- colMeans(
      X_train
    )
    
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
    
    y_mean <- mean(
      y_train
    )
    
    y_centered <- y_train -
      y_mean
    
    
    for (i in seq_along(lambda_grid)) {
      
      beta_hat <- ridge_fit(
        X_train_scaled,
        y_centered,
        lambda_grid[i]
      )
      
      predictions <-
        y_mean +
        X_valid_scaled %*%
        beta_hat
      
      cv_error[i, fold] <-
        mean(
          (y_valid - predictions)^2
        )
    }
  }
  
  mean_cv_error <- rowMeans(
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
# 23. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)

cv_results <- ridge_cv(
  X,
  y,
  lambda_grid,
  k = 10
)

head(
  cv_results$cv_error
)


# ------------------------------------------------------------------------------
# 24. Plot Cross-Validation Error
# ------------------------------------------------------------------------------

plot(
  log(cv_results$lambda),
  cv_results$cv_error,
  type = "l",
  xlab = "log(lambda)",
  ylab = "Cross-Validated MSE",
  main = "Ridge Regression Cross-Validation"
)


# ------------------------------------------------------------------------------
# 25. Select Lambda by Cross-Validation
# ------------------------------------------------------------------------------

cv_best_index <- which.min(
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
# 26. Fit Final Ridge Model
# ------------------------------------------------------------------------------

X_means_final <- colMeans(X)

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

y_mean_final <- mean(y)

y_centered_final <-
  y - y_mean_final

beta_final_scaled <- ridge_fit(
  X_scaled_final,
  y_centered_final,
  cv_best_lambda
)

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
# 27. Final Coefficient Comparison
# ------------------------------------------------------------------------------

final_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  Ridge = beta_final
)

final_comparison


# ------------------------------------------------------------------------------
# 28. Ridge vs OLS
# ------------------------------------------------------------------------------

ols_model <- lm(
  y ~ .,
  data = data
)

ols_coefficients <-
  coef(ols_model)[-1]

comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  OLS = ols_coefficients,
  Ridge = beta_final
)

comparison


# ------------------------------------------------------------------------------
# 29. Visualize Coefficient Comparison
# ------------------------------------------------------------------------------

matplot(
  cbind(
    beta_true,
    ols_coefficients,
    beta_final
  ),
  type = "b",
  pch = 19,
  lty = 1,
  xaxt = "n",
  xlab = "Predictor",
  ylab = "Coefficient",
  main = "True vs OLS vs Ridge Coefficients"
)

axis(
  1,
  at = 1:p,
  labels = colnames(X)
)

legend(
  "topright",
  legend = c(
    "True",
    "OLS",
    "Ridge"
  ),
  lty = 1,
  pch = 19
)


# ------------------------------------------------------------------------------
# 30. Singular Value Interpretation
# ------------------------------------------------------------------------------

svd_X <- svd(
  X_scaled
)

singular_values <- svd_X$d

shrinkage_factors <-
  singular_values^2 /
  (
    singular_values^2 +
      cv_best_lambda
  )

data.frame(
  Singular_Value = singular_values,
  Shrinkage_Factor = shrinkage_factors
)


# ------------------------------------------------------------------------------
# 31. Plot Shrinkage Factors
# ------------------------------------------------------------------------------

plot(
  singular_values,
  shrinkage_factors,
  pch = 19,
  xlab = "Singular Value",
  ylab = "Shrinkage Factor",
  main = "Ridge Shrinkage by Principal Direction"
)