# ==============================================================================
# Elastic Net Regression
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
  main = "Elastic Net Regression Data"
)

cor(data)


# ------------------------------------------------------------------------------
# 3. Standardize Predictors
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# 6. Elastic Net Coordinate Descent
# ------------------------------------------------------------------------------

elastic_net_fit <- function(
    X,
    y,
    lambda,
    alpha,
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
  
  
  # Column squared norms
  
  z <- colSums(
    X^2
  ) / n
  
  
  for (iteration in 1:max_iter) {
    
    beta_old <- beta
    
    
    for (j in 1:p) {
      
      residual_j <-
        y -
        X %*% beta +
        X[, j] * beta[j]
      
      
      rho_j <-
        sum(
          X[, j] *
            residual_j
        ) / n
      
      
      beta[j] <-
        soft_threshold(
          rho_j,
          lambda * alpha
        ) /
        (
          z[j] +
            lambda * (1 - alpha)
        )
    }
    
    
    if (
      max(
        abs(
          beta -
          beta_old
        )
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
# 7. Fit Elastic Net for a Single Lambda and Alpha
# ------------------------------------------------------------------------------

lambda <- 0.5
alpha <- 0.5

elastic_model <- elastic_net_fit(
  X_scaled,
  y_centered,
  lambda,
  alpha
)

beta_elastic_scaled <-
  elastic_model$coefficients

beta_elastic_scaled

elastic_model$iterations


# ------------------------------------------------------------------------------
# 8. Convert Coefficients Back to Original Scale
# ------------------------------------------------------------------------------

beta_elastic <-
  beta_elastic_scaled /
  X_sds

intercept_elastic <-
  y_mean -
  sum(
    beta_elastic *
      X_means
  )

intercept_elastic
beta_elastic


# ------------------------------------------------------------------------------
# 9. Compare with OLS
# ------------------------------------------------------------------------------

ols_model <- lm(
  y ~ .,
  data = data
)

beta_ols <-
  coef(
    ols_model
  )[-1]

comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  OLS = beta_ols,
  Elastic_Net = beta_elastic
)

comparison


# ------------------------------------------------------------------------------
# 10. Selected Predictors
# ------------------------------------------------------------------------------

selected <- abs(
  beta_elastic
) > 1e-8

colnames(X)[
  selected
]

sum(selected)


# ------------------------------------------------------------------------------
# 11. Prediction
# ------------------------------------------------------------------------------

y_hat <-
  intercept_elastic +
  X %*% beta_elastic

y_hat <- as.vector(
  y_hat
)

head(y_hat)


# ------------------------------------------------------------------------------
# 12. Training Error
# ------------------------------------------------------------------------------

residuals <-
  y -
  y_hat

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

# For alpha > 0:
#
# lambda_max = max(|X'y| / n) / alpha

lambda_max <- max(
  abs(
    as.vector(
      crossprod(
        X_scaled,
        y_centered
      )
    )
  )
) /
  (
    n *
      alpha
  )

lambda_max


# ------------------------------------------------------------------------------
# 14. Lambda Grid
# ------------------------------------------------------------------------------

lambda_grid <- exp(
  seq(
    log(lambda_max),
    log(lambda_max * 0.001),
    length.out = 200
  )
)

head(lambda_grid)
tail(lambda_grid)


# ------------------------------------------------------------------------------
# 15. Elastic Net Regularization Path
# ------------------------------------------------------------------------------

elastic_path <- matrix(
  NA,
  nrow = length(lambda_grid),
  ncol = p
)

colnames(elastic_path) <-
  colnames(X)

beta_previous <- rep(
  0,
  p
)

for (i in seq_along(lambda_grid)) {
  
  fit <- elastic_net_fit(
    X_scaled,
    y_centered,
    lambda = lambda_grid[i],
    alpha = alpha,
    beta_init = beta_previous
  )
  
  elastic_path[i, ] <-
    fit$coefficients
  
  beta_previous <-
    fit$coefficients
}


# ------------------------------------------------------------------------------
# 16. Plot Coefficient Paths
# ------------------------------------------------------------------------------

matplot(
  log(lambda_grid),
  elastic_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Standardized Coefficient",
  main = "Elastic Net Coefficient Paths"
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
# 17. Number of Nonzero Coefficients
# ------------------------------------------------------------------------------

nonzero_count <- apply(
  elastic_path,
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
  main = "Elastic Net Model Complexity"
)


# ------------------------------------------------------------------------------
# 18. Compare Different Alpha Values
# ------------------------------------------------------------------------------

alpha_grid <- c(
  0.25,
  0.50,
  0.75,
  1.00
)

alpha_results <- list()

for (a in alpha_grid) {
  
  lambda_max_a <- max(
    abs(
      as.vector(
        crossprod(
          X_scaled,
          y_centered
        )
      )
    )
  ) /
    (
      n * a
    )
  
  lambda_grid_a <- exp(
    seq(
      log(lambda_max_a),
      log(lambda_max_a * 0.001),
      length.out = 100
    )
  )
  
  path_a <- matrix(
    NA,
    nrow = length(lambda_grid_a),
    ncol = p
  )
  
  beta_previous <- rep(
    0,
    p
  )
  
  for (i in seq_along(lambda_grid_a)) {
    
    fit <- elastic_net_fit(
      X_scaled,
      y_centered,
      lambda_grid_a[i],
      alpha = a,
      beta_init = beta_previous
    )
    
    path_a[i, ] <-
      fit$coefficients
    
    beta_previous <-
      fit$coefficients
  }
  
  alpha_results[[as.character(a)]] <- list(
    lambda = lambda_grid_a,
    path = path_a
  )
}


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
# 21. Cross-Validation Function
# ------------------------------------------------------------------------------

elastic_net_cv <- function(
    X,
    y,
    alpha_grid,
    n_lambda = 100,
    k = 10
) {
  
  n <- nrow(X)
  p <- ncol(X)
  
  folds <- sample(
    rep(
      1:k,
      length.out = n
    )
  )
  
  results <- list()
  
  
  for (a in alpha_grid) {
    
    if (a == 0) {
      
      next
    }
    
    
    lambda_max <- max(
      abs(
        as.vector(
          crossprod(
            scale(X),
            y - mean(y)
          )
        )
      )
    ) /
      (
        n * a
      )
    
    
    lambda_grid <- exp(
      seq(
        log(lambda_max),
        log(lambda_max * 0.001),
        length.out = n_lambda
      )
    )
    
    
    cv_error <- matrix(
      NA,
      nrow = n_lambda,
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
      
      
      y_mean <-
        mean(y_train)
      
      y_centered <-
        y_train -
        y_mean
      
      
      beta_previous <- rep(
        0,
        p
      )
      
      
      for (i in seq_along(lambda_grid)) {
        
        fit <- elastic_net_fit(
          X_train_scaled,
          y_centered,
          lambda = lambda_grid[i],
          alpha = a,
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
    
    
    results[[as.character(a)]] <- list(
      alpha = a,
      lambda = lambda_grid,
      mean_cv_error = rowMeans(
        cv_error
      ),
      fold_errors = cv_error
    )
  }
  
  
  return(results)
}


# ------------------------------------------------------------------------------
# 22. Run Cross-Validation
# ------------------------------------------------------------------------------

alpha_grid <- c(
  0.25,
  0.50,
  0.75,
  1.00
)

set.seed(123)

cv_results <- elastic_net_cv(
  X,
  y,
  alpha_grid = alpha_grid,
  n_lambda = 100,
  k = 10
)


# ------------------------------------------------------------------------------
# 23. Find Best Alpha and Lambda
# ------------------------------------------------------------------------------

best_error <- Inf
best_alpha <- NA
best_lambda <- NA

for (name in names(cv_results)) {
  
  result <-
    cv_results[[name]]
  
  index <- which.min(
    result$mean_cv_error
  )
  
  error <-
    result$mean_cv_error[
      index
    ]
  
  if (error < best_error) {
    
    best_error <-
      error
    
    best_alpha <-
      result$alpha
    
    best_lambda <-
      result$lambda[
        index
      ]
  }
}

best_alpha
best_lambda
best_error


# ------------------------------------------------------------------------------
# 24. Plot CV Curves
# ------------------------------------------------------------------------------

first_result <-
  cv_results[[1]]

plot(
  log(first_result$lambda),
  first_result$mean_cv_error,
  type = "l",
  xlab = "log(lambda)",
  ylab = "Cross-Validated MSE",
  main = "Elastic Net Cross-Validation"
)

for (i in 2:length(cv_results)) {
  
  result <-
    cv_results[[i]]
  
  lines(
    log(result$lambda),
    result$mean_cv_error,
    lty = i
  )
}

legend(
  "topright",
  legend = paste0(
    "alpha = ",
    names(cv_results)
  ),
  lty = seq_along(
    cv_results
  )
)


# ------------------------------------------------------------------------------
# 25. Fit Final Elastic Net Model
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
  y -
  y_mean_final


final_fit <- elastic_net_fit(
  X_scaled_final,
  y_centered_final,
  lambda = best_lambda,
  alpha = best_alpha
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
# 26. Selected Predictors
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
# 27. Final Coefficient Comparison
# ------------------------------------------------------------------------------

final_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  OLS = beta_ols,
  Elastic_Net = beta_final
)

final_comparison


# ------------------------------------------------------------------------------
# 28. Compare Sparsity
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "True Model",
    "OLS",
    "Elastic Net"
  ),
  
  Nonzero = c(
    sum(
      beta_true != 0
    ),
    
    sum(
      abs(beta_ols) > 1e-8
    ),
    
    sum(
      abs(beta_final) > 1e-8
    )
  )
)


# ------------------------------------------------------------------------------
# 29. Visualize Final Coefficients
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
  main = "True vs OLS vs Elastic Net"
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
    "Elastic Net"
  ),
  pch = 19,
  lty = 1
)


# ------------------------------------------------------------------------------
# 30. Compare Ridge, Lasso, and Elastic Net
# ------------------------------------------------------------------------------

ridge_fit <- function(
    X,
    y,
    lambda
) {
  
  p <- ncol(X)
  
  solve(
    t(X) %*% X +
      lambda * diag(p),
    t(X) %*% y
  )
}


lasso_fit <- function(
    X,
    y,
    lambda,
    beta_init = NULL,
    max_iter = 10000,
    tol = 1e-8
) {
  
  elastic_net_fit(
    X = X,
    y = y,
    lambda = lambda,
    alpha = 1,
    beta_init = beta_init,
    max_iter = max_iter,
    tol = tol
  )
}


beta_ridge_scaled <- ridge_fit(
  X_scaled_final,
  y_centered_final,
  best_lambda
)

beta_lasso_scaled <- lasso_fit(
  X_scaled_final,
  y_centered_final,
  best_lambda
)$coefficients


beta_ridge <-
  as.vector(
    beta_ridge_scaled
  ) /
  X_sds_final

beta_lasso <-
  beta_lasso_scaled /
  X_sds_final


regularization_comparison <- data.frame(
  Predictor = colnames(X),
  Ridge = beta_ridge,
  Lasso = beta_lasso,
  Elastic_Net = beta_final
)

regularization_comparison