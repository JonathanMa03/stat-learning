# ==============================================================================
# Principal Components Regression
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

# Create correlated predictors

z1 <- rnorm(n)
z2 <- rnorm(n)
z3 <- rnorm(n)

x1 <- z1 + rnorm(n, sd = 0.2)
x2 <- 0.8 * z1 + rnorm(n, sd = 0.2)
x3 <- 0.6 * z1 + rnorm(n, sd = 0.3)

x4 <- z2 + rnorm(n, sd = 0.2)
x5 <- 0.7 * z2 + rnorm(n, sd = 0.3)

x6 <- z3 + rnorm(n, sd = 0.2)
x7 <- 0.5 * z3 + rnorm(n, sd = 0.4)

x8 <- rnorm(n)

X <- cbind(
  x1,
  x2,
  x3,
  x4,
  x5,
  x6,
  x7,
  x8
)

colnames(X) <- paste0("x", 1:p)

beta_true <- c(
  2.5,
  2,
  1.5,
  -2,
  -1.5,
  1,
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
# 2. Explore Predictor Correlation
# ------------------------------------------------------------------------------

cor(X)

pairs(
  X,
  pch = 19,
  main = "Correlated Predictors"
)


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
# 4. Center Response
# ------------------------------------------------------------------------------

y_mean <- mean(y)

y_centered <- y - y_mean


# ------------------------------------------------------------------------------
# 5. Compute PCA Using SVD
# ------------------------------------------------------------------------------

# X = U D V'
#
# Principal component scores:
# Z = X V = U D

svd_X <- svd(
  X_scaled
)

U <- svd_X$u
D <- svd_X$d
V <- svd_X$v

dim(U)
length(D)
dim(V)


# ------------------------------------------------------------------------------
# 6. Principal Component Scores
# ------------------------------------------------------------------------------

Z <- X_scaled %*% V

colnames(Z) <- paste0(
  "PC",
  1:p
)

head(Z)


# ------------------------------------------------------------------------------
# 7. Verify SVD Relationship
# ------------------------------------------------------------------------------

Z_svd <- U %*% diag(D)

max(
  abs(
    Z -
      Z_svd
  )
)


# ------------------------------------------------------------------------------
# 8. Variance Explained
# ------------------------------------------------------------------------------

eigenvalues <- D^2 / (n - 1)

proportion_variance <- eigenvalues /
  sum(eigenvalues)

cumulative_variance <- cumsum(
  proportion_variance
)

variance_table <- data.frame(
  Component = paste0(
    "PC",
    1:p
  ),
  Eigenvalue = eigenvalues,
  Proportion = proportion_variance,
  Cumulative = cumulative_variance
)

variance_table


# ------------------------------------------------------------------------------
# 9. Scree Plot
# ------------------------------------------------------------------------------

plot(
  1:p,
  proportion_variance,
  type = "b",
  pch = 19,
  xlab = "Principal Component",
  ylab = "Proportion of Variance Explained",
  main = "Scree Plot"
)


# ------------------------------------------------------------------------------
# 10. Cumulative Variance Explained
# ------------------------------------------------------------------------------

plot(
  1:p,
  cumulative_variance,
  type = "b",
  pch = 19,
  ylim = c(0, 1),
  xlab = "Number of Components",
  ylab = "Cumulative Variance Explained",
  main = "Cumulative Variance Explained"
)

abline(
  h = 0.90,
  lty = 2
)


# ------------------------------------------------------------------------------
# 11. Inspect Principal Component Loadings
# ------------------------------------------------------------------------------

loadings <- V

rownames(loadings) <- colnames(X)

colnames(loadings) <- paste0(
  "PC",
  1:p
)

loadings


# ------------------------------------------------------------------------------
# 12. PCR Fitting Function
# ------------------------------------------------------------------------------

pcr_fit <- function(
    X,
    y,
    M
) {
  
  svd_X <- svd(X)
  
  V <- svd_X$v
  
  Z <- X %*% V
  
  Z_M <- Z[
    ,
    1:M,
    drop = FALSE
  ]
  
  
  # Regression of y on first M principal components
  
  gamma_hat <- solve(
    crossprod(Z_M),
    crossprod(
      Z_M,
      y
    )
  )
  
  
  # Convert back to original standardized predictor space
  
  beta_hat <- V[
    ,
    1:M,
    drop = FALSE
  ] %*%
    gamma_hat
  
  
  fitted <- X %*%
    beta_hat
  
  
  return(
    list(
      beta = as.vector(beta_hat),
      gamma = as.vector(gamma_hat),
      fitted = as.vector(fitted),
      loadings = V,
      scores = Z
    )
  )
}


# ------------------------------------------------------------------------------
# 13. Fit PCR with a Fixed Number of Components
# ------------------------------------------------------------------------------

M <- 3

pcr_model <- pcr_fit(
  X_scaled,
  y_centered,
  M
)

pcr_model$beta
pcr_model$gamma


# ------------------------------------------------------------------------------
# 14. Convert Coefficients Back to Original Scale
# ------------------------------------------------------------------------------

beta_pcr <- pcr_model$beta /
  X_sds

intercept_pcr <- y_mean -
  sum(
    beta_pcr *
      X_means
  )

intercept_pcr
beta_pcr


# ------------------------------------------------------------------------------
# 15. Predictions
# ------------------------------------------------------------------------------

y_hat <- intercept_pcr +
  X %*%
  beta_pcr

y_hat <- as.vector(
  y_hat
)

head(y_hat)


# ------------------------------------------------------------------------------
# 16. Training Error
# ------------------------------------------------------------------------------

residuals <- y -
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
# 17. Fit PCR for All Possible Numbers of Components
# ------------------------------------------------------------------------------

training_mse <- numeric(
  p
)

beta_path <- matrix(
  NA,
  nrow = p,
  ncol = p
)

colnames(beta_path) <- colnames(X)

for (M in 1:p) {
  
  fit <- pcr_fit(
    X_scaled,
    y_centered,
    M
  )
  
  training_mse[M] <- mean(
    (
      y_centered -
        fit$fitted
    )^2
  )
  
  beta_path[M, ] <- fit$beta
}


# ------------------------------------------------------------------------------
# 18. Plot Training Error
# ------------------------------------------------------------------------------

plot(
  1:p,
  training_mse,
  type = "b",
  pch = 19,
  xlab = "Number of Principal Components",
  ylab = "Training MSE",
  main = "PCR Training Error"
)


# ------------------------------------------------------------------------------
# 19. Coefficient Paths
# ------------------------------------------------------------------------------

matplot(
  1:p,
  beta_path,
  type = "l",
  lty = 1,
  xlab = "Number of Principal Components",
  ylab = "Standardized Coefficient",
  main = "PCR Coefficient Paths"
)

legend(
  "topright",
  legend = colnames(X),
  lty = 1,
  cex = 0.7
)


# ------------------------------------------------------------------------------
# 20. Compare PCR with OLS
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
  PCR = beta_pcr
)

comparison


# ------------------------------------------------------------------------------
# 21. Verify Full PCR Equals OLS
# ------------------------------------------------------------------------------

full_pcr <- pcr_fit(
  X_scaled,
  y_centered,
  M = p
)

beta_full_pcr <- full_pcr$beta

beta_ols_scaled <- solve(
  crossprod(X_scaled),
  crossprod(
    X_scaled,
    y_centered
  )
)

max(
  abs(
    beta_full_pcr -
      beta_ols_scaled
  )
)


# ------------------------------------------------------------------------------
# 22. Train-Test Split
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
# 23. Standardize Using Training Data Only
# ------------------------------------------------------------------------------

train_means <- colMeans(
  X_train
)

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

y_train_mean <- mean(
  y_train
)

y_train_centered <- y_train -
  y_train_mean


# ------------------------------------------------------------------------------
# 24. PCA on Training Data Only
# ------------------------------------------------------------------------------

svd_train <- svd(
  X_train_scaled
)

V_train <- svd_train$v

Z_train <- X_train_scaled %*%
  V_train

Z_test <- X_test_scaled %*%
  V_train


# ------------------------------------------------------------------------------
# 25. Test Error for Different Numbers of Components
# ------------------------------------------------------------------------------

test_mse <- numeric(
  p
)

for (M in 1:p) {
  
  Z_train_M <- Z_train[
    ,
    1:M,
    drop = FALSE
  ]
  
  Z_test_M <- Z_test[
    ,
    1:M,
    drop = FALSE
  ]
  
  gamma_hat <- solve(
    crossprod(Z_train_M),
    crossprod(
      Z_train_M,
      y_train_centered
    )
  )
  
  prediction <- y_train_mean +
    Z_test_M %*%
    gamma_hat
  
  test_mse[M] <- mean(
    (
      y_test -
        prediction
    )^2
  )
}


# ------------------------------------------------------------------------------
# 26. Plot Test Error
# ------------------------------------------------------------------------------

plot(
  1:p,
  test_mse,
  type = "b",
  pch = 19,
  xlab = "Number of Principal Components",
  ylab = "Test MSE",
  main = "PCR Test Error"
)


# ------------------------------------------------------------------------------
# 27. Best Number of Components
# ------------------------------------------------------------------------------

best_M <- which.min(
  test_mse
)

best_M

test_mse[
  best_M
]


# ------------------------------------------------------------------------------
# 28. K-Fold Cross-Validation Function
# ------------------------------------------------------------------------------

pcr_cv <- function(
    X,
    y,
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
  
  cv_error <- matrix(
    NA,
    nrow = p,
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
    
    
    # PCA must also be fit only on training fold
    
    svd_train <- svd(
      X_train_scaled
    )
    
    V <- svd_train$v
    
    Z_train <- X_train_scaled %*%
      V
    
    Z_valid <- X_valid_scaled %*%
      V
    
    
    for (M in 1:p) {
      
      Z_train_M <- Z_train[
        ,
        1:M,
        drop = FALSE
      ]
      
      Z_valid_M <- Z_valid[
        ,
        1:M,
        drop = FALSE
      ]
      
      
      gamma_hat <- solve(
        crossprod(Z_train_M),
        crossprod(
          Z_train_M,
          y_centered
        )
      )
      
      
      prediction <- y_mean +
        Z_valid_M %*%
        gamma_hat
      
      
      cv_error[M, fold] <- mean(
        (
          y_valid -
            prediction
        )^2
      )
    }
  }
  
  
  return(
    list(
      mean_cv_error = rowMeans(
        cv_error
      ),
      fold_errors = cv_error
    )
  )
}


# ------------------------------------------------------------------------------
# 29. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)

cv_results <- pcr_cv(
  X,
  y,
  k = 10
)

cv_results$mean_cv_error


# ------------------------------------------------------------------------------
# 30. Plot Cross-Validation Error
# ------------------------------------------------------------------------------

plot(
  1:p,
  cv_results$mean_cv_error,
  type = "b",
  pch = 19,
  xlab = "Number of Principal Components",
  ylab = "Cross-Validated MSE",
  main = "PCR Cross-Validation"
)


# ------------------------------------------------------------------------------
# 31. Select Number of Components by CV
# ------------------------------------------------------------------------------

cv_best_M <- which.min(
  cv_results$mean_cv_error
)

cv_best_M

cv_results$mean_cv_error[
  cv_best_M
]


# ------------------------------------------------------------------------------
# 32. Fit Final PCR Model
# ------------------------------------------------------------------------------

X_means_final <- colMeans(
  X
)

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

y_mean_final <- mean(
  y
)

y_centered_final <- y -
  y_mean_final


final_fit <- pcr_fit(
  X_scaled_final,
  y_centered_final,
  M = cv_best_M
)

beta_final_scaled <- final_fit$beta

beta_final <- beta_final_scaled /
  X_sds_final

intercept_final <- y_mean_final -
  sum(
    beta_final *
      X_means_final
  )

intercept_final
beta_final


# ------------------------------------------------------------------------------
# 33. Final Coefficient Comparison
# ------------------------------------------------------------------------------

final_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  OLS = beta_ols,
  PCR = beta_final
)

final_comparison


# ------------------------------------------------------------------------------
# 34. Visualize Final Coefficients
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
  main = "True vs OLS vs PCR"
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
    "PCR"
  ),
  pch = 19,
  lty = 1
)


# ------------------------------------------------------------------------------
# 35. Inspect Selected Principal Components
# ------------------------------------------------------------------------------

selected_loadings <- loadings[
  ,
  1:cv_best_M,
  drop = FALSE
]

selected_loadings


# ------------------------------------------------------------------------------
# 36. Variance Explained by Selected Components
# ------------------------------------------------------------------------------

sum(
  proportion_variance[
    1:cv_best_M
  ]
)