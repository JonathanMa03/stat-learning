# ==============================================================================
# Piecewise Linear Path Algorithms
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Focus:
#   - Lasso solution paths
#   - Regularization parameter lambda
#   - Active sets
#   - Knots in the solution path
#   - Piecewise linear coefficient trajectories
#   - Comparison with ridge regression
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 150
p <- 6


# Create some correlated predictors

z1 <- rnorm(n)
z2 <- rnorm(n)

x1 <- z1 + rnorm(n, sd = 0.25)
x2 <- 0.8 * z1 + rnorm(n, sd = 0.25)

x3 <- z2 + rnorm(n, sd = 0.25)
x4 <- 0.7 * z2 + rnorm(n, sd = 0.25)

x5 <- rnorm(n)
x6 <- rnorm(n)


X <- cbind(
  x1,
  x2,
  x3,
  x4,
  x5,
  x6
)

colnames(X) <- paste0(
  "x",
  1:p
)


beta_true <- c(
  3,
  2,
  -2.5,
  0,
  1.5,
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
  X %*%
  beta_true +
  epsilon

y <- as.vector(y)


data <- data.frame(
  y = y,
  X
)

head(data)

cor(X)


# ------------------------------------------------------------------------------
# 2. Standardize Predictors
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
# 3. Center Response
# ------------------------------------------------------------------------------

y_mean <- mean(y)

y_centered <- y -
  y_mean


# ------------------------------------------------------------------------------
# 4. Soft-Thresholding Operator
# ------------------------------------------------------------------------------

soft_threshold <- function(
    z,
    gamma
) {
  
  sign(z) *
    pmax(
      abs(z) - gamma,
      0
    )
}


# ------------------------------------------------------------------------------
# 5. Lasso Coordinate Descent
# ------------------------------------------------------------------------------

lasso_fit <- function(
    X,
    y,
    lambda,
    beta_init = NULL,
    max_iter = 10000,
    tol = 1e-10
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
  
  
  # Squared norm of each predictor
  
  z <- colSums(
    X^2
  ) / n
  
  
  for (iteration in 1:max_iter) {
    
    beta_old <- beta
    
    
    for (j in 1:p) {
      
      # Partial residual
      
      residual_j <-
        y -
        X %*% beta +
        X[, j] * beta[j]
      
      
      # Partial correlation
      
      rho_j <- sum(
        X[, j] *
          residual_j
      ) / n
      
      
      # Lasso coordinate update
      
      beta[j] <-
        soft_threshold(
          rho_j,
          lambda
        ) /
        z[j]
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
  
  
  list(
    coefficients = beta,
    iterations = iteration
  )
}


# ------------------------------------------------------------------------------
# 6. Maximum Lambda
# ------------------------------------------------------------------------------

# lambda_max is the smallest lambda for which
# every coefficient is exactly zero.

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
# 7. Construct Lambda Grid
# ------------------------------------------------------------------------------

lambda_min <-
  lambda_max * 0.001


lambda_grid <- exp(
  seq(
    log(lambda_max),
    log(lambda_min),
    length.out = 500
  )
)


head(lambda_grid)

tail(lambda_grid)


# ------------------------------------------------------------------------------
# 8. Compute the Lasso Solution Path
# ------------------------------------------------------------------------------

beta_path <- matrix(
  0,
  nrow = length(lambda_grid),
  ncol = p
)

colnames(beta_path) <-
  colnames(X)


# Warm starts make path algorithms much faster.

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
  
  
  beta_path[i, ] <-
    fit$coefficients
  
  
  beta_previous <-
    fit$coefficients
}


# ------------------------------------------------------------------------------
# 9. Inspect Beginning of Path
# ------------------------------------------------------------------------------

head(
  beta_path
)


# ------------------------------------------------------------------------------
# 10. Plot Coefficient Path Against log(lambda)
# ------------------------------------------------------------------------------

matplot(
  log(lambda_grid),
  beta_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Standardized Coefficient",
  main = "Piecewise Linear Lasso Path"
)

abline(
  h = 0,
  lty = 2
)

legend(
  "topright",
  legend = colnames(X),
  lty = 1,
  cex = 0.8
)


# ------------------------------------------------------------------------------
# 11. L1 Norm Along Path
# ------------------------------------------------------------------------------

L1_norm <- apply(
  beta_path,
  1,
  function(beta) {
    
    sum(
      abs(beta)
    )
  }
)


# ------------------------------------------------------------------------------
# 12. Plot Path Against L1 Norm
# ------------------------------------------------------------------------------

matplot(
  L1_norm,
  beta_path,
  type = "l",
  lty = 1,
  xlab = "L1 Norm",
  ylab = "Standardized Coefficient",
  main = "Lasso Path vs L1 Norm"
)

abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 13. Determine Active Set
# ------------------------------------------------------------------------------

tolerance <- 1e-6


active_matrix <- abs(
  beta_path
) > tolerance


head(
  active_matrix
)


# ------------------------------------------------------------------------------
# 14. Number of Active Variables
# ------------------------------------------------------------------------------

active_count <- rowSums(
  active_matrix
)


plot(
  log(lambda_grid),
  active_count,
  type = "s",
  xlab = "log(lambda)",
  ylab = "Number of Active Variables",
  main = "Active Set Along Lasso Path"
)


# ------------------------------------------------------------------------------
# 15. Detect Knots
# ------------------------------------------------------------------------------

# A knot occurs when the active set changes.

knot_index <- c(1)


for (i in 2:nrow(active_matrix)) {
  
  if (
    any(
      active_matrix[i, ] !=
      active_matrix[i - 1, ]
    )
  ) {
    
    knot_index <- c(
      knot_index,
      i
    )
  }
}


knot_index


# ------------------------------------------------------------------------------
# 16. Lambda Values at Knots
# ------------------------------------------------------------------------------

knot_lambda <-
  lambda_grid[
    knot_index
  ]


knot_lambda


# ------------------------------------------------------------------------------
# 17. Coefficients at Knots
# ------------------------------------------------------------------------------

knot_coefficients <-
  beta_path[
    knot_index,
    ,
    drop = FALSE
  ]


knot_table <- data.frame(
  Knot = seq_along(
    knot_index
  ),
  Lambda = knot_lambda,
  knot_coefficients
)


knot_table


# ------------------------------------------------------------------------------
# 18. Active Set at Each Knot
# ------------------------------------------------------------------------------

for (k in seq_along(knot_index)) {
  
  index <- knot_index[k]
  
  
  active_variables <- colnames(X)[
    active_matrix[
      index,
    ]
  ]
  
  
  cat(
    "Knot",
    k,
    "\n"
  )
  
  cat(
    "Lambda:",
    lambda_grid[index],
    "\n"
  )
  
  cat(
    "Active variables:",
    active_variables,
    "\n\n"
  )
}


# ------------------------------------------------------------------------------
# 19. Variable Entry Lambda
# ------------------------------------------------------------------------------

entry_lambda <- rep(
  NA,
  p
)

names(entry_lambda) <-
  colnames(X)


for (j in 1:p) {
  
  first_active <- which(
    active_matrix[, j]
  )
  
  
  if (
    length(first_active) > 0
  ) {
    
    entry_lambda[j] <-
      lambda_grid[
        first_active[1]
      ]
  }
}


entry_lambda


# ------------------------------------------------------------------------------
# 20. Variable Entry Order
# ------------------------------------------------------------------------------

entry_order <- order(
  entry_lambda,
  decreasing = TRUE,
  na.last = TRUE
)


data.frame(
  Predictor = names(
    entry_lambda
  )[entry_order],
  
  Entry_Lambda = entry_lambda[
    entry_order
  ]
)


# ------------------------------------------------------------------------------
# 21. Plot Knots on Coefficient Path
# ------------------------------------------------------------------------------

matplot(
  log(lambda_grid),
  beta_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Standardized Coefficient",
  main = "Lasso Path with Estimated Knots"
)


for (lambda in knot_lambda) {
  
  abline(
    v = log(lambda),
    lty = 3
  )
}


abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 22. Examine a Single Coefficient Path
# ------------------------------------------------------------------------------

j <- 1


plot(
  lambda_grid,
  beta_path[, j],
  type = "l",
  xlab = "lambda",
  ylab = paste0(
    "Coefficient for ",
    colnames(X)[j]
  ),
  main = "Single Lasso Coefficient Path"
)

abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 23. Reverse Lambda Axis
# ------------------------------------------------------------------------------

# This makes the plot easier to interpret:
#
# left = stronger regularization
# right = weaker regularization

plot(
  rev(lambda_grid),
  rev(beta_path[, j]),
  type = "l",
  xlab = "Decreasing Regularization",
  ylab = paste0(
    "Coefficient for ",
    colnames(X)[j]
  ),
  main = "Coefficient Evolution"
)


# ------------------------------------------------------------------------------
# 24. Approximate Path Slopes
# ------------------------------------------------------------------------------

# Between knots, the coefficient path should behave
# approximately linearly as a function of lambda.


path_slopes <- matrix(
  NA,
  nrow = length(lambda_grid) - 1,
  ncol = p
)

colnames(path_slopes) <-
  colnames(X)


for (j in 1:p) {
  
  path_slopes[, j] <-
    diff(
      beta_path[, j]
    ) /
    diff(
      lambda_grid
    )
}


head(
  path_slopes
)


# ------------------------------------------------------------------------------
# 25. Plot Local Path Slope
# ------------------------------------------------------------------------------

j <- 1


plot(
  lambda_grid[-1],
  path_slopes[, j],
  type = "l",
  xlab = "lambda",
  ylab = "Local Path Slope",
  main = paste(
    "Path Slope:",
    colnames(X)[j]
  )
)


# ------------------------------------------------------------------------------
# 26. Examine Linear Segments Between Knots
# ------------------------------------------------------------------------------

# Because the lambda grid is numerical rather than exact,
# the segments are approximately rather than perfectly linear.


if (length(knot_index) >= 2) {
  
  segment_start <-
    knot_index[1]
  
  segment_end <-
    knot_index[2]
  
  
  if (
    segment_end -
    segment_start >
    2
  ) {
    
    lambda_segment <-
      lambda_grid[
        segment_start:segment_end
      ]
    
    
    beta_segment <-
      beta_path[
        segment_start:segment_end,
        1
      ]
    
    
    segment_model <- lm(
      beta_segment ~
        lambda_segment
    )
    
    
    summary(
      segment_model
    )
    
    
    plot(
      lambda_segment,
      beta_segment,
      pch = 19,
      xlab = "lambda",
      ylab = "Coefficient",
      main = "Linear Segment Between Knots"
    )
    
    
    abline(
      segment_model
    )
  }
}


# ------------------------------------------------------------------------------
# 27. Residual Sum of Squares Along Path
# ------------------------------------------------------------------------------

RSS_path <- numeric(
  length(lambda_grid)
)


for (i in seq_along(lambda_grid)) {
  
  prediction <-
    X_scaled %*%
    beta_path[i, ]
  
  
  residuals <-
    y_centered -
    prediction
  
  
  RSS_path[i] <-
    sum(
      residuals^2
    )
}


# ------------------------------------------------------------------------------
# 28. Plot RSS Along Path
# ------------------------------------------------------------------------------

plot(
  log(lambda_grid),
  RSS_path,
  type = "l",
  xlab = "log(lambda)",
  ylab = "RSS",
  main = "Training RSS Along Regularization Path"
)


# ------------------------------------------------------------------------------
# 29. Compare with Ordinary Least Squares
# ------------------------------------------------------------------------------

beta_ols_scaled <- solve(
  crossprod(
    X_scaled
  ),
  crossprod(
    X_scaled,
    y_centered
  )
)

beta_ols_scaled <-
  as.vector(
    beta_ols_scaled
  )


comparison <- data.frame(
  Predictor = colnames(X),
  OLS = beta_ols_scaled,
  Lasso_End = beta_path[
    nrow(beta_path),
  ]
)


comparison


# ------------------------------------------------------------------------------
# 30. Ridge Path for Comparison
# ------------------------------------------------------------------------------

ridge_fit <- function(
    X,
    y,
    lambda
) {
  
  p <- ncol(X)
  
  
  beta <- solve(
    crossprod(X) +
      lambda *
      diag(p),
    crossprod(
      X,
      y
    )
  )
  
  
  as.vector(beta)
}


ridge_path <- matrix(
  NA,
  nrow = length(lambda_grid),
  ncol = p
)

colnames(ridge_path) <-
  colnames(X)


for (i in seq_along(lambda_grid)) {
  
  ridge_path[i, ] <-
    ridge_fit(
      X_scaled,
      y_centered,
      lambda_grid[i]
    )
}


# ------------------------------------------------------------------------------
# 31. Plot Ridge Path
# ------------------------------------------------------------------------------

matplot(
  log(lambda_grid),
  ridge_path,
  type = "l",
  lty = 1,
  xlab = "log(lambda)",
  ylab = "Standardized Coefficient",
  main = "Ridge Coefficient Path"
)

abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 32. Compare Lasso and Ridge Sparsity
# ------------------------------------------------------------------------------

lasso_nonzero <- apply(
  beta_path,
  1,
  function(beta) {
    
    sum(
      abs(beta) > tolerance
    )
  }
)


ridge_nonzero <- apply(
  ridge_path,
  1,
  function(beta) {
    
    sum(
      abs(beta) > tolerance
    )
  }
)


plot(
  log(lambda_grid),
  lasso_nonzero,
  type = "s",
  ylim = c(0, p),
  xlab = "log(lambda)",
  ylab = "Number of Nonzero Coefficients",
  main = "Lasso vs Ridge Sparsity"
)

lines(
  log(lambda_grid),
  ridge_nonzero,
  type = "s",
  lty = 2
)

legend(
  "bottomleft",
  legend = c(
    "Lasso",
    "Ridge"
  ),
  lty = c(
    1,
    2
  )
)


# ------------------------------------------------------------------------------
# 33. Train-Test Split
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
# 34. Standardize Using Training Data Only
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

y_train_centered <-
  y_train -
  y_train_mean


# ------------------------------------------------------------------------------
# 35. Training Lambda Grid
# ------------------------------------------------------------------------------

n_train <- nrow(
  X_train
)


lambda_max_train <- max(
  abs(
    as.vector(
      crossprod(
        X_train_scaled,
        y_train_centered
      )
    )
  )
) / n_train


lambda_grid_train <- exp(
  seq(
    log(lambda_max_train),
    log(
      lambda_max_train *
        0.001
    ),
    length.out = 300
  )
)


# ------------------------------------------------------------------------------
# 36. Compute Training Path
# ------------------------------------------------------------------------------

train_path <- matrix(
  0,
  nrow = length(lambda_grid_train),
  ncol = p
)


beta_previous <- rep(
  0,
  p
)


for (i in seq_along(
  lambda_grid_train
)) {
  
  fit <- lasso_fit(
    X_train_scaled,
    y_train_centered,
    lambda_grid_train[i],
    beta_init = beta_previous
  )
  
  
  train_path[i, ] <-
    fit$coefficients
  
  
  beta_previous <-
    fit$coefficients
}


# ------------------------------------------------------------------------------
# 37. Test MSE Along Path
# ------------------------------------------------------------------------------

test_mse <- numeric(
  length(lambda_grid_train)
)


for (i in seq_along(
  lambda_grid_train
)) {
  
  prediction <-
    y_train_mean +
    X_test_scaled %*%
    train_path[i, ]
  
  
  test_mse[i] <- mean(
    (
      y_test -
        prediction
    )^2
  )
}


# ------------------------------------------------------------------------------
# 38. Plot Test Error
# ------------------------------------------------------------------------------

plot(
  log(lambda_grid_train),
  test_mse,
  type = "l",
  xlab = "log(lambda)",
  ylab = "Test MSE",
  main = "Prediction Error Along Lasso Path"
)


# ------------------------------------------------------------------------------
# 39. Best Lambda
# ------------------------------------------------------------------------------

best_index <- which.min(
  test_mse
)

best_lambda <-
  lambda_grid_train[
    best_index
  ]

best_lambda

test_mse[
  best_index
]


# ------------------------------------------------------------------------------
# 40. Coefficients at Best Lambda
# ------------------------------------------------------------------------------

best_beta_scaled <-
  train_path[
    best_index,
  ]


best_beta <-
  best_beta_scaled /
  train_sds


best_intercept <-
  y_train_mean -
  sum(
    best_beta *
      train_means
  )


best_intercept

best_beta


# ------------------------------------------------------------------------------
# 41. Compare with True Coefficients
# ------------------------------------------------------------------------------

final_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  Estimated = best_beta
)

final_comparison


# ------------------------------------------------------------------------------
# 42. Active Variables at Best Lambda
# ------------------------------------------------------------------------------

selected <- abs(
  best_beta
) > tolerance


colnames(X)[
  selected
]


# ------------------------------------------------------------------------------
# 43. Summary of Estimated Knots
# ------------------------------------------------------------------------------

cat(
  "Number of estimated knots:",
  length(knot_index),
  "\n"
)

cat(
  "Number of predictors:",
  p,
  "\n\n"
)


for (k in seq_along(
  knot_index
)) {
  
  i <- knot_index[k]
  
  active_variables <-
    colnames(X)[
      active_matrix[
        i,
      ]
    ]
  
  
  cat(
    "Knot",
    k,
    "\n"
  )
  
  cat(
    "Lambda:",
    round(
      lambda_grid[i],
      4
    ),
    "\n"
  )
  
  cat(
    "Active set:",
    active_variables,
    "\n\n"
  )
}