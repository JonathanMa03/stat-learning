# ==============================================================================
# Least Angle Regression
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
  0,
  0,
  1.5,
  0,
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
# 2. Standardize Predictors
# ------------------------------------------------------------------------------

# LARS assumes centered and standardized predictors.

X_means <- colMeans(X)

X_centered <- sweep(
  X,
  2,
  X_means,
  "-"
)

X_norms <- sqrt(
  colSums(
    X_centered^2
  )
)

X_scaled <- sweep(
  X_centered,
  2,
  X_norms,
  "/"
)

colMeans(X_scaled)

sqrt(
  colSums(
    X_scaled^2
  )
)


# ------------------------------------------------------------------------------
# 3. Center Response
# ------------------------------------------------------------------------------

y_mean <- mean(y)

y_centered <- y - y_mean


# ------------------------------------------------------------------------------
# 4. Initial State
# ------------------------------------------------------------------------------

beta <- rep(
  0,
  p
)

names(beta) <- colnames(X)

mu <- rep(
  0,
  n
)

active <- integer(0)

inactive <- 1:p


# Store coefficient path

beta_path <- matrix(
  0,
  nrow = 1,
  ncol = p
)

colnames(beta_path) <- colnames(X)

active_path <- list(
  character(0)
)


# ------------------------------------------------------------------------------
# 5. Least Angle Regression Function
# ------------------------------------------------------------------------------

lar_fit <- function(
    X,
    y,
    max_steps = ncol(X),
    tol = 1e-10
) {
  
  n <- nrow(X)
  p <- ncol(X)
  
  beta <- rep(
    0,
    p
  )
  
  mu <- rep(
    0,
    n
  )
  
  active <- integer(0)
  
  beta_path <- matrix(
    beta,
    nrow = 1
  )
  
  active_path <- list(
    character(0)
  )
  
  
  for (step in 1:max_steps) {
    
    # --------------------------------------------------------------------------
    # Current Residual
    # --------------------------------------------------------------------------
    
    residual <- y - mu
    
    
    # --------------------------------------------------------------------------
    # Correlations
    # --------------------------------------------------------------------------
    
    correlations <- as.vector(
      crossprod(
        X,
        residual
      )
    )
    
    C <- max(
      abs(correlations)
    )
    
    
    if (C < tol) {
      break
    }
    
    
    # --------------------------------------------------------------------------
    # Add Most Correlated Predictor
    # --------------------------------------------------------------------------
    
    inactive <- setdiff(
      1:p,
      active
    )
    
    if (length(active) == 0) {
      
      new_variable <- which.max(
        abs(correlations)
      )
      
      active <- c(
        active,
        new_variable
      )
      
    } else {
      
      # Add any variable tied with maximum correlation
      
      candidates <- inactive[
        abs(
          abs(correlations[inactive]) -
            C
        ) < tol
      ]
      
      if (length(candidates) > 0) {
        
        active <- c(
          active,
          candidates[1]
        )
      }
    }
    
    
    # --------------------------------------------------------------------------
    # Signs of Active Correlations
    # --------------------------------------------------------------------------
    
    signs <- sign(
      correlations[active]
    )
    
    signs[
      signs == 0
    ] <- 1
    
    
    # --------------------------------------------------------------------------
    # Signed Active Design Matrix
    # --------------------------------------------------------------------------
    
    X_active <- sweep(
      X[, active, drop = FALSE],
      2,
      signs,
      "*"
    )
    
    
    # --------------------------------------------------------------------------
    # Gram Matrix
    # --------------------------------------------------------------------------
    
    G_active <- crossprod(
      X_active
    )
    
    
    G_inv <- solve(
      G_active
    )
    
    
    # --------------------------------------------------------------------------
    # Equiangular Direction
    # --------------------------------------------------------------------------
    
    ones <- rep(
      1,
      length(active)
    )
    
    A <- sqrt(
      1 /
        as.numeric(
          t(ones) %*%
            G_inv %*%
            ones
        )
    )
    
    
    w <- as.vector(
      A *
        G_inv %*%
        ones
    )
    
    
    u <- as.vector(
      X_active %*%
        w
    )
    
    
    # --------------------------------------------------------------------------
    # Correlation of All Predictors with Equiangular Direction
    # --------------------------------------------------------------------------
    
    a <- as.vector(
      crossprod(
        X,
        u
      )
    )
    
    
    # --------------------------------------------------------------------------
    # Determine Step Size
    # --------------------------------------------------------------------------
    
    inactive <- setdiff(
      1:p,
      active
    )
    
    
    if (length(inactive) == 0) {
      
      gamma <- C / A
      
    } else {
      
      gamma_candidates <- numeric(0)
      
      
      for (j in inactive) {
        
        candidate_1 <- (
          C - correlations[j]
        ) / (
          A - a[j]
        )
        
        candidate_2 <- (
          C + correlations[j]
        ) / (
          A + a[j]
        )
        
        
        if (
          is.finite(candidate_1) &&
          candidate_1 > tol
        ) {
          
          gamma_candidates <- c(
            gamma_candidates,
            candidate_1
          )
        }
        
        
        if (
          is.finite(candidate_2) &&
          candidate_2 > tol
        ) {
          
          gamma_candidates <- c(
            gamma_candidates,
            candidate_2
          )
        }
      }
      
      
      if (length(gamma_candidates) == 0) {
        
        gamma <- C / A
        
      } else {
        
        gamma <- min(
          gamma_candidates,
          C / A
        )
      }
    }
    
    
    # --------------------------------------------------------------------------
    # Update Fitted Values
    # --------------------------------------------------------------------------
    
    mu <- mu +
      gamma * u
    
    
    # --------------------------------------------------------------------------
    # Update Coefficients
    # --------------------------------------------------------------------------
    
    beta_update <- gamma *
      w *
      signs
    
    beta[active] <- beta[active] +
      beta_update
    
    
    # --------------------------------------------------------------------------
    # Store Path
    # --------------------------------------------------------------------------
    
    beta_path <- rbind(
      beta_path,
      beta
    )
    
    active_path[[length(active_path) + 1]] <- colnames(X)[active]
  }
  
  
  colnames(beta_path) <-
    colnames(X)
  
  
  return(
    list(
      coefficients = beta,
      beta_path = beta_path,
      active_path = active_path,
      fitted = mu,
      residuals = y - mu
    )
  )
}


# ------------------------------------------------------------------------------
# 6. Fit Least Angle Regression
# ------------------------------------------------------------------------------

lar_model <- lar_fit(
  X_scaled,
  y_centered,
  max_steps = p
)

lar_model$coefficients

lar_model$active_path


# ------------------------------------------------------------------------------
# 7. Coefficient Path
# ------------------------------------------------------------------------------

beta_path <-
  lar_model$beta_path

beta_path


# ------------------------------------------------------------------------------
# 8. Plot Coefficient Path by Step
# ------------------------------------------------------------------------------

matplot(
  0:(nrow(beta_path) - 1),
  beta_path,
  type = "l",
  lty = 1,
  xlab = "LARS Step",
  ylab = "Standardized Coefficient",
  main = "Least Angle Regression Path"
)

abline(
  h = 0,
  lty = 2
)

legend(
  "topleft",
  legend = colnames(X),
  lty = 1,
  cex = 0.7
)


# ------------------------------------------------------------------------------
# 9. Plot Coefficient Path by L1 Norm
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

matplot(
  L1_norm,
  beta_path,
  type = "l",
  lty = 1,
  xlab = "L1 Norm",
  ylab = "Standardized Coefficient",
  main = "LARS Coefficient Path"
)

abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 10. Variable Entry Order
# ------------------------------------------------------------------------------

entry_order <- character(0)

for (i in 2:length(
  lar_model$active_path
)) {
  
  previous <- lar_model$active_path[[i - 1]]
  
  current <- lar_model$active_path[[i]]
  
  new_variable <- setdiff(
    current,
    previous
  )
  
  if (length(new_variable) > 0) {
    
    entry_order <- c(
      entry_order,
      new_variable
    )
  }
}

entry_order


# ------------------------------------------------------------------------------
# 11. Inspect Active Set by Step
# ------------------------------------------------------------------------------

for (i in seq_along(
  lar_model$active_path
)) {
  
  cat(
    "Step",
    i - 1,
    ":",
    lar_model$active_path[[i]],
    "\n"
  )
}


# ------------------------------------------------------------------------------
# 12. Convert Final Coefficients Back to Original Scale
# ------------------------------------------------------------------------------

beta_lar_scaled <-
  lar_model$coefficients

beta_lar <-
  beta_lar_scaled /
  X_norms

intercept_lar <-
  y_mean -
  sum(
    beta_lar *
      X_means
  )

intercept_lar
beta_lar


# ------------------------------------------------------------------------------
# 13. Predictions
# ------------------------------------------------------------------------------

y_hat <-
  intercept_lar +
  X %*%
  beta_lar

y_hat <- as.vector(
  y_hat
)

head(y_hat)


# ------------------------------------------------------------------------------
# 14. Training Error
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
# 15. Compare with Ordinary Least Squares
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
  LAR = beta_lar
)

comparison


# ------------------------------------------------------------------------------
# 16. Compute Predictions at Each LARS Step
# ------------------------------------------------------------------------------

path_predictions <- matrix(
  NA,
  nrow = n,
  ncol = nrow(beta_path)
)

for (i in 1:nrow(beta_path)) {
  
  path_predictions[, i] <-
    y_mean +
    X_scaled %*%
    beta_path[i, ]
}


# ------------------------------------------------------------------------------
# 17. Training MSE Along the Path
# ------------------------------------------------------------------------------

path_mse <- numeric(
  nrow(beta_path)
)

for (i in 1:nrow(beta_path)) {
  
  path_mse[i] <- mean(
    (
      y -
        path_predictions[, i]
    )^2
  )
}

path_mse


# ------------------------------------------------------------------------------
# 18. Plot Training Error Along Path
# ------------------------------------------------------------------------------

plot(
  0:(length(path_mse) - 1),
  path_mse,
  type = "b",
  pch = 19,
  xlab = "LARS Step",
  ylab = "Training MSE",
  main = "Training Error Along LARS Path"
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
# 20. Standardize Using Training Data
# ------------------------------------------------------------------------------

train_means <-
  colMeans(X_train)

X_train_centered <- sweep(
  X_train,
  2,
  train_means,
  "-"
)

train_norms <- sqrt(
  colSums(
    X_train_centered^2
  )
)

X_train_scaled <- sweep(
  X_train_centered,
  2,
  train_norms,
  "/"
)

X_test_scaled <- sweep(
  X_test,
  2,
  train_means,
  "-"
)

X_test_scaled <- sweep(
  X_test_scaled,
  2,
  train_norms,
  "/"
)

y_train_mean <-
  mean(y_train)

y_train_centered <-
  y_train -
  y_train_mean


# ------------------------------------------------------------------------------
# 21. Fit LARS on Training Data
# ------------------------------------------------------------------------------

lar_train <- lar_fit(
  X_train_scaled,
  y_train_centered,
  max_steps = p
)

train_path <-
  lar_train$beta_path


# ------------------------------------------------------------------------------
# 22. Test MSE Along LARS Path
# ------------------------------------------------------------------------------

test_mse <- numeric(
  nrow(train_path)
)

for (i in 1:nrow(train_path)) {
  
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

test_mse


# ------------------------------------------------------------------------------
# 23. Plot Test MSE
# ------------------------------------------------------------------------------

plot(
  0:(length(test_mse) - 1),
  test_mse,
  type = "b",
  pch = 19,
  xlab = "LARS Step",
  ylab = "Test MSE",
  main = "LARS Test Error"
)


# ------------------------------------------------------------------------------
# 24. Select Best Step
# ------------------------------------------------------------------------------

best_step <- which.min(
  test_mse
)

# First row represents step 0

best_step_number <-
  best_step - 1

best_step_number

test_mse[
  best_step
]


# ------------------------------------------------------------------------------
# 25. Coefficients at Best Step
# ------------------------------------------------------------------------------

best_beta_scaled <-
  train_path[
    best_step,
  ]

best_beta_scaled


# ------------------------------------------------------------------------------
# 26. Convert Best Coefficients to Original Scale
# ------------------------------------------------------------------------------

best_beta <-
  best_beta_scaled /
  train_norms

best_intercept <-
  y_train_mean -
  sum(
    best_beta *
      train_means
  )

best_intercept
best_beta


# ------------------------------------------------------------------------------
# 27. Selected Predictors
# ------------------------------------------------------------------------------

selected <- abs(
  best_beta
) > 1e-10

colnames(X)[
  selected
]


# ------------------------------------------------------------------------------
# 28. Compare True and LARS Coefficients
# ------------------------------------------------------------------------------

best_comparison <- data.frame(
  Predictor = colnames(X),
  True = beta_true,
  LARS = best_beta
)

best_comparison


# ------------------------------------------------------------------------------
# 29. Visualize Coefficient Comparison
# ------------------------------------------------------------------------------

matplot(
  cbind(
    beta_true,
    best_beta
  ),
  type = "b",
  pch = 19,
  lty = 1,
  xaxt = "n",
  xlab = "Predictor",
  ylab = "Coefficient",
  main = "True vs LARS Coefficients"
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
    "LARS"
  ),
  pch = 19,
  lty = 1
)


# ------------------------------------------------------------------------------
# 30. Correlations During Initial Step
# ------------------------------------------------------------------------------

initial_correlations <- as.vector(
  crossprod(
    X_scaled,
    y_centered
  )
)

names(initial_correlations) <-
  colnames(X)

initial_correlations

initial_correlations[
  order(
    abs(initial_correlations),
    decreasing = TRUE
  )
]