# ==============================================================================
# Support Vector Regression
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Linear Support Vector Regression
#   - Epsilon-insensitive loss
#   - Regularization parameter C
#   - Epsilon tube
#   - Support vectors
#   - Subgradient optimization
#   - Cross-validation for C and epsilon
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 250

x <- runif(
  n,
  min = -3,
  max = 3
)

epsilon_noise <- rnorm(
  n,
  mean = 0,
  sd = 0.8
)

y <- 2 +
  1.5 * x +
  epsilon_noise


X <- matrix(
  x,
  ncol = 1
)

colnames(X) <- "x"


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
  main = "Support Vector Regression Data"
)


# ------------------------------------------------------------------------------
# 3. Standardize Predictor
# ------------------------------------------------------------------------------

X_mean <- mean(
  X[, 1]
)

X_sd <- sd(
  X[, 1]
)


X_scaled <- scale(
  X,
  center = X_mean,
  scale = X_sd
)

X_scaled <- as.matrix(
  X_scaled
)


# ------------------------------------------------------------------------------
# 4. Epsilon-Insensitive Loss
# ------------------------------------------------------------------------------

epsilon_loss <- function(
    residuals,
    epsilon
) {
  
  pmax(
    0,
    abs(residuals) - epsilon
  )
}


# Example

epsilon_loss(
  residuals = c(
    -2,
    -0.5,
    0,
    0.5,
    2
  ),
  epsilon = 1
)


# ------------------------------------------------------------------------------
# 5. SVR Objective Function
# ------------------------------------------------------------------------------

# We minimize:
#
#   1/2 ||w||^2
#   +
#   C * sum_i max(0, |y_i - f(x_i)| - epsilon)
#
# where:
#
#   f(x) = w'x + b


svr_objective <- function(
    X,
    y,
    w,
    b,
    C,
    epsilon
) {
  
  prediction <- as.vector(
    X %*% w + b
  )
  
  residuals <- y -
    prediction
  
  
  loss <- epsilon_loss(
    residuals,
    epsilon
  )
  
  
  objective <- 0.5 *
    sum(
      w^2
    ) +
    C *
    sum(loss)
  
  
  return(objective)
}


# ------------------------------------------------------------------------------
# 6. Linear SVR via Subgradient Descent
# ------------------------------------------------------------------------------

svr_fit <- function(
    X,
    y,
    C = 1,
    epsilon = 0.5,
    learning_rate = 0.001,
    max_iter = 20000,
    tol = 1e-8,
    decay = 1e-4
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  # Initialize parameters
  
  w <- rep(
    0,
    p
  )
  
  b <- mean(y)
  
  
  objective_history <- numeric(
    max_iter
  )
  
  
  for (iteration in seq_len(
    max_iter
  )) {
    
    prediction <- as.vector(
      X %*% w + b
    )
    
    
    residuals <- y -
      prediction
    
    
    # ------------------------------------------------------------
    # Outside epsilon tube
    # ------------------------------------------------------------
    
    above <- residuals > epsilon
    
    below <- residuals < -epsilon
    
    
    # ------------------------------------------------------------
    # Gradient of regularization
    # ------------------------------------------------------------
    
    grad_w <- w
    
    grad_b <- 0
    
    
    # ------------------------------------------------------------
    # Cases where:
    #
    # y_i - f_i > epsilon
    #
    # We need to increase prediction.
    # ------------------------------------------------------------
    
    if (
      any(above)
    ) {
      
      grad_w <- grad_w -
        C *
        colSums(
          X[
            above,
            ,
            drop = FALSE
          ]
        )
      
      grad_b <- grad_b -
        C *
        sum(above)
    }
    
    
    # ------------------------------------------------------------
    # Cases where:
    #
    # y_i - f_i < -epsilon
    #
    # We need to decrease prediction.
    # ------------------------------------------------------------
    
    if (
      any(below)
    ) {
      
      grad_w <- grad_w +
        C *
        colSums(
          X[
            below,
            ,
            drop = FALSE
          ]
        )
      
      grad_b <- grad_b +
        C *
        sum(below)
    }
    
    
    # Decaying learning rate
    
    eta <- learning_rate /
      (
        1 +
          decay *
          iteration
      )
    
    
    w_new <- w -
      eta *
      grad_w
    
    
    b_new <- b -
      eta *
      grad_b
    
    
    objective_history[iteration] <-
      svr_objective(
        X = X,
        y = y,
        w = w_new,
        b = b_new,
        C = C,
        epsilon = epsilon
      )
    
    
    parameter_change <- max(
      c(
        abs(
          w_new -
            w
        ),
        abs(
          b_new -
            b
        )
      )
    )
    
    
    w <- w_new
    
    b <- b_new
    
    
    if (
      parameter_change <
      tol
    ) {
      
      break
    }
  }
  
  
  objective_history <- objective_history[
    seq_len(iteration)
  ]
  
  
  prediction <- as.vector(
    X %*% w + b
  )
  
  
  residuals <- y -
    prediction
  
  
  return(
    list(
      weights = w,
      intercept = b,
      iterations = iteration,
      objective_history = objective_history,
      prediction = prediction,
      residuals = residuals,
      C = C,
      epsilon = epsilon
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Fit Support Vector Regression
# ------------------------------------------------------------------------------

set.seed(123)


svr_model <- svr_fit(
  X = X_scaled,
  y = y,
  C = 1,
  epsilon = 0.5,
  learning_rate = 0.001,
  max_iter = 20000
)


svr_model$weights

svr_model$intercept

svr_model$iterations


# ------------------------------------------------------------------------------
# 8. Prediction Function
# ------------------------------------------------------------------------------

svr_predict <- function(
    X_new,
    model
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  prediction <- as.vector(
    X_new %*%
      model$weights +
      model$intercept
  )
  
  
  return(prediction)
}


# ------------------------------------------------------------------------------
# 9. Training Predictions
# ------------------------------------------------------------------------------

y_hat <- svr_predict(
  X_scaled,
  svr_model
)


head(
  y_hat
)


# ------------------------------------------------------------------------------
# 10. Training Error Metrics
# ------------------------------------------------------------------------------

residuals <- y -
  y_hat


MSE <- mean(
  residuals^2
)


RMSE <- sqrt(
  MSE
)


MAE <- mean(
  abs(
    residuals
  )
)


data.frame(
  Metric = c(
    "MSE",
    "RMSE",
    "MAE"
  ),
  Value = c(
    MSE,
    RMSE,
    MAE
  )
)


# ------------------------------------------------------------------------------
# 11. Identify Observations Inside Epsilon Tube
# ------------------------------------------------------------------------------

inside_tube <- abs(
  residuals
) <= svr_model$epsilon


sum(
  inside_tube
)


mean(
  inside_tube
)


# ------------------------------------------------------------------------------
# 12. Approximate Support Vectors
# ------------------------------------------------------------------------------

# Points on or outside the epsilon tube influence the SVR solution.

support_vector_index <- which(
  abs(
    residuals
  ) >=
    svr_model$epsilon -
    1e-4
)


length(
  support_vector_index
)


# ------------------------------------------------------------------------------
# 13. Plot Objective Function
# ------------------------------------------------------------------------------

plot(
  seq_along(
    svr_model$objective_history
  ),
  svr_model$objective_history,
  type = "l",
  xlab = "Iteration",
  ylab = "Objective",
  main = "SVR Optimization"
)


# ------------------------------------------------------------------------------
# 14. Plot SVR Regression Line
# ------------------------------------------------------------------------------

plot(
  X_scaled[, 1],
  y,
  pch = 19,
  xlab = "Standardized x",
  ylab = "y",
  main = "Support Vector Regression"
)


x_grid <- seq(
  min(
    X_scaled[, 1]
  ),
  max(
    X_scaled[, 1]
  ),
  length.out = 300
)


X_grid <- matrix(
  x_grid,
  ncol = 1
)


y_grid <- svr_predict(
  X_grid,
  svr_model
)


lines(
  x_grid,
  y_grid,
  lwd = 2
)


# Upper epsilon tube

lines(
  x_grid,
  y_grid +
    svr_model$epsilon,
  lty = 2
)


# Lower epsilon tube

lines(
  x_grid,
  y_grid -
    svr_model$epsilon,
  lty = 2
)


# Highlight approximate support vectors

points(
  X_scaled[
    support_vector_index,
    1
  ],
  y[
    support_vector_index
  ],
  pch = 1,
  cex = 1.5,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "SVR Fit",
    "Epsilon Tube",
    "Support Vectors"
  ),
  lty = c(
    1,
    2,
    NA
  ),
  pch = c(
    NA,
    NA,
    1
  )
)


# ------------------------------------------------------------------------------
# 15. Plot Residuals
# ------------------------------------------------------------------------------

plot(
  y_hat,
  residuals,
  pch = 19,
  xlab = "Fitted Values",
  ylab = "Residuals",
  main = "SVR Residual Plot"
)


abline(
  h = 0,
  lty = 2
)


abline(
  h = svr_model$epsilon,
  lty = 3
)


abline(
  h = -svr_model$epsilon,
  lty = 3
)


# ------------------------------------------------------------------------------
# 16. Compare Epsilon-Insensitive Loss and Squared Error
# ------------------------------------------------------------------------------

residual_grid <- seq(
  -3,
  3,
  length.out = 500
)


epsilon_value <- 0.5


svr_loss <- epsilon_loss(
  residual_grid,
  epsilon_value
)


squared_error_loss <- residual_grid^2


plot(
  residual_grid,
  svr_loss,
  type = "l",
  lwd = 2,
  xlab = "Residual",
  ylab = "Loss",
  main = "SVR Loss vs Squared Error"
)


lines(
  residual_grid,
  squared_error_loss,
  lty = 2,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Epsilon-Insensitive Loss",
    "Squared Error"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 17. Compare Different Values of Epsilon
# ------------------------------------------------------------------------------

epsilon_grid <- c(
  0.1,
  0.25,
  0.5,
  1,
  1.5
)


epsilon_results <- data.frame(
  Epsilon = epsilon_grid,
  RMSE = NA,
  MAE = NA,
  Inside_Tube = NA,
  Support_Vectors = NA
)


epsilon_models <- vector(
  "list",
  length(
    epsilon_grid
  )
)


for (i in seq_along(
  epsilon_grid
)) {
  
  epsilon_i <- epsilon_grid[i]
  
  
  model_i <- svr_fit(
    X = X_scaled,
    y = y,
    C = 1,
    epsilon = epsilon_i,
    learning_rate = 0.001,
    max_iter = 20000
  )
  
  
  prediction_i <- svr_predict(
    X_scaled,
    model_i
  )
  
  
  residual_i <- y -
    prediction_i
  
  
  epsilon_results$RMSE[i] <- sqrt(
    mean(
      residual_i^2
    )
  )
  
  
  epsilon_results$MAE[i] <- mean(
    abs(
      residual_i
    )
  )
  
  
  epsilon_results$Inside_Tube[i] <- mean(
    abs(
      residual_i
    ) <=
      epsilon_i
  )
  
  
  epsilon_results$Support_Vectors[i] <- sum(
    abs(
      residual_i
    ) >=
      epsilon_i -
      1e-4
  )
  
  
  epsilon_models[[i]] <- model_i
}


epsilon_results


# ------------------------------------------------------------------------------
# 18. Plot Effect of Epsilon
# ------------------------------------------------------------------------------

plot(
  epsilon_results$Epsilon,
  epsilon_results$Support_Vectors,
  type = "b",
  pch = 19,
  xlab = "Epsilon",
  ylab = "Approximate Support Vectors",
  main = "Effect of Epsilon"
)


# ------------------------------------------------------------------------------
# 19. Compare Different Values of C
# ------------------------------------------------------------------------------

C_grid <- c(
  0.01,
  0.1,
  1,
  10,
  100
)


C_results <- data.frame(
  C = C_grid,
  RMSE = NA,
  MAE = NA,
  Weight_Norm = NA,
  Support_Vectors = NA
)


for (i in seq_along(
  C_grid
)) {
  
  C_i <- C_grid[i]
  
  
  model_i <- svr_fit(
    X = X_scaled,
    y = y,
    C = C_i,
    epsilon = 0.5,
    learning_rate = 0.001,
    max_iter = 20000
  )
  
  
  prediction_i <- svr_predict(
    X_scaled,
    model_i
  )
  
  
  residual_i <- y -
    prediction_i
  
  
  C_results$RMSE[i] <- sqrt(
    mean(
      residual_i^2
    )
  )
  
  
  C_results$MAE[i] <- mean(
    abs(
      residual_i
    )
  )
  
  
  C_results$Weight_Norm[i] <- sqrt(
    sum(
      model_i$weights^2
    )
  )
  
  
  C_results$Support_Vectors[i] <- sum(
    abs(
      residual_i
    ) >=
      model_i$epsilon -
      1e-4
  )
}


C_results


# ------------------------------------------------------------------------------
# 20. Plot Effect of C
# ------------------------------------------------------------------------------

plot(
  log10(
    C_results$C
  ),
  C_results$RMSE,
  type = "b",
  pch = 19,
  xlab = "log10(C)",
  ylab = "Training RMSE",
  main = "Effect of C on SVR"
)


# ------------------------------------------------------------------------------
# 21. Train-Test Split
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
# 22. Standardize Using Training Data Only
# ------------------------------------------------------------------------------

train_mean <- colMeans(
  X_train_raw
)


train_sd <- apply(
  X_train_raw,
  2,
  sd
)


X_train <- scale(
  X_train_raw,
  center = train_mean,
  scale = train_sd
)


X_test <- scale(
  X_test_raw,
  center = train_mean,
  scale = train_sd
)


X_train <- as.matrix(
  X_train
)

X_test <- as.matrix(
  X_test
)


# ------------------------------------------------------------------------------
# 23. Fit Training SVR
# ------------------------------------------------------------------------------

train_model <- svr_fit(
  X = X_train,
  y = y_train,
  C = 1,
  epsilon = 0.5,
  learning_rate = 0.001,
  max_iter = 20000
)


# ------------------------------------------------------------------------------
# 24. Test Predictions
# ------------------------------------------------------------------------------

test_prediction <- svr_predict(
  X_test,
  train_model
)


test_residuals <- y_test -
  test_prediction


test_RMSE <- sqrt(
  mean(
    test_residuals^2
  )
)


test_MAE <- mean(
  abs(
    test_residuals
  )
)


data.frame(
  Metric = c(
    "RMSE",
    "MAE"
  ),
  Value = c(
    test_RMSE,
    test_MAE
  )
)


# ------------------------------------------------------------------------------
# 25. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

svr_cv <- function(
    X,
    y,
    C_grid,
    epsilon_grid,
    k = 5,
    learning_rate = 0.001,
    max_iter = 10000
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  n <- nrow(X)
  
  
  folds <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  results <- expand.grid(
    C = C_grid,
    Epsilon = epsilon_grid
  )
  
  
  results$CV_RMSE <- NA
  
  
  for (r in seq_len(
    nrow(results)
  )) {
    
    C_value <- results$C[r]
    
    epsilon_value <- results$Epsilon[r]
    
    
    fold_RMSE <- numeric(
      k
    )
    
    
    for (fold in seq_len(k)) {
      
      train_index <- which(
        folds != fold
      )
      
      valid_index <- which(
        folds == fold
      )
      
      
      X_train_raw <- X[
        train_index,
        ,
        drop = FALSE
      ]
      
      X_valid_raw <- X[
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
      
      
      X_valid <- scale(
        X_valid_raw,
        center = means,
        scale = sds
      )
      
      
      X_train <- as.matrix(
        X_train
      )
      
      X_valid <- as.matrix(
        X_valid
      )
      
      
      model <- svr_fit(
        X = X_train,
        y = y_train,
        C = C_value,
        epsilon = epsilon_value,
        learning_rate = learning_rate,
        max_iter = max_iter
      )
      
      
      prediction <- svr_predict(
        X_valid,
        model
      )
      
      
      fold_RMSE[fold] <- sqrt(
        mean(
          (
            y_valid -
              prediction
          )^2
        )
      )
    }
    
    
    results$CV_RMSE[r] <- mean(
      fold_RMSE
    )
  }
  
  
  return(results)
}


# ------------------------------------------------------------------------------
# 26. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


cv_results <- svr_cv(
  X = X,
  y = y,
  C_grid = c(
    0.1,
    1,
    10
  ),
  epsilon_grid = c(
    0.1,
    0.5,
    1
  ),
  k = 5,
  learning_rate = 0.001,
  max_iter = 10000
)


cv_results


# ------------------------------------------------------------------------------
# 27. Select Best Hyperparameters
# ------------------------------------------------------------------------------

best_index <- which.min(
  cv_results$CV_RMSE
)


best_C <- cv_results$C[
  best_index
]


best_epsilon <- cv_results$Epsilon[
  best_index
]


best_C

best_epsilon


# ------------------------------------------------------------------------------
# 28. Fit Final Model
# ------------------------------------------------------------------------------

final_model <- svr_fit(
  X = X_scaled,
  y = y,
  C = best_C,
  epsilon = best_epsilon,
  learning_rate = 0.001,
  max_iter = 20000
)


final_prediction <- svr_predict(
  X_scaled,
  final_model
)


# ------------------------------------------------------------------------------
# 29. Final Performance
# ------------------------------------------------------------------------------

final_residuals <- y -
  final_prediction


final_RMSE <- sqrt(
  mean(
    final_residuals^2
  )
)


final_MAE <- mean(
  abs(
    final_residuals
  )
)


final_support_vectors <- sum(
  abs(
    final_residuals
  ) >=
    best_epsilon -
    1e-4
)


data.frame(
  RMSE = final_RMSE,
  MAE = final_MAE,
  Support_Vectors = final_support_vectors
)


# ------------------------------------------------------------------------------
# 30. Compare with Ordinary Linear Regression
# ------------------------------------------------------------------------------

lm_model <- lm(
  y ~ x,
  data = data
)


summary(
  lm_model
)


lm_prediction <- predict(
  lm_model
)


lm_RMSE <- sqrt(
  mean(
    (
      y -
        lm_prediction
    )^2
  )
)


lm_MAE <- mean(
  abs(
    y -
      lm_prediction
  )
)


comparison <- data.frame(
  Method = c(
    "Support Vector Regression",
    "Linear Regression"
  ),
  RMSE = c(
    final_RMSE,
    lm_RMSE
  ),
  MAE = c(
    final_MAE,
    lm_MAE
  )
)


comparison


# ------------------------------------------------------------------------------
# 31. Plot SVR and Linear Regression
# ------------------------------------------------------------------------------

plot(
  X_scaled[, 1],
  y,
  pch = 19,
  xlab = "Standardized x",
  ylab = "y",
  main = "SVR vs Linear Regression"
)


x_grid <- seq(
  min(
    X_scaled[, 1]
  ),
  max(
    X_scaled[, 1]
  ),
  length.out = 300
)


X_grid <- matrix(
  x_grid,
  ncol = 1
)


svr_grid_prediction <- svr_predict(
  X_grid,
  final_model
)


lines(
  x_grid,
  svr_grid_prediction,
  lwd = 2
)


# Convert standardized grid back to original x

x_grid_original <- x_grid *
  X_sd +
  X_mean


lm_grid_prediction <- predict(
  lm_model,
  newdata = data.frame(
    x = x_grid_original
  )
)


lines(
  x_grid,
  lm_grid_prediction,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "SVR",
    "Linear Regression"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 32. Summary
# ------------------------------------------------------------------------------

cat(
  "Best C:",
  best_C,
  "\n"
)

cat(
  "Best epsilon:",
  best_epsilon,
  "\n"
)

cat(
  "Training RMSE:",
  round(
    final_RMSE,
    4
  ),
  "\n"
)

cat(
  "Training MAE:",
  round(
    final_MAE,
    4
  ),
  "\n"
)

cat(
  "Test RMSE:",
  round(
    test_RMSE,
    4
  ),
  "\n"
)

cat(
  "Approximate Support Vectors:",
  final_support_vectors,
  "\n"
)