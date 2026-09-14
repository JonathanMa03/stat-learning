# ==============================================================================
# Generalized Additive Models
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Additive models
#   - Nonlinear component functions
#   - Backfitting
#   - Partial residuals
#   - Local regression smoothers
#   - Identifiability constraints
#   - Generalized additive models
#   - Logistic GAMs
#   - Iteratively reweighted least squares
#   - Bias-variance tradeoff
#
# Basic additive regression model:
#
#   E[Y | X]
#   =
#   alpha
#   +
#   f_1(X_1)
#   +
#   ...
#   +
#   f_p(X_p)
#
#
# Generalized additive model:
#
#   g(E[Y | X])
#   =
#   alpha
#   +
#   f_1(X_1)
#   +
#   ...
#   +
#   f_p(X_p)
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR ADDITIVE DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 400

x1 <- runif(
  n,
  min = -3,
  max = 3
)

x2 <- runif(
  n,
  min = -3,
  max = 3
)

x3 <- runif(
  n,
  min = -3,
  max = 3
)


# True component functions

f1_true <- function(x) {
  2 * sin(x)
}

f2_true <- function(x) {
  0.8 * x^2
}

f3_true <- function(x) {
  -1.2 * cos(1.5 * x)
}


alpha_true <- 2


mu_true <-
  alpha_true +
  f1_true(x1) +
  f2_true(x2) +
  f3_true(x3)


sigma <- 1


y <- mu_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
  )


data <- data.frame(
  x1 = x1,
  x2 = x2,
  x3 = x3,
  y = y
)


# ------------------------------------------------------------------------------
# 2. Explore the Data
# ------------------------------------------------------------------------------

pairs(
  data,
  main = "Generalized Additive Model Data"
)


# ==============================================================================
# PART II
#
# LOCAL LINEAR SMOOTHER
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Tricube Kernel
# ------------------------------------------------------------------------------

tricube <- function(u) {
  
  w <- (1 - abs(u)^3)^3
  
  w[abs(u) >= 1] <- 0
  
  return(w)
}


# ------------------------------------------------------------------------------
# 4. Local Linear Regression at One Point
# ------------------------------------------------------------------------------

local_linear_point <- function(
    x,
    y,
    x0,
    span = 0.4,
    weights = NULL
) {
  
  n <- length(x)
  
  if (is.null(weights)) {
    weights <- rep(1, n)
  }
  
  
  distances <- abs(
    x - x0
  )
  
  
  number_neighbors <- max(
    2,
    ceiling(
      span * n
    )
  )
  
  
  sorted_distances <- sort(
    distances
  )
  
  
  bandwidth <- sorted_distances[
    min(
      number_neighbors,
      length(sorted_distances)
    )
  ]
  
  
  # Numerical safeguard if multiple observations have x = x0.
  
  if (bandwidth <= 0) {
    
    positive_distances <- distances[
      distances > 0
    ]
    
    if (length(positive_distances) == 0) {
      return(
        weighted.mean(
          y,
          weights
        )
      )
    }
    
    bandwidth <- min(
      positive_distances
    )
  }
  
  
  # Slight inflation prevents the kth observation from receiving
  # exactly zero tricube weight.
  
  bandwidth <- bandwidth *
    (1 + 1e-8)
  
  
  u <- distances /
    bandwidth
  
  
  kernel_weights <- tricube(
    u
  )
  
  
  total_weights <-
    kernel_weights *
    weights
  
  
  positive <- total_weights > 0
  
  
  if (sum(positive) < 2) {
    
    return(
      weighted.mean(
        y,
        weights
      )
    )
  }
  
  
  x_centered <-
    x[positive] -
    x0
  
  
  X_local <- cbind(
    1,
    x_centered
  )
  
  
  sqrt_weights <- sqrt(
    total_weights[
      positive
    ]
  )
  
  
  X_weighted <-
    X_local *
    sqrt_weights
  
  
  y_weighted <-
    y[positive] *
    sqrt_weights
  
  
  beta <- tryCatch(
    qr.solve(
      X_weighted,
      y_weighted
    ),
    error = function(e) {
      NULL
    }
  )
  
  
  if (is.null(beta)) {
    
    return(
      weighted.mean(
        y[positive],
        total_weights[positive]
      )
    )
  }
  
  
  # Because predictors were centered at x0,
  # the fitted value at x0 is the intercept.
  
  return(
    beta[1]
  )
}


# ------------------------------------------------------------------------------
# 5. Local Linear Smoother
# ------------------------------------------------------------------------------

local_linear_smoother <- function(
    x,
    y,
    x_eval = x,
    span = 0.4,
    weights = NULL
) {
  
  sapply(
    x_eval,
    function(x0) {
      
      local_linear_point(
        x = x,
        y = y,
        x0 = x0,
        span = span,
        weights = weights
      )
    }
  )
}


# ------------------------------------------------------------------------------
# 6. Demonstrate One-Dimensional Smoothing
# ------------------------------------------------------------------------------

smooth_x1 <- local_linear_smoother(
  x = x1,
  y = y,
  x_eval = x1,
  span = 0.4
)


order_x1 <- order(
  x1
)


plot(
  x1,
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x1",
  ylab = "y",
  main = "Marginal Local Regression"
)


lines(
  x1[order_x1],
  smooth_x1[order_x1],
  lwd = 2
)


# ==============================================================================
# PART III
#
# MANUAL ADDITIVE MODEL USING BACKFITTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Backfitting Algorithm
# ------------------------------------------------------------------------------

# For:
#
# y =
# alpha +
# f1(x1) +
# f2(x2) +
# ... +
# fp(xp) +
# epsilon
#
#
# Backfitting repeatedly updates one component at a time:
#
# r_j =
# y -
# alpha -
# sum_{k != j} f_k(x_k)
#
# Then:
#
# f_j <- Smooth(r_j against x_j)
#
# Each component is centered so that:
#
# mean(f_j) = 0
#
# This provides identifiability.


additive_backfit <- function(
    X,
    y,
    spans = 0.4,
    max_iter = 100,
    tol = 1e-6,
    verbose = FALSE
) {
  
  X <- as.matrix(X)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  if (length(spans) == 1) {
    spans <- rep(
      spans,
      p
    )
  }
  
  
  alpha <- mean(
    y
  )
  
  
  components <- matrix(
    0,
    nrow = n,
    ncol = p
  )
  
  
  colnames(
    components
  ) <- colnames(X)
  
  
  convergence_history <- numeric(
    max_iter
  )
  
  
  for (iteration in seq_len(max_iter)) {
    
    old_components <- components
    
    
    for (j in seq_len(p)) {
      
      other_components <- rowSums(
        components[
          ,
          -j,
          drop = FALSE
        ]
      )
      
      
      partial_residual <-
        y -
        alpha -
        other_components
      
      
      updated_component <- local_linear_smoother(
        x = X[, j],
        y = partial_residual,
        x_eval = X[, j],
        span = spans[j]
      )
      
      
      # Identifiability constraint:
      #
      # sum_i f_j(x_ij) = 0
      
      updated_component <-
        updated_component -
        mean(
          updated_component
        )
      
      
      components[, j] <-
        updated_component
    }
    
    
    maximum_change <- max(
      abs(
        components -
          old_components
      )
    )
    
    
    convergence_history[
      iteration
    ] <- maximum_change
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        iteration,
        "Maximum Change:",
        maximum_change,
        "\n"
      )
    }
    
    
    if (maximum_change < tol) {
      break
    }
  }
  
  
  fitted_values <-
    alpha +
    rowSums(
      components
    )
  
  
  residuals <-
    y -
    fitted_values
  
  
  return(
    list(
      alpha = alpha,
      components = components,
      fitted_values = fitted_values,
      residuals = residuals,
      spans = spans,
      X = X,
      y = y,
      iterations = iteration,
      convergence_history =
        convergence_history[
          seq_len(iteration)
        ]
    )
  )
}


# ------------------------------------------------------------------------------
# 8. Fit the Additive Model
# ------------------------------------------------------------------------------

X <- cbind(
  x1 = x1,
  x2 = x2,
  x3 = x3
)


gam_fit <- additive_backfit(
  X = X,
  y = y,
  spans = c(
    0.35,
    0.35,
    0.35
  ),
  max_iter = 100,
  tol = 1e-5,
  verbose = FALSE
)


gam_fit$iterations


gam_fit$alpha


# ------------------------------------------------------------------------------
# 9. Training Error
# ------------------------------------------------------------------------------

gam_training_MSE <- mean(
  gam_fit$residuals^2
)


gam_training_RMSE <- sqrt(
  gam_training_MSE
)


gam_training_MSE


gam_training_RMSE


# ==============================================================================
# PART IV
#
# BACKFITTING CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Plot Convergence
# ------------------------------------------------------------------------------

plot(
  seq_along(
    gam_fit$convergence_history
  ),
  gam_fit$convergence_history,
  type = "b",
  pch = 19,
  log = "y",
  xlab = "Backfitting Iteration",
  ylab = "Maximum Component Change",
  main = "Backfitting Convergence"
)


abline(
  h = 1e-5,
  lty = 2
)


# ==============================================================================
# PART V
#
# ESTIMATED COMPONENT FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Center True Functions
# ------------------------------------------------------------------------------

true_component_1 <- f1_true(
  x1
)

true_component_1 <-
  true_component_1 -
  mean(
    true_component_1
  )


true_component_2 <- f2_true(
  x2
)

true_component_2 <-
  true_component_2 -
  mean(
    true_component_2
  )


true_component_3 <- f3_true(
  x3
)

true_component_3 <-
  true_component_3 -
  mean(
    true_component_3
  )


# ------------------------------------------------------------------------------
# 12. Plot Estimated f1
# ------------------------------------------------------------------------------

order_1 <- order(
  x1
)


plot(
  x1,
  gam_fit$components[, 1],
  pch = 19,
  cex = 0.5,
  xlab = "x1",
  ylab = "Component Effect",
  main = "Estimated f1(x1)"
)


lines(
  x1[order_1],
  true_component_1[order_1],
  lwd = 2,
  lty = 2
)


# ------------------------------------------------------------------------------
# 13. Plot Estimated f2
# ------------------------------------------------------------------------------

order_2 <- order(
  x2
)


plot(
  x2,
  gam_fit$components[, 2],
  pch = 19,
  cex = 0.5,
  xlab = "x2",
  ylab = "Component Effect",
  main = "Estimated f2(x2)"
)


lines(
  x2[order_2],
  true_component_2[order_2],
  lwd = 2,
  lty = 2
)


# ------------------------------------------------------------------------------
# 14. Plot Estimated f3
# ------------------------------------------------------------------------------

order_3 <- order(
  x3
)


plot(
  x3,
  gam_fit$components[, 3],
  pch = 19,
  cex = 0.5,
  xlab = "x3",
  ylab = "Component Effect",
  main = "Estimated f3(x3)"
)


lines(
  x3[order_3],
  true_component_3[order_3],
  lwd = 2,
  lty = 2
)


# ------------------------------------------------------------------------------
# 15. Plot All Components
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    3
  )
)


plot(
  x1,
  gam_fit$components[, 1],
  pch = 19,
  cex = 0.4,
  xlab = "x1",
  ylab = "Effect",
  main = "f1(x1)"
)

lines(
  x1[order_1],
  true_component_1[order_1],
  lwd = 2,
  lty = 2
)


plot(
  x2,
  gam_fit$components[, 2],
  pch = 19,
  cex = 0.4,
  xlab = "x2",
  ylab = "Effect",
  main = "f2(x2)"
)

lines(
  x2[order_2],
  true_component_2[order_2],
  lwd = 2,
  lty = 2
)


plot(
  x3,
  gam_fit$components[, 3],
  pch = 19,
  cex = 0.4,
  xlab = "x3",
  ylab = "Effect",
  main = "f3(x3)"
)

lines(
  x3[order_3],
  true_component_3[order_3],
  lwd = 2,
  lty = 2
)


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART VI
#
# COMPARE WITH LINEAR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Fit Ordinary Linear Model
# ------------------------------------------------------------------------------

linear_model <- lm(
  y ~ x1 + x2 + x3,
  data = data
)


linear_prediction <- predict(
  linear_model,
  newdata = data
)


linear_training_MSE <- mean(
  (
    y -
      linear_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 17. Compare Training Error
# ------------------------------------------------------------------------------

training_results <- data.frame(
  Model = c(
    "Linear Regression",
    "Additive Model"
  ),
  MSE = c(
    linear_training_MSE,
    gam_training_MSE
  )
)


training_results


# ==============================================================================
# PART VII
#
# PREDICTING NEW OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Predict One Component at New Points
# ------------------------------------------------------------------------------

# The training component estimates can be viewed as pseudo-responses:
#
# estimated f_j(x_ij)
#
# For new observations, smooth those component values against X_j.


predict_additive <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  p <- ncol(
    model$X
  )
  
  
  component_predictions <- matrix(
    0,
    nrow = nrow(X_new),
    ncol = p
  )
  
  
  for (j in seq_len(p)) {
    
    component_predictions[, j] <-
      local_linear_smoother(
        x = model$X[, j],
        y = model$components[, j],
        x_eval = X_new[, j],
        span = model$spans[j]
      )
  }
  
  
  prediction <-
    model$alpha +
    rowSums(
      component_predictions
    )
  
  
  return(
    list(
      prediction = prediction,
      components = component_predictions
    )
  )
}


# ------------------------------------------------------------------------------
# 19. Verify Training Prediction
# ------------------------------------------------------------------------------

training_prediction_check <- predict_additive(
  gam_fit,
  X
)


mean(
  (
    training_prediction_check$prediction -
      gam_fit$fitted_values
  )^2
)


# Small differences are expected because component functions are re-smoothed
# when prediction is performed.


# ==============================================================================
# PART VIII
#
# TRAIN-TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(456)


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


X_train <- X[
  train_index,
  ,
  drop = FALSE
]


y_train <- y[
  train_index
]


X_test <- X[
  test_index,
  ,
  drop = FALSE
]


y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 21. Fit GAM on Training Data
# ------------------------------------------------------------------------------

gam_train <- additive_backfit(
  X = X_train,
  y = y_train,
  spans = c(
    0.35,
    0.35,
    0.35
  ),
  max_iter = 100,
  tol = 1e-5
)


gam_test_prediction <- predict_additive(
  gam_train,
  X_test
)$prediction


gam_test_MSE <- mean(
  (
    y_test -
      gam_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 22. Linear Regression Baseline
# ------------------------------------------------------------------------------

training_data <- data.frame(
  y = y_train,
  x1 = X_train[, 1],
  x2 = X_train[, 2],
  x3 = X_train[, 3]
)


testing_data <- data.frame(
  x1 = X_test[, 1],
  x2 = X_test[, 2],
  x3 = X_test[, 3]
)


linear_train <- lm(
  y ~ x1 + x2 + x3,
  data = training_data
)


linear_test_prediction <- predict(
  linear_train,
  newdata = testing_data
)


linear_test_MSE <- mean(
  (
    y_test -
      linear_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 23. Test Comparison
# ------------------------------------------------------------------------------

test_results <- data.frame(
  Model = c(
    "Linear Regression",
    "Additive Model"
  ),
  Test_MSE = c(
    linear_test_MSE,
    gam_test_MSE
  )
)


test_results$Test_RMSE <- sqrt(
  test_results$Test_MSE
)


test_results


# ==============================================================================
# PART IX
#
# CHOOSING SMOOTHNESS BY CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. K-Fold Cross Validation for Span
# ------------------------------------------------------------------------------

gam_cv <- function(
    X,
    y,
    spans,
    k = 5,
    seed = 123
) {
  
  set.seed(seed)
  
  n <- nrow(X)
  
  
  fold_id <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  results <- data.frame(
    Span = spans,
    Mean_CV_MSE = NA,
    SE_CV_MSE = NA
  )
  
  
  for (s in seq_along(spans)) {
    
    fold_MSE <- numeric(
      k
    )
    
    
    for (fold in seq_len(k)) {
      
      train_idx <- which(
        fold_id != fold
      )
      
      validation_idx <- which(
        fold_id == fold
      )
      
      
      model <- additive_backfit(
        X = X[
          train_idx,
          ,
          drop = FALSE
        ],
        y = y[
          train_idx
        ],
        spans = spans[s],
        max_iter = 50,
        tol = 1e-4
      )
      
      
      prediction <- predict_additive(
        model,
        X[
          validation_idx,
          ,
          drop = FALSE
        ]
      )$prediction
      
      
      fold_MSE[fold] <- mean(
        (
          y[validation_idx] -
            prediction
        )^2
      )
    }
    
    
    results$Mean_CV_MSE[s] <- mean(
      fold_MSE
    )
    
    
    results$SE_CV_MSE[s] <-
      sd(
        fold_MSE
      ) /
      sqrt(k)
  }
  
  
  return(results)
}


# ------------------------------------------------------------------------------
# 25. Run Cross Validation
# ------------------------------------------------------------------------------

candidate_spans <- c(
  0.15,
  0.20,
  0.30,
  0.40,
  0.50,
  0.65,
  0.80
)


cv_results <- gam_cv(
  X = X_train,
  y = y_train,
  spans = candidate_spans,
  k = 5,
  seed = 123
)


cv_results


# ------------------------------------------------------------------------------
# 26. Best Span
# ------------------------------------------------------------------------------

best_span <- cv_results$Span[
  which.min(
    cv_results$Mean_CV_MSE
  )
]


best_span


# ------------------------------------------------------------------------------
# 27. Plot Cross-Validation Curve
# ------------------------------------------------------------------------------

plot(
  cv_results$Span,
  cv_results$Mean_CV_MSE,
  type = "b",
  pch = 19,
  xlab = "Span",
  ylab = "Cross-Validated MSE",
  main = "GAM Smoothness Selection"
)


arrows(
  x0 = cv_results$Span,
  y0 = cv_results$Mean_CV_MSE -
    cv_results$SE_CV_MSE,
  x1 = cv_results$Span,
  y1 = cv_results$Mean_CV_MSE +
    cv_results$SE_CV_MSE,
  angle = 90,
  code = 3,
  length = 0.05
)


# ------------------------------------------------------------------------------
# 28. Refit with CV-Selected Span
# ------------------------------------------------------------------------------

gam_cv_model <- additive_backfit(
  X = X_train,
  y = y_train,
  spans = best_span,
  max_iter = 100,
  tol = 1e-5
)


cv_test_prediction <- predict_additive(
  gam_cv_model,
  X_test
)$prediction


cv_test_MSE <- mean(
  (
    y_test -
      cv_test_prediction
  )^2
)


cv_test_MSE


# ==============================================================================
# PART X
#
# DIFFERENT SMOOTHNESS FOR DIFFERENT VARIABLES
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Variable-Specific Spans
# ------------------------------------------------------------------------------

# GAMs do not require every component to have the same complexity.


variable_span_model <- additive_backfit(
  X = X_train,
  y = y_train,
  spans = c(
    0.25,
    0.45,
    0.30
  ),
  max_iter = 100,
  tol = 1e-5
)


variable_span_prediction <- predict_additive(
  variable_span_model,
  X_test
)$prediction


variable_span_MSE <- mean(
  (
    y_test -
      variable_span_prediction
  )^2
)


variable_span_MSE


# ==============================================================================
# PART XI
#
# PARTIAL RESIDUALS
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Partial Residuals for x1
# ------------------------------------------------------------------------------

# For component j:
#
# partial residual =
#
# y -
# alpha -
# sum_{k != j} f_k(x_k)
#
# Equivalently:
#
# residual + fitted component j


partial_residual_1 <-
  gam_fit$residuals +
  gam_fit$components[, 1]


plot(
  x1,
  partial_residual_1,
  pch = 19,
  cex = 0.5,
  xlab = "x1",
  ylab = "Partial Residual",
  main = "Partial Residual Plot for x1"
)


lines(
  x1[order_1],
  gam_fit$components[
    order_1,
    1
  ],
  lwd = 2
)


# ------------------------------------------------------------------------------
# 31. Partial Residuals for x2
# ------------------------------------------------------------------------------

partial_residual_2 <-
  gam_fit$residuals +
  gam_fit$components[, 2]


plot(
  x2,
  partial_residual_2,
  pch = 19,
  cex = 0.5,
  xlab = "x2",
  ylab = "Partial Residual",
  main = "Partial Residual Plot for x2"
)


lines(
  x2[order_2],
  gam_fit$components[
    order_2,
    2
  ],
  lwd = 2
)


# ==============================================================================
# PART XII
#
# ADDITIVE MODEL LIMITATION: INTERACTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Simulate Interaction Data
# ------------------------------------------------------------------------------

set.seed(789)


n_interaction <- 400


z1 <- runif(
  n_interaction,
  -3,
  3
)


z2 <- runif(
  n_interaction,
  -3,
  3
)


interaction_truth <- function(
    z1,
    z2
) {
  
  2 +
    sin(
      z1 * z2
    )
}


y_interaction <-
  interaction_truth(
    z1,
    z2
  ) +
  rnorm(
    n_interaction,
    sd = 0.5
  )


X_interaction <- cbind(
  z1 = z1,
  z2 = z2
)


# ------------------------------------------------------------------------------
# 33. Fit Additive Model to Interaction Data
# ------------------------------------------------------------------------------

interaction_additive_model <- additive_backfit(
  X = X_interaction,
  y = y_interaction,
  spans = 0.3,
  max_iter = 100,
  tol = 1e-5
)


interaction_MSE <- mean(
  interaction_additive_model$residuals^2
)


interaction_MSE


# ------------------------------------------------------------------------------
# 34. Compare Against Additive Data
# ------------------------------------------------------------------------------

cat(
  "Training MSE on additive DGP:",
  gam_training_MSE,
  "\n"
)


cat(
  "Training MSE on interaction DGP:",
  interaction_MSE,
  "\n"
)


# The additive model cannot directly represent:
#
# sin(z1 * z2)
#
# using only:
#
# f1(z1) + f2(z2)


# ==============================================================================
# PART XIII
#
# LOGISTIC GENERALIZED ADDITIVE MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Generate Binary Data
# ------------------------------------------------------------------------------

set.seed(321)


n_binary <- 500


b1 <- runif(
  n_binary,
  -3,
  3
)


b2 <- runif(
  n_binary,
  -3,
  3
)


true_logit <- function(
    b1,
    b2
) {
  
  -0.5 +
    1.5 * sin(b1) +
    0.5 * b2^2 -
    1
}


eta_true <- true_logit(
  b1,
  b2
)


stable_sigmoid <- function(eta) {
  
  output <- numeric(
    length(eta)
  )
  
  
  positive <- eta >= 0
  
  
  output[positive] <-
    1 /
    (
      1 +
        exp(
          -eta[positive]
        )
    )
  
  
  exp_eta <- exp(
    eta[
      !positive
    ]
  )
  
  
  output[
    !positive
  ] <-
    exp_eta /
    (
      1 +
        exp_eta
    )
  
  
  return(output)
}


probability_true <- stable_sigmoid(
  eta_true
)


y_binary <- rbinom(
  n_binary,
  size = 1,
  prob = probability_true
)


X_binary <- cbind(
  b1 = b1,
  b2 = b2
)


# ------------------------------------------------------------------------------
# 36. Binary Outcome Proportion
# ------------------------------------------------------------------------------

mean(
  y_binary
)


# ==============================================================================
# PART XIV
#
# LOGISTIC GAM USING LOCAL SCORING
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Weighted Additive Backfitting
# ------------------------------------------------------------------------------

weighted_additive_backfit <- function(
    X,
    z,
    weights,
    spans,
    initial_components = NULL,
    initial_alpha = NULL,
    max_iter = 50,
    tol = 1e-5
) {
  
  X <- as.matrix(X)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  if (length(spans) == 1) {
    spans <- rep(
      spans,
      p
    )
  }
  
  
  if (is.null(initial_alpha)) {
    
    alpha <- weighted.mean(
      z,
      weights
    )
    
  } else {
    
    alpha <- initial_alpha
  }
  
  
  if (is.null(initial_components)) {
    
    components <- matrix(
      0,
      nrow = n,
      ncol = p
    )
    
  } else {
    
    components <- initial_components
  }
  
  
  for (iteration in seq_len(max_iter)) {
    
    old_components <- components
    
    
    for (j in seq_len(p)) {
      
      other_components <- rowSums(
        components[
          ,
          -j,
          drop = FALSE
        ]
      )
      
      
      partial_response <-
        z -
        alpha -
        other_components
      
      
      updated_component <- local_linear_smoother(
        x = X[, j],
        y = partial_response,
        x_eval = X[, j],
        span = spans[j],
        weights = weights
      )
      
      
      component_mean <- weighted.mean(
        updated_component,
        weights
      )
      
      
      updated_component <-
        updated_component -
        component_mean
      
      
      alpha <-
        alpha +
        component_mean
      
      
      components[, j] <-
        updated_component
    }
    
    
    maximum_change <- max(
      abs(
        components -
          old_components
      )
    )
    
    
    if (maximum_change < tol) {
      break
    }
  }
  
  
  fitted_eta <-
    alpha +
    rowSums(
      components
    )
  
  
  return(
    list(
      alpha = alpha,
      components = components,
      fitted_eta = fitted_eta,
      iterations = iteration
    )
  )
}


# ------------------------------------------------------------------------------
# 38. Logistic GAM
# ------------------------------------------------------------------------------

logistic_gam <- function(
    X,
    y,
    spans = 0.4,
    max_outer = 50,
    max_inner = 50,
    tol = 1e-5,
    epsilon = 1e-6,
    verbose = FALSE
) {
  
  X <- as.matrix(X)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  if (length(spans) == 1) {
    spans <- rep(
      spans,
      p
    )
  }
  
  
  starting_probability <- mean(
    y
  )
  
  
  starting_probability <- min(
    max(
      starting_probability,
      epsilon
    ),
    1 - epsilon
  )
  
  
  alpha <- qlogis(
    starting_probability
  )
  
  
  components <- matrix(
    0,
    nrow = n,
    ncol = p
  )
  
  
  eta <- rep(
    alpha,
    n
  )
  
  
  deviance_history <- numeric(
    max_outer
  )
  
  
  for (outer in seq_len(max_outer)) {
    
    probability <- stable_sigmoid(
      eta
    )
    
    
    probability <- pmin(
      pmax(
        probability,
        epsilon
      ),
      1 - epsilon
    )
    
    
    weights <-
      probability *
      (
        1 -
          probability
      )
    
    
    weights <- pmax(
      weights,
      epsilon
    )
    
    
    working_response <-
      eta +
      (
        y -
          probability
      ) /
      weights
    
    
    additive_update <- weighted_additive_backfit(
      X = X,
      z = working_response,
      weights = weights,
      spans = spans,
      initial_components = components,
      initial_alpha = alpha,
      max_iter = max_inner,
      tol = tol
    )
    
    
    eta_new <-
      additive_update$fitted_eta
    
    
    probability_new <- stable_sigmoid(
      eta_new
    )
    
    
    probability_new <- pmin(
      pmax(
        probability_new,
        epsilon
      ),
      1 - epsilon
    )
    
    
    deviance <- -2 *
      sum(
        y *
          log(
            probability_new
          ) +
          (
            1 -
              y
          ) *
          log(
            1 -
              probability_new
          )
      )
    
    
    deviance_history[
      outer
    ] <- deviance
    
    
    maximum_change <- max(
      abs(
        eta_new -
          eta
      )
    )
    
    
    if (verbose) {
      
      cat(
        "Outer iteration:",
        outer,
        "Deviance:",
        deviance,
        "Maximum change:",
        maximum_change,
        "\n"
      )
    }
    
    
    alpha <- additive_update$alpha
    
    components <- additive_update$components
    
    eta <- eta_new
    
    
    if (maximum_change < tol) {
      break
    }
  }
  
  
  probability <- stable_sigmoid(
    eta
  )
  
  
  return(
    list(
      alpha = alpha,
      components = components,
      eta = eta,
      probability = probability,
      spans = spans,
      X = X,
      y = y,
      iterations = outer,
      deviance_history =
        deviance_history[
          seq_len(outer)
        ]
    )
  )
}


# ------------------------------------------------------------------------------
# 39. Fit Logistic GAM
# ------------------------------------------------------------------------------

logistic_gam_fit <- logistic_gam(
  X = X_binary,
  y = y_binary,
  spans = c(
    0.35,
    0.35
  ),
  max_outer = 50,
  max_inner = 30,
  tol = 1e-4
)


logistic_gam_fit$iterations


# ------------------------------------------------------------------------------
# 40. Binary Predictions
# ------------------------------------------------------------------------------

predicted_class <- ifelse(
  logistic_gam_fit$probability >= 0.5,
  1,
  0
)


confusion_matrix <- table(
  Actual = y_binary,
  Predicted = predicted_class
)


confusion_matrix


classification_accuracy <- mean(
  predicted_class ==
    y_binary
)


classification_accuracy


# ------------------------------------------------------------------------------
# 41. Binary Log-Loss
# ------------------------------------------------------------------------------

logistic_probabilities <- pmin(
  pmax(
    logistic_gam_fit$probability,
    1e-10
  ),
  1 - 1e-10
)


gam_log_loss <- -mean(
  y_binary *
    log(
      logistic_probabilities
    ) +
    (
      1 -
        y_binary
    ) *
    log(
      1 -
        logistic_probabilities
    )
)


gam_log_loss


# ------------------------------------------------------------------------------
# 42. Brier Score
# ------------------------------------------------------------------------------

gam_brier <- mean(
  (
    y_binary -
      logistic_gam_fit$probability
  )^2
)


gam_brier


# ------------------------------------------------------------------------------
# 43. Plot Logistic GAM Deviance
# ------------------------------------------------------------------------------

plot(
  seq_along(
    logistic_gam_fit$deviance_history
  ),
  logistic_gam_fit$deviance_history,
  type = "b",
  pch = 19,
  xlab = "Local-Scoring Iteration",
  ylab = "Deviance",
  main = "Logistic GAM Convergence"
)


# ==============================================================================
# PART XV
#
# COMPARE WITH LINEAR LOGISTIC REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Standard Logistic Regression
# ------------------------------------------------------------------------------

binary_data <- data.frame(
  y = y_binary,
  b1 = b1,
  b2 = b2
)


linear_logistic <- glm(
  y ~ b1 + b2,
  data = binary_data,
  family = binomial()
)


linear_probability <- predict(
  linear_logistic,
  type = "response"
)


linear_probability <- pmin(
  pmax(
    linear_probability,
    1e-10
  ),
  1 - 1e-10
)


linear_log_loss <- -mean(
  y_binary *
    log(
      linear_probability
    ) +
    (
      1 -
        y_binary
    ) *
    log(
      1 -
        linear_probability
    )
)


linear_accuracy <- mean(
  ifelse(
    linear_probability >= 0.5,
    1,
    0
  ) ==
    y_binary
)


# ------------------------------------------------------------------------------
# 45. Binary Model Comparison
# ------------------------------------------------------------------------------

binary_results <- data.frame(
  Model = c(
    "Linear Logistic Regression",
    "Logistic GAM"
  ),
  Accuracy = c(
    linear_accuracy,
    classification_accuracy
  ),
  Log_Loss = c(
    linear_log_loss,
    gam_log_loss
  )
)


binary_results


# ==============================================================================
# PART XVI
#
# VISUALIZE LOGISTIC COMPONENT FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. First Logistic Component
# ------------------------------------------------------------------------------

order_b1 <- order(
  b1
)


plot(
  b1,
  logistic_gam_fit$components[, 1],
  pch = 19,
  cex = 0.4,
  xlab = "b1",
  ylab = "Estimated Log-Odds Effect",
  main = "Logistic GAM: f1(b1)"
)


lines(
  b1[order_b1],
  logistic_gam_fit$components[
    order_b1,
    1
  ],
  lwd = 2
)


# ------------------------------------------------------------------------------
# 47. Second Logistic Component
# ------------------------------------------------------------------------------

order_b2 <- order(
  b2
)


plot(
  b2,
  logistic_gam_fit$components[, 2],
  pch = 19,
  cex = 0.4,
  xlab = "b2",
  ylab = "Estimated Log-Odds Effect",
  main = "Logistic GAM: f2(b2)"
)


lines(
  b2[order_b2],
  logistic_gam_fit$components[
    order_b2,
    2
  ],
  lwd = 2
)


# ==============================================================================
# PART XVII
#
# OPTIONAL VERIFICATION WITH MGCV
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Gaussian GAM Verification
# ------------------------------------------------------------------------------

# mgcv is not required for this demo.
#
# If installed, compare the manually implemented backfitting GAM against
# a production-quality penalized spline GAM.


if (
  requireNamespace(
    "mgcv",
    quietly = TRUE
  )
) {
  
  mgcv_model <- mgcv::gam(
    y ~
      s(x1) +
      s(x2) +
      s(x3),
    data = data,
    method = "REML"
  )
  
  
  mgcv_prediction <- predict(
    mgcv_model,
    newdata = data
  )
  
  
  mgcv_MSE <- mean(
    (
      y -
        mgcv_prediction
    )^2
  )
  
  
  cat(
    "Manual GAM training MSE:",
    gam_training_MSE,
    "\n"
  )
  
  
  cat(
    "mgcv GAM training MSE:",
    mgcv_MSE,
    "\n"
  )
  
  
  print(
    summary(
      mgcv_model
    )
  )
}


# ------------------------------------------------------------------------------
# 49. Logistic GAM Verification
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "mgcv",
    quietly = TRUE
  )
) {
  
  mgcv_binary_data <- data.frame(
    y = y_binary,
    b1 = b1,
    b2 = b2
  )
  
  
  mgcv_logistic <- mgcv::gam(
    y ~
      s(b1) +
      s(b2),
    data = mgcv_binary_data,
    family = binomial(),
    method = "REML"
  )
  
  
  mgcv_probability <- predict(
    mgcv_logistic,
    type = "response"
  )
  
  
  mgcv_probability <- pmin(
    pmax(
      mgcv_probability,
      1e-10
    ),
    1 - 1e-10
  )
  
  
  mgcv_log_loss <- -mean(
    y_binary *
      log(
        mgcv_probability
      ) +
      (
        1 -
          y_binary
      ) *
      log(
        1 -
          mgcv_probability
      )
  )
  
  
  cat(
    "Manual logistic GAM log-loss:",
    gam_log_loss,
    "\n"
  )
  
  
  cat(
    "mgcv logistic GAM log-loss:",
    mgcv_log_loss,
    "\n"
  )
}


# ==============================================================================
# PART XVIII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Summary
# ------------------------------------------------------------------------------

cat(
  "Generalized Additive Models Summary\n"
)


cat(
  "-----------------------------------\n"
)


cat(
  "Additive backfitting iterations:",
  gam_fit$iterations,
  "\n"
)


cat(
  "Linear-regression training MSE:",
  round(
    linear_training_MSE,
    4
  ),
  "\n"
)


cat(
  "Additive-model training MSE:",
  round(
    gam_training_MSE,
    4
  ),
  "\n"
)


cat(
  "Linear-regression test MSE:",
  round(
    linear_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Additive-model test MSE:",
  round(
    gam_test_MSE,
    4
  ),
  "\n"
)


cat(
  "CV-selected span:",
  best_span,
  "\n"
)


cat(
  "CV-selected GAM test MSE:",
  round(
    cv_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Logistic GAM accuracy:",
  round(
    classification_accuracy,
    4
  ),
  "\n"
)


cat(
  "Logistic GAM log-loss:",
  round(
    gam_log_loss,
    4
  ),
  "\n"
)