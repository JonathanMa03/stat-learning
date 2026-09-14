# ==============================================================================
# Local Regression
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Local constant regression
#   - Local linear regression
#   - Local polynomial regression
#   - Distance-based weighting
#   - Tricube kernel
#   - Bandwidth / span
#   - Bias-variance tradeoff
#   - Cross-validation for span selection
#   - Comparison with loess()
#
# Local linear regression at x0:
#
#   minimize_{beta0, beta1}
#
#       sum_i w_i(x0)
#       [y_i - beta0 - beta1(x_i - x0)]^2
#
# where observations close to x0 receive larger weights.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Nonlinear Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 250

x <- sort(
  runif(
    n,
    min = 0,
    max = 10
  )
)


true_function <- function(x) {
  
  2 +
    0.5 * x +
    2 * sin(x)
}


y_true <- true_function(
  x
)


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = 0.8
  )


data <- data.frame(
  x = x,
  y = y
)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Nonlinear Regression Data"
)


lines(
  x,
  y_true,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Tricube Weight Function
# ------------------------------------------------------------------------------

# LOESS commonly uses the tricube kernel:
#
# K(u) = (1 - |u|^3)^3     if |u| < 1
#        0                  otherwise


tricube <- function(u) {
  
  output <- (
    1 -
      abs(u)^3
  )^3
  
  
  output[
    abs(u) >= 1
  ] <- 0
  
  
  return(
    output
  )
}


# ------------------------------------------------------------------------------
# 4. Visualize Tricube Kernel
# ------------------------------------------------------------------------------

u_grid <- seq(
  -1.5,
  1.5,
  length.out = 500
)


plot(
  u_grid,
  tricube(
    u_grid
  ),
  type = "l",
  lwd = 2,
  xlab = "Standardized Distance",
  ylab = "Weight",
  main = "Tricube Kernel"
)


# ------------------------------------------------------------------------------
# 5. Local Constant Regression
# ------------------------------------------------------------------------------

# At prediction location x0:
#
# y_hat(x0)
# =
# weighted average of nearby responses.


local_constant <- function(
    x,
    y,
    x0,
    span = 0.3
) {
  
  n <- length(
    x
  )
  
  
  number_neighbors <- max(
    2,
    ceiling(
      span * n
    )
  )
  
  
  distances <- abs(
    x -
      x0
  )
  
  
  bandwidth <- sort(
    distances
  )[
    number_neighbors
  ]
  
  
  if (
    bandwidth == 0
  ) {
    
    bandwidth <- max(
      distances[
        distances > 0
      ]
    )
  }
  
  
  u <- distances /
    bandwidth
  
  
  weights <- tricube(
    u
  )
  
  
  if (
    sum(
      weights
    ) == 0
  ) {
    
    return(
      mean(
        y
      )
    )
  }
  
  
  sum(
    weights *
      y
  ) /
    sum(
      weights
    )
}


# ------------------------------------------------------------------------------
# 6. Fit Local Constant Regression
# ------------------------------------------------------------------------------

x_grid <- seq(
  min(x),
  max(x),
  length.out = 300
)


local_constant_prediction <- sapply(
  x_grid,
  function(x0) {
    
    local_constant(
      x = x,
      y = y,
      x0 = x0,
      span = 0.3
    )
  }
)


# ------------------------------------------------------------------------------
# 7. Plot Local Constant Fit
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Local Constant Regression"
)


lines(
  x_grid,
  local_constant_prediction,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 8. Local Linear Regression
# ------------------------------------------------------------------------------

# At each x0 fit:
#
# y_i =
#
# beta0 +
# beta1(x_i - x0)
#
# using distance-based weights.
#
# Prediction at x0 is beta0.


local_linear <- function(
    x,
    y,
    x0,
    span = 0.3
) {
  
  n <- length(
    x
  )
  
  
  number_neighbors <- max(
    2,
    ceiling(
      span * n
    )
  )
  
  
  distances <- abs(
    x -
      x0
  )
  
  
  bandwidth <- sort(
    distances
  )[
    number_neighbors
  ]
  
  
  if (
    bandwidth == 0
  ) {
    
    positive_distances <- distances[
      distances > 0
    ]
    
    
    if (
      length(
        positive_distances
      ) == 0
    ) {
      
      return(
        mean(
          y
        )
      )
      
      
    } else {
      
      bandwidth <- min(
        positive_distances
      )
    }
  }
  
  
  u <- distances /
    bandwidth
  
  
  weights <- tricube(
    u
  )
  
  
  X_local <- cbind(
    Intercept = 1,
    Centered_x =
      x -
      x0
  )
  
  
  W <- diag(
    weights
  )
  
  
  XtWX <- t(
    X_local
  ) %*%
    W %*%
    X_local
  
  
  XtWy <- t(
    X_local
  ) %*%
    W %*%
    y
  
  
  beta <- tryCatch(
    solve(
      XtWX,
      XtWy
    ),
    error = function(e) {
      
      qr.solve(
        XtWX,
        XtWy
      )
    }
  )
  
  
  return(
    as.numeric(
      beta[1]
    )
  )
}


# ------------------------------------------------------------------------------
# 9. Fit Local Linear Regression
# ------------------------------------------------------------------------------

local_linear_prediction <- sapply(
  x_grid,
  function(x0) {
    
    local_linear(
      x = x,
      y = y,
      x0 = x0,
      span = 0.3
    )
  }
)


# ------------------------------------------------------------------------------
# 10. Plot Local Linear Fit
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Local Linear Regression"
)


lines(
  x_grid,
  local_linear_prediction,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 11. Generic Local Polynomial Regression
# ------------------------------------------------------------------------------

local_polynomial <- function(
    x,
    y,
    x0,
    span = 0.3,
    degree = 1
) {
  
  n <- length(
    x
  )
  
  
  number_neighbors <- max(
    degree + 2,
    ceiling(
      span * n
    )
  )
  
  
  distances <- abs(
    x -
      x0
  )
  
  
  bandwidth <- sort(
    distances
  )[
    min(
      number_neighbors,
      n
    )
  ]
  
  
  if (
    bandwidth == 0
  ) {
    
    positive_distances <- distances[
      distances > 0
    ]
    
    
    if (
      length(
        positive_distances
      ) == 0
    ) {
      
      return(
        mean(
          y
        )
      )
      
      
    } else {
      
      bandwidth <- min(
        positive_distances
      )
    }
  }
  
  
  u <- distances /
    bandwidth
  
  
  weights <- tricube(
    u
  )
  
  
  centered_x <- x -
    x0
  
  
  X_local <- matrix(
    1,
    nrow = n,
    ncol = degree + 1
  )
  
  
  if (
    degree >= 1
  ) {
    
    for (
      d in seq_len(
        degree
      )
    ) {
      
      X_local[
        ,
        d + 1
      ] <-
        centered_x^d
    }
  }
  
  
  sqrt_w <- sqrt(
    weights
  )
  
  
  X_weighted <- X_local *
    sqrt_w
  
  
  y_weighted <- y *
    sqrt_w
  
  
  beta <- tryCatch(
    qr.solve(
      X_weighted,
      y_weighted
    ),
    error = function(e) {
      
      return(
        c(
          weighted.mean(
            y,
            weights
          ),
          rep(
            0,
            degree
          )
        )
      )
    }
  )
  
  
  return(
    as.numeric(
      beta[1]
    )
  )
}


# ------------------------------------------------------------------------------
# 12. Compare Polynomial Degrees
# ------------------------------------------------------------------------------

degree_0_prediction <- sapply(
  x_grid,
  function(x0) {
    
    local_polynomial(
      x,
      y,
      x0,
      span = 0.3,
      degree = 0
    )
  }
)


degree_1_prediction <- sapply(
  x_grid,
  function(x0) {
    
    local_polynomial(
      x,
      y,
      x0,
      span = 0.3,
      degree = 1
    )
  }
)


degree_2_prediction <- sapply(
  x_grid,
  function(x0) {
    
    local_polynomial(
      x,
      y,
      x0,
      span = 0.3,
      degree = 2
    )
  }
)


# ------------------------------------------------------------------------------
# 13. Plot Degree Comparison
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Local Polynomial Regression"
)


lines(
  x_grid,
  degree_0_prediction,
  lty = 2,
  lwd = 2
)


lines(
  x_grid,
  degree_1_prediction,
  lty = 1,
  lwd = 2
)


lines(
  x_grid,
  degree_2_prediction,
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Degree 0",
    "Degree 1",
    "Degree 2"
  ),
  lty = c(
    2,
    1,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 14. Inspect Local Weights at One Prediction Point
# ------------------------------------------------------------------------------

x0 <- 5

span <- 0.3


number_neighbors <- ceiling(
  span *
    n
)


distances <- abs(
  x -
    x0
)


bandwidth <- sort(
  distances
)[
  number_neighbors
]


weights <- tricube(
  distances /
    bandwidth
)


plot(
  x,
  weights,
  type = "h",
  xlab = "x",
  ylab = "Weight",
  main = paste(
    "Local Weights Around x0 =",
    x0
  )
)


abline(
  v = x0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 15. Visualize One Local Linear Fit
# ------------------------------------------------------------------------------

X_local <- cbind(
  1,
  x -
    x0
)


sqrt_w <- sqrt(
  weights
)


beta_local <- qr.solve(
  X_local *
    sqrt_w,
  y *
    sqrt_w
)


local_line <- beta_local[1] +
  beta_local[2] *
  (
    x_grid -
      x0
  )


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "One Local Linear Regression"
)


points(
  x,
  y,
  cex = 0.5 +
    2 *
    weights
)


lines(
  x_grid,
  local_line,
  lty = 2,
  lwd = 2
)


points(
  x0,
  beta_local[1],
  pch = 19,
  cex = 1.5
)


# ------------------------------------------------------------------------------
# 16. Effect of Span
# ------------------------------------------------------------------------------

span_values <- c(
  0.1,
  0.2,
  0.4,
  0.7,
  1
)


span_predictions <- vector(
  "list",
  length(
    span_values
  )
)


for (
  i in seq_along(
    span_values
  )
) {
  
  span_i <- span_values[i]
  
  
  span_predictions[[i]] <- sapply(
    x_grid,
    function(x0) {
      
      local_polynomial(
        x,
        y,
        x0,
        span = span_i,
        degree = 1
      )
    }
  )
}


# ------------------------------------------------------------------------------
# 17. Plot Fits for Different Spans
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    2,
    3
  )
)


for (
  i in seq_along(
    span_values
  )
) {
  
  plot(
    x,
    y,
    pch = 19,
    cex = 0.5,
    xlab = "x",
    ylab = "y",
    main = paste(
      "Span =",
      span_values[i]
    )
  )
  
  
  lines(
    x_grid,
    span_predictions[[i]],
    lwd = 2
  )
  
  
  lines(
    x,
    y_true,
    lty = 2
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 18. Bias-Variance Interpretation
# ------------------------------------------------------------------------------

# Small span:
#
#   fewer observations in each local fit
#   lower bias
#   higher variance
#
# Large span:
#
#   more observations in each local fit
#   higher bias
#   lower variance


# ------------------------------------------------------------------------------
# 19. Prediction Function
# ------------------------------------------------------------------------------

local_regression_predict <- function(
    x_train,
    y_train,
    x_new,
    span = 0.3,
    degree = 1
) {
  
  sapply(
    x_new,
    function(x0) {
      
      local_polynomial(
        x = x_train,
        y = y_train,
        x0 = x0,
        span = span,
        degree = degree
      )
    }
  )
}


# ------------------------------------------------------------------------------
# 20. Training Predictions
# ------------------------------------------------------------------------------

training_prediction <- local_regression_predict(
  x_train = x,
  y_train = y,
  x_new = x,
  span = 0.3,
  degree = 1
)


training_MSE <- mean(
  (
    y -
      training_prediction
  )^2
)


training_MSE


# ------------------------------------------------------------------------------
# 21. Compare with Global Linear Regression
# ------------------------------------------------------------------------------

linear_model <- lm(
  y ~ x
)


linear_prediction <- predict(
  linear_model
)


linear_MSE <- mean(
  (
    y -
      linear_prediction
  )^2
)


data.frame(
  Model = c(
    "Global Linear Regression",
    "Local Linear Regression"
  ),
  Training_MSE = c(
    linear_MSE,
    training_MSE
  )
)


# ------------------------------------------------------------------------------
# 22. Compare Global and Local Fits
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Global vs Local Regression"
)


abline(
  linear_model,
  lty = 2,
  lwd = 2
)


lines(
  x_grid,
  local_linear_prediction,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Local Linear",
    "Global Linear",
    "True Function"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 23. Train-Test Split
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


x_train <- x[
  train_index
]


y_train <- y[
  train_index
]


x_test <- x[
  test_index
]


y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 24. Test Local Regression
# ------------------------------------------------------------------------------

test_prediction <- local_regression_predict(
  x_train = x_train,
  y_train = y_train,
  x_new = x_test,
  span = 0.3,
  degree = 1
)


test_MSE <- mean(
  (
    y_test -
      test_prediction
  )^2
)


test_RMSE <- sqrt(
  test_MSE
)


data.frame(
  Test_MSE = test_MSE,
  Test_RMSE = test_RMSE
)


# ------------------------------------------------------------------------------
# 25. K-Fold Cross-Validation for Span
# ------------------------------------------------------------------------------

local_regression_cv <- function(
    x,
    y,
    span_values,
    degree = 1,
    k = 5
) {
  
  n <- length(
    y
  )
  
  
  folds <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  CV_error <- matrix(
    NA,
    nrow = length(
      span_values
    ),
    ncol = k
  )
  
  
  for (
    fold in seq_len(
      k
    )
  ) {
    
    train_index <- which(
      folds != fold
    )
    
    
    validation_index <- which(
      folds == fold
    )
    
    
    x_train <- x[
      train_index
    ]
    
    
    y_train <- y[
      train_index
    ]
    
    
    x_validation <- x[
      validation_index
    ]
    
    
    y_validation <- y[
      validation_index
    ]
    
    
    for (
      i in seq_along(
        span_values
      )
    ) {
      
      prediction <- local_regression_predict(
        x_train = x_train,
        y_train = y_train,
        x_new = x_validation,
        span = span_values[i],
        degree = degree
      )
      
      
      CV_error[
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
      Span = span_values,
      Fold_MSE = CV_error,
      Mean_MSE = rowMeans(
        CV_error
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 26. Run Cross-Validation
# ------------------------------------------------------------------------------

span_grid <- seq(
  0.1,
  1,
  by = 0.05
)


set.seed(123)


CV_results <- local_regression_cv(
  x = x,
  y = y,
  span_values = span_grid,
  degree = 1,
  k = 5
)


# ------------------------------------------------------------------------------
# 27. Best Span
# ------------------------------------------------------------------------------

best_index <- which.min(
  CV_results$Mean_MSE
)


best_span <- CV_results$Span[
  best_index
]


best_span


# ------------------------------------------------------------------------------
# 28. Plot CV Error
# ------------------------------------------------------------------------------

plot(
  CV_results$Span,
  CV_results$Mean_MSE,
  type = "b",
  pch = 19,
  xlab = "Span",
  ylab = "Cross-Validated MSE",
  main = "Selecting Local Regression Span"
)


abline(
  v = best_span,
  lty = 2
)


# ------------------------------------------------------------------------------
# 29. Fit Final Local Regression
# ------------------------------------------------------------------------------

final_prediction <- local_regression_predict(
  x_train = x,
  y_train = y,
  x_new = x_grid,
  span = best_span,
  degree = 1
)


# ------------------------------------------------------------------------------
# 30. Plot Final Model
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Cross-Validated Local Regression"
)


lines(
  x_grid,
  final_prediction,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Local Regression",
    "True Function"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 31. Boundary Bias: Local Constant vs Local Linear
# ------------------------------------------------------------------------------

# One major advantage of local linear regression is improved
# behavior near the boundaries.


left_region <- x_grid <= 2


plot(
  x,
  y,
  pch = 19,
  cex = 0.5,
  xlim = c(
    0,
    2
  ),
  xlab = "x",
  ylab = "y",
  main = "Boundary Behavior"
)


lines(
  x_grid[
    left_region
  ],
  degree_0_prediction[
    left_region
  ],
  lty = 2,
  lwd = 2
)


lines(
  x_grid[
    left_region
  ],
  degree_1_prediction[
    left_region
  ],
  lty = 1,
  lwd = 2
)


lines(
  x[
    x <= 2
  ],
  y_true[
    x <= 2
  ],
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Local Linear",
    "Local Constant",
    "True Function"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 32. Approximate Smoother Matrix
# ------------------------------------------------------------------------------

# Local regression is a linear smoother:
#
# y_hat = S y
#
# if the neighborhoods and weights depend only on x.
#
# We can construct S by seeing how each observation contributes
# to each fitted value.


local_linear_weights <- function(
    x,
    x0,
    span = 0.3
) {
  
  n <- length(
    x
  )
  
  
  number_neighbors <- max(
    2,
    ceiling(
      span * n
    )
  )
  
  
  distances <- abs(
    x -
      x0
  )
  
  
  bandwidth <- sort(
    distances
  )[
    number_neighbors
  ]
  
  
  if (
    bandwidth == 0
  ) {
    
    positive_distances <- distances[
      distances > 0
    ]
    
    
    bandwidth <- min(
      positive_distances
    )
  }
  
  
  weights <- tricube(
    distances /
      bandwidth
  )
  
  
  X_local <- cbind(
    1,
    x -
      x0
  )
  
  
  W <- diag(
    weights
  )
  
  
  XtWX_inv <- tryCatch(
    solve(
      t(
        X_local
      ) %*%
        W %*%
        X_local
    ),
    error = function(e) {
      
      qr.solve(
        t(
          X_local
        ) %*%
          W %*%
          X_local
      )
    }
  )
  
  
  # Prediction at x0 selects the intercept:
  #
  # [1, 0] beta_hat
  
  selector <- matrix(
    c(
      1,
      0
    ),
    nrow = 1
  )
  
  
  smoother_weights <-
    selector %*%
    XtWX_inv %*%
    t(
      X_local
    ) %*%
    W
  
  
  return(
    as.vector(
      smoother_weights
    )
  )
}


# ------------------------------------------------------------------------------
# 33. Construct Local Regression Smoother Matrix
# ------------------------------------------------------------------------------

S <- matrix(
  0,
  nrow = n,
  ncol = n
)


for (
  i in seq_len(
    n
  )
) {
  
  S[
    i,
  ] <- local_linear_weights(
    x = x,
    x0 = x[i],
    span = best_span
  )
}


# ------------------------------------------------------------------------------
# 34. Verify Smoother Matrix
# ------------------------------------------------------------------------------

fitted_from_S <- as.vector(
  S %*%
    y
)


fitted_direct <- local_regression_predict(
  x_train = x,
  y_train = y,
  x_new = x,
  span = best_span,
  degree = 1
)


max(
  abs(
    fitted_from_S -
      fitted_direct
  )
)


# ------------------------------------------------------------------------------
# 35. Effective Degrees of Freedom
# ------------------------------------------------------------------------------

# For a linear smoother:
#
# df = trace(S)


effective_df <- sum(
  diag(
    S
  )
)


effective_df


# ------------------------------------------------------------------------------
# 36. Leverage
# ------------------------------------------------------------------------------

leverage <- diag(
  S
)


plot(
  x,
  leverage,
  type = "h",
  xlab = "x",
  ylab = "S_ii",
  main = "Local Regression Leverage"
)


# ------------------------------------------------------------------------------
# 37. LOOCV Shortcut for Linear Smoothers
# ------------------------------------------------------------------------------

# LOOCV residual:
#
# e_i^(-i)
# =
# (y_i - y_hat_i) / (1 - S_ii)


residuals <- y -
  fitted_from_S


LOOCV_residuals <- residuals /
  (
    1 -
      leverage
  )


LOOCV_MSE <- mean(
  LOOCV_residuals^2
)


LOOCV_MSE


# ------------------------------------------------------------------------------
# 38. Verify Against Base R loess()
# ------------------------------------------------------------------------------

builtin_loess <- loess(
  y ~ x,
  span = best_span,
  degree = 1,
  family = "gaussian"
)


builtin_prediction <- predict(
  builtin_loess,
  newdata = data.frame(
    x = x_grid
  )
)


# ------------------------------------------------------------------------------
# 39. Plot Manual vs Built-In LOESS
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Manual Local Regression vs loess()"
)


lines(
  x_grid,
  final_prediction,
  lwd = 2
)


lines(
  x_grid,
  builtin_prediction,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Manual",
    "loess()"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 40. Compare Manual and Built-In Predictions
# ------------------------------------------------------------------------------

valid <- !is.na(
  builtin_prediction
)


mean(
  (
    final_prediction[
      valid
    ] -
      builtin_prediction[
        valid
      ]
  )^2
)


# ------------------------------------------------------------------------------
# 41. Robust LOESS
# ------------------------------------------------------------------------------

# Base R loess() can optionally use robust reweighting.
#
# family = "symmetric"
#
# downweights observations with large residuals.


robust_loess <- loess(
  y ~ x,
  span = best_span,
  degree = 1,
  family = "symmetric"
)


robust_prediction <- predict(
  robust_loess,
  newdata = data.frame(
    x = x_grid
  )
)


# ------------------------------------------------------------------------------
# 42. Add Outliers to Demonstrate Robustness
# ------------------------------------------------------------------------------

set.seed(456)


y_outlier <- y


outlier_index <- sample(
  seq_len(n),
  size = 10
)


y_outlier[
  outlier_index
] <- y_outlier[
  outlier_index
] +
  rnorm(
    10,
    mean = 8,
    sd = 2
  )


ordinary_outlier_fit <- loess(
  y_outlier ~ x,
  span = best_span,
  degree = 1,
  family = "gaussian"
)


robust_outlier_fit <- loess(
  y_outlier ~ x,
  span = best_span,
  degree = 1,
  family = "symmetric"
)


ordinary_outlier_prediction <- predict(
  ordinary_outlier_fit,
  data.frame(
    x = x_grid
  )
)


robust_outlier_prediction <- predict(
  robust_outlier_fit,
  data.frame(
    x = x_grid
  )
)


# ------------------------------------------------------------------------------
# 43. Plot Robust LOESS
# ------------------------------------------------------------------------------

plot(
  x,
  y_outlier,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Robust Local Regression"
)


points(
  x[
    outlier_index
  ],
  y_outlier[
    outlier_index
  ],
  pch = 1,
  cex = 1.5,
  lwd = 2
)


lines(
  x_grid,
  ordinary_outlier_prediction,
  lty = 2,
  lwd = 2
)


lines(
  x_grid,
  robust_outlier_prediction,
  lty = 1,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Robust LOESS",
    "Ordinary LOESS",
    "True Function"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 44. Summary
# ------------------------------------------------------------------------------

cat(
  "Cross-validated span:",
  best_span,
  "\n"
)


cat(
  "Effective degrees of freedom:",
  round(
    effective_df,
    3
  ),
  "\n"
)


cat(
  "Training MSE:",
  round(
    training_MSE,
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
  "LOOCV MSE:",
  round(
    LOOCV_MSE,
    4
  ),
  "\n"
)