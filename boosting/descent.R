# ==============================================================================
# Steepest Descent
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Gradient-based optimization
#   - Steepest descent
#   - Direction of steepest decrease
#   - Learning rate / step size
#   - Exact line search
#   - Convergence
#   - Conditioning
#   - Feature scaling
#   - Linear regression via gradient descent
#   - Logistic regression via gradient descent
#   - Connection to gradient boosting
#
#
# General optimization problem:
#
#       minimize f(beta)
#
#
# Steepest descent update:
#
#       beta^(k+1)
#       =
#       beta^(k)
#       -
#       alpha_k * gradient f(beta^(k))
#
#
# The negative gradient
#
#       -gradient f(beta)
#
# is the direction of steepest local decrease under the Euclidean norm.
#
#
# In gradient boosting, the same idea is moved from parameter space
# into function space.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# LINEAR REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 200


x1 <- rnorm(
  n
)


x2 <- rnorm(
  n
)


beta_0_true <- 2

beta_1_true <- 3

beta_2_true <- -2


sigma <- 1.5


epsilon <- rnorm(
  n,
  mean = 0,
  sd = sigma
)


y <- beta_0_true +
  beta_1_true * x1 +
  beta_2_true * x2 +
  epsilon


# ------------------------------------------------------------------------------
# 2. Design Matrix
# ------------------------------------------------------------------------------

X <- cbind(
  Intercept = 1,
  x1 = x1,
  x2 = x2
)


p <- ncol(
  X
)


# ------------------------------------------------------------------------------
# 3. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data.frame(
    y = y,
    x1 = x1,
    x2 = x2
  ),
  main = "Linear Regression Data"
)


# ==============================================================================
# PART II
#
# LEAST-SQUARES OBJECTIVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Define Objective Function
# ------------------------------------------------------------------------------

# We use:
#
#       L(beta)
#       =
#       1 / (2n) * ||y - X beta||^2
#
#
# The factor 1/(2n) does not change the minimizer.
#
# It makes the gradient especially convenient:
#
#       gradient L(beta)
#       =
#       1/n X'(X beta - y)


least_squares_loss <- function(
    beta,
    X,
    y
) {
  
  n <- nrow(
    X
  )
  
  
  residuals <- y -
    as.vector(
      X %*% beta
    )
  
  
  sum(
    residuals^2
  ) /
    (
      2 * n
    )
}


# ------------------------------------------------------------------------------
# 5. Analytical Gradient
# ------------------------------------------------------------------------------

least_squares_gradient <- function(
    beta,
    X,
    y
) {
  
  n <- nrow(
    X
  )
  
  
  fitted_values <- as.vector(
    X %*% beta
  )
  
  
  gradient <- as.vector(
    crossprod(
      X,
      fitted_values -
        y
    )
  ) /
    n
  
  
  return(
    gradient
  )
}


# ==============================================================================
# PART III
#
# VERIFY THE GRADIENT NUMERICALLY
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Finite-Difference Gradient
# ------------------------------------------------------------------------------

numerical_gradient <- function(
    f,
    beta,
    h = 1e-6
) {
  
  gradient <- numeric(
    length(
      beta
    )
  )
  
  
  for (
    j in seq_along(
      beta
    )
  ) {
    
    beta_plus <- beta
    
    beta_minus <- beta
    
    
    beta_plus[j] <-
      beta_plus[j] +
      h
    
    
    beta_minus[j] <-
      beta_minus[j] -
      h
    
    
    gradient[j] <- (
      f(beta_plus) -
        f(beta_minus)
    ) /
      (
        2 * h
      )
  }
  
  
  return(
    gradient
  )
}


# ------------------------------------------------------------------------------
# 7. Compare Analytical and Numerical Gradients
# ------------------------------------------------------------------------------

beta_test <- c(
  0.5,
  1,
  -1
)


analytical_gradient <- least_squares_gradient(
  beta_test,
  X,
  y
)


numerical_gradient_result <- numerical_gradient(
  f = function(beta) {
    
    least_squares_loss(
      beta,
      X,
      y
    )
  },
  beta = beta_test
)


gradient_check <- data.frame(
  Parameter = colnames(X),
  Analytical = analytical_gradient,
  Numerical = numerical_gradient_result,
  Difference =
    analytical_gradient -
    numerical_gradient_result
)


gradient_check


# ==============================================================================
# PART IV
#
# MANUAL STEEPEST DESCENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Steepest Descent Function
# ------------------------------------------------------------------------------

steepest_descent <- function(
    X,
    y,
    beta_initial = NULL,
    learning_rate = 0.1,
    max_iter = 1000,
    tolerance = 1e-8,
    verbose = FALSE
) {
  
  X <- as.matrix(
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
    
    beta <- rep(
      0,
      p
    )
  } else {
    
    beta <- beta_initial
  }
  
  
  beta_history <- matrix(
    NA_real_,
    nrow = max_iter + 1,
    ncol = p
  )
  
  
  colnames(
    beta_history
  ) <- colnames(X)
  
  
  loss_history <- numeric(
    max_iter + 1
  )
  
  
  gradient_norm_history <- numeric(
    max_iter
  )
  
  
  beta_history[
    1,
  ] <- beta
  
  
  loss_history[1] <- least_squares_loss(
    beta,
    X,
    y
  )
  
  
  actual_iterations <- max_iter
  
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    gradient <- least_squares_gradient(
      beta,
      X,
      y
    )
    
    
    gradient_norm <- sqrt(
      sum(
        gradient^2
      )
    )
    
    
    gradient_norm_history[
      iteration
    ] <- gradient_norm
    
    
    # ------------------------------------------------------------------------
    # Steepest descent update
    # ------------------------------------------------------------------------
    
    beta_new <- beta -
      learning_rate *
      gradient
    
    
    beta_history[
      iteration + 1,
    ] <- beta_new
    
    
    loss_history[
      iteration + 1
    ] <- least_squares_loss(
      beta_new,
      X,
      y
    )
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        iteration,
        "| Loss:",
        round(
          loss_history[
            iteration + 1
          ],
          8
        ),
        "| Gradient Norm:",
        round(
          gradient_norm,
          8
        ),
        "\n"
      )
    }
    
    
    # ------------------------------------------------------------------------
    # Convergence criterion
    # ------------------------------------------------------------------------
    
    if (
      sqrt(
        sum(
          (
            beta_new -
            beta
          )^2
        )
      ) <
      tolerance
    ) {
      
      actual_iterations <- iteration
      
      beta <- beta_new
      
      break
    }
    
    
    beta <- beta_new
  }
  
  
  beta_history <- beta_history[
    seq_len(
      actual_iterations + 1
    ),
    ,
    drop = FALSE
  ]
  
  
  loss_history <- loss_history[
    seq_len(
      actual_iterations + 1
    )
  ]
  
  
  gradient_norm_history <-
    gradient_norm_history[
      seq_len(
        actual_iterations
      )
    ]
  
  
  return(
    list(
      coefficients = beta,
      beta_history = beta_history,
      loss_history = loss_history,
      gradient_norm_history =
        gradient_norm_history,
      iterations =
        actual_iterations,
      learning_rate =
        learning_rate
    )
  )
}


# ------------------------------------------------------------------------------
# 9. Fit Model
# ------------------------------------------------------------------------------

SD_model <- steepest_descent(
  X = X,
  y = y,
  learning_rate = 0.1,
  max_iter = 5000,
  tolerance = 1e-10
)


SD_model$coefficients


SD_model$iterations


# ==============================================================================
# PART V
#
# CLOSED-FORM OLS SOLUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. OLS
# ------------------------------------------------------------------------------

beta_OLS <- as.vector(
  solve(
    crossprod(X),
    crossprod(
      X,
      y
    )
  )
)


names(
  beta_OLS
) <- colnames(X)


beta_OLS


# ------------------------------------------------------------------------------
# 11. Compare Estimates
# ------------------------------------------------------------------------------

coefficient_comparison <- data.frame(
  Parameter = colnames(X),
  
  True = c(
    beta_0_true,
    beta_1_true,
    beta_2_true
  ),
  
  Steepest_Descent =
    SD_model$coefficients,
  
  Closed_Form_OLS =
    beta_OLS
)


coefficient_comparison


# ------------------------------------------------------------------------------
# 12. Maximum Difference
# ------------------------------------------------------------------------------

max(
  abs(
    SD_model$coefficients -
      beta_OLS
  )
)


# ==============================================================================
# PART VI
#
# CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Loss Across Iterations
# ------------------------------------------------------------------------------

plot(
  0:SD_model$iterations,
  SD_model$loss_history,
  type = "l",
  lwd = 2,
  xlab = "Iteration",
  ylab = "Least-Squares Loss",
  main = "Steepest Descent Convergence"
)


# ------------------------------------------------------------------------------
# 14. Log Loss Gap
# ------------------------------------------------------------------------------

optimal_loss <- least_squares_loss(
  beta_OLS,
  X,
  y
)


loss_gap <- SD_model$loss_history -
  optimal_loss


plot(
  0:SD_model$iterations,
  pmax(
    loss_gap,
    .Machine$double.eps
  ),
  type = "l",
  log = "y",
  lwd = 2,
  xlab = "Iteration",
  ylab = "Loss - Optimal Loss",
  main = "Convergence Toward OLS"
)


# ------------------------------------------------------------------------------
# 15. Gradient Norm
# ------------------------------------------------------------------------------

plot(
  seq_along(
    SD_model$gradient_norm_history
  ),
  SD_model$gradient_norm_history,
  type = "l",
  log = "y",
  lwd = 2,
  xlab = "Iteration",
  ylab = "Gradient Norm",
  main = "Gradient Norm During Optimization"
)


# ==============================================================================
# PART VII
#
# COEFFICIENT PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Plot Parameter Estimates
# ------------------------------------------------------------------------------

matplot(
  0:SD_model$iterations,
  SD_model$beta_history,
  type = "l",
  lty = 1,
  lwd = 2,
  xlab = "Iteration",
  ylab = "Coefficient",
  main = "Steepest Descent Coefficient Path"
)


abline(
  h = beta_OLS,
  lty = 2
)


legend(
  "right",
  legend = colnames(X),
  lty = 1,
  lwd = 2
)


# ==============================================================================
# PART VIII
#
# GEOMETRIC INTERPRETATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Two-Parameter Example
# ------------------------------------------------------------------------------

# To visualize the optimization path, use a model without an intercept:
#
#       y = beta1*x1 + beta2*x2 + noise
#
# We center y and X so an intercept is unnecessary.


X_centered <- scale(
  cbind(
    x1,
    x2
  ),
  center = TRUE,
  scale = FALSE
)


y_centered <- y -
  mean(y)


geometry_model <- steepest_descent(
  X = X_centered,
  y = y_centered,
  learning_rate = 0.1,
  max_iter = 500,
  tolerance = 1e-10
)


# ------------------------------------------------------------------------------
# 18. Loss Surface
# ------------------------------------------------------------------------------

beta1_grid <- seq(
  -1,
  5,
  length.out = 100
)


beta2_grid <- seq(
  -5,
  2,
  length.out = 100
)


loss_surface <- outer(
  beta1_grid,
  beta2_grid,
  Vectorize(
    function(b1, b2) {
      
      least_squares_loss(
        c(
          b1,
          b2
        ),
        X_centered,
        y_centered
      )
    }
  )
)


# ------------------------------------------------------------------------------
# 19. Contour Plot
# ------------------------------------------------------------------------------

contour(
  beta1_grid,
  beta2_grid,
  loss_surface,
  nlevels = 25,
  xlab = expression(beta[1]),
  ylab = expression(beta[2]),
  main = "Steepest Descent Optimization Path"
)


lines(
  geometry_model$beta_history[
    ,
    1
  ],
  geometry_model$beta_history[
    ,
    2
  ],
  type = "o",
  pch = 19,
  cex = 0.4,
  lwd = 2
)


points(
  geometry_model$beta_history[
    1,
    1
  ],
  geometry_model$beta_history[
    1,
    2
  ],
  pch = 19,
  cex = 1.5
)


points(
  tail(
    geometry_model$beta_history[
      ,
      1
    ],
    1
  ),
  tail(
    geometry_model$beta_history[
      ,
      2
    ],
    1
  ),
  pch = 4,
  lwd = 3,
  cex = 1.5
)


# ==============================================================================
# PART IX
#
# EFFECT OF LEARNING RATE
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Compare Step Sizes
# ------------------------------------------------------------------------------

learning_rates <- c(
  0.01,
  0.05,
  0.1,
  0.5,
  1
)


learning_rate_results <- data.frame(
  Learning_Rate =
    learning_rates,
  Iterations =
    NA_integer_,
  Final_Loss =
    NA_real_,
  Distance_From_OLS =
    NA_real_
)


learning_rate_models <- vector(
  "list",
  length(
    learning_rates
  )
)


for (
  i in seq_along(
    learning_rates
  )
) {
  
  model_i <- steepest_descent(
    X = X,
    y = y,
    learning_rate =
      learning_rates[i],
    max_iter = 5000,
    tolerance = 1e-8
  )
  
  
  learning_rate_models[[i]] <-
    model_i
  
  
  learning_rate_results$Iterations[i] <-
    model_i$iterations
  
  
  learning_rate_results$Final_Loss[i] <-
    tail(
      model_i$loss_history,
      1
    )
  
  
  learning_rate_results$Distance_From_OLS[i] <-
    sqrt(
      sum(
        (
          model_i$coefficients -
            beta_OLS
        )^2
      )
    )
}


learning_rate_results


# ------------------------------------------------------------------------------
# 21. Compare Loss Paths
# ------------------------------------------------------------------------------

plot(
  NULL,
  xlim = c(
    0,
    500
  ),
  ylim = range(
    unlist(
      lapply(
        learning_rate_models,
        function(model) {
          
          model$loss_history[
            seq_len(
              min(
                length(
                  model$loss_history
                ),
                501
              )
            )
          ]
        }
      )
    ),
    finite = TRUE
  ),
  xlab = "Iteration",
  ylab = "Loss",
  main = "Effect of Learning Rate"
)


for (
  i in seq_along(
    learning_rate_models
  )
) {
  
  loss_i <-
    learning_rate_models[[i]]$loss_history
  
  
  maximum_plot_iteration <- min(
    500,
    length(
      loss_i
    ) -
      1
  )
  
  
  lines(
    0:maximum_plot_iteration,
    loss_i[
      seq_len(
        maximum_plot_iteration + 1
      )
    ],
    lty = i,
    lwd = 2
  )
}


legend(
  "topright",
  legend = paste0(
    "alpha = ",
    learning_rates
  ),
  lty = seq_along(
    learning_rates
  ),
  lwd = 2
)


# ==============================================================================
# PART X
#
# WHY STEP SIZE MATTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Hessian of Least-Squares Loss
# ------------------------------------------------------------------------------

# For:
#
#       L(beta) = 1/(2n) ||y-X beta||^2
#
# the Hessian is:
#
#       H = X'X / n


H <- crossprod(
  X
) /
  n


H


# ------------------------------------------------------------------------------
# 23. Hessian Eigenvalues
# ------------------------------------------------------------------------------

H_eigenvalues <- eigen(
  H,
  symmetric = TRUE,
  only.values = TRUE
)$values


H_eigenvalues


lambda_max <- max(
  H_eigenvalues
)


lambda_min <- min(
  H_eigenvalues
)


condition_number <- lambda_max /
  lambda_min


condition_number


# ------------------------------------------------------------------------------
# 24. Stability Limit
# ------------------------------------------------------------------------------

# For this quadratic objective, fixed-step gradient descent converges when:
#
#       0 < alpha < 2 / lambda_max(H)


maximum_stable_step <- 2 /
  lambda_max


maximum_stable_step


cat(
  "Approximate maximum stable fixed step size:",
  maximum_stable_step,
  "\n"
)


# ==============================================================================
# PART XI
#
# EXACT LINE SEARCH
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Exact Step Size for Least Squares
# ------------------------------------------------------------------------------

# At beta_k:
#
#       g_k = gradient L(beta_k)
#
# and steepest descent direction:
#
#       d_k = -g_k
#
# For a quadratic objective, the exact line-search step is:
#
#       alpha_k
#       =
#       (g_k' g_k)
#       /
#       (g_k' H g_k)


exact_step_size <- function(
    gradient,
    H
) {
  
  numerator <- sum(
    gradient^2
  )
  
  
  denominator <- as.numeric(
    crossprod(
      gradient,
      H %*% gradient
    )
  )
  
  
  if (
    denominator <=
    0
  ) {
    
    stop(
      "Non-positive line-search denominator."
    )
  }
  
  
  numerator /
    denominator
}


# ------------------------------------------------------------------------------
# 26. Steepest Descent with Exact Line Search
# ------------------------------------------------------------------------------

steepest_descent_line_search <- function(
    X,
    y,
    beta_initial = NULL,
    max_iter = 1000,
    tolerance = 1e-10
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
  
  
  H <- crossprod(
    X
  ) /
    n
  
  
  if (
    is.null(
      beta_initial
    )
  ) {
    
    beta <- rep(
      0,
      p
    )
  } else {
    
    beta <- beta_initial
  }
  
  
  beta_history <- matrix(
    NA_real_,
    nrow = max_iter + 1,
    ncol = p
  )
  
  
  loss_history <- numeric(
    max_iter + 1
  )
  
  
  step_history <- numeric(
    max_iter
  )
  
  
  beta_history[
    1,
  ] <- beta
  
  
  loss_history[1] <- least_squares_loss(
    beta,
    X,
    y
  )
  
  
  actual_iterations <- max_iter
  
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    gradient <- least_squares_gradient(
      beta,
      X,
      y
    )
    
    
    if (
      sqrt(
        sum(
          gradient^2
        )
      ) <
      tolerance
    ) {
      
      actual_iterations <-
        iteration - 1
      
      break
    }
    
    
    alpha <- exact_step_size(
      gradient,
      H
    )
    
    
    beta <- beta -
      alpha *
      gradient
    
    
    beta_history[
      iteration + 1,
    ] <- beta
    
    
    loss_history[
      iteration + 1
    ] <- least_squares_loss(
      beta,
      X,
      y
    )
    
    
    step_history[
      iteration
    ] <- alpha
  }
  
  
  beta_history <- beta_history[
    seq_len(
      actual_iterations + 1
    ),
    ,
    drop = FALSE
  ]
  
  
  loss_history <- loss_history[
    seq_len(
      actual_iterations + 1
    )
  ]
  
  
  if (
    actual_iterations >
    0
  ) {
    
    step_history <- step_history[
      seq_len(
        actual_iterations
      )
    ]
    
  } else {
    
    step_history <- numeric(0)
  }
  
  
  return(
    list(
      coefficients = beta,
      beta_history =
        beta_history,
      loss_history =
        loss_history,
      step_history =
        step_history,
      iterations =
        actual_iterations
    )
  )
}


# ------------------------------------------------------------------------------
# 27. Fit with Exact Line Search
# ------------------------------------------------------------------------------

line_search_model <- steepest_descent_line_search(
  X = X,
  y = y
)


line_search_model$coefficients


line_search_model$iterations


# ------------------------------------------------------------------------------
# 28. Compare Fixed and Adaptive Step Sizes
# ------------------------------------------------------------------------------

plot(
  0:SD_model$iterations,
  SD_model$loss_history,
  type = "l",
  lwd = 2,
  xlab = "Iteration",
  ylab = "Loss",
  main = "Fixed Step vs Exact Line Search"
)


lines(
  0:line_search_model$iterations,
  line_search_model$loss_history,
  lty = 2,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Fixed Step",
    "Exact Line Search"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 29. Line-Search Step Sizes
# ------------------------------------------------------------------------------

if (
  length(
    line_search_model$step_history
  ) >
  0
) {
  
  plot(
    seq_along(
      line_search_model$step_history
    ),
    line_search_model$step_history,
    type = "b",
    pch = 19,
    xlab = "Iteration",
    ylab = "Step Size",
    main = "Exact Line-Search Step Sizes"
  )
}


# ==============================================================================
# PART XII
#
# CONDITIONING AND FEATURE SCALING
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Create Poorly Scaled Predictors
# ------------------------------------------------------------------------------

set.seed(321)


n_scale <- 300


z1 <- rnorm(
  n_scale,
  sd = 1
)


z2 <- rnorm(
  n_scale,
  sd = 100
)


z3 <- rnorm(
  n_scale,
  sd = 0.01
)


y_scale <- 2 +
  3 * z1 -
  0.02 * z2 +
  150 * z3 +
  rnorm(
    n_scale
  )


X_unscaled <- cbind(
  Intercept = 1,
  z1 = z1,
  z2 = z2,
  z3 = z3
)


# ------------------------------------------------------------------------------
# 31. Condition Number Before Scaling
# ------------------------------------------------------------------------------

H_unscaled <- crossprod(
  X_unscaled
) /
  n_scale


eigenvalues_unscaled <- eigen(
  H_unscaled,
  symmetric = TRUE,
  only.values = TRUE
)$values


condition_unscaled <- max(
  eigenvalues_unscaled
) /
  min(
    eigenvalues_unscaled
  )


condition_unscaled


# ------------------------------------------------------------------------------
# 32. Standardize Predictors
# ------------------------------------------------------------------------------

Z_scaled <- scale(
  cbind(
    z1,
    z2,
    z3
  )
)


X_scaled <- cbind(
  Intercept = 1,
  Z_scaled
)


# ------------------------------------------------------------------------------
# 33. Condition Number After Scaling
# ------------------------------------------------------------------------------

H_scaled <- crossprod(
  X_scaled
) /
  n_scale


eigenvalues_scaled <- eigen(
  H_scaled,
  symmetric = TRUE,
  only.values = TRUE
)$values


condition_scaled <- max(
  eigenvalues_scaled
) /
  min(
    eigenvalues_scaled
  )


data.frame(
  Design = c(
    "Unscaled",
    "Scaled"
  ),
  Condition_Number = c(
    condition_unscaled,
    condition_scaled
  )
)


# ------------------------------------------------------------------------------
# 34. Fit Scaled Model
# ------------------------------------------------------------------------------

scaled_SD <- steepest_descent(
  X = X_scaled,
  y = y_scale,
  learning_rate = 0.1,
  max_iter = 5000,
  tolerance = 1e-8
)


scaled_SD$iterations


# ==============================================================================
# PART XIII
#
# LOGISTIC REGRESSION WITH STEEPEST DESCENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Generate Binary Data
# ------------------------------------------------------------------------------

set.seed(456)


n_class <- 400


u1 <- rnorm(
  n_class
)


u2 <- rnorm(
  n_class
)


true_log_odds <- -0.5 +
  2 * u1 -
  1.5 * u2


sigmoid <- function(z) {
  
  positive <- z >= 0
  
  
  result <- numeric(
    length(z)
  )
  
  
  result[
    positive
  ] <- 1 /
    (
      1 +
        exp(
          -z[
            positive
          ]
        )
    )
  
  
  exp_z <- exp(
    z[
      !positive
    ]
  )
  
  
  result[
    !positive
  ] <- exp_z /
    (
      1 +
        exp_z
    )
  
  
  return(
    result
  )
}


probability <- sigmoid(
  true_log_odds
)


y_class <- rbinom(
  n_class,
  size = 1,
  prob = probability
)


X_class <- cbind(
  Intercept = 1,
  u1 = u1,
  u2 = u2
)


# ==============================================================================
# PART XIV
#
# LOGISTIC LOSS AND GRADIENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Stable Logistic Loss
# ------------------------------------------------------------------------------

softplus <- function(z) {
  
  pmax(
    z,
    0
  ) +
    log1p(
      exp(
        -abs(z)
      )
    )
}


logistic_loss <- function(
    beta,
    X,
    y
) {
  
  eta <- as.vector(
    X %*% beta
  )
  
  
  mean(
    softplus(
      eta
    ) -
      y * eta
  )
}


# ------------------------------------------------------------------------------
# 37. Logistic Gradient
# ------------------------------------------------------------------------------

logistic_gradient <- function(
    beta,
    X,
    y
) {
  
  n <- nrow(
    X
  )
  
  
  eta <- as.vector(
    X %*% beta
  )
  
  
  probability <- sigmoid(
    eta
  )
  
  
  as.vector(
    crossprod(
      X,
      probability -
        y
    )
  ) /
    n
}


# ------------------------------------------------------------------------------
# 38. Verify Logistic Gradient
# ------------------------------------------------------------------------------

beta_logistic_test <- c(
  0.1,
  0.5,
  -0.5
)


analytical_logistic_gradient <-
  logistic_gradient(
    beta_logistic_test,
    X_class,
    y_class
  )


numerical_logistic_gradient <-
  numerical_gradient(
    f = function(beta) {
      
      logistic_loss(
        beta,
        X_class,
        y_class
      )
    },
    beta =
      beta_logistic_test
  )


data.frame(
  Parameter = colnames(X_class),
  Analytical =
    analytical_logistic_gradient,
  Numerical =
    numerical_logistic_gradient,
  Difference =
    analytical_logistic_gradient -
    numerical_logistic_gradient
)


# ==============================================================================
# PART XV
#
# LOGISTIC STEEPEST DESCENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Fit Logistic Regression Manually
# ------------------------------------------------------------------------------

logistic_steepest_descent <- function(
    X,
    y,
    learning_rate = 0.1,
    max_iter = 10000,
    tolerance = 1e-8
) {
  
  X <- as.matrix(
    X
  )
  
  
  beta <- rep(
    0,
    ncol(X)
  )
  
  
  beta_history <- matrix(
    NA_real_,
    nrow = max_iter + 1,
    ncol = ncol(X)
  )
  
  
  loss_history <- numeric(
    max_iter + 1
  )
  
  
  beta_history[
    1,
  ] <- beta
  
  
  loss_history[1] <- logistic_loss(
    beta,
    X,
    y
  )
  
  
  actual_iterations <- max_iter
  
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    gradient <- logistic_gradient(
      beta,
      X,
      y
    )
    
    
    beta_new <- beta -
      learning_rate *
      gradient
    
    
    beta_history[
      iteration + 1,
    ] <- beta_new
    
    
    loss_history[
      iteration + 1
    ] <- logistic_loss(
      beta_new,
      X,
      y
    )
    
    
    if (
      sqrt(
        sum(
          (
            beta_new -
            beta
          )^2
        )
      ) <
      tolerance
    ) {
      
      actual_iterations <- iteration
      
      beta <- beta_new
      
      break
    }
    
    
    beta <- beta_new
  }
  
  
  return(
    list(
      coefficients = beta,
      
      beta_history =
        beta_history[
          seq_len(
            actual_iterations + 1
          ),
          ,
          drop = FALSE
        ],
      
      loss_history =
        loss_history[
          seq_len(
            actual_iterations + 1
          )
        ],
      
      iterations =
        actual_iterations
    )
  )
}


# ------------------------------------------------------------------------------
# 40. Fit Model
# ------------------------------------------------------------------------------

logistic_SD <- logistic_steepest_descent(
  X = X_class,
  y = y_class,
  learning_rate = 0.1
)


logistic_SD$coefficients


# ==============================================================================
# PART XVI
#
# VERIFY AGAINST GLM
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Built-In Logistic Regression
# ------------------------------------------------------------------------------

glm_model <- glm(
  y_class ~ u1 + u2,
  family = binomial()
)


coef(
  glm_model
)


# ------------------------------------------------------------------------------
# 42. Compare Coefficients
# ------------------------------------------------------------------------------

data.frame(
  Parameter = colnames(X_class),
  Steepest_Descent =
    logistic_SD$coefficients,
  GLM =
    as.vector(
      coef(
        glm_model
      )
    )
)


# ------------------------------------------------------------------------------
# 43. Logistic Loss Path
# ------------------------------------------------------------------------------

plot(
  0:logistic_SD$iterations,
  logistic_SD$loss_history,
  type = "l",
  lwd = 2,
  xlab = "Iteration",
  ylab = "Logistic Loss",
  main = "Logistic Regression via Steepest Descent"
)


# ==============================================================================
# PART XVII
#
# LOGISTIC CLASSIFICATION PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Predictions
# ------------------------------------------------------------------------------

logistic_probability_SD <- sigmoid(
  as.vector(
    X_class %*%
      logistic_SD$coefficients
  )
)


logistic_prediction_SD <- ifelse(
  logistic_probability_SD >=
    0.5,
  1,
  0
)


accuracy_SD <- mean(
  logistic_prediction_SD ==
    y_class
)


accuracy_SD


# ------------------------------------------------------------------------------
# 45. Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y_class,
  Predicted =
    logistic_prediction_SD
)


# ==============================================================================
# PART XVIII
#
# CONNECTION TO BOOSTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Ordinary Parameter-Space Gradient Descent
# ------------------------------------------------------------------------------

# Suppose:
#
#       f_beta(x) = x' beta
#
# and we minimize:
#
#       L(beta)
#
#
# Steepest descent computes:
#
#       g_k = gradient_beta L(beta_k)
#
# and updates:
#
#       beta_(k+1)
#       =
#       beta_k
#       -
#       alpha_k g_k
#
#
# The search direction is therefore:
#
#       -g_k


# ------------------------------------------------------------------------------
# 47. Function-Space Analogy
# ------------------------------------------------------------------------------

# Gradient boosting instead optimizes over functions:
#
#       F(x)
#
# rather than directly over a finite-dimensional beta.
#
#
# At iteration m:
#
#       pseudo_residual_i
#
#       =
#
#       - [
#           partial L(y_i, F(x_i))
#           -----------------------
#                partial F(x_i)
#         ]
#
#
# A weak learner h_m(x) is then fitted to these negative gradients.
#
#
# The model is updated:
#
#       F_m(x)
#       =
#       F_(m-1)(x)
#       +
#       nu * h_m(x)
#
#
# Therefore:
#
#       PARAMETER SPACE:
#
#           beta <- beta - alpha * gradient
#
#
#       FUNCTION SPACE:
#
#           F <- F + nu * weak_learner
#
#
# where the weak learner approximates the negative gradient.


# ==============================================================================
# PART XIX
#
# SQUARED-ERROR NEGATIVE GRADIENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Verify Residual Connection
# ------------------------------------------------------------------------------

# For:
#
#       L(y, F)
#       =
#       1/2 (y-F)^2
#
#
# derivative:
#
#       dL/dF
#       =
#       F-y
#
#
# negative gradient:
#
#       -dL/dF
#       =
#       y-F
#
#
# which is exactly the residual.


initial_prediction <- rep(
  mean(y),
  n
)


residuals <- y -
  initial_prediction


negative_gradient_squared_error <-
  y -
  initial_prediction


max(
  abs(
    residuals -
      negative_gradient_squared_error
  )
)


# ==============================================================================
# PART XX
#
# LOGISTIC NEGATIVE GRADIENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Binary Log-Loss Gradient
# ------------------------------------------------------------------------------

# For logistic loss with y in {0,1}:
#
#       L(y,F)
#       =
#       log(1 + exp(F)) - yF
#
#
# where:
#
#       p = sigmoid(F)
#
#
# derivative:
#
#       dL/dF
#       =
#       p-y
#
#
# negative gradient:
#
#       y-p
#
#
# Thus gradient boosting for logistic loss fits weak learners to:
#
#       y_i - p_i
#
# rather than ordinary regression residuals.


initial_log_odds <- log(
  mean(y_class) /
    (
      1 -
        mean(y_class)
    )
)


initial_F <- rep(
  initial_log_odds,
  n_class
)


initial_probability <- sigmoid(
  initial_F
)


logistic_pseudo_residual <- y_class -
  initial_probability


summary(
  logistic_pseudo_residual
)


# ==============================================================================
# PART XXI
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Summary
# ------------------------------------------------------------------------------

cat(
  "Steepest Descent Summary\n"
)


cat(
  "------------------------\n"
)


cat(
  "True linear coefficients:",
  c(
    beta_0_true,
    beta_1_true,
    beta_2_true
  ),
  "\n"
)


cat(
  "Steepest descent coefficients:",
  round(
    SD_model$coefficients,
    4
  ),
  "\n"
)


cat(
  "OLS coefficients:",
  round(
    beta_OLS,
    4
  ),
  "\n"
)


cat(
  "Steepest descent iterations:",
  SD_model$iterations,
  "\n"
)


cat(
  "Exact line-search iterations:",
  line_search_model$iterations,
  "\n"
)


cat(
  "Hessian condition number:",
  round(
    condition_number,
    4
  ),
  "\n"
)


cat(
  "Maximum stable fixed step:",
  round(
    maximum_stable_step,
    4
  ),
  "\n"
)


cat(
  "\nLogistic regression coefficients:\n"
)


print(
  round(
    logistic_SD$coefficients,
    4
  )
)


cat(
  "\nLogistic training accuracy:",
  round(
    accuracy_SD,
    4
  ),
  "\n"
)