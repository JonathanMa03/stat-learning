# ==============================================================================
# Fused Lasso for Functional Data
# Jonathan Ma
#
# Package-free pedagogical implementation.
#
# Main ideas:
#   - Functional linear regression
#   - Discretized functional predictors
#   - Ordered coefficient vectors
#   - Ordinary Lasso
#   - Total variation / fusion penalty
#   - Fused Lasso
#   - Difference matrices
#   - ADMM optimization
#   - Sparsity versus fusion
#   - Piecewise-constant coefficient functions
#   - Cross-validation
#   - Functional coefficient recovery
#   - Prediction
#
#
# Functional linear model:
#
#       y_i
#
#       =
#
#       beta_0
#
#       +
#
#       integral X_i(t) beta(t) dt
#
#       +
#
#       epsilon_i
#
#
# After observing each function on a grid:
#
#       t_1, ..., t_p
#
# we obtain approximately:
#
#       y_i
#
#       =
#
#       beta_0
#
#       +
#
#       sum_j X_ij beta_j
#
#       +
#
#       epsilon_i
#
#
# Fused Lasso objective:
#
#       minimize_beta
#
#           (1/(2n)) ||y - X beta||_2^2
#
#           +
#
#           lambda_1 sum_j |beta_j|
#
#           +
#
#           lambda_2 sum_j |beta_j - beta_{j-1}|
#
#
# lambda_1:
#
#       sparsity
#
# lambda_2:
#
#       fusion / piecewise constancy
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE FUNCTIONAL DATA
# ==============================================================================


set.seed(123)


n <- 300

p <- 80


time_grid <- seq(
  0,
  1,
  length.out = p
)


delta_t <- time_grid[2] -
  time_grid[1]


# ==============================================================================
# PART II
#
# GENERATE SMOOTH FUNCTIONAL PREDICTORS
# ==============================================================================


# ------------------------------------------------------------------------------
# Each observation is a complete function X_i(t).
#
# We construct functions from a few smooth latent basis functions.
# ------------------------------------------------------------------------------

basis_1 <- sin(
  2 * pi * time_grid
)


basis_2 <- cos(
  2 * pi * time_grid
)


basis_3 <- sin(
  4 * pi * time_grid
)


basis_4 <- cos(
  6 * pi * time_grid
)


basis_5 <- exp(
  -20 *
    (
      time_grid -
        0.5
    )^2
)


basis_matrix <- cbind(
  basis_1,
  basis_2,
  basis_3,
  basis_4,
  basis_5
)


number_basis <- ncol(
  basis_matrix
)


latent_scores <- matrix(
  rnorm(
    n *
      number_basis
  ),
  nrow = n,
  ncol =
    number_basis
)


X <- latent_scores %*%
  t(
    basis_matrix
  )


# ------------------------------------------------------------------------------
# Add smooth observation-level noise.
# ------------------------------------------------------------------------------

smooth_noise <- matrix(
  rnorm(
    n * p,
    sd = 0.20
  ),
  nrow = n,
  ncol = p
)


for (
  i in seq_len(n)
) {
  
  smooth_noise[i, ] <- as.numeric(
    stats::filter(
      smooth_noise[i, ],
      filter = rep(
        1 / 5,
        5
      ),
      sides = 2
    )
  )
  
  
  missing_values <- is.na(
    smooth_noise[i, ]
  )
  
  
  smooth_noise[
    i,
    missing_values
  ] <- 0
}


X <- X +
  smooth_noise


# ==============================================================================
# PART III
#
# TRUE FUNCTIONAL COEFFICIENT
# ==============================================================================


# ------------------------------------------------------------------------------
# The true beta(t) contains several contiguous regions.
#
# This is exactly the kind of structure fused lasso is designed to recover.
# ------------------------------------------------------------------------------

beta_true <- rep(
  0,
  p
)


beta_true[
  time_grid >=
    0.15 &
    time_grid <
    0.30
] <- 2.5


beta_true[
  time_grid >=
    0.45 &
    time_grid <
    0.60
] <- -3.0


beta_true[
  time_grid >=
    0.72 &
    time_grid <
    0.85
] <- 1.75


plot(
  time_grid,
  beta_true,
  type = "s",
  lwd = 3,
  xlab = "t",
  ylab = expression(
    beta(t)
  ),
  main = "True Functional Coefficient"
)


# ==============================================================================
# PART IV
#
# GENERATE RESPONSE
# ==============================================================================


# ------------------------------------------------------------------------------
# Numerical approximation:
#
#       integral X_i(t) beta(t) dt
#
# approximately:
#
#       delta_t * sum_j X_ij beta_j
#
#
# For optimization, we absorb delta_t into the design matrix.
# ------------------------------------------------------------------------------

X_functional <- X *
  delta_t


beta_0_true <- 2


signal <- beta_0_true +
  as.numeric(
    X_functional %*%
      beta_true
  )


sigma <- 0.50


y <- signal +
  rnorm(
    n,
    sd = sigma
  )


# ==============================================================================
# PART V
#
# VISUALIZE FUNCTIONAL DATA
# ==============================================================================


set.seed(321)


example_functions <- sample(
  seq_len(n),
  20
)


matplot(
  time_grid,
  t(
    X[
      example_functions,
      ,
      drop = FALSE
    ]
  ),
  type = "l",
  lty = 1,
  xlab = "t",
  ylab = "X(t)",
  main = "Example Functional Predictors"
)


# ==============================================================================
# PART VI
#
# TRAIN / TEST SPLIT
# ==============================================================================


set.seed(456)


train_indices <- sample(
  seq_len(n),
  size = floor(
    0.70 * n
  )
)


test_indices <- setdiff(
  seq_len(n),
  train_indices
)


X_train_raw <- X_functional[
  train_indices,
  ,
  drop = FALSE
]


X_test_raw <- X_functional[
  test_indices,
  ,
  drop = FALSE
]


y_train <- y[
  train_indices
]


y_test <- y[
  test_indices
]


n_train <- nrow(
  X_train_raw
)


# ==============================================================================
# PART VII
#
# CENTER THE DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# We leave the intercept unpenalized by centering X and y.
#
# IMPORTANT:
#
# Functional columns are NOT individually variance-standardized here.
#
# Each column represents the same physical quantity observed at neighboring
# grid locations. Scaling every location differently would change the meaning
# of the fusion penalty:
#
#       |beta_j - beta_{j-1}|
#
# because adjacent coefficients would no longer be on the same scale.
# ------------------------------------------------------------------------------

x_means <- colMeans(
  X_train_raw
)


y_mean <- mean(
  y_train
)


X_train <- sweep(
  X_train_raw,
  2,
  x_means,
  "-"
)


X_test <- sweep(
  X_test_raw,
  2,
  x_means,
  "-"
)


y_train_centered <- y_train -
  y_mean


# ==============================================================================
# PART VIII
#
# REGRESSION METRICS
# ==============================================================================


regression_metrics <- function(
    observed,
    predicted
) {
  
  mse <- mean(
    (
      observed -
        predicted
    )^2
  )
  
  
  rmse <- sqrt(
    mse
  )
  
  
  mae <- mean(
    abs(
      observed -
        predicted
    )
  )
  
  
  r_squared <- 1 -
    sum(
      (
        observed -
          predicted
      )^2
    ) /
    sum(
      (
        observed -
          mean(
            observed
          )
      )^2
    )
  
  
  c(
    MSE = mse,
    RMSE = rmse,
    MAE = mae,
    R2 = r_squared
  )
}


# ==============================================================================
# PART IX
#
# FIRST-DIFFERENCE MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# Define D so that:
#
#       D beta
#
#       =
#
#       (
#           beta_2 - beta_1,
#           beta_3 - beta_2,
#           ...
#           beta_p - beta_{p-1}
#       )
#
#
# D has dimension:
#
#       (p - 1) x p
# ------------------------------------------------------------------------------

first_difference_matrix <- function(p) {
  
  D <- matrix(
    0,
    nrow = p - 1,
    ncol = p
  )
  
  
  for (
    j in seq_len(
      p - 1
    )
  ) {
    
    D[
      j,
      j
    ] <- -1
    
    
    D[
      j,
      j + 1
    ] <- 1
  }
  
  
  D
}


D <- first_difference_matrix(
  p
)


dim(
  D
)


# ------------------------------------------------------------------------------
# Verify Difference Operator
# ------------------------------------------------------------------------------

test_beta <- c(
  1,
  1,
  3,
  3,
  0
)


test_D <- first_difference_matrix(
  length(
    test_beta
  )
)


as.numeric(
  test_D %*%
    test_beta
)


# Expected:
#
#       0, 2, 0, -3


# ==============================================================================
# PART X
#
# TOTAL VARIATION
# ==============================================================================


total_variation <- function(beta) {
  
  sum(
    abs(
      diff(
        beta
      )
    )
  )
}


total_variation(
  beta_true
)


# ==============================================================================
# PART XI
#
# SOFT THRESHOLDING
# ==============================================================================


soft_threshold <- function(
    x,
    threshold
) {
  
  sign(x) *
    pmax(
      abs(x) -
        threshold,
      0
    )
}


# ==============================================================================
# PART XII
#
# FUSED LASSO AS A GENERALIZED LASSO
# ==============================================================================


# ------------------------------------------------------------------------------
# Objective:
#
#       (1/(2n)) ||y - X beta||^2
#
#       +
#
#       lambda_1 ||beta||_1
#
#       +
#
#       lambda_2 ||D beta||_1
#
#
# Introduce auxiliary variables:
#
#       z = beta
#
#       u = D beta
#
#
# Then solve:
#
#       minimize
#
#           loss(beta)
#
#           +
#
#           lambda_1 ||z||_1
#
#           +
#
#           lambda_2 ||u||_1
#
#
# subject to:
#
#       beta = z
#
#       D beta = u
#
#
# ADMM makes these pieces easy to optimize separately.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XIII
#
# MANUAL ADMM FUSED LASSO
# ==============================================================================


fused_lasso_admm <- function(
    X,
    y,
    lambda_1,
    lambda_2,
    rho = 1,
    maximum_iterations = 5000,
    absolute_tolerance = 1e-5,
    relative_tolerance = 1e-4,
    beta_initial = NULL,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  y <- as.numeric(
    y
  )
  
  
  n <- nrow(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  D <- first_difference_matrix(
    p
  )
  
  
  # --------------------------------------------------------------------------
  # Initialize Variables
  # --------------------------------------------------------------------------
  
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
    
    beta <- as.numeric(
      beta_initial
    )
  }
  
  
  z <- beta
  
  
  u <- as.numeric(
    D %*%
      beta
  )
  
  
  dual_z <- rep(
    0,
    p
  )
  
  
  dual_u <- rep(
    0,
    p - 1
  )
  
  
  # --------------------------------------------------------------------------
  # Matrices That Do Not Change Across Iterations
  #
  # beta update solves:
  #
  #   [X'X/n + rho I + rho D'D] beta
  #
  #       =
  #
  #   X'y/n
  #
  #       +
  #
  #   rho(z - dual_z)
  #
  #       +
  #
  #   rho D'(u - dual_u)
  # --------------------------------------------------------------------------
  
  system_matrix <- crossprod(
    X
  ) /
    n +
    rho *
    diag(p) +
    rho *
    crossprod(
      D
    )
  
  
  Xty <- as.numeric(
    crossprod(
      X,
      y
    )
  ) /
    n
  
  
  # --------------------------------------------------------------------------
  # Cholesky factorization.
  #
  # This is preferable to explicitly computing the inverse.
  # --------------------------------------------------------------------------
  
  chol_system <- chol(
    system_matrix
  )
  
  
  solve_system <- function(rhs) {
    
    backsolve(
      chol_system,
      forwardsolve(
        t(
          chol_system
        ),
        rhs
      )
    )
  }
  
  
  objective_history <- numeric(
    maximum_iterations
  )
  
  
  primal_history <- numeric(
    maximum_iterations
  )
  
  
  dual_history <- numeric(
    maximum_iterations
  )
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      maximum_iterations
    )
  ) {
    
    z_old <- z
    
    u_old <- u
    
    
    # ========================================================================
    # BETA UPDATE
    # ========================================================================
    
    rhs <- Xty +
      rho *
      (
        z -
          dual_z
      ) +
      rho *
      as.numeric(
        crossprod(
          D,
          u -
            dual_u
        )
      )
    
    
    beta <- as.numeric(
      solve_system(
        rhs
      )
    )
    
    
    # ========================================================================
    # Z UPDATE
    #
    # Handles:
    #
    #       lambda_1 ||beta||_1
    # ========================================================================
    
    z <- soft_threshold(
      beta +
        dual_z,
      lambda_1 /
        rho
    )
    
    
    # ========================================================================
    # U UPDATE
    #
    # Handles:
    #
    #       lambda_2 ||D beta||_1
    # ========================================================================
    
    D_beta <- as.numeric(
      D %*%
        beta
    )
    
    
    u <- soft_threshold(
      D_beta +
        dual_u,
      lambda_2 /
        rho
    )
    
    
    # ========================================================================
    # DUAL UPDATES
    # ========================================================================
    
    dual_z <- dual_z +
      beta -
      z
    
    
    dual_u <- dual_u +
      D_beta -
      u
    
    
    # ========================================================================
    # OBJECTIVE
    # ========================================================================
    
    residual <- y -
      as.numeric(
        X %*%
          beta
      )
    
    
    objective <- mean(
      residual^2
    ) /
      2 +
      lambda_1 *
      sum(
        abs(
          beta
        )
      ) +
      lambda_2 *
      sum(
        abs(
          D_beta
        )
      )
    
    
    objective_history[
      iteration
    ] <- objective
    
    
    # ========================================================================
    # ADMM RESIDUALS
    # ========================================================================
    
    primal_residual <- sqrt(
      sum(
        (
          beta -
            z
        )^2
      ) +
        sum(
          (
            D_beta -
              u
          )^2
        )
    )
    
    
    dual_residual <- rho *
      sqrt(
        sum(
          (
            z -
              z_old
          )^2
        ) +
          sum(
            as.numeric(
              crossprod(
                D,
                u -
                  u_old
              )
            )^2
          )
      )
    
    
    primal_history[
      iteration
    ] <- primal_residual
    
    
    dual_history[
      iteration
    ] <- dual_residual
    
    
    # ========================================================================
    # STOPPING TOLERANCES
    # ========================================================================
    
    primal_dimension <- p +
      (
        p -
          1
      )
    
    
    epsilon_primal <- sqrt(
      primal_dimension
    ) *
      absolute_tolerance +
      relative_tolerance *
      max(
        sqrt(
          sum(
            beta^2
          ) +
            sum(
              D_beta^2
            )
        ),
        sqrt(
          sum(
            z^2
          ) +
            sum(
              u^2
            )
        )
      )
    
    
    dual_quantity <- rho *
      (
        dual_z +
          as.numeric(
            crossprod(
              D,
              dual_u
            )
          )
      )
    
    
    epsilon_dual <- sqrt(
      p
    ) *
      absolute_tolerance +
      relative_tolerance *
      sqrt(
        sum(
          dual_quantity^2
        )
      )
    
    
    if (
      verbose &&
      (
        iteration ==
        1 ||
        iteration %% 100 ==
        0
      )
    ) {
      
      cat(
        "Iteration:",
        iteration,
        " Objective:",
        round(
          objective,
          6
        ),
        " Primal:",
        round(
          primal_residual,
          6
        ),
        " Dual:",
        round(
          dual_residual,
          6
        ),
        "\n"
      )
    }
    
    
    if (
      primal_residual <=
      epsilon_primal &&
      dual_residual <=
      epsilon_dual
    ) {
      
      converged <- TRUE
      
      break
    }
  }
  
  
  list(
    beta =
      beta,
    z =
      z,
    fused_difference =
      u,
    objective_history =
      objective_history[
        seq_len(
          iteration
        )
      ],
    primal_history =
      primal_history[
        seq_len(
          iteration
        )
      ],
    dual_history =
      dual_history[
        seq_len(
          iteration
        )
      ],
    iterations =
      iteration,
    converged =
      converged,
    lambda_1 =
      lambda_1,
    lambda_2 =
      lambda_2,
    rho =
      rho
  )
}


# ==============================================================================
# PART XIV
#
# FIRST FUSED LASSO FIT
# ==============================================================================


initial_fit <- fused_lasso_admm(
  X_train,
  y_train_centered,
  lambda_1 = 0.002,
  lambda_2 = 0.02,
  rho = 1,
  maximum_iterations = 5000,
  verbose = TRUE
)


initial_fit$converged


initial_fit$iterations


# ==============================================================================
# PART XV
#
# ADMM CONVERGENCE
# ==============================================================================


par(
  mfrow = c(1, 3)
)


plot(
  initial_fit$objective_history,
  type = "l",
  xlab = "Iteration",
  ylab = "Objective",
  main = "Objective"
)


plot(
  initial_fit$primal_history,
  type = "l",
  log = "y",
  xlab = "Iteration",
  ylab = "Primal Residual",
  main = "Primal Residual"
)


plot(
  initial_fit$dual_history,
  type = "l",
  log = "y",
  xlab = "Iteration",
  ylab = "Dual Residual",
  main = "Dual Residual"
)


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XVI
#
# ESTIMATED COEFFICIENT FUNCTION
# ==============================================================================


plot(
  time_grid,
  beta_true,
  type = "s",
  lwd = 3,
  xlab = "t",
  ylab = expression(
    beta(t)
  ),
  main = "True vs Fused-Lasso Coefficient"
)


lines(
  time_grid,
  initial_fit$beta,
  type = "s",
  lwd = 2,
  lty = 2
)


legend(
  "topright",
  legend = c(
    "True",
    "Fused Lasso"
  ),
  lty = c(
    1,
    2
  ),
  lwd = c(
    3,
    2
  )
)


# ==============================================================================
# PART XVII
#
# PREDICTION
# ==============================================================================


predict_functional_model <- function(
    X_new_raw,
    x_means,
    y_mean,
    beta
) {
  
  X_new <- sweep(
    X_new_raw,
    2,
    x_means,
    "-"
  )
  
  
  y_mean +
    as.numeric(
      X_new %*%
        beta
    )
}


initial_prediction <- predict_functional_model(
  X_test_raw,
  x_means,
  y_mean,
  initial_fit$beta
)


regression_metrics(
  y_test,
  initial_prediction
)


# ==============================================================================
# PART XVIII
#
# COEFFICIENT RECOVERY METRICS
# ==============================================================================


coefficient_mse <- function(
    beta_hat,
    beta_true
) {
  
  mean(
    (
      beta_hat -
        beta_true
    )^2
  )
}


coefficient_correlation <- function(
    beta_hat,
    beta_true
) {
  
  cor(
    beta_hat,
    beta_true
  )
}


coefficient_mse(
  initial_fit$beta,
  beta_true
)


coefficient_correlation(
  initial_fit$beta,
  beta_true
)


# ==============================================================================
# PART XIX
#
# SPARSITY AND FUSION DIAGNOSTICS
# ==============================================================================


number_nonzero <- function(
    beta,
    tolerance = 1e-3
) {
  
  sum(
    abs(
      beta
    ) >
      tolerance
  )
}


number_jumps <- function(
    beta,
    tolerance = 1e-3
) {
  
  sum(
    abs(
      diff(
        beta
      )
    ) >
      tolerance
  )
}


data.frame(
  Quantity = c(
    "Nonzero coefficients",
    "Adjacent jumps",
    "Total variation"
  ),
  Value = c(
    number_nonzero(
      initial_fit$beta
    ),
    number_jumps(
      initial_fit$beta
    ),
    total_variation(
      initial_fit$beta
    )
  )
)


# ==============================================================================
# PART XX
#
# ORDINARY LEAST SQUARES
# ==============================================================================


# ------------------------------------------------------------------------------
# Functional predictors are strongly correlated across nearby grid points.
#
# OLS can therefore be unstable.
#
# Use QR-based lm.fit rather than explicitly inverting X'X.
# ------------------------------------------------------------------------------

ols_fit <- lm.fit(
  x = cbind(
    1,
    X_train_raw
  ),
  y = y_train
)


ols_coefficients <- coef(
  ols_fit
)


ols_intercept <- ols_coefficients[1]


ols_beta <- ols_coefficients[
  -1
]


# ------------------------------------------------------------------------------
# lm.fit may return NA coefficients under exact rank deficiency.
# ------------------------------------------------------------------------------

ols_beta[
  is.na(
    ols_beta
  )
] <- 0


if (
  is.na(
    ols_intercept
  )
) {
  
  ols_intercept <- mean(
    y_train
  )
}


ols_prediction <- ols_intercept +
  as.numeric(
    X_test_raw %*%
      ols_beta
  )


regression_metrics(
  y_test,
  ols_prediction
)


# ==============================================================================
# PART XXI
#
# ORDINARY LASSO
# ==============================================================================


# ------------------------------------------------------------------------------
# Setting:
#
#       lambda_2 = 0
#
# gives ordinary Lasso.
#
# This is useful for understanding what the fusion penalty adds.
# ------------------------------------------------------------------------------

lasso_fit <- fused_lasso_admm(
  X_train,
  y_train_centered,
  lambda_1 = 0.002,
  lambda_2 = 0,
  rho = 1,
  maximum_iterations = 5000
)


lasso_prediction <- predict_functional_model(
  X_test_raw,
  x_means,
  y_mean,
  lasso_fit$beta
)


regression_metrics(
  y_test,
  lasso_prediction
)


# ==============================================================================
# PART XXII
#
# PURE FUSION PENALTY
# ==============================================================================


# ------------------------------------------------------------------------------
# Setting:
#
#       lambda_1 = 0
#
# leaves only the total-variation penalty:
#
#       lambda_2 ||D beta||_1
#
#
# This encourages piecewise constancy but does not directly encourage zero
# coefficients.
# ------------------------------------------------------------------------------

fusion_only_fit <- fused_lasso_admm(
  X_train,
  y_train_centered,
  lambda_1 = 0,
  lambda_2 = 0.02,
  rho = 1,
  maximum_iterations = 5000
)


fusion_prediction <- predict_functional_model(
  X_test_raw,
  x_means,
  y_mean,
  fusion_only_fit$beta
)


regression_metrics(
  y_test,
  fusion_prediction
)


# ==============================================================================
# PART XXIII
#
# COMPARE COEFFICIENT FUNCTIONS
# ==============================================================================


par(
  mfrow = c(2, 2)
)


plot(
  time_grid,
  beta_true,
  type = "s",
  lwd = 3,
  xlab = "t",
  ylab = expression(
    beta(t)
  ),
  main = "True Coefficient"
)


plot(
  time_grid,
  ols_beta,
  type = "l",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = "OLS"
)


abline(
  h = 0,
  lty = 3
)


plot(
  time_grid,
  lasso_fit$beta,
  type = "s",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = "Lasso"
)


abline(
  h = 0,
  lty = 3
)


plot(
  time_grid,
  initial_fit$beta,
  type = "s",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = "Fused Lasso"
)


abline(
  h = 0,
  lty = 3
)


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XXIV
#
# LASSO VS FUSED LASSO
# ==============================================================================


plot(
  time_grid,
  beta_true,
  type = "s",
  lwd = 3,
  xlab = "t",
  ylab = expression(
    beta(t)
  ),
  main = "Lasso vs Fused Lasso"
)


lines(
  time_grid,
  lasso_fit$beta,
  type = "s",
  lwd = 2,
  lty = 2
)


lines(
  time_grid,
  initial_fit$beta,
  type = "s",
  lwd = 2,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "True",
    "Lasso",
    "Fused Lasso"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = c(
    3,
    2,
    2
  )
)


# ==============================================================================
# PART XXV
#
# EFFECT OF LAMBDA_1
# ==============================================================================


# ------------------------------------------------------------------------------
# Hold fusion penalty fixed.
#
# Increase lambda_1:
#
#       -> more coefficients pushed toward zero.
# ------------------------------------------------------------------------------

lambda_1_values <- c(
  0,
  0.001,
  0.003,
  0.01,
  0.03
)


lambda_1_fits <- vector(
  "list",
  length(
    lambda_1_values
  )
)


for (
  index in seq_along(
    lambda_1_values
  )
) {
  
  lambda_1_fits[[index]] <- fused_lasso_admm(
    X_train,
    y_train_centered,
    lambda_1 =
      lambda_1_values[
        index
      ],
    lambda_2 = 0.02,
    rho = 1,
    maximum_iterations = 4000
  )
}


matplot(
  time_grid,
  sapply(
    lambda_1_fits,
    function(fit) {
      
      fit$beta
    }
  ),
  type = "l",
  lty = seq_along(
    lambda_1_values
  ),
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = expression(
    "Effect of " * lambda[1]
  )
)


legend(
  "topright",
  legend = paste0(
    "lambda1 = ",
    lambda_1_values
  ),
  lty = seq_along(
    lambda_1_values
  ),
  lwd = 2,
  cex = 0.8
)


# ------------------------------------------------------------------------------
# Sparsity Path
# ------------------------------------------------------------------------------

lambda_1_nonzero <- vapply(
  lambda_1_fits,
  function(fit) {
    
    number_nonzero(
      fit$beta
    )
  },
  numeric(1)
)


plot(
  lambda_1_values,
  lambda_1_nonzero,
  type = "b",
  pch = 19,
  xlab = expression(
    lambda[1]
  ),
  ylab = "Number of Nonzero Coefficients",
  main = "Sparsity from L1 Penalty"
)


# ==============================================================================
# PART XXVI
#
# EFFECT OF LAMBDA_2
# ==============================================================================


# ------------------------------------------------------------------------------
# Hold sparsity penalty fixed.
#
# Increase lambda_2:
#
#       -> adjacent coefficients increasingly fused together.
# ------------------------------------------------------------------------------

lambda_2_values <- c(
  0,
  0.002,
  0.01,
  0.03,
  0.10
)


lambda_2_fits <- vector(
  "list",
  length(
    lambda_2_values
  )
)


for (
  index in seq_along(
    lambda_2_values
  )
) {
  
  lambda_2_fits[[index]] <- fused_lasso_admm(
    X_train,
    y_train_centered,
    lambda_1 = 0.002,
    lambda_2 =
      lambda_2_values[
        index
      ],
    rho = 1,
    maximum_iterations = 4000
  )
}


matplot(
  time_grid,
  sapply(
    lambda_2_fits,
    function(fit) {
      
      fit$beta
    }
  ),
  type = "l",
  lty = seq_along(
    lambda_2_values
  ),
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = expression(
    "Effect of " * lambda[2]
  )
)


legend(
  "topright",
  legend = paste0(
    "lambda2 = ",
    lambda_2_values
  ),
  lty = seq_along(
    lambda_2_values
  ),
  lwd = 2,
  cex = 0.8
)


# ------------------------------------------------------------------------------
# Fusion Path
# ------------------------------------------------------------------------------

lambda_2_jumps <- vapply(
  lambda_2_fits,
  function(fit) {
    
    number_jumps(
      fit$beta
    )
  },
  numeric(1)
)


plot(
  lambda_2_values,
  lambda_2_jumps,
  type = "b",
  pch = 19,
  xlab = expression(
    lambda[2]
  ),
  ylab = "Number of Adjacent Jumps",
  main = "Fusion from Total Variation Penalty"
)


# ==============================================================================
# PART XXVII
#
# TWO-DIMENSIONAL REGULARIZATION GRID
# ==============================================================================


# ------------------------------------------------------------------------------
# Fused Lasso has TWO tuning parameters.
#
# We therefore tune over combinations:
#
#       (lambda_1, lambda_2)
# ------------------------------------------------------------------------------

lambda_1_grid <- c(
  0,
  0.001,
  0.003,
  0.01
)


lambda_2_grid <- c(
  0,
  0.003,
  0.01,
  0.03
)


# ==============================================================================
# PART XXVIII
#
# MANUAL K-FOLD CROSS-VALIDATION
# ==============================================================================


make_folds <- function(
    n,
    number_folds = 5,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  sample(
    rep(
      seq_len(
        number_folds
      ),
      length.out = n
    )
  )
}


folds <- make_folds(
  n_train,
  number_folds = 5,
  seed = 999
)


cv_results <- expand.grid(
  lambda_1 =
    lambda_1_grid,
  lambda_2 =
    lambda_2_grid
)


cv_results$CV_MSE <- NA_real_

cv_results$CV_SE <- NA_real_


# ==============================================================================
# PART XXIX
#
# CROSS-VALIDATE
# ==============================================================================


for (
  combination_index in seq_len(
    nrow(
      cv_results
    )
  )
) {
  
  lambda_1 <- cv_results$lambda_1[
    combination_index
  ]
  
  
  lambda_2 <- cv_results$lambda_2[
    combination_index
  ]
  
  
  fold_mse <- numeric(
    5
  )
  
  
  for (
    fold in seq_len(5)
  ) {
    
    fold_validation_indices <- which(
      folds ==
        fold
    )
    
    
    fold_training_indices <- which(
      folds !=
        fold
    )
    
    
    X_fold_train_raw <- X_train_raw[
      fold_training_indices,
      ,
      drop = FALSE
    ]
    
    
    X_fold_validation_raw <- X_train_raw[
      fold_validation_indices,
      ,
      drop = FALSE
    ]
    
    
    y_fold_train <- y_train[
      fold_training_indices
    ]
    
    
    y_fold_validation <- y_train[
      fold_validation_indices
    ]
    
    
    # ------------------------------------------------------------------------
    # Fold-specific centering.
    # ------------------------------------------------------------------------
    
    fold_x_means <- colMeans(
      X_fold_train_raw
    )
    
    
    fold_y_mean <- mean(
      y_fold_train
    )
    
    
    X_fold_train <- sweep(
      X_fold_train_raw,
      2,
      fold_x_means,
      "-"
    )
    
    
    X_fold_validation <- sweep(
      X_fold_validation_raw,
      2,
      fold_x_means,
      "-"
    )
    
    
    y_fold_train_centered <- y_fold_train -
      fold_y_mean
    
    
    fit <- fused_lasso_admm(
      X_fold_train,
      y_fold_train_centered,
      lambda_1 =
        lambda_1,
      lambda_2 =
        lambda_2,
      rho = 1,
      maximum_iterations = 3000
    )
    
    
    prediction <- fold_y_mean +
      as.numeric(
        X_fold_validation %*%
          fit$beta
      )
    
    
    fold_mse[
      fold
    ] <- mean(
      (
        y_fold_validation -
          prediction
      )^2
    )
  }
  
  
  cv_results$CV_MSE[
    combination_index
  ] <- mean(
    fold_mse
  )
  
  
  cv_results$CV_SE[
    combination_index
  ] <- sd(
    fold_mse
  ) /
    sqrt(
      length(
        fold_mse
      )
    )
  
  
  cat(
    "lambda1 =",
    lambda_1,
    "lambda2 =",
    lambda_2,
    "CV MSE =",
    round(
      mean(
        fold_mse
      ),
      5
    ),
    "\n"
  )
}


cv_results


# ==============================================================================
# PART XXX
#
# SELECT BEST TUNING PARAMETERS
# ==============================================================================


best_index <- which.min(
  cv_results$CV_MSE
)


best_lambda_1 <- cv_results$lambda_1[
  best_index
]


best_lambda_2 <- cv_results$lambda_2[
  best_index
]


best_lambda_1


best_lambda_2


# ==============================================================================
# PART XXXI
#
# CV SURFACE
# ==============================================================================


cv_matrix <- matrix(
  NA_real_,
  nrow =
    length(
      lambda_1_grid
    ),
  ncol =
    length(
      lambda_2_grid
    )
)


for (
  i in seq_along(
    lambda_1_grid
  )
) {
  
  for (
    j in seq_along(
      lambda_2_grid
    )
  ) {
    
    index <- which(
      cv_results$lambda_1 ==
        lambda_1_grid[i] &
        cv_results$lambda_2 ==
        lambda_2_grid[j]
    )
    
    
    cv_matrix[
      i,
      j
    ] <- cv_results$CV_MSE[
      index
    ]
  }
}


image(
  x =
    lambda_1_grid,
  y =
    lambda_2_grid,
  z =
    cv_matrix,
  xlab = expression(
    lambda[1]
  ),
  ylab = expression(
    lambda[2]
  ),
  main = "Cross-Validated MSE"
)


# ==============================================================================
# PART XXXII
#
# FINAL MODEL
# ==============================================================================


final_fit <- fused_lasso_admm(
  X_train,
  y_train_centered,
  lambda_1 =
    best_lambda_1,
  lambda_2 =
    best_lambda_2,
  rho = 1,
  maximum_iterations = 5000,
  verbose = FALSE
)


final_fit$converged


# ==============================================================================
# PART XXXIII
#
# FINAL TEST PERFORMANCE
# ==============================================================================


final_prediction <- predict_functional_model(
  X_test_raw,
  x_means,
  y_mean,
  final_fit$beta
)


final_metrics <- regression_metrics(
  y_test,
  final_prediction
)


final_metrics


# ==============================================================================
# PART XXXIV
#
# FINAL COEFFICIENT RECOVERY
# ==============================================================================


plot(
  time_grid,
  beta_true,
  type = "s",
  lwd = 3,
  xlab = "t",
  ylab = expression(
    beta(t)
  ),
  main = "Final Fused-Lasso Estimate"
)


lines(
  time_grid,
  final_fit$beta,
  type = "s",
  lwd = 2,
  lty = 2
)


abline(
  h = 0,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "True",
    "Estimated"
  ),
  lty = c(
    1,
    2
  ),
  lwd = c(
    3,
    2
  )
)


# ==============================================================================
# PART XXXV
#
# FINAL STRUCTURAL DIAGNOSTICS
# ==============================================================================


final_structure <- data.frame(
  Metric = c(
    "Coefficient MSE",
    "Coefficient Correlation",
    "Nonzero Coefficients",
    "Adjacent Jumps",
    "Total Variation"
  ),
  Value = c(
    coefficient_mse(
      final_fit$beta,
      beta_true
    ),
    coefficient_correlation(
      final_fit$beta,
      beta_true
    ),
    number_nonzero(
      final_fit$beta
    ),
    number_jumps(
      final_fit$beta
    ),
    total_variation(
      final_fit$beta
    )
  )
)


final_structure


# ==============================================================================
# PART XXXVI
#
# COMPARE MODELS
# ==============================================================================


fusion_only_metrics <- regression_metrics(
  y_test,
  fusion_prediction
)


lasso_metrics <- regression_metrics(
  y_test,
  lasso_prediction
)


ols_metrics <- regression_metrics(
  y_test,
  ols_prediction
)


comparison <- data.frame(
  Model = c(
    "OLS",
    "Lasso",
    "Fusion Only",
    "Fused Lasso"
  ),
  MSE = c(
    ols_metrics["MSE"],
    lasso_metrics["MSE"],
    fusion_only_metrics["MSE"],
    final_metrics["MSE"]
  ),
  RMSE = c(
    ols_metrics["RMSE"],
    lasso_metrics["RMSE"],
    fusion_only_metrics["RMSE"],
    final_metrics["RMSE"]
  ),
  R2 = c(
    ols_metrics["R2"],
    lasso_metrics["R2"],
    fusion_only_metrics["R2"],
    final_metrics["R2"]
  )
)


comparison


# ==============================================================================
# PART XXXVII
#
# IDENTIFY ESTIMATED CONSTANT SEGMENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# Because numerical ADMM estimates will not be exactly identical at every
# location, define a tolerance for deciding whether adjacent coefficients
# belong to the same fused segment.
# ------------------------------------------------------------------------------

identify_segments <- function(
    beta,
    tolerance = 0.01
) {
  
  p <- length(
    beta
  )
  
  
  segment_id <- integer(
    p
  )
  
  
  current_segment <- 1
  
  
  segment_id[1] <- current_segment
  
  
  for (
    j in 2:p
  ) {
    
    if (
      abs(
        beta[j] -
        beta[j - 1]
      ) >
      tolerance
    ) {
      
      current_segment <- current_segment +
        1
    }
    
    
    segment_id[j] <- current_segment
  }
  
  
  segment_id
}


segment_id <- identify_segments(
  final_fit$beta,
  tolerance = 0.02
)


number_segments <- max(
  segment_id
)


number_segments


# ==============================================================================
# PART XXXVIII
#
# SEGMENT TABLE
# ==============================================================================


segment_table <- data.frame(
  Segment = seq_len(
    number_segments
  ),
  Start_t = NA_real_,
  End_t = NA_real_,
  Mean_Beta = NA_real_,
  Number_Grid_Points = NA_integer_
)


for (
  segment in seq_len(
    number_segments
  )
) {
  
  indices <- which(
    segment_id ==
      segment
  )
  
  
  segment_table$Start_t[
    segment
  ] <- min(
    time_grid[
      indices
    ]
  )
  
  
  segment_table$End_t[
    segment
  ] <- max(
    time_grid[
      indices
    ]
  )
  
  
  segment_table$Mean_Beta[
    segment
  ] <- mean(
    final_fit$beta[
      indices
    ]
  )
  
  
  segment_table$Number_Grid_Points[
    segment
  ] <- length(
    indices
  )
}


segment_table


# ==============================================================================
# PART XXXIX
#
# SELECTED FUNCTIONAL REGIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# Which parts of the functional domain have substantial estimated effects?
# ------------------------------------------------------------------------------

effect_tolerance <- 0.10


active_locations <- abs(
  final_fit$beta
) >
  effect_tolerance


plot(
  time_grid,
  final_fit$beta,
  type = "s",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = "Estimated Active Functional Regions"
)


abline(
  h = c(
    -effect_tolerance,
    effect_tolerance
  ),
  lty = 3
)


points(
  time_grid[
    active_locations
  ],
  final_fit$beta[
    active_locations
  ],
  pch = 19
)


# ==============================================================================
# PART XL
#
# FIRST-DIFFERENCE SPARSITY
# ==============================================================================


# ------------------------------------------------------------------------------
# Fused Lasso does not merely make beta sparse.
#
# It also makes:
#
#       D beta
#
# sparse.
#
# Sparse D beta means there are only a few change points.
# ------------------------------------------------------------------------------

estimated_differences <- as.numeric(
  D %*%
    final_fit$beta
)


par(
  mfrow = c(2, 1)
)


plot(
  time_grid,
  final_fit$beta,
  type = "s",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = "Estimated Coefficient"
)


plot(
  time_grid[
    -1
  ],
  estimated_differences,
  type = "h",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    Delta * hat(beta)
  ),
  main = "Estimated First Differences"
)


abline(
  h = 0,
  lty = 3
)


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XLI
#
# CHANGE-POINT LOCATIONS
# ==============================================================================


change_tolerance <- 0.02


change_points <- which(
  abs(
    estimated_differences
  ) >
    change_tolerance
) +
  1


time_grid[
  change_points
]


# ==============================================================================
# PART XLII
#
# WHY ORDINARY LASSO DOES NOT FUSE
# ==============================================================================


# ------------------------------------------------------------------------------
# Ordinary Lasso penalizes:
#
#       |beta_1| + ... + |beta_p|
#
#
# It knows nothing about ordering.
#
# Swapping columns 5 and 70 leaves the Lasso penalty unchanged.
#
#
# Fused Lasso also penalizes:
#
#       |beta_2 - beta_1|
#
#       +
#
#       ...
#
#       +
#
#       |beta_p - beta_{p-1}|
#
#
# Therefore predictor ordering is fundamental.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XLIII
#
# PERMUTATION DEMONSTRATION
# ==============================================================================


set.seed(2025)


permutation <- sample(
  seq_len(p)
)


X_train_permuted <- X_train[
  ,
  permutation,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# Fit with same fused penalty after destroying functional ordering.
# ------------------------------------------------------------------------------

permuted_fit <- fused_lasso_admm(
  X_train_permuted,
  y_train_centered,
  lambda_1 =
    best_lambda_1,
  lambda_2 =
    best_lambda_2,
  rho = 1,
  maximum_iterations = 5000
)


# ------------------------------------------------------------------------------
# Undo permutation to compare coefficients in original time order.
# ------------------------------------------------------------------------------

permuted_beta_original_order <- numeric(
  p
)


permuted_beta_original_order[
  permutation
] <- permuted_fit$beta


plot(
  time_grid,
  beta_true,
  type = "s",
  lwd = 3,
  xlab = "t",
  ylab = expression(
    beta(t)
  ),
  main = "Why Functional Ordering Matters"
)


lines(
  time_grid,
  final_fit$beta,
  type = "s",
  lwd = 2,
  lty = 2
)


lines(
  time_grid,
  permuted_beta_original_order,
  type = "s",
  lwd = 2,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "True",
    "Correct Ordering",
    "Permuted Ordering"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = c(
    3,
    2,
    2
  )
)


# ==============================================================================
# PART XLIV
#
# HIGHLY CORRELATED FUNCTIONAL PREDICTORS
# ==============================================================================


# ------------------------------------------------------------------------------
# Adjacent functional measurements are strongly correlated.
# ------------------------------------------------------------------------------

adjacent_correlations <- numeric(
  p - 1
)


for (
  j in seq_len(
    p - 1
  )
) {
  
  adjacent_correlations[j] <- cor(
    X_train_raw[
      ,
      j
    ],
    X_train_raw[
      ,
      j + 1
    ]
  )
}


summary(
  adjacent_correlations
)


plot(
  time_grid[
    -1
  ],
  adjacent_correlations,
  type = "l",
  lwd = 2,
  ylim = c(
    -1,
    1
  ),
  xlab = "t",
  ylab = "Correlation",
  main = "Correlation Between Adjacent Functional Measurements"
)


abline(
  h = 0,
  lty = 3
)


# ==============================================================================
# PART XLV
#
# SECOND-DIFFERENCE PENALTY
# ==============================================================================


# ------------------------------------------------------------------------------
# Fused Lasso penalizes FIRST differences:
#
#       beta_j - beta_{j-1}
#
# giving piecewise-constant coefficients.
#
#
# If instead we penalize SECOND differences:
#
#       beta_{j+1} - 2 beta_j + beta_{j-1}
#
# we encourage piecewise-LINEAR coefficients.
#
#
# This leads toward trend filtering.
# ------------------------------------------------------------------------------

second_difference_matrix <- function(p) {
  
  D2 <- matrix(
    0,
    nrow = p - 2,
    ncol = p
  )
  
  
  for (
    j in seq_len(
      p - 2
    )
  ) {
    
    D2[
      j,
      j
    ] <- 1
    
    
    D2[
      j,
      j + 1
    ] <- -2
    
    
    D2[
      j,
      j + 2
    ] <- 1
  }
  
  
  D2
}


D2 <- second_difference_matrix(
  p
)


dim(
  D2
)


# ==============================================================================
# PART XLVI
#
# GENERALIZED DIFFERENCE-PENALTY ADMM
# ==============================================================================


# ------------------------------------------------------------------------------
# General objective:
#
#       (1/(2n)) ||y - X beta||^2
#
#       +
#
#       lambda_1 ||beta||_1
#
#       +
#
#       lambda_D ||D beta||_1
#
#
# This lets us replace the first-difference matrix with another structural
# operator.
# ------------------------------------------------------------------------------

generalized_fused_admm <- function(
    X,
    y,
    difference_matrix,
    lambda_1,
    lambda_D,
    rho = 1,
    maximum_iterations = 5000,
    tolerance = 1e-5
) {
  
  X <- as.matrix(
    X
  )
  
  
  y <- as.numeric(
    y
  )
  
  
  D <- as.matrix(
    difference_matrix
  )
  
  
  n <- nrow(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  q <- nrow(
    D
  )
  
  
  beta <- rep(
    0,
    p
  )
  
  
  z <- rep(
    0,
    p
  )
  
  
  u <- rep(
    0,
    q
  )
  
  
  dual_z <- rep(
    0,
    p
  )
  
  
  dual_u <- rep(
    0,
    q
  )
  
  
  system_matrix <- crossprod(
    X
  ) /
    n +
    rho *
    diag(p) +
    rho *
    crossprod(
      D
    )
  
  
  chol_system <- chol(
    system_matrix
  )
  
  
  Xty <- as.numeric(
    crossprod(
      X,
      y
    )
  ) /
    n
  
  
  solve_system <- function(rhs) {
    
    backsolve(
      chol_system,
      forwardsolve(
        t(
          chol_system
        ),
        rhs
      )
    )
  }
  
  
  for (
    iteration in seq_len(
      maximum_iterations
    )
  ) {
    
    beta_old <- beta
    
    
    rhs <- Xty +
      rho *
      (
        z -
          dual_z
      ) +
      rho *
      as.numeric(
        crossprod(
          D,
          u -
            dual_u
        )
      )
    
    
    beta <- as.numeric(
      solve_system(
        rhs
      )
    )
    
    
    D_beta <- as.numeric(
      D %*%
        beta
    )
    
    
    z <- soft_threshold(
      beta +
        dual_z,
      lambda_1 /
        rho
    )
    
    
    u <- soft_threshold(
      D_beta +
        dual_u,
      lambda_D /
        rho
    )
    
    
    dual_z <- dual_z +
      beta -
      z
    
    
    dual_u <- dual_u +
      D_beta -
      u
    
    
    if (
      max(
        abs(
          beta -
          beta_old
        )
      ) <
      tolerance
    ) {
      
      break
    }
  }
  
  
  list(
    beta =
      beta,
    iterations =
      iteration
  )
}


# ==============================================================================
# PART XLVII
#
# FIRST VS SECOND DIFFERENCE PENALTY
# ==============================================================================


first_difference_fit <- generalized_fused_admm(
  X_train,
  y_train_centered,
  difference_matrix =
    D,
  lambda_1 = 0.002,
  lambda_D = 0.02
)


second_difference_fit <- generalized_fused_admm(
  X_train,
  y_train_centered,
  difference_matrix =
    D2,
  lambda_1 = 0.002,
  lambda_D = 0.02
)


plot(
  time_grid,
  beta_true,
  type = "s",
  lwd = 3,
  xlab = "t",
  ylab = expression(
    beta(t)
  ),
  main = "First vs Second Difference Penalties"
)


lines(
  time_grid,
  first_difference_fit$beta,
  lwd = 2,
  lty = 2
)


lines(
  time_grid,
  second_difference_fit$beta,
  lwd = 2,
  lty = 3
)


legend(
  "topright",
  legend = c(
    "True",
    "First Difference",
    "Second Difference"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = c(
    3,
    2,
    2
  )
)


# ==============================================================================
# PART XLVIII
#
# FUSED LASSO AS CHANGE-POINT DETECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# If:
#
#       beta_j = beta_{j-1}
#
# then:
#
#       (D beta)_j = 0.
#
#
# Thus nonzero entries of D beta identify estimated boundaries between
# coefficient regions.
# ------------------------------------------------------------------------------

estimated_change_indicator <- abs(
  diff(
    final_fit$beta
  )
) >
  0.02


plot(
  time_grid,
  final_fit$beta,
  type = "s",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = "Estimated Change Points"
)


estimated_change_locations <- time_grid[
  which(
    estimated_change_indicator
  ) +
    1
]


if (
  length(
    estimated_change_locations
  ) >
  0
) {
  
  abline(
    v =
      estimated_change_locations,
    lty = 3
  )
}


# ==============================================================================
# PART XLIX
#
# TRUE CHANGE POINTS
# ==============================================================================


true_change_points <- which(
  abs(
    diff(
      beta_true
    )
  ) >
    0
) +
  1


time_grid[
  true_change_points
]


# ==============================================================================
# PART L
#
# WHY FUSED LASSO IS USEFUL FOR FUNCTIONAL DATA
# ==============================================================================


# Functional measurements often satisfy:
#
#       X(t_j) approximately X(t_{j+1})
#
# in the sense that neighboring locations are naturally ordered and highly
# related.
#
#
# Ordinary Lasso treats:
#
#       beta_1, ..., beta_p
#
# as unrelated coefficients.
#
#
# Fused Lasso explicitly uses the geometry:
#
#       t_1 < t_2 < ... < t_p.
#
#
# It therefore estimates REGIONS rather than isolated variables.


# ==============================================================================
# PART LI
#
# EXAMPLE: FUNCTIONAL VARIABLE SELECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# Instead of saying:
#
#       "grid point 17 is selected"
#
# functional interpretation is often:
#
#       "the interval t in [0.15, 0.30] is associated with the response."
#
#
# This is often much more scientifically meaningful.
# ------------------------------------------------------------------------------

active <- abs(
  final_fit$beta
) >
  0.10


# ------------------------------------------------------------------------------
# Convert contiguous TRUE values into intervals.
# ------------------------------------------------------------------------------

extract_active_intervals <- function(
    active,
    grid
) {
  
  if (
    !any(
      active
    )
  ) {
    
    return(
      data.frame(
        Start = numeric(0),
        End = numeric(0)
      )
    )
  }
  
  
  runs <- rle(
    active
  )
  
  
  end_indices <- cumsum(
    runs$lengths
  )
  
  
  start_indices <- c(
    1,
    head(
      end_indices,
      -1
    ) +
      1
  )
  
  
  selected_runs <- which(
    runs$values
  )
  
  
  data.frame(
    Start =
      grid[
        start_indices[
          selected_runs
        ]
      ],
    End =
      grid[
        end_indices[
          selected_runs
        ]
      ]
  )
}


active_intervals <- extract_active_intervals(
  active,
  time_grid
)


active_intervals


# ==============================================================================
# PART LII
#
# FUNCTIONAL CONTRIBUTIONS FOR INDIVIDUAL OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# The predicted functional contribution is:
#
#       integral X_i(t) beta(t) dt
#
#
# We can inspect pointwise contributions:
#
#       X_i(t_j) beta_j delta_t.
# ------------------------------------------------------------------------------

example_index <- 1


example_function <- X[
  test_indices[
    example_index
  ],
]


pointwise_contribution <- example_function *
  final_fit$beta *
  delta_t


par(
  mfrow = c(3, 1)
)


plot(
  time_grid,
  example_function,
  type = "l",
  lwd = 2,
  xlab = "t",
  ylab = "X(t)",
  main = "Example Functional Predictor"
)


plot(
  time_grid,
  final_fit$beta,
  type = "s",
  lwd = 2,
  xlab = "t",
  ylab = expression(
    hat(beta)(t)
  ),
  main = "Estimated Coefficient"
)


plot(
  time_grid,
  pointwise_contribution,
  type = "h",
  lwd = 2,
  xlab = "t",
  ylab = "Contribution",
  main = "Pointwise Contribution to Prediction"
)


abline(
  h = 0,
  lty = 3
)


par(
  mfrow = c(1, 1)
)


sum(
  pointwise_contribution
)


# ==============================================================================
# PART LIII
#
# PREDICTED VS OBSERVED
# ==============================================================================


plot(
  y_test,
  final_prediction,
  pch = 19,
  xlab = "Observed Y",
  ylab = "Predicted Y",
  main = "Fused-Lasso Predictions"
)


abline(
  0,
  1,
  lwd = 2,
  lty = 2
)


# ==============================================================================
# PART LIV
#
# RESIDUAL DIAGNOSTICS
# ==============================================================================


final_residuals <- y_test -
  final_prediction


par(
  mfrow = c(1, 3)
)


plot(
  final_prediction,
  final_residuals,
  pch = 19,
  xlab = "Predicted",
  ylab = "Residual",
  main = "Residuals vs Predicted"
)


abline(
  h = 0,
  lty = 2
)


hist(
  final_residuals,
  breaks = 20,
  xlab = "Residual",
  main = "Residual Distribution"
)


qqnorm(
  final_residuals,
  main = "Residual Q-Q Plot"
)


qqline(
  final_residuals
)


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART LV
#
# OPTIONAL PACKAGE VERIFICATION
# ==============================================================================


# Production fused-lasso / generalized-lasso implementations are available in
# packages such as:
#
#       genlasso
#
# and related optimization packages.
#
#
# A generalized-lasso formulation is:
#
#       minimize
#
#           1/2 ||y - X beta||^2
#
#           +
#
#           lambda ||D beta||_1
#
#
# Our implementation additionally includes the separate:
#
#       lambda_1 ||beta||_1
#
# term.
#
#
# Exact tuning-parameter conventions can differ across packages, especially
# because this script uses:
#
#       (1/(2n)) ||y - X beta||^2.
#
#
# Therefore package lambda values should not automatically be expected to match
# this script numerically.


# ==============================================================================
# PART LVI
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nFused Lasso for Functional Data Summary\n"
)


cat(
  "---------------------------------------\n"
)


cat(
  "Training observations:",
  n_train,
  "\n"
)


cat(
  "Functional grid points:",
  p,
  "\n"
)


cat(
  "Best lambda_1:",
  best_lambda_1,
  "\n"
)


cat(
  "Best lambda_2:",
  best_lambda_2,
  "\n"
)


cat(
  "Final ADMM converged:",
  final_fit$converged,
  "\n"
)


cat(
  "Final ADMM iterations:",
  final_fit$iterations,
  "\n"
)


cat(
  "Test MSE:",
  round(
    final_metrics["MSE"],
    5
  ),
  "\n"
)


cat(
  "Test R2:",
  round(
    final_metrics["R2"],
    5
  ),
  "\n"
)


cat(
  "Coefficient MSE:",
  round(
    coefficient_mse(
      final_fit$beta,
      beta_true
    ),
    5
  ),
  "\n"
)


cat(
  "Coefficient correlation:",
  round(
    coefficient_correlation(
      final_fit$beta,
      beta_true
    ),
    5
  ),
  "\n"
)


cat(
  "Estimated nonzero coefficients:",
  number_nonzero(
    final_fit$beta
  ),
  "\n"
)


cat(
  "Estimated adjacent jumps:",
  number_jumps(
    final_fit$beta
  ),
  "\n"
)


cat(
  "Estimated active intervals:\n"
)


print(
  active_intervals
)