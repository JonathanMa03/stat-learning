# ==============================================================================
# Hierarchical Mixtures of Experts
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Mixture of Experts
#   - Hierarchical Mixture of Experts
#   - Soft gating networks
#   - Logistic gates
#   - Regression experts
#   - Latent expert assignments
#   - EM algorithm
#   - Weighted least squares
#   - Soft partitioning of predictor space
#   - Hierarchical conditional probabilities
#
#
# Two-level HME:
#
#                       Root Gate
#                      /         \
#                   Gate 1       Gate 2
#                  /     \       /     \
#                E11     E12   E21     E22
#
#
# Final predictive density:
#
# p(y | x)
# =
# g_1(x)
# [
#   g_11(x) p(y | x, E11)
#   +
#   g_12(x) p(y | x, E12)
# ]
# +
# g_2(x)
# [
#   g_21(x) p(y | x, E21)
#   +
#   g_22(x) p(y | x, E22)
# ]
#
#
# For regression, each expert is:
#
# y | x, expert k
# ~
# Normal(mu_k(x), sigma_k^2)
#
# with:
#
# mu_k(x) = beta_k0 + beta_k1 x1 + ... + beta_kp xp
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE DATA WITH REGIME-SPECIFIC RELATIONSHIPS
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Predictors
# ------------------------------------------------------------------------------

set.seed(123)

n <- 600


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


X <- cbind(
  x1 = x1,
  x2 = x2
)


# ------------------------------------------------------------------------------
# 2. Construct Four Smooth Regimes
# ------------------------------------------------------------------------------

# The true regression structure changes depending on x1 and x2.
#
# We deliberately create four different local relationships.


mu_11 <- function(x1, x2) {
  
  -3 +
    1.5 * x1 +
    0.5 * x2
}


mu_12 <- function(x1, x2) {
  
  2 +
    0.5 * x1 -
    1.5 * x2
}


mu_21 <- function(x1, x2) {
  
  3 -
    1.5 * x1 +
    0.75 * x2
}


mu_22 <- function(x1, x2) {
  
  -1 -
    0.5 * x1 -
    1.25 * x2
}


# ------------------------------------------------------------------------------
# 3. True Soft Gates
# ------------------------------------------------------------------------------

stable_sigmoid <- function(z) {
  
  output <- numeric(
    length(z)
  )
  
  
  positive <- z >= 0
  
  
  output[positive] <-
    1 /
    (
      1 +
        exp(
          -z[positive]
        )
    )
  
  
  exp_z <- exp(
    z[
      !positive
    ]
  )
  
  
  output[
    !positive
  ] <-
    exp_z /
    (
      1 +
        exp_z
    )
  
  
  return(output)
}


# Root gate:
#
# approximately separates by x1.


root_probability <- stable_sigmoid(
  2 * x1
)


# Lower gate on left branch.

left_probability <- stable_sigmoid(
  2 * x2
)


# Lower gate on right branch.

right_probability <- stable_sigmoid(
  -2 * x2
)


# ------------------------------------------------------------------------------
# 4. Four Leaf Probabilities
# ------------------------------------------------------------------------------

pi_11 <-
  root_probability *
  left_probability


pi_12 <-
  root_probability *
  (
    1 -
      left_probability
  )


pi_21 <-
  (
    1 -
      root_probability
  ) *
  right_probability


pi_22 <-
  (
    1 -
      root_probability
  ) *
  (
    1 -
      right_probability
  )


leaf_probabilities <- cbind(
  E11 = pi_11,
  E12 = pi_12,
  E21 = pi_21,
  E22 = pi_22
)


# Check probabilities sum to one.

range(
  rowSums(
    leaf_probabilities
  )
)


# ------------------------------------------------------------------------------
# 5. Sample Latent Experts
# ------------------------------------------------------------------------------

sample_expert <- function(probability_row) {
  
  sample(
    1:4,
    size = 1,
    prob = probability_row
  )
}


true_expert <- apply(
  leaf_probabilities,
  1,
  sample_expert
)


table(
  true_expert
)


# ------------------------------------------------------------------------------
# 6. Generate Response
# ------------------------------------------------------------------------------

expert_means <- cbind(
  mu_11(
    x1,
    x2
  ),
  
  mu_12(
    x1,
    x2
  ),
  
  mu_21(
    x1,
    x2
  ),
  
  mu_22(
    x1,
    x2
  )
)


mu <- expert_means[
  cbind(
    seq_len(n),
    true_expert
  )
]


sigma_true <- c(
  0.8,
  0.8,
  1,
  1
)


y <- mu +
  rnorm(
    n,
    mean = 0,
    sd = sigma_true[
      true_expert
    ]
  )


data <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  true_expert = true_expert
)


# ------------------------------------------------------------------------------
# 7. Explore Data
# ------------------------------------------------------------------------------

plot(
  x1,
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x1",
  ylab = "y",
  main = "Hierarchical Mixture of Experts Data"
)


plot(
  x2,
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x2",
  ylab = "y",
  main = "Response vs x2"
)


# ==============================================================================
# PART II
#
# LOGISTIC GATING NETWORKS
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Add Intercept
# ------------------------------------------------------------------------------

add_intercept <- function(X) {
  
  cbind(
    Intercept = 1,
    X
  )
}


G <- add_intercept(
  X
)


# ------------------------------------------------------------------------------
# 9. Logistic Probability
# ------------------------------------------------------------------------------

logistic_probability <- function(
    G,
    gamma
) {
  
  stable_sigmoid(
    as.vector(
      G %*%
        gamma
    )
  )
}


# ------------------------------------------------------------------------------
# 10. Weighted Logistic Regression
# ------------------------------------------------------------------------------

# The gate M-step needs to fit logistic models to SOFT targets.
#
# If r_i is the expected class-1 membership, maximize:
#
# sum_i [
#   r_i log(p_i)
#   +
#   (1-r_i) log(1-p_i)
# ]
#
# This is logistic regression with fractional outcomes.


weighted_logistic_soft <- function(
    G,
    target,
    observation_weights = NULL,
    initial_gamma = NULL,
    max_iter = 100,
    tol = 1e-7,
    ridge = 1e-8
) {
  
  G <- as.matrix(
    G
  )
  
  
  n <- nrow(
    G
  )
  
  
  p <- ncol(
    G
  )
  
  
  if (
    is.null(
      observation_weights
    )
  ) {
    
    observation_weights <- rep(
      1,
      n
    )
  }
  
  
  if (
    is.null(
      initial_gamma
    )
  ) {
    
    gamma <- rep(
      0,
      p
    )
    
  } else {
    
    gamma <- initial_gamma
  }
  
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    gamma_old <- gamma
    
    
    eta <- as.vector(
      G %*%
        gamma
    )
    
    
    probability <- stable_sigmoid(
      eta
    )
    
    
    probability <- pmin(
      pmax(
        probability,
        1e-8
      ),
      1 - 1e-8
    )
    
    
    variance <- probability *
      (
        1 -
          probability
      )
    
    
    W <- observation_weights *
      variance
    
    
    z <- eta +
      (
        target -
          probability
      ) /
      variance
    
    
    sqrt_W <- sqrt(
      W
    )
    
    
    G_weighted <- G *
      sqrt_W
    
    
    z_weighted <- z *
      sqrt_W
    
    
    penalty <- ridge *
      diag(
        p
      )
    
    
    penalty[
      1,
      1
    ] <- 0
    
    
    gamma <- tryCatch(
      
      solve(
        crossprod(
          G_weighted
        ) +
          penalty,
        
        crossprod(
          G_weighted,
          z_weighted
        )
      ),
      
      error = function(e) {
        
        qr.solve(
          crossprod(
            G_weighted
          ) +
            penalty,
          
          crossprod(
            G_weighted,
            z_weighted
          )
        )
      }
    )
    
    
    gamma <- as.vector(
      gamma
    )
    
    
    if (
      max(
        abs(
          gamma -
          gamma_old
        )
      ) <
      tol
    ) {
      
      break
    }
  }
  
  
  return(
    list(
      coefficients = gamma,
      probability = logistic_probability(
        G,
        gamma
      ),
      iterations = iteration
    )
  )
}


# ==============================================================================
# PART III
#
# REGRESSION EXPERTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Weighted Linear Regression
# ------------------------------------------------------------------------------

weighted_linear_fit <- function(
    X,
    y,
    weights,
    ridge = 1e-8
) {
  
  X <- as.matrix(
    X
  )
  
  
  weights <- pmax(
    weights,
    1e-12
  )
  
  
  sqrt_weights <- sqrt(
    weights
  )
  
  
  X_weighted <- X *
    sqrt_weights
  
  
  y_weighted <- y *
    sqrt_weights
  
  
  p <- ncol(
    X
  )
  
  
  penalty <- ridge *
    diag(
      p
    )
  
  
  penalty[
    1,
    1
  ] <- 0
  
  
  beta <- solve(
    crossprod(
      X_weighted
    ) +
      penalty,
    
    crossprod(
      X_weighted,
      y_weighted
    )
  )
  
  
  beta <- as.vector(
    beta
  )
  
  
  fitted <- as.vector(
    X %*%
      beta
  )
  
  
  return(
    list(
      coefficients = beta,
      fitted = fitted
    )
  )
}


# ------------------------------------------------------------------------------
# 12. Weighted Gaussian Variance
# ------------------------------------------------------------------------------

weighted_variance <- function(
    y,
    mu,
    weights,
    minimum_variance = 1e-4
) {
  
  variance <- sum(
    weights *
      (
        y -
          mu
      )^2
  ) /
    sum(
      weights
    )
  
  
  max(
    variance,
    minimum_variance
  )
}


# ==============================================================================
# PART IV
#
# GAUSSIAN EXPERT DENSITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Gaussian Density Matrix
# ------------------------------------------------------------------------------

expert_density_matrix <- function(
    y,
    expert_means,
    expert_variances
) {
  
  K <- ncol(
    expert_means
  )
  
  
  densities <- matrix(
    NA_real_,
    nrow = length(y),
    ncol = K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    densities[
      ,
      k
    ] <- dnorm(
      y,
      mean = expert_means[
        ,
        k
      ],
      sd = sqrt(
        expert_variances[k]
      )
    )
  }
  
  
  return(
    densities
  )
}


# ==============================================================================
# PART V
#
# HIERARCHICAL GATE PROBABILITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Compute Leaf Probabilities
# ------------------------------------------------------------------------------

hme_gate_probabilities <- function(
    G,
    gamma_root,
    gamma_left,
    gamma_right
) {
  
  g_root <- logistic_probability(
    G,
    gamma_root
  )
  
  
  g_left <- logistic_probability(
    G,
    gamma_left
  )
  
  
  g_right <- logistic_probability(
    G,
    gamma_right
  )
  
  
  leaf_probabilities <- cbind(
    
    E11 =
      g_root *
      g_left,
    
    E12 =
      g_root *
      (
        1 -
          g_left
      ),
    
    E21 =
      (
        1 -
          g_root
      ) *
      g_right,
    
    E22 =
      (
        1 -
          g_root
      ) *
      (
        1 -
          g_right
      )
  )
  
  
  return(
    list(
      root = g_root,
      left = g_left,
      right = g_right,
      leaf = leaf_probabilities
    )
  )
}


# ==============================================================================
# PART VI
#
# INITIALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Initialize Responsibilities
# ------------------------------------------------------------------------------

# EM is sensitive to initialization.
#
# A simple deterministic initialization based on predictor quadrants gives
# the four experts different starting regions.


initialize_responsibilities <- function(
    X
) {
  
  x1 <- X[
    ,
    1
  ]
  
  
  x2 <- X[
    ,
    2
  ]
  
  
  groups <- ifelse(
    
    x1 >= median(
      x1
    ),
    
    ifelse(
      x2 >= median(
        x2
      ),
      1,
      2
    ),
    
    ifelse(
      x2 >= median(
        x2
      ),
      3,
      4
    )
  )
  
  
  responsibilities <- matrix(
    0.05,
    nrow = nrow(X),
    ncol = 4
  )
  
  
  for (
    i in seq_len(
      nrow(X)
    )
  ) {
    
    responsibilities[
      i,
      groups[i]
    ] <- 0.85
  }
  
  
  responsibilities <-
    responsibilities /
    rowSums(
      responsibilities
    )
  
  
  return(
    responsibilities
  )
}


# ==============================================================================
# PART VII
#
# EM ALGORITHM
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. HME EM Algorithm
# ------------------------------------------------------------------------------

fit_hme <- function(
    X,
    y,
    max_iter = 200,
    tol = 1e-6,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  G <- add_intercept(
    X
  )
  
  
  E <- add_intercept(
    X
  )
  
  
  K <- 4
  
  
  responsibilities <- initialize_responsibilities(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Initial expert fits
  # --------------------------------------------------------------------------
  
  expert_beta <- matrix(
    NA_real_,
    nrow = ncol(E),
    ncol = K
  )
  
  
  expert_variance <- numeric(
    K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    fit_k <- weighted_linear_fit(
      E,
      y,
      responsibilities[
        ,
        k
      ]
    )
    
    
    expert_beta[
      ,
      k
    ] <- fit_k$coefficients
    
    
    expert_variance[k] <- weighted_variance(
      y,
      fit_k$fitted,
      responsibilities[
        ,
        k
      ]
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Initial gate coefficients
  # --------------------------------------------------------------------------
  
  gamma_root <- rep(
    0,
    ncol(G)
  )
  
  
  gamma_left <- rep(
    0,
    ncol(G)
  )
  
  
  gamma_right <- rep(
    0,
    ncol(G)
  )
  
  
  log_likelihood_history <- numeric(
    max_iter
  )
  
  
  # ==========================================================================
  # EM LOOP
  # ==========================================================================
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    # ========================================================================
    # M-STEP: EXPERTS
    # ========================================================================
    
    for (
      k in seq_len(K)
    ) {
      
      expert_fit <- weighted_linear_fit(
        E,
        y,
        responsibilities[
          ,
          k
        ]
      )
      
      
      expert_beta[
        ,
        k
      ] <- expert_fit$coefficients
      
      
      expert_variance[k] <- weighted_variance(
        y = y,
        mu = expert_fit$fitted,
        weights = responsibilities[
          ,
          k
        ]
      )
    }
    
    
    # ========================================================================
    # M-STEP: GATES
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # Root gate
    #
    # Branch 1 contains E11 + E12.
    # Branch 2 contains E21 + E22.
    # ------------------------------------------------------------------------
    
    root_target <-
      responsibilities[
        ,
        1
      ] +
      responsibilities[
        ,
        2
      ]
    
    
    root_fit <- weighted_logistic_soft(
      G = G,
      target = root_target,
      initial_gamma = gamma_root
    )
    
    
    gamma_root <-
      root_fit$coefficients
    
    
    # ------------------------------------------------------------------------
    # Left child gate
    #
    # Conditional probability:
    #
    # P(E11 | branch 1)
    # =
    # r11 / (r11 + r12)
    # ------------------------------------------------------------------------
    
    left_weight <-
      responsibilities[
        ,
        1
      ] +
      responsibilities[
        ,
        2
      ]
    
    
    left_target <-
      responsibilities[
        ,
        1
      ] /
      pmax(
        left_weight,
        1e-12
      )
    
    
    left_fit <- weighted_logistic_soft(
      G = G,
      target = left_target,
      observation_weights = left_weight,
      initial_gamma = gamma_left
    )
    
    
    gamma_left <-
      left_fit$coefficients
    
    
    # ------------------------------------------------------------------------
    # Right child gate
    #
    # P(E21 | branch 2)
    # =
    # r21 / (r21 + r22)
    # ------------------------------------------------------------------------
    
    right_weight <-
      responsibilities[
        ,
        3
      ] +
      responsibilities[
        ,
        4
      ]
    
    
    right_target <-
      responsibilities[
        ,
        3
      ] /
      pmax(
        right_weight,
        1e-12
      )
    
    
    right_fit <- weighted_logistic_soft(
      G = G,
      target = right_target,
      observation_weights = right_weight,
      initial_gamma = gamma_right
    )
    
    
    gamma_right <-
      right_fit$coefficients
    
    
    # ========================================================================
    # E-STEP
    # ========================================================================
    
    gate_probabilities <- hme_gate_probabilities(
      G = G,
      gamma_root = gamma_root,
      gamma_left = gamma_left,
      gamma_right = gamma_right
    )
    
    
    expert_means <- E %*%
      expert_beta
    
    
    expert_densities <- expert_density_matrix(
      y = y,
      expert_means = expert_means,
      expert_variances = expert_variance
    )
    
    
    joint_probability <-
      gate_probabilities$leaf *
      expert_densities
    
    
    marginal_density <- rowSums(
      joint_probability
    )
    
    
    marginal_density <- pmax(
      marginal_density,
      1e-300
    )
    
    
    responsibilities_new <-
      joint_probability /
      marginal_density
    
    
    log_likelihood <- sum(
      log(
        marginal_density
      )
    )
    
    
    log_likelihood_history[
      iteration
    ] <- log_likelihood
    
    
    responsibility_change <- max(
      abs(
        responsibilities_new -
          responsibilities
      )
    )
    
    
    responsibilities <-
      responsibilities_new
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        iteration,
        "| Log-Likelihood:",
        round(
          log_likelihood,
          4
        ),
        "| Max Responsibility Change:",
        round(
          responsibility_change,
          6
        ),
        "\n"
      )
    }
    
    
    if (
      iteration > 1
    ) {
      
      likelihood_change <- abs(
        log_likelihood_history[
          iteration
        ] -
          log_likelihood_history[
            iteration - 1
          ]
      )
      
      
      if (
        likelihood_change <
        tol
      ) {
        
        break
      }
    }
  }
  
  
  # --------------------------------------------------------------------------
  # Final values
  # --------------------------------------------------------------------------
  
  gate_probabilities <- hme_gate_probabilities(
    G,
    gamma_root,
    gamma_left,
    gamma_right
  )
  
  
  expert_means <- E %*%
    expert_beta
  
  
  fitted_values <- rowSums(
    gate_probabilities$leaf *
      expert_means
  )
  
  
  return(
    list(
      gamma_root = gamma_root,
      gamma_left = gamma_left,
      gamma_right = gamma_right,
      expert_beta = expert_beta,
      expert_variance = expert_variance,
      responsibilities = responsibilities,
      gate_probabilities = gate_probabilities,
      expert_means = expert_means,
      fitted_values = fitted_values,
      log_likelihood =
        log_likelihood_history[
          seq_len(
            iteration
          )
        ],
      iterations = iteration
    )
  )
}


# ------------------------------------------------------------------------------
# 17. Fit HME
# ------------------------------------------------------------------------------

set.seed(123)


hme_model <- fit_hme(
  X = X,
  y = y,
  max_iter = 200,
  tol = 1e-5,
  verbose = TRUE
)


hme_model$iterations


# ==============================================================================
# PART VIII
#
# CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Plot Log-Likelihood
# ------------------------------------------------------------------------------

plot(
  seq_along(
    hme_model$log_likelihood
  ),
  hme_model$log_likelihood,
  type = "b",
  pch = 19,
  xlab = "EM Iteration",
  ylab = "Log-Likelihood",
  main = "HME EM Convergence"
)


# ------------------------------------------------------------------------------
# 19. Check Monotonicity Approximately
# ------------------------------------------------------------------------------

diff(
  hme_model$log_likelihood
)


# Small numerical decreases can occur because gate updates use iterative
# numerical optimization rather than an exact closed-form maximizer.


# ==============================================================================
# PART IX
#
# INSPECT THE GATING NETWORK
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Gate Coefficients
# ------------------------------------------------------------------------------

gate_coefficients <- data.frame(
  Parameter = colnames(
    G
  ),
  Root = hme_model$gamma_root,
  Left = hme_model$gamma_left,
  Right = hme_model$gamma_right
)


gate_coefficients


# ------------------------------------------------------------------------------
# 21. Root Routing Probabilities
# ------------------------------------------------------------------------------

summary(
  hme_model$gate_probabilities$root
)


hist(
  hme_model$gate_probabilities$root,
  breaks = 30,
  xlab = "Root Gate Probability",
  main = "Root Gating Network"
)


# ------------------------------------------------------------------------------
# 22. Child Routing Probabilities
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    2
  )
)


hist(
  hme_model$gate_probabilities$left,
  breaks = 30,
  xlab = "Probability",
  main = "Left Child Gate"
)


hist(
  hme_model$gate_probabilities$right,
  breaks = 30,
  xlab = "Probability",
  main = "Right Child Gate"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART X
#
# EXPERT MODELS
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Expert Coefficients
# ------------------------------------------------------------------------------

expert_coefficients <- data.frame(
  Parameter = colnames(
    G
  ),
  
  E11 = hme_model$expert_beta[
    ,
    1
  ],
  
  E12 = hme_model$expert_beta[
    ,
    2
  ],
  
  E21 = hme_model$expert_beta[
    ,
    3
  ],
  
  E22 = hme_model$expert_beta[
    ,
    4
  ]
)


expert_coefficients


# ------------------------------------------------------------------------------
# 24. Expert Standard Deviations
# ------------------------------------------------------------------------------

sqrt(
  hme_model$expert_variance
)


# ==============================================================================
# PART XI
#
# SOFT AND HARD EXPERT ASSIGNMENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Most Probable Expert
# ------------------------------------------------------------------------------

predicted_expert <- max.col(
  hme_model$responsibilities
)


table(
  Predicted = predicted_expert
)


# ------------------------------------------------------------------------------
# 26. Responsibility Summary
# ------------------------------------------------------------------------------

head(
  hme_model$responsibilities
)


# ------------------------------------------------------------------------------
# 27. Assignment Confidence
# ------------------------------------------------------------------------------

assignment_confidence <- apply(
  hme_model$responsibilities,
  1,
  max
)


hist(
  assignment_confidence,
  breaks = 30,
  xlab = "Maximum Posterior Responsibility",
  main = "Expert Assignment Confidence"
)


# ==============================================================================
# PART XII
#
# TRAINING PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. HME Training Error
# ------------------------------------------------------------------------------

hme_training_MSE <- mean(
  (
    y -
      hme_model$fitted_values
  )^2
)


hme_training_RMSE <- sqrt(
  hme_training_MSE
)


hme_training_MSE


hme_training_RMSE


# ------------------------------------------------------------------------------
# 29. Compare Against One Global Linear Model
# ------------------------------------------------------------------------------

global_linear <- lm(
  y ~ x1 + x2,
  data = data
)


global_prediction <- predict(
  global_linear
)


global_training_MSE <- mean(
  (
    y -
      global_prediction
  )^2
)


data.frame(
  Model = c(
    "Global Linear Regression",
    "Hierarchical Mixture of Experts"
  ),
  
  Training_MSE = c(
    global_training_MSE,
    hme_training_MSE
  )
)


# ==============================================================================
# PART XIII
#
# PREDICTION ON NEW DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. HME Prediction Function
# ------------------------------------------------------------------------------

predict_hme <- function(
    model,
    X_new,
    return_details = FALSE
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  G_new <- add_intercept(
    X_new
  )
  
  
  gate_probabilities <- hme_gate_probabilities(
    G = G_new,
    gamma_root = model$gamma_root,
    gamma_left = model$gamma_left,
    gamma_right = model$gamma_right
  )
  
  
  expert_means <- G_new %*%
    model$expert_beta
  
  
  prediction <- rowSums(
    gate_probabilities$leaf *
      expert_means
  )
  
  
  if (
    return_details
  ) {
    
    return(
      list(
        prediction = prediction,
        gate_probabilities =
          gate_probabilities,
        expert_means =
          expert_means
      )
    )
  }
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 31. Verify Training Prediction
# ------------------------------------------------------------------------------

prediction_check <- predict_hme(
  hme_model,
  X
)


max(
  abs(
    prediction_check -
      hme_model$fitted_values
  )
)


# ==============================================================================
# PART XIV
#
# VISUALIZE SOFT ROUTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Root Gate as Function of x1
# ------------------------------------------------------------------------------

x1_grid <- seq(
  -3,
  3,
  length.out = 400
)


root_slice <- cbind(
  x1 = x1_grid,
  x2 = 0
)


root_details <- predict_hme(
  hme_model,
  root_slice,
  return_details = TRUE
)


plot(
  x1_grid,
  root_details$gate_probabilities$root,
  type = "l",
  lwd = 2,
  ylim = c(
    0,
    1
  ),
  xlab = "x1",
  ylab = "Routing Probability",
  main = "Root Gating Network"
)


abline(
  h = 0.5,
  lty = 2
)


# ------------------------------------------------------------------------------
# 33. Lower-Level Gates as Function of x2
# ------------------------------------------------------------------------------

x2_grid <- seq(
  -3,
  3,
  length.out = 400
)


child_slice <- cbind(
  x1 = 0,
  x2 = x2_grid
)


child_details <- predict_hme(
  hme_model,
  child_slice,
  return_details = TRUE
)


matplot(
  x2_grid,
  cbind(
    child_details$gate_probabilities$left,
    child_details$gate_probabilities$right
  ),
  type = "l",
  lty = c(
    1,
    2
  ),
  lwd = 2,
  ylim = c(
    0,
    1
  ),
  xlab = "x2",
  ylab = "Routing Probability",
  main = "Lower-Level Gating Networks"
)


legend(
  "topright",
  legend = c(
    "Left Gate",
    "Right Gate"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART XV
#
# HIERARCHICAL LEAF PROBABILITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Plot Expert Probabilities Along x1 Slice
# ------------------------------------------------------------------------------

slice_details <- predict_hme(
  hme_model,
  root_slice,
  return_details = TRUE
)


matplot(
  x1_grid,
  slice_details$gate_probabilities$leaf,
  type = "l",
  lty = 1:4,
  lwd = 2,
  ylim = c(
    0,
    1
  ),
  xlab = "x1",
  ylab = "Leaf Probability",
  main = "Hierarchical Expert Probabilities"
)


legend(
  "topright",
  legend = c(
    "E11",
    "E12",
    "E21",
    "E22"
  ),
  lty = 1:4,
  lwd = 2
)


# ==============================================================================
# PART XVI
#
# PREDICTION SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Two-Dimensional Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 75


grid_x1 <- seq(
  -3,
  3,
  length.out = grid_size
)


grid_x2 <- seq(
  -3,
  3,
  length.out = grid_size
)


prediction_grid <- expand.grid(
  x1 = grid_x1,
  x2 = grid_x2
)


grid_prediction <- predict_hme(
  hme_model,
  as.matrix(
    prediction_grid
  )
)


prediction_matrix <- matrix(
  grid_prediction,
  nrow = grid_size,
  ncol = grid_size
)


image(
  grid_x1,
  grid_x2,
  prediction_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "HME Prediction Surface"
)


contour(
  grid_x1,
  grid_x2,
  prediction_matrix,
  add = TRUE
)


# ==============================================================================
# PART XVII
#
# HARD ROUTING VERSUS SOFT ROUTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Hard-Expert Prediction
# ------------------------------------------------------------------------------

hard_expert <- max.col(
  hme_model$gate_probabilities$leaf
)


hard_prediction <- hme_model$expert_means[
  cbind(
    seq_len(n),
    hard_expert
  )
]


hard_training_MSE <- mean(
  (
    y -
      hard_prediction
  )^2
)


data.frame(
  Model = c(
    "Hard Routing",
    "Soft HME Routing"
  ),
  
  Training_MSE = c(
    hard_training_MSE,
    hme_training_MSE
  )
)


# ==============================================================================
# PART XVIII
#
# TRAIN-TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(456)


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
# 38. Fit HME on Training Data
# ------------------------------------------------------------------------------

hme_train <- fit_hme(
  X = X_train,
  y = y_train,
  max_iter = 200,
  tol = 1e-5
)


hme_test_prediction <- predict_hme(
  hme_train,
  X_test
)


hme_test_MSE <- mean(
  (
    y_test -
      hme_test_prediction
  )^2
)


hme_test_RMSE <- sqrt(
  hme_test_MSE
)


hme_test_MSE


# ------------------------------------------------------------------------------
# 39. Global Linear Baseline
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = y_train,
  x1 = X_train[
    ,
    1
  ],
  x2 = X_train[
    ,
    2
  ]
)


test_data <- data.frame(
  x1 = X_test[
    ,
    1
  ],
  x2 = X_test[
    ,
    2
  ]
)


global_train <- lm(
  y ~ x1 + x2,
  data = train_data
)


global_test_prediction <- predict(
  global_train,
  newdata = test_data
)


global_test_MSE <- mean(
  (
    y_test -
      global_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 40. Quadratic Regression Baseline
# ------------------------------------------------------------------------------

quadratic_model <- lm(
  y ~
    x1 +
    x2 +
    I(x1^2) +
    I(x2^2) +
    x1:x2,
  data = train_data
)


quadratic_test_prediction <- predict(
  quadratic_model,
  newdata = test_data
)


quadratic_test_MSE <- mean(
  (
    y_test -
      quadratic_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 41. Test Comparison
# ------------------------------------------------------------------------------

test_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Quadratic Regression",
    "Hierarchical Mixture of Experts"
  ),
  
  Test_MSE = c(
    global_test_MSE,
    quadratic_test_MSE,
    hme_test_MSE
  )
)


test_comparison$Test_RMSE <- sqrt(
  test_comparison$Test_MSE
)


test_comparison


# ==============================================================================
# PART XIX
#
# FLAT MIXTURE OF EXPERTS FOR COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Softmax
# ------------------------------------------------------------------------------

softmax <- function(eta) {
  
  row_max <- apply(
    eta,
    1,
    max
  )
  
  
  shifted <- eta -
    row_max
  
  
  exp_eta <- exp(
    shifted
  )
  
  
  exp_eta /
    rowSums(
      exp_eta
    )
}


# ------------------------------------------------------------------------------
# 43. Illustrate Flat vs Hierarchical Probabilities
# ------------------------------------------------------------------------------

# A flat four-expert model would directly estimate:
#
#   P(E1 | x)
#   P(E2 | x)
#   P(E3 | x)
#   P(E4 | x)
#
# with one multinomial gating network.
#
# The HME instead factorizes:
#
#   P(E11 | x)
#   =
#   P(branch 1 | x)
#   *
#   P(E11 | branch 1, x)
#
# and similarly for the other leaves.


head(
  hme_model$gate_probabilities$leaf
)


# ==============================================================================
# PART XX
#
# MULTIPLE RANDOM INITIALIZATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Motivation
# ------------------------------------------------------------------------------

# Like Gaussian mixtures, HME likelihood is non-convex.
#
# Different initializations can lead to different local maxima.
#
# Here we demonstrate repeated fits using small perturbations of X to alter
# the deterministic starting responsibilities.


number_starts <- 5


multi_start_models <- vector(
  "list",
  number_starts
)


multi_start_log_likelihood <- numeric(
  number_starts
)


for (
  s in seq_len(
    number_starts
  )
) {
  
  set.seed(
    100 + s
  )
  
  
  X_start <- X_train +
    matrix(
      rnorm(
        length(
          X_train
        ),
        sd = 1e-4
      ),
      nrow = nrow(
        X_train
      )
    )
  
  
  model_s <- fit_hme(
    X = X_start,
    y = y_train,
    max_iter = 150,
    tol = 1e-5
  )
  
  
  multi_start_models[[s]] <- model_s
  
  
  multi_start_log_likelihood[s] <- tail(
    model_s$log_likelihood,
    1
  )
}


multi_start_log_likelihood


best_start <- which.max(
  multi_start_log_likelihood
)


best_start


# ==============================================================================
# PART XXI
#
# VISUALIZE DOMINANT EXPERT REGIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Dominant Expert Grid
# ------------------------------------------------------------------------------

grid_details <- predict_hme(
  hme_train,
  as.matrix(
    prediction_grid
  ),
  return_details = TRUE
)


dominant_expert <- max.col(
  grid_details$gate_probabilities$leaf
)


expert_region_matrix <- matrix(
  dominant_expert,
  nrow = grid_size,
  ncol = grid_size
)


image(
  grid_x1,
  grid_x2,
  expert_region_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "Dominant HME Expert Regions"
)


# ==============================================================================
# PART XXII
#
# PREDICTIVE UNCERTAINTY
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Mixture Predictive Variance
# ------------------------------------------------------------------------------

# For mixture components:
#
# E[Y | x]
# =
# sum_k pi_k mu_k
#
#
# Var(Y | x)
# =
# sum_k pi_k (sigma_k^2 + mu_k^2)
# -
# E[Y | x]^2


predict_hme_distribution <- function(
    model,
    X_new
) {
  
  details <- predict_hme(
    model,
    X_new,
    return_details = TRUE
  )
  
  
  probabilities <-
    details$gate_probabilities$leaf
  
  
  expert_means <-
    details$expert_means
  
  
  mixture_mean <- details$prediction
  
  
  second_moment <- rowSums(
    probabilities *
      sweep(
        expert_means^2,
        2,
        model$expert_variance,
        FUN = "+"
      )
  )
  
  
  mixture_variance <-
    second_moment -
    mixture_mean^2
  
  
  mixture_variance <- pmax(
    mixture_variance,
    0
  )
  
  
  return(
    list(
      mean = mixture_mean,
      variance = mixture_variance,
      sd = sqrt(
        mixture_variance
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 47. Prediction Intervals on a Slice
# ------------------------------------------------------------------------------

distribution_slice <- predict_hme_distribution(
  hme_model,
  root_slice
)


upper_slice <-
  distribution_slice$mean +
  1.96 *
  distribution_slice$sd


lower_slice <-
  distribution_slice$mean -
  1.96 *
  distribution_slice$sd


plot(
  x1_grid,
  distribution_slice$mean,
  type = "l",
  lwd = 2,
  xlab = "x1",
  ylab = "Response",
  main = "HME Predictive Mean and Uncertainty"
)


lines(
  x1_grid,
  upper_slice,
  lty = 2
)


lines(
  x1_grid,
  lower_slice,
  lty = 2
)


# Note:
#
# mean +/- 1.96 * SD is only a rough interval for a mixture distribution.
# A Gaussian mixture is not generally Gaussian.
#
# Exact predictive quantiles should be obtained from the mixture CDF or by
# simulation.


# ==============================================================================
# PART XXIII
#
# MONTE CARLO PREDICTIVE DISTRIBUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Draw from HME Predictive Distribution
# ------------------------------------------------------------------------------

simulate_hme <- function(
    model,
    X_new,
    draws_per_observation = 1
) {
  
  details <- predict_hme(
    model,
    X_new,
    return_details = TRUE
  )
  
  
  n_new <- nrow(
    as.matrix(
      X_new
    )
  )
  
  
  simulations <- matrix(
    NA_real_,
    nrow = n_new,
    ncol = draws_per_observation
  )
  
  
  for (
    i in seq_len(
      n_new
    )
  ) {
    
    for (
      b in seq_len(
        draws_per_observation
      )
    ) {
      
      expert <- sample(
        1:4,
        size = 1,
        prob =
          details$gate_probabilities$leaf[
            i,
          ]
      )
      
      
      simulations[
        i,
        b
      ] <- rnorm(
        1,
        mean =
          details$expert_means[
            i,
            expert
          ],
        sd = sqrt(
          model$expert_variance[
            expert
          ]
        )
      )
    }
  }
  
  
  return(
    simulations
  )
}


# ------------------------------------------------------------------------------
# 49. Predictive Distribution at One Point
# ------------------------------------------------------------------------------

new_point <- matrix(
  c(
    0,
    0
  ),
  nrow = 1
)


colnames(
  new_point
) <- c(
  "x1",
  "x2"
)


set.seed(999)


predictive_draws <- simulate_hme(
  hme_model,
  new_point,
  draws_per_observation = 5000
)


hist(
  predictive_draws,
  breaks = 50,
  xlab = "Predicted y",
  main = "HME Predictive Distribution"
)


# ==============================================================================
# PART XXIV
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Final Summary
# ------------------------------------------------------------------------------

cat(
  "Hierarchical Mixture of Experts Summary\n"
)


cat(
  "---------------------------------------\n"
)


cat(
  "EM iterations:",
  hme_model$iterations,
  "\n"
)


cat(
  "Final training log-likelihood:",
  round(
    tail(
      hme_model$log_likelihood,
      1
    ),
    4
  ),
  "\n"
)


cat(
  "HME training MSE:",
  round(
    hme_training_MSE,
    4
  ),
  "\n"
)


cat(
  "HME test MSE:",
  round(
    hme_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Global linear test MSE:",
  round(
    global_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Quadratic regression test MSE:",
  round(
    quadratic_test_MSE,
    4
  ),
  "\n"
)


cat(
  "\nRoot gate coefficients:\n"
)


print(
  hme_model$gamma_root
)


cat(
  "\nLeft child gate coefficients:\n"
)


print(
  hme_model$gamma_left
)


cat(
  "\nRight child gate coefficients:\n"
)


print(
  hme_model$gamma_right
)


cat(
  "\nExpert coefficients:\n"
)


print(
  expert_coefficients
)