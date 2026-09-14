# ==============================================================================
# Projection Pursuit Regression
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Projection Pursuit Regression
#   - Ridge functions
#   - Dimension reduction
#   - Nonlinear regression
#   - Projection directions
#   - Local linear smoothing
#   - Sequential additive fitting
#   - Numerical optimization
#   - Number of projection terms
#   - Cross-validation
#
#
# Projection Pursuit Regression model:
#
#       f(x)
#       =
#       beta_0
#       +
#       sum_{m=1}^M
#           g_m(alpha_m' x)
#
#
# where:
#
#       alpha_m
#
# is a projection direction and
#
#       g_m(.)
#
# is a one-dimensional nonlinear ridge function.
#
#
# Each term therefore has the form:
#
#       g(alpha' x)
#
#
# Even when x is high-dimensional, each ridge function only requires
# smoothing in ONE dimension.
#
#
# This script implements a pedagogical stagewise version of PPR:
#
#   1. Start with F_0(x) = mean(y)
#
#   2. Compute residuals
#
#   3. Search for a direction alpha that produces a useful projection
#
#          z = X alpha
#
#   4. Fit a one-dimensional smoother:
#
#          residual ~ g(z)
#
#   5. Add the ridge function to the current model
#
#   6. Repeat
#
#
# This captures the core projection-pursuit idea, although production PPR
# algorithms generally perform additional joint optimization/backfitting of
# previously fitted ridge functions.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE DATA WITH LOW-DIMENSIONAL PROJECTION STRUCTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Predictors
# ------------------------------------------------------------------------------

set.seed(123)

n <- 500

p <- 5


X <- matrix(
  rnorm(
    n * p
  ),
  nrow = n,
  ncol = p
)


colnames(X) <- paste0(
  "x",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 2. True Projection Directions
# ------------------------------------------------------------------------------

alpha_1_true <- c(
  0.70,
  0.70,
  0,
  0,
  0
)


alpha_1_true <- alpha_1_true /
  sqrt(
    sum(
      alpha_1_true^2
    )
  )


alpha_2_true <- c(
  0,
  0.20,
  0.75,
  -0.60,
  0
)


alpha_2_true <- alpha_2_true /
  sqrt(
    sum(
      alpha_2_true^2
    )
  )


# ------------------------------------------------------------------------------
# 3. True Projections
# ------------------------------------------------------------------------------

z1_true <- as.vector(
  X %*%
    alpha_1_true
)


z2_true <- as.vector(
  X %*%
    alpha_2_true
)


# ------------------------------------------------------------------------------
# 4. True Ridge Functions
# ------------------------------------------------------------------------------

g1_true <- function(z) {
  
  3 *
    sin(
      1.5 * z
    )
}


g2_true <- function(z) {
  
  1.5 *
    (
      z^2 -
        1
    )
}


beta_0_true <- 2


signal <- beta_0_true +
  g1_true(
    z1_true
  ) +
  g2_true(
    z2_true
  )


sigma <- 1


y <- signal +
  rnorm(
    n,
    sd = sigma
  )


# ------------------------------------------------------------------------------
# 5. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data.frame(
    y = y,
    X
  ),
  main = "Projection Pursuit Regression Data"
)


# ==============================================================================
# PART II
#
# STANDARDIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Standardize Predictors
# ------------------------------------------------------------------------------

X_mean <- colMeans(
  X
)


X_sd <- apply(
  X,
  2,
  sd
)


X_scaled <- sweep(
  X,
  2,
  X_mean,
  "-"
)


X_scaled <- sweep(
  X_scaled,
  2,
  X_sd,
  "/"
)


# Projection directions are much easier to interpret and optimize if the
# predictors operate on comparable scales.


# ==============================================================================
# PART III
#
# LOCAL LINEAR SMOOTHER
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Gaussian Kernel
# ------------------------------------------------------------------------------

gaussian_kernel <- function(u) {
  
  exp(
    -0.5 *
      u^2
  )
}


# ------------------------------------------------------------------------------
# 8. Local Linear Prediction at One Point
# ------------------------------------------------------------------------------

local_linear_one <- function(
    z_train,
    y_train,
    z0,
    bandwidth
) {
  
  distance <- (
    z_train -
      z0
  ) /
    bandwidth
  
  
  weights <- gaussian_kernel(
    distance
  )
  
  
  if (
    sum(weights) <
    1e-12
  ) {
    
    return(
      mean(
        y_train
      )
    )
  }
  
  
  centered_z <- z_train -
    z0
  
  
  X_local <- cbind(
    1,
    centered_z
  )
  
  
  sqrt_weights <- sqrt(
    weights
  )
  
  
  X_weighted <- X_local *
    sqrt_weights
  
  
  y_weighted <- y_train *
    sqrt_weights
  
  
  fit <- tryCatch(
    qr.solve(
      X_weighted,
      y_weighted
    ),
    error = function(e) {
      
      c(
        weighted.mean(
          y_train,
          weights
        ),
        0
      )
    }
  )
  
  
  return(
    fit[1]
  )
}


# ------------------------------------------------------------------------------
# 9. Local Linear Smoother
# ------------------------------------------------------------------------------

local_linear_predict <- function(
    z_train,
    y_train,
    z_new,
    bandwidth = NULL
) {
  
  z_train <- as.numeric(
    z_train
  )
  
  
  z_new <- as.numeric(
    z_new
  )
  
  
  if (
    is.null(
      bandwidth
    )
  ) {
    
    bandwidth <- 0.5 *
      sd(
        z_train
      )
  }
  
  
  bandwidth <- max(
    bandwidth,
    1e-4
  )
  
  
  sapply(
    z_new,
    function(z0) {
      
      local_linear_one(
        z_train =
          z_train,
        y_train =
          y_train,
        z0 =
          z0,
        bandwidth =
          bandwidth
      )
    }
  )
}


# ==============================================================================
# PART IV
#
# PROJECTION DIRECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Normalize Direction
# ------------------------------------------------------------------------------

normalize_direction <- function(alpha) {
  
  norm_alpha <- sqrt(
    sum(
      alpha^2
    )
  )
  
  
  if (
    norm_alpha <
    1e-12
  ) {
    
    alpha <- rep(
      0,
      length(alpha)
    )
    
    
    alpha[1] <- 1
    
    
    return(
      alpha
    )
  }
  
  
  alpha /
    norm_alpha
}


# ------------------------------------------------------------------------------
# 11. Sign Convention
# ------------------------------------------------------------------------------

# alpha and -alpha describe the same one-dimensional subspace:
#
#       alpha' x
#
# versus
#
#       -alpha' x
#
# because the ridge function can absorb the sign reversal.
#
# For reproducibility we orient the direction so its largest absolute
# coefficient is positive.


orient_direction <- function(alpha) {
  
  alpha <- normalize_direction(
    alpha
  )
  
  
  largest_index <- which.max(
    abs(
      alpha
    )
  )
  
  
  if (
    alpha[
      largest_index
    ] <
    0
  ) {
    
    alpha <- -alpha
  }
  
  
  alpha
}


# ==============================================================================
# PART V
#
# OBJECTIVE FOR ONE PROJECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Projection Objective
# ------------------------------------------------------------------------------

projection_objective <- function(
    alpha_raw,
    X,
    target,
    bandwidth_multiplier = 0.5
) {
  
  alpha <- normalize_direction(
    alpha_raw
  )
  
  
  z <- as.vector(
    X %*%
      alpha
  )
  
  
  bandwidth <- bandwidth_multiplier *
    sd(
      z
    )
  
  
  fitted <- local_linear_predict(
    z_train = z,
    y_train = target,
    z_new = z,
    bandwidth =
      bandwidth
  )
  
  
  mean(
    (
      target -
        fitted
    )^2
  )
}


# ==============================================================================
# PART VI
#
# INITIAL PROJECTION SEARCH
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Linear Initialization
# ------------------------------------------------------------------------------

# A useful initialization comes from the direction most linearly associated
# with the current residuals.


initial_projection_direction <- function(
    X,
    target
) {
  
  direction <- as.vector(
    crossprod(
      X,
      target
    )
  )
  
  
  if (
    sqrt(
      sum(
        direction^2
      )
    ) <
    1e-10
  ) {
    
    direction <- rnorm(
      ncol(X)
    )
  }
  
  
  orient_direction(
    direction
  )
}


# ------------------------------------------------------------------------------
# 14. Random Direction
# ------------------------------------------------------------------------------

random_direction <- function(p) {
  
  orient_direction(
    rnorm(p)
  )
}


# ==============================================================================
# PART VII
#
# OPTIMIZE ONE RIDGE DIRECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Direction Search
# ------------------------------------------------------------------------------

optimize_projection_direction <- function(
    X,
    target,
    bandwidth_multiplier = 0.5,
    number_starts = 5,
    maxit = 100
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  starting_directions <- vector(
    "list",
    number_starts
  )
  
  
  starting_directions[[1]] <-
    initial_projection_direction(
      X,
      target
    )
  
  
  if (
    number_starts >
    1
  ) {
    
    for (
      s in 2:number_starts
    ) {
      
      starting_directions[[s]] <-
        random_direction(
          p
        )
    }
  }
  
  
  best_value <- Inf
  
  best_alpha <- NULL
  
  best_optimization <- NULL
  
  
  for (
    s in seq_len(
      number_starts
    )
  ) {
    
    optimization <- optim(
      par =
        starting_directions[[s]],
      
      fn = projection_objective,
      
      X = X,
      
      target =
        target,
      
      bandwidth_multiplier =
        bandwidth_multiplier,
      
      method = "BFGS",
      
      control = list(
        maxit =
          maxit,
        reltol =
          1e-6
      )
    )
    
    
    if (
      optimization$value <
      best_value
    ) {
      
      best_value <-
        optimization$value
      
      best_alpha <-
        orient_direction(
          optimization$par
        )
      
      best_optimization <-
        optimization
    }
  }
  
  
  return(
    list(
      alpha =
        best_alpha,
      objective =
        best_value,
      optimization =
        best_optimization
    )
  )
}


# ==============================================================================
# PART VIII
#
# FIT ONE RIDGE FUNCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Ridge Function Fit
# ------------------------------------------------------------------------------

fit_ridge_function <- function(
    X,
    target,
    bandwidth_multiplier = 0.5,
    number_starts = 5,
    maxit = 100
) {
  
  direction_fit <-
    optimize_projection_direction(
      X = X,
      target =
        target,
      bandwidth_multiplier =
        bandwidth_multiplier,
      number_starts =
        number_starts,
      maxit =
        maxit
    )
  
  
  alpha <- direction_fit$alpha
  
  
  z <- as.vector(
    X %*%
      alpha
  )
  
  
  bandwidth <-
    bandwidth_multiplier *
    sd(
      z
    )
  
  
  ridge_prediction <- local_linear_predict(
    z_train = z,
    y_train =
      target,
    z_new = z,
    bandwidth =
      bandwidth
  )
  
  
  # Center the ridge function.
  #
  # This makes the decomposition more identifiable because any global
  # constant stays in beta_0 rather than individual ridge functions.
  
  ridge_mean <- mean(
    ridge_prediction
  )
  
  
  ridge_prediction <-
    ridge_prediction -
    ridge_mean
  
  
  return(
    list(
      alpha =
        alpha,
      z_train =
        z,
      target_train =
        target,
      bandwidth =
        bandwidth,
      ridge_mean =
        ridge_mean,
      fitted_values =
        ridge_prediction,
      objective =
        direction_fit$objective
    )
  )
}


# ==============================================================================
# PART IX
#
# MANUAL STAGEWISE PPR
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Fit Projection Pursuit Regression
# ------------------------------------------------------------------------------

fit_ppr_manual <- function(
    X,
    y,
    M = 2,
    bandwidth_multiplier = 0.5,
    number_starts = 5,
    maxit = 100,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Store scaling parameters
  # --------------------------------------------------------------------------
  
  X_mean <- colMeans(
    X
  )
  
  
  X_sd <- apply(
    X,
    2,
    sd
  )
  
  
  X_scaled <- sweep(
    X,
    2,
    X_mean,
    "-"
  )
  
  
  X_scaled <- sweep(
    X_scaled,
    2,
    X_sd,
    "/"
  )
  
  
  # --------------------------------------------------------------------------
  # Initial model
  # --------------------------------------------------------------------------
  
  beta_0 <- mean(
    y
  )
  
  
  fitted_values <- rep(
    beta_0,
    length(y)
  )
  
  
  ridge_functions <- vector(
    "list",
    M
  )
  
  
  training_MSE <- numeric(
    M
  )
  
  
  improvements <- numeric(
    M
  )
  
  
  previous_MSE <- mean(
    (
      y -
        fitted_values
    )^2
  )
  
  
  # ==========================================================================
  # STAGEWISE PPR LOOP
  # ==========================================================================
  
  for (
    m in seq_len(M)
  ) {
    
    # ------------------------------------------------------------------------
    # Current residual
    # ------------------------------------------------------------------------
    
    residual <- y -
      fitted_values
    
    
    # ------------------------------------------------------------------------
    # Find a projection and smooth along it
    # ------------------------------------------------------------------------
    
    ridge_fit <- fit_ridge_function(
      X = X_scaled,
      target =
        residual,
      bandwidth_multiplier =
        bandwidth_multiplier,
      number_starts =
        number_starts,
      maxit =
        maxit
    )
    
    
    # ------------------------------------------------------------------------
    # Add ridge function
    # ------------------------------------------------------------------------
    
    fitted_values <-
      fitted_values +
      ridge_fit$fitted_values
    
    
    ridge_functions[[m]] <-
      ridge_fit
    
    
    current_MSE <- mean(
      (
        y -
          fitted_values
      )^2
    )
    
    
    training_MSE[m] <-
      current_MSE
    
    
    improvements[m] <-
      previous_MSE -
      current_MSE
    
    
    previous_MSE <-
      current_MSE
    
    
    if (verbose) {
      
      cat(
        "Projection:",
        m,
        "| Training MSE:",
        round(
          current_MSE,
          4
        ),
        "\n"
      )
      
      
      print(
        round(
          ridge_fit$alpha,
          3
        )
      )
    }
  }
  
  
  return(
    list(
      beta_0 =
        beta_0,
      ridge_functions =
        ridge_functions,
      fitted_values =
        fitted_values,
      training_MSE =
        training_MSE,
      improvements =
        improvements,
      X_mean =
        X_mean,
      X_sd =
        X_sd,
      feature_names =
        colnames(X),
      M =
        M
    )
  )
}


# ------------------------------------------------------------------------------
# 18. Fit Two-Term PPR
# ------------------------------------------------------------------------------

ppr_model <- fit_ppr_manual(
  X = X,
  y = y,
  M = 2,
  bandwidth_multiplier = 0.45,
  number_starts = 5,
  maxit = 100,
  verbose = TRUE
)


# ==============================================================================
# PART X
#
# PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Predict One Ridge Function
# ------------------------------------------------------------------------------

predict_ridge_function <- function(
    ridge,
    X_scaled_new
) {
  
  z_new <- as.vector(
    X_scaled_new %*%
      ridge$alpha
  )
  
  
  prediction <- local_linear_predict(
    z_train =
      ridge$z_train,
    y_train =
      ridge$target_train,
    z_new =
      z_new,
    bandwidth =
      ridge$bandwidth
  )
  
  
  prediction -
    ridge$ridge_mean
}


# ------------------------------------------------------------------------------
# 20. PPR Prediction
# ------------------------------------------------------------------------------

predict_ppr_manual <- function(
    model,
    X_new,
    M = model$M
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  X_scaled_new <- sweep(
    X_new,
    2,
    model$X_mean,
    "-"
  )
  
  
  X_scaled_new <- sweep(
    X_scaled_new,
    2,
    model$X_sd,
    "/"
  )
  
  
  M <- min(
    M,
    length(
      model$ridge_functions
    )
  )
  
  
  prediction <- rep(
    model$beta_0,
    nrow(
      X_new
    )
  )
  
  
  if (
    M == 0
  ) {
    
    return(
      prediction
    )
  }
  
  
  for (
    m in seq_len(M)
  ) {
    
    prediction <-
      prediction +
      predict_ridge_function(
        model$ridge_functions[[m]],
        X_scaled_new
      )
  }
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 21. Verify Training Prediction
# ------------------------------------------------------------------------------

prediction_check <- predict_ppr_manual(
  ppr_model,
  X
)


max(
  abs(
    prediction_check -
      ppr_model$fitted_values
  )
)


# ==============================================================================
# PART XI
#
# TRAINING PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Training MSE
# ------------------------------------------------------------------------------

training_MSE <- mean(
  (
    y -
      ppr_model$fitted_values
  )^2
)


training_RMSE <- sqrt(
  training_MSE
)


training_MSE


training_RMSE


# ------------------------------------------------------------------------------
# 23. Stagewise Error
# ------------------------------------------------------------------------------

plot(
  seq_along(
    ppr_model$training_MSE
  ),
  ppr_model$training_MSE,
  type = "b",
  pch = 19,
  lwd = 2,
  xlab = "Number of Projection Terms",
  ylab = "Training MSE",
  main = "PPR Training Error"
)


# ==============================================================================
# PART XII
#
# EXAMINE LEARNED DIRECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Direction Matrix
# ------------------------------------------------------------------------------

direction_matrix <- do.call(
  rbind,
  lapply(
    ppr_model$ridge_functions,
    function(ridge) {
      
      ridge$alpha
    }
  )
)


colnames(
  direction_matrix
) <- colnames(X)


rownames(
  direction_matrix
) <- paste0(
  "Projection_",
  seq_len(
    nrow(
      direction_matrix
    )
  )
)


round(
  direction_matrix,
  3
)


# ------------------------------------------------------------------------------
# 25. True Directions
# ------------------------------------------------------------------------------

true_directions <- rbind(
  True_Projection_1 =
    alpha_1_true,
  True_Projection_2 =
    alpha_2_true
)


colnames(
  true_directions
) <- colnames(X)


round(
  true_directions,
  3
)


# ==============================================================================
# PART XIII
#
# DIRECTION SIMILARITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Absolute Cosine Similarity
# ------------------------------------------------------------------------------

# Directions alpha and -alpha represent the same projection subspace, so
# absolute cosine similarity is appropriate.


direction_similarity <- function(
    a,
    b
) {
  
  a <- normalize_direction(
    a
  )
  
  
  b <- normalize_direction(
    b
  )
  
  
  abs(
    sum(
      a * b
    )
  )
}


similarity_matrix <- matrix(
  NA_real_,
  nrow =
    nrow(
      direction_matrix
    ),
  ncol =
    nrow(
      true_directions
    )
)


for (
  i in seq_len(
    nrow(
      direction_matrix
    )
  )
) {
  
  for (
    j in seq_len(
      nrow(
        true_directions
      )
    )
  ) {
    
    similarity_matrix[
      i,
      j
    ] <- direction_similarity(
      direction_matrix[
        i,
      ],
      true_directions[
        j,
      ]
    )
  }
}


rownames(
  similarity_matrix
) <- rownames(
  direction_matrix
)


colnames(
  similarity_matrix
) <- rownames(
  true_directions
)


round(
  similarity_matrix,
  3
)


# ==============================================================================
# PART XIV
#
# VISUALIZE RIDGE FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Plot Each Learned Ridge Function
# ------------------------------------------------------------------------------

for (
  m in seq_along(
    ppr_model$ridge_functions
  )
) {
  
  ridge <- ppr_model$ridge_functions[[m]]
  
  
  order_z <- order(
    ridge$z_train
  )
  
  
  plot(
    ridge$z_train[
      order_z
    ],
    ridge$fitted_values[
      order_z
    ],
    type = "l",
    lwd = 2,
    xlab = expression(
      alpha^T * x
    ),
    ylab = "Estimated Ridge Function",
    main = paste(
      "Projection Pursuit Ridge Function",
      m
    )
  )
}


# ==============================================================================
# PART XV
#
# VISUALIZE RAW RESIDUALS AGAINST PROJECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. First Projection
# ------------------------------------------------------------------------------

ridge_1 <- ppr_model$ridge_functions[[1]]


plot(
  ridge_1$z_train,
  y -
    ppr_model$beta_0,
  pch = 19,
  cex = 0.5,
  xlab = expression(
    alpha[1]^T * x
  ),
  ylab = "Initial Residual",
  main = "First PPR Projection"
)


order_1 <- order(
  ridge_1$z_train
)


lines(
  ridge_1$z_train[
    order_1
  ],
  ridge_1$fitted_values[
    order_1
  ],
  lwd = 3
)


# ------------------------------------------------------------------------------
# 29. Second Projection
# ------------------------------------------------------------------------------

first_stage_fit <- ppr_model$beta_0 +
  ridge_1$fitted_values


second_stage_residual <- y -
  first_stage_fit


ridge_2 <- ppr_model$ridge_functions[[2]]


plot(
  ridge_2$z_train,
  second_stage_residual,
  pch = 19,
  cex = 0.5,
  xlab = expression(
    alpha[2]^T * x
  ),
  ylab = "Residual After First Projection",
  main = "Second PPR Projection"
)


order_2 <- order(
  ridge_2$z_train
)


lines(
  ridge_2$z_train[
    order_2
  ],
  ridge_2$fitted_values[
    order_2
  ],
  lwd = 3
)


# ==============================================================================
# PART XVI
#
# NUMBER OF PROJECTION TERMS
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Fit Models with Different M
# ------------------------------------------------------------------------------

M_values <- 1:5


complexity_results <- data.frame(
  M =
    M_values,
  Training_MSE =
    NA_real_
)


complexity_models <- vector(
  "list",
  length(
    M_values
  )
)


for (
  i in seq_along(
    M_values
  )
) {
  
  model_i <- fit_ppr_manual(
    X = X,
    y = y,
    M =
      M_values[i],
    bandwidth_multiplier =
      0.45,
    number_starts = 3,
    maxit = 75,
    verbose = FALSE
  )
  
  
  complexity_models[[i]] <-
    model_i
  
  
  complexity_results$Training_MSE[i] <-
    mean(
      (
        y -
          model_i$fitted_values
      )^2
    )
}


complexity_results


plot(
  complexity_results$M,
  complexity_results$Training_MSE,
  type = "b",
  pch = 19,
  xlab = "Number of Projection Terms",
  ylab = "Training MSE",
  main = "PPR Complexity"
)


# ==============================================================================
# PART XVII
#
# TRAIN-TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Split Data
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


# ==============================================================================
# PART XVIII
#
# TEST PERFORMANCE VS NUMBER OF PROJECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Compare M
# ------------------------------------------------------------------------------

M_test_values <- 1:4


M_test_results <- data.frame(
  M =
    M_test_values,
  Train_MSE =
    NA_real_,
  Test_MSE =
    NA_real_
)


for (
  i in seq_along(
    M_test_values
  )
) {
  
  model_i <- fit_ppr_manual(
    X = X_train,
    y = y_train,
    M =
      M_test_values[i],
    bandwidth_multiplier =
      0.45,
    number_starts = 3,
    maxit = 75
  )
  
  
  train_prediction_i <- predict_ppr_manual(
    model_i,
    X_train
  )
  
  
  test_prediction_i <- predict_ppr_manual(
    model_i,
    X_test
  )
  
  
  M_test_results$Train_MSE[i] <-
    mean(
      (
        y_train -
          train_prediction_i
      )^2
    )
  
  
  M_test_results$Test_MSE[i] <-
    mean(
      (
        y_test -
          test_prediction_i
      )^2
    )
}


M_test_results


matplot(
  M_test_results$M,
  cbind(
    M_test_results$Train_MSE,
    M_test_results$Test_MSE
  ),
  type = "b",
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2,
  xlab = "Number of Projection Terms",
  ylab = "MSE",
  main = "PPR Train/Test Error"
)


legend(
  "topright",
  legend = c(
    "Training",
    "Test"
  ),
  lty = c(
    1,
    2
  ),
  pch = c(
    19,
    17
  ),
  lwd = 2
)


# IMPORTANT:
#
# Test error is displayed here only for illustration.
# Model complexity should normally be selected by validation or CV.


# ==============================================================================
# PART XIX
#
# CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Manual K-Fold Cross Validation
# ------------------------------------------------------------------------------

ppr_cv <- function(
    X,
    y,
    M_values = 1:4,
    k = 5,
    bandwidth_multiplier = 0.45,
    number_starts = 2,
    maxit = 60,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  n <- nrow(
    X
  )
  
  
  fold_id <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  CV_MSE <- matrix(
    NA_real_,
    nrow =
      length(
        M_values
      ),
    ncol = k
  )
  
  
  for (
    fold in seq_len(k)
  ) {
    
    training_indices <- which(
      fold_id != fold
    )
    
    
    validation_indices <- which(
      fold_id == fold
    )
    
    
    for (
      i in seq_along(
        M_values
      )
    ) {
      
      model_fold <- fit_ppr_manual(
        X = X[
          training_indices,
          ,
          drop = FALSE
        ],
        y = y[
          training_indices
        ],
        M =
          M_values[i],
        bandwidth_multiplier =
          bandwidth_multiplier,
        number_starts =
          number_starts,
        maxit =
          maxit
      )
      
      
      validation_prediction <-
        predict_ppr_manual(
          model_fold,
          X[
            validation_indices,
            ,
            drop = FALSE
          ]
        )
      
      
      CV_MSE[
        i,
        fold
      ] <- mean(
        (
          y[
            validation_indices
          ] -
            validation_prediction
        )^2
      )
    }
  }
  
  
  mean_CV <- rowMeans(
    CV_MSE
  )
  
  
  SE_CV <- apply(
    CV_MSE,
    1,
    sd
  ) /
    sqrt(k)
  
  
  return(
    list(
      M_values =
        M_values,
      mean =
        mean_CV,
      SE =
        SE_CV,
      fold_MSE =
        CV_MSE
    )
  )
}


# ------------------------------------------------------------------------------
# 34. Run CV
# ------------------------------------------------------------------------------

CV_result <- ppr_cv(
  X = X_train,
  y = y_train,
  M_values = 1:4,
  k = 5,
  bandwidth_multiplier = 0.45,
  number_starts = 2,
  maxit = 60,
  seed = 123
)


CV_table <- data.frame(
  M =
    CV_result$M_values,
  CV_MSE =
    CV_result$mean,
  CV_SE =
    CV_result$SE
)


CV_table


# ------------------------------------------------------------------------------
# 35. Select M
# ------------------------------------------------------------------------------

best_index <- which.min(
  CV_result$mean
)


best_M <- CV_result$M_values[
  best_index
]


best_M


# ------------------------------------------------------------------------------
# 36. One-SE Rule
# ------------------------------------------------------------------------------

one_SE_threshold <-
  CV_result$mean[
    best_index
  ] +
  CV_result$SE[
    best_index
  ]


eligible <- which(
  CV_result$mean <=
    one_SE_threshold
)


one_SE_M <- min(
  CV_result$M_values[
    eligible
  ]
)


one_SE_M


# ------------------------------------------------------------------------------
# 37. Plot CV
# ------------------------------------------------------------------------------

plot(
  CV_result$M_values,
  CV_result$mean,
  type = "b",
  pch = 19,
  lwd = 2,
  xlab = "Number of Projection Terms",
  ylab = "Cross-Validated MSE",
  main = "PPR Cross Validation"
)


abline(
  h =
    one_SE_threshold,
  lty = 2
)


abline(
  v =
    best_M,
  lty = 3
)


abline(
  v =
    one_SE_M,
  lty = 4
)


# ==============================================================================
# PART XX
#
# FINAL PPR MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Fit CV-Selected Model
# ------------------------------------------------------------------------------

final_ppr <- fit_ppr_manual(
  X = X_train,
  y = y_train,
  M =
    one_SE_M,
  bandwidth_multiplier =
    0.45,
  number_starts = 5,
  maxit = 100
)


# ------------------------------------------------------------------------------
# 39. Final Predictions
# ------------------------------------------------------------------------------

final_train_prediction <- predict_ppr_manual(
  final_ppr,
  X_train
)


final_test_prediction <- predict_ppr_manual(
  final_ppr,
  X_test
)


final_train_MSE <- mean(
  (
    y_train -
      final_train_prediction
  )^2
)


final_test_MSE <- mean(
  (
    y_test -
      final_test_prediction
  )^2
)


final_test_RMSE <- sqrt(
  final_test_MSE
)


final_train_MSE


final_test_MSE


# ==============================================================================
# PART XXI
#
# COMPARE WITH LINEAR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Linear Regression
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = y_train,
  X_train
)


test_data <- data.frame(
  X_test
)


linear_model <- lm(
  y ~ .,
  data =
    train_data
)


linear_prediction <- predict(
  linear_model,
  newdata =
    test_data
)


linear_test_MSE <- mean(
  (
    y_test -
      linear_prediction
  )^2
)


# ==============================================================================
# PART XXII
#
# ADDITIVE NONLINEAR BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Polynomial Additive Model
# ------------------------------------------------------------------------------

polynomial_model <- lm(
  y ~
    poly(x1, 3) +
    poly(x2, 3) +
    poly(x3, 3) +
    poly(x4, 3) +
    poly(x5, 3),
  data =
    train_data
)


polynomial_prediction <- predict(
  polynomial_model,
  newdata =
    test_data
)


polynomial_test_MSE <- mean(
  (
    y_test -
      polynomial_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 42. Comparison
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Additive Cubic Model",
    "Projection Pursuit Regression"
  ),
  
  Test_MSE = c(
    linear_test_MSE,
    polynomial_test_MSE,
    final_test_MSE
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XXIII
#
# WHY PROJECTIONS MATTER
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Compare True Projection with Individual Predictors
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    3
  )
)


plot(
  X[, 1],
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x1",
  ylab = "y",
  main = "Response vs x1"
)


plot(
  X[, 2],
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x2",
  ylab = "y",
  main = "Response vs x2"
)


plot(
  z1_true,
  y,
  pch = 19,
  cex = 0.5,
  xlab = expression(
    alpha[1]^T * x
  ),
  ylab = "y",
  main = "Response vs True Projection"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART XXIV
#
# TWO-DIMENSIONAL RESPONSE SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Vary x1 and x2
# ------------------------------------------------------------------------------

grid_size <- 80


grid_x1 <- seq(
  -3,
  3,
  length.out =
    grid_size
)


grid_x2 <- seq(
  -3,
  3,
  length.out =
    grid_size
)


grid <- expand.grid(
  x1 =
    grid_x1,
  x2 =
    grid_x2
)


X_grid <- cbind(
  x1 =
    grid$x1,
  x2 =
    grid$x2,
  x3 = 0,
  x4 = 0,
  x5 = 0
)


grid_prediction <- predict_ppr_manual(
  final_ppr,
  X_grid
)


prediction_matrix <- matrix(
  grid_prediction,
  nrow =
    grid_size,
  ncol =
    grid_size
)


image(
  grid_x1,
  grid_x2,
  prediction_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "PPR Prediction Surface"
)


contour(
  grid_x1,
  grid_x2,
  prediction_matrix,
  add = TRUE
)


# ==============================================================================
# PART XXV
#
# OPTIONAL VERIFICATION WITH R'S PPR
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Built-In Projection Pursuit Regression
# ------------------------------------------------------------------------------

# ppr() is included with R's stats package.
#
# Its fitting procedure is substantially more sophisticated than the simple
# stagewise implementation above, so numerical results need not match.


built_in_ppr <- ppr(
  x = X_train,
  y = y_train,
  nterms =
    one_SE_M,
  max.terms =
    max(
      one_SE_M,
      5
    )
)


built_in_prediction <- predict(
  built_in_ppr,
  newdata =
    X_test
)


built_in_test_MSE <- mean(
  (
    y_test -
      built_in_prediction
  )^2
)


data.frame(
  Model = c(
    "Manual PPR",
    "stats::ppr"
  ),
  Test_MSE = c(
    final_test_MSE,
    built_in_test_MSE
  )
)


# ==============================================================================
# PART XXVI
#
# INSPECT BUILT-IN PPR
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Built-In Projection Directions
# ------------------------------------------------------------------------------

built_in_ppr$alpha


# ------------------------------------------------------------------------------
# 47. Built-In Summary
# ------------------------------------------------------------------------------

summary(
  built_in_ppr
)


# ==============================================================================
# PART XXVII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Output
# ------------------------------------------------------------------------------

cat(
  "Projection Pursuit Regression Summary\n"
)


cat(
  "-------------------------------------\n"
)


cat(
  "Best CV number of terms:",
  best_M,
  "\n"
)


cat(
  "One-SE number of terms:",
  one_SE_M,
  "\n"
)


cat(
  "Linear regression test MSE:",
  round(
    linear_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Additive polynomial test MSE:",
  round(
    polynomial_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Manual PPR test MSE:",
  round(
    final_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Manual PPR test RMSE:",
  round(
    final_test_RMSE,
    4
  ),
  "\n"
)


cat(
  "\nLearned projection directions:\n"
)


final_direction_matrix <- do.call(
  rbind,
  lapply(
    final_ppr$ridge_functions,
    function(ridge) {
      
      ridge$alpha
    }
  )
)


colnames(
  final_direction_matrix
) <- final_ppr$feature_names


print(
  round(
    final_direction_matrix,
    3
  )
)