# ==============================================================================
# Principal Curves
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   Hastie and Stuetzle (1989), Principal Curves
#
# Main ideas:
#   - Nonlinear dimensionality reduction
#   - PCA as a linear special case
#   - One-dimensional nonlinear manifolds
#   - Self-consistency
#   - Projection index
#   - Curve smoothing
#   - Projection onto a curve
#   - Iterative fitting
#   - Reconstruction error
#
#
# A principal curve is a smooth curve:
#
#       f(lambda)
#       =
#       (f_1(lambda), ..., f_p(lambda))
#
#
# passing through the "middle" of a multidimensional data cloud.
#
#
# The central self-consistency property is:
#
#       f(lambda)
#       =
#       E[X | lambda_f(X) = lambda]
#
#
# where:
#
#       lambda_f(x)
#
# is the parameter value corresponding to the point on the curve closest
# to x.
#
#
# This script implements a pedagogical approximation:
#
#   1. Initialize with PC1
#   2. Smooth each coordinate X_j against lambda
#   3. Construct a dense discretized curve
#   4. Project observations onto the curve
#   5. Update lambda
#   6. Repeat
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Curved Two-Dimensional Data
# ------------------------------------------------------------------------------

set.seed(123)


n <- 350


latent_t <- runif(
  n,
  min = -2.5,
  max = 2.5
)


true_x1 <- 2.0 *
  latent_t


true_x2 <- 2.5 *
  sin(
    latent_t
  )


x1 <- true_x1 +
  rnorm(
    n,
    sd = 0.35
  )


x2 <- true_x2 +
  rnorm(
    n,
    sd = 0.35
  )


X <- cbind(
  x1,
  x2
)


colnames(
  X
) <- c(
  "X1",
  "X2"
)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex = 0.6,
  xlab = "X1",
  ylab = "X2",
  main = "Nonlinear Data for Principal Curves"
)


# ------------------------------------------------------------------------------
# 3. Add True Generating Curve
# ------------------------------------------------------------------------------

true_grid <- seq(
  min(latent_t),
  max(latent_t),
  length.out = 300
)


lines(
  2.0 *
    true_grid,
  2.5 *
    sin(
      true_grid
    ),
  lwd = 2
)


# ==============================================================================
# PART II
#
# PCA INITIALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Center Data
# ------------------------------------------------------------------------------

feature_means <- colMeans(
  X
)


X_centered <- sweep(
  X,
  2,
  feature_means,
  "-"
)


# ------------------------------------------------------------------------------
# 5. First Principal Component
# ------------------------------------------------------------------------------

covariance_matrix <- cov(
  X_centered
)


eigen_result <- eigen(
  covariance_matrix,
  symmetric = TRUE
)


PC1_loading <- eigen_result$vectors[
  ,
  1
]


PC1_scores <- as.numeric(
  X_centered %*%
    PC1_loading
)


PC1_loading


# ------------------------------------------------------------------------------
# 6. PCA Line
# ------------------------------------------------------------------------------

lambda_line <- seq(
  min(
    PC1_scores
  ),
  max(
    PC1_scores
  ),
  length.out = 300
)


PCA_line_centered <- outer(
  lambda_line,
  PC1_loading
)


PCA_line <- sweep(
  PCA_line_centered,
  2,
  feature_means,
  "+"
)


plot(
  X,
  pch = 19,
  cex = 0.6,
  xlab = "X1",
  ylab = "X2",
  main = "PCA Initialization"
)


lines(
  PCA_line[
    ,
    1
  ],
  PCA_line[
    ,
    2
  ],
  lwd = 2
)


# ==============================================================================
# PART III
#
# MANUAL LOCAL-LINEAR SMOOTHER
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Tricube Kernel
# ------------------------------------------------------------------------------

tricube <- function(
    u
) {
  
  u <- abs(
    u
  )
  
  
  weights <- (
    1 -
      pmin(
        u,
        1
      )^3
  )^3
  
  
  weights[
    u >= 1
  ] <- 0
  
  
  weights
}


# ------------------------------------------------------------------------------
# 8. Local Linear Prediction at One Point
# ------------------------------------------------------------------------------

local_linear_predict_one <- function(
    x,
    y,
    x0,
    span = 0.30,
    ridge = 1e-8
) {
  
  n <- length(
    x
  )
  
  
  number_neighbors <- max(
    3,
    ceiling(
      span *
        n
    )
  )
  
  
  distances <- abs(
    x -
      x0
  )
  
  
  sorted_distances <- sort(
    distances
  )
  
  
  bandwidth <- sorted_distances[
    min(
      number_neighbors,
      n
    )
  ]
  
  
  if (
    bandwidth <=
    .Machine$double.eps
  ) {
    
    positive_distances <- sorted_distances[
      sorted_distances >
        .Machine$double.eps
    ]
    
    
    if (
      length(
        positive_distances
      ) >
      0
    ) {
      
      bandwidth <- min(
        positive_distances
      )
      
    } else {
      
      return(
        mean(y)
      )
    }
  }
  
  
  weights <- tricube(
    distances /
      bandwidth
  )
  
  
  design <- cbind(
    1,
    x -
      x0
  )
  
  
  weighted_design <- design *
    sqrt(
      weights
    )
  
  
  weighted_y <- y *
    sqrt(
      weights
    )
  
  
  XtX <- crossprod(
    weighted_design
  )
  
  
  XtY <- crossprod(
    weighted_design,
    weighted_y
  )
  
  
  # Penalize only the local slope very slightly for numerical stability.
  
  penalty <- diag(
    c(
      0,
      ridge
    )
  )
  
  
  coefficients <- tryCatch(
    
    solve(
      XtX +
        penalty,
      XtY
    ),
    
    error = function(e) {
      
      c(
        weighted.mean(
          y,
          pmax(
            weights,
            1e-12
          )
        ),
        0
      )
    }
  )
  
  
  as.numeric(
    coefficients[1]
  )
}


# ------------------------------------------------------------------------------
# 9. Local Linear Smoother
# ------------------------------------------------------------------------------

local_linear_predict <- function(
    x,
    y,
    x_new,
    span = 0.30
) {
  
  sapply(
    x_new,
    function(x0) {
      
      local_linear_predict_one(
        x,
        y,
        x0,
        span =
          span
      )
    }
  )
}


# ==============================================================================
# PART IV
#
# CONSTRUCT A CURVE FROM A PROJECTION INDEX
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Smooth Each Coordinate as a Function of Lambda
# ------------------------------------------------------------------------------

fit_curve_given_lambda <- function(
    X,
    lambda,
    span = 0.30,
    grid_size = 400
) {
  
  X <- as.matrix(
    X
  )
  
  
  ordering <- order(
    lambda
  )
  
  
  lambda_sorted <- lambda[
    ordering
  ]
  
  
  X_sorted <- X[
    ordering,
    ,
    drop = FALSE
  ]
  
  
  lambda_grid <- seq(
    min(
      lambda_sorted
    ),
    max(
      lambda_sorted
    ),
    length.out =
      grid_size
  )
  
  
  curve_grid <- matrix(
    NA_real_,
    nrow =
      grid_size,
    ncol =
      ncol(X)
  )
  
  
  for (
    j in seq_len(
      ncol(X)
    )
  ) {
    
    curve_grid[
      ,
      j
    ] <- local_linear_predict(
      lambda_sorted,
      X_sorted[
        ,
        j
      ],
      lambda_grid,
      span =
        span
    )
  }
  
  
  colnames(
    curve_grid
  ) <- colnames(
    X
  )
  
  
  list(
    lambda_grid =
      lambda_grid,
    curve_grid =
      curve_grid
  )
}


# ==============================================================================
# PART V
#
# PROJECT OBSERVATIONS ONTO A DISCRETIZED CURVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Project One Observation onto Curve Segments
# ------------------------------------------------------------------------------

project_point_to_curve <- function(
    x,
    curve_grid,
    lambda_grid
) {
  
  number_segments <- nrow(
    curve_grid
  ) -
    1
  
  
  best_distance_squared <- Inf
  
  best_point <- NULL
  
  best_lambda <- NA_real_
  
  best_segment <- NA_integer_
  
  
  for (
    segment in seq_len(
      number_segments
    )
  ) {
    
    a <- curve_grid[
      segment,
    ]
    
    
    b <- curve_grid[
      segment + 1,
    ]
    
    
    direction <- b -
      a
    
    
    denominator <- sum(
      direction^2
    )
    
    
    if (
      denominator <=
      .Machine$double.eps
    ) {
      
      t_segment <- 0
      
    } else {
      
      t_segment <- sum(
        (
          x -
            a
        ) *
          direction
      ) /
        denominator
      
      
      t_segment <- max(
        0,
        min(
          1,
          t_segment
        )
      )
    }
    
    
    projected_point <- a +
      t_segment *
      direction
    
    
    distance_squared <- sum(
      (
        x -
          projected_point
      )^2
    )
    
    
    if (
      distance_squared <
      best_distance_squared
    ) {
      
      best_distance_squared <-
        distance_squared
      
      
      best_point <-
        projected_point
      
      
      best_lambda <-
        lambda_grid[
          segment
        ] +
        t_segment *
        (
          lambda_grid[
            segment + 1
          ] -
            lambda_grid[
              segment
            ]
        )
      
      
      best_segment <-
        segment
    }
  }
  
  
  list(
    lambda =
      best_lambda,
    projection =
      best_point,
    distance_squared =
      best_distance_squared,
    segment =
      best_segment
  )
}


# ------------------------------------------------------------------------------
# 12. Project Entire Dataset
# ------------------------------------------------------------------------------

project_data_to_curve <- function(
    X,
    curve_grid,
    lambda_grid
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  projected <- matrix(
    NA_real_,
    nrow =
      n,
    ncol =
      ncol(X)
  )
  
  
  lambda_new <- numeric(
    n
  )
  
  
  distance_squared <- numeric(
    n
  )
  
  
  segment <- integer(
    n
  )
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    projection_i <- project_point_to_curve(
      X[
        i,
      ],
      curve_grid,
      lambda_grid
    )
    
    
    lambda_new[i] <-
      projection_i$lambda
    
    
    projected[
      i,
    ] <-
      projection_i$projection
    
    
    distance_squared[i] <-
      projection_i$distance_squared
    
    
    segment[i] <-
      projection_i$segment
  }
  
  
  colnames(
    projected
  ) <- colnames(
    X
  )
  
  
  list(
    lambda =
      lambda_new,
    projected =
      projected,
    distance_squared =
      distance_squared,
    segment =
      segment
  )
}


# ==============================================================================
# PART VI
#
# MANUAL PRINCIPAL CURVE ALGORITHM
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Fit Principal Curve
# ------------------------------------------------------------------------------

fit_principal_curve <- function(
    X,
    span = 0.30,
    grid_size = 400,
    max_iterations = 30,
    tolerance = 1e-4,
    verbose = TRUE
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Initialize using first principal component
  # --------------------------------------------------------------------------
  
  means <- colMeans(
    X
  )
  
  
  X_centered <- sweep(
    X,
    2,
    means,
    "-"
  )
  
  
  covariance_matrix <- cov(
    X_centered
  )
  
  
  eigen_result <- eigen(
    covariance_matrix,
    symmetric = TRUE
  )
  
  
  first_loading <- eigen_result$vectors[
    ,
    1
  ]
  
  
  lambda <- as.numeric(
    X_centered %*%
      first_loading
  )
  
  
  # Give lambda a consistent orientation.
  
  if (
    cor(
      lambda,
      X[
        ,
        1
      ]
    ) <
    0
  ) {
    
    lambda <- -lambda
  }
  
  
  error_history <- numeric(
    max_iterations
  )
  
  
  lambda_change_history <- numeric(
    max_iterations
  )
  
  
  converged <- FALSE
  
  
  previous_error <- Inf
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    # ------------------------------------------------------------------------
    # Smoothing step
    # ------------------------------------------------------------------------
    
    curve_fit <- fit_curve_given_lambda(
      X,
      lambda,
      span =
        span,
      grid_size =
        grid_size
    )
    
    
    # ------------------------------------------------------------------------
    # Projection step
    # ------------------------------------------------------------------------
    
    projection <- project_data_to_curve(
      X,
      curve_fit$curve_grid,
      curve_fit$lambda_grid
    )
    
    
    lambda_new <- projection$lambda
    
    
    # ------------------------------------------------------------------------
    # Reparameterize lambda by rank / normalized position
    #
    # This does not change the ordering of points along the curve and helps
    # prevent arbitrary parameter-scale drift between iterations.
    # ------------------------------------------------------------------------
    
    lambda_old_scaled <- (
      lambda -
        min(lambda)
    ) /
      max(
        max(lambda) -
          min(lambda),
        1e-12
      )
    
    
    lambda_new_scaled <- (
      lambda_new -
        min(lambda_new)
    ) /
      max(
        max(lambda_new) -
          min(lambda_new),
        1e-12
      )
    
    
    lambda_change <- sqrt(
      mean(
        (
          lambda_new_scaled -
            lambda_old_scaled
        )^2
      )
    )
    
    
    reconstruction_error <- mean(
      projection$distance_squared
    )
    
    
    error_history[
      iteration
    ] <- reconstruction_error
    
    
    lambda_change_history[
      iteration
    ] <- lambda_change
    
    
    if (
      verbose
    ) {
      
      cat(
        "Iteration:",
        iteration,
        "| Projection MSE:",
        round(
          reconstruction_error,
          6
        ),
        "| Lambda change:",
        round(
          lambda_change,
          6
        ),
        "\n"
      )
    }
    
    
    lambda <- lambda_new
    
    
    if (
      abs(
        previous_error -
        reconstruction_error
      ) <
      tolerance *
      (
        1 +
        previous_error
      ) &&
      lambda_change <
      sqrt(
        tolerance
      )
    ) {
      
      converged <- TRUE
      
      break
    }
    
    
    previous_error <-
      reconstruction_error
  }
  
  
  iterations_used <-
    iteration
  
  
  error_history <-
    error_history[
      seq_len(
        iterations_used
      )
    ]
  
  
  lambda_change_history <-
    lambda_change_history[
      seq_len(
        iterations_used
      )
    ]
  
  
  # --------------------------------------------------------------------------
  # Final smoothing and projection
  # --------------------------------------------------------------------------
  
  final_curve <- fit_curve_given_lambda(
    X,
    lambda,
    span =
      span,
    grid_size =
      grid_size
  )
  
  
  final_projection <- project_data_to_curve(
    X,
    final_curve$curve_grid,
    final_curve$lambda_grid
  )
  
  
  lambda <- final_projection$lambda
  
  
  list(
    lambda =
      lambda,
    curve =
      final_curve$curve_grid,
    lambda_grid =
      final_curve$lambda_grid,
    projected =
      final_projection$projected,
    distance_squared =
      final_projection$distance_squared,
    projection_MSE =
      mean(
        final_projection$distance_squared
      ),
    error_history =
      error_history,
    lambda_change_history =
      lambda_change_history,
    span =
      span,
    grid_size =
      grid_size,
    iterations =
      iterations_used,
    converged =
      converged,
    initial_loading =
      first_loading
  )
}


# ==============================================================================
# PART VII
#
# FIT PRINCIPAL CURVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Fit
# ------------------------------------------------------------------------------

principal_curve_model <- fit_principal_curve(
  X,
  span = 0.25,
  grid_size = 350,
  max_iterations = 25,
  tolerance = 1e-4,
  verbose = TRUE
)


principal_curve_model$converged


principal_curve_model$iterations


principal_curve_model$projection_MSE


# ==============================================================================
# PART VIII
#
# VISUALIZE FITTED CURVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Principal Curve
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex = 0.6,
  xlab = "X1",
  ylab = "X2",
  main = "Estimated Principal Curve"
)


lines(
  principal_curve_model$curve[
    ,
    1
  ],
  principal_curve_model$curve[
    ,
    2
  ],
  lwd = 3
)


# ------------------------------------------------------------------------------
# 16. Compare with PCA
# ------------------------------------------------------------------------------

lines(
  PCA_line[
    ,
    1
  ],
  PCA_line[
    ,
    2
  ],
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Principal Curve",
    "First Principal Component"
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
# PART IX
#
# SHOW PROJECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Observation-to-Curve Projection
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex = 0.5,
  xlab = "X1",
  ylab = "X2",
  main = "Projection onto Principal Curve"
)


lines(
  principal_curve_model$curve[
    ,
    1
  ],
  principal_curve_model$curve[
    ,
    2
  ],
  lwd = 3
)


# Draw projection segments for a subset to avoid visual clutter.

projection_indices <- seq(
  1,
  n,
  length.out = 40
)


projection_indices <- unique(
  round(
    projection_indices
  )
)


for (
  i in projection_indices
) {
  
  segments(
    X[
      i,
      1
    ],
    X[
      i,
      2
    ],
    principal_curve_model$projected[
      i,
      1
    ],
    principal_curve_model$projected[
      i,
      2
    ]
  )
}


points(
  principal_curve_model$projected[
    projection_indices,
    1
  ],
  principal_curve_model$projected[
    projection_indices,
    2
  ],
  pch = 4
)


# ==============================================================================
# PART X
#
# PROJECTION INDEX
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Inspect Lambda
# ------------------------------------------------------------------------------

summary(
  principal_curve_model$lambda
)


# ------------------------------------------------------------------------------
# 19. Compare Estimated Lambda with True Latent Variable
# ------------------------------------------------------------------------------

cor(
  principal_curve_model$lambda,
  latent_t,
  method = "spearman"
)


# The orientation of lambda is arbitrary, so the correlation can sometimes have
# opposite sign if the curve orientation is reversed.


# ------------------------------------------------------------------------------
# 20. Lambda vs True Latent Coordinate
# ------------------------------------------------------------------------------

plot(
  latent_t,
  principal_curve_model$lambda,
  pch = 19,
  cex = 0.6,
  xlab = "True Latent t",
  ylab = "Estimated Principal-Curve Lambda",
  main = "Recovery of One-Dimensional Ordering"
)


# ==============================================================================
# PART XI
#
# COLOR DATA BY POSITION ALONG CURVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Divide Lambda into Ordered Groups
# ------------------------------------------------------------------------------

lambda_groups <- cut(
  principal_curve_model$lambda,
  breaks = quantile(
    principal_curve_model$lambda,
    probs = seq(
      0,
      1,
      length.out = 6
    )
  ),
  include.lowest = TRUE,
  labels = FALSE
)


plot(
  X,
  pch = lambda_groups,
  cex = 0.7,
  xlab = "X1",
  ylab = "X2",
  main = "Position Along Principal Curve"
)


lines(
  principal_curve_model$curve[
    ,
    1
  ],
  principal_curve_model$curve[
    ,
    2
  ],
  lwd = 3
)


# ==============================================================================
# PART XII
#
# CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Projection Error by Iteration
# ------------------------------------------------------------------------------

plot(
  seq_along(
    principal_curve_model$error_history
  ),
  principal_curve_model$error_history,
  type = "b",
  pch = 19,
  xlab = "Iteration",
  ylab = "Mean Squared Distance to Curve",
  main = "Principal Curve Convergence"
)


# ------------------------------------------------------------------------------
# 23. Lambda Change by Iteration
# ------------------------------------------------------------------------------

plot(
  seq_along(
    principal_curve_model$lambda_change_history
  ),
  principal_curve_model$lambda_change_history,
  type = "b",
  pch = 19,
  xlab = "Iteration",
  ylab = "Change in Projection Index",
  main = "Projection Index Convergence"
)


# ==============================================================================
# PART XIII
#
# PCA RECONSTRUCTION ERROR
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Projection onto First PC
# ------------------------------------------------------------------------------

PCA_projection_centered <- outer(
  PC1_scores,
  PC1_loading
)


PCA_projection <- sweep(
  PCA_projection_centered,
  2,
  feature_means,
  "+"
)


PCA_distance_squared <- rowSums(
  (
    X -
      PCA_projection
  )^2
)


PCA_projection_MSE <- mean(
  PCA_distance_squared
)


PCA_projection_MSE


# ==============================================================================
# PART XIV
#
# PRINCIPAL CURVE VS PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Reconstruction Comparison
# ------------------------------------------------------------------------------

comparison_table <- data.frame(
  Method = c(
    "First Principal Component",
    "Principal Curve"
  ),
  Projection_MSE = c(
    PCA_projection_MSE,
    principal_curve_model$projection_MSE
  )
)


comparison_table


# On genuinely curved data, the nonlinear principal curve should usually achieve
# a smaller reconstruction error than a single straight PCA axis.


# ==============================================================================
# PART XV
#
# SMOOTHING PARAMETER
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Compare Different Spans
# ------------------------------------------------------------------------------

span_values <- c(
  0.15,
  0.25,
  0.40,
  0.60
)


span_models <- vector(
  "list",
  length(
    span_values
  )
)


span_results <- data.frame(
  Span =
    span_values,
  Projection_MSE =
    NA_real_,
  Iterations =
    NA_integer_
)


for (
  i in seq_along(
    span_values
  )
) {
  
  span_models[[i]] <- fit_principal_curve(
    X,
    span =
      span_values[i],
    grid_size = 300,
    max_iterations = 20,
    tolerance = 1e-4,
    verbose = FALSE
  )
  
  
  span_results$Projection_MSE[i] <-
    span_models[[i]]$projection_MSE
  
  
  span_results$Iterations[i] <-
    span_models[[i]]$iterations
}


span_results


# ------------------------------------------------------------------------------
# 27. Overlay Different Smoothness Levels
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex = 0.4,
  xlab = "X1",
  ylab = "X2",
  main = "Principal Curve Smoothing"
)


for (
  i in seq_along(
    span_models
  )
) {
  
  lines(
    span_models[[i]]$curve[
      ,
      1
    ],
    span_models[[i]]$curve[
      ,
      2
    ],
    lty = i,
    lwd = 2
  )
}


legend(
  "topleft",
  legend = paste(
    "Span =",
    span_values
  ),
  lty = seq_along(
    span_values
  ),
  lwd = 2
)


# ==============================================================================
# PART XVI
#
# OVERFITTING WITH TOO LITTLE SMOOTHING
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Very Flexible Curve
# ------------------------------------------------------------------------------

flexible_curve <- fit_principal_curve(
  X,
  span = 0.08,
  grid_size = 400,
  max_iterations = 20,
  verbose = FALSE
)


# ------------------------------------------------------------------------------
# 29. Very Smooth Curve
# ------------------------------------------------------------------------------

smooth_curve <- fit_principal_curve(
  X,
  span = 0.70,
  grid_size = 400,
  max_iterations = 20,
  verbose = FALSE
)


plot(
  X,
  pch = 19,
  cex = 0.4,
  xlab = "X1",
  ylab = "X2",
  main = "Principal Curve Bias-Variance Tradeoff"
)


lines(
  flexible_curve$curve[
    ,
    1
  ],
  flexible_curve$curve[
    ,
    2
  ],
  lwd = 2
)


lines(
  smooth_curve$curve[
    ,
    1
  ],
  smooth_curve$curve[
    ,
    2
  ],
  lwd = 2,
  lty = 2
)


legend(
  "topleft",
  legend = c(
    "Flexible",
    "Smooth"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART XVII
#
# TRAIN / TEST ASSESSMENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Split Data
# ------------------------------------------------------------------------------

set.seed(456)


train_indices <- sample(
  seq_len(n),
  size = floor(
    0.70 *
      n
  )
)


test_indices <- setdiff(
  seq_len(n),
  train_indices
)


X_train <- X[
  train_indices,
  ,
  drop = FALSE
]


X_test <- X[
  test_indices,
  ,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# 31. Fit on Training Data
# ------------------------------------------------------------------------------

principal_curve_train <- fit_principal_curve(
  X_train,
  span = 0.25,
  grid_size = 350,
  max_iterations = 25,
  verbose = FALSE
)


# ------------------------------------------------------------------------------
# 32. Project Test Observations onto Learned Curve
# ------------------------------------------------------------------------------

test_projection <- project_data_to_curve(
  X_test,
  principal_curve_train$curve,
  principal_curve_train$lambda_grid
)


train_projection_MSE <-
  principal_curve_train$projection_MSE


test_projection_MSE <- mean(
  test_projection$distance_squared
)


data.frame(
  Dataset = c(
    "Training",
    "Test"
  ),
  Projection_MSE = c(
    train_projection_MSE,
    test_projection_MSE
  )
)


# ==============================================================================
# PART XVIII
#
# CHOOSE SPAN USING TEST-INDEPENDENT VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Train / Validation / Test Split
# ------------------------------------------------------------------------------

set.seed(789)


all_indices <- sample(
  seq_len(n)
)


n_train <- floor(
  0.60 *
    n
)


n_validation <- floor(
  0.20 *
    n
)


train_idx <- all_indices[
  seq_len(
    n_train
  )
]


validation_idx <- all_indices[
  (
    n_train + 1
  ):(
    n_train +
      n_validation
  )
]


test_idx <- all_indices[
  (
    n_train +
      n_validation +
      1
  ):n
]


X_train2 <- X[
  train_idx,
  ,
  drop = FALSE
]


X_validation <- X[
  validation_idx,
  ,
  drop = FALSE
]


X_test2 <- X[
  test_idx,
  ,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# 34. Validation Search
# ------------------------------------------------------------------------------

candidate_spans <- c(
  0.12,
  0.18,
  0.25,
  0.35,
  0.50,
  0.70
)


validation_error <- numeric(
  length(
    candidate_spans
  )
)


for (
  i in seq_along(
    candidate_spans
  )
) {
  
  model_i <- fit_principal_curve(
    X_train2,
    span =
      candidate_spans[i],
    grid_size = 300,
    max_iterations = 20,
    verbose = FALSE
  )
  
  
  projection_i <- project_data_to_curve(
    X_validation,
    model_i$curve,
    model_i$lambda_grid
  )
  
  
  validation_error[i] <- mean(
    projection_i$distance_squared
  )
}


validation_table <- data.frame(
  Span =
    candidate_spans,
  Validation_MSE =
    validation_error
)


validation_table


best_span <- candidate_spans[
  which.min(
    validation_error
  )
]


best_span


# ------------------------------------------------------------------------------
# 35. Refit on Training + Validation
# ------------------------------------------------------------------------------

X_train_validation <- rbind(
  X_train2,
  X_validation
)


final_curve <- fit_principal_curve(
  X_train_validation,
  span =
    best_span,
  grid_size = 400,
  max_iterations = 25,
  verbose = FALSE
)


# ------------------------------------------------------------------------------
# 36. Final Test Error
# ------------------------------------------------------------------------------

final_test_projection <- project_data_to_curve(
  X_test2,
  final_curve$curve,
  final_curve$lambda_grid
)


final_test_MSE <- mean(
  final_test_projection$distance_squared
)


final_test_MSE


# ==============================================================================
# PART XIX
#
# THREE-DIMENSIONAL PRINCIPAL CURVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Generate a 3D Helix-Like Dataset
# ------------------------------------------------------------------------------

set.seed(999)


n3 <- 350


t3 <- runif(
  n3,
  0,
  2.5 *
    pi
)


X3 <- cbind(
  cos(t3) +
    rnorm(
      n3,
      sd = 0.12
    ),
  sin(t3) +
    rnorm(
      n3,
      sd = 0.12
    ),
  0.35 *
    t3 +
    rnorm(
      n3,
      sd = 0.12
    )
)


colnames(
  X3
) <- c(
  "X1",
  "X2",
  "X3"
)


# ------------------------------------------------------------------------------
# 38. Fit Principal Curve in 3D
# ------------------------------------------------------------------------------

principal_curve_3D <- fit_principal_curve(
  X3,
  span = 0.20,
  grid_size = 350,
  max_iterations = 25,
  verbose = FALSE
)


principal_curve_3D$projection_MSE


# ------------------------------------------------------------------------------
# 39. Inspect Pairwise Views
# ------------------------------------------------------------------------------

pairs(
  X3,
  main = "Three-Dimensional Curved Data"
)


# X1 vs X2

plot(
  X3[
    ,
    1
  ],
  X3[
    ,
    2
  ],
  pch = 19,
  cex = 0.4,
  xlab = "X1",
  ylab = "X2",
  main = "3D Principal Curve: X1 vs X2"
)


lines(
  principal_curve_3D$curve[
    ,
    1
  ],
  principal_curve_3D$curve[
    ,
    2
  ],
  lwd = 3
)


# X1 vs X3

plot(
  X3[
    ,
    1
  ],
  X3[
    ,
    3
  ],
  pch = 19,
  cex = 0.4,
  xlab = "X1",
  ylab = "X3",
  main = "3D Principal Curve: X1 vs X3"
)


lines(
  principal_curve_3D$curve[
    ,
    1
  ],
  principal_curve_3D$curve[
    ,
    3
  ],
  lwd = 3
)


# X2 vs X3

plot(
  X3[
    ,
    2
  ],
  X3[
    ,
    3
  ],
  pch = 19,
  cex = 0.4,
  xlab = "X2",
  ylab = "X3",
  main = "3D Principal Curve: X2 vs X3"
)


lines(
  principal_curve_3D$curve[
    ,
    2
  ],
  principal_curve_3D$curve[
    ,
    3
  ],
  lwd = 3
)


# ==============================================================================
# PART XX
#
# PRINCIPAL CURVE AS A ONE-DIMENSIONAL REPRESENTATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Sort Observations Along Curve
# ------------------------------------------------------------------------------

curve_order <- order(
  principal_curve_model$lambda
)


ordered_data <- data.frame(
  Lambda =
    principal_curve_model$lambda[
      curve_order
    ],
  X1 =
    X[
      curve_order,
      1
    ],
  X2 =
    X[
      curve_order,
      2
    ]
)


head(
  ordered_data,
  20
)


# lambda acts as a one-dimensional coordinate for the observations.


# ==============================================================================
# PART XXI
#
# RECONSTRUCTION FROM LAMBDA
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Reconstruct Observations from Curve Projection
# ------------------------------------------------------------------------------

X_curve_reconstructed <-
  principal_curve_model$projected


head(
  X_curve_reconstructed
)


# ------------------------------------------------------------------------------
# 42. Reconstruction MSE
# ------------------------------------------------------------------------------

mean(
  rowSums(
    (
      X -
        X_curve_reconstructed
    )^2
  )
)


# ==============================================================================
# PART XXII
#
# DISTANCE-TO-CURVE AS AN OUTLIER MEASURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Largest Projection Distances
# ------------------------------------------------------------------------------

distance_to_curve <- sqrt(
  principal_curve_model$distance_squared
)


outlier_order <- order(
  distance_to_curve,
  decreasing = TRUE
)


data.frame(
  Observation =
    outlier_order[
      1:10
    ],
  X[
    outlier_order[
      1:10
    ],
    ,
    drop = FALSE
  ],
  Lambda =
    principal_curve_model$lambda[
      outlier_order[
        1:10
      ]
    ],
  Distance_to_Curve =
    distance_to_curve[
      outlier_order[
        1:10
      ]
    ]
)


# ------------------------------------------------------------------------------
# 44. Highlight Large-Distance Points
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex = 0.5,
  xlab = "X1",
  ylab = "X2",
  main = "Distance from Principal Curve"
)


lines(
  principal_curve_model$curve[
    ,
    1
  ],
  principal_curve_model$curve[
    ,
    2
  ],
  lwd = 3
)


points(
  X[
    outlier_order[
      1:10
    ],
    ,
    drop = FALSE
  ],
  pch = 8,
  cex = 1.5
)


# ==============================================================================
# PART XXIII
#
# LINEAR DATA: PRINCIPAL CURVE APPROACHES PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Generate Linear Data
# ------------------------------------------------------------------------------

set.seed(111)


n_linear <- 300


linear_x1 <- rnorm(
  n_linear
)


linear_x2 <- 2.5 *
  linear_x1 +
  rnorm(
    n_linear,
    sd = 0.40
  )


X_linear <- cbind(
  linear_x1,
  linear_x2
)


colnames(
  X_linear
) <- c(
  "X1",
  "X2"
)


# ------------------------------------------------------------------------------
# 46. Fit Principal Curve
# ------------------------------------------------------------------------------

linear_curve <- fit_principal_curve(
  X_linear,
  span = 0.40,
  grid_size = 300,
  max_iterations = 20,
  verbose = FALSE
)


# ------------------------------------------------------------------------------
# 47. Fit PCA Line
# ------------------------------------------------------------------------------

linear_mean <- colMeans(
  X_linear
)


linear_centered <- sweep(
  X_linear,
  2,
  linear_mean,
  "-"
)


linear_eigen <- eigen(
  cov(
    linear_centered
  ),
  symmetric = TRUE
)


linear_loading <- linear_eigen$vectors[
  ,
  1
]


linear_scores <- as.numeric(
  linear_centered %*%
    linear_loading
)


linear_grid <- seq(
  min(
    linear_scores
  ),
  max(
    linear_scores
  ),
  length.out = 300
)


linear_PCA_line <- sweep(
  outer(
    linear_grid,
    linear_loading
  ),
  2,
  linear_mean,
  "+"
)


# ------------------------------------------------------------------------------
# 48. Compare
# ------------------------------------------------------------------------------

plot(
  X_linear,
  pch = 19,
  cex = 0.5,
  xlab = "X1",
  ylab = "X2",
  main = "Principal Curve on Approximately Linear Data"
)


lines(
  linear_curve$curve[
    ,
    1
  ],
  linear_curve$curve[
    ,
    2
  ],
  lwd = 3
)


lines(
  linear_PCA_line[
    ,
    1
  ],
  linear_PCA_line[
    ,
    2
  ],
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Principal Curve",
    "PC1"
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
# PART XXIV
#
# CAVEAT: BRANCHING STRUCTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Generate Y-Shaped Data
# ------------------------------------------------------------------------------

set.seed(222)


n_branch <- 300


branch <- sample(
  1:3,
  n_branch,
  replace = TRUE
)


radius <- runif(
  n_branch,
  0,
  3
)


angles <- c(
  0,
  2 * pi / 3,
  4 * pi / 3
)


X_branch <- cbind(
  radius *
    cos(
      angles[
        branch
      ]
    ) +
    rnorm(
      n_branch,
      sd = 0.12
    ),
  radius *
    sin(
      angles[
        branch
      ]
    ) +
    rnorm(
      n_branch,
      sd = 0.12
    )
)


colnames(
  X_branch
) <- c(
  "X1",
  "X2"
)


plot(
  X_branch,
  pch = 19,
  cex = 0.5,
  xlab = "X1",
  ylab = "X2",
  main = "Branching Data"
)


# A single principal curve is fundamentally a one-dimensional non-branching
# object. It cannot naturally represent a Y-shaped topology with three branches.


# ==============================================================================
# PART XXV
#
# OPTIONAL VERIFICATION WITH princurve
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Compare with princurve Package if Installed
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "princurve",
    quietly = TRUE
  )
) {
  
  package_curve <- princurve::principal_curve(
    X
  )
  
  
  plot(
    X,
    pch = 19,
    cex = 0.5,
    xlab = "X1",
    ylab = "X2",
    main = "Manual vs princurve Implementation"
  )
  
  
  lines(
    principal_curve_model$curve[
      ,
      1
    ],
    principal_curve_model$curve[
      ,
      2
    ],
    lwd = 3
  )
  
  
  package_order <- order(
    package_curve$lambda
  )
  
  
  lines(
    package_curve$s[
      package_order,
      1
    ],
    package_curve$s[
      package_order,
      2
    ],
    lwd = 2,
    lty = 2
  )
  
  
  legend(
    "topleft",
    legend = c(
      "Manual",
      "princurve"
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
  
  
  cat(
    "Manual projection MSE:",
    principal_curve_model$projection_MSE,
    "\n"
  )
  
  
  cat(
    "princurve average squared distance:",
    mean(
      rowSums(
        (
          X -
            package_curve$s
        )^2
      )
    ),
    "\n"
  )
}


# ==============================================================================
# PART XXVI
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Output
# ------------------------------------------------------------------------------

cat(
  "Principal Curves Summary\n"
)


cat(
  "------------------------\n"
)


cat(
  "Observations:",
  n,
  "\n"
)


cat(
  "Dimensions:",
  ncol(X),
  "\n"
)


cat(
  "Smoothing span:",
  principal_curve_model$span,
  "\n"
)


cat(
  "Iterations:",
  principal_curve_model$iterations,
  "\n"
)


cat(
  "Converged:",
  principal_curve_model$converged,
  "\n"
)


cat(
  "PCA projection MSE:",
  round(
    PCA_projection_MSE,
    5
  ),
  "\n"
)


cat(
  "Principal curve projection MSE:",
  round(
    principal_curve_model$projection_MSE,
    5
  ),
  "\n"
)


cat(
  "Spearman correlation between true latent t and estimated lambda:",
  round(
    cor(
      latent_t,
      principal_curve_model$lambda,
      method = "spearman"
    ),
    4
  ),
  "\n"
)


cat(
  "Validation-selected span:",
  best_span,
  "\n"
)


cat(
  "Final held-out projection MSE:",
  round(
    final_test_MSE,
    5
  ),
  "\n"
)