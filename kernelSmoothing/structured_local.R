# ==============================================================================
# Structured Local Regression
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   Chapter 6: Kernel Smoothing Methods
#
# Main ideas:
#   - Multivariate local regression
#   - Curse of dimensionality
#   - Structured kernels
#   - Mahalanobis / weighted distance metrics
#   - Predictor relevance through kernel weights
#   - Additive regression functions
#   - Backfitting
#   - One-dimensional local smoothers inside a multivariate model
#   - Cross-validation
#
# Two approaches:
#
#   1. Structured kernel:
#
#      d_A(x, x0)^2 =
#
#          (x - x0)' A (x - x0)
#
#
#   2. Structured regression function:
#
#      f(x1, ..., xp)
#      =
#      alpha + g1(x1) + ... + gp(xp)
#
# ==============================================================================
#
#
# ------------------------------------------------------------------------------
# 1. Generate Multivariate Nonlinear Data
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


# x1 and x2 matter strongly.
#
# x3 is deliberately irrelevant.
#
# This makes the structured-kernel idea easy to demonstrate.


true_function <- function(
    x1,
    x2,
    x3 = 0
) {
  
  2 +
    2 * sin(x1) +
    0.75 * x2^2
}


y_true <- true_function(
  x1,
  x2,
  x3
)


sigma <- 1


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
  )


X <- cbind(
  x1 = x1,
  x2 = x2,
  x3 = x3
)


data <- data.frame(
  x1 = x1,
  x2 = x2,
  x3 = x3,
  y = y
)


head(data)


# ------------------------------------------------------------------------------
# 2. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data,
  main = "Structured Local Regression Data"
)


par(
  mfrow = c(
    1,
    3
  )
)


plot(
  x1,
  y,
  pch = 19,
  xlab = "x1",
  ylab = "y",
  main = "Response vs x1"
)


plot(
  x2,
  y,
  pch = 19,
  xlab = "x2",
  ylab = "y",
  main = "Response vs x2"
)


plot(
  x3,
  y,
  pch = 19,
  xlab = "x3",
  ylab = "y",
  main = "Response vs x3"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 3. Standardize Predictors
# ------------------------------------------------------------------------------

# Distance-based methods are sensitive to measurement scale.
#
# Standardization ensures that a variable measured in large units does
# not dominate purely because of its numerical scale.


X_means <- colMeans(
  X
)


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


X_scaled <- as.matrix(
  X_scaled
)


# ------------------------------------------------------------------------------
# 4. Tricube Kernel
# ------------------------------------------------------------------------------

tricube <- function(u) {
  
  result <- (
    1 -
      abs(u)^3
  )^3
  
  
  result[
    abs(u) >= 1
  ] <- 0
  
  
  return(
    result
  )
}


# ------------------------------------------------------------------------------
# 5. Structured Distance
# ------------------------------------------------------------------------------

# Standard Euclidean distance corresponds to:
#
# A = I
#
#
# Structured distance:
#
# d_A(x, x0)
# =
# sqrt(
#   (x - x0)' A (x - x0)
# )
#
#
# A must be positive semidefinite.


structured_distance <- function(
    X,
    x0,
    A
) {
  
  X <- as.matrix(
    X
  )
  
  
  x0 <- as.numeric(
    x0
  )
  
  
  differences <- sweep(
    X,
    MARGIN = 2,
    STATS = x0,
    FUN = "-"
  )
  
  
  squared_distance <- rowSums(
    (
      differences %*%
        A
    ) *
      differences
  )
  
  
  squared_distance <- pmax(
    squared_distance,
    0
  )
  
  
  sqrt(
    squared_distance
  )
}


# ------------------------------------------------------------------------------
# 6. Standard Spherical Kernel
# ------------------------------------------------------------------------------

# Equal importance for every standardized predictor.

A_spherical <- diag(
  ncol(
    X_scaled
  )
)


A_spherical


# ------------------------------------------------------------------------------
# 7. Structured Kernel
# ------------------------------------------------------------------------------

# We know from simulation that:
#
# x1 = highly important
# x2 = important
# x3 = irrelevant
#
# So define a metric that strongly downweights x3.


A_structured <- diag(
  c(
    1,
    1,
    0.05
  )
)


A_structured


# ------------------------------------------------------------------------------
# 8. Extreme Structured Kernel
# ------------------------------------------------------------------------------

# Setting the x3 weight exactly equal to zero removes x3 from the
# neighborhood calculation.


A_ignore_x3 <- diag(
  c(
    1,
    1,
    0
  )
)


# ------------------------------------------------------------------------------
# 9. Visualize the Effect of the Metric
# ------------------------------------------------------------------------------

x0 <- c(
  0,
  0,
  0
)


distance_spherical <- structured_distance(
  X_scaled,
  x0,
  A_spherical
)


distance_structured <- structured_distance(
  X_scaled,
  x0,
  A_structured
)


plot(
  distance_spherical,
  distance_structured,
  pch = 19,
  xlab = "Spherical Distance",
  ylab = "Structured Distance",
  main = "Effect of Structured Kernel"
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# ------------------------------------------------------------------------------
# 10. Multivariate Local Linear Regression
# ------------------------------------------------------------------------------

# At x0 fit:
#
# y_i =
#
# beta0 +
# beta1(x_i1 - x01) +
# ...
# beta_p(x_ip - x0p)
#
# using kernel weights based on structured distance.
#
# The prediction at x0 is beta0.


structured_local_linear <- function(
    X,
    y,
    x0,
    span = 0.3,
    A = diag(
      ncol(
        X
      )
    )
) {
  
  X <- as.matrix(
    X
  )
  
  
  y <- as.numeric(
    y
  )
  
  
  x0 <- as.numeric(
    x0
  )
  
  
  n <- nrow(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  number_neighbors <- max(
    p + 2,
    ceiling(
      span *
        n
    )
  )
  
  
  distances <- structured_distance(
    X,
    x0,
    A
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
    bandwidth <= 0
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
  
  
  weights <- tricube(
    distances /
      bandwidth
  )
  
  
  centered_X <- sweep(
    X,
    MARGIN = 2,
    STATS = x0,
    FUN = "-"
  )
  
  
  X_local <- cbind(
    Intercept = 1,
    centered_X
  )
  
  
  sqrt_weights <- sqrt(
    weights
  )
  
  
  X_weighted <- X_local *
    sqrt_weights
  
  
  y_weighted <- y *
    sqrt_weights
  
  
  beta_hat <- tryCatch(
    
    qr.solve(
      X_weighted,
      y_weighted
    ),
    
    error = function(e) {
      
      # Mild ridge stabilization
      
      solve(
        crossprod(
          X_weighted
        ) +
          1e-8 *
          diag(
            ncol(
              X_weighted
            )
          ),
        crossprod(
          X_weighted,
          y_weighted
        )
      )
    }
  )
  
  
  return(
    as.numeric(
      beta_hat[1]
    )
  )
}


# ------------------------------------------------------------------------------
# 11. Prediction Function
# ------------------------------------------------------------------------------

structured_local_predict <- function(
    X_train,
    y_train,
    X_new,
    span = 0.3,
    A = diag(
      ncol(
        X_train
      )
    )
) {
  
  X_train <- as.matrix(
    X_train
  )
  
  
  X_new <- as.matrix(
    X_new
  )
  
  
  prediction <- numeric(
    nrow(
      X_new
    )
  )
  
  
  for (
    i in seq_len(
      nrow(
        X_new
      )
    )
  ) {
    
    prediction[i] <- structured_local_linear(
      X = X_train,
      y = y_train,
      x0 = X_new[
        i,
        ,
        drop = TRUE
      ],
      span = span,
      A = A
    )
  }
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 12. Fit Spherical Local Regression
# ------------------------------------------------------------------------------

prediction_spherical <- structured_local_predict(
  X_train = X_scaled,
  y_train = y,
  X_new = X_scaled,
  span = 0.3,
  A = A_spherical
)


MSE_spherical <- mean(
  (
    y -
      prediction_spherical
  )^2
)


MSE_spherical


# ------------------------------------------------------------------------------
# 13. Fit Structured Local Regression
# ------------------------------------------------------------------------------

prediction_structured <- structured_local_predict(
  X_train = X_scaled,
  y_train = y,
  X_new = X_scaled,
  span = 0.3,
  A = A_structured
)


MSE_structured <- mean(
  (
    y -
      prediction_structured
  )^2
)


MSE_structured


# ------------------------------------------------------------------------------
# 14. Ignore Irrelevant Variable
# ------------------------------------------------------------------------------

prediction_ignore_x3 <- structured_local_predict(
  X_train = X_scaled,
  y_train = y,
  X_new = X_scaled,
  span = 0.3,
  A = A_ignore_x3
)


MSE_ignore_x3 <- mean(
  (
    y -
      prediction_ignore_x3
  )^2
)


data.frame(
  Kernel = c(
    "Spherical",
    "Downweight x3",
    "Ignore x3"
  ),
  Training_MSE = c(
    MSE_spherical,
    MSE_structured,
    MSE_ignore_x3
  )
)


# ------------------------------------------------------------------------------
# 15. Kernel Weight Interpretation
# ------------------------------------------------------------------------------

# Consider one target observation.

target <- 100


target_point <- X_scaled[
  target,
  ,
  drop = TRUE
]


distance_ordinary <- structured_distance(
  X_scaled,
  target_point,
  A_spherical
)


distance_weighted <- structured_distance(
  X_scaled,
  target_point,
  A_structured
)


# ------------------------------------------------------------------------------
# 16. Plot Neighborhood Changes
# ------------------------------------------------------------------------------

plot(
  distance_ordinary,
  distance_weighted,
  pch = 19,
  xlab = "Ordinary Distance",
  ylab = "Structured Distance",
  main = "Structured Neighborhood Geometry"
)


abline(
  0,
  1,
  lty = 2
)


# ------------------------------------------------------------------------------
# 17. Diagonal Structured Kernels
# ------------------------------------------------------------------------------

# If:
#
# A = diag(a1, a2, ..., ap)
#
# then:
#
# distance^2 =
#
# a1(x1-x01)^2 +
# ...
# ap(xp-x0p)^2
#
#
# Large a_j:
#
# differences in variable j increase distance strongly.
#
#
# Small a_j:
#
# variable j matters less for neighborhood selection.


# ------------------------------------------------------------------------------
# 18. Explore the Weight Assigned to x3
# ------------------------------------------------------------------------------

x3_weights <- c(
  0,
  0.01,
  0.05,
  0.1,
  0.25,
  0.5,
  1,
  2
)


x3_weight_MSE <- numeric(
  length(
    x3_weights
  )
)


for (
  i in seq_along(
    x3_weights
  )
) {
  
  A_i <- diag(
    c(
      1,
      1,
      x3_weights[i]
    )
  )
  
  
  prediction_i <- structured_local_predict(
    X_train = X_scaled,
    y_train = y,
    X_new = X_scaled,
    span = 0.3,
    A = A_i
  )
  
  
  x3_weight_MSE[i] <- mean(
    (
      y -
        prediction_i
    )^2
  )
}


weight_results <- data.frame(
  x3_Weight = x3_weights,
  Training_MSE = x3_weight_MSE
)


weight_results


# ------------------------------------------------------------------------------
# 19. Plot Metric Weight vs Error
# ------------------------------------------------------------------------------

plot(
  x3_weights,
  x3_weight_MSE,
  type = "b",
  pch = 19,
  xlab = "Kernel Weight for x3",
  ylab = "Training MSE",
  main = "Effect of an Irrelevant Predictor"
)


# ==============================================================================
# PART II
#
# STRUCTURED REGRESSION FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Additive Model Structure
# ------------------------------------------------------------------------------

# Instead of estimating an unrestricted:
#
# f(x1, x2, x3)
#
# assume:
#
# f(x1, x2, x3)
#
# =
#
# alpha +
# g1(x1) +
# g2(x2) +
# g3(x3)
#
#
# This eliminates interaction terms and avoids fitting a full
# three-dimensional smoother.


# ------------------------------------------------------------------------------
# 21. One-Dimensional Tricube Local Linear Smoother
# ------------------------------------------------------------------------------

local_linear_1d <- function(
    x,
    y,
    x_new,
    span = 0.3
) {
  
  predictions <- numeric(
    length(
      x_new
    )
  )
  
  
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
  
  
  for (
    m in seq_along(
      x_new
    )
  ) {
    
    x0 <- x_new[m]
    
    
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
      bandwidth <= 0
    ) {
      
      positive_distances <- distances[
        distances > 0
      ]
      
      
      if (
        length(
          positive_distances
        ) == 0
      ) {
        
        predictions[m] <- mean(
          y
        )
        
        
        next
      }
      
      
      bandwidth <- min(
        positive_distances
      )
    }
    
    
    weights <- tricube(
      distances /
        bandwidth
    )
    
    
    centered_x <- x -
      x0
    
    
    X_local <- cbind(
      1,
      centered_x
    )
    
    
    sqrt_weights <- sqrt(
      weights
    )
    
    
    X_weighted <- X_local *
      sqrt_weights
    
    
    y_weighted <- y *
      sqrt_weights
    
    
    beta_hat <- tryCatch(
      
      qr.solve(
        X_weighted,
        y_weighted
      ),
      
      error = function(e) {
        
        c(
          weighted.mean(
            y,
            weights
          ),
          0
        )
      }
    )
    
    
    predictions[m] <- beta_hat[1]
  }
  
  
  return(
    predictions
  )
}


# ------------------------------------------------------------------------------
# 22. Additive Backfitting Algorithm
# ------------------------------------------------------------------------------

# For:
#
# y =
# alpha + g1(x1) + g2(x2) + ... + gp(xp) + error
#
#
# update one function at a time:
#
# residual for variable j =
#
# y - alpha - sum_{k != j} g_k(x_k)
#
#
# Then smooth this partial residual against x_j.


additive_backfit <- function(
    X,
    y,
    span = 0.3,
    max_iter = 100,
    tol = 1e-6
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
  ) <- colnames(
    X
  )
  
  
  change_history <- numeric(
    max_iter
  )
  
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    old_components <- components
    
    
    for (
      j in seq_len(
        p
      )
    ) {
      
      other_components <- rowSums(
        components[
          ,
          -j,
          drop = FALSE
        ]
      )
      
      
      partial_residual <- y -
        alpha -
        other_components
      
      
      component_j <- local_linear_1d(
        x = X[
          ,
          j
        ],
        y = partial_residual,
        x_new = X[
          ,
          j
        ],
        span = span
      )
      
      
      # Identifiability constraint:
      #
      # each g_j has mean zero.
      
      component_j <- component_j -
        mean(
          component_j
        )
      
      
      components[
        ,
        j
      ] <- component_j
    }
    
    
    change <- max(
      abs(
        components -
          old_components
      )
    )
    
    
    change_history[
      iteration
    ] <- change
    
    
    if (
      change <
      tol
    ) {
      
      break
    }
  }
  
  
  fitted <- alpha +
    rowSums(
      components
    )
  
  
  return(
    list(
      intercept = alpha,
      components = components,
      fitted = fitted,
      iterations = iteration,
      change_history = change_history[
        seq_len(
          iteration
        )
      ],
      span = span
    )
  )
}


# ------------------------------------------------------------------------------
# 23. Fit Additive Structured Local Regression
# ------------------------------------------------------------------------------

additive_model <- additive_backfit(
  X = X,
  y = y,
  span = 0.3
)


additive_model$iterations


# ------------------------------------------------------------------------------
# 24. Backfitting Convergence
# ------------------------------------------------------------------------------

plot(
  seq_along(
    additive_model$change_history
  ),
  additive_model$change_history,
  type = "l",
  xlab = "Iteration",
  ylab = "Maximum Component Change",
  main = "Backfitting Convergence"
)


# ------------------------------------------------------------------------------
# 25. Training Error
# ------------------------------------------------------------------------------

additive_MSE <- mean(
  (
    y -
      additive_model$fitted
  )^2
)


additive_MSE


# ------------------------------------------------------------------------------
# 26. Plot Estimated Component g1(x1)
# ------------------------------------------------------------------------------

order_x1 <- order(
  x1
)


plot(
  x1[
    order_x1
  ],
  additive_model$components[
    order_x1,
    1
  ],
  type = "l",
  lwd = 2,
  xlab = "x1",
  ylab = "g1(x1)",
  main = "Estimated Effect of x1"
)


# True centered component

true_g1 <- 2 *
  sin(x1)


true_g1 <- true_g1 -
  mean(
    true_g1
  )


lines(
  x1[
    order_x1
  ],
  true_g1[
    order_x1
  ],
  lty = 2,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Estimated",
    "True"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 27. Plot Estimated Component g2(x2)
# ------------------------------------------------------------------------------

order_x2 <- order(
  x2
)


plot(
  x2[
    order_x2
  ],
  additive_model$components[
    order_x2,
    2
  ],
  type = "l",
  lwd = 2,
  xlab = "x2",
  ylab = "g2(x2)",
  main = "Estimated Effect of x2"
)


true_g2 <- 0.75 *
  x2^2


true_g2 <- true_g2 -
  mean(
    true_g2
  )


lines(
  x2[
    order_x2
  ],
  true_g2[
    order_x2
  ],
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 28. Irrelevant Predictor Component
# ------------------------------------------------------------------------------

order_x3 <- order(
  x3
)


plot(
  x3[
    order_x3
  ],
  additive_model$components[
    order_x3,
    3
  ],
  type = "l",
  lwd = 2,
  xlab = "x3",
  ylab = "g3(x3)",
  main = "Estimated Effect of Irrelevant x3"
)


abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 29. Compare Models
# ------------------------------------------------------------------------------

global_linear <- lm(
  y ~ x1 + x2 + x3
)


linear_prediction <- predict(
  global_linear
)


linear_MSE <- mean(
  (
    y -
      linear_prediction
  )^2
)


comparison <- data.frame(
  Model = c(
    "Global Linear",
    "3D Spherical Local",
    "Structured Kernel",
    "Ignore x3 Kernel",
    "Additive Local Regression"
  ),
  Training_MSE = c(
    linear_MSE,
    MSE_spherical,
    MSE_structured,
    MSE_ignore_x3,
    additive_MSE
  )
)


comparison


# ------------------------------------------------------------------------------
# 30. Additive Prediction for New Data
# ------------------------------------------------------------------------------

# For new observations, each fitted component must be smoothed/predicted
# separately.
#
# To make prediction easier, save the partial-residual training targets
# for each component.


additive_fit_with_targets <- function(
    X,
    y,
    span = 0.3,
    max_iter = 100,
    tol = 1e-6
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
  
  
  alpha <- mean(
    y
  )
  
  
  components <- matrix(
    0,
    nrow = n,
    ncol = p
  )
  
  
  component_targets <- matrix(
    0,
    nrow = n,
    ncol = p
  )
  
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    old_components <- components
    
    
    for (
      j in seq_len(
        p
      )
    ) {
      
      other_components <- rowSums(
        components[
          ,
          -j,
          drop = FALSE
        ]
      )
      
      
      partial_residual <- y -
        alpha -
        other_components
      
      
      component_targets[
        ,
        j
      ] <- partial_residual
      
      
      component_j <- local_linear_1d(
        X[
          ,
          j
        ],
        partial_residual,
        X[
          ,
          j
        ],
        span
      )
      
      
      component_j <- component_j -
        mean(
          component_j
        )
      
      
      components[
        ,
        j
      ] <- component_j
    }
    
    
    if (
      max(
        abs(
          components -
          old_components
        )
      ) <
      tol
    ) {
      
      break
    }
  }
  
  
  return(
    list(
      intercept = alpha,
      X = X,
      y = y,
      components = components,
      targets = component_targets,
      component_means = colMeans(
        components
      ),
      span = span
    )
  )
}


# ------------------------------------------------------------------------------
# 31. Additive Prediction Function
# ------------------------------------------------------------------------------

additive_predict <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  p <- ncol(
    model$X
  )
  
  
  prediction_components <- matrix(
    0,
    nrow = nrow(
      X_new
    ),
    ncol = p
  )
  
  
  for (
    j in seq_len(
      p
    )
  ) {
    
    component_prediction <- local_linear_1d(
      x = model$X[
        ,
        j
      ],
      y = model$targets[
        ,
        j
      ],
      x_new = X_new[
        ,
        j
      ],
      span = model$span
    )
    
    
    # Center using the training component estimate.
    
    training_component <- local_linear_1d(
      x = model$X[
        ,
        j
      ],
      y = model$targets[
        ,
        j
      ],
      x_new = model$X[
        ,
        j
      ],
      span = model$span
    )
    
    
    center_value <- mean(
      training_component
    )
    
    
    prediction_components[
      ,
      j
    ] <- component_prediction -
      center_value
  }
  
  
  model$intercept +
    rowSums(
      prediction_components
    )
}


# ------------------------------------------------------------------------------
# 32. Train-Test Split
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
# 33. Standardize Multivariate Kernel Inputs Using Training Data
# ------------------------------------------------------------------------------

train_means <- colMeans(
  X_train_raw
)


train_sds <- apply(
  X_train_raw,
  2,
  sd
)


X_train_scaled <- scale(
  X_train_raw,
  center = train_means,
  scale = train_sds
)


X_test_scaled <- scale(
  X_test_raw,
  center = train_means,
  scale = train_sds
)


X_train_scaled <- as.matrix(
  X_train_scaled
)


X_test_scaled <- as.matrix(
  X_test_scaled
)


# ------------------------------------------------------------------------------
# 34. Test Spherical Kernel
# ------------------------------------------------------------------------------

spherical_test_prediction <- structured_local_predict(
  X_train = X_train_scaled,
  y_train = y_train,
  X_new = X_test_scaled,
  span = 0.3,
  A = A_spherical
)


# ------------------------------------------------------------------------------
# 35. Test Structured Kernel
# ------------------------------------------------------------------------------

structured_test_prediction <- structured_local_predict(
  X_train = X_train_scaled,
  y_train = y_train,
  X_new = X_test_scaled,
  span = 0.3,
  A = A_structured
)


# ------------------------------------------------------------------------------
# 36. Test Kernel Ignoring x3
# ------------------------------------------------------------------------------

ignore_x3_test_prediction <- structured_local_predict(
  X_train = X_train_scaled,
  y_train = y_train,
  X_new = X_test_scaled,
  span = 0.3,
  A = A_ignore_x3
)


# ------------------------------------------------------------------------------
# 37. Test Additive Structured Model
# ------------------------------------------------------------------------------

additive_train_model <- additive_fit_with_targets(
  X = X_train_raw,
  y = y_train,
  span = 0.3
)


additive_test_prediction <- additive_predict(
  additive_train_model,
  X_test_raw
)


# ------------------------------------------------------------------------------
# 38. Test Error Comparison
# ------------------------------------------------------------------------------

test_results <- data.frame(
  Model = c(
    "Spherical Local Regression",
    "Structured Kernel",
    "Ignore x3 Kernel",
    "Additive Local Regression"
  ),
  
  Test_MSE = c(
    
    mean(
      (
        y_test -
          spherical_test_prediction
      )^2
    ),
    
    mean(
      (
        y_test -
          structured_test_prediction
      )^2
    ),
    
    mean(
      (
        y_test -
          ignore_x3_test_prediction
      )^2
    ),
    
    mean(
      (
        y_test -
          additive_test_prediction
      )^2
    )
  )
)


test_results$Test_RMSE <- sqrt(
  test_results$Test_MSE
)


test_results


# ------------------------------------------------------------------------------
# 39. Cross-Validation for Span
# ------------------------------------------------------------------------------

structured_local_cv <- function(
    X,
    y,
    spans,
    A,
    k = 5
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  folds <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  errors <- matrix(
    NA,
    nrow = length(
      spans
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
    
    
    X_train_raw <- X[
      train_index,
      ,
      drop = FALSE
    ]
    
    
    X_validation_raw <- X[
      validation_index,
      ,
      drop = FALSE
    ]
    
    
    y_train <- y[
      train_index
    ]
    
    
    y_validation <- y[
      validation_index
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
    
    
    X_validation <- scale(
      X_validation_raw,
      center = means,
      scale = sds
    )
    
    
    for (
      i in seq_along(
        spans
      )
    ) {
      
      prediction <- structured_local_predict(
        X_train = X_train,
        y_train = y_train,
        X_new = X_validation,
        span = spans[i],
        A = A
      )
      
      
      errors[
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
      Span = spans,
      Fold_MSE = errors,
      Mean_MSE = rowMeans(
        errors
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 40. Run CV for Structured Kernel
# ------------------------------------------------------------------------------

span_grid <- seq(
  0.15,
  0.7,
  by = 0.05
)


set.seed(123)


CV_structured <- structured_local_cv(
  X = X,
  y = y,
  spans = span_grid,
  A = A_structured,
  k = 5
)


best_index <- which.min(
  CV_structured$Mean_MSE
)


best_span <- CV_structured$Span[
  best_index
]


best_span


# ------------------------------------------------------------------------------
# 41. Plot CV Error
# ------------------------------------------------------------------------------

plot(
  CV_structured$Span,
  CV_structured$Mean_MSE,
  type = "b",
  pch = 19,
  xlab = "Span",
  ylab = "Cross-Validated MSE",
  main = "Structured Local Regression CV"
)


abline(
  v = best_span,
  lty = 2
)


# ------------------------------------------------------------------------------
# 42. Interaction Example
# ------------------------------------------------------------------------------

# Additive structure works well only if interactions are absent or weak.
#
# Consider:
#
# f(x1, x2) =
#
# sin(x1 * x2)
#
#
# This cannot generally be written as:
#
# g1(x1) + g2(x2).
#
#
# An unrestricted multivariate local smoother can represent such
# interactions more naturally.


interaction_function <- function(
    x1,
    x2
) {
  
  sin(
    x1 *
      x2
  )
}


# ------------------------------------------------------------------------------
# 43. Curse of Dimensionality Illustration
# ------------------------------------------------------------------------------

# Suppose a one-dimensional neighborhood contains 30% of the data
# along each coordinate.
#
# A corresponding hypercube in p dimensions contains approximately:
#
# 0.3^p
#
# of the observations.


dimensions <- 1:10


local_fraction <- 0.3^dimensions


curse_table <- data.frame(
  Dimension = dimensions,
  Approximate_Fraction = local_fraction,
  Expected_Observations =
    n *
    local_fraction
)


curse_table


# ------------------------------------------------------------------------------
# 44. Plot Curse of Dimensionality
# ------------------------------------------------------------------------------

plot(
  dimensions,
  local_fraction,
  type = "b",
  pch = 19,
  log = "y",
  xlab = "Number of Predictors",
  ylab = "Approximate Local Fraction",
  main = "Curse of Dimensionality"
)


# ------------------------------------------------------------------------------
# 45. Summary
# ------------------------------------------------------------------------------

cat(
  "Structured Local Regression\n"
)


cat(
  "---------------------------\n"
)


cat(
  "Spherical kernel training MSE:",
  round(
    MSE_spherical,
    4
  ),
  "\n"
)


cat(
  "Structured kernel training MSE:",
  round(
    MSE_structured,
    4
  ),
  "\n"
)


cat(
  "Kernel ignoring x3 training MSE:",
  round(
    MSE_ignore_x3,
    4
  ),
  "\n"
)


cat(
  "Additive local regression training MSE:",
  round(
    additive_MSE,
    4
  ),
  "\n"
)


cat(
  "Best structured-kernel CV span:",
  best_span,
  "\n"
)


cat(
  "\nTest results:\n"
)


print(
  test_results
)