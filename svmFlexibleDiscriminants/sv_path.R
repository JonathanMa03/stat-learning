# ==============================================================================
# Path Algorithm for Support Vector Machines
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Linear support vector classifier
#   - Regularization path
#   - Warm starts
#   - Support-vector set changes
#   - Approximate knots
#   - Piecewise-linear coefficient behavior
#   - Effect of C on margin and model complexity
#
# Note:
#   This is a pedagogical approximation to the SVM path.
#   We solve the primal SVC over a dense grid of C values and use warm starts.
#   Exact SVM path algorithms move directly from knot to knot.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Binary Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 200


X_negative <- cbind(
  rnorm(
    n / 2,
    mean = -1.2,
    sd = 1
  ),
  rnorm(
    n / 2,
    mean = -1,
    sd = 1
  )
)


X_positive <- cbind(
  rnorm(
    n / 2,
    mean = 1.2,
    sd = 1
  ),
  rnorm(
    n / 2,
    mean = 1,
    sd = 1
  )
)


X <- rbind(
  X_negative,
  X_positive
)


colnames(X) <- c(
  "x1",
  "x2"
)


y <- c(
  rep(-1, n / 2),
  rep(1, n / 2)
)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

plot(
  X[, 1],
  X[, 2],
  pch = ifelse(
    y == 1,
    19,
    1
  ),
  xlab = "x1",
  ylab = "x2",
  main = "Support Vector Classification Data"
)


legend(
  "topleft",
  legend = c(
    "Class -1",
    "Class +1"
  ),
  pch = c(
    1,
    19
  )
)


# ------------------------------------------------------------------------------
# 3. Standardize Predictors
# ------------------------------------------------------------------------------

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
# 4. Hinge Loss
# ------------------------------------------------------------------------------

hinge_loss <- function(
    margin
) {
  
  pmax(
    0,
    1 - margin
  )
}


# ------------------------------------------------------------------------------
# 5. SVC Objective
# ------------------------------------------------------------------------------

# We use:
#
#   1/2 ||w||^2
#       +
#   C * sum_i max(0, 1 - y_i f_i)
#
# where
#
#   f_i = w'x_i + b


svc_objective <- function(
    X,
    y,
    w,
    b,
    C
) {
  
  scores <- as.vector(
    X %*% w + b
  )
  
  
  margins <- y *
    scores
  
  
  objective <- 0.5 *
    sum(
      w^2
    ) +
    C *
    sum(
      hinge_loss(
        margins
      )
    )
  
  
  return(objective)
}


# ------------------------------------------------------------------------------
# 6. Warm-Start SVC Solver
# ------------------------------------------------------------------------------

svc_fit <- function(
    X,
    y,
    C,
    w_init = NULL,
    b_init = 0,
    learning_rate = 0.001,
    max_iter = 20000,
    tol = 1e-8,
    decay = 1e-4
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  
  p <- ncol(X)
  
  
  if (
    is.null(
      w_init
    )
  ) {
    
    w <- rep(
      0,
      p
    )
    
  } else {
    
    w <- as.numeric(
      w_init
    )
  }
  
  
  b <- b_init
  
  
  objective_history <- numeric(
    max_iter
  )
  
  
  for (iteration in seq_len(
    max_iter
  )) {
    
    scores <- as.vector(
      X %*% w + b
    )
    
    
    margins <- y *
      scores
    
    
    active <- margins < 1
    
    
    # ------------------------------------------------------------
    # Subgradient
    # ------------------------------------------------------------
    
    grad_w <- w
    
    grad_b <- 0
    
    
    if (
      any(active)
    ) {
      
      weighted_X <- sweep(
        X[
          active,
          ,
          drop = FALSE
        ],
        MARGIN = 1,
        STATS = y[active],
        FUN = "*"
      )
      
      
      grad_w <- grad_w -
        C *
        colSums(
          weighted_X
        )
      
      
      grad_b <- -C *
        sum(
          y[active]
        )
    }
    
    
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
      svc_objective(
        X,
        y,
        w_new,
        b_new,
        C
      )
    
    
    change <- max(
      c(
        abs(
          w_new - w
        ),
        abs(
          b_new - b
        )
      )
    )
    
    
    w <- w_new
    
    b <- b_new
    
    
    if (
      change < tol
    ) {
      
      break
    }
  }
  
  
  scores <- as.vector(
    X %*% w + b
  )
  
  
  margins <- y *
    scores
  
  
  return(
    list(
      weights = w,
      intercept = b,
      margins = margins,
      scores = scores,
      iterations = iteration,
      objective = objective_history[
        seq_len(iteration)
      ]
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Construct Regularization Path
# ------------------------------------------------------------------------------

# Start with strong regularization:
#
# small C
#
# and gradually increase C.


C_grid <- 10^seq(
  -3,
  3,
  length.out = 300
)


p <- ncol(
  X_scaled
)


weight_path <- matrix(
  NA,
  nrow = length(
    C_grid
  ),
  ncol = p
)


colnames(weight_path) <- colnames(
  X_scaled
)


intercept_path <- numeric(
  length(
    C_grid
  )
)


support_count <- numeric(
  length(
    C_grid
  )
)


violation_count <- numeric(
  length(
    C_grid
  )
)


misclassification_count <- numeric(
  length(
    C_grid
  )
)


margin_width <- numeric(
  length(
    C_grid
  )
)


training_error <- numeric(
  length(
    C_grid
  )
)


support_sets <- vector(
  "list",
  length(
    C_grid
  )
)


# ------------------------------------------------------------------------------
# 8. Warm Starts Along Path
# ------------------------------------------------------------------------------

w_previous <- rep(
  0,
  p
)


b_previous <- 0


for (i in seq_along(
  C_grid
)) {
  
  C_i <- C_grid[i]
  
  
  model <- svc_fit(
    X = X_scaled,
    y = y,
    C = C_i,
    w_init = w_previous,
    b_init = b_previous,
    learning_rate = 0.001,
    max_iter = 15000
  )
  
  
  weight_path[i, ] <-
    model$weights
  
  
  intercept_path[i] <-
    model$intercept
  
  
  margins_i <- model$margins
  
  
  # Approximate active support-vector set
  
  support_sets[[i]] <- which(
    margins_i <=
      1 +
      1e-4
  )
  
  
  support_count[i] <- length(
    support_sets[[i]]
  )
  
  
  violation_count[i] <- sum(
    margins_i < 1
  )
  
  
  misclassification_count[i] <- sum(
    margins_i < 0
  )
  
  
  predictions_i <- ifelse(
    model$scores >= 0,
    1,
    -1
  )
  
  
  training_error[i] <- mean(
    predictions_i != y
  )
  
  
  w_norm <- sqrt(
    sum(
      model$weights^2
    )
  )
  
  
  if (
    w_norm >
    1e-12
  ) {
    
    margin_width[i] <- 2 /
      w_norm
    
  } else {
    
    margin_width[i] <- NA
  }
  
  
  # Warm start next fit
  
  w_previous <- model$weights
  
  b_previous <- model$intercept
}


# ------------------------------------------------------------------------------
# 9. Inspect Path
# ------------------------------------------------------------------------------

head(
  weight_path
)


head(
  intercept_path
)


# ------------------------------------------------------------------------------
# 10. Plot Coefficient Path
# ------------------------------------------------------------------------------

matplot(
  log10(
    C_grid
  ),
  weight_path,
  type = "l",
  lty = 1,
  xlab = "log10(C)",
  ylab = "Coefficient",
  main = "SVM Coefficient Path"
)


legend(
  "topleft",
  legend = colnames(
    weight_path
  ),
  lty = 1
)


# ------------------------------------------------------------------------------
# 11. Plot Intercept Path
# ------------------------------------------------------------------------------

plot(
  log10(
    C_grid
  ),
  intercept_path,
  type = "l",
  xlab = "log10(C)",
  ylab = "Intercept",
  main = "SVM Intercept Path"
)


# ------------------------------------------------------------------------------
# 12. Number of Support Vectors Along Path
# ------------------------------------------------------------------------------

plot(
  log10(
    C_grid
  ),
  support_count,
  type = "l",
  xlab = "log10(C)",
  ylab = "Approximate Support Vectors",
  main = "Support Vectors Along SVM Path"
)


# ------------------------------------------------------------------------------
# 13. Margin Violations Along Path
# ------------------------------------------------------------------------------

plot(
  log10(
    C_grid
  ),
  violation_count,
  type = "l",
  xlab = "log10(C)",
  ylab = "Margin Violations",
  main = "Margin Violations Along Path"
)


# ------------------------------------------------------------------------------
# 14. Misclassifications Along Path
# ------------------------------------------------------------------------------

plot(
  log10(
    C_grid
  ),
  misclassification_count,
  type = "l",
  xlab = "log10(C)",
  ylab = "Misclassified Observations",
  main = "Training Errors Along Path"
)


# ------------------------------------------------------------------------------
# 15. Margin Width Along Path
# ------------------------------------------------------------------------------

plot(
  log10(
    C_grid
  ),
  margin_width,
  type = "l",
  xlab = "log10(C)",
  ylab = "Margin Width",
  main = "Margin Width Along SVM Path"
)


# ------------------------------------------------------------------------------
# 16. Detect Support-Set Changes
# ------------------------------------------------------------------------------

support_set_changed <- logical(
  length(
    C_grid
  )
)


support_set_changed[1] <- TRUE


for (i in 2:length(
  C_grid
)) {
  
  previous_set <- sort(
    support_sets[[i - 1]]
  )
  
  
  current_set <- sort(
    support_sets[[i]]
  )
  
  
  support_set_changed[i] <- !identical(
    previous_set,
    current_set
  )
}


# ------------------------------------------------------------------------------
# 17. Approximate Knots
# ------------------------------------------------------------------------------

knot_index <- which(
  support_set_changed
)


knot_C <- C_grid[
  knot_index
]


knot_table <- data.frame(
  Path_Index = knot_index,
  C = knot_C,
  log10_C = log10(
    knot_C
  ),
  Support_Vectors = support_count[
    knot_index
  ],
  Margin_Violations = violation_count[
    knot_index
  ],
  Misclassified = misclassification_count[
    knot_index
  ]
)


head(
  knot_table,
  20
)


# ------------------------------------------------------------------------------
# 18. Number of Approximate Knots
# ------------------------------------------------------------------------------

length(
  knot_index
)


# ------------------------------------------------------------------------------
# 19. Overlay Knots on Coefficient Path
# ------------------------------------------------------------------------------

matplot(
  log10(
    C_grid
  ),
  weight_path,
  type = "l",
  lty = 1,
  xlab = "log10(C)",
  ylab = "Coefficient",
  main = "Approximate SVM Path Knots"
)


abline(
  v = log10(
    knot_C
  ),
  lty = 3
)


# ------------------------------------------------------------------------------
# 20. Inspect Changes in Support Vectors
# ------------------------------------------------------------------------------

if (
  length(
    knot_index
  ) >= 2
) {
  
  first_knot <- knot_index[1]
  
  second_knot <- knot_index[2]
  
  
  support_sets[[
    first_knot
  ]]
  
  
  support_sets[[
    second_knot
  ]]
}


# ------------------------------------------------------------------------------
# 21. Observations Entering or Leaving Support Set
# ------------------------------------------------------------------------------

support_changes <- vector(
  "list",
  length(
    knot_index
  )
)


if (
  length(
    knot_index
  ) > 1
) {
  
  for (j in 2:length(
    knot_index
  )) {
    
    previous_index <- knot_index[
      j - 1
    ]
    
    current_index <- knot_index[j]
    
    
    previous_set <- support_sets[[
      previous_index
    ]]
    
    
    current_set <- support_sets[[
      current_index
    ]]
    
    
    entered <- setdiff(
      current_set,
      previous_set
    )
    
    
    left <- setdiff(
      previous_set,
      current_set
    )
    
    
    support_changes[[j]] <- list(
      C = C_grid[
        current_index
      ],
      Entered = entered,
      Left = left
    )
  }
}


support_changes[
  1:min(
    10,
    length(
      support_changes
    )
  )
]


# ------------------------------------------------------------------------------
# 22. Examine Local Slopes
# ------------------------------------------------------------------------------

# For an exact SVM path algorithm, coefficients are piecewise linear
# in the appropriate regularization parameter between knots.
#
# Here we estimate local slopes numerically.


lambda_grid <- 1 /
  C_grid


weight_slopes <- matrix(
  NA,
  nrow = length(
    C_grid
  ) - 1,
  ncol = p
)


colnames(weight_slopes) <- colnames(
  weight_path
)


for (j in seq_len(p)) {
  
  weight_slopes[, j] <-
    diff(
      weight_path[, j]
    ) /
    diff(
      lambda_grid
    )
}


# ------------------------------------------------------------------------------
# 23. Plot Local Slope for First Coefficient
# ------------------------------------------------------------------------------

plot(
  lambda_grid[-1],
  weight_slopes[, 1],
  type = "l",
  xlab = "lambda = 1/C",
  ylab = "Approximate Local Slope",
  main = "Local Slope of First SVM Coefficient"
)


# ------------------------------------------------------------------------------
# 24. Path Against lambda = 1/C
# ------------------------------------------------------------------------------

matplot(
  lambda_grid,
  weight_path,
  type = "l",
  lty = 1,
  xlab = "lambda = 1/C",
  ylab = "Coefficient",
  main = "SVM Regularization Path"
)


legend(
  "topright",
  legend = colnames(
    weight_path
  ),
  lty = 1
)


# ------------------------------------------------------------------------------
# 25. Training Error Along Path
# ------------------------------------------------------------------------------

plot(
  log10(
    C_grid
  ),
  training_error,
  type = "l",
  xlab = "log10(C)",
  ylab = "Training Error",
  main = "Training Error Along SVM Path"
)


# ------------------------------------------------------------------------------
# 26. Decision Boundaries at Selected C Values
# ------------------------------------------------------------------------------

selected_C <- c(
  0.01,
  0.1,
  1,
  10,
  100
)


selected_indices <- sapply(
  selected_C,
  function(value) {
    
    which.min(
      abs(
        C_grid - value
      )
    )
  }
)


par(
  mfrow = c(
    2,
    3
  )
)


for (index in selected_indices) {
  
  w <- weight_path[
    index,
  ]
  
  
  b <- intercept_path[
    index
  ]
  
  
  plot(
    X_scaled[, 1],
    X_scaled[, 2],
    pch = ifelse(
      y == 1,
      19,
      1
    ),
    xlab = "x1",
    ylab = "x2",
    main = paste(
      "C =",
      round(
        C_grid[index],
        3
      )
    )
  )
  
  
  if (
    abs(
      w[2]
    ) >
    1e-12
  ) {
    
    # Decision boundary
    
    abline(
      a = -b /
        w[2],
      b = -w[1] /
        w[2],
      lwd = 2
    )
    
    
    # Positive margin
    
    abline(
      a = (1 - b) /
        w[2],
      b = -w[1] /
        w[2],
      lty = 2
    )
    
    
    # Negative margin
    
    abline(
      a = (-1 - b) /
        w[2],
      b = -w[1] /
        w[2],
      lty = 2
    )
  }
  
  
  support_index <- support_sets[[
    index
  ]]
  
  
  points(
    X_scaled[
      support_index,
      1
    ],
    X_scaled[
      support_index,
      2
    ],
    pch = 1,
    cex = 1.5
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 27. Train-Test Split
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
# 28. Training-Only Standardization
# ------------------------------------------------------------------------------

train_means <- colMeans(
  X_train_raw
)


train_sds <- apply(
  X_train_raw,
  2,
  sd
)


X_train <- scale(
  X_train_raw,
  center = train_means,
  scale = train_sds
)


X_test <- scale(
  X_test_raw,
  center = train_means,
  scale = train_sds
)


X_train <- as.matrix(
  X_train
)


X_test <- as.matrix(
  X_test
)


# ------------------------------------------------------------------------------
# 29. Construct Path on Training Data
# ------------------------------------------------------------------------------

train_error_path <- numeric(
  length(
    C_grid
  )
)


test_error_path <- numeric(
  length(
    C_grid
  )
)


train_support_path <- numeric(
  length(
    C_grid
  )
)


w_previous <- rep(
  0,
  ncol(
    X_train
  )
)


b_previous <- 0


for (i in seq_along(
  C_grid
)) {
  
  model <- svc_fit(
    X = X_train,
    y = y_train,
    C = C_grid[i],
    w_init = w_previous,
    b_init = b_previous,
    learning_rate = 0.001,
    max_iter = 15000
  )
  
  
  train_score <- as.vector(
    X_train %*%
      model$weights +
      model$intercept
  )
  
  
  test_score <- as.vector(
    X_test %*%
      model$weights +
      model$intercept
  )
  
  
  train_prediction <- ifelse(
    train_score >= 0,
    1,
    -1
  )
  
  
  test_prediction <- ifelse(
    test_score >= 0,
    1,
    -1
  )
  
  
  train_error_path[i] <- mean(
    train_prediction !=
      y_train
  )
  
  
  test_error_path[i] <- mean(
    test_prediction !=
      y_test
  )
  
  
  train_margins <- y_train *
    train_score
  
  
  train_support_path[i] <- sum(
    train_margins <=
      1 +
      1e-4
  )
  
  
  w_previous <- model$weights
  
  b_previous <- model$intercept
}


# ------------------------------------------------------------------------------
# 30. Training and Test Error Path
# ------------------------------------------------------------------------------

plot(
  log10(
    C_grid
  ),
  train_error_path,
  type = "l",
  xlab = "log10(C)",
  ylab = "Classification Error",
  main = "Training and Test Error Along SVM Path"
)


lines(
  log10(
    C_grid
  ),
  test_error_path,
  lty = 2,
  lwd = 2
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
  )
)


# ------------------------------------------------------------------------------
# 31. Best Test C
# ------------------------------------------------------------------------------

best_test_index <- which.min(
  test_error_path
)


best_test_C <- C_grid[
  best_test_index
]


best_test_C


test_error_path[
  best_test_index
]


# ------------------------------------------------------------------------------
# 32. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

svc_path_cv <- function(
    X,
    y,
    C_grid,
    k = 5
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
  
  
  cv_errors <- matrix(
    NA,
    nrow = length(
      C_grid
    ),
    ncol = k
  )
  
  
  for (fold in seq_len(k)) {
    
    training_index <- which(
      folds != fold
    )
    
    
    validation_index <- which(
      folds == fold
    )
    
    
    X_training_raw <- X[
      training_index,
      ,
      drop = FALSE
    ]
    
    
    X_validation_raw <- X[
      validation_index,
      ,
      drop = FALSE
    ]
    
    
    y_training <- y[
      training_index
    ]
    
    
    y_validation <- y[
      validation_index
    ]
    
    
    means <- colMeans(
      X_training_raw
    )
    
    
    sds <- apply(
      X_training_raw,
      2,
      sd
    )
    
    
    X_training <- scale(
      X_training_raw,
      center = means,
      scale = sds
    )
    
    
    X_validation <- scale(
      X_validation_raw,
      center = means,
      scale = sds
    )
    
    
    X_training <- as.matrix(
      X_training
    )
    
    
    X_validation <- as.matrix(
      X_validation
    )
    
    
    w_previous <- rep(
      0,
      ncol(
        X_training
      )
    )
    
    
    b_previous <- 0
    
    
    for (i in seq_along(
      C_grid
    )) {
      
      model <- svc_fit(
        X = X_training,
        y = y_training,
        C = C_grid[i],
        w_init = w_previous,
        b_init = b_previous,
        learning_rate = 0.001,
        max_iter = 10000
      )
      
      
      validation_scores <- as.vector(
        X_validation %*%
          model$weights +
          model$intercept
      )
      
      
      validation_prediction <- ifelse(
        validation_scores >= 0,
        1,
        -1
      )
      
      
      cv_errors[
        i,
        fold
      ] <- mean(
        validation_prediction !=
          y_validation
      )
      
      
      w_previous <- model$weights
      
      b_previous <- model$intercept
    }
  }
  
  
  return(
    list(
      C = C_grid,
      errors = cv_errors,
      mean_error = rowMeans(
        cv_errors
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 33. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


cv_results <- svc_path_cv(
  X = X,
  y = y,
  C_grid = C_grid,
  k = 5
)


# ------------------------------------------------------------------------------
# 34. Best C by Cross-Validation
# ------------------------------------------------------------------------------

best_cv_index <- which.min(
  cv_results$mean_error
)


best_C <- cv_results$C[
  best_cv_index
]


best_C


# ------------------------------------------------------------------------------
# 35. Plot Cross-Validation Path
# ------------------------------------------------------------------------------

plot(
  log10(
    cv_results$C
  ),
  cv_results$mean_error,
  type = "l",
  xlab = "log10(C)",
  ylab = "Cross-Validated Error",
  main = "SVM Regularization Path"
)


abline(
  v = log10(
    best_C
  ),
  lty = 2
)


# ------------------------------------------------------------------------------
# 36. Fit Final Model
# ------------------------------------------------------------------------------

final_model <- svc_fit(
  X = X_scaled,
  y = y,
  C = best_C,
  learning_rate = 0.001,
  max_iter = 20000
)


final_scores <- as.vector(
  X_scaled %*%
    final_model$weights +
    final_model$intercept
)


final_prediction <- ifelse(
  final_scores >= 0,
  1,
  -1
)


final_accuracy <- mean(
  final_prediction ==
    y
)


final_margins <- y *
  final_scores


final_support_vectors <- which(
  final_margins <=
    1 +
    1e-4
)


# ------------------------------------------------------------------------------
# 37. Final Model Summary
# ------------------------------------------------------------------------------

cat(
  "Best C:",
  best_C,
  "\n"
)


cat(
  "Training Accuracy:",
  round(
    final_accuracy,
    4
  ),
  "\n"
)


cat(
  "Approximate Support Vectors:",
  length(
    final_support_vectors
  ),
  "\n"
)


cat(
  "Approximate Path Knots:",
  length(
    knot_index
  ),
  "\n"
)


cat(
  "Margin Width:",
  round(
    2 /
      sqrt(
        sum(
          final_model$weights^2
        )
      ),
    4
  ),
  "\n"
)