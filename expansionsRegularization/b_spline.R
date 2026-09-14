# ==============================================================================
# B-Splines
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - B-spline basis functions
#   - Cox-de Boor recursion
#   - Local support
#   - Degree and knots
#   - Regression using B-spline bases
#   - Comparison with truncated-power splines
#   - Cross-validation for spline complexity
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
  main = "Nonlinear Regression Data"
)


lines(
  x,
  y_true,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Choose Degree and Interior Knots
# ------------------------------------------------------------------------------

degree <- 3


interior_knots <- c(
  2.5,
  5,
  7.5
)


boundary_knots <- c(
  min(x),
  max(x)
)


# ------------------------------------------------------------------------------
# 4. Construct Full Knot Vector
# ------------------------------------------------------------------------------

# For a B-spline of degree d, repeat each boundary knot d + 1 times.

full_knots <- c(
  rep(
    boundary_knots[1],
    degree + 1
  ),
  interior_knots,
  rep(
    boundary_knots[2],
    degree + 1
  )
)


full_knots


# ------------------------------------------------------------------------------
# 5. Degree-Zero B-Spline Basis Function
# ------------------------------------------------------------------------------

# B_{i,0}(x) = 1 if t_i <= x < t_{i+1}
#              0 otherwise
#
# Special handling is used for the final endpoint.


bspline_zero <- function(
    x,
    i,
    knots
) {
  
  left <- knots[i]
  
  right <- knots[i + 1]
  
  
  basis <- ifelse(
    x >= left &
      x < right,
    1,
    0
  )
  
  
  # Include final endpoint in last nonempty interval
  
  if (
    right ==
    max(knots)
  ) {
    
    basis[
      x ==
        max(knots)
    ] <- 1
  }
  
  
  return(
    basis
  )
}


# ------------------------------------------------------------------------------
# 6. Cox-de Boor Recursive B-Spline Function
# ------------------------------------------------------------------------------

bspline_basis_function <- function(
    x,
    i,
    degree,
    knots
) {
  
  if (
    degree == 0
  ) {
    
    return(
      bspline_zero(
        x,
        i,
        knots
      )
    )
  }
  
  
  left_denominator <-
    knots[i + degree] -
    knots[i]
  
  
  right_denominator <-
    knots[i + degree + 1] -
    knots[i + 1]
  
  
  left_term <- rep(
    0,
    length(x)
  )
  
  
  right_term <- rep(
    0,
    length(x)
  )
  
  
  if (
    left_denominator >
    0
  ) {
    
    left_term <-
      (
        x -
          knots[i]
      ) /
      left_denominator *
      bspline_basis_function(
        x,
        i,
        degree - 1,
        knots
      )
  }
  
  
  if (
    right_denominator >
    0
  ) {
    
    right_term <-
      (
        knots[i + degree + 1] -
          x
      ) /
      right_denominator *
      bspline_basis_function(
        x,
        i + 1,
        degree - 1,
        knots
      )
  }
  
  
  return(
    left_term +
      right_term
  )
}


# ------------------------------------------------------------------------------
# 7. Number of B-Spline Basis Functions
# ------------------------------------------------------------------------------

# If the full knot vector has length m,
#
# number of basis functions =
#
# m - degree - 1


number_basis <- length(
  full_knots
) -
  degree -
  1


number_basis


# ------------------------------------------------------------------------------
# 8. Construct B-Spline Design Matrix
# ------------------------------------------------------------------------------

bspline_basis <- function(
    x,
    interior_knots,
    degree = 3,
    boundary_knots = range(x)
) {
  
  full_knots <- c(
    rep(
      boundary_knots[1],
      degree + 1
    ),
    interior_knots,
    rep(
      boundary_knots[2],
      degree + 1
    )
  )
  
  
  number_basis <-
    length(full_knots) -
    degree -
    1
  
  
  B <- matrix(
    0,
    nrow = length(x),
    ncol = number_basis
  )
  
  
  for (j in seq_len(
    number_basis
  )) {
    
    B[, j] <-
      bspline_basis_function(
        x = x,
        i = j,
        degree = degree,
        knots = full_knots
      )
  }
  
  
  colnames(B) <- paste0(
    "B",
    seq_len(
      number_basis
    )
  )
  
  
  return(
    B
  )
}


# ------------------------------------------------------------------------------
# 9. Build Cubic B-Spline Basis
# ------------------------------------------------------------------------------

B <- bspline_basis(
  x = x,
  interior_knots = interior_knots,
  degree = degree,
  boundary_knots = boundary_knots
)


dim(
  B
)


head(
  B
)


# ------------------------------------------------------------------------------
# 10. Check Partition of Unity
# ------------------------------------------------------------------------------

# Within the spline domain, B-spline basis functions should sum to 1.

basis_sum <- rowSums(
  B
)


summary(
  basis_sum
)


max(
  abs(
    basis_sum -
      1
  )
)


# ------------------------------------------------------------------------------
# 11. Plot B-Spline Basis Functions
# ------------------------------------------------------------------------------

matplot(
  x,
  B,
  type = "l",
  lty = 1,
  xlab = "x",
  ylab = "Basis Value",
  main = "Cubic B-Spline Basis Functions"
)


abline(
  v = interior_knots,
  lty = 3
)


legend(
  "topright",
  legend = colnames(B),
  lty = 1,
  cex = 0.8
)


# ------------------------------------------------------------------------------
# 12. Demonstrate Local Support
# ------------------------------------------------------------------------------

# Each B-spline basis function is nonzero only over a limited region.

support_ranges <- data.frame(
  Basis = colnames(B),
  Minimum_x = NA,
  Maximum_x = NA
)


for (j in seq_len(
  ncol(B)
)) {
  
  active <- which(
    B[, j] >
      1e-10
  )
  
  
  support_ranges$Minimum_x[j] <-
    min(
      x[active]
    )
  
  
  support_ranges$Maximum_x[j] <-
    max(
      x[active]
    )
}


support_ranges


# ------------------------------------------------------------------------------
# 13. Fit Regression Using B-Spline Basis
# ------------------------------------------------------------------------------

# Since the B-spline basis sums to 1, an explicit intercept is not required
# for representing constants.
#
# Here we fit directly using the basis matrix.


beta_hat <- solve(
  crossprod(
    B
  ),
  crossprod(
    B,
    y
  )
)


beta_hat


# ------------------------------------------------------------------------------
# 14. Fitted Values
# ------------------------------------------------------------------------------

y_hat <- as.vector(
  B %*%
    beta_hat
)


# ------------------------------------------------------------------------------
# 15. Training Error
# ------------------------------------------------------------------------------

MSE <- mean(
  (
    y -
      y_hat
  )^2
)


RMSE <- sqrt(
  MSE
)


data.frame(
  MSE = MSE,
  RMSE = RMSE
)


# ------------------------------------------------------------------------------
# 16. Plot B-Spline Regression Fit
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "B-Spline Regression"
)


lines(
  x,
  y_hat,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


abline(
  v = interior_knots,
  lty = 3
)


legend(
  "topleft",
  legend = c(
    "B-Spline Fit",
    "True Function",
    "Knots"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = c(
    2,
    2,
    1
  )
)


# ------------------------------------------------------------------------------
# 17. Generic B-Spline Fit Function
# ------------------------------------------------------------------------------

bspline_fit <- function(
    x,
    y,
    interior_knots,
    degree = 3,
    boundary_knots = range(x)
) {
  
  B <- bspline_basis(
    x = x,
    interior_knots = interior_knots,
    degree = degree,
    boundary_knots = boundary_knots
  )
  
  
  beta_hat <- solve(
    crossprod(B),
    crossprod(
      B,
      y
    )
  )
  
  
  return(
    list(
      coefficients = as.vector(
        beta_hat
      ),
      interior_knots = interior_knots,
      degree = degree,
      boundary_knots = boundary_knots,
      basis = B
    )
  )
}


# ------------------------------------------------------------------------------
# 18. Generic B-Spline Prediction
# ------------------------------------------------------------------------------

bspline_predict <- function(
    x_new,
    model
) {
  
  B_new <- bspline_basis(
    x = x_new,
    interior_knots = model$interior_knots,
    degree = model$degree,
    boundary_knots = model$boundary_knots
  )
  
  
  prediction <- as.vector(
    B_new %*%
      model$coefficients
  )
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 19. Verify Generic Fit
# ------------------------------------------------------------------------------

model <- bspline_fit(
  x = x,
  y = y,
  interior_knots = interior_knots,
  degree = 3,
  boundary_knots = boundary_knots
)


prediction <- bspline_predict(
  x,
  model
)


max(
  abs(
    prediction -
      y_hat
  )
)


# ------------------------------------------------------------------------------
# 20. Compare Different Polynomial Degrees
# ------------------------------------------------------------------------------

degrees <- c(
  0,
  1,
  2,
  3
)


degree_results <- data.frame(
  Degree = degrees,
  Number_Basis = NA,
  Training_MSE = NA
)


degree_models <- vector(
  "list",
  length(
    degrees
  )
)


for (i in seq_along(
  degrees
)) {
  
  degree_i <- degrees[i]
  
  
  model_i <- bspline_fit(
    x = x,
    y = y,
    interior_knots = interior_knots,
    degree = degree_i,
    boundary_knots = boundary_knots
  )
  
  
  prediction_i <- bspline_predict(
    x,
    model_i
  )
  
  
  degree_results$Number_Basis[i] <-
    length(
      model_i$coefficients
    )
  
  
  degree_results$Training_MSE[i] <-
    mean(
      (
        y -
          prediction_i
      )^2
    )
  
  
  degree_models[[i]] <- model_i
}


degree_results


# ------------------------------------------------------------------------------
# 21. Plot Fits by Degree
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    2,
    2
  )
)


for (i in seq_along(
  degrees
)) {
  
  prediction_i <- bspline_predict(
    x,
    degree_models[[i]]
  )
  
  
  plot(
    x,
    y,
    pch = 19,
    xlab = "x",
    ylab = "y",
    main = paste(
      "Degree",
      degrees[i]
    )
  )
  
  
  lines(
    x,
    prediction_i,
    lwd = 2
  )
  
  
  abline(
    v = interior_knots,
    lty = 3
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 22. Effect of Number of Interior Knots
# ------------------------------------------------------------------------------

knot_counts <- c(
  0,
  1,
  2,
  3,
  5,
  8
)


training_MSE <- numeric(
  length(
    knot_counts
  )
)


number_basis_functions <- numeric(
  length(
    knot_counts
  )
)


models_by_knots <- vector(
  "list",
  length(
    knot_counts
  )
)


for (i in seq_along(
  knot_counts
)) {
  
  number_knots <- knot_counts[i]
  
  
  if (
    number_knots == 0
  ) {
    
    knots_i <- numeric(
      0
    )
    
  } else {
    
    probabilities <- seq(
      0,
      1,
      length.out = number_knots + 2
    )
    
    
    probabilities <- probabilities[
      -c(
        1,
        length(
          probabilities
        )
      )
    ]
    
    
    knots_i <- as.numeric(
      quantile(
        x,
        probs = probabilities
      )
    )
  }
  
  
  model_i <- bspline_fit(
    x = x,
    y = y,
    interior_knots = knots_i,
    degree = 3,
    boundary_knots = range(x)
  )
  
  
  prediction_i <- bspline_predict(
    x,
    model_i
  )
  
  
  training_MSE[i] <- mean(
    (
      y -
        prediction_i
    )^2
  )
  
  
  number_basis_functions[i] <-
    length(
      model_i$coefficients
    )
  
  
  models_by_knots[[i]] <- model_i
}


knot_results <- data.frame(
  Number_of_Knots = knot_counts,
  Number_of_Basis_Functions =
    number_basis_functions,
  Training_MSE = training_MSE
)


knot_results


# ------------------------------------------------------------------------------
# 23. Plot Training Error vs Knot Count
# ------------------------------------------------------------------------------

plot(
  knot_counts,
  training_MSE,
  type = "b",
  pch = 19,
  xlab = "Number of Interior Knots",
  ylab = "Training MSE",
  main = "B-Spline Complexity"
)


# ------------------------------------------------------------------------------
# 24. Compare Few vs Many Knots
# ------------------------------------------------------------------------------

few_knots_model <- bspline_fit(
  x,
  y,
  interior_knots = c(
    3.33,
    6.67
  ),
  degree = 3
)


many_knots <- as.numeric(
  quantile(
    x,
    probs = seq(
      0.1,
      0.9,
      by = 0.1
    )
  )
)


many_knots_model <- bspline_fit(
  x,
  y,
  interior_knots = many_knots,
  degree = 3
)


few_prediction <- bspline_predict(
  x,
  few_knots_model
)


many_prediction <- bspline_predict(
  x,
  many_knots_model
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Effect of Number of Knots"
)


lines(
  x,
  few_prediction,
  lty = 2,
  lwd = 2
)


lines(
  x,
  many_prediction,
  lty = 1,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Many Knots",
    "Few Knots"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 25. Truncated-Power Cubic Spline for Comparison
# ------------------------------------------------------------------------------

positive_part <- function(
    x
) {
  
  pmax(
    x,
    0
  )
}


truncated_power_basis <- function(
    x,
    knots
) {
  
  basis <- cbind(
    Intercept = 1,
    x = x,
    x2 = x^2,
    x3 = x^3
  )
  
  
  if (
    length(
      knots
    ) > 0
  ) {
    
    for (j in seq_along(
      knots
    )) {
      
      basis <- cbind(
        basis,
        positive_part(
          x -
            knots[j]
        )^3
      )
    }
  }
  
  
  return(
    basis
  )
}


X_truncated <- truncated_power_basis(
  x,
  interior_knots
)


beta_truncated <- solve(
  crossprod(
    X_truncated
  ),
  crossprod(
    X_truncated,
    y
  )
)


prediction_truncated <- as.vector(
  X_truncated %*%
    beta_truncated
)


# ------------------------------------------------------------------------------
# 26. Compare Fits
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "B-Spline vs Truncated-Power Basis"
)


lines(
  x,
  y_hat,
  lwd = 2
)


lines(
  x,
  prediction_truncated,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "B-Spline Basis",
    "Truncated-Power Basis"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 27. Compare Predictions Numerically
# ------------------------------------------------------------------------------

# These bases span the same cubic spline space when the same
# degree and knots are used.
#
# Therefore the fitted values should be nearly identical.


max(
  abs(
    y_hat -
      prediction_truncated
  )
)


data.frame(
  Method = c(
    "B-Spline",
    "Truncated-Power"
  ),
  MSE = c(
    mean(
      (
        y -
          y_hat
      )^2
    ),
    mean(
      (
        y -
          prediction_truncated
      )^2
    )
  )
)


# ------------------------------------------------------------------------------
# 28. Compare Condition Numbers
# ------------------------------------------------------------------------------

# One major advantage of the B-spline basis is numerical stability.


condition_bspline <- kappa(
  B
)


condition_truncated <- kappa(
  X_truncated
)


data.frame(
  Basis = c(
    "B-Spline",
    "Truncated-Power"
  ),
  Condition_Number = c(
    condition_bspline,
    condition_truncated
  )
)


# ------------------------------------------------------------------------------
# 29. Train-Test Split
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
# 30. Evaluate Different Knot Counts
# ------------------------------------------------------------------------------

test_MSE <- numeric(
  length(
    knot_counts
  )
)


for (i in seq_along(
  knot_counts
)) {
  
  number_knots <- knot_counts[i]
  
  
  if (
    number_knots == 0
  ) {
    
    knots_i <- numeric(
      0
    )
    
  } else {
    
    probabilities <- seq(
      0,
      1,
      length.out = number_knots + 2
    )
    
    
    probabilities <- probabilities[
      -c(
        1,
        length(
          probabilities
        )
      )
    ]
    
    
    knots_i <- as.numeric(
      quantile(
        x_train,
        probs = probabilities
      )
    )
  }
  
  
  model_i <- bspline_fit(
    x = x_train,
    y = y_train,
    interior_knots = knots_i,
    degree = 3,
    boundary_knots = range(
      x_train
    )
  )
  
  
  # Only evaluate points inside training boundary range.
  
  valid_test <- x_test >=
    model_i$boundary_knots[1] &
    x_test <=
    model_i$boundary_knots[2]
  
  
  prediction_i <- bspline_predict(
    x_test[
      valid_test
    ],
    model_i
  )
  
  
  test_MSE[i] <- mean(
    (
      y_test[
        valid_test
      ] -
        prediction_i
    )^2
  )
}


test_results <- data.frame(
  Number_of_Knots = knot_counts,
  Test_MSE = test_MSE
)


test_results


# ------------------------------------------------------------------------------
# 31. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

bspline_cv <- function(
    x,
    y,
    knot_counts,
    degree = 3,
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
      knot_counts
    ),
    ncol = k
  )
  
  
  for (fold in seq_len(
    k
  )) {
    
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
    
    
    boundary_knots <- range(
      x_train
    )
    
    
    for (i in seq_along(
      knot_counts
    )) {
      
      number_knots <- knot_counts[i]
      
      
      if (
        number_knots == 0
      ) {
        
        knots_i <- numeric(
          0
        )
        
      } else {
        
        probabilities <- seq(
          0,
          1,
          length.out = number_knots + 2
        )
        
        
        probabilities <- probabilities[
          -c(
            1,
            length(
              probabilities
            )
          )
        ]
        
        
        knots_i <- as.numeric(
          quantile(
            x_train,
            probs = probabilities
          )
        )
      }
      
      
      model_i <- bspline_fit(
        x = x_train,
        y = y_train,
        interior_knots = knots_i,
        degree = degree,
        boundary_knots = boundary_knots
      )
      
      
      # Restrict to validation points inside training boundaries.
      
      valid <- x_validation >=
        boundary_knots[1] &
        x_validation <=
        boundary_knots[2]
      
      
      prediction_i <- bspline_predict(
        x_validation[
          valid
        ],
        model_i
      )
      
      
      CV_error[
        i,
        fold
      ] <- mean(
        (
          y_validation[
            valid
          ] -
            prediction_i
        )^2
      )
    }
  }
  
  
  return(
    list(
      Knot_Count = knot_counts,
      Fold_Error = CV_error,
      Mean_Error = rowMeans(
        CV_error
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 32. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


CV_results <- bspline_cv(
  x = x,
  y = y,
  knot_counts = knot_counts,
  degree = 3,
  k = 5
)


CV_table <- data.frame(
  Number_of_Knots = CV_results$Knot_Count,
  CV_MSE = CV_results$Mean_Error
)


CV_table


# ------------------------------------------------------------------------------
# 33. Select Best Number of Knots
# ------------------------------------------------------------------------------

best_index <- which.min(
  CV_results$Mean_Error
)


best_knot_count <- CV_results$Knot_Count[
  best_index
]


best_knot_count


# ------------------------------------------------------------------------------
# 34. Plot Cross-Validation Error
# ------------------------------------------------------------------------------

plot(
  CV_results$Knot_Count,
  CV_results$Mean_Error,
  type = "b",
  pch = 19,
  xlab = "Number of Interior Knots",
  ylab = "Cross-Validated MSE",
  main = "Choosing B-Spline Complexity"
)


# ------------------------------------------------------------------------------
# 35. Construct Final Knot Set
# ------------------------------------------------------------------------------

if (
  best_knot_count == 0
) {
  
  final_knots <- numeric(
    0
  )
  
} else {
  
  probabilities <- seq(
    0,
    1,
    length.out = best_knot_count + 2
  )
  
  
  probabilities <- probabilities[
    -c(
      1,
      length(
        probabilities
      )
    )
  ]
  
  
  final_knots <- as.numeric(
    quantile(
      x,
      probs = probabilities
    )
  )
}


final_knots


# ------------------------------------------------------------------------------
# 36. Fit Final Model
# ------------------------------------------------------------------------------

final_model <- bspline_fit(
  x = x,
  y = y,
  interior_knots = final_knots,
  degree = 3,
  boundary_knots = range(x)
)


final_prediction <- bspline_predict(
  x,
  final_model
)


final_MSE <- mean(
  (
    y -
      final_prediction
  )^2
)


final_MSE


# ------------------------------------------------------------------------------
# 37. Plot Final B-Spline
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Final B-Spline Regression Model"
)


lines(
  x,
  final_prediction,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


if (
  length(
    final_knots
  ) >
  0
) {
  
  abline(
    v = final_knots,
    lty = 3
  )
}


# ------------------------------------------------------------------------------
# 38. Verify Against splines::bs()
# ------------------------------------------------------------------------------

# splines is included with standard R.
# We use it only for verification.


B_builtin <- splines::bs(
  x,
  knots = interior_knots,
  degree = 3,
  Boundary.knots = boundary_knots,
  intercept = TRUE
)


dim(
  B_builtin
)


# Fit regression using built-in B-spline basis

beta_builtin <- solve(
  crossprod(
    B_builtin
  ),
  crossprod(
    B_builtin,
    y
  )
)


prediction_builtin <- as.vector(
  B_builtin %*%
    beta_builtin
)


# ------------------------------------------------------------------------------
# 39. Compare Manual and Built-In Predictions
# ------------------------------------------------------------------------------

max_difference <- max(
  abs(
    y_hat -
      prediction_builtin
  )
)


max_difference


data.frame(
  Method = c(
    "Manual B-Spline",
    "splines::bs()"
  ),
  MSE = c(
    mean(
      (
        y -
          y_hat
      )^2
    ),
    mean(
      (
        y -
          prediction_builtin
      )^2
    )
  )
)


# ------------------------------------------------------------------------------
# 40. Compare Basis Spaces
# ------------------------------------------------------------------------------

# The individual basis functions do not have to match exactly.
# Different bases can represent the same spline space.
#
# What matters is whether the fitted values from the same spline space agree.


plot(
  y_hat,
  prediction_builtin,
  pch = 19,
  xlab = "Manual B-Spline Prediction",
  ylab = "Built-In B-Spline Prediction",
  main = "Manual vs Built-In B-Spline"
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# ------------------------------------------------------------------------------
# 41. Summary
# ------------------------------------------------------------------------------

cat(
  "Degree:",
  degree,
  "\n"
)


cat(
  "Interior knots:",
  interior_knots,
  "\n"
)


cat(
  "Number of basis functions:",
  number_basis,
  "\n"
)


cat(
  "Training MSE:",
  round(
    MSE,
    4
  ),
  "\n"
)


cat(
  "Best knot count by CV:",
  best_knot_count,
  "\n"
)


cat(
  "Final training MSE:",
  round(
    final_MSE,
    4
  ),
  "\n"
)


cat(
  "B-spline condition number:",
  round(
    condition_bspline,
    2
  ),
  "\n"
)


cat(
  "Truncated-power condition number:",
  round(
    condition_truncated,
    2
  ),
  "\n"
)