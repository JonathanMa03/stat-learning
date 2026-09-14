# ==============================================================================
# Piecewise Polynomials and Splines
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Polynomial regression
#   - Piecewise polynomial regression
#   - Knots
#   - Continuity constraints
#   - Linear splines
#   - Cubic splines
#   - Truncated power basis
#   - Bias-variance tradeoff
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
    2 * sin(
      x
    )
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
# 3. Ordinary Polynomial Regression
# ------------------------------------------------------------------------------

# Begin with a global cubic polynomial:
#
# y =
# beta0 +
# beta1*x +
# beta2*x^2 +
# beta3*x^3


X_poly <- cbind(
  Intercept = 1,
  x = x,
  x2 = x^2,
  x3 = x^3
)


beta_poly <- solve(
  crossprod(
    X_poly
  ),
  crossprod(
    X_poly,
    y
  )
)


beta_poly


# ------------------------------------------------------------------------------
# 4. Polynomial Predictions
# ------------------------------------------------------------------------------

y_hat_poly <- as.vector(
  X_poly %*%
    beta_poly
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Global Cubic Polynomial"
)


lines(
  x,
  y_hat_poly,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 5. Polynomial Training Error
# ------------------------------------------------------------------------------

poly_MSE <- mean(
  (
    y -
      y_hat_poly
  )^2
)


poly_MSE


# ------------------------------------------------------------------------------
# 6. Introduce a Knot
# ------------------------------------------------------------------------------

# A piecewise polynomial fits separate functions in different regions.
#
# Start with one knot at:
#
# x = 5


knot <- 5


left_index <- x <= knot

right_index <- x > knot


# ------------------------------------------------------------------------------
# 7. Separate Piecewise Linear Fits
# ------------------------------------------------------------------------------

# Fit:
#
# y = beta0 + beta1*x
#
# separately on either side of the knot.


X_left <- cbind(
  1,
  x[left_index]
)


beta_left <- solve(
  crossprod(
    X_left
  ),
  crossprod(
    X_left,
    y[left_index]
  )
)


X_right <- cbind(
  1,
  x[right_index]
)


beta_right <- solve(
  crossprod(
    X_right
  ),
  crossprod(
    X_right,
    y[right_index]
  )
)


beta_left

beta_right


# ------------------------------------------------------------------------------
# 8. Piecewise Linear Predictions
# ------------------------------------------------------------------------------

y_hat_piecewise <- numeric(
  n
)


y_hat_piecewise[left_index] <-
  beta_left[1] +
  beta_left[2] *
  x[left_index]


y_hat_piecewise[right_index] <-
  beta_right[1] +
  beta_right[2] *
  x[right_index]


# ------------------------------------------------------------------------------
# 9. Plot Unconstrained Piecewise Linear Fit
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Unconstrained Piecewise Linear Regression"
)


lines(
  x[left_index],
  y_hat_piecewise[left_index],
  lwd = 2
)


lines(
  x[right_index],
  y_hat_piecewise[right_index],
  lwd = 2
)


abline(
  v = knot,
  lty = 2
)


# ------------------------------------------------------------------------------
# 10. Check Discontinuity at Knot
# ------------------------------------------------------------------------------

left_at_knot <- beta_left[1] +
  beta_left[2] *
  knot


right_at_knot <- beta_right[1] +
  beta_right[2] *
  knot


data.frame(
  Side = c(
    "Left",
    "Right"
  ),
  Prediction_at_Knot = c(
    left_at_knot,
    right_at_knot
  )
)


# Separate piecewise fits do not automatically meet at the knot.


# ------------------------------------------------------------------------------
# 11. Truncated Power Function
# ------------------------------------------------------------------------------

positive_part <- function(
    x
) {
  
  pmax(
    x,
    0
  )
}


# For a linear spline:
#
# (x - knot)_+
#
# equals:
#
# 0               if x <= knot
# x - knot        if x > knot


truncated_linear <- positive_part(
  x - knot
)


# ------------------------------------------------------------------------------
# 12. Linear Spline Basis
# ------------------------------------------------------------------------------

# A linear spline with one knot:
#
# f(x) =
# beta0 +
# beta1*x +
# beta2*(x-knot)_+
#
#
# This is automatically continuous at the knot.


X_linear_spline <- cbind(
  Intercept = 1,
  x = x,
  Knot1 = positive_part(
    x - knot
  )
)


beta_linear_spline <- solve(
  crossprod(
    X_linear_spline
  ),
  crossprod(
    X_linear_spline,
    y
  )
)


beta_linear_spline


# ------------------------------------------------------------------------------
# 13. Linear Spline Predictions
# ------------------------------------------------------------------------------

y_hat_linear_spline <- as.vector(
  X_linear_spline %*%
    beta_linear_spline
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Linear Spline"
)


lines(
  x,
  y_hat_linear_spline,
  lwd = 2
)


abline(
  v = knot,
  lty = 2
)


# ------------------------------------------------------------------------------
# 14. Interpret Slopes
# ------------------------------------------------------------------------------

# Before knot:
#
# slope = beta1
#
# After knot:
#
# slope = beta1 + beta2


slope_before <- beta_linear_spline[2]


slope_after <- beta_linear_spline[2] +
  beta_linear_spline[3]


data.frame(
  Region = c(
    "Before Knot",
    "After Knot"
  ),
  Slope = c(
    slope_before,
    slope_after
  )
)


# ------------------------------------------------------------------------------
# 15. Multiple-Knot Linear Spline
# ------------------------------------------------------------------------------

knots_linear <- c(
  2.5,
  5,
  7.5
)


X_linear_multi <- cbind(
  Intercept = 1,
  x = x
)


for (j in seq_along(
  knots_linear
)) {
  
  X_linear_multi <- cbind(
    X_linear_multi,
    positive_part(
      x -
        knots_linear[j]
    )
  )
}


colnames(
  X_linear_multi
) <- c(
  "Intercept",
  "x",
  paste0(
    "Knot_",
    knots_linear
  )
)


beta_linear_multi <- solve(
  crossprod(
    X_linear_multi
  ),
  crossprod(
    X_linear_multi,
    y
  )
)


# ------------------------------------------------------------------------------
# 16. Predictions from Multiple-Knot Linear Spline
# ------------------------------------------------------------------------------

y_hat_linear_multi <- as.vector(
  X_linear_multi %*%
    beta_linear_multi
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Piecewise Linear Spline"
)


lines(
  x,
  y_hat_linear_multi,
  lwd = 2
)


abline(
  v = knots_linear,
  lty = 2
)


# ------------------------------------------------------------------------------
# 17. Cubic Spline Basis
# ------------------------------------------------------------------------------

# Cubic spline:
#
# f(x) =
#
# beta0 +
# beta1*x +
# beta2*x^2 +
# beta3*x^3
#
# +
#
# sum_j theta_j (x-knot_j)_+^3
#
#
# This construction guarantees continuity of:
#
# f(x)
# f'(x)
# f''(x)
#
# at each knot.


knots_cubic <- c(
  2.5,
  5,
  7.5
)


X_cubic_spline <- cbind(
  Intercept = 1,
  x = x,
  x2 = x^2,
  x3 = x^3
)


for (j in seq_along(
  knots_cubic
)) {
  
  X_cubic_spline <- cbind(
    X_cubic_spline,
    positive_part(
      x -
        knots_cubic[j]
    )^3
  )
}


colnames(
  X_cubic_spline
) <- c(
  "Intercept",
  "x",
  "x2",
  "x3",
  paste0(
    "Knot_",
    knots_cubic
  )
)


# ------------------------------------------------------------------------------
# 18. Fit Cubic Spline
# ------------------------------------------------------------------------------

beta_cubic_spline <- solve(
  crossprod(
    X_cubic_spline
  ),
  crossprod(
    X_cubic_spline,
    y
  )
)


beta_cubic_spline


# ------------------------------------------------------------------------------
# 19. Cubic Spline Predictions
# ------------------------------------------------------------------------------

y_hat_cubic_spline <- as.vector(
  X_cubic_spline %*%
    beta_cubic_spline
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Cubic Regression Spline"
)


lines(
  x,
  y_hat_cubic_spline,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


abline(
  v = knots_cubic,
  lty = 3
)


# ------------------------------------------------------------------------------
# 20. Helper Function for Cubic Spline Basis
# ------------------------------------------------------------------------------

cubic_spline_basis <- function(
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


# ------------------------------------------------------------------------------
# 21. Verify Helper Function
# ------------------------------------------------------------------------------

X_check <- cubic_spline_basis(
  x,
  knots_cubic
)


max(
  abs(
    X_check -
      X_cubic_spline
  )
)


# ------------------------------------------------------------------------------
# 22. Generic Spline Fit
# ------------------------------------------------------------------------------

cubic_spline_fit <- function(
    x,
    y,
    knots
) {
  
  X <- cubic_spline_basis(
    x,
    knots
  )
  
  
  beta_hat <- solve(
    crossprod(
      X
    ),
    crossprod(
      X,
      y
    )
  )
  
  
  return(
    list(
      coefficients = beta_hat,
      knots = knots
    )
  )
}


# ------------------------------------------------------------------------------
# 23. Generic Spline Prediction
# ------------------------------------------------------------------------------

cubic_spline_predict <- function(
    x_new,
    model
) {
  
  X_new <- cubic_spline_basis(
    x_new,
    model$knots
  )
  
  
  prediction <- as.vector(
    X_new %*%
      model$coefficients
  )
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 24. Fit Model Using Helper
# ------------------------------------------------------------------------------

spline_model <- cubic_spline_fit(
  x = x,
  y = y,
  knots = knots_cubic
)


spline_prediction <- cubic_spline_predict(
  x,
  spline_model
)


max(
  abs(
    spline_prediction -
      y_hat_cubic_spline
  )
)


# ------------------------------------------------------------------------------
# 25. Compare Models
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Global Cubic Polynomial",
    "Unconstrained Piecewise Linear",
    "Linear Spline",
    "Linear Spline - Multiple Knots",
    "Cubic Spline"
  ),
  
  MSE = c(
    mean(
      (
        y -
          y_hat_poly
      )^2
    ),
    
    mean(
      (
        y -
          y_hat_piecewise
      )^2
    ),
    
    mean(
      (
        y -
          y_hat_linear_spline
      )^2
    ),
    
    mean(
      (
        y -
          y_hat_linear_multi
      )^2
    ),
    
    mean(
      (
        y -
          y_hat_cubic_spline
      )^2
    )
  )
)


model_comparison


# ------------------------------------------------------------------------------
# 26. Compare Fitted Curves
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Polynomial and Spline Fits"
)


lines(
  x,
  y_hat_poly,
  lty = 2,
  lwd = 2
)


lines(
  x,
  y_hat_linear_multi,
  lty = 3,
  lwd = 2
)


lines(
  x,
  y_hat_cubic_spline,
  lty = 1,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Cubic Spline",
    "Global Cubic Polynomial",
    "Linear Spline"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 27. Knot Placement Using Quantiles
# ------------------------------------------------------------------------------

# Instead of choosing knots manually, use empirical quantiles.


quantile_knots <- as.numeric(
  quantile(
    x,
    probs = c(
      0.25,
      0.5,
      0.75
    )
  )
)


quantile_knots


quantile_spline_model <- cubic_spline_fit(
  x,
  y,
  quantile_knots
)


quantile_prediction <- cubic_spline_predict(
  x,
  quantile_spline_model
)


# ------------------------------------------------------------------------------
# 28. Plot Quantile-Knot Spline
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Cubic Spline with Quantile Knots"
)


lines(
  x,
  quantile_prediction,
  lwd = 2
)


abline(
  v = quantile_knots,
  lty = 2
)


# ------------------------------------------------------------------------------
# 29. Effect of Number of Knots
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
    )[
      -c(
        1,
        number_knots + 2
      )
    ]
    
    
    knots_i <- as.numeric(
      quantile(
        x,
        probs = probabilities
      )
    )
  }
  
  
  model_i <- cubic_spline_fit(
    x,
    y,
    knots_i
  )
  
  
  prediction_i <- cubic_spline_predict(
    x,
    model_i
  )
  
  
  training_MSE[i] <- mean(
    (
      y -
        prediction_i
    )^2
  )
  
  
  models_by_knots[[i]] <- model_i
}


data.frame(
  Number_of_Knots = knot_counts,
  Training_MSE = training_MSE
)


# ------------------------------------------------------------------------------
# 30. Plot Training Error vs Number of Knots
# ------------------------------------------------------------------------------

plot(
  knot_counts,
  training_MSE,
  type = "b",
  pch = 19,
  xlab = "Number of Knots",
  ylab = "Training MSE",
  main = "Spline Complexity"
)


# ------------------------------------------------------------------------------
# 31. Train-Test Split
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
# 32. Compare Knot Counts on Test Data
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
    )[
      -c(
        1,
        number_knots + 2
      )
    ]
    
    
    # Knots determined using training data only
    
    knots_i <- as.numeric(
      quantile(
        x_train,
        probs = probabilities
      )
    )
  }
  
  
  model_i <- cubic_spline_fit(
    x_train,
    y_train,
    knots_i
  )
  
  
  prediction_i <- cubic_spline_predict(
    x_test,
    model_i
  )
  
  
  test_MSE[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


complexity_results <- data.frame(
  Number_of_Knots = knot_counts,
  Test_MSE = test_MSE
)


complexity_results


# ------------------------------------------------------------------------------
# 33. Select Number of Knots
# ------------------------------------------------------------------------------

best_index <- which.min(
  test_MSE
)


best_number_knots <- knot_counts[
  best_index
]


best_number_knots


# ------------------------------------------------------------------------------
# 34. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

spline_cv <- function(
    x,
    y,
    knot_counts,
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
        )[
          -c(
            1,
            number_knots + 2
          )
        ]
        
        
        knots_i <- as.numeric(
          quantile(
            x_train,
            probs = probabilities
          )
        )
      }
      
      
      model_i <- cubic_spline_fit(
        x_train,
        y_train,
        knots_i
      )
      
      
      prediction_i <- cubic_spline_predict(
        x_validation,
        model_i
      )
      
      
      CV_error[
        i,
        fold
      ] <- mean(
        (
          y_validation -
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
# 35. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


CV_results <- spline_cv(
  x = x,
  y = y,
  knot_counts = knot_counts,
  k = 5
)


CV_table <- data.frame(
  Number_of_Knots = CV_results$Knot_Count,
  CV_MSE = CV_results$Mean_Error
)


CV_table


# ------------------------------------------------------------------------------
# 36. Best Number of Knots by CV
# ------------------------------------------------------------------------------

best_CV_index <- which.min(
  CV_results$Mean_Error
)


best_CV_knots <- CV_results$Knot_Count[
  best_CV_index
]


best_CV_knots


# ------------------------------------------------------------------------------
# 37. Plot Cross-Validation Error
# ------------------------------------------------------------------------------

plot(
  CV_results$Knot_Count,
  CV_results$Mean_Error,
  type = "b",
  pch = 19,
  xlab = "Number of Knots",
  ylab = "Cross-Validated MSE",
  main = "Choosing Spline Complexity"
)


# ------------------------------------------------------------------------------
# 38. Fit Final Spline
# ------------------------------------------------------------------------------

if (
  best_CV_knots == 0
) {
  
  final_knots <- numeric(
    0
  )
  
} else {
  
  final_probabilities <- seq(
    0,
    1,
    length.out = best_CV_knots + 2
  )[
    -c(
      1,
      best_CV_knots + 2
    )
  ]
  
  
  final_knots <- as.numeric(
    quantile(
      x,
      probs = final_probabilities
    )
  )
}


final_model <- cubic_spline_fit(
  x,
  y,
  final_knots
)


final_prediction <- cubic_spline_predict(
  x,
  final_model
)


# ------------------------------------------------------------------------------
# 39. Plot Final Spline
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Final Cubic Regression Spline"
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
  ) > 0
) {
  
  abline(
    v = final_knots,
    lty = 3
  )
}


# ------------------------------------------------------------------------------
# 40. Verify Against Base R splines()
# ------------------------------------------------------------------------------

# splines is part of the standard R distribution.
# It is useful here only as a verification of the spline idea.

if (
  requireNamespace(
    "splines",
    quietly = TRUE
  )
) {
  
  builtin_model <- lm(
    y ~ splines::bs(
      x,
      knots = knots_cubic,
      degree = 3
    )
  )
  
  
  builtin_prediction <- predict(
    builtin_model
  )
  
  
  builtin_MSE <- mean(
    (
      y -
        builtin_prediction
    )^2
  )
  
  
  manual_MSE <- mean(
    (
      y -
        y_hat_cubic_spline
    )^2
  )
  
  
  data.frame(
    Method = c(
      "Manual Truncated-Power Cubic Spline",
      "Base R B-Spline"
    ),
    Training_MSE = c(
      manual_MSE,
      builtin_MSE
    )
  )
}


# ------------------------------------------------------------------------------
# 41. Summary
# ------------------------------------------------------------------------------

cat(
  "Global cubic polynomial MSE:",
  round(
    poly_MSE,
    4
  ),
  "\n"
)


cat(
  "Cubic spline MSE:",
  round(
    mean(
      (
        y -
          y_hat_cubic_spline
      )^2
    ),
    4
  ),
  "\n"
)


cat(
  "Best number of knots by CV:",
  best_CV_knots,
  "\n"
)


cat(
  "Final knot locations:",
  final_knots,
  "\n"
)