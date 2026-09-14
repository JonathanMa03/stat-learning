# ==============================================================================
# Cross Validation
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Training error vs test error
#   - Validation-set approach
#   - K-fold cross-validation
#   - Leave-one-out cross-validation
#   - Model selection
#   - Bias-variance tradeoff
#   - Data leakage
#   - Repeated cross-validation
#
# Example problem:
#
#   Choose the degree of a polynomial regression model.
#
# Candidate models:
#
#   degree = 1, 2, ..., 15
#
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Nonlinear Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 250


x <- runif(
  n,
  min = -3,
  max = 3
)


true_function <- function(x) {
  
  2 +
    1.5 * x -
    0.8 * x^2 +
    0.4 * x^3
}


y_true <- true_function(
  x
)


sigma <- 3


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
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


x_grid <- seq(
  min(x),
  max(x),
  length.out = 500
)


lines(
  x_grid,
  true_function(
    x_grid
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Polynomial Design Matrix
# ------------------------------------------------------------------------------

# Create:
#
# [1, x, x^2, ..., x^degree]


polynomial_design <- function(
    x,
    degree
) {
  
  X <- matrix(
    1,
    nrow = length(x),
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
      
      X[
        ,
        d + 1
      ] <- x^d
    }
  }
  
  
  colnames(
    X
  ) <- c(
    "Intercept",
    paste0(
      "x^",
      seq_len(
        degree
      )
    )
  )
  
  
  return(
    X
  )
}


# ------------------------------------------------------------------------------
# 4. Polynomial Regression Fit
# ------------------------------------------------------------------------------

polynomial_fit <- function(
    x,
    y,
    degree
) {
  
  X <- polynomial_design(
    x,
    degree
  )
  
  
  beta_hat <- qr.solve(
    X,
    y
  )
  
  
  return(
    list(
      coefficients = as.vector(
        beta_hat
      ),
      degree = degree
    )
  )
}


# ------------------------------------------------------------------------------
# 5. Polynomial Prediction
# ------------------------------------------------------------------------------

polynomial_predict <- function(
    model,
    x_new
) {
  
  X_new <- polynomial_design(
    x_new,
    model$degree
  )
  
  
  as.vector(
    X_new %*%
      model$coefficients
  )
}


# ------------------------------------------------------------------------------
# 6. Training Error Across Model Complexity
# ------------------------------------------------------------------------------

degrees <- 1:15


training_MSE <- numeric(
  length(
    degrees
  )
)


for (
  i in seq_along(
    degrees
  )
) {
  
  model_i <- polynomial_fit(
    x,
    y,
    degrees[i]
  )
  
  
  prediction_i <- polynomial_predict(
    model_i,
    x
  )
  
  
  training_MSE[i] <- mean(
    (
      y -
        prediction_i
    )^2
  )
}


# ------------------------------------------------------------------------------
# 7. Plot Training Error
# ------------------------------------------------------------------------------

plot(
  degrees,
  training_MSE,
  type = "b",
  pch = 19,
  xlab = "Polynomial Degree",
  ylab = "Training MSE",
  main = "Training Error"
)


# ------------------------------------------------------------------------------
# 8. Validation-Set Approach
# ------------------------------------------------------------------------------

# Split data once into:
#
# training set
# validation set


set.seed(123)


train_index <- sample(
  seq_len(n),
  size = floor(
    0.7 *
      n
  )
)


validation_index <- setdiff(
  seq_len(n),
  train_index
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


# ------------------------------------------------------------------------------
# 9. Validation Error Across Degrees
# ------------------------------------------------------------------------------

validation_MSE <- numeric(
  length(
    degrees
  )
)


for (
  i in seq_along(
    degrees
  )
) {
  
  model_i <- polynomial_fit(
    x_train,
    y_train,
    degrees[i]
  )
  
  
  prediction_i <- polynomial_predict(
    model_i,
    x_validation
  )
  
  
  validation_MSE[i] <- mean(
    (
      y_validation -
        prediction_i
    )^2
  )
}


# ------------------------------------------------------------------------------
# 10. Select Degree Using Validation Set
# ------------------------------------------------------------------------------

best_validation_index <- which.min(
  validation_MSE
)


best_validation_degree <- degrees[
  best_validation_index
]


best_validation_degree


# ------------------------------------------------------------------------------
# 11. Plot Validation Error
# ------------------------------------------------------------------------------

plot(
  degrees,
  validation_MSE,
  type = "b",
  pch = 19,
  xlab = "Polynomial Degree",
  ylab = "Validation MSE",
  main = "Validation-Set Model Selection"
)


abline(
  v = best_validation_degree,
  lty = 2
)


# ------------------------------------------------------------------------------
# 12. Why a Single Validation Split Can Be Unstable
# ------------------------------------------------------------------------------

# Repeat the random train-validation split many times.


number_repeats <- 50


validation_results <- matrix(
  NA,
  nrow = number_repeats,
  ncol = length(
    degrees
  )
)


selected_degrees <- numeric(
  number_repeats
)


for (
  repeat_id in seq_len(
    number_repeats
  )
) {
  
  set.seed(
    1000 +
      repeat_id
  )
  
  
  train_index <- sample(
    seq_len(n),
    size = floor(
      0.7 *
        n
    )
  )
  
  
  validation_index <- setdiff(
    seq_len(n),
    train_index
  )
  
  
  for (
    i in seq_along(
      degrees
    )
  ) {
    
    model_i <- polynomial_fit(
      x[
        train_index
      ],
      y[
        train_index
      ],
      degrees[i]
    )
    
    
    prediction_i <- polynomial_predict(
      model_i,
      x[
        validation_index
      ]
    )
    
    
    validation_results[
      repeat_id,
      i
    ] <- mean(
      (
        y[
          validation_index
        ] -
          prediction_i
      )^2
    )
  }
  
  
  selected_degrees[
    repeat_id
  ] <- degrees[
    which.min(
      validation_results[
        repeat_id,
      ]
    )
  ]
}


# ------------------------------------------------------------------------------
# 13. Distribution of Selected Degrees
# ------------------------------------------------------------------------------

table(
  selected_degrees
)


barplot(
  table(
    selected_degrees
  ),
  xlab = "Selected Degree",
  ylab = "Frequency",
  main = "Instability of a Single Validation Split"
)


# ------------------------------------------------------------------------------
# 14. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

k_fold_cv <- function(
    x,
    y,
    degrees,
    k = 10,
    seed = NULL
) {
  
  n <- length(
    y
  )
  
  
  if (
    !is.null(
      seed
    )
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  fold_id <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  fold_MSE <- matrix(
    NA,
    nrow = length(
      degrees
    ),
    ncol = k
  )
  
  
  for (
    fold in seq_len(
      k
    )
  ) {
    
    train_index <- which(
      fold_id != fold
    )
    
    
    validation_index <- which(
      fold_id == fold
    )
    
    
    for (
      i in seq_along(
        degrees
      )
    ) {
      
      model_i <- polynomial_fit(
        x[
          train_index
        ],
        y[
          train_index
        ],
        degrees[i]
      )
      
      
      prediction_i <- polynomial_predict(
        model_i,
        x[
          validation_index
        ]
      )
      
      
      fold_MSE[
        i,
        fold
      ] <- mean(
        (
          y[
            validation_index
          ] -
            prediction_i
        )^2
      )
    }
  }
  
  
  return(
    list(
      Degrees = degrees,
      Fold_MSE = fold_MSE,
      Mean_MSE = rowMeans(
        fold_MSE
      ),
      SE_MSE = apply(
        fold_MSE,
        1,
        sd
      ) /
        sqrt(k),
      Fold_ID = fold_id
    )
  )
}


# ------------------------------------------------------------------------------
# 15. Run 10-Fold Cross-Validation
# ------------------------------------------------------------------------------

CV_10 <- k_fold_cv(
  x = x,
  y = y,
  degrees = degrees,
  k = 10,
  seed = 123
)


CV_10_table <- data.frame(
  Degree = degrees,
  CV_MSE = CV_10$Mean_MSE,
  SE = CV_10$SE_MSE
)


CV_10_table


# ------------------------------------------------------------------------------
# 16. Best Degree by 10-Fold CV
# ------------------------------------------------------------------------------

best_CV_index <- which.min(
  CV_10$Mean_MSE
)


best_CV_degree <- degrees[
  best_CV_index
]


best_CV_degree


# ------------------------------------------------------------------------------
# 17. Plot 10-Fold CV Error
# ------------------------------------------------------------------------------

plot(
  degrees,
  CV_10$Mean_MSE,
  type = "b",
  pch = 19,
  ylim = range(
    CV_10$Mean_MSE -
      CV_10$SE_MSE,
    CV_10$Mean_MSE +
      CV_10$SE_MSE
  ),
  xlab = "Polynomial Degree",
  ylab = "Cross-Validated MSE",
  main = "10-Fold Cross-Validation"
)


arrows(
  x0 = degrees,
  y0 = CV_10$Mean_MSE -
    CV_10$SE_MSE,
  x1 = degrees,
  y1 = CV_10$Mean_MSE +
    CV_10$SE_MSE,
  angle = 90,
  code = 3,
  length = 0.03
)


abline(
  v = best_CV_degree,
  lty = 2
)


# ------------------------------------------------------------------------------
# 18. One-Standard-Error Rule
# ------------------------------------------------------------------------------

# Instead of always choosing the model with the minimum CV error,
# choose the simplest model whose CV error is within one standard error
# of the minimum.


minimum_error <- CV_10$Mean_MSE[
  best_CV_index
]


minimum_SE <- CV_10$SE_MSE[
  best_CV_index
]


one_SE_threshold <- minimum_error +
  minimum_SE


eligible_indices <- which(
  CV_10$Mean_MSE <=
    one_SE_threshold
)


one_SE_degree <- min(
  degrees[
    eligible_indices
  ]
)


one_SE_degree


# ------------------------------------------------------------------------------
# 19. Plot One-SE Rule
# ------------------------------------------------------------------------------

plot(
  degrees,
  CV_10$Mean_MSE,
  type = "b",
  pch = 19,
  xlab = "Polynomial Degree",
  ylab = "Cross-Validated MSE",
  main = "One-Standard-Error Rule"
)


abline(
  h = one_SE_threshold,
  lty = 2
)


abline(
  v = best_CV_degree,
  lty = 3
)


abline(
  v = one_SE_degree,
  lty = 4
)


legend(
  "topright",
  legend = c(
    "Minimum CV Degree",
    "One-SE Degree",
    "One-SE Threshold"
  ),
  lty = c(
    3,
    4,
    2
  )
)


# ------------------------------------------------------------------------------
# 20. Leave-One-Out Cross-Validation
# ------------------------------------------------------------------------------

# LOOCV:
#
# train on n - 1 observations
# test on the remaining observation
#
# repeat for every observation.


loocv <- function(
    x,
    y,
    degrees
) {
  
  n <- length(
    y
  )
  
  
  errors <- matrix(
    NA,
    nrow = length(
      degrees
    ),
    ncol = n
  )
  
  
  for (
    observation in seq_len(
      n
    )
  ) {
    
    train_index <- setdiff(
      seq_len(n),
      observation
    )
    
    
    for (
      i in seq_along(
        degrees
      )
    ) {
      
      model_i <- polynomial_fit(
        x[
          train_index
        ],
        y[
          train_index
        ],
        degrees[i]
      )
      
      
      prediction_i <- polynomial_predict(
        model_i,
        x[
          observation
        ]
      )
      
      
      errors[
        i,
        observation
      ] <- (
        y[
          observation
        ] -
          prediction_i
      )^2
    }
  }
  
  
  return(
    rowMeans(
      errors
    )
  )
}


# ------------------------------------------------------------------------------
# 21. Run LOOCV
# ------------------------------------------------------------------------------

LOOCV_MSE <- loocv(
  x,
  y,
  degrees
)


best_LOOCV_degree <- degrees[
  which.min(
    LOOCV_MSE
  )
]


best_LOOCV_degree


# ------------------------------------------------------------------------------
# 22. Plot LOOCV Error
# ------------------------------------------------------------------------------

plot(
  degrees,
  LOOCV_MSE,
  type = "b",
  pch = 19,
  xlab = "Polynomial Degree",
  ylab = "LOOCV MSE",
  main = "Leave-One-Out Cross-Validation"
)


abline(
  v = best_LOOCV_degree,
  lty = 2
)


# ------------------------------------------------------------------------------
# 23. Compare Different Values of K
# ------------------------------------------------------------------------------

K_values <- c(
  2,
  5,
  10,
  20
)


K_results <- matrix(
  NA,
  nrow = length(
    K_values
  ),
  ncol = length(
    degrees
  )
)


for (
  k_index in seq_along(
    K_values
  )
) {
  
  result <- k_fold_cv(
    x,
    y,
    degrees,
    k = K_values[
      k_index
    ],
    seed = 123
  )
  
  
  K_results[
    k_index,
  ] <- result$Mean_MSE
}


rownames(
  K_results
) <- paste0(
  "K=",
  K_values
)


colnames(
  K_results
) <- paste0(
  "Degree",
  degrees
)


# ------------------------------------------------------------------------------
# 24. Plot CV Curves for Different K
# ------------------------------------------------------------------------------

matplot(
  degrees,
  t(
    K_results
  ),
  type = "l",
  lty = 1:length(
    K_values
  ),
  xlab = "Polynomial Degree",
  ylab = "Cross-Validated MSE",
  main = "Effect of K in K-Fold CV"
)


legend(
  "topright",
  legend = paste0(
    "K = ",
    K_values
  ),
  lty = 1:length(
    K_values
  )
)


# ------------------------------------------------------------------------------
# 25. Selected Degree for Different K
# ------------------------------------------------------------------------------

selected_by_K <- data.frame(
  K = K_values,
  Selected_Degree = apply(
    K_results,
    1,
    function(error) {
      
      degrees[
        which.min(
          error
        )
      ]
    }
  )
)


selected_by_K


# ------------------------------------------------------------------------------
# 26. Repeated K-Fold Cross-Validation
# ------------------------------------------------------------------------------

# K-fold CV itself depends on the random fold assignment.
#
# Repeating CV and averaging reduces this randomness.


repeated_k_fold_cv <- function(
    x,
    y,
    degrees,
    k = 10,
    repeats = 20
) {
  
  all_errors <- matrix(
    NA,
    nrow = length(
      degrees
    ),
    ncol = repeats
  )
  
  
  for (
    r in seq_len(
      repeats
    )
  ) {
    
    result <- k_fold_cv(
      x,
      y,
      degrees,
      k = k,
      seed = 1000 +
        r
    )
    
    
    all_errors[
      ,
      r
    ] <- result$Mean_MSE
  }
  
  
  return(
    list(
      Mean_MSE = rowMeans(
        all_errors
      ),
      SD_MSE = apply(
        all_errors,
        1,
        sd
      ),
      All_MSE = all_errors
    )
  )
}


# ------------------------------------------------------------------------------
# 27. Run Repeated 10-Fold CV
# ------------------------------------------------------------------------------

repeated_CV <- repeated_k_fold_cv(
  x,
  y,
  degrees,
  k = 10,
  repeats = 30
)


best_repeated_degree <- degrees[
  which.min(
    repeated_CV$Mean_MSE
  )
]


best_repeated_degree


# ------------------------------------------------------------------------------
# 28. Plot Repeated CV
# ------------------------------------------------------------------------------

plot(
  degrees,
  repeated_CV$Mean_MSE,
  type = "b",
  pch = 19,
  xlab = "Polynomial Degree",
  ylab = "Repeated CV MSE",
  main = "Repeated 10-Fold Cross-Validation"
)


# ------------------------------------------------------------------------------
# 29. Compare Model-Selection Methods
# ------------------------------------------------------------------------------

selection_comparison <- data.frame(
  Method = c(
    "Validation Set",
    "10-Fold CV",
    "10-Fold One-SE Rule",
    "LOOCV",
    "Repeated 10-Fold CV"
  ),
  
  Selected_Degree = c(
    best_validation_degree,
    best_CV_degree,
    one_SE_degree,
    best_LOOCV_degree,
    best_repeated_degree
  )
)


selection_comparison


# ------------------------------------------------------------------------------
# 30. Independent Test Set
# ------------------------------------------------------------------------------

# Cross-validation estimates test performance.
#
# For simulation, we can generate a completely independent test set
# and see how well CV performs.


set.seed(999)


n_test <- 10000


x_test <- runif(
  n_test,
  min = -3,
  max = 3
)


y_test <- true_function(
  x_test
) +
  rnorm(
    n_test,
    mean = 0,
    sd = sigma
  )


# ------------------------------------------------------------------------------
# 31. True Test Error Across Degrees
# ------------------------------------------------------------------------------

test_MSE <- numeric(
  length(
    degrees
  )
)


for (
  i in seq_along(
    degrees
  )
) {
  
  model_i <- polynomial_fit(
    x,
    y,
    degrees[i]
  )
  
  
  prediction_i <- polynomial_predict(
    model_i,
    x_test
  )
  
  
  test_MSE[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


# ------------------------------------------------------------------------------
# 32. Best Degree According to Actual Test Error
# ------------------------------------------------------------------------------

best_test_degree <- degrees[
  which.min(
    test_MSE
  )
]


best_test_degree


# ------------------------------------------------------------------------------
# 33. Compare Training, CV, and Test Error
# ------------------------------------------------------------------------------

plot(
  degrees,
  training_MSE,
  type = "l",
  lwd = 2,
  ylim = range(
    training_MSE,
    CV_10$Mean_MSE,
    test_MSE
  ),
  xlab = "Polynomial Degree",
  ylab = "MSE",
  main = "Training vs CV vs Test Error"
)


lines(
  degrees,
  CV_10$Mean_MSE,
  lty = 2,
  lwd = 2
)


lines(
  degrees,
  test_MSE,
  lty = 3,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Training Error",
    "10-Fold CV Error",
    "True Test Error"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 34. Compare CV Estimate and True Test Error
# ------------------------------------------------------------------------------

comparison_table <- data.frame(
  Degree = degrees,
  Training_MSE = training_MSE,
  CV_MSE = CV_10$Mean_MSE,
  LOOCV_MSE = LOOCV_MSE,
  Test_MSE = test_MSE
)


comparison_table


# ------------------------------------------------------------------------------
# 35. Fit Final Model
# ------------------------------------------------------------------------------

# Once model complexity is selected, refit the model using ALL available
# training observations.


final_model <- polynomial_fit(
  x,
  y,
  degree = best_CV_degree
)


final_grid_prediction <- polynomial_predict(
  final_model,
  x_grid
)


# ------------------------------------------------------------------------------
# 36. Plot Final Selected Model
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = paste(
    "Final Model: Degree",
    best_CV_degree
  )
)


lines(
  x_grid,
  final_grid_prediction,
  lwd = 2
)


lines(
  x_grid,
  true_function(
    x_grid
  ),
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "CV-Selected Model",
    "True Function"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 37. Data Leakage Example
# ------------------------------------------------------------------------------

# Cross-validation must include every data-dependent preprocessing step
# inside each training fold.
#
# Example:
#
# WRONG:
#
#   1. Standardize using all observations.
#   2. Perform CV.
#
# Information from validation observations entered the training process.
#
#
# CORRECT:
#
#   for each fold:
#
#       compute means / SDs on training fold
#       transform training fold
#       transform validation fold using training statistics
#       fit model
#       evaluate validation observations


# ------------------------------------------------------------------------------
# 38. General K-Fold CV Template
# ------------------------------------------------------------------------------

# This is the generic structure to reuse for other methods.


general_k_fold_cv <- function(
    X,
    y,
    k,
    fit_function,
    predict_function,
    seed = NULL
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
  
  
  if (
    !is.null(
      seed
    )
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  folds <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  fold_error <- numeric(
    k
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
    
    
    model <- fit_function(
      X[
        train_index,
        ,
        drop = FALSE
      ],
      y[
        train_index
      ]
    )
    
    
    prediction <- predict_function(
      model,
      X[
        validation_index,
        ,
        drop = FALSE
      ]
    )
    
    
    fold_error[
      fold
    ] <- mean(
      (
        y[
          validation_index
        ] -
          prediction
      )^2
    )
  }
  
  
  return(
    list(
      Fold_Error = fold_error,
      Mean_Error = mean(
        fold_error
      ),
      SE = sd(
        fold_error
      ) /
        sqrt(k)
    )
  )
}


# ------------------------------------------------------------------------------
# 39. K-Fold Geometry
# ------------------------------------------------------------------------------

# Example of how observations are assigned to folds.


set.seed(123)


example_folds <- sample(
  rep(
    1:5,
    length.out = n
  )
)


table(
  example_folds
)


plot(
  seq_len(n),
  example_folds,
  pch = 19,
  xlab = "Observation",
  ylab = "Fold",
  main = "Example 5-Fold Assignment"
)


# ------------------------------------------------------------------------------
# 40. Summary
# ------------------------------------------------------------------------------

cat(
  "Cross Validation Summary\n"
)


cat(
  "------------------------\n"
)


cat(
  "Validation-set selected degree:",
  best_validation_degree,
  "\n"
)


cat(
  "10-fold CV selected degree:",
  best_CV_degree,
  "\n"
)


cat(
  "One-SE selected degree:",
  one_SE_degree,
  "\n"
)


cat(
  "LOOCV selected degree:",
  best_LOOCV_degree,
  "\n"
)


cat(
  "Repeated 10-fold CV selected degree:",
  best_repeated_degree,
  "\n"
)


cat(
  "True test-error minimizing degree:",
  best_test_degree,
  "\n"
)