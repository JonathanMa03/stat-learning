# ==============================================================================
# Boosting
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Boosting
#   - Sequential additive modeling
#   - Weak learners
#   - Regression stumps
#   - Residual fitting
#   - Learning rate / shrinkage
#   - Number of boosting iterations
#   - Stagewise fitting
#   - Training and test error
#   - Variable importance
#   - Bias-variance tradeoff
#
#
# General additive boosting model:
#
#   F_M(x)
#   =
#   F_0(x)
#   +
#   sum_{m=1}^M
#       nu * h_m(x)
#
# where:
#
#   F_0(x) = initial prediction
#   h_m(x) = weak learner fitted at boosting iteration m
#   nu     = learning rate
#
#
# For squared-error loss:
#
#   L(y, F(x)) = 1/2 * (y - F(x))^2
#
# the negative gradient is simply:
#
#   y - F(x)
#
# Therefore each new learner is fitted to the current residuals.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 500


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


# True function combines:
#
#   - nonlinear x1 effect
#   - threshold effect in x2
#   - smooth x3 effect


true_function <- function(
    x1,
    x2,
    x3
) {
  
  2 +
    2 * sin(x1) +
    ifelse(
      x2 > 0.5,
      2.5,
      -1
    ) +
    0.75 * x3^2
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
  y = y,
  x1 = x1,
  x2 = x2,
  x3 = x3
)


# ------------------------------------------------------------------------------
# 2. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data,
  main = "Boosting Regression Data"
)


# ==============================================================================
# PART II
#
# REGRESSION STUMPS
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Residual Sum of Squares
# ------------------------------------------------------------------------------

node_rss <- function(y) {
  
  if (length(y) == 0) {
    return(0)
  }
  
  
  sum(
    (
      y -
        mean(y)
    )^2
  )
}


# ------------------------------------------------------------------------------
# 4. Candidate Split Points
# ------------------------------------------------------------------------------

candidate_splits <- function(
    x,
    maximum_splits = 50
) {
  
  unique_values <- sort(
    unique(
      x
    )
  )
  
  
  if (
    length(unique_values) <
    2
  ) {
    
    return(
      numeric(0)
    )
  }
  
  
  all_splits <- (
    unique_values[
      -length(
        unique_values
      )
    ] +
      unique_values[
        -1
      ]
  ) / 2
  
  
  # If there are many possible splits, retain an evenly spaced subset.
  # This keeps the demo computationally manageable.
  
  if (
    length(all_splits) >
    maximum_splits
  ) {
    
    indices <- round(
      seq(
        1,
        length(all_splits),
        length.out = maximum_splits
      )
    )
    
    
    all_splits <- all_splits[
      unique(
        indices
      )
    ]
  }
  
  
  return(
    all_splits
  )
}


# ------------------------------------------------------------------------------
# 5. Fit One Regression Stump
# ------------------------------------------------------------------------------

# A regression stump has one split:
#
#               X_j <= s ?
#               /        \
#          left mean   right mean
#
# We choose j and s to minimize squared error.


fit_regression_stump <- function(
    X,
    y,
    min_leaf = 5,
    maximum_splits = 50
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  best_RSS <- Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  best_left_value <- NA_real_
  
  best_right_value <- NA_real_
  
  
  for (
    j in seq_len(p)
  ) {
    
    splits <- candidate_splits(
      X[, j],
      maximum_splits = maximum_splits
    )
    
    
    if (
      length(splits) == 0
    ) {
      
      next
    }
    
    
    for (
      split_value in splits
    ) {
      
      left <- X[, j] <=
        split_value
      
      
      right <- !left
      
      
      if (
        sum(left) <
        min_leaf ||
        sum(right) <
        min_leaf
      ) {
        
        next
      }
      
      
      left_value <- mean(
        y[
          left
        ]
      )
      
      
      right_value <- mean(
        y[
          right
        ]
      )
      
      
      prediction <- ifelse(
        left,
        left_value,
        right_value
      )
      
      
      RSS <- sum(
        (
          y -
            prediction
        )^2
      )
      
      
      if (
        RSS <
        best_RSS
      ) {
        
        best_RSS <- RSS
        
        best_variable <- j
        
        best_split <- split_value
        
        best_left_value <- left_value
        
        best_right_value <- right_value
      }
    }
  }
  
  
  if (
    is.na(
      best_variable
    )
  ) {
    
    stop(
      "No valid stump could be fitted."
    )
  }
  
  
  return(
    list(
      variable = best_variable,
      split = best_split,
      left_value = best_left_value,
      right_value = best_right_value,
      RSS = best_RSS
    )
  )
}


# ------------------------------------------------------------------------------
# 6. Predict with One Stump
# ------------------------------------------------------------------------------

predict_stump <- function(
    stump,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  ifelse(
    X_new[, stump$variable] <=
      stump$split,
    stump$left_value,
    stump$right_value
  )
}


# ==============================================================================
# PART III
#
# INSPECT ONE WEAK LEARNER
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Fit One Stump to y
# ------------------------------------------------------------------------------

single_stump <- fit_regression_stump(
  X = X,
  y = y,
  min_leaf = 10
)


single_stump


colnames(X)[
  single_stump$variable
]


# ------------------------------------------------------------------------------
# 8. Single-Stump Predictions
# ------------------------------------------------------------------------------

single_stump_prediction <- predict_stump(
  single_stump,
  X
)


single_stump_MSE <- mean(
  (
    y -
      single_stump_prediction
  )^2
)


single_stump_MSE


# ==============================================================================
# PART IV
#
# MANUAL BOOSTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Squared-Error Boosting Algorithm
# ------------------------------------------------------------------------------

boost_regression <- function(
    X,
    y,
    M = 200,
    learning_rate = 0.05,
    min_leaf = 5,
    maximum_splits = 50,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Initial model
  # --------------------------------------------------------------------------
  
  initial_prediction <- mean(
    y
  )
  
  
  fitted_values <- rep(
    initial_prediction,
    n
  )
  
  
  # --------------------------------------------------------------------------
  # Storage
  # --------------------------------------------------------------------------
  
  learners <- vector(
    "list",
    M
  )
  
  
  training_MSE <- numeric(
    M
  )
  
  
  residual_MSE <- numeric(
    M
  )
  
  
  improvement <- numeric(
    M
  )
  
  
  previous_MSE <- mean(
    (
      y -
        fitted_values
    )^2
  )
  
  
  # ==========================================================================
  # BOOSTING LOOP
  # ==========================================================================
  
  for (
    m in seq_len(M)
  ) {
    
    # ------------------------------------------------------------------------
    # Current residuals
    #
    # For squared-error loss these are also the negative gradients.
    # ------------------------------------------------------------------------
    
    residuals <- y -
      fitted_values
    
    
    residual_MSE[m] <- mean(
      residuals^2
    )
    
    
    # ------------------------------------------------------------------------
    # Fit weak learner to residuals
    # ------------------------------------------------------------------------
    
    learner <- fit_regression_stump(
      X = X,
      y = residuals,
      min_leaf = min_leaf,
      maximum_splits = maximum_splits
    )
    
    
    learner_prediction <- predict_stump(
      learner,
      X
    )
    
    
    # ------------------------------------------------------------------------
    # Stagewise update
    # ------------------------------------------------------------------------
    
    fitted_values <-
      fitted_values +
      learning_rate *
      learner_prediction
    
    
    learners[[m]] <- learner
    
    
    current_MSE <- mean(
      (
        y -
          fitted_values
      )^2
    )
    
    
    training_MSE[m] <- current_MSE
    
    
    improvement[m] <-
      previous_MSE -
      current_MSE
    
    
    previous_MSE <- current_MSE
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        m,
        "| Variable:",
        colnames(X)[
          learner$variable
        ],
        "| Split:",
        round(
          learner$split,
          3
        ),
        "| Training MSE:",
        round(
          current_MSE,
          4
        ),
        "\n"
      )
    }
  }
  
  
  return(
    list(
      initial_prediction =
        initial_prediction,
      learners = learners,
      learning_rate =
        learning_rate,
      fitted_values =
        fitted_values,
      training_MSE =
        training_MSE,
      residual_MSE =
        residual_MSE,
      improvement =
        improvement,
      feature_names =
        colnames(X)
    )
  )
}


# ------------------------------------------------------------------------------
# 10. Fit Boosted Model
# ------------------------------------------------------------------------------

boost_model <- boost_regression(
  X = X,
  y = y,
  M = 300,
  learning_rate = 0.05,
  min_leaf = 10,
  maximum_splits = 50,
  verbose = FALSE
)


# ==============================================================================
# PART V
#
# BOOSTING CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Training Error
# ------------------------------------------------------------------------------

boost_training_MSE <- mean(
  (
    y -
      boost_model$fitted_values
  )^2
)


boost_training_RMSE <- sqrt(
  boost_training_MSE
)


boost_training_MSE


boost_training_RMSE


# ------------------------------------------------------------------------------
# 12. Plot Training Error Across Iterations
# ------------------------------------------------------------------------------

plot(
  seq_along(
    boost_model$training_MSE
  ),
  boost_model$training_MSE,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Training MSE",
  main = "Boosting Training Error"
)


# ------------------------------------------------------------------------------
# 13. Plot Improvement Per Iteration
# ------------------------------------------------------------------------------

plot(
  seq_along(
    boost_model$improvement
  ),
  boost_model$improvement,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Decrease in Training MSE",
  main = "Stagewise Improvement"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART VI
#
# HOW BOOSTING BUILDS THE FIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Prediction Function
# ------------------------------------------------------------------------------

predict_boosting <- function(
    model,
    X_new,
    M = length(
      model$learners
    )
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  M <- min(
    M,
    length(
      model$learners
    )
  )
  
  
  prediction <- rep(
    model$initial_prediction,
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
      model$learning_rate *
      predict_stump(
        model$learners[[m]],
        X_new
      )
  }
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 15. Verify Training Predictions
# ------------------------------------------------------------------------------

prediction_check <- predict_boosting(
  boost_model,
  X
)


max(
  abs(
    prediction_check -
      boost_model$fitted_values
  )
)


# ------------------------------------------------------------------------------
# 16. Predictions at Different Boosting Stages
# ------------------------------------------------------------------------------

iteration_values <- c(
  1,
  5,
  20,
  100,
  300
)


stage_predictions <- lapply(
  iteration_values,
  function(M) {
    
    predict_boosting(
      boost_model,
      X,
      M = M
    )
  }
)


stage_MSE <- sapply(
  stage_predictions,
  function(prediction) {
    
    mean(
      (
        y -
          prediction
      )^2
    )
  }
)


data.frame(
  Iterations = iteration_values,
  Training_MSE = stage_MSE
)


# ==============================================================================
# PART VII
#
# VISUALIZE STAGEWISE LEARNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. One-Dimensional Slice
# ------------------------------------------------------------------------------

x1_grid <- seq(
  -3,
  3,
  length.out = 500
)


slice_data <- cbind(
  x1 = x1_grid,
  x2 = 0,
  x3 = 0
)


true_slice <- true_function(
  x1_grid,
  0,
  0
)


# ------------------------------------------------------------------------------
# 18. Plot Predictions as Boosting Proceeds
# ------------------------------------------------------------------------------

plot(
  x1_grid,
  true_slice,
  type = "l",
  lwd = 3,
  xlab = "x1",
  ylab = "Predicted y",
  main = "Stagewise Boosting"
)


for (
  i in seq_along(
    iteration_values
  )
) {
  
  prediction_i <- predict_boosting(
    boost_model,
    slice_data,
    M = iteration_values[i]
  )
  
  
  lines(
    x1_grid,
    prediction_i,
    lty = i,
    lwd = 2
  )
}


legend(
  "topleft",
  legend = c(
    "True Function",
    paste0(
      "M = ",
      iteration_values
    )
  ),
  lty = c(
    1,
    seq_along(
      iteration_values
    )
  ),
  lwd = c(
    3,
    rep(
      2,
      length(
        iteration_values
      )
    )
  )
)


# ==============================================================================
# PART VIII
#
# RESIDUAL FITTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Examine Residuals at Different Iterations
# ------------------------------------------------------------------------------

residual_iteration_values <- c(
  0,
  1,
  10,
  50,
  300
)


residual_summary <- data.frame(
  Iteration =
    residual_iteration_values,
  Residual_Mean = NA_real_,
  Residual_SD = NA_real_,
  Residual_MSE = NA_real_
)


for (
  i in seq_along(
    residual_iteration_values
  )
) {
  
  M <- residual_iteration_values[i]
  
  
  prediction_M <- predict_boosting(
    boost_model,
    X,
    M = M
  )
  
  
  residual_M <- y -
    prediction_M
  
  
  residual_summary$Residual_Mean[i] <-
    mean(
      residual_M
    )
  
  
  residual_summary$Residual_SD[i] <-
    sd(
      residual_M
    )
  
  
  residual_summary$Residual_MSE[i] <-
    mean(
      residual_M^2
    )
}


residual_summary


# ==============================================================================
# PART IX
#
# VARIABLE USAGE AND IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Count Variable Usage
# ------------------------------------------------------------------------------

selected_variables <- sapply(
  boost_model$learners,
  function(learner) {
    
    learner$variable
  }
)


variable_usage <- table(
  factor(
    selected_variables,
    levels = seq_len(
      ncol(X)
    ),
    labels = colnames(X)
  )
)


variable_usage


barplot(
  variable_usage,
  ylab = "Number of Selected Stumps",
  main = "Boosting Variable Usage"
)


# ------------------------------------------------------------------------------
# 21. RSS-Reduction Importance
# ------------------------------------------------------------------------------

# At each iteration, the weak learner reduces training error.
#
# Allocate that reduction to the variable used by the stump.


boost_variable_importance <- function(
    model
) {
  
  p <- length(
    model$feature_names
  )
  
  
  importance <- numeric(
    p
  )
  
  
  for (
    m in seq_along(
      model$learners
    )
  ) {
    
    variable <- model$learners[[m]]$variable
    
    
    importance[variable] <-
      importance[variable] +
      max(
        model$improvement[m],
        0
      )
  }
  
  
  names(
    importance
  ) <- model$feature_names
  
  
  if (
    sum(importance) >
    0
  ) {
    
    importance <-
      importance /
      sum(
        importance
      )
  }
  
  
  return(
    importance
  )
}


variable_importance <- boost_variable_importance(
  boost_model
)


variable_importance


barplot(
  variable_importance,
  ylab = "Relative Importance",
  main = "Boosting Variable Importance"
)


# ==============================================================================
# PART X
#
# TRAIN-TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Train-Test Split
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
# 23. Fit Boosting on Training Data
# ------------------------------------------------------------------------------

boost_train <- boost_regression(
  X = X_train,
  y = y_train,
  M = 500,
  learning_rate = 0.05,
  min_leaf = 8,
  maximum_splits = 50
)


# ==============================================================================
# PART XI
#
# TEST ERROR AS BOOSTING CONTINUES
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Evaluate Every Boosting Iteration
# ------------------------------------------------------------------------------

maximum_iterations <- length(
  boost_train$learners
)


train_MSE_path <- numeric(
  maximum_iterations
)


test_MSE_path <- numeric(
  maximum_iterations
)


for (
  M in seq_len(
    maximum_iterations
  )
) {
  
  train_prediction_M <- predict_boosting(
    boost_train,
    X_train,
    M = M
  )
  
  
  test_prediction_M <- predict_boosting(
    boost_train,
    X_test,
    M = M
  )
  
  
  train_MSE_path[M] <- mean(
    (
      y_train -
        train_prediction_M
    )^2
  )
  
  
  test_MSE_path[M] <- mean(
    (
      y_test -
        test_prediction_M
    )^2
  )
}


# ------------------------------------------------------------------------------
# 25. Best Test Iteration
# ------------------------------------------------------------------------------

best_test_iteration <- which.min(
  test_MSE_path
)


best_test_iteration


min(
  test_MSE_path
)


# NOTE:
#
# This is an ORACLE diagnostic because the test set should not normally be
# used to select the stopping iteration.
#
# Cross-validation is used later for proper model selection.


# ------------------------------------------------------------------------------
# 26. Plot Train and Test Error
# ------------------------------------------------------------------------------

matplot(
  seq_len(
    maximum_iterations
  ),
  cbind(
    train_MSE_path,
    test_MSE_path
  ),
  type = "l",
  lty = c(
    1,
    2
  ),
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "MSE",
  main = "Boosting Bias-Variance Tradeoff"
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
  lwd = 2
)


# ==============================================================================
# PART XII
#
# LEARNING RATE / SHRINKAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Compare Learning Rates
# ------------------------------------------------------------------------------

learning_rates <- c(
  1,
  0.25,
  0.10,
  0.05,
  0.01
)


learning_rate_results <- data.frame(
  Learning_Rate =
    learning_rates,
  Training_MSE = NA_real_,
  Test_MSE = NA_real_
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
  
  model_i <- boost_regression(
    X = X_train,
    y = y_train,
    M = 300,
    learning_rate =
      learning_rates[i],
    min_leaf = 8,
    maximum_splits = 50
  )
  
  
  train_prediction_i <- predict_boosting(
    model_i,
    X_train
  )
  
  
  test_prediction_i <- predict_boosting(
    model_i,
    X_test
  )
  
  
  learning_rate_results$Training_MSE[i] <-
    mean(
      (
        y_train -
          train_prediction_i
      )^2
    )
  
  
  learning_rate_results$Test_MSE[i] <-
    mean(
      (
        y_test -
          test_prediction_i
      )^2
    )
  
  
  learning_rate_models[[i]] <-
    model_i
}


learning_rate_results


# ------------------------------------------------------------------------------
# 28. Plot Learning-Rate Comparison
# ------------------------------------------------------------------------------

plot(
  learning_rate_results$Learning_Rate,
  learning_rate_results$Test_MSE,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Learning Rate",
  ylab = "Test MSE",
  main = "Boosting Shrinkage"
)


# ==============================================================================
# PART XIII
#
# MANUAL CROSS-VALIDATION FOR NUMBER OF ITERATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. K-Fold Cross Validation
# ------------------------------------------------------------------------------

boosting_cv <- function(
    X,
    y,
    M = 300,
    learning_rate = 0.05,
    k = 5,
    min_leaf = 5,
    maximum_splits = 40,
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
  
  
  fold_MSE <- matrix(
    NA_real_,
    nrow = M,
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
    
    
    model_fold <- boost_regression(
      X = X[
        training_indices,
        ,
        drop = FALSE
      ],
      y = y[
        training_indices
      ],
      M = M,
      learning_rate =
        learning_rate,
      min_leaf =
        min_leaf,
      maximum_splits =
        maximum_splits
    )
    
    
    # Instead of recomputing from scratch for every M,
    # update validation predictions sequentially.
    
    validation_prediction <- rep(
      model_fold$initial_prediction,
      length(
        validation_indices
      )
    )
    
    
    for (
      m in seq_len(M)
    ) {
      
      validation_prediction <-
        validation_prediction +
        learning_rate *
        predict_stump(
          model_fold$learners[[m]],
          X[
            validation_indices,
            ,
            drop = FALSE
          ]
        )
      
      
      fold_MSE[
        m,
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
  
  
  mean_CV_MSE <- rowMeans(
    fold_MSE
  )
  
  
  SE_CV_MSE <- apply(
    fold_MSE,
    1,
    sd
  ) /
    sqrt(k)
  
  
  return(
    list(
      mean = mean_CV_MSE,
      SE = SE_CV_MSE,
      fold_MSE = fold_MSE
    )
  )
}


# ------------------------------------------------------------------------------
# 30. Run Cross Validation
# ------------------------------------------------------------------------------

CV_result <- boosting_cv(
  X = X_train,
  y = y_train,
  M = 300,
  learning_rate = 0.05,
  k = 5,
  min_leaf = 8,
  maximum_splits = 40,
  seed = 123
)


# ------------------------------------------------------------------------------
# 31. Best Boosting Iteration
# ------------------------------------------------------------------------------

best_CV_iteration <- which.min(
  CV_result$mean
)


best_CV_iteration


CV_result$mean[
  best_CV_iteration
]


# ------------------------------------------------------------------------------
# 32. Plot CV Error
# ------------------------------------------------------------------------------

plot(
  seq_along(
    CV_result$mean
  ),
  CV_result$mean,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Cross-Validated MSE",
  main = "Boosting Cross Validation"
)


# ==============================================================================
# PART XIV
#
# ONE-STANDARD-ERROR RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Calculate One-SE Iteration
# ------------------------------------------------------------------------------

minimum_CV_error <- CV_result$mean[
  best_CV_iteration
]


minimum_CV_SE <- CV_result$SE[
  best_CV_iteration
]


one_SE_threshold <-
  minimum_CV_error +
  minimum_CV_SE


eligible_iterations <- which(
  CV_result$mean <=
    one_SE_threshold
)


one_SE_iteration <- min(
  eligible_iterations
)


one_SE_iteration


# ------------------------------------------------------------------------------
# 34. Display Selection
# ------------------------------------------------------------------------------

abline(
  h = one_SE_threshold,
  lty = 2
)


abline(
  v = best_CV_iteration,
  lty = 3
)


abline(
  v = one_SE_iteration,
  lty = 4
)


# ==============================================================================
# PART XV
#
# FINAL BOOSTED MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Predict Using CV-Selected Number of Iterations
# ------------------------------------------------------------------------------

final_test_prediction <- predict_boosting(
  boost_train,
  X_test,
  M = one_SE_iteration
)


final_train_prediction <- predict_boosting(
  boost_train,
  X_train,
  M = one_SE_iteration
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
# PART XVI
#
# COMPARE WITH SINGLE TREE STUMP
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Fit One Stump on Training Data
# ------------------------------------------------------------------------------

training_stump <- fit_regression_stump(
  X = X_train,
  y = y_train,
  min_leaf = 8
)


stump_test_prediction <- predict_stump(
  training_stump,
  X_test
)


stump_test_MSE <- mean(
  (
    y_test -
      stump_test_prediction
  )^2
)


# ==============================================================================
# PART XVII
#
# COMPARE WITH LINEAR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Linear Model
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = y_train,
  x1 = X_train[, "x1"],
  x2 = X_train[, "x2"],
  x3 = X_train[, "x3"]
)


test_data <- data.frame(
  x1 = X_test[, "x1"],
  x2 = X_test[, "x2"],
  x3 = X_test[, "x3"]
)


linear_model <- lm(
  y ~ x1 + x2 + x3,
  data = train_data
)


linear_test_prediction <- predict(
  linear_model,
  newdata = test_data
)


linear_test_MSE <- mean(
  (
    y_test -
      linear_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 38. Quadratic Linear Model
# ------------------------------------------------------------------------------

quadratic_model <- lm(
  y ~
    x1 +
    I(x1^2) +
    x2 +
    I(x2^2) +
    x3 +
    I(x3^2),
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
# 39. Compare Models
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Regression Stump",
    "Linear Regression",
    "Quadratic Regression",
    "Boosting"
  ),
  
  Test_MSE = c(
    stump_test_MSE,
    linear_test_MSE,
    quadratic_test_MSE,
    final_test_MSE
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XVIII
#
# TWO-DIMENSIONAL BOOSTING SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 80


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


X_grid <- cbind(
  x1 = prediction_grid$x1,
  x2 = prediction_grid$x2,
  x3 = 0
)


grid_prediction <- predict_boosting(
  boost_train,
  X_grid,
  M = one_SE_iteration
)


prediction_matrix <- matrix(
  grid_prediction,
  nrow = grid_size,
  ncol = grid_size
)


# ------------------------------------------------------------------------------
# 41. Plot Prediction Surface
# ------------------------------------------------------------------------------

image(
  grid_x1,
  grid_x2,
  prediction_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "Boosted Stump Prediction Surface"
)


contour(
  grid_x1,
  grid_x2,
  prediction_matrix,
  add = TRUE
)


# ==============================================================================
# PART XIX
#
# UNDERSTANDING SHRINKAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Compare Fast and Slow Boosting
# ------------------------------------------------------------------------------

fast_boost <- boost_regression(
  X = X_train,
  y = y_train,
  M = 100,
  learning_rate = 1,
  min_leaf = 8
)


slow_boost <- boost_regression(
  X = X_train,
  y = y_train,
  M = 100,
  learning_rate = 0.05,
  min_leaf = 8
)


fast_train_MSE <- fast_boost$training_MSE


slow_train_MSE <- slow_boost$training_MSE


matplot(
  1:100,
  cbind(
    fast_train_MSE,
    slow_train_MSE
  ),
  type = "l",
  lty = c(
    1,
    2
  ),
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Training MSE",
  main = "Effect of Shrinkage"
)


legend(
  "topright",
  legend = c(
    "Learning Rate = 1",
    "Learning Rate = 0.05"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART XX
#
# BOOSTING AS AN ADDITIVE MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Inspect First Few Learners
# ------------------------------------------------------------------------------

number_to_show <- min(
  10,
  length(
    boost_train$learners
  )
)


first_learners <- data.frame(
  Iteration = seq_len(
    number_to_show
  ),
  Variable = NA_character_,
  Split = NA_real_,
  Left_Value = NA_real_,
  Right_Value = NA_real_
)


for (
  m in seq_len(
    number_to_show
  )
) {
  
  learner <- boost_train$learners[[m]]
  
  
  first_learners$Variable[m] <-
    boost_train$feature_names[
      learner$variable
    ]
  
  
  first_learners$Split[m] <-
    learner$split
  
  
  first_learners$Left_Value[m] <-
    learner$left_value
  
  
  first_learners$Right_Value[m] <-
    learner$right_value
}


first_learners


# ==============================================================================
# PART XXI
#
# EFFECT OF NUMBER OF ITERATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Selected Iteration Performance
# ------------------------------------------------------------------------------

M_values <- c(
  1,
  5,
  10,
  25,
  50,
  100,
  200,
  300,
  500
)


iteration_results <- data.frame(
  Iterations = M_values,
  Training_MSE = NA_real_,
  Test_MSE = NA_real_
)


for (
  i in seq_along(
    M_values
  )
) {
  
  M_i <- M_values[i]
  
  
  train_prediction_i <- predict_boosting(
    boost_train,
    X_train,
    M = M_i
  )
  
  
  test_prediction_i <- predict_boosting(
    boost_train,
    X_test,
    M = M_i
  )
  
  
  iteration_results$Training_MSE[i] <-
    mean(
      (
        y_train -
          train_prediction_i
      )^2
    )
  
  
  iteration_results$Test_MSE[i] <-
    mean(
      (
        y_test -
          test_prediction_i
      )^2
    )
}


iteration_results


# ==============================================================================
# PART XXII
#
# OPTIONAL VERIFICATION USING GBM
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Production Boosting Comparison
# ------------------------------------------------------------------------------

# gbm is not required for this demonstration.
#
# This section is only an optional comparison.


if (
  requireNamespace(
    "gbm",
    quietly = TRUE
  )
) {
  
  gbm_model <- gbm::gbm(
    formula =
      y ~ x1 + x2 + x3,
    data = train_data,
    distribution = "gaussian",
    n.trees = 500,
    interaction.depth = 1,
    shrinkage = 0.05,
    bag.fraction = 1,
    train.fraction = 1,
    verbose = FALSE
  )
  
  
  gbm_prediction <- predict(
    gbm_model,
    newdata = test_data,
    n.trees = one_SE_iteration
  )
  
  
  gbm_test_MSE <- mean(
    (
      y_test -
        gbm_prediction
    )^2
  )
  
  
  cat(
    "Manual boosting test MSE:",
    round(
      final_test_MSE,
      4
    ),
    "\n"
  )
  
  
  cat(
    "gbm test MSE:",
    round(
      gbm_test_MSE,
      4
    ),
    "\n"
  )
}


# ==============================================================================
# PART XXIII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Final Output
# ------------------------------------------------------------------------------

cat(
  "Boosting Summary\n"
)


cat(
  "----------------\n"
)


cat(
  "Learning rate:",
  boost_train$learning_rate,
  "\n"
)


cat(
  "Total fitted boosting iterations:",
  length(
    boost_train$learners
  ),
  "\n"
)


cat(
  "CV minimum iteration:",
  best_CV_iteration,
  "\n"
)


cat(
  "One-SE selected iteration:",
  one_SE_iteration,
  "\n"
)


cat(
  "Single-stump test MSE:",
  round(
    stump_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Linear-regression test MSE:",
  round(
    linear_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Quadratic-regression test MSE:",
  round(
    quadratic_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Boosting test MSE:",
  round(
    final_test_MSE,
    4
  ),
  "\n"
)


cat(
  "\nBoosting variable importance:\n"
)


print(
  round(
    boost_variable_importance(
      boost_train
    ),
    4
  )
)