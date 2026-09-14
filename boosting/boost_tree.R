# ==============================================================================
# Boosting Trees
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Tree boosting
#   - Small regression trees as weak learners
#   - Sequential residual fitting
#   - Additive models
#   - Interaction depth
#   - Learning rate / shrinkage
#   - Number of boosting iterations
#   - Cross-validation
#   - Variable importance
#
#
# Squared-error tree boosting:
#
#   F_0(x) = mean(y)
#
# For m = 1, ..., M:
#
#   r_im = y_i - F_{m-1}(x_i)
#
#   Fit regression tree T_m(x) to r_im
#
#   F_m(x)
#   =
#   F_{m-1}(x)
#   +
#   nu T_m(x)
#
#
# Final model:
#
#   F_M(x)
#   =
#   F_0(x)
#   +
#   nu sum_{m=1}^M T_m(x)
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR DATA WITH INTERACTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Predictors
# ------------------------------------------------------------------------------

set.seed(123)

n <- 500


x1 <- runif(
  n,
  -3,
  3
)


x2 <- runif(
  n,
  -3,
  3
)


x3 <- runif(
  n,
  -3,
  3
)


# ------------------------------------------------------------------------------
# 2. True Regression Function
# ------------------------------------------------------------------------------

true_function <- function(
    x1,
    x2,
    x3
) {
  
  2 +
    2 * sin(x1) +
    ifelse(
      x2 > 0,
      2,
      -1.5
    ) +
    1.5 *
    (x1 > 0) *
    (x2 > 0) +
    0.5 * x3^2
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
# 3. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data,
  main = "Boosting Trees Data"
)


# ==============================================================================
# PART II
#
# REGRESSION TREE HELPERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Node RSS
# ------------------------------------------------------------------------------

node_rss <- function(y) {
  
  if (
    length(y) == 0
  ) {
    
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
# 5. Candidate Splits
# ------------------------------------------------------------------------------

candidate_splits <- function(
    x,
    maximum_splits = 40
) {
  
  values <- sort(
    unique(
      x
    )
  )
  
  
  if (
    length(values) <
    2
  ) {
    
    return(
      numeric(0)
    )
  }
  
  
  splits <- (
    values[
      -length(values)
    ] +
      values[
        -1
      ]
  ) / 2
  
  
  if (
    length(splits) >
    maximum_splits
  ) {
    
    indices <- round(
      seq(
        1,
        length(splits),
        length.out =
          maximum_splits
      )
    )
    
    
    splits <- splits[
      unique(
        indices
      )
    ]
  }
  
  
  return(
    splits
  )
}


# ------------------------------------------------------------------------------
# 6. Find Best Split
# ------------------------------------------------------------------------------

find_best_split <- function(
    X,
    y,
    min_leaf = 5,
    maximum_splits = 40
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  parent_RSS <- node_rss(
    y
  )
  
  
  best_RSS <- Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  
  for (
    j in seq_len(p)
  ) {
    
    split_values <- candidate_splits(
      X[, j],
      maximum_splits =
        maximum_splits
    )
    
    
    for (
      split_value in split_values
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
      
      
      split_RSS <-
        node_rss(
          y[left]
        ) +
        node_rss(
          y[right]
        )
      
      
      if (
        split_RSS <
        best_RSS
      ) {
        
        best_RSS <-
          split_RSS
        
        best_variable <-
          j
        
        best_split <-
          split_value
      }
    }
  }
  
  
  if (
    is.na(
      best_variable
    )
  ) {
    
    return(
      list(
        variable =
          NA_integer_,
        split =
          NA_real_,
        improvement = 0
      )
    )
  }
  
  
  return(
    list(
      variable =
        best_variable,
      split =
        best_split,
      improvement =
        parent_RSS -
        best_RSS
    )
  )
}


# ==============================================================================
# PART III
#
# SMALL REGRESSION TREES
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Grow Depth-Limited Regression Tree
# ------------------------------------------------------------------------------

grow_small_tree <- function(
    X,
    y,
    indices = seq_len(
      nrow(X)
    ),
    depth = 0,
    max_depth = 2,
    min_leaf = 5,
    min_split = 10,
    maximum_splits = 40
) {
  
  X <- as.matrix(
    X
  )
  
  
  y_node <- y[
    indices
  ]
  
  
  node <- list(
    terminal = TRUE,
    prediction =
      mean(y_node),
    n =
      length(indices),
    variable =
      NA_integer_,
    split =
      NA_real_,
    improvement = 0,
    left = NULL,
    right = NULL
  )
  
  
  # --------------------------------------------------------------------------
  # Stopping rules
  # --------------------------------------------------------------------------
  
  if (
    depth >=
    max_depth
  ) {
    
    return(node)
  }
  
  
  if (
    length(indices) <
    min_split
  ) {
    
    return(node)
  }
  
  
  # --------------------------------------------------------------------------
  # Best split
  # --------------------------------------------------------------------------
  
  split_result <- find_best_split(
    X = X[
      indices,
      ,
      drop = FALSE
    ],
    y = y_node,
    min_leaf =
      min_leaf,
    maximum_splits =
      maximum_splits
  )
  
  
  if (
    is.na(
      split_result$variable
    ) ||
    split_result$improvement <= 0
  ) {
    
    return(node)
  }
  
  
  j <- split_result$variable
  
  split_value <- split_result$split
  
  
  left_local <-
    X[
      indices,
      j
    ] <=
    split_value
  
  
  left_indices <- indices[
    left_local
  ]
  
  
  right_indices <- indices[
    !left_local
  ]
  
  
  node$terminal <- FALSE
  
  node$variable <- j
  
  node$split <-
    split_value
  
  node$improvement <-
    split_result$improvement
  
  
  node$left <- grow_small_tree(
    X = X,
    y = y,
    indices =
      left_indices,
    depth =
      depth + 1,
    max_depth =
      max_depth,
    min_leaf =
      min_leaf,
    min_split =
      min_split,
    maximum_splits =
      maximum_splits
  )
  
  
  node$right <- grow_small_tree(
    X = X,
    y = y,
    indices =
      right_indices,
    depth =
      depth + 1,
    max_depth =
      max_depth,
    min_leaf =
      min_leaf,
    min_split =
      min_split,
    maximum_splits =
      maximum_splits
  )
  
  
  return(node)
}


# ------------------------------------------------------------------------------
# 8. Predict One Observation
# ------------------------------------------------------------------------------

predict_tree_one <- function(
    tree,
    x
) {
  
  if (
    tree$terminal
  ) {
    
    return(
      tree$prediction
    )
  }
  
  
  if (
    x[
      tree$variable
    ] <=
    tree$split
  ) {
    
    return(
      predict_tree_one(
        tree$left,
        x
      )
    )
  }
  
  
  predict_tree_one(
    tree$right,
    x
  )
}


# ------------------------------------------------------------------------------
# 9. Predict Matrix
# ------------------------------------------------------------------------------

predict_tree <- function(
    tree,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  apply(
    X_new,
    1,
    function(x) {
      
      predict_tree_one(
        tree,
        x
      )
    }
  )
}


# ==============================================================================
# PART IV
#
# TREE BOOSTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Fit Boosted Trees
# ------------------------------------------------------------------------------

fit_boosted_trees <- function(
    X,
    y,
    M = 300,
    learning_rate = 0.05,
    tree_depth = 2,
    min_leaf = 5,
    min_split = 10,
    maximum_splits = 40,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Initial prediction
  # --------------------------------------------------------------------------
  
  initial_prediction <- mean(
    y
  )
  
  
  fitted_values <- rep(
    initial_prediction,
    n
  )
  
  
  trees <- vector(
    "list",
    M
  )
  
  
  training_MSE <- numeric(
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
    # Residuals
    # ------------------------------------------------------------------------
    
    residuals <- y -
      fitted_values
    
    
    # ------------------------------------------------------------------------
    # Fit small regression tree to residuals
    # ------------------------------------------------------------------------
    
    tree_m <- grow_small_tree(
      X = X,
      y = residuals,
      max_depth =
        tree_depth,
      min_leaf =
        min_leaf,
      min_split =
        min_split,
      maximum_splits =
        maximum_splits
    )
    
    
    tree_prediction <- predict_tree(
      tree_m,
      X
    )
    
    
    # ------------------------------------------------------------------------
    # Add tree to ensemble
    # ------------------------------------------------------------------------
    
    fitted_values <-
      fitted_values +
      learning_rate *
      tree_prediction
    
    
    current_MSE <- mean(
      (
        y -
          fitted_values
      )^2
    )
    
    
    trees[[m]] <- tree_m
    
    
    training_MSE[m] <-
      current_MSE
    
    
    improvement[m] <-
      previous_MSE -
      current_MSE
    
    
    previous_MSE <-
      current_MSE
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        m,
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
      trees =
        trees,
      learning_rate =
        learning_rate,
      tree_depth =
        tree_depth,
      fitted_values =
        fitted_values,
      training_MSE =
        training_MSE,
      improvement =
        improvement,
      feature_names =
        colnames(X)
    )
  )
}


# ------------------------------------------------------------------------------
# 11. Fit Boosting Tree Model
# ------------------------------------------------------------------------------

boosted_tree_model <- fit_boosted_trees(
  X = X,
  y = y,
  M = 300,
  learning_rate = 0.05,
  tree_depth = 2,
  min_leaf = 8,
  min_split = 16,
  maximum_splits = 40
)


# ==============================================================================
# PART V
#
# PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Boosted Tree Prediction
# ------------------------------------------------------------------------------

predict_boosted_trees <- function(
    model,
    X_new,
    M = length(
      model$trees
    )
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  M <- min(
    M,
    length(
      model$trees
    )
  )
  
  
  prediction <- rep(
    model$initial_prediction,
    nrow(
      X_new
    )
  )
  
  
  for (
    m in seq_len(M)
  ) {
    
    prediction <-
      prediction +
      model$learning_rate *
      predict_tree(
        model$trees[[m]],
        X_new
      )
  }
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 13. Check Training Prediction
# ------------------------------------------------------------------------------

prediction_check <- predict_boosted_trees(
  boosted_tree_model,
  X
)


max(
  abs(
    prediction_check -
      boosted_tree_model$fitted_values
  )
)


# ==============================================================================
# PART VI
#
# TRAINING PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Final Training MSE
# ------------------------------------------------------------------------------

training_MSE <- mean(
  (
    y -
      boosted_tree_model$fitted_values
  )^2
)


training_RMSE <- sqrt(
  training_MSE
)


training_MSE


training_RMSE


# ------------------------------------------------------------------------------
# 15. Training Error Path
# ------------------------------------------------------------------------------

plot(
  seq_along(
    boosted_tree_model$training_MSE
  ),
  boosted_tree_model$training_MSE,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Training MSE",
  main = "Boosting Trees Training Error"
)


# ==============================================================================
# PART VII
#
# STAGEWISE FIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. View Different Numbers of Trees
# ------------------------------------------------------------------------------

M_values <- c(
  1,
  5,
  20,
  50,
  100,
  300
)


stage_results <- data.frame(
  Trees = M_values,
  Training_MSE =
    NA_real_
)


for (
  i in seq_along(
    M_values
  )
) {
  
  prediction_i <- predict_boosted_trees(
    boosted_tree_model,
    X,
    M = M_values[i]
  )
  
  
  stage_results$Training_MSE[i] <- mean(
    (
      y -
        prediction_i
    )^2
  )
}


stage_results


# ==============================================================================
# PART VIII
#
# ONE-DIMENSIONAL SLICE
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Prediction Slice
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


plot(
  x1_grid,
  true_slice,
  type = "l",
  lwd = 3,
  xlab = "x1",
  ylab = "Response",
  main = "Boosting Tree Prediction"
)


for (
  i in seq_along(
    M_values
  )
) {
  
  prediction_i <- predict_boosted_trees(
    boosted_tree_model,
    slice_data,
    M = M_values[i]
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
    "True",
    paste0(
      "M = ",
      M_values
    )
  ),
  lty = c(
    1,
    seq_along(
      M_values
    )
  ),
  lwd = c(
    3,
    rep(
      2,
      length(M_values)
    )
  )
)


# ==============================================================================
# PART IX
#
# INTERACTION DEPTH
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Compare Tree Depths
# ------------------------------------------------------------------------------

depth_values <- c(
  1,
  2,
  3,
  4
)


depth_results <- data.frame(
  Tree_Depth =
    depth_values,
  Training_MSE =
    NA_real_
)


depth_models <- vector(
  "list",
  length(
    depth_values
  )
)


for (
  i in seq_along(
    depth_values
  )
) {
  
  model_i <- fit_boosted_trees(
    X = X,
    y = y,
    M = 200,
    learning_rate = 0.05,
    tree_depth =
      depth_values[i],
    min_leaf = 8,
    min_split = 16,
    maximum_splits = 40
  )
  
  
  depth_models[[i]] <-
    model_i
  
  
  depth_results$Training_MSE[i] <-
    tail(
      model_i$training_MSE,
      1
    )
}


depth_results


# ==============================================================================
# PART X
#
# TRAIN-TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Split Data
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
# 20. Fit Training Boosted Trees
# ------------------------------------------------------------------------------

boost_train <- fit_boosted_trees(
  X = X_train,
  y = y_train,
  M = 500,
  learning_rate = 0.05,
  tree_depth = 2,
  min_leaf = 8,
  min_split = 16,
  maximum_splits = 40
)


# ==============================================================================
# PART XI
#
# TRAIN AND TEST ERROR PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Evaluate Every Boosting Iteration
# ------------------------------------------------------------------------------

M_max <- length(
  boost_train$trees
)


train_MSE_path <- numeric(
  M_max
)


test_MSE_path <- numeric(
  M_max
)


train_prediction <- rep(
  boost_train$initial_prediction,
  nrow(
    X_train
  )
)


test_prediction <- rep(
  boost_train$initial_prediction,
  nrow(
    X_test
  )
)


for (
  m in seq_len(
    M_max
  )
) {
  
  train_prediction <-
    train_prediction +
    boost_train$learning_rate *
    predict_tree(
      boost_train$trees[[m]],
      X_train
    )
  
  
  test_prediction <-
    test_prediction +
    boost_train$learning_rate *
    predict_tree(
      boost_train$trees[[m]],
      X_test
    )
  
  
  train_MSE_path[m] <- mean(
    (
      y_train -
        train_prediction
    )^2
  )
  
  
  test_MSE_path[m] <- mean(
    (
      y_test -
        test_prediction
    )^2
  )
}


# ------------------------------------------------------------------------------
# 22. Plot Train/Test Error
# ------------------------------------------------------------------------------

matplot(
  seq_len(
    M_max
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
  xlab = "Number of Trees",
  ylab = "MSE",
  main = "Boosting Trees Train/Test Error"
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


# ------------------------------------------------------------------------------
# 23. Oracle Test Iteration
# ------------------------------------------------------------------------------

best_test_iteration <- which.min(
  test_MSE_path
)


best_test_iteration


test_MSE_path[
  best_test_iteration
]


# Test-set selection is diagnostic only.


# ==============================================================================
# PART XII
#
# CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. K-Fold Cross Validation
# ------------------------------------------------------------------------------

boosted_tree_cv <- function(
    X,
    y,
    M = 300,
    k = 5,
    learning_rate = 0.05,
    tree_depth = 2,
    min_leaf = 5,
    min_split = 10,
    maximum_splits = 30,
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
    
    
    model_fold <- fit_boosted_trees(
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
      tree_depth =
        tree_depth,
      min_leaf =
        min_leaf,
      min_split =
        min_split,
      maximum_splits =
        maximum_splits
    )
    
    
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
        predict_tree(
          model_fold$trees[[m]],
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
  
  
  mean_MSE <- rowMeans(
    fold_MSE
  )
  
  
  SE_MSE <- apply(
    fold_MSE,
    1,
    sd
  ) /
    sqrt(k)
  
  
  return(
    list(
      mean =
        mean_MSE,
      SE =
        SE_MSE,
      fold_MSE =
        fold_MSE
    )
  )
}


# ------------------------------------------------------------------------------
# 25. Run CV
# ------------------------------------------------------------------------------

CV_result <- boosted_tree_cv(
  X = X_train,
  y = y_train,
  M = 300,
  k = 5,
  learning_rate = 0.05,
  tree_depth = 2,
  min_leaf = 8,
  min_split = 16,
  maximum_splits = 30,
  seed = 123
)


# ------------------------------------------------------------------------------
# 26. Best CV Iteration
# ------------------------------------------------------------------------------

best_CV_iteration <- which.min(
  CV_result$mean
)


best_CV_iteration


# ------------------------------------------------------------------------------
# 27. Plot CV Error
# ------------------------------------------------------------------------------

plot(
  seq_along(
    CV_result$mean
  ),
  CV_result$mean,
  type = "l",
  lwd = 2,
  xlab = "Number of Trees",
  ylab = "Cross-Validated MSE",
  main = "Boosting Trees Cross Validation"
)


abline(
  v = best_CV_iteration,
  lty = 2
)


# ==============================================================================
# PART XIII
#
# ONE-STANDARD-ERROR RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. One-SE Selection
# ------------------------------------------------------------------------------

minimum_error <- CV_result$mean[
  best_CV_iteration
]


minimum_SE <- CV_result$SE[
  best_CV_iteration
]


threshold <-
  minimum_error +
  minimum_SE


eligible_iterations <- which(
  CV_result$mean <=
    threshold
)


one_SE_iteration <- min(
  eligible_iterations
)


one_SE_iteration


# ==============================================================================
# PART XIV
#
# FINAL TEST PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Final Model Prediction
# ------------------------------------------------------------------------------

final_test_prediction <- predict_boosted_trees(
  boost_train,
  X_test,
  M = one_SE_iteration
)


final_train_prediction <- predict_boosted_trees(
  boost_train,
  X_train,
  M = one_SE_iteration
)


final_test_MSE <- mean(
  (
    y_test -
      final_test_prediction
  )^2
)


final_train_MSE <- mean(
  (
    y_train -
      final_train_prediction
  )^2
)


final_test_RMSE <- sqrt(
  final_test_MSE
)


final_train_MSE


final_test_MSE


# ==============================================================================
# PART XV
#
# VARIABLE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Accumulate Split Improvements
# ------------------------------------------------------------------------------

tree_importance <- function(
    tree,
    p
) {
  
  importance <- numeric(
    p
  )
  
  
  traverse <- function(node) {
    
    if (
      node$terminal
    ) {
      
      return(
        invisible(NULL)
      )
    }
    
    
    importance[
      node$variable
    ] <<-
      importance[
        node$variable
      ] +
      node$improvement
    
    
    traverse(
      node$left
    )
    
    
    traverse(
      node$right
    )
    
    
    invisible(NULL)
  }
  
  
  traverse(
    tree
  )
  
  
  return(
    importance
  )
}


# ------------------------------------------------------------------------------
# 31. Ensemble Importance
# ------------------------------------------------------------------------------

boosted_tree_importance <- function(
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
      model$trees
    )
  ) {
    
    importance <-
      importance +
      tree_importance(
        model$trees[[m]],
        p
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


importance <- boosted_tree_importance(
  boost_train
)


importance


barplot(
  importance,
  ylab = "Relative Importance",
  main = "Boosting Trees Variable Importance"
)


# ==============================================================================
# PART XVI
#
# DEPTH AND INTERACTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Compare Depths on Test Data
# ------------------------------------------------------------------------------

depth_values <- 1:4


depth_test_results <- data.frame(
  Depth = depth_values,
  Test_MSE = NA_real_
)


for (
  i in seq_along(
    depth_values
  )
) {
  
  model_i <- fit_boosted_trees(
    X = X_train,
    y = y_train,
    M = 300,
    learning_rate = 0.05,
    tree_depth =
      depth_values[i],
    min_leaf = 8,
    min_split = 16,
    maximum_splits = 30
  )
  
  
  prediction_i <- predict_boosted_trees(
    model_i,
    X_test
  )
  
  
  depth_test_results$Test_MSE[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


depth_test_results


plot(
  depth_test_results$Depth,
  depth_test_results$Test_MSE,
  type = "b",
  pch = 19,
  xlab = "Tree Depth",
  ylab = "Test MSE",
  main = "Interaction Depth in Tree Boosting"
)


# ==============================================================================
# PART XVII
#
# LEARNING RATE
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Compare Shrinkage Values
# ------------------------------------------------------------------------------

learning_rates <- c(
  0.01,
  0.05,
  0.1,
  0.25
)


learning_rate_results <- data.frame(
  Learning_Rate =
    learning_rates,
  Test_MSE =
    NA_real_
)


for (
  i in seq_along(
    learning_rates
  )
) {
  
  model_i <- fit_boosted_trees(
    X = X_train,
    y = y_train,
    M = 300,
    learning_rate =
      learning_rates[i],
    tree_depth = 2,
    min_leaf = 8,
    min_split = 16,
    maximum_splits = 30
  )
  
  
  prediction_i <- predict_boosted_trees(
    model_i,
    X_test
  )
  
  
  learning_rate_results$Test_MSE[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


learning_rate_results


# ==============================================================================
# PART XVIII
#
# TWO-DIMENSIONAL PREDICTION SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Grid
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


grid <- expand.grid(
  x1 = grid_x1,
  x2 = grid_x2
)


X_grid <- cbind(
  x1 = grid$x1,
  x2 = grid$x2,
  x3 = 0
)


grid_prediction <- predict_boosted_trees(
  boost_train,
  X_grid,
  M = one_SE_iteration
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
  main = "Boosting Trees Prediction Surface"
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
# COMPARE WITH A SINGLE TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Fit One Regression Tree
# ------------------------------------------------------------------------------

single_tree <- grow_small_tree(
  X = X_train,
  y = y_train,
  max_depth = 4,
  min_leaf = 8,
  min_split = 16,
  maximum_splits = 40
)


single_tree_prediction <- predict_tree(
  single_tree,
  X_test
)


single_tree_MSE <- mean(
  (
    y_test -
      single_tree_prediction
  )^2
)


# ==============================================================================
# PART XX
#
# COMPARE WITH LINEAR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Linear Model
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


linear_prediction <- predict(
  linear_model,
  newdata = test_data
)


linear_MSE <- mean(
  (
    y_test -
      linear_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 37. Comparison
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Single Regression Tree",
    "Boosting Trees"
  ),
  
  Test_MSE = c(
    linear_MSE,
    single_tree_MSE,
    final_test_MSE
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XXI
#
# OPTIONAL VERIFICATION WITH GBM
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Compare with gbm
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "gbm",
    quietly = TRUE
  )
) {
  
  gbm_model <- gbm::gbm(
    formula =
      y ~ x1 + x2 + x3,
    data =
      train_data,
    distribution =
      "gaussian",
    n.trees = 500,
    interaction.depth = 2,
    shrinkage = 0.05,
    bag.fraction = 1,
    train.fraction = 1,
    verbose = FALSE
  )
  
  
  gbm_prediction <- predict(
    gbm_model,
    newdata =
      test_data,
    n.trees =
      one_SE_iteration
  )
  
  
  gbm_test_MSE <- mean(
    (
      y_test -
        gbm_prediction
    )^2
  )
  
  
  cat(
    "Manual boosted-tree test MSE:",
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
# PART XXII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Final Output
# ------------------------------------------------------------------------------

cat(
  "Boosting Trees Summary\n"
)


cat(
  "----------------------\n"
)


cat(
  "Learning rate:",
  boost_train$learning_rate,
  "\n"
)


cat(
  "Tree depth:",
  boost_train$tree_depth,
  "\n"
)


cat(
  "Total fitted trees:",
  length(
    boost_train$trees
  ),
  "\n"
)


cat(
  "Best CV iteration:",
  best_CV_iteration,
  "\n"
)


cat(
  "One-SE iteration:",
  one_SE_iteration,
  "\n"
)


cat(
  "Single tree test MSE:",
  round(
    single_tree_MSE,
    4
  ),
  "\n"
)


cat(
  "Boosted trees test MSE:",
  round(
    final_test_MSE,
    4
  ),
  "\n"
)


cat(
  "\nVariable importance:\n"
)


print(
  round(
    importance,
    4
  )
)