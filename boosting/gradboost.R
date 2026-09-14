# ==============================================================================
# Gradient Boosting
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Gradient boosting
#   - Functional gradient descent
#   - Pseudo-residuals
#   - Regression trees as base learners
#   - Squared-error loss
#   - Huber loss
#   - Logistic loss
#   - Learning rate / shrinkage
#   - Interaction depth
#   - Early stopping
#   - Cross-validation
#   - Variable importance
#
#
# General gradient boosting algorithm:
#
#   1. Initialize:
#
#        F_0(x)
#        =
#        argmin_gamma
#        sum_i L(y_i, gamma)
#
#
#   2. For m = 1, ..., M:
#
#        r_im
#        =
#        - [
#            partial L(y_i, F(x_i))
#            -----------------------
#                 partial F(x_i)
#          ] evaluated at F = F_(m-1)
#
#
#   3. Fit a regression tree to the pseudo-residuals r_im.
#
#
#   4. Update:
#
#        F_m(x)
#        =
#        F_(m-1)(x)
#        +
#        nu * h_m(x)
#
#
# where:
#
#   nu = learning rate
#   h_m(x) = weak learner approximating the negative gradient
#
#
# For squared error:
#
#        L(y,F) = 1/2 (y-F)^2
#
#        r = y-F
#
#
# For logistic loss:
#
#        L(y,F) = log(1+exp(F)) - yF
#
#        r = y-p
#
# where:
#
#        p = sigmoid(F)
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Predictors
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
      x2 > 0.5,
      2.5,
      -1
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
# 3. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data,
  main = "Gradient Boosting Regression Data"
)


# ==============================================================================
# PART II
#
# REGRESSION TREE IMPLEMENTATION
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
# 5. Candidate Split Points
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
  
  
  best_child_RSS <- Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  
  for (
    j in seq_len(p)
  ) {
    
    splits <- candidate_splits(
      X[, j],
      maximum_splits =
        maximum_splits
    )
    
    
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
      
      
      child_RSS <-
        node_rss(
          y[left]
        ) +
        node_rss(
          y[right]
        )
      
      
      if (
        child_RSS <
        best_child_RSS
      ) {
        
        best_child_RSS <-
          child_RSS
        
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
        best_child_RSS
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Grow a Small Regression Tree
# ------------------------------------------------------------------------------

grow_tree <- function(
    X,
    target,
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
  
  
  target_node <- target[
    indices
  ]
  
  
  node <- list(
    terminal = TRUE,
    prediction =
      mean(
        target_node
      ),
    indices =
      indices,
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
  # Stopping Conditions
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
  # Search for Split
  # --------------------------------------------------------------------------
  
  best_split <- find_best_split(
    X = X[
      indices,
      ,
      drop = FALSE
    ],
    y = target_node,
    min_leaf =
      min_leaf,
    maximum_splits =
      maximum_splits
  )
  
  
  if (
    is.na(
      best_split$variable
    ) ||
    best_split$improvement <=
    0
  ) {
    
    return(node)
  }
  
  
  j <- best_split$variable
  
  split_value <- best_split$split
  
  
  left_condition <-
    X[
      indices,
      j
    ] <=
    split_value
  
  
  left_indices <- indices[
    left_condition
  ]
  
  
  right_indices <- indices[
    !left_condition
  ]
  
  
  node$terminal <- FALSE
  
  node$variable <- j
  
  node$split <- split_value
  
  node$improvement <-
    best_split$improvement
  
  
  node$left <- grow_tree(
    X = X,
    target = target,
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
  
  
  node$right <- grow_tree(
    X = X,
    target = target,
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
  
  
  return(
    predict_tree_one(
      tree$right,
      x
    )
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
# PART III
#
# LOSS FUNCTIONS AND NEGATIVE GRADIENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Squared Error Loss
# ------------------------------------------------------------------------------

squared_error_loss <- function(
    y,
    F
) {
  
  0.5 *
    (
      y -
        F
    )^2
}


# ------------------------------------------------------------------------------
# 11. Squared Error Negative Gradient
# ------------------------------------------------------------------------------

squared_error_gradient <- function(
    y,
    F
) {
  
  # Negative gradient:
  #
  #   -(F-y) = y-F
  
  y -
    F
}


# ------------------------------------------------------------------------------
# 12. Huber Loss
# ------------------------------------------------------------------------------

huber_loss <- function(
    y,
    F,
    delta = 1
) {
  
  residual <- y -
    F
  
  
  ifelse(
    abs(residual) <=
      delta,
    
    0.5 *
      residual^2,
    
    delta *
      (
        abs(residual) -
          0.5 *
          delta
      )
  )
}


# ------------------------------------------------------------------------------
# 13. Huber Negative Gradient
# ------------------------------------------------------------------------------

huber_gradient <- function(
    y,
    F,
    delta = 1
) {
  
  residual <- y -
    F
  
  
  pmax(
    pmin(
      residual,
      delta
    ),
    -delta
  )
}


# ==============================================================================
# PART IV
#
# GENERIC GRADIENT BOOSTING FOR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Gradient Boosting Function
# ------------------------------------------------------------------------------

fit_gradient_boosting <- function(
    X,
    y,
    loss = c(
      "squared_error",
      "huber"
    ),
    M = 300,
    learning_rate = 0.05,
    tree_depth = 2,
    min_leaf = 5,
    min_split = 10,
    maximum_splits = 40,
    huber_delta = 1,
    verbose = FALSE
) {
  
  loss <- match.arg(
    loss
  )
  
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Initial Model
  # --------------------------------------------------------------------------
  
  if (
    loss ==
    "squared_error"
  ) {
    
    initial_prediction <- mean(
      y
    )
    
  } else {
    
    # Median is robust and provides a useful initialization for Huber loss.
    
    initial_prediction <- median(
      y
    )
  }
  
  
  F <- rep(
    initial_prediction,
    n
  )
  
  
  trees <- vector(
    "list",
    M
  )
  
  
  training_loss <- numeric(
    M
  )
  
  
  training_MSE <- numeric(
    M
  )
  
  
  gradient_norm <- numeric(
    M
  )
  
  
  # ==========================================================================
  # BOOSTING LOOP
  # ==========================================================================
  
  for (
    m in seq_len(M)
  ) {
    
    # ------------------------------------------------------------------------
    # Compute Negative Gradient
    # ------------------------------------------------------------------------
    
    if (
      loss ==
      "squared_error"
    ) {
      
      pseudo_residual <- squared_error_gradient(
        y,
        F
      )
      
    } else {
      
      pseudo_residual <- huber_gradient(
        y,
        F,
        delta =
          huber_delta
      )
    }
    
    
    gradient_norm[m] <- sqrt(
      mean(
        pseudo_residual^2
      )
    )
    
    
    # ------------------------------------------------------------------------
    # Fit Tree to Negative Gradient
    # ------------------------------------------------------------------------
    
    tree_m <- grow_tree(
      X = X,
      target =
        pseudo_residual,
      max_depth =
        tree_depth,
      min_leaf =
        min_leaf,
      min_split =
        min_split,
      maximum_splits =
        maximum_splits
    )
    
    
    # ------------------------------------------------------------------------
    # Weak Learner Prediction
    # ------------------------------------------------------------------------
    
    h_m <- predict_tree(
      tree_m,
      X
    )
    
    
    # ------------------------------------------------------------------------
    # Functional Gradient Update
    # ------------------------------------------------------------------------
    
    F <- F +
      learning_rate *
      h_m
    
    
    trees[[m]] <- tree_m
    
    
    # ------------------------------------------------------------------------
    # Store Loss
    # ------------------------------------------------------------------------
    
    if (
      loss ==
      "squared_error"
    ) {
      
      training_loss[m] <- mean(
        squared_error_loss(
          y,
          F
        )
      )
      
    } else {
      
      training_loss[m] <- mean(
        huber_loss(
          y,
          F,
          delta =
            huber_delta
        )
      )
    }
    
    
    training_MSE[m] <- mean(
      (
        y -
          F
      )^2
    )
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        m,
        "| Loss:",
        round(
          training_loss[m],
          5
        ),
        "| Gradient Norm:",
        round(
          gradient_norm[m],
          5
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
      fitted_values =
        F,
      learning_rate =
        learning_rate,
      loss =
        loss,
      huber_delta =
        huber_delta,
      tree_depth =
        tree_depth,
      training_loss =
        training_loss,
      training_MSE =
        training_MSE,
      gradient_norm =
        gradient_norm,
      feature_names =
        colnames(X)
    )
  )
}


# ------------------------------------------------------------------------------
# 15. Fit Squared-Error Gradient Boosting
# ------------------------------------------------------------------------------

GB_model <- fit_gradient_boosting(
  X = X,
  y = y,
  loss = "squared_error",
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
# 16. Prediction Function
# ------------------------------------------------------------------------------

predict_gradient_boosting <- function(
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
    
    prediction <- prediction +
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
# 17. Verify Training Predictions
# ------------------------------------------------------------------------------

prediction_check <- predict_gradient_boosting(
  GB_model,
  X
)


max(
  abs(
    prediction_check -
      GB_model$fitted_values
  )
)


# ==============================================================================
# PART VI
#
# CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Loss Path
# ------------------------------------------------------------------------------

plot(
  seq_along(
    GB_model$training_loss
  ),
  GB_model$training_loss,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Average Loss",
  main = "Gradient Boosting Loss"
)


# ------------------------------------------------------------------------------
# 19. MSE Path
# ------------------------------------------------------------------------------

plot(
  seq_along(
    GB_model$training_MSE
  ),
  GB_model$training_MSE,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Training MSE",
  main = "Gradient Boosting Training MSE"
)


# ------------------------------------------------------------------------------
# 20. Negative Gradient Magnitude
# ------------------------------------------------------------------------------

plot(
  seq_along(
    GB_model$gradient_norm
  ),
  GB_model$gradient_norm,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "RMS Negative Gradient",
  main = "Functional Gradient Magnitude"
)


# ==============================================================================
# PART VII
#
# VERIFY THE SQUARED-ERROR PSEUDO-RESIDUAL
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Initial Predictions
# ------------------------------------------------------------------------------

F0 <- rep(
  mean(y),
  n
)


ordinary_residual <- y -
  F0


pseudo_residual <- squared_error_gradient(
  y,
  F0
)


max(
  abs(
    ordinary_residual -
      pseudo_residual
  )
)


# For squared error:
#
#   residual
#
# and
#
#   negative gradient
#
# are exactly identical.


# ==============================================================================
# PART VIII
#
# STAGEWISE PREDICTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Different Numbers of Trees
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
  Trees =
    M_values,
  Training_MSE =
    NA_real_
)


for (
  i in seq_along(
    M_values
  )
) {
  
  prediction_i <- predict_gradient_boosting(
    GB_model,
    X,
    M =
      M_values[i]
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
# PART IX
#
# VISUALIZE STAGEWISE LEARNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. x1 Prediction Slice
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
  main = "Stagewise Gradient Boosting"
)


for (
  i in seq_along(
    M_values
  )
) {
  
  prediction_i <- predict_gradient_boosting(
    GB_model,
    slice_data,
    M =
      M_values[i]
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
      length(
        M_values
      )
    )
  )
)


# ==============================================================================
# PART X
#
# TRAIN-TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Train-Test Split
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
# 25. Fit Training Model
# ------------------------------------------------------------------------------

GB_train <- fit_gradient_boosting(
  X = X_train,
  y = y_train,
  loss = "squared_error",
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
# 26. Sequential Evaluation
# ------------------------------------------------------------------------------

M_max <- length(
  GB_train$trees
)


train_prediction <- rep(
  GB_train$initial_prediction,
  nrow(
    X_train
  )
)


test_prediction <- rep(
  GB_train$initial_prediction,
  nrow(
    X_test
  )
)


train_MSE_path <- numeric(
  M_max
)


test_MSE_path <- numeric(
  M_max
)


for (
  m in seq_len(
    M_max
  )
) {
  
  train_prediction <- train_prediction +
    GB_train$learning_rate *
    predict_tree(
      GB_train$trees[[m]],
      X_train
    )
  
  
  test_prediction <- test_prediction +
    GB_train$learning_rate *
    predict_tree(
      GB_train$trees[[m]],
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
# 27. Plot Paths
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
  main = "Gradient Boosting Train/Test Error"
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
# 28. Oracle Test Iteration
# ------------------------------------------------------------------------------

best_test_iteration <- which.min(
  test_MSE_path
)


best_test_iteration


# Diagnostic only:
#
# Never use the final test set to select M in a real analysis.


# ==============================================================================
# PART XII
#
# CROSS-VALIDATION FOR NUMBER OF TREES
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Manual K-Fold CV
# ------------------------------------------------------------------------------

gradient_boosting_cv <- function(
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
    
    
    model_fold <- fit_gradient_boosting(
      X = X[
        training_indices,
        ,
        drop = FALSE
      ],
      y = y[
        training_indices
      ],
      loss =
        "squared_error",
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
# 30. Run CV
# ------------------------------------------------------------------------------

CV_result <- gradient_boosting_cv(
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
# 31. Minimum-CV Model
# ------------------------------------------------------------------------------

best_CV_iteration <- which.min(
  CV_result$mean
)


best_CV_iteration


# ------------------------------------------------------------------------------
# 32. One-Standard-Error Rule
# ------------------------------------------------------------------------------

CV_threshold <-
  CV_result$mean[
    best_CV_iteration
  ] +
  CV_result$SE[
    best_CV_iteration
  ]


eligible_iterations <- which(
  CV_result$mean <=
    CV_threshold
)


one_SE_iteration <- min(
  eligible_iterations
)


one_SE_iteration


# ------------------------------------------------------------------------------
# 33. Plot CV Path
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
  main = "Gradient Boosting Cross Validation"
)


abline(
  v = best_CV_iteration,
  lty = 2
)


abline(
  v = one_SE_iteration,
  lty = 3
)


# ==============================================================================
# PART XIII
#
# FINAL REGRESSION MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. CV-Selected Test Prediction
# ------------------------------------------------------------------------------

final_test_prediction <- predict_gradient_boosting(
  GB_train,
  X_test,
  M =
    one_SE_iteration
)


final_train_prediction <- predict_gradient_boosting(
  GB_train,
  X_train,
  M =
    one_SE_iteration
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


final_train_MSE <- mean(
  (
    y_train -
      final_train_prediction
  )^2
)


final_train_MSE


final_test_MSE


# ==============================================================================
# PART XIV
#
# LEARNING RATE
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Compare Shrinkage
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
  
  model_i <- fit_gradient_boosting(
    X = X_train,
    y = y_train,
    loss =
      "squared_error",
    M = 300,
    learning_rate =
      learning_rates[i],
    tree_depth = 2,
    min_leaf = 8,
    min_split = 16,
    maximum_splits = 30
  )
  
  
  prediction_i <- predict_gradient_boosting(
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
# PART XV
#
# INTERACTION DEPTH
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Compare Tree Depth
# ------------------------------------------------------------------------------

depth_values <- 1:4


depth_results <- data.frame(
  Tree_Depth =
    depth_values,
  Test_MSE =
    NA_real_
)


for (
  i in seq_along(
    depth_values
  )
) {
  
  model_i <- fit_gradient_boosting(
    X = X_train,
    y = y_train,
    loss =
      "squared_error",
    M = 300,
    learning_rate = 0.05,
    tree_depth =
      depth_values[i],
    min_leaf = 8,
    min_split = 16,
    maximum_splits = 30
  )
  
  
  prediction_i <- predict_gradient_boosting(
    model_i,
    X_test
  )
  
  
  depth_results$Test_MSE[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


depth_results


# ==============================================================================
# PART XVI
#
# ROBUST GRADIENT BOOSTING WITH HUBER LOSS
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Add Outliers
# ------------------------------------------------------------------------------

set.seed(789)


y_outlier <- y


number_outliers <- round(
  0.05 *
    n
)


outlier_indices <- sample(
  seq_len(n),
  size =
    number_outliers
)


y_outlier[
  outlier_indices
] <-
  y_outlier[
    outlier_indices
  ] +
  rnorm(
    number_outliers,
    mean = 0,
    sd = 15
  )


# ------------------------------------------------------------------------------
# 38. Squared-Error Boosting with Outliers
# ------------------------------------------------------------------------------

GB_squared_outlier <- fit_gradient_boosting(
  X = X,
  y = y_outlier,
  loss =
    "squared_error",
  M = 300,
  learning_rate = 0.05,
  tree_depth = 2,
  min_leaf = 8,
  min_split = 16
)


# ------------------------------------------------------------------------------
# 39. Huber Boosting
# ------------------------------------------------------------------------------

GB_huber <- fit_gradient_boosting(
  X = X,
  y = y_outlier,
  loss =
    "huber",
  M = 300,
  learning_rate = 0.05,
  tree_depth = 2,
  min_leaf = 8,
  min_split = 16,
  huber_delta = 1.5
)


# ------------------------------------------------------------------------------
# 40. Compare Against Clean Signal
# ------------------------------------------------------------------------------

squared_clean_MSE <- mean(
  (
    y_true -
      predict_gradient_boosting(
        GB_squared_outlier,
        X
      )
  )^2
)


huber_clean_MSE <- mean(
  (
    y_true -
      predict_gradient_boosting(
        GB_huber,
        X
      )
  )^2
)


data.frame(
  Loss = c(
    "Squared Error",
    "Huber"
  ),
  MSE_Against_True_Function = c(
    squared_clean_MSE,
    huber_clean_MSE
  )
)


# ==============================================================================
# PART XVII
#
# HUBER PSEUDO-RESIDUALS
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Compare Gradients
# ------------------------------------------------------------------------------

initial_squared <- rep(
  mean(
    y_outlier
  ),
  n
)


initial_huber <- rep(
  median(
    y_outlier
  ),
  n
)


squared_gradient <- squared_error_gradient(
  y_outlier,
  initial_squared
)


huber_gradient_values <- huber_gradient(
  y_outlier,
  initial_huber,
  delta = 1.5
)


plot(
  squared_gradient,
  huber_gradient_values,
  pch = 19,
  cex = 0.6,
  xlab = "Squared-Error Negative Gradient",
  ylab = "Huber Negative Gradient",
  main = "Robust Gradient Clipping"
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# Huber loss clips extreme pseudo-residuals.
#
# Squared-error gradients can become arbitrarily large.


# ==============================================================================
# PART XVIII
#
# VARIABLE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Importance from Split Improvements
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


gradient_boosting_importance <- function(
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
    
    importance <- importance +
      tree_importance(
        model$trees[[m]],
        p
      )
  }
  
  
  names(
    importance
  ) <- model$feature_names
  
  
  if (
    sum(
      importance
    ) >
    0
  ) {
    
    importance <- importance /
      sum(
        importance
      )
  }
  
  
  return(
    importance
  )
}


importance <- gradient_boosting_importance(
  GB_train
)


importance


barplot(
  importance,
  ylab = "Relative Importance",
  main = "Gradient Boosting Variable Importance"
)


# ==============================================================================
# PART XIX
#
# TWO-DIMENSIONAL REGRESSION SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Prediction Grid
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


grid_prediction <- predict_gradient_boosting(
  GB_train,
  X_grid,
  M =
    one_SE_iteration
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
  main = "Gradient Boosting Prediction Surface"
)


contour(
  grid_x1,
  grid_x2,
  prediction_matrix,
  add = TRUE
)


# ==============================================================================
# PART XX
#
# LOGISTIC GRADIENT BOOSTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Generate Binary Classification Data
# ------------------------------------------------------------------------------

set.seed(321)


n_class <- 500


z1 <- runif(
  n_class,
  -3,
  3
)


z2 <- runif(
  n_class,
  -3,
  3
)


classification_score <-
  1.5 * sin(z1) +
  1.25 * z2 -
  0.5 * z1 * z2


stable_sigmoid <- function(z) {
  
  result <- numeric(
    length(z)
  )
  
  
  positive <- z >= 0
  
  
  result[
    positive
  ] <- 1 /
    (
      1 +
        exp(
          -z[
            positive
          ]
        )
    )
  
  
  exp_z <- exp(
    z[
      !positive
    ]
  )
  
  
  result[
    !positive
  ] <- exp_z /
    (
      1 +
        exp_z
    )
  
  
  return(
    result
  )
}


class_probability <- stable_sigmoid(
  classification_score
)


y_class <- rbinom(
  n_class,
  size = 1,
  prob =
    class_probability
)


X_class <- cbind(
  z1 = z1,
  z2 = z2
)


# ==============================================================================
# PART XXI
#
# LOGISTIC LOSS
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Stable Softplus
# ------------------------------------------------------------------------------

softplus <- function(z) {
  
  pmax(
    z,
    0
  ) +
    log1p(
      exp(
        -abs(z)
      )
    )
}


# ------------------------------------------------------------------------------
# 46. Logistic Loss
# ------------------------------------------------------------------------------

logistic_loss <- function(
    y,
    F
) {
  
  softplus(
    F
  ) -
    y * F
}


# ------------------------------------------------------------------------------
# 47. Logistic Negative Gradient
# ------------------------------------------------------------------------------

logistic_negative_gradient <- function(
    y,
    F
) {
  
  probability <- stable_sigmoid(
    F
  )
  
  
  y -
    probability
}


# ==============================================================================
# PART XXII
#
# LOGISTIC GRADIENT BOOSTING MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Fit Classification Gradient Boosting
# ------------------------------------------------------------------------------

fit_logistic_gradient_boosting <- function(
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
  
  
  # --------------------------------------------------------------------------
  # Initial Log-Odds
  # --------------------------------------------------------------------------
  
  class_mean <- mean(
    y
  )
  
  
  class_mean <- pmin(
    pmax(
      class_mean,
      1e-8
    ),
    1 - 1e-8
  )
  
  
  F0 <- log(
    class_mean /
      (
        1 -
          class_mean
      )
  )
  
  
  F <- rep(
    F0,
    length(y)
  )
  
  
  trees <- vector(
    "list",
    M
  )
  
  
  training_loss <- numeric(
    M
  )
  
  
  training_error <- numeric(
    M
  )
  
  
  gradient_norm <- numeric(
    M
  )
  
  
  for (
    m in seq_len(M)
  ) {
    
    # ------------------------------------------------------------------------
    # Negative Gradient
    # ------------------------------------------------------------------------
    
    pseudo_residual <- logistic_negative_gradient(
      y,
      F
    )
    
    
    gradient_norm[m] <- sqrt(
      mean(
        pseudo_residual^2
      )
    )
    
    
    # ------------------------------------------------------------------------
    # Fit Regression Tree to Pseudo-Residuals
    # ------------------------------------------------------------------------
    
    tree_m <- grow_tree(
      X = X,
      target =
        pseudo_residual,
      max_depth =
        tree_depth,
      min_leaf =
        min_leaf,
      min_split =
        min_split,
      maximum_splits =
        maximum_splits
    )
    
    
    h_m <- predict_tree(
      tree_m,
      X
    )
    
    
    # ------------------------------------------------------------------------
    # Update Log-Odds Function
    # ------------------------------------------------------------------------
    
    F <- F +
      learning_rate *
      h_m
    
    
    trees[[m]] <- tree_m
    
    
    probability <- stable_sigmoid(
      F
    )
    
    
    prediction <- ifelse(
      probability >=
        0.5,
      1,
      0
    )
    
    
    training_loss[m] <- mean(
      logistic_loss(
        y,
        F
      )
    )
    
    
    training_error[m] <- mean(
      prediction !=
        y
    )
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        m,
        "| Log Loss:",
        round(
          training_loss[m],
          5
        ),
        "| Error:",
        round(
          training_error[m],
          5
        ),
        "\n"
      )
    }
  }
  
  
  return(
    list(
      initial_score =
        F0,
      trees =
        trees,
      fitted_score =
        F,
      fitted_probability =
        stable_sigmoid(F),
      learning_rate =
        learning_rate,
      tree_depth =
        tree_depth,
      training_loss =
        training_loss,
      training_error =
        training_error,
      gradient_norm =
        gradient_norm,
      feature_names =
        colnames(X)
    )
  )
}


# ------------------------------------------------------------------------------
# 49. Fit Logistic Gradient Boosting
# ------------------------------------------------------------------------------

GB_class <- fit_logistic_gradient_boosting(
  X = X_class,
  y = y_class,
  M = 300,
  learning_rate = 0.05,
  tree_depth = 2,
  min_leaf = 8,
  min_split = 16
)


# ==============================================================================
# PART XXIII
#
# LOGISTIC PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Predict Logistic Score
# ------------------------------------------------------------------------------

predict_logistic_GB_score <- function(
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
  
  
  F <- rep(
    model$initial_score,
    nrow(
      X_new
    )
  )
  
  
  for (
    m in seq_len(M)
  ) {
    
    F <- F +
      model$learning_rate *
      predict_tree(
        model$trees[[m]],
        X_new
      )
  }
  
  
  return(
    F
  )
}


# ------------------------------------------------------------------------------
# 51. Probability Prediction
# ------------------------------------------------------------------------------

predict_logistic_GB_probability <- function(
    model,
    X_new,
    M = length(
      model$trees
    )
) {
  
  stable_sigmoid(
    predict_logistic_GB_score(
      model,
      X_new,
      M
    )
  )
}


# ------------------------------------------------------------------------------
# 52. Class Prediction
# ------------------------------------------------------------------------------

predict_logistic_GB <- function(
    model,
    X_new,
    M = length(
      model$trees
    ),
    threshold = 0.5
) {
  
  probability <-
    predict_logistic_GB_probability(
      model,
      X_new,
      M
    )
  
  
  ifelse(
    probability >=
      threshold,
    1,
    0
  )
}


# ==============================================================================
# PART XXIV
#
# CLASSIFICATION PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. Training Performance
# ------------------------------------------------------------------------------

class_prediction <- predict_logistic_GB(
  GB_class,
  X_class
)


class_accuracy <- mean(
  class_prediction ==
    y_class
)


class_error <- mean(
  class_prediction !=
    y_class
)


class_accuracy


# ------------------------------------------------------------------------------
# 54. Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y_class,
  Predicted =
    class_prediction
)


# ------------------------------------------------------------------------------
# 55. Logistic Loss Path
# ------------------------------------------------------------------------------

plot(
  seq_along(
    GB_class$training_loss
  ),
  GB_class$training_loss,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Log Loss",
  main = "Logistic Gradient Boosting"
)


# ------------------------------------------------------------------------------
# 56. Classification Error Path
# ------------------------------------------------------------------------------

plot(
  seq_along(
    GB_class$training_error
  ),
  GB_class$training_error,
  type = "l",
  lwd = 2,
  xlab = "Boosting Iteration",
  ylab = "Classification Error",
  main = "Gradient Boosting Classification Error"
)


# ==============================================================================
# PART XXV
#
# LOGISTIC PSEUDO-RESIDUALS
# ==============================================================================


# ------------------------------------------------------------------------------
# 57. Initial Classification Gradient
# ------------------------------------------------------------------------------

initial_class_mean <- mean(
  y_class
)


initial_log_odds <- log(
  initial_class_mean /
    (
      1 -
        initial_class_mean
    )
)


initial_score <- rep(
  initial_log_odds,
  n_class
)


initial_probability <- stable_sigmoid(
  initial_score
)


classification_pseudo_residual <-
  logistic_negative_gradient(
    y_class,
    initial_score
  )


head(
  data.frame(
    y = y_class,
    Probability =
      initial_probability,
    Pseudo_Residual =
      classification_pseudo_residual
  )
)


# Since:
#
#   r_i = y_i - p_i
#
# positive-class observations tend to have positive pseudo-residuals,
# while negative-class observations tend to have negative pseudo-residuals.


# ==============================================================================
# PART XXVI
#
# CLASSIFICATION DECISION SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 58. Grid
# ------------------------------------------------------------------------------

class_grid_size <- 100


class_grid_z1 <- seq(
  -3,
  3,
  length.out =
    class_grid_size
)


class_grid_z2 <- seq(
  -3,
  3,
  length.out =
    class_grid_size
)


class_grid <- expand.grid(
  z1 = class_grid_z1,
  z2 = class_grid_z2
)


class_grid_probability <-
  predict_logistic_GB_probability(
    GB_class,
    as.matrix(
      class_grid
    )
  )


probability_matrix <- matrix(
  class_grid_probability,
  nrow =
    class_grid_size,
  ncol =
    class_grid_size
)


image(
  class_grid_z1,
  class_grid_z2,
  probability_matrix,
  xlab = "z1",
  ylab = "z2",
  main = "Gradient Boosting Class Probability"
)


contour(
  class_grid_z1,
  class_grid_z2,
  probability_matrix,
  levels = 0.5,
  add = TRUE,
  lwd = 3
)


points(
  z1,
  z2,
  pch = ifelse(
    y_class == 1,
    19,
    1
  ),
  cex = 0.5
)


# ==============================================================================
# PART XXVII
#
# LOGISTIC REGRESSION COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 59. Global Logistic Regression
# ------------------------------------------------------------------------------

class_data <- data.frame(
  y = y_class,
  z1 = z1,
  z2 = z2
)


logistic_model <- glm(
  y ~ z1 + z2,
  data =
    class_data,
  family =
    binomial()
)


logistic_probability <- predict(
  logistic_model,
  type = "response"
)


logistic_prediction <- ifelse(
  logistic_probability >=
    0.5,
  1,
  0
)


logistic_accuracy <- mean(
  logistic_prediction ==
    y_class
)


data.frame(
  Model = c(
    "Logistic Regression",
    "Logistic Gradient Boosting"
  ),
  Accuracy = c(
    logistic_accuracy,
    class_accuracy
  )
)


# ==============================================================================
# PART XXVIII
#
# COMPARE WITH LINEAR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 60. Regression Baseline
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


# ------------------------------------------------------------------------------
# 61. Model Comparison
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Gradient Boosting"
  ),
  Test_MSE = c(
    linear_test_MSE,
    final_test_MSE
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XXIX
#
# OPTIONAL VERIFICATION WITH GBM
# ==============================================================================


# ------------------------------------------------------------------------------
# 62. Production Gradient Boosting Comparison
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
    "Manual gradient boosting test MSE:",
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
# PART XXX
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 63. Summary
# ------------------------------------------------------------------------------

cat(
  "Gradient Boosting Summary\n"
)


cat(
  "-------------------------\n"
)


cat(
  "Regression loss:",
  GB_train$loss,
  "\n"
)


cat(
  "Learning rate:",
  GB_train$learning_rate,
  "\n"
)


cat(
  "Tree depth:",
  GB_train$tree_depth,
  "\n"
)


cat(
  "Total fitted trees:",
  length(
    GB_train$trees
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
  "Gradient boosting test MSE:",
  round(
    final_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Gradient boosting test RMSE:",
  round(
    final_test_RMSE,
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


cat(
  "\nClassification gradient boosting accuracy:",
  round(
    class_accuracy,
    4
  ),
  "\n"
)