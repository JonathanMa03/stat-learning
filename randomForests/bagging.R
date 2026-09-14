# ==============================================================================
# Bagging: Bootstrap Aggregating
# Jonathan Ma
#
# Main ideas:
#   - Bootstrap sampling
#   - Unstable estimators
#   - Variance reduction through averaging
#   - Bagged regression trees
#   - Out-of-bag observations
#   - Out-of-bag prediction and error
#   - Number of bootstrap replicates / trees
#   - Tree correlation
#   - Variable importance
#   - Bagging vs a single tree
#   - Bagging vs Random Forests
#
#
# Core regression bagging estimator:
#
#       f_hat_bag(x)
#
#       =
#
#       (1 / B) * sum_{b=1}^B f_hat^{*b}(x)
#
#
# where each f_hat^{*b} is trained on a bootstrap sample.
#
#
# For classification:
#
#       C_hat_bag(x)
#
#       =
#
#       majority vote of C_hat^{*1}(x), ..., C_hat^{*B}(x)
#
#
# Bagging is especially useful for high-variance estimators such as deep
# decision trees.
# ==============================================================================


# ==============================================================================
# PART I
#
# SIMULATE NONLINEAR REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Predictors
# ------------------------------------------------------------------------------

set.seed(123)

n <- 600

p <- 8


X <- matrix(
  rnorm(
    n * p
  ),
  nrow = n,
  ncol = p
)


colnames(X) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 2. True Regression Function
#
# Only X1:X5 matter.
#
# The function contains:
#
#   - smooth nonlinear effects
#   - threshold effects
#   - interactions
#
# A single linear model will therefore be misspecified.
# ------------------------------------------------------------------------------

true_function <- function(X) {
  
  3 *
    sin(
      X[, 1]
    ) +
    2 *
    (
      X[, 2] >
        0
    ) +
    1.5 *
    X[, 3]^2 -
    2 *
    X[, 4] *
    X[, 5]
}


signal <- true_function(
  X
)


sigma <- 1.5


y <- signal +
  rnorm(
    n,
    sd = sigma
  )


# ------------------------------------------------------------------------------
# 3. Quick Exploration
# ------------------------------------------------------------------------------

par(
  mfrow = c(2, 2)
)


plot(
  X[, 1],
  y,
  pch = 19,
  cex = 0.6,
  xlab = "X1",
  ylab = "Y",
  main = "Response vs X1"
)


plot(
  X[, 2],
  y,
  pch = 19,
  cex = 0.6,
  xlab = "X2",
  ylab = "Y",
  main = "Response vs X2"
)


plot(
  X[, 3],
  y,
  pch = 19,
  cex = 0.6,
  xlab = "X3",
  ylab = "Y",
  main = "Response vs X3"
)


hist(
  y,
  breaks = 30,
  main = "Response Distribution",
  xlab = "Y"
)


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART II
#
# TRAIN / TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Split Data
# ------------------------------------------------------------------------------

set.seed(456)


train_indices <- sample(
  seq_len(n),
  size = floor(
    0.70 * n
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


y_train <- y[
  train_indices
]


X_test <- X[
  test_indices,
  ,
  drop = FALSE
]


y_test <- y[
  test_indices
]


n_train <- nrow(
  X_train
)


n_test <- nrow(
  X_test
)


# ==============================================================================
# PART III
#
# REGRESSION METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Evaluation Function
# ------------------------------------------------------------------------------

regression_metrics <- function(
    y,
    prediction
) {
  
  mse <- mean(
    (
      y -
        prediction
    )^2
  )
  
  
  rmse <- sqrt(
    mse
  )
  
  
  mae <- mean(
    abs(
      y -
        prediction
    )
  )
  
  
  r_squared <- 1 -
    sum(
      (
        y -
          prediction
      )^2
    ) /
    sum(
      (
        y -
          mean(y)
      )^2
    )
  
  
  c(
    MSE = mse,
    RMSE = rmse,
    MAE = mae,
    R2 = r_squared
  )
}


# ==============================================================================
# PART IV
#
# MANUAL REGRESSION TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. RSS Within a Node
# ------------------------------------------------------------------------------

node_rss <- function(y) {
  
  if (
    length(y) ==
    0
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
# 7. Candidate Split Values
#
# Evaluating every possible split can be slow when this function is repeatedly
# called inside hundreds of bootstrap trees.
#
# We therefore use every midpoint when the number is small and otherwise use
# a quantile-based approximation.
# ------------------------------------------------------------------------------

candidate_splits <- function(
    x,
    maximum_splits = 30
) {
  
  unique_values <- sort(
    unique(x)
  )
  
  
  if (
    length(unique_values) <=
    1
  ) {
    
    return(
      numeric(0)
    )
  }
  
  
  midpoints <- (
    unique_values[
      -length(unique_values)
    ] +
      unique_values[
        -1
      ]
  ) /
    2
  
  
  if (
    length(midpoints) <=
    maximum_splits
  ) {
    
    return(
      midpoints
    )
  }
  
  
  probabilities <- seq(
    0,
    1,
    length.out =
      maximum_splits + 2
  )
  
  
  probabilities <- probabilities[
    -c(
      1,
      length(probabilities)
    )
  ]
  
  
  split_values <- as.numeric(
    quantile(
      x,
      probs = probabilities,
      names = FALSE
    )
  )
  
  
  sort(
    unique(
      split_values
    )
  )
}


# ==============================================================================
# PART V
#
# BEST CART SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Find Best Split
#
# IMPORTANT:
#
# Bagging uses ALL predictors at each node.
#
# This is the key difference from Random Forests.
# ------------------------------------------------------------------------------

find_best_split <- function(
    X,
    y,
    minimum_leaf_size = 5,
    maximum_splits = 30
) {
  
  X <- as.matrix(X)
  
  
  p <- ncol(X)
  
  
  parent_rss <- node_rss(
    y
  )
  
  
  best_improvement <- -Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  best_left <- NULL
  
  best_right <- NULL
  
  
  for (
    variable in seq_len(p)
  ) {
    
    x <- X[, variable]
    
    
    splits <- candidate_splits(
      x,
      maximum_splits =
        maximum_splits
    )
    
    
    if (
      length(splits) ==
      0
    ) {
      
      next
    }
    
    
    for (
      split_value in splits
    ) {
      
      left <- which(
        x <=
          split_value
      )
      
      
      right <- which(
        x >
          split_value
      )
      
      
      if (
        length(left) <
        minimum_leaf_size ||
        length(right) <
        minimum_leaf_size
      ) {
        
        next
      }
      
      
      child_rss <- node_rss(
        y[left]
      ) +
        node_rss(
          y[right]
        )
      
      
      improvement <- parent_rss -
        child_rss
      
      
      if (
        improvement >
        best_improvement
      ) {
        
        best_improvement <- improvement
        
        best_variable <- variable
        
        best_split <- split_value
        
        best_left <- left
        
        best_right <- right
      }
    }
  }
  
  
  if (
    is.na(best_variable)
  ) {
    
    return(NULL)
  }
  
  
  list(
    variable =
      best_variable,
    split =
      best_split,
    improvement =
      best_improvement,
    left =
      best_left,
    right =
      best_right
  )
}


# ==============================================================================
# PART VI
#
# GROW A DEEP REGRESSION TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Recursive CART-Style Tree
# ------------------------------------------------------------------------------

grow_regression_tree <- function(
    X,
    y,
    minimum_leaf_size = 5,
    maximum_depth = 20,
    maximum_splits = 30,
    current_depth = 0
) {
  
  X <- as.matrix(X)
  
  
  n_node <- nrow(X)
  
  
  node_prediction <- mean(
    y
  )
  
  
  # --------------------------------------------------------------------------
  # Stopping Rules
  # --------------------------------------------------------------------------
  
  if (
    n_node <
    2 * minimum_leaf_size ||
    current_depth >=
    maximum_depth ||
    length(
      unique(y)
    ) ==
    1
  ) {
    
    return(
      list(
        terminal = TRUE,
        prediction =
          node_prediction,
        n =
          n_node
      )
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Search ALL Variables
  # --------------------------------------------------------------------------
  
  best_split <- find_best_split(
    X,
    y,
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_splits =
      maximum_splits
  )
  
  
  if (
    is.null(best_split)
  ) {
    
    return(
      list(
        terminal = TRUE,
        prediction =
          node_prediction,
        n =
          n_node
      )
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Recursively Grow Children
  # --------------------------------------------------------------------------
  
  left_tree <- grow_regression_tree(
    X[
      best_split$left,
      ,
      drop = FALSE
    ],
    y[
      best_split$left
    ],
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_depth =
      maximum_depth,
    maximum_splits =
      maximum_splits,
    current_depth =
      current_depth + 1
  )
  
  
  right_tree <- grow_regression_tree(
    X[
      best_split$right,
      ,
      drop = FALSE
    ],
    y[
      best_split$right
    ],
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_depth =
      maximum_depth,
    maximum_splits =
      maximum_splits,
    current_depth =
      current_depth + 1
  )
  
  
  list(
    terminal = FALSE,
    prediction =
      node_prediction,
    variable =
      best_split$variable,
    split =
      best_split$split,
    improvement =
      best_split$improvement,
    left =
      left_tree,
    right =
      right_tree,
    n =
      n_node
  )
}


# ==============================================================================
# PART VII
#
# TREE PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Predict One Observation
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
    
  } else {
    
    return(
      predict_tree_one(
        tree$right,
        x
      )
    )
  }
}


# ------------------------------------------------------------------------------
# 11. Predict Multiple Observations
# ------------------------------------------------------------------------------

predict_tree <- function(
    tree,
    X
) {
  
  X <- as.matrix(X)
  
  
  predictions <- numeric(
    nrow(X)
  )
  
  
  for (
    i in seq_len(
      nrow(X)
    )
  ) {
    
    predictions[i] <- predict_tree_one(
      tree,
      X[
        i,
      ]
    )
  }
  
  
  predictions
}


# ==============================================================================
# PART VIII
#
# SINGLE TREE BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Fit One Deep Tree
# ------------------------------------------------------------------------------

single_tree <- grow_regression_tree(
  X_train,
  y_train,
  minimum_leaf_size = 5,
  maximum_depth = 20,
  maximum_splits = 30
)


# ------------------------------------------------------------------------------
# 13. Predictions
# ------------------------------------------------------------------------------

single_tree_train_prediction <- predict_tree(
  single_tree,
  X_train
)


single_tree_test_prediction <- predict_tree(
  single_tree,
  X_test
)


# ------------------------------------------------------------------------------
# 14. Performance
# ------------------------------------------------------------------------------

regression_metrics(
  y_train,
  single_tree_train_prediction
)


regression_metrics(
  y_test,
  single_tree_test_prediction
)


# A deep tree should generally have much lower training error than test error.
#
# The tree is flexible but unstable.


# ==============================================================================
# PART IX
#
# BOOTSTRAP SAMPLING
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. One Bootstrap Sample
# ------------------------------------------------------------------------------

set.seed(123)


bootstrap_indices <- sample(
  seq_len(n_train),
  size = n_train,
  replace = TRUE
)


head(
  bootstrap_indices
)


# ------------------------------------------------------------------------------
# 16. Number of Unique Observations
# ------------------------------------------------------------------------------

number_unique <- length(
  unique(
    bootstrap_indices
  )
)


proportion_unique <- number_unique /
  n_train


proportion_unique


# Approximately 63.2% of the original observations appear at least once.


# ------------------------------------------------------------------------------
# 17. Out-of-Bag Observations
# ------------------------------------------------------------------------------

oob_indices <- setdiff(
  seq_len(n_train),
  unique(
    bootstrap_indices
  )
)


length(oob_indices) /
  n_train


# Approximately 36.8% are OOB.


# ==============================================================================
# PART X
#
# WHY 63.2%?
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Theoretical Probability
#
# Probability observation i is NOT selected in n bootstrap draws:
#
#       (1 - 1/n)^n
#
# As n -> infinity:
#
#       -> exp(-1)
#       ~= 0.367879
#
# Therefore:
#
#       P(selected at least once)
#
#       ~= 1 - exp(-1)
#
#       ~= 0.632121
# ------------------------------------------------------------------------------

probability_not_selected <- (
  1 -
    1 /
    n_train
)^n_train


probability_selected <- 1 -
  probability_not_selected


probability_not_selected


probability_selected


exp(-1)


1 -
  exp(-1)


# ==============================================================================
# PART XI
#
# FIT ONE BOOTSTRAP TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Bootstrap Tree
# ------------------------------------------------------------------------------

bootstrap_tree <- grow_regression_tree(
  X_train[
    bootstrap_indices,
    ,
    drop = FALSE
  ],
  y_train[
    bootstrap_indices
  ],
  minimum_leaf_size = 5,
  maximum_depth = 20,
  maximum_splits = 30
)


# ------------------------------------------------------------------------------
# 20. Compare Original and Bootstrap Tree Predictions
# ------------------------------------------------------------------------------

original_prediction <- predict_tree(
  single_tree,
  X_test
)


bootstrap_prediction <- predict_tree(
  bootstrap_tree,
  X_test
)


plot(
  original_prediction,
  bootstrap_prediction,
  pch = 19,
  cex = 0.7,
  xlab = "Original Tree Prediction",
  ylab = "Bootstrap Tree Prediction",
  main = "Instability of Regression Trees"
)


abline(
  0,
  1,
  lty = 2
)


cor(
  original_prediction,
  bootstrap_prediction
)


# The two trees use almost the same underlying population but can produce
# noticeably different fitted functions.


# ==============================================================================
# PART XII
#
# MANUAL BAGGING
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Fit Bagged Regression Trees
# ------------------------------------------------------------------------------

fit_bagging <- function(
    X,
    y,
    number_trees = 200,
    minimum_leaf_size = 5,
    maximum_depth = 20,
    maximum_splits = 30,
    seed = 123,
    verbose = FALSE
) {
  
  X <- as.matrix(X)
  
  
  n <- nrow(X)
  
  
  trees <- vector(
    "list",
    number_trees
  )
  
  
  bootstrap_samples <- vector(
    "list",
    number_trees
  )
  
  
  oob_samples <- vector(
    "list",
    number_trees
  )
  
  
  # Running OOB predictions.
  
  oob_prediction_sum <- numeric(
    n
  )
  
  
  oob_prediction_count <- integer(
    n
  )
  
  
  oob_mse_path <- rep(
    NA_real_,
    number_trees
  )
  
  
  set.seed(seed)
  
  
  for (
    tree_index in seq_len(
      number_trees
    )
  ) {
    
    # ------------------------------------------------------------------------
    # Bootstrap Sample
    # ------------------------------------------------------------------------
    
    bootstrap_indices <- sample(
      seq_len(n),
      size = n,
      replace = TRUE
    )
    
    
    bootstrap_samples[[tree_index]] <-
      bootstrap_indices
    
    
    # ------------------------------------------------------------------------
    # OOB Sample
    # ------------------------------------------------------------------------
    
    oob_indices <- setdiff(
      seq_len(n),
      unique(
        bootstrap_indices
      )
    )
    
    
    oob_samples[[tree_index]] <-
      oob_indices
    
    
    # ------------------------------------------------------------------------
    # Fit Deep Tree
    #
    # Every variable is available at every split.
    # ------------------------------------------------------------------------
    
    tree <- grow_regression_tree(
      X[
        bootstrap_indices,
        ,
        drop = FALSE
      ],
      y[
        bootstrap_indices
      ],
      minimum_leaf_size =
        minimum_leaf_size,
      maximum_depth =
        maximum_depth,
      maximum_splits =
        maximum_splits
    )
    
    
    trees[[tree_index]] <-
      tree
    
    
    # ------------------------------------------------------------------------
    # Update OOB Predictions
    # ------------------------------------------------------------------------
    
    if (
      length(oob_indices) >
      0
    ) {
      
      tree_oob_prediction <- predict_tree(
        tree,
        X[
          oob_indices,
          ,
          drop = FALSE
        ]
      )
      
      
      oob_prediction_sum[
        oob_indices
      ] <- oob_prediction_sum[
        oob_indices
      ] +
        tree_oob_prediction
      
      
      oob_prediction_count[
        oob_indices
      ] <- oob_prediction_count[
        oob_indices
      ] +
        1
    }
    
    
    # ------------------------------------------------------------------------
    # Current OOB Error
    # ------------------------------------------------------------------------
    
    available <- oob_prediction_count >
      0
    
    
    if (
      any(available)
    ) {
      
      current_oob_prediction <-
        oob_prediction_sum[
          available
        ] /
        oob_prediction_count[
          available
        ]
      
      
      oob_mse_path[
        tree_index
      ] <- mean(
        (
          y[
            available
          ] -
            current_oob_prediction
        )^2
      )
    }
    
    
    if (
      verbose &&
      tree_index %% 25 ==
      0
    ) {
      
      cat(
        "Finished tree",
        tree_index,
        "of",
        number_trees,
        "\n"
      )
    }
  }
  
  
  # --------------------------------------------------------------------------
  # Final OOB Predictions
  # --------------------------------------------------------------------------
  
  oob_predictions <- rep(
    NA_real_,
    n
  )
  
  
  available <- oob_prediction_count >
    0
  
  
  oob_predictions[
    available
  ] <- oob_prediction_sum[
    available
  ] /
    oob_prediction_count[
      available
    ]
  
  
  list(
    trees =
      trees,
    bootstrap_samples =
      bootstrap_samples,
    oob_samples =
      oob_samples,
    number_trees =
      number_trees,
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_depth =
      maximum_depth,
    maximum_splits =
      maximum_splits,
    oob_predictions =
      oob_predictions,
    oob_counts =
      oob_prediction_count,
    oob_mse_path =
      oob_mse_path
  )
}


# ==============================================================================
# PART XIII
#
# FIT THE BAGGED ENSEMBLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Train
# ------------------------------------------------------------------------------

bagged_model <- fit_bagging(
  X_train,
  y_train,
  number_trees = 300,
  minimum_leaf_size = 5,
  maximum_depth = 20,
  maximum_splits = 30,
  seed = 123,
  verbose = TRUE
)


# ==============================================================================
# PART XIV
#
# BAGGED PREDICTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Prediction Function
# ------------------------------------------------------------------------------

predict_bagging <- function(
    model,
    X,
    number_trees = NULL
) {
  
  X <- as.matrix(X)
  
  
  if (
    is.null(number_trees)
  ) {
    
    number_trees <- model$number_trees
  }
  
  
  number_trees <- min(
    number_trees,
    model$number_trees
  )
  
  
  prediction_sum <- numeric(
    nrow(X)
  )
  
  
  for (
    tree_index in seq_len(
      number_trees
    )
  ) {
    
    prediction_sum <- prediction_sum +
      predict_tree(
        model$trees[[tree_index]],
        X
      )
  }
  
  
  prediction_sum /
    number_trees
}


# ------------------------------------------------------------------------------
# 24. Train and Test Predictions
# ------------------------------------------------------------------------------

bagged_train_prediction <- predict_bagging(
  bagged_model,
  X_train
)


bagged_test_prediction <- predict_bagging(
  bagged_model,
  X_test
)


# ------------------------------------------------------------------------------
# 25. Metrics
# ------------------------------------------------------------------------------

bagged_train_metrics <- regression_metrics(
  y_train,
  bagged_train_prediction
)


bagged_test_metrics <- regression_metrics(
  y_test,
  bagged_test_prediction
)


bagged_train_metrics


bagged_test_metrics


# ==============================================================================
# PART XV
#
# SINGLE TREE VS BAGGING
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Performance Comparison
# ------------------------------------------------------------------------------

single_tree_metrics <- regression_metrics(
  y_test,
  single_tree_test_prediction
)


comparison <- data.frame(
  Method = c(
    "Single Tree",
    "Bagging"
  ),
  MSE = c(
    single_tree_metrics["MSE"],
    bagged_test_metrics["MSE"]
  ),
  RMSE = c(
    single_tree_metrics["RMSE"],
    bagged_test_metrics["RMSE"]
  ),
  MAE = c(
    single_tree_metrics["MAE"],
    bagged_test_metrics["MAE"]
  ),
  R2 = c(
    single_tree_metrics["R2"],
    bagged_test_metrics["R2"]
  )
)


comparison


# ==============================================================================
# PART XVI
#
# OUT-OF-BAG PREDICTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Number of OOB Trees Per Observation
# ------------------------------------------------------------------------------

summary(
  bagged_model$oob_counts
)


hist(
  bagged_model$oob_counts,
  breaks = 25,
  xlab = "Number of Trees",
  main = "Number of OOB Trees per Training Observation"
)


# For B trees, the expected number is approximately:
#
#       0.368 * B


0.368 *
  bagged_model$number_trees


# ==============================================================================
# PART XVII
#
# OOB ERROR
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. OOB Performance
# ------------------------------------------------------------------------------

available_oob <- !is.na(
  bagged_model$oob_predictions
)


oob_metrics <- regression_metrics(
  y_train[
    available_oob
  ],
  bagged_model$oob_predictions[
    available_oob
  ]
)


oob_metrics


# ------------------------------------------------------------------------------
# 29. Compare OOB and Test Error
# ------------------------------------------------------------------------------

data.frame(
  Error_Estimate = c(
    "OOB",
    "Independent Test"
  ),
  MSE = c(
    oob_metrics["MSE"],
    bagged_test_metrics["MSE"]
  )
)


# ==============================================================================
# PART XVIII
#
# OOB ERROR PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. OOB MSE as Trees Accumulate
# ------------------------------------------------------------------------------

plot(
  seq_len(
    bagged_model$number_trees
  ),
  bagged_model$oob_mse_path,
  type = "l",
  xlab = "Number of Trees",
  ylab = "OOB MSE",
  main = "Bagging: OOB Error vs Number of Trees"
)


abline(
  h =
    oob_metrics["MSE"],
  lty = 2
)


# ==============================================================================
# PART XIX
#
# TEST ERROR VS NUMBER OF TREES
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Evaluate Growing Ensemble
# ------------------------------------------------------------------------------

tree_grid <- unique(
  round(
    seq(
      1,
      bagged_model$number_trees,
      length.out = 40
    )
  )
)


test_mse_path <- numeric(
  length(tree_grid)
)


for (
  i in seq_along(
    tree_grid
  )
) {
  
  prediction_i <- predict_bagging(
    bagged_model,
    X_test,
    number_trees =
      tree_grid[i]
  )
  
  
  test_mse_path[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


# ------------------------------------------------------------------------------
# 32. Plot
# ------------------------------------------------------------------------------

plot(
  tree_grid,
  test_mse_path,
  type = "b",
  pch = 19,
  xlab = "Number of Trees",
  ylab = "Test MSE",
  main = "Bagging Test Error vs Number of Trees"
)


# Adding trees usually causes the prediction to stabilize rather than creating
# classical overfitting.


# ==============================================================================
# PART XX
#
# HOW ENSEMBLE PREDICTIONS STABILIZE
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Follow One Test Observation
# ------------------------------------------------------------------------------

example_x <- X_test[
  1,
  ,
  drop = FALSE
]


individual_predictions <- numeric(
  bagged_model$number_trees
)


cumulative_predictions <- numeric(
  bagged_model$number_trees
)


for (
  tree_index in seq_len(
    bagged_model$number_trees
  )
) {
  
  individual_predictions[
    tree_index
  ] <- predict_tree(
    bagged_model$trees[[tree_index]],
    example_x
  )
  
  
  cumulative_predictions[
    tree_index
  ] <- mean(
    individual_predictions[
      seq_len(tree_index)
    ]
  )
}


# ------------------------------------------------------------------------------
# 34. Individual Tree Predictions
# ------------------------------------------------------------------------------

plot(
  individual_predictions,
  pch = 19,
  cex = 0.5,
  xlab = "Tree",
  ylab = "Prediction",
  main = "Individual Bootstrap-Tree Predictions"
)


abline(
  h =
    bagged_test_prediction[1],
  lty = 2
)


# ------------------------------------------------------------------------------
# 35. Cumulative Average
# ------------------------------------------------------------------------------

plot(
  cumulative_predictions,
  type = "l",
  xlab = "Number of Trees",
  ylab = "Cumulative Prediction",
  main = "Bagged Prediction Stabilization"
)


abline(
  h =
    bagged_test_prediction[1],
  lty = 2
)


# ==============================================================================
# PART XXI
#
# VARIABILITY ACROSS TREES
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Tree Prediction Distribution
# ------------------------------------------------------------------------------

mean(
  individual_predictions
)


sd(
  individual_predictions
)


bagged_test_prediction[1]


hist(
  individual_predictions,
  breaks = 25,
  xlab = "Individual Tree Prediction",
  main = "Bootstrap Distribution of Tree Predictions"
)


abline(
  v =
    bagged_test_prediction[1],
  lwd = 2
)


# ==============================================================================
# PART XXII
#
# TREE CORRELATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Predictions from Individual Trees on Test Data
# ------------------------------------------------------------------------------

number_correlation_trees <- min(
  75,
  bagged_model$number_trees
)


tree_prediction_matrix <- matrix(
  NA_real_,
  nrow = n_test,
  ncol =
    number_correlation_trees
)


for (
  tree_index in seq_len(
    number_correlation_trees
  )
) {
  
  tree_prediction_matrix[
    ,
    tree_index
  ] <- predict_tree(
    bagged_model$trees[[tree_index]],
    X_test
  )
}


# ------------------------------------------------------------------------------
# 38. Pairwise Correlation
# ------------------------------------------------------------------------------

tree_correlation_matrix <- cor(
  tree_prediction_matrix
)


average_tree_correlation <- mean(
  tree_correlation_matrix[
    upper.tri(
      tree_correlation_matrix
    )
  ]
)


average_tree_correlation


hist(
  tree_correlation_matrix[
    upper.tri(
      tree_correlation_matrix
    )
  ],
  breaks = 25,
  xlab = "Pairwise Prediction Correlation",
  main = "Correlation Among Bagged Trees"
)


# This motivates Random Forests:
#
# Bagging reduces variance through averaging, but if the trees are highly
# correlated, averaging is less effective.


# ==============================================================================
# PART XXIII
#
# VARIANCE OF AN AVERAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Theoretical Illustration
#
# Suppose each estimator has variance sigma_tree^2 and every pair has
# correlation rho.
#
# Then:
#
# Var(mean(T_1,...,T_B))
#
#     =
#
# sigma_tree^2 *
#
# [ rho + (1-rho)/B ]
# ------------------------------------------------------------------------------

ensemble_variance_ratio <- function(
    B,
    rho
) {
  
  rho +
    (
      1 -
        rho
    ) /
    B
}


B_values <- 1:500


plot(
  B_values,
  ensemble_variance_ratio(
    B_values,
    rho = 0
  ),
  type = "l",
  ylim = c(
    0,
    1
  ),
  xlab = "Number of Estimators",
  ylab = "Variance / Individual Variance",
  main = "Effect of Correlation on Ensemble Variance"
)


lines(
  B_values,
  ensemble_variance_ratio(
    B_values,
    rho = 0.25
  )
)


lines(
  B_values,
  ensemble_variance_ratio(
    B_values,
    rho = 0.50
  )
)


lines(
  B_values,
  ensemble_variance_ratio(
    B_values,
    rho = 0.75
  )
)


legend(
  "topright",
  legend = c(
    "rho = 0",
    "rho = 0.25",
    "rho = 0.50",
    "rho = 0.75"
  ),
  lty = 1
)


# ==============================================================================
# PART XXIV
#
# DEMONSTRATE TREE INSTABILITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Fit Many Trees to Slightly Different Bootstrap Samples
# ------------------------------------------------------------------------------

number_instability_trees <- 50


instability_predictions <- matrix(
  NA_real_,
  nrow =
    n_test,
  ncol =
    number_instability_trees
)


set.seed(789)


for (
  b in seq_len(
    number_instability_trees
  )
) {
  
  indices_b <- sample(
    seq_len(n_train),
    size = n_train,
    replace = TRUE
  )
  
  
  tree_b <- grow_regression_tree(
    X_train[
      indices_b,
      ,
      drop = FALSE
    ],
    y_train[
      indices_b
    ],
    minimum_leaf_size = 5,
    maximum_depth = 20,
    maximum_splits = 30
  )
  
  
  instability_predictions[
    ,
    b
  ] <- predict_tree(
    tree_b,
    X_test
  )
}


# ------------------------------------------------------------------------------
# 41. Standard Deviation Across Trees for Each Test Observation
# ------------------------------------------------------------------------------

prediction_sd <- apply(
  instability_predictions,
  1,
  sd
)


summary(
  prediction_sd
)


hist(
  prediction_sd,
  breaks = 30,
  xlab = "SD Across Bootstrap Trees",
  main = "Prediction Instability of Deep Trees"
)


# ==============================================================================
# PART XXV
#
# BAGGING WORKS BEST FOR UNSTABLE ESTIMATORS
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Stable Estimator Example: Sample Mean
#
# Bagging the sample mean changes essentially nothing.
#
# Each bootstrap mean fluctuates, but their average converges back toward the
# original sample mean.
# ------------------------------------------------------------------------------

set.seed(123)


z <- rnorm(
  200,
  mean = 5,
  sd = 2
)


original_mean <- mean(
  z
)


B <- 5000


bootstrap_means <- numeric(
  B
)


for (
  b in seq_len(B)
) {
  
  z_star <- sample(
    z,
    size = length(z),
    replace = TRUE
  )
  
  
  bootstrap_means[b] <- mean(
    z_star
  )
}


bagged_mean <- mean(
  bootstrap_means
)


c(
  Original_Mean =
    original_mean,
  Bagged_Mean =
    bagged_mean
)


# Bagging does not magically improve every estimator.
#
# It is especially valuable for unstable/high-variance procedures.


# ==============================================================================
# PART XXVI
#
# SPLIT-BASED VARIABLE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Collect Split Improvements
# ------------------------------------------------------------------------------

collect_tree_importance <- function(
    tree,
    p
) {
  
  importance <- numeric(
    p
  )
  
  
  recursive_collect <- function(
    node
  ) {
    
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
    
    
    recursive_collect(
      node$left
    )
    
    
    recursive_collect(
      node$right
    )
    
    
    invisible(NULL)
  }
  
  
  recursive_collect(
    tree
  )
  
  
  importance
}


# ------------------------------------------------------------------------------
# 44. Aggregate Across Trees
# ------------------------------------------------------------------------------

bagging_split_importance <- numeric(
  p
)


for (
  tree_index in seq_len(
    bagged_model$number_trees
  )
) {
  
  bagging_split_importance <-
    bagging_split_importance +
    collect_tree_importance(
      bagged_model$trees[[tree_index]],
      p
    )
}


bagging_split_importance <-
  bagging_split_importance /
  sum(
    bagging_split_importance
  )


importance_table <- data.frame(
  Variable =
    colnames(X_train),
  Importance =
    bagging_split_importance
)


importance_table <- importance_table[
  order(
    -importance_table$Importance
  ),
]


importance_table


# ------------------------------------------------------------------------------
# 45. Plot
# ------------------------------------------------------------------------------

barplot(
  importance_table$Importance,
  names.arg =
    importance_table$Variable,
  las = 2,
  ylab = "Relative RSS Improvement",
  main = "Bagging Variable Importance"
)


# ==============================================================================
# PART XXVII
#
# OOB PERMUTATION IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Permutation Importance
#
# For each tree:
#
#   1. evaluate its OOB MSE;
#   2. shuffle predictor j among its OOB observations;
#   3. evaluate OOB MSE again;
#   4. measure increase in error.
# ------------------------------------------------------------------------------

oob_permutation_importance <- function(
    model,
    X,
    y,
    seed = 123
) {
  
  X <- as.matrix(X)
  
  
  p <- ncol(X)
  
  
  importance_sum <- numeric(
    p
  )
  
  
  importance_count <- integer(
    p
  )
  
  
  set.seed(seed)
  
  
  for (
    tree_index in seq_len(
      model$number_trees
    )
  ) {
    
    oob_indices <- model$oob_samples[[tree_index]]
    
    
    if (
      length(oob_indices) <
      2
    ) {
      
      next
    }
    
    
    tree <- model$trees[[tree_index]]
    
    
    X_oob <- X[
      oob_indices,
      ,
      drop = FALSE
    ]
    
    
    y_oob <- y[
      oob_indices
    ]
    
    
    baseline_prediction <- predict_tree(
      tree,
      X_oob
    )
    
    
    baseline_mse <- mean(
      (
        y_oob -
          baseline_prediction
      )^2
    )
    
    
    for (
      variable in seq_len(p)
    ) {
      
      X_permuted <- X_oob
      
      
      X_permuted[
        ,
        variable
      ] <- sample(
        X_permuted[
          ,
          variable
        ]
      )
      
      
      permuted_prediction <- predict_tree(
        tree,
        X_permuted
      )
      
      
      permuted_mse <- mean(
        (
          y_oob -
            permuted_prediction
        )^2
      )
      
      
      importance_sum[
        variable
      ] <- importance_sum[
        variable
      ] +
        (
          permuted_mse -
            baseline_mse
        )
      
      
      importance_count[
        variable
      ] <- importance_count[
        variable
      ] +
        1
    }
  }
  
  
  importance_sum /
    pmax(
      importance_count,
      1
    )
}


# ------------------------------------------------------------------------------
# 47. Compute OOB Importance
# ------------------------------------------------------------------------------

permutation_importance <- oob_permutation_importance(
  bagged_model,
  X_train,
  y_train,
  seed = 123
)


permutation_importance_table <- data.frame(
  Variable =
    colnames(X_train),
  Importance =
    permutation_importance
)


permutation_importance_table <-
  permutation_importance_table[
    order(
      -permutation_importance_table$Importance
    ),
  ]


permutation_importance_table


# ------------------------------------------------------------------------------
# 48. Plot
# ------------------------------------------------------------------------------

barplot(
  permutation_importance_table$Importance,
  names.arg =
    permutation_importance_table$Variable,
  las = 2,
  ylab = "Increase in OOB MSE",
  main = "OOB Permutation Importance"
)


# ==============================================================================
# PART XXVIII
#
# PARTIAL DEPENDENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Manual Partial Dependence
# ------------------------------------------------------------------------------

partial_dependence <- function(
    model,
    X_reference,
    variable,
    grid
) {
  
  X_reference <- as.matrix(
    X_reference
  )
  
  
  pd <- numeric(
    length(grid)
  )
  
  
  for (
    i in seq_along(grid)
  ) {
    
    X_modified <- X_reference
    
    
    X_modified[
      ,
      variable
    ] <- grid[i]
    
    
    pd[i] <- mean(
      predict_bagging(
        model,
        X_modified
      )
    )
  }
  
  
  pd
}


# ------------------------------------------------------------------------------
# 50. X1 Partial Dependence
# ------------------------------------------------------------------------------

x1_grid <- seq(
  quantile(
    X_train[, 1],
    0.02
  ),
  quantile(
    X_train[, 1],
    0.98
  ),
  length.out = 60
)


pd_x1 <- partial_dependence(
  bagged_model,
  X_train,
  variable = 1,
  grid = x1_grid
)


plot(
  x1_grid,
  pd_x1,
  type = "l",
  lwd = 2,
  xlab = "X1",
  ylab = "Partial Dependence",
  main = "Bagging Partial Dependence: X1"
)


# ==============================================================================
# PART XXIX
#
# 1D ILLUSTRATION OF SMOOTHING THROUGH BAGGING
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Generate One-Dimensional Data
# ------------------------------------------------------------------------------

set.seed(999)


n_1d <- 200


x_1d <- sort(
  runif(
    n_1d,
    -3,
    3
  )
)


f_1d <- 2 *
  sin(
    1.5 *
      x_1d
  )


y_1d <- f_1d +
  rnorm(
    n_1d,
    sd = 0.8
  )


X_1d <- matrix(
  x_1d,
  ncol = 1
)


colnames(
  X_1d
) <- "X1"


# ------------------------------------------------------------------------------
# 52. Single Tree
# ------------------------------------------------------------------------------

tree_1d <- grow_regression_tree(
  X_1d,
  y_1d,
  minimum_leaf_size = 5,
  maximum_depth = 15,
  maximum_splits = 50
)


# ------------------------------------------------------------------------------
# 53. Bagged Trees
# ------------------------------------------------------------------------------

bag_1d <- fit_bagging(
  X_1d,
  y_1d,
  number_trees = 200,
  minimum_leaf_size = 5,
  maximum_depth = 15,
  maximum_splits = 50,
  seed = 999
)


# ------------------------------------------------------------------------------
# 54. Prediction Grid
# ------------------------------------------------------------------------------

grid_1d <- seq(
  -3,
  3,
  length.out = 400
)


X_grid_1d <- matrix(
  grid_1d,
  ncol = 1
)


single_prediction_1d <- predict_tree(
  tree_1d,
  X_grid_1d
)


bagged_prediction_1d <- predict_bagging(
  bag_1d,
  X_grid_1d
)


# ------------------------------------------------------------------------------
# 55. Plot
# ------------------------------------------------------------------------------

plot(
  x_1d,
  y_1d,
  pch = 19,
  cex = 0.6,
  xlab = "X",
  ylab = "Y",
  main = "Single Tree vs Bagged Trees"
)


lines(
  grid_1d,
  2 *
    sin(
      1.5 *
        grid_1d
    ),
  lwd = 3,
  lty = 2
)


lines(
  grid_1d,
  single_prediction_1d,
  lwd = 2
)


lines(
  grid_1d,
  bagged_prediction_1d,
  lwd = 3
)


legend(
  "topright",
  legend = c(
    "True Function",
    "Single Tree",
    "Bagging"
  ),
  lty = c(
    2,
    1,
    1
  ),
  lwd = c(
    3,
    2,
    3
  )
)


# ==============================================================================
# PART XXX
#
# NUMBER OF TREES AND MONTE CARLO STABILITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Compare Predictions from Different B
# ------------------------------------------------------------------------------

B_values <- c(
  1,
  5,
  10,
  25,
  50,
  100,
  200,
  300
)


B_results <- data.frame(
  Trees =
    B_values,
  Test_MSE =
    NA_real_
)


for (
  i in seq_along(
    B_values
  )
) {
  
  prediction_i <- predict_bagging(
    bagged_model,
    X_test,
    number_trees =
      B_values[i]
  )
  
  
  B_results$Test_MSE[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


B_results


plot(
  B_results$Trees,
  B_results$Test_MSE,
  type = "b",
  pch = 19,
  xlab = "Number of Trees",
  ylab = "Test MSE",
  main = "Bagging Performance vs Number of Trees"
)


# ==============================================================================
# PART XXXI
#
# BAGGING VS RANDOM FOREST
# ==============================================================================


# ------------------------------------------------------------------------------
# 57. Conceptual Comparison
#
# BAGGING:
#
#       Bootstrap observations
#       +
#       ALL p predictors available at every split
#
#
# RANDOM FOREST:
#
#       Bootstrap observations
#       +
#       only mtry < p randomly chosen predictors available at each split
#
#
# Therefore:
#
#       Bagging = Random Forest with mtry = p
#
# in the usual tree-ensemble formulation.
# ------------------------------------------------------------------------------


cat(
  "\nBagging uses all",
  p,
  "predictors at every split.\n"
)


cat(
  "A Random Forest deliberately restricts each split to a random subset.\n"
)


# ==============================================================================
# PART XXXII
#
# OPTIONAL RANDOM FOREST COMPARISON
# ==============================================================================


# If the randomForest package is available, compare:
#
#       mtry = p
#
# against
#
#       mtry < p.
#
# The first is bagging.


if (
  requireNamespace(
    "randomForest",
    quietly = TRUE
  )
) {
  
  set.seed(123)
  
  
  package_bagging <- randomForest::randomForest(
    x =
      X_train,
    y =
      y_train,
    ntree = 300,
    mtry = p,
    nodesize = 5,
    importance = TRUE
  )
  
  
  package_bagging_prediction <- predict(
    package_bagging,
    X_test
  )
  
  
  cat(
    "\nPackage Bagging Test MSE:\n"
  )
  
  
  print(
    mean(
      (
        y_test -
          package_bagging_prediction
      )^2
    )
  )
  
  
  set.seed(123)
  
  
  package_random_forest <- randomForest::randomForest(
    x =
      X_train,
    y =
      y_train,
    ntree = 300,
    mtry = max(
      1,
      floor(
        p /
          3
      )
    ),
    nodesize = 5
  )
  
  
  package_rf_prediction <- predict(
    package_random_forest,
    X_test
  )
  
  
  cat(
    "\nPackage Random Forest Test MSE:\n"
  )
  
  
  print(
    mean(
      (
        y_test -
          package_rf_prediction
      )^2
    )
  )
}


# ==============================================================================
# PART XXXIII
#
# BAGGING A STABLE LINEAR ESTIMATOR
# ==============================================================================


# ------------------------------------------------------------------------------
# 58. Linear Regression
#
# Linear least squares is much more stable than a deep tree.
#
# Bagging it usually changes much less.
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y =
    y_train,
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


linear_mse <- mean(
  (
    y_test -
      linear_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 59. Bootstrap Linear Models
# ------------------------------------------------------------------------------

set.seed(321)


B_linear <- 300


bagged_linear_prediction_sum <- numeric(
  n_test
)


for (
  b in seq_len(
    B_linear
  )
) {
  
  indices_b <- sample(
    seq_len(n_train),
    size = n_train,
    replace = TRUE
  )
  
  
  data_b <- data.frame(
    y =
      y_train[
        indices_b
      ],
    X_train[
      indices_b,
      ,
      drop = FALSE
    ]
  )
  
  
  fit_b <- lm(
    y ~ .,
    data =
      data_b
  )
  
  
  prediction_b <- predict(
    fit_b,
    newdata =
      test_data
  )
  
  
  bagged_linear_prediction_sum <-
    bagged_linear_prediction_sum +
    prediction_b
}


bagged_linear_prediction <-
  bagged_linear_prediction_sum /
  B_linear


bagged_linear_mse <- mean(
  (
    y_test -
      bagged_linear_prediction
  )^2
)


data.frame(
  Method = c(
    "Linear Regression",
    "Bagged Linear Regression",
    "Single Tree",
    "Bagged Trees"
  ),
  Test_MSE = c(
    linear_mse,
    bagged_linear_mse,
    single_tree_metrics["MSE"],
    bagged_test_metrics["MSE"]
  )
)


# The important point is not that bagging can only be used with trees.
#
# It can be applied to many estimators.
#
# But the gains are generally greatest for unstable/high-variance procedures.


# ==============================================================================
# PART XXXIV
#
# CLASSIFICATION BAGGING: SMALL MANUAL DEMO
# ==============================================================================


# ------------------------------------------------------------------------------
# 60. Classification Setup
#
# Rather than rebuild the full classification CART algorithm here, this section
# demonstrates the aggregation rule conceptually.
#
# Suppose five bootstrap classifiers predict:
#
#       1, 1, 0, 1, 0
#
# Bagging uses majority vote.
# ------------------------------------------------------------------------------

classification_predictions <- c(
  1,
  1,
  0,
  1,
  0
)


bagged_class_prediction <- as.integer(
  mean(
    classification_predictions
  ) >=
    0.5
)


bagged_class_prediction


# ------------------------------------------------------------------------------
# 61. Probability Averaging
#
# If trees return class probabilities, they can also be averaged.
# ------------------------------------------------------------------------------

tree_probabilities <- c(
  0.80,
  0.65,
  0.40,
  0.72,
  0.35
)


bagged_probability <- mean(
  tree_probabilities
)


bagged_probability


as.integer(
  bagged_probability >=
    0.5
)


# ==============================================================================
# PART XXXV
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nBagging Summary\n"
)


cat(
  "---------------\n"
)


cat(
  "Training observations:",
  n_train,
  "\n"
)


cat(
  "Test observations:",
  n_test,
  "\n"
)


cat(
  "Predictors:",
  p,
  "\n"
)


cat(
  "Bootstrap trees:",
  bagged_model$number_trees,
  "\n"
)


cat(
  "Approximate unique fraction per bootstrap sample:",
  round(
    probability_selected,
    4
  ),
  "\n"
)


cat(
  "Approximate OOB fraction per tree:",
  round(
    probability_not_selected,
    4
  ),
  "\n"
)


cat(
  "Single-tree test MSE:",
  round(
    single_tree_metrics["MSE"],
    5
  ),
  "\n"
)


cat(
  "Bagging test MSE:",
  round(
    bagged_test_metrics["MSE"],
    5
  ),
  "\n"
)


cat(
  "Bagging OOB MSE:",
  round(
    oob_metrics["MSE"],
    5
  ),
  "\n"
)


cat(
  "Average pairwise tree correlation:",
  round(
    average_tree_correlation,
    4
  ),
  "\n"
)


cat(
  "Most important variable by OOB permutation:",
  permutation_importance_table$Variable[1],
  "\n"
)