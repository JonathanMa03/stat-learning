# ==============================================================================
# Random Forests
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Decision trees
#   - Bootstrap aggregation
#   - Random feature selection
#   - Tree decorrelation
#   - Ensemble averaging
#   - Out-of-bag prediction
#   - Out-of-bag error
#   - Permutation importance
#   - Split-improvement importance
#   - Bias-variance tradeoff
#   - Number of trees
#   - mtry
#   - Bagging vs random forests
#
#
# Regression Random Forest:
#
#   For b = 1,...,B:
#
#       1. Draw bootstrap sample from training observations.
#
#       2. Grow a deep regression tree.
#
#       3. At each split, randomly select mtry predictors.
#
#       4. Choose the best split only among those predictors.
#
#
# Prediction:
#
#       f_hat_RF(x)
#
#       =
#
#       (1/B) sum_b T_b(x)
#
#
# Random forests differ from ordinary bagging because bagging considers ALL
# predictors at every split, whereas random forests only consider a random
# subset.
# ==============================================================================


# ==============================================================================
# PART I
#
# SIMULATE NONLINEAR REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Data
# ------------------------------------------------------------------------------

set.seed(123)


n <- 600

p <- 10


X <- matrix(
  rnorm(
    n * p
  ),
  nrow = n,
  ncol = p
)


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 2. True Regression Function
#
# Only the first five variables matter.
#
# Includes:
#
#   - nonlinear terms
#   - threshold effects
#   - interactions
# ------------------------------------------------------------------------------

true_function <- function(
    X
) {
  
  3 *
    sin(
      X[
        ,
        1
      ]
    ) +
    2 *
    (
      X[
        ,
        2
      ] >
        0
    ) +
    1.5 *
    X[
      ,
      3
    ]^2 -
    2 *
    X[
      ,
      4
    ] *
    X[
      ,
      5
    ]
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
# 3. Data Frame
# ------------------------------------------------------------------------------

data <- data.frame(
  y = y,
  X
)


head(
  data
)


# ==============================================================================
# PART II
#
# TRAIN / TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Split
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


# ==============================================================================
# PART III
#
# REGRESSION TREE UTILITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Node RSS
# ------------------------------------------------------------------------------

node_rss <- function(
    y
) {
  
  if (
    length(y) ==
    0
  ) {
    
    return(
      0
    )
  }
  
  
  sum(
    (
      y -
        mean(y)
    )^2
  )
}


# ------------------------------------------------------------------------------
# 6. Candidate Split Points
#
# For computational convenience, if a predictor has many unique values,
# evaluate only a grid of quantile-based candidate splits.
# ------------------------------------------------------------------------------

candidate_splits <- function(
    x,
    maximum_splits = 30
) {
  
  unique_values <- sort(
    unique(
      x
    )
  )
  
  
  if (
    length(
      unique_values
    ) <=
    1
  ) {
    
    return(
      numeric(0)
    )
  }
  
  
  midpoints <- (
    unique_values[
      -length(
        unique_values
      )
    ] +
      unique_values[
        -1
      ]
  ) /
    2
  
  
  if (
    length(
      midpoints
    ) <=
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
      maximum_splits +
      2
  )[
    -c(
      1,
      maximum_splits +
        2
    )
  ]
  
  
  split_values <- as.numeric(
    quantile(
      x,
      probs =
        probabilities,
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
# PART IV
#
# FIND BEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Best Split Among a GIVEN SET of Predictors
#
# This is where random forests differ from ordinary trees.
# ------------------------------------------------------------------------------

find_best_split <- function(
    X,
    y,
    predictor_indices,
    minimum_leaf_size = 5,
    maximum_splits = 30
) {
  
  parent_rss <- node_rss(
    y
  )
  
  
  best_improvement <- -Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  best_left <- NULL
  
  best_right <- NULL
  
  
  for (
    variable in predictor_indices
  ) {
    
    x <- X[
      ,
      variable
    ]
    
    
    splits <- candidate_splits(
      x,
      maximum_splits =
        maximum_splits
    )
    
    
    if (
      length(
        splits
      ) ==
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
        length(
          left
        ) <
        minimum_leaf_size ||
        length(
          right
        ) <
        minimum_leaf_size
      ) {
        
        next
      }
      
      
      split_rss <- node_rss(
        y[
          left
        ]
      ) +
        node_rss(
          y[
            right
          ]
        )
      
      
      improvement <- parent_rss -
        split_rss
      
      
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
    is.na(
      best_variable
    )
  ) {
    
    return(
      NULL
    )
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
# PART V
#
# GROW A RANDOMIZED REGRESSION TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Grow Tree
#
# At every node:
#
#       sample mtry predictors
#
# then search for the best split only among those predictors.
# ------------------------------------------------------------------------------

grow_random_tree <- function(
    X,
    y,
    mtry,
    minimum_leaf_size = 5,
    maximum_depth = 20,
    maximum_splits = 30,
    current_depth = 0
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  node_prediction <- mean(
    y
  )
  
  
  # --------------------------------------------------------------------------
  # Stopping Conditions
  # --------------------------------------------------------------------------
  
  if (
    n <
    2 *
    minimum_leaf_size ||
    current_depth >=
    maximum_depth ||
    length(
      unique(
        y
      )
    ) ==
    1
  ) {
    
    return(
      list(
        terminal =
          TRUE,
        prediction =
          node_prediction,
        n =
          n
      )
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Randomly Select Candidate Predictors
  # --------------------------------------------------------------------------
  
  mtry <- min(
    mtry,
    p
  )
  
  
  candidate_variables <- sample(
    seq_len(p),
    size =
      mtry,
    replace = FALSE
  )
  
  
  # --------------------------------------------------------------------------
  # Find Best Split Among Random Candidate Variables
  # --------------------------------------------------------------------------
  
  best_split <- find_best_split(
    X,
    y,
    predictor_indices =
      candidate_variables,
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_splits =
      maximum_splits
  )
  
  
  if (
    is.null(
      best_split
    )
  ) {
    
    return(
      list(
        terminal =
          TRUE,
        prediction =
          node_prediction,
        n =
          n
      )
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Recursive Children
  # --------------------------------------------------------------------------
  
  left_tree <- grow_random_tree(
    X[
      best_split$left,
      ,
      drop = FALSE
    ],
    y[
      best_split$left
    ],
    mtry =
      mtry,
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_depth =
      maximum_depth,
    maximum_splits =
      maximum_splits,
    current_depth =
      current_depth +
      1
  )
  
  
  right_tree <- grow_random_tree(
    X[
      best_split$right,
      ,
      drop = FALSE
    ],
    y[
      best_split$right
    ],
    mtry =
      mtry,
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_depth =
      maximum_depth,
    maximum_splits =
      maximum_splits,
    current_depth =
      current_depth +
      1
  )
  
  
  list(
    terminal =
      FALSE,
    prediction =
      node_prediction,
    variable =
      best_split$variable,
    split =
      best_split$split,
    improvement =
      best_split$improvement,
    candidate_variables =
      candidate_variables,
    left =
      left_tree,
    right =
      right_tree,
    n =
      n
  )
}


# ==============================================================================
# PART VI
#
# TREE PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Predict One Observation
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
    
    predict_tree_one(
      tree$left,
      x
    )
    
  } else {
    
    predict_tree_one(
      tree$right,
      x
    )
  }
}


# ------------------------------------------------------------------------------
# 10. Predict Matrix
# ------------------------------------------------------------------------------

predict_tree <- function(
    tree,
    X
) {
  
  X <- as.matrix(
    X
  )
  
  
  predictions <- numeric(
    nrow(
      X
    )
  )
  
  
  for (
    i in seq_len(
      nrow(
        X
      )
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
# PART VII
#
# FIT ONE RANDOMIZED TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. One Tree
# ------------------------------------------------------------------------------

set.seed(123)


one_tree <- grow_random_tree(
  X_train,
  y_train,
  mtry = 3,
  minimum_leaf_size = 5,
  maximum_depth = 12
)


# ------------------------------------------------------------------------------
# 12. Training and Test Prediction
# ------------------------------------------------------------------------------

one_tree_train_prediction <- predict_tree(
  one_tree,
  X_train
)


one_tree_test_prediction <- predict_tree(
  one_tree,
  X_test
)


# ------------------------------------------------------------------------------
# 13. MSE
# ------------------------------------------------------------------------------

mean(
  (
    y_train -
      one_tree_train_prediction
  )^2
)


mean(
  (
    y_test -
      one_tree_test_prediction
  )^2
)


# ==============================================================================
# PART VIII
#
# BOOTSTRAP SAMPLING
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Demonstrate Bootstrap Sample
# ------------------------------------------------------------------------------

set.seed(123)


bootstrap_indices <- sample(
  seq_len(
    nrow(
      X_train
    )
  ),
  size =
    nrow(
      X_train
    ),
  replace = TRUE
)


# ------------------------------------------------------------------------------
# 15. Unique Observations in Bootstrap Sample
# ------------------------------------------------------------------------------

length(
  unique(
    bootstrap_indices
  )
) /
  nrow(
    X_train
  )


# This should often be around 0.632 for a large sample.


# ------------------------------------------------------------------------------
# 16. Out-of-Bag Observations
# ------------------------------------------------------------------------------

oob_indices <- setdiff(
  seq_len(
    nrow(
      X_train
    )
  ),
  unique(
    bootstrap_indices
  )
)


length(
  oob_indices
) /
  nrow(
    X_train
  )


# Roughly 36.8% of training observations are OOB for a particular tree.


# ==============================================================================
# PART IX
#
# RANDOM FOREST FITTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Manual Random Forest
# ------------------------------------------------------------------------------

fit_random_forest <- function(
    X,
    y,
    number_trees = 200,
    mtry = NULL,
    minimum_leaf_size = 5,
    maximum_depth = 20,
    maximum_splits = 30,
    seed = 123,
    keep_oob_predictions = TRUE,
    verbose = FALSE
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
  
  
  if (
    is.null(
      mtry
    )
  ) {
    
    mtry <- max(
      1,
      floor(
        p /
          3
      )
    )
  }
  
  
  set.seed(
    seed
  )
  
  
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
  
  
  if (
    keep_oob_predictions
  ) {
    
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
    
  } else {
    
    oob_prediction_sum <- NULL
    
    oob_prediction_count <- NULL
    
    oob_mse_path <- NULL
  }
  
  
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
    
    
    unique_bootstrap <- unique(
      bootstrap_indices
    )
    
    
    oob_indices <- setdiff(
      seq_len(n),
      unique_bootstrap
    )
    
    
    bootstrap_samples[[tree_index]] <-
      bootstrap_indices
    
    
    oob_samples[[tree_index]] <-
      oob_indices
    
    
    # ------------------------------------------------------------------------
    # Grow Randomized Tree
    # ------------------------------------------------------------------------
    
    tree <- grow_random_tree(
      X[
        bootstrap_indices,
        ,
        drop = FALSE
      ],
      y[
        bootstrap_indices
      ],
      mtry =
        mtry,
      minimum_leaf_size =
        minimum_leaf_size,
      maximum_depth =
        maximum_depth,
      maximum_splits =
        maximum_splits,
      current_depth = 0
    )
    
    
    trees[[tree_index]] <-
      tree
    
    
    # ------------------------------------------------------------------------
    # OOB Prediction
    # ------------------------------------------------------------------------
    
    if (
      keep_oob_predictions &&
      length(
        oob_indices
      ) >
      0
    ) {
      
      predictions_oob <- predict_tree(
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
        predictions_oob
      
      
      oob_prediction_count[
        oob_indices
      ] <- oob_prediction_count[
        oob_indices
      ] +
        1
      
      
      available <- oob_prediction_count >
        0
      
      
      current_oob_predictions <-
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
            current_oob_predictions
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
  
  
  if (
    keep_oob_predictions
  ) {
    
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
    
  } else {
    
    oob_predictions <- NULL
  }
  
  
  list(
    trees =
      trees,
    bootstrap_samples =
      bootstrap_samples,
    oob_samples =
      oob_samples,
    number_trees =
      number_trees,
    mtry =
      mtry,
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
      oob_mse_path,
    training_X =
      X,
    training_y =
      y
  )
}


# ==============================================================================
# PART X
#
# RANDOM FOREST PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Predict Forest
# ------------------------------------------------------------------------------

predict_random_forest <- function(
    forest,
    X,
    number_trees = NULL
) {
  
  X <- as.matrix(
    X
  )
  
  
  if (
    is.null(
      number_trees
    )
  ) {
    
    number_trees <- forest$number_trees
  }
  
  
  number_trees <- min(
    number_trees,
    forest$number_trees
  )
  
  
  prediction_sum <- numeric(
    nrow(
      X
    )
  )
  
  
  for (
    tree_index in seq_len(
      number_trees
    )
  ) {
    
    prediction_sum <- prediction_sum +
      predict_tree(
        forest$trees[[tree_index]],
        X
      )
  }
  
  
  prediction_sum /
    number_trees
}


# ==============================================================================
# PART XI
#
# FIT RANDOM FOREST
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Fit
#
# For regression, p/3 is a traditional default-style choice for mtry.
# ------------------------------------------------------------------------------

forest <- fit_random_forest(
  X_train,
  y_train,
  number_trees = 300,
  mtry = 3,
  minimum_leaf_size = 5,
  maximum_depth = 20,
  maximum_splits = 30,
  seed = 123,
  verbose = TRUE
)


# ==============================================================================
# PART XII
#
# TRAIN / TEST PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Predictions
# ------------------------------------------------------------------------------

forest_train_prediction <- predict_random_forest(
  forest,
  X_train
)


forest_test_prediction <- predict_random_forest(
  forest,
  X_test
)


# ------------------------------------------------------------------------------
# 21. Metrics
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
    MSE =
      mse,
    RMSE =
      rmse,
    MAE =
      mae,
    R2 =
      r_squared
  )
}


regression_metrics(
  y_train,
  forest_train_prediction
)


regression_metrics(
  y_test,
  forest_test_prediction
)


# ==============================================================================
# PART XIII
#
# OUT-OF-BAG ERROR
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. OOB Predictions
# ------------------------------------------------------------------------------

available_oob <- !is.na(
  forest$oob_predictions
)


oob_metrics <- regression_metrics(
  y_train[
    available_oob
  ],
  forest$oob_predictions[
    available_oob
  ]
)


oob_metrics


# ------------------------------------------------------------------------------
# 23. Compare OOB and Test MSE
# ------------------------------------------------------------------------------

data.frame(
  Dataset = c(
    "OOB",
    "Test"
  ),
  MSE = c(
    oob_metrics[
      "MSE"
    ],
    regression_metrics(
      y_test,
      forest_test_prediction
    )[
      "MSE"
    ]
  )
)


# OOB error acts as an internal validation estimate without requiring a separate
# validation split.


# ==============================================================================
# PART XIV
#
# OOB ERROR VS NUMBER OF TREES
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Plot
# ------------------------------------------------------------------------------

plot(
  seq_len(
    forest$number_trees
  ),
  forest$oob_mse_path,
  type = "l",
  xlab = "Number of Trees",
  ylab = "OOB MSE",
  main = "Random Forest OOB Error"
)


# Random forests generally do not overfit merely because more trees are added.
# Instead, the ensemble error usually stabilizes.


# ==============================================================================
# PART XV
#
# TEST ERROR VS NUMBER OF TREES
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Cumulative Prediction Path
# ------------------------------------------------------------------------------

tree_grid <- unique(
  round(
    seq(
      1,
      forest$number_trees,
      length.out = 40
    )
  )
)


test_mse_path <- numeric(
  length(
    tree_grid
  )
)


for (
  i in seq_along(
    tree_grid
  )
) {
  
  prediction_i <- predict_random_forest(
    forest,
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


plot(
  tree_grid,
  test_mse_path,
  type = "b",
  pch = 19,
  xlab = "Number of Trees",
  ylab = "Test MSE",
  main = "Test Error vs Number of Trees"
)


# ==============================================================================
# PART XVI
#
# BAGGING AS A SPECIAL CASE
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Bagging
#
# Bagging is equivalent to:
#
#       mtry = p
#
# Every predictor is available at every split.
# ------------------------------------------------------------------------------

bagged_forest <- fit_random_forest(
  X_train,
  y_train,
  number_trees = 300,
  mtry = p,
  minimum_leaf_size = 5,
  maximum_depth = 20,
  maximum_splits = 30,
  seed = 123
)


bagged_test_prediction <- predict_random_forest(
  bagged_forest,
  X_test
)


# ------------------------------------------------------------------------------
# 27. Compare
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Single Randomized Tree",
    "Bagging",
    "Random Forest"
  ),
  Test_MSE = c(
    mean(
      (
        y_test -
          one_tree_test_prediction
      )^2
    ),
    mean(
      (
        y_test -
          bagged_test_prediction
      )^2
    ),
    mean(
      (
        y_test -
          forest_test_prediction
      )^2
    )
  )
)


# ==============================================================================
# PART XVII
#
# WHY RANDOM FEATURE SELECTION HELPS
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Predictions from Individual Trees
# ------------------------------------------------------------------------------

number_tree_predictions <- 50


individual_tree_predictions_bagging <- matrix(
  NA_real_,
  nrow =
    nrow(
      X_test
    ),
  ncol =
    number_tree_predictions
)


individual_tree_predictions_rf <- matrix(
  NA_real_,
  nrow =
    nrow(
      X_test
    ),
  ncol =
    number_tree_predictions
)


for (
  tree_index in seq_len(
    number_tree_predictions
  )
) {
  
  individual_tree_predictions_bagging[
    ,
    tree_index
  ] <- predict_tree(
    bagged_forest$trees[[tree_index]],
    X_test
  )
  
  
  individual_tree_predictions_rf[
    ,
    tree_index
  ] <- predict_tree(
    forest$trees[[tree_index]],
    X_test
  )
}


# ------------------------------------------------------------------------------
# 29. Average Pairwise Tree Correlation
# ------------------------------------------------------------------------------

average_tree_correlation <- function(
    prediction_matrix
) {
  
  correlation_matrix <- cor(
    prediction_matrix
  )
  
  
  mean(
    correlation_matrix[
      upper.tri(
        correlation_matrix
      )
    ]
  )
}


bagging_tree_correlation <- average_tree_correlation(
  individual_tree_predictions_bagging
)


rf_tree_correlation <- average_tree_correlation(
  individual_tree_predictions_rf
)


data.frame(
  Method = c(
    "Bagging",
    "Random Forest"
  ),
  Average_Tree_Correlation = c(
    bagging_tree_correlation,
    rf_tree_correlation
  )
)


# Random feature selection is intended to reduce correlation between trees.


# ==============================================================================
# PART XVIII
#
# EFFECT OF MTRY
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. mtry Grid
# ------------------------------------------------------------------------------

mtry_values <- c(
  1,
  2,
  3,
  5,
  7,
  p
)


mtry_results <- data.frame(
  mtry =
    mtry_values,
  OOB_MSE =
    NA_real_,
  Test_MSE =
    NA_real_
)


mtry_models <- vector(
  "list",
  length(
    mtry_values
  )
)


for (
  i in seq_along(
    mtry_values
  )
) {
  
  fit_i <- fit_random_forest(
    X_train,
    y_train,
    number_trees = 150,
    mtry =
      mtry_values[i],
    minimum_leaf_size = 5,
    maximum_depth = 18,
    maximum_splits = 25,
    seed =
      100 +
      i
  )
  
  
  mtry_models[[i]] <-
    fit_i
  
  
  available <- !is.na(
    fit_i$oob_predictions
  )
  
  
  mtry_results$OOB_MSE[i] <- mean(
    (
      y_train[
        available
      ] -
        fit_i$oob_predictions[
          available
        ]
    )^2
  )
  
  
  test_prediction_i <- predict_random_forest(
    fit_i,
    X_test
  )
  
  
  mtry_results$Test_MSE[i] <- mean(
    (
      y_test -
        test_prediction_i
    )^2
  )
}


mtry_results


# ------------------------------------------------------------------------------
# 31. Plot
# ------------------------------------------------------------------------------

plot(
  mtry_results$mtry,
  mtry_results$OOB_MSE,
  type = "b",
  pch = 19,
  xlab = "mtry",
  ylab = "OOB MSE",
  main = "Random Forest Performance vs mtry"
)


# ==============================================================================
# PART XIX
#
# SPLIT-IMPROVEMENT VARIABLE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Collect RSS Improvements from One Tree
# ------------------------------------------------------------------------------

collect_split_importance <- function(
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
        NULL
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
  }
  
  
  recursive_collect(
    tree
  )
  
  
  importance
}


# ------------------------------------------------------------------------------
# 33. Forest Split Importance
# ------------------------------------------------------------------------------

forest_split_importance <- function(
    forest,
    p
) {
  
  total_importance <- numeric(
    p
  )
  
  
  for (
    tree in forest$trees
  ) {
    
    total_importance <- total_importance +
      collect_split_importance(
        tree,
        p
      )
  }
  
  
  total_importance
}


split_importance <- forest_split_importance(
  forest,
  p
)


split_importance <- split_importance /
  sum(
    split_importance
  )


split_importance_table <- data.frame(
  Variable =
    colnames(
      X_train
    ),
  Importance =
    split_importance
)


split_importance_table <- split_importance_table[
  order(
    -split_importance_table$Importance
  ),
]


split_importance_table


# ------------------------------------------------------------------------------
# 34. Plot
# ------------------------------------------------------------------------------

barplot(
  split_importance_table$Importance,
  names.arg =
    split_importance_table$Variable,
  las = 2,
  ylab = "Relative Split Improvement",
  main = "Random Forest Split Importance"
)


# ==============================================================================
# PART XX
#
# PERMUTATION IMPORTANCE USING TEST SET
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Generic Permutation Importance
#
# This version uses a holdout/test dataset for clarity.
# Later we implement OOB permutation importance.
# ------------------------------------------------------------------------------

permutation_importance <- function(
    forest,
    X,
    y,
    repetitions = 5,
    seed = 123
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  baseline_prediction <- predict_random_forest(
    forest,
    X
  )
  
  
  baseline_mse <- mean(
    (
      y -
        baseline_prediction
    )^2
  )
  
  
  importance <- numeric(
    p
  )
  
  
  set.seed(
    seed
  )
  
  
  for (
    variable in seq_len(
      p
    )
  ) {
    
    increases <- numeric(
      repetitions
    )
    
    
    for (
      repetition in seq_len(
        repetitions
      )
    ) {
      
      X_permuted <- X
      
      
      X_permuted[
        ,
        variable
      ] <- sample(
        X_permuted[
          ,
          variable
        ]
      )
      
      
      prediction_permuted <- predict_random_forest(
        forest,
        X_permuted
      )
      
      
      permuted_mse <- mean(
        (
          y -
            prediction_permuted
        )^2
      )
      
      
      increases[
        repetition
      ] <- permuted_mse -
        baseline_mse
    }
    
    
    importance[
      variable
    ] <- mean(
      increases
    )
  }
  
  
  list(
    baseline_mse =
      baseline_mse,
    importance =
      importance
  )
}


# ------------------------------------------------------------------------------
# 36. Compute
# ------------------------------------------------------------------------------

test_permutation_importance <- permutation_importance(
  forest,
  X_test,
  y_test,
  repetitions = 10,
  seed = 123
)


permutation_table <- data.frame(
  Variable =
    colnames(
      X_test
    ),
  Importance =
    test_permutation_importance$importance
)


permutation_table <- permutation_table[
  order(
    -permutation_table$Importance
  ),
]


permutation_table


# ------------------------------------------------------------------------------
# 37. Plot
# ------------------------------------------------------------------------------

barplot(
  permutation_table$Importance,
  names.arg =
    permutation_table$Variable,
  las = 2,
  ylab = "Increase in MSE",
  main = "Permutation Variable Importance"
)


# ==============================================================================
# PART XXI
#
# TRUE VARIABLE IMPORTANCE STRUCTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. True Signal Variables
# ------------------------------------------------------------------------------

true_signal_variables <- c(
  "X1",
  "X2",
  "X3",
  "X4",
  "X5"
)


true_signal_variables


# Variables X6:X10 are pure noise.


# ==============================================================================
# PART XXII
#
# OOB PERMUTATION IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. OOB Permutation Importance
#
# For each tree:
#
#   1. Predict its OOB observations.
#   2. Permute variable j within those OOB observations.
#   3. Predict again.
#   4. Measure increase in OOB error.
#
# Average across trees.
# ------------------------------------------------------------------------------

oob_permutation_importance <- function(
    forest,
    X,
    y,
    seed = 123
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  importance_sum <- numeric(
    p
  )
  
  
  importance_count <- integer(
    p
  )
  
  
  set.seed(
    seed
  )
  
  
  for (
    tree_index in seq_len(
      forest$number_trees
    )
  ) {
    
    oob_indices <- forest$oob_samples[[tree_index]]
    
    
    if (
      length(
        oob_indices
      ) <
      2
    ) {
      
      next
    }
    
    
    tree <- forest$trees[[tree_index]]
    
    
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
      variable in seq_len(
        p
      )
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
# 40. Compute
# ------------------------------------------------------------------------------

oob_importance <- oob_permutation_importance(
  forest,
  X_train,
  y_train,
  seed = 456
)


oob_importance_table <- data.frame(
  Variable =
    colnames(
      X_train
    ),
  Importance =
    oob_importance
)


oob_importance_table <- oob_importance_table[
  order(
    -oob_importance_table$Importance
  ),
]


oob_importance_table


# ==============================================================================
# PART XXIII
#
# PARTIAL DEPENDENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Manual Partial Dependence
#
# For feature j:
#
#       PD_j(z)
#
#       =
#
#       (1/n) sum_i f_hat(z, x_i,-j)
#
# ------------------------------------------------------------------------------

partial_dependence <- function(
    forest,
    X_reference,
    variable,
    grid
) {
  
  X_reference <- as.matrix(
    X_reference
  )
  
  
  pd <- numeric(
    length(
      grid
    )
  )
  
  
  for (
    i in seq_along(
      grid
    )
  ) {
    
    X_modified <- X_reference
    
    
    X_modified[
      ,
      variable
    ] <- grid[i]
    
    
    predictions <- predict_random_forest(
      forest,
      X_modified
    )
    
    
    pd[i] <- mean(
      predictions
    )
  }
  
  
  pd
}


# ------------------------------------------------------------------------------
# 42. X1 Partial Dependence
# ------------------------------------------------------------------------------

x1_grid <- seq(
  quantile(
    X_train[
      ,
      1
    ],
    0.02
  ),
  quantile(
    X_train[
      ,
      1
    ],
    0.98
  ),
  length.out = 60
)


pd_x1 <- partial_dependence(
  forest,
  X_train,
  variable = 1,
  grid =
    x1_grid
)


plot(
  x1_grid,
  pd_x1,
  type = "l",
  lwd = 2,
  xlab = "X1",
  ylab = "Partial Dependence",
  main = "Random Forest Partial Dependence: X1"
)


# ------------------------------------------------------------------------------
# 43. X2 Partial Dependence
# ------------------------------------------------------------------------------

x2_grid <- seq(
  quantile(
    X_train[
      ,
      2
    ],
    0.02
  ),
  quantile(
    X_train[
      ,
      2
    ],
    0.98
  ),
  length.out = 60
)


pd_x2 <- partial_dependence(
  forest,
  X_train,
  variable = 2,
  grid =
    x2_grid
)


plot(
  x2_grid,
  pd_x2,
  type = "l",
  lwd = 2,
  xlab = "X2",
  ylab = "Partial Dependence",
  main = "Random Forest Partial Dependence: X2"
)


# ==============================================================================
# PART XXIV
#
# TWO-WAY PARTIAL DEPENDENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Interaction Between X4 and X5
# ------------------------------------------------------------------------------

x4_grid <- seq(
  quantile(
    X_train[
      ,
      4
    ],
    0.05
  ),
  quantile(
    X_train[
      ,
      4
    ],
    0.95
  ),
  length.out = 25
)


x5_grid <- seq(
  quantile(
    X_train[
      ,
      5
    ],
    0.05
  ),
  quantile(
    X_train[
      ,
      5
    ],
    0.95
  ),
  length.out = 25
)


pd_surface <- matrix(
  NA_real_,
  nrow =
    length(
      x4_grid
    ),
  ncol =
    length(
      x5_grid
    )
)


for (
  i in seq_along(
    x4_grid
  )
) {
  
  for (
    j in seq_along(
      x5_grid
    )
  ) {
    
    X_modified <- X_train
    
    
    X_modified[
      ,
      4
    ] <- x4_grid[i]
    
    
    X_modified[
      ,
      5
    ] <- x5_grid[j]
    
    
    pd_surface[
      i,
      j
    ] <- mean(
      predict_random_forest(
        forest,
        X_modified
      )
    )
  }
}


persp(
  x4_grid,
  x5_grid,
  pd_surface,
  xlab = "X4",
  ylab = "X5",
  zlab = "Partial Dependence",
  main = "Two-Way Partial Dependence: X4 and X5"
)


# ==============================================================================
# PART XXV
#
# RANDOM FOREST VS LINEAR MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Linear Regression Baseline
# ------------------------------------------------------------------------------

linear_model <- lm(
  y ~ .,
  data = data.frame(
    y =
      y_train,
    X_train
  )
)


linear_test_prediction <- predict(
  linear_model,
  newdata = data.frame(
    X_test
  )
)


# ------------------------------------------------------------------------------
# 46. Compare
# ------------------------------------------------------------------------------

comparison_table <- data.frame(
  Method = c(
    "Linear Regression",
    "Single Tree",
    "Bagging",
    "Random Forest"
  ),
  Test_MSE = c(
    mean(
      (
        y_test -
          linear_test_prediction
      )^2
    ),
    mean(
      (
        y_test -
          one_tree_test_prediction
      )^2
    ),
    mean(
      (
        y_test -
          bagged_test_prediction
      )^2
    ),
    mean(
      (
        y_test -
          forest_test_prediction
      )^2
    )
  )
)


comparison_table


# ==============================================================================
# PART XXVI
#
# FOREST VARIANCE REDUCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Predictions from Growing Ensemble
# ------------------------------------------------------------------------------

example_observation <- X_test[
  1,
  ,
  drop = FALSE
]


cumulative_prediction <- numeric(
  forest$number_trees
)


running_sum <- 0


for (
  tree_index in seq_len(
    forest$number_trees
  )
) {
  
  tree_prediction <- predict_tree(
    forest$trees[[tree_index]],
    example_observation
  )
  
  
  running_sum <- running_sum +
    tree_prediction
  
  
  cumulative_prediction[
    tree_index
  ] <- running_sum /
    tree_index
}


plot(
  cumulative_prediction,
  type = "l",
  xlab = "Number of Trees",
  ylab = "Ensemble Prediction",
  main = "Stabilization of a Random Forest Prediction"
)


abline(
  h =
    forest_test_prediction[1],
  lty = 2
)


# ==============================================================================
# PART XXVII
#
# RANDOMNESS FROM BOOTSTRAP VS FEATURE SUBSAMPLING
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. A Deterministic-Like Forest without Feature Randomness
#
# mtry = p gives bagging.
# ------------------------------------------------------------------------------

# Already fitted as bagged_forest.


# ------------------------------------------------------------------------------
# 49. More Aggressive Feature Randomness
# ------------------------------------------------------------------------------

very_random_forest <- fit_random_forest(
  X_train,
  y_train,
  number_trees = 200,
  mtry = 1,
  minimum_leaf_size = 5,
  maximum_depth = 20,
  maximum_splits = 30,
  seed = 321
)


very_random_prediction <- predict_random_forest(
  very_random_forest,
  X_test
)


data.frame(
  mtry = c(
    1,
    3,
    p
  ),
  Method = c(
    "Strong Feature Randomization",
    "Random Forest",
    "Bagging"
  ),
  Test_MSE = c(
    mean(
      (
        y_test -
          very_random_prediction
      )^2
    ),
    mean(
      (
        y_test -
          forest_test_prediction
      )^2
    ),
    mean(
      (
        y_test -
          bagged_test_prediction
      )^2
    )
  )
)


# Too much decorrelation can weaken individual trees.
#
# Random forests balance:
#
#       tree strength
#
# against
#
#       tree correlation.


# ==============================================================================
# PART XXVIII
#
# CORRELATED PREDICTORS
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Create Strongly Correlated Copies
# ------------------------------------------------------------------------------

set.seed(222)


X_correlated <- cbind(
  X_train,
  X1_copy =
    X_train[
      ,
      1
    ] +
    rnorm(
      nrow(
        X_train
      ),
      sd = 0.05
    ),
  X3_copy =
    X_train[
      ,
      3
    ] +
    rnorm(
      nrow(
        X_train
      ),
      sd = 0.05
    )
)


X_test_correlated <- cbind(
  X_test,
  X1_copy =
    X_test[
      ,
      1
    ] +
    rnorm(
      nrow(
        X_test
      ),
      sd = 0.05
    ),
  X3_copy =
    X_test[
      ,
      3
    ] +
    rnorm(
      nrow(
        X_test
      ),
      sd = 0.05
    )
)


# ------------------------------------------------------------------------------
# 51. Forest
# ------------------------------------------------------------------------------

forest_correlated <- fit_random_forest(
  X_correlated,
  y_train,
  number_trees = 200,
  mtry = 4,
  minimum_leaf_size = 5,
  maximum_depth = 18,
  seed = 123
)


# ------------------------------------------------------------------------------
# 52. Importance
# ------------------------------------------------------------------------------

importance_correlated <- permutation_importance(
  forest_correlated,
  X_test_correlated,
  y_test,
  repetitions = 5,
  seed = 123
)


data.frame(
  Variable =
    colnames(
      X_correlated
    ),
  Importance =
    importance_correlated$importance
)[
  order(
    -importance_correlated$importance
  ),
]


# Correlated predictors can share / dilute permutation importance.


# ==============================================================================
# PART XXIX
#
# OPTIONAL CLASSIFICATION RANDOM FOREST CONCEPT
# ==============================================================================


# For classification, almost everything stays the same except:
#
#   - terminal nodes predict a class / class probability;
#   - splits use classification impurity such as:
#
#         Gini = 1 - sum_k p_k^2
#
#   - the forest predicts by majority vote or averaging class probabilities.
#
#
# Typical classification default:
#
#       mtry approximately sqrt(p)
#
#
# Typical regression default:
#
#       mtry approximately p / 3
#
# These are heuristics, not mathematical requirements.


# ==============================================================================
# PART XXX
#
# OPTIONAL PACKAGE VERIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. randomForest
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "randomForest",
    quietly = TRUE
  )
) {
  
  set.seed(
    123
  )
  
  
  package_forest <- randomForest::randomForest(
    x =
      X_train,
    y =
      y_train,
    ntree = 300,
    mtry = 3,
    nodesize = 5,
    importance = TRUE
  )
  
  
  package_prediction <- predict(
    package_forest,
    X_test
  )
  
  
  cat(
    "\nrandomForest package test MSE:\n"
  )
  
  
  print(
    mean(
      (
        y_test -
          package_prediction
      )^2
    )
  )
  
  
  cat(
    "\nPackage importance:\n"
  )
  
  
  print(
    randomForest::importance(
      package_forest
    )
  )
  
  
  # Exact results will differ because:
  #
  #   - our tree-growing implementation is simplified;
  #   - candidate split search is approximate;
  #   - package internals use optimized CART routines;
  #   - stopping and randomization details differ.
  
}


# ==============================================================================
# PART XXXI
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nRandom Forest Summary\n"
)


cat(
  "---------------------\n"
)


cat(
  "Training observations:",
  nrow(
    X_train
  ),
  "\n"
)


cat(
  "Test observations:",
  nrow(
    X_test
  ),
  "\n"
)


cat(
  "Predictors:",
  p,
  "\n"
)


cat(
  "Trees:",
  forest$number_trees,
  "\n"
)


cat(
  "mtry:",
  forest$mtry,
  "\n"
)


cat(
  "OOB MSE:",
  round(
    oob_metrics[
      "MSE"
    ],
    5
  ),
  "\n"
)


cat(
  "Test MSE:",
  round(
    regression_metrics(
      y_test,
      forest_test_prediction
    )[
      "MSE"
    ],
    5
  ),
  "\n"
)


cat(
  "Bagging tree correlation:",
  round(
    bagging_tree_correlation,
    4
  ),
  "\n"
)


cat(
  "Random forest tree correlation:",
  round(
    rf_tree_correlation,
    4
  ),
  "\n"
)


cat(
  "Most important variable by permutation:",
  permutation_table$Variable[1],
  "\n"
)