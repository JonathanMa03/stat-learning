# ==============================================================================
# Regression Trees
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Recursive binary splitting
#   - CART regression trees
#   - Residual sum of squares
#   - Terminal nodes / leaves
#   - Greedy tree construction
#   - Tree depth
#   - Minimum node size
#   - Variable importance
#   - Cost-complexity pruning
#   - Weakest-link pruning
#   - Cross-validation
#   - Bias-variance tradeoff
#
# Regression tree model:
#
#   f(x) = sum_{m=1}^M c_m I(x in R_m)
#
# where:
#
#   R_m = terminal regions
#   c_m = mean response inside region m
#
#
# CART chooses splits by minimizing:
#
#   RSS =
#   sum_{i: x_i in R_1} (y_i - mean(y_R1))^2
#   +
#   sum_{i: x_i in R_2} (y_i - mean(y_R2))^2
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


true_function <- function(
    x1,
    x2,
    x3
) {
  
  3 +
    ifelse(
      x1 < 0,
      -2,
      2
    ) +
    ifelse(
      x2 < 1,
      1.5 * x2,
      3
    ) +
    0.75 * sin(
      2 * x3
    )
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
  main = "Regression Tree Data"
)


# ==============================================================================
# PART II
#
# BASIC TREE QUANTITIES
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
# 4. Constant Prediction Inside a Node
# ------------------------------------------------------------------------------

node_prediction <- function(y) {
  
  mean(y)
}


# ------------------------------------------------------------------------------
# 5. Root Node Error
# ------------------------------------------------------------------------------

root_prediction <- node_prediction(
  y
)


root_RSS <- node_rss(
  y
)


root_prediction


root_RSS


# ==============================================================================
# PART III
#
# FIND THE BEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Candidate Split Points
# ------------------------------------------------------------------------------

# For a continuous predictor, it is enough to consider midpoints between
# consecutive unique sorted values.


candidate_splits <- function(x) {
  
  unique_values <- sort(
    unique(
      x
    )
  )
  
  
  if (length(unique_values) < 2) {
    return(numeric(0))
  }
  
  
  (
    unique_values[
      -length(unique_values)
    ] +
      unique_values[
        -1
      ]
  ) / 2
}


# ------------------------------------------------------------------------------
# 7. Evaluate One Split
# ------------------------------------------------------------------------------

evaluate_split <- function(
    x,
    y,
    split_value,
    min_leaf = 5
) {
  
  left <- x <= split_value
  
  right <- x > split_value
  
  
  if (
    sum(left) < min_leaf ||
    sum(right) < min_leaf
  ) {
    
    return(Inf)
  }
  
  
  left_RSS <- node_rss(
    y[left]
  )
  
  
  right_RSS <- node_rss(
    y[right]
  )
  
  
  left_RSS +
    right_RSS
}


# ------------------------------------------------------------------------------
# 8. Find Best Split Across All Predictors
# ------------------------------------------------------------------------------

find_best_split <- function(
    X,
    y,
    min_leaf = 5
) {
  
  X <- as.matrix(X)
  
  p <- ncol(X)
  
  
  parent_RSS <- node_rss(
    y
  )
  
  
  best_RSS <- Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  
  for (j in seq_len(p)) {
    
    split_values <- candidate_splits(
      X[, j]
    )
    
    
    if (length(split_values) == 0) {
      next
    }
    
    
    for (split_value in split_values) {
      
      split_RSS <- evaluate_split(
        x = X[, j],
        y = y,
        split_value = split_value,
        min_leaf = min_leaf
      )
      
      
      if (split_RSS < best_RSS) {
        
        best_RSS <- split_RSS
        
        best_variable <- j
        
        best_split <- split_value
      }
    }
  }
  
  
  if (
    is.infinite(best_RSS) ||
    is.na(best_variable)
  ) {
    
    return(
      list(
        variable = NA_integer_,
        split = NA_real_,
        RSS = parent_RSS,
        improvement = 0
      )
    )
  }
  
  
  improvement <- parent_RSS -
    best_RSS
  
  
  return(
    list(
      variable = best_variable,
      split = best_split,
      RSS = best_RSS,
      improvement = improvement
    )
  )
}


# ------------------------------------------------------------------------------
# 9. Inspect Best Root Split
# ------------------------------------------------------------------------------

root_split <- find_best_split(
  X = X,
  y = y,
  min_leaf = 10
)


root_split


colnames(X)[
  root_split$variable
]


# ==============================================================================
# PART IV
#
# MANUAL CART REGRESSION TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Grow Regression Tree Recursively
# ------------------------------------------------------------------------------

grow_regression_tree <- function(
    X,
    y,
    indices = seq_len(nrow(X)),
    depth = 0,
    max_depth = 5,
    min_split = 20,
    min_leaf = 10,
    min_improvement = 0
) {
  
  X <- as.matrix(X)
  
  y_node <- y[
    indices
  ]
  
  
  node <- list(
    prediction = mean(y_node),
    RSS = node_rss(y_node),
    n = length(indices),
    depth = depth,
    indices = indices,
    terminal = TRUE,
    variable = NA_integer_,
    split = NA_real_,
    improvement = 0,
    left = NULL,
    right = NULL
  )
  
  
  # --------------------------------------------------------------------------
  # Stopping conditions
  # --------------------------------------------------------------------------
  
  if (depth >= max_depth) {
    return(node)
  }
  
  
  if (length(indices) < min_split) {
    return(node)
  }
  
  
  if (length(unique(y_node)) <= 1) {
    return(node)
  }
  
  
  # --------------------------------------------------------------------------
  # Search for best split
  # --------------------------------------------------------------------------
  
  best_split <- find_best_split(
    X = X[
      indices,
      ,
      drop = FALSE
    ],
    y = y_node,
    min_leaf = min_leaf
  )
  
  
  if (is.na(best_split$variable)) {
    return(node)
  }
  
  
  if (best_split$improvement <= min_improvement) {
    return(node)
  }
  
  
  split_variable <- best_split$variable
  
  split_value <- best_split$split
  
  
  left_local <- X[
    indices,
    split_variable
  ] <= split_value
  
  
  right_local <- !left_local
  
  
  left_indices <- indices[
    left_local
  ]
  
  
  right_indices <- indices[
    right_local
  ]
  
  
  if (
    length(left_indices) < min_leaf ||
    length(right_indices) < min_leaf
  ) {
    
    return(node)
  }
  
  
  # --------------------------------------------------------------------------
  # Convert node into internal node
  # --------------------------------------------------------------------------
  
  node$terminal <- FALSE
  
  node$variable <- split_variable
  
  node$split <- split_value
  
  node$improvement <- best_split$improvement
  
  
  node$left <- grow_regression_tree(
    X = X,
    y = y,
    indices = left_indices,
    depth = depth + 1,
    max_depth = max_depth,
    min_split = min_split,
    min_leaf = min_leaf,
    min_improvement = min_improvement
  )
  
  
  node$right <- grow_regression_tree(
    X = X,
    y = y,
    indices = right_indices,
    depth = depth + 1,
    max_depth = max_depth,
    min_split = min_split,
    min_leaf = min_leaf,
    min_improvement = min_improvement
  )
  
  
  return(node)
}


# ------------------------------------------------------------------------------
# 11. Grow Tree
# ------------------------------------------------------------------------------

tree <- grow_regression_tree(
  X = X,
  y = y,
  max_depth = 5,
  min_split = 20,
  min_leaf = 10
)


# ==============================================================================
# PART V
#
# TREE INSPECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Print Tree
# ------------------------------------------------------------------------------

print_tree <- function(
    tree,
    feature_names,
    indent = ""
) {
  
  if (tree$terminal) {
    
    cat(
      indent,
      "Leaf: n =",
      tree$n,
      "| prediction =",
      round(
        tree$prediction,
        3
      ),
      "| RSS =",
      round(
        tree$RSS,
        3
      ),
      "\n"
    )
    
    return(
      invisible(NULL)
    )
  }
  
  
  variable_name <- feature_names[
    tree$variable
  ]
  
  
  cat(
    indent,
    variable_name,
    "<=",
    round(
      tree$split,
      3
    ),
    "| improvement =",
    round(
      tree$improvement,
      3
    ),
    "\n"
  )
  
  
  cat(
    indent,
    "LEFT:\n"
  )
  
  
  print_tree(
    tree$left,
    feature_names,
    paste0(
      indent,
      "  "
    )
  )
  
  
  cat(
    indent,
    "RIGHT:\n"
  )
  
  
  print_tree(
    tree$right,
    feature_names,
    paste0(
      indent,
      "  "
    )
  )
  
  
  invisible(NULL)
}


print_tree(
  tree,
  colnames(X)
)


# ------------------------------------------------------------------------------
# 13. Count Terminal Nodes
# ------------------------------------------------------------------------------

count_leaves <- function(tree) {
  
  if (tree$terminal) {
    return(1)
  }
  
  
  count_leaves(
    tree$left
  ) +
    count_leaves(
      tree$right
    )
}


count_leaves(
  tree
)


# ------------------------------------------------------------------------------
# 14. Count Total Nodes
# ------------------------------------------------------------------------------

count_nodes <- function(tree) {
  
  if (tree$terminal) {
    return(1)
  }
  
  
  1 +
    count_nodes(
      tree$left
    ) +
    count_nodes(
      tree$right
    )
}


count_nodes(
  tree
)


# ------------------------------------------------------------------------------
# 15. Maximum Tree Depth
# ------------------------------------------------------------------------------

tree_depth <- function(tree) {
  
  if (tree$terminal) {
    return(tree$depth)
  }
  
  
  max(
    tree_depth(
      tree$left
    ),
    tree_depth(
      tree$right
    )
  )
}


tree_depth(
  tree
)


# ==============================================================================
# PART VI
#
# TREE PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Predict One Observation
# ------------------------------------------------------------------------------

predict_tree_one <- function(
    tree,
    x
) {
  
  if (tree$terminal) {
    return(tree$prediction)
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
# 17. Predict Multiple Observations
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


# ------------------------------------------------------------------------------
# 18. Training Predictions
# ------------------------------------------------------------------------------

tree_prediction <- predict_tree(
  tree,
  X
)


tree_training_MSE <- mean(
  (
    y -
      tree_prediction
  )^2
)


tree_training_RMSE <- sqrt(
  tree_training_MSE
)


tree_training_MSE


tree_training_RMSE


# ==============================================================================
# PART VII
#
# TERMINAL REGIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Obtain Leaf Assignment
# ------------------------------------------------------------------------------

predict_leaf_one <- function(
    tree,
    x,
    path = "root"
) {
  
  if (tree$terminal) {
    return(path)
  }
  
  
  if (
    x[
      tree$variable
    ] <=
    tree$split
  ) {
    
    predict_leaf_one(
      tree$left,
      x,
      paste0(
        path,
        "L"
      )
    )
    
  } else {
    
    predict_leaf_one(
      tree$right,
      x,
      paste0(
        path,
        "R"
      )
    )
  }
}


leaf_assignment <- apply(
  X,
  1,
  function(x) {
    
    predict_leaf_one(
      tree,
      x
    )
  }
)


table(
  leaf_assignment
)


# ------------------------------------------------------------------------------
# 20. Check Constant Prediction Within Leaves
# ------------------------------------------------------------------------------

head(
  data.frame(
    y = y,
    prediction = tree_prediction,
    leaf = leaf_assignment
  )
)


# ==============================================================================
# PART VIII
#
# VARIABLE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. CART Variable Importance
# ------------------------------------------------------------------------------

# Each split reduces RSS.
#
# Sum the RSS reduction contributed by each predictor.


tree_variable_importance <- function(
    tree,
    p
) {
  
  importance <- numeric(
    p
  )
  
  
  accumulate_importance <- function(node) {
    
    if (node$terminal) {
      return(invisible(NULL))
    }
    
    
    importance[
      node$variable
    ] <<-
      importance[
        node$variable
      ] +
      node$improvement
    
    
    accumulate_importance(
      node$left
    )
    
    
    accumulate_importance(
      node$right
    )
    
    
    invisible(NULL)
  }
  
  
  accumulate_importance(
    tree
  )
  
  
  importance
}


importance <- tree_variable_importance(
  tree,
  ncol(X)
)


names(
  importance
) <- colnames(X)


importance


# ------------------------------------------------------------------------------
# 22. Normalize Variable Importance
# ------------------------------------------------------------------------------

if (sum(importance) > 0) {
  
  normalized_importance <-
    importance /
    sum(importance)
  
} else {
  
  normalized_importance <-
    importance
}


normalized_importance


barplot(
  normalized_importance,
  ylab = "Relative Importance",
  main = "Regression Tree Variable Importance"
)


# ==============================================================================
# PART IX
#
# EFFECT OF TREE DEPTH
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Fit Trees of Different Depth
# ------------------------------------------------------------------------------

depth_values <- 1:10


depth_results <- data.frame(
  Depth = depth_values,
  Leaves = NA,
  Training_MSE = NA
)


depth_trees <- vector(
  "list",
  length(
    depth_values
  )
)


for (i in seq_along(depth_values)) {
  
  tree_i <- grow_regression_tree(
    X = X,
    y = y,
    max_depth = depth_values[i],
    min_split = 20,
    min_leaf = 5
  )
  
  
  prediction_i <- predict_tree(
    tree_i,
    X
  )
  
  
  depth_trees[[i]] <- tree_i
  
  
  depth_results$Leaves[i] <- count_leaves(
    tree_i
  )
  
  
  depth_results$Training_MSE[i] <- mean(
    (
      y -
        prediction_i
    )^2
  )
}


depth_results


# ------------------------------------------------------------------------------
# 24. Plot Training Error vs Tree Depth
# ------------------------------------------------------------------------------

plot(
  depth_results$Depth,
  depth_results$Training_MSE,
  type = "b",
  pch = 19,
  xlab = "Maximum Tree Depth",
  ylab = "Training MSE",
  main = "Tree Complexity and Training Error"
)


# ==============================================================================
# PART X
#
# TRAIN-TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(456)


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
# 26. Grow Large Tree on Training Data
# ------------------------------------------------------------------------------

large_tree <- grow_regression_tree(
  X = X_train,
  y = y_train,
  max_depth = 10,
  min_split = 10,
  min_leaf = 5
)


large_train_prediction <- predict_tree(
  large_tree,
  X_train
)


large_test_prediction <- predict_tree(
  large_tree,
  X_test
)


large_train_MSE <- mean(
  (
    y_train -
      large_train_prediction
  )^2
)


large_test_MSE <- mean(
  (
    y_test -
      large_test_prediction
  )^2
)


data.frame(
  Quantity = c(
    "Training MSE",
    "Test MSE"
  ),
  Value = c(
    large_train_MSE,
    large_test_MSE
  )
)


# ==============================================================================
# PART XI
#
# TREE DEPTH AS REGULARIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Compare Depth on Training and Test Data
# ------------------------------------------------------------------------------

depth_test_results <- data.frame(
  Depth = depth_values,
  Leaves = NA,
  Training_MSE = NA,
  Test_MSE = NA
)


for (i in seq_along(depth_values)) {
  
  tree_i <- grow_regression_tree(
    X = X_train,
    y = y_train,
    max_depth = depth_values[i],
    min_split = 10,
    min_leaf = 5
  )
  
  
  train_prediction_i <- predict_tree(
    tree_i,
    X_train
  )
  
  
  test_prediction_i <- predict_tree(
    tree_i,
    X_test
  )
  
  
  depth_test_results$Leaves[i] <- count_leaves(
    tree_i
  )
  
  
  depth_test_results$Training_MSE[i] <- mean(
    (
      y_train -
        train_prediction_i
    )^2
  )
  
  
  depth_test_results$Test_MSE[i] <- mean(
    (
      y_test -
        test_prediction_i
    )^2
  )
}


depth_test_results


# ------------------------------------------------------------------------------
# 28. Plot Bias-Variance Tradeoff
# ------------------------------------------------------------------------------

matplot(
  depth_test_results$Depth,
  cbind(
    depth_test_results$Training_MSE,
    depth_test_results$Test_MSE
  ),
  type = "b",
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  ),
  xlab = "Maximum Tree Depth",
  ylab = "MSE",
  main = "Regression Tree Bias-Variance Tradeoff"
)


legend(
  "topright",
  legend = c(
    "Training",
    "Test"
  ),
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  )
)


# ==============================================================================
# PART XII
#
# COST-COMPLEXITY PRUNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Subtree RSS
# ------------------------------------------------------------------------------

# For a tree T:
#
#   R(T)
#
# is the total RSS across terminal nodes.


subtree_RSS <- function(tree) {
  
  if (tree$terminal) {
    return(tree$RSS)
  }
  
  
  subtree_RSS(
    tree$left
  ) +
    subtree_RSS(
      tree$right
    )
}


# ------------------------------------------------------------------------------
# 30. Cost-Complexity Criterion
# ------------------------------------------------------------------------------

# CART pruning minimizes:
#
#   R_alpha(T)
#   =
#   R(T) + alpha |T|
#
# where:
#
#   |T| = number of terminal nodes.


cost_complexity <- function(
    tree,
    alpha
) {
  
  subtree_RSS(
    tree
  ) +
    alpha *
    count_leaves(
      tree
    )
}


# ------------------------------------------------------------------------------
# 31. Effective Alpha for an Internal Node
# ------------------------------------------------------------------------------

# Suppose an internal node t currently contains a whole subtree T_t.
#
# If we collapse that subtree into one leaf:
#
#   g(t)
#
#   =
#   [R(t) - R(T_t)]
#   /
#   [|T_t| - 1]
#
# The smallest g(t) identifies the weakest link.


node_effective_alpha <- function(tree) {
  
  if (tree$terminal) {
    return(Inf)
  }
  
  
  number_leaves <- count_leaves(
    tree
  )
  
  
  if (number_leaves <= 1) {
    return(Inf)
  }
  
  
  (
    tree$RSS -
      subtree_RSS(
        tree
      )
  ) /
    (
      number_leaves -
        1
    )
}


# ------------------------------------------------------------------------------
# 32. Find Weakest Link
# ------------------------------------------------------------------------------

find_weakest_link <- function(
    tree,
    path = ""
) {
  
  if (tree$terminal) {
    
    return(
      data.frame(
        path = character(0),
        alpha = numeric(0)
      )
    )
  }
  
  
  current <- data.frame(
    path = path,
    alpha = node_effective_alpha(
      tree
    )
  )
  
  
  left_nodes <- find_weakest_link(
    tree$left,
    paste0(
      path,
      "L"
    )
  )
  
  
  right_nodes <- find_weakest_link(
    tree$right,
    paste0(
      path,
      "R"
    )
  )
  
  
  rbind(
    current,
    left_nodes,
    right_nodes
  )
}


# ------------------------------------------------------------------------------
# 33. Prune Tree at a Path
# ------------------------------------------------------------------------------

prune_at_path <- function(
    tree,
    path
) {
  
  # Empty path means prune root.
  
  if (path == "") {
    
    tree$terminal <- TRUE
    
    tree$variable <- NA_integer_
    
    tree$split <- NA_real_
    
    tree$improvement <- 0
    
    tree$left <- NULL
    
    tree$right <- NULL
    
    return(tree)
  }
  
  
  direction <- substr(
    path,
    1,
    1
  )
  
  
  remaining_path <- substr(
    path,
    2,
    nchar(path)
  )
  
  
  if (direction == "L") {
    
    tree$left <- prune_at_path(
      tree$left,
      remaining_path
    )
    
  } else {
    
    tree$right <- prune_at_path(
      tree$right,
      remaining_path
    )
  }
  
  
  return(tree)
}


# ------------------------------------------------------------------------------
# 34. Weakest-Link Pruning Sequence
# ------------------------------------------------------------------------------

weakest_link_sequence <- function(tree) {
  
  trees <- list(
    tree
  )
  
  
  alphas <- 0
  
  
  current_tree <- tree
  
  
  while (
    !current_tree$terminal
  ) {
    
    candidate_nodes <- find_weakest_link(
      current_tree
    )
    
    
    minimum_alpha <- min(
      candidate_nodes$alpha
    )
    
    
    # If several nodes tie, prune all weakest links at that alpha.
    #
    # Prune deeper paths first so that parent paths remain valid.
    
    weakest_paths <- candidate_nodes$path[
      abs(
        candidate_nodes$alpha -
          minimum_alpha
      ) <
        1e-10
    ]
    
    
    weakest_paths <- weakest_paths[
      order(
        nchar(
          weakest_paths
        ),
        decreasing = TRUE
      )
    ]
    
    
    for (path in weakest_paths) {
      
      current_tree <- prune_at_path(
        current_tree,
        path
      )
    }
    
    
    trees[[length(trees) + 1]] <- current_tree
    
    
    alphas <- c(
      alphas,
      minimum_alpha
    )
  }
  
  
  return(
    list(
      trees = trees,
      alphas = alphas
    )
  )
}


# ------------------------------------------------------------------------------
# 35. Generate Pruning Sequence
# ------------------------------------------------------------------------------

pruning_sequence <- weakest_link_sequence(
  large_tree
)


length(
  pruning_sequence$trees
)


pruning_sequence$alphas


# ------------------------------------------------------------------------------
# 36. Inspect Pruned Tree Sizes
# ------------------------------------------------------------------------------

pruning_results <- data.frame(
  Step = seq_along(
    pruning_sequence$trees
  ),
  Alpha = pruning_sequence$alphas,
  Leaves = sapply(
    pruning_sequence$trees,
    count_leaves
  ),
  Training_RSS = sapply(
    pruning_sequence$trees,
    subtree_RSS
  )
)


pruning_results


# ------------------------------------------------------------------------------
# 37. Plot Pruning Path
# ------------------------------------------------------------------------------

plot(
  pruning_results$Leaves,
  pruning_results$Training_RSS,
  type = "b",
  pch = 19,
  xlab = "Number of Terminal Nodes",
  ylab = "Training RSS",
  main = "Cost-Complexity Pruning Path"
)


# ==============================================================================
# PART XIII
#
# TEST PERFORMANCE ALONG PRUNING PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Evaluate Every Pruned Tree
# ------------------------------------------------------------------------------

pruning_test_results <- pruning_results


pruning_test_results$Training_MSE <- NA

pruning_test_results$Test_MSE <- NA


for (
  i in seq_along(
    pruning_sequence$trees
  )
) {
  
  tree_i <- pruning_sequence$trees[[i]]
   
  
  train_prediction_i <- predict_tree(
    tree_i,
    X_train
  )
  
  
  test_prediction_i <- predict_tree(
    tree_i,
    X_test
  )
  
  
  pruning_test_results$Training_MSE[i] <- mean(
    (
      y_train -
        train_prediction_i
    )^2
  )
  
  
  pruning_test_results$Test_MSE[i] <- mean(
    (
      y_test -
        test_prediction_i
    )^2
  )
}


pruning_test_results


# ------------------------------------------------------------------------------
# 39. Best Pruned Tree on Test Data
# ------------------------------------------------------------------------------

best_pruned_index <- which.min(
  pruning_test_results$Test_MSE
)


best_pruned_tree <- pruning_sequence$trees[[best_pruned_index]]


pruning_test_results[
  best_pruned_index,
]


# ------------------------------------------------------------------------------
# 40. Plot Error Along Pruning Path
# ------------------------------------------------------------------------------

matplot(
  pruning_test_results$Leaves,
  cbind(
    pruning_test_results$Training_MSE,
    pruning_test_results$Test_MSE
  ),
  type = "b",
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  ),
  xlab = "Number of Terminal Nodes",
  ylab = "MSE",
  main = "Cost-Complexity Pruning"
)


legend(
  "topright",
  legend = c(
    "Training",
    "Test"
  ),
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  )
)


# ==============================================================================
# PART XIV
#
# CROSS-VALIDATING TREE COMPLEXITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. K-Fold CV for Maximum Depth
# ------------------------------------------------------------------------------

tree_depth_cv <- function(
    X,
    y,
    depths = 1:8,
    k = 5,
    min_split = 10,
    min_leaf = 5,
    seed = 123
) {
  
  set.seed(seed)
  
  
  n <- nrow(X)
  
  
  fold_id <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  fold_error <- matrix(
    NA,
    nrow = length(depths),
    ncol = k
  )
  
  
  for (
    d in seq_along(depths)
  ) {
    
    for (
      fold in seq_len(k)
    ) {
      
      training_indices <- which(
        fold_id != fold
      )
      
      
      validation_indices <- which(
        fold_id == fold
      )
      
      
      tree_fold <- grow_regression_tree(
        X = X[
          training_indices,
          ,
          drop = FALSE
        ],
        y = y[
          training_indices
        ],
        max_depth = depths[d],
        min_split = min_split,
        min_leaf = min_leaf
      )
      
      
      prediction_fold <- predict_tree(
        tree_fold,
        X[
          validation_indices,
          ,
          drop = FALSE
        ]
      )
      
      
      fold_error[
        d,
        fold
      ] <- mean(
        (
          y[
            validation_indices
          ] -
            prediction_fold
        )^2
      )
    }
  }
  
  
  mean_error <- rowMeans(
    fold_error
  )
  
  
  SE_error <- apply(
    fold_error,
    1,
    sd
  ) /
    sqrt(k)
  
  
  return(
    data.frame(
      Depth = depths,
      CV_MSE = mean_error,
      CV_SE = SE_error
    )
  )
}


# ------------------------------------------------------------------------------
# 42. Run Cross Validation
# ------------------------------------------------------------------------------

CV_results <- tree_depth_cv(
  X = X_train,
  y = y_train,
  depths = 1:8,
  k = 5,
  min_split = 10,
  min_leaf = 5,
  seed = 123
)


CV_results


# ------------------------------------------------------------------------------
# 43. Select Best Depth
# ------------------------------------------------------------------------------

best_depth <- CV_results$Depth[
  which.min(
    CV_results$CV_MSE
  )
]


best_depth


# ------------------------------------------------------------------------------
# 44. Plot CV Error
# ------------------------------------------------------------------------------

plot(
  CV_results$Depth,
  CV_results$CV_MSE,
  type = "b",
  pch = 19,
  xlab = "Maximum Tree Depth",
  ylab = "Cross-Validated MSE",
  main = "Regression Tree Cross Validation"
)


arrows(
  x0 = CV_results$Depth,
  y0 = CV_results$CV_MSE -
    CV_results$CV_SE,
  x1 = CV_results$Depth,
  y1 = CV_results$CV_MSE +
    CV_results$CV_SE,
  angle = 90,
  code = 3,
  length = 0.05
)


# ------------------------------------------------------------------------------
# 45. One-Standard-Error Rule
# ------------------------------------------------------------------------------

minimum_index <- which.min(
  CV_results$CV_MSE
)


minimum_error <- CV_results$CV_MSE[
  minimum_index
]


minimum_SE <- CV_results$CV_SE[
  minimum_index
]


threshold <- minimum_error +
  minimum_SE


eligible_depths <- CV_results$Depth[
  CV_results$CV_MSE <= threshold
]


one_SE_depth <- min(
  eligible_depths
)


one_SE_depth


# ------------------------------------------------------------------------------
# 46. Fit CV-Selected Tree
# ------------------------------------------------------------------------------

CV_tree <- grow_regression_tree(
  X = X_train,
  y = y_train,
  max_depth = one_SE_depth,
  min_split = 10,
  min_leaf = 5
)


CV_tree_prediction <- predict_tree(
  CV_tree,
  X_test
)


CV_tree_test_MSE <- mean(
  (
    y_test -
      CV_tree_prediction
  )^2
)


CV_tree_test_MSE


# ==============================================================================
# PART XV
#
# VISUALIZE TREE PREDICTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. One-Dimensional Slice
# ------------------------------------------------------------------------------

# Fix x2 = 0 and x3 = 0.
#
# Vary x1 to see the characteristic step-function prediction.


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


tree_slice_prediction <- predict_tree(
  CV_tree,
  slice_data
)


true_slice <- true_function(
  x1_grid,
  0,
  0
)


# ------------------------------------------------------------------------------
# 48. Plot Step-Function Prediction
# ------------------------------------------------------------------------------

plot(
  x1_grid,
  tree_slice_prediction,
  type = "s",
  lwd = 2,
  xlab = "x1",
  ylab = "Predicted y",
  main = "Regression Tree Prediction"
)


lines(
  x1_grid,
  true_slice,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Tree",
    "True Function"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART XVI
#
# TWO-DIMENSIONAL PARTITION
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Fit Tree Using Only x1 and x2
# ------------------------------------------------------------------------------

X_2D <- X_train[
  ,
  c(
    "x1",
    "x2"
  ),
  drop = FALSE
]


tree_2D <- grow_regression_tree(
  X = X_2D,
  y = y_train,
  max_depth = 4,
  min_split = 15,
  min_leaf = 8
)


# ------------------------------------------------------------------------------
# 50. Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 100


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


grid_prediction <- predict_tree(
  tree_2D,
  as.matrix(
    prediction_grid
  )
)


prediction_matrix <- matrix(
  grid_prediction,
  nrow = grid_size,
  ncol = grid_size
)


# ------------------------------------------------------------------------------
# 51. Plot Rectangular Prediction Regions
# ------------------------------------------------------------------------------

image(
  grid_x1,
  grid_x2,
  prediction_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "Regression Tree Prediction Regions"
)


points(
  X_train[
    ,
    "x1"
  ],
  X_train[
    ,
    "x2"
  ],
  pch = 19,
  cex = 0.25
)


# ==============================================================================
# PART XVII
#
# COMPARE WITH LINEAR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 52. Linear Regression Baseline
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = y_train,
  x1 = X_train[
    ,
    "x1"
  ],
  x2 = X_train[
    ,
    "x2"
  ],
  x3 = X_train[
    ,
    "x3"
  ]
)


test_data <- data.frame(
  x1 = X_test[
    ,
    "x1"
  ],
  x2 = X_test[
    ,
    "x2"
  ],
  x3 = X_test[
    ,
    "x3"
  ]
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
# 53. More Flexible Polynomial Baseline
# ------------------------------------------------------------------------------

polynomial_model <- lm(
  y ~
    x1 +
    I(x1^2) +
    x2 +
    I(x2^2) +
    x3 +
    I(x3^2),
  data = train_data
)


polynomial_test_prediction <- predict(
  polynomial_model,
  newdata = test_data
)


polynomial_test_MSE <- mean(
  (
    y_test -
      polynomial_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 54. Comparison
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Quadratic Regression",
    "Large Regression Tree",
    "CV-Selected Regression Tree"
  ),
  
  Test_MSE = c(
    linear_test_MSE,
    polynomial_test_MSE,
    large_test_MSE,
    CV_tree_test_MSE
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XVIII
#
# OPTIONAL VERIFICATION WITH RPART
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Compare with rpart
# ------------------------------------------------------------------------------

# rpart is NOT required for this demo.
#
# It is used only as an optional check against a production CART
# implementation.


if (
  requireNamespace(
    "rpart",
    quietly = TRUE
  )
) {
  
  rpart_model <- rpart::rpart(
    y ~ x1 + x2 + x3,
    data = train_data,
    method = "anova",
    control = rpart::rpart.control(
      cp = 0,
      minsplit = 10,
      minbucket = 5,
      maxdepth = 10
    )
  )
  
  
  rpart_prediction <- predict(
    rpart_model,
    newdata = test_data
  )
  
  
  rpart_test_MSE <- mean(
    (
      y_test -
        rpart_prediction
    )^2
  )
  
  
  cat(
    "Manual tree test MSE:",
    round(
      large_test_MSE,
      4
    ),
    "\n"
  )
  
  
  cat(
    "rpart test MSE:",
    round(
      rpart_test_MSE,
      4
    ),
    "\n"
  )
  
  
  print(
    rpart_model
  )
  
  
  print(
    rpart::printcp(
      rpart_model
    )
  )
}


# ==============================================================================
# PART XIX
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Final Tree Statistics
# ------------------------------------------------------------------------------

cat(
  "Regression Trees Summary\n"
)


cat(
  "------------------------\n"
)


cat(
  "Large tree leaves:",
  count_leaves(
    large_tree
  ),
  "\n"
)


cat(
  "Large tree maximum depth:",
  tree_depth(
    large_tree
  ),
  "\n"
)


cat(
  "Large tree training MSE:",
  round(
    large_train_MSE,
    4
  ),
  "\n"
)


cat(
  "Large tree test MSE:",
  round(
    large_test_MSE,
    4
  ),
  "\n"
)


cat(
  "CV-selected depth:",
  best_depth,
  "\n"
)


cat(
  "One-SE selected depth:",
  one_SE_depth,
  "\n"
)


cat(
  "CV-selected tree leaves:",
  count_leaves(
    CV_tree
  ),
  "\n"
)


cat(
  "CV-selected tree test MSE:",
  round(
    CV_tree_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Best pruning-path tree leaves:",
  count_leaves(
    best_pruned_tree
  ),
  "\n"
)


cat(
  "Best pruning-path test MSE:",
  round(
    pruning_test_results$Test_MSE[
      best_pruned_index
    ],
    4
  ),
  "\n"
)


cat(
  "\nVariable importance:\n"
)


print(
  round(
    normalized_importance,
    4
  )
)