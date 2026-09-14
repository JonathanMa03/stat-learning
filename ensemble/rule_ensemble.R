# ==============================================================================
# Rule Ensembles
# Jonathan Ma
#
# Package-free pedagogical implementation inspired by RuleFit.
#
# Main ideas:
#   - Decision trees as rule generators
#   - Extracting rules from internal and terminal tree nodes
#   - Randomized shallow trees
#   - Rule indicator matrices
#   - Linear terms + rule terms
#   - Rule standardization
#   - L1 regularization
#   - Coordinate descent
#   - Cross-validation
#   - Sparse rule selection
#   - Rule importance
#   - Variable importance
#   - Interpretability
#
#
# General rule ensemble:
#
#   f(x)
#
#     =
#
#   beta_0
#
#     +
#
#   sum_j beta_j x_j
#
#     +
#
#   sum_k alpha_k r_k(x)
#
#
# where:
#
#   r_k(x) in {0,1}
#
# is a decision rule generated from a tree.
#
#
# We then estimate coefficients with an L1 penalty:
#
#   minimize
#
#       (1/(2n)) ||y - Z theta||^2
#
#       +
#
#       lambda ||theta||_1
#
#
# where Z contains both linear predictors and tree rules.
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR DATA
# ==============================================================================


set.seed(123)


n <- 600

p <- 6


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
# True Regression Function
#
# Contains:
#
#   - linear effects
#   - threshold effects
#   - interactions
#   - nonlinear structure
#
# This is exactly the kind of setting where rule ensembles can be useful.
# ------------------------------------------------------------------------------

true_function <- function(X) {
  
  2 +
    1.5 * X[, 1] -
    1.0 * X[, 2] +
    3.0 *
    (
      X[, 1] >
        0.5
    ) *
    (
      X[, 3] <
        0
    ) -
    2.5 *
    (
      X[, 2] <
        -0.75
    ) +
    2.0 *
    (
      X[, 4] >
        0
    ) *
    (
      X[, 5] >
        0
    )
}


signal <- true_function(
  X
)


y <- signal +
  rnorm(
    n,
    sd = 1.5
  )


# ------------------------------------------------------------------------------
# Exploration
# ------------------------------------------------------------------------------

par(
  mfrow = c(2, 2)
)


plot(
  X[, 1],
  y,
  pch = 19,
  cex = 0.5,
  xlab = "X1",
  ylab = "Y",
  main = "Y vs X1"
)


plot(
  X[, 2],
  y,
  pch = 19,
  cex = 0.5,
  xlab = "X2",
  ylab = "Y",
  main = "Y vs X2"
)


plot(
  X[, 3],
  y,
  pch = 19,
  cex = 0.5,
  xlab = "X3",
  ylab = "Y",
  main = "Y vs X3"
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


set.seed(456)


train_indices <- sample(
  seq_len(n),
  floor(
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


# ==============================================================================
# PART III
#
# METRICS
# ==============================================================================


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
# BASIC TREE FUNCTIONS
# ==============================================================================


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
# Candidate Split Values
# ------------------------------------------------------------------------------

candidate_splits <- function(
    x,
    maximum_splits = 20
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
  
  
  split_values <- quantile(
    x,
    probs =
      probabilities,
    names = FALSE
  )
  
  
  sort(
    unique(
      as.numeric(
        split_values
      )
    )
  )
}


# ==============================================================================
# PART V
#
# RANDOMIZED TREE SPLITS
# ==============================================================================


# ------------------------------------------------------------------------------
# RuleFit obtains a diverse collection of trees.
#
# Here we create diversity using:
#
#   1. bootstrap / subsampling;
#   2. random subsets of predictors;
#   3. shallow random tree depths.
#
# This is a pedagogical approximation to the tree-generation stage.
# ------------------------------------------------------------------------------


find_best_random_split <- function(
    X,
    y,
    minimum_leaf_size = 10,
    maximum_splits = 20,
    mtry = NULL
) {
  
  X <- as.matrix(X)
  
  
  p <- ncol(X)
  
  
  if (
    is.null(mtry)
  ) {
    
    mtry <- max(
      1,
      floor(
        sqrt(p)
      )
    )
  }
  
  
  mtry <- min(
    mtry,
    p
  )
  
  
  variables <- sample(
    seq_len(p),
    size = mtry,
    replace = FALSE
  )
  
  
  parent_rss <- node_rss(
    y
  )
  
  
  best_improvement <- -Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  best_left <- NULL
  
  best_right <- NULL
  
  
  for (
    variable in variables
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
    is.na(
      best_variable
    )
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
# GROW SHALLOW RANDOMIZED TREES
# ==============================================================================


grow_rule_tree <- function(
    X,
    y,
    minimum_leaf_size = 10,
    maximum_depth = 3,
    maximum_splits = 20,
    mtry = NULL,
    current_depth = 0
) {
  
  X <- as.matrix(X)
  
  
  n_node <- nrow(X)
  
  
  prediction <- mean(
    y
  )
  
  
  if (
    n_node <
    2 * minimum_leaf_size ||
    current_depth >=
    maximum_depth
  ) {
    
    return(
      list(
        terminal = TRUE,
        prediction =
          prediction,
        n =
          n_node
      )
    )
  }
  
  
  best_split <- find_best_random_split(
    X,
    y,
    minimum_leaf_size =
      minimum_leaf_size,
    maximum_splits =
      maximum_splits,
    mtry =
      mtry
  )
  
  
  if (
    is.null(
      best_split
    )
  ) {
    
    return(
      list(
        terminal = TRUE,
        prediction =
          prediction,
        n =
          n_node
      )
    )
  }
  
  
  left_tree <- grow_rule_tree(
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
    mtry =
      mtry,
    current_depth =
      current_depth + 1
  )
  
  
  right_tree <- grow_rule_tree(
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
    mtry =
      mtry,
    current_depth =
      current_depth + 1
  )
  
  
  list(
    terminal = FALSE,
    prediction =
      prediction,
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
# GENERATE TREE ENSEMBLE
# ==============================================================================


generate_tree_ensemble <- function(
    X,
    y,
    number_trees = 150,
    sample_fraction = 0.70,
    minimum_leaf_size = 10,
    minimum_depth = 2,
    maximum_depth = 4,
    maximum_splits = 20,
    mtry = NULL,
    seed = 123
) {
  
  X <- as.matrix(X)
  
  
  n <- nrow(X)
  
  
  trees <- vector(
    "list",
    number_trees
  )
  
  
  set.seed(
    seed
  )
  
  
  for (
    tree_index in seq_len(
      number_trees
    )
  ) {
    
    sample_indices <- sample(
      seq_len(n),
      size =
        floor(
          sample_fraction *
            n
        ),
      replace = FALSE
    )
    
    
    depth <- sample(
      minimum_depth:maximum_depth,
      size = 1
    )
    
    
    trees[[tree_index]] <- grow_rule_tree(
      X[
        sample_indices,
        ,
        drop = FALSE
      ],
      y[
        sample_indices
      ],
      minimum_leaf_size =
        minimum_leaf_size,
      maximum_depth =
        depth,
      maximum_splits =
        maximum_splits,
      mtry =
        mtry
    )
  }
  
  
  trees
}


# ==============================================================================
# PART VIII
#
# BUILD TREE LIBRARY
# ==============================================================================


trees <- generate_tree_ensemble(
  X_train,
  y_train,
  number_trees = 200,
  sample_fraction = 0.70,
  minimum_leaf_size = 10,
  minimum_depth = 2,
  maximum_depth = 4,
  maximum_splits = 20,
  mtry = 3,
  seed = 123
)


length(
  trees
)


# ==============================================================================
# PART IX
#
# REPRESENT A RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# A rule is represented by a list of conditions.
#
# Example:
#
#       X1 <= 0.5
#       X3 > -0.2
#
# becomes:
#
#       list(
#           list(variable=1, operator="<=", value=0.5),
#           list(variable=3, operator=">", value=-0.2)
#       )
# ------------------------------------------------------------------------------


# ==============================================================================
# PART X
#
# APPLY A RULE
# ==============================================================================


apply_rule <- function(
    X,
    conditions
) {
  
  X <- as.matrix(X)
  
  
  result <- rep(
    TRUE,
    nrow(X)
  )
  
  
  if (
    length(conditions) ==
    0
  ) {
    
    return(
      as.numeric(
        result
      )
    )
  }
  
  
  for (
    condition in conditions
  ) {
    
    variable <- condition$variable
    
    value <- condition$value
    
    
    if (
      condition$operator ==
      "<="
    ) {
      
      result <- result &
        (
          X[
            ,
            variable
          ] <=
            value
        )
      
    } else {
      
      result <- result &
        (
          X[
            ,
            variable
          ] >
            value
        )
    }
  }
  
  
  as.numeric(
    result
  )
}


# ==============================================================================
# PART XI
#
# RULE DESCRIPTION
# ==============================================================================


rule_to_text <- function(
    conditions,
    variable_names
) {
  
  pieces <- character(
    length(
      conditions
    )
  )
  
  
  for (
    i in seq_along(
      conditions
    )
  ) {
    
    condition <- conditions[[i]]
    
    
    pieces[i] <- paste0(
      variable_names[
        condition$variable
      ],
      " ",
      condition$operator,
      " ",
      round(
        condition$value,
        3
      )
    )
  }
  
  
  paste(
    pieces,
    collapse = " & "
  )
}


# ==============================================================================
# PART XII
#
# EXTRACT RULES FROM A TREE
# ==============================================================================


# ------------------------------------------------------------------------------
# Every non-root node corresponds to a rule.
#
# Example:
#
#                 X1 <= 0
#                 /      \
#              left      right
#
# Left child corresponds to:
#
#       I(X1 <= 0)
#
#
# If the next split is X2 > 1, the descendant rule becomes:
#
#       I(X1 <= 0 AND X2 > 1)
#
#
# We extract both internal and terminal node rules.
# ------------------------------------------------------------------------------

extract_rules_from_tree <- function(
    tree,
    current_conditions = list(),
    include_internal = TRUE
) {
  
  rules <- list()
  
  
  recursive_extract <- function(
    node,
    conditions,
    is_root = FALSE
  ) {
    
    if (
      !is_root
    ) {
      
      if (
        include_internal ||
        node$terminal
      ) {
        
        rules[[length(rules) + 1]] <<-
          conditions
      }
    }
    
    
    if (
      node$terminal
    ) {
      
      return(
        invisible(NULL)
      )
    }
    
    
    left_condition <- list(
      variable =
        node$variable,
      operator =
        "<=",
      value =
        node$split
    )
    
    
    right_condition <- list(
      variable =
        node$variable,
      operator =
        ">",
      value =
        node$split
    )
    
    
    recursive_extract(
      node$left,
      c(
        conditions,
        list(
          left_condition
        )
      ),
      FALSE
    )
    
    
    recursive_extract(
      node$right,
      c(
        conditions,
        list(
          right_condition
        )
      ),
      FALSE
    )
    
    
    invisible(NULL)
  }
  
  
  recursive_extract(
    tree,
    current_conditions,
    TRUE
  )
  
  
  rules
}


# ==============================================================================
# PART XIII
#
# EXTRACT RULES FROM ALL TREES
# ==============================================================================


all_rules <- list()


for (
  tree_index in seq_along(
    trees
  )
) {
  
  tree_rules <- extract_rules_from_tree(
    trees[[tree_index]]
  )
  
  
  all_rules <- c(
    all_rules,
    tree_rules
  )
}


length(
  all_rules
)


# ==============================================================================
# PART XIV
#
# RULE CANONICALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# Different trees may generate identical rules.
#
# We need a stable textual representation so duplicate rules can be removed.
# ------------------------------------------------------------------------------

canonical_rule <- function(
    conditions
) {
  
  pieces <- character(
    length(
      conditions
    )
  )
  
  
  for (
    i in seq_along(
      conditions
    )
  ) {
    
    condition <- conditions[[i]]
    
    
    pieces[i] <- paste(
      condition$variable,
      condition$operator,
      format(
        condition$value,
        digits = 15
      ),
      sep = ":"
    )
  }
  
  
  paste(
    pieces,
    collapse = "|"
  )
}


rule_keys <- vapply(
  all_rules,
  canonical_rule,
  character(1)
)


unique_indices <- !duplicated(
  rule_keys
)


rules <- all_rules[
  unique_indices
]


length(
  rules
)


# ==============================================================================
# PART XV
#
# RULE MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# Each rule becomes a binary predictor:
#
#       R_ik = r_k(x_i)
#
# ------------------------------------------------------------------------------

build_rule_matrix <- function(
    X,
    rules
) {
  
  X <- as.matrix(X)
  
  
  R <- matrix(
    0,
    nrow =
      nrow(X),
    ncol =
      length(rules)
  )
  
  
  for (
    rule_index in seq_along(
      rules
    )
  ) {
    
    R[
      ,
      rule_index
    ] <- apply_rule(
      X,
      rules[[rule_index]]
    )
  }
  
  
  R
}


R_train <- build_rule_matrix(
  X_train,
  rules
)


R_test <- build_rule_matrix(
  X_test,
  rules
)


dim(
  R_train
)


# ==============================================================================
# PART XVI
#
# RULE SUPPORT
# ==============================================================================


# ------------------------------------------------------------------------------
# Support:
#
#       s_k = mean(r_k(x))
#
# It measures the fraction of observations satisfying the rule.
# ------------------------------------------------------------------------------

rule_support <- colMeans(
  R_train
)


summary(
  rule_support
)


hist(
  rule_support,
  breaks = 30,
  xlab = "Rule Support",
  main = "Distribution of Rule Support"
)


# ==============================================================================
# PART XVII
#
# FILTER DEGENERATE RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# Rules that almost never or almost always occur contain little information.
# ------------------------------------------------------------------------------

minimum_support <- 0.025

maximum_support <- 0.975


keep_rules <- (
  rule_support >=
    minimum_support
) &
  (
    rule_support <=
      maximum_support
  )


rules <- rules[
  keep_rules
]


R_train <- R_train[
  ,
  keep_rules,
  drop = FALSE
]


R_test <- R_test[
  ,
  keep_rules,
  drop = FALSE
]


rule_support <- colMeans(
  R_train
)


length(
  rules
)


# ==============================================================================
# PART XVIII
#
# PRINT SAMPLE RULES
# ==============================================================================


number_to_print <- min(
  20,
  length(rules)
)


for (
  i in seq_len(
    number_to_print
  )
) {
  
  cat(
    "Rule",
    i,
    ":",
    rule_to_text(
      rules[[i]],
      colnames(X_train)
    ),
    "\n"
  )
}


# ==============================================================================
# PART XIX
#
# STANDARDIZE LINEAR TERMS
# ==============================================================================


# ------------------------------------------------------------------------------
# Rule ensembles can contain:
#
#   1. original linear predictor terms;
#   2. tree-generated rules.
#
# Including linear terms lets simple additive relationships be represented
# efficiently without requiring many threshold rules.
# ------------------------------------------------------------------------------

x_means <- colMeans(
  X_train
)


x_sds <- apply(
  X_train,
  2,
  sd
)


x_sds[
  x_sds <
    1e-12
] <- 1


X_train_standardized <- sweep(
  X_train,
  2,
  x_means,
  "-"
)


X_train_standardized <- sweep(
  X_train_standardized,
  2,
  x_sds,
  "/"
)


X_test_standardized <- sweep(
  X_test,
  2,
  x_means,
  "-"
)


X_test_standardized <- sweep(
  X_test_standardized,
  2,
  x_sds,
  "/"
)


# ==============================================================================
# PART XX
#
# STANDARDIZE RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# Binary rule variance:
#
#       Var(R_k) = s_k (1-s_k)
#
# Rules with very different support naturally have different scales.
#
# We standardize them before Lasso so penalty strength is comparable.
# ------------------------------------------------------------------------------

rule_means <- colMeans(
  R_train
)


rule_sds <- apply(
  R_train,
  2,
  sd
)


rule_sds[
  rule_sds <
    1e-12
] <- 1


R_train_standardized <- sweep(
  R_train,
  2,
  rule_means,
  "-"
)


R_train_standardized <- sweep(
  R_train_standardized,
  2,
  rule_sds,
  "/"
)


R_test_standardized <- sweep(
  R_test,
  2,
  rule_means,
  "-"
)


R_test_standardized <- sweep(
  R_test_standardized,
  2,
  rule_sds,
  "/"
)


# ==============================================================================
# PART XXI
#
# DESIGN MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# Combine:
#
#       [linear terms | rule terms]
# ------------------------------------------------------------------------------

Z_train <- cbind(
  X_train_standardized,
  R_train_standardized
)


Z_test <- cbind(
  X_test_standardized,
  R_test_standardized
)


number_linear_terms <- ncol(
  X_train_standardized
)


number_rule_terms <- ncol(
  R_train_standardized
)


cat(
  "Linear terms:",
  number_linear_terms,
  "\n"
)


cat(
  "Rule terms:",
  number_rule_terms,
  "\n"
)


cat(
  "Total candidate features:",
  ncol(Z_train),
  "\n"
)


# ==============================================================================
# PART XXII
#
# CENTER RESPONSE
# ==============================================================================


y_mean <- mean(
  y_train
)


y_train_centered <- y_train -
  y_mean


# ==============================================================================
# PART XXIII
#
# SOFT THRESHOLDING
# ==============================================================================


soft_threshold <- function(
    z,
    gamma
) {
  
  sign(z) *
    pmax(
      abs(z) -
        gamma,
      0
    )
}


# ==============================================================================
# PART XXIV
#
# MANUAL LASSO COORDINATE DESCENT
# ==============================================================================


# ------------------------------------------------------------------------------
# Objective:
#
#       (1/(2n)) ||y - Z theta||^2
#
#       +
#
#       lambda ||theta||_1
# ------------------------------------------------------------------------------

lasso_fit <- function(
    Z,
    y,
    lambda,
    beta_initial = NULL,
    tolerance = 1e-7,
    maximum_iterations = 5000
) {
  
  Z <- as.matrix(Z)
  
  
  n <- nrow(Z)
  
  q <- ncol(Z)
  
  
  if (
    is.null(
      beta_initial
    )
  ) {
    
    beta <- numeric(
      q
    )
    
  } else {
    
    beta <- beta_initial
  }
  
  
  column_squared_mean <- colSums(
    Z^2
  ) /
    n
  
  
  fitted <- as.numeric(
    Z %*%
      beta
  )
  
  
  for (
    iteration in seq_len(
      maximum_iterations
    )
  ) {
    
    beta_old <- beta
    
    
    for (
      j in seq_len(
        q
      )
    ) {
      
      # Remove old contribution from fitted values.
      
      fitted_without_j <- fitted -
        Z[
          ,
          j
        ] *
        beta[j]
      
      
      residual_without_j <- y -
        fitted_without_j
      
      
      rho_j <- sum(
        Z[
          ,
          j
        ] *
          residual_without_j
      ) /
        n
      
      
      if (
        column_squared_mean[j] >
        1e-12
      ) {
        
        beta_new <- soft_threshold(
          rho_j,
          lambda
        ) /
          column_squared_mean[j]
        
      } else {
        
        beta_new <- 0
      }
      
      
      fitted <- fitted_without_j +
        Z[
          ,
          j
        ] *
        beta_new
      
      
      beta[j] <- beta_new
    }
    
    
    if (
      max(
        abs(
          beta -
          beta_old
        )
      ) <
      tolerance
    ) {
      
      break
    }
  }
  
  
  list(
    beta =
      beta,
    iterations =
      iteration,
    converged =
      iteration <
      maximum_iterations
  )
}


# ==============================================================================
# PART XXV
#
# LAMBDA MAX
# ==============================================================================


lambda_max <- max(
  abs(
    as.numeric(
      t(Z_train) %*%
        y_train_centered
    )
  )
) /
  n_train


lambda_max


lambda_grid <- exp(
  seq(
    log(
      lambda_max
    ),
    log(
      lambda_max *
        0.005
    ),
    length.out = 80
  )
)


# ==============================================================================
# PART XXVI
#
# REGULARIZATION PATH
# ==============================================================================


coefficient_path <- matrix(
  0,
  nrow =
    length(lambda_grid),
  ncol =
    ncol(Z_train)
)


beta_previous <- numeric(
  ncol(Z_train)
)


for (
  lambda_index in seq_along(
    lambda_grid
  )
) {
  
  fit <- lasso_fit(
    Z_train,
    y_train_centered,
    lambda =
      lambda_grid[
        lambda_index
      ],
    beta_initial =
      beta_previous
  )
  
  
  beta_previous <- fit$beta
  
  
  coefficient_path[
    lambda_index,
  ] <- beta_previous
}


# ==============================================================================
# PART XXVII
#
# SPARSITY PATH
# ==============================================================================


number_nonzero <- apply(
  coefficient_path,
  1,
  function(beta) {
    
    sum(
      abs(beta) >
        1e-8
    )
  }
)


plot(
  log(
    lambda_grid
  ),
  number_nonzero,
  type = "s",
  xlab = "log(lambda)",
  ylab = "Number of Nonzero Terms",
  main = "Rule Ensemble Sparsity Path"
)


# ==============================================================================
# PART XXVIII
#
# LINEAR VS RULE TERM SELECTION
# ==============================================================================


linear_nonzero <- apply(
  coefficient_path[
    ,
    seq_len(
      number_linear_terms
    ),
    drop = FALSE
  ],
  1,
  function(beta) {
    
    sum(
      abs(beta) >
        1e-8
    )
  }
)


rule_columns <- number_linear_terms +
  seq_len(
    number_rule_terms
  )


rule_nonzero <- apply(
  coefficient_path[
    ,
    rule_columns,
    drop = FALSE
  ],
  1,
  function(beta) {
    
    sum(
      abs(beta) >
        1e-8
    )
  }
)


plot(
  log(
    lambda_grid
  ),
  linear_nonzero,
  type = "l",
  lwd = 2,
  ylim = range(
    c(
      linear_nonzero,
      rule_nonzero
    )
  ),
  xlab = "log(lambda)",
  ylab = "Number Selected",
  main = "Linear Terms vs Rules"
)


lines(
  log(
    lambda_grid
  ),
  rule_nonzero,
  lwd = 2,
  lty = 2
)


legend(
  "topright",
  legend = c(
    "Linear Terms",
    "Rules"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART XXIX
#
# CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# IMPORTANT:
#
# For a completely leakage-free production implementation, the entire rule
# generation stage should occur independently inside each CV training fold.
#
# That would require regenerating hundreds of trees/rules for every fold and
# lambda combination.
#
# To keep this educational script computationally manageable, we:
#
#   1. generate the rule library using the training set;
#   2. cross-validate the sparse linear combination of those candidate rules.
#
# This is therefore an approximation to fully nested RuleFit tuning.
# ------------------------------------------------------------------------------


make_folds <- function(
    n,
    number_folds = 5,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  sample(
    rep(
      seq_len(
        number_folds
      ),
      length.out = n
    )
  )
}


folds <- make_folds(
  n_train,
  number_folds = 5,
  seed = 123
)


cv_mse <- matrix(
  NA_real_,
  nrow = 5,
  ncol =
    length(lambda_grid)
)


for (
  fold in seq_len(5)
) {
  
  validation_indices <- which(
    folds ==
      fold
  )
  
  
  training_indices <- which(
    folds !=
      fold
  )
  
  
  Z_fold_train_raw <- Z_train[
    training_indices,
    ,
    drop = FALSE
  ]
  
  
  Z_fold_validation_raw <- Z_train[
    validation_indices,
    ,
    drop = FALSE
  ]
  
  
  y_fold_train_raw <- y_train[
    training_indices
  ]
  
  
  y_fold_validation <- y_train[
    validation_indices
  ]
  
  
  # --------------------------------------------------------------------------
  # Re-standardize candidate features within fold.
  #
  # Z_train was already standardized globally, but we re-center and re-scale
  # within the fold to prevent the Lasso penalty from using validation-fold
  # means and scales.
  # --------------------------------------------------------------------------
  
  fold_means <- colMeans(
    Z_fold_train_raw
  )
  
  
  fold_sds <- apply(
    Z_fold_train_raw,
    2,
    sd
  )
  
  
  fold_sds[
    fold_sds <
      1e-12
  ] <- 1
  
  
  Z_fold_train <- sweep(
    Z_fold_train_raw,
    2,
    fold_means,
    "-"
  )
  
  
  Z_fold_train <- sweep(
    Z_fold_train,
    2,
    fold_sds,
    "/"
  )
  
  
  Z_fold_validation <- sweep(
    Z_fold_validation_raw,
    2,
    fold_means,
    "-"
  )
  
  
  Z_fold_validation <- sweep(
    Z_fold_validation,
    2,
    fold_sds,
    "/"
  )
  
  
  fold_y_mean <- mean(
    y_fold_train_raw
  )
  
  
  y_fold_train <- y_fold_train_raw -
    fold_y_mean
  
  
  beta_previous <- numeric(
    ncol(
      Z_fold_train
    )
  )
  
  
  for (
    lambda_index in seq_along(
      lambda_grid
    )
  ) {
    
    fit <- lasso_fit(
      Z_fold_train,
      y_fold_train,
      lambda =
        lambda_grid[
          lambda_index
        ],
      beta_initial =
        beta_previous
    )
    
    
    beta_previous <- fit$beta
    
    
    prediction <- fold_y_mean +
      as.numeric(
        Z_fold_validation %*%
          fit$beta
      )
    
    
    cv_mse[
      fold,
      lambda_index
    ] <- mean(
      (
        y_fold_validation -
          prediction
      )^2
    )
  }
}


mean_cv_mse <- colMeans(
  cv_mse
)


se_cv_mse <- apply(
  cv_mse,
  2,
  sd
) /
  sqrt(5)


# ==============================================================================
# PART XXX
#
# SELECT LAMBDA
# ==============================================================================


minimum_index <- which.min(
  mean_cv_mse
)


lambda_min <- lambda_grid[
  minimum_index
]


minimum_threshold <- mean_cv_mse[
  minimum_index
] +
  se_cv_mse[
    minimum_index
  ]


eligible <- which(
  mean_cv_mse <=
    minimum_threshold
)


# lambda_grid runs from large to small.
#
# First eligible lambda = strongest regularization satisfying one-SE rule.

one_se_index <- min(
  eligible
)


lambda_one_se <- lambda_grid[
  one_se_index
]


lambda_min


lambda_one_se


# ==============================================================================
# PART XXXI
#
# CV PLOT
# ==============================================================================


plot(
  log(
    lambda_grid
  ),
  mean_cv_mse,
  type = "l",
  lwd = 2,
  xlab = "log(lambda)",
  ylab = "Cross-Validated MSE",
  main = "Rule Ensemble Cross-Validation"
)


lines(
  log(
    lambda_grid
  ),
  mean_cv_mse +
    se_cv_mse,
  lty = 2
)


lines(
  log(
    lambda_grid
  ),
  mean_cv_mse -
    se_cv_mse,
  lty = 2
)


abline(
  v =
    log(
      lambda_min
    ),
  lty = 3
)


abline(
  v =
    log(
      lambda_one_se
    ),
  lty = 3
)


# ==============================================================================
# PART XXXII
#
# FINAL MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# Use one-SE model for a sparser, more interpretable rule ensemble.
# ------------------------------------------------------------------------------

final_fit <- lasso_fit(
  Z_train,
  y_train_centered,
  lambda =
    lambda_one_se
)


final_beta <- final_fit$beta


# ==============================================================================
# PART XXXIII
#
# TEST PREDICTIONS
# ==============================================================================


rule_ensemble_prediction <- y_mean +
  as.numeric(
    Z_test %*%
      final_beta
  )


regression_metrics(
  y_test,
  rule_ensemble_prediction
)


# ==============================================================================
# PART XXXIV
#
# SELECTED LINEAR TERMS
# ==============================================================================


linear_beta <- final_beta[
  seq_len(
    number_linear_terms
  )
]


selected_linear <- which(
  abs(
    linear_beta
  ) >
    1e-8
)


data.frame(
  Variable =
    colnames(X_train)[
      selected_linear
    ],
  Standardized_Coefficient =
    linear_beta[
      selected_linear
    ]
)


# ==============================================================================
# PART XXXV
#
# SELECTED RULES
# ==============================================================================


rule_beta <- final_beta[
  rule_columns
]


selected_rule_indices <- which(
  abs(
    rule_beta
  ) >
    1e-8
)


length(
  selected_rule_indices
)


# ------------------------------------------------------------------------------
# Rule Table
# ------------------------------------------------------------------------------

selected_rule_table <- data.frame(
  Rule_Index =
    selected_rule_indices,
  Rule =
    vapply(
      selected_rule_indices,
      function(index) {
        
        rule_to_text(
          rules[[index]],
          colnames(X_train)
        )
      },
      character(1)
    ),
  Support =
    rule_support[
      selected_rule_indices
    ],
  Standardized_Coefficient =
    rule_beta[
      selected_rule_indices
    ]
)


selected_rule_table <- selected_rule_table[
  order(
    -abs(
      selected_rule_table$Standardized_Coefficient
    )
  ),
]


head(
  selected_rule_table,
  20
)


# ==============================================================================
# PART XXXVI
#
# RULE COEFFICIENTS IN ORIGINAL BINARY-RULE SCALE
# ==============================================================================


# ------------------------------------------------------------------------------
# We fit:
#
#       alpha_standardized *
#
#       (r - mean(r)) / sd(r)
#
#
# Therefore coefficient multiplying the raw 0/1 rule is:
#
#       alpha_raw
#
#       =
#
#       alpha_standardized / sd(r)
# ------------------------------------------------------------------------------

rule_beta_raw <- rule_beta /
  rule_sds


selected_rule_table$Raw_Rule_Coefficient <-
  rule_beta_raw[
    selected_rule_indices
  ]


head(
  selected_rule_table,
  20
)


# ==============================================================================
# PART XXXVII
#
# RULE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# Friedman-style rule importance idea:
#
#       I_k
#
#       =
#
#       |alpha_k|
#
#       sqrt(
#
#           s_k (1-s_k)
#
#       )
#
#
# where alpha_k is the coefficient on the RAW binary rule.
#
# Since our rule features were standardized, this quantity is essentially the
# magnitude of the standardized coefficient, apart from sample-SD convention.
# ------------------------------------------------------------------------------

rule_importance <- abs(
  rule_beta_raw
) *
  sqrt(
    rule_support *
      (
        1 -
          rule_support
      )
  )


selected_rule_table$Importance <-
  rule_importance[
    selected_rule_indices
  ]


selected_rule_table <- selected_rule_table[
  order(
    -selected_rule_table$Importance
  ),
]


head(
  selected_rule_table,
  20
)


# ==============================================================================
# PART XXXVIII
#
# PLOT RULE IMPORTANCE
# ==============================================================================


number_plot_rules <- min(
  15,
  nrow(
    selected_rule_table
  )
)


if (
  number_plot_rules >
  0
) {
  
  top_rules <- selected_rule_table[
    seq_len(
      number_plot_rules
    ),
  ]
  
  
  barplot(
    rev(
      top_rules$Importance
    ),
    names.arg =
      rev(
        paste0(
          "Rule ",
          top_rules$Rule_Index
        )
      ),
    horiz = TRUE,
    las = 1,
    xlab = "Importance",
    main = "Most Important Rules"
  )
}


# ==============================================================================
# PART XXXIX
#
# VARIABLE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# A predictor can contribute through:
#
#   1. its linear term;
#   2. every selected rule containing that predictor.
#
# We distribute a rule's importance across variables appearing in that rule.
# ------------------------------------------------------------------------------

variables_in_rule <- function(
    conditions
) {
  
  unique(
    vapply(
      conditions,
      function(condition) {
        
        condition$variable
      },
      integer(1)
    )
  )
}


variable_importance <- numeric(
  p
)


# ------------------------------------------------------------------------------
# Linear contributions
# ------------------------------------------------------------------------------

for (
  j in seq_len(p)
) {
  
  variable_importance[j] <-
    abs(
      linear_beta[j]
    )
}


# ------------------------------------------------------------------------------
# Rule contributions
# ------------------------------------------------------------------------------

for (
  rule_index in seq_along(
    rules
  )
) {
  
  if (
    abs(
      rule_beta[rule_index]
    ) <=
    1e-8
  ) {
    
    next
  }
  
  
  variables <- variables_in_rule(
    rules[[rule_index]]
  )
  
  
  contribution <- rule_importance[
    rule_index
  ] /
    length(
      variables
    )
  
  
  variable_importance[
    variables
  ] <- variable_importance[
    variables
  ] +
    contribution
}


variable_importance_table <- data.frame(
  Variable =
    colnames(X_train),
  Importance =
    variable_importance
)


variable_importance_table <-
  variable_importance_table[
    order(
      -variable_importance_table$Importance
    ),
  ]


variable_importance_table


barplot(
  variable_importance_table$Importance,
  names.arg =
    variable_importance_table$Variable,
  las = 2,
  ylab = "Importance",
  main = "Rule Ensemble Variable Importance"
)


# ==============================================================================
# PART XL
#
# SINGLE RULE INTERPRETATION
# ==============================================================================


if (
  length(
    selected_rule_indices
  ) >
  0
) {
  
  best_rule_index <- selected_rule_table$Rule_Index[1]
  
  
  best_rule <- rules[[
    best_rule_index
  ]]
  
  
  best_rule_indicator <- apply_rule(
    X_train,
    best_rule
  )
  
  
  cat(
    "\nMost important rule:\n"
  )
  
  
  cat(
    rule_to_text(
      best_rule,
      colnames(X_train)
    ),
    "\n"
  )
  
  
  cat(
    "Support:",
    mean(
      best_rule_indicator
    ),
    "\n"
  )
  
  
  cat(
    "Mean Y when rule = 1:",
    mean(
      y_train[
        best_rule_indicator ==
          1
      ]
    ),
    "\n"
  )
  
  
  cat(
    "Mean Y when rule = 0:",
    mean(
      y_train[
        best_rule_indicator ==
          0
      ]
    ),
    "\n"
  )
}


# ==============================================================================
# PART XLI
#
# BASELINE: LINEAR REGRESSION
# ==============================================================================


linear_data_train <- data.frame(
  y =
    y_train,
  X_train
)


linear_data_test <- data.frame(
  X_test
)


linear_model <- lm(
  y ~ .,
  data =
    linear_data_train
)


linear_prediction <- predict(
  linear_model,
  newdata =
    linear_data_test
)


linear_metrics <- regression_metrics(
  y_test,
  linear_prediction
)


linear_metrics


# ==============================================================================
# PART XLII
#
# BASELINE: LASSO USING ONLY ORIGINAL LINEAR TERMS
# ==============================================================================


# ------------------------------------------------------------------------------
# Same sparse regression machinery, but without tree rules.
# ------------------------------------------------------------------------------

linear_lambda_max <- max(
  abs(
    as.numeric(
      t(
        X_train_standardized
      ) %*%
        y_train_centered
    )
  ) /
    n_train
)


linear_lambda <- 0.05 *
  linear_lambda_max


linear_lasso_fit <- lasso_fit(
  X_train_standardized,
  y_train_centered,
  lambda =
    linear_lambda
)


linear_lasso_prediction <- y_mean +
  as.numeric(
    X_test_standardized %*%
      linear_lasso_fit$beta
  )


regression_metrics(
  y_test,
  linear_lasso_prediction
)


# ==============================================================================
# PART XLIII
#
# BASELINE: TREE ENSEMBLE
# ==============================================================================


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


predict_tree <- function(
    tree,
    X
) {
  
  X <- as.matrix(X)
  
  
  prediction <- numeric(
    nrow(X)
  )
  
  
  for (
    i in seq_len(
      nrow(X)
    )
  ) {
    
    prediction[i] <- predict_tree_one(
      tree,
      X[
        i,
      ]
    )
  }
  
  
  prediction
}


predict_tree_ensemble <- function(
    trees,
    X
) {
  
  predictions <- matrix(
    0,
    nrow =
      nrow(X),
    ncol =
      length(trees)
  )
  
  
  for (
    tree_index in seq_along(
      trees
    )
  ) {
    
    predictions[
      ,
      tree_index
    ] <- predict_tree(
      trees[[tree_index]],
      X
    )
  }
  
  
  rowMeans(
    predictions
  )
}


tree_ensemble_prediction <- predict_tree_ensemble(
  trees,
  X_test
)


tree_ensemble_metrics <- regression_metrics(
  y_test,
  tree_ensemble_prediction
)


tree_ensemble_metrics


# ==============================================================================
# PART XLIV
#
# MODEL COMPARISON
# ==============================================================================


rule_metrics <- regression_metrics(
  y_test,
  rule_ensemble_prediction
)


linear_lasso_metrics <- regression_metrics(
  y_test,
  linear_lasso_prediction
)


comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Linear Lasso",
    "Tree Ensemble",
    "Rule Ensemble"
  ),
  MSE = c(
    linear_metrics["MSE"],
    linear_lasso_metrics["MSE"],
    tree_ensemble_metrics["MSE"],
    rule_metrics["MSE"]
  ),
  RMSE = c(
    linear_metrics["RMSE"],
    linear_lasso_metrics["RMSE"],
    tree_ensemble_metrics["RMSE"],
    rule_metrics["RMSE"]
  ),
  R2 = c(
    linear_metrics["R2"],
    linear_lasso_metrics["R2"],
    tree_ensemble_metrics["R2"],
    rule_metrics["R2"]
  )
)


comparison


# ==============================================================================
# PART XLV
#
# RULE COMPLEXITY
# ==============================================================================


# ------------------------------------------------------------------------------
# Number of conditions in each rule.
#
# A one-condition rule represents a main threshold effect.
#
# A multi-condition rule can encode an interaction.
# ------------------------------------------------------------------------------

rule_complexity <- vapply(
  rules,
  length,
  integer(1)
)


table(
  rule_complexity
)


hist(
  rule_complexity,
  breaks =
    seq(
      0.5,
      max(
        rule_complexity
      ) +
        0.5,
      by = 1
    ),
  xlab = "Number of Conditions",
  main = "Rule Complexity"
)


# ==============================================================================
# PART XLVI
#
# COMPLEXITY OF SELECTED RULES
# ==============================================================================


if (
  length(
    selected_rule_indices
  ) >
  0
) {
  
  selected_complexity <- rule_complexity[
    selected_rule_indices
  ]
  
  
  table(
    selected_complexity
  )
  
  
  mean(
    selected_complexity
  )
}


# ==============================================================================
# PART XLVII
#
# MAIN EFFECTS VS INTERACTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# Rules involving one distinct predictor are threshold main effects.
#
# Rules involving multiple distinct predictors represent interactions.
# ------------------------------------------------------------------------------

number_rule_variables <- vapply(
  rules,
  function(rule) {
    
    length(
      variables_in_rule(
        rule
      )
    )
  },
  integer(1)
)


table(
  number_rule_variables
)


if (
  length(
    selected_rule_indices
  ) >
  0
) {
  
  table(
    number_rule_variables[
      selected_rule_indices
    ]
  )
}


# ==============================================================================
# PART XLVIII
#
# PREDICTION CONTRIBUTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# Because the final model is linear in the expanded feature space, we can
# decompose each prediction into term-level contributions.
# ------------------------------------------------------------------------------

example_index <- 1


example_linear_contributions <-
  X_test_standardized[
    example_index,
  ] *
  linear_beta


example_rule_contributions <-
  R_test_standardized[
    example_index,
  ] *
  rule_beta


cat(
  "\nPrediction decomposition for test observation 1\n"
)


cat(
  "Intercept:",
  y_mean,
  "\n"
)


cat(
  "Linear contribution:",
  sum(
    example_linear_contributions
  ),
  "\n"
)


cat(
  "Rule contribution:",
  sum(
    example_rule_contributions
  ),
  "\n"
)


cat(
  "Final prediction:",
  rule_ensemble_prediction[
    example_index
  ],
  "\n"
)


# ==============================================================================
# PART XLIX
#
# TOP CONTRIBUTING RULES FOR ONE OBSERVATION
# ==============================================================================


example_contribution <- example_rule_contributions


top_contribution_indices <- order(
  abs(
    example_contribution
  ),
  decreasing = TRUE
)


top_contribution_indices <- top_contribution_indices[
  seq_len(
    min(
      10,
      length(
        top_contribution_indices
      )
    )
  )
]


example_rule_table <- data.frame(
  Rule =
    vapply(
      top_contribution_indices,
      function(index) {
        
        rule_to_text(
          rules[[index]],
          colnames(X_train)
        )
      },
      character(1)
    ),
  Active =
    R_test[
      example_index,
      top_contribution_indices
    ],
  Contribution =
    example_contribution[
      top_contribution_indices
    ]
)


example_rule_table


# ==============================================================================
# PART L
#
# EFFECT OF TREE DEPTH
# ==============================================================================


# ------------------------------------------------------------------------------
# Depth controls interaction complexity.
#
# depth 1:
#       single-condition threshold rules
#
# depth 2:
#       up to two-condition paths
#
# depth 3+:
#       increasingly complex interactions
#
#
# We demonstrate the size of the rule library generated by different depths.
# ------------------------------------------------------------------------------

depth_values <- 1:5


rule_count_by_depth <- numeric(
  length(
    depth_values
  )
)


set.seed(321)


for (
  depth_index in seq_along(
    depth_values
  )
) {
  
  temporary_trees <- generate_tree_ensemble(
    X_train,
    y_train,
    number_trees = 30,
    sample_fraction = 0.70,
    minimum_leaf_size = 10,
    minimum_depth =
      depth_values[
        depth_index
      ],
    maximum_depth =
      depth_values[
        depth_index
      ],
    maximum_splits = 15,
    mtry = 3,
    seed =
      100 +
      depth_index
  )
  
  
  temporary_rules <- list()
  
  
  for (
    tree_index in seq_along(
      temporary_trees
    )
  ) {
    
    temporary_rules <- c(
      temporary_rules,
      extract_rules_from_tree(
        temporary_trees[[tree_index]]
      )
    )
  }
  
  
  temporary_keys <- vapply(
    temporary_rules,
    canonical_rule,
    character(1)
  )
  
  
  rule_count_by_depth[
    depth_index
  ] <- length(
    unique(
      temporary_keys
    )
  )
}


data.frame(
  Depth =
    depth_values,
  Rules =
    rule_count_by_depth
)


plot(
  depth_values,
  rule_count_by_depth,
  type = "b",
  pch = 19,
  xlab = "Tree Depth",
  ylab = "Number of Unique Rules",
  main = "Tree Depth and Rule Library Size"
)


# ==============================================================================
# PART LI
#
# RULE SUPPORT AND IMPORTANCE
# ==============================================================================


plot(
  rule_support,
  rule_importance,
  pch = 19,
  cex = 0.5,
  xlab = "Rule Support",
  ylab = "Rule Importance",
  main = "Support vs Rule Importance"
)


# Rules that are almost always true or almost always false have little
# variability and therefore tend to have limited effective importance.


# ==============================================================================
# PART LII
#
# RULE CORRELATION
# ==============================================================================


# ------------------------------------------------------------------------------
# Tree-generated rules are often highly correlated.
#
# Example:
#
#       X1 <= 0.5
#
# and
#
#       X1 <= 0.6
#
# will usually produce very similar indicator vectors.
#
# Lasso must select among this large correlated dictionary.
# ------------------------------------------------------------------------------

if (
  ncol(
    R_train
  ) >
  1
) {
  
  number_correlation_rules <- min(
    100,
    ncol(
      R_train
    )
  )
  
  
  rule_correlation <- cor(
    R_train[
      ,
      seq_len(
        number_correlation_rules
      ),
      drop = FALSE
    ]
  )
  
  
  rule_correlation_values <- rule_correlation[
    upper.tri(
      rule_correlation
    )
  ]
  
  
  rule_correlation_values <- rule_correlation_values[
    is.finite(
      rule_correlation_values
    )
  ]
  
  
  hist(
    rule_correlation_values,
    breaks = 30,
    xlab = "Pairwise Rule Correlation",
    main = "Correlation Among Generated Rules"
  )
}


# ==============================================================================
# PART LIII
#
# TRUE STRUCTURE COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# The true simulation contains:
#
#   X1 > 0.5 AND X3 < 0
#
#   X2 < -0.75
#
#   X4 > 0 AND X5 > 0
#
#
# Search selected rule descriptions for related structures.
# ------------------------------------------------------------------------------

if (
  nrow(
    selected_rule_table
  ) >
  0
) {
  
  print(
    selected_rule_table[
      seq_len(
        min(
          25,
          nrow(
            selected_rule_table
          )
        )
      ),
      c(
        "Rule",
        "Support",
        "Raw_Rule_Coefficient",
        "Importance"
      )
    ]
  )
}


# ==============================================================================
# PART LIV
#
# OPTIONAL RULEFIT PACKAGE NOTE
# ==============================================================================


# A production RuleFit implementation usually contains substantially more
# machinery than this demonstration, including:
#
#   - carefully generated tree ensembles;
#   - controlled tree-size distributions;
#   - winsorized linear terms;
#   - standardized linear terms;
#   - efficient sparse rule matrices;
#   - optimized Lasso fitting;
#   - classification losses;
#   - interaction statistics;
#   - partial dependence / interpretability tools.
#
#
# Packages implementing RuleFit-style procedures vary across R ecosystems.
#
# The goal here is to expose the algorithm rather than hide it behind a package.


# ==============================================================================
# PART LV
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nRule Ensemble Summary\n"
)


cat(
  "---------------------\n"
)


cat(
  "Training observations:",
  n_train,
  "\n"
)


cat(
  "Original predictors:",
  p,
  "\n"
)


cat(
  "Trees generated:",
  length(
    trees
  ),
  "\n"
)


cat(
  "Candidate rules after filtering:",
  length(
    rules
  ),
  "\n"
)


cat(
  "Total candidate features:",
  ncol(
    Z_train
  ),
  "\n"
)


cat(
  "Selected lambda:",
  round(
    lambda_one_se,
    6
  ),
  "\n"
)


cat(
  "Selected linear terms:",
  length(
    selected_linear
  ),
  "\n"
)


cat(
  "Selected rules:",
  length(
    selected_rule_indices
  ),
  "\n"
)


cat(
  "Rule ensemble test MSE:",
  round(
    rule_metrics["MSE"],
    5
  ),
  "\n"
)


cat(
  "Rule ensemble test R2:",
  round(
    rule_metrics["R2"],
    5
  ),
  "\n"
)