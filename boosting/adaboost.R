# ==============================================================================
# AdaBoost
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - AdaBoost
#   - Binary classification
#   - Weak learners
#   - Classification stumps
#   - Observation weights
#   - Weighted classification error
#   - Sequential reweighting
#   - Additive classification
#   - Exponential loss
#   - Classification margins
#   - Learning rate / shrinkage
#
#
# Binary labels:
#
#   y_i in {-1, +1}
#
#
# AdaBoost constructs:
#
#   F_M(x) = sum_{m=1}^M alpha_m G_m(x)
#
# where:
#
#   G_m(x) in {-1, +1}
#
#
# Final classifier:
#
#   G(x) = sign(F_M(x))
#
#
# At each iteration:
#
#   1. Fit weak classifier using observation weights.
#
#   2. Calculate weighted error:
#
#        err_m =
#        sum_i w_i I(y_i != G_m(x_i))
#        --------------------------------
#                 sum_i w_i
#
#   3. Compute learner weight:
#
#        alpha_m =
#        1/2 log((1 - err_m) / err_m)
#
#   4. Update observation weights:
#
#        w_i <- w_i exp(-alpha_m y_i G_m(x_i))
#
#   5. Normalize weights.
#
#
# Misclassified observations therefore receive MORE weight.
# Correctly classified observations receive LESS weight.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE BINARY CLASSIFICATION DATA
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


# ------------------------------------------------------------------------------
# 2. Construct Nonlinear Class Boundary
# ------------------------------------------------------------------------------

# We deliberately use a nonlinear decision boundary so one classification
# stump will not be sufficient.


latent_score <- (
  1.5 * sin(x1) +
    0.8 * x2 -
    0.35 * x1 * x2
)


probability <- 1 / (
  1 +
    exp(
      -latent_score
    )
)


y_binary <- rbinom(
  n,
  size = 1,
  prob = probability
)


# AdaBoost convention:
#
#   Class 0 -> -1
#   Class 1 -> +1


y <- ifelse(
  y_binary == 1,
  1,
  -1
)


X <- cbind(
  x1 = x1,
  x2 = x2
)


# ------------------------------------------------------------------------------
# 3. Class Distribution
# ------------------------------------------------------------------------------

table(
  y
)


prop.table(
  table(
    y
  )
)


# ------------------------------------------------------------------------------
# 4. Plot Data
# ------------------------------------------------------------------------------

plot(
  x1,
  x2,
  pch = ifelse(
    y == 1,
    19,
    1
  ),
  xlab = "x1",
  ylab = "x2",
  main = "AdaBoost Classification Data"
)


legend(
  "topright",
  legend = c(
    "Class -1",
    "Class +1"
  ),
  pch = c(
    1,
    19
  )
)


# ==============================================================================
# PART II
#
# CLASSIFICATION STUMPS
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Candidate Split Points
# ------------------------------------------------------------------------------

candidate_splits <- function(
    x,
    maximum_splits = 100
) {
  
  unique_values <- sort(
    unique(
      x
    )
  )
  
  
  if (
    length(
      unique_values
    ) <
    2
  ) {
    
    return(
      numeric(0)
    )
  }
  
  
  splits <- (
    unique_values[
      -length(
        unique_values
      )
    ] +
      unique_values[
        -1
      ]
  ) / 2
  
  
  if (
    length(
      splits
    ) >
    maximum_splits
  ) {
    
    indices <- round(
      seq(
        1,
        length(
          splits
        ),
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
# 6. Classification Stump Prediction
# ------------------------------------------------------------------------------

# polarity = 1:
#
#   x <= split -> -1
#   x >  split -> +1
#
#
# polarity = -1:
#
#   x <= split -> +1
#   x >  split -> -1


predict_classification_stump <- function(
    stump,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  x <- X_new[
    ,
    stump$variable
  ]
  
  
  prediction <- ifelse(
    x <= stump$split,
    -1,
    1
  )
  
  
  prediction <-
    stump$polarity *
    prediction
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 7. Fit Weighted Classification Stump
# ------------------------------------------------------------------------------

fit_classification_stump <- function(
    X,
    y,
    weights = NULL,
    maximum_splits = 100
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
      weights
    )
  ) {
    
    weights <- rep(
      1 / n,
      n
    )
  }
  
  
  weights <- weights /
    sum(
      weights
    )
  
  
  best_error <- Inf
  
  best_variable <- NA_integer_
  
  best_split <- NA_real_
  
  best_polarity <- NA_integer_
  
  
  for (
    j in seq_len(p)
  ) {
    
    splits <- candidate_splits(
      X[
        ,
        j
      ],
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
      
      for (
        polarity in c(
          -1,
          1
        )
      ) {
        
        prediction <- ifelse(
          X[
            ,
            j
          ] <=
            split_value,
          -1,
          1
        )
        
        
        prediction <-
          polarity *
          prediction
        
        
        weighted_error <- sum(
          weights *
            (
              prediction !=
                y
            )
        )
        
        
        if (
          weighted_error <
          best_error
        ) {
          
          best_error <-
            weighted_error
          
          best_variable <-
            j
          
          best_split <-
            split_value
          
          best_polarity <-
            polarity
        }
      }
    }
  }
  
  
  if (
    is.na(
      best_variable
    )
  ) {
    
    stop(
      "No valid classification stump found."
    )
  }
  
  
  return(
    list(
      variable =
        best_variable,
      split =
        best_split,
      polarity =
        best_polarity,
      error =
        best_error
    )
  )
}


# ==============================================================================
# PART III
#
# FIT ONE WEAK CLASSIFIER
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Equal Initial Weights
# ------------------------------------------------------------------------------

initial_weights <- rep(
  1 / n,
  n
)


# ------------------------------------------------------------------------------
# 9. Fit First Stump
# ------------------------------------------------------------------------------

first_stump <- fit_classification_stump(
  X = X,
  y = y,
  weights =
    initial_weights
)


first_stump


cat(
  "First selected variable:",
  colnames(X)[
    first_stump$variable
  ],
  "\n"
)


# ------------------------------------------------------------------------------
# 10. First Stump Predictions
# ------------------------------------------------------------------------------

first_prediction <-
  predict_classification_stump(
    first_stump,
    X
  )


first_error <- mean(
  first_prediction !=
    y
)


first_accuracy <- mean(
  first_prediction ==
    y
)


first_error


first_accuracy


# ==============================================================================
# PART IV
#
# FIRST ADABOOST UPDATE BY HAND
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Calculate Alpha
# ------------------------------------------------------------------------------

epsilon <- 1e-12


first_weighted_error <- pmin(
  pmax(
    first_stump$error,
    epsilon
  ),
  1 - epsilon
)


first_alpha <- 0.5 *
  log(
    (
      1 -
        first_weighted_error
    ) /
      first_weighted_error
  )


first_alpha


# ------------------------------------------------------------------------------
# 12. Update Observation Weights
# ------------------------------------------------------------------------------

updated_weights <-
  initial_weights *
  exp(
    -first_alpha *
      y *
      first_prediction
  )


updated_weights <-
  updated_weights /
  sum(
    updated_weights
  )


# ------------------------------------------------------------------------------
# 13. Compare Correct and Incorrect Observation Weights
# ------------------------------------------------------------------------------

first_correct <- (
  first_prediction ==
    y
)


mean(
  updated_weights[
    first_correct
  ]
)


mean(
  updated_weights[
    !first_correct
  ]
)


# Misclassified observations should now have greater average weight.


# ------------------------------------------------------------------------------
# 14. Visualize Updated Weights
# ------------------------------------------------------------------------------

plot(
  x1,
  x2,
  pch = ifelse(
    y == 1,
    19,
    1
  ),
  cex = 0.5 +
    8 *
    updated_weights /
    max(
      updated_weights
    ),
  xlab = "x1",
  ylab = "x2",
  main = "Weights After First AdaBoost Iteration"
)


# ==============================================================================
# PART V
#
# MANUAL ADABOOST
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. AdaBoost Training Function
# ------------------------------------------------------------------------------

fit_adaboost <- function(
    X,
    y,
    M = 200,
    learning_rate = 1,
    maximum_splits = 100,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Initial observation weights
  # --------------------------------------------------------------------------
  
  weights <- rep(
    1 / n,
    n
  )
  
  
  # --------------------------------------------------------------------------
  # Ensemble score
  # --------------------------------------------------------------------------
  
  ensemble_score <- rep(
    0,
    n
  )
  
  
  # --------------------------------------------------------------------------
  # Storage
  # --------------------------------------------------------------------------
  
  learners <- vector(
    "list",
    M
  )
  
  
  alpha <- numeric(
    M
  )
  
  
  weighted_error <- numeric(
    M
  )
  
  
  training_error <- numeric(
    M
  )
  
  
  exponential_loss <- numeric(
    M
  )
  
  
  minimum_margin <- numeric(
    M
  )
  
  
  mean_margin <- numeric(
    M
  )
  
  
  weight_history <- matrix(
    NA_real_,
    nrow = n,
    ncol = M
  )
  
  
  actual_iterations <- 0
  
  
  # ==========================================================================
  # BOOSTING LOOP
  # ==========================================================================
  
  for (
    m in seq_len(M)
  ) {
    
    # ------------------------------------------------------------------------
    # Fit weighted weak learner
    # ------------------------------------------------------------------------
    
    stump <- fit_classification_stump(
      X = X,
      y = y,
      weights = weights,
      maximum_splits =
        maximum_splits
    )
    
    
    prediction <-
      predict_classification_stump(
        stump,
        X
      )
    
    
    error_m <- sum(
      weights *
        (
          prediction !=
            y
        )
    )
    
    
    # ------------------------------------------------------------------------
    # A weak classifier must perform better than random guessing.
    # ------------------------------------------------------------------------
    
    if (
      error_m >=
      0.5
    ) {
      
      if (verbose) {
        
        cat(
          "Stopped at iteration",
          m,
          "because weak learner error >= 0.5.\n"
        )
      }
      
      
      break
    }
    
    
    # ------------------------------------------------------------------------
    # Perfect weak classifier
    # ------------------------------------------------------------------------
    
    error_for_alpha <- pmin(
      pmax(
        error_m,
        1e-12
      ),
      1 - 1e-12
    )
    
    
    alpha_m <- 0.5 *
      log(
        (
          1 -
            error_for_alpha
        ) /
          error_for_alpha
      )
    
    
    # Optional shrinkage.
    
    alpha_m <-
      learning_rate *
      alpha_m
    
    
    # ------------------------------------------------------------------------
    # Update ensemble score
    # ------------------------------------------------------------------------
    
    ensemble_score <-
      ensemble_score +
      alpha_m *
      prediction
    
    
    # ------------------------------------------------------------------------
    # Update observation weights
    # ------------------------------------------------------------------------
    
    weights <-
      weights *
      exp(
        -alpha_m *
          y *
          prediction
      )
    
    
    weights <-
      weights /
      sum(
        weights
      )
    
    
    # ------------------------------------------------------------------------
    # Ensemble classification
    # ------------------------------------------------------------------------
    
    ensemble_prediction <- ifelse(
      ensemble_score >= 0,
      1,
      -1
    )
    
    
    # ------------------------------------------------------------------------
    # Margins
    # ------------------------------------------------------------------------
    
    margin <- y *
      ensemble_score
    
    
    # ------------------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------------------
    
    learners[[m]] <- stump
    
    alpha[m] <- alpha_m
    
    weighted_error[m] <- error_m
    
    training_error[m] <- mean(
      ensemble_prediction !=
        y
    )
    
    
    exponential_loss[m] <- mean(
      exp(
        -margin
      )
    )
    
    
    minimum_margin[m] <- min(
      margin
    )
    
    
    mean_margin[m] <- mean(
      margin
    )
    
    
    weight_history[
      ,
      m
    ] <- weights
    
    
    actual_iterations <- m
    
    
    if (verbose) {
      
      cat(
        "Iteration:",
        m,
        "| Variable:",
        colnames(X)[
          stump$variable
        ],
        "| Error:",
        round(
          error_m,
          4
        ),
        "| Alpha:",
        round(
          alpha_m,
          4
        ),
        "| Training Error:",
        round(
          training_error[m],
          4
        ),
        "\n"
      )
    }
    
    
    # If the stump perfectly classifies the weighted sample,
    # continuing is unnecessary.
    
    if (
      error_m <=
      1e-12
    ) {
      
      break
    }
  }
  
  
  if (
    actual_iterations ==
    0
  ) {
    
    stop(
      "AdaBoost failed to find a weak learner with error below 0.5."
    )
  }
  
  
  learners <- learners[
    seq_len(
      actual_iterations
    )
  ]
  
  
  alpha <- alpha[
    seq_len(
      actual_iterations
    )
  ]
  
  
  weighted_error <- weighted_error[
    seq_len(
      actual_iterations
    )
  ]
  
  
  training_error <- training_error[
    seq_len(
      actual_iterations
    )
  ]
  
  
  exponential_loss <- exponential_loss[
    seq_len(
      actual_iterations
    )
  ]
  
  
  minimum_margin <- minimum_margin[
    seq_len(
      actual_iterations
    )
  ]
  
  
  mean_margin <- mean_margin[
    seq_len(
      actual_iterations
    )
  ]
  
  
  weight_history <- weight_history[
    ,
    seq_len(
      actual_iterations
    ),
    drop = FALSE
  ]
  
  
  return(
    list(
      learners = learners,
      alpha = alpha,
      weighted_error =
        weighted_error,
      training_error =
        training_error,
      exponential_loss =
        exponential_loss,
      minimum_margin =
        minimum_margin,
      mean_margin =
        mean_margin,
      weight_history =
        weight_history,
      learning_rate =
        learning_rate,
      feature_names =
        colnames(X),
      iterations =
        actual_iterations
    )
  )
}


# ------------------------------------------------------------------------------
# 16. Fit AdaBoost
# ------------------------------------------------------------------------------

adaboost_model <- fit_adaboost(
  X = X,
  y = y,
  M = 300,
  learning_rate = 1,
  maximum_splits = 100,
  verbose = FALSE
)


adaboost_model$iterations


# ==============================================================================
# PART VI
#
# ADABOOST PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Ensemble Score
# ------------------------------------------------------------------------------

predict_adaboost_score <- function(
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
  
  
  score <- rep(
    0,
    nrow(
      X_new
    )
  )
  
  
  if (
    M ==
    0
  ) {
    
    return(
      score
    )
  }
  
  
  for (
    m in seq_len(M)
  ) {
    
    stump_prediction <-
      predict_classification_stump(
        model$learners[[m]],
        X_new
      )
    
    
    score <-
      score +
      model$alpha[m] *
      stump_prediction
  }
  
  
  return(
    score
  )
}


# ------------------------------------------------------------------------------
# 18. Class Prediction
# ------------------------------------------------------------------------------

predict_adaboost <- function(
    model,
    X_new,
    M = length(
      model$learners
    )
) {
  
  score <- predict_adaboost_score(
    model,
    X_new,
    M
  )
  
  
  ifelse(
    score >= 0,
    1,
    -1
  )
}


# ------------------------------------------------------------------------------
# 19. Training Predictions
# ------------------------------------------------------------------------------

training_prediction <- predict_adaboost(
  adaboost_model,
  X
)


training_accuracy <- mean(
  training_prediction ==
    y
)


training_error <- mean(
  training_prediction !=
    y
)


training_accuracy


training_error


# ==============================================================================
# PART VII
#
# BOOSTING PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Training Classification Error
# ------------------------------------------------------------------------------

plot(
  seq_along(
    adaboost_model$training_error
  ),
  adaboost_model$training_error,
  type = "l",
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Classification Error",
  main = "AdaBoost Training Error"
)


# ------------------------------------------------------------------------------
# 21. Weak-Learner Error
# ------------------------------------------------------------------------------

plot(
  seq_along(
    adaboost_model$weighted_error
  ),
  adaboost_model$weighted_error,
  type = "l",
  lwd = 2,
  ylim = c(
    0,
    0.5
  ),
  xlab = "AdaBoost Iteration",
  ylab = "Weighted Weak-Learner Error",
  main = "Weak Learner Error"
)


abline(
  h = 0.5,
  lty = 2
)


# ------------------------------------------------------------------------------
# 22. Learner Alpha
# ------------------------------------------------------------------------------

plot(
  seq_along(
    adaboost_model$alpha
  ),
  adaboost_model$alpha,
  type = "h",
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Alpha",
  main = "Weak Learner Contributions"
)


# ==============================================================================
# PART VIII
#
# EXPONENTIAL LOSS
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Exponential Loss Path
# ------------------------------------------------------------------------------

plot(
  seq_along(
    adaboost_model$exponential_loss
  ),
  adaboost_model$exponential_loss,
  type = "l",
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Exponential Loss",
  main = "AdaBoost Exponential Loss"
)


# ==============================================================================
# PART IX
#
# CLASSIFICATION MARGINS
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Final Margins
# ------------------------------------------------------------------------------

final_score <- predict_adaboost_score(
  adaboost_model,
  X
)


final_margin <- y *
  final_score


summary(
  final_margin
)


# ------------------------------------------------------------------------------
# 25. Margin Interpretation
# ------------------------------------------------------------------------------

# margin > 0:
#
#   correctly classified
#
# margin < 0:
#
#   incorrectly classified
#
# large positive margin:
#
#   confident correct classification


hist(
  final_margin,
  breaks = 40,
  xlab = "y * F(x)",
  main = "AdaBoost Classification Margins"
)


abline(
  v = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 26. Mean Margin Across Iterations
# ------------------------------------------------------------------------------

plot(
  seq_along(
    adaboost_model$mean_margin
  ),
  adaboost_model$mean_margin,
  type = "l",
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Mean Margin",
  main = "Mean Classification Margin"
)


# ==============================================================================
# PART X
#
# HOW OBSERVATION WEIGHTS CHANGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Maximum Observation Weight
# ------------------------------------------------------------------------------

maximum_weight <- apply(
  adaboost_model$weight_history,
  2,
  max
)


plot(
  seq_along(
    maximum_weight
  ),
  maximum_weight,
  type = "l",
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Maximum Observation Weight",
  main = "Concentration of AdaBoost Weights"
)


# ------------------------------------------------------------------------------
# 28. Effective Sample Size of Weights
# ------------------------------------------------------------------------------

# For normalized weights:
#
#   ESS = 1 / sum(w_i^2)
#
# Equal weights produce ESS = n.
#
# Concentrated weights produce smaller ESS.


weight_ESS <- apply(
  adaboost_model$weight_history,
  2,
  function(w) {
    
    1 /
      sum(
        w^2
      )
  }
)


plot(
  seq_along(
    weight_ESS
  ),
  weight_ESS,
  type = "l",
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Effective Sample Size",
  main = "Effective Sample Size of AdaBoost Weights"
)


abline(
  h = n,
  lty = 2
)


# ------------------------------------------------------------------------------
# 29. Highest-Weight Observations
# ------------------------------------------------------------------------------

final_weights <-
  adaboost_model$weight_history[
    ,
    ncol(
      adaboost_model$weight_history
    )
  ]


highest_weight_indices <- order(
  final_weights,
  decreasing = TRUE
)[
  1:10
]


data.frame(
  Index =
    highest_weight_indices,
  x1 =
    x1[
      highest_weight_indices
    ],
  x2 =
    x2[
      highest_weight_indices
    ],
  y =
    y[
      highest_weight_indices
    ],
  Weight =
    final_weights[
      highest_weight_indices
    ],
  Margin =
    final_margin[
      highest_weight_indices
    ]
)


# ==============================================================================
# PART XI
#
# DECISION BOUNDARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 150


grid_x1 <- seq(
  min(
    x1
  ),
  max(
    x1
  ),
  length.out = grid_size
)


grid_x2 <- seq(
  min(
    x2
  ),
  max(
    x2
  ),
  length.out = grid_size
)


prediction_grid <- expand.grid(
  x1 = grid_x1,
  x2 = grid_x2
)


X_grid <- as.matrix(
  prediction_grid
)


grid_score <- predict_adaboost_score(
  adaboost_model,
  X_grid
)


score_matrix <- matrix(
  grid_score,
  nrow = grid_size,
  ncol = grid_size
)


# ------------------------------------------------------------------------------
# 31. Plot Decision Function
# ------------------------------------------------------------------------------

image(
  grid_x1,
  grid_x2,
  score_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "AdaBoost Decision Function"
)


contour(
  grid_x1,
  grid_x2,
  score_matrix,
  levels = 0,
  add = TRUE,
  lwd = 3
)


points(
  x1,
  x2,
  pch = ifelse(
    y == 1,
    19,
    1
  )
)


# ==============================================================================
# PART XII
#
# EVOLUTION OF THE DECISION BOUNDARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Compare Different Numbers of Weak Learners
# ------------------------------------------------------------------------------

M_values <- c(
  1,
  5,
  20,
  100,
  min(
    300,
    adaboost_model$iterations
  )
)


M_values <- unique(
  M_values[
    M_values <=
      adaboost_model$iterations
  ]
)


for (
  M in M_values
) {
  
  grid_score_M <- predict_adaboost_score(
    adaboost_model,
    X_grid,
    M = M
  )
  
  
  score_matrix_M <- matrix(
    grid_score_M,
    nrow = grid_size,
    ncol = grid_size
  )
  
  
  image(
    grid_x1,
    grid_x2,
    score_matrix_M,
    xlab = "x1",
    ylab = "x2",
    main = paste(
      "AdaBoost Decision Function: M =",
      M
    )
  )
  
  
  contour(
    grid_x1,
    grid_x2,
    score_matrix_M,
    levels = 0,
    add = TRUE,
    lwd = 3
  )
  
  
  points(
    x1,
    x2,
    pch = ifelse(
      y == 1,
      19,
      1
    ),
    cex = 0.5
  )
}


# ==============================================================================
# PART XIII
#
# VARIABLE USAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Variables Selected by Stumps
# ------------------------------------------------------------------------------

selected_variables <- sapply(
  adaboost_model$learners,
  function(stump) {
    
    stump$variable
  }
)


variable_names <- adaboost_model$feature_names[
  selected_variables
]


table(
  variable_names
)


# ------------------------------------------------------------------------------
# 34. Alpha-Weighted Variable Importance
# ------------------------------------------------------------------------------

variable_importance <- numeric(
  ncol(X)
)


for (
  m in seq_along(
    adaboost_model$learners
  )
) {
  
  j <-
    adaboost_model$learners[[m]]$variable
  
  
  variable_importance[j] <-
    variable_importance[j] +
    abs(
      adaboost_model$alpha[m]
    )
}


names(
  variable_importance
) <- colnames(X)


variable_importance <-
  variable_importance /
  sum(
    variable_importance
  )


variable_importance


barplot(
  variable_importance,
  ylab = "Relative Importance",
  main = "AdaBoost Variable Importance"
)


# ==============================================================================
# PART XIV
#
# TRAIN-TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Split Data
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
# 36. Fit Training Model
# ------------------------------------------------------------------------------

adaboost_train <- fit_adaboost(
  X = X_train,
  y = y_train,
  M = 500,
  learning_rate = 1,
  maximum_splits = 100
)


# ==============================================================================
# PART XV
#
# TRAIN AND TEST ERROR PATHS
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Evaluate Every Iteration
# ------------------------------------------------------------------------------

M_max <- adaboost_train$iterations


train_error_path <- numeric(
  M_max
)


test_error_path <- numeric(
  M_max
)


train_exponential_loss <- numeric(
  M_max
)


test_exponential_loss <- numeric(
  M_max
)


for (
  M in seq_len(
    M_max
  )
) {
  
  train_score <- predict_adaboost_score(
    adaboost_train,
    X_train,
    M
  )
  
  
  test_score <- predict_adaboost_score(
    adaboost_train,
    X_test,
    M
  )
  
  
  train_prediction <- ifelse(
    train_score >= 0,
    1,
    -1
  )
  
  
  test_prediction <- ifelse(
    test_score >= 0,
    1,
    -1
  )
  
  
  train_error_path[M] <- mean(
    train_prediction !=
      y_train
  )
  
  
  test_error_path[M] <- mean(
    test_prediction !=
      y_test
  )
  
  
  train_exponential_loss[M] <- mean(
    exp(
      -y_train *
        train_score
    )
  )
  
  
  test_exponential_loss[M] <- mean(
    exp(
      -y_test *
        test_score
    )
  )
}


# ------------------------------------------------------------------------------
# 38. Plot Classification Error
# ------------------------------------------------------------------------------

matplot(
  seq_len(
    M_max
  ),
  cbind(
    train_error_path,
    test_error_path
  ),
  type = "l",
  lty = c(
    1,
    2
  ),
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Classification Error",
  main = "AdaBoost Train and Test Error"
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
# PART XVI
#
# ORACLE TEST ITERATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Best Test Iteration
# ------------------------------------------------------------------------------

best_test_iteration <- which.min(
  test_error_path
)


best_test_iteration


test_error_path[
  best_test_iteration
]


# IMPORTANT:
#
# This is only a diagnostic.
#
# The test set should not be used to select M.
# Cross-validation is used below.


# ==============================================================================
# PART XVII
#
# CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. K-Fold Cross Validation
# ------------------------------------------------------------------------------

adaboost_cv <- function(
    X,
    y,
    M = 300,
    k = 5,
    learning_rate = 1,
    maximum_splits = 75,
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
  
  
  fold_error <- matrix(
    NA_real_,
    nrow = M,
    ncol = k
  )
  
  
  for (
    fold in seq_len(k)
  ) {
    
    train_indices <- which(
      fold_id != fold
    )
    
    
    validation_indices <- which(
      fold_id == fold
    )
    
    
    model_fold <- fit_adaboost(
      X = X[
        train_indices,
        ,
        drop = FALSE
      ],
      y = y[
        train_indices
      ],
      M = M,
      learning_rate =
        learning_rate,
      maximum_splits =
        maximum_splits
    )
    
    
    fold_iterations <-
      model_fold$iterations
    
    
    validation_score <- rep(
      0,
      length(
        validation_indices
      )
    )
    
    
    for (
      m in seq_len(
        fold_iterations
      )
    ) {
      
      validation_score <-
        validation_score +
        model_fold$alpha[m] *
        predict_classification_stump(
          model_fold$learners[[m]],
          X[
            validation_indices,
            ,
            drop = FALSE
          ]
        )
      
      
      validation_prediction <- ifelse(
        validation_score >= 0,
        1,
        -1
      )
      
      
      fold_error[
        m,
        fold
      ] <- mean(
        validation_prediction !=
          y[
            validation_indices
          ]
      )
    }
    
    
    # If fitting terminated before M, carry forward the final model.
    
    if (
      fold_iterations <
      M
    ) {
      
      final_error <- fold_error[
        fold_iterations,
        fold
      ]
      
      
      fold_error[
        seq.int(
          fold_iterations + 1,
          M
        ),
        fold
      ] <- final_error
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
    list(
      mean = mean_error,
      SE = SE_error,
      fold_error = fold_error
    )
  )
}


# ------------------------------------------------------------------------------
# 41. Run Cross Validation
# ------------------------------------------------------------------------------

CV_result <- adaboost_cv(
  X = X_train,
  y = y_train,
  M = 300,
  k = 5,
  learning_rate = 1,
  maximum_splits = 75,
  seed = 123
)


# ------------------------------------------------------------------------------
# 42. Best CV Iteration
# ------------------------------------------------------------------------------

best_CV_iteration <- which.min(
  CV_result$mean
)


best_CV_iteration


CV_result$mean[
  best_CV_iteration
]


# ------------------------------------------------------------------------------
# 43. Plot CV Error
# ------------------------------------------------------------------------------

plot(
  seq_along(
    CV_result$mean
  ),
  CV_result$mean,
  type = "l",
  lwd = 2,
  xlab = "AdaBoost Iteration",
  ylab = "Cross-Validated Error",
  main = "AdaBoost Cross Validation"
)


abline(
  v = best_CV_iteration,
  lty = 2
)


# ==============================================================================
# PART XVIII
#
# ONE-STANDARD-ERROR RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Calculate One-SE Iteration
# ------------------------------------------------------------------------------

minimum_CV_error <-
  CV_result$mean[
    best_CV_iteration
  ]


minimum_CV_SE <-
  CV_result$SE[
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


# ==============================================================================
# PART XIX
#
# FINAL TEST PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. CV-Selected Predictions
# ------------------------------------------------------------------------------

final_test_prediction <- predict_adaboost(
  adaboost_train,
  X_test,
  M = one_SE_iteration
)


final_test_accuracy <- mean(
  final_test_prediction ==
    y_test
)


final_test_error <- mean(
  final_test_prediction !=
    y_test
)


final_test_accuracy


final_test_error


# ------------------------------------------------------------------------------
# 46. Confusion Matrix
# ------------------------------------------------------------------------------

confusion_matrix <- table(
  Actual = y_test,
  Predicted =
    final_test_prediction
)


confusion_matrix


# ------------------------------------------------------------------------------
# 47. Classification Metrics
# ------------------------------------------------------------------------------

TP <- sum(
  y_test == 1 &
    final_test_prediction == 1
)


TN <- sum(
  y_test == -1 &
    final_test_prediction == -1
)


FP <- sum(
  y_test == -1 &
    final_test_prediction == 1
)


FN <- sum(
  y_test == 1 &
    final_test_prediction == -1
)


accuracy <- (
  TP +
    TN
) /
  length(
    y_test
  )


sensitivity <- TP /
  (
    TP +
      FN
  )


specificity <- TN /
  (
    TN +
      FP
  )


precision <- TP /
  (
    TP +
      FP
  )


F1 <- 2 *
  precision *
  sensitivity /
  (
    precision +
      sensitivity
  )


data.frame(
  Metric = c(
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1"
  ),
  Value = c(
    accuracy,
    sensitivity,
    specificity,
    precision,
    F1
  )
)


# ==============================================================================
# PART XX
#
# COMPARE WITH LOGISTIC REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Logistic Baseline
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = ifelse(
    y_train == 1,
    1,
    0
  ),
  x1 = X_train[
    ,
    "x1"
  ],
  x2 = X_train[
    ,
    "x2"
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
  ]
)


logistic_model <- glm(
  y ~ x1 + x2,
  data = train_data,
  family = binomial()
)


logistic_probability <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)


logistic_prediction <- ifelse(
  logistic_probability >=
    0.5,
  1,
  -1
)


logistic_test_error <- mean(
  logistic_prediction !=
    y_test
)


logistic_test_error


# ==============================================================================
# PART XXI
#
# SINGLE STUMP BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Fit One Classification Stump
# ------------------------------------------------------------------------------

single_training_stump <-
  fit_classification_stump(
    X = X_train,
    y = y_train
  )


single_stump_prediction <-
  predict_classification_stump(
    single_training_stump,
    X_test
  )


single_stump_error <- mean(
  single_stump_prediction !=
    y_test
)


single_stump_error


# ------------------------------------------------------------------------------
# 50. Model Comparison
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Single Classification Stump",
    "Logistic Regression",
    "AdaBoost"
  ),
  Test_Error = c(
    single_stump_error,
    logistic_test_error,
    final_test_error
  )
)


model_comparison$Accuracy <-
  1 -
  model_comparison$Test_Error


model_comparison


# ==============================================================================
# PART XXII
#
# LEARNING RATE / SHRINKAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Compare Learning Rates
# ------------------------------------------------------------------------------

learning_rates <- c(
  1,
  0.5,
  0.25,
  0.1,
  0.05
)


learning_rate_results <- data.frame(
  Learning_Rate =
    learning_rates,
  Train_Error =
    NA_real_,
  Test_Error =
    NA_real_
)


for (
  i in seq_along(
    learning_rates
  )
) {
  
  model_i <- fit_adaboost(
    X = X_train,
    y = y_train,
    M = 300,
    learning_rate =
      learning_rates[i],
    maximum_splits = 75
  )
  
  
  train_prediction_i <- predict_adaboost(
    model_i,
    X_train
  )
  
  
  test_prediction_i <- predict_adaboost(
    model_i,
    X_test
  )
  
  
  learning_rate_results$Train_Error[i] <-
    mean(
      train_prediction_i !=
        y_train
    )
  
  
  learning_rate_results$Test_Error[i] <-
    mean(
      test_prediction_i !=
        y_test
    )
}


learning_rate_results


# ==============================================================================
# PART XXIII
#
# NOISY LABELS
# ==============================================================================


# ------------------------------------------------------------------------------
# 52. Introduce Label Noise
# ------------------------------------------------------------------------------

set.seed(999)


noise_fraction <- 0.10


number_flipped <- round(
  noise_fraction *
    length(
      y_train
    )
)


flipped_indices <- sample(
  seq_along(
    y_train
  ),
  size =
    number_flipped
)


y_train_noisy <- y_train


y_train_noisy[
  flipped_indices
] <-
  -y_train_noisy[
    flipped_indices
  ]


# ------------------------------------------------------------------------------
# 53. Fit AdaBoost to Noisy Labels
# ------------------------------------------------------------------------------

noisy_model <- fit_adaboost(
  X = X_train,
  y = y_train_noisy,
  M = 300,
  learning_rate = 1,
  maximum_splits = 75
)


# ------------------------------------------------------------------------------
# 54. Examine Final Weights
# ------------------------------------------------------------------------------

noisy_final_weights <-
  noisy_model$weight_history[
    ,
    ncol(
      noisy_model$weight_history
    )
  ]


mean_weight_flipped <- mean(
  noisy_final_weights[
    flipped_indices
  ]
)


mean_weight_unflipped <- mean(
  noisy_final_weights[
    -flipped_indices
  ]
)


data.frame(
  Group = c(
    "Flipped Labels",
    "Unflipped Labels"
  ),
  Mean_Final_Weight = c(
    mean_weight_flipped,
    mean_weight_unflipped
  )
)


# AdaBoost may concentrate substantial weight on noisy or difficult cases.
#
# This illustrates one reason exponential-loss boosting can be sensitive
# to label noise and outliers.


# ==============================================================================
# PART XXIV
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Final Output
# ------------------------------------------------------------------------------

cat(
  "AdaBoost Summary\n"
)


cat(
  "----------------\n"
)


cat(
  "Number of fitted learners:",
  adaboost_train$iterations,
  "\n"
)


cat(
  "Learning rate:",
  adaboost_train$learning_rate,
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
  "Single-stump test error:",
  round(
    single_stump_error,
    4
  ),
  "\n"
)


cat(
  "Logistic-regression test error:",
  round(
    logistic_test_error,
    4
  ),
  "\n"
)


cat(
  "AdaBoost test error:",
  round(
    final_test_error,
    4
  ),
  "\n"
)


cat(
  "AdaBoost test accuracy:",
  round(
    final_test_accuracy,
    4
  ),
  "\n"
)