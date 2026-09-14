# ==============================================================================
# Nearest Shrunken Centroids
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   Tibshirani et al. - Nearest Shrunken Centroids / PAM
#
# Main ideas:
#   - Centroid-based classification
#   - Standardized class centroids
#   - Soft thresholding
#   - Embedded feature selection
#   - High-dimensional classification
#   - Cross-validation for shrinkage threshold
#   - Comparison with nearest-centroid classification
#   - Relationship with diagonal LDA
#
#
# Standardized class difference:
#
#       d_kj
#       =
#       (xbar_kj - xbar_j)
#       -------------------
#       m_k (s_j + s0)
#
#
# where:
#
#       m_k = sqrt(1 / n_k - 1 / n)
#
#
# Soft threshold:
#
#       d'_kj
#       =
#       sign(d_kj) (|d_kj| - Delta)_+
#
#
# Reconstructed shrunken centroid:
#
#       xbar'_kj
#       =
#       xbar_j
#       +
#       m_k (s_j + s0) d'_kj
#
#
# A feature is effectively removed when its class-specific standardized
# differences are all shrunk to zero.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE HIGH-DIMENSIONAL CLASSIFICATION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulation Setup
# ------------------------------------------------------------------------------

set.seed(123)

K <- 3

n_per_class <- 80

n <- K *
  n_per_class

p <- 50


# Only the first 8 variables contain signal.

signal_features <- 1:8


# ------------------------------------------------------------------------------
# 2. Class Means
# ------------------------------------------------------------------------------

class_means <- matrix(
  0,
  nrow = K,
  ncol = p
)


class_means[
  1,
  1:4
] <- c(
  2.0,
  1.5,
  -1.5,
  -2.0
)


class_means[
  2,
  1:4
] <- c(
  -2.0,
  -1.5,
  1.5,
  2.0
)


class_means[
  3,
  5:8
] <- c(
  2.0,
  -2.0,
  1.5,
  -1.5
)


# ------------------------------------------------------------------------------
# 3. Generate Observations
# ------------------------------------------------------------------------------

X <- matrix(
  NA_real_,
  nrow = n,
  ncol = p
)


y <- rep(
  seq_len(K),
  each = n_per_class
)


row_index <- 1


for (
  k in seq_len(K)
) {
  
  rows_k <- row_index:
    (
      row_index +
        n_per_class -
        1
    )
  
  
  X[
    rows_k,
  ] <- matrix(
    rnorm(
      n_per_class *
        p,
      mean = 0,
      sd = 1
    ),
    nrow =
      n_per_class,
    ncol =
      p
  )
  
  
  X[
    rows_k,
  ] <- sweep(
    X[
      rows_k,
      ,
      drop = FALSE
    ],
    2,
    class_means[
      k,
    ],
    "+"
  )
  
  
  row_index <- row_index +
    n_per_class
}


colnames(
  X
) <- paste0(
  "x",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 4. Examine First Few Variables
# ------------------------------------------------------------------------------

pairs(
  data.frame(
    Class =
      factor(y),
    X[
      ,
      1:5,
      drop = FALSE
    ]
  ),
  main = "Nearest Shrunken Centroids Data"
)


# ==============================================================================
# PART II
#
# TRAIN / TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Stratified Split
# ------------------------------------------------------------------------------

set.seed(456)


train_index <- integer(0)


for (
  k in seq_len(K)
) {
  
  class_indices <- which(
    y == k
  )
  
  
  train_index <- c(
    train_index,
    sample(
      class_indices,
      size = floor(
        0.7 *
          length(
            class_indices
          )
      )
    )
  )
}


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


# ==============================================================================
# PART III
#
# CLASS CENTROIDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Overall Centroid
# ------------------------------------------------------------------------------

overall_centroid <- colMeans(
  X_train
)


# ------------------------------------------------------------------------------
# 7. Class-Specific Centroids
# ------------------------------------------------------------------------------

compute_class_centroids <- function(
    X,
    y
) {
  
  classes <- sort(
    unique(y)
  )
  
  
  centroids <- matrix(
    NA_real_,
    nrow =
      length(
        classes
      ),
    ncol =
      ncol(X)
  )
  
  
  for (
    k in seq_along(
      classes
    )
  ) {
    
    centroids[
      k,
    ] <- colMeans(
      X[
        y ==
          classes[k],
        ,
        drop = FALSE
      ]
    )
  }
  
  
  rownames(
    centroids
  ) <- paste0(
    "Class_",
    classes
  )
  
  
  colnames(
    centroids
  ) <- colnames(X)
  
  
  centroids
}


class_centroids <- compute_class_centroids(
  X_train,
  y_train
)


class_centroids[
  ,
  1:10,
  drop = FALSE
]


# ==============================================================================
# PART IV
#
# WITHIN-CLASS FEATURE STANDARD DEVIATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Pooled Within-Class Variance
# ------------------------------------------------------------------------------

pooled_within_sd <- function(
    X,
    y
) {
  
  classes <- sort(
    unique(y)
  )
  
  
  n <- nrow(
    X
  )
  
  
  K <- length(
    classes
  )
  
  
  pooled_sum_squares <- rep(
    0,
    ncol(X)
  )
  
  
  for (
    k in classes
  ) {
    
    X_k <- X[
      y == k,
      ,
      drop = FALSE
    ]
    
    
    mean_k <- colMeans(
      X_k
    )
    
    
    centered_k <- sweep(
      X_k,
      2,
      mean_k,
      "-"
    )
    
    
    pooled_sum_squares <-
      pooled_sum_squares +
      colSums(
        centered_k^2
      )
  }
  
  
  sqrt(
    pooled_sum_squares /
      (
        n -
          K
      )
  )
}


s_j <- pooled_within_sd(
  X_train,
  y_train
)


summary(
  s_j
)


# ==============================================================================
# PART V
#
# FUDGE FACTOR s0
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Stabilization Constant
# ------------------------------------------------------------------------------

s0 <- median(
  s_j
)


s0


# The original nearest-shrunken-centroid method uses a small stabilizing
# quantity so variables with tiny within-class variance do not receive
# enormous standardized differences.


# ==============================================================================
# PART VI
#
# CLASS SIZE FACTORS
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Compute m_k
# ------------------------------------------------------------------------------

compute_mk <- function(
    y
) {
  
  classes <- sort(
    unique(y)
  )
  
  
  n <- length(
    y
  )
  
  
  n_k <- sapply(
    classes,
    function(k) {
      
      sum(
        y == k
      )
    }
  )
  
  
  # Standard PAM / nearest-shrunken-centroid scaling:
  #
  #       m_k = sqrt(1 / n_k - 1 / n)
  
  m_k <- sqrt(
    1 /
      n_k -
      1 /
      n
  )
  
  
  names(
    m_k
  ) <- classes
  
  
  list(
    n_k =
      n_k,
    m_k =
      m_k
  )
}


class_size_info <- compute_mk(
  y_train
)


class_size_info$n_k


class_size_info$m_k


# ==============================================================================
# PART VII
#
# STANDARDIZED CENTROID DIFFERENCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Compute d_kj
# ------------------------------------------------------------------------------

compute_standardized_differences <- function(
    X,
    y,
    s0 = NULL
) {
  
  classes <- sort(
    unique(y)
  )
  
  
  K <- length(
    classes
  )
  
  
  overall_centroid <- colMeans(
    X
  )
  
  
  class_centroids <- compute_class_centroids(
    X,
    y
  )
  
  
  s_j <- pooled_within_sd(
    X,
    y
  )
  
  
  if (
    is.null(
      s0
    )
  ) {
    
    s0 <- median(
      s_j
    )
  }
  
  
  size_info <- compute_mk(
    y
  )
  
  
  m_k <- size_info$m_k
  
  
  d <- matrix(
    NA_real_,
    nrow = K,
    ncol =
      ncol(X)
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    d[
      k,
    ] <- (
      class_centroids[
        k,
      ] -
        overall_centroid
    ) /
      (
        m_k[k] *
          (
            s_j +
              s0
          )
      )
  }
  
  
  rownames(
    d
  ) <- rownames(
    class_centroids
  )
  
  
  colnames(
    d
  ) <- colnames(X)
  
  
  list(
    d =
      d,
    overall_centroid =
      overall_centroid,
    class_centroids =
      class_centroids,
    s_j =
      s_j,
    s0 =
      s0,
    m_k =
      m_k,
    classes =
      classes
  )
}


centroid_statistics <- compute_standardized_differences(
  X_train,
  y_train
)


d <- centroid_statistics$d


round(
  d[
    ,
    1:12,
    drop = FALSE
  ],
  2
)


# ==============================================================================
# PART VIII
#
# SOFT THRESHOLDING
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Soft Threshold Function
# ------------------------------------------------------------------------------

soft_threshold <- function(
    z,
    threshold
) {
  
  sign(z) *
    pmax(
      abs(z) -
        threshold,
      0
    )
}


# ------------------------------------------------------------------------------
# 13. Example Shrinkage
# ------------------------------------------------------------------------------

example_values <- c(
  -4,
  -2,
  -0.5,
  0,
  0.5,
  2,
  4
)


soft_threshold(
  example_values,
  threshold = 1
)


# ==============================================================================
# PART IX
#
# SHRUNKEN CENTROIDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Construct Shrunken Centroids
# ------------------------------------------------------------------------------

shrink_centroids <- function(
    centroid_statistics,
    threshold
) {
  
  d <- centroid_statistics$d
  
  
  d_shrunken <- soft_threshold(
    d,
    threshold
  )
  
  
  K <- nrow(
    d_shrunken
  )
  
  
  p <- ncol(
    d_shrunken
  )
  
  
  shrunken_centroids <- matrix(
    NA_real_,
    nrow = K,
    ncol = p
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    shrunken_centroids[
      k,
    ] <-
      centroid_statistics$overall_centroid +
      centroid_statistics$m_k[k] *
      (
        centroid_statistics$s_j +
          centroid_statistics$s0
      ) *
      d_shrunken[
        k,
      ]
  }
  
  
  rownames(
    shrunken_centroids
  ) <- rownames(
    d_shrunken
  )
  
  
  colnames(
    shrunken_centroids
  ) <- colnames(
    d_shrunken
  )
  
  
  list(
    centroids =
      shrunken_centroids,
    d_shrunken =
      d_shrunken,
    threshold =
      threshold
  )
}


# ------------------------------------------------------------------------------
# 15. Example Threshold
# ------------------------------------------------------------------------------

example_threshold <- 2


shrunk_example <- shrink_centroids(
  centroid_statistics,
  threshold =
    example_threshold
)


round(
  shrunk_example$d_shrunken[
    ,
    1:12,
    drop = FALSE
  ],
  2
)


# ==============================================================================
# PART X
#
# FEATURE SELECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Determine Active Features
# ------------------------------------------------------------------------------

active_features <- function(
    d_shrunken
) {
  
  apply(
    abs(
      d_shrunken
    ) >
      0,
    2,
    any
  )
}


active_example <- active_features(
  shrunk_example$d_shrunken
)


sum(
  active_example
)


colnames(X_train)[
  active_example
]


# A feature is removed when every class-specific standardized difference is
# shrunk to zero.


# ==============================================================================
# PART XI
#
# CLASSIFICATION RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Fit Nearest Shrunken Centroid Model
# ------------------------------------------------------------------------------

fit_nsc <- function(
    X,
    y,
    threshold = 0,
    s0 = NULL
) {
  
  X <- as.matrix(
    X
  )
  
  
  classes <- sort(
    unique(y)
  )
  
  
  centroid_statistics <-
    compute_standardized_differences(
      X,
      y,
      s0 =
        s0
    )
  
  
  shrinkage <- shrink_centroids(
    centroid_statistics,
    threshold =
      threshold
  )
  
  
  priors <- sapply(
    classes,
    function(k) {
      
      mean(
        y == k
      )
    }
  )
  
  
  list(
    threshold =
      threshold,
    classes =
      classes,
    priors =
      priors,
    overall_centroid =
      centroid_statistics$overall_centroid,
    original_centroids =
      centroid_statistics$class_centroids,
    shrunken_centroids =
      shrinkage$centroids,
    d =
      centroid_statistics$d,
    d_shrunken =
      shrinkage$d_shrunken,
    s_j =
      centroid_statistics$s_j,
    s0 =
      centroid_statistics$s0,
    m_k =
      centroid_statistics$m_k,
    active_features =
      active_features(
        shrinkage$d_shrunken
      )
  )
}


# ------------------------------------------------------------------------------
# 18. Prediction Scores
# ------------------------------------------------------------------------------

predict_nsc <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  K <- length(
    model$classes
  )
  
  
  n_new <- nrow(
    X_new
  )
  
  
  scores <- matrix(
    NA_real_,
    nrow =
      n_new,
    ncol =
      K
  )
  
  
  denominator <- (
    model$s_j +
      model$s0
  )^2
  
  
  # PAM-style nearest-centroid discriminant score:
  #
  #       delta_k(x)
  #       =
  #       sum_j
  #       (x_j - centroid_kj)^2
  #       / (s_j + s0)^2
  #       - 2 log pi_k
  #
  #
  # Smaller score is better.
  
  
  for (
    k in seq_len(K)
  ) {
    
    differences <- sweep(
      X_new,
      2,
      model$shrunken_centroids[
        k,
      ],
      "-"
    )
    
    
    scores[
      ,
      k
    ] <- rowSums(
      sweep(
        differences^2,
        2,
        denominator,
        "/"
      )
    ) -
      2 *
      log(
        model$priors[k]
      )
  }
  
  
  class_index <- max.col(
    -scores,
    ties.method = "first"
  )
  
  
  predicted_class <- model$classes[
    class_index
  ]
  
  
  list(
    class =
      predicted_class,
    scores =
      scores
  )
}


# ==============================================================================
# PART XII
#
# FIT EXAMPLE MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Fit with Delta = 2
# ------------------------------------------------------------------------------

NSC_model <- fit_nsc(
  X_train,
  y_train,
  threshold = 2
)


sum(
  NSC_model$active_features
)


colnames(X_train)[
  NSC_model$active_features
]


# ------------------------------------------------------------------------------
# 20. Test Prediction
# ------------------------------------------------------------------------------

NSC_test <- predict_nsc(
  NSC_model,
  X_test
)


# ------------------------------------------------------------------------------
# 21. Accuracy
# ------------------------------------------------------------------------------

NSC_accuracy <- mean(
  NSC_test$class ==
    y_test
)


NSC_accuracy


# ------------------------------------------------------------------------------
# 22. Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual =
    y_test,
  Predicted =
    NSC_test$class
)


# ==============================================================================
# PART XIII
#
# NEAREST CENTROID BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Threshold = 0
# ------------------------------------------------------------------------------

nearest_centroid_model <- fit_nsc(
  X_train,
  y_train,
  threshold = 0
)


nearest_centroid_prediction <- predict_nsc(
  nearest_centroid_model,
  X_test
)


nearest_centroid_accuracy <- mean(
  nearest_centroid_prediction$class ==
    y_test
)


nearest_centroid_accuracy


# Threshold zero means:
#
#       d'_kj = d_kj
#
# so there is no shrinkage.


# ==============================================================================
# PART XIV
#
# EFFECT OF SHRINKAGE THRESHOLD
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Threshold Grid
# ------------------------------------------------------------------------------

threshold_values <- seq(
  0,
  6,
  by = 0.25
)


threshold_results <- data.frame(
  Threshold =
    threshold_values,
  Number_Active_Features =
    NA_integer_,
  Train_Accuracy =
    NA_real_,
  Test_Accuracy =
    NA_real_
)


for (
  i in seq_along(
    threshold_values
  )
) {
  
  model_i <- fit_nsc(
    X_train,
    y_train,
    threshold =
      threshold_values[i]
  )
  
  
  train_prediction_i <- predict_nsc(
    model_i,
    X_train
  )
  
  
  test_prediction_i <- predict_nsc(
    model_i,
    X_test
  )
  
  
  threshold_results$Number_Active_Features[i] <-
    sum(
      model_i$active_features
    )
  
  
  threshold_results$Train_Accuracy[i] <-
    mean(
      train_prediction_i$class ==
        y_train
    )
  
  
  threshold_results$Test_Accuracy[i] <-
    mean(
      test_prediction_i$class ==
        y_test
    )
}


threshold_results


# ------------------------------------------------------------------------------
# 25. Accuracy vs Threshold
# ------------------------------------------------------------------------------

matplot(
  threshold_results$Threshold,
  cbind(
    threshold_results$Train_Accuracy,
    threshold_results$Test_Accuracy
  ),
  type = "l",
  lty = c(
    1,
    2
  ),
  lwd = 2,
  xlab = "Shrinkage Threshold Delta",
  ylab = "Accuracy",
  main = "Nearest Shrunken Centroids"
)


legend(
  "bottomleft",
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
# 26. Number of Active Features
# ------------------------------------------------------------------------------

plot(
  threshold_results$Threshold,
  threshold_results$Number_Active_Features,
  type = "b",
  pch = 19,
  xlab = "Shrinkage Threshold Delta",
  ylab = "Number of Active Features",
  main = "Feature Selection by Centroid Shrinkage"
)


# ==============================================================================
# PART XV
#
# SHRINKAGE PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Standardized Difference Paths
# ------------------------------------------------------------------------------

selected_features <- 1:12


shrinkage_path <- array(
  NA_real_,
  dim = c(
    length(
      threshold_values
    ),
    K,
    length(
      selected_features
    )
  )
)


for (
  i in seq_along(
    threshold_values
  )
) {
  
  shrinkage_i <- shrink_centroids(
    centroid_statistics,
    threshold_values[i]
  )
  
  
  shrinkage_path[
    i,
    ,
  ] <- shrinkage_i$d_shrunken[
    ,
    selected_features,
    drop = FALSE
  ]
}


# ------------------------------------------------------------------------------
# 28. Plot Paths for Class 1
# ------------------------------------------------------------------------------

matplot(
  threshold_values,
  shrinkage_path[
    ,
    1,
  ],
  type = "l",
  lty = 1,
  xlab = "Threshold Delta",
  ylab = "Shrunken Standardized Difference",
  main = "Nearest Shrunken Centroid Paths: Class 1"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART XVI
#
# MANUAL K-FOLD CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Stratified Fold Assignment
# ------------------------------------------------------------------------------

make_stratified_folds <- function(
    y,
    folds = 5,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  fold_id <- integer(
    length(y)
  )
  
  
  classes <- sort(
    unique(y)
  )
  
  
  for (
    k in classes
  ) {
    
    indices <- which(
      y == k
    )
    
    
    fold_labels <- sample(
      rep(
        seq_len(folds),
        length.out =
          length(
            indices
          )
      )
    )
    
    
    fold_id[
      indices
    ] <- fold_labels
  }
  
  
  fold_id
}


# ------------------------------------------------------------------------------
# 30. Cross Validation
# ------------------------------------------------------------------------------

nsc_cv <- function(
    X,
    y,
    threshold_values,
    folds = 5,
    seed = 123
) {
  
  fold_id <- make_stratified_folds(
    y,
    folds =
      folds,
    seed =
      seed
  )
  
  
  accuracy_matrix <- matrix(
    NA_real_,
    nrow =
      length(
        threshold_values
      ),
    ncol =
      folds
  )
  
  
  feature_matrix <- matrix(
    NA_real_,
    nrow =
      length(
        threshold_values
      ),
    ncol =
      folds
  )
  
  
  for (
    fold in seq_len(
      folds
    )
  ) {
    
    training_indices <- which(
      fold_id != fold
    )
    
    
    validation_indices <- which(
      fold_id == fold
    )
    
    
    for (
      j in seq_along(
        threshold_values
      )
    ) {
      
      model_j <- fit_nsc(
        X[
          training_indices,
          ,
          drop = FALSE
        ],
        y[
          training_indices
        ],
        threshold =
          threshold_values[j]
      )
      
      
      prediction_j <- predict_nsc(
        model_j,
        X[
          validation_indices,
          ,
          drop = FALSE
        ]
      )
      
      
      accuracy_matrix[
        j,
        fold
      ] <- mean(
        prediction_j$class ==
          y[
            validation_indices
          ]
      )
      
      
      feature_matrix[
        j,
        fold
      ] <- sum(
        model_j$active_features
      )
    }
  }
  
  
  mean_accuracy <- rowMeans(
    accuracy_matrix
  )
  
  
  SE_accuracy <- apply(
    accuracy_matrix,
    1,
    sd
  ) /
    sqrt(
      folds
    )
  
  
  mean_features <- rowMeans(
    feature_matrix
  )
  
  
  list(
    thresholds =
      threshold_values,
    mean_accuracy =
      mean_accuracy,
    SE_accuracy =
      SE_accuracy,
    mean_features =
      mean_features,
    fold_accuracy =
      accuracy_matrix
  )
}


# ------------------------------------------------------------------------------
# 31. Run CV
# ------------------------------------------------------------------------------

CV_thresholds <- seq(
  0,
  6,
  by = 0.25
)


CV_result <- nsc_cv(
  X_train,
  y_train,
  threshold_values =
    CV_thresholds,
  folds = 5,
  seed = 123
)


CV_table <- data.frame(
  Threshold =
    CV_result$thresholds,
  Mean_Accuracy =
    CV_result$mean_accuracy,
  SE =
    CV_result$SE_accuracy,
  Mean_Active_Features =
    CV_result$mean_features
)


CV_table


# ------------------------------------------------------------------------------
# 32. Best Threshold
# ------------------------------------------------------------------------------

best_index <- which.max(
  CV_result$mean_accuracy
)


best_threshold <- CV_result$thresholds[
  best_index
]


best_threshold


# ==============================================================================
# PART XVII
#
# ONE-STANDARD-ERROR RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Prefer Simpler Model Within One SE
# ------------------------------------------------------------------------------

best_accuracy <- CV_result$mean_accuracy[
  best_index
]


best_SE <- CV_result$SE_accuracy[
  best_index
]


accuracy_cutoff <- best_accuracy -
  best_SE


eligible_indices <- which(
  CV_result$mean_accuracy >=
    accuracy_cutoff
)


# Larger threshold means more shrinkage and generally fewer selected features.
#
# Therefore choose the LARGEST threshold satisfying the one-SE rule.

one_SE_index <- eligible_indices[
  which.max(
    CV_result$thresholds[
      eligible_indices
    ]
  )
]


one_SE_threshold <- CV_result$thresholds[
  one_SE_index
]


one_SE_threshold


# ------------------------------------------------------------------------------
# 34. Plot CV Accuracy
# ------------------------------------------------------------------------------

plot(
  CV_result$thresholds,
  CV_result$mean_accuracy,
  type = "b",
  pch = 19,
  ylim = range(
    CV_result$mean_accuracy -
      CV_result$SE_accuracy,
    CV_result$mean_accuracy +
      CV_result$SE_accuracy
  ),
  xlab = "Shrinkage Threshold Delta",
  ylab = "Cross-Validated Accuracy",
  main = "NSC Cross Validation"
)


arrows(
  CV_result$thresholds,
  CV_result$mean_accuracy -
    CV_result$SE_accuracy,
  CV_result$thresholds,
  CV_result$mean_accuracy +
    CV_result$SE_accuracy,
  angle = 90,
  code = 3,
  length = 0.03
)


abline(
  v =
    best_threshold,
  lty = 2
)


abline(
  v =
    one_SE_threshold,
  lty = 3
)


# ==============================================================================
# PART XVIII
#
# FINAL MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Fit One-SE Model
# ------------------------------------------------------------------------------

final_NSC <- fit_nsc(
  X_train,
  y_train,
  threshold =
    one_SE_threshold
)


# ------------------------------------------------------------------------------
# 36. Selected Features
# ------------------------------------------------------------------------------

selected_features_final <- colnames(
  X_train
)[
  final_NSC$active_features
]


selected_features_final


length(
  selected_features_final
)


# ------------------------------------------------------------------------------
# 37. Compare with True Signal Variables
# ------------------------------------------------------------------------------

true_signal_names <- colnames(
  X_train
)[
  signal_features
]


true_signal_names


# ------------------------------------------------------------------------------
# 38. Final Test Prediction
# ------------------------------------------------------------------------------

final_prediction <- predict_nsc(
  final_NSC,
  X_test
)


final_accuracy <- mean(
  final_prediction$class ==
    y_test
)


final_accuracy


table(
  Actual =
    y_test,
  Predicted =
    final_prediction$class
)


# ==============================================================================
# PART XIX
#
# FEATURE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Maximum Absolute Standardized Difference
# ------------------------------------------------------------------------------

feature_importance <- apply(
  abs(
    final_NSC$d
  ),
  2,
  max
)


importance_order <- order(
  feature_importance,
  decreasing = TRUE
)


feature_importance_table <- data.frame(
  Feature =
    colnames(X_train)[
      importance_order
    ],
  Importance =
    feature_importance[
      importance_order
    ],
  Selected =
    final_NSC$active_features[
      importance_order
    ]
)


head(
  feature_importance_table,
  20
)


# ------------------------------------------------------------------------------
# 40. Plot Feature Importance
# ------------------------------------------------------------------------------

top_features <- importance_order[
  1:20
]


barplot(
  feature_importance[
    top_features
  ],
  names.arg =
    colnames(X_train)[
      top_features
    ],
  las = 2,
  ylab = "Maximum |d_kj|",
  main = "Nearest Shrunken Centroid Feature Importance"
)


abline(
  h =
    one_SE_threshold,
  lty = 2
)


# ==============================================================================
# PART XX
#
# VISUALIZE ORIGINAL VS SHRUNKEN CENTROIDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. First 12 Features
# ------------------------------------------------------------------------------

features_to_plot <- 1:12


par(
  mfrow = c(
    1,
    2
  )
)


matplot(
  features_to_plot,
  t(
    final_NSC$original_centroids[
      ,
      features_to_plot,
      drop = FALSE
    ]
  ),
  type = "b",
  pch = 1:K,
  lty = 1,
  xlab = "Feature",
  ylab = "Centroid Value",
  main = "Original Centroids"
)


matplot(
  features_to_plot,
  t(
    final_NSC$shrunken_centroids[
      ,
      features_to_plot,
      drop = FALSE
    ]
  ),
  type = "b",
  pch = 1:K,
  lty = 1,
  xlab = "Feature",
  ylab = "Centroid Value",
  main = "Shrunken Centroids"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART XXI
#
# COMPARISON WITH DIAGONAL LDA
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Manual Diagonal LDA
# ------------------------------------------------------------------------------

fit_diagonal_lda <- function(
    X,
    y
) {
  
  classes <- sort(
    unique(y)
  )
  
  
  centroids <- compute_class_centroids(
    X,
    y
  )
  
  
  pooled_sd <- pooled_within_sd(
    X,
    y
  )
  
  
  priors <- sapply(
    classes,
    function(k) {
      
      mean(
        y == k
      )
    }
  )
  
  
  list(
    classes =
      classes,
    centroids =
      centroids,
    variance =
      pooled_sd^2,
    priors =
      priors
  )
}


predict_diagonal_lda <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  K <- length(
    model$classes
  )
  
  
  scores <- matrix(
    NA_real_,
    nrow =
      nrow(
        X_new
      ),
    ncol =
      K
  )
  
  
  variance <- pmax(
    model$variance,
    1e-8
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    differences <- sweep(
      X_new,
      2,
      model$centroids[
        k,
      ],
      "-"
    )
    
    
    scores[
      ,
      k
    ] <- rowSums(
      sweep(
        differences^2,
        2,
        variance,
        "/"
      )
    ) -
      2 *
      log(
        model$priors[k]
      )
  }
  
  
  model$classes[
    max.col(
      -scores,
      ties.method = "first"
    )
  ]
}


# ------------------------------------------------------------------------------
# 43. Fit Diagonal LDA
# ------------------------------------------------------------------------------

DLDA_model <- fit_diagonal_lda(
  X_train,
  y_train
)


DLDA_prediction <- predict_diagonal_lda(
  DLDA_model,
  X_test
)


DLDA_accuracy <- mean(
  DLDA_prediction ==
    y_test
)


# ==============================================================================
# PART XXII
#
# MODEL COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Compare Methods
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Nearest Centroid",
    "Nearest Shrunken Centroid",
    "Diagonal LDA"
  ),
  Test_Accuracy = c(
    nearest_centroid_accuracy,
    final_accuracy,
    DLDA_accuracy
  ),
  Selected_Features = c(
    p,
    sum(
      final_NSC$active_features
    ),
    p
  )
)


model_comparison


# ==============================================================================
# PART XXIII
#
# HIGH-DIMENSIONAL p > n EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Generate p >> n Data
# ------------------------------------------------------------------------------

set.seed(999)


n_hd_per_class <- 25

K_hd <- 3

n_hd <- n_hd_per_class *
  K_hd

p_hd <- 500


X_hd <- matrix(
  rnorm(
    n_hd *
      p_hd
  ),
  nrow =
    n_hd,
  ncol =
    p_hd
)


y_hd <- rep(
  1:K_hd,
  each =
    n_hd_per_class
)


# Add signal only to first 10 features.

X_hd[
  y_hd == 1,
  1:5
] <- X_hd[
  y_hd == 1,
  1:5
] + 2


X_hd[
  y_hd == 2,
  1:5
] <- X_hd[
  y_hd == 2,
  1:5
] - 2


X_hd[
  y_hd == 3,
  6:10
] <- X_hd[
  y_hd == 3,
  6:10
] + 2


colnames(
  X_hd
) <- paste0(
  "g",
  seq_len(
    p_hd
  )
)


# ------------------------------------------------------------------------------
# 46. Split
# ------------------------------------------------------------------------------

set.seed(1000)


hd_train <- integer(0)


for (
  k in seq_len(
    K_hd
  )
) {
  
  indices_k <- which(
    y_hd == k
  )
  
  
  hd_train <- c(
    hd_train,
    sample(
      indices_k,
      size = 18
    )
  )
}


hd_test <- setdiff(
  seq_len(
    n_hd
  ),
  hd_train
)


# ------------------------------------------------------------------------------
# 47. Cross-Validate Threshold
# ------------------------------------------------------------------------------

hd_thresholds <- seq(
  0,
  8,
  by = 0.5
)


hd_CV <- nsc_cv(
  X_hd[
    hd_train,
    ,
    drop = FALSE
  ],
  y_hd[
    hd_train
  ],
  threshold_values =
    hd_thresholds,
  folds = 3,
  seed = 123
)


hd_best_threshold <- hd_CV$thresholds[
  which.max(
    hd_CV$mean_accuracy
  )
]


hd_best_threshold


# ------------------------------------------------------------------------------
# 48. Fit High-Dimensional NSC
# ------------------------------------------------------------------------------

hd_model <- fit_nsc(
  X_hd[
    hd_train,
    ,
    drop = FALSE
  ],
  y_hd[
    hd_train
  ],
  threshold =
    hd_best_threshold
)


hd_prediction <- predict_nsc(
  hd_model,
  X_hd[
    hd_test,
    ,
    drop = FALSE
  ]
)


hd_accuracy <- mean(
  hd_prediction$class ==
    y_hd[
      hd_test
    ]
)


hd_accuracy


sum(
  hd_model$active_features
)


colnames(
  X_hd
)[
  hd_model$active_features
][
  1:min(
    30,
    sum(
      hd_model$active_features
    )
  )
]


# ==============================================================================
# PART XXIV
#
# EFFECT OF MANY NOISE VARIABLES
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Compare Using All vs Selected Features
# ------------------------------------------------------------------------------

if (
  sum(
    final_NSC$active_features
  ) >
  0
) {
  
  selected_nearest_centroid <- fit_nsc(
    X_train[
      ,
      final_NSC$active_features,
      drop = FALSE
    ],
    y_train,
    threshold = 0
  )
  
  
  selected_prediction <- predict_nsc(
    selected_nearest_centroid,
    X_test[
      ,
      final_NSC$active_features,
      drop = FALSE
    ]
  )
  
  
  selected_centroid_accuracy <- mean(
    selected_prediction$class ==
      y_test
  )
  
  
  data.frame(
    Method = c(
      "Nearest Centroid: All Features",
      "Nearest Centroid: NSC Selected Features",
      "Nearest Shrunken Centroid"
    ),
    Accuracy = c(
      nearest_centroid_accuracy,
      selected_centroid_accuracy,
      final_accuracy
    )
  )
}


# ==============================================================================
# PART XXV
#
# OPTIONAL VERIFICATION WITH pamr
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Verify with pamr if Installed
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "pamr",
    quietly = TRUE
  )
) {
  
  pam_data <- list(
    x =
      t(
        X_train
      ),
    y =
      as.character(
        y_train
      ),
    geneid =
      colnames(
        X_train
      ),
    genenames =
      colnames(
        X_train
      )
  )
  
  
  pam_model <- pamr::pamr.train(
    pam_data
  )
  
  
  print(
    pam_model
  )
  
  
  cat(
    "\nOptional pamr model fitted successfully.\n"
  )
  
  
  cat(
    "Note: pamr uses its own threshold path and implementation details,\n"
  )
  
  
  cat(
    "so exact selected features and predictions need not match the\n"
  )
  
  
  cat(
    "manual educational implementation.\n"
  )
}


# ==============================================================================
# PART XXVI
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Output
# ------------------------------------------------------------------------------

cat(
  "Nearest Shrunken Centroids Summary\n"
)


cat(
  "----------------------------------\n"
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
  "Number of predictors:",
  p,
  "\n"
)


cat(
  "True signal predictors:",
  length(
    signal_features
  ),
  "\n"
)


cat(
  "CV-optimal threshold:",
  round(
    best_threshold,
    4
  ),
  "\n"
)


cat(
  "One-SE threshold:",
  round(
    one_SE_threshold,
    4
  ),
  "\n"
)


cat(
  "Selected predictors:",
  sum(
    final_NSC$active_features
  ),
  "\n"
)


cat(
  "Nearest centroid test accuracy:",
  round(
    nearest_centroid_accuracy,
    4
  ),
  "\n"
)


cat(
  "Nearest shrunken centroid test accuracy:",
  round(
    final_accuracy,
    4
  ),
  "\n"
)


cat(
  "Diagonal LDA test accuracy:",
  round(
    DLDA_accuracy,
    4
  ),
  "\n"
)


cat(
  "High-dimensional p > n accuracy:",
  round(
    hd_accuracy,
    4
  ),
  "\n"
)