# ==============================================================================
# k-Nearest Neighbors
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Instance-based learning
#   - Euclidean distance
#   - Classification
#   - Regression
#   - Choice of k
#   - Bias-variance tradeoff
#   - Feature scaling
#   - Cross-validation
#   - Decision boundaries
#   - Curse of dimensionality
#
#
# Classification:
#
#       y_hat(x0)
#       =
#       majority class among the k nearest observations
#
#
# Regression:
#
#       y_hat(x0)
#       =
#       average response among the k nearest observations
#
#
# No explicit parametric model is fitted.
#
# Prediction is based directly on nearby observations.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE CLASSIFICATION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Nonlinear Binary Data
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


latent_score <-
  1.5 * sin(x1) +
  0.8 * x2 -
  0.4 * x1 * x2


probability <- 1 /
  (
    1 +
      exp(
        -latent_score
      )
  )


y <- rbinom(
  n,
  size = 1,
  prob = probability
)


X <- cbind(
  x1 = x1,
  x2 = x2
)


# ------------------------------------------------------------------------------
# 2. Plot Data
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
  main = "k-NN Classification Data"
)


legend(
  "topright",
  legend = c(
    "Class 0",
    "Class 1"
  ),
  pch = c(
    1,
    19
  )
)


# ==============================================================================
# PART II
#
# DISTANCE CALCULATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Euclidean Distance
# ------------------------------------------------------------------------------

euclidean_distance <- function(
    x,
    z
) {
  
  sqrt(
    sum(
      (
        x -
          z
      )^2
    )
  )
}


# ------------------------------------------------------------------------------
# 4. Verify One Distance
# ------------------------------------------------------------------------------

euclidean_distance(
  X[1, ],
  X[2, ]
)


# ==============================================================================
# PART III
#
# MANUAL k-NN CLASSIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Predict One Observation
# ------------------------------------------------------------------------------

knn_classify_one <- function(
    X_train,
    y_train,
    x_new,
    k = 5
) {
  
  X_train <- as.matrix(
    X_train
  )
  
  
  distances <- apply(
    X_train,
    1,
    function(x) {
      
      euclidean_distance(
        x,
        x_new
      )
    }
  )
  
  
  nearest_indices <- order(
    distances
  )[
    seq_len(k)
  ]
  
  
  nearest_classes <- y_train[
    nearest_indices
  ]
  
  
  class_probability <- mean(
    nearest_classes
  )
  
  
  predicted_class <- ifelse(
    class_probability >=
      0.5,
    1,
    0
  )
  
  
  return(
    list(
      class =
        predicted_class,
      probability =
        class_probability,
      neighbors =
        nearest_indices,
      distances =
        distances[
          nearest_indices
        ]
    )
  )
}


# ------------------------------------------------------------------------------
# 6. Example Prediction
# ------------------------------------------------------------------------------

example_point <- c(
  0.5,
  0.5
)


example_prediction <- knn_classify_one(
  X_train = X,
  y_train = y,
  x_new =
    example_point,
  k = 7
)


example_prediction


# ==============================================================================
# PART IV
#
# VECTOR PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Predict Many Observations
# ------------------------------------------------------------------------------

knn_classify <- function(
    X_train,
    y_train,
    X_new,
    k = 5
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  predicted_class <- numeric(
    nrow(
      X_new
    )
  )
  
  
  predicted_probability <- numeric(
    nrow(
      X_new
    )
  )
  
  
  for (
    i in seq_len(
      nrow(
        X_new
      )
    )
  ) {
    
    result <- knn_classify_one(
      X_train =
        X_train,
      y_train =
        y_train,
      x_new =
        X_new[i, ],
      k = k
    )
    
    
    predicted_class[i] <-
      result$class
    
    
    predicted_probability[i] <-
      result$probability
  }
  
  
  return(
    list(
      class =
        predicted_class,
      probability =
        predicted_probability
    )
  )
}


# ==============================================================================
# PART V
#
# TRAIN / TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Split Data
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


# ==============================================================================
# PART VI
#
# FIT k-NN CLASSIFIER
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Test Prediction
# ------------------------------------------------------------------------------

k <- 15


KNN_test <- knn_classify(
  X_train =
    X_train,
  y_train =
    y_train,
  X_new =
    X_test,
  k = k
)


# ------------------------------------------------------------------------------
# 10. Accuracy
# ------------------------------------------------------------------------------

accuracy <- mean(
  KNN_test$class ==
    y_test
)


accuracy


# ------------------------------------------------------------------------------
# 11. Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y_test,
  Predicted =
    KNN_test$class
)


# ==============================================================================
# PART VII
#
# CLASSIFICATION METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Metrics Function
# ------------------------------------------------------------------------------

classification_metrics <- function(
    actual,
    predicted
) {
  
  TP <- sum(
    actual == 1 &
      predicted == 1
  )
  
  
  TN <- sum(
    actual == 0 &
      predicted == 0
  )
  
  
  FP <- sum(
    actual == 0 &
      predicted == 1
  )
  
  
  FN <- sum(
    actual == 1 &
      predicted == 0
  )
  
  
  accuracy <- (
    TP +
      TN
  ) /
    length(actual)
  
  
  precision <- ifelse(
    TP + FP == 0,
    NA,
    TP /
      (
        TP +
          FP
      )
  )
  
  
  recall <- ifelse(
    TP + FN == 0,
    NA,
    TP /
      (
        TP +
          FN
      )
  )
  
  
  specificity <- ifelse(
    TN + FP == 0,
    NA,
    TN /
      (
        TN +
          FP
      )
  )
  
  
  data.frame(
    Accuracy =
      accuracy,
    Precision =
      precision,
    Recall =
      recall,
    Specificity =
      specificity
  )
}


classification_metrics(
  y_test,
  KNN_test$class
)


# ==============================================================================
# PART VIII
#
# DECISION BOUNDARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 100


grid_x1 <- seq(
  -3,
  3,
  length.out =
    grid_size
)


grid_x2 <- seq(
  -3,
  3,
  length.out =
    grid_size
)


grid <- expand.grid(
  x1 =
    grid_x1,
  x2 =
    grid_x2
)


grid_prediction <- knn_classify(
  X_train =
    X_train,
  y_train =
    y_train,
  X_new =
    as.matrix(
      grid
    ),
  k = k
)


probability_matrix <- matrix(
  grid_prediction$probability,
  nrow =
    grid_size,
  ncol =
    grid_size
)


# ------------------------------------------------------------------------------
# 14. Plot Estimated Probability Surface
# ------------------------------------------------------------------------------

image(
  grid_x1,
  grid_x2,
  probability_matrix,
  xlab = "x1",
  ylab = "x2",
  main = paste(
    "k-NN Decision Surface, k =",
    k
  )
)


contour(
  grid_x1,
  grid_x2,
  probability_matrix,
  levels = 0.5,
  add = TRUE,
  lwd = 3
)


points(
  X_train[, 1],
  X_train[, 2],
  pch = ifelse(
    y_train == 1,
    19,
    1
  ),
  cex = 0.4
)


# ==============================================================================
# PART IX
#
# EFFECT OF k
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Compare k Values
# ------------------------------------------------------------------------------

k_values <- c(
  1,
  3,
  5,
  10,
  20,
  40,
  80
)


k_results <- data.frame(
  k =
    k_values,
  Train_Accuracy =
    NA_real_,
  Test_Accuracy =
    NA_real_
)


for (
  i in seq_along(
    k_values
  )
) {
  
  k_i <- k_values[i]
  
  
  train_prediction_i <- knn_classify(
    X_train =
      X_train,
    y_train =
      y_train,
    X_new =
      X_train,
    k = k_i
  )
  
  
  test_prediction_i <- knn_classify(
    X_train =
      X_train,
    y_train =
      y_train,
    X_new =
      X_test,
    k = k_i
  )
  
  
  k_results$Train_Accuracy[i] <-
    mean(
      train_prediction_i$class ==
        y_train
    )
  
  
  k_results$Test_Accuracy[i] <-
    mean(
      test_prediction_i$class ==
        y_test
    )
}


k_results


# ------------------------------------------------------------------------------
# 16. Plot Accuracy
# ------------------------------------------------------------------------------

matplot(
  k_results$k,
  cbind(
    k_results$Train_Accuracy,
    k_results$Test_Accuracy
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
  lwd = 2,
  xlab = "k",
  ylab = "Accuracy",
  main = "Effect of k"
)


legend(
  "bottomright",
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
  ),
  lwd = 2
)


# ==============================================================================
# PART X
#
# LEAVE-ONE-OUT TRAINING ERROR
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Why k = 1 Gives Perfect Apparent Training Accuracy
# ------------------------------------------------------------------------------

# If we predict an observation using the same training set containing that
# observation, its closest neighbor is itself.
#
# Therefore k = 1 gives training accuracy = 1.
#
# A fairer training estimate excludes the observation being predicted.


knn_leave_one_out <- function(
    X,
    y,
    k
) {
  
  n <- nrow(
    X
  )
  
  
  predictions <- numeric(
    n
  )
  
  
  for (
    i in seq_len(n)
  ) {
    
    result <- knn_classify_one(
      X_train =
        X[
          -i,
          ,
          drop = FALSE
        ],
      y_train =
        y[-i],
      x_new =
        X[i, ],
      k = k
    )
    
    
    predictions[i] <-
      result$class
  }
  
  
  predictions
}


# ------------------------------------------------------------------------------
# 18. Leave-One-Out Accuracy
# ------------------------------------------------------------------------------

LOO_prediction <- knn_leave_one_out(
  X_train,
  y_train,
  k = 1
)


mean(
  LOO_prediction ==
    y_train
)


# ==============================================================================
# PART XI
#
# MANUAL K-FOLD CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Cross Validation
# ------------------------------------------------------------------------------

knn_cv <- function(
    X,
    y,
    k_values,
    folds = 5,
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
      seq_len(
        folds
      ),
      length.out = n
    )
  )
  
  
  CV_accuracy <- matrix(
    NA_real_,
    nrow =
      length(
        k_values
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
        k_values
      )
    ) {
      
      prediction <- knn_classify(
        X_train =
          X[
            training_indices,
            ,
            drop = FALSE
          ],
        y_train =
          y[
            training_indices
          ],
        X_new =
          X[
            validation_indices,
            ,
            drop = FALSE
          ],
        k =
          k_values[j]
      )
      
      
      CV_accuracy[
        j,
        fold
      ] <- mean(
        prediction$class ==
          y[
            validation_indices
          ]
      )
    }
  }
  
  
  mean_accuracy <- rowMeans(
    CV_accuracy
  )
  
  
  SE_accuracy <- apply(
    CV_accuracy,
    1,
    sd
  ) /
    sqrt(
      folds
    )
  
  
  return(
    list(
      k_values =
        k_values,
      mean =
        mean_accuracy,
      SE =
        SE_accuracy,
      fold_accuracy =
        CV_accuracy
    )
  )
}


# ------------------------------------------------------------------------------
# 20. Run CV
# ------------------------------------------------------------------------------

CV_k_values <- seq(
  1,
  41,
  by = 2
)


CV_result <- knn_cv(
  X = X_train,
  y = y_train,
  k_values =
    CV_k_values,
  folds = 5,
  seed = 123
)


CV_table <- data.frame(
  k =
    CV_result$k_values,
  CV_Accuracy =
    CV_result$mean,
  CV_SE =
    CV_result$SE
)


CV_table


# ------------------------------------------------------------------------------
# 21. Best k
# ------------------------------------------------------------------------------

best_index <- which.max(
  CV_result$mean
)


best_k <- CV_result$k_values[
  best_index
]


best_k


# ------------------------------------------------------------------------------
# 22. CV Plot
# ------------------------------------------------------------------------------

plot(
  CV_result$k_values,
  CV_result$mean,
  type = "b",
  pch = 19,
  xlab = "k",
  ylab = "Cross-Validated Accuracy",
  main = "k-NN Cross Validation"
)


abline(
  v =
    best_k,
  lty = 2
)


# ==============================================================================
# PART XII
#
# FINAL CLASSIFICATION MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Final Test Prediction
# ------------------------------------------------------------------------------

final_prediction <- knn_classify(
  X_train =
    X_train,
  y_train =
    y_train,
  X_new =
    X_test,
  k =
    best_k
)


final_accuracy <- mean(
  final_prediction$class ==
    y_test
)


final_accuracy


classification_metrics(
  y_test,
  final_prediction$class
)


# ==============================================================================
# PART XIII
#
# FEATURE SCALING
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Construct Badly Scaled Data
# ------------------------------------------------------------------------------

set.seed(321)


x_small <- rnorm(
  300,
  sd = 1
)


x_large <- rnorm(
  300,
  sd = 100
)


class_scale <- ifelse(
  x_small >
    0,
  1,
  0
)


X_scale <- cbind(
  x_small =
    x_small,
  x_large =
    x_large
)


# x_large is noise but dominates Euclidean distance because of its scale.


# ------------------------------------------------------------------------------
# 25. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(654)


scale_train <- sample(
  seq_len(
    nrow(
      X_scale
    )
  ),
  size = 200
)


scale_test <- setdiff(
  seq_len(
    nrow(
      X_scale
    )
  ),
  scale_train
)


# ------------------------------------------------------------------------------
# 26. Unscaled k-NN
# ------------------------------------------------------------------------------

unscaled_prediction <- knn_classify(
  X_train =
    X_scale[
      scale_train,
      ,
      drop = FALSE
    ],
  y_train =
    class_scale[
      scale_train
    ],
  X_new =
    X_scale[
      scale_test,
      ,
      drop = FALSE
    ],
  k = 7
)


unscaled_accuracy <- mean(
  unscaled_prediction$class ==
    class_scale[
      scale_test
    ]
)


# ------------------------------------------------------------------------------
# 27. Standardize Using Training Set
# ------------------------------------------------------------------------------

scale_mean <- colMeans(
  X_scale[
    scale_train,
    ,
    drop = FALSE
  ]
)


scale_sd <- apply(
  X_scale[
    scale_train,
    ,
    drop = FALSE
  ],
  2,
  sd
)


X_scale_train_standardized <- sweep(
  X_scale[
    scale_train,
    ,
    drop = FALSE
  ],
  2,
  scale_mean,
  "-"
)


X_scale_train_standardized <- sweep(
  X_scale_train_standardized,
  2,
  scale_sd,
  "/"
)


X_scale_test_standardized <- sweep(
  X_scale[
    scale_test,
    ,
    drop = FALSE
  ],
  2,
  scale_mean,
  "-"
)


X_scale_test_standardized <- sweep(
  X_scale_test_standardized,
  2,
  scale_sd,
  "/"
)


# ------------------------------------------------------------------------------
# 28. Scaled k-NN
# ------------------------------------------------------------------------------

scaled_prediction <- knn_classify(
  X_train =
    X_scale_train_standardized,
  y_train =
    class_scale[
      scale_train
    ],
  X_new =
    X_scale_test_standardized,
  k = 7
)


scaled_accuracy <- mean(
  scaled_prediction$class ==
    class_scale[
      scale_test
    ]
)


data.frame(
  Method = c(
    "Unscaled k-NN",
    "Scaled k-NN"
  ),
  Accuracy = c(
    unscaled_accuracy,
    scaled_accuracy
  )
)


# ==============================================================================
# PART XIV
#
# DISTANCE-WEIGHTED k-NN
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Weighted Classifier
# ------------------------------------------------------------------------------

knn_weighted_classify_one <- function(
    X_train,
    y_train,
    x_new,
    k = 5,
    epsilon = 1e-8
) {
  
  distances <- apply(
    X_train,
    1,
    function(x) {
      
      euclidean_distance(
        x,
        x_new
      )
    }
  )
  
  
  nearest_indices <- order(
    distances
  )[
    seq_len(k)
  ]
  
  
  nearest_distances <- distances[
    nearest_indices
  ]
  
  
  weights <- 1 /
    (
      nearest_distances +
        epsilon
    )
  
  
  probability <- weighted.mean(
    y_train[
      nearest_indices
    ],
    weights
  )
  
  
  predicted_class <- ifelse(
    probability >=
      0.5,
    1,
    0
  )
  
  
  list(
    class =
      predicted_class,
    probability =
      probability
  )
}


# ------------------------------------------------------------------------------
# 30. Weighted Vector Prediction
# ------------------------------------------------------------------------------

knn_weighted_classify <- function(
    X_train,
    y_train,
    X_new,
    k = 5
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  classes <- numeric(
    nrow(
      X_new
    )
  )
  
  
  probabilities <- numeric(
    nrow(
      X_new
    )
  )
  
  
  for (
    i in seq_len(
      nrow(
        X_new
      )
    )
  ) {
    
    result <- knn_weighted_classify_one(
      X_train,
      y_train,
      X_new[i, ],
      k
    )
    
    
    classes[i] <-
      result$class
    
    
    probabilities[i] <-
      result$probability
  }
  
  
  list(
    class =
      classes,
    probability =
      probabilities
  )
}


# ------------------------------------------------------------------------------
# 31. Compare Weighted and Unweighted
# ------------------------------------------------------------------------------

weighted_test <- knn_weighted_classify(
  X_train,
  y_train,
  X_test,
  k =
    best_k
)


data.frame(
  Method = c(
    "Unweighted k-NN",
    "Distance-Weighted k-NN"
  ),
  Accuracy = c(
    final_accuracy,
    mean(
      weighted_test$class ==
        y_test
    )
  )
)


# ==============================================================================
# PART XV
#
# k-NN REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Generate Regression Data
# ------------------------------------------------------------------------------

set.seed(777)


n_reg <- 300


xr <- runif(
  n_reg,
  min = -3,
  max = 3
)


yr_true <- 2 +
  2 *
  sin(
    1.5 *
      xr
  )


yr <- yr_true +
  rnorm(
    n_reg,
    sd = 0.8
  )


X_reg <- matrix(
  xr,
  ncol = 1
)


colnames(
  X_reg
) <- "x"


# ------------------------------------------------------------------------------
# 33. Plot Regression Data
# ------------------------------------------------------------------------------

plot(
  xr,
  yr,
  pch = 19,
  cex = 0.6,
  xlab = "x",
  ylab = "y",
  main = "k-NN Regression"
)


lines(
  sort(xr),
  yr_true[
    order(xr)
  ],
  lwd = 3
)


# ==============================================================================
# PART XVI
#
# MANUAL k-NN REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Predict One Observation
# ------------------------------------------------------------------------------

knn_regress_one <- function(
    X_train,
    y_train,
    x_new,
    k = 5
) {
  
  distances <- apply(
    X_train,
    1,
    function(x) {
      
      euclidean_distance(
        x,
        x_new
      )
    }
  )
  
  
  nearest_indices <- order(
    distances
  )[
    seq_len(k)
  ]
  
  
  mean(
    y_train[
      nearest_indices
    ]
  )
}


# ------------------------------------------------------------------------------
# 35. Predict Many Observations
# ------------------------------------------------------------------------------

knn_regress <- function(
    X_train,
    y_train,
    X_new,
    k = 5
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  sapply(
    seq_len(
      nrow(
        X_new
      )
    ),
    function(i) {
      
      knn_regress_one(
        X_train =
          X_train,
        y_train =
          y_train,
        x_new =
          X_new[i, ],
        k = k
      )
    }
  )
}


# ==============================================================================
# PART XVII
#
# REGRESSION SMOOTHING EFFECT
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Prediction Grid
# ------------------------------------------------------------------------------

xr_grid <- seq(
  -3,
  3,
  length.out = 300
)


Xr_grid <- matrix(
  xr_grid,
  ncol = 1
)


# ------------------------------------------------------------------------------
# 37. Compare k Values
# ------------------------------------------------------------------------------

regression_k_values <- c(
  1,
  5,
  20,
  60
)


plot(
  xr,
  yr,
  pch = 19,
  cex = 0.4,
  xlab = "x",
  ylab = "y",
  main = "k-NN Regression: Effect of k"
)


for (
  i in seq_along(
    regression_k_values
  )
) {
  
  prediction_i <- knn_regress(
    X_train =
      X_reg,
    y_train =
      yr,
    X_new =
      Xr_grid,
    k =
      regression_k_values[i]
  )
  
  
  lines(
    xr_grid,
    prediction_i,
    lty = i,
    lwd = 2
  )
}


legend(
  "topleft",
  legend = paste0(
    "k = ",
    regression_k_values
  ),
  lty =
    seq_along(
      regression_k_values
    ),
  lwd = 2
)


# ==============================================================================
# PART XVIII
#
# REGRESSION TRAIN / TEST
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Split Regression Data
# ------------------------------------------------------------------------------

set.seed(888)


reg_train <- sample(
  seq_len(
    n_reg
  ),
  size = 200
)


reg_test <- setdiff(
  seq_len(
    n_reg
  ),
  reg_train
)


X_reg_train <- X_reg[
  reg_train,
  ,
  drop = FALSE
]


y_reg_train <- yr[
  reg_train
]


X_reg_test <- X_reg[
  reg_test,
  ,
  drop = FALSE
]


y_reg_test <- yr[
  reg_test
]


# ==============================================================================
# PART XIX
#
# CROSS-VALIDATION FOR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Regression CV
# ------------------------------------------------------------------------------

knn_regression_cv <- function(
    X,
    y,
    k_values,
    folds = 5,
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
      seq_len(
        folds
      ),
      length.out = n
    )
  )
  
  
  fold_MSE <- matrix(
    NA_real_,
    nrow =
      length(
        k_values
      ),
    ncol =
      folds
  )
  
  
  for (
    fold in seq_len(
      folds
    )
  ) {
    
    train_indices <- which(
      fold_id != fold
    )
    
    
    validation_indices <- which(
      fold_id == fold
    )
    
    
    for (
      j in seq_along(
        k_values
      )
    ) {
      
      prediction <- knn_regress(
        X_train =
          X[
            train_indices,
            ,
            drop = FALSE
          ],
        y_train =
          y[
            train_indices
          ],
        X_new =
          X[
            validation_indices,
            ,
            drop = FALSE
          ],
        k =
          k_values[j]
      )
      
      
      fold_MSE[
        j,
        fold
      ] <- mean(
        (
          y[
            validation_indices
          ] -
            prediction
        )^2
      )
    }
  }
  
  
  list(
    k_values =
      k_values,
    mean_MSE =
      rowMeans(
        fold_MSE
      ),
    SE_MSE =
      apply(
        fold_MSE,
        1,
        sd
      ) /
      sqrt(
        folds
      ),
    fold_MSE =
      fold_MSE
  )
}


# ------------------------------------------------------------------------------
# 40. Run Regression CV
# ------------------------------------------------------------------------------

reg_CV_k <- seq(
  1,
  51,
  by = 2
)


reg_CV <- knn_regression_cv(
  X_reg_train,
  y_reg_train,
  k_values =
    reg_CV_k,
  folds = 5
)


best_regression_k <- reg_CV$k_values[
  which.min(
    reg_CV$mean_MSE
  )
]


best_regression_k


# ------------------------------------------------------------------------------
# 41. Plot Regression CV
# ------------------------------------------------------------------------------

plot(
  reg_CV$k_values,
  reg_CV$mean_MSE,
  type = "b",
  pch = 19,
  xlab = "k",
  ylab = "Cross-Validated MSE",
  main = "k-NN Regression Cross Validation"
)


abline(
  v =
    best_regression_k,
  lty = 2
)


# ------------------------------------------------------------------------------
# 42. Final Regression Prediction
# ------------------------------------------------------------------------------

regression_test_prediction <- knn_regress(
  X_train =
    X_reg_train,
  y_train =
    y_reg_train,
  X_new =
    X_reg_test,
  k =
    best_regression_k
)


regression_test_MSE <- mean(
  (
    y_reg_test -
      regression_test_prediction
  )^2
)


regression_test_RMSE <- sqrt(
  regression_test_MSE
)


regression_test_MSE


regression_test_RMSE


# ==============================================================================
# PART XX
#
# CURSE OF DIMENSIONALITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Distance Concentration Experiment
# ------------------------------------------------------------------------------

set.seed(999)


dimensions <- c(
  1,
  2,
  5,
  10,
  20,
  50,
  100
)


distance_results <- data.frame(
  Dimension =
    dimensions,
  Nearest_Distance =
    NA_real_,
  Farthest_Distance =
    NA_real_,
  Ratio =
    NA_real_
)


for (
  i in seq_along(
    dimensions
  )
) {
  
  dimension <- dimensions[i]
  
  
  sample_points <- matrix(
    runif(
      1000 *
        dimension
    ),
    nrow = 1000,
    ncol =
      dimension
  )
  
  
  query <- rep(
    0.5,
    dimension
  )
  
  
  distances <- apply(
    sample_points,
    1,
    function(x) {
      
      euclidean_distance(
        x,
        query
      )
    }
  )
  
  
  nearest <- min(
    distances
  )
  
  
  farthest <- max(
    distances
  )
  
  
  distance_results$Nearest_Distance[i] <-
    nearest
  
  
  distance_results$Farthest_Distance[i] <-
    farthest
  
  
  distance_results$Ratio[i] <-
    nearest /
    farthest
}


distance_results


# ------------------------------------------------------------------------------
# 44. Plot Distance Concentration
# ------------------------------------------------------------------------------

plot(
  distance_results$Dimension,
  distance_results$Ratio,
  type = "b",
  pch = 19,
  xlab = "Dimension",
  ylab = "Nearest / Farthest Distance",
  main = "Distance Concentration in High Dimensions"
)


# ==============================================================================
# PART XXI
#
# IRRELEVANT FEATURES
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Demonstrate Harm from Noise Predictors
# ------------------------------------------------------------------------------

set.seed(222)


n_noise <- 400


signal_x <- rnorm(
  n_noise
)


y_noise_demo <- ifelse(
  signal_x > 0,
  1,
  0
)


noise_matrix <- matrix(
  rnorm(
    n_noise *
      20
  ),
  nrow =
    n_noise,
  ncol = 20
)


X_signal_only <- matrix(
  signal_x,
  ncol = 1
)


X_with_noise <- cbind(
  signal_x,
  noise_matrix
)


set.seed(333)


noise_train <- sample(
  seq_len(
    n_noise
  ),
  size = 280
)


noise_test <- setdiff(
  seq_len(
    n_noise
  ),
  noise_train
)


signal_prediction <- knn_classify(
  X_signal_only[
    noise_train,
    ,
    drop = FALSE
  ],
  y_noise_demo[
    noise_train
  ],
  X_signal_only[
    noise_test,
    ,
    drop = FALSE
  ],
  k = 7
)


noise_prediction <- knn_classify(
  X_with_noise[
    noise_train,
    ,
    drop = FALSE
  ],
  y_noise_demo[
    noise_train
  ],
  X_with_noise[
    noise_test,
    ,
    drop = FALSE
  ],
  k = 7
)


data.frame(
  Predictors = c(
    "Signal Only",
    "Signal + 20 Noise Variables"
  ),
  Accuracy = c(
    mean(
      signal_prediction$class ==
        y_noise_demo[
          noise_test
        ]
    ),
    mean(
      noise_prediction$class ==
        y_noise_demo[
          noise_test
        ]
    )
  )
)


# ==============================================================================
# PART XXII
#
# LOGISTIC REGRESSION COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Logistic Baseline
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = y_train,
  x1 =
    X_train[, 1],
  x2 =
    X_train[, 2]
)


test_data <- data.frame(
  x1 =
    X_test[, 1],
  x2 =
    X_test[, 2]
)


logistic_model <- glm(
  y ~ x1 + x2,
  data =
    train_data,
  family =
    binomial()
)


logistic_probability <- predict(
  logistic_model,
  newdata =
    test_data,
  type =
    "response"
)


logistic_class <- ifelse(
  logistic_probability >=
    0.5,
  1,
  0
)


logistic_accuracy <- mean(
  logistic_class ==
    y_test
)


data.frame(
  Model = c(
    "Logistic Regression",
    "k-Nearest Neighbors"
  ),
  Test_Accuracy = c(
    logistic_accuracy,
    final_accuracy
  )
)


# ==============================================================================
# PART XXIII
#
# OPTIONAL VERIFICATION WITH CLASS PACKAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Verify Classification
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "class",
    quietly = TRUE
  )
) {
  
  built_in_prediction <- class::knn(
    train =
      X_train,
    test =
      X_test,
    cl =
      factor(
        y_train
      ),
    k =
      best_k,
    prob = TRUE
  )
  
  
  built_in_accuracy <- mean(
    as.numeric(
      as.character(
        built_in_prediction
      )
    ) ==
      y_test
  )
  
  
  cat(
    "Manual k-NN accuracy:",
    round(
      final_accuracy,
      4
    ),
    "\n"
  )
  
  
  cat(
    "class::knn accuracy:",
    round(
      built_in_accuracy,
      4
    ),
    "\n"
  )
}


# ==============================================================================
# PART XXIV
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Output
# ------------------------------------------------------------------------------

cat(
  "k-Nearest Neighbors Summary\n"
)


cat(
  "---------------------------\n"
)


cat(
  "Best classification k:",
  best_k,
  "\n"
)


cat(
  "Classification test accuracy:",
  round(
    final_accuracy,
    4
  ),
  "\n"
)


cat(
  "Best regression k:",
  best_regression_k,
  "\n"
)


cat(
  "Regression test MSE:",
  round(
    regression_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Regression test RMSE:",
  round(
    regression_test_RMSE,
    4
  ),
  "\n"
)