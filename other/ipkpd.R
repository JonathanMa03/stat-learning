# ==============================================================================
# Classification with Inner Product Kernels and Pairwise Distances
# Jonathan Ma
#
# Package-free pedagogical implementation.
#
# Main ideas:
#   - Inner products and distances
#   - Gram matrices
#   - Kernel trick
#   - Linear kernel
#   - Polynomial kernel
#   - Gaussian / RBF kernel
#   - Kernel nearest-centroid classification
#   - Kernel similarity classifiers
#   - Pairwise-distance classification
#   - k-Nearest Neighbors from a distance matrix
#   - Nearest-centroid classification from distances
#   - Kernel-distance relationship
#   - Kernel PCA connection
#   - Positive semidefinite kernels
#   - Non-Euclidean dissimilarities
#
#
# Inner-product kernel:
#
#       K(x,z)
#
#       =
#
#       <phi(x), phi(z)>
#
#
# Squared distance in feature space:
#
#       ||phi(x)-phi(z)||^2
#
#       =
#
#       K(x,x)
#
#       +
#
#       K(z,z)
#
#       -
#
#       2 K(x,z)
#
#
# This means we can classify using distances in an implicit feature space
# without explicitly constructing phi(x).
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR CLASSIFICATION DATA
# ==============================================================================


set.seed(123)


n <- 500


# ------------------------------------------------------------------------------
# Two concentric classes.
#
# A linear classifier in the original coordinates will struggle because the
# decision boundary is nonlinear.
# ------------------------------------------------------------------------------

n_class_1 <- n / 2

n_class_2 <- n / 2


angle_1 <- runif(
  n_class_1,
  0,
  2 * pi
)


angle_2 <- runif(
  n_class_2,
  0,
  2 * pi
)


radius_1 <- rnorm(
  n_class_1,
  mean = 1.5,
  sd = 0.15
)


radius_2 <- rnorm(
  n_class_2,
  mean = 3.0,
  sd = 0.20
)


X_1 <- cbind(
  radius_1 *
    cos(
      angle_1
    ),
  radius_1 *
    sin(
      angle_1
    )
)


X_2 <- cbind(
  radius_2 *
    cos(
      angle_2
    ),
  radius_2 *
    sin(
      angle_2
    )
)


X <- rbind(
  X_1,
  X_2
)


y <- c(
  rep(
    -1,
    n_class_1
  ),
  rep(
    1,
    n_class_2
  )
)


colnames(X) <- c(
  "X1",
  "X2"
)


# ------------------------------------------------------------------------------
# Shuffle observations.
# ------------------------------------------------------------------------------

set.seed(456)


permutation <- sample(
  seq_len(n)
)


X <- X[
  permutation,
  ,
  drop = FALSE
]


y <- y[
  permutation
]


# ==============================================================================
# PART II
#
# VISUALIZE DATA
# ==============================================================================


plot(
  X[
    y == -1,
    1
  ],
  X[
    y == -1,
    2
  ],
  pch = 19,
  xlab = "X1",
  ylab = "X2",
  xlim = range(
    X[, 1]
  ),
  ylim = range(
    X[, 2]
  ),
  main = "Nonlinear Binary Classification"
)


points(
  X[
    y == 1,
    1
  ],
  X[
    y == 1,
    2
  ],
  pch = 1
)


legend(
  "topright",
  legend = c(
    "Class -1",
    "Class +1"
  ),
  pch = c(
    19,
    1
  )
)


# ==============================================================================
# PART III
#
# TRAIN / TEST SPLIT
# ==============================================================================


set.seed(789)


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


X_train_raw <- X[
  train_indices,
  ,
  drop = FALSE
]


X_test_raw <- X[
  test_indices,
  ,
  drop = FALSE
]


y_train <- y[
  train_indices
]


y_test <- y[
  test_indices
]


# ==============================================================================
# PART IV
#
# STANDARDIZATION
# ==============================================================================


x_means <- colMeans(
  X_train_raw
)


x_sds <- apply(
  X_train_raw,
  2,
  sd
)


X_train <- sweep(
  X_train_raw,
  2,
  x_means,
  "-"
)


X_train <- sweep(
  X_train,
  2,
  x_sds,
  "/"
)


X_test <- sweep(
  X_test_raw,
  2,
  x_means,
  "-"
)


X_test <- sweep(
  X_test,
  2,
  x_sds,
  "/"
)


# ==============================================================================
# PART V
#
# CLASSIFICATION METRICS
# ==============================================================================


classification_metrics <- function(
    truth,
    prediction
) {
  
  truth <- as.numeric(
    truth
  )
  
  
  prediction <- as.numeric(
    prediction
  )
  
  
  TP <- sum(
    truth == 1 &
      prediction == 1
  )
  
  
  TN <- sum(
    truth == -1 &
      prediction == -1
  )
  
  
  FP <- sum(
    truth == -1 &
      prediction == 1
  )
  
  
  FN <- sum(
    truth == 1 &
      prediction == -1
  )
  
  
  accuracy <- (
    TP +
      TN
  ) /
    length(
      truth
    )
  
  
  precision <- if (
    TP + FP >
    0
  ) {
    
    TP /
      (
        TP +
          FP
      )
    
  } else {
    
    NA_real_
  }
  
  
  recall <- if (
    TP + FN >
    0
  ) {
    
    TP /
      (
        TP +
          FN
      )
    
  } else {
    
    NA_real_
  }
  
  
  specificity <- if (
    TN + FP >
    0
  ) {
    
    TN /
      (
        TN +
          FP
      )
    
  } else {
    
    NA_real_
  }
  
  
  c(
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


# ==============================================================================
# PART VI
#
# INNER PRODUCTS
# ==============================================================================


# ------------------------------------------------------------------------------
# Standard Euclidean inner product:
#
#       <x,z> = x'z
# ------------------------------------------------------------------------------

inner_product <- function(
    x,
    z
) {
  
  sum(
    x *
      z
  )
}


inner_product(
  X_train[
    1,
  ],
  X_train[
    2,
  ]
)


# ==============================================================================
# PART VII
#
# EUCLIDEAN DISTANCE
# ==============================================================================


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


euclidean_distance(
  X_train[
    1,
  ],
  X_train[
    2,
  ]
)


# ==============================================================================
# PART VIII
#
# VERIFY INNER PRODUCT / DISTANCE IDENTITY
# ==============================================================================


x_example <- X_train[
  1,
]


z_example <- X_train[
  2,
]


distance_squared_direct <- sum(
  (
    x_example -
      z_example
  )^2
)


distance_squared_inner_product <-
  inner_product(
    x_example,
    x_example
  ) +
  inner_product(
    z_example,
    z_example
  ) -
  2 *
  inner_product(
    x_example,
    z_example
  )


c(
  Direct =
    distance_squared_direct,
  Inner_Product_Formula =
    distance_squared_inner_product
)


# ==============================================================================
# PART IX
#
# PAIRWISE DISTANCE MATRIX
# ==============================================================================


pairwise_squared_distance <- function(X) {
  
  X <- as.matrix(
    X
  )
  
  
  squared_norm <- rowSums(
    X^2
  )
  
  
  D2 <- outer(
    squared_norm,
    squared_norm,
    "+"
  ) -
    2 *
    X %*%
    t(
      X
    )
  
  
  pmax(
    D2,
    0
  )
}


pairwise_distance <- function(X) {
  
  sqrt(
    pairwise_squared_distance(
      X
    )
  )
}


D_train <- pairwise_distance(
  X_train
)


dim(
  D_train
)


image(
  D_train,
  axes = FALSE,
  main = "Training Pairwise Distance Matrix"
)


# ==============================================================================
# PART X
#
# CROSS-DISTANCE MATRIX
# ==============================================================================


pairwise_cross_squared_distance <- function(
    X_new,
    X_reference
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  X_reference <- as.matrix(
    X_reference
  )
  
  
  norm_new <- rowSums(
    X_new^2
  )
  
  
  norm_reference <- rowSums(
    X_reference^2
  )
  
  
  D2 <- outer(
    norm_new,
    norm_reference,
    "+"
  ) -
    2 *
    X_new %*%
    t(
      X_reference
    )
  
  
  pmax(
    D2,
    0
  )
}


pairwise_cross_distance <- function(
    X_new,
    X_reference
) {
  
  sqrt(
    pairwise_cross_squared_distance(
      X_new,
      X_reference
    )
  )
}


# ==============================================================================
# PART XI
#
# LINEAR KERNEL
# ==============================================================================


linear_kernel <- function(
    X,
    Z = X
) {
  
  as.matrix(X) %*%
    t(
      as.matrix(Z)
    )
}


K_linear <- linear_kernel(
  X_train
)


# ------------------------------------------------------------------------------
# The Gram matrix contains all pairwise inner products.
# ------------------------------------------------------------------------------

dim(
  K_linear
)


# ==============================================================================
# PART XII
#
# DISTANCE FROM A GRAM MATRIX
# ==============================================================================


gram_to_squared_distance <- function(K) {
  
  diagonal <- diag(
    K
  )
  
  
  D2 <- outer(
    diagonal,
    diagonal,
    "+"
  ) -
    2 *
    K
  
  
  pmax(
    D2,
    0
  )
}


D2_from_linear_kernel <- gram_to_squared_distance(
  K_linear
)


max(
  abs(
    D2_from_linear_kernel -
      pairwise_squared_distance(
        X_train
      )
  )
)


# For a linear kernel, kernel-induced distances equal ordinary Euclidean
# distances in standardized predictor space.


# ==============================================================================
# PART XIII
#
# POLYNOMIAL KERNEL
# ==============================================================================


# ------------------------------------------------------------------------------
# Polynomial kernel:
#
#       K(x,z)
#
#       =
#
#       (gamma x'z + c_0)^degree
# ------------------------------------------------------------------------------

polynomial_kernel <- function(
    X,
    Z = X,
    degree = 2,
    gamma = 1,
    c0 = 1
) {
  
  (
    gamma *
      as.matrix(X) %*%
      t(
        as.matrix(Z)
      ) +
      c0
  )^degree
}


K_poly <- polynomial_kernel(
  X_train,
  degree = 2,
  gamma = 1,
  c0 = 1
)


# ==============================================================================
# PART XIV
#
# RBF / GAUSSIAN KERNEL
# ==============================================================================


# ------------------------------------------------------------------------------
# Gaussian kernel:
#
#       K(x,z)
#
#       =
#
#       exp(
#           -gamma ||x-z||^2
#       )
#
#
# Nearby points:
#
#       K close to 1
#
# Distant points:
#
#       K close to 0
# ------------------------------------------------------------------------------

rbf_kernel <- function(
    X,
    Z = X,
    gamma = 1
) {
  
  D2 <- pairwise_cross_squared_distance(
    X,
    Z
  )
  
  
  exp(
    -gamma *
      D2
  )
}


K_rbf <- rbf_kernel(
  X_train,
  gamma = 1
)


summary(
  as.numeric(
    K_rbf
  )
)


# ==============================================================================
# PART XV
#
# POSITIVE SEMIDEFINITE CHECK
# ==============================================================================


# ------------------------------------------------------------------------------
# A valid inner-product kernel has a positive-semidefinite Gram matrix:
#
#       c' K c >= 0
#
# for every vector c.
#
# Numerically, all eigenvalues should be nonnegative up to roundoff error.
# ------------------------------------------------------------------------------

linear_eigenvalues <- eigen(
  K_linear,
  symmetric = TRUE,
  only.values = TRUE
)$values


poly_eigenvalues <- eigen(
  K_poly,
  symmetric = TRUE,
  only.values = TRUE
)$values


rbf_eigenvalues <- eigen(
  K_rbf,
  symmetric = TRUE,
  only.values = TRUE
)$values


min(
  linear_eigenvalues
)


min(
  poly_eigenvalues
)


min(
  rbf_eigenvalues
)


# Tiny negative values around numerical precision are not substantive.


# ==============================================================================
# PART XVI
#
# BASELINE: LINEAR LOGISTIC REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# Convert {-1,+1} to {0,1}.
# ------------------------------------------------------------------------------

y_train_binary <- as.numeric(
  y_train ==
    1
)


linear_data <- data.frame(
  Y =
    y_train_binary,
  X_train
)


linear_logistic <- glm(
  Y ~ .,
  data =
    linear_data,
  family =
    binomial()
)


linear_probability <- predict(
  linear_logistic,
  newdata =
    data.frame(
      X_test
    ),
  type = "response"
)


linear_prediction <- ifelse(
  linear_probability >=
    0.5,
  1,
  -1
)


classification_metrics(
  y_test,
  linear_prediction
)


# Linear logistic regression should struggle with concentric rings.


# ==============================================================================
# PART XVII
#
# k-NEAREST NEIGHBORS USING ONLY PAIRWISE DISTANCES
# ==============================================================================


# ------------------------------------------------------------------------------
# Notice:
#
# Once we have the distance matrix between test points and training points,
# we no longer need the original coordinates for kNN classification.
# ------------------------------------------------------------------------------

knn_from_distance <- function(
    distance_matrix,
    y_train,
    k = 5
) {
  
  distance_matrix <- as.matrix(
    distance_matrix
  )
  
  
  predictions <- integer(
    nrow(
      distance_matrix
    )
  )
  
  
  for (
    i in seq_len(
      nrow(
        distance_matrix
      )
    )
  ) {
    
    nearest <- order(
      distance_matrix[
        i,
      ]
    )[
      seq_len(k)
    ]
    
    
    vote <- mean(
      y_train[
        nearest
      ]
    )
    
    
    predictions[i] <- ifelse(
      vote >=
        0,
      1,
      -1
    )
  }
  
  
  predictions
}


# ------------------------------------------------------------------------------
# Test-to-training Euclidean distances.
# ------------------------------------------------------------------------------

D_test_train <- pairwise_cross_distance(
  X_test,
  X_train
)


knn_prediction <- knn_from_distance(
  D_test_train,
  y_train,
  k = 7
)


classification_metrics(
  y_test,
  knn_prediction
)


# ==============================================================================
# PART XVIII
#
# CROSS-VALIDATE k
# ==============================================================================


k_values <- seq(
  1,
  31,
  by = 2
)


# ------------------------------------------------------------------------------
# Leave-one-out kNN directly from the training distance matrix.
#
# Set diagonal to Inf so an observation cannot classify itself.
# ------------------------------------------------------------------------------

D_train_loocv <- D_train


diag(
  D_train_loocv
) <- Inf


loocv_accuracy <- numeric(
  length(
    k_values
  )
)


for (
  k_index in seq_along(
    k_values
  )
) {
  
  prediction <- knn_from_distance(
    D_train_loocv,
    y_train,
    k =
      k_values[
        k_index
      ]
  )
  
  
  loocv_accuracy[
    k_index
  ] <- mean(
    prediction ==
      y_train
  )
}


best_k <- k_values[
  which.max(
    loocv_accuracy
  )
]


best_k


plot(
  k_values,
  loocv_accuracy,
  type = "b",
  pch = 19,
  xlab = "k",
  ylab = "LOOCV Accuracy",
  main = "Distance-Based kNN"
)


# ==============================================================================
# PART XIX
#
# FINAL DISTANCE-BASED kNN
# ==============================================================================


knn_final_prediction <- knn_from_distance(
  D_test_train,
  y_train,
  k =
    best_k
)


knn_metrics <- classification_metrics(
  y_test,
  knn_final_prediction
)


knn_metrics


# ==============================================================================
# PART XX
#
# KERNEL-INDUCED DISTANCES
# ==============================================================================


# ------------------------------------------------------------------------------
# For kernel K:
#
#       d_K^2(x,z)
#
#       =
#
#       K(x,x)
#
#       +
#
#       K(z,z)
#
#       -
#
#       2 K(x,z)
#
#
# This is Euclidean distance in the implicit feature space phi(x).
# ------------------------------------------------------------------------------

kernel_cross_squared_distance <- function(
    K_test_train,
    K_test_diag,
    K_train_diag
) {
  
  D2 <- outer(
    K_test_diag,
    K_train_diag,
    "+"
  ) -
    2 *
    K_test_train
  
  
  pmax(
    D2,
    0
  )
}


# ==============================================================================
# PART XXI
#
# RBF KERNEL DISTANCES
# ==============================================================================


gamma <- 1


K_train_rbf <- rbf_kernel(
  X_train,
  X_train,
  gamma =
    gamma
)


K_test_train_rbf <- rbf_kernel(
  X_test,
  X_train,
  gamma =
    gamma
)


K_test_test_rbf <- rbf_kernel(
  X_test,
  X_test,
  gamma =
    gamma
)


rbf_train_diagonal <- diag(
  K_train_rbf
)


rbf_test_diagonal <- diag(
  K_test_test_rbf
)


D2_rbf_test_train <- kernel_cross_squared_distance(
  K_test_train_rbf,
  rbf_test_diagonal,
  rbf_train_diagonal
)


D_rbf_test_train <- sqrt(
  D2_rbf_test_train
)


# ==============================================================================
# PART XXII
#
# kNN IN RBF FEATURE SPACE
# ==============================================================================


rbf_knn_prediction <- knn_from_distance(
  D_rbf_test_train,
  y_train,
  k =
    best_k
)


rbf_knn_metrics <- classification_metrics(
  y_test,
  rbf_knn_prediction
)


rbf_knn_metrics


# Important subtlety:
#
# For a fixed gamma, Gaussian kernel distance is a monotone transformation of
# ordinary Euclidean distance:
#
#       d_K^2
#
#       =
#
#       2 - 2 exp(-gamma ||x-z||^2)
#
# because K(x,x)=1.
#
# Therefore nearest-neighbor RANKING is unchanged.
#
# So RBF-kNN will choose the same neighbors as Euclidean kNN.
#
# Kernelization becomes more meaningful when the classifier uses global
# geometry, class centroids, weighted similarities, margins, etc.


# ==============================================================================
# PART XXIII
#
# KERNEL NEAREST-CENTROID CLASSIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# In feature space:
#
#       mu_c
#
#       =
#
#       (1/n_c) sum_{i:y_i=c} phi(x_i)
#
#
# Classify x by smallest:
#
#       ||phi(x) - mu_c||^2
#
#
# Expand using kernels:
#
#       K(x,x)
#
#       -
#
#       (2/n_c) sum_{i in c} K(x,x_i)
#
#       +
#
#       (1/n_c^2)
#
#       sum_{i in c}
#       sum_{j in c}
#
#       K(x_i,x_j)
#
#
# No explicit phi(x) is needed.
# ------------------------------------------------------------------------------

kernel_nearest_centroid <- function(
    K_train,
    K_test_train,
    K_test_diag,
    y_train
) {
  
  classes <- sort(
    unique(
      y_train
    )
  )
  
  
  n_test <- nrow(
    K_test_train
  )
  
  
  distance_matrix <- matrix(
    0,
    nrow =
      n_test,
    ncol =
      length(
        classes
      )
  )
  
  
  colnames(
    distance_matrix
  ) <- classes
  
  
  for (
    class_index in seq_along(
      classes
    )
  ) {
    
    class_value <- classes[
      class_index
    ]
    
    
    indices <- which(
      y_train ==
        class_value
    )
    
    
    n_class <- length(
      indices
    )
    
    
    # ------------------------------------------------------------------------
    # Class centroid norm:
    #
    #       ||mu_c||^2
    # ------------------------------------------------------------------------
    
    centroid_norm_squared <- sum(
      K_train[
        indices,
        indices,
        drop = FALSE
      ]
    ) /
      (
        n_class^2
      )
    
    
    # ------------------------------------------------------------------------
    # <phi(x), mu_c>
    # ------------------------------------------------------------------------
    
    x_centroid_inner_product <- rowSums(
      K_test_train[
        ,
        indices,
        drop = FALSE
      ]
    ) /
      n_class
    
    
    # ------------------------------------------------------------------------
    # Distance to centroid.
    # ------------------------------------------------------------------------
    
    distance_matrix[
      ,
      class_index
    ] <- K_test_diag -
      2 *
      x_centroid_inner_product +
      centroid_norm_squared
  }
  
  
  predictions <- classes[
    max.col(
      -distance_matrix,
      ties.method = "first"
    )
  ]
  
  
  list(
    prediction =
      predictions,
    distance =
      distance_matrix
  )
}


# ==============================================================================
# PART XXIV
#
# LINEAR-KERNEL NEAREST CENTROID
# ==============================================================================


K_train_linear <- linear_kernel(
  X_train
)


K_test_train_linear <- linear_kernel(
  X_test,
  X_train
)


K_test_linear_diag <- rowSums(
  X_test^2
)


linear_kernel_centroid <- kernel_nearest_centroid(
  K_train_linear,
  K_test_train_linear,
  K_test_linear_diag,
  y_train
)


linear_centroid_metrics <- classification_metrics(
  y_test,
  linear_kernel_centroid$prediction
)


linear_centroid_metrics


# ==============================================================================
# PART XXV
#
# RBF-KERNEL NEAREST CENTROID
# ==============================================================================


rbf_kernel_centroid <- kernel_nearest_centroid(
  K_train_rbf,
  K_test_train_rbf,
  rbf_test_diagonal,
  y_train
)


rbf_centroid_metrics <- classification_metrics(
  y_test,
  rbf_kernel_centroid$prediction
)


rbf_centroid_metrics


# ==============================================================================
# PART XXVI
#
# POLYNOMIAL-KERNEL NEAREST CENTROID
# ==============================================================================


degree <- 2


K_train_poly <- polynomial_kernel(
  X_train,
  X_train,
  degree =
    degree,
  gamma = 1,
  c0 = 1
)


K_test_train_poly <- polynomial_kernel(
  X_test,
  X_train,
  degree =
    degree,
  gamma = 1,
  c0 = 1
)


K_test_test_poly <- polynomial_kernel(
  X_test,
  X_test,
  degree =
    degree,
  gamma = 1,
  c0 = 1
)


poly_test_diag <- diag(
  K_test_test_poly
)


poly_kernel_centroid <- kernel_nearest_centroid(
  K_train_poly,
  K_test_train_poly,
  poly_test_diag,
  y_train
)


poly_centroid_metrics <- classification_metrics(
  y_test,
  poly_kernel_centroid$prediction
)


poly_centroid_metrics


# ==============================================================================
# PART XXVII
#
# SIMPLE KERNEL SIMILARITY CLASSIFIER
# ==============================================================================


# ------------------------------------------------------------------------------
# Another simple kernel classifier compares average similarity to each class:
#
#       score_c(x)
#
#       =
#
#       (1/n_c)
#       sum_{i:y_i=c}
#       K(x,x_i)
#
#
# Choose class with greatest average similarity.
#
# This is closely related to kernel density ideas, although this exact score is
# not generally normalized as a probability density.
# ------------------------------------------------------------------------------

kernel_average_similarity_classifier <- function(
    K_test_train,
    y_train
) {
  
  classes <- sort(
    unique(
      y_train
    )
  )
  
  
  scores <- matrix(
    0,
    nrow =
      nrow(
        K_test_train
      ),
    ncol =
      length(
        classes
      )
  )
  
  
  colnames(
    scores
  ) <- classes
  
  
  for (
    class_index in seq_along(
      classes
    )
  ) {
    
    indices <- which(
      y_train ==
        classes[
          class_index
        ]
    )
    
    
    scores[
      ,
      class_index
    ] <- rowMeans(
      K_test_train[
        ,
        indices,
        drop = FALSE
      ]
    )
  }
  
  
  prediction <- classes[
    max.col(
      scores,
      ties.method = "first"
    )
  ]
  
  
  list(
    prediction =
      prediction,
    scores =
      scores
  )
}


# ==============================================================================
# PART XXVIII
#
# RBF SIMILARITY CLASSIFIER
# ==============================================================================


rbf_similarity <- kernel_average_similarity_classifier(
  K_test_train_rbf,
  y_train
)


rbf_similarity_metrics <- classification_metrics(
  y_test,
  rbf_similarity$prediction
)


rbf_similarity_metrics


# ==============================================================================
# PART XXIX
#
# CROSS-VALIDATE RBF GAMMA
# ==============================================================================


# ------------------------------------------------------------------------------
# gamma controls locality:
#
# small gamma:
#
#       broad similarities
#
# large gamma:
#
#       very local similarities
# ------------------------------------------------------------------------------

gamma_values <- c(
  0.05,
  0.10,
  0.25,
  0.50,
  1,
  2,
  5,
  10
)


# ------------------------------------------------------------------------------
# 5-fold CV for kernel nearest centroid.
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
  nrow(
    X_train
  ),
  number_folds = 5,
  seed = 321
)


gamma_cv_accuracy <- numeric(
  length(
    gamma_values
  )
)


for (
  gamma_index in seq_along(
    gamma_values
  )
) {
  
  gamma_value <- gamma_values[
    gamma_index
  ]
  
  
  fold_accuracy <- numeric(
    5
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
    
    
    X_fold_train <- X_train[
      training_indices,
      ,
      drop = FALSE
    ]
    
    
    X_fold_validation <- X_train[
      validation_indices,
      ,
      drop = FALSE
    ]
    
    
    y_fold_train <- y_train[
      training_indices
    ]
    
    
    y_fold_validation <- y_train[
      validation_indices
    ]
    
    
    K_fold_train <- rbf_kernel(
      X_fold_train,
      X_fold_train,
      gamma =
        gamma_value
    )
    
    
    K_fold_validation <- rbf_kernel(
      X_fold_validation,
      X_fold_train,
      gamma =
        gamma_value
    )
    
    
    K_fold_validation_self <- rbf_kernel(
      X_fold_validation,
      X_fold_validation,
      gamma =
        gamma_value
    )
    
    
    fold_fit <- kernel_nearest_centroid(
      K_fold_train,
      K_fold_validation,
      diag(
        K_fold_validation_self
      ),
      y_fold_train
    )
    
    
    fold_accuracy[
      fold
    ] <- mean(
      fold_fit$prediction ==
        y_fold_validation
    )
  }
  
  
  gamma_cv_accuracy[
    gamma_index
  ] <- mean(
    fold_accuracy
  )
}


gamma_results <- data.frame(
  Gamma =
    gamma_values,
  CV_Accuracy =
    gamma_cv_accuracy
)


gamma_results


best_gamma <- gamma_values[
  which.max(
    gamma_cv_accuracy
  )
]


best_gamma


plot(
  gamma_values,
  gamma_cv_accuracy,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "gamma",
  ylab = "CV Accuracy",
  main = "RBF Kernel Tuning"
)


# ==============================================================================
# PART XXX
#
# FINAL RBF KERNEL CENTROID MODEL
# ==============================================================================


K_train_rbf_final <- rbf_kernel(
  X_train,
  X_train,
  gamma =
    best_gamma
)


K_test_train_rbf_final <- rbf_kernel(
  X_test,
  X_train,
  gamma =
    best_gamma
)


K_test_self_rbf_final <- rbf_kernel(
  X_test,
  X_test,
  gamma =
    best_gamma
)


rbf_centroid_final <- kernel_nearest_centroid(
  K_train_rbf_final,
  K_test_train_rbf_final,
  diag(
    K_test_self_rbf_final
  ),
  y_train
)


final_rbf_metrics <- classification_metrics(
  y_test,
  rbf_centroid_final$prediction
)


final_rbf_metrics


# ==============================================================================
# PART XXXI
#
# DECISION REGIONS
# ==============================================================================


x1_grid <- seq(
  min(
    X_train[, 1]
  ) -
    0.5,
  max(
    X_train[, 1]
  ) +
    0.5,
  length.out = 150
)


x2_grid <- seq(
  min(
    X_train[, 2]
  ) -
    0.5,
  max(
    X_train[, 2]
  ) +
    0.5,
  length.out = 150
)


grid <- expand.grid(
  X1 =
    x1_grid,
  X2 =
    x2_grid
)


grid_matrix <- as.matrix(
  grid
)


K_grid_train <- rbf_kernel(
  grid_matrix,
  X_train,
  gamma =
    best_gamma
)


# RBF self-kernel K(x,x)=1.

grid_self_diag <- rep(
  1,
  nrow(
    grid_matrix
  )
)


grid_prediction <- kernel_nearest_centroid(
  K_train_rbf_final,
  K_grid_train,
  grid_self_diag,
  y_train
)$prediction


grid_prediction_matrix <- matrix(
  grid_prediction,
  nrow =
    length(
      x1_grid
    ),
  ncol =
    length(
      x2_grid
    )
)


image(
  x1_grid,
  x2_grid,
  grid_prediction_matrix,
  xlab = "X1",
  ylab = "X2",
  main = "RBF Kernel Nearest-Centroid Decision Regions"
)


points(
  X_train[
    y_train == -1,
    1
  ],
  X_train[
    y_train == -1,
    2
  ],
  pch = 19,
  cex = 0.5
)


points(
  X_train[
    y_train == 1,
    1
  ],
  X_train[
    y_train == 1,
    2
  ],
  pch = 1,
  cex = 0.5
)


# ==============================================================================
# PART XXXII
#
# DISTANCE-ONLY NEAREST-CENTROID CLASSIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# Suppose we no longer have coordinates X.
#
# We have only pairwise distances between training observations.
#
# Can we classify?
#
# Yes, if the distances are Euclidean, distances to class centroids can be
# recovered from pairwise distances.
#
#
# For a point x and class c with n_c members:
#
#   ||x - mu_c||^2
#
#   =
#
#   (1/n_c) sum_i ||x-x_i||^2
#
#   -
#
#   (1/(2 n_c^2))
#   sum_i sum_j ||x_i-x_j||^2
#
#
# where i,j belong to class c.
# ------------------------------------------------------------------------------

distance_only_nearest_centroid <- function(
    D_train_squared,
    D_test_train_squared,
    y_train
) {
  
  classes <- sort(
    unique(
      y_train
    )
  )
  
  
  distance_to_centroid <- matrix(
    0,
    nrow =
      nrow(
        D_test_train_squared
      ),
    ncol =
      length(
        classes
      )
  )
  
  
  colnames(
    distance_to_centroid
  ) <- classes
  
  
  for (
    class_index in seq_along(
      classes
    )
  ) {
    
    indices <- which(
      y_train ==
        classes[
          class_index
        ]
    )
    
    
    n_class <- length(
      indices
    )
    
    
    mean_test_to_class <- rowMeans(
      D_test_train_squared[
        ,
        indices,
        drop = FALSE
      ]
    )
    
    
    within_class_adjustment <- sum(
      D_train_squared[
        indices,
        indices,
        drop = FALSE
      ]
    ) /
      (
        2 *
          n_class^2
      )
    
    
    distance_to_centroid[
      ,
      class_index
    ] <- mean_test_to_class -
      within_class_adjustment
  }
  
  
  prediction <- classes[
    max.col(
      -distance_to_centroid,
      ties.method = "first"
    )
  ]
  
  
  list(
    prediction =
      prediction,
    distance =
      distance_to_centroid
  )
}


# ==============================================================================
# PART XXXIII
#
# TEST DISTANCE-ONLY CENTROID CLASSIFIER
# ==============================================================================


D2_train <- pairwise_squared_distance(
  X_train
)


D2_test_train <- pairwise_cross_squared_distance(
  X_test,
  X_train
)


distance_centroid <- distance_only_nearest_centroid(
  D2_train,
  D2_test_train,
  y_train
)


distance_centroid_metrics <- classification_metrics(
  y_test,
  distance_centroid$prediction
)


distance_centroid_metrics


# ==============================================================================
# PART XXXIV
#
# VERIFY DISTANCE-ONLY FORMULA
# ==============================================================================


# ------------------------------------------------------------------------------
# Compare distance-only centroid classifier to ordinary coordinate-based
# nearest-centroid classifier.
# ------------------------------------------------------------------------------

coordinate_centroid_classifier <- function(
    X_train,
    y_train,
    X_test
) {
  
  classes <- sort(
    unique(
      y_train
    )
  )
  
  
  centroids <- matrix(
    0,
    nrow =
      length(
        classes
      ),
    ncol =
      ncol(
        X_train
      )
  )
  
  
  for (
    class_index in seq_along(
      classes
    )
  ) {
    
    centroids[
      class_index,
    ] <- colMeans(
      X_train[
        y_train ==
          classes[
            class_index
          ],
        ,
        drop = FALSE
      ]
    )
  }
  
  
  distance <- pairwise_cross_squared_distance(
    X_test,
    centroids
  )
  
  
  classes[
    max.col(
      -distance,
      ties.method = "first"
    )
  ]
}


coordinate_centroid_prediction <- coordinate_centroid_classifier(
  X_train,
  y_train,
  X_test
)


mean(
  coordinate_centroid_prediction ==
    distance_centroid$prediction
)


# For Euclidean distances, they should agree up to numerical/tie effects.


# ==============================================================================
# PART XXXV
#
# DOUBLE-CENTERING DISTANCES INTO INNER PRODUCTS
# ==============================================================================


# ------------------------------------------------------------------------------
# Classical MDS identity:
#
#       B
#
#       =
#
#       -1/2 J D^2 J
#
#
# where:
#
#       J = I - 11'/n
#
#
# If D is Euclidean, B is the centered Gram matrix:
#
#       B = X_c X_c'
# ------------------------------------------------------------------------------

distance_to_gram <- function(
    D_squared
) {
  
  n <- nrow(
    D_squared
  )
  
  
  J <- diag(n) -
    matrix(
      1 / n,
      nrow = n,
      ncol = n
    )
  
  
  -0.5 *
    J %*%
    D_squared %*%
    J
}


B_from_distance <- distance_to_gram(
  D2_train
)


X_train_centered <- scale(
  X_train,
  center = TRUE,
  scale = FALSE
)


B_direct <- X_train_centered %*%
  t(
    X_train_centered
  )


max(
  abs(
    B_from_distance -
      B_direct
  )
)


# ==============================================================================
# PART XXXVI
#
# KERNEL TO DISTANCE AND DISTANCE TO KERNEL
# ==============================================================================


# ------------------------------------------------------------------------------
# These are two sides of the same geometry.
#
# Kernel -> squared distance:
#
#       d_ij^2
#
#       =
#
#       K_ii + K_jj - 2 K_ij
#
#
# Euclidean distance -> centered kernel:
#
#       K_c
#
#       =
#
#       -1/2 J D^2 J
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XXXVII
#
# NON-EUCLIDEAN DISTANCE EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# Not every dissimilarity matrix corresponds to Euclidean geometry.
#
# Manhattan distance is a valid metric, but its squared-distance matrix does
# not necessarily produce a positive-semidefinite centered Gram matrix.
# ------------------------------------------------------------------------------

pairwise_manhattan <- function(X) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  D <- matrix(
    0,
    nrow = n,
    ncol = n
  )
  
  
  for (
    i in seq_len(n)
  ) {
    
    for (
      j in seq_len(n)
    ) {
      
      D[
        i,
        j
      ] <- sum(
        abs(
          X[
            i,
          ] -
            X[
              j,
            ]
        )
      )
    }
  }
  
  
  D
}


D_manhattan <- pairwise_manhattan(
  X_train[
    1:100,
    ,
    drop = FALSE
  ]
)


B_manhattan <- distance_to_gram(
  D_manhattan^2
)


manhattan_eigenvalues <- eigen(
  B_manhattan,
  symmetric = TRUE,
  only.values = TRUE
)$values


min(
  manhattan_eigenvalues
)


# Negative eigenvalues indicate that this squared dissimilarity geometry is not
# exactly representable as ordinary Euclidean coordinates after centering.


# ==============================================================================
# PART XXXVIII
#
# CLASSIFICATION WITH AN ARBITRARY DISTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# kNN does NOT require the distance to arise from an inner product.
#
# It only needs a useful notion of proximity.
#
# Therefore pairwise-distance methods can work with Manhattan distance,
# edit distance, graph distance, etc.
# ------------------------------------------------------------------------------

cross_manhattan <- function(
    X_new,
    X_reference
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  X_reference <- as.matrix(
    X_reference
  )
  
  
  D <- matrix(
    0,
    nrow =
      nrow(
        X_new
      ),
    ncol =
      nrow(
        X_reference
      )
  )
  
  
  for (
    i in seq_len(
      nrow(
        X_new
      )
    )
  ) {
    
    for (
      j in seq_len(
        nrow(
          X_reference
        )
      )
    ) {
      
      D[
        i,
        j
      ] <- sum(
        abs(
          X_new[
            i,
          ] -
            X_reference[
              j,
            ]
        )
      )
    }
  }
  
  
  D
}


D_manhattan_test_train <- cross_manhattan(
  X_test,
  X_train
)


manhattan_knn_prediction <- knn_from_distance(
  D_manhattan_test_train,
  y_train,
  k =
    best_k
)


manhattan_knn_metrics <- classification_metrics(
  y_test,
  manhattan_knn_prediction
)


manhattan_knn_metrics


# ==============================================================================
# PART XXXIX
#
# SIMILARITY FROM DISTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# A common construction:
#
#       K_ij = exp(-gamma d_ij^2)
#
#
# For Euclidean distance this is the Gaussian RBF kernel.
#
# For arbitrary dissimilarities, however, exp(-gamma d^2) is NOT automatically
# guaranteed to be positive semidefinite.
# ------------------------------------------------------------------------------

distance_to_rbf_similarity <- function(
    D,
    gamma = 1
) {
  
  exp(
    -gamma *
      D^2
  )
}


K_from_euclidean_distance <- distance_to_rbf_similarity(
  D_train,
  gamma = 1
)


max(
  abs(
    K_from_euclidean_distance -
      K_rbf
  )
)


# ==============================================================================
# PART XL
#
# INDEFINITE SIMILARITY EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# Apply the same formula to Manhattan distances and check PSD.
# ------------------------------------------------------------------------------

K_manhattan_similarity <- distance_to_rbf_similarity(
  D_manhattan,
  gamma = 1
)


manhattan_kernel_eigenvalues <- eigen(
  K_manhattan_similarity,
  symmetric = TRUE,
  only.values = TRUE
)$values


min(
  manhattan_kernel_eigenvalues
)


# Depending on the distance and sample, the matrix may or may not be PSD.
#
# The broader lesson:
#
#       a useful similarity is not automatically a valid Mercer kernel.


# ==============================================================================
# PART XLI
#
# KERNEL PCA CONNECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# Kernel PCA begins with the Gram matrix:
#
#       K_ij = K(x_i,x_j)
#
# It then centers K and eigendecomposes it.
#
# So the same kernel geometry used for classification can also be used for
# nonlinear dimension reduction.
# ------------------------------------------------------------------------------

center_kernel <- function(K) {
  
  n <- nrow(
    K
  )
  
  
  J <- diag(n) -
    matrix(
      1 / n,
      nrow = n,
      ncol = n
    )
  
  
  J %*%
    K %*%
    J
}


K_rbf_centered <- center_kernel(
  K_train_rbf_final
)


kpca_eigen <- eigen(
  K_rbf_centered,
  symmetric = TRUE
)


positive_indices <- which(
  kpca_eigen$values >
    1e-8
)


kpca_scores <- kpca_eigen$vectors[
  ,
  positive_indices[
    1:2
  ],
  drop = FALSE
] %*%
  diag(
    sqrt(
      kpca_eigen$values[
        positive_indices[
          1:2
        ]
      ]
    )
  )


plot(
  kpca_scores[
    y_train == -1,
    1
  ],
  kpca_scores[
    y_train == -1,
    2
  ],
  pch = 19,
  xlab = "Kernel PC1",
  ylab = "Kernel PC2",
  main = "RBF Kernel PCA Representation"
)


points(
  kpca_scores[
    y_train == 1,
    1
  ],
  kpca_scores[
    y_train == 1,
    2
  ],
  pch = 1
)


# ==============================================================================
# PART XLII
#
# SIMPLE KERNEL RIDGE CLASSIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# A classifier can also be written directly as a linear combination of kernel
# similarities:
#
#       f(x)
#
#       =
#
#       sum_i alpha_i K(x, x_i)
#
#
# Kernel ridge solves:
#
#       alpha
#
#       =
#
#       (K + lambda I)^(-1) y
#
#
# This is regression on {-1,+1} labels followed by sign thresholding.
# ------------------------------------------------------------------------------

kernel_ridge_fit <- function(
    K_train,
    y,
    lambda
) {
  
  n <- nrow(
    K_train
  )
  
  
  alpha <- solve(
    K_train +
      lambda *
      diag(n),
    y
  )
  
  
  list(
    alpha =
      as.numeric(
        alpha
      ),
    lambda =
      lambda
  )
}


kernel_ridge_predict <- function(
    fit,
    K_test_train
) {
  
  score <- as.numeric(
    K_test_train %*%
      fit$alpha
  )
  
  
  prediction <- ifelse(
    score >=
      0,
    1,
    -1
  )
  
  
  list(
    score =
      score,
    prediction =
      prediction
  )
}


# ==============================================================================
# PART XLIII
#
# RBF KERNEL RIDGE
# ==============================================================================


kernel_ridge_model <- kernel_ridge_fit(
  K_train_rbf_final,
  y_train,
  lambda = 1
)


kernel_ridge_result <- kernel_ridge_predict(
  kernel_ridge_model,
  K_test_train_rbf_final
)


kernel_ridge_metrics <- classification_metrics(
  y_test,
  kernel_ridge_result$prediction
)


kernel_ridge_metrics


# ==============================================================================
# PART XLIV
#
# CROSS-VALIDATE KERNEL RIDGE LAMBDA
# ==============================================================================


lambda_values <- c(
  0.001,
  0.01,
  0.1,
  0.5,
  1,
  5,
  10,
  50
)


kernel_ridge_cv_accuracy <- numeric(
  length(
    lambda_values
  )
)


# ------------------------------------------------------------------------------
# Use fixed best gamma from earlier only for pedagogy.
#
# In a full production workflow, gamma and lambda should be tuned jointly
# inside the same cross-validation procedure.
# ------------------------------------------------------------------------------

for (
  lambda_index in seq_along(
    lambda_values
  )
) {
  
  lambda_value <- lambda_values[
    lambda_index
  ]
  
  
  fold_accuracy <- numeric(
    5
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
    
    
    X_fold_train <- X_train[
      training_indices,
      ,
      drop = FALSE
    ]
    
    
    X_fold_validation <- X_train[
      validation_indices,
      ,
      drop = FALSE
    ]
    
    
    y_fold_train <- y_train[
      training_indices
    ]
    
    
    y_fold_validation <- y_train[
      validation_indices
    ]
    
    
    K_fold_train <- rbf_kernel(
      X_fold_train,
      X_fold_train,
      gamma =
        best_gamma
    )
    
    
    K_fold_validation <- rbf_kernel(
      X_fold_validation,
      X_fold_train,
      gamma =
        best_gamma
    )
    
    
    fit <- kernel_ridge_fit(
      K_fold_train,
      y_fold_train,
      lambda =
        lambda_value
    )
    
    
    prediction <- kernel_ridge_predict(
      fit,
      K_fold_validation
    )$prediction
    
    
    fold_accuracy[
      fold
    ] <- mean(
      prediction ==
        y_fold_validation
    )
  }
  
  
  kernel_ridge_cv_accuracy[
    lambda_index
  ] <- mean(
    fold_accuracy
  )
}


kernel_ridge_lambda_results <- data.frame(
  Lambda =
    lambda_values,
  CV_Accuracy =
    kernel_ridge_cv_accuracy
)


kernel_ridge_lambda_results


best_lambda <- lambda_values[
  which.max(
    kernel_ridge_cv_accuracy
  )
]


# ==============================================================================
# PART XLV
#
# FINAL KERNEL RIDGE MODEL
# ==============================================================================


kernel_ridge_final <- kernel_ridge_fit(
  K_train_rbf_final,
  y_train,
  lambda =
    best_lambda
)


kernel_ridge_final_prediction <- kernel_ridge_predict(
  kernel_ridge_final,
  K_test_train_rbf_final
)


kernel_ridge_final_metrics <- classification_metrics(
  y_test,
  kernel_ridge_final_prediction$prediction
)


kernel_ridge_final_metrics


# ==============================================================================
# PART XLVI
#
# DECISION FUNCTION FOR KERNEL RIDGE
# ==============================================================================


K_grid_train_final <- rbf_kernel(
  grid_matrix,
  X_train,
  gamma =
    best_gamma
)


grid_kernel_ridge_score <- as.numeric(
  K_grid_train_final %*%
    kernel_ridge_final$alpha
)


grid_kernel_ridge_prediction <- ifelse(
  grid_kernel_ridge_score >=
    0,
  1,
  -1
)


score_matrix <- matrix(
  grid_kernel_ridge_score,
  nrow =
    length(
      x1_grid
    ),
  ncol =
    length(
      x2_grid
    )
)


contour(
  x1_grid,
  x2_grid,
  score_matrix,
  levels = 0,
  drawlabels = FALSE,
  xlab = "X1",
  ylab = "X2",
  main = "Kernel Ridge Decision Boundary"
)


points(
  X_train[
    y_train == -1,
    1
  ],
  X_train[
    y_train == -1,
    2
  ],
  pch = 19,
  cex = 0.5
)


points(
  X_train[
    y_train == 1,
    1
  ],
  X_train[
    y_train == 1,
    2
  ],
  pch = 1,
  cex = 0.5
)


# ==============================================================================
# PART XLVII
#
# KERNEL MATRIX AS FEATURES
# ==============================================================================


# ------------------------------------------------------------------------------
# One interpretation of a kernel classifier:
#
# Each observation x is represented by its similarities to all training points:
#
#       phi_empirical(x)
#
#       =
#
#       (
#           K(x,x_1),
#           ...
#           K(x,x_n)
#       )
#
#
# Kernel ridge then forms a weighted linear combination of those similarities.
# ------------------------------------------------------------------------------

dim(
  K_test_train_rbf_final
)


# ==============================================================================
# PART XLVIII
#
# DISTANCE PROFILE AS FEATURES
# ==============================================================================


# ------------------------------------------------------------------------------
# Similarly, an observation may be represented only by its distances to the
# training observations:
#
#       (
#           d(x,x_1),
#           ...
#           d(x,x_n)
#       )
#
#
# This becomes useful when original coordinates do not exist.
# ------------------------------------------------------------------------------

dim(
  D_test_train
)


# ==============================================================================
# PART XLIX
#
# CLASSIFICATION FROM DISTANCE-TO-REFERENCE FEATURES
# ==============================================================================


# ------------------------------------------------------------------------------
# Use each observation's distances to a small set of reference observations as
# ordinary derived predictors.
#
# This is a simple example of converting relational data into coordinates.
# ------------------------------------------------------------------------------

set.seed(2026)


number_references <- 20


reference_indices <- sample(
  seq_len(
    nrow(
      X_train
    )
  ),
  number_references
)


D_train_reference <- D_train[
  ,
  reference_indices,
  drop = FALSE
]


D_test_reference <- D_test_train[
  ,
  reference_indices,
  drop = FALSE
]


distance_feature_data <- data.frame(
  Y =
    y_train_binary,
  D_train_reference
)


distance_feature_logistic <- glm(
  Y ~ .,
  data =
    distance_feature_data,
  family =
    binomial()
)


distance_feature_probability <- predict(
  distance_feature_logistic,
  newdata =
    data.frame(
      D_test_reference
    ),
  type = "response"
)


distance_feature_prediction <- ifelse(
  distance_feature_probability >=
    0.5,
  1,
  -1
)


distance_feature_metrics <- classification_metrics(
  y_test,
  distance_feature_prediction
)


distance_feature_metrics


# ==============================================================================
# PART L
#
# CLASSIFICATION COMPARISON
# ==============================================================================


comparison <- data.frame(
  Method = c(
    "Linear Logistic",
    "Euclidean kNN",
    "Manhattan kNN",
    "Linear Kernel Centroid",
    "Polynomial Kernel Centroid",
    "RBF Kernel Centroid",
    "RBF Similarity Classifier",
    "RBF Kernel Ridge",
    "Distance-Feature Logistic"
  ),
  Accuracy = c(
    classification_metrics(
      y_test,
      linear_prediction
    )["Accuracy"],
    knn_metrics["Accuracy"],
    manhattan_knn_metrics["Accuracy"],
    linear_centroid_metrics["Accuracy"],
    poly_centroid_metrics["Accuracy"],
    final_rbf_metrics["Accuracy"],
    rbf_similarity_metrics["Accuracy"],
    kernel_ridge_final_metrics["Accuracy"],
    distance_feature_metrics["Accuracy"]
  )
)


comparison


# ==============================================================================
# PART LI
#
# RANDOM PROJECTION / ROTATION INVARIANCE OF DISTANCES
# ==============================================================================


# ------------------------------------------------------------------------------
# Euclidean distance is invariant under orthogonal rotations.
#
# If:
#
#       Q'Q = I
#
# then:
#
#       ||xQ - zQ||
#
#       =
#
#       ||x-z||
# ------------------------------------------------------------------------------

set.seed(111)


random_matrix <- matrix(
  rnorm(
    4
  ),
  nrow = 2
)


Q <- qr.Q(
  qr(
    random_matrix
  )
)


X_rotated <- X_train %*%
  Q


D_rotated <- pairwise_distance(
  X_rotated
)


max(
  abs(
    D_rotated -
      D_train
  )
)


# ==============================================================================
# PART LII
#
# KERNEL SCALE SENSITIVITY
# ==============================================================================


# ------------------------------------------------------------------------------
# RBF kernels are highly scale-sensitive.
#
# Without standardization, one large-scale predictor can dominate:
#
#       ||x-z||^2
#
# and therefore the kernel.
# ------------------------------------------------------------------------------

set.seed(555)


X_scale_demo <- cbind(
  X_train,
  Huge_Noise =
    rnorm(
      nrow(
        X_train
      ),
      sd = 100
    )
)


K_bad_scale <- rbf_kernel(
  X_scale_demo,
  gamma = 1
)


summary(
  K_bad_scale[
    upper.tri(
      K_bad_scale
    )
  ]
)


# Most off-diagonal similarities may collapse near zero.


# ==============================================================================
# PART LIII
#
# MEDIAN DISTANCE HEURISTIC
# ==============================================================================


# ------------------------------------------------------------------------------
# One common heuristic chooses kernel scale based on a typical pairwise
# distance.
#
# If:
#
#       K(x,z) = exp(-gamma ||x-z||^2)
#
# one possible heuristic is:
#
#       gamma = 1 / median(||x_i-x_j||^2)
#
#
# Exact conventions differ.
# ------------------------------------------------------------------------------

upper_squared_distances <- D2_train[
  upper.tri(
    D2_train
  )
]


median_squared_distance <- median(
  upper_squared_distances
)


gamma_median <- 1 /
  median_squared_distance


gamma_median


# ==============================================================================
# PART LIV
#
# CLASSIFICATION WITH ONLY A GRAM MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# Important conceptual experiment:
#
# Once K_train and K_test_train are available, kernel ridge does not need the
# original X coordinates at all.
# ------------------------------------------------------------------------------

kernel_only_fit <- kernel_ridge_fit(
  K_train_rbf_final,
  y_train,
  lambda =
    best_lambda
)


kernel_only_prediction <- kernel_ridge_predict(
  kernel_only_fit,
  K_test_train_rbf_final
)$prediction


mean(
  kernel_only_prediction ==
    kernel_ridge_final_prediction$prediction
)


# ==============================================================================
# PART LV
#
# CLASSIFICATION WITH ONLY A DISTANCE MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# Similarly, kNN needs only:
#
#       D_test_train
#
# and:
#
#       y_train.
#
# Original coordinates are unnecessary.
# ------------------------------------------------------------------------------

distance_only_prediction <- knn_from_distance(
  D_test_train,
  y_train,
  k =
    best_k
)


mean(
  distance_only_prediction ==
    knn_final_prediction
)


# ==============================================================================
# PART LVI
#
# WHEN DISTANCES ARE MORE NATURAL THAN VECTORS
# ==============================================================================


# Examples:
#
#   - strings:
#
#       edit distance
#
#   - DNA sequences:
#
#       sequence alignment distance
#
#   - graphs:
#
#       graph edit / shortest-path based distances
#
#   - shapes:
#
#       Procrustes distance
#
#   - probability distributions:
#
#       Wasserstein distance
#
#   - time series:
#
#       dynamic time warping
#
#
# In such problems, an observation may not naturally be a vector in R^p.
#
# A pairwise dissimilarity matrix can therefore be the primary data object.


# ==============================================================================
# PART LVII
#
# WHEN KERNELS ARE MORE NATURAL THAN EXPLICIT FEATURES
# ==============================================================================


# Examples:
#
#   - polynomial features
#   - string kernels
#   - graph kernels
#   - sequence kernels
#   - diffusion kernels
#
#
# Rather than explicitly construct:
#
#       phi(x)
#
# we calculate:
#
#       K(x,z)
#
# directly.
#
# This is the kernel trick.


# ==============================================================================
# PART LVIII
#
# KEY RELATIONSHIPS
# ==============================================================================


cat(
  "\nCore relationships\n"
)


cat(
  "------------------\n"
)


cat(
  "Euclidean squared distance:\n"
)


cat(
  "d^2(x,z) = <x,x> + <z,z> - 2<x,z>\n\n"
)


cat(
  "Kernel feature-space distance:\n"
)


cat(
  "d_K^2(x,z) = K(x,x) + K(z,z) - 2K(x,z)\n\n"
)


cat(
  "Centered Gram matrix from Euclidean distances:\n"
)


cat(
  "K_c = -1/2 J D^2 J\n"
)


# ==============================================================================
# PART LIX
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nClassification with Kernels and Pairwise Distances Summary\n"
)


cat(
  "----------------------------------------------------------\n"
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
  "Best kNN k:",
  best_k,
  "\n"
)


cat(
  "Best RBF gamma:",
  best_gamma,
  "\n"
)


cat(
  "Best kernel-ridge lambda:",
  best_lambda,
  "\n"
)


cat(
  "\nModel comparison:\n"
)


print(
  comparison
)