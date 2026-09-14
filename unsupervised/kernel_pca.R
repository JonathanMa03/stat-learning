# ==============================================================================
# Kernel Principal Components Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Nonlinear dimensionality reduction
#   - Kernel trick
#   - Gram matrix
#   - Feature-space centering
#   - Kernel eigenvectors
#   - Kernel principal component scores
#   - RBF / Gaussian kernel
#   - Polynomial kernel
#   - Relationship to ordinary PCA
#   - Out-of-sample transformation
#
#
# Ordinary PCA:
#
#       X -> covariance matrix -> eigenvectors
#
#
# Kernel PCA:
#
#       x_i -> phi(x_i)
#
#       K_ij = <phi(x_i), phi(x_j)>
#
#
# We never need to construct phi(x) explicitly.
#
# For a centered kernel matrix K_c:
#
#       K_c alpha_j = lambda_j alpha_j
#
#
# The score of training observation i on kernel PC j is proportional to:
#
#       sqrt(lambda_j) * alpha_ij
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Two Concentric Rings
# ------------------------------------------------------------------------------

set.seed(123)

n_inner <- 180
n_outer <- 220


theta_inner <- runif(
  n_inner,
  0,
  2 * pi
)


theta_outer <- runif(
  n_outer,
  0,
  2 * pi
)


radius_inner <- 1 +
  rnorm(
    n_inner,
    sd = 0.08
  )


radius_outer <- 2.5 +
  rnorm(
    n_outer,
    sd = 0.12
  )


inner <- cbind(
  radius_inner *
    cos(
      theta_inner
    ),
  radius_inner *
    sin(
      theta_inner
    )
)


outer <- cbind(
  radius_outer *
    cos(
      theta_outer
    ),
  radius_outer *
    sin(
      theta_outer
    )
)


X <- rbind(
  inner,
  outer
)


true_group <- c(
  rep(
    1,
    n_inner
  ),
  rep(
    2,
    n_outer
  )
)


colnames(
  X
) <- c(
  "X1",
  "X2"
)


n <- nrow(
  X
)


# ------------------------------------------------------------------------------
# 2. Plot
# ------------------------------------------------------------------------------

plot(
  X,
  pch = true_group,
  xlab = "X1",
  ylab = "X2",
  main = "Nonlinear Ring Data"
)


# ==============================================================================
# PART II
#
# ORDINARY PCA BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Ordinary PCA
# ------------------------------------------------------------------------------

X_centered <- sweep(
  X,
  2,
  colMeans(X),
  "-"
)


ordinary_PCA <- svd(
  X_centered
)


ordinary_scores <- X_centered %*%
  ordinary_PCA$v


# ------------------------------------------------------------------------------
# 4. First Two Ordinary PCs
# ------------------------------------------------------------------------------

plot(
  ordinary_scores[
    ,
    1
  ],
  ordinary_scores[
    ,
    2
  ],
  pch = true_group,
  xlab = "PC1",
  ylab = "PC2",
  main = "Ordinary PCA"
)


# PCA cannot unfold concentric rings because the structure is nonlinear.


# ==============================================================================
# PART III
#
# KERNEL FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Linear Kernel
# ------------------------------------------------------------------------------

linear_kernel <- function(
    X,
    Y = NULL
) {
  
  X <- as.matrix(
    X
  )
  
  
  if (
    is.null(
      Y
    )
  ) {
    
    Y <- X
    
  } else {
    
    Y <- as.matrix(
      Y
    )
  }
  
  
  X %*%
    t(Y)
}


# ------------------------------------------------------------------------------
# 6. Polynomial Kernel
# ------------------------------------------------------------------------------

polynomial_kernel <- function(
    X,
    Y = NULL,
    degree = 2,
    constant = 1
) {
  
  if (
    is.null(
      Y
    )
  ) {
    
    Y <- X
  }
  
  
  (
    X %*%
      t(Y) +
      constant
  )^degree
}


# ------------------------------------------------------------------------------
# 7. Squared Euclidean Distance Matrix
# ------------------------------------------------------------------------------

squared_distance_matrix <- function(
    X,
    Y = NULL
) {
  
  X <- as.matrix(
    X
  )
  
  
  if (
    is.null(
      Y
    )
  ) {
    
    Y <- X
    
  } else {
    
    Y <- as.matrix(
      Y
    )
  }
  
  
  X_squared <- rowSums(
    X^2
  )
  
  
  Y_squared <- rowSums(
    Y^2
  )
  
  
  distances <- outer(
    X_squared,
    Y_squared,
    "+"
  ) -
    2 *
    X %*%
    t(Y)
  
  
  pmax(
    distances,
    0
  )
}


# ------------------------------------------------------------------------------
# 8. Gaussian / RBF Kernel
# ------------------------------------------------------------------------------

rbf_kernel <- function(
    X,
    Y = NULL,
    gamma = 1
) {
  
  exp(
    -gamma *
      squared_distance_matrix(
        X,
        Y
      )
  )
}


# ==============================================================================
# PART IV
#
# BUILD GRAM MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. RBF Kernel Matrix
# ------------------------------------------------------------------------------

gamma <- 1


K <- rbf_kernel(
  X,
  gamma =
    gamma
)


dim(
  K
)


K[
  1:5,
  1:5
]


# ==============================================================================
# PART V
#
# CENTER KERNEL MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Manual Kernel Centering
# ------------------------------------------------------------------------------

center_kernel_matrix <- function(
    K
) {
  
  n <- nrow(
    K
  )
  
  
  one_n <- matrix(
    1 /
      n,
    nrow =
      n,
    ncol =
      n
  )
  
  
  K_centered <- K -
    one_n %*%
    K -
    K %*%
    one_n +
    one_n %*%
    K %*%
    one_n
  
  
  K_centered
}


K_centered <- center_kernel_matrix(
  K
)


# ------------------------------------------------------------------------------
# 11. Verify Row and Column Means Near Zero
# ------------------------------------------------------------------------------

max(
  abs(
    rowMeans(
      K_centered
    )
  )
)


max(
  abs(
    colMeans(
      K_centered
    )
  )
)


# ==============================================================================
# PART VI
#
# EIGENDECOMPOSITION
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Kernel Eigenvectors
# ------------------------------------------------------------------------------

kernel_eigen <- eigen(
  K_centered,
  symmetric = TRUE
)


kernel_eigenvalues <- kernel_eigen$values


kernel_eigenvectors <- kernel_eigen$vectors


# ------------------------------------------------------------------------------
# 13. Keep Positive Eigenvalues
# ------------------------------------------------------------------------------

positive <- kernel_eigenvalues >
  1e-10


kernel_eigenvalues <- kernel_eigenvalues[
  positive
]


kernel_eigenvectors <- kernel_eigenvectors[
  ,
  positive,
  drop = FALSE
]


length(
  kernel_eigenvalues
)


# ==============================================================================
# PART VII
#
# NORMALIZE KERNEL EIGENVECTORS
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Feature-Space Normalization
# ------------------------------------------------------------------------------

alpha <- kernel_eigenvectors


for (
  j in seq_len(
    ncol(alpha)
  )
) {
  
  alpha[
    ,
    j
  ] <- alpha[
    ,
    j
  ] /
    sqrt(
      kernel_eigenvalues[j]
    )
}


# ==============================================================================
# PART VIII
#
# TRAINING KERNEL PC SCORES
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Scores
# ------------------------------------------------------------------------------

kernel_scores <- K_centered %*%
  alpha


colnames(
  kernel_scores
) <- paste0(
  "KPC",
  seq_len(
    ncol(
      kernel_scores
    )
  )
)


head(
  kernel_scores[
    ,
    1:3,
    drop = FALSE
  ]
)


# Equivalent up to sign to:
#
#       eigenvector * sqrt(eigenvalue)


# ------------------------------------------------------------------------------
# 16. Verify Equivalent Formula
# ------------------------------------------------------------------------------

scores_alternative <- sweep(
  kernel_eigenvectors,
  2,
  sqrt(
    kernel_eigenvalues
  ),
  "*"
)


max(
  abs(
    abs(
      kernel_scores
    ) -
      abs(
        scores_alternative
      )
  )
)


# ==============================================================================
# PART IX
#
# VISUALIZE KERNEL PCS
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. First Two Kernel Principal Components
# ------------------------------------------------------------------------------

plot(
  kernel_scores[
    ,
    1
  ],
  kernel_scores[
    ,
    2
  ],
  pch = true_group,
  xlab = "Kernel PC1",
  ylab = "Kernel PC2",
  main = "RBF Kernel PCA"
)


# ------------------------------------------------------------------------------
# 18. Individual Kernel Components
# ------------------------------------------------------------------------------

plot(
  kernel_scores[
    ,
    1
  ],
  kernel_scores[
    ,
    2
  ],
  pch = 19,
  cex = 0.6,
  xlab = "KPC1",
  ylab = "KPC2",
  main = "Kernel PCA Embedding"
)


# ==============================================================================
# PART X
#
# EIGENVALUE / VARIANCE STRUCTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Kernel Eigenvalues
# ------------------------------------------------------------------------------

plot(
  seq_along(
    kernel_eigenvalues
  ),
  kernel_eigenvalues,
  type = "b",
  pch = 19,
  xlab = "Kernel Principal Component",
  ylab = "Kernel Eigenvalue",
  main = "Kernel PCA Eigenvalues"
)


# ------------------------------------------------------------------------------
# 20. Normalized Kernel Variance
# ------------------------------------------------------------------------------

kernel_variance_proportion <- kernel_eigenvalues /
  sum(
    kernel_eigenvalues
  )


kernel_cumulative_variance <- cumsum(
  kernel_variance_proportion
)


plot(
  seq_along(
    kernel_cumulative_variance
  ),
  kernel_cumulative_variance,
  type = "b",
  pch = 19,
  ylim = c(
    0,
    1
  ),
  xlab = "Number of Kernel PCs",
  ylab = "Cumulative Kernel Variance",
  main = "Kernel PCA Cumulative Variance"
)


# Important:
#
# This is variance in the implicit FEATURE SPACE, not directly variance in the
# original X coordinates.


# ==============================================================================
# PART XI
#
# GENERAL MANUAL KERNEL PCA FUNCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Fit Kernel PCA
# ------------------------------------------------------------------------------

fit_kernel_pca <- function(
    X,
    kernel = c(
      "rbf",
      "linear",
      "polynomial"
    ),
    gamma = 1,
    degree = 2,
    constant = 1,
    tolerance = 1e-10
) {
  
  X <- as.matrix(
    X
  )
  
  
  kernel <- match.arg(
    kernel
  )
  
  
  if (
    kernel ==
    "rbf"
  ) {
    
    K <- rbf_kernel(
      X,
      gamma =
        gamma
    )
    
  } else if (
    kernel ==
    "linear"
  ) {
    
    K <- linear_kernel(
      X
    )
    
  } else {
    
    K <- polynomial_kernel(
      X,
      degree =
        degree,
      constant =
        constant
    )
  }
  
  
  n <- nrow(
    X
  )
  
  
  row_means <- rowMeans(
    K
  )
  
  
  grand_mean <- mean(
    K
  )
  
  
  K_centered <- K -
    matrix(
      row_means,
      nrow = n,
      ncol = n
    ) -
    matrix(
      row_means,
      nrow = n,
      ncol = n,
      byrow = TRUE
    ) +
    grand_mean
  
  
  decomposition <- eigen(
    K_centered,
    symmetric = TRUE
  )
  
  
  eigenvalues <- decomposition$values
  
  
  eigenvectors <- decomposition$vectors
  
  
  keep <- eigenvalues >
    tolerance
  
  
  eigenvalues <- eigenvalues[
    keep
  ]
  
  
  eigenvectors <- eigenvectors[
    ,
    keep,
    drop = FALSE
  ]
  
  
  alpha <- eigenvectors
  
  
  for (
    j in seq_len(
      ncol(alpha)
    )
  ) {
    
    alpha[
      ,
      j
    ] <- alpha[
      ,
      j
    ] /
      sqrt(
        eigenvalues[j]
      )
  }
  
  
  scores <- K_centered %*%
    alpha
  
  
  colnames(
    scores
  ) <- paste0(
    "KPC",
    seq_len(
      ncol(scores)
    )
  )
  
  
  list(
    X_train =
      X,
    kernel =
      kernel,
    gamma =
      gamma,
    degree =
      degree,
    constant =
      constant,
    K =
      K,
    K_centered =
      K_centered,
    training_kernel_row_means =
      row_means,
    training_kernel_grand_mean =
      grand_mean,
    eigenvalues =
      eigenvalues,
    eigenvectors =
      eigenvectors,
    alpha =
      alpha,
    scores =
      scores,
    variance_proportion =
      eigenvalues /
      sum(
        eigenvalues
      ),
    cumulative_variance =
      cumsum(
        eigenvalues /
          sum(
            eigenvalues
          )
      )
  )
}


# ------------------------------------------------------------------------------
# 22. Fit General Model
# ------------------------------------------------------------------------------

KPCA_model <- fit_kernel_pca(
  X,
  kernel = "rbf",
  gamma = 1
)


# ==============================================================================
# PART XII
#
# OUT-OF-SAMPLE KERNEL CENTERING
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Kernel Between New and Training Observations
# ------------------------------------------------------------------------------

compute_cross_kernel <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  if (
    model$kernel ==
    "rbf"
  ) {
    
    rbf_kernel(
      X_new,
      model$X_train,
      gamma =
        model$gamma
    )
    
  } else if (
    model$kernel ==
    "linear"
  ) {
    
    linear_kernel(
      X_new,
      model$X_train
    )
    
  } else {
    
    polynomial_kernel(
      X_new,
      model$X_train,
      degree =
        model$degree,
      constant =
        model$constant
    )
  }
}


# ------------------------------------------------------------------------------
# 24. Center New Kernel Rows
# ------------------------------------------------------------------------------

center_new_kernel <- function(
    model,
    K_new
) {
  
  # For a new observation x:
  #
  # k_c(x, x_i)
  #
  # =
  #
  # k(x, x_i)
  # - mean_j k(x, x_j)
  # - mean_j k(x_j, x_i)
  # + mean_{j,l} k(x_j, x_l)
  
  
  new_row_means <- rowMeans(
    K_new
  )
  
  
  centered <- K_new
  
  
  centered <- sweep(
    centered,
    1,
    new_row_means,
    "-"
  )
  
  
  centered <- sweep(
    centered,
    2,
    model$training_kernel_row_means,
    "-"
  )
  
  
  centered <- centered +
    model$training_kernel_grand_mean
  
  
  centered
}


# ------------------------------------------------------------------------------
# 25. Predict Kernel PC Scores for New Data
# ------------------------------------------------------------------------------

predict_kernel_pca <- function(
    model,
    X_new,
    number_components = NULL
) {
  
  if (
    is.null(
      number_components
    )
  ) {
    
    number_components <- ncol(
      model$alpha
    )
  }
  
  
  K_new <- compute_cross_kernel(
    model,
    X_new
  )
  
  
  K_new_centered <- center_new_kernel(
    model,
    K_new
  )
  
  
  K_new_centered %*%
    model$alpha[
      ,
      seq_len(
        number_components
      ),
      drop = FALSE
    ]
}


# ------------------------------------------------------------------------------
# 26. Verify Training Projection
# ------------------------------------------------------------------------------

training_scores_again <- predict_kernel_pca(
  KPCA_model,
  X,
  number_components = 3
)


max(
  abs(
    training_scores_again -
      KPCA_model$scores[
        ,
        1:3,
        drop = FALSE
      ]
  )
)


# ==============================================================================
# PART XIII
#
# NEW OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Example New Points
# ------------------------------------------------------------------------------

X_new <- rbind(
  c(
    1,
    0
  ),
  c(
    2.5,
    0
  ),
  c(
    0,
    1
  ),
  c(
    0,
    2.5
  ),
  c(
    0,
    0
  )
)


colnames(
  X_new
) <- colnames(
  X
)


new_kernel_scores <- predict_kernel_pca(
  KPCA_model,
  X_new,
  number_components = 2
)


new_kernel_scores


# ==============================================================================
# PART XIV
#
# GAMMA EFFECT
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Compare RBF Bandwidths
# ------------------------------------------------------------------------------

gamma_values <- c(
  0.05,
  0.20,
  1,
  5,
  20
)


gamma_models <- vector(
  "list",
  length(
    gamma_values
  )
)


par(
  mfrow = c(
    2,
    3
  )
)


for (
  i in seq_along(
    gamma_values
  )
) {
  
  gamma_models[[i]] <- fit_kernel_pca(
    X,
    kernel = "rbf",
    gamma =
      gamma_values[i]
  )
  
  
  plot(
    gamma_models[[i]]$scores[
      ,
      1
    ],
    gamma_models[[i]]$scores[
      ,
      2
    ],
    pch = true_group,
    xlab = "KPC1",
    ylab = "KPC2",
    main = paste(
      "Gamma =",
      gamma_values[i]
    )
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART XV
#
# LINEAR KERNEL RECOVERS ORDINARY PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Fit Linear Kernel PCA
# ------------------------------------------------------------------------------

linear_KPCA <- fit_kernel_pca(
  X_centered,
  kernel = "linear"
)


# ------------------------------------------------------------------------------
# 30. Compare Pairwise Geometry of Scores
# ------------------------------------------------------------------------------

ordinary_scores_2 <- ordinary_scores[
  ,
  1:2,
  drop = FALSE
]


linear_kernel_scores_2 <- linear_KPCA$scores[
  ,
  1:2,
  drop = FALSE
]


# Eigenvectors may differ in sign, so compare pairwise distances.

ordinary_distance <- as.matrix(
  dist(
    ordinary_scores_2
  )
)


kernel_distance <- as.matrix(
  dist(
    linear_kernel_scores_2
  )
)


cor(
  ordinary_distance[
    upper.tri(
      ordinary_distance
    )
  ],
  kernel_distance[
    upper.tri(
      kernel_distance
    )
  ]
)


# Linear-kernel PCA is equivalent to ordinary PCA up to rotation/sign/scaling
# conventions in score representation.


# ==============================================================================
# PART XVI
#
# POLYNOMIAL KERNEL PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Degree-2 Polynomial Kernel
# ------------------------------------------------------------------------------

polynomial_KPCA <- fit_kernel_pca(
  X,
  kernel = "polynomial",
  degree = 2,
  constant = 1
)


plot(
  polynomial_KPCA$scores[
    ,
    1
  ],
  polynomial_KPCA$scores[
    ,
    2
  ],
  pch = true_group,
  xlab = "KPC1",
  ylab = "KPC2",
  main = "Polynomial Kernel PCA"
)


# ==============================================================================
# PART XVII
#
# NONLINEAR FEATURE MAP EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Explicit Quadratic Features
# ------------------------------------------------------------------------------

quadratic_features <- cbind(
  X1 =
    X[
      ,
      1
    ],
  X2 =
    X[
      ,
      2
    ],
  X1_squared =
    X[
      ,
      1
    ]^2,
  Interaction =
    sqrt(2) *
    X[
      ,
      1
    ] *
    X[
      ,
      2
    ],
  X2_squared =
    X[
      ,
      2
    ]^2
)


# ------------------------------------------------------------------------------
# 33. PCA in Explicit Nonlinear Feature Space
# ------------------------------------------------------------------------------

quadratic_centered <- sweep(
  quadratic_features,
  2,
  colMeans(
    quadratic_features
  ),
  "-"
)


quadratic_PCA <- svd(
  quadratic_centered
)


quadratic_scores <- quadratic_centered %*%
  quadratic_PCA$v


plot(
  quadratic_scores[
    ,
    1
  ],
  quadratic_scores[
    ,
    2
  ],
  pch = true_group,
  xlab = "Feature-Space PC1",
  ylab = "Feature-Space PC2",
  main = "PCA after Explicit Nonlinear Mapping"
)


# This illustrates the basic idea behind kernel PCA:
#
# transform first, then perform ordinary PCA.
#
# Kernels allow us to compute inner products in that transformed space without
# explicitly constructing all transformed features.


# ==============================================================================
# PART XVIII
#
# KERNEL MATRIX AS SIMILARITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Inspect RBF Similarities
# ------------------------------------------------------------------------------

example_indices <- c(
  1,
  2,
  n_inner + 1
)


K[
  example_indices,
  example_indices
]


# Nearby observations have kernel values near 1.
#
# Distant observations have values near 0 for the RBF kernel.


# ==============================================================================
# PART XIX
#
# KERNEL PCA ON SWISS-ROLL-LIKE 2D CURVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Generate Spiral Data
# ------------------------------------------------------------------------------

set.seed(456)


n_spiral <- 400


spiral_t <- seq(
  0.5,
  4 * pi,
  length.out =
    n_spiral
)


spiral_radius <- 0.25 *
  spiral_t


X_spiral <- cbind(
  spiral_radius *
    cos(
      spiral_t
    ) +
    rnorm(
      n_spiral,
      sd = 0.08
    ),
  spiral_radius *
    sin(
      spiral_t
    ) +
    rnorm(
      n_spiral,
      sd = 0.08
    )
)


colnames(
  X_spiral
) <- c(
  "X1",
  "X2"
)


# ------------------------------------------------------------------------------
# 36. Ordinary PCA
# ------------------------------------------------------------------------------

spiral_centered <- sweep(
  X_spiral,
  2,
  colMeans(
    X_spiral
  ),
  "-"
)


spiral_PCA <- svd(
  spiral_centered
)


spiral_PCA_scores <- spiral_centered %*%
  spiral_PCA$v


plot(
  spiral_PCA_scores[
    ,
    1
  ],
  spiral_PCA_scores[
    ,
    2
  ],
  pch = 19,
  cex = 0.4,
  xlab = "PC1",
  ylab = "PC2",
  main = "Spiral: Ordinary PCA"
)


# ------------------------------------------------------------------------------
# 37. Kernel PCA
# ------------------------------------------------------------------------------

spiral_KPCA <- fit_kernel_pca(
  X_spiral,
  kernel = "rbf",
  gamma = 2
)


plot(
  spiral_KPCA$scores[
    ,
    1
  ],
  spiral_KPCA$scores[
    ,
    2
  ],
  pch = 19,
  cex = 0.4,
  xlab = "KPC1",
  ylab = "KPC2",
  main = "Spiral: Kernel PCA"
)


# ==============================================================================
# PART XX
#
# TRAIN / TEST TRANSFORMATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Split Data
# ------------------------------------------------------------------------------

set.seed(999)


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


X_test <- X[
  test_indices,
  ,
  drop = FALSE
]


y_train <- true_group[
  train_indices
]


y_test <- true_group[
  test_indices
]


# ------------------------------------------------------------------------------
# 39. Fit Kernel PCA on Training Set Only
# ------------------------------------------------------------------------------

KPCA_train <- fit_kernel_pca(
  X_train,
  kernel = "rbf",
  gamma = 1
)


# ------------------------------------------------------------------------------
# 40. Transform Test Data
# ------------------------------------------------------------------------------

train_embedding <- KPCA_train$scores[
  ,
  1:2,
  drop = FALSE
]


test_embedding <- predict_kernel_pca(
  KPCA_train,
  X_test,
  number_components = 2
)


plot(
  train_embedding,
  pch = y_train,
  xlab = "KPC1",
  ylab = "KPC2",
  main = "Training Kernel PCA Embedding"
)


plot(
  test_embedding,
  pch = y_test,
  xlab = "KPC1",
  ylab = "KPC2",
  main = "Held-Out Observations in Kernel PCA Space"
)


# ==============================================================================
# PART XXI
#
# SIMPLE CLUSTERING AFTER KPCA
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. K-Means in Original Space
# ------------------------------------------------------------------------------

set.seed(123)


kmeans_original <- kmeans(
  X,
  centers = 2,
  nstart = 50
)


# ------------------------------------------------------------------------------
# 42. K-Means after Kernel PCA
# ------------------------------------------------------------------------------

set.seed(123)


kmeans_kernel <- kmeans(
  KPCA_model$scores[
    ,
    1:2,
    drop = FALSE
  ],
  centers = 2,
  nstart = 50
)


# ------------------------------------------------------------------------------
# 43. Rand Index
# ------------------------------------------------------------------------------

rand_index <- function(
    labels_1,
    labels_2
) {
  
  n <- length(
    labels_1
  )
  
  
  agreement <- 0
  
  total <- 0
  
  
  for (
    i in seq_len(
      n - 1
    )
  ) {
    
    for (
      j in (
        i + 1
      ):n
    ) {
      
      same_1 <- labels_1[i] ==
        labels_1[j]
      
      
      same_2 <- labels_2[i] ==
        labels_2[j]
      
      
      agreement <- agreement +
        (
          same_1 ==
            same_2
        )
      
      
      total <- total +
        1
    }
  }
  
  
  agreement /
    total
}


data.frame(
  Method = c(
    "K-Means Original Space",
    "K-Means after Kernel PCA"
  ),
  Rand_Index = c(
    rand_index(
      true_group,
      kmeans_original$cluster
    ),
    rand_index(
      true_group,
      kmeans_kernel$cluster
    )
  )
)


# ==============================================================================
# PART XXII
#
# GAMMA SELECTION USING CLUSTER SEPARATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Demonstration Only: True-Label-Based Gamma Comparison
# ------------------------------------------------------------------------------

gamma_grid <- c(
  0.02,
  0.05,
  0.1,
  0.2,
  0.5,
  1,
  2,
  5,
  10
)


gamma_results <- data.frame(
  Gamma =
    gamma_grid,
  Rand_Index =
    NA_real_
)


for (
  i in seq_along(
    gamma_grid
  )
) {
  
  model_i <- fit_kernel_pca(
    X,
    kernel = "rbf",
    gamma =
      gamma_grid[i]
  )
  
  
  embedding_i <- model_i$scores[
    ,
    1:2,
    drop = FALSE
  ]
  
  
  set.seed(123)
  
  
  cluster_i <- kmeans(
    embedding_i,
    centers = 2,
    nstart = 20
  )$cluster
  
  
  gamma_results$Rand_Index[i] <-
    rand_index(
      true_group,
      cluster_i
    )
}


gamma_results


plot(
  gamma_results$Gamma,
  gamma_results$Rand_Index,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "RBF Gamma",
  ylab = "Rand Index",
  main = "Kernel Parameter Sensitivity"
)


# IMPORTANT:
#
# Using true labels to select gamma is only valid here because this is simulated
# educational data.
#
# In a real unsupervised problem, the true cluster labels are unavailable.


# ==============================================================================
# PART XXIII
#
# MEDIAN DISTANCE HEURISTIC
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Pairwise Distances
# ------------------------------------------------------------------------------

distance_squared <- squared_distance_matrix(
  X
)


upper_distances <- distance_squared[
  upper.tri(
    distance_squared
  )
]


median_squared_distance <- median(
  upper_distances
)


# ------------------------------------------------------------------------------
# 46. Heuristic Gamma
# ------------------------------------------------------------------------------

gamma_median <- 1 /
  (
    2 *
      median_squared_distance
  )


gamma_median


# ------------------------------------------------------------------------------
# 47. Fit Using Heuristic
# ------------------------------------------------------------------------------

KPCA_median <- fit_kernel_pca(
  X,
  kernel = "rbf",
  gamma =
    gamma_median
)


plot(
  KPCA_median$scores[
    ,
    1
  ],
  KPCA_median$scores[
    ,
    2
  ],
  pch = true_group,
  xlab = "KPC1",
  ylab = "KPC2",
  main = "Kernel PCA: Median Distance Heuristic"
)


# ==============================================================================
# PART XXIV
#
# PRE-IMAGE PROBLEM
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Conceptual Note
# ------------------------------------------------------------------------------

# Ordinary PCA gives:
#
#       z = V' (x - mean)
#
# and has a simple inverse:
#
#       x_hat = mean + V z
#
#
# Kernel PCA maps implicitly into a nonlinear feature space:
#
#       x -> phi(x)
#
# and performs PCA there.
#
# Given a reduced kernel representation, finding an original-space x such that:
#
#       phi(x)
#
# corresponds to the reduced feature-space point is generally nontrivial.
#
# This is the "pre-image problem".
#
# Therefore kernel PCA does not have the same simple exact inverse transform as
# ordinary PCA.


# ==============================================================================
# PART XXV
#
# KERNEL PCA COMPUTATIONAL COST
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Kernel Matrix Size
# ------------------------------------------------------------------------------

n_values <- c(
  100,
  500,
  1000,
  5000,
  10000
)


kernel_entries <- n_values^2


memory_MB <- kernel_entries *
  8 /
  1024^2


data.frame(
  Observations =
    n_values,
  Kernel_Entries =
    kernel_entries,
  Approx_Memory_MB =
    memory_MB
)


# Kernel PCA requires an n x n Gram matrix.
#
# Standard eigendecomposition is also expensive for large n.


# ==============================================================================
# PART XXVI
#
# OPTIONAL VERIFICATION WITH kernlab
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Compare if kernlab is Installed
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "kernlab",
    quietly = TRUE
  )
) {
  
  kernlab_model <- kernlab::kpca(
    X,
    kernel = "rbfdot",
    kpar = list(
      sigma = gamma
    ),
    features = 2
  )
  
  
  kernlab_scores <- kernlab::rotated(
    kernlab_model
  )
  
  
  plot(
    kernlab_scores[
      ,
      1
    ],
    kernlab_scores[
      ,
      2
    ],
    pch = true_group,
    xlab = "KPC1",
    ylab = "KPC2",
    main = "kernlab Kernel PCA"
  )
  
  
  cat(
    "Manual KPCA and kernlab may differ by signs, scaling, and kernel\n"
  )
  
  
  cat(
    "parameter conventions even when representing similar geometry.\n"
  )
}


# ==============================================================================
# PART XXVII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Output
# ------------------------------------------------------------------------------

cat(
  "Kernel PCA Summary\n"
)


cat(
  "------------------\n"
)


cat(
  "Observations:",
  n,
  "\n"
)


cat(
  "Original dimensions:",
  ncol(X),
  "\n"
)


cat(
  "Kernel:",
  KPCA_model$kernel,
  "\n"
)


cat(
  "RBF gamma:",
  KPCA_model$gamma,
  "\n"
)


cat(
  "Positive kernel eigenvalues:",
  length(
    KPCA_model$eigenvalues
  ),
  "\n"
)


cat(
  "Feature-space variance explained by KPC1:",
  round(
    KPCA_model$variance_proportion[1],
    4
  ),
  "\n"
)


cat(
  "Feature-space variance explained by first 2 KPCs:",
  round(
    KPCA_model$cumulative_variance[2],
    4
  ),
  "\n"
)


cat(
  "Median-distance heuristic gamma:",
  round(
    gamma_median,
    6
  ),
  "\n"
)