# ==============================================================================
# Principal Components Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Unsupervised dimensionality reduction
#   - Centering and scaling
#   - Covariance matrix
#   - Eigenvalues and eigenvectors
#   - Principal component loadings
#   - Principal component scores
#   - Explained variance
#   - Scree plots
#   - Reconstruction
#   - Low-rank approximation
#   - Singular value decomposition (SVD)
#   - Biplots
#
#
# PCA finds orthogonal directions:
#
#       v_1, v_2, ..., v_p
#
# such that the projected data have successively maximum variance.
#
#
# First principal component:
#
#       v_1
#       =
#       arg max   Var(X v)
#         ||v||=1
#
#
# Subsequent components:
#
#       v_m
#       =
#       arg max   Var(X v)
#         ||v||=1
#
# subject to:
#
#       v_m' v_l = 0
#
# for all l < m.
#
#
# If S is the sample covariance matrix:
#
#       S v_j = lambda_j v_j
#
#
# then:
#
#       v_j      = principal component loading
#       lambda_j = variance of principal component j
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE CORRELATED DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulation
# ------------------------------------------------------------------------------

set.seed(123)

n <- 300

p <- 6


# Generate a few latent variables.

z1 <- rnorm(
  n
)

z2 <- rnorm(
  n
)

z3 <- rnorm(
  n
)


# Observed predictors are noisy combinations of latent variables.

x1 <- 2.0 * z1 +
  0.3 * z2 +
  rnorm(
    n,
    sd = 0.30
  )


x2 <- 1.8 * z1 +
  0.2 * z2 +
  rnorm(
    n,
    sd = 0.35
  )


x3 <- -1.5 * z1 +
  0.5 * z2 +
  rnorm(
    n,
    sd = 0.40
  )


x4 <- 0.2 * z1 +
  2.0 * z2 +
  rnorm(
    n,
    sd = 0.30
  )


x5 <- 0.1 * z1 +
  1.7 * z2 +
  0.5 * z3 +
  rnorm(
    n,
    sd = 0.35
  )


x6 <- 1.8 * z3 +
  rnorm(
    n,
    sd = 0.50
  )


X <- cbind(
  x1,
  x2,
  x3,
  x4,
  x5,
  x6
)


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 2. Examine Data
# ------------------------------------------------------------------------------

head(
  X
)


summary(
  X
)


# ------------------------------------------------------------------------------
# 3. Pairwise Plots
# ------------------------------------------------------------------------------

pairs(
  X,
  main = "Correlated Predictor Data"
)


# ------------------------------------------------------------------------------
# 4. Correlation Matrix
# ------------------------------------------------------------------------------

correlation_matrix <- cor(
  X
)


round(
  correlation_matrix,
  2
)


# ==============================================================================
# PART II
#
# CENTERING THE DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Feature Means
# ------------------------------------------------------------------------------

feature_means <- colMeans(
  X
)


feature_means


# ------------------------------------------------------------------------------
# 6. Center Data Manually
# ------------------------------------------------------------------------------

X_centered <- sweep(
  X,
  2,
  feature_means,
  "-"
)


# ------------------------------------------------------------------------------
# 7. Verify Centering
# ------------------------------------------------------------------------------

round(
  colMeans(
    X_centered
  ),
  10
)


# PCA should generally be performed on centered data.
#
# Otherwise the origin influences the directions of maximum variation.


# ==============================================================================
# PART III
#
# SAMPLE COVARIANCE MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Compute Covariance Matrix Manually
# ------------------------------------------------------------------------------

covariance_manual <- (
  t(
    X_centered
  ) %*%
    X_centered
) /
  (
    n -
      1
  )


round(
  covariance_manual,
  3
)


# ------------------------------------------------------------------------------
# 9. Verify Against cov()
# ------------------------------------------------------------------------------

covariance_builtin <- cov(
  X
)


max(
  abs(
    covariance_manual -
      covariance_builtin
  )
)


# ==============================================================================
# PART IV
#
# PCA VIA EIGENDECOMPOSITION
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Eigendecomposition
# ------------------------------------------------------------------------------

eigen_result <- eigen(
  covariance_manual,
  symmetric = TRUE
)


eigenvalues <- eigen_result$values


loadings <- eigen_result$vectors


colnames(
  loadings
) <- paste0(
  "PC",
  seq_len(p)
)


rownames(
  loadings
) <- colnames(
  X
)


# ------------------------------------------------------------------------------
# 11. Eigenvalues
# ------------------------------------------------------------------------------

eigenvalues


# Eigenvalues are sorted from largest to smallest by eigen().


# ------------------------------------------------------------------------------
# 12. Loadings
# ------------------------------------------------------------------------------

round(
  loadings,
  3
)


# Each column is a principal component direction.


# ==============================================================================
# PART V
#
# VERIFY EIGENVECTOR EQUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Verify S v = lambda v
# ------------------------------------------------------------------------------

PC1_loading <- loadings[
  ,
  1
]


left_side <- covariance_manual %*%
  PC1_loading


right_side <- eigenvalues[1] *
  PC1_loading


max(
  abs(
    left_side -
      right_side
  )
)


# ==============================================================================
# PART VI
#
# ORTHONORMALITY OF LOADINGS
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. V'V = I
# ------------------------------------------------------------------------------

orthogonality_check <- t(
  loadings
) %*%
  loadings


round(
  orthogonality_check,
  8
)


# Principal component directions are orthonormal:
#
#       v_i' v_j = 0   if i != j
#
#       ||v_j|| = 1


# ==============================================================================
# PART VII
#
# PRINCIPAL COMPONENT SCORES
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Project Data onto Principal Component Directions
# ------------------------------------------------------------------------------

scores <- X_centered %*%
  loadings


colnames(
  scores
) <- paste0(
  "PC",
  seq_len(p)
)


head(
  scores
)


# Each row now represents one observation in principal-component coordinates.


# ==============================================================================
# PART VIII
#
# VARIANCE OF PRINCIPAL COMPONENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Variance of Scores
# ------------------------------------------------------------------------------

score_variances <- apply(
  scores,
  2,
  var
)


score_variances


# ------------------------------------------------------------------------------
# 17. Compare with Eigenvalues
# ------------------------------------------------------------------------------

data.frame(
  Principal_Component =
    paste0(
      "PC",
      seq_len(p)
    ),
  Eigenvalue =
    eigenvalues,
  Score_Variance =
    score_variances,
  Difference =
    eigenvalues -
    score_variances
)


# The variance of PC_j is exactly lambda_j.


# ==============================================================================
# PART IX
#
# PRINCIPAL COMPONENTS ARE UNCORRELATED
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Score Covariance Matrix
# ------------------------------------------------------------------------------

score_covariance <- cov(
  scores
)


round(
  score_covariance,
  6
)


# ------------------------------------------------------------------------------
# 19. Score Correlation Matrix
# ------------------------------------------------------------------------------

round(
  cor(
    scores
  ),
  6
)


# PCA rotates the coordinate system so the covariance matrix becomes diagonal.


# ==============================================================================
# PART X
#
# EXPLAINED VARIANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Total Variance
# ------------------------------------------------------------------------------

total_variance <- sum(
  eigenvalues
)


total_variance


# This should equal the trace of the covariance matrix.

sum(
  diag(
    covariance_manual
  )
)


# ------------------------------------------------------------------------------
# 21. Proportion of Variance Explained
# ------------------------------------------------------------------------------

proportion_variance <- eigenvalues /
  sum(
    eigenvalues
  )


# ------------------------------------------------------------------------------
# 22. Cumulative Explained Variance
# ------------------------------------------------------------------------------

cumulative_variance <- cumsum(
  proportion_variance
)


variance_table <- data.frame(
  PC =
    paste0(
      "PC",
      seq_len(p)
    ),
  Eigenvalue =
    eigenvalues,
  Proportion =
    proportion_variance,
  Cumulative =
    cumulative_variance
)


variance_table


# ==============================================================================
# PART XI
#
# SCREE PLOT
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Eigenvalue Scree Plot
# ------------------------------------------------------------------------------

plot(
  seq_len(p),
  eigenvalues,
  type = "b",
  pch = 19,
  xlab = "Principal Component",
  ylab = "Eigenvalue",
  main = "PCA Scree Plot"
)


# ==============================================================================
# PART XII
#
# CUMULATIVE EXPLAINED VARIANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Cumulative Variance Plot
# ------------------------------------------------------------------------------

plot(
  seq_len(p),
  cumulative_variance,
  type = "b",
  pch = 19,
  ylim = c(
    0,
    1
  ),
  xlab = "Number of Principal Components",
  ylab = "Cumulative Proportion of Variance",
  main = "Cumulative Explained Variance"
)


abline(
  h = c(
    0.80,
    0.90,
    0.95
  ),
  lty = c(
    2,
    3,
    4
  )
)


legend(
  "bottomright",
  legend = c(
    "80%",
    "90%",
    "95%"
  ),
  lty = c(
    2,
    3,
    4
  )
)


# ==============================================================================
# PART XIII
#
# SELECT NUMBER OF COMPONENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Components Needed for 90% Variance
# ------------------------------------------------------------------------------

M_90 <- which(
  cumulative_variance >=
    0.90
)[1]


M_90


# ------------------------------------------------------------------------------
# 26. Components Needed for 95% Variance
# ------------------------------------------------------------------------------

M_95 <- which(
  cumulative_variance >=
    0.95
)[1]


M_95


# ==============================================================================
# PART XIV
#
# FIRST TWO PRINCIPAL COMPONENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. PC1 vs PC2
# ------------------------------------------------------------------------------

plot(
  scores[
    ,
    1
  ],
  scores[
    ,
    2
  ],
  pch = 19,
  xlab = "PC1",
  ylab = "PC2",
  main = "Observations in Principal Component Space"
)


abline(
  h = 0,
  lty = 2
)


abline(
  v = 0,
  lty = 2
)


# ==============================================================================
# PART XV
#
# INTERPRETING LOADINGS
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. First Two Loading Vectors
# ------------------------------------------------------------------------------

loading_table <- data.frame(
  Variable =
    rownames(
      loadings
    ),
  PC1 =
    loadings[
      ,
      1
    ],
  PC2 =
    loadings[
      ,
      2
    ],
  PC3 =
    loadings[
      ,
      3
    ]
)


loading_table


# ------------------------------------------------------------------------------
# 29. Absolute PC1 Contributions
# ------------------------------------------------------------------------------

PC1_absolute_loadings <- abs(
  loadings[
    ,
    1
  ]
)


PC1_order <- order(
  PC1_absolute_loadings,
  decreasing = TRUE
)


data.frame(
  Variable =
    rownames(
      loadings
    )[
      PC1_order
    ],
  Loading =
    loadings[
      PC1_order,
      1
    ],
  Absolute_Loading =
    PC1_absolute_loadings[
      PC1_order
    ]
)


# ------------------------------------------------------------------------------
# 30. Plot Loadings
# ------------------------------------------------------------------------------

barplot(
  loadings[
    ,
    1
  ],
  names.arg =
    rownames(
      loadings
    ),
  las = 2,
  ylab = "PC1 Loading",
  main = "First Principal Component Loadings"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART XVI
#
# PCA AS ROTATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Preserve Pairwise Euclidean Distances Using All PCs
# ------------------------------------------------------------------------------

distance_original <- as.matrix(
  dist(
    X_centered
  )
)


distance_rotated <- as.matrix(
  dist(
    scores
  )
)


max(
  abs(
    distance_original -
      distance_rotated
  )
)


# Because V is orthogonal, using all principal components is only a rotation.
#
# No information is lost.


# ==============================================================================
# PART XVII
#
# RECONSTRUCT DATA USING ALL COMPONENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Inverse Transformation
# ------------------------------------------------------------------------------

X_centered_reconstructed <- scores %*%
  t(
    loadings
  )


X_reconstructed <- sweep(
  X_centered_reconstructed,
  2,
  feature_means,
  "+"
)


# ------------------------------------------------------------------------------
# 33. Reconstruction Error
# ------------------------------------------------------------------------------

max(
  abs(
    X -
      X_reconstructed
  )
)


# Using all PCs reconstructs the original data up to numerical precision.


# ==============================================================================
# PART XVIII
#
# LOW-RANK PCA RECONSTRUCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Reconstruction Function
# ------------------------------------------------------------------------------

reconstruct_pca <- function(
    X_centered,
    loadings,
    number_components
) {
  
  V_M <- loadings[
    ,
    seq_len(
      number_components
    ),
    drop = FALSE
  ]
  
  
  scores_M <- X_centered %*%
    V_M
  
  
  reconstruction <- scores_M %*%
    t(
      V_M
    )
  
  
  list(
    scores =
      scores_M,
    reconstruction =
      reconstruction
  )
}


# ------------------------------------------------------------------------------
# 35. Reconstruction Error for Different M
# ------------------------------------------------------------------------------

reconstruction_error <- numeric(
  p
)


for (
  M in seq_len(p)
) {
  
  reconstruction_M <- reconstruct_pca(
    X_centered,
    loadings,
    number_components = M
  )
  
  
  reconstruction_error[M] <- mean(
    (
      X_centered -
        reconstruction_M$reconstruction
    )^2
  )
}


data.frame(
  Components =
    seq_len(p),
  Reconstruction_MSE =
    reconstruction_error
)


# ------------------------------------------------------------------------------
# 36. Plot Reconstruction Error
# ------------------------------------------------------------------------------

plot(
  seq_len(p),
  reconstruction_error,
  type = "b",
  pch = 19,
  xlab = "Number of Principal Components",
  ylab = "Reconstruction MSE",
  main = "PCA Reconstruction Error"
)


# ==============================================================================
# PART XIX
#
# THEORETICAL RECONSTRUCTION ERROR
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Sum of Discarded Eigenvalues
# ------------------------------------------------------------------------------

discarded_variance <- sapply(
  seq_len(p),
  function(M) {
    
    if (
      M ==
      p
    ) {
      
      return(
        0
      )
    }
    
    
    sum(
      eigenvalues[
        (
          M + 1
        ):p
      ]
    )
  }
)


# Mean squared error across all matrix entries should equal:
#
#       sum(discarded eigenvalues) / p
#
# because the total squared reconstruction error is:
#
#       (n - 1) * sum(discarded eigenvalues)


theoretical_MSE <- discarded_variance /
  p


data.frame(
  Components =
    seq_len(p),
  Empirical_MSE =
    reconstruction_error,
  Theoretical_MSE =
    theoretical_MSE
)


# ==============================================================================
# PART XX
#
# PCA VIA SINGULAR VALUE DECOMPOSITION
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. SVD of Centered Data
# ------------------------------------------------------------------------------

svd_result <- svd(
  X_centered
)


U <- svd_result$u


singular_values <- svd_result$d


V <- svd_result$v


# ------------------------------------------------------------------------------
# 39. Compare SVD Loadings with Eigenvectors
# ------------------------------------------------------------------------------

# Eigenvectors are unique only up to sign.
#
# Therefore compare absolute values.

round(
  abs(
    V
  ),
  6
)


round(
  abs(
    loadings
  ),
  6
)


max(
  abs(
    abs(V) -
      abs(loadings)
  )
)


# ------------------------------------------------------------------------------
# 40. Eigenvalues from Singular Values
# ------------------------------------------------------------------------------

eigenvalues_from_svd <- singular_values^2 /
  (
    n -
      1
  )


data.frame(
  Eigen =
    eigenvalues,
  SVD =
    eigenvalues_from_svd,
  Difference =
    eigenvalues -
    eigenvalues_from_svd
)


# ==============================================================================
# PART XXI
#
# PCA SCORES VIA SVD
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Scores = U D
# ------------------------------------------------------------------------------

scores_from_svd <- U %*%
  diag(
    singular_values,
    nrow =
      length(
        singular_values
      )
  )


# Scores may differ by column signs.

max(
  abs(
    abs(
      scores_from_svd
    ) -
      abs(
        scores
      )
  )
)


# ==============================================================================
# PART XXII
#
# LOW-RANK APPROXIMATION VIA SVD
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Rank-2 Approximation
# ------------------------------------------------------------------------------

M <- 2


U_M <- U[
  ,
  seq_len(M),
  drop = FALSE
]


D_M <- diag(
  singular_values[
    seq_len(M)
  ],
  nrow = M
)


V_M <- V[
  ,
  seq_len(M),
  drop = FALSE
]


X_rank_M <- U_M %*%
  D_M %*%
  t(
    V_M
  )


mean(
  (
    X_centered -
      X_rank_M
  )^2
)


# This is the optimal rank-M approximation under squared Frobenius error.


# ==============================================================================
# PART XXIII
#
# MANUAL FIRST PC OPTIMIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Variance of a Projection
# ------------------------------------------------------------------------------

projection_variance <- function(
    direction,
    covariance_matrix
) {
  
  direction <- direction /
    sqrt(
      sum(
        direction^2
      )
    )
  
  
  as.numeric(
    t(direction) %*%
      covariance_matrix %*%
      direction
  )
}


# ------------------------------------------------------------------------------
# 44. Random Directions
# ------------------------------------------------------------------------------

set.seed(999)

number_random_directions <- 10000


random_variances <- numeric(
  number_random_directions
)


best_random_variance <- -Inf

best_random_direction <- NULL


for (
  i in seq_len(
    number_random_directions
  )
) {
  
  direction <- rnorm(
    p
  )
  
  
  direction <- direction /
    sqrt(
      sum(
        direction^2
      )
    )
  
  
  variance_i <- projection_variance(
    direction,
    covariance_manual
  )
  
  
  random_variances[i] <-
    variance_i
  
  
  if (
    variance_i >
    best_random_variance
  ) {
    
    best_random_variance <-
      variance_i
    
    
    best_random_direction <-
      direction
  }
}


# ------------------------------------------------------------------------------
# 45. Compare with PCA
# ------------------------------------------------------------------------------

best_random_variance


eigenvalues[1]


# No random direction should systematically exceed the first eigenvalue.


hist(
  random_variances,
  breaks = 40,
  xlab = "Variance of Projection",
  main = "Variance Across Random Projection Directions"
)


abline(
  v = eigenvalues[1],
  lwd = 2
)


# ==============================================================================
# PART XXIV
#
# LOADINGS ARE SIGN-INDETERMINATE
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Sign Flip
# ------------------------------------------------------------------------------

v1 <- loadings[
  ,
  1
]


v1_negative <- -v1


projection_original <- X_centered %*%
  v1


projection_flipped <- X_centered %*%
  v1_negative


max(
  abs(
    projection_original +
      projection_flipped
  )
)


# v and -v represent the same principal component axis.


# ==============================================================================
# PART XXV
#
# EFFECT OF FEATURE SCALING
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Create Variables on Very Different Scales
# ------------------------------------------------------------------------------

set.seed(321)


X_scale_demo <- cbind(
  Small_1 =
    rnorm(
      300,
      sd = 1
    ),
  Small_2 =
    rnorm(
      300,
      sd = 1
    ),
  Large =
    rnorm(
      300,
      sd = 100
    )
)


# Add correlation between first two variables.

X_scale_demo[
  ,
  2
] <- 0.8 *
  X_scale_demo[
    ,
    1
  ] +
  rnorm(
    300,
    sd = 0.4
  )


# ------------------------------------------------------------------------------
# 48. PCA on Covariance Matrix
# ------------------------------------------------------------------------------

covariance_scale <- cov(
  X_scale_demo
)


PCA_covariance <- eigen(
  covariance_scale,
  symmetric = TRUE
)


round(
  PCA_covariance$vectors,
  3
)


# The variable measured on the large scale tends to dominate variance.


# ------------------------------------------------------------------------------
# 49. Standardize Features
# ------------------------------------------------------------------------------

scale_means <- colMeans(
  X_scale_demo
)


scale_sds <- apply(
  X_scale_demo,
  2,
  sd
)


X_standardized <- sweep(
  X_scale_demo,
  2,
  scale_means,
  "-"
)


X_standardized <- sweep(
  X_standardized,
  2,
  scale_sds,
  "/"
)


# ------------------------------------------------------------------------------
# 50. PCA on Correlation Matrix
# ------------------------------------------------------------------------------

correlation_scale <- cov(
  X_standardized
)


PCA_correlation <- eigen(
  correlation_scale,
  symmetric = TRUE
)


round(
  PCA_correlation$vectors,
  3
)


# Standardizing gives each original variable unit variance before PCA.


# ==============================================================================
# PART XXVI
#
# GENERAL PCA FUNCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Manual PCA
# ------------------------------------------------------------------------------

pca_manual <- function(
    X,
    scale_features = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  means <- colMeans(
    X
  )
  
  
  X_processed <- sweep(
    X,
    2,
    means,
    "-"
  )
  
  
  if (
    scale_features
  ) {
    
    standard_deviations <- apply(
      X,
      2,
      sd
    )
    
    
    if (
      any(
        standard_deviations <=
        .Machine$double.eps
      )
    ) {
      
      stop(
        "Cannot scale a feature with zero variance."
      )
    }
    
    
    X_processed <- sweep(
      X_processed,
      2,
      standard_deviations,
      "/"
    )
    
  } else {
    
    standard_deviations <- rep(
      1,
      ncol(X)
    )
  }
  
  
  decomposition <- svd(
    X_processed
  )
  
  
  loadings <- decomposition$v
  
  
  scores <- X_processed %*%
    loadings
  
  
  eigenvalues <- decomposition$d^2 /
    (
      nrow(X) -
        1
    )
  
  
  proportion_variance <- eigenvalues /
    sum(
      eigenvalues
    )
  
  
  cumulative_variance <- cumsum(
    proportion_variance
  )
  
  
  colnames(
    loadings
  ) <- paste0(
    "PC",
    seq_len(
      ncol(loadings)
    )
  )
  
  
  rownames(
    loadings
  ) <- colnames(
    X
  )
  
  
  colnames(
    scores
  ) <- paste0(
    "PC",
    seq_len(
      ncol(scores)
    )
  )
  
  
  list(
    means =
      means,
    standard_deviations =
      standard_deviations,
    scale_features =
      scale_features,
    loadings =
      loadings,
    scores =
      scores,
    eigenvalues =
      eigenvalues,
    proportion_variance =
      proportion_variance,
    cumulative_variance =
      cumulative_variance
  )
}


# ------------------------------------------------------------------------------
# 52. Fit Manual PCA
# ------------------------------------------------------------------------------

manual_model <- pca_manual(
  X,
  scale_features = FALSE
)


manual_model$eigenvalues


round(
  manual_model$loadings,
  3
)


# ==============================================================================
# PART XXVII
#
# PREDICT PRINCIPAL COMPONENT SCORES FOR NEW DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. PCA Transform Function
# ------------------------------------------------------------------------------

predict_pca <- function(
    model,
    X_new,
    number_components = NULL
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  X_processed <- sweep(
    X_new,
    2,
    model$means,
    "-"
  )
  
  
  X_processed <- sweep(
    X_processed,
    2,
    model$standard_deviations,
    "/"
  )
  
  
  if (
    is.null(
      number_components
    )
  ) {
    
    number_components <- ncol(
      model$loadings
    )
  }
  
  
  X_processed %*%
    model$loadings[
      ,
      seq_len(
        number_components
      ),
      drop = FALSE
    ]
}


# ------------------------------------------------------------------------------
# 54. Example New Observations
# ------------------------------------------------------------------------------

X_new <- X[
  1:5,
  ,
  drop = FALSE
]


new_scores <- predict_pca(
  manual_model,
  X_new,
  number_components = 3
)


new_scores


# ------------------------------------------------------------------------------
# 55. Verify Against Original Scores
# ------------------------------------------------------------------------------

manual_model$scores[
  1:5,
  1:3,
  drop = FALSE
]


# ==============================================================================
# PART XXVIII
#
# RECONSTRUCT NEW OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Inverse PCA Transformation
# ------------------------------------------------------------------------------

inverse_pca <- function(
    model,
    scores,
    number_components = NULL
) {
  
  scores <- as.matrix(
    scores
  )
  
  
  if (
    is.null(
      number_components
    )
  ) {
    
    number_components <- ncol(
      scores
    )
  }
  
  
  V_M <- model$loadings[
    ,
    seq_len(
      number_components
    ),
    drop = FALSE
  ]
  
  
  X_processed <- scores[
    ,
    seq_len(
      number_components
    ),
    drop = FALSE
  ] %*%
    t(
      V_M
    )
  
  
  X_unscaled <- sweep(
    X_processed,
    2,
    model$standard_deviations,
    "*"
  )
  
  
  sweep(
    X_unscaled,
    2,
    model$means,
    "+"
  )
}


# ------------------------------------------------------------------------------
# 57. Reconstruct with Two Components
# ------------------------------------------------------------------------------

scores_2 <- predict_pca(
  manual_model,
  X_new,
  number_components = 2
)


X_new_reconstructed <- inverse_pca(
  manual_model,
  scores_2,
  number_components = 2
)


X_new


round(
  X_new_reconstructed,
  3
)


# ==============================================================================
# PART XXIX
#
# BIPLOT
# ==============================================================================


# ------------------------------------------------------------------------------
# 58. Manual PC1-PC2 Biplot
# ------------------------------------------------------------------------------

plot(
  scores[
    ,
    1
  ],
  scores[
    ,
    2
  ],
  pch = 19,
  cex = 0.6,
  xlab = "PC1",
  ylab = "PC2",
  main = "PCA Biplot"
)


# Scale loading arrows for visualization only.

arrow_scale <- 3


for (
  j in seq_len(p)
) {
  
  arrows(
    0,
    0,
    arrow_scale *
      loadings[
        j,
        1
      ],
    arrow_scale *
      loadings[
        j,
        2
      ],
    length = 0.08
  )
  
  
  text(
    arrow_scale *
      loadings[
        j,
        1
      ],
    arrow_scale *
      loadings[
        j,
        2
      ],
    labels =
      rownames(
        loadings
      )[j],
    pos = 3
  )
}


# ==============================================================================
# PART XXX
#
# PCA AND CORRELATED FEATURES
# ==============================================================================


# ------------------------------------------------------------------------------
# 59. Original Correlations
# ------------------------------------------------------------------------------

round(
  cor(
    X
  ),
  2
)


# ------------------------------------------------------------------------------
# 60. Principal Component Correlations
# ------------------------------------------------------------------------------

round(
  cor(
    scores
  ),
  6
)


# Highly correlated original features have been replaced by orthogonal
# principal component coordinates.


# ==============================================================================
# PART XXXI
#
# PCA AS DATA COMPRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 61. Compression Ratios
# ------------------------------------------------------------------------------

compression_table <- data.frame(
  Components =
    seq_len(p),
  Original_Values =
    n *
    p,
  Score_Values =
    n *
    seq_len(p),
  Loading_Values =
    p *
    seq_len(p)
)


compression_table$Total_PCA_Values <-
  compression_table$Score_Values +
  compression_table$Loading_Values


compression_table$Ratio <-
  compression_table$Total_PCA_Values /
  compression_table$Original_Values


compression_table


# This simple calculation ignores the feature means and other metadata, but it
# illustrates the low-rank representation:
#
#       X approximately Z_M V_M'


# ==============================================================================
# PART XXXII
#
# NOISY HIGH-DIMENSIONAL EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 62. Generate Low-Rank Signal in Many Dimensions
# ------------------------------------------------------------------------------

set.seed(777)

n_hd <- 250

p_hd <- 50


latent_hd <- matrix(
  rnorm(
    n_hd *
      3
  ),
  nrow =
    n_hd,
  ncol =
    3
)


loading_hd <- matrix(
  rnorm(
    p_hd *
      3
  ),
  nrow =
    p_hd,
  ncol =
    3
)


X_hd <- latent_hd %*%
  t(
    loading_hd
  ) +
  matrix(
    rnorm(
      n_hd *
        p_hd,
      sd = 0.5
    ),
    nrow =
      n_hd,
    ncol =
      p_hd
  )


colnames(
  X_hd
) <- paste0(
  "X",
  seq_len(
    p_hd
  )
)


# ------------------------------------------------------------------------------
# 63. PCA
# ------------------------------------------------------------------------------

PCA_hd <- pca_manual(
  X_hd
)


# ------------------------------------------------------------------------------
# 64. Scree Plot
# ------------------------------------------------------------------------------

plot(
  seq_len(
    p_hd
  ),
  PCA_hd$proportion_variance,
  type = "b",
  pch = 19,
  xlab = "Principal Component",
  ylab = "Proportion of Variance Explained",
  main = "Low-Rank Signal in High-Dimensional Data"
)


# ------------------------------------------------------------------------------
# 65. Cumulative Variance
# ------------------------------------------------------------------------------

plot(
  seq_len(
    p_hd
  ),
  PCA_hd$cumulative_variance,
  type = "l",
  xlab = "Number of Components",
  ylab = "Cumulative Variance",
  main = "High-Dimensional PCA"
)


abline(
  h = 0.90,
  lty = 2
)


# ==============================================================================
# PART XXXIII
#
# OPTIONAL VERIFICATION WITH prcomp()
# ==============================================================================


# ------------------------------------------------------------------------------
# 66. Built-In PCA
# ------------------------------------------------------------------------------

PCA_builtin <- prcomp(
  X,
  center = TRUE,
  scale. = FALSE
)


# ------------------------------------------------------------------------------
# 67. Compare Standard Deviations
# ------------------------------------------------------------------------------

data.frame(
  Manual_Eigenvalue =
    eigenvalues,
  prcomp_Eigenvalue =
    PCA_builtin$sdev^2
)


# ------------------------------------------------------------------------------
# 68. Compare Loadings
# ------------------------------------------------------------------------------

round(
  abs(
    loadings
  ),
  6
)


round(
  abs(
    PCA_builtin$rotation
  ),
  6
)


# ------------------------------------------------------------------------------
# 69. Compare Scores
# ------------------------------------------------------------------------------

max(
  abs(
    abs(
      scores
    ) -
      abs(
        PCA_builtin$x
      )
  )
)


# Sign differences between implementations are irrelevant.


# ==============================================================================
# PART XXXIV
#
# OPTIONAL VERIFICATION WITH princomp()
# ==============================================================================


# ------------------------------------------------------------------------------
# 70. princomp()
# ------------------------------------------------------------------------------

PCA_princomp <- princomp(
  X,
  cor = FALSE
)


summary(
  PCA_princomp
)


# Note:
#
# prcomp() uses SVD and is generally preferred numerically.
#
# princomp() uses an eigendecomposition approach and its variance convention can
# differ, so its reported component variances need not exactly equal prcomp().


# ==============================================================================
# PART XXXV
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 71. Output
# ------------------------------------------------------------------------------

cat(
  "Principal Components Analysis Summary\n"
)


cat(
  "-------------------------------------\n"
)


cat(
  "Observations:",
  n,
  "\n"
)


cat(
  "Original variables:",
  p,
  "\n"
)


cat(
  "Total variance:",
  round(
    total_variance,
    4
  ),
  "\n"
)


cat(
  "PC1 variance explained:",
  round(
    proportion_variance[1],
    4
  ),
  "\n"
)


cat(
  "PC2 variance explained:",
  round(
    proportion_variance[2],
    4
  ),
  "\n"
)


cat(
  "Variance explained by first 2 PCs:",
  round(
    cumulative_variance[2],
    4
  ),
  "\n"
)


cat(
  "Components for >= 90% variance:",
  M_90,
  "\n"
)


cat(
  "Components for >= 95% variance:",
  M_95,
  "\n"
)


cat(
  "Full reconstruction maximum error:",
  format(
    max(
      abs(
        X -
          X_reconstructed
      )
    ),
    scientific = TRUE
  ),
  "\n"
)