# ==============================================================================
# Sparse Principal Components Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - PCA
#   - Sparse loadings
#   - L1 regularization
#   - Soft thresholding
#   - Penalized matrix decomposition
#   - Alternating optimization
#   - Deflation
#   - Explained variance
#   - Reconstruction
#   - Interpretability
#
#
# Ordinary PCA:
#
#       max_v Var(X v)
#
# subject to:
#
#       ||v||_2 = 1
#
#
# Sparse PCA adds a sparsity preference so that many entries of v are zero.
#
#
# This script uses a pedagogical penalized rank-one formulation:
#
#       max_{u,v}
#
#           u' X v - lambda ||v||_1
#
# subject to:
#
#       ||u||_2 <= 1
#       ||v||_2 <= 1
#
#
# Alternating updates:
#
#       u <- X v / ||X v||
#
#       v <- S(X' u, lambda)
#       v <- v / ||v||
#
# where S() is the soft-thresholding operator.
#
#
# Multiple sparse PCs are obtained using sequential deflation.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# SIMULATE DATA WITH A SPARSE LATENT STRUCTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------------------

set.seed(123)


n <- 400

p <- 20


# Three latent factors.

z1 <- rnorm(
  n
)

z2 <- rnorm(
  n
)

z3 <- rnorm(
  n
)


# --------------------------------------------------------------------------
# Sparse population loading patterns
#
# Factor 1:
#   variables 1:5
#
# Factor 2:
#   variables 6:10
#
# Factor 3:
#   variables 11:14
#
# Variables 15:20 are mostly noise.
# --------------------------------------------------------------------------

true_loading_1 <- c(
  1.2,
  1.0,
  0.9,
  -1.1,
  -0.8,
  rep(
    0,
    p - 5
  )
)


true_loading_2 <- c(
  rep(
    0,
    5
  ),
  1.1,
  -1.0,
  0.9,
  0.8,
  -1.2,
  rep(
    0,
    p - 10
  )
)


true_loading_3 <- c(
  rep(
    0,
    10
  ),
  1.0,
  0.9,
  -1.1,
  0.8,
  rep(
    0,
    p - 14
  )
)


# ------------------------------------------------------------------------------
# 2. Construct Observed Variables
# ------------------------------------------------------------------------------

X <- matrix(
  0,
  nrow = n,
  ncol = p
)


for (
  j in seq_len(p)
) {
  
  X[
    ,
    j
  ] <-
    true_loading_1[j] *
    z1 +
    true_loading_2[j] *
    z2 +
    true_loading_3[j] *
    z3 +
    rnorm(
      n,
      sd = 0.60
    )
}


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 3. Inspect
# ------------------------------------------------------------------------------

head(
  X
)


round(
  cor(
    X
  )[
    1:10,
    1:10
  ],
  2
)


# ==============================================================================
# PART II
#
# CENTER AND SCALE
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Means and Standard Deviations
# ------------------------------------------------------------------------------

feature_means <- colMeans(
  X
)


feature_sds <- apply(
  X,
  2,
  sd
)


# ------------------------------------------------------------------------------
# 5. Standardize
# ------------------------------------------------------------------------------

X_standardized <- sweep(
  X,
  2,
  feature_means,
  "-"
)


X_standardized <- sweep(
  X_standardized,
  2,
  feature_sds,
  "/"
)


round(
  colMeans(
    X_standardized
  ),
  8
)


round(
  apply(
    X_standardized,
    2,
    sd
  ),
  8
)


# ==============================================================================
# PART III
#
# ORDINARY PCA BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. SVD
# ------------------------------------------------------------------------------

PCA <- svd(
  X_standardized
)


PCA_loadings <- PCA$v


PCA_scores <- X_standardized %*%
  PCA_loadings


PCA_eigenvalues <- PCA$d^2 /
  (
    n -
      1
  )


# ------------------------------------------------------------------------------
# 7. First Three Ordinary PCA Loadings
# ------------------------------------------------------------------------------

ordinary_loading_table <- data.frame(
  Variable =
    colnames(
      X
    ),
  PC1 =
    PCA_loadings[
      ,
      1
    ],
  PC2 =
    PCA_loadings[
      ,
      2
    ],
  PC3 =
    PCA_loadings[
      ,
      3
    ]
)


round(
  ordinary_loading_table,
  3
)


# Ordinary PCA generally assigns nonzero weight to almost every variable.


# ==============================================================================
# PART IV
#
# SOFT THRESHOLDING
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Soft Threshold Function
# ------------------------------------------------------------------------------

soft_threshold <- function(
    z,
    lambda
) {
  
  sign(z) *
    pmax(
      abs(z) -
        lambda,
      0
    )
}


# ------------------------------------------------------------------------------
# 9. Example
# ------------------------------------------------------------------------------

example_values <- c(
  -3,
  -1,
  -0.4,
  0,
  0.3,
  1.5,
  4
)


data.frame(
  Original =
    example_values,
  Thresholded =
    soft_threshold(
      example_values,
      lambda = 1
    )
)


# ==============================================================================
# PART V
#
# ONE SPARSE PRINCIPAL COMPONENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Rank-One Sparse PCA
# ------------------------------------------------------------------------------

fit_sparse_pc <- function(
    X,
    lambda,
    initial_v = NULL,
    max_iterations = 500,
    tolerance = 1e-7,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Initialization
  # --------------------------------------------------------------------------
  
  if (
    is.null(
      initial_v
    )
  ) {
    
    svd_initial <- svd(
      X,
      nu = 0,
      nv = 1
    )
    
    
    v <- as.numeric(
      svd_initial$v[
        ,
        1
      ]
    )
    
  } else {
    
    v <- as.numeric(
      initial_v
    )
    
    
    if (
      length(v) !=
      p
    ) {
      
      stop(
        "initial_v has the wrong length."
      )
    }
    
    
    v_norm <- sqrt(
      sum(
        v^2
      )
    )
    
    
    if (
      v_norm <=
      .Machine$double.eps
    ) {
      
      stop(
        "initial_v cannot be the zero vector."
      )
    }
    
    
    v <- v /
      v_norm
  }
  
  
  objective_history <- numeric(
    max_iterations
  )
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    v_old <- v
    
    
    # ------------------------------------------------------------------------
    # Update u
    # ------------------------------------------------------------------------
    
    Xv <- as.numeric(
      X %*%
        v
    )
    
    
    Xv_norm <- sqrt(
      sum(
        Xv^2
      )
    )
    
    
    if (
      Xv_norm <=
      .Machine$double.eps
    ) {
      
      break
    }
    
    
    u <- Xv /
      Xv_norm
    
    
    # ------------------------------------------------------------------------
    # Update v
    # ------------------------------------------------------------------------
    
    raw_v <- as.numeric(
      crossprod(
        X,
        u
      )
    )
    
    
    v_thresholded <- soft_threshold(
      raw_v,
      lambda =
        lambda
    )
    
    
    v_norm <- sqrt(
      sum(
        v_thresholded^2
      )
    )
    
    
    if (
      v_norm <=
      .Machine$double.eps
    ) {
      
      # lambda is too large: all loadings were thresholded to zero.
      
      v <- rep(
        0,
        p
      )
      
      
      break
    }
    
    
    v <- v_thresholded /
      v_norm
    
    
    # ------------------------------------------------------------------------
    # Sign alignment for stable convergence diagnostics
    # ------------------------------------------------------------------------
    
    if (
      sum(
        v *
        v_old
      ) <
      0
    ) {
      
      v <- -v
    }
    
    
    # ------------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------------
    
    objective_history[
      iteration
    ] <- as.numeric(
      crossprod(
        u,
        X %*%
          v
      )
    ) -
      lambda *
      sum(
        abs(v)
      )
    
    
    # ------------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------------
    
    change <- sqrt(
      sum(
        (
          v -
            v_old
        )^2
      )
    )
    
    
    if (
      verbose
    ) {
      
      cat(
        "Iteration:",
        iteration,
        "| Nonzero loadings:",
        sum(
          abs(v) >
            1e-10
        ),
        "| Change:",
        round(
          change,
          8
        ),
        "\n"
      )
    }
    
    
    if (
      change <
      tolerance
    ) {
      
      converged <- TRUE
      
      break
    }
  }
  
  
  iterations_used <- iteration
  
  
  objective_history <- objective_history[
    seq_len(
      iterations_used
    )
  ]
  
  
  # --------------------------------------------------------------------------
  # Scores
  # --------------------------------------------------------------------------
  
  scores <- as.numeric(
    X %*%
      v
  )
  
  
  # --------------------------------------------------------------------------
  # Rank-one approximation
  #
  # Given unit v, regress X onto score t = Xv:
  #
  #       X_hat = t p'
  #
  # where p = X't / t't.
  #
  # This is used only for reconstruction diagnostics.
  # --------------------------------------------------------------------------
  
  score_denominator <- sum(
    scores^2
  )
  
  
  if (
    score_denominator >
    .Machine$double.eps
  ) {
    
    reconstruction_loading <- as.numeric(
      crossprod(
        X,
        scores
      ) /
        score_denominator
    )
    
    
    X_hat <- outer(
      scores,
      reconstruction_loading
    )
    
  } else {
    
    reconstruction_loading <- rep(
      0,
      p
    )
    
    
    X_hat <- matrix(
      0,
      nrow =
        nrow(X),
      ncol =
        p
    )
  }
  
  
  list(
    loading =
      v,
    scores =
      scores,
    reconstruction_loading =
      reconstruction_loading,
    reconstruction =
      X_hat,
    lambda =
      lambda,
    objective_history =
      objective_history,
    iterations =
      iterations_used,
    converged =
      converged,
    nonzero =
      sum(
        abs(v) >
          1e-10
      )
  )
}


# ==============================================================================
# PART VI
#
# FIT FIRST SPARSE PC
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Fit
# ------------------------------------------------------------------------------

lambda <- 2.5


sparse_PC1 <- fit_sparse_pc(
  X_standardized,
  lambda =
    lambda,
  verbose = TRUE
)


# ------------------------------------------------------------------------------
# 12. Sparse Loading
# ------------------------------------------------------------------------------

data.frame(
  Variable =
    colnames(
      X
    ),
  Ordinary_PC1 =
    PCA_loadings[
      ,
      1
    ],
  Sparse_PC1 =
    sparse_PC1$loading
)


# ------------------------------------------------------------------------------
# 13. Number of Nonzero Loadings
# ------------------------------------------------------------------------------

sparse_PC1$nonzero


# ==============================================================================
# PART VII
#
# LOADING COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Ordinary PC1
# ------------------------------------------------------------------------------

barplot(
  PCA_loadings[
    ,
    1
  ],
  names.arg =
    colnames(
      X
    ),
  las = 2,
  ylab = "Loading",
  main = "Ordinary PCA: PC1"
)


abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 15. Sparse PC1
# ------------------------------------------------------------------------------

barplot(
  sparse_PC1$loading,
  names.arg =
    colnames(
      X
    ),
  las = 2,
  ylab = "Loading",
  main = "Sparse PCA: SPC1"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART VIII
#
# SPARSITY PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Lambda Grid
# ------------------------------------------------------------------------------

lambda_grid <- seq(
  0,
  8,
  length.out = 60
)


loading_path <- matrix(
  NA_real_,
  nrow =
    length(
      lambda_grid
    ),
  ncol =
    p
)


nonzero_path <- integer(
  length(
    lambda_grid
  )
)


for (
  i in seq_along(
    lambda_grid
  )
) {
  
  fit_i <- fit_sparse_pc(
    X_standardized,
    lambda =
      lambda_grid[i],
    max_iterations = 300
  )
  
  
  loading_path[
    i,
  ] <- fit_i$loading
  
  
  nonzero_path[i] <-
    fit_i$nonzero
}


# ------------------------------------------------------------------------------
# 17. Loading Paths
# ------------------------------------------------------------------------------

matplot(
  lambda_grid,
  loading_path,
  type = "l",
  lty = 1,
  xlab = "Lambda",
  ylab = "Sparse Loading",
  main = "Sparse PCA Loading Path"
)


abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 18. Number of Active Variables
# ------------------------------------------------------------------------------

plot(
  lambda_grid,
  nonzero_path,
  type = "b",
  pch = 19,
  xlab = "Lambda",
  ylab = "Number of Nonzero Loadings",
  main = "Sparsity Path"
)


# ==============================================================================
# PART IX
#
# SPARSITY-VARIANCE TRADEOFF
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Variance Captured by Sparse Component
# ------------------------------------------------------------------------------

variance_captured <- numeric(
  length(
    lambda_grid
  )
)


for (
  i in seq_along(
    lambda_grid
  )
) {
  
  v_i <- loading_path[
    i,
  ]
  
  
  if (
    sum(
      v_i^2
    ) >
    0
  ) {
    
    scores_i <- X_standardized %*%
      v_i
    
    
    variance_captured[i] <- var(
      as.numeric(
        scores_i
      )
    )
    
  } else {
    
    variance_captured[i] <- 0
  }
}


# ------------------------------------------------------------------------------
# 20. Compare with Ordinary PC1
# ------------------------------------------------------------------------------

ordinary_PC1_variance <- var(
  PCA_scores[
    ,
    1
  ]
)


plot(
  nonzero_path,
  variance_captured,
  pch = 19,
  xlab = "Number of Nonzero Loadings",
  ylab = "Score Variance",
  main = "Interpretability vs Variance"
)


abline(
  h =
    ordinary_PC1_variance,
  lty = 2
)


# ==============================================================================
# PART X
#
# SEQUENTIAL SPARSE PCA WITH DEFLATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Fit Multiple Sparse Components
# ------------------------------------------------------------------------------

fit_sparse_pca <- function(
    X,
    number_components = 3,
    lambda = 2.5,
    max_iterations = 500,
    tolerance = 1e-7
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
    length(lambda) ==
    1
  ) {
    
    lambda <- rep(
      lambda,
      number_components
    )
  }
  
  
  if (
    length(lambda) !=
    number_components
  ) {
    
    stop(
      "lambda must have length 1 or number_components."
    )
  }
  
  
  residual_matrix <- X
  
  
  loadings <- matrix(
    0,
    nrow =
      p,
    ncol =
      number_components
  )
  
  
  scores <- matrix(
    0,
    nrow =
      n,
    ncol =
      number_components
  )
  
  
  reconstruction_loadings <- matrix(
    0,
    nrow =
      p,
    ncol =
      number_components
  )
  
  
  component_variance <- numeric(
    number_components
  )
  
  
  nonzero_counts <- integer(
    number_components
  )
  
  
  reconstruction_history <- vector(
    "list",
    number_components
  )
  
  
  for (
    component in seq_len(
      number_components
    )
  ) {
    
    fit_component <- fit_sparse_pc(
      residual_matrix,
      lambda =
        lambda[
          component
        ],
      max_iterations =
        max_iterations,
      tolerance =
        tolerance
    )
    
    
    loadings[
      ,
      component
    ] <-
      fit_component$loading
    
    
    scores[
      ,
      component
    ] <-
      fit_component$scores
    
    
    reconstruction_loadings[
      ,
      component
    ] <-
      fit_component$reconstruction_loading
    
    
    component_variance[
      component
    ] <-
      var(
        fit_component$scores
      )
    
    
    nonzero_counts[
      component
    ] <-
      fit_component$nonzero
    
    
    # ------------------------------------------------------------------------
    # Deflation
    #
    # Remove the rank-one approximation found by this component.
    # ------------------------------------------------------------------------
    
    residual_matrix <- residual_matrix -
      fit_component$reconstruction
    
    
    reconstruction_history[[component]] <-
      residual_matrix
  }
  
  
  colnames(
    loadings
  ) <- paste0(
    "SPC",
    seq_len(
      number_components
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
    "SPC",
    seq_len(
      number_components
    )
  )
  
  
  list(
    loadings =
      loadings,
    scores =
      scores,
    reconstruction_loadings =
      reconstruction_loadings,
    component_variance =
      component_variance,
    nonzero_counts =
      nonzero_counts,
    residual =
      residual_matrix,
    lambda =
      lambda
  )
}


# ==============================================================================
# PART XI
#
# FIT THREE SPARSE PRINCIPAL COMPONENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Fit
# ------------------------------------------------------------------------------

sparse_PCA <- fit_sparse_pca(
  X_standardized,
  number_components = 3,
  lambda = c(
    2.5,
    2.5,
    2.0
  )
)


# ------------------------------------------------------------------------------
# 23. Loadings
# ------------------------------------------------------------------------------

round(
  sparse_PCA$loadings,
  3
)


# ------------------------------------------------------------------------------
# 24. Active Variable Counts
# ------------------------------------------------------------------------------

data.frame(
  Component =
    paste0(
      "SPC",
      1:3
    ),
  Nonzero_Loadings =
    sparse_PCA$nonzero_counts,
  Score_Variance =
    sparse_PCA$component_variance
)


# ==============================================================================
# PART XII
#
# ACTIVE VARIABLES
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Extract Active Variables
# ------------------------------------------------------------------------------

active_variables <- function(
    loading,
    tolerance = 1e-8
) {
  
  names(
    loading
  )[
    abs(
      loading
    ) >
      tolerance
  ]
}


for (
  component in 1:3
) {
  
  loading_component <-
    sparse_PCA$loadings[
      ,
      component
    ]
  
  
  names(
    loading_component
  ) <-
    rownames(
      sparse_PCA$loadings
    )
  
  
  cat(
    "\nSPC",
    component,
    "active variables:\n",
    sep = ""
  )
  
  
  print(
    active_variables(
      loading_component
    )
  )
}


# ==============================================================================
# PART XIII
#
# COMPARE WITH TRUE SPARSE STRUCTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. True Active Sets
# ------------------------------------------------------------------------------

true_active_sets <- list(
  Factor1 =
    paste0(
      "X",
      1:5
    ),
  Factor2 =
    paste0(
      "X",
      6:10
    ),
  Factor3 =
    paste0(
      "X",
      11:14
    )
)


true_active_sets


# ------------------------------------------------------------------------------
# 27. Loading Heatmap-Like Plot
# ------------------------------------------------------------------------------

image(
  t(
    sparse_PCA$loadings
  ),
  axes = FALSE,
  xlab = "Variable",
  ylab = "Sparse Component",
  main = "Sparse PCA Loading Structure"
)


axis(
  1,
  at = seq(
    0,
    1,
    length.out = p
  ),
  labels =
    colnames(
      X
    ),
  las = 2
)


axis(
  2,
  at = seq(
    0,
    1,
    length.out = 3
  ),
  labels = paste0(
    "SPC",
    1:3
  )
)


# ==============================================================================
# PART XIV
#
# ORDINARY VS SPARSE LOADINGS
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. PC1 Comparison
# ------------------------------------------------------------------------------

comparison_PC1 <- data.frame(
  Variable =
    colnames(
      X
    ),
  Ordinary =
    PCA_loadings[
      ,
      1
    ],
  Sparse =
    sparse_PCA$loadings[
      ,
      1
    ]
)


comparison_PC1


# ------------------------------------------------------------------------------
# 29. PC2 Comparison
# ------------------------------------------------------------------------------

comparison_PC2 <- data.frame(
  Variable =
    colnames(
      X
    ),
  Ordinary =
    PCA_loadings[
      ,
      2
    ],
  Sparse =
    sparse_PCA$loadings[
      ,
      2
    ]
)


comparison_PC2


# ==============================================================================
# PART XV
#
# SPARSE SCORES
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. First Two Sparse Components
# ------------------------------------------------------------------------------

plot(
  sparse_PCA$scores[
    ,
    1
  ],
  sparse_PCA$scores[
    ,
    2
  ],
  pch = 19,
  cex = 0.5,
  xlab = "Sparse PC1",
  ylab = "Sparse PC2",
  main = "Sparse PCA Scores"
)


# ==============================================================================
# PART XVI
#
# COMPONENT CORRELATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Ordinary PCA Scores
# ------------------------------------------------------------------------------

round(
  cor(
    PCA_scores[
      ,
      1:3,
      drop = FALSE
    ]
  ),
  4
)


# ------------------------------------------------------------------------------
# 32. Sparse PCA Scores
# ------------------------------------------------------------------------------

round(
  cor(
    sparse_PCA$scores
  ),
  4
)


# Important:
#
# Sequential sparse components are generally NOT guaranteed to be exactly
# orthogonal or uncorrelated.
#
# Sparsity changes the PCA optimization problem.


# ==============================================================================
# PART XVII
#
# RECONSTRUCTION ERROR
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Sparse PCA Reconstruction
# ------------------------------------------------------------------------------

reconstruct_sparse_pca <- function(
    model,
    number_components = NULL
) {
  
  if (
    is.null(
      number_components
    )
  ) {
    
    number_components <- ncol(
      model$scores
    )
  }
  
  
  reconstruction <- matrix(
    0,
    nrow =
      nrow(
        model$scores
      ),
    ncol =
      nrow(
        model$reconstruction_loadings
      )
  )
  
  
  for (
    component in seq_len(
      number_components
    )
  ) {
    
    reconstruction <- reconstruction +
      outer(
        model$scores[
          ,
          component
        ],
        model$reconstruction_loadings[
          ,
          component
        ]
      )
  }
  
  
  reconstruction
}


X_sparse_reconstructed <- reconstruct_sparse_pca(
  sparse_PCA
)


sparse_reconstruction_MSE <- mean(
  (
    X_standardized -
      X_sparse_reconstructed
  )^2
)


sparse_reconstruction_MSE


# ------------------------------------------------------------------------------
# 34. Ordinary PCA Reconstruction with Three Components
# ------------------------------------------------------------------------------

ordinary_reconstruction <- PCA_scores[
  ,
  1:3,
  drop = FALSE
] %*%
  t(
    PCA_loadings[
      ,
      1:3,
      drop = FALSE
    ]
  )


ordinary_reconstruction_MSE <- mean(
  (
    X_standardized -
      ordinary_reconstruction
  )^2
)


ordinary_reconstruction_MSE


data.frame(
  Method = c(
    "Ordinary PCA",
    "Sparse PCA"
  ),
  Components = c(
    3,
    3
  ),
  Reconstruction_MSE = c(
    ordinary_reconstruction_MSE,
    sparse_reconstruction_MSE
  )
)


# Ordinary PCA should usually reconstruct better for a fixed number of
# components because minimizing squared reconstruction error is exactly what
# ordinary PCA is optimal for.


# ==============================================================================
# PART XVIII
#
# EXPLAINED VARIANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Total Variance
# ------------------------------------------------------------------------------

total_variance <- sum(
  apply(
    X_standardized,
    2,
    var
  )
)


# ------------------------------------------------------------------------------
# 36. Ordinary PCA Explained Variance
# ------------------------------------------------------------------------------

ordinary_PVE <- PCA_eigenvalues /
  total_variance


ordinary_PVE[
  1:5
]


# ------------------------------------------------------------------------------
# 37. Naive Sparse Score Variance Fractions
# ------------------------------------------------------------------------------

sparse_score_variance_fraction <-
  sparse_PCA$component_variance /
  total_variance


data.frame(
  Component =
    paste0(
      "SPC",
      1:3
    ),
  Score_Variance =
    sparse_PCA$component_variance,
  Fraction_Total_Variance =
    sparse_score_variance_fraction
)


# CAUTION:
#
# Sparse components are not generally orthogonal.
#
# Therefore their score variances should NOT simply be added and interpreted
# exactly like ordinary PCA eigenvalues.
#
# Reconstruction-based explained variance is safer.


# ==============================================================================
# PART XIX
#
# RECONSTRUCTION-BASED EXPLAINED VARIANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Reconstruction Fraction
# ------------------------------------------------------------------------------

total_sum_squares <- sum(
  X_standardized^2
)


sparse_residual_sum_squares <- sum(
  (
    X_standardized -
      X_sparse_reconstructed
  )^2
)


sparse_reconstruction_fraction <- 1 -
  sparse_residual_sum_squares /
  total_sum_squares


ordinary_residual_sum_squares <- sum(
  (
    X_standardized -
      ordinary_reconstruction
  )^2
)


ordinary_reconstruction_fraction <- 1 -
  ordinary_residual_sum_squares /
  total_sum_squares


data.frame(
  Method = c(
    "Ordinary PCA",
    "Sparse PCA"
  ),
  Reconstruction_Variance_Fraction = c(
    ordinary_reconstruction_fraction,
    sparse_reconstruction_fraction
  )
)


# ==============================================================================
# PART XX
#
# CHOOSE LAMBDA
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Train / Validation Split
# ------------------------------------------------------------------------------

set.seed(456)


train_indices <- sample(
  seq_len(n),
  size = floor(
    0.70 *
      n
  )
)


validation_indices <- setdiff(
  seq_len(n),
  train_indices
)


X_train <- X[
  train_indices,
  ,
  drop = FALSE
]


X_validation <- X[
  validation_indices,
  ,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# 40. Training-Only Standardization
# ------------------------------------------------------------------------------

train_means <- colMeans(
  X_train
)


train_sds <- apply(
  X_train,
  2,
  sd
)


X_train_standardized <- sweep(
  X_train,
  2,
  train_means,
  "-"
)


X_train_standardized <- sweep(
  X_train_standardized,
  2,
  train_sds,
  "/"
)


X_validation_standardized <- sweep(
  X_validation,
  2,
  train_means,
  "-"
)


X_validation_standardized <- sweep(
  X_validation_standardized,
  2,
  train_sds,
  "/"
)


# ==============================================================================
# PART XXI
#
# ONE-COMPONENT VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Lambda Grid
# ------------------------------------------------------------------------------

lambda_validation_grid <- seq(
  0,
  7,
  by = 0.25
)


validation_MSE <- numeric(
  length(
    lambda_validation_grid
  )
)


validation_nonzero <- integer(
  length(
    lambda_validation_grid
  )
)


# ------------------------------------------------------------------------------
# 42. Validation Reconstruction
# ------------------------------------------------------------------------------

for (
  i in seq_along(
    lambda_validation_grid
  )
) {
  
  fit_i <- fit_sparse_pc(
    X_train_standardized,
    lambda =
      lambda_validation_grid[i]
  )
  
  
  v_i <- fit_i$loading
  
  
  validation_nonzero[i] <-
    fit_i$nonzero
  
  
  if (
    sum(
      v_i^2
    ) <=
    .Machine$double.eps
  ) {
    
    validation_MSE[i] <- mean(
      X_validation_standardized^2
    )
    
    
    next
  }
  
  
  # Project validation observations onto learned sparse direction.
  
  validation_scores <- as.numeric(
    X_validation_standardized %*%
      v_i
  )
  
  
  # Use training reconstruction loading.
  
  validation_reconstruction <- outer(
    validation_scores,
    fit_i$reconstruction_loading
  )
  
  
  validation_MSE[i] <- mean(
    (
      X_validation_standardized -
        validation_reconstruction
    )^2
  )
}


validation_results <- data.frame(
  Lambda =
    lambda_validation_grid,
  Nonzero_Loadings =
    validation_nonzero,
  Validation_MSE =
    validation_MSE
)


validation_results


# ------------------------------------------------------------------------------
# 43. Best Lambda
# ------------------------------------------------------------------------------

best_lambda <- lambda_validation_grid[
  which.min(
    validation_MSE
  )
]


best_lambda


# ------------------------------------------------------------------------------
# 44. Validation Error Plot
# ------------------------------------------------------------------------------

plot(
  lambda_validation_grid,
  validation_MSE,
  type = "b",
  pch = 19,
  xlab = "Lambda",
  ylab = "Validation Reconstruction MSE",
  main = "Sparse PCA Lambda Selection"
)


abline(
  v =
    best_lambda,
  lty = 2
)


# ==============================================================================
# PART XXII
#
# SPARSITY-CONSTRAINED SELECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Best Model with at Most 5 Variables
# ------------------------------------------------------------------------------

eligible <- which(
  validation_nonzero <=
    5 &
    validation_nonzero >
    0
)


if (
  length(
    eligible
  ) >
  0
) {
  
  best_sparse_index <- eligible[
    which.min(
      validation_MSE[
        eligible
      ]
    )
  ]
  
  
  best_sparse_lambda <-
    lambda_validation_grid[
      best_sparse_index
    ]
  
  
  best_sparse_lambda
  
}


# This demonstrates an important point:
#
# Sparse PCA often involves an explicit tradeoff:
#
#       slightly more reconstruction error
#
# in exchange for:
#
#       far fewer variables.


# ==============================================================================
# PART XXIII
#
# HIGH-DIMENSIONAL EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. p >> n
# ------------------------------------------------------------------------------

set.seed(789)


n_hd <- 80

p_hd <- 300


latent_hd <- rnorm(
  n_hd
)


X_hd <- matrix(
  rnorm(
    n_hd *
      p_hd,
    sd = 1
  ),
  nrow =
    n_hd,
  ncol =
    p_hd
)


# Only first 10 variables carry strong common structure.

for (
  j in 1:10
) {
  
  X_hd[
    ,
    j
  ] <- (
    1 +
      runif(
        1,
        -0.2,
        0.2
      )
  ) *
    latent_hd +
    rnorm(
      n_hd,
      sd = 0.5
    )
}


colnames(
  X_hd
) <- paste0(
  "X",
  seq_len(
    p_hd
  )
)


# ------------------------------------------------------------------------------
# 47. Standardize
# ------------------------------------------------------------------------------

X_hd_standardized <- scale(
  X_hd
)


# ------------------------------------------------------------------------------
# 48. Sparse PC
# ------------------------------------------------------------------------------

sparse_hd <- fit_sparse_pc(
  X_hd_standardized,
  lambda = 4
)


sparse_hd$nonzero


hd_loading <- sparse_hd$loading


names(
  hd_loading
) <- colnames(
  X_hd
)


sort(
  abs(
    hd_loading
  ),
  decreasing = TRUE
)[1:20]


# Sparse PCA can be especially useful when p is large and only a small subset of
# variables drives a dominant latent factor.


# ==============================================================================
# PART XXIV
#
# STABILITY ACROSS BOOTSTRAP SAMPLES
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Bootstrap Variable Selection Frequency
# ------------------------------------------------------------------------------

set.seed(999)


B <- 100


selection_frequency <- numeric(
  p
)


for (
  b in seq_len(
    B
  )
) {
  
  bootstrap_indices <- sample(
    seq_len(n),
    size = n,
    replace = TRUE
  )
  
  
  X_bootstrap <- X_standardized[
    bootstrap_indices,
    ,
    drop = FALSE
  ]
  
  
  fit_b <- fit_sparse_pc(
    X_bootstrap,
    lambda = 2.5,
    max_iterations = 300
  )
  
  
  selection_frequency <- selection_frequency +
    (
      abs(
        fit_b$loading
      ) >
        1e-8
    )
}


selection_frequency <- selection_frequency /
  B


selection_table <- data.frame(
  Variable =
    colnames(
      X
    ),
  Selection_Frequency =
    selection_frequency
)


selection_table <- selection_table[
  order(
    selection_table$Selection_Frequency,
    decreasing = TRUE
  ),
]


selection_table


# ------------------------------------------------------------------------------
# 50. Plot Stability
# ------------------------------------------------------------------------------

barplot(
  selection_frequency,
  names.arg =
    colnames(
      X
    ),
  las = 2,
  ylim = c(
    0,
    1
  ),
  ylab = "Bootstrap Selection Frequency",
  main = "Sparse PCA Variable-Selection Stability"
)


# ==============================================================================
# PART XXV
#
# SPARSE PCA VS VARIABLE SELECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Variables Selected by First Sparse PC
# ------------------------------------------------------------------------------

SPC1_loading <- sparse_PCA$loadings[
  ,
  1
]


selected_SPC1 <- rownames(
  sparse_PCA$loadings
)[
  abs(
    SPC1_loading
  ) >
    1e-8
]


selected_SPC1


# Sparse PCA performs an unsupervised form of variable selection for each
# latent component.
#
# It does NOT select variables based on a response Y.


# ==============================================================================
# PART XXVI
#
# CORRELATED SIGNAL VARIABLES
# ==============================================================================


# ------------------------------------------------------------------------------
# 52. Highly Correlated Variables
# ------------------------------------------------------------------------------

set.seed(321)


latent_correlated <- rnorm(
  300
)


X_correlated <- cbind(
  X1 =
    latent_correlated +
    rnorm(
      300,
      sd = 0.2
    ),
  X2 =
    latent_correlated +
    rnorm(
      300,
      sd = 0.2
    ),
  X3 =
    latent_correlated +
    rnorm(
      300,
      sd = 0.2
    ),
  X4 =
    rnorm(
      300
    ),
  X5 =
    rnorm(
      300
    )
)


X_correlated <- scale(
  X_correlated
)


# ------------------------------------------------------------------------------
# 53. Different Lambda Values
# ------------------------------------------------------------------------------

correlated_lambda_values <- c(
  0,
  1,
  3,
  6
)


correlated_loadings <- matrix(
  NA_real_,
  nrow = 5,
  ncol =
    length(
      correlated_lambda_values
    )
)


for (
  i in seq_along(
    correlated_lambda_values
  )
) {
  
  fit_i <- fit_sparse_pc(
    X_correlated,
    lambda =
      correlated_lambda_values[i]
  )
  
  
  correlated_loadings[
    ,
    i
  ] <- fit_i$loading
}


rownames(
  correlated_loadings
) <- colnames(
  X_correlated
)


colnames(
  correlated_loadings
) <- paste0(
  "Lambda_",
  correlated_lambda_values
)


round(
  correlated_loadings,
  3
)


# L1-type sparsity may sometimes select only some members of a highly correlated
# group instead of retaining the whole group.


# ==============================================================================
# PART XXVII
#
# OPTIONAL PACKAGE VERIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. PMA Verification if Installed
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "PMA",
    quietly = TRUE
  )
) {
  
  cat(
    "\nPMA package is installed.\n"
  )
  
  
  cat(
    "PMA::SPC() can be used for comparison with a more established\n"
  )
  
  
  cat(
    "penalized matrix decomposition implementation.\n"
  )
  
  
  # Example:
  #
  # package_fit <- PMA::SPC(
  #   X_standardized,
  #   sumabsv = 2,
  #   K = 3,
  #   center = FALSE
  # )
  #
  # package_fit$v
  #
  # Note:
  # PMA uses its own penalty parameterization, so sumabsv is not numerically
  # equivalent to the lambda used in this manual implementation.
  
}


# ==============================================================================
# PART XXVIII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Summary
# ------------------------------------------------------------------------------

cat(
  "Sparse PCA Summary\n"
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
  "Variables:",
  p,
  "\n"
)


cat(
  "Sparse components fitted:",
  ncol(
    sparse_PCA$loadings
  ),
  "\n"
)


cat(
  "Nonzero loadings by component:",
  sparse_PCA$nonzero_counts,
  "\n"
)


cat(
  "Ordinary PCA 3-component reconstruction MSE:",
  round(
    ordinary_reconstruction_MSE,
    5
  ),
  "\n"
)


cat(
  "Sparse PCA 3-component reconstruction MSE:",
  round(
    sparse_reconstruction_MSE,
    5
  ),
  "\n"
)


cat(
  "Ordinary PCA reconstruction fraction:",
  round(
    ordinary_reconstruction_fraction,
    4
  ),
  "\n"
)


cat(
  "Sparse PCA reconstruction fraction:",
  round(
    sparse_reconstruction_fraction,
    4
  ),
  "\n"
)


cat(
  "Validation-selected first-component lambda:",
  best_lambda,
  "\n"
)