# ==============================================================================
# Independent Component Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Blind source separation
#   - Latent independent components
#   - Linear mixing model
#   - Centering and whitening
#   - PCA as preprocessing
#   - Non-Gaussianity
#   - Central Limit Theorem intuition
#   - Kurtosis
#   - Negentropy
#   - FastICA fixed-point iteration
#   - Deflation
#   - Symmetric FastICA
#   - Sign and permutation ambiguity
#   - Source recovery
#
#
# ICA model:
#
#       X = S A'
#
# where:
#
#       S = latent independent sources
#       A = mixing matrix
#       X = observed mixtures
#
#
# Equivalently, for a single observation:
#
#       x = A s
#
#
# ICA seeks an unmixing matrix W such that:
#
#       s_hat = W x
#
#
# or, with observations stored in rows:
#
#       S_hat = X W'
#
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE INDEPENDENT LATENT SOURCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Sample Size
# ------------------------------------------------------------------------------

set.seed(123)

n <- 3000


# ------------------------------------------------------------------------------
# 2. Source 1: Sinusoidal Signal
# ------------------------------------------------------------------------------

time <- seq(
  0,
  8 * pi,
  length.out = n
)


source_1 <- sin(
  time
)


# ------------------------------------------------------------------------------
# 3. Source 2: Square-Wave-Like Signal
# ------------------------------------------------------------------------------

source_2 <- sign(
  sin(
    1.7 * time
  )
)


# ------------------------------------------------------------------------------
# 4. Source 3: Sawtooth-Like Signal
# ------------------------------------------------------------------------------

source_3 <- 2 * (
  (
    time / (2 * pi)
  ) -
    floor(
      time / (2 * pi) + 0.5
    )
)


# ------------------------------------------------------------------------------
# 5. Add Small Independent Noise
# ------------------------------------------------------------------------------

source_1 <- source_1 +
  rnorm(
    n,
    sd = 0.03
  )


source_2 <- source_2 +
  rnorm(
    n,
    sd = 0.03
  )


source_3 <- source_3 +
  rnorm(
    n,
    sd = 0.03
  )


# ------------------------------------------------------------------------------
# 6. Standardize Sources
# ------------------------------------------------------------------------------

standardize_vector <- function(
    x
) {
  
  as.numeric(
    (
      x -
        mean(x)
    ) /
      sd(x)
  )
}


source_1 <- standardize_vector(
  source_1
)


source_2 <- standardize_vector(
  source_2
)


source_3 <- standardize_vector(
  source_3
)


# ------------------------------------------------------------------------------
# 7. Source Matrix
#
# Observations are rows.
# Sources are columns.
# ------------------------------------------------------------------------------

S_true <- cbind(
  Source1 =
    source_1,
  Source2 =
    source_2,
  Source3 =
    source_3
)


# ------------------------------------------------------------------------------
# 8. Plot True Sources
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    3,
    1
  )
)


for (
  j in 1:3
) {
  
  plot(
    time,
    S_true[
      ,
      j
    ],
    type = "l",
    xlab = "Time",
    ylab = paste0(
      "Source ",
      j
    ),
    main = paste(
      "True Independent Source",
      j
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
# PART II
#
# CHECK SOURCE DEPENDENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Pairwise Correlations
# ------------------------------------------------------------------------------

round(
  cor(
    S_true
  ),
  3
)


# Low correlation is consistent with independence, but it does NOT prove
# independence.


# ------------------------------------------------------------------------------
# 10. Pairwise Scatterplots
# ------------------------------------------------------------------------------

pairs(
  S_true,
  pch = 19,
  cex = 0.25,
  main = "True Latent Sources"
)


# ==============================================================================
# PART III
#
# MIX THE SOURCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Mixing Matrix
# ------------------------------------------------------------------------------

A_true <- matrix(
  c(
    1.0,  0.5,  0.3,
    0.4,  1.0,  0.6,
    0.7,  0.2,  1.0
  ),
  nrow = 3,
  byrow = TRUE
)


A_true


det(
  A_true
)


# ------------------------------------------------------------------------------
# 12. Observed Mixtures
#
# With observations in rows:
#
#       X = S A'
# ------------------------------------------------------------------------------

X <- S_true %*%
  t(
    A_true
  )


colnames(
  X
) <- paste0(
  "Mixture",
  1:3
)


# ------------------------------------------------------------------------------
# 13. Plot Observed Mixtures
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    3,
    1
  )
)


for (
  j in 1:3
) {
  
  plot(
    time,
    X[
      ,
      j
    ],
    type = "l",
    xlab = "Time",
    ylab = paste0(
      "X",
      j
    ),
    main = paste(
      "Observed Mixture",
      j
    )
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 14. Mixture Correlations
# ------------------------------------------------------------------------------

round(
  cor(
    X
  ),
  3
)


# ==============================================================================
# PART IV
#
# PCA BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Center Data
# ------------------------------------------------------------------------------

X_means <- colMeans(
  X
)


X_centered <- sweep(
  X,
  2,
  X_means,
  "-"
)


# ------------------------------------------------------------------------------
# 16. PCA
# ------------------------------------------------------------------------------

PCA <- svd(
  X_centered
)


PCA_scores <- X_centered %*%
  PCA$v


# ------------------------------------------------------------------------------
# 17. PCA Components
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    3,
    1
  )
)


for (
  j in 1:3
) {
  
  plot(
    time,
    PCA_scores[
      ,
      j
    ],
    type = "l",
    xlab = "Time",
    ylab = paste0(
      "PC",
      j
    ),
    main = paste(
      "PCA Component",
      j
    )
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# PCA decorrelates the mixtures, but it does not necessarily recover the
# independent sources.


# ==============================================================================
# PART V
#
# KURTOSIS AND NON-GAUSSIANITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Manual Excess Kurtosis
# ------------------------------------------------------------------------------

excess_kurtosis <- function(
    x
) {
  
  x <- x -
    mean(x)
  
  
  variance_population <- mean(
    x^2
  )
  
  
  if (
    variance_population <=
    .Machine$double.eps
  ) {
    
    return(
      NA_real_
    )
  }
  
  
  mean(
    x^4
  ) /
    variance_population^2 -
    3
}


# ------------------------------------------------------------------------------
# 19. Source Kurtosis
# ------------------------------------------------------------------------------

source_kurtosis <- apply(
  S_true,
  2,
  excess_kurtosis
)


source_kurtosis


# ------------------------------------------------------------------------------
# 20. Mixture Kurtosis
# ------------------------------------------------------------------------------

mixture_kurtosis <- apply(
  X,
  2,
  excess_kurtosis
)


mixture_kurtosis


# ==============================================================================
# PART VI
#
# CENTRAL LIMIT THEOREM INTUITION
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Random Linear Combinations of Sources
# ------------------------------------------------------------------------------

set.seed(456)


number_random_directions <- 1000


random_kurtosis <- numeric(
  number_random_directions
)


for (
  b in seq_len(
    number_random_directions
  )
) {
  
  w <- rnorm(
    3
  )
  
  
  w <- w /
    sqrt(
      sum(
        w^2
      )
    )
  
  
  projection <- as.numeric(
    S_true %*%
      w
  )
  
  
  random_kurtosis[b] <- excess_kurtosis(
    projection
  )
}


hist(
  random_kurtosis,
  breaks = 40,
  xlab = "Excess Kurtosis",
  main = "Non-Gaussianity of Random Source Mixtures"
)


abline(
  v = source_kurtosis,
  lty = 2
)


# Random mixtures tend to be more Gaussian than the strongly non-Gaussian
# original sources. This is the core intuition behind ICA.


# ==============================================================================
# PART VII
#
# WHITENING
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Manual Whitening Function
# ------------------------------------------------------------------------------

whiten_data <- function(
    X,
    tolerance = 1e-10
) {
  
  X <- as.matrix(
    X
  )
  
  
  means <- colMeans(
    X
  )
  
  
  X_centered <- sweep(
    X,
    2,
    means,
    "-"
  )
  
  
  covariance_matrix <- crossprod(
    X_centered
  ) /
    nrow(
      X_centered
    )
  
  
  eig <- eigen(
    covariance_matrix,
    symmetric = TRUE
  )
  
  
  keep <- eig$values >
    tolerance
  
  
  eigenvalues <- eig$values[
    keep
  ]
  
  
  eigenvectors <- eig$vectors[
    ,
    keep,
    drop = FALSE
  ]
  
  
  # --------------------------------------------------------------------------
  # Whitening matrix:
  #
  #       V = E D^(-1/2) E'
  #
  # With observations stored in rows:
  #
  #       Z = X_centered V
  # --------------------------------------------------------------------------
  
  whitening_matrix <- eigenvectors %*%
    diag(
      1 /
        sqrt(
          eigenvalues
        ),
      nrow =
        length(
          eigenvalues
        )
    ) %*%
    t(
      eigenvectors
    )
  
  
  dewhitening_matrix <- eigenvectors %*%
    diag(
      sqrt(
        eigenvalues
      ),
      nrow =
        length(
          eigenvalues
        )
    ) %*%
    t(
      eigenvectors
    )
  
  
  Z <- X_centered %*%
    whitening_matrix
  
  
  list(
    centered =
      X_centered,
    means =
      means,
    covariance =
      covariance_matrix,
    eigenvalues =
      eigenvalues,
    eigenvectors =
      eigenvectors,
    whitening_matrix =
      whitening_matrix,
    dewhitening_matrix =
      dewhitening_matrix,
    whitened =
      Z
  )
}


# ------------------------------------------------------------------------------
# 23. Whiten Mixtures
# ------------------------------------------------------------------------------

whitening_result <- whiten_data(
  X
)


Z <- whitening_result$whitened


# ------------------------------------------------------------------------------
# 24. Verify Whitened Covariance
# ------------------------------------------------------------------------------

round(
  crossprod(
    Z
  ) /
    nrow(Z),
  4
)


# This should be approximately the identity matrix.


# ==============================================================================
# PART VIII
#
# VISUALIZE WHITENING
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Before Whitening
# ------------------------------------------------------------------------------

plot(
  X[
    ,
    1
  ],
  X[
    ,
    2
  ],
  pch = 19,
  cex = 0.3,
  xlab = "Mixture 1",
  ylab = "Mixture 2",
  main = "Before Whitening"
)


# ------------------------------------------------------------------------------
# 26. After Whitening
# ------------------------------------------------------------------------------

plot(
  Z[
    ,
    1
  ],
  Z[
    ,
    2
  ],
  pch = 19,
  cex = 0.3,
  xlab = "Whitened Coordinate 1",
  ylab = "Whitened Coordinate 2",
  main = "After Whitening"
)


# ==============================================================================
# PART IX
#
# FASTICA NONLINEARITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. tanh Nonlinearity
#
# FastICA commonly uses:
#
#       g(u) = tanh(alpha * u)
#
#       g'(u) = alpha * (1 - tanh(alpha*u)^2)
#
# ------------------------------------------------------------------------------

fastica_g <- function(
    u,
    alpha = 1
) {
  
  tanh(
    alpha *
      u
  )
}


fastica_g_prime <- function(
    u,
    alpha = 1
) {
  
  alpha *
    (
      1 -
        tanh(
          alpha *
            u
        )^2
    )
}


# ==============================================================================
# PART X
#
# ONE-COMPONENT FASTICA
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Fit One Independent Component
# ------------------------------------------------------------------------------

fit_one_fastica_component <- function(
    Z,
    previous_weights = NULL,
    alpha = 1,
    max_iterations = 1000,
    tolerance = 1e-7,
    seed = NULL
) {
  
  Z <- as.matrix(
    Z
  )
  
  
  p <- ncol(
    Z
  )
  
  
  if (
    !is.null(
      seed
    )
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  # Random unit vector.
  
  w <- rnorm(
    p
  )
  
  
  w <- w /
    sqrt(
      sum(
        w^2
      )
    )
  
  
  convergence_history <- numeric(
    max_iterations
  )
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    w_old <- w
    
    
    # ------------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------------
    
    projection <- as.numeric(
      Z %*%
        w
    )
    
    
    # ------------------------------------------------------------------------
    # FastICA fixed-point update
    #
    #       w_new =
    #
    #       E[ Z g(w'Z) ]
    #       -
    #       E[g'(w'Z)] w
    #
    # ------------------------------------------------------------------------
    
    g_values <- fastica_g(
      projection,
      alpha =
        alpha
    )
    
    
    g_prime_values <- fastica_g_prime(
      projection,
      alpha =
        alpha
    )
    
    
    w <- as.numeric(
      colMeans(
        Z *
          g_values
      ) -
        mean(
          g_prime_values
        ) *
        w
    )
    
    
    # ------------------------------------------------------------------------
    # Deflationary Orthogonalization
    #
    # Remove directions already discovered.
    # ------------------------------------------------------------------------
    
    if (
      !is.null(
        previous_weights
      )
    ) {
      
      for (
        j in seq_len(
          nrow(
            previous_weights
          )
        )
      ) {
        
        previous_w <- previous_weights[
          j,
        ]
        
        
        w <- w -
          sum(
            w *
              previous_w
          ) *
          previous_w
      }
    }
    
    
    # ------------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------------
    
    w_norm <- sqrt(
      sum(
        w^2
      )
    )
    
    
    if (
      w_norm <=
      .Machine$double.eps
    ) {
      
      stop(
        "FastICA encountered a near-zero weight vector."
      )
    }
    
    
    w <- w /
      w_norm
    
    
    # ------------------------------------------------------------------------
    # Sign-Invariant Convergence
    #
    # w and -w represent the same independent component.
    #
    # Therefore use:
    #
    #       1 - |w_new' w_old|
    #
    # ------------------------------------------------------------------------
    
    convergence_measure <- 1 -
      abs(
        sum(
          w *
            w_old
        )
      )
    
    
    convergence_history[
      iteration
    ] <- convergence_measure
    
    
    if (
      convergence_measure <
      tolerance
    ) {
      
      converged <- TRUE
      
      break
    }
  }
  
  
  convergence_history <- convergence_history[
    seq_len(
      iteration
    )
  ]
  
  
  list(
    weight =
      w,
    component =
      as.numeric(
        Z %*%
          w
      ),
    iterations =
      iteration,
    converged =
      converged,
    convergence_history =
      convergence_history
  )
}


# ==============================================================================
# PART XI
#
# FIRST INDEPENDENT COMPONENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Fit
# ------------------------------------------------------------------------------

ICA_component_1 <- fit_one_fastica_component(
  Z,
  seed = 123
)


ICA_component_1$weight


ICA_component_1$iterations


ICA_component_1$converged


# ------------------------------------------------------------------------------
# 30. Plot
# ------------------------------------------------------------------------------

plot(
  time,
  ICA_component_1$component,
  type = "l",
  xlab = "Time",
  ylab = "Recovered Signal",
  main = "First FastICA Component"
)


# ------------------------------------------------------------------------------
# 31. Convergence
# ------------------------------------------------------------------------------

plot(
  ICA_component_1$convergence_history,
  type = "b",
  pch = 19,
  xlab = "Iteration",
  ylab = "1 - |w_new' w_old|",
  main = "FastICA Convergence"
)


# ==============================================================================
# PART XII
#
# DEFLATIONARY FASTICA
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Fit Multiple Components
# ------------------------------------------------------------------------------

fit_fastica_deflation <- function(
    X,
    number_components = NULL,
    alpha = 1,
    max_iterations = 1000,
    tolerance = 1e-7,
    seed = 123
) {
  
  X <- as.matrix(
    X
  )
  
  
  whitening <- whiten_data(
    X
  )
  
  
  Z <- whitening$whitened
  
  
  p_whitened <- ncol(
    Z
  )
  
  
  if (
    is.null(
      number_components
    )
  ) {
    
    number_components <- p_whitened
  }
  
  
  number_components <- min(
    number_components,
    p_whitened
  )
  
  
  W_whitened <- matrix(
    0,
    nrow =
      number_components,
    ncol =
      p_whitened
  )
  
  
  components <- matrix(
    0,
    nrow =
      nrow(X),
    ncol =
      number_components
  )
  
  
  iterations <- integer(
    number_components
  )
  
  
  converged <- logical(
    number_components
  )
  
  
  convergence_histories <- vector(
    "list",
    number_components
  )
  
  
  for (
    component_index in seq_len(
      number_components
    )
  ) {
    
    previous_weights <- NULL
    
    
    if (
      component_index >
      1
    ) {
      
      previous_weights <- W_whitened[
        seq_len(
          component_index -
            1
        ),
        ,
        drop = FALSE
      ]
    }
    
    
    fit_component <- fit_one_fastica_component(
      Z,
      previous_weights =
        previous_weights,
      alpha =
        alpha,
      max_iterations =
        max_iterations,
      tolerance =
        tolerance,
      seed =
        seed +
        component_index
    )
    
    
    W_whitened[
      component_index,
    ] <-
      fit_component$weight
    
    
    components[
      ,
      component_index
    ] <-
      fit_component$component
    
    
    iterations[
      component_index
    ] <-
      fit_component$iterations
    
    
    converged[
      component_index
    ] <-
      fit_component$converged
    
    
    convergence_histories[[component_index]] <-
      fit_component$convergence_history
  }
  
  
  # --------------------------------------------------------------------------
  # Overall unmixing matrix for centered original data.
  #
  #       Z = X_centered V
  #
  #       S_hat = Z W_white'
  #
  # Therefore:
  #
  #       S_hat = X_centered V W_white'
  #
  # If we want:
  #
  #       S_hat = X_centered W_original'
  #
  # then:
  #
  #       W_original = W_white V'
  #
  # --------------------------------------------------------------------------
  
  W_original <- W_whitened %*%
    t(
      whitening$whitening_matrix
    )
  
  
  colnames(
    components
  ) <- paste0(
    "IC",
    seq_len(
      number_components
    )
  )
  
  
  list(
    components =
      components,
    W_whitened =
      W_whitened,
    W_original =
      W_original,
    whitening =
      whitening,
    iterations =
      iterations,
    converged =
      converged,
    convergence_histories =
      convergence_histories
  )
}


# ==============================================================================
# PART XIII
#
# FIT ICA
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Three Independent Components
# ------------------------------------------------------------------------------

ICA_model <- fit_fastica_deflation(
  X,
  number_components = 3,
  seed = 123
)


S_estimated <- ICA_model$components


ICA_model$converged


ICA_model$iterations


# ------------------------------------------------------------------------------
# 34. Plot Recovered Components
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    3,
    1
  )
)


for (
  j in 1:3
) {
  
  plot(
    time,
    S_estimated[
      ,
      j
    ],
    type = "l",
    xlab = "Time",
    ylab = paste0(
      "IC",
      j
    ),
    main = paste(
      "Recovered Independent Component",
      j
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
# PART XIV
#
# MATCH RECOVERED COMPONENTS TO TRUE SOURCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Absolute Correlation Matrix
#
# ICA cannot identify the original ordering or signs.
# Therefore compare absolute correlations.
# ------------------------------------------------------------------------------

source_correlation <- cor(
  S_true,
  S_estimated
)


round(
  source_correlation,
  3
)


round(
  abs(
    source_correlation
  ),
  3
)


# ==============================================================================
# PART XV
#
# MANUAL COMPONENT MATCHING
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Find Best Permutation
#
# Since there are only 3 components, manually enumerate all 3! permutations.
# ------------------------------------------------------------------------------

all_permutations_3 <- rbind(
  c(
    1,
    2,
    3
  ),
  c(
    1,
    3,
    2
  ),
  c(
    2,
    1,
    3
  ),
  c(
    2,
    3,
    1
  ),
  c(
    3,
    1,
    2
  ),
  c(
    3,
    2,
    1
  )
)


permutation_scores <- numeric(
  nrow(
    all_permutations_3
  )
)


for (
  i in seq_len(
    nrow(
      all_permutations_3
    )
  )
) {
  
  permutation_i <- all_permutations_3[
    i,
  ]
  
  
  permutation_scores[i] <- sum(
    abs(
      source_correlation[
        cbind(
          1:3,
          permutation_i
        )
      ]
    )
  )
}


best_permutation <- all_permutations_3[
  which.max(
    permutation_scores
  ),
]


best_permutation


# ------------------------------------------------------------------------------
# 37. Reorder Estimated Components
# ------------------------------------------------------------------------------

S_matched <- S_estimated[
  ,
  best_permutation,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# 38. Fix Sign Ambiguity
# ------------------------------------------------------------------------------

for (
  j in 1:3
) {
  
  if (
    cor(
      S_true[
        ,
        j
      ],
      S_matched[
        ,
        j
      ]
    ) <
    0
  ) {
    
    S_matched[
      ,
      j
    ] <- -S_matched[
      ,
      j
    ]
  }
}


# ------------------------------------------------------------------------------
# 39. Final Correlations
# ------------------------------------------------------------------------------

recovery_correlations <- diag(
  cor(
    S_true,
    S_matched
  )
)


recovery_correlations


# ==============================================================================
# PART XVI
#
# TRUE VS RECOVERED SOURCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Plot Each Source
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    3,
    1
  )
)


for (
  j in 1:3
) {
  
  plot(
    time,
    S_true[
      ,
      j
    ],
    type = "l",
    lwd = 2,
    xlab = "Time",
    ylab = "Signal",
    main = paste(
      "True vs Recovered Source",
      j
    )
  )
  
  
  lines(
    time,
    S_matched[
      ,
      j
    ],
    lty = 2
  )
  
  
  legend(
    "topright",
    legend = c(
      "True",
      "Recovered"
    ),
    lty = c(
      1,
      2
    ),
    bty = "n"
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART XVII
#
# RECOVERY METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Standardize Estimated Components Before RMSE
#
# ICA scale is not identifiable either, so align scale before calculating RMSE.
# ------------------------------------------------------------------------------

S_matched_standardized <- apply(
  S_matched,
  2,
  standardize_vector
)


source_RMSE <- numeric(
  3
)


for (
  j in 1:3
) {
  
  source_RMSE[j] <- sqrt(
    mean(
      (
        S_true[
          ,
          j
        ] -
          S_matched_standardized[
            ,
            j
          ]
      )^2
    )
  )
}


data.frame(
  Source =
    paste0(
      "Source",
      1:3
    ),
  Absolute_Correlation =
    abs(
      recovery_correlations
    ),
  RMSE =
    source_RMSE
)


# ==============================================================================
# PART XVIII
#
# PCA VS ICA SOURCE RECOVERY
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Best Correlation of Each True Source with Any PCA Component
# ------------------------------------------------------------------------------

PCA_standardized <- apply(
  PCA_scores[
    ,
    1:3,
    drop = FALSE
  ],
  2,
  standardize_vector
)


PCA_source_correlations <- abs(
  cor(
    S_true,
    PCA_standardized
  )
)


ICA_source_correlations <- abs(
  cor(
    S_true,
    S_estimated
  )
)


comparison_recovery <- data.frame(
  Source =
    paste0(
      "Source",
      1:3
    ),
  Best_PCA_Correlation =
    apply(
      PCA_source_correlations,
      1,
      max
    ),
  Best_ICA_Correlation =
    apply(
      ICA_source_correlations,
      1,
      max
    )
)


comparison_recovery


# ==============================================================================
# PART XIX
#
# UNCORRELATED VS INDEPENDENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Construct Uncorrelated but Dependent Variables
# ------------------------------------------------------------------------------

set.seed(555)


u <- rnorm(
  5000
)


v <- u^2 -
  1


cor(
  u,
  v
)


# Correlation should be approximately zero.


plot(
  u,
  v,
  pch = 19,
  cex = 0.25,
  xlab = "U",
  ylab = "U^2 - 1",
  main = "Uncorrelated Does Not Mean Independent"
)


# V is completely determined by U despite approximately zero correlation.


# ==============================================================================
# PART XX
#
# KURTOSIS PROJECTION PURSUIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Two-Dimensional Example
#
# Search directions on a circle and calculate non-Gaussianity.
# ------------------------------------------------------------------------------

Z_2d <- Z[
  ,
  1:2,
  drop = FALSE
]


angles <- seq(
  0,
  pi,
  length.out = 500
)


projection_kurtosis <- numeric(
  length(
    angles
  )
)


for (
  i in seq_along(
    angles
  )
) {
  
  direction <- c(
    cos(
      angles[i]
    ),
    sin(
      angles[i]
    )
  )
  
  
  projection <- as.numeric(
    Z_2d %*%
      direction
  )
  
  
  projection_kurtosis[i] <- excess_kurtosis(
    projection
  )
}


# ------------------------------------------------------------------------------
# 45. Plot Absolute Kurtosis
# ------------------------------------------------------------------------------

plot(
  angles,
  abs(
    projection_kurtosis
  ),
  type = "l",
  xlab = "Projection Angle",
  ylab = "|Excess Kurtosis|",
  main = "Projection Pursuit for Non-Gaussianity"
)


# ICA can be viewed as projection pursuit where the projection index measures
# non-Gaussianity.


# ==============================================================================
# PART XXI
#
# NEGENTROPY APPROXIMATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Log-Cosh Contrast
#
# A commonly used approximation to negentropy uses:
#
#       G(u) = log cosh(u)
#
# Compare E[G(Y)] to a Gaussian variable with the same standardization.
# ------------------------------------------------------------------------------

log_cosh_stable <- function(
    x
) {
  
  abs(x) +
    log1p(
      exp(
        -2 *
          abs(x)
      )
    ) -
    log(2)
}


approximate_negentropy <- function(
    y,
    gaussian_reference = NULL
) {
  
  y <- standardize_vector(
    y
  )
  
  
  if (
    is.null(
      gaussian_reference
    )
  ) {
    
    set.seed(12345)
    
    
    gaussian_reference <- rnorm(
      length(y)
    )
  }
  
  
  gaussian_reference <- standardize_vector(
    gaussian_reference
  )
  
  
  (
    mean(
      log_cosh_stable(
        y
      )
    ) -
      mean(
        log_cosh_stable(
          gaussian_reference
        )
      )
  )^2
}


# ------------------------------------------------------------------------------
# 47. Compare Sources, Mixtures, PCA, and ICA
# ------------------------------------------------------------------------------

negentropy_table <- data.frame(
  Signal = character(0),
  Approx_Negentropy = numeric(0)
)


for (
  j in 1:3
) {
  
  negentropy_table <- rbind(
    negentropy_table,
    data.frame(
      Signal =
        paste0(
          "True Source ",
          j
        ),
      Approx_Negentropy =
        approximate_negentropy(
          S_true[
            ,
            j
          ]
        )
    )
  )
  
  
  negentropy_table <- rbind(
    negentropy_table,
    data.frame(
      Signal =
        paste0(
          "Mixture ",
          j
        ),
      Approx_Negentropy =
        approximate_negentropy(
          X[
            ,
            j
          ]
        )
    )
  )
  
  
  negentropy_table <- rbind(
    negentropy_table,
    data.frame(
      Signal =
        paste0(
          "PCA ",
          j
        ),
      Approx_Negentropy =
        approximate_negentropy(
          PCA_scores[
            ,
            j
          ]
        )
    )
  )
  
  
  negentropy_table <- rbind(
    negentropy_table,
    data.frame(
      Signal =
        paste0(
          "ICA ",
          j
        ),
      Approx_Negentropy =
        approximate_negentropy(
          S_estimated[
            ,
            j
          ]
        )
    )
  )
}


negentropy_table


# ==============================================================================
# PART XXII
#
# WHY GAUSSIAN SOURCES ARE A PROBLEM
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Two Gaussian Sources
# ------------------------------------------------------------------------------

set.seed(777)


n_gaussian <- 5000


S_gaussian <- cbind(
  rnorm(
    n_gaussian
  ),
  rnorm(
    n_gaussian
  )
)


A_gaussian <- matrix(
  c(
    1,
    0.7,
    0.4,
    1
  ),
  nrow = 2,
  byrow = TRUE
)


X_gaussian <- S_gaussian %*%
  t(
    A_gaussian
  )


# ------------------------------------------------------------------------------
# 49. ICA
# ------------------------------------------------------------------------------

ICA_gaussian <- fit_fastica_deflation(
  X_gaussian,
  number_components = 2,
  seed = 123
)


# ------------------------------------------------------------------------------
# 50. Recovery Correlations
# ------------------------------------------------------------------------------

abs(
  cor(
    S_gaussian,
    ICA_gaussian$components
  )
)


# With multiple Gaussian independent components, ICA cannot uniquely determine
# their orientation after whitening.


# ==============================================================================
# PART XXIII
#
# SYMMETRIC FASTICA
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Symmetric Decorrelation
#
# Given W, transform it so:
#
#       W W' = I
#
# using:
#
#       W <- (W W')^(-1/2) W
# ------------------------------------------------------------------------------

symmetric_decorrelation <- function(
    W,
    tolerance = 1e-12
) {
  
  WWt <- W %*%
    t(W)
  
  
  eig <- eigen(
    WWt,
    symmetric = TRUE
  )
  
  
  eigenvalues <- pmax(
    eig$values,
    tolerance
  )
  
  
  inverse_square_root <- eig$vectors %*%
    diag(
      1 /
        sqrt(
          eigenvalues
        ),
      nrow =
        length(
          eigenvalues
        )
    ) %*%
    t(
      eig$vectors
    )
  
  
  inverse_square_root %*%
    W
}


# ------------------------------------------------------------------------------
# 52. Symmetric FastICA
# ------------------------------------------------------------------------------

fit_fastica_symmetric <- function(
    X,
    number_components = NULL,
    alpha = 1,
    max_iterations = 1000,
    tolerance = 1e-7,
    seed = 123
) {
  
  X <- as.matrix(
    X
  )
  
  
  whitening <- whiten_data(
    X
  )
  
  
  Z_full <- whitening$whitened
  
  
  p_whitened <- ncol(
    Z_full
  )
  
  
  if (
    is.null(
      number_components
    )
  ) {
    
    number_components <- p_whitened
  }
  
  
  if (
    number_components !=
    p_whitened
  ) {
    
    stop(
      paste(
        "This simple symmetric implementation extracts all",
        "whitened dimensions."
      )
    )
  }
  
  
  set.seed(
    seed
  )
  
  
  W <- matrix(
    rnorm(
      number_components^2
    ),
    nrow =
      number_components
  )
  
  
  W <- symmetric_decorrelation(
    W
  )
  
  
  convergence_history <- numeric(
    max_iterations
  )
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    W_old <- W
    
    
    # ------------------------------------------------------------------------
    # Project all observations onto all current directions.
    #
    # Z is n x p.
    # W' is p x p.
    #
    # projections is n x p.
    # ------------------------------------------------------------------------
    
    projections <- Z_full %*%
      t(W)
    
    
    G <- tanh(
      alpha *
        projections
    )
    
    
    G_prime <- alpha *
      (
        1 -
          G^2
      )
    
    
    # ------------------------------------------------------------------------
    # Simultaneous FastICA update:
    #
    # Each row w_j:
    #
    # E[Z g(w_j'Z)] - E[g'(w_j'Z)] w_j
    # ------------------------------------------------------------------------
    
    W_new <- G %>%
      crossprod(
        Z_full,
        .
      )
    
    
    # The previous expression would require magrittr and is intentionally not
    # used. Compute directly in base R below.
    
    
    W_new <- t(G) %*%
      Z_full /
      nrow(
        Z_full
      )
    
    
    derivative_means <- colMeans(
      G_prime
    )
    
    
    W_new <- W_new -
      derivative_means *
      W
    
    
    # ------------------------------------------------------------------------
    # Symmetric orthogonalization
    # ------------------------------------------------------------------------
    
    W <- symmetric_decorrelation(
      W_new
    )
    
    
    # ------------------------------------------------------------------------
    # Convergence is invariant to sign.
    # ------------------------------------------------------------------------
    
    alignment <- abs(
      diag(
        W %*%
          t(
            W_old
          )
      )
    )
    
    
    convergence_measure <- max(
      abs(
        1 -
          alignment
      )
    )
    
    
    convergence_history[
      iteration
    ] <- convergence_measure
    
    
    if (
      convergence_measure <
      tolerance
    ) {
      
      converged <- TRUE
      
      break
    }
  }
  
  
  convergence_history <- convergence_history[
    seq_len(
      iteration
    )
  ]
  
  
  components <- Z_full %*%
    t(W)
  
  
  W_original <- W %*%
    t(
      whitening$whitening_matrix
    )
  
  
  colnames(
    components
  ) <- paste0(
    "IC",
    seq_len(
      ncol(
        components
      )
    )
  )
  
  
  list(
    components =
      components,
    W_whitened =
      W,
    W_original =
      W_original,
    whitening =
      whitening,
    converged =
      converged,
    iterations =
      iteration,
    convergence_history =
      convergence_history
  )
}


# IMPORTANT:
#
# The function above intentionally stays package-free.
#
# Remove the following dead demonstration fragment if your R installation
# evaluates it during parsing:
#
#       W_new <- G %>% crossprod(...)
#
# We will define the clean version below without any pipe/package syntax.


# ==============================================================================
# PART XXIV
#
# CLEAN SYMMETRIC FASTICA IMPLEMENTATION
# ==============================================================================


fit_fastica_symmetric <- function(
    X,
    alpha = 1,
    max_iterations = 1000,
    tolerance = 1e-7,
    seed = 123
) {
  
  X <- as.matrix(
    X
  )
  
  
  whitening <- whiten_data(
    X
  )
  
  
  Z <- whitening$whitened
  
  
  p <- ncol(
    Z
  )
  
  
  set.seed(
    seed
  )
  
  
  W <- matrix(
    rnorm(
      p *
        p
    ),
    nrow =
      p,
    ncol =
      p
  )
  
  
  W <- symmetric_decorrelation(
    W
  )
  
  
  convergence_history <- numeric(
    max_iterations
  )
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    W_old <- W
    
    
    projections <- Z %*%
      t(W)
    
    
    G <- tanh(
      alpha *
        projections
    )
    
    
    G_prime <- alpha *
      (
        1 -
          G^2
      )
    
    
    W_new <- t(G) %*%
      Z /
      nrow(
        Z
      )
    
    
    derivative_means <- colMeans(
      G_prime
    )
    
    
    W_new <- W_new -
      sweep(
        W,
        1,
        derivative_means,
        "*"
      )
    
    
    W <- symmetric_decorrelation(
      W_new
    )
    
    
    alignment <- abs(
      diag(
        W %*%
          t(
            W_old
          )
      )
    )
    
    
    convergence_measure <- max(
      abs(
        1 -
          alignment
      )
    )
    
    
    convergence_history[
      iteration
    ] <- convergence_measure
    
    
    if (
      convergence_measure <
      tolerance
    ) {
      
      converged <- TRUE
      
      break
    }
  }
  
  
  convergence_history <- convergence_history[
    seq_len(
      iteration
    )
  ]
  
  
  components <- Z %*%
    t(W)
  
  
  W_original <- W %*%
    t(
      whitening$whitening_matrix
    )
  
  
  colnames(
    components
  ) <- paste0(
    "IC",
    seq_len(
      p
    )
  )
  
  
  list(
    components =
      components,
    W_whitened =
      W,
    W_original =
      W_original,
    whitening =
      whitening,
    converged =
      converged,
    iterations =
      iteration,
    convergence_history =
      convergence_history
  )
}


# ==============================================================================
# PART XXV
#
# FIT SYMMETRIC FASTICA
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. Fit
# ------------------------------------------------------------------------------

ICA_symmetric <- fit_fastica_symmetric(
  X,
  seed = 123
)


ICA_symmetric$converged


ICA_symmetric$iterations


# ------------------------------------------------------------------------------
# 54. Recovery
# ------------------------------------------------------------------------------

symmetric_correlations <- abs(
  cor(
    S_true,
    ICA_symmetric$components
  )
)


round(
  symmetric_correlations,
  3
)


# ==============================================================================
# PART XXVI
#
# MULTIPLE RANDOM STARTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Run Symmetric ICA from Several Initializations
# ------------------------------------------------------------------------------

ICA_starts <- vector(
  "list",
  10
)


start_scores <- numeric(
  10
)


for (
  start in 1:10
) {
  
  fit_start <- fit_fastica_symmetric(
    X,
    seed =
      100 +
      start
  )
  
  
  ICA_starts[[start]] <-
    fit_start
  
  
  correlation_start <- abs(
    cor(
      S_true,
      fit_start$components
    )
  )
  
  
  # For this simulation only, use true sources to evaluate recovery.
  # This is NOT available in real blind-source-separation problems.
  
  start_scores[start] <- sum(
    apply(
      correlation_start,
      1,
      max
    )
  )
}


data.frame(
  Start =
    1:10,
  Recovery_Score =
    start_scores
)


# ==============================================================================
# PART XXVII
#
# ICA AS DIMENSION REDUCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Higher-Dimensional Data with Three Sources
# ------------------------------------------------------------------------------

set.seed(888)


p_high <- 10


A_high <- matrix(
  runif(
    p_high *
      3,
    min = -1,
    max = 1
  ),
  nrow =
    p_high,
  ncol =
    3
)


X_high <- S_true %*%
  t(
    A_high
  )


X_high <- X_high +
  matrix(
    rnorm(
      n *
        p_high,
      sd = 0.05
    ),
    nrow =
      n
  )


# ------------------------------------------------------------------------------
# 57. Inspect Eigenvalues
# ------------------------------------------------------------------------------

X_high_centered <- sweep(
  X_high,
  2,
  colMeans(
    X_high
  ),
  "-"
)


high_covariance <- crossprod(
  X_high_centered
) /
  n


high_eigenvalues <- eigen(
  high_covariance,
  symmetric = TRUE
)$values


plot(
  high_eigenvalues,
  type = "b",
  pch = 19,
  xlab = "Component",
  ylab = "Eigenvalue",
  main = "Eigenvalues Before ICA"
)


# In noisy high-dimensional settings, PCA is often first used to reduce the
# data to the estimated latent dimension before ICA.


# ==============================================================================
# PART XXVIII
#
# PCA REDUCTION FOLLOWED BY ICA
# ==============================================================================


# ------------------------------------------------------------------------------
# 58. Reduce to Three Dimensions
# ------------------------------------------------------------------------------

PCA_high <- svd(
  X_high_centered
)


X_high_reduced <- X_high_centered %*%
  PCA_high$v[
    ,
    1:3,
    drop = FALSE
  ]


# ------------------------------------------------------------------------------
# 59. ICA on Reduced Representation
# ------------------------------------------------------------------------------

ICA_high <- fit_fastica_symmetric(
  X_high_reduced,
  seed = 123
)


# ------------------------------------------------------------------------------
# 60. Source Recovery
# ------------------------------------------------------------------------------

round(
  abs(
    cor(
      S_true,
      ICA_high$components
    )
  ),
  3
)


# ==============================================================================
# PART XXIX
#
# ESTIMATED UNMIXING MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 61. Inspect
# ------------------------------------------------------------------------------

ICA_model$W_original


# ------------------------------------------------------------------------------
# 62. Apply Directly to Centered Observations
# ------------------------------------------------------------------------------

S_direct <- ICA_model$whitening$centered %*%
  t(
    ICA_model$W_original
  )


max(
  abs(
    S_direct -
      ICA_model$components
  )
)


# ==============================================================================
# PART XXX
#
# APPROXIMATE MIXING MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 63. Recover Mixing Matrix
#
# For a square full-rank system:
#
#       W approximately A^{-1}
#
# so:
#
#       A_hat approximately W^{-1}
#
# Scale and permutation remain unidentified.
# ------------------------------------------------------------------------------

A_estimated <- solve(
  ICA_model$W_original
)


A_estimated


A_true


# Do not compare entries directly without first resolving component scaling,
# signs, and permutations.


# ==============================================================================
# PART XXXI
#
# INDEPENDENCE DIAGNOSTICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 64. Pairwise Correlation
# ------------------------------------------------------------------------------

round(
  cor(
    S_estimated
  ),
  4
)


# ------------------------------------------------------------------------------
# 65. Nonlinear Dependence Diagnostic
#
# Compare correlation between squared components.
# This is not a complete independence test, but it can detect some dependence
# invisible to ordinary correlation.
# ------------------------------------------------------------------------------

squared_component_correlation <- cor(
  S_estimated^2
)


round(
  squared_component_correlation,
  4
)


# ------------------------------------------------------------------------------
# 66. Compare PCA
# ------------------------------------------------------------------------------

round(
  cor(
    PCA_scores[
      ,
      1:3,
      drop = FALSE
    ]^2
  ),
  4
)


# ==============================================================================
# PART XXXII
#
# HISTOGRAMS
# ==============================================================================


# ------------------------------------------------------------------------------
# 67. True Sources
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    3,
    3
  )
)


for (
  j in 1:3
) {
  
  hist(
    S_true[
      ,
      j
    ],
    breaks = 40,
    main = paste(
      "True Source",
      j
    ),
    xlab = ""
  )
  
  
  hist(
    X[
      ,
      j
    ],
    breaks = 40,
    main = paste(
      "Mixture",
      j
    ),
    xlab = ""
  )
  
  
  hist(
    S_estimated[
      ,
      j
    ],
    breaks = 40,
    main = paste(
      "ICA Component",
      j
    ),
    xlab = ""
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART XXXIII
#
# OPTIONAL VERIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 68. fastICA Package
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "fastICA",
    quietly = TRUE
  )
) {
  
  package_ICA <- fastICA::fastICA(
    X,
    n.comp = 3,
    alg.typ = "parallel",
    fun = "logcosh",
    alpha = 1
  )
  
  
  package_components <- package_ICA$S
  
  
  cat(
    "\nAbsolute correlations between true sources and fastICA package:\n"
  )
  
  
  print(
    round(
      abs(
        cor(
          S_true,
          package_components
        )
      ),
      3
    )
  )
  
  
  cat(
    "\nPackage and manual components may appear in different orders\n"
  )
  
  
  cat(
    "and with opposite signs because ICA has permutation and sign ambiguity.\n"
  )
  
}


# ==============================================================================
# PART XXXIV
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 69. Summary
# ------------------------------------------------------------------------------

cat(
  "\nIndependent Component Analysis Summary\n"
)


cat(
  "--------------------------------------\n"
)


cat(
  "Observations:",
  n,
  "\n"
)


cat(
  "Observed mixtures:",
  ncol(
    X
  ),
  "\n"
)


cat(
  "Independent components:",
  ncol(
    S_estimated
  ),
  "\n"
)


cat(
  "Deflationary FastICA converged:",
  all(
    ICA_model$converged
  ),
  "\n"
)


cat(
  "Iterations by component:",
  ICA_model$iterations,
  "\n"
)


cat(
  "Matched source correlations:",
  round(
    recovery_correlations,
    4
  ),
  "\n"
)


cat(
  "Source RMSE after sign/scale/permutation alignment:",
  round(
    source_RMSE,
    4
  ),
  "\n"
)


cat(
  "Symmetric FastICA converged:",
  ICA_symmetric$converged,
  "\n"
)


cat(
  "Symmetric FastICA iterations:",
  ICA_symmetric$iterations,
  "\n"
)