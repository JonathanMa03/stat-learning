# ==============================================================================
# Multidimensional Scaling
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Pairwise distances and dissimilarities
#   - Classical Multidimensional Scaling
#   - Double centering
#   - Gram matrices
#   - Eigendecomposition
#   - Low-dimensional embeddings
#   - Distance reconstruction
#   - Stress
#   - Choosing embedding dimension
#   - Euclidean vs non-Euclidean dissimilarities
#   - Relationship between MDS and PCA
#   - Metric MDS
#
#
# Classical MDS begins with a dissimilarity matrix:
#
#       D = [d_ij]
#
# and seeks low-dimensional coordinates:
#
#       z_1, ..., z_n
#
# such that:
#
#       ||z_i - z_j|| approximately d_ij
#
#
# For Euclidean distances, classical MDS uses:
#
#       B = -1/2 J D^2 J
#
# where:
#
#       J = I - (1/n) 11'
#
#
# Then:
#
#       B = V Lambda V'
#
# and the q-dimensional coordinates are:
#
#       Z_q = V_q Lambda_q^(1/2)
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE HIGH-DIMENSIONAL DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulation
# ------------------------------------------------------------------------------

set.seed(123)

n <- 150


# Three latent groups in two dimensions.

group <- rep(
  1:3,
  each = n / 3
)


latent <- matrix(
  0,
  nrow = n,
  ncol = 2
)


latent[
  group == 1,
] <- cbind(
  rnorm(
    sum(group == 1),
    mean = -3,
    sd = 0.7
  ),
  rnorm(
    sum(group == 1),
    mean = 0,
    sd = 0.7
  )
)


latent[
  group == 2,
] <- cbind(
  rnorm(
    sum(group == 2),
    mean = 3,
    sd = 0.7
  ),
  rnorm(
    sum(group == 2),
    mean = 0,
    sd = 0.7
  )
)


latent[
  group == 3,
] <- cbind(
  rnorm(
    sum(group == 3),
    mean = 0,
    sd = 0.7
  ),
  rnorm(
    sum(group == 3),
    mean = 4,
    sd = 0.7
  )
)


colnames(
  latent
) <- c(
  "Latent1",
  "Latent2"
)


# ------------------------------------------------------------------------------
# 2. Plot Latent Structure
# ------------------------------------------------------------------------------

plot(
  latent[
    ,
    1
  ],
  latent[
    ,
    2
  ],
  pch = 19,
  xlab = "Latent Dimension 1",
  ylab = "Latent Dimension 2",
  main = "Underlying Two-Dimensional Structure"
)


# ==============================================================================
# PART II
#
# EMBED THE LATENT DATA INTO A HIGHER-DIMENSIONAL FEATURE SPACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Linear Transformation
# ------------------------------------------------------------------------------

set.seed(456)


p <- 8


loading_matrix <- matrix(
  rnorm(
    2 *
      p
  ),
  nrow = 2,
  ncol = p
)


X <- latent %*%
  loading_matrix


# Add small noise.

X <- X +
  matrix(
    rnorm(
      n *
        p,
      sd = 0.15
    ),
    nrow = n,
    ncol = p
  )


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 4. Standardize Features
# ------------------------------------------------------------------------------

X <- scale(
  X
)


# ==============================================================================
# PART III
#
# PAIRWISE EUCLIDEAN DISTANCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Manual Euclidean Distance
# ------------------------------------------------------------------------------

euclidean_distance <- function(
    x,
    y
) {
  
  sqrt(
    sum(
      (x - y)^2
    )
  )
}


# ------------------------------------------------------------------------------
# 6. Manual Pairwise Distance Matrix
# ------------------------------------------------------------------------------

pairwise_euclidean <- function(
    X
) {
  
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
  
  
  if (
    n <= 1
  ) {
    
    return(
      D
    )
  }
  
  
  for (
    i in 1:(n - 1)
  ) {
    
    for (
      j in (i + 1):n
    ) {
      
      distance_ij <- euclidean_distance(
        X[
          i,
        ],
        X[
          j,
        ]
      )
      
      
      D[
        i,
        j
      ] <- distance_ij
      
      
      D[
        j,
        i
      ] <- distance_ij
    }
  }
  
  
  D
}


# ------------------------------------------------------------------------------
# 7. Calculate Distances
# ------------------------------------------------------------------------------

D <- pairwise_euclidean(
  X
)


round(
  D[
    1:5,
    1:5
  ],
  3
)


# ------------------------------------------------------------------------------
# 8. Verify Against dist()
# ------------------------------------------------------------------------------

D_builtin <- as.matrix(
  dist(
    X
  )
)


max(
  abs(
    D -
      D_builtin
  )
)


# ==============================================================================
# PART IV
#
# DISTANCE MATRIX VISUALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Reorder Observations by Group
# ------------------------------------------------------------------------------

ordering <- order(
  group
)


image(
  D[
    ordering,
    ordering
  ],
  axes = FALSE,
  main = "Pairwise Distance Matrix"
)


# Blocks of small distances correspond to observations from similar groups.


# ==============================================================================
# PART V
#
# CLASSICAL MULTIDIMENSIONAL SCALING
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Manual Classical MDS
# ------------------------------------------------------------------------------

classical_mds <- function(
    D,
    dimensions = 2,
    tolerance = 1e-10
) {
  
  D <- as.matrix(
    D
  )
  
  
  n <- nrow(
    D
  )
  
  
  if (
    ncol(D) !=
    n
  ) {
    
    stop(
      "D must be a square dissimilarity matrix."
    )
  }
  
  
  if (
    dimensions <
    1
  ) {
    
    stop(
      "dimensions must be at least 1."
    )
  }
  
  
  if (
    max(
      abs(
        D -
        t(D)
      )
    ) >
    1e-8
  ) {
    
    stop(
      "D must be symmetric."
    )
  }
  
  
  if (
    any(
      D <
      -1e-12
    )
  ) {
    
    stop(
      "D cannot contain negative dissimilarities."
    )
  }
  
  
  if (
    max(
      abs(
        diag(D)
      )
    ) >
    1e-8
  ) {
    
    stop(
      "The diagonal of D should be zero."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Squared distances
  # --------------------------------------------------------------------------
  
  D_squared <- D^2
  
  
  # --------------------------------------------------------------------------
  # Centering matrix
  #
  #       J = I - (1/n) 11'
  # --------------------------------------------------------------------------
  
  J <- diag(
    n
  ) -
    matrix(
      1 / n,
      nrow = n,
      ncol = n
    )
  
  
  # --------------------------------------------------------------------------
  # Double-centered Gram matrix
  #
  #       B = -1/2 J D^2 J
  # --------------------------------------------------------------------------
  
  B <- -0.5 *
    J %*%
    D_squared %*%
    J
  
  
  # Remove tiny numerical asymmetry.
  
  B <- (
    B +
      t(B)
  ) /
    2
  
  
  # --------------------------------------------------------------------------
  # Eigendecomposition
  # --------------------------------------------------------------------------
  
  eig <- eigen(
    B,
    symmetric = TRUE
  )
  
  
  eigenvalues <- eig$values
  
  eigenvectors <- eig$vectors
  
  
  # --------------------------------------------------------------------------
  # Positive eigenvalues correspond to Euclidean embedding dimensions.
  # --------------------------------------------------------------------------
  
  positive <- which(
    eigenvalues >
      tolerance
  )
  
  
  available_dimensions <- length(
    positive
  )
  
  
  dimensions_used <- min(
    dimensions,
    available_dimensions
  )
  
  
  if (
    dimensions_used ==
    0
  ) {
    
    coordinates <- matrix(
      0,
      nrow = n,
      ncol = 0
    )
    
  } else {
    
    selected <- positive[
      seq_len(
        dimensions_used
      )
    ]
    
    
    coordinates <- eigenvectors[
      ,
      selected,
      drop = FALSE
    ] %*%
      diag(
        sqrt(
          eigenvalues[
            selected
          ]
        ),
        nrow =
          dimensions_used
      )
    
    
    colnames(
      coordinates
    ) <- paste0(
      "MDS",
      seq_len(
        dimensions_used
      )
    )
  }
  
  
  list(
    coordinates =
      coordinates,
    eigenvalues =
      eigenvalues,
    eigenvectors =
      eigenvectors,
    gram_matrix =
      B,
    centering_matrix =
      J,
    squared_distances =
      D_squared,
    dimensions_used =
      dimensions_used
  )
}


# ==============================================================================
# PART VI
#
# FIT TWO-DIMENSIONAL CLASSICAL MDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Fit
# ------------------------------------------------------------------------------

MDS <- classical_mds(
  D,
  dimensions = 2
)


Z_mds <- MDS$coordinates


# ------------------------------------------------------------------------------
# 12. Coordinates
# ------------------------------------------------------------------------------

head(
  Z_mds
)


# ------------------------------------------------------------------------------
# 13. Plot
# ------------------------------------------------------------------------------

plot(
  Z_mds[
    ,
    1
  ],
  Z_mds[
    ,
    2
  ],
  pch = 19,
  xlab = "MDS Dimension 1",
  ylab = "MDS Dimension 2",
  main = "Classical Multidimensional Scaling"
)


# ==============================================================================
# PART VII
#
# UNDERSTAND DOUBLE CENTERING
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Row / Column Means of B
# ------------------------------------------------------------------------------

round(
  rowMeans(
    MDS$gram_matrix
  )[
    1:10
  ],
  10
)


round(
  colMeans(
    MDS$gram_matrix
  )[
    1:10
  ],
  10
)


# B is centered.


# ------------------------------------------------------------------------------
# 15. Verify Coordinates Reconstruct B
# ------------------------------------------------------------------------------

B_reconstructed_2D <- Z_mds %*%
  t(
    Z_mds
  )


mean(
  (
    MDS$gram_matrix -
      B_reconstructed_2D
  )^2
)


# Since the original data contain small higher-dimensional noise, the first
# two dimensions do not necessarily reconstruct B perfectly.


# ==============================================================================
# PART VIII
#
# EIGENVALUES
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Plot Eigenvalues
# ------------------------------------------------------------------------------

plot(
  MDS$eigenvalues,
  type = "b",
  pch = 19,
  xlab = "Dimension",
  ylab = "Eigenvalue",
  main = "Classical MDS Eigenvalues"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART IX
#
# EXPLAINED POSITIVE EIGENVALUE FRACTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Positive Eigenvalues
# ------------------------------------------------------------------------------

positive_eigenvalues <- MDS$eigenvalues[
  MDS$eigenvalues >
    1e-10
]


positive_fraction <- positive_eigenvalues /
  sum(
    positive_eigenvalues
  )


cumulative_positive_fraction <- cumsum(
  positive_fraction
)


data.frame(
  Dimension =
    seq_along(
      positive_eigenvalues
    ),
  Eigenvalue =
    positive_eigenvalues,
  Fraction =
    positive_fraction,
  Cumulative =
    cumulative_positive_fraction
)


# ------------------------------------------------------------------------------
# 18. Cumulative Plot
# ------------------------------------------------------------------------------

plot(
  cumulative_positive_fraction,
  type = "b",
  pch = 19,
  ylim = c(
    0,
    1
  ),
  xlab = "Number of Dimensions",
  ylab = "Cumulative Positive Eigenvalue Fraction",
  main = "Classical MDS Dimension Selection"
)


abline(
  h = 0.90,
  lty = 2
)


abline(
  h = 0.95,
  lty = 2
)


# ==============================================================================
# PART X
#
# DISTANCE RECONSTRUCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Distances in MDS Space
# ------------------------------------------------------------------------------

D_mds <- pairwise_euclidean(
  Z_mds
)


# ------------------------------------------------------------------------------
# 20. Original vs Embedded Distances
# ------------------------------------------------------------------------------

upper_triangle <- upper.tri(
  D
)


plot(
  D[
    upper_triangle
  ],
  D_mds[
    upper_triangle
  ],
  pch = 19,
  cex = 0.4,
  xlab = "Original Distance",
  ylab = "Embedded Distance",
  main = "Original vs MDS Distances"
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# ==============================================================================
# PART XI
#
# STRESS
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Raw Stress
#
#       Stress_raw = sum_{i<j} (d_ij - delta_ij)^2
# ------------------------------------------------------------------------------

raw_stress <- function(
    D_original,
    D_embedded
) {
  
  upper <- upper.tri(
    D_original
  )
  
  
  sum(
    (
      D_original[
        upper
      ] -
        D_embedded[
          upper
        ]
    )^2
  )
}


# ------------------------------------------------------------------------------
# 22. Normalized Stress
#
# A common normalized form:
#
#       sqrt[
#           sum(d_ij - delta_ij)^2
#           /
#           sum(d_ij^2)
#       ]
# ------------------------------------------------------------------------------

normalized_stress <- function(
    D_original,
    D_embedded
) {
  
  upper <- upper.tri(
    D_original
  )
  
  
  numerator <- sum(
    (
      D_original[
        upper
      ] -
        D_embedded[
          upper
        ]
    )^2
  )
  
  
  denominator <- sum(
    D_original[
      upper
    ]^2
  )
  
  
  sqrt(
    numerator /
      denominator
  )
}


# ------------------------------------------------------------------------------
# 23. Two-Dimensional Stress
# ------------------------------------------------------------------------------

raw_stress(
  D,
  D_mds
)


normalized_stress(
  D,
  D_mds
)


# ==============================================================================
# PART XII
#
# STRESS BY EMBEDDING DIMENSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Compare q = 1,...,8
# ------------------------------------------------------------------------------

maximum_dimension <- min(
  8,
  length(
    positive_eigenvalues
  )
)


stress_by_dimension <- numeric(
  maximum_dimension
)


for (
  q in seq_len(
    maximum_dimension
  )
) {
  
  fit_q <- classical_mds(
    D,
    dimensions = q
  )
  
  
  D_q <- pairwise_euclidean(
    fit_q$coordinates
  )
  
  
  stress_by_dimension[q] <- normalized_stress(
    D,
    D_q
  )
}


data.frame(
  Dimension =
    seq_len(
      maximum_dimension
    ),
  Stress =
    stress_by_dimension
)


# ------------------------------------------------------------------------------
# 25. Scree-Like Stress Plot
# ------------------------------------------------------------------------------

plot(
  seq_len(
    maximum_dimension
  ),
  stress_by_dimension,
  type = "b",
  pch = 19,
  xlab = "Embedding Dimension",
  ylab = "Normalized Stress",
  main = "MDS Stress by Dimension"
)


# ==============================================================================
# PART XIII
#
# BUILT-IN CLASSICAL MDS VERIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. cmdscale()
# ------------------------------------------------------------------------------

MDS_builtin <- cmdscale(
  as.dist(
    D
  ),
  k = 2,
  eig = TRUE
)


Z_builtin <- MDS_builtin$points


# ------------------------------------------------------------------------------
# 27. Distance Comparison
#
# Coordinates may differ by rotation/reflection, so compare pairwise distances.
# ------------------------------------------------------------------------------

D_builtin_embedding <- as.matrix(
  dist(
    Z_builtin
  )
)


max(
  abs(
    D_mds -
      D_builtin_embedding
  )
)


# ==============================================================================
# PART XIV
#
# ROTATION AND REFLECTION AMBIGUITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Rotate the Embedding
# ------------------------------------------------------------------------------

theta <- pi / 4


rotation_matrix <- matrix(
  c(
    cos(theta),
    -sin(theta),
    sin(theta),
    cos(theta)
  ),
  nrow = 2,
  byrow = TRUE
)


Z_rotated <- Z_mds %*%
  rotation_matrix


# ------------------------------------------------------------------------------
# 29. Distances Remain Unchanged
# ------------------------------------------------------------------------------

D_rotated <- pairwise_euclidean(
  Z_rotated
)


max(
  abs(
    D_mds -
      D_rotated
  )
)


# ------------------------------------------------------------------------------
# 30. Plot Rotated Version
# ------------------------------------------------------------------------------

plot(
  Z_rotated[
    ,
    1
  ],
  Z_rotated[
    ,
    2
  ],
  pch = 19,
  xlab = "Rotated Dimension 1",
  ylab = "Rotated Dimension 2",
  main = "Rotated MDS Embedding"
)


# MDS coordinates themselves are not unique.
# The pairwise distances are the important object.


# ==============================================================================
# PART XV
#
# CLASSICAL MDS AND PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Center X
# ------------------------------------------------------------------------------

X_centered <- sweep(
  X,
  2,
  colMeans(
    X
  ),
  "-"
)


# ------------------------------------------------------------------------------
# 32. PCA via SVD
# ------------------------------------------------------------------------------

PCA <- svd(
  X_centered
)


PCA_scores <- X_centered %*%
  PCA$v


# ------------------------------------------------------------------------------
# 33. First Two PCA Coordinates
# ------------------------------------------------------------------------------

Z_pca <- PCA_scores[
  ,
  1:2,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# 34. Compare Pairwise Distances
# ------------------------------------------------------------------------------

D_pca_2D <- pairwise_euclidean(
  Z_pca
)


max(
  abs(
    D_mds -
      D_pca_2D
  )
)


# For Euclidean distances computed from X, classical MDS and PCA recover the
# same principal coordinate geometry, up to sign / rotation / reflection.


# ==============================================================================
# PART XVI
#
# VERIFY GRAM MATRIX IDENTITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. X X' Gram Matrix
# ------------------------------------------------------------------------------

B_from_X <- X_centered %*%
  t(
    X_centered
  )


# ------------------------------------------------------------------------------
# 36. Compare with Double-Centered Distance Matrix
# ------------------------------------------------------------------------------

max(
  abs(
    B_from_X -
      MDS$gram_matrix
  )
)


# This is the key identity connecting PCA and classical MDS:
#
#       X_centered X_centered'
#
#              =
#
#       -1/2 J D^2 J


# ==============================================================================
# PART XVII
#
# MANHATTAN DISSIMILARITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Manual Manhattan Distance
# ------------------------------------------------------------------------------

manhattan_distance <- function(
    x,
    y
) {
  
  sum(
    abs(
      x -
        y
    )
  )
}


# ------------------------------------------------------------------------------
# 38. Pairwise Manhattan Matrix
# ------------------------------------------------------------------------------

pairwise_manhattan <- function(
    X
) {
  
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
  
  
  if (
    n <= 1
  ) {
    
    return(
      D
    )
  }
  
  
  for (
    i in 1:(n - 1)
  ) {
    
    for (
      j in (i + 1):n
    ) {
      
      distance_ij <- manhattan_distance(
        X[
          i,
        ],
        X[
          j,
        ]
      )
      
      
      D[
        i,
        j
      ] <- distance_ij
      
      
      D[
        j,
        i
      ] <- distance_ij
    }
  }
  
  
  D
}


# ------------------------------------------------------------------------------
# 39. Manhattan Dissimilarity Matrix
# ------------------------------------------------------------------------------

D_manhattan <- pairwise_manhattan(
  X
)


# ==============================================================================
# PART XVIII
#
# CLASSICAL MDS ON NON-EUCLIDEAN DISSIMILARITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Fit
# ------------------------------------------------------------------------------

MDS_manhattan <- classical_mds(
  D_manhattan,
  dimensions = 2
)


# ------------------------------------------------------------------------------
# 41. Eigenvalues
# ------------------------------------------------------------------------------

plot(
  MDS_manhattan$eigenvalues,
  type = "b",
  pch = 19,
  xlab = "Dimension",
  ylab = "Eigenvalue",
  main = "Eigenvalues: Manhattan Dissimilarities"
)


abline(
  h = 0,
  lty = 2
)


# Negative eigenvalues indicate that the supplied dissimilarities cannot be
# represented exactly as ordinary Euclidean distances in the reconstructed
# coordinate space.


# ------------------------------------------------------------------------------
# 42. Manhattan MDS Plot
# ------------------------------------------------------------------------------

Z_manhattan <- MDS_manhattan$coordinates


plot(
  Z_manhattan[
    ,
    1
  ],
  Z_manhattan[
    ,
    2
  ],
  pch = 19,
  xlab = "MDS Dimension 1",
  ylab = "MDS Dimension 2",
  main = "Classical MDS of Manhattan Dissimilarities"
)


# ==============================================================================
# PART XIX
#
# QUANTIFY NEGATIVE EIGENVALUES
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Positive and Negative Eigenvalue Mass
# ------------------------------------------------------------------------------

positive_mass <- sum(
  MDS_manhattan$eigenvalues[
    MDS_manhattan$eigenvalues >
      0
  ]
)


negative_mass <- sum(
  abs(
    MDS_manhattan$eigenvalues[
      MDS_manhattan$eigenvalues <
        0
    ]
  )
)


data.frame(
  Positive_Eigenvalue_Mass =
    positive_mass,
  Negative_Eigenvalue_Mass =
    negative_mass,
  Negative_to_Positive_Ratio =
    negative_mass /
    positive_mass
)


# ==============================================================================
# PART XX
#
# GENERIC DISSIMILITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. MDS Does Not Require Raw Features
#
# Suppose all we have is a dissimilarity matrix.
# ------------------------------------------------------------------------------

example_names <- c(
  "A",
  "B",
  "C",
  "D",
  "E"
)


D_example <- matrix(
  c(
    0, 1, 2, 4, 5,
    1, 0, 1.5, 3.5, 4,
    2, 1.5, 0, 2, 3,
    4, 3.5, 2, 0, 1,
    5, 4, 3, 1, 0
  ),
  nrow = 5,
  byrow = TRUE
)


rownames(
  D_example
) <- example_names


colnames(
  D_example
) <- example_names


# ------------------------------------------------------------------------------
# 45. Embed
# ------------------------------------------------------------------------------

example_mds <- classical_mds(
  D_example,
  dimensions = 2
)


# ------------------------------------------------------------------------------
# 46. Plot with Labels
# ------------------------------------------------------------------------------

plot(
  example_mds$coordinates[
    ,
    1
  ],
  example_mds$coordinates[
    ,
    2
  ],
  type = "n",
  xlab = "MDS Dimension 1",
  ylab = "MDS Dimension 2",
  main = "MDS from Dissimilarities Only"
)


text(
  example_mds$coordinates[
    ,
    1
  ],
  example_mds$coordinates[
    ,
    2
  ],
  labels =
    example_names
)


# ==============================================================================
# PART XXI
#
# METRIC STRESS-MINIMIZATION MDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Coordinate Vector Helpers
# ------------------------------------------------------------------------------

coordinates_to_vector <- function(
    Z
) {
  
  as.numeric(
    Z
  )
}


vector_to_coordinates <- function(
    parameters,
    n,
    dimensions
) {
  
  matrix(
    parameters,
    nrow = n,
    ncol = dimensions
  )
}


# ------------------------------------------------------------------------------
# 48. Metric MDS Stress Objective
#
# Here we directly minimize:
#
#       sum_{i<j} (delta_ij - d_ij(Z))^2
#
# where:
#
#       delta_ij = supplied dissimilarity
#       d_ij(Z)  = Euclidean distance in embedding
# ------------------------------------------------------------------------------

metric_mds_objective <- function(
    parameters,
    D_target,
    dimensions
) {
  
  n <- nrow(
    D_target
  )
  
  
  Z <- vector_to_coordinates(
    parameters,
    n,
    dimensions
  )
  
  
  # Translation does not affect distances.
  # Recenter to improve numerical stability.
  
  Z <- sweep(
    Z,
    2,
    colMeans(
      Z
    ),
    "-"
  )
  
  
  D_current <- pairwise_euclidean(
    Z
  )
  
  
  upper <- upper.tri(
    D_target
  )
  
  
  sum(
    (
      D_target[
        upper
      ] -
        D_current[
          upper
        ]
    )^2
  )
}


# ------------------------------------------------------------------------------
# 49. Manual Metric MDS
# ------------------------------------------------------------------------------

fit_metric_mds <- function(
    D,
    dimensions = 2,
    initial_coordinates = NULL,
    maxit = 1000
) {
  
  D <- as.matrix(
    D
  )
  
  
  n <- nrow(
    D
  )
  
  
  if (
    is.null(
      initial_coordinates
    )
  ) {
    
    initial_fit <- classical_mds(
      D,
      dimensions =
        dimensions
    )
    
    
    initial_coordinates <- initial_fit$coordinates
    
    
    # If classical MDS provides fewer positive dimensions than requested,
    # append small random coordinates.
    
    if (
      ncol(
        initial_coordinates
      ) <
      dimensions
    ) {
      
      set.seed(
        123
      )
      
      
      missing_dimensions <- dimensions -
        ncol(
          initial_coordinates
        )
      
      
      initial_coordinates <- cbind(
        initial_coordinates,
        matrix(
          rnorm(
            n *
              missing_dimensions,
            sd = 0.01
          ),
          nrow = n
        )
      )
    }
  }
  
  
  initial_coordinates <- initial_coordinates[
    ,
    seq_len(
      dimensions
    ),
    drop = FALSE
  ]
  
  
  initial_coordinates <- sweep(
    initial_coordinates,
    2,
    colMeans(
      initial_coordinates
    ),
    "-"
  )
  
  
  result <- optim(
    par =
      coordinates_to_vector(
        initial_coordinates
      ),
    fn =
      metric_mds_objective,
    D_target =
      D,
    dimensions =
      dimensions,
    method = "BFGS",
    control = list(
      maxit =
        maxit,
      reltol =
        1e-10
    )
  )
  
  
  coordinates <- vector_to_coordinates(
    result$par,
    n,
    dimensions
  )
  
  
  coordinates <- sweep(
    coordinates,
    2,
    colMeans(
      coordinates
    ),
    "-"
  )
  
  
  D_embedding <- pairwise_euclidean(
    coordinates
  )
  
  
  list(
    coordinates =
      coordinates,
    distance_matrix =
      D_embedding,
    raw_stress =
      raw_stress(
        D,
        D_embedding
      ),
    normalized_stress =
      normalized_stress(
        D,
        D_embedding
      ),
    optim_result =
      result
  )
}


# ==============================================================================
# PART XXII
#
# METRIC MDS ON MANHATTAN DISSIMILARITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Use a Smaller Subset for Optimization Speed
# ------------------------------------------------------------------------------

set.seed(123)


metric_indices <- sample(
  seq_len(n),
  60
)


D_metric <- D_manhattan[
  metric_indices,
  metric_indices
]


# ------------------------------------------------------------------------------
# 51. Classical Initialization
# ------------------------------------------------------------------------------

classical_metric <- classical_mds(
  D_metric,
  dimensions = 2
)


D_classical_metric <- pairwise_euclidean(
  classical_metric$coordinates
)


classical_metric_stress <- normalized_stress(
  D_metric,
  D_classical_metric
)


# ------------------------------------------------------------------------------
# 52. Direct Stress Optimization
# ------------------------------------------------------------------------------

metric_fit <- fit_metric_mds(
  D_metric,
  dimensions = 2,
  initial_coordinates =
    classical_metric$coordinates,
  maxit = 500
)


metric_fit$normalized_stress


# ------------------------------------------------------------------------------
# 53. Compare
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Classical MDS",
    "Metric Stress MDS"
  ),
  Normalized_Stress = c(
    classical_metric_stress,
    metric_fit$normalized_stress
  )
)


# Directly minimizing stress can improve distance matching because classical
# MDS optimizes a different algebraic criterion.


# ==============================================================================
# PART XXIII
#
# SHEPARD-STYLE DISTANCE PLOT
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Target vs Embedded Dissimilarity
# ------------------------------------------------------------------------------

upper_metric <- upper.tri(
  D_metric
)


plot(
  D_metric[
    upper_metric
  ],
  metric_fit$distance_matrix[
    upper_metric
  ],
  pch = 19,
  cex = 0.5,
  xlab = "Target Dissimilarity",
  ylab = "Embedding Distance",
  main = "Metric MDS Distance Fit"
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# ==============================================================================
# PART XXIV
#
# MULTIPLE RANDOM STARTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Metric MDS Can Have Local Minima
# ------------------------------------------------------------------------------

set.seed(999)


number_starts <- 5


start_stress <- numeric(
  number_starts
)


start_models <- vector(
  "list",
  number_starts
)


for (
  start in seq_len(
    number_starts
  )
) {
  
  if (
    start ==
    1
  ) {
    
    initial <- classical_metric$coordinates
    
  } else {
    
    initial <- matrix(
      rnorm(
        nrow(D_metric) *
          2
      ),
      nrow =
        nrow(D_metric),
      ncol = 2
    )
  }
  
  
  fit_start <- fit_metric_mds(
    D_metric,
    dimensions = 2,
    initial_coordinates =
      initial,
    maxit = 500
  )
  
  
  start_models[[start]] <-
    fit_start
  
  
  start_stress[start] <-
    fit_start$normalized_stress
}


data.frame(
  Start =
    seq_len(
      number_starts
    ),
  Normalized_Stress =
    start_stress
)


best_start <- which.min(
  start_stress
)


best_metric_model <- start_models[[best_start]]


# ==============================================================================
# PART XXV
#
# OUTLIER EFFECT
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Add an Extreme Observation
# ------------------------------------------------------------------------------

X_outlier <- rbind(
  X,
  rep(
    8,
    ncol(X)
  )
)


D_outlier <- pairwise_euclidean(
  X_outlier
)


MDS_outlier <- classical_mds(
  D_outlier,
  dimensions = 2
)


# ------------------------------------------------------------------------------
# 57. Plot
# ------------------------------------------------------------------------------

plot(
  MDS_outlier$coordinates[
    ,
    1
  ],
  MDS_outlier$coordinates[
    ,
    2
  ],
  pch = 19,
  xlab = "MDS Dimension 1",
  ylab = "MDS Dimension 2",
  main = "Effect of a Distant Observation"
)


points(
  MDS_outlier$coordinates[
    n + 1,
    1
  ],
  MDS_outlier$coordinates[
    n + 1,
    2
  ],
  pch = 8,
  cex = 2
)


# A very distant point can strongly affect the overall geometry.


# ==============================================================================
# PART XXVI
#
# FEATURE SCALING
# ==============================================================================


# ------------------------------------------------------------------------------
# 58. Unscaled Example
# ------------------------------------------------------------------------------

set.seed(222)


X_scale_demo <- cbind(
  Useful1 =
    latent[
      ,
      1
    ],
  Useful2 =
    latent[
      ,
      2
    ],
  LargeScaleNoise =
    rnorm(
      n,
      sd = 100
    )
)


# ------------------------------------------------------------------------------
# 59. MDS Without Standardization
# ------------------------------------------------------------------------------

D_unscaled <- pairwise_euclidean(
  X_scale_demo
)


MDS_unscaled <- classical_mds(
  D_unscaled,
  dimensions = 2
)


plot(
  MDS_unscaled$coordinates[
    ,
    1
  ],
  MDS_unscaled$coordinates[
    ,
    2
  ],
  pch = 19,
  xlab = "MDS1",
  ylab = "MDS2",
  main = "MDS Without Feature Scaling"
)


# ------------------------------------------------------------------------------
# 60. MDS After Standardization
# ------------------------------------------------------------------------------

X_scale_standardized <- scale(
  X_scale_demo
)


D_scaled <- pairwise_euclidean(
  X_scale_standardized
)


MDS_scaled <- classical_mds(
  D_scaled,
  dimensions = 2
)


plot(
  MDS_scaled$coordinates[
    ,
    1
  ],
  MDS_scaled$coordinates[
    ,
    2
  ],
  pch = 19,
  xlab = "MDS1",
  ylab = "MDS2",
  main = "MDS After Feature Scaling"
)


# ==============================================================================
# PART XXVII
#
# DISTANCE PRESERVATION CORRELATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 61. Correlation Between Original and Embedded Distances
# ------------------------------------------------------------------------------

distance_correlation_measure <- function(
    D_original,
    D_embedding
) {
  
  upper <- upper.tri(
    D_original
  )
  
  
  cor(
    D_original[
      upper
    ],
    D_embedding[
      upper
    ]
  )
}


distance_correlation_measure(
  D,
  D_mds
)


# This is a descriptive distance-preservation measure, not the statistical
# distance correlation dependence measure.


# ==============================================================================
# PART XXVIII
#
# RANK PRESERVATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 62. Spearman Correlation of Pairwise Distances
# ------------------------------------------------------------------------------

rank_distance_preservation <- function(
    D_original,
    D_embedding
) {
  
  upper <- upper.tri(
    D_original
  )
  
  
  cor(
    D_original[
      upper
    ],
    D_embedding[
      upper
    ],
    method = "spearman"
  )
}


rank_distance_preservation(
  D,
  D_mds
)


# Rank preservation becomes particularly important when we later consider
# nonmetric MDS.


# ==============================================================================
# PART XXIX
#
# DIMENSION SELECTION TABLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 63. Combine Eigenvalue and Stress Information
# ------------------------------------------------------------------------------

dimension_table <- data.frame(
  Dimension =
    seq_len(
      maximum_dimension
    ),
  Eigenvalue =
    positive_eigenvalues[
      seq_len(
        maximum_dimension
      )
    ],
  Cumulative_Eigenvalue_Fraction =
    cumulative_positive_fraction[
      seq_len(
        maximum_dimension
      )
    ],
  Normalized_Stress =
    stress_by_dimension
)


dimension_table


# ==============================================================================
# PART XXX
#
# CLASSICAL MDS FROM THE GRAM MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 64. Direct Eigendecomposition Demonstration
# ------------------------------------------------------------------------------

eig_B <- eigen(
  MDS$gram_matrix,
  symmetric = TRUE
)


Z_direct <- eig_B$vectors[
  ,
  1:2,
  drop = FALSE
] %*%
  diag(
    sqrt(
      eig_B$values[
        1:2
      ]
    ),
    nrow = 2
  )


max(
  abs(
    pairwise_euclidean(
      Z_direct
    ) -
      D_mds
  )
)


# ==============================================================================
# PART XXXI
#
# WHY SQUARED DISTANCES?
# ==============================================================================


# ------------------------------------------------------------------------------
# 65. Recover Inner Products from Distances
# ------------------------------------------------------------------------------

# For centered points:
#
#       ||x_i - x_j||^2
#
#       =
#
#       ||x_i||^2
#       +
#       ||x_j||^2
#       -
#       2 x_i' x_j
#
#
# Double centering removes the individual squared-norm terms and recovers:
#
#       x_i' x_j
#
#
# Therefore:
#
#       B = -1/2 J D^2 J
#
# is a Gram matrix.


# ==============================================================================
# PART XXXII
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nMultidimensional Scaling Summary\n"
)


cat(
  "--------------------------------\n"
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
  "Requested MDS dimensions:",
  2,
  "\n"
)


cat(
  "Positive classical MDS eigenvalues:",
  length(
    positive_eigenvalues
  ),
  "\n"
)


cat(
  "Two-dimensional normalized stress:",
  round(
    normalized_stress(
      D,
      D_mds
    ),
    6
  ),
  "\n"
)


cat(
  "Distance correlation:",
  round(
    distance_correlation_measure(
      D,
      D_mds
    ),
    6
  ),
  "\n"
)


cat(
  "Distance rank correlation:",
  round(
    rank_distance_preservation(
      D,
      D_mds
    ),
    6
  ),
  "\n"
)


cat(
  "Maximum difference between X X' and double-centered D^2:",
  format(
    max(
      abs(
        B_from_X -
          MDS$gram_matrix
      )
    ),
    scientific = TRUE
  ),
  "\n"
)


cat(
  "Metric MDS stress on Manhattan subset:",
  round(
    metric_fit$normalized_stress,
    6
  ),
  "\n"
)