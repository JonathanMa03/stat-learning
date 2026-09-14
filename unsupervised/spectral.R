# ==============================================================================
# Spectral Clustering
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Similarity graphs
#   - Gaussian / RBF affinity
#   - K-nearest-neighbor graph
#   - Degree matrix
#   - Graph Laplacian
#   - Unnormalized Laplacian
#   - Normalized Laplacian
#   - Spectral embedding
#   - Connected components
#   - Eigengap
#   - K-means in spectral space
#   - Nonconvex cluster geometry
#   - Relationship to graph cuts
#
#
# Spectral clustering:
#
#   1. Build similarity matrix W
#   2. Compute degree matrix D
#   3. Construct graph Laplacian
#   4. Compute selected eigenvectors
#   5. Treat eigenvectors as a low-dimensional embedding
#   6. Run K-means in that embedding
#
#
# Normalized symmetric Laplacian:
#
#       L_sym = I - D^(-1/2) W D^(-1/2)
#
#
# Equivalently:
#
#       S = D^(-1/2) W D^(-1/2)
#
# and the smallest eigenvectors of L_sym are the largest eigenvectors of S.
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONCONVEX CLUSTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Two-Moons Data
# ------------------------------------------------------------------------------

set.seed(123)


n_per_cluster <- 200


theta_1 <- runif(
  n_per_cluster,
  0,
  pi
)


theta_2 <- runif(
  n_per_cluster,
  0,
  pi
)


cluster_1 <- cbind(
  cos(
    theta_1
  ),
  sin(
    theta_1
  )
)


cluster_2 <- cbind(
  1 -
    cos(
      theta_2
    ),
  -sin(
    theta_2
  ) -
    0.5
)


# Add noise.

cluster_1 <- cluster_1 +
  matrix(
    rnorm(
      2 *
        n_per_cluster,
      sd = 0.08
    ),
    ncol = 2
  )


cluster_2 <- cluster_2 +
  matrix(
    rnorm(
      2 *
        n_per_cluster,
      sd = 0.08
    ),
    ncol = 2
  )


X <- rbind(
  cluster_1,
  cluster_2
)


true_cluster <- c(
  rep(
    1,
    n_per_cluster
  ),
  rep(
    2,
    n_per_cluster
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
  pch = true_cluster,
  xlab = "X1",
  ylab = "X2",
  main = "Two-Moons Data"
)


# K-means has difficulty because these clusters are not approximately spherical.


# ==============================================================================
# PART II
#
# BASELINE K-MEANS
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. K-Means in Original Space
# ------------------------------------------------------------------------------

set.seed(123)


kmeans_original <- kmeans(
  X,
  centers = 2,
  nstart = 50
)


# ------------------------------------------------------------------------------
# 4. Plot
# ------------------------------------------------------------------------------

plot(
  X,
  pch = kmeans_original$cluster,
  xlab = "X1",
  ylab = "X2",
  main = "K-Means in Original Space"
)


# ==============================================================================
# PART III
#
# PAIRWISE DISTANCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Squared Distance Matrix
# ------------------------------------------------------------------------------

squared_distance_matrix <- function(
    X
) {
  
  X <- as.matrix(
    X
  )
  
  
  squared_norms <- rowSums(
    X^2
  )
  
  
  distances_squared <- outer(
    squared_norms,
    squared_norms,
    "+"
  ) -
    2 *
    X %*%
    t(
      X
    )
  
  
  pmax(
    distances_squared,
    0
  )
}


# ------------------------------------------------------------------------------
# 6. Distances
# ------------------------------------------------------------------------------

D_squared <- squared_distance_matrix(
  X
)


D_euclidean <- sqrt(
  D_squared
)


# ==============================================================================
# PART IV
#
# GAUSSIAN AFFINITY MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. RBF Similarity
#
#       W_ij = exp(-||x_i-x_j||^2 / (2 sigma^2))
# ------------------------------------------------------------------------------

rbf_affinity <- function(
    X,
    sigma
) {
  
  D_squared <- squared_distance_matrix(
    X
  )
  
  
  W <- exp(
    -D_squared /
      (
        2 *
          sigma^2
      )
  )
  
  
  diag(
    W
  ) <- 0
  
  
  W
}


# ------------------------------------------------------------------------------
# 8. Choose Sigma
# ------------------------------------------------------------------------------

sigma <- 0.25


W <- rbf_affinity(
  X,
  sigma =
    sigma
)


# ------------------------------------------------------------------------------
# 9. Inspect
# ------------------------------------------------------------------------------

round(
  W[
    1:5,
    1:5
  ],
  3
)


# ==============================================================================
# PART V
#
# K-NEAREST-NEIGHBOR GRAPH
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. KNN Mask
# ------------------------------------------------------------------------------

knn_mask <- function(
    D,
    k
) {
  
  n <- nrow(
    D
  )
  
  
  mask <- matrix(
    FALSE,
    nrow = n,
    ncol = n
  )
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    distances_i <- D[
      i,
    ]
    
    
    distances_i[i] <- Inf
    
    
    neighbors <- order(
      distances_i
    )[
      seq_len(
        k
      )
    ]
    
    
    mask[
      i,
      neighbors
    ] <- TRUE
  }
  
  
  # Symmetrize.
  
  mask <- mask |
    t(
      mask
    )
  
  
  diag(
    mask
  ) <- FALSE
  
  
  mask
}


# ------------------------------------------------------------------------------
# 11. Sparse Local Graph
# ------------------------------------------------------------------------------

k_neighbors <- 12


neighbor_mask <- knn_mask(
  D_euclidean,
  k =
    k_neighbors
)


W_knn <- W


W_knn[
  !neighbor_mask
] <- 0


# ==============================================================================
# PART VI
#
# VISUALIZE GRAPH
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. KNN Graph
# ------------------------------------------------------------------------------

plot(
  X,
  type = "n",
  xlab = "X1",
  ylab = "X2",
  main = "Similarity Graph"
)


for (
  i in 1:(n - 1)
) {
  
  for (
    j in (i + 1):n
  ) {
    
    if (
      W_knn[
        i,
        j
      ] >
      0
    ) {
      
      segments(
        X[
          i,
          1
        ],
        X[
          i,
          2
        ],
        X[
          j,
          1
        ],
        X[
          j,
          2
        ]
      )
    }
  }
}


points(
  X,
  pch = 19,
  cex = 0.5
)


# ==============================================================================
# PART VII
#
# DEGREE MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Node Degrees
# ------------------------------------------------------------------------------

degrees <- rowSums(
  W_knn
)


summary(
  degrees
)


# ------------------------------------------------------------------------------
# 14. Degree Matrix
# ------------------------------------------------------------------------------

D_degree <- diag(
  degrees
)


# ==============================================================================
# PART VIII
#
# GRAPH LAPLACIANS
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Unnormalized Laplacian
#
#       L = D - W
# ------------------------------------------------------------------------------

L_unnormalized <- D_degree -
  W_knn


# ------------------------------------------------------------------------------
# 16. Random-Walk Laplacian
#
#       L_rw = I - D^(-1) W
# ------------------------------------------------------------------------------

inverse_degrees <- 1 /
  pmax(
    degrees,
    1e-12
  )


L_random_walk <- diag(
  n
) -
  diag(
    inverse_degrees
  ) %*%
  W_knn


# ------------------------------------------------------------------------------
# 17. Symmetric Normalized Laplacian
#
#       L_sym = I - D^(-1/2) W D^(-1/2)
# ------------------------------------------------------------------------------

inverse_sqrt_degrees <- 1 /
  sqrt(
    pmax(
      degrees,
      1e-12
    )
  )


D_inverse_sqrt <- diag(
  inverse_sqrt_degrees
)


L_symmetric <- diag(
  n
) -
  D_inverse_sqrt %*%
  W_knn %*%
  D_inverse_sqrt


# ==============================================================================
# PART IX
#
# LAPLACIAN PROPERTIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Symmetry
# ------------------------------------------------------------------------------

max(
  abs(
    L_symmetric -
      t(
        L_symmetric
      )
  )
)


# ------------------------------------------------------------------------------
# 19. Eigenvalues Should Be Nonnegative
# ------------------------------------------------------------------------------

laplacian_eigen <- eigen(
  L_symmetric,
  symmetric = TRUE
)


laplacian_values <- sort(
  laplacian_eigen$values
)


head(
  laplacian_values,
  10
)


# Numerical values may be slightly below zero due to floating-point error.


# ==============================================================================
# PART X
#
# CONNECTED COMPONENTS AND ZERO EIGENVALUES
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Count Near-Zero Eigenvalues
# ------------------------------------------------------------------------------

sum(
  abs(
    laplacian_values
  ) <
    1e-8
)


# The multiplicity of eigenvalue 0 equals the number of connected components
# in the graph.


# ==============================================================================
# PART XI
#
# MANUAL NORMALIZED SPECTRAL CLUSTERING
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Spectral Clustering Function
# ------------------------------------------------------------------------------

spectral_clustering <- function(
    X,
    number_clusters,
    sigma,
    k_neighbors = NULL,
    nstart = 50,
    seed = 123
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  D_squared <- squared_distance_matrix(
    X
  )
  
  
  D <- sqrt(
    D_squared
  )
  
  
  W <- exp(
    -D_squared /
      (
        2 *
          sigma^2
      )
  )
  
  
  diag(
    W
  ) <- 0
  
  
  # --------------------------------------------------------------------------
  # Optional KNN sparsification
  # --------------------------------------------------------------------------
  
  if (
    !is.null(
      k_neighbors
    )
  ) {
    
    mask <- knn_mask(
      D,
      k =
        k_neighbors
    )
    
    
    W[
      !mask
    ] <- 0
  }
  
  
  # --------------------------------------------------------------------------
  # Degree matrix
  # --------------------------------------------------------------------------
  
  degrees <- rowSums(
    W
  )
  
  
  if (
    any(
      degrees <=
      1e-12
    )
  ) {
    
    warning(
      "Graph contains nearly isolated observations."
    )
  }
  
  
  inverse_sqrt_degrees <- 1 /
    sqrt(
      pmax(
        degrees,
        1e-12
      )
    )
  
  
  D_inverse_sqrt <- diag(
    inverse_sqrt_degrees
  )
  
  
  # --------------------------------------------------------------------------
  # Symmetric normalized similarity matrix
  #
  #       S = D^(-1/2) W D^(-1/2)
  #
  # Largest eigenvectors of S correspond to smallest eigenvectors of L_sym.
  # --------------------------------------------------------------------------
  
  normalized_similarity <- D_inverse_sqrt %*%
    W %*%
    D_inverse_sqrt
  
  
  eig <- eigen(
    normalized_similarity,
    symmetric = TRUE
  )
  
  
  # eigen() already returns eigenvalues in decreasing order.
  
  selected_vectors <- eig$vectors[
    ,
    seq_len(
      number_clusters
    ),
    drop = FALSE
  ]
  
  
  selected_values <- eig$values[
    seq_len(
      number_clusters
    )
  ]
  
  
  # --------------------------------------------------------------------------
  # Row normalization
  #
  # Ng-Jordan-Weiss normalized spectral clustering.
  # --------------------------------------------------------------------------
  
  row_norms <- sqrt(
    rowSums(
      selected_vectors^2
    )
  )
  
  
  spectral_embedding <- selected_vectors /
    pmax(
      row_norms,
      1e-12
    )
  
  
  # --------------------------------------------------------------------------
  # Cluster spectral coordinates
  # --------------------------------------------------------------------------
  
  set.seed(
    seed
  )
  
  
  clustering <- kmeans(
    spectral_embedding,
    centers =
      number_clusters,
    nstart =
      nstart
  )
  
  
  # --------------------------------------------------------------------------
  # L_sym eigenvalues for diagnostics
  # --------------------------------------------------------------------------
  
  laplacian_eigenvalues <- 1 -
    eig$values
  
  
  list(
    cluster =
      clustering$cluster,
    embedding =
      spectral_embedding,
    affinity =
      W,
    degrees =
      degrees,
    normalized_similarity =
      normalized_similarity,
    laplacian_eigenvalues =
      laplacian_eigenvalues,
    selected_similarity_eigenvalues =
      selected_values,
    kmeans =
      clustering,
    sigma =
      sigma,
    k_neighbors =
      k_neighbors
  )
}


# ==============================================================================
# PART XII
#
# FIT SPECTRAL CLUSTERING
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Fit
# ------------------------------------------------------------------------------

spectral_fit <- spectral_clustering(
  X,
  number_clusters = 2,
  sigma = 0.25,
  k_neighbors = 12,
  nstart = 50,
  seed = 123
)


# ------------------------------------------------------------------------------
# 23. Plot Clusters
# ------------------------------------------------------------------------------

plot(
  X,
  pch = spectral_fit$cluster,
  xlab = "X1",
  ylab = "X2",
  main = "Spectral Clustering"
)


# ==============================================================================
# PART XIII
#
# SPECTRAL EMBEDDING
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Plot First Two Spectral Coordinates
# ------------------------------------------------------------------------------

plot(
  spectral_fit$embedding[
    ,
    1
  ],
  spectral_fit$embedding[
    ,
    2
  ],
  pch = spectral_fit$cluster,
  xlab = "Spectral Coordinate 1",
  ylab = "Spectral Coordinate 2",
  main = "Spectral Embedding"
)


# In spectral space, the nonconvex moons become much easier to separate using
# ordinary K-means.


# ==============================================================================
# PART XIV
#
# CLUSTER EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Rand Index
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
    i in 1:(n - 1)
  ) {
    
    for (
      j in (i + 1):n
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


# ------------------------------------------------------------------------------
# 26. Compare with K-Means
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "K-Means Original Space",
    "Spectral Clustering"
  ),
  Rand_Index = c(
    rand_index(
      true_cluster,
      kmeans_original$cluster
    ),
    rand_index(
      true_cluster,
      spectral_fit$cluster
    )
  )
)


# ==============================================================================
# PART XV
#
# CONFUSION TABLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Spectral Clustering vs Truth
# ------------------------------------------------------------------------------

table(
  True =
    true_cluster,
  Spectral =
    spectral_fit$cluster
)


# Cluster labels themselves are arbitrary.


# ==============================================================================
# PART XVI
#
# EIGENGAP
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. First Laplacian Eigenvalues
# ------------------------------------------------------------------------------

laplacian_values_fit <- spectral_fit$laplacian_eigenvalues


plot(
  seq_len(
    min(
      20,
      length(
        laplacian_values_fit
      )
    )
  ),
  laplacian_values_fit[
    seq_len(
      min(
        20,
        length(
          laplacian_values_fit
        )
      )
    )
  ],
  type = "b",
  pch = 19,
  xlab = "Eigenvalue Index",
  ylab = "Laplacian Eigenvalue",
  main = "Spectral Clustering Eigengap"
)


# For K clusters, we often look for K small eigenvalues followed by a larger
# jump.


# ------------------------------------------------------------------------------
# 29. Eigaps
# ------------------------------------------------------------------------------

number_eigenvalues <- min(
  15,
  length(
    laplacian_values_fit
  )
)


eigengaps <- diff(
  laplacian_values_fit[
    seq_len(
      number_eigenvalues
    )
  ]
)


data.frame(
  Between = paste0(
    1:(number_eigenvalues - 1),
    " and ",
    2:number_eigenvalues
  ),
  Eigengap =
    eigengaps
)


# ==============================================================================
# PART XVII
#
# EFFECT OF SIGMA
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Sigma Grid
# ------------------------------------------------------------------------------

sigma_values <- c(
  0.05,
  0.10,
  0.20,
  0.30,
  0.50,
  1.0
)


sigma_results <- data.frame(
  Sigma =
    sigma_values,
  Rand_Index =
    NA_real_,
  Minimum_Degree =
    NA_real_,
  Median_Degree =
    NA_real_
)


sigma_models <- vector(
  "list",
  length(
    sigma_values
  )
)


for (
  i in seq_along(
    sigma_values
  )
) {
  
  fit_i <- spectral_clustering(
    X,
    number_clusters = 2,
    sigma =
      sigma_values[i],
    k_neighbors = 12,
    nstart = 30,
    seed = 123
  )
  
  
  sigma_models[[i]] <-
    fit_i
  
  
  sigma_results$Rand_Index[i] <-
    rand_index(
      true_cluster,
      fit_i$cluster
    )
  
  
  sigma_results$Minimum_Degree[i] <-
    min(
      fit_i$degrees
    )
  
  
  sigma_results$Median_Degree[i] <-
    median(
      fit_i$degrees
    )
}


sigma_results


# ------------------------------------------------------------------------------
# 31. Performance vs Sigma
# ------------------------------------------------------------------------------

plot(
  sigma_results$Sigma,
  sigma_results$Rand_Index,
  type = "b",
  pch = 19,
  xlab = "RBF Sigma",
  ylab = "Rand Index",
  main = "Spectral Clustering Sensitivity to Sigma"
)


# ==============================================================================
# PART XVIII
#
# EFFECT OF NUMBER OF NEIGHBORS
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. KNN Grid
# ------------------------------------------------------------------------------

neighbor_values <- c(
  4,
  6,
  8,
  12,
  20,
  40
)


neighbor_results <- data.frame(
  KNN =
    neighbor_values,
  Rand_Index =
    NA_real_,
  Zero_Eigenvalues =
    NA_integer_
)


for (
  i in seq_along(
    neighbor_values
  )
) {
  
  fit_i <- spectral_clustering(
    X,
    number_clusters = 2,
    sigma = 0.25,
    k_neighbors =
      neighbor_values[i],
    nstart = 30,
    seed = 123
  )
  
  
  neighbor_results$Rand_Index[i] <-
    rand_index(
      true_cluster,
      fit_i$cluster
    )
  
  
  neighbor_results$Zero_Eigenvalues[i] <-
    sum(
      abs(
        fit_i$laplacian_eigenvalues
      ) <
        1e-8
    )
}


neighbor_results


# ==============================================================================
# PART XIX
#
# GRAPH TOO SPARSE
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Very Small Neighborhood
# ------------------------------------------------------------------------------

spectral_sparse <- spectral_clustering(
  X,
  number_clusters = 2,
  sigma = 0.25,
  k_neighbors = 2,
  seed = 123
)


sum(
  abs(
    spectral_sparse$laplacian_eigenvalues
  ) <
    1e-8
)


# Too-small K can break the graph into many disconnected pieces.


# ==============================================================================
# PART XX
#
# GRAPH TOO DENSE
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Fully Connected RBF Graph
# ------------------------------------------------------------------------------

spectral_dense <- spectral_clustering(
  X,
  number_clusters = 2,
  sigma = 0.25,
  k_neighbors = NULL,
  seed = 123
)


rand_index(
  true_cluster,
  spectral_dense$cluster
)


# A dense graph can blur local manifold structure if sigma is too broad.


# ==============================================================================
# PART XXI
#
# SELF-TUNING LOCAL SCALE
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Local Scaling Affinity
#
# Zelnik-Manor-style intuition:
#
#       W_ij = exp(
#           -||x_i-x_j||^2 /
#           (sigma_i sigma_j)
#       )
#
# where sigma_i is distance to the local-scale neighbor.
# ------------------------------------------------------------------------------

local_scale_affinity <- function(
    X,
    local_k = 7,
    graph_k = NULL
) {
  
  D_squared <- squared_distance_matrix(
    X
  )
  
  
  D <- sqrt(
    D_squared
  )
  
  
  n <- nrow(
    X
  )
  
  
  local_scale <- numeric(
    n
  )
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    distances_i <- sort(
      D[
        i,
        -i
      ]
    )
    
    
    local_scale[i] <-
      distances_i[
        min(
          local_k,
          length(
            distances_i
          )
        )
      ]
  }
  
  
  scale_matrix <- outer(
    local_scale,
    local_scale,
    "*"
  )
  
  
  W <- exp(
    -D_squared /
      pmax(
        scale_matrix,
        1e-12
      )
  )
  
  
  diag(
    W
  ) <- 0
  
  
  if (
    !is.null(
      graph_k
    )
  ) {
    
    mask <- knn_mask(
      D,
      graph_k
    )
    
    
    W[
      !mask
    ] <- 0
  }
  
  
  list(
    affinity =
      W,
    local_scale =
      local_scale
  )
}


# ==============================================================================
# PART XXII
#
# SPECTRAL CLUSTERING FROM PRECOMPUTED AFFINITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. General Affinity-Based Spectral Clustering
# ------------------------------------------------------------------------------

spectral_from_affinity <- function(
    W,
    number_clusters,
    nstart = 50,
    seed = 123
) {
  
  W <- as.matrix(
    W
  )
  
  
  n <- nrow(
    W
  )
  
  
  W <- (
    W +
      t(W)
  ) /
    2
  
  
  diag(
    W
  ) <- 0
  
  
  degrees <- rowSums(
    W
  )
  
  
  D_inverse_sqrt <- diag(
    1 /
      sqrt(
        pmax(
          degrees,
          1e-12
        )
      )
  )
  
  
  normalized_similarity <- D_inverse_sqrt %*%
    W %*%
    D_inverse_sqrt
  
  
  eig <- eigen(
    normalized_similarity,
    symmetric = TRUE
  )
  
  
  U <- eig$vectors[
    ,
    seq_len(
      number_clusters
    ),
    drop = FALSE
  ]
  
  
  row_norms <- sqrt(
    rowSums(
      U^2
    )
  )
  
  
  U_normalized <- U /
    pmax(
      row_norms,
      1e-12
    )
  
  
  set.seed(
    seed
  )
  
  
  clustering <- kmeans(
    U_normalized,
    centers =
      number_clusters,
    nstart =
      nstart
  )
  
  
  list(
    cluster =
      clustering$cluster,
    embedding =
      U_normalized,
    degrees =
      degrees,
    laplacian_eigenvalues =
      1 -
      eig$values
  )
}


# ------------------------------------------------------------------------------
# 37. Self-Tuning Spectral Clustering
# ------------------------------------------------------------------------------

local_affinity <- local_scale_affinity(
  X,
  local_k = 7,
  graph_k = 12
)


spectral_self_tuning <- spectral_from_affinity(
  local_affinity$affinity,
  number_clusters = 2,
  seed = 123
)


rand_index(
  true_cluster,
  spectral_self_tuning$cluster
)


# ==============================================================================
# PART XXIII
#
# THREE NONCONVEX CLUSTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Concentric Rings
# ------------------------------------------------------------------------------

set.seed(777)


n_ring <- 150


theta_1 <- runif(
  n_ring,
  0,
  2 * pi
)


theta_2 <- runif(
  n_ring,
  0,
  2 * pi
)


theta_3 <- runif(
  n_ring,
  0,
  2 * pi
)


ring_1 <- cbind(
  cos(theta_1),
  sin(theta_1)
)


ring_2 <- 2 *
  cbind(
    cos(theta_2),
    sin(theta_2)
  )


ring_3 <- 3 *
  cbind(
    cos(theta_3),
    sin(theta_3)
  )


X_rings <- rbind(
  ring_1,
  ring_2,
  ring_3
)


X_rings <- X_rings +
  matrix(
    rnorm(
      length(
        X_rings
      ),
      sd = 0.08
    ),
    nrow =
      nrow(
        X_rings
      )
  )


true_rings <- rep(
  1:3,
  each =
    n_ring
)


# ------------------------------------------------------------------------------
# 39. Spectral Fit
# ------------------------------------------------------------------------------

spectral_rings <- spectral_clustering(
  X_rings,
  number_clusters = 3,
  sigma = 0.20,
  k_neighbors = 10,
  seed = 123
)


# ------------------------------------------------------------------------------
# 40. Plot
# ------------------------------------------------------------------------------

plot(
  X_rings,
  pch =
    spectral_rings$cluster,
  xlab = "X1",
  ylab = "X2",
  main = "Spectral Clustering: Concentric Rings"
)


rand_index(
  true_rings,
  spectral_rings$cluster
)


# ==============================================================================
# PART XXIV
#
# K-MEANS ON RINGS
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Baseline
# ------------------------------------------------------------------------------

set.seed(123)


kmeans_rings <- kmeans(
  X_rings,
  centers = 3,
  nstart = 50
)


rand_index(
  true_rings,
  kmeans_rings$cluster
)


# Spectral clustering can capture nonconvex connected sets that K-means cannot.


# ==============================================================================
# PART XXV
#
# GRAPH CUT INTERPRETATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Cut Weight
#
# Sum of affinities connecting two cluster sets.
# ------------------------------------------------------------------------------

cut_weight <- function(
    W,
    cluster_labels
) {
  
  n <- length(
    cluster_labels
  )
  
  
  cut <- 0
  
  
  for (
    i in 1:(n - 1)
  ) {
    
    for (
      j in (i + 1):n
    ) {
      
      if (
        cluster_labels[i] !=
        cluster_labels[j]
      ) {
        
        cut <- cut +
          W[
            i,
            j
          ]
      }
    }
  }
  
  
  cut
}


# ------------------------------------------------------------------------------
# 43. Compare Cut Values
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "K-Means",
    "Spectral"
  ),
  Graph_Cut = c(
    cut_weight(
      spectral_fit$affinity,
      kmeans_original$cluster
    ),
    cut_weight(
      spectral_fit$affinity,
      spectral_fit$cluster
    )
  )
)


# A good graph clustering tends to avoid cutting high-affinity edges.


# ==============================================================================
# PART XXVI
#
# NORMALIZED CUT
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Two-Cluster Normalized Cut
# ------------------------------------------------------------------------------

normalized_cut_two_clusters <- function(
    W,
    labels
) {
  
  unique_labels <- unique(
    labels
  )
  
  
  if (
    length(
      unique_labels
    ) !=
    2
  ) {
    
    stop(
      "This helper expects exactly two clusters."
    )
  }
  
  
  A <- which(
    labels ==
      unique_labels[1]
  )
  
  
  B <- which(
    labels ==
      unique_labels[2]
  )
  
  
  cut_AB <- sum(
    W[
      A,
      B,
      drop = FALSE
    ]
  )
  
  
  degree <- rowSums(
    W
  )
  
  
  volume_A <- sum(
    degree[
      A
    ]
  )
  
  
  volume_B <- sum(
    degree[
      B
    ]
  )
  
  
  cut_AB /
    volume_A +
    cut_AB /
    volume_B
}


# ------------------------------------------------------------------------------
# 45. Compare
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "K-Means",
    "Spectral"
  ),
  Normalized_Cut = c(
    normalized_cut_two_clusters(
      spectral_fit$affinity,
      kmeans_original$cluster
    ),
    normalized_cut_two_clusters(
      spectral_fit$affinity,
      spectral_fit$cluster
    )
  )
)


# ==============================================================================
# PART XXVII
#
# WHY NORMALIZATION MATTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Unequal-Density Clusters
# ------------------------------------------------------------------------------

set.seed(888)


dense_cluster <- cbind(
  rnorm(
    200,
    mean = -2,
    sd = 0.25
  ),
  rnorm(
    200,
    mean = 0,
    sd = 0.25
  )
)


diffuse_cluster <- cbind(
  rnorm(
    200,
    mean = 2,
    sd = 0.80
  ),
  rnorm(
    200,
    mean = 0,
    sd = 0.80
  )
)


X_density <- rbind(
  dense_cluster,
  diffuse_cluster
)


true_density <- rep(
  1:2,
  each = 200
)


# ------------------------------------------------------------------------------
# 47. Normalized Spectral Clustering
# ------------------------------------------------------------------------------

spectral_density <- spectral_clustering(
  X_density,
  number_clusters = 2,
  sigma = 0.50,
  k_neighbors = 15,
  seed = 123
)


rand_index(
  true_density,
  spectral_density$cluster
)


# Degree normalization prevents high-degree regions from dominating as strongly
# as they would in an unnormalized graph formulation.


# ==============================================================================
# PART XXVIII
#
# CLUSTER NUMBER SELECTION VIA EIGENGAP
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Helper
# ------------------------------------------------------------------------------

estimate_cluster_number <- function(
    laplacian_eigenvalues,
    maximum_clusters = 10
) {
  
  maximum_clusters <- min(
    maximum_clusters,
    length(
      laplacian_eigenvalues
    ) -
      1
  )
  
  
  values <- laplacian_eigenvalues[
    seq_len(
      maximum_clusters +
        1
    )
  ]
  
  
  gaps <- diff(
    values
  )
  
  
  which.max(
    gaps
  )
}


# ------------------------------------------------------------------------------
# 49. Estimate
# ------------------------------------------------------------------------------

estimated_K <- estimate_cluster_number(
  spectral_fit$laplacian_eigenvalues,
  maximum_clusters = 8
)


estimated_K


# Eigengap is a heuristic, not a guaranteed cluster-count estimator.


# ==============================================================================
# PART XXIX
#
# CLUSTER STABILITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Different K-Means Starts in Spectral Space
# ------------------------------------------------------------------------------

spectral_seed_results <- data.frame(
  Seed =
    1:20,
  Rand_Index =
    NA_real_
)


for (
  seed_i in 1:20
) {
  
  fit_i <- spectral_clustering(
    X,
    number_clusters = 2,
    sigma = 0.25,
    k_neighbors = 12,
    nstart = 1,
    seed =
      seed_i
  )
  
  
  spectral_seed_results$Rand_Index[
    seed_i
  ] <-
    rand_index(
      true_cluster,
      fit_i$cluster
    )
}


spectral_seed_results


# Spectral embedding can be excellent while the final K-means stage still has
# its own local-optimum issue.


# ==============================================================================
# PART XXX
#
# FEATURE SCALING
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Add Large-Scale Irrelevant Feature
# ------------------------------------------------------------------------------

set.seed(999)


X_scale_problem <- cbind(
  X,
  Noise =
    rnorm(
      n,
      sd = 100
    )
)


# ------------------------------------------------------------------------------
# 52. Spectral Clustering Without Scaling
# ------------------------------------------------------------------------------

spectral_bad_scale <- spectral_clustering(
  X_scale_problem,
  number_clusters = 2,
  sigma = 10,
  k_neighbors = 12,
  seed = 123
)


rand_index(
  true_cluster,
  spectral_bad_scale$cluster
)


# ------------------------------------------------------------------------------
# 53. Standardize Features
# ------------------------------------------------------------------------------

X_scale_fixed <- scale(
  X_scale_problem
)


spectral_scale_fixed <- spectral_clustering(
  X_scale_fixed,
  number_clusters = 2,
  sigma = 0.8,
  k_neighbors = 12,
  seed = 123
)


rand_index(
  true_cluster,
  spectral_scale_fixed$cluster
)


# ==============================================================================
# PART XXXI
#
# HIGH-DIMENSIONAL EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Embed Moons into Higher Dimension
# ------------------------------------------------------------------------------

set.seed(111)


random_projection_matrix <- matrix(
  rnorm(
    2 *
      10
  ),
  nrow = 2,
  ncol = 10
)


X_high <- X %*%
  random_projection_matrix


X_high <- X_high +
  matrix(
    rnorm(
      n *
        10,
      sd = 0.05
    ),
    nrow = n
  )


X_high <- scale(
  X_high
)


# ------------------------------------------------------------------------------
# 55. Spectral Clustering
# ------------------------------------------------------------------------------

spectral_high <- spectral_clustering(
  X_high,
  number_clusters = 2,
  sigma = 1.2,
  k_neighbors = 12,
  seed = 123
)


rand_index(
  true_cluster,
  spectral_high$cluster
)


# ==============================================================================
# PART XXXII
#
# OPTIONAL PACKAGE VERIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. kernlab::specc
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "kernlab",
    quietly = TRUE
  )
) {
  
  package_fit <- kernlab::specc(
    X,
    centers = 2
  )
  
  
  package_cluster <- as.integer(
    package_fit
  )
  
  
  cat(
    "\nRand index against kernlab::specc result:\n"
  )
  
  
  print(
    rand_index(
      true_cluster,
      package_cluster
    )
  )
  
  
  # The package may use different automatic kernel and scaling conventions, so
  # exact cluster assignments need not match the manual implementation.
  
}


# ==============================================================================
# PART XXXIII
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nSpectral Clustering Summary\n"
)


cat(
  "---------------------------\n"
)


cat(
  "Observations:",
  n,
  "\n"
)


cat(
  "Requested clusters:",
  2,
  "\n"
)


cat(
  "RBF sigma:",
  spectral_fit$sigma,
  "\n"
)


cat(
  "KNN graph size:",
  spectral_fit$k_neighbors,
  "\n"
)


cat(
  "Minimum graph degree:",
  round(
    min(
      spectral_fit$degrees
    ),
    6
  ),
  "\n"
)


cat(
  "Near-zero Laplacian eigenvalues:",
  sum(
    abs(
      spectral_fit$laplacian_eigenvalues
    ) <
      1e-8
  ),
  "\n"
)


cat(
  "K-means original-space Rand index:",
  round(
    rand_index(
      true_cluster,
      kmeans_original$cluster
    ),
    4
  ),
  "\n"
)


cat(
  "Spectral clustering Rand index:",
  round(
    rand_index(
      true_cluster,
      spectral_fit$cluster
    ),
    4
  ),
  "\n"
)


cat(
  "Estimated cluster count from eigengap:",
  estimated_K,
  "\n"
)