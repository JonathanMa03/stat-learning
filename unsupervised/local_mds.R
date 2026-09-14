# ==============================================================================
# Local Multidimensional Scaling
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   Chen and Buja (2009), Local Multidimensional Scaling for Nonlinear
#   Dimension Reduction, Graph Drawing, and Proximity Analysis
#
# Main ideas:
#   - Multidimensional Scaling
#   - Local neighborhoods
#   - K-nearest-neighbor graph
#   - Local distance preservation
#   - Repulsion between non-neighbors
#   - Nonlinear dimensionality reduction
#   - Local stress
#   - Gradient-based optimization
#   - Neighborhood preservation
#   - Trustworthiness intuition
#   - Comparison with classical/global MDS
#
#
# Classical / metric MDS:
#
#       minimize
#
#       sum_{i<j} (Delta_ij - d_ij)^2
#
#
# Local MDS:
#
#       minimize
#
#       sum_{(i,j) in N_K}
#           (Delta_ij - d_ij)^2
#
#       -
#
#       tau *
#       sum_{(i,j) not in N_K}
#           d_ij
#
#
# where:
#
#       Delta_ij = original-space distance
#       d_ij     = embedding-space distance
#       N_K      = symmetric K-nearest-neighbor graph
#       tau      = nonlocal repulsion strength
#
#
# The local term preserves nearby relationships.
#
# The repulsion term discourages unrelated observations from collapsing
# together in the low-dimensional representation.
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE A NONLINEAR MANIFOLD
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate an S-Curve in Three Dimensions
# ------------------------------------------------------------------------------

set.seed(123)


n <- 140


latent_t <- runif(
  n,
  min = -1.5 * pi,
  max = 1.5 * pi
)


height <- runif(
  n,
  min = -2,
  max = 2
)


# Nonlinear 2D manifold embedded in 3D.

X <- cbind(
  X1 =
    sin(
      latent_t
    ),
  X2 =
    height,
  X3 =
    sign(
      latent_t
    ) *
    (
      cos(
        latent_t
      ) -
        1
    )
)


# Add a little measurement noise.

X <- X +
  matrix(
    rnorm(
      n * 3,
      sd = 0.03
    ),
    nrow = n,
    ncol = 3
  )


# ------------------------------------------------------------------------------
# 2. Pairwise Views of the 3D Data
# ------------------------------------------------------------------------------

pairs(
  X,
  pch = 19,
  cex = 0.5,
  main = "Nonlinear S-Curve Data"
)


# ==============================================================================
# PART II
#
# MANUAL PAIRWISE DISTANCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Euclidean Distance
# ------------------------------------------------------------------------------

euclidean_distance <- function(
    x,
    y
) {
  
  sqrt(
    sum(
      (
        x -
          y
      )^2
    )
  )
}


# ------------------------------------------------------------------------------
# 4. Pairwise Euclidean Distance Matrix
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
      
      distance_ij <- sqrt(
        sum(
          (
            X[
              i,
            ] -
              X[
                j,
              ]
          )^2
        )
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
# 5. Original Distance Matrix
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


# ==============================================================================
# PART III
#
# K-NEAREST-NEIGHBOR GRAPH
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. KNN Indices from Distance Matrix
# ------------------------------------------------------------------------------

knn_indices <- function(
    D,
    k
) {
  
  D <- as.matrix(
    D
  )
  
  
  n <- nrow(
    D
  )
  
  
  neighbors <- matrix(
    NA_integer_,
    nrow = n,
    ncol = k
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
    
    
    neighbors[
      i,
    ] <- order(
      distances_i
    )[
      seq_len(
        k
      )
    ]
  }
  
  
  neighbors
}


# ------------------------------------------------------------------------------
# 7. Symmetric KNN Adjacency Matrix
#
# Pair i,j is local if either:
#
#   j is among i's K nearest neighbors
#
# OR
#
#   i is among j's K nearest neighbors
# ------------------------------------------------------------------------------

symmetric_knn_graph <- function(
    D,
    k
) {
  
  D <- as.matrix(
    D
  )
  
  
  n <- nrow(
    D
  )
  
  
  neighbors <- knn_indices(
    D,
    k
  )
  
  
  adjacency <- matrix(
    FALSE,
    nrow = n,
    ncol = n
  )
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    adjacency[
      i,
      neighbors[
        i,
      ]
    ] <- TRUE
  }
  
  
  adjacency <- adjacency |
    t(
      adjacency
    )
  
  
  diag(
    adjacency
  ) <- FALSE
  
  
  adjacency
}


# ------------------------------------------------------------------------------
# 8. Choose Neighborhood Size
# ------------------------------------------------------------------------------

k <- 8


neighbor_graph <- symmetric_knn_graph(
  D,
  k =
    k
)


sum(
  neighbor_graph
) /
  2


# ==============================================================================
# PART IV
#
# VISUALIZE THE LOCAL GRAPH
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Use Latent Coordinates Only for Simulation Visualization
#
# These are NOT supplied to Local MDS.
# ------------------------------------------------------------------------------

plot(
  latent_t,
  height,
  pch = 19,
  cex = 0.6,
  xlab = "True Latent Coordinate",
  ylab = "Height",
  main = "True Manifold Coordinates with KNN Graph"
)


for (
  i in 1:(n - 1)
) {
  
  for (
    j in (i + 1):n
  ) {
    
    if (
      neighbor_graph[
        i,
        j
      ]
    ) {
      
      segments(
        latent_t[i],
        height[i],
        latent_t[j],
        height[j]
      )
    }
  }
}


points(
  latent_t,
  height,
  pch = 19,
  cex = 0.5
)


# ==============================================================================
# PART V
#
# CLASSICAL MDS INITIALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Classical MDS
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
  
  
  J <- diag(
    n
  ) -
    matrix(
      1 / n,
      nrow = n,
      ncol = n
    )
  
  
  B <- -0.5 *
    J %*%
    (
      D^2
    ) %*%
    J
  
  
  B <- (
    B +
      t(B)
  ) /
    2
  
  
  eig <- eigen(
    B,
    symmetric = TRUE
  )
  
  
  positive <- which(
    eig$values >
      tolerance
  )
  
  
  dimensions_used <- min(
    dimensions,
    length(
      positive
    )
  )
  
  
  selected <- positive[
    seq_len(
      dimensions_used
    )
  ]
  
  
  coordinates <- eig$vectors[
    ,
    selected,
    drop = FALSE
  ] %*%
    diag(
      sqrt(
        eig$values[
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
  
  
  list(
    coordinates =
      coordinates,
    eigenvalues =
      eig$values,
    gram =
      B
  )
}


# ------------------------------------------------------------------------------
# 11. Initial Embedding
# ------------------------------------------------------------------------------

classical_fit <- classical_mds(
  D,
  dimensions = 2
)


Z_initial <- classical_fit$coordinates


plot(
  Z_initial,
  pch = 19,
  cex = 0.6,
  xlab = "MDS1",
  ylab = "MDS2",
  main = "Classical MDS Initialization"
)


# ==============================================================================
# PART VI
#
# LOCAL MDS OBJECTIVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Convert Coordinates Between Matrix and Vector
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
# 13. Local MDS Stress
#
#
#       L(Z)
#
#       =
#
#       sum_local (Delta_ij - d_ij)^2
#
#       -
#
#       tau * sum_nonlocal d_ij
#
#
# The first term is an attraction / local fidelity term.
#
# The second term is a repulsion term.
# ------------------------------------------------------------------------------

local_mds_objective <- function(
    parameters,
    D,
    neighbor_graph,
    dimensions,
    tau
) {
  
  n <- nrow(
    D
  )
  
  
  Z <- vector_to_coordinates(
    parameters,
    n =
      n,
    dimensions =
      dimensions
  )
  
  
  # Translation does not affect distances.
  # Keep the configuration centered for numerical stability.
  
  Z <- sweep(
    Z,
    2,
    colMeans(
      Z
    ),
    "-"
  )
  
  
  D_embedding <- pairwise_euclidean(
    Z
  )
  
  
  upper <- upper.tri(
    D
  )
  
  
  local_pairs <- upper &
    neighbor_graph
  
  
  nonlocal_pairs <- upper &
    !neighbor_graph
  
  
  local_stress <- sum(
    (
      D[
        local_pairs
      ] -
        D_embedding[
          local_pairs
        ]
    )^2
  )
  
  
  repulsion <- sum(
    D_embedding[
      nonlocal_pairs
    ]
  )
  
  
  local_stress -
    tau *
    repulsion
}


# ==============================================================================
# PART VII
#
# ANALYTIC LOCAL MDS GRADIENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Gradient
#
# For a local pair:
#
#       (Delta - d)^2
#
# gradient with respect to z_i:
#
#       2 (d - Delta) (z_i - z_j) / d
#
#
# For a nonlocal pair:
#
#       -tau d
#
# gradient:
#
#       -tau (z_i - z_j) / d
# ------------------------------------------------------------------------------

local_mds_gradient <- function(
    parameters,
    D,
    neighbor_graph,
    dimensions,
    tau,
    distance_floor = 1e-10
) {
  
  n <- nrow(
    D
  )
  
  
  Z <- vector_to_coordinates(
    parameters,
    n =
      n,
    dimensions =
      dimensions
  )
  
  
  Z <- sweep(
    Z,
    2,
    colMeans(
      Z
    ),
    "-"
  )
  
  
  gradient <- matrix(
    0,
    nrow = n,
    ncol = dimensions
  )
  
  
  if (
    n <= 1
  ) {
    
    return(
      as.numeric(
        gradient
      )
    )
  }
  
  
  for (
    i in 1:(n - 1)
  ) {
    
    for (
      j in (i + 1):n
    ) {
      
      difference <- Z[
        i,
      ] -
        Z[
          j,
        ]
      
      
      distance_ij <- sqrt(
        sum(
          difference^2
        )
      )
      
      
      safe_distance <- max(
        distance_ij,
        distance_floor
      )
      
      
      unit_direction <- difference /
        safe_distance
      
      
      if (
        neighbor_graph[
          i,
          j
        ]
      ) {
        
        coefficient <- 2 *
          (
            distance_ij -
              D[
                i,
                j
              ]
          )
        
        
      } else {
        
        coefficient <- -tau
      }
      
      
      contribution <- coefficient *
        unit_direction
      
      
      gradient[
        i,
      ] <- gradient[
        i,
      ] +
        contribution
      
      
      gradient[
        j,
      ] <- gradient[
        j,
      ] -
        contribution
    }
  }
  
  
  # The objective is translation invariant.
  # Remove any small numerical common gradient component.
  
  gradient <- sweep(
    gradient,
    2,
    colMeans(
      gradient
    ),
    "-"
  )
  
  
  as.numeric(
    gradient
  )
}


# ==============================================================================
# PART VIII
#
# CHECK THE GRADIENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Numerical Gradient for a Few Coordinates
# ------------------------------------------------------------------------------

finite_difference_gradient <- function(
    parameters,
    objective_function,
    indices,
    epsilon = 1e-6,
    ...
) {
  
  result <- numeric(
    length(
      indices
    )
  )
  
  
  for (
    m in seq_along(
      indices
    )
  ) {
    
    j <- indices[m]
    
    
    plus <- parameters
    
    minus <- parameters
    
    
    plus[j] <- plus[j] +
      epsilon
    
    
    minus[j] <- minus[j] -
      epsilon
    
    
    result[m] <- (
      objective_function(
        plus,
        ...
      ) -
        objective_function(
          minus,
          ...
        )
    ) /
      (
        2 *
          epsilon
      )
  }
  
  
  result
}


initial_parameters <- coordinates_to_vector(
  Z_initial
)


set.seed(321)


gradient_check_indices <- sample(
  seq_along(
    initial_parameters
  ),
  10
)


analytic_gradient_check <- local_mds_gradient(
  initial_parameters,
  D =
    D,
  neighbor_graph =
    neighbor_graph,
  dimensions = 2,
  tau = 0.01
)[
  gradient_check_indices
]


numerical_gradient_check <- finite_difference_gradient(
  initial_parameters,
  objective_function =
    local_mds_objective,
  indices =
    gradient_check_indices,
  D =
    D,
  neighbor_graph =
    neighbor_graph,
  dimensions = 2,
  tau = 0.01
)


data.frame(
  Coordinate =
    gradient_check_indices,
  Analytic =
    analytic_gradient_check,
  Numerical =
    numerical_gradient_check,
  Difference =
    analytic_gradient_check -
    numerical_gradient_check
)


# ==============================================================================
# PART IX
#
# FIT LOCAL MDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Fitting Function
# ------------------------------------------------------------------------------

fit_local_mds <- function(
    D,
    k = 8,
    tau = 0.01,
    dimensions = 2,
    initial_coordinates = NULL,
    max_iterations = 1000,
    tolerance = 1e-9
) {
  
  D <- as.matrix(
    D
  )
  
  
  n <- nrow(
    D
  )
  
  
  neighbor_graph <- symmetric_knn_graph(
    D,
    k =
      k
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
    
    
    initial_coordinates <-
      initial_fit$coordinates
  }
  
  
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
      local_mds_objective,
    gr =
      local_mds_gradient,
    D =
      D,
    neighbor_graph =
      neighbor_graph,
    dimensions =
      dimensions,
    tau =
      tau,
    method = "BFGS",
    control = list(
      maxit =
        max_iterations,
      reltol =
        tolerance
    )
  )
  
  
  coordinates <- vector_to_coordinates(
    result$par,
    n =
      n,
    dimensions =
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
    D_embedding =
      D_embedding,
    neighbor_graph =
      neighbor_graph,
    k =
      k,
    tau =
      tau,
    objective =
      result$value,
    convergence =
      result$convergence,
    iterations =
      result$counts,
    optim_result =
      result
  )
}


# ------------------------------------------------------------------------------
# 17. Fit
# ------------------------------------------------------------------------------

local_fit <- fit_local_mds(
  D,
  k = 8,
  tau = 0.01,
  dimensions = 2,
  initial_coordinates =
    Z_initial,
  max_iterations = 1000
)


Z_local <- local_fit$coordinates


local_fit$convergence


local_fit$objective


# ==============================================================================
# PART X
#
# VISUALIZE LOCAL MDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Local MDS Embedding
# ------------------------------------------------------------------------------

plot(
  Z_local[
    ,
    1
  ],
  Z_local[
    ,
    2
  ],
  pch = 19,
  cex = 0.6,
  xlab = "Local MDS Dimension 1",
  ylab = "Local MDS Dimension 2",
  main = "Local Multidimensional Scaling"
)


# ------------------------------------------------------------------------------
# 19. Draw Neighbor Graph
# ------------------------------------------------------------------------------

plot(
  Z_local[
    ,
    1
  ],
  Z_local[
    ,
    2
  ],
  type = "n",
  xlab = "Local MDS Dimension 1",
  ylab = "Local MDS Dimension 2",
  main = "Local MDS with Original KNN Graph"
)


for (
  i in 1:(n - 1)
) {
  
  for (
    j in (i + 1):n
  ) {
    
    if (
      neighbor_graph[
        i,
        j
      ]
    ) {
      
      segments(
        Z_local[
          i,
          1
        ],
        Z_local[
          i,
          2
        ],
        Z_local[
          j,
          1
        ],
        Z_local[
          j,
          2
        ]
      )
    }
  }
}


points(
  Z_local,
  pch = 19,
  cex = 0.5
)


# ==============================================================================
# PART XI
#
# GLOBAL METRIC MDS COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Raw Metric MDS Stress
# ------------------------------------------------------------------------------

metric_mds_objective <- function(
    parameters,
    D,
    dimensions
) {
  
  n <- nrow(
    D
  )
  
  
  Z <- vector_to_coordinates(
    parameters,
    n,
    dimensions
  )
  
  
  Z <- sweep(
    Z,
    2,
    colMeans(
      Z
    ),
    "-"
  )
  
  
  D_embedding <- pairwise_euclidean(
    Z
  )
  
  
  upper <- upper.tri(
    D
  )
  
  
  sum(
    (
      D[
        upper
      ] -
        D_embedding[
          upper
        ]
    )^2
  )
}


# ------------------------------------------------------------------------------
# 21. Fit Global Metric MDS
# ------------------------------------------------------------------------------

global_metric_result <- optim(
  par =
    coordinates_to_vector(
      Z_initial
    ),
  fn =
    metric_mds_objective,
  D =
    D,
  dimensions = 2,
  method = "BFGS",
  control = list(
    maxit = 1000
  )
)


Z_global <- vector_to_coordinates(
  global_metric_result$par,
  n,
  2
)


Z_global <- sweep(
  Z_global,
  2,
  colMeans(
    Z_global
  ),
  "-"
)


# ------------------------------------------------------------------------------
# 22. Plot
# ------------------------------------------------------------------------------

plot(
  Z_global,
  pch = 19,
  cex = 0.6,
  xlab = "Global MDS Dimension 1",
  ylab = "Global MDS Dimension 2",
  main = "Global Metric MDS"
)


# ==============================================================================
# PART XII
#
# GLOBAL DISTANCE STRESS
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Normalized Stress
# ------------------------------------------------------------------------------

normalized_stress <- function(
    D_original,
    D_embedding
) {
  
  upper <- upper.tri(
    D_original
  )
  
  
  numerator <- sum(
    (
      D_original[
        upper
      ] -
        D_embedding[
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


D_global <- pairwise_euclidean(
  Z_global
)


D_local_embedding <- pairwise_euclidean(
  Z_local
)


data.frame(
  Method = c(
    "Global Metric MDS",
    "Local MDS"
  ),
  Global_Stress = c(
    normalized_stress(
      D,
      D_global
    ),
    normalized_stress(
      D,
      D_local_embedding
    )
  )
)


# Global MDS should usually win according to GLOBAL distance stress because that
# is exactly the criterion it optimizes.
#
# Local MDS deliberately sacrifices global distance fidelity to better preserve
# neighborhoods.


# ==============================================================================
# PART XIII
#
# LOCAL DISTANCE ERROR
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Local Stress Only
# ------------------------------------------------------------------------------

local_distance_rmse <- function(
    D_original,
    D_embedding,
    neighbor_graph
) {
  
  local_pairs <- upper.tri(
    D_original
  ) &
    neighbor_graph
  
  
  sqrt(
    mean(
      (
        D_original[
          local_pairs
        ] -
          D_embedding[
            local_pairs
          ]
      )^2
    )
  )
}


# ------------------------------------------------------------------------------
# 25. Compare
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Classical MDS",
    "Global Metric MDS",
    "Local MDS"
  ),
  Local_RMSE = c(
    local_distance_rmse(
      D,
      pairwise_euclidean(
        Z_initial
      ),
      neighbor_graph
    ),
    local_distance_rmse(
      D,
      D_global,
      neighbor_graph
    ),
    local_distance_rmse(
      D,
      D_local_embedding,
      neighbor_graph
    )
  )
)


# ==============================================================================
# PART XIV
#
# KNN NEIGHBORHOOD PRESERVATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Mean Neighborhood Overlap
#
# For every observation:
#
#       overlap_i =
#
#       |N_original(i) intersect N_embedding(i)| / K
#
# Then average over observations.
# ------------------------------------------------------------------------------

knn_overlap <- function(
    D_original,
    D_embedding,
    k
) {
  
  original_neighbors <- knn_indices(
    D_original,
    k
  )
  
  
  embedding_neighbors <- knn_indices(
    D_embedding,
    k
  )
  
  
  overlap <- numeric(
    nrow(
      D_original
    )
  )
  
  
  for (
    i in seq_len(
      nrow(
        D_original
      )
    )
  ) {
    
    overlap[i] <- length(
      intersect(
        original_neighbors[
          i,
        ],
        embedding_neighbors[
          i,
        ]
      )
    ) /
      k
  }
  
  
  list(
    average =
      mean(
        overlap
      ),
    pointwise =
      overlap
  )
}


# ------------------------------------------------------------------------------
# 27. Compare Neighborhood Preservation
# ------------------------------------------------------------------------------

overlap_classical <- knn_overlap(
  D,
  pairwise_euclidean(
    Z_initial
  ),
  k
)


overlap_global <- knn_overlap(
  D,
  D_global,
  k
)


overlap_local <- knn_overlap(
  D,
  D_local_embedding,
  k
)


data.frame(
  Method = c(
    "Classical MDS",
    "Global Metric MDS",
    "Local MDS"
  ),
  Mean_KNN_Overlap = c(
    overlap_classical$average,
    overlap_global$average,
    overlap_local$average
  )
)


# ==============================================================================
# PART XV
#
# TRUSTWORTHINESS
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Rank Matrix
#
# rank_matrix[i,j] =
#
# rank of observation j by distance from observation i.
# ------------------------------------------------------------------------------

distance_rank_matrix <- function(
    D
) {
  
  n <- nrow(
    D
  )
  
  
  ranks <- matrix(
    NA_integer_,
    nrow = n,
    ncol = n
  )
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    ordering <- order(
      D[
        i,
      ]
    )
    
    
    ordering <- ordering[
      ordering !=
        i
    ]
    
    
    ranks[
      i,
      ordering
    ] <- seq_along(
      ordering
    )
    
    
    ranks[
      i,
      i
    ] <- 0
  }
  
  
  ranks
}


# ------------------------------------------------------------------------------
# 29. Trustworthiness
#
# Penalizes observations that become neighbors in the embedding even though
# they were not neighbors in the original data.
# ------------------------------------------------------------------------------

trustworthiness <- function(
    D_original,
    D_embedding,
    k
) {
  
  n <- nrow(
    D_original
  )
  
  
  original_neighbors <- knn_indices(
    D_original,
    k
  )
  
  
  embedded_neighbors <- knn_indices(
    D_embedding,
    k
  )
  
  
  original_ranks <- distance_rank_matrix(
    D_original
  )
  
  
  penalty <- 0
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    intrusions <- setdiff(
      embedded_neighbors[
        i,
      ],
      original_neighbors[
        i,
      ]
    )
    
    
    if (
      length(
        intrusions
      ) >
      0
    ) {
      
      penalty <- penalty +
        sum(
          original_ranks[
            i,
            intrusions
          ] -
            k
        )
    }
  }
  
  
  denominator <- n *
    k *
    (
      2 * n -
        3 * k -
        1
    )
  
  
  1 -
    (
      2 /
        denominator
    ) *
    penalty
}


# ==============================================================================
# PART XVI
#
# CONTINUITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Continuity
#
# Penalizes original neighbors that are pushed too far away in the embedding.
# ------------------------------------------------------------------------------

continuity <- function(
    D_original,
    D_embedding,
    k
) {
  
  n <- nrow(
    D_original
  )
  
  
  original_neighbors <- knn_indices(
    D_original,
    k
  )
  
  
  embedded_neighbors <- knn_indices(
    D_embedding,
    k
  )
  
  
  embedded_ranks <- distance_rank_matrix(
    D_embedding
  )
  
  
  penalty <- 0
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    missing_neighbors <- setdiff(
      original_neighbors[
        i,
      ],
      embedded_neighbors[
        i,
      ]
    )
    
    
    if (
      length(
        missing_neighbors
      ) >
      0
    ) {
      
      penalty <- penalty +
        sum(
          embedded_ranks[
            i,
            missing_neighbors
          ] -
            k
        )
    }
  }
  
  
  denominator <- n *
    k *
    (
      2 * n -
        3 * k -
        1
    )
  
  
  1 -
    (
      2 /
        denominator
    ) *
    penalty
}


# ------------------------------------------------------------------------------
# 31. Compare
# ------------------------------------------------------------------------------

embedding_quality <- data.frame(
  Method = c(
    "Classical MDS",
    "Global Metric MDS",
    "Local MDS"
  ),
  Trustworthiness = c(
    trustworthiness(
      D,
      pairwise_euclidean(
        Z_initial
      ),
      k
    ),
    trustworthiness(
      D,
      D_global,
      k
    ),
    trustworthiness(
      D,
      D_local_embedding,
      k
    )
  ),
  Continuity = c(
    continuity(
      D,
      pairwise_euclidean(
        Z_initial
      ),
      k
    ),
    continuity(
      D,
      D_global,
      k
    ),
    continuity(
      D,
      D_local_embedding,
      k
    )
  ),
  KNN_Overlap = c(
    overlap_classical$average,
    overlap_global$average,
    overlap_local$average
  )
)


embedding_quality


# ==============================================================================
# PART XVII
#
# POINTWISE NEIGHBORHOOD QUALITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Plot Pointwise KNN Preservation
# ------------------------------------------------------------------------------

plot(
  Z_local[
    ,
    1
  ],
  Z_local[
    ,
    2
  ],
  pch = 19,
  cex =
    0.5 +
    overlap_local$pointwise,
  xlab = "Local MDS Dimension 1",
  ylab = "Local MDS Dimension 2",
  main = "Pointwise Neighborhood Preservation"
)


# Larger points have better KNN overlap.


# ==============================================================================
# PART XVIII
#
# EFFECT OF TAU
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Repulsion Strength Grid
# ------------------------------------------------------------------------------

tau_values <- c(
  0,
  0.001,
  0.005,
  0.01,
  0.02,
  0.05
)


tau_models <- vector(
  "list",
  length(
    tau_values
  )
)


tau_results <- data.frame(
  Tau =
    tau_values,
  Objective =
    NA_real_,
  Global_Stress =
    NA_real_,
  Local_RMSE =
    NA_real_,
  Trustworthiness =
    NA_real_,
  Continuity =
    NA_real_,
  KNN_Overlap =
    NA_real_
)


for (
  i in seq_along(
    tau_values
  )
) {
  
  fit_i <- fit_local_mds(
    D,
    k =
      k,
    tau =
      tau_values[i],
    dimensions = 2,
    initial_coordinates =
      Z_initial,
    max_iterations = 700
  )
  
  
  tau_models[[i]] <-
    fit_i
  
  
  D_i <- fit_i$D_embedding
  
  
  overlap_i <- knn_overlap(
    D,
    D_i,
    k
  )
  
  
  tau_results$Objective[i] <-
    fit_i$objective
  
  
  tau_results$Global_Stress[i] <-
    normalized_stress(
      D,
      D_i
    )
  
  
  tau_results$Local_RMSE[i] <-
    local_distance_rmse(
      D,
      D_i,
      neighbor_graph
    )
  
  
  tau_results$Trustworthiness[i] <-
    trustworthiness(
      D,
      D_i,
      k
    )
  
  
  tau_results$Continuity[i] <-
    continuity(
      D,
      D_i,
      k
    )
  
  
  tau_results$KNN_Overlap[i] <-
    overlap_i$average
}


tau_results


# ==============================================================================
# PART XIX
#
# TAU TRADEOFF PLOTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Trustworthiness
# ------------------------------------------------------------------------------

plot(
  tau_results$Tau,
  tau_results$Trustworthiness,
  type = "b",
  pch = 19,
  xlab = "Tau",
  ylab = "Trustworthiness",
  main = "Effect of Nonlocal Repulsion"
)


# ------------------------------------------------------------------------------
# 35. Continuity
# ------------------------------------------------------------------------------

plot(
  tau_results$Tau,
  tau_results$Continuity,
  type = "b",
  pch = 19,
  xlab = "Tau",
  ylab = "Continuity",
  main = "Continuity vs Repulsion"
)


# ------------------------------------------------------------------------------
# 36. KNN Overlap
# ------------------------------------------------------------------------------

plot(
  tau_results$Tau,
  tau_results$KNN_Overlap,
  type = "b",
  pch = 19,
  xlab = "Tau",
  ylab = "Mean KNN Overlap",
  main = "Neighborhood Preservation vs Tau"
)


# ==============================================================================
# PART XX
#
# EFFECT OF K
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Neighborhood Size Grid
# ------------------------------------------------------------------------------

k_values <- c(
  4,
  6,
  8,
  12,
  20,
  30
)


k_results <- data.frame(
  K =
    k_values,
  Global_Stress =
    NA_real_,
  Local_RMSE =
    NA_real_,
  Trustworthiness =
    NA_real_,
  Continuity =
    NA_real_
)


k_models <- vector(
  "list",
  length(
    k_values
  )
)


for (
  i in seq_along(
    k_values
  )
) {
  
  graph_i <- symmetric_knn_graph(
    D,
    k =
      k_values[i]
  )
  
  
  fit_i <- fit_local_mds(
    D,
    k =
      k_values[i],
    tau = 0.01,
    dimensions = 2,
    initial_coordinates =
      Z_initial,
    max_iterations = 700
  )
  
  
  k_models[[i]] <-
    fit_i
  
  
  D_i <- fit_i$D_embedding
  
  
  k_results$Global_Stress[i] <-
    normalized_stress(
      D,
      D_i
    )
  
  
  k_results$Local_RMSE[i] <-
    local_distance_rmse(
      D,
      D_i,
      graph_i
    )
  
  
  k_results$Trustworthiness[i] <-
    trustworthiness(
      D,
      D_i,
      k_values[i]
    )
  
  
  k_results$Continuity[i] <-
    continuity(
      D,
      D_i,
      k_values[i]
    )
}


k_results


# ==============================================================================
# PART XXI
#
# SMALL K VS LARGE K
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Small Local Neighborhood
# ------------------------------------------------------------------------------

small_k_model <- k_models[[
  which(
    k_values ==
      4
  )
]]


# ------------------------------------------------------------------------------
# 39. Large Neighborhood
# ------------------------------------------------------------------------------

large_k_model <- k_models[[
  which(
    k_values ==
      30
  )
]]


# ------------------------------------------------------------------------------
# 40. Plot
# ------------------------------------------------------------------------------

plot(
  small_k_model$coordinates,
  pch = 19,
  cex = 0.5,
  xlab = "LMDS1",
  ylab = "LMDS2",
  main = "Local MDS: K = 4"
)


plot(
  large_k_model$coordinates,
  pch = 19,
  cex = 0.5,
  xlab = "LMDS1",
  ylab = "LMDS2",
  main = "Local MDS: K = 30"
)


# As K becomes large, Local MDS becomes increasingly global.


# ==============================================================================
# PART XXII
#
# SELECT HYPERPARAMETERS USING NEIGHBORHOOD AGREEMENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Grid Search
# ------------------------------------------------------------------------------

candidate_k <- c(
  5,
  8,
  12
)


candidate_tau <- c(
  0.002,
  0.005,
  0.01,
  0.02
)


hyperparameter_results <- data.frame(
  K = integer(0),
  Tau = numeric(0),
  Trustworthiness = numeric(0),
  Continuity = numeric(0),
  KNN_Overlap = numeric(0)
)


hyperparameter_models <- list()


model_counter <- 0


for (
  k_candidate in candidate_k
) {
  
  for (
    tau_candidate in candidate_tau
  ) {
    
    model_counter <- model_counter +
      1
    
    
    fit_candidate <- fit_local_mds(
      D,
      k =
        k_candidate,
      tau =
        tau_candidate,
      dimensions = 2,
      initial_coordinates =
        Z_initial,
      max_iterations = 600
    )
    
    
    D_candidate <- fit_candidate$D_embedding
    
    
    overlap_candidate <- knn_overlap(
      D,
      D_candidate,
      k_candidate
    )
    
    
    trust_candidate <- trustworthiness(
      D,
      D_candidate,
      k_candidate
    )
    
    
    continuity_candidate <- continuity(
      D,
      D_candidate,
      k_candidate
    )
    
    
    hyperparameter_results <- rbind(
      hyperparameter_results,
      data.frame(
        K =
          k_candidate,
        Tau =
          tau_candidate,
        Trustworthiness =
          trust_candidate,
        Continuity =
          continuity_candidate,
        KNN_Overlap =
          overlap_candidate$average
      )
    )
    
    
    hyperparameter_models[[model_counter]] <-
      fit_candidate
  }
}


hyperparameter_results


# ------------------------------------------------------------------------------
# 42. Simple Combined Neighborhood Criterion
#
# Equal-weight harmonic mean of trustworthiness and continuity.
#
# This is a convenient pedagogical summary, not the exact Chen-Buja
# meta-criterion.
# ------------------------------------------------------------------------------

hyperparameter_results$Neighborhood_Score <-
  2 *
  hyperparameter_results$Trustworthiness *
  hyperparameter_results$Continuity /
  (
    hyperparameter_results$Trustworthiness +
      hyperparameter_results$Continuity
  )


best_index <- which.max(
  hyperparameter_results$Neighborhood_Score
)


hyperparameter_results[
  best_index,
]


best_local_model <- hyperparameter_models[[best_index]]


# ==============================================================================
# PART XXIII
#
# VISUALIZE TUNED LOCAL MDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Best Embedding
# ------------------------------------------------------------------------------

plot(
  best_local_model$coordinates,
  pch = 19,
  cex = 0.6,
  xlab = "Local MDS Dimension 1",
  ylab = "Local MDS Dimension 2",
  main = "Tuned Local MDS"
)


# ==============================================================================
# PART XXIV
#
# MANIFOLD ORDERING
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Compare Embedding Coordinate with True Latent Position
#
# This is available only because this is a simulation.
# ------------------------------------------------------------------------------

cor(
  latent_t,
  best_local_model$coordinates[
    ,
    1
  ],
  method = "spearman"
)


cor(
  latent_t,
  best_local_model$coordinates[
    ,
    2
  ],
  method = "spearman"
)


# Since MDS embeddings may rotate or reflect, individual coordinate correlations
# are not necessarily the best invariant measure.


# ==============================================================================
# PART XXV
#
# LOCAL DISTANCE SCATTERPLOT
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Target vs Embedded LOCAL Distances
# ------------------------------------------------------------------------------

local_pairs <- upper.tri(
  D
) &
  best_local_model$neighbor_graph


plot(
  D[
    local_pairs
  ],
  best_local_model$D_embedding[
    local_pairs
  ],
  pch = 19,
  cex = 0.5,
  xlab = "Original Local Distance",
  ylab = "Embedded Local Distance",
  main = "Local Distance Preservation"
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# ==============================================================================
# PART XXVI
#
# NONLOCAL DISTANCES
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Compare Nonlocal Distances
# ------------------------------------------------------------------------------

nonlocal_pairs <- upper.tri(
  D
) &
  !best_local_model$neighbor_graph


summary(
  best_local_model$D_embedding[
    nonlocal_pairs
  ]
)


summary(
  D[
    nonlocal_pairs
  ]
)


# Local MDS is NOT trying to faithfully reproduce these original large
# distances. It mainly tries to stop these pairs from collapsing together.


# ==============================================================================
# PART XXVII
#
# REPULSION DEMONSTRATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Tau = 0
#
# With no repulsion, preserving only local edges can permit unrelated portions
# of the manifold to collapse into the same region of the map.
# ------------------------------------------------------------------------------

no_repulsion_model <- fit_local_mds(
  D,
  k = 8,
  tau = 0,
  dimensions = 2,
  initial_coordinates =
    Z_initial,
  max_iterations = 700
)


plot(
  no_repulsion_model$coordinates,
  pch = 19,
  cex = 0.5,
  xlab = "LMDS1",
  ylab = "LMDS2",
  main = "Local Stress Only: Tau = 0"
)


# ------------------------------------------------------------------------------
# 48. Positive Repulsion
# ------------------------------------------------------------------------------

plot(
  Z_local,
  pch = 19,
  cex = 0.5,
  xlab = "LMDS1",
  ylab = "LMDS2",
  main = "Local MDS with Nonlocal Repulsion"
)


# ==============================================================================
# PART XXVIII
#
# MULTIPLE RANDOM STARTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Local MDS Is Nonconvex
# ------------------------------------------------------------------------------

set.seed(999)


number_starts <- 5


start_results <- data.frame(
  Start =
    seq_len(
      number_starts
    ),
  Objective =
    NA_real_,
  Trustworthiness =
    NA_real_,
  Continuity =
    NA_real_
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
    
    initial_start <- Z_initial
    
  } else {
    
    initial_start <- matrix(
      rnorm(
        n * 2
      ),
      nrow = n,
      ncol = 2
    )
  }
  
  
  fit_start <- fit_local_mds(
    D,
    k = 8,
    tau = 0.01,
    dimensions = 2,
    initial_coordinates =
      initial_start,
    max_iterations = 700
  )
  
  
  start_models[[start]] <-
    fit_start
  
  
  start_results$Objective[start] <-
    fit_start$objective
  
  
  start_results$Trustworthiness[start] <-
    trustworthiness(
      D,
      fit_start$D_embedding,
      8
    )
  
  
  start_results$Continuity[start] <-
    continuity(
      D,
      fit_start$D_embedding,
      8
    )
}


start_results


# ==============================================================================
# PART XXIX
#
# LINEAR DATA EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Approximately Linear Data
# ------------------------------------------------------------------------------

set.seed(111)


n_linear <- 120


linear_x <- rnorm(
  n_linear
)


X_linear <- cbind(
  X1 =
    linear_x +
    rnorm(
      n_linear,
      sd = 0.2
    ),
  X2 =
    2 *
    linear_x +
    rnorm(
      n_linear,
      sd = 0.2
    ),
  X3 =
    -1.5 *
    linear_x +
    rnorm(
      n_linear,
      sd = 0.2
    )
)


D_linear <- pairwise_euclidean(
  X_linear
)


classical_linear <- classical_mds(
  D_linear,
  dimensions = 2
)


local_linear <- fit_local_mds(
  D_linear,
  k = 8,
  tau = 0.01,
  dimensions = 2,
  initial_coordinates =
    classical_linear$coordinates,
  max_iterations = 600
)


# ------------------------------------------------------------------------------
# 51. Classical MDS
# ------------------------------------------------------------------------------

plot(
  classical_linear$coordinates,
  pch = 19,
  cex = 0.5,
  xlab = "MDS1",
  ylab = "MDS2",
  main = "Classical MDS: Nearly Linear Data"
)


# ------------------------------------------------------------------------------
# 52. Local MDS
# ------------------------------------------------------------------------------

plot(
  local_linear$coordinates,
  pch = 19,
  cex = 0.5,
  xlab = "LMDS1",
  ylab = "LMDS2",
  main = "Local MDS: Nearly Linear Data"
)


# On simple globally Euclidean structure, localization may provide little
# benefit over ordinary MDS.


# ==============================================================================
# PART XXX
#
# HIGH-DIMENSIONAL MANIFOLD
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. Add Irrelevant Dimensions
# ------------------------------------------------------------------------------

set.seed(777)


noise_dimensions <- matrix(
  rnorm(
    n * 7,
    sd = 0.10
  ),
  nrow = n,
  ncol = 7
)


X_high <- cbind(
  X,
  noise_dimensions
)


D_high <- pairwise_euclidean(
  X_high
)


classical_high <- classical_mds(
  D_high,
  dimensions = 2
)


local_high <- fit_local_mds(
  D_high,
  k = 8,
  tau = 0.01,
  dimensions = 2,
  initial_coordinates =
    classical_high$coordinates,
  max_iterations = 700
)


plot(
  local_high$coordinates,
  pch = 19,
  cex = 0.5,
  xlab = "LMDS1",
  ylab = "LMDS2",
  main = "Local MDS from 10-Dimensional Observations"
)


# ==============================================================================
# PART XXXI
#
# SCALING MATTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Large-Scale Irrelevant Variable
# ------------------------------------------------------------------------------

set.seed(888)


X_scale_problem <- cbind(
  X,
  Large_Noise =
    rnorm(
      n,
      sd = 50
    )
)


D_bad_scale <- pairwise_euclidean(
  X_scale_problem
)


# The neighborhood graph itself is now corrupted because the irrelevant
# variable dominates Euclidean distance.


# ------------------------------------------------------------------------------
# 55. Standardize Before Constructing Distances
# ------------------------------------------------------------------------------

X_scale_fixed <- scale(
  X_scale_problem
)


D_scale_fixed <- pairwise_euclidean(
  X_scale_fixed
)


local_scale_fixed <- fit_local_mds(
  D_scale_fixed,
  k = 8,
  tau = 0.01,
  dimensions = 2,
  max_iterations = 700
)


plot(
  local_scale_fixed$coordinates,
  pch = 19,
  cex = 0.5,
  xlab = "LMDS1",
  ylab = "LMDS2",
  main = "Local MDS after Feature Standardization"
)


# ==============================================================================
# PART XXXII
#
# OPTIONAL PACKAGE VERIFICATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. smacofx
#
# The smacofx package contains an implementation of Local MDS.
# Its exact normalization / parameterization may differ from this manual
# pedagogical implementation.
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "smacofx",
    quietly = TRUE
  )
) {
  
  cat(
    "\nsmacofx is installed.\n"
  )
  
  
  cat(
    "You can compare against smacofx::lmds().\n"
  )
  
  
  # Example:
  #
  # package_fit <- smacofx::lmds(
  #   delta = D,
  #   k = 8,
  #   tau = 0.01,
  #   ndim = 2
  # )
  #
  # package_fit$conf
  #
  # Do not expect the tau parameter to match numerically with this manual
  # implementation because package normalization conventions can differ.
  
}


# ==============================================================================
# PART XXXIII
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nLocal Multidimensional Scaling Summary\n"
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
  "Original dimensions:",
  ncol(
    X
  ),
  "\n"
)


cat(
  "Neighborhood size K:",
  k,
  "\n"
)


cat(
  "Repulsion parameter tau:",
  local_fit$tau,
  "\n"
)


cat(
  "Number of local graph edges:",
  sum(
    neighbor_graph
  ) /
    2,
  "\n"
)


cat(
  "Local MDS convergence code:",
  local_fit$convergence,
  "\n"
)


cat(
  "Classical MDS KNN overlap:",
  round(
    overlap_classical$average,
    4
  ),
  "\n"
)


cat(
  "Global metric MDS KNN overlap:",
  round(
    overlap_global$average,
    4
  ),
  "\n"
)


cat(
  "Local MDS KNN overlap:",
  round(
    overlap_local$average,
    4
  ),
  "\n"
)


cat(
  "Local MDS trustworthiness:",
  round(
    trustworthiness(
      D,
      D_local_embedding,
      k
    ),
    4
  ),
  "\n"
)


cat(
  "Local MDS continuity:",
  round(
    continuity(
      D,
      D_local_embedding,
      k
    ),
    4
  ),
  "\n"
)