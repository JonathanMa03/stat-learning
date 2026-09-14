# ==============================================================================
# K-Medoids Clustering
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Unsupervised learning
#   - Medoids
#   - Pairwise dissimilarities
#   - Partitioning Around Medoids (PAM)
#   - Assignment step
#   - Swap step
#   - Multiple random starts
#   - Robustness to outliers
#   - Arbitrary distance measures
#   - Silhouette analysis
#   - Comparison with K-means
#
#
# K-medoids objective:
#
#       minimize
#
#       sum_i min_k d(x_i, m_k)
#
#
# where:
#
#       m_k
#
# is the medoid of cluster k and must be one of the observed data points.
#
#
# Unlike K-means:
#
#       K-means:
#
#           center = arithmetic mean
#
#       K-medoids:
#
#           center = actual observation
#
#
# This script implements a simplified PAM-style algorithm manually.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE CLUSTERED DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Three Clusters
# ------------------------------------------------------------------------------

set.seed(123)

n_per_cluster <- 100


cluster_1 <- cbind(
  x1 = rnorm(
    n_per_cluster,
    mean = -3,
    sd = 0.7
  ),
  x2 = rnorm(
    n_per_cluster,
    mean = 0,
    sd = 0.7
  )
)


cluster_2 <- cbind(
  x1 = rnorm(
    n_per_cluster,
    mean = 3,
    sd = 0.7
  ),
  x2 = rnorm(
    n_per_cluster,
    mean = 0,
    sd = 0.7
  )
)


cluster_3 <- cbind(
  x1 = rnorm(
    n_per_cluster,
    mean = 0,
    sd = 0.7
  ),
  x2 = rnorm(
    n_per_cluster,
    mean = 4,
    sd = 0.7
  )
)


X <- rbind(
  cluster_1,
  cluster_2,
  cluster_3
)


true_cluster <- rep(
  1:3,
  each = n_per_cluster
)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  xlab = "x1",
  ylab = "x2",
  main = "Unlabeled Data"
)


# ==============================================================================
# PART II
#
# DISTANCE FUNCTIONS
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
# 4. Manhattan Distance
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
# 5. General Pairwise Distance Matrix
# ------------------------------------------------------------------------------

pairwise_distance_matrix <- function(
    X,
    distance = c(
      "euclidean",
      "manhattan"
    )
) {
  
  distance <- match.arg(
    distance
  )
  
  
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
  
  
  distance_function <- if (
    distance ==
    "euclidean"
  ) {
    
    euclidean_distance
    
  } else {
    
    manhattan_distance
  }
  
  
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
      
      distance_ij <- distance_function(
        X[i, ],
        X[j, ]
      )
      
      
      D[i, j] <- distance_ij
      
      D[j, i] <- distance_ij
    }
  }
  
  
  D
}


# ------------------------------------------------------------------------------
# 6. Compute Euclidean Distance Matrix
# ------------------------------------------------------------------------------

D <- pairwise_distance_matrix(
  X,
  distance = "euclidean"
)


# ==============================================================================
# PART III
#
# MEDOID CONCEPT
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Medoid of One Set of Observations
# ------------------------------------------------------------------------------

find_medoid <- function(
    D,
    indices
) {
  
  if (
    length(
      indices
    ) ==
    1
  ) {
    
    return(
      indices
    )
  }
  
  
  within_distances <- D[
    indices,
    indices,
    drop = FALSE
  ]
  
  
  total_distance <- rowSums(
    within_distances
  )
  
  
  indices[
    which.min(
      total_distance
    )
  ]
}


# ------------------------------------------------------------------------------
# 8. Example Medoid of Entire Dataset
# ------------------------------------------------------------------------------

global_medoid <- find_medoid(
  D,
  seq_len(
    nrow(X)
  )
)


global_medoid


X[
  global_medoid,
]


# ==============================================================================
# PART IV
#
# INITIALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Random Medoid Initialization
# ------------------------------------------------------------------------------

initialize_medoids <- function(
    n,
    K
) {
  
  sample(
    seq_len(n),
    size = K,
    replace = FALSE
  )
}


# ------------------------------------------------------------------------------
# 10. Example Initialization
# ------------------------------------------------------------------------------

set.seed(100)


initial_medoids <- initialize_medoids(
  nrow(X),
  K = 3
)


initial_medoids


plot(
  X,
  pch = 19,
  cex = 0.5,
  xlab = "x1",
  ylab = "x2",
  main = "Initial Random Medoids"
)


points(
  X[
    initial_medoids,
    ,
    drop = FALSE
  ],
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART V
#
# ASSIGNMENT STEP
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Assign Observations to Closest Medoid
# ------------------------------------------------------------------------------

assign_to_medoids <- function(
    D,
    medoids
) {
  
  medoid_distances <- D[
    ,
    medoids,
    drop = FALSE
  ]
  
  
  max.col(
    -medoid_distances,
    ties.method = "first"
  )
}


# ------------------------------------------------------------------------------
# 12. First Assignment
# ------------------------------------------------------------------------------

first_assignment <- assign_to_medoids(
  D,
  initial_medoids
)


table(
  first_assignment
)


# ==============================================================================
# PART VI
#
# K-MEDOIDS OBJECTIVE
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Total Dissimilarity
# ------------------------------------------------------------------------------

kmedoids_objective <- function(
    D,
    medoids
) {
  
  nearest_distance <- apply(
    D[
      ,
      medoids,
      drop = FALSE
    ],
    1,
    min
  )
  
  
  sum(
    nearest_distance
  )
}


# ------------------------------------------------------------------------------
# 14. Initial Objective
# ------------------------------------------------------------------------------

initial_objective <- kmedoids_objective(
  D,
  initial_medoids
)


initial_objective


# ==============================================================================
# PART VII
#
# UPDATE MEDOIDS WITHIN CLUSTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Recompute Cluster Medoids
# ------------------------------------------------------------------------------

update_cluster_medoids <- function(
    D,
    cluster_assignment,
    K,
    old_medoids
) {
  
  new_medoids <- integer(
    K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    indices <- which(
      cluster_assignment == k
    )
    
    
    if (
      length(indices) ==
      0
    ) {
      
      new_medoids[k] <-
        old_medoids[k]
      
    } else {
      
      new_medoids[k] <- find_medoid(
        D,
        indices
      )
    }
  }
  
  
  new_medoids
}


# ------------------------------------------------------------------------------
# 16. First Update
# ------------------------------------------------------------------------------

first_updated_medoids <- update_cluster_medoids(
  D,
  first_assignment,
  K = 3,
  old_medoids =
    initial_medoids
)


first_updated_medoids


X[
  first_updated_medoids,
  ,
  drop = FALSE
]


# ==============================================================================
# PART VIII
#
# SIMPLE ALTERNATING K-MEDOIDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Fit Alternating K-Medoids
# ------------------------------------------------------------------------------

kmedoids_alternating <- function(
    X,
    K,
    distance = "euclidean",
    max_iterations = 100,
    initial_medoids = NULL,
    seed = NULL,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  D <- pairwise_distance_matrix(
    X,
    distance =
      distance
  )
  
  
  n <- nrow(
    X
  )
  
  
  if (
    !is.null(seed)
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  if (
    is.null(
      initial_medoids
    )
  ) {
    
    medoids <- initialize_medoids(
      n,
      K
    )
    
  } else {
    
    medoids <- initial_medoids
  }
  
  
  objective_history <- numeric(
    max_iterations
  )
  
  
  medoid_history <- vector(
    "list",
    max_iterations + 1
  )
  
  
  medoid_history[[1]] <-
    medoids
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    # ------------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------------
    
    cluster_assignment <- assign_to_medoids(
      D,
      medoids
    )
    
    
    # ------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------
    
    new_medoids <- update_cluster_medoids(
      D,
      cluster_assignment,
      K,
      old_medoids =
        medoids
    )
    
    
    objective_history[
      iteration
    ] <- kmedoids_objective(
      D,
      new_medoids
    )
    
    
    medoid_history[[
      iteration + 1
    ]] <- new_medoids
    
    
    if (
      verbose
    ) {
      
      cat(
        "Iteration:",
        iteration,
        "| Objective:",
        round(
          objective_history[
            iteration
          ],
          4
        ),
        "| Medoids:",
        paste(
          new_medoids,
          collapse = ", "
        ),
        "\n"
      )
    }
    
    
    if (
      identical(
        sort(
          new_medoids
        ),
        sort(
          medoids
        )
      )
    ) {
      
      medoids <- new_medoids
      
      converged <- TRUE
      
      break
    }
    
    
    medoids <- new_medoids
  }
  
  
  iterations_used <- iteration
  
  
  objective_history <- objective_history[
    seq_len(
      iterations_used
    )
  ]
  
  
  medoid_history <- medoid_history[
    seq_len(
      iterations_used + 1
    )
  ]
  
  
  final_cluster <- assign_to_medoids(
    D,
    medoids
  )
  
  
  return(
    list(
      medoids =
        medoids,
      centers =
        X[
          medoids,
          ,
          drop = FALSE
        ],
      cluster =
        final_cluster,
      objective =
        kmedoids_objective(
          D,
          medoids
        ),
      objective_history =
        objective_history,
      medoid_history =
        medoid_history,
      distance_matrix =
        D,
      distance =
        distance,
      iterations =
        iterations_used,
      converged =
        converged
    )
  )
}


# ==============================================================================
# PART IX
#
# FIT ONE MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Run K-Medoids
# ------------------------------------------------------------------------------

KMedoids_model <- kmedoids_alternating(
  X,
  K = 3,
  distance = "euclidean",
  max_iterations = 100,
  seed = 123,
  verbose = TRUE
)


KMedoids_model$medoids


KMedoids_model$centers


KMedoids_model$objective


# ==============================================================================
# PART X
#
# VISUALIZE CLUSTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Estimated Clusters
# ------------------------------------------------------------------------------

plot(
  X,
  pch =
    KMedoids_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "Manual K-Medoids Clustering"
)


points(
  KMedoids_model$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XI
#
# CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Objective Path
# ------------------------------------------------------------------------------

plot(
  seq_along(
    KMedoids_model$objective_history
  ),
  KMedoids_model$objective_history,
  type = "b",
  pch = 19,
  xlab = "Iteration",
  ylab = "Total Dissimilarity",
  main = "K-Medoids Convergence"
)


# ==============================================================================
# PART XII
#
# PAM-STYLE SWAP SEARCH
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Best Single Swap
# ------------------------------------------------------------------------------

best_medoid_swap <- function(
    D,
    medoids
) {
  
  n <- nrow(
    D
  )
  
  
  current_objective <- kmedoids_objective(
    D,
    medoids
  )
  
  
  non_medoids <- setdiff(
    seq_len(n),
    medoids
  )
  
  
  best_objective <- current_objective
  
  best_medoids <- medoids
  
  best_removed <- NA_integer_
  
  best_added <- NA_integer_
  
  
  for (
    medoid_position in seq_along(
      medoids
    )
  ) {
    
    for (
      candidate in non_medoids
    ) {
      
      proposed_medoids <- medoids
      
      
      proposed_medoids[
        medoid_position
      ] <- candidate
      
      
      proposed_objective <- kmedoids_objective(
        D,
        proposed_medoids
      )
      
      
      if (
        proposed_objective <
        best_objective
      ) {
        
        best_objective <-
          proposed_objective
        
        best_medoids <-
          proposed_medoids
        
        best_removed <-
          medoids[
            medoid_position
          ]
        
        best_added <-
          candidate
      }
    }
  }
  
  
  list(
    medoids =
      best_medoids,
    objective =
      best_objective,
    improvement =
      current_objective -
      best_objective,
    removed =
      best_removed,
    added =
      best_added
  )
}


# ------------------------------------------------------------------------------
# 22. PAM-Style Swap Algorithm
# ------------------------------------------------------------------------------

kmedoids_pam_manual <- function(
    X,
    K,
    distance = "euclidean",
    max_swaps = 100,
    initial_medoids = NULL,
    seed = NULL,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  D <- pairwise_distance_matrix(
    X,
    distance =
      distance
  )
  
  
  n <- nrow(
    X
  )
  
  
  if (
    !is.null(seed)
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  if (
    is.null(
      initial_medoids
    )
  ) {
    
    medoids <- initialize_medoids(
      n,
      K
    )
    
  } else {
    
    medoids <- initial_medoids
  }
  
  
  objective_history <- numeric(
    max_swaps + 1
  )
  
  
  objective_history[1] <- kmedoids_objective(
    D,
    medoids
  )
  
  
  medoid_history <- vector(
    "list",
    max_swaps + 1
  )
  
  
  medoid_history[[1]] <-
    medoids
  
  
  swaps_used <- 0
  
  
  for (
    swap_iteration in seq_len(
      max_swaps
    )
  ) {
    
    swap_result <- best_medoid_swap(
      D,
      medoids
    )
    
    
    if (
      swap_result$improvement <=
      1e-12
    ) {
      
      break
    }
    
    
    medoids <- swap_result$medoids
    
    
    swaps_used <- swap_iteration
    
    
    objective_history[
      swap_iteration + 1
    ] <- swap_result$objective
    
    
    medoid_history[[
      swap_iteration + 1
    ]] <- medoids
    
    
    if (
      verbose
    ) {
      
      cat(
        "Swap:",
        swap_iteration,
        "| Removed:",
        swap_result$removed,
        "| Added:",
        swap_result$added,
        "| Objective:",
        round(
          swap_result$objective,
          4
        ),
        "\n"
      )
    }
  }
  
  
  history_length <- swaps_used + 1
  
  
  objective_history <- objective_history[
    seq_len(
      history_length
    )
  ]
  
  
  medoid_history <- medoid_history[
    seq_len(
      history_length
    )
  ]
  
  
  cluster_assignment <- assign_to_medoids(
    D,
    medoids
  )
  
  
  list(
    medoids =
      medoids,
    centers =
      X[
        medoids,
        ,
        drop = FALSE
      ],
    cluster =
      cluster_assignment,
    objective =
      kmedoids_objective(
        D,
        medoids
      ),
    objective_history =
      objective_history,
    medoid_history =
      medoid_history,
    swaps =
      swaps_used,
    distance =
      distance,
    distance_matrix =
      D
  )
}


# ==============================================================================
# PART XIII
#
# FIT PAM-STYLE MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Run Manual PAM
# ------------------------------------------------------------------------------

PAM_model <- kmedoids_pam_manual(
  X,
  K = 3,
  distance = "euclidean",
  max_swaps = 50,
  seed = 123,
  verbose = TRUE
)


PAM_model$medoids


PAM_model$centers


PAM_model$objective


# ------------------------------------------------------------------------------
# 24. Plot PAM Result
# ------------------------------------------------------------------------------

plot(
  X,
  pch =
    PAM_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "Manual PAM-Style K-Medoids"
)


points(
  PAM_model$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XIV
#
# MULTIPLE RANDOM STARTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Multi-Start Wrapper
# ------------------------------------------------------------------------------

kmedoids_multistart <- function(
    X,
    K,
    distance = "euclidean",
    nstart = 20,
    max_swaps = 50,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  start_seeds <- sample(
    seq_len(
      1000000
    ),
    nstart
  )
  
  
  models <- vector(
    "list",
    nstart
  )
  
  
  objectives <- numeric(
    nstart
  )
  
  
  for (
    s in seq_len(
      nstart
    )
  ) {
    
    models[[s]] <- kmedoids_pam_manual(
      X,
      K = K,
      distance =
        distance,
      max_swaps =
        max_swaps,
      seed =
        start_seeds[s]
    )
    
    
    objectives[s] <-
      models[[s]]$objective
  }
  
  
  best <- which.min(
    objectives
  )
  
  
  best_model <- models[[best]]
  
  
  best_model$all_objectives <-
    objectives
  
  
  best_model$best_start <-
    best
  
  
  best_model$nstart <-
    nstart
  
  
  best_model
}


# ------------------------------------------------------------------------------
# 26. Final Multi-Start Model
# ------------------------------------------------------------------------------

final_KMedoids <- kmedoids_multistart(
  X,
  K = 3,
  distance = "euclidean",
  nstart = 30,
  max_swaps = 50,
  seed = 123
)


final_KMedoids$objective


final_KMedoids$medoids


# ==============================================================================
# PART XV
#
# LOCAL MINIMA
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Distribution of Random-Start Solutions
# ------------------------------------------------------------------------------

hist(
  final_KMedoids$all_objectives,
  breaks = 20,
  xlab = "Final Objective",
  main = "K-Medoids Random Starts"
)


# ==============================================================================
# PART XVI
#
# CLUSTER RECOVERY
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Contingency Table
# ------------------------------------------------------------------------------

table(
  True =
    true_cluster,
  Estimated =
    final_KMedoids$cluster
)


# ==============================================================================
# PART XVII
#
# RAND INDEX
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Rand Index
# ------------------------------------------------------------------------------

rand_index <- function(
    true_labels,
    estimated_labels
) {
  
  n <- length(
    true_labels
  )
  
  
  agreements <- 0
  
  total_pairs <- 0
  
  
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
      
      same_true <-
        true_labels[i] ==
        true_labels[j]
      
      
      same_estimated <-
        estimated_labels[i] ==
        estimated_labels[j]
      
      
      agreements <- agreements +
        (
          same_true ==
            same_estimated
        )
      
      
      total_pairs <- total_pairs +
        1
    }
  }
  
  
  agreements /
    total_pairs
}


rand_index(
  true_cluster,
  final_KMedoids$cluster
)


# ==============================================================================
# PART XVIII
#
# SILHOUETTE ANALYSIS
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Manual Silhouette Values
# ------------------------------------------------------------------------------

silhouette_manual <- function(
    D,
    cluster
) {
  
  n <- nrow(
    D
  )
  
  
  cluster_values <- sort(
    unique(
      cluster
    )
  )
  
  
  silhouette <- numeric(
    n
  )
  
  
  for (
    i in seq_len(n)
  ) {
    
    own_cluster <- cluster[i]
    
    
    own_indices <- which(
      cluster ==
        own_cluster
    )
    
    
    own_indices <- own_indices[
      own_indices != i
    ]
    
    
    if (
      length(
        own_indices
      ) ==
      0
    ) {
      
      silhouette[i] <- 0
      
      next
    }
    
    
    a_i <- mean(
      D[
        i,
        own_indices
      ]
    )
    
    
    other_clusters <- cluster_values[
      cluster_values !=
        own_cluster
    ]
    
    
    other_average_distances <- sapply(
      other_clusters,
      function(k) {
        
        indices_k <- which(
          cluster ==
            k
        )
        
        
        mean(
          D[
            i,
            indices_k
          ]
        )
      }
    )
    
    
    b_i <- min(
      other_average_distances
    )
    
    
    silhouette[i] <- (
      b_i -
        a_i
    ) /
      max(
        a_i,
        b_i
      )
  }
  
  
  silhouette
}


# ------------------------------------------------------------------------------
# 31. Silhouette for Final Model
# ------------------------------------------------------------------------------

final_silhouette <- silhouette_manual(
  final_KMedoids$distance_matrix,
  final_KMedoids$cluster
)


mean(
  final_silhouette
)


summary(
  final_silhouette
)


hist(
  final_silhouette,
  breaks = 30,
  xlab = "Silhouette",
  main = "K-Medoids Silhouette Values"
)


# ==============================================================================
# PART XIX
#
# CHOOSING K
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Compare K by Silhouette
# ------------------------------------------------------------------------------

K_values <- 2:7


K_results <- data.frame(
  K =
    K_values,
  Objective =
    NA_real_,
  Average_Silhouette =
    NA_real_
)


for (
  i in seq_along(
    K_values
  )
) {
  
  model_i <- kmedoids_multistart(
    X,
    K =
      K_values[i],
    distance =
      "euclidean",
    nstart = 15,
    max_swaps = 30,
    seed =
      100 + i
  )
  
  
  silhouette_i <- silhouette_manual(
    model_i$distance_matrix,
    model_i$cluster
  )
  
  
  K_results$Objective[i] <-
    model_i$objective
  
  
  K_results$Average_Silhouette[i] <-
    mean(
      silhouette_i
    )
}


K_results


# ------------------------------------------------------------------------------
# 33. Objective Plot
# ------------------------------------------------------------------------------

plot(
  K_results$K,
  K_results$Objective,
  type = "b",
  pch = 19,
  xlab = "K",
  ylab = "Total Dissimilarity",
  main = "K-Medoids Objective vs K"
)


# ------------------------------------------------------------------------------
# 34. Silhouette Plot
# ------------------------------------------------------------------------------

plot(
  K_results$K,
  K_results$Average_Silhouette,
  type = "b",
  pch = 19,
  xlab = "K",
  ylab = "Average Silhouette",
  main = "Choosing K by Silhouette"
)


best_K <- K_results$K[
  which.max(
    K_results$Average_Silhouette
  )
]


best_K


# ==============================================================================
# PART XX
#
# K-MEANS COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Fit Base R K-Means
# ------------------------------------------------------------------------------

set.seed(123)


KMeans_model <- kmeans(
  X,
  centers = 3,
  nstart = 50
)


# ------------------------------------------------------------------------------
# 36. Compare Representatives
# ------------------------------------------------------------------------------

KMeans_model$centers


final_KMedoids$centers


# K-means centers need not correspond to observed data.
#
# K-medoids centers always correspond to actual observations.


# ==============================================================================
# PART XXI
#
# VERIFY MEDOIDS ARE ACTUAL OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Medoid Check
# ------------------------------------------------------------------------------

for (
  k in seq_len(
    nrow(
      final_KMedoids$centers
    )
  )
) {
  
  match_found <- any(
    apply(
      X,
      1,
      function(row) {
        
        all(
          row ==
            final_KMedoids$centers[
              k,
            ]
        )
      }
    )
  )
  
  
  print(
    match_found
  )
}


# ==============================================================================
# PART XXII
#
# OUTLIER ROBUSTNESS
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Add Extreme Outliers
# ------------------------------------------------------------------------------

outliers <- rbind(
  c(
    15,
    15
  ),
  c(
    16,
    14
  ),
  c(
    14,
    16
  )
)


X_outlier <- rbind(
  X,
  outliers
)


# ------------------------------------------------------------------------------
# 39. K-Means with Outliers
# ------------------------------------------------------------------------------

set.seed(123)


KMeans_outlier <- kmeans(
  X_outlier,
  centers = 3,
  nstart = 50
)


# ------------------------------------------------------------------------------
# 40. K-Medoids with Outliers
# ------------------------------------------------------------------------------

KMedoids_outlier <- kmedoids_multistart(
  X_outlier,
  K = 3,
  distance =
    "euclidean",
  nstart = 30,
  max_swaps = 50,
  seed = 123
)


# ------------------------------------------------------------------------------
# 41. Compare Representatives
# ------------------------------------------------------------------------------

KMeans_model$centers


KMeans_outlier$centers


final_KMedoids$centers


KMedoids_outlier$centers


# ------------------------------------------------------------------------------
# 42. Plot Outlier K-Means
# ------------------------------------------------------------------------------

plot(
  X_outlier,
  pch =
    KMeans_outlier$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "K-Means with Outliers"
)


points(
  KMeans_outlier$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ------------------------------------------------------------------------------
# 43. Plot Outlier K-Medoids
# ------------------------------------------------------------------------------

plot(
  X_outlier,
  pch =
    KMedoids_outlier$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "K-Medoids with Outliers"
)


points(
  KMedoids_outlier$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXIII
#
# MANHATTAN-DISTANCE K-MEDOIDS
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Fit with Manhattan Distance
# ------------------------------------------------------------------------------

manhattan_model <- kmedoids_multistart(
  X,
  K = 3,
  distance =
    "manhattan",
  nstart = 30,
  max_swaps = 50,
  seed = 123
)


manhattan_model$medoids


manhattan_model$objective


# ------------------------------------------------------------------------------
# 45. Plot Manhattan Result
# ------------------------------------------------------------------------------

plot(
  X,
  pch =
    manhattan_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "K-Medoids with Manhattan Distance"
)


points(
  manhattan_model$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXIV
#
# FEATURE SCALING
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Generate Scale-Sensitive Example
# ------------------------------------------------------------------------------

set.seed(333)


n_scale <- 300


true_scale_cluster <- rep(
  1:3,
  each = 100
)


signal <- c(
  rnorm(
    100,
    -3,
    0.5
  ),
  rnorm(
    100,
    0,
    0.5
  ),
  rnorm(
    100,
    3,
    0.5
  )
)


large_noise <- rnorm(
  n_scale,
  mean = 0,
  sd = 100
)


X_scale <- cbind(
  Signal =
    signal,
  Large_Noise =
    large_noise
)


# ------------------------------------------------------------------------------
# 47. Unscaled Model
# ------------------------------------------------------------------------------

unscaled_model <- kmedoids_multistart(
  X_scale,
  K = 3,
  distance =
    "euclidean",
  nstart = 20,
  seed = 123
)


# ------------------------------------------------------------------------------
# 48. Standardize
# ------------------------------------------------------------------------------

scale_mean <- colMeans(
  X_scale
)


scale_sd <- apply(
  X_scale,
  2,
  sd
)


X_scale_standardized <- sweep(
  X_scale,
  2,
  scale_mean,
  "-"
)


X_scale_standardized <- sweep(
  X_scale_standardized,
  2,
  scale_sd,
  "/"
)


# ------------------------------------------------------------------------------
# 49. Scaled Model
# ------------------------------------------------------------------------------

scaled_model <- kmedoids_multistart(
  X_scale_standardized,
  K = 3,
  distance =
    "euclidean",
  nstart = 20,
  seed = 123
)


data.frame(
  Method = c(
    "Unscaled",
    "Standardized"
  ),
  Rand_Index = c(
    rand_index(
      true_scale_cluster,
      unscaled_model$cluster
    ),
    rand_index(
      true_scale_cluster,
      scaled_model$cluster
    )
  )
)


# ==============================================================================
# PART XXV
#
# PREDICT CLUSTER FOR NEW DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Prediction Function
# ------------------------------------------------------------------------------

predict_kmedoids <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  medoid_points <- model$centers
  
  
  distance_function <- if (
    model$distance ==
    "euclidean"
  ) {
    
    euclidean_distance
    
  } else {
    
    manhattan_distance
  }
  
  
  prediction <- integer(
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
    
    distances <- apply(
      medoid_points,
      1,
      function(medoid) {
        
        distance_function(
          X_new[i, ],
          medoid
        )
      }
    )
    
    
    prediction[i] <- which.min(
      distances
    )
  }
  
  
  prediction
}


# ------------------------------------------------------------------------------
# 51. Example New Observations
# ------------------------------------------------------------------------------

new_points <- rbind(
  c(
    -3,
    0
  ),
  c(
    3,
    0
  ),
  c(
    0,
    4
  ),
  c(
    0,
    0
  )
)


colnames(
  new_points
) <- colnames(X)


predict_kmedoids(
  final_KMedoids,
  new_points
)


# ==============================================================================
# PART XXVI
#
# DECISION REGIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 52. Grid
# ------------------------------------------------------------------------------

grid_size <- 150


x1_grid <- seq(
  min(X[, 1]) - 1,
  max(X[, 1]) + 1,
  length.out =
    grid_size
)


x2_grid <- seq(
  min(X[, 2]) - 1,
  max(X[, 2]) + 1,
  length.out =
    grid_size
)


grid <- expand.grid(
  x1 = x1_grid,
  x2 = x2_grid
)


grid_cluster <- predict_kmedoids(
  final_KMedoids,
  grid
)


grid_matrix <- matrix(
  grid_cluster,
  nrow =
    grid_size,
  ncol =
    grid_size
)


# ------------------------------------------------------------------------------
# 53. Plot Regions
# ------------------------------------------------------------------------------

image(
  x1_grid,
  x2_grid,
  grid_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "K-Medoids Nearest-Medoid Regions"
)


points(
  X,
  pch =
    final_KMedoids$cluster,
  cex = 0.5
)


points(
  final_KMedoids$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXVII
#
# MEDOID INTERPRETABILITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Display Medoid Observations
# ------------------------------------------------------------------------------

medoid_table <- data.frame(
  Cluster =
    seq_len(
      length(
        final_KMedoids$medoids
      )
    ),
  Observation_Index =
    final_KMedoids$medoids,
  X[
    final_KMedoids$medoids,
    ,
    drop = FALSE
  ]
)


medoid_table


# Because a medoid is a real observation, it can often serve as an interpretable
# representative example of the cluster.


# ==============================================================================
# PART XXVIII
#
# OPTIONAL VERIFICATION WITH cluster::pam()
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Built-In PAM
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "cluster",
    quietly = TRUE
  )
) {
  
  pam_model <- cluster::pam(
    X,
    k = 3,
    metric =
      "euclidean"
  )
  
  
  print(
    pam_model$medoids
  )
  
  
  print(
    table(
      Manual =
        final_KMedoids$cluster,
      PAM =
        pam_model$clustering
    )
  )
  
  
  cat(
    "Manual objective:",
    round(
      final_KMedoids$objective,
      4
    ),
    "\n"
  )
  
  
  cat(
    "cluster::pam objective:",
    round(
      pam_model$objective[2],
      4
    ),
    "\n"
  )
}


# ==============================================================================
# PART XXIX
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Output
# ------------------------------------------------------------------------------

cat(
  "K-Medoids Clustering Summary\n"
)


cat(
  "----------------------------\n"
)


cat(
  "Number of observations:",
  nrow(X),
  "\n"
)


cat(
  "Number of variables:",
  ncol(X),
  "\n"
)


cat(
  "Number of clusters:",
  3,
  "\n"
)


cat(
  "Distance metric:",
  final_KMedoids$distance,
  "\n"
)


cat(
  "Final medoid indices:",
  paste(
    final_KMedoids$medoids,
    collapse = ", "
  ),
  "\n"
)


cat(
  "Final objective:",
  round(
    final_KMedoids$objective,
    4
  ),
  "\n"
)


cat(
  "Average silhouette:",
  round(
    mean(
      final_silhouette
    ),
    4
  ),
  "\n"
)


cat(
  "Best K by silhouette:",
  best_K,
  "\n"
)


cat(
  "Rand index against simulated truth:",
  round(
    rand_index(
      true_cluster,
      final_KMedoids$cluster
    ),
    4
  ),
  "\n"
)