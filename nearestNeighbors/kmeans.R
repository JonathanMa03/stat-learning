# ==============================================================================
# K-Means Clustering
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Unsupervised learning
#   - Clustering
#   - Cluster centroids
#   - Within-cluster sum of squares
#   - Lloyd's algorithm
#   - Assignment step
#   - Update step
#   - Local minima
#   - Random initialization
#   - Multiple random starts
#   - Choosing K
#   - Elbow method
#   - Silhouette analysis
#   - Feature scaling
#   - Sensitivity to outliers
#   - Relationship with nearest-neighbor methods
#
#
# K-means objective:
#
#       minimize
#
#       sum_{k=1}^K
#       sum_{i in C_k}
#       ||x_i - mu_k||^2
#
#
# where:
#
#       C_k  = cluster k
#       mu_k = centroid of cluster k
#
#
# Lloyd's algorithm alternates:
#
#   1. Assignment:
#
#       assign each observation to its nearest centroid
#
#   2. Update:
#
#       replace each centroid with the mean of its assigned observations
#
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
    sd = 0.8
  ),
  x2 = rnorm(
    n_per_cluster,
    mean = 0,
    sd = 0.8
  )
)


cluster_2 <- cbind(
  x1 = rnorm(
    n_per_cluster,
    mean = 3,
    sd = 0.8
  ),
  x2 = rnorm(
    n_per_cluster,
    mean = 0,
    sd = 0.8
  )
)


cluster_3 <- cbind(
  x1 = rnorm(
    n_per_cluster,
    mean = 0,
    sd = 0.8
  ),
  x2 = rnorm(
    n_per_cluster,
    mean = 4,
    sd = 0.8
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
# 2. Plot Data Without Labels
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  xlab = "x1",
  ylab = "x2",
  main = "Unlabeled Data"
)


# ------------------------------------------------------------------------------
# 3. True Clusters
# ------------------------------------------------------------------------------

plot(
  X,
  pch = true_cluster,
  xlab = "x1",
  ylab = "x2",
  main = "True Data-Generating Clusters"
)


# ==============================================================================
# PART II
#
# DISTANCE FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Squared Euclidean Distance
# ------------------------------------------------------------------------------

squared_euclidean_distance <- function(
    x,
    centroid
) {
  
  sum(
    (
      x -
        centroid
    )^2
  )
}


# ------------------------------------------------------------------------------
# 5. Distance from Every Observation to Every Centroid
# ------------------------------------------------------------------------------

distance_matrix <- function(
    X,
    centroids
) {
  
  X <- as.matrix(
    X
  )
  
  
  centroids <- as.matrix(
    centroids
  )
  
  
  n <- nrow(
    X
  )
  
  
  K <- nrow(
    centroids
  )
  
  
  distances <- matrix(
    NA_real_,
    nrow = n,
    ncol = K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    differences <- sweep(
      X,
      2,
      centroids[k, ],
      "-"
    )
    
    
    distances[, k] <- rowSums(
      differences^2
    )
  }
  
  
  distances
}


# ==============================================================================
# PART III
#
# INITIALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Random Observation Initialization
# ------------------------------------------------------------------------------

initialize_centroids <- function(
    X,
    K
) {
  
  indices <- sample(
    seq_len(
      nrow(X)
    ),
    size = K,
    replace = FALSE
  )
  
  
  X[
    indices,
    ,
    drop = FALSE
  ]
}


# ------------------------------------------------------------------------------
# 7. Example Initialization
# ------------------------------------------------------------------------------

set.seed(100)


initial_centroids <- initialize_centroids(
  X,
  K = 3
)


initial_centroids


plot(
  X,
  pch = 19,
  cex = 0.6,
  xlab = "x1",
  ylab = "x2",
  main = "Random Initial Centroids"
)


points(
  initial_centroids,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART IV
#
# ASSIGNMENT STEP
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Assign Observations to Closest Centroid
# ------------------------------------------------------------------------------

assign_clusters <- function(
    X,
    centroids
) {
  
  distances <- distance_matrix(
    X,
    centroids
  )
  
  
  max.col(
    -distances,
    ties.method = "first"
  )
}


# ------------------------------------------------------------------------------
# 9. First Assignment
# ------------------------------------------------------------------------------

first_assignment <- assign_clusters(
  X,
  initial_centroids
)


table(
  first_assignment
)


# ==============================================================================
# PART V
#
# UPDATE STEP
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Recalculate Centroids
# ------------------------------------------------------------------------------

update_centroids <- function(
    X,
    cluster_assignment,
    K,
    old_centroids = NULL
) {
  
  p <- ncol(
    X
  )
  
  
  centroids <- matrix(
    NA_real_,
    nrow = K,
    ncol = p
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    cluster_indices <- which(
      cluster_assignment == k
    )
    
    
    if (
      length(
        cluster_indices
      ) >
      0
    ) {
      
      centroids[k, ] <- colMeans(
        X[
          cluster_indices,
          ,
          drop = FALSE
        ]
      )
      
    } else {
      
      # If a cluster becomes empty, retain the previous centroid if available.
      #
      # A more sophisticated implementation could reinitialize the centroid.
      
      if (
        !is.null(
          old_centroids
        )
      ) {
        
        centroids[k, ] <-
          old_centroids[k, ]
        
      } else {
        
        centroids[k, ] <-
          X[
            sample(
              seq_len(
                nrow(X)
              ),
              1
            ),
          ]
      }
    }
  }
  
  
  colnames(
    centroids
  ) <- colnames(X)
  
  
  centroids
}


# ------------------------------------------------------------------------------
# 11. First Update
# ------------------------------------------------------------------------------

first_updated_centroids <- update_centroids(
  X,
  first_assignment,
  K = 3,
  old_centroids =
    initial_centroids
)


first_updated_centroids


# ==============================================================================
# PART VI
#
# K-MEANS OBJECTIVE FUNCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Within-Cluster Sum of Squares
# ------------------------------------------------------------------------------

within_cluster_sum_squares <- function(
    X,
    cluster_assignment,
    centroids
) {
  
  total <- 0
  
  
  for (
    k in seq_len(
      nrow(
        centroids
      )
    )
  ) {
    
    indices <- which(
      cluster_assignment == k
    )
    
    
    if (
      length(indices) >
      0
    ) {
      
      differences <- sweep(
        X[
          indices,
          ,
          drop = FALSE
        ],
        2,
        centroids[k, ],
        "-"
      )
      
      
      total <- total +
        sum(
          differences^2
        )
    }
  }
  
  
  total
}


# ------------------------------------------------------------------------------
# 13. Initial Objective
# ------------------------------------------------------------------------------

initial_WCSS <- within_cluster_sum_squares(
  X,
  first_assignment,
  initial_centroids
)


updated_WCSS <- within_cluster_sum_squares(
  X,
  first_assignment,
  first_updated_centroids
)


initial_WCSS


updated_WCSS


# Updating the centroid to the cluster mean cannot increase WCSS for a fixed
# cluster assignment.


# ==============================================================================
# PART VII
#
# MANUAL LLOYD'S ALGORITHM
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Fit K-Means
# ------------------------------------------------------------------------------

kmeans_manual <- function(
    X,
    K,
    max_iterations = 100,
    tolerance = 1e-8,
    initial_centroids = NULL,
    seed = NULL,
    verbose = FALSE
) {
  
  X <- as.matrix(
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
      initial_centroids
    )
  ) {
    
    centroids <- initialize_centroids(
      X,
      K
    )
    
  } else {
    
    centroids <- as.matrix(
      initial_centroids
    )
  }
  
  
  objective_history <- numeric(
    max_iterations
  )
  
  
  centroid_history <- vector(
    "list",
    max_iterations + 1
  )
  
  
  centroid_history[[1]] <-
    centroids
  
  
  old_assignment <- rep(
    NA_integer_,
    nrow(X)
  )
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    # ------------------------------------------------------------------------
    # Assignment step
    # ------------------------------------------------------------------------
    
    cluster_assignment <- assign_clusters(
      X,
      centroids
    )
    
    
    # ------------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------------
    
    new_centroids <- update_centroids(
      X,
      cluster_assignment,
      K,
      old_centroids =
        centroids
    )
    
    
    # ------------------------------------------------------------------------
    # Reassign after centroid update
    # ------------------------------------------------------------------------
    
    new_assignment <- assign_clusters(
      X,
      new_centroids
    )
    
    
    # ------------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------------
    
    objective_history[
      iteration
    ] <- within_cluster_sum_squares(
      X,
      new_assignment,
      new_centroids
    )
    
    
    centroid_history[[
      iteration + 1
    ]] <- new_centroids
    
    
    # ------------------------------------------------------------------------
    # Centroid movement
    # ------------------------------------------------------------------------
    
    centroid_change <- sqrt(
      sum(
        (
          new_centroids -
            centroids
        )^2
      )
    )
    
    
    assignment_unchanged <- identical(
      as.integer(
        new_assignment
      ),
      as.integer(
        old_assignment
      )
    )
    
    
    if (
      verbose
    ) {
      
      cat(
        "Iteration:",
        iteration,
        "| WCSS:",
        round(
          objective_history[
            iteration
          ],
          4
        ),
        "| Centroid Change:",
        round(
          centroid_change,
          6
        ),
        "\n"
      )
    }
    
    
    centroids <- new_centroids
    
    old_assignment <- new_assignment
    
    
    if (
      centroid_change <
      tolerance ||
      assignment_unchanged
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
  
  
  centroid_history <- centroid_history[
    seq_len(
      iterations_used + 1
    )
  ]
  
  
  final_assignment <- assign_clusters(
    X,
    centroids
  )
  
  
  final_WCSS <- within_cluster_sum_squares(
    X,
    final_assignment,
    centroids
  )
  
  
  return(
    list(
      cluster =
        final_assignment,
      centers =
        centroids,
      total_withinss =
        final_WCSS,
      objective_history =
        objective_history,
      centroid_history =
        centroid_history,
      iterations =
        iterations_used,
      converged =
        converged
    )
  )
}


# ==============================================================================
# PART VIII
#
# FIT ONE K-MEANS MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Run K-Means
# ------------------------------------------------------------------------------

K <- 3


KMeans_model <- kmeans_manual(
  X,
  K = K,
  max_iterations = 100,
  tolerance = 1e-8,
  seed = 123,
  verbose = TRUE
)


KMeans_model$centers


KMeans_model$total_withinss


KMeans_model$iterations


KMeans_model$converged


# ==============================================================================
# PART IX
#
# VISUALIZE CLUSTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Estimated Clusters
# ------------------------------------------------------------------------------

plot(
  X,
  pch =
    KMeans_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "Manual K-Means Clustering"
)


points(
  KMeans_model$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART X
#
# CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Objective Function by Iteration
# ------------------------------------------------------------------------------

plot(
  seq_along(
    KMeans_model$objective_history
  ),
  KMeans_model$objective_history,
  type = "b",
  pch = 19,
  xlab = "Iteration",
  ylab = "Within-Cluster Sum of Squares",
  main = "K-Means Convergence"
)


# ==============================================================================
# PART XI
#
# CENTROID MOVEMENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Plot Centroid Paths
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex = 0.4,
  xlab = "x1",
  ylab = "x2",
  main = "K-Means Centroid Paths"
)


for (
  k in seq_len(K)
) {
  
  centroid_path <- do.call(
    rbind,
    lapply(
      KMeans_model$centroid_history,
      function(centers) {
        
        centers[
          k,
          ,
          drop = FALSE
        ]
      }
    )
  )
  
  
  lines(
    centroid_path[
      ,
      1
    ],
    centroid_path[
      ,
      2
    ],
    type = "b",
    pch = 8,
    lwd = 2
  )
}


# ==============================================================================
# PART XII
#
# CLUSTER SIZES AND WITHIN-CLUSTER VARIATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Cluster Sizes
# ------------------------------------------------------------------------------

cluster_sizes <- table(
  KMeans_model$cluster
)


cluster_sizes


# ------------------------------------------------------------------------------
# 20. WCSS by Cluster
# ------------------------------------------------------------------------------

cluster_WCSS <- numeric(
  K
)


for (
  k in seq_len(K)
) {
  
  indices <- which(
    KMeans_model$cluster ==
      k
  )
  
  
  differences <- sweep(
    X[
      indices,
      ,
      drop = FALSE
    ],
    2,
    KMeans_model$centers[k, ],
    "-"
  )
  
  
  cluster_WCSS[k] <- sum(
    differences^2
  )
}


cluster_WCSS


sum(
  cluster_WCSS
)


KMeans_model$total_withinss


# ==============================================================================
# PART XIII
#
# CLUSTER LABELS ARE ARBITRARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Compare Estimated and True Labels
# ------------------------------------------------------------------------------

table(
  True =
    true_cluster,
  Estimated =
    KMeans_model$cluster
)


# IMPORTANT:
#
# Cluster labels have no intrinsic meaning.
#
# Estimated cluster 1 does NOT need to correspond to true cluster 1.
#
# A permutation such as:
#
#       true 1 -> estimated 3
#       true 2 -> estimated 1
#       true 3 -> estimated 2
#
# can still represent perfect clustering.


# ==============================================================================
# PART XIV
#
# PAIRWISE CLUSTER AGREEMENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Rand Index
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


# ------------------------------------------------------------------------------
# 23. Rand Index for Simulated Data
# ------------------------------------------------------------------------------

rand_index(
  true_cluster,
  KMeans_model$cluster
)


# ==============================================================================
# PART XV
#
# LOCAL MINIMA
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Multiple Random Initializations
# ------------------------------------------------------------------------------

number_starts <- 50


start_results <- data.frame(
  Start =
    seq_len(
      number_starts
    ),
  WCSS =
    NA_real_,
  Iterations =
    NA_integer_
)


start_models <- vector(
  "list",
  number_starts
)


for (
  s in seq_len(
    number_starts
  )
) {
  
  model_s <- kmeans_manual(
    X,
    K = 3,
    seed =
      1000 + s
  )
  
  
  start_models[[s]] <-
    model_s
  
  
  start_results$WCSS[s] <-
    model_s$total_withinss
  
  
  start_results$Iterations[s] <-
    model_s$iterations
}


head(
  start_results
)


# ------------------------------------------------------------------------------
# 25. Distribution of Solutions
# ------------------------------------------------------------------------------

hist(
  start_results$WCSS,
  breaks = 20,
  xlab = "Final WCSS",
  main = "Sensitivity to Random Initialization"
)


# ------------------------------------------------------------------------------
# 26. Best Random Start
# ------------------------------------------------------------------------------

best_start_index <- which.min(
  start_results$WCSS
)


best_KMeans_model <- start_models[[
  best_start_index
]]


best_KMeans_model$total_withinss


# ------------------------------------------------------------------------------
# 27. Best Solution
# ------------------------------------------------------------------------------

plot(
  X,
  pch =
    best_KMeans_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "Best of Multiple K-Means Starts"
)


points(
  best_KMeans_model$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XVI
#
# MULTI-START K-MEANS WRAPPER
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Fit K-Means with nstart
# ------------------------------------------------------------------------------

kmeans_multistart <- function(
    X,
    K,
    nstart = 20,
    max_iterations = 100,
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
    
    models[[s]] <- kmeans_manual(
      X,
      K = K,
      max_iterations =
        max_iterations,
      seed =
        start_seeds[s]
    )
    
    
    objectives[s] <-
      models[[s]]$total_withinss
  }
  
  
  best <- which.min(
    objectives
  )
  
  
  model <- models[[best]]
  
  
  model$all_objectives <-
    objectives
  
  
  model$best_start <-
    best
  
  
  model$nstart <-
    nstart
  
  
  model
}


# ------------------------------------------------------------------------------
# 29. Fit Robust Multi-Start Model
# ------------------------------------------------------------------------------

final_KMeans <- kmeans_multistart(
  X,
  K = 3,
  nstart = 50,
  seed = 123
)


final_KMeans$total_withinss


final_KMeans$centers


# ==============================================================================
# PART XVII
#
# CHOOSING K: ELBOW METHOD
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Fit Different Numbers of Clusters
# ------------------------------------------------------------------------------

K_values <- 1:10


elbow_WCSS <- numeric(
  length(
    K_values
  )
)


for (
  i in seq_along(
    K_values
  )
) {
  
  model_i <- kmeans_multistart(
    X,
    K =
      K_values[i],
    nstart = 20,
    seed =
      200 + i
  )
  
  
  elbow_WCSS[i] <-
    model_i$total_withinss
}


elbow_results <- data.frame(
  K =
    K_values,
  WCSS =
    elbow_WCSS
)


elbow_results


# ------------------------------------------------------------------------------
# 31. Elbow Plot
# ------------------------------------------------------------------------------

plot(
  K_values,
  elbow_WCSS,
  type = "b",
  pch = 19,
  xlab = "Number of Clusters K",
  ylab = "Within-Cluster Sum of Squares",
  main = "K-Means Elbow Method"
)


# WCSS must decrease as K increases.
#
# We look for a point after which adding clusters produces relatively small
# additional improvement.


# ==============================================================================
# PART XVIII
#
# TOTAL, WITHIN, AND BETWEEN SUM OF SQUARES
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Total Sum of Squares
# ------------------------------------------------------------------------------

global_centroid <- colMeans(
  X
)


TSS <- sum(
  sweep(
    X,
    2,
    global_centroid,
    "-"
  )^2
)


# ------------------------------------------------------------------------------
# 33. Between-Cluster Sum of Squares
# ------------------------------------------------------------------------------

WCSS <- final_KMeans$total_withinss


BCSS <- TSS -
  WCSS


TSS


WCSS


BCSS


WCSS + BCSS


# Therefore:
#
#       TSS = WCSS + BCSS
#
#
# Minimizing WCSS is equivalent to maximizing BCSS for fixed data.


# ==============================================================================
# PART XIX
#
# PROPORTION OF VARIATION EXPLAINED
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Between-Cluster Fraction
# ------------------------------------------------------------------------------

between_fraction <- BCSS /
  TSS


between_fraction


# ------------------------------------------------------------------------------
# 35. Fraction Across K
# ------------------------------------------------------------------------------

between_fraction_by_K <- 1 -
  elbow_WCSS /
  TSS


plot(
  K_values,
  between_fraction_by_K,
  type = "b",
  pch = 19,
  xlab = "K",
  ylab = "Between-Cluster / Total Variation",
  main = "Variation Explained by Clustering"
)


# ==============================================================================
# PART XX
#
# MANUAL SILHOUETTE ANALYSIS
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Pairwise Euclidean Distance Matrix
# ------------------------------------------------------------------------------

pairwise_distance_matrix <- function(
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
      
      distance_ij <- sqrt(
        sum(
          (
            X[i, ] -
              X[j, ]
          )^2
        )
      )
      
      
      D[i, j] <- distance_ij
      
      D[j, i] <- distance_ij
    }
  }
  
  
  D
}


# ------------------------------------------------------------------------------
# 37. Silhouette Values
# ------------------------------------------------------------------------------

silhouette_manual <- function(
    X,
    cluster
) {
  
  D <- pairwise_distance_matrix(
    X
  )
  
  
  n <- nrow(
    X
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
    
    
    # ------------------------------------------------------------------------
    # a(i): average distance to own cluster
    # ------------------------------------------------------------------------
    
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
    
    
    # ------------------------------------------------------------------------
    # b(i): smallest average distance to another cluster
    # ------------------------------------------------------------------------
    
    other_clusters <- cluster_values[
      cluster_values !=
        own_cluster
    ]
    
    
    average_other_distance <- sapply(
      other_clusters,
      function(k) {
        
        indices_k <- which(
          cluster == k
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
      average_other_distance
    )
    
    
    # ------------------------------------------------------------------------
    # Silhouette
    # ------------------------------------------------------------------------
    
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
# 38. Silhouette for K = 3
# ------------------------------------------------------------------------------

silhouette_values <- silhouette_manual(
  X,
  final_KMeans$cluster
)


summary(
  silhouette_values
)


mean(
  silhouette_values
)


# ------------------------------------------------------------------------------
# 39. Plot Silhouette Distribution
# ------------------------------------------------------------------------------

hist(
  silhouette_values,
  breaks = 30,
  xlab = "Silhouette Value",
  main = "K-Means Silhouette Values"
)


abline(
  v = 0,
  lty = 2
)


# ==============================================================================
# PART XXI
#
# SILHOUETTE CHOICE OF K
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Average Silhouette for Different K
# ------------------------------------------------------------------------------

silhouette_K <- 2:8


average_silhouette <- numeric(
  length(
    silhouette_K
  )
)


for (
  i in seq_along(
    silhouette_K
  )
) {
  
  model_i <- kmeans_multistart(
    X,
    K =
      silhouette_K[i],
    nstart = 20,
    seed =
      500 + i
  )
  
  
  silhouette_i <- silhouette_manual(
    X,
    model_i$cluster
  )
  
  
  average_silhouette[i] <- mean(
    silhouette_i
  )
}


silhouette_results <- data.frame(
  K =
    silhouette_K,
  Average_Silhouette =
    average_silhouette
)


silhouette_results


# ------------------------------------------------------------------------------
# 41. Plot Average Silhouette
# ------------------------------------------------------------------------------

plot(
  silhouette_K,
  average_silhouette,
  type = "b",
  pch = 19,
  xlab = "Number of Clusters K",
  ylab = "Average Silhouette",
  main = "Choosing K by Silhouette"
)


best_silhouette_K <- silhouette_K[
  which.max(
    average_silhouette
  )
]


best_silhouette_K


# ==============================================================================
# PART XXII
#
# PREDICT CLUSTERS FOR NEW OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Prediction Function
# ------------------------------------------------------------------------------

predict_kmeans <- function(
    model,
    X_new
) {
  
  assign_clusters(
    as.matrix(
      X_new
    ),
    model$centers
  )
}


# ------------------------------------------------------------------------------
# 43. New Observations
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


predict_kmeans(
  final_KMeans,
  new_points
)


# ==============================================================================
# PART XXIII
#
# K-MEANS DECISION REGIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Grid
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


grid_cluster <- predict_kmeans(
  final_KMeans,
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
# 45. Plot Voronoi-Like Regions
# ------------------------------------------------------------------------------

image(
  x1_grid,
  x2_grid,
  grid_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "K-Means Nearest-Centroid Regions"
)


points(
  X,
  pch =
    final_KMeans$cluster,
  cex = 0.5
)


points(
  final_KMeans$centers,
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

set.seed(777)


n_scale <- 300


scale_group <- rep(
  1:3,
  each = 100
)


x_small <- c(
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


x_large_noise <- rnorm(
  n_scale,
  mean = 0,
  sd = 100
)


X_scale <- cbind(
  Signal =
    x_small,
  Large_Noise =
    x_large_noise
)


# ------------------------------------------------------------------------------
# 47. K-Means Without Scaling
# ------------------------------------------------------------------------------

unscaled_model <- kmeans_multistart(
  X_scale,
  K = 3,
  nstart = 30,
  seed = 123
)


unscaled_rand <- rand_index(
  scale_group,
  unscaled_model$cluster
)


# ------------------------------------------------------------------------------
# 48. Standardize Variables
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
# 49. K-Means After Scaling
# ------------------------------------------------------------------------------

scaled_model <- kmeans_multistart(
  X_scale_standardized,
  K = 3,
  nstart = 30,
  seed = 123
)


scaled_rand <- rand_index(
  scale_group,
  scaled_model$cluster
)


data.frame(
  Method = c(
    "Unscaled K-Means",
    "Standardized K-Means"
  ),
  Rand_Index = c(
    unscaled_rand,
    scaled_rand
  )
)


# ==============================================================================
# PART XXV
#
# SENSITIVITY TO OUTLIERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Add Extreme Outliers
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
# 51. Fit K-Means with Outliers
# ------------------------------------------------------------------------------

outlier_model <- kmeans_multistart(
  X_outlier,
  K = 3,
  nstart = 50,
  seed = 123
)


# ------------------------------------------------------------------------------
# 52. Compare Centroids
# ------------------------------------------------------------------------------

final_KMeans$centers


outlier_model$centers


# ------------------------------------------------------------------------------
# 53. Plot Outlier Result
# ------------------------------------------------------------------------------

plot(
  X_outlier,
  pch =
    outlier_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "K-Means Sensitivity to Outliers"
)


points(
  outlier_model$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


# The squared-distance objective makes K-means sensitive to extreme values.


# ==============================================================================
# PART XXVI
#
# NON-SPHERICAL CLUSTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Generate Elongated Clusters
# ------------------------------------------------------------------------------

set.seed(888)


n_shape <- 150


elongated_1 <- cbind(
  rnorm(
    n_shape,
    mean = -2,
    sd = 2
  ),
  rnorm(
    n_shape,
    mean = 0,
    sd = 0.25
  )
)


elongated_2 <- cbind(
  rnorm(
    n_shape,
    mean = 2,
    sd = 2
  ),
  rnorm(
    n_shape,
    mean = 2,
    sd = 0.25
  )
)


X_elongated <- rbind(
  elongated_1,
  elongated_2
)


true_elongated <- rep(
  1:2,
  each = n_shape
)


elongated_model <- kmeans_multistart(
  X_elongated,
  K = 2,
  nstart = 50,
  seed = 123
)


plot(
  X_elongated,
  pch =
    elongated_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "K-Means with Elongated Clusters"
)


points(
  elongated_model$centers,
  pch = 8,
  cex = 2,
  lwd = 3
)


rand_index(
  true_elongated,
  elongated_model$cluster
)


# K-means works best when clusters are reasonably compact around their means.
#
# It is less natural for elongated, curved, unequal-density, or strongly
# unequal-variance clusters.


# ==============================================================================
# PART XXVII
#
# K-MEANS AFTER PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Generate Higher-Dimensional Data
# ------------------------------------------------------------------------------

set.seed(999)


n_hd <- 450

p_hd <- 20


hd_cluster <- rep(
  1:3,
  each = 150
)


X_hd <- matrix(
  rnorm(
    n_hd *
      p_hd
  ),
  nrow =
    n_hd,
  ncol =
    p_hd
)


X_hd[
  hd_cluster == 1,
  1:3
] <- X_hd[
  hd_cluster == 1,
  1:3
] - 2


X_hd[
  hd_cluster == 2,
  1:3
] <- X_hd[
  hd_cluster == 2,
  1:3
] + 2


X_hd[
  hd_cluster == 3,
  4:6
] <- X_hd[
  hd_cluster == 3,
  4:6
] + 2


# ------------------------------------------------------------------------------
# 56. Standardize
# ------------------------------------------------------------------------------

X_hd_scaled <- scale(
  X_hd
)


# ------------------------------------------------------------------------------
# 57. Manual PCA with SVD
# ------------------------------------------------------------------------------

X_hd_centered <- sweep(
  X_hd_scaled,
  2,
  colMeans(
    X_hd_scaled
  ),
  "-"
)


PCA_svd <- svd(
  X_hd_centered
)


PC_scores <- X_hd_centered %*%
  PCA_svd$v


# ------------------------------------------------------------------------------
# 58. K-Means on First PCs
# ------------------------------------------------------------------------------

X_PC <- PC_scores[
  ,
  1:5,
  drop = FALSE
]


PCA_KMeans <- kmeans_multistart(
  X_PC,
  K = 3,
  nstart = 30,
  seed = 123
)


rand_index(
  hd_cluster,
  PCA_KMeans$cluster
)


# ------------------------------------------------------------------------------
# 59. Visualize First Two PCs
# ------------------------------------------------------------------------------

plot(
  PC_scores[, 1],
  PC_scores[, 2],
  pch =
    PCA_KMeans$cluster,
  xlab = "PC1",
  ylab = "PC2",
  main = "K-Means after PCA"
)


# ==============================================================================
# PART XXVIII
#
# VERIFY AGAINST BASE R kmeans()
# ==============================================================================


# ------------------------------------------------------------------------------
# 60. Built-In K-Means
# ------------------------------------------------------------------------------

set.seed(123)


built_in_model <- kmeans(
  X,
  centers = 3,
  nstart = 50,
  iter.max = 100
)


built_in_model$centers


built_in_model$tot.withinss


# ------------------------------------------------------------------------------
# 61. Compare Objective Values
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Manual K-Means",
    "Base R kmeans()"
  ),
  WCSS = c(
    final_KMeans$total_withinss,
    built_in_model$tot.withinss
  )
)


# Because the implementations and random starts may differ, the solutions do
# not have to have identical cluster labels or exactly identical WCSS.
#
# With well-separated data and enough starts, they should generally find
# equivalent high-quality solutions.


# ==============================================================================
# PART XXIX
#
# VERIFY CENTROID PROPERTY
# ==============================================================================


# ------------------------------------------------------------------------------
# 62. Cluster Mean Must Equal Centroid
# ------------------------------------------------------------------------------

manual_cluster_means <- matrix(
  NA_real_,
  nrow = K,
  ncol = ncol(X)
)


for (
  k in seq_len(K)
) {
  
  manual_cluster_means[k, ] <- colMeans(
    X[
      final_KMeans$cluster ==
        k,
      ,
      drop = FALSE
    ]
  )
}


manual_cluster_means


final_KMeans$centers


max(
  abs(
    manual_cluster_means -
      final_KMeans$centers
  )
)


# ==============================================================================
# PART XXX
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 63. Output
# ------------------------------------------------------------------------------

cat(
  "K-Means Clustering Summary\n"
)


cat(
  "--------------------------\n"
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
  "Chosen K:",
  K,
  "\n"
)


cat(
  "Final WCSS:",
  round(
    final_KMeans$total_withinss,
    4
  ),
  "\n"
)


cat(
  "Total sum of squares:",
  round(
    TSS,
    4
  ),
  "\n"
)


cat(
  "Between-cluster sum of squares:",
  round(
    BCSS,
    4
  ),
  "\n"
)


cat(
  "Between-cluster fraction:",
  round(
    between_fraction,
    4
  ),
  "\n"
)


cat(
  "Average silhouette:",
  round(
    mean(
      silhouette_values
    ),
    4
  ),
  "\n"
)


cat(
  "Best silhouette K:",
  best_silhouette_K,
  "\n"
)


cat(
  "Rand index against simulated truth:",
  round(
    rand_index(
      true_cluster,
      final_KMeans$cluster
    ),
    4
  ),
  "\n"
)