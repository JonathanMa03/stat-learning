# ==============================================================================
# Gaussian Mixture Models
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Model-based clustering
#   - Finite mixture models
#   - Multivariate Gaussian distributions
#   - Latent cluster membership
#   - Soft clustering
#   - Posterior responsibilities
#   - Expectation-Maximization (EM)
#   - Mixing proportions
#   - Component means
#   - Component covariance matrices
#   - Log-likelihood
#   - Multiple random starts
#   - AIC / BIC
#   - Classification uncertainty
#   - Relationship to K-means
#
#
# Gaussian mixture density:
#
#             K
#       p(x) = sum pi_k N(x | mu_k, Sigma_k)
#            k=1
#
#
# where:
#
#       pi_k >= 0
#
#       sum pi_k = 1
#
#
# Latent cluster:
#
#       Z_i in {1, ..., K}
#
#
# Responsibilities:
#
#                       pi_k N(x_i | mu_k, Sigma_k)
#       gamma_ik = ---------------------------------------
#                   sum_j pi_j N(x_i | mu_j, Sigma_j)
#
#
# EM alternates:
#
#       E-step:
#           calculate gamma_ik
#
#       M-step:
#           update pi_k, mu_k, Sigma_k
#
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE GAUSSIAN MIXTURE DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Manual Multivariate Normal Generator
# ------------------------------------------------------------------------------

rmvnorm_manual <- function(
    n,
    mean,
    covariance
) {
  
  mean <- as.numeric(
    mean
  )
  
  
  p <- length(
    mean
  )
  
  
  eigen_decomposition <- eigen(
    covariance,
    symmetric = TRUE
  )
  
  
  eigenvalues <- pmax(
    eigen_decomposition$values,
    0
  )
  
  
  square_root_covariance <-
    eigen_decomposition$vectors %*%
    diag(
      sqrt(
        eigenvalues
      ),
      nrow = p
    ) %*%
    t(
      eigen_decomposition$vectors
    )
  
  
  Z <- matrix(
    rnorm(
      n * p
    ),
    nrow = n,
    ncol = p
  )
  
  
  X <- Z %*%
    square_root_covariance
  
  
  X <- sweep(
    X,
    2,
    mean,
    "+"
  )
  
  
  X
}


# ------------------------------------------------------------------------------
# 2. Define Three Gaussian Components
# ------------------------------------------------------------------------------

set.seed(123)


n1 <- 150
n2 <- 120
n3 <- 130


mu1 <- c(
  -3,
  0
)


mu2 <- c(
  3,
  0
)


mu3 <- c(
  0,
  4
)


Sigma1 <- matrix(
  c(
    1.0,  0.7,
    0.7,  1.2
  ),
  nrow = 2,
  byrow = TRUE
)


Sigma2 <- matrix(
  c(
    1.4, -0.6,
    -0.6,  0.8
  ),
  nrow = 2,
  byrow = TRUE
)


Sigma3 <- matrix(
  c(
    0.7, 0.0,
    0.0, 1.5
  ),
  nrow = 2,
  byrow = TRUE
)


# ------------------------------------------------------------------------------
# 3. Generate Data
# ------------------------------------------------------------------------------

X1 <- rmvnorm_manual(
  n1,
  mu1,
  Sigma1
)


X2 <- rmvnorm_manual(
  n2,
  mu2,
  Sigma2
)


X3 <- rmvnorm_manual(
  n3,
  mu3,
  Sigma3
)


X <- rbind(
  X1,
  X2,
  X3
)


colnames(
  X
) <- c(
  "x1",
  "x2"
)


true_cluster <- c(
  rep(
    1,
    n1
  ),
  rep(
    2,
    n2
  ),
  rep(
    3,
    n3
  )
)


n <- nrow(
  X
)


p <- ncol(
  X
)


# ------------------------------------------------------------------------------
# 4. Plot Unlabeled Data
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex = 0.6,
  xlab = "x1",
  ylab = "x2",
  main = "Gaussian Mixture Data"
)


# ------------------------------------------------------------------------------
# 5. Plot True Components
# ------------------------------------------------------------------------------

plot(
  X,
  pch = true_cluster,
  xlab = "x1",
  ylab = "x2",
  main = "True Gaussian Components"
)


# ==============================================================================
# PART II
#
# MULTIVARIATE GAUSSIAN DENSITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Manual Multivariate Gaussian Log-Density
# ------------------------------------------------------------------------------

dmvnorm_log_manual <- function(
    X,
    mean,
    covariance,
    regularization = 1e-8
) {
  
  X <- as.matrix(
    X
  )
  
  
  mean <- as.numeric(
    mean
  )
  
  
  p <- ncol(
    X
  )
  
  
  covariance <- covariance +
    diag(
      regularization,
      p
    )
  
  
  centered <- sweep(
    X,
    2,
    mean,
    "-"
  )
  
  
  covariance_chol <- tryCatch(
    chol(
      covariance
    ),
    error = function(e) {
      
      chol(
        covariance +
          diag(
            1e-6,
            p
          )
      )
    }
  )
  
  
  log_determinant <- 2 *
    sum(
      log(
        diag(
          covariance_chol
        )
      )
    )
  
  
  transformed <- forwardsolve(
    t(
      covariance_chol
    ),
    t(
      centered
    )
  )
  
  
  mahalanobis_squared <- colSums(
    transformed^2
  )
  
  
  log_density <- -0.5 *
    (
      p *
        log(
          2 * pi
        ) +
        log_determinant +
        mahalanobis_squared
    )
  
  
  log_density
}


# ------------------------------------------------------------------------------
# 7. Density Function
# ------------------------------------------------------------------------------

dmvnorm_manual <- function(
    X,
    mean,
    covariance
) {
  
  exp(
    dmvnorm_log_manual(
      X,
      mean,
      covariance
    )
  )
}


# ------------------------------------------------------------------------------
# 8. Evaluate First Component Density
# ------------------------------------------------------------------------------

head(
  dmvnorm_manual(
    X,
    mu1,
    Sigma1
  )
)


# ==============================================================================
# PART III
#
# LOG-SUM-EXP
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Numerically Stable Log-Sum-Exp
# ------------------------------------------------------------------------------

log_sum_exp <- function(
    values
) {
  
  maximum_value <- max(
    values
  )
  
  
  maximum_value +
    log(
      sum(
        exp(
          values -
            maximum_value
        )
      )
    )
}


# ==============================================================================
# PART IV
#
# INITIALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Initialize with K-Means
# ------------------------------------------------------------------------------

initialize_gmm <- function(
    X,
    K,
    seed = NULL
) {
  
  if (
    !is.null(seed)
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  n <- nrow(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  kmeans_initial <- kmeans(
    X,
    centers = K,
    nstart = 1
  )
  
  
  cluster <- kmeans_initial$cluster
  
  
  means <- matrix(
    NA_real_,
    nrow = K,
    ncol = p
  )
  
  
  covariances <- vector(
    "list",
    K
  )
  
  
  mixing_proportions <- numeric(
    K
  )
  
  
  global_covariance <- cov(
    X
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    indices <- which(
      cluster == k
    )
    
    
    means[k, ] <- colMeans(
      X[
        indices,
        ,
        drop = FALSE
      ]
    )
    
    
    mixing_proportions[k] <-
      length(
        indices
      ) /
      n
    
    
    if (
      length(
        indices
      ) >
      p
    ) {
      
      covariance_k <- cov(
        X[
          indices,
          ,
          drop = FALSE
        ]
      )
      
    } else {
      
      covariance_k <- global_covariance
    }
    
    
    covariances[[k]] <-
      covariance_k +
      diag(
        1e-6,
        p
      )
  }
  
  
  list(
    mixing_proportions =
      mixing_proportions,
    means =
      means,
    covariances =
      covariances
  )
}


# ------------------------------------------------------------------------------
# 11. Example Initialization
# ------------------------------------------------------------------------------

initial_parameters <- initialize_gmm(
  X,
  K = 3,
  seed = 100
)


initial_parameters$mixing_proportions


initial_parameters$means


initial_parameters$covariances


# ==============================================================================
# PART V
#
# E-STEP
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Calculate Responsibilities
# ------------------------------------------------------------------------------

gmm_expectation_step <- function(
    X,
    mixing_proportions,
    means,
    covariances
) {
  
  n <- nrow(
    X
  )
  
  
  K <- length(
    mixing_proportions
  )
  
  
  log_weighted_density <- matrix(
    NA_real_,
    nrow = n,
    ncol = K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    log_weighted_density[, k] <-
      log(
        pmax(
          mixing_proportions[k],
          1e-300
        )
      ) +
      dmvnorm_log_manual(
        X,
        means[k, ],
        covariances[[k]]
      )
  }
  
  
  log_normalizer <- apply(
    log_weighted_density,
    1,
    log_sum_exp
  )
  
  
  responsibilities <- exp(
    sweep(
      log_weighted_density,
      1,
      log_normalizer,
      "-"
    )
  )
  
  
  log_likelihood <- sum(
    log_normalizer
  )
  
  
  list(
    responsibilities =
      responsibilities,
    log_likelihood =
      log_likelihood
  )
}


# ------------------------------------------------------------------------------
# 13. First E-Step
# ------------------------------------------------------------------------------

first_E_step <- gmm_expectation_step(
  X,
  initial_parameters$mixing_proportions,
  initial_parameters$means,
  initial_parameters$covariances
)


head(
  first_E_step$responsibilities
)


rowSums(
  first_E_step$responsibilities[
    1:10,
    ,
    drop = FALSE
  ]
)


first_E_step$log_likelihood


# ==============================================================================
# PART VI
#
# M-STEP
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Update Parameters
# ------------------------------------------------------------------------------

gmm_maximization_step <- function(
    X,
    responsibilities,
    covariance_regularization = 1e-6
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
  
  
  K <- ncol(
    responsibilities
  )
  
  
  effective_counts <- colSums(
    responsibilities
  )
  
  
  mixing_proportions <- effective_counts /
    n
  
  
  means <- matrix(
    NA_real_,
    nrow = K,
    ncol = p
  )
  
  
  covariances <- vector(
    "list",
    K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    weights <- responsibilities[
      ,
      k
    ]
    
    
    N_k <- max(
      effective_counts[k],
      1e-12
    )
    
    
    means[k, ] <- colSums(
      X *
        weights
    ) /
      N_k
    
    
    centered <- sweep(
      X,
      2,
      means[k, ],
      "-"
    )
    
    
    weighted_centered <- centered *
      sqrt(
        weights
      )
    
    
    covariance_k <- crossprod(
      weighted_centered
    ) /
      N_k
    
    
    covariance_k <- covariance_k +
      diag(
        covariance_regularization,
        p
      )
    
    
    covariances[[k]] <-
      covariance_k
  }
  
  
  list(
    mixing_proportions =
      mixing_proportions,
    means =
      means,
    covariances =
      covariances,
    effective_counts =
      effective_counts
  )
}


# ------------------------------------------------------------------------------
# 15. First M-Step
# ------------------------------------------------------------------------------

first_M_step <- gmm_maximization_step(
  X,
  first_E_step$responsibilities
)


first_M_step$mixing_proportions


first_M_step$means


first_M_step$effective_counts


# ==============================================================================
# PART VII
#
# MANUAL EM ALGORITHM
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Fit Gaussian Mixture Model
# ------------------------------------------------------------------------------

fit_gmm <- function(
    X,
    K,
    max_iterations = 500,
    tolerance = 1e-6,
    covariance_regularization = 1e-6,
    seed = NULL,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  initialization <- initialize_gmm(
    X,
    K,
    seed =
      seed
  )
  
  
  mixing_proportions <-
    initialization$mixing_proportions
  
  
  means <-
    initialization$means
  
  
  covariances <-
    initialization$covariances
  
  
  log_likelihood_history <- numeric(
    max_iterations
  )
  
  
  converged <- FALSE
  
  
  previous_log_likelihood <- -Inf
  
  
  for (
    iteration in seq_len(
      max_iterations
    )
  ) {
    
    # ------------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------------
    
    E_step <- gmm_expectation_step(
      X,
      mixing_proportions,
      means,
      covariances
    )
    
    
    responsibilities <-
      E_step$responsibilities
    
    
    # ------------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------------
    
    M_step <- gmm_maximization_step(
      X,
      responsibilities,
      covariance_regularization =
        covariance_regularization
    )
    
    
    mixing_proportions <-
      M_step$mixing_proportions
    
    
    means <-
      M_step$means
    
    
    covariances <-
      M_step$covariances
    
    
    # ------------------------------------------------------------------------
    # Evaluate likelihood after parameter update
    # ------------------------------------------------------------------------
    
    updated_E_step <- gmm_expectation_step(
      X,
      mixing_proportions,
      means,
      covariances
    )
    
    
    current_log_likelihood <-
      updated_E_step$log_likelihood
    
    
    log_likelihood_history[
      iteration
    ] <- current_log_likelihood
    
    
    if (
      verbose
    ) {
      
      cat(
        "Iteration:",
        iteration,
        "| Log-Likelihood:",
        round(
          current_log_likelihood,
          6
        ),
        "| Change:",
        round(
          current_log_likelihood -
            previous_log_likelihood,
          8
        ),
        "\n"
      )
    }
    
    
    if (
      is.finite(
        previous_log_likelihood
      ) &&
      abs(
        current_log_likelihood -
        previous_log_likelihood
      ) <
      tolerance *
      (
        1 +
        abs(
          previous_log_likelihood
        )
      )
    ) {
      
      converged <- TRUE
      
      break
    }
    
    
    previous_log_likelihood <-
      current_log_likelihood
  }
  
  
  iterations_used <- iteration
  
  
  log_likelihood_history <-
    log_likelihood_history[
      seq_len(
        iterations_used
      )
    ]
  
  
  final_E_step <- gmm_expectation_step(
    X,
    mixing_proportions,
    means,
    covariances
  )
  
  
  responsibilities <-
    final_E_step$responsibilities
  
  
  cluster <- max.col(
    responsibilities,
    ties.method = "first"
  )
  
  
  list(
    K =
      K,
    mixing_proportions =
      mixing_proportions,
    means =
      means,
    covariances =
      covariances,
    responsibilities =
      responsibilities,
    cluster =
      cluster,
    log_likelihood =
      final_E_step$log_likelihood,
    log_likelihood_history =
      log_likelihood_history,
    iterations =
      iterations_used,
    converged =
      converged,
    covariance_regularization =
      covariance_regularization
  )
}


# ==============================================================================
# PART VIII
#
# FIT THREE-COMPONENT GMM
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Fit Model
# ------------------------------------------------------------------------------

GMM_model <- fit_gmm(
  X,
  K = 3,
  max_iterations = 500,
  tolerance = 1e-7,
  covariance_regularization = 1e-6,
  seed = 123,
  verbose = TRUE
)


# ------------------------------------------------------------------------------
# 18. Estimated Mixing Proportions
# ------------------------------------------------------------------------------

GMM_model$mixing_proportions


# ------------------------------------------------------------------------------
# 19. Estimated Means
# ------------------------------------------------------------------------------

GMM_model$means


# ------------------------------------------------------------------------------
# 20. Estimated Covariances
# ------------------------------------------------------------------------------

GMM_model$covariances


# ------------------------------------------------------------------------------
# 21. Final Log-Likelihood
# ------------------------------------------------------------------------------

GMM_model$log_likelihood


# ==============================================================================
# PART IX
#
# EM CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Plot Log-Likelihood
# ------------------------------------------------------------------------------

plot(
  seq_along(
    GMM_model$log_likelihood_history
  ),
  GMM_model$log_likelihood_history,
  type = "b",
  pch = 19,
  xlab = "EM Iteration",
  ylab = "Log-Likelihood",
  main = "Gaussian Mixture EM Convergence"
)


# ------------------------------------------------------------------------------
# 23. Verify Approximate Monotonicity
# ------------------------------------------------------------------------------

likelihood_changes <- diff(
  GMM_model$log_likelihood_history
)


min(
  likelihood_changes
)


# Tiny negative differences can occur because of numerical precision.


# ==============================================================================
# PART X
#
# HARD CLUSTER ASSIGNMENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Plot Estimated Components
# ------------------------------------------------------------------------------

plot(
  X,
  pch =
    GMM_model$cluster,
  xlab = "x1",
  ylab = "x2",
  main = "Gaussian Mixture Hard Assignments"
)


points(
  GMM_model$means,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XI
#
# SOFT CLUSTER ASSIGNMENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Inspect Responsibilities
# ------------------------------------------------------------------------------

head(
  GMM_model$responsibilities,
  20
)


# ------------------------------------------------------------------------------
# 26. Verify Responsibilities Sum to One
# ------------------------------------------------------------------------------

summary(
  rowSums(
    GMM_model$responsibilities
  )
)


# ------------------------------------------------------------------------------
# 27. Maximum Responsibility
# ------------------------------------------------------------------------------

maximum_responsibility <- apply(
  GMM_model$responsibilities,
  1,
  max
)


summary(
  maximum_responsibility
)


# A value close to 1 means the model is highly confident about component
# membership.
#
# A value near 1/K indicates substantial ambiguity.


# ==============================================================================
# PART XII
#
# CLASSIFICATION UNCERTAINTY
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Uncertainty
# ------------------------------------------------------------------------------

classification_uncertainty <- 1 -
  maximum_responsibility


summary(
  classification_uncertainty
)


# ------------------------------------------------------------------------------
# 29. Most Uncertain Observations
# ------------------------------------------------------------------------------

most_uncertain <- order(
  classification_uncertainty,
  decreasing = TRUE
)[
  1:10
]


data.frame(
  Observation =
    most_uncertain,
  X[
    most_uncertain,
    ,
    drop = FALSE
  ],
  Cluster =
    GMM_model$cluster[
      most_uncertain
    ],
  Maximum_Responsibility =
    maximum_responsibility[
      most_uncertain
    ],
  Uncertainty =
    classification_uncertainty[
      most_uncertain
    ]
)


# ------------------------------------------------------------------------------
# 30. Plot Uncertainty
# ------------------------------------------------------------------------------

plot(
  X,
  pch = 19,
  cex =
    0.5 +
    2 *
    classification_uncertainty,
  xlab = "x1",
  ylab = "x2",
  main = "Gaussian Mixture Classification Uncertainty"
)


points(
  GMM_model$means,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XIII
#
# ENTROPY OF CLUSTER ASSIGNMENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Posterior Entropy
# ------------------------------------------------------------------------------

posterior_entropy <- -rowSums(
  GMM_model$responsibilities *
    log(
      pmax(
        GMM_model$responsibilities,
        1e-15
      )
    )
)


summary(
  posterior_entropy
)


# ------------------------------------------------------------------------------
# 32. Normalized Entropy
# ------------------------------------------------------------------------------

normalized_entropy <- posterior_entropy /
  log(
    GMM_model$K
  )


summary(
  normalized_entropy
)


# 0 = essentially certain assignment
# 1 = responsibilities approximately equal across all components


# ==============================================================================
# PART XIV
#
# CLUSTER RECOVERY
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Contingency Table
# ------------------------------------------------------------------------------

table(
  True =
    true_cluster,
  Estimated =
    GMM_model$cluster
)


# Component labels are arbitrary.
#
# Estimated component 1 does not necessarily correspond to true component 1.


# ==============================================================================
# PART XV
#
# RAND INDEX
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Rand Index
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
  GMM_model$cluster
)


# ==============================================================================
# PART XVI
#
# MULTIPLE RANDOM STARTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Multi-Start GMM
# ------------------------------------------------------------------------------

fit_gmm_multistart <- function(
    X,
    K,
    nstart = 10,
    max_iterations = 500,
    tolerance = 1e-6,
    covariance_regularization = 1e-6,
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
  
  
  log_likelihoods <- numeric(
    nstart
  )
  
  
  for (
    s in seq_len(
      nstart
    )
  ) {
    
    models[[s]] <- fit_gmm(
      X,
      K = K,
      max_iterations =
        max_iterations,
      tolerance =
        tolerance,
      covariance_regularization =
        covariance_regularization,
      seed =
        start_seeds[s],
      verbose = FALSE
    )
    
    
    log_likelihoods[s] <-
      models[[s]]$log_likelihood
  }
  
  
  best_index <- which.max(
    log_likelihoods
  )
  
  
  best_model <- models[[
    best_index
  ]]
  
  
  best_model$all_log_likelihoods <-
    log_likelihoods
  
  
  best_model$best_start <-
    best_index
  
  
  best_model$nstart <-
    nstart
  
  
  best_model
}


# ------------------------------------------------------------------------------
# 36. Fit Multi-Start Model
# ------------------------------------------------------------------------------

final_GMM <- fit_gmm_multistart(
  X,
  K = 3,
  nstart = 20,
  max_iterations = 500,
  tolerance = 1e-7,
  covariance_regularization = 1e-6,
  seed = 123
)


final_GMM$log_likelihood


# ------------------------------------------------------------------------------
# 37. Distribution of Local Solutions
# ------------------------------------------------------------------------------

hist(
  final_GMM$all_log_likelihoods,
  breaks = 15,
  xlab = "Final Log-Likelihood",
  main = "GMM Multiple Initializations"
)


# ==============================================================================
# PART XVII
#
# PARAMETER COUNT
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Number of Free Parameters
# ------------------------------------------------------------------------------

gmm_parameter_count <- function(
    K,
    p
) {
  
  # Mixing proportions:
  #
  # K proportions subject to sum(pi) = 1
  #
  # => K - 1 free parameters
  
  
  mixing_parameters <- K -
    1
  
  
  # Means:
  #
  # p parameters for each component
  
  
  mean_parameters <- K *
    p
  
  
  # Full symmetric covariance:
  #
  # p(p+1)/2 parameters per component
  
  
  covariance_parameters <- K *
    p *
    (
      p +
        1
    ) /
    2
  
  
  mixing_parameters +
    mean_parameters +
    covariance_parameters
}


gmm_parameter_count(
  K = 3,
  p = 2
)


# ==============================================================================
# PART XVIII
#
# AIC AND BIC
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Information Criteria
# ------------------------------------------------------------------------------

gmm_information_criteria <- function(
    model,
    n,
    p
) {
  
  parameter_count <- gmm_parameter_count(
    model$K,
    p
  )
  
  
  log_likelihood <-
    model$log_likelihood
  
  
  AIC <- -2 *
    log_likelihood +
    2 *
    parameter_count
  
  
  BIC <- -2 *
    log_likelihood +
    log(n) *
    parameter_count
  
  
  list(
    parameters =
      parameter_count,
    AIC =
      AIC,
    BIC =
      BIC
  )
}


# ------------------------------------------------------------------------------
# 40. Current Model AIC/BIC
# ------------------------------------------------------------------------------

final_information <- gmm_information_criteria(
  final_GMM,
  n,
  p
)


final_information


# ==============================================================================
# PART XIX
#
# CHOOSING NUMBER OF COMPONENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Fit K = 1 Through 6
# ------------------------------------------------------------------------------

K_values <- 1:6


model_selection_results <- data.frame(
  K =
    K_values,
  Log_Likelihood =
    NA_real_,
  Parameters =
    NA_integer_,
  AIC =
    NA_real_,
  BIC =
    NA_real_
)


GMM_models <- vector(
  "list",
  length(
    K_values
  )
)


for (
  i in seq_along(
    K_values
  )
) {
  
  K_i <- K_values[i]
  
  
  model_i <- fit_gmm_multistart(
    X,
    K =
      K_i,
    nstart = 10,
    max_iterations = 500,
    tolerance = 1e-6,
    covariance_regularization = 1e-5,
    seed =
      100 + K_i
  )
  
  
  GMM_models[[i]] <-
    model_i
  
  
  information_i <- gmm_information_criteria(
    model_i,
    n,
    p
  )
  
  
  model_selection_results$Log_Likelihood[i] <-
    model_i$log_likelihood
  
  
  model_selection_results$Parameters[i] <-
    information_i$parameters
  
  
  model_selection_results$AIC[i] <-
    information_i$AIC
  
  
  model_selection_results$BIC[i] <-
    information_i$BIC
}


model_selection_results


# ------------------------------------------------------------------------------
# 42. Plot Log-Likelihood
# ------------------------------------------------------------------------------

plot(
  model_selection_results$K,
  model_selection_results$Log_Likelihood,
  type = "b",
  pch = 19,
  xlab = "Number of Components K",
  ylab = "Log-Likelihood",
  main = "GMM Log-Likelihood"
)


# ------------------------------------------------------------------------------
# 43. Plot AIC
# ------------------------------------------------------------------------------

plot(
  model_selection_results$K,
  model_selection_results$AIC,
  type = "b",
  pch = 19,
  xlab = "Number of Components K",
  ylab = "AIC",
  main = "Gaussian Mixture AIC"
)


# ------------------------------------------------------------------------------
# 44. Plot BIC
# ------------------------------------------------------------------------------

plot(
  model_selection_results$K,
  model_selection_results$BIC,
  type = "b",
  pch = 19,
  xlab = "Number of Components K",
  ylab = "BIC",
  main = "Gaussian Mixture BIC"
)


best_K_BIC <- model_selection_results$K[
  which.min(
    model_selection_results$BIC
  )
]


best_K_BIC


# ==============================================================================
# PART XX
#
# PREDICTION FOR NEW OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Predict Responsibilities
# ------------------------------------------------------------------------------

predict_gmm <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  E_step <- gmm_expectation_step(
    X_new,
    model$mixing_proportions,
    model$means,
    model$covariances
  )
  
  
  responsibilities <-
    E_step$responsibilities
  
  
  cluster <- max.col(
    responsibilities,
    ties.method = "first"
  )
  
  
  list(
    cluster =
      cluster,
    responsibilities =
      responsibilities,
    uncertainty =
      1 -
      apply(
        responsibilities,
        1,
        max
      )
  )
}


# ------------------------------------------------------------------------------
# 46. Example Predictions
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


new_predictions <- predict_gmm(
  final_GMM,
  new_points
)


new_predictions$cluster


new_predictions$responsibilities


new_predictions$uncertainty


# ==============================================================================
# PART XXI
#
# MIXTURE DENSITY FOR NEW OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Calculate Mixture Log-Density
# ------------------------------------------------------------------------------

gmm_log_density <- function(
    model,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  n_new <- nrow(
    X_new
  )
  
  
  K <- model$K
  
  
  log_component_density <- matrix(
    NA_real_,
    nrow = n_new,
    ncol = K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    log_component_density[, k] <-
      log(
        model$mixing_proportions[k]
      ) +
      dmvnorm_log_manual(
        X_new,
        model$means[k, ],
        model$covariances[[k]]
      )
  }
  
  
  apply(
    log_component_density,
    1,
    log_sum_exp
  )
}


# ------------------------------------------------------------------------------
# 48. Density of New Points
# ------------------------------------------------------------------------------

exp(
  gmm_log_density(
    final_GMM,
    new_points
  )
)


# ==============================================================================
# PART XXII
#
# DECISION REGIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 150


x1_grid <- seq(
  min(X[, 1]) - 2,
  max(X[, 1]) + 2,
  length.out =
    grid_size
)


x2_grid <- seq(
  min(X[, 2]) - 2,
  max(X[, 2]) + 2,
  length.out =
    grid_size
)


grid <- expand.grid(
  x1 =
    x1_grid,
  x2 =
    x2_grid
)


grid_prediction <- predict_gmm(
  final_GMM,
  grid
)


cluster_matrix <- matrix(
  grid_prediction$cluster,
  nrow =
    grid_size,
  ncol =
    grid_size
)


# ------------------------------------------------------------------------------
# 50. Plot GMM Decision Regions
# ------------------------------------------------------------------------------

image(
  x1_grid,
  x2_grid,
  cluster_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "Gaussian Mixture Decision Regions"
)


points(
  X,
  pch =
    final_GMM$cluster,
  cex = 0.4
)


points(
  final_GMM$means,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXIII
#
# UNCERTAINTY SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Grid Uncertainty
# ------------------------------------------------------------------------------

uncertainty_matrix <- matrix(
  grid_prediction$uncertainty,
  nrow =
    grid_size,
  ncol =
    grid_size
)


image(
  x1_grid,
  x2_grid,
  uncertainty_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "GMM Classification Uncertainty"
)


points(
  final_GMM$means,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXIV
#
# MIXTURE DENSITY SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 52. Evaluate Mixture Density on Grid
# ------------------------------------------------------------------------------

grid_log_density <- gmm_log_density(
  final_GMM,
  grid
)


density_matrix <- matrix(
  exp(
    grid_log_density
  ),
  nrow =
    grid_size,
  ncol =
    grid_size
)


# ------------------------------------------------------------------------------
# 53. Density Contours
# ------------------------------------------------------------------------------

contour(
  x1_grid,
  x2_grid,
  density_matrix,
  nlevels = 12,
  xlab = "x1",
  ylab = "x2",
  main = "Estimated Gaussian Mixture Density"
)


points(
  X,
  pch = 19,
  cex = 0.25
)


points(
  final_GMM$means,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXV
#
# COVARIANCE ELLIPSES
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Gaussian Covariance Ellipse
# ------------------------------------------------------------------------------

covariance_ellipse <- function(
    mean,
    covariance,
    probability = 0.95,
    number_points = 200
) {
  
  eigen_decomposition <- eigen(
    covariance,
    symmetric = TRUE
  )
  
  
  angle <- seq(
    0,
    2 * pi,
    length.out =
      number_points
  )
  
  
  unit_circle <- rbind(
    cos(angle),
    sin(angle)
  )
  
  
  radius <- sqrt(
    qchisq(
      probability,
      df = 2
    )
  )
  
  
  transformation <-
    eigen_decomposition$vectors %*%
    diag(
      sqrt(
        pmax(
          eigen_decomposition$values,
          0
        )
      ),
      nrow = 2
    )
  
  
  ellipse <- t(
    radius *
      transformation %*%
      unit_circle
  )
  
  
  ellipse <- sweep(
    ellipse,
    2,
    mean,
    "+"
  )
  
  
  ellipse
}


# ------------------------------------------------------------------------------
# 55. Plot Estimated Gaussian Ellipses
# ------------------------------------------------------------------------------

plot(
  X,
  pch =
    final_GMM$cluster,
  cex = 0.5,
  xlab = "x1",
  ylab = "x2",
  main = "Estimated Gaussian Components"
)


for (
  k in seq_len(
    final_GMM$K
  )
) {
  
  ellipse_k <- covariance_ellipse(
    final_GMM$means[k, ],
    final_GMM$covariances[[k]],
    probability = 0.95
  )
  
  
  lines(
    ellipse_k,
    lwd = 2
  )
}


points(
  final_GMM$means,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXVI
#
# K-MEANS COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Fit K-Means
# ------------------------------------------------------------------------------

set.seed(123)


KMeans_model <- kmeans(
  X,
  centers = 3,
  nstart = 50
)


# ------------------------------------------------------------------------------
# 57. Compare Cluster Recovery
# ------------------------------------------------------------------------------

KMeans_rand <- rand_index(
  true_cluster,
  KMeans_model$cluster
)


GMM_rand <- rand_index(
  true_cluster,
  final_GMM$cluster
)


data.frame(
  Method = c(
    "K-Means",
    "Gaussian Mixture"
  ),
  Rand_Index = c(
    KMeans_rand,
    GMM_rand
  )
)


# ------------------------------------------------------------------------------
# 58. Compare Centers
# ------------------------------------------------------------------------------

KMeans_model$centers


final_GMM$means


# ==============================================================================
# PART XXVII
#
# K-MEANS AS A RESTRICTED GMM
# ==============================================================================


# ------------------------------------------------------------------------------
# 59. Spherical Equal-Covariance Example
# ------------------------------------------------------------------------------

set.seed(456)


spherical_covariance <- diag(
  c(
    0.8,
    0.8
  )
)


S1 <- rmvnorm_manual(
  100,
  c(
    -3,
    0
  ),
  spherical_covariance
)


S2 <- rmvnorm_manual(
  100,
  c(
    3,
    0
  ),
  spherical_covariance
)


S3 <- rmvnorm_manual(
  100,
  c(
    0,
    4
  ),
  spherical_covariance
)


X_spherical <- rbind(
  S1,
  S2,
  S3
)


true_spherical <- rep(
  1:3,
  each = 100
)


set.seed(123)


KMeans_spherical <- kmeans(
  X_spherical,
  centers = 3,
  nstart = 50
)


GMM_spherical <- fit_gmm_multistart(
  X_spherical,
  K = 3,
  nstart = 10,
  seed = 123
)


data.frame(
  Method = c(
    "K-Means",
    "GMM"
  ),
  Rand_Index = c(
    rand_index(
      true_spherical,
      KMeans_spherical$cluster
    ),
    rand_index(
      true_spherical,
      GMM_spherical$cluster
    )
  )
)


# ==============================================================================
# PART XXVIII
#
# OVERLAPPING COMPONENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 60. Generate Overlapping Mixture
# ------------------------------------------------------------------------------

set.seed(789)


overlap_covariance <- matrix(
  c(
    2, 0.5,
    0.5, 2
  ),
  nrow = 2
)


O1 <- rmvnorm_manual(
  200,
  c(
    -1,
    0
  ),
  overlap_covariance
)


O2 <- rmvnorm_manual(
  200,
  c(
    1,
    0
  ),
  overlap_covariance
)


X_overlap <- rbind(
  O1,
  O2
)


true_overlap <- rep(
  1:2,
  each = 200
)


overlap_GMM <- fit_gmm_multistart(
  X_overlap,
  K = 2,
  nstart = 20,
  seed = 123
)


overlap_uncertainty <- 1 -
  apply(
    overlap_GMM$responsibilities,
    1,
    max
  )


# ------------------------------------------------------------------------------
# 61. Plot Overlapping Components
# ------------------------------------------------------------------------------

plot(
  X_overlap,
  pch =
    overlap_GMM$cluster,
  cex =
    0.5 +
    2 *
    overlap_uncertainty,
  xlab = "x1",
  ylab = "x2",
  main = "Overlapping GMM Components and Uncertainty"
)


points(
  overlap_GMM$means,
  pch = 8,
  cex = 2,
  lwd = 3
)


# ==============================================================================
# PART XXIX
#
# SAMPLING FROM THE FITTED MIXTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 62. Sample from GMM
# ------------------------------------------------------------------------------

sample_gmm <- function(
    model,
    n,
    seed = NULL
) {
  
  if (
    !is.null(seed)
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  component <- sample(
    seq_len(
      model$K
    ),
    size = n,
    replace = TRUE,
    prob =
      model$mixing_proportions
  )
  
  
  X_sample <- matrix(
    NA_real_,
    nrow = n,
    ncol =
      ncol(
        model$means
      )
  )
  
  
  for (
    k in seq_len(
      model$K
    )
  ) {
    
    indices <- which(
      component == k
    )
    
    
    if (
      length(
        indices
      ) >
      0
    ) {
      
      X_sample[
        indices,
      ] <- rmvnorm_manual(
        length(
          indices
        ),
        model$means[k, ],
        model$covariances[[k]]
      )
    }
  }
  
  
  colnames(
    X_sample
  ) <- colnames(
    model$means
  )
  
  
  list(
    X =
      X_sample,
    component =
      component
  )
}


# ------------------------------------------------------------------------------
# 63. Generate Synthetic Data from Fitted Model
# ------------------------------------------------------------------------------

generated_data <- sample_gmm(
  final_GMM,
  n = 400,
  seed = 999
)


plot(
  generated_data$X,
  pch =
    generated_data$component,
  xlab = "x1",
  ylab = "x2",
  main = "Samples from Fitted Gaussian Mixture"
)


# ==============================================================================
# PART XXX
#
# SINGULARITY / VARIANCE COLLAPSE
# ==============================================================================


# ------------------------------------------------------------------------------
# 64. Demonstrate Why Covariance Regularization Matters
# ------------------------------------------------------------------------------

# Gaussian mixture likelihoods can become unbounded.
#
# Suppose one component mean approaches a single observation x_i and its
# covariance approaches zero.
#
# Then:
#
#       N(x_i | mu_k, Sigma_k)
#
# can become arbitrarily large.
#
# Therefore unrestricted maximum likelihood estimation of a finite Gaussian
# mixture has singular solutions.
#
# Our implementation protects against this by adding:
#
#       lambda * I
#
# to each covariance matrix.


regularization_values <- c(
  1e-2,
  1e-4,
  1e-6
)


regularization_results <- data.frame(
  Regularization =
    regularization_values,
  Log_Likelihood =
    NA_real_
)


for (
  i in seq_along(
    regularization_values
  )
) {
  
  model_i <- fit_gmm_multistart(
    X,
    K = 3,
    nstart = 5,
    covariance_regularization =
      regularization_values[i],
    seed =
      700 + i
  )
  
  
  regularization_results$Log_Likelihood[i] <-
    model_i$log_likelihood
}


regularization_results


# ==============================================================================
# PART XXXI
#
# OPTIONAL VERIFICATION WITH MCLUST
# ==============================================================================


# ------------------------------------------------------------------------------
# 65. Verify with mclust if Installed
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "mclust",
    quietly = TRUE
  )
) {
  
  mclust_model <- mclust::Mclust(
    X,
    G = 3,
    modelNames = "VVV"
  )
  
  
  cat(
    "Manual GMM log-likelihood:",
    round(
      final_GMM$log_likelihood,
      4
    ),
    "\n"
  )
  
  
  cat(
    "mclust log-likelihood:",
    round(
      mclust_model$loglik,
      4
    ),
    "\n"
  )
  
  
  cat(
    "Manual vs mclust Rand index:",
    round(
      rand_index(
        final_GMM$cluster,
        mclust_model$classification
      ),
      4
    ),
    "\n"
  )
}


# ==============================================================================
# PART XXXII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 66. Output
# ------------------------------------------------------------------------------

cat(
  "Gaussian Mixture Model Summary\n"
)


cat(
  "------------------------------\n"
)


cat(
  "Number of observations:",
  n,
  "\n"
)


cat(
  "Dimension:",
  p,
  "\n"
)


cat(
  "Number of components:",
  final_GMM$K,
  "\n"
)


cat(
  "Final log-likelihood:",
  round(
    final_GMM$log_likelihood,
    4
  ),
  "\n"
)


cat(
  "Mixing proportions:",
  paste(
    round(
      final_GMM$mixing_proportions,
      4
    ),
    collapse = ", "
  ),
  "\n"
)


cat(
  "Iterations:",
  final_GMM$iterations,
  "\n"
)


cat(
  "Converged:",
  final_GMM$converged,
  "\n"
)


cat(
  "Rand index:",
  round(
    rand_index(
      true_cluster,
      final_GMM$cluster
    ),
    4
  ),
  "\n"
)


cat(
  "Best K by BIC:",
  best_K_BIC,
  "\n"
)


cat(
  "Mean classification uncertainty:",
  round(
    mean(
      1 -
        apply(
          final_GMM$responsibilities,
          1,
          max
        )
    ),
    4
  ),
  "\n"
)