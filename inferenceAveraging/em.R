# ==============================================================================
# Expectation-Maximization (EM) Algorithm
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Latent variables
#   - Incomplete-data likelihood
#   - Complete-data likelihood
#   - E-step
#   - M-step
#   - Responsibilities
#   - Gaussian mixture models
#   - Log-likelihood monotonicity
#   - Local optima
#   - Sensitivity to initialization
#
# Example:
#
#   Fit a two-component Gaussian mixture:
#
#       p(x)
#       =
#       pi_1 N(x | mu_1, sigma_1^2)
#       +
#       pi_2 N(x | mu_2, sigma_2^2)
#
# The component labels are unobserved latent variables.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Gaussian Mixture Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 500


true_pi <- c(
  0.4,
  0.6
)


true_mu <- c(
  -2,
  3
)


true_sigma <- c(
  1,
  1.5
)


# Latent component memberships

z <- sample(
  1:2,
  size = n,
  replace = TRUE,
  prob = true_pi
)


# Generate observations conditional on latent component

x <- numeric(
  n
)


for (
  i in seq_len(n)
) {
  
  x[i] <- rnorm(
    1,
    mean = true_mu[
      z[i]
    ],
    sd = true_sigma[
      z[i]
    ]
  )
}


# ------------------------------------------------------------------------------
# 2. Plot Mixture Data
# ------------------------------------------------------------------------------

hist(
  x,
  breaks = 40,
  probability = TRUE,
  xlab = "x",
  main = "Gaussian Mixture Data"
)


x_grid <- seq(
  min(x) - 2,
  max(x) + 2,
  length.out = 500
)


true_density <-
  true_pi[1] *
  dnorm(
    x_grid,
    true_mu[1],
    true_sigma[1]
  ) +
  true_pi[2] *
  dnorm(
    x_grid,
    true_mu[2],
    true_sigma[2]
  )


lines(
  x_grid,
  true_density,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Gaussian Mixture Density
# ------------------------------------------------------------------------------

mixture_density <- function(
    x,
    pi,
    mu,
    sigma
) {
  
  K <- length(
    pi
  )
  
  
  density <- rep(
    0,
    length(x)
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    density <- density +
      pi[k] *
      dnorm(
        x,
        mean = mu[k],
        sd = sigma[k]
      )
  }
  
  
  return(
    density
  )
}


# ------------------------------------------------------------------------------
# 4. Observed-Data Log-Likelihood
# ------------------------------------------------------------------------------

# Since latent component memberships are unknown:
#
# log L(theta)
#
# =
# sum_i log[
#     sum_k pi_k N(x_i | mu_k, sigma_k^2)
# ]


mixture_log_likelihood <- function(
    x,
    pi,
    mu,
    sigma
) {
  
  density <- mixture_density(
    x,
    pi,
    mu,
    sigma
  )
  
  
  # Numerical safeguard
  
  density <- pmax(
    density,
    .Machine$double.xmin
  )
  
  
  sum(
    log(
      density
    )
  )
}


# ------------------------------------------------------------------------------
# 5. Initial Parameter Values
# ------------------------------------------------------------------------------

# EM requires starting values.
#
# Here we deliberately choose imperfect initial values.

pi_est <- c(
  0.5,
  0.5
)


mu_est <- c(
  -1,
  1
)


sigma_est <- c(
  2,
  2
)


# ------------------------------------------------------------------------------
# 6. E-Step
# ------------------------------------------------------------------------------

# Responsibility:
#
# gamma_ik
#
# =
# P(Z_i = k | x_i)
#
#
# =
# pi_k N(x_i | mu_k, sigma_k^2)
# -----------------------------------
# sum_j pi_j N(x_i | mu_j, sigma_j^2)


e_step <- function(
    x,
    pi,
    mu,
    sigma
) {
  
  n <- length(
    x
  )
  
  
  K <- length(
    pi
  )
  
  
  weighted_density <- matrix(
    0,
    nrow = n,
    ncol = K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    weighted_density[
      ,
      k
    ] <-
      pi[k] *
      dnorm(
        x,
        mean = mu[k],
        sd = sigma[k]
      )
  }
  
  
  denominator <- rowSums(
    weighted_density
  )
  
  
  denominator <- pmax(
    denominator,
    .Machine$double.xmin
  )
  
  
  responsibilities <-
    weighted_density /
    denominator
  
  
  colnames(
    responsibilities
  ) <- paste0(
    "Component",
    seq_len(K)
  )
  
  
  return(
    responsibilities
  )
}


# ------------------------------------------------------------------------------
# 7. Perform One E-Step
# ------------------------------------------------------------------------------

responsibilities <- e_step(
  x,
  pi_est,
  mu_est,
  sigma_est
)


head(
  responsibilities
)


# Responsibilities for each observation sum to 1.

head(
  rowSums(
    responsibilities
  )
)


# ------------------------------------------------------------------------------
# 8. Inspect Responsibilities
# ------------------------------------------------------------------------------

plot(
  x,
  responsibilities[
    ,
    1
  ],
  pch = 19,
  xlab = "x",
  ylab = "P(Component 1 | x)",
  main = "Responsibilities After E-Step"
)


abline(
  h = 0.5,
  lty = 2
)


# ------------------------------------------------------------------------------
# 9. M-Step
# ------------------------------------------------------------------------------

# Effective membership of component k:
#
# N_k = sum_i gamma_ik
#
#
# Update mixing probability:
#
# pi_k = N_k / n
#
#
# Update mean:
#
# mu_k =
# sum_i gamma_ik x_i / N_k
#
#
# Update variance:
#
# sigma_k^2 =
# sum_i gamma_ik (x_i - mu_k)^2 / N_k


m_step <- function(
    x,
    responsibilities,
    min_sigma = 1e-6
) {
  
  n <- length(
    x
  )
  
  
  K <- ncol(
    responsibilities
  )
  
  
  Nk <- colSums(
    responsibilities
  )
  
  
  pi_new <- Nk /
    n
  
  
  mu_new <- numeric(
    K
  )
  
  
  sigma_new <- numeric(
    K
  )
  
  
  for (
    k in seq_len(K)
  ) {
    
    mu_new[k] <-
      sum(
        responsibilities[
          ,
          k
        ] *
          x
      ) /
      Nk[k]
    
    
    variance_k <-
      sum(
        responsibilities[
          ,
          k
        ] *
          (
            x -
              mu_new[k]
          )^2
      ) /
      Nk[k]
    
    
    sigma_new[k] <-
      sqrt(
        max(
          variance_k,
          min_sigma^2
        )
      )
  }
  
  
  return(
    list(
      pi = pi_new,
      mu = mu_new,
      sigma = sigma_new,
      Nk = Nk
    )
  )
}


# ------------------------------------------------------------------------------
# 10. Perform One M-Step
# ------------------------------------------------------------------------------

updated_parameters <- m_step(
  x,
  responsibilities
)


updated_parameters


# ------------------------------------------------------------------------------
# 11. Full EM Algorithm
# ------------------------------------------------------------------------------

gaussian_mixture_em <- function(
    x,
    K = 2,
    pi_init = NULL,
    mu_init = NULL,
    sigma_init = NULL,
    max_iter = 1000,
    tol = 1e-8,
    min_sigma = 1e-6,
    verbose = FALSE
) {
  
  x <- as.numeric(
    x
  )
  
  
  n <- length(
    x
  )
  
  
  # --------------------------------------------------------------------------
  # Initialization
  # --------------------------------------------------------------------------
  
  if (
    is.null(
      pi_init
    )
  ) {
    
    pi <- rep(
      1 / K,
      K
    )
    
  } else {
    
    pi <- pi_init /
      sum(
        pi_init
      )
  }
  
  
  if (
    is.null(
      mu_init
    )
  ) {
    
    mu <- as.numeric(
      quantile(
        x,
        probs = seq(
          0.2,
          0.8,
          length.out = K
        )
      )
    )
    
  } else {
    
    mu <- as.numeric(
      mu_init
    )
  }
  
  
  if (
    is.null(
      sigma_init
    )
  ) {
    
    sigma <- rep(
      sd(x),
      K
    )
    
  } else {
    
    sigma <- as.numeric(
      sigma_init
    )
  }
  
  
  log_likelihood_history <- numeric(
    max_iter
  )
  
  
  parameter_history <- vector(
    "list",
    max_iter
  )
  
  
  # --------------------------------------------------------------------------
  # EM iterations
  # --------------------------------------------------------------------------
  
  for (
    iteration in seq_len(
      max_iter
    )
  ) {
    
    # E-step
    
    responsibilities <- e_step(
      x,
      pi,
      mu,
      sigma
    )
    
    
    # M-step
    
    update <- m_step(
      x,
      responsibilities,
      min_sigma = min_sigma
    )
    
    
    pi_new <- update$pi
    
    mu_new <- update$mu
    
    sigma_new <- update$sigma
    
    
    # Observed-data log-likelihood
    
    log_likelihood_history[
      iteration
    ] <- mixture_log_likelihood(
      x,
      pi_new,
      mu_new,
      sigma_new
    )
    
    
    parameter_history[[iteration]] <-
      list(
        pi = pi_new,
        mu = mu_new,
        sigma = sigma_new
      )
    
    
    if (
      verbose
    ) {
      
      cat(
        "Iteration:",
        iteration,
        "Log-Likelihood:",
        round(
          log_likelihood_history[
            iteration
          ],
          6
        ),
        "\n"
      )
    }
    
    
    # Convergence check based on log-likelihood change
    
    if (
      iteration > 1
    ) {
      
      improvement <-
        abs(
          log_likelihood_history[
            iteration
          ] -
            log_likelihood_history[
              iteration - 1
            ]
        )
      
      
      if (
        improvement <
        tol
      ) {
        
        pi <- pi_new
        
        mu <- mu_new
        
        sigma <- sigma_new
        
        
        break
      }
    }
    
    
    pi <- pi_new
    
    mu <- mu_new
    
    sigma <- sigma_new
  }
  
  
  # Final responsibilities
  
  responsibilities <- e_step(
    x,
    pi,
    mu,
    sigma
  )
  
  
  return(
    list(
      pi = pi,
      mu = mu,
      sigma = sigma,
      responsibilities = responsibilities,
      log_likelihood =
        log_likelihood_history[
          seq_len(
            iteration
          )
        ],
      parameter_history =
        parameter_history[
          seq_len(
            iteration
          )
        ],
      iterations = iteration
    )
  )
}


# ------------------------------------------------------------------------------
# 12. Fit Gaussian Mixture with EM
# ------------------------------------------------------------------------------

em_model <- gaussian_mixture_em(
  x = x,
  K = 2,
  pi_init = c(
    0.5,
    0.5
  ),
  mu_init = c(
    -1,
    1
  ),
  sigma_init = c(
    2,
    2
  ),
  max_iter = 1000,
  tol = 1e-8
)


em_model$pi


em_model$mu


em_model$sigma


em_model$iterations


# ------------------------------------------------------------------------------
# 13. Resolve Label Switching for Comparison
# ------------------------------------------------------------------------------

# Mixture labels themselves have no intrinsic meaning.
#
# Sort components by estimated mean before comparing with truth.

component_order <- order(
  em_model$mu
)


pi_sorted <- em_model$pi[
  component_order
]


mu_sorted <- em_model$mu[
  component_order
]


sigma_sorted <- em_model$sigma[
  component_order
]


data.frame(
  Parameter = c(
    "Mixing Weight",
    "Mean",
    "Standard Deviation"
  ),
  
  True_Component_1 = c(
    true_pi[1],
    true_mu[1],
    true_sigma[1]
  ),
  
  Estimated_Component_1 = c(
    pi_sorted[1],
    mu_sorted[1],
    sigma_sorted[1]
  ),
  
  True_Component_2 = c(
    true_pi[2],
    true_mu[2],
    true_sigma[2]
  ),
  
  Estimated_Component_2 = c(
    pi_sorted[2],
    mu_sorted[2],
    sigma_sorted[2]
  )
)


# ------------------------------------------------------------------------------
# 14. Plot Estimated Mixture Density
# ------------------------------------------------------------------------------

estimated_density <- mixture_density(
  x_grid,
  em_model$pi,
  em_model$mu,
  em_model$sigma
)


hist(
  x,
  breaks = 40,
  probability = TRUE,
  xlab = "x",
  main = "Gaussian Mixture Estimated by EM"
)


lines(
  x_grid,
  estimated_density,
  lwd = 2
)


lines(
  x_grid,
  true_density,
  lty = 2,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "EM Estimate",
    "True Density"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 15. Plot Individual Mixture Components
# ------------------------------------------------------------------------------

component_1_density <-
  em_model$pi[1] *
  dnorm(
    x_grid,
    em_model$mu[1],
    em_model$sigma[1]
  )


component_2_density <-
  em_model$pi[2] *
  dnorm(
    x_grid,
    em_model$mu[2],
    em_model$sigma[2]
  )


plot(
  x_grid,
  estimated_density,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "Density",
  main = "Estimated Mixture Components"
)


lines(
  x_grid,
  component_1_density,
  lty = 2,
  lwd = 2
)


lines(
  x_grid,
  component_2_density,
  lty = 3,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Mixture",
    "Component 1",
    "Component 2"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 16. Plot Log-Likelihood Convergence
# ------------------------------------------------------------------------------

plot(
  seq_along(
    em_model$log_likelihood
  ),
  em_model$log_likelihood,
  type = "b",
  pch = 19,
  xlab = "EM Iteration",
  ylab = "Log-Likelihood",
  main = "EM Log-Likelihood Convergence"
)


# ------------------------------------------------------------------------------
# 17. Check Monotonicity
# ------------------------------------------------------------------------------

likelihood_changes <- diff(
  em_model$log_likelihood
)


min(
  likelihood_changes
)


all(
  likelihood_changes >=
    -1e-10
)


# ------------------------------------------------------------------------------
# 18. Plot Parameter Convergence
# ------------------------------------------------------------------------------

number_iterations <- em_model$iterations


pi_history <- matrix(
  NA,
  nrow = number_iterations,
  ncol = 2
)


mu_history <- matrix(
  NA,
  nrow = number_iterations,
  ncol = 2
)


sigma_history <- matrix(
  NA,
  nrow = number_iterations,
  ncol = 2
)


for (
  i in seq_len(
    number_iterations
  )
) {
  
  pi_history[
    i,
  ] <-
    em_model$parameter_history[[i]]$pi
  
  
  mu_history[
    i,
  ] <-
    em_model$parameter_history[[i]]$mu
  
  
  sigma_history[
    i,
  ] <-
    em_model$parameter_history[[i]]$sigma
}


par(
  mfrow = c(
    1,
    3
  )
)


matplot(
  pi_history,
  type = "l",
  lty = 1:2,
  xlab = "Iteration",
  ylab = "Mixing Probability",
  main = "Mixing Weights"
)


matplot(
  mu_history,
  type = "l",
  lty = 1:2,
  xlab = "Iteration",
  ylab = "Mean",
  main = "Component Means"
)


matplot(
  sigma_history,
  type = "l",
  lty = 1:2,
  xlab = "Iteration",
  ylab = "Standard Deviation",
  main = "Component SDs"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 19. Posterior Responsibilities
# ------------------------------------------------------------------------------

responsibilities <- em_model$responsibilities


head(
  responsibilities
)


# ------------------------------------------------------------------------------
# 20. Classify Observations by Most Likely Component
# ------------------------------------------------------------------------------

predicted_component <- max.col(
  responsibilities
)


table(
  predicted_component
)


# ------------------------------------------------------------------------------
# 21. Compare Estimated and True Latent Classes
# ------------------------------------------------------------------------------

# Need to account for possible label switching.
#
# Sort estimated classes by component mean.

rank_of_component <- rank(
  em_model$mu
)


predicted_sorted <- rank_of_component[
  predicted_component
]


confusion_matrix <- table(
  True = z,
  Estimated = predicted_sorted
)


confusion_matrix


classification_accuracy <- mean(
  predicted_sorted ==
    z
)


classification_accuracy


# ------------------------------------------------------------------------------
# 22. Plot Responsibilities
# ------------------------------------------------------------------------------

order_x <- order(
  x
)


plot(
  x[
    order_x
  ],
  responsibilities[
    order_x,
    component_order[1]
  ],
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "Posterior Responsibility",
  main = "Posterior Component Probabilities"
)


lines(
  x[
    order_x
  ],
  responsibilities[
    order_x,
    component_order[2]
  ],
  lty = 2,
  lwd = 2
)


abline(
  h = 0.5,
  lty = 3
)


legend(
  "right",
  legend = c(
    "Component 1",
    "Component 2"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 23. Uncertainty of Component Assignment
# ------------------------------------------------------------------------------

# Maximum responsibility close to 1:
#
# confident assignment.
#
# Close to 0.5 in the two-component case:
#
# ambiguous assignment.


assignment_confidence <- apply(
  responsibilities,
  1,
  max
)


hist(
  assignment_confidence,
  breaks = 30,
  xlab = "Maximum Responsibility",
  main = "Certainty of Mixture Assignments"
)


# ------------------------------------------------------------------------------
# 24. Identify Ambiguous Observations
# ------------------------------------------------------------------------------

ambiguous_index <- which(
  assignment_confidence <
    0.7
)


length(
  ambiguous_index
)


plot(
  x,
  rep(
    0,
    n
  ),
  pch = 19,
  xlab = "x",
  ylab = "",
  yaxt = "n",
  main = "Ambiguous Mixture Assignments"
)


points(
  x[
    ambiguous_index
  ],
  rep(
    0,
    length(
      ambiguous_index
    )
  ),
  pch = 1,
  cex = 1.5,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 25. Complete-Data Log-Likelihood
# ------------------------------------------------------------------------------

# If latent memberships z_ik were observed:
#
# log L_c(theta)
#
# =
# sum_i sum_k
# z_ik [
#   log pi_k +
#   log N(x_i | mu_k, sigma_k^2)
# ]
#
#
# EM replaces unknown z_ik with responsibilities gamma_ik
# during the E-step.


expected_complete_log_likelihood <- function(
    x,
    responsibilities,
    pi,
    mu,
    sigma
) {
  
  n <- length(
    x
  )
  
  
  K <- length(
    pi
  )
  
  
  Q <- 0
  
  
  for (
    k in seq_len(K)
  ) {
    
    Q <- Q +
      sum(
        responsibilities[
          ,
          k
        ] *
          (
            log(
              pi[k]
            ) +
              dnorm(
                x,
                mean = mu[k],
                sd = sigma[k],
                log = TRUE
              )
          )
      )
  }
  
  
  return(
    Q
  )
}


Q_final <- expected_complete_log_likelihood(
  x,
  em_model$responsibilities,
  em_model$pi,
  em_model$mu,
  em_model$sigma
)


Q_final


# ------------------------------------------------------------------------------
# 26. E-Step as Soft Classification
# ------------------------------------------------------------------------------

# Hard classification would assign:
#
# z_i = argmax_k gamma_ik
#
# EM instead uses fractional membership.
#
# Example observation near overlap:

overlap_index <- which.min(
  abs(
    responsibilities[
      ,
      1
    ] -
      0.5
  )
)


x[
  overlap_index
]


responsibilities[
  overlap_index,
]


# ------------------------------------------------------------------------------
# 27. Effective Component Sample Sizes
# ------------------------------------------------------------------------------

effective_counts <- colSums(
  responsibilities
)


effective_counts


effective_counts /
  n


# ------------------------------------------------------------------------------
# 28. Sensitivity to Initialization
# ------------------------------------------------------------------------------

# EM is not guaranteed to find the global maximum.
#
# Run multiple random initializations.


set.seed(123)


number_starts <- 20


initialization_results <- data.frame(
  Start = seq_len(
    number_starts
  ),
  Final_LogLikelihood = NA,
  Mean_1 = NA,
  Mean_2 = NA,
  SD_1 = NA,
  SD_2 = NA,
  Pi_1 = NA,
  Pi_2 = NA,
  Iterations = NA
)


models <- vector(
  "list",
  number_starts
)


for (
  start in seq_len(
    number_starts
  )
) {
  
  mu_init <- sample(
    x,
    size = 2
  )
  
  
  sigma_init <- runif(
    2,
    min = 0.5,
    max = 3
  )
  
  
  pi_init <- runif(
    2
  )
  
  
  pi_init <- pi_init /
    sum(
      pi_init
    )
  
  
  model_start <- gaussian_mixture_em(
    x = x,
    K = 2,
    pi_init = pi_init,
    mu_init = mu_init,
    sigma_init = sigma_init,
    max_iter = 1000,
    tol = 1e-8
  )
  
  
  models[[start]] <- model_start
  
  
  sorted_index <- order(
    model_start$mu
  )
  
  
  initialization_results$Final_LogLikelihood[start] <-
    tail(
      model_start$log_likelihood,
      1
    )
  
  
  initialization_results$Mean_1[start] <-
    model_start$mu[
      sorted_index[1]
    ]
  
  
  initialization_results$Mean_2[start] <-
    model_start$mu[
      sorted_index[2]
    ]
  
  
  initialization_results$SD_1[start] <-
    model_start$sigma[
      sorted_index[1]
    ]
  
  
  initialization_results$SD_2[start] <-
    model_start$sigma[
      sorted_index[2]
    ]
  
  
  initialization_results$Pi_1[start] <-
    model_start$pi[
      sorted_index[1]
    ]
  
  
  initialization_results$Pi_2[start] <-
    model_start$pi[
      sorted_index[2]
    ]
  
  
  initialization_results$Iterations[start] <-
    model_start$iterations
}


initialization_results


# ------------------------------------------------------------------------------
# 29. Best Initialization
# ------------------------------------------------------------------------------

best_start <- which.max(
  initialization_results$Final_LogLikelihood
)


best_model <- models[[best_start]]


best_start


initialization_results[
  best_start,
]


# ------------------------------------------------------------------------------
# 30. Plot Final Likelihood Across Initializations
# ------------------------------------------------------------------------------

plot(
  initialization_results$Start,
  initialization_results$Final_LogLikelihood,
  type = "h",
  lwd = 2,
  xlab = "Random Initialization",
  ylab = "Final Log-Likelihood",
  main = "Sensitivity to Initialization"
)


points(
  best_start,
  initialization_results$Final_LogLikelihood[
    best_start
  ],
  pch = 19,
  cex = 1.5
)


# ------------------------------------------------------------------------------
# 31. K-Means Initialization
# ------------------------------------------------------------------------------

# K-means can provide a useful initialization for Gaussian mixture EM.


set.seed(123)


kmeans_result <- kmeans(
  x,
  centers = 2
)


mu_kmeans <- as.numeric(
  kmeans_result$centers
)


pi_kmeans <- as.numeric(
  table(
    kmeans_result$cluster
  ) /
    n
)


sigma_kmeans <- numeric(
  2
)


for (
  k in 1:2
) {
  
  sigma_kmeans[k] <- sd(
    x[
      kmeans_result$cluster ==
        k
    ]
  )
}


kmeans_em_model <- gaussian_mixture_em(
  x = x,
  K = 2,
  pi_init = pi_kmeans,
  mu_init = mu_kmeans,
  sigma_init = sigma_kmeans
)


# ------------------------------------------------------------------------------
# 32. Compare Initialization Strategies
# ------------------------------------------------------------------------------

data.frame(
  Initialization = c(
    "Manual",
    "K-Means",
    "Best Random Start"
  ),
  
  Final_LogLikelihood = c(
    tail(
      em_model$log_likelihood,
      1
    ),
    
    tail(
      kmeans_em_model$log_likelihood,
      1
    ),
    
    tail(
      best_model$log_likelihood,
      1
    )
  ),
  
  Iterations = c(
    em_model$iterations,
    kmeans_em_model$iterations,
    best_model$iterations
  )
)


# ------------------------------------------------------------------------------
# 33. Compare EM with K-Means
# ------------------------------------------------------------------------------

# K-means uses hard assignments.
#
# EM for Gaussian mixtures uses soft assignments.


plot(
  x,
  rep(
    0,
    n
  ),
  col = kmeans_result$cluster,
  pch = 19,
  xlab = "x",
  ylab = "",
  yaxt = "n",
  main = "K-Means Hard Assignments"
)


# EM responsibilities vary continuously.

plot(
  x,
  responsibilities[
    ,
    component_order[1]
  ],
  pch = 19,
  xlab = "x",
  ylab = "Responsibility",
  main = "EM Soft Assignments"
)


# ------------------------------------------------------------------------------
# 34. Fit Different Numbers of Mixture Components
# ------------------------------------------------------------------------------

# EM itself does not automatically determine K.
#
# Compare models with K = 1, 2, 3, 4.


fit_mixture_multiple_starts <- function(
    x,
    K,
    starts = 10,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  best_model <- NULL
  
  best_log_likelihood <- -Inf
  
  
  for (
    s in seq_len(
      starts
    )
  ) {
    
    mu_init <- sample(
      x,
      K
    )
    
    
    sigma_init <- rep(
      sd(x),
      K
    )
    
    
    pi_init <- rep(
      1 / K,
      K
    )
    
    
    model <- gaussian_mixture_em(
      x = x,
      K = K,
      pi_init = pi_init,
      mu_init = mu_init,
      sigma_init = sigma_init
    )
    
    
    final_ll <- tail(
      model$log_likelihood,
      1
    )
    
    
    if (
      final_ll >
      best_log_likelihood
    ) {
      
      best_log_likelihood <- final_ll
      
      best_model <- model
    }
  }
  
  
  return(
    best_model
  )
}


# ------------------------------------------------------------------------------
# 35. AIC and BIC for Number of Components
# ------------------------------------------------------------------------------

K_values <- 1:4


model_selection <- data.frame(
  K = K_values,
  LogLikelihood = NA,
  Parameters = NA,
  AIC = NA,
  BIC = NA
)


mixture_models <- vector(
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
  
  K <- K_values[i]
  
  
  model_K <- fit_mixture_multiple_starts(
    x,
    K,
    starts = 10,
    seed = 100 +
      K
  )
  
  
  mixture_models[[i]] <- model_K
  
  
  log_likelihood <- tail(
    model_K$log_likelihood,
    1
  )
  
  
  # Parameters:
  #
  # K means
  # K variances
  # K - 1 free mixing probabilities
  #
  # total = 3K - 1
  
  number_parameters <- 3 *
    K -
    1
  
  
  model_selection$LogLikelihood[i] <-
    log_likelihood
  
  
  model_selection$Parameters[i] <-
    number_parameters
  
  
  model_selection$AIC[i] <-
    -2 *
    log_likelihood +
    2 *
    number_parameters
  
  
  model_selection$BIC[i] <-
    -2 *
    log_likelihood +
    log(n) *
    number_parameters
}


model_selection


# ------------------------------------------------------------------------------
# 36. Best Number of Components
# ------------------------------------------------------------------------------

best_K_AIC <- model_selection$K[
  which.min(
    model_selection$AIC
  )
]


best_K_BIC <- model_selection$K[
  which.min(
    model_selection$BIC
  )
]


best_K_AIC


best_K_BIC


# ------------------------------------------------------------------------------
# 37. Plot BIC
# ------------------------------------------------------------------------------

plot(
  model_selection$K,
  model_selection$BIC,
  type = "b",
  pch = 19,
  xlab = "Number of Components",
  ylab = "BIC",
  main = "Gaussian Mixture Model Selection"
)


# ------------------------------------------------------------------------------
# 38. Numerical Stability: Log-Sum-Exp
# ------------------------------------------------------------------------------

# For more extreme values or high-dimensional mixtures, directly computing:
#
# pi_k * density
#
# can underflow.
#
# A numerically stable implementation works on the log scale.


log_sum_exp <- function(
    values
) {
  
  maximum <- max(
    values
  )
  
  
  maximum +
    log(
      sum(
        exp(
          values -
            maximum
        )
      )
    )
}


# ------------------------------------------------------------------------------
# 39. Stable Mixture Log-Likelihood
# ------------------------------------------------------------------------------

stable_mixture_log_likelihood <- function(
    x,
    pi,
    mu,
    sigma
) {
  
  K <- length(
    pi
  )
  
  
  log_likelihood <- 0
  
  
  for (
    i in seq_along(
      x
    )
  ) {
    
    log_terms <- numeric(
      K
    )
    
    
    for (
      k in seq_len(K)
    ) {
      
      log_terms[k] <-
        log(
          pi[k]
        ) +
        dnorm(
          x[i],
          mean = mu[k],
          sd = sigma[k],
          log = TRUE
        )
    }
    
    
    log_likelihood <-
      log_likelihood +
      log_sum_exp(
        log_terms
      )
  }
  
  
  return(
    log_likelihood
  )
}


# ------------------------------------------------------------------------------
# 40. Verify Stable and Direct Likelihood
# ------------------------------------------------------------------------------

direct_ll <- mixture_log_likelihood(
  x,
  em_model$pi,
  em_model$mu,
  em_model$sigma
)


stable_ll <- stable_mixture_log_likelihood(
  x,
  em_model$pi,
  em_model$mu,
  em_model$sigma
)


c(
  Direct = direct_ll,
  Stable = stable_ll,
  Difference = direct_ll -
    stable_ll
)


# ------------------------------------------------------------------------------
# 41. EM as Missing-Data Estimation
# ------------------------------------------------------------------------------

# Conceptually:
#
# observed data:
#
#     x_i
#
# missing / latent data:
#
#     z_i
#
# If z_i were observed, maximum likelihood would be straightforward.
#
# EM alternates:
#
# E-step:
#
#     estimate the distribution of z_i given x_i
#
# M-step:
#
#     maximize the expected complete-data log-likelihood


# ------------------------------------------------------------------------------
# 42. Demonstrate One Full EM Update
# ------------------------------------------------------------------------------

pi_demo <- c(
  0.5,
  0.5
)


mu_demo <- c(
  -1,
  1
)


sigma_demo <- c(
  2,
  2
)


ll_before <- mixture_log_likelihood(
  x,
  pi_demo,
  mu_demo,
  sigma_demo
)


gamma_demo <- e_step(
  x,
  pi_demo,
  mu_demo,
  sigma_demo
)


update_demo <- m_step(
  x,
  gamma_demo
)


ll_after <- mixture_log_likelihood(
  x,
  update_demo$pi,
  update_demo$mu,
  update_demo$sigma
)


data.frame(
  Stage = c(
    "Before EM Update",
    "After EM Update"
  ),
  LogLikelihood = c(
    ll_before,
    ll_after
  )
)


# ------------------------------------------------------------------------------
# 43. Summary
# ------------------------------------------------------------------------------

cat(
  "EM Algorithm Summary\n"
)


cat(
  "--------------------\n"
)


cat(
  "Iterations:",
  em_model$iterations,
  "\n"
)


cat(
  "Final log-likelihood:",
  round(
    tail(
      em_model$log_likelihood,
      1
    ),
    4
  ),
  "\n"
)


cat(
  "\nEstimated mixing weights:\n"
)


print(
  round(
    pi_sorted,
    4
  )
)


cat(
  "\nEstimated means:\n"
)


print(
  round(
    mu_sorted,
    4
  )
)


cat(
  "\nEstimated standard deviations:\n"
)


print(
  round(
    sigma_sorted,
    4
  )
)


cat(
  "\nTrue means:\n"
)


print(
  true_mu
)


cat(
  "\nClassification accuracy after resolving labels:",
  round(
    classification_accuracy,
    4
  ),
  "\n"
)


cat(
  "\nBIC-selected number of components:",
  best_K_BIC,
  "\n"
)