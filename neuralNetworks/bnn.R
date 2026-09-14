# ==============================================================================
# Bayesian Neural Networks
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Bayesian neural networks
#   - Priors on neural-network weights
#   - Gaussian likelihood
#   - Posterior inference
#   - Metropolis-Hastings MCMC
#   - Posterior predictive distribution
#   - Parameter uncertainty
#   - Predictive uncertainty
#   - Credible intervals
#   - Prior predictive simulation
#   - Posterior predictive simulation
#
#
# Single-hidden-layer regression network:
#
#       z_m
#       =
#       b1_m + x' w1_m
#
#
#       h_m
#       =
#       tanh(z_m)
#
#
#       f_theta(x)
#       =
#       b2 + sum_m w2_m h_m
#
#
# Observation model:
#
#       y_i
#       =
#       f_theta(x_i)
#       +
#       epsilon_i
#
#
#       epsilon_i ~ Normal(0, sigma^2)
#
#
# Priors:
#
#       weights ~ Normal(0, tau^2)
#
#
#       log(sigma) ~ Normal(mu_sigma, sd_sigma^2)
#
#
# Posterior:
#
#       p(theta, sigma | y, X)
#       proportional to
#
#       p(y | X, theta, sigma)
#       p(theta)
#       p(sigma)
#
#
# Instead of returning ONE optimized neural network, Bayesian inference
# produces MANY plausible networks drawn from the posterior distribution.
#
# Predictions are therefore distributions rather than single values.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Predictors
# ------------------------------------------------------------------------------

set.seed(123)

n <- 150

p <- 2


x1 <- runif(
  n,
  min = -3,
  max = 3
)


x2 <- runif(
  n,
  min = -3,
  max = 3
)


X <- cbind(
  x1 = x1,
  x2 = x2
)


# ------------------------------------------------------------------------------
# 2. True Regression Function
# ------------------------------------------------------------------------------

true_function <- function(
    x1,
    x2
) {
  
  2 +
    2.5 * sin(
      x1
    ) +
    1.25 * tanh(
      x1 + x2
    )
}


y_true <- true_function(
  x1,
  x2
)


sigma_true <- 0.75


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma_true
  )


# ------------------------------------------------------------------------------
# 3. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data.frame(
    y = y,
    x1 = x1,
    x2 = x2
  ),
  main = "Bayesian Neural Network Data"
)


# ==============================================================================
# PART II
#
# TRAIN-TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Split Data
# ------------------------------------------------------------------------------

set.seed(456)


train_index <- sample(
  seq_len(n),
  size = floor(
    0.75 *
      n
  )
)


test_index <- setdiff(
  seq_len(n),
  train_index
)


X_train <- X[
  train_index,
  ,
  drop = FALSE
]


y_train <- y[
  train_index
]


X_test <- X[
  test_index,
  ,
  drop = FALSE
]


y_test <- y[
  test_index
]


# ==============================================================================
# PART III
#
# STANDARDIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Predictor Scaling
# ------------------------------------------------------------------------------

X_mean <- colMeans(
  X_train
)


X_sd <- apply(
  X_train,
  2,
  sd
)


X_train_scaled <- sweep(
  X_train,
  2,
  X_mean,
  "-"
)


X_train_scaled <- sweep(
  X_train_scaled,
  2,
  X_sd,
  "/"
)


X_test_scaled <- sweep(
  X_test,
  2,
  X_mean,
  "-"
)


X_test_scaled <- sweep(
  X_test_scaled,
  2,
  X_sd,
  "/"
)


# ------------------------------------------------------------------------------
# 6. Response Scaling
# ------------------------------------------------------------------------------

y_mean <- mean(
  y_train
)


y_sd <- sd(
  y_train
)


y_train_scaled <- (
  y_train -
    y_mean
) /
  y_sd


y_test_scaled <- (
  y_test -
    y_mean
) /
  y_sd


# ==============================================================================
# PART IV
#
# BAYESIAN NEURAL NETWORK ARCHITECTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Network Dimensions
# ------------------------------------------------------------------------------

input_size <- ncol(
  X_train_scaled
)


hidden_size <- 3


# We deliberately use a very small network.
#
# MCMC over neural-network parameters becomes expensive quickly.
#
# Parameters:
#
#       W1: p x H
#       b1: H
#       W2: H
#       b2: 1
#       log_sigma: 1


# ------------------------------------------------------------------------------
# 8. Number of Parameters
# ------------------------------------------------------------------------------

number_network_parameters <-
  input_size *
  hidden_size +
  hidden_size +
  hidden_size +
  1


number_total_parameters <-
  number_network_parameters +
  1


number_network_parameters


number_total_parameters


# ==============================================================================
# PART V
#
# PARAMETER PACKING AND UNPACKING
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Convert Parameter Vector into Network Parameters
# ------------------------------------------------------------------------------

unpack_parameters <- function(
    theta,
    input_size,
    hidden_size
) {
  
  index <- 1
  
  
  # --------------------------------------------------------------------------
  # W1
  # --------------------------------------------------------------------------
  
  W1_length <- input_size *
    hidden_size
  
  
  W1 <- matrix(
    theta[
      index:
        (
          index +
            W1_length -
            1
        )
    ],
    nrow =
      input_size,
    ncol =
      hidden_size
  )
  
  
  index <- index +
    W1_length
  
  
  # --------------------------------------------------------------------------
  # b1
  # --------------------------------------------------------------------------
  
  b1 <- theta[
    index:
      (
        index +
          hidden_size -
          1
      )
  ]
  
  
  index <- index +
    hidden_size
  
  
  # --------------------------------------------------------------------------
  # W2
  # --------------------------------------------------------------------------
  
  W2 <- theta[
    index:
      (
        index +
          hidden_size -
          1
      )
  ]
  
  
  index <- index +
    hidden_size
  
  
  # --------------------------------------------------------------------------
  # b2
  # --------------------------------------------------------------------------
  
  b2 <- theta[
    index
  ]
  
  
  index <- index +
    1
  
  
  # --------------------------------------------------------------------------
  # log sigma
  # --------------------------------------------------------------------------
  
  log_sigma <- theta[
    index
  ]
  
  
  return(
    list(
      W1 = W1,
      b1 = b1,
      W2 = W2,
      b2 = b2,
      log_sigma =
        log_sigma
    )
  )
}


# ------------------------------------------------------------------------------
# 10. Pack Parameters into Vector
# ------------------------------------------------------------------------------

pack_parameters <- function(
    W1,
    b1,
    W2,
    b2,
    log_sigma
) {
  
  c(
    as.vector(W1),
    as.vector(b1),
    as.vector(W2),
    b2,
    log_sigma
  )
}


# ==============================================================================
# PART VI
#
# FORWARD PROPAGATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Neural Network Forward Pass
# ------------------------------------------------------------------------------

network_forward <- function(
    X,
    theta,
    input_size,
    hidden_size
) {
  
  parameters <- unpack_parameters(
    theta,
    input_size,
    hidden_size
  )
  
  
  X <- as.matrix(
    X
  )
  
  
  Z1 <- sweep(
    X %*%
      parameters$W1,
    2,
    parameters$b1,
    "+"
  )
  
  
  H <- tanh(
    Z1
  )
  
  
  prediction <- as.vector(
    H %*%
      parameters$W2 +
      parameters$b2
  )
  
  
  return(
    prediction
  )
}


# ==============================================================================
# PART VII
#
# PRIOR DISTRIBUTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Prior Hyperparameters
# ------------------------------------------------------------------------------

weight_prior_sd <- 1


log_sigma_prior_mean <- log(
  0.5
)


log_sigma_prior_sd <- 1


# ------------------------------------------------------------------------------
# 13. Log Prior
# ------------------------------------------------------------------------------

log_prior <- function(
    theta,
    input_size,
    hidden_size,
    weight_prior_sd = 1,
    log_sigma_prior_mean = log(0.5),
    log_sigma_prior_sd = 1
) {
  
  parameters <- unpack_parameters(
    theta,
    input_size,
    hidden_size
  )
  
  
  # --------------------------------------------------------------------------
  # Neural-network weights and biases
  # --------------------------------------------------------------------------
  
  network_parameters <- c(
    as.vector(
      parameters$W1
    ),
    parameters$b1,
    parameters$W2,
    parameters$b2
  )
  
  
  network_log_prior <- sum(
    dnorm(
      network_parameters,
      mean = 0,
      sd =
        weight_prior_sd,
      log = TRUE
    )
  )
  
  
  # --------------------------------------------------------------------------
  # Prior on log sigma
  # --------------------------------------------------------------------------
  
  sigma_log_prior <- dnorm(
    parameters$log_sigma,
    mean =
      log_sigma_prior_mean,
    sd =
      log_sigma_prior_sd,
    log = TRUE
  )
  
  
  network_log_prior +
    sigma_log_prior
}


# ==============================================================================
# PART VIII
#
# LIKELIHOOD
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Log Likelihood
# ------------------------------------------------------------------------------

log_likelihood <- function(
    theta,
    X,
    y,
    input_size,
    hidden_size
) {
  
  parameters <- unpack_parameters(
    theta,
    input_size,
    hidden_size
  )
  
  
  prediction <- network_forward(
    X,
    theta,
    input_size,
    hidden_size
  )
  
  
  sigma <- exp(
    parameters$log_sigma
  )
  
  
  sum(
    dnorm(
      y,
      mean =
        prediction,
      sd =
        sigma,
      log = TRUE
    )
  )
}


# ==============================================================================
# PART IX
#
# POSTERIOR DENSITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Log Posterior
# ------------------------------------------------------------------------------

log_posterior <- function(
    theta,
    X,
    y,
    input_size,
    hidden_size,
    weight_prior_sd = 1,
    log_sigma_prior_mean = log(0.5),
    log_sigma_prior_sd = 1
) {
  
  prior_value <- log_prior(
    theta,
    input_size,
    hidden_size,
    weight_prior_sd =
      weight_prior_sd,
    log_sigma_prior_mean =
      log_sigma_prior_mean,
    log_sigma_prior_sd =
      log_sigma_prior_sd
  )
  
  
  if (
    !is.finite(
      prior_value
    )
  ) {
    
    return(
      -Inf
    )
  }
  
  
  likelihood_value <- log_likelihood(
    theta,
    X,
    y,
    input_size,
    hidden_size
  )
  
  
  prior_value +
    likelihood_value
}


# ==============================================================================
# PART X
#
# PRIOR PREDICTIVE DISTRIBUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Draw Parameters from Prior
# ------------------------------------------------------------------------------

sample_prior_theta <- function(
    input_size,
    hidden_size,
    weight_prior_sd = 1,
    log_sigma_prior_mean = log(0.5),
    log_sigma_prior_sd = 1
) {
  
  W1 <- matrix(
    rnorm(
      input_size *
        hidden_size,
      sd =
        weight_prior_sd
    ),
    nrow =
      input_size,
    ncol =
      hidden_size
  )
  
  
  b1 <- rnorm(
    hidden_size,
    sd =
      weight_prior_sd
  )
  
  
  W2 <- rnorm(
    hidden_size,
    sd =
      weight_prior_sd
  )
  
  
  b2 <- rnorm(
    1,
    sd =
      weight_prior_sd
  )
  
  
  log_sigma <- rnorm(
    1,
    mean =
      log_sigma_prior_mean,
    sd =
      log_sigma_prior_sd
  )
  
  
  pack_parameters(
    W1,
    b1,
    W2,
    b2,
    log_sigma
  )
}


# ------------------------------------------------------------------------------
# 17. Prior Predictive Functions
# ------------------------------------------------------------------------------

x1_grid_prior <- seq(
  -3,
  3,
  length.out = 200
)


X_prior_grid <- cbind(
  x1 =
    x1_grid_prior,
  x2 = 0
)


X_prior_grid_scaled <- sweep(
  X_prior_grid,
  2,
  X_mean,
  "-"
)


X_prior_grid_scaled <- sweep(
  X_prior_grid_scaled,
  2,
  X_sd,
  "/"
)


plot(
  x1_grid_prior,
  rep(
    y_mean,
    length(
      x1_grid_prior
    )
  ),
  type = "n",
  ylim = range(y),
  xlab = "x1",
  ylab = "Response",
  main = "Prior Predictive Neural Network Functions"
)


for (
  s in seq_len(20)
) {
  
  theta_prior <- sample_prior_theta(
    input_size =
      input_size,
    hidden_size =
      hidden_size,
    weight_prior_sd =
      weight_prior_sd,
    log_sigma_prior_mean =
      log_sigma_prior_mean,
    log_sigma_prior_sd =
      log_sigma_prior_sd
  )
  
  
  prediction_scaled <-
    network_forward(
      X_prior_grid_scaled,
      theta_prior,
      input_size,
      hidden_size
    )
  
  
  prediction_original <-
    y_mean +
    y_sd *
    prediction_scaled
  
  
  lines(
    x1_grid_prior,
    prediction_original
  )
}


# ==============================================================================
# PART XI
#
# INITIALIZE MCMC
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Initial Parameter Vector
# ------------------------------------------------------------------------------

set.seed(789)


initial_W1 <- matrix(
  rnorm(
    input_size *
      hidden_size,
    mean = 0,
    sd = 0.2
  ),
  nrow =
    input_size,
  ncol =
    hidden_size
)


initial_b1 <- rep(
  0,
  hidden_size
)


initial_W2 <- rnorm(
  hidden_size,
  mean = 0,
  sd = 0.2
)


initial_b2 <- 0


initial_log_sigma <- log(
  0.5
)


theta_initial <- pack_parameters(
  initial_W1,
  initial_b1,
  initial_W2,
  initial_b2,
  initial_log_sigma
)


length(
  theta_initial
)


# ==============================================================================
# PART XII
#
# RANDOM-WALK METROPOLIS-HASTINGS
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Metropolis-Hastings Sampler
# ------------------------------------------------------------------------------

metropolis_hastings_bnn <- function(
    X,
    y,
    theta_initial,
    input_size,
    hidden_size,
    iterations = 20000,
    proposal_sd = 0.02,
    weight_prior_sd = 1,
    log_sigma_prior_mean = log(0.5),
    log_sigma_prior_sd = 1,
    verbose = FALSE
) {
  
  number_parameters <- length(
    theta_initial
  )
  
  
  samples <- matrix(
    NA_real_,
    nrow =
      iterations,
    ncol =
      number_parameters
  )
  
  
  log_posterior_history <- numeric(
    iterations
  )
  
  
  accepted <- logical(
    iterations
  )
  
  
  theta_current <- theta_initial
  
  
  log_post_current <- log_posterior(
    theta_current,
    X,
    y,
    input_size,
    hidden_size,
    weight_prior_sd =
      weight_prior_sd,
    log_sigma_prior_mean =
      log_sigma_prior_mean,
    log_sigma_prior_sd =
      log_sigma_prior_sd
  )
  
  
  for (
    iteration in seq_len(
      iterations
    )
  ) {
    
    # ------------------------------------------------------------------------
    # Random-walk proposal
    # ------------------------------------------------------------------------
    
    theta_proposed <-
      theta_current +
      rnorm(
        number_parameters,
        mean = 0,
        sd = proposal_sd
      )
    
    
    # ------------------------------------------------------------------------
    # Proposed posterior
    # ------------------------------------------------------------------------
    
    log_post_proposed <- log_posterior(
      theta_proposed,
      X,
      y,
      input_size,
      hidden_size,
      weight_prior_sd =
        weight_prior_sd,
      log_sigma_prior_mean =
        log_sigma_prior_mean,
      log_sigma_prior_sd =
        log_sigma_prior_sd
    )
    
    
    # ------------------------------------------------------------------------
    # Metropolis acceptance probability
    # ------------------------------------------------------------------------
    
    log_acceptance_ratio <-
      log_post_proposed -
      log_post_current
    
    
    if (
      log(
        runif(1)
      ) <
      log_acceptance_ratio
    ) {
      
      theta_current <-
        theta_proposed
      
      
      log_post_current <-
        log_post_proposed
      
      
      accepted[
        iteration
      ] <- TRUE
    }
    
    
    samples[
      iteration,
    ] <- theta_current
    
    
    log_posterior_history[
      iteration
    ] <- log_post_current
    
    
    if (
      verbose &&
      iteration %% 1000 == 0
    ) {
      
      cat(
        "Iteration:",
        iteration,
        "| Acceptance Rate:",
        round(
          mean(
            accepted[
              seq_len(
                iteration
              )
            ]
          ),
          3
        ),
        "| Log Posterior:",
        round(
          log_post_current,
          2
        ),
        "\n"
      )
    }
  }
  
  
  return(
    list(
      samples =
        samples,
      accepted =
        accepted,
      log_posterior =
        log_posterior_history,
      acceptance_rate =
        mean(
          accepted
        ),
      proposal_sd =
        proposal_sd
    )
  )
}


# ==============================================================================
# PART XIII
#
# RUN MCMC
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Fit Bayesian Neural Network
# ------------------------------------------------------------------------------

set.seed(999)


MCMC_result <- metropolis_hastings_bnn(
  X = X_train_scaled,
  y = y_train_scaled,
  theta_initial =
    theta_initial,
  input_size =
    input_size,
  hidden_size =
    hidden_size,
  iterations = 20000,
  proposal_sd = 0.02,
  weight_prior_sd =
    weight_prior_sd,
  log_sigma_prior_mean =
    log_sigma_prior_mean,
  log_sigma_prior_sd =
    log_sigma_prior_sd,
  verbose = TRUE
)


MCMC_result$acceptance_rate


# ==============================================================================
# PART XIV
#
# MCMC DIAGNOSTICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Log Posterior Trace
# ------------------------------------------------------------------------------

plot(
  MCMC_result$log_posterior,
  type = "l",
  xlab = "MCMC Iteration",
  ylab = "Log Posterior",
  main = "BNN Log Posterior Trace"
)


# ------------------------------------------------------------------------------
# 22. Running Acceptance Rate
# ------------------------------------------------------------------------------

running_acceptance_rate <- cumsum(
  MCMC_result$accepted
) /
  seq_along(
    MCMC_result$accepted
  )


plot(
  running_acceptance_rate,
  type = "l",
  xlab = "MCMC Iteration",
  ylab = "Running Acceptance Rate",
  main = "Metropolis-Hastings Acceptance Rate"
)


# ==============================================================================
# PART XV
#
# REMOVE BURN-IN
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Burn-In
# ------------------------------------------------------------------------------

burn_in <- 5000


posterior_samples <- MCMC_result$samples[
  (
    burn_in + 1
  ):
    nrow(
      MCMC_result$samples
    ),
  ,
  drop = FALSE
]


nrow(
  posterior_samples
)


# ==============================================================================
# PART XVI
#
# THINNING FOR PLOTTING / COMPUTATIONAL CONVENIENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 24. Thin Posterior Samples
# ------------------------------------------------------------------------------

thin_every <- 10


posterior_samples_thinned <-
  posterior_samples[
    seq(
      1,
      nrow(
        posterior_samples
      ),
      by =
        thin_every
    ),
    ,
    drop = FALSE
  ]


nrow(
  posterior_samples_thinned
)


# NOTE:
#
# Thinning is not generally necessary for valid MCMC inference.
#
# Here it is used only to reduce the computational burden of repeated
# posterior-predictive calculations and plotting.


# ==============================================================================
# PART XVII
#
# PARAMETER TRACE PLOTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Selected Network Parameters
# ------------------------------------------------------------------------------

number_to_plot <- min(
  6,
  ncol(
    posterior_samples
  )
)


par(
  mfrow = c(
    3,
    2
  )
)


for (
  j in seq_len(
    number_to_plot
  )
) {
  
  plot(
    posterior_samples[
      ,
      j
    ],
    type = "l",
    xlab = "Post-Burn-In Iteration",
    ylab = paste0(
      "Theta ",
      j
    ),
    main = paste(
      "Parameter",
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
# PART XVIII
#
# SIGMA POSTERIOR
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Extract Noise Standard Deviation
# ------------------------------------------------------------------------------

log_sigma_index <- ncol(
  posterior_samples
)


sigma_samples_scaled <- exp(
  posterior_samples[
    ,
    log_sigma_index
  ]
)


# Transform sigma back to original response scale.

sigma_samples_original <-
  sigma_samples_scaled *
  y_sd


summary(
  sigma_samples_original
)


quantile(
  sigma_samples_original,
  probs = c(
    0.025,
    0.5,
    0.975
  )
)


hist(
  sigma_samples_original,
  breaks = 40,
  xlab = "Sigma",
  main = "Posterior Distribution of Observation Noise"
)


abline(
  v = sigma_true,
  lty = 2
)


# ==============================================================================
# PART XIX
#
# POSTERIOR PREDICTIVE MEAN FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Prediction Grid
# ------------------------------------------------------------------------------

x1_grid <- seq(
  -3,
  3,
  length.out = 200
)


X_grid <- cbind(
  x1 =
    x1_grid,
  x2 = 0
)


X_grid_scaled <- sweep(
  X_grid,
  2,
  X_mean,
  "-"
)


X_grid_scaled <- sweep(
  X_grid_scaled,
  2,
  X_sd,
  "/"
)


# ------------------------------------------------------------------------------
# 28. Draw Posterior Functions
# ------------------------------------------------------------------------------

number_posterior_draws <- min(
  100,
  nrow(
    posterior_samples_thinned
  )
)


posterior_draw_indices <- sample(
  seq_len(
    nrow(
      posterior_samples_thinned
    )
  ),
  size =
    number_posterior_draws
)


posterior_function_draws <- matrix(
  NA_real_,
  nrow =
    number_posterior_draws,
  ncol =
    length(
      x1_grid
    )
)


for (
  s in seq_len(
    number_posterior_draws
  )
) {
  
  theta_s <-
    posterior_samples_thinned[
      posterior_draw_indices[s],
    ]
  
  
  prediction_scaled <- network_forward(
    X_grid_scaled,
    theta_s,
    input_size,
    hidden_size
  )
  
  
  posterior_function_draws[
    s,
  ] <-
    y_mean +
    y_sd *
    prediction_scaled
}


# ------------------------------------------------------------------------------
# 29. Plot Posterior Functions
# ------------------------------------------------------------------------------

plot(
  x1_grid,
  true_function(
    x1_grid,
    0
  ),
  type = "l",
  lwd = 3,
  xlab = "x1",
  ylab = "Response",
  main = "Posterior Neural Network Functions"
)


for (
  s in seq_len(
    number_posterior_draws
  )
) {
  
  lines(
    x1_grid,
    posterior_function_draws[
      s,
    ]
  )
}


lines(
  x1_grid,
  true_function(
    x1_grid,
    0
  ),
  lwd = 3
)


# ==============================================================================
# PART XX
#
# POSTERIOR MEAN AND CREDIBLE INTERVAL
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Posterior Mean Function
# ------------------------------------------------------------------------------

posterior_mean_function <- colMeans(
  posterior_function_draws
)


# ------------------------------------------------------------------------------
# 31. Pointwise Credible Intervals for Mean Function
# ------------------------------------------------------------------------------

posterior_lower_function <- apply(
  posterior_function_draws,
  2,
  quantile,
  probs = 0.025
)


posterior_upper_function <- apply(
  posterior_function_draws,
  2,
  quantile,
  probs = 0.975
)


# ------------------------------------------------------------------------------
# 32. Plot Credible Band
# ------------------------------------------------------------------------------

plot(
  x1_grid,
  posterior_mean_function,
  type = "l",
  lwd = 3,
  ylim = range(
    posterior_lower_function,
    posterior_upper_function
  ),
  xlab = "x1",
  ylab = "Mean Response",
  main = "BNN Posterior Mean Function"
)


lines(
  x1_grid,
  posterior_lower_function,
  lty = 2,
  lwd = 2
)


lines(
  x1_grid,
  posterior_upper_function,
  lty = 2,
  lwd = 2
)


lines(
  x1_grid,
  true_function(
    x1_grid,
    0
  ),
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Posterior Mean",
    "95% Credible Band",
    "True Function"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = c(
    3,
    2,
    2
  )
)


# ==============================================================================
# PART XXI
#
# POSTERIOR PREDICTIVE DISTRIBUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Posterior Predictive Function
# ------------------------------------------------------------------------------

posterior_predictive_draws <- function(
    X_new,
    posterior_samples,
    input_size,
    hidden_size,
    y_mean,
    y_sd
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  number_samples <- nrow(
    posterior_samples
  )
  
  
  number_observations <- nrow(
    X_new
  )
  
  
  predictive_draws <- matrix(
    NA_real_,
    nrow =
      number_samples,
    ncol =
      number_observations
  )
  
  
  mean_draws <- matrix(
    NA_real_,
    nrow =
      number_samples,
    ncol =
      number_observations
  )
  
  
  for (
    s in seq_len(
      number_samples
    )
  ) {
    
    theta_s <- posterior_samples[
      s,
    ]
    
    
    parameters_s <- unpack_parameters(
      theta_s,
      input_size,
      hidden_size
    )
    
    
    mean_scaled <- network_forward(
      X_new,
      theta_s,
      input_size,
      hidden_size
    )
    
    
    sigma_scaled <- exp(
      parameters_s$log_sigma
    )
    
    
    predictive_scaled <- rnorm(
      number_observations,
      mean =
        mean_scaled,
      sd =
        sigma_scaled
    )
    
    
    mean_draws[
      s,
    ] <- y_mean +
      y_sd *
      mean_scaled
    
    
    predictive_draws[
      s,
    ] <- y_mean +
      y_sd *
      predictive_scaled
  }
  
  
  return(
    list(
      mean_draws =
        mean_draws,
      predictive_draws =
        predictive_draws
    )
  )
}


# ==============================================================================
# PART XXII
#
# TEST-SET POSTERIOR PREDICTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Posterior Predictive Draws on Test Data
# ------------------------------------------------------------------------------

test_posterior <- posterior_predictive_draws(
  X_new =
    X_test_scaled,
  posterior_samples =
    posterior_samples_thinned,
  input_size =
    input_size,
  hidden_size =
    hidden_size,
  y_mean =
    y_mean,
  y_sd =
    y_sd
)


# ------------------------------------------------------------------------------
# 35. Posterior Predictive Mean
# ------------------------------------------------------------------------------

test_prediction_mean <- colMeans(
  test_posterior$predictive_draws
)


# ------------------------------------------------------------------------------
# 36. Posterior Mean Function Prediction
# ------------------------------------------------------------------------------

test_mean_function <- colMeans(
  test_posterior$mean_draws
)


# ==============================================================================
# PART XXIII
#
# PREDICTIVE INTERVALS
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. 95% Posterior Predictive Intervals
# ------------------------------------------------------------------------------

test_predictive_lower <- apply(
  test_posterior$predictive_draws,
  2,
  quantile,
  probs = 0.025
)


test_predictive_upper <- apply(
  test_posterior$predictive_draws,
  2,
  quantile,
  probs = 0.975
)


# ------------------------------------------------------------------------------
# 38. Mean-Function Credible Intervals
# ------------------------------------------------------------------------------

test_mean_lower <- apply(
  test_posterior$mean_draws,
  2,
  quantile,
  probs = 0.025
)


test_mean_upper <- apply(
  test_posterior$mean_draws,
  2,
  quantile,
  probs = 0.975
)


# ==============================================================================
# PART XXIV
#
# TEST PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Metrics
# ------------------------------------------------------------------------------

MSE <- function(
    actual,
    predicted
) {
  
  mean(
    (
      actual -
        predicted
    )^2
  )
}


RMSE <- function(
    actual,
    predicted
) {
  
  sqrt(
    MSE(
      actual,
      predicted
    )
  )
}


R_squared <- function(
    actual,
    predicted
) {
  
  1 -
    sum(
      (
        actual -
          predicted
      )^2
    ) /
    sum(
      (
        actual -
          mean(actual)
      )^2
    )
}


# ------------------------------------------------------------------------------
# 40. Test Metrics
# ------------------------------------------------------------------------------

BNN_test_MSE <- MSE(
  y_test,
  test_mean_function
)


BNN_test_RMSE <- RMSE(
  y_test,
  test_mean_function
)


BNN_test_R_squared <- R_squared(
  y_test,
  test_mean_function
)


BNN_test_MSE


BNN_test_RMSE


BNN_test_R_squared


# ==============================================================================
# PART XXV
#
# PREDICTIVE INTERVAL COVERAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Coverage
# ------------------------------------------------------------------------------

predictive_coverage <- mean(
  y_test >=
    test_predictive_lower &
    y_test <=
    test_predictive_upper
)


predictive_coverage


# ------------------------------------------------------------------------------
# 42. Average Interval Width
# ------------------------------------------------------------------------------

average_predictive_width <- mean(
  test_predictive_upper -
    test_predictive_lower
)


average_predictive_width


# ==============================================================================
# PART XXVI
#
# VISUALIZE TEST PREDICTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Order Test Observations
# ------------------------------------------------------------------------------

test_order <- order(
  y_test
)


plot(
  seq_along(
    y_test
  ),
  y_test[
    test_order
  ],
  pch = 19,
  ylim = range(
    y_test,
    test_predictive_lower,
    test_predictive_upper
  ),
  xlab = "Ordered Test Observation",
  ylab = "Response",
  main = "BNN Posterior Predictive Intervals"
)


segments(
  x0 =
    seq_along(
      y_test
    ),
  y0 =
    test_predictive_lower[
      test_order
    ],
  x1 =
    seq_along(
      y_test
    ),
  y1 =
    test_predictive_upper[
      test_order
    ]
)


points(
  seq_along(
    y_test
  ),
  test_mean_function[
    test_order
  ],
  pch = 4
)


# ==============================================================================
# PART XXVII
#
# EPISTEMIC VS ALEATORIC UNCERTAINTY
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Posterior Mean Uncertainty
# ------------------------------------------------------------------------------

epistemic_variance <- apply(
  test_posterior$mean_draws,
  2,
  var
)


# ------------------------------------------------------------------------------
# 45. Predictive Variance
# ------------------------------------------------------------------------------

total_predictive_variance <- apply(
  test_posterior$predictive_draws,
  2,
  var
)


# ------------------------------------------------------------------------------
# 46. Approximate Aleatoric Component
# ------------------------------------------------------------------------------

aleatoric_variance <- pmax(
  total_predictive_variance -
    epistemic_variance,
  0
)


uncertainty_summary <- data.frame(
  Mean_Epistemic_Variance =
    mean(
      epistemic_variance
    ),
  
  Mean_Aleatoric_Variance =
    mean(
      aleatoric_variance
    ),
  
  Mean_Total_Variance =
    mean(
      total_predictive_variance
    )
)


uncertainty_summary


# ==============================================================================
# PART XXVIII
#
# UNCERTAINTY AWAY FROM THE TRAINING DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Extended x1 Grid
# ------------------------------------------------------------------------------

x1_extended <- seq(
  -6,
  6,
  length.out = 250
)


X_extended <- cbind(
  x1 =
    x1_extended,
  x2 = 0
)


X_extended_scaled <- sweep(
  X_extended,
  2,
  X_mean,
  "-"
)


X_extended_scaled <- sweep(
  X_extended_scaled,
  2,
  X_sd,
  "/"
)


# ------------------------------------------------------------------------------
# 48. Posterior Mean Draws
# ------------------------------------------------------------------------------

extended_posterior <- posterior_predictive_draws(
  X_new =
    X_extended_scaled,
  posterior_samples =
    posterior_samples_thinned,
  input_size =
    input_size,
  hidden_size =
    hidden_size,
  y_mean =
    y_mean,
  y_sd =
    y_sd
)


extended_mean <- colMeans(
  extended_posterior$mean_draws
)


extended_lower <- apply(
  extended_posterior$mean_draws,
  2,
  quantile,
  probs = 0.025
)


extended_upper <- apply(
  extended_posterior$mean_draws,
  2,
  quantile,
  probs = 0.975
)


# ------------------------------------------------------------------------------
# 49. Extrapolation Uncertainty
# ------------------------------------------------------------------------------

plot(
  x1_extended,
  extended_mean,
  type = "l",
  lwd = 3,
  ylim = range(
    extended_lower,
    extended_upper
  ),
  xlab = "x1",
  ylab = "Mean Response",
  main = "BNN Epistemic Uncertainty"
)


lines(
  x1_extended,
  extended_lower,
  lty = 2,
  lwd = 2
)


lines(
  x1_extended,
  extended_upper,
  lty = 2,
  lwd = 2
)


rug(
  X_train[
    ,
    "x1"
  ]
)


# ==============================================================================
# PART XXIX
#
# POSTERIOR PARAMETER SUMMARIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Posterior Mean Parameters
# ------------------------------------------------------------------------------

posterior_parameter_mean <- colMeans(
  posterior_samples
)


posterior_parameter_sd <- apply(
  posterior_samples,
  2,
  sd
)


parameter_summary <- data.frame(
  Parameter =
    seq_along(
      posterior_parameter_mean
    ),
  
  Posterior_Mean =
    posterior_parameter_mean,
  
  Posterior_SD =
    posterior_parameter_sd
)


head(
  parameter_summary,
  10
)


# IMPORTANT:
#
# Neural-network weights are not individually identifiable in the same way
# as ordinary regression coefficients.
#
# Hidden units can be permuted and signs can sometimes be transformed without
# changing the represented function.
#
# Therefore function-space predictions are usually more interpretable than
# marginal posterior summaries of individual weights.


# ==============================================================================
# PART XXX
#
# POSTERIOR CORRELATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Selected Parameter Correlations
# ------------------------------------------------------------------------------

selected_parameter_count <- min(
  8,
  ncol(
    posterior_samples
  )
)


posterior_correlation <- cor(
  posterior_samples[
    ,
    seq_len(
      selected_parameter_count
    ),
    drop = FALSE
  ]
)


round(
  posterior_correlation,
  2
)


# ==============================================================================
# PART XXXI
#
# SIMPLE EFFECTIVE SAMPLE SIZE
# ==============================================================================


# ------------------------------------------------------------------------------
# 52. Autocorrelation-Based ESS
# ------------------------------------------------------------------------------

simple_ESS <- function(
    x,
    max_lag = 100
) {
  
  n <- length(
    x
  )
  
  
  acf_values <- acf(
    x,
    plot = FALSE,
    lag.max =
      min(
        max_lag,
        n - 1
      )
  )$acf[
    -1
  ]
  
  
  positive_acf <- acf_values[
    acf_values >
      0
  ]
  
  
  if (
    length(
      positive_acf
    ) ==
    0
  ) {
    
    return(n)
  }
  
  
  n /
    (
      1 +
        2 *
        sum(
          positive_acf
        )
    )
}


# ------------------------------------------------------------------------------
# 53. ESS for Selected Parameters
# ------------------------------------------------------------------------------

ESS_values <- sapply(
  seq_len(
    min(
      6,
      ncol(
        posterior_samples
      )
    )
  ),
  function(j) {
    
    simple_ESS(
      posterior_samples[
        ,
        j
      ]
    )
  }
)


ESS_values


# ==============================================================================
# PART XXXII
#
# AUTOCORRELATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. ACF of First Parameter
# ------------------------------------------------------------------------------

acf(
  posterior_samples[
    ,
    1
  ],
  main = "MCMC Autocorrelation: Parameter 1"
)


# ==============================================================================
# PART XXXIII
#
# POSTERIOR PREDICTIVE CHECKS
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Replicated Dataset Summary
# ------------------------------------------------------------------------------

number_check_draws <- min(
  500,
  nrow(
    posterior_samples_thinned
  )
)


check_indices <- sample(
  seq_len(
    nrow(
      posterior_samples_thinned
    )
  ),
  size =
    number_check_draws
)


replicated_means <- numeric(
  number_check_draws
)


replicated_sds <- numeric(
  number_check_draws
)


for (
  s in seq_len(
    number_check_draws
  )
) {
  
  theta_s <-
    posterior_samples_thinned[
      check_indices[s],
    ]
  
  
  parameters_s <- unpack_parameters(
    theta_s,
    input_size,
    hidden_size
  )
  
  
  mean_scaled <- network_forward(
    X_train_scaled,
    theta_s,
    input_size,
    hidden_size
  )
  
  
  replicated_scaled <- rnorm(
    length(
      y_train_scaled
    ),
    mean =
      mean_scaled,
    sd =
      exp(
        parameters_s$log_sigma
      )
  )
  
  
  replicated_original <- y_mean +
    y_sd *
    replicated_scaled
  
  
  replicated_means[s] <- mean(
    replicated_original
  )
  
  
  replicated_sds[s] <- sd(
    replicated_original
  )
}


# ------------------------------------------------------------------------------
# 56. Compare Means
# ------------------------------------------------------------------------------

hist(
  replicated_means,
  breaks = 30,
  xlab = "Replicated Dataset Mean",
  main = "Posterior Predictive Check: Mean"
)


abline(
  v = mean(
    y_train
  ),
  lwd = 2,
  lty = 2
)


# ------------------------------------------------------------------------------
# 57. Compare Standard Deviations
# ------------------------------------------------------------------------------

hist(
  replicated_sds,
  breaks = 30,
  xlab = "Replicated Dataset SD",
  main = "Posterior Predictive Check: SD"
)


abline(
  v = sd(
    y_train
  ),
  lwd = 2,
  lty = 2
)


# ==============================================================================
# PART XXXIV
#
# ORDINARY NEURAL NETWORK COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 58. Simple Deterministic Neural Network
# ------------------------------------------------------------------------------

# For comparison, fit the same basic architecture by ordinary gradient descent.


initialize_deterministic_network <- function(
    input_size,
    hidden_size,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  list(
    W1 = matrix(
      rnorm(
        input_size *
          hidden_size,
        sd = 0.3
      ),
      nrow =
        input_size,
      ncol =
        hidden_size
    ),
    
    b1 = rep(
      0,
      hidden_size
    ),
    
    W2 = rnorm(
      hidden_size,
      sd = 0.3
    ),
    
    b2 = 0
  )
}


# ------------------------------------------------------------------------------
# 59. Deterministic Forward Pass
# ------------------------------------------------------------------------------

deterministic_forward <- function(
    X,
    parameters
) {
  
  Z <- sweep(
    X %*%
      parameters$W1,
    2,
    parameters$b1,
    "+"
  )
  
  
  H <- tanh(
    Z
  )
  
  
  prediction <- as.vector(
    H %*%
      parameters$W2 +
      parameters$b2
  )
  
  
  list(
    Z = Z,
    H = H,
    prediction =
      prediction
  )
}


# ------------------------------------------------------------------------------
# 60. Train Deterministic Network
# ------------------------------------------------------------------------------

fit_deterministic_network <- function(
    X,
    y,
    hidden_size = 3,
    epochs = 5000,
    learning_rate = 0.03,
    lambda = 0.01
) {
  
  X <- as.matrix(
    X
  )
  
  
  parameters <-
    initialize_deterministic_network(
      ncol(X),
      hidden_size
    )
  
  
  n <- nrow(
    X
  )
  
  
  for (
    epoch in seq_len(
      epochs
    )
  ) {
    
    cache <- deterministic_forward(
      X,
      parameters
    )
    
    
    dY <- (
      cache$prediction -
        y
    ) /
      n
    
    
    dW2 <- as.vector(
      crossprod(
        cache$H,
        dY
      )
    ) +
      lambda *
      parameters$W2
    
    
    db2 <- sum(
      dY
    )
    
    
    dH <- outer(
      dY,
      parameters$W2
    )
    
    
    dZ <- dH *
      (
        1 -
          tanh(
            cache$Z
          )^2
      )
    
    
    dW1 <- crossprod(
      X,
      dZ
    ) +
      lambda *
      parameters$W1
    
    
    db1 <- colSums(
      dZ
    )
    
    
    parameters$W1 <-
      parameters$W1 -
      learning_rate *
      dW1
    
    
    parameters$b1 <-
      parameters$b1 -
      learning_rate *
      db1
    
    
    parameters$W2 <-
      parameters$W2 -
      learning_rate *
      dW2
    
    
    parameters$b2 <-
      parameters$b2 -
      learning_rate *
      db2
  }
  
  
  parameters
}


# ------------------------------------------------------------------------------
# 61. Fit Deterministic Model
# ------------------------------------------------------------------------------

deterministic_model <- fit_deterministic_network(
  X_train_scaled,
  y_train_scaled,
  hidden_size =
    hidden_size
)


# ------------------------------------------------------------------------------
# 62. Deterministic Test Prediction
# ------------------------------------------------------------------------------

deterministic_test_scaled <-
  deterministic_forward(
    X_test_scaled,
    deterministic_model
  )$prediction


deterministic_test_prediction <-
  y_mean +
  y_sd *
  deterministic_test_scaled


deterministic_test_MSE <- MSE(
  y_test,
  deterministic_test_prediction
)


# ==============================================================================
# PART XXXV
#
# MODEL COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 63. Compare Point Prediction
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Deterministic Neural Network",
    "Bayesian Neural Network"
  ),
  
  Test_MSE = c(
    deterministic_test_MSE,
    BNN_test_MSE
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XXXVI
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 64. Output
# ------------------------------------------------------------------------------

cat(
  "Bayesian Neural Network Summary\n"
)


cat(
  "-------------------------------\n"
)


cat(
  "Training observations:",
  nrow(
    X_train
  ),
  "\n"
)


cat(
  "Input variables:",
  input_size,
  "\n"
)


cat(
  "Hidden units:",
  hidden_size,
  "\n"
)


cat(
  "Total unknown parameters:",
  number_total_parameters,
  "\n"
)


cat(
  "MCMC acceptance rate:",
  round(
    MCMC_result$acceptance_rate,
    4
  ),
  "\n"
)


cat(
  "Posterior median sigma:",
  round(
    median(
      sigma_samples_original
    ),
    4
  ),
  "\n"
)


cat(
  "True sigma:",
  sigma_true,
  "\n"
)


cat(
  "Bayesian NN test MSE:",
  round(
    BNN_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Bayesian NN test RMSE:",
  round(
    BNN_test_RMSE,
    4
  ),
  "\n"
)


cat(
  "Bayesian NN test R-squared:",
  round(
    BNN_test_R_squared,
    4
  ),
  "\n"
)


cat(
  "95% predictive interval coverage:",
  round(
    predictive_coverage,
    4
  ),
  "\n"
)


cat(
  "Average predictive interval width:",
  round(
    average_predictive_width,
    4
  ),
  "\n"
)


cat(
  "Deterministic NN test MSE:",
  round(
    deterministic_test_MSE,
    4
  ),
  "\n"
)