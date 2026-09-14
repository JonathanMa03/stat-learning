# ==============================================================================
# Markov Chain Monte Carlo (MCMC) Sampling
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Monte Carlo approximation
#   - Markov chains
#   - Stationary distributions
#   - Metropolis-Hastings
#   - Acceptance probabilities
#   - Random-walk proposals
#   - Burn-in
#   - Autocorrelation
#   - Effective sample size
#   - Monte Carlo standard error
#   - Gibbs sampling
#   - Bayesian posterior sampling
#   - Trace plots
#   - Posterior summaries
#
# Main example:
#
#   y_i ~ N(mu, sigma^2)
#
# with sigma^2 known and prior:
#
#   mu ~ N(mu_0, tau_0^2)
#
# The posterior distribution is analytically available, allowing us to
# verify the MCMC implementation.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 100

mu_true <- 3

sigma <- 2


y <- rnorm(
  n,
  mean = mu_true,
  sd = sigma
)


mean(
  y
)


sd(
  y
)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

hist(
  y,
  breaks = 25,
  probability = TRUE,
  xlab = "y",
  main = "Observed Data"
)


curve(
  dnorm(
    x,
    mean = mu_true,
    sd = sigma
  ),
  add = TRUE,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Bayesian Model
# ------------------------------------------------------------------------------

# Likelihood:
#
# y_i | mu ~ N(mu, sigma^2)
#
#
# Prior:
#
# mu ~ N(mu_0, tau_0^2)


mu_0 <- 0

tau_0 <- 5


# ------------------------------------------------------------------------------
# 4. Analytic Posterior
# ------------------------------------------------------------------------------

# Because the normal likelihood and normal prior are conjugate:
#
#
# posterior variance:
#
# tau_n^2 =
#
# 1 /
# (
#   1/tau_0^2 +
#   n/sigma^2
# )
#
#
# posterior mean:
#
# mu_n =
#
# tau_n^2 *
# (
#   mu_0/tau_0^2 +
#   n*y_bar/sigma^2
# )


posterior_variance <-
  1 /
  (
    1 /
      tau_0^2 +
      n /
      sigma^2
  )


posterior_sd <- sqrt(
  posterior_variance
)


posterior_mean <-
  posterior_variance *
  (
    mu_0 /
      tau_0^2 +
      sum(y) /
      sigma^2
  )


posterior_mean


posterior_sd


# ------------------------------------------------------------------------------
# 5. Exact Posterior Distribution
# ------------------------------------------------------------------------------

posterior_grid <- seq(
  posterior_mean -
    5 * posterior_sd,
  posterior_mean +
    5 * posterior_sd,
  length.out = 500
)


posterior_density <- dnorm(
  posterior_grid,
  mean = posterior_mean,
  sd = posterior_sd
)


plot(
  posterior_grid,
  posterior_density,
  type = "l",
  lwd = 2,
  xlab = expression(mu),
  ylab = "Posterior Density",
  main = "Exact Posterior Distribution"
)


abline(
  v = mu_true,
  lty = 2
)


# ==============================================================================
# PART I
#
# ORDINARY MONTE CARLO
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Monte Carlo Approximation
# ------------------------------------------------------------------------------

# If we can directly sample:
#
# theta^(1), ..., theta^(M) ~ p(theta)
#
# then:
#
# E[g(theta)]
#
# approximately equals
#
# 1/M sum_m g(theta^(m))


set.seed(123)


M <- 10000


direct_samples <- rnorm(
  M,
  mean = posterior_mean,
  sd = posterior_sd
)


# ------------------------------------------------------------------------------
# 7. Estimate Posterior Mean with Monte Carlo
# ------------------------------------------------------------------------------

mean(
  direct_samples
)


posterior_mean


# ------------------------------------------------------------------------------
# 8. Estimate Posterior Variance
# ------------------------------------------------------------------------------

var(
  direct_samples
)


posterior_variance


# ------------------------------------------------------------------------------
# 9. Posterior Probability
# ------------------------------------------------------------------------------

# Example:
#
# P(mu > 3 | y)


mean(
  direct_samples >
    3
)


# Exact calculation

1 -
  pnorm(
    3,
    posterior_mean,
    posterior_sd
  )


# ------------------------------------------------------------------------------
# 10. Monte Carlo Error
# ------------------------------------------------------------------------------

# For independent Monte Carlo samples:
#
# SE_MC(mean)
#
# approximately:
#
# sd(samples) / sqrt(M)


MC_SE_direct <-
  sd(
    direct_samples
  ) /
  sqrt(
    M
  )


MC_SE_direct


# ==============================================================================
# PART II
#
# METROPOLIS-HASTINGS
# ==============================================================================


# ------------------------------------------------------------------------------
# 11. Log Prior
# ------------------------------------------------------------------------------

log_prior <- function(
    mu
) {
  
  dnorm(
    mu,
    mean = mu_0,
    sd = tau_0,
    log = TRUE
  )
}


# ------------------------------------------------------------------------------
# 12. Log Likelihood
# ------------------------------------------------------------------------------

log_likelihood <- function(
    mu,
    y
) {
  
  sum(
    dnorm(
      y,
      mean = mu,
      sd = sigma,
      log = TRUE
    )
  )
}


# ------------------------------------------------------------------------------
# 13. Log Posterior Kernel
# ------------------------------------------------------------------------------

# We only need the posterior up to proportionality.
#
# p(mu | y)
#
# proportional to
#
# p(y | mu) p(mu)


log_posterior <- function(
    mu,
    y
) {
  
  log_prior(
    mu
  ) +
    log_likelihood(
      mu,
      y
    )
}


# ------------------------------------------------------------------------------
# 14. Visualize Posterior Kernel
# ------------------------------------------------------------------------------

log_posterior_values <- sapply(
  posterior_grid,
  function(mu) {
    
    log_posterior(
      mu,
      y
    )
  }
)


# Convert to relative density for plotting.

relative_density <- exp(
  log_posterior_values -
    max(
      log_posterior_values
    )
)


relative_density <-
  relative_density /
  sum(
    relative_density
  )


plot(
  posterior_grid,
  relative_density,
  type = "l",
  lwd = 2,
  xlab = expression(mu),
  ylab = "Relative Posterior Density",
  main = "Unnormalized Posterior"
)


# ------------------------------------------------------------------------------
# 15. Random-Walk Metropolis-Hastings
# ------------------------------------------------------------------------------

# Current state:
#
# theta_t
#
#
# Proposal:
#
# theta* ~ N(theta_t, proposal_sd^2)
#
#
# Acceptance probability:
#
# alpha =
#
# min(
#   1,
#   p(theta* | y) / p(theta_t | y)
# )
#
#
# Since proposal is symmetric, q terms cancel.


metropolis_hastings <- function(
    log_target,
    initial_value,
    iterations,
    proposal_sd,
    ...
) {
  
  samples <- numeric(
    iterations
  )
  
  
  samples[1] <- initial_value
  
  
  accepted <- logical(
    iterations
  )
  
  
  accepted[1] <- TRUE
  
  
  current <- initial_value
  
  
  current_log_density <- log_target(
    current,
    ...
  )
  
  
  for (
    iteration in 2:iterations
  ) {
    
    proposal <- rnorm(
      1,
      mean = current,
      sd = proposal_sd
    )
    
    
    proposal_log_density <- log_target(
      proposal,
      ...
    )
    
    
    log_acceptance_ratio <-
      proposal_log_density -
      current_log_density
    
    
    log_alpha <- min(
      0,
      log_acceptance_ratio
    )
    
    
    if (
      log(
        runif(
          1
        )
      ) <
      log_alpha
    ) {
      
      current <- proposal
      
      current_log_density <-
        proposal_log_density
      
      accepted[
        iteration
      ] <- TRUE
    }
    
    
    samples[
      iteration
    ] <- current
  }
  
  
  return(
    list(
      samples = samples,
      accepted = accepted,
      acceptance_rate = mean(
        accepted[
          -1
        ]
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 16. Run Metropolis-Hastings
# ------------------------------------------------------------------------------

set.seed(123)


iterations <- 20000


MH_model <- metropolis_hastings(
  log_target = log_posterior,
  initial_value = 0,
  iterations = iterations,
  proposal_sd = 0.5,
  y = y
)


MH_samples <- MH_model$samples


MH_model$acceptance_rate


# ------------------------------------------------------------------------------
# 17. Trace Plot
# ------------------------------------------------------------------------------

plot(
  seq_len(
    iterations
  ),
  MH_samples,
  type = "l",
  xlab = "Iteration",
  ylab = expression(mu),
  main = "Metropolis-Hastings Trace Plot"
)


abline(
  h = posterior_mean,
  lty = 2
)


# ------------------------------------------------------------------------------
# 18. Burn-In
# ------------------------------------------------------------------------------

# Early samples may reflect the arbitrary starting value.
#
# Discard an initial portion of the chain.


burn_in <- 2000


MH_post_burn <- MH_samples[
  (burn_in + 1):iterations
]


length(
  MH_post_burn
)


# ------------------------------------------------------------------------------
# 19. Trace Plot After Burn-In
# ------------------------------------------------------------------------------

plot(
  seq_along(
    MH_post_burn
  ),
  MH_post_burn,
  type = "l",
  xlab = "Post-Burn-In Iteration",
  ylab = expression(mu),
  main = "Trace Plot After Burn-In"
)


abline(
  h = posterior_mean,
  lty = 2
)


# ------------------------------------------------------------------------------
# 20. Posterior Histogram
# ------------------------------------------------------------------------------

hist(
  MH_post_burn,
  breaks = 50,
  probability = TRUE,
  xlab = expression(mu),
  main = "MCMC Approximation to Posterior"
)


lines(
  posterior_grid,
  posterior_density,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 21. Posterior Mean from MCMC
# ------------------------------------------------------------------------------

MH_mean <- mean(
  MH_post_burn
)


MH_mean


posterior_mean


# ------------------------------------------------------------------------------
# 22. Posterior SD from MCMC
# ------------------------------------------------------------------------------

MH_sd <- sd(
  MH_post_burn
)


MH_sd


posterior_sd


# ------------------------------------------------------------------------------
# 23. Posterior Credible Interval
# ------------------------------------------------------------------------------

MH_CI <- quantile(
  MH_post_burn,
  probs = c(
    0.025,
    0.975
  )
)


MH_CI


# Exact posterior interval

exact_CI <- qnorm(
  c(
    0.025,
    0.975
  ),
  mean = posterior_mean,
  sd = posterior_sd
)


exact_CI


# ------------------------------------------------------------------------------
# 24. Posterior Probability
# ------------------------------------------------------------------------------

MH_probability <- mean(
  MH_post_burn >
    3
)


exact_probability <-
  1 -
  pnorm(
    3,
    posterior_mean,
    posterior_sd
  )


data.frame(
  Method = c(
    "MCMC",
    "Exact"
  ),
  Probability_mu_Greater_3 = c(
    MH_probability,
    exact_probability
  )
)


# ==============================================================================
# PART III
#
# MARKOV-CHAIN DEPENDENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Consecutive MCMC Draws Are Dependent
# ------------------------------------------------------------------------------

plot(
  MH_post_burn[
    -length(
      MH_post_burn
    )
  ],
  MH_post_burn[
    -1
  ],
  pch = 19,
  cex = 0.4,
  xlab = expression(mu[t]),
  ylab = expression(mu[t + 1]),
  main = "Dependence Between Consecutive MCMC Samples"
)


# ------------------------------------------------------------------------------
# 26. Autocorrelation Function
# ------------------------------------------------------------------------------

acf(
  MH_post_burn,
  lag.max = 50,
  main = "MCMC Autocorrelation"
)


# ------------------------------------------------------------------------------
# 27. Manual Autocorrelation
# ------------------------------------------------------------------------------

sample_autocorrelation <- function(
    samples,
    lag
) {
  
  n <- length(
    samples
  )
  
  
  sample_mean <- mean(
    samples
  )
  
  
  numerator <- sum(
    (
      samples[
        1:(n - lag)
      ] -
        sample_mean
    ) *
      (
        samples[
          (lag + 1):n
        ] -
          sample_mean
      )
  )
  
  
  denominator <- sum(
    (
      samples -
        sample_mean
    )^2
  )
  
  
  numerator /
    denominator
}


lags <- 0:50


autocorrelations <- sapply(
  lags,
  function(lag) {
    
    if (
      lag == 0
    ) {
      
      1
      
    } else {
      
      sample_autocorrelation(
        MH_post_burn,
        lag
      )
    }
  }
)


plot(
  lags,
  autocorrelations,
  type = "h",
  xlab = "Lag",
  ylab = "Autocorrelation",
  main = "Manual Autocorrelation Function"
)


abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 28. Effective Sample Size
# ------------------------------------------------------------------------------

# For correlated MCMC samples:
#
# ESS approximately:
#
# M /
# (
#   1 + 2 sum_k rho_k
# )
#
# where rho_k are autocorrelations.
#
# In practice, truncate the sum when correlations become negligible
# or unstable.


effective_sample_size <- function(
    samples,
    max_lag = 1000
) {
  
  n <- length(
    samples
  )
  
  
  max_lag <- min(
    max_lag,
    n - 1
  )
  
  
  rho <- numeric(
    max_lag
  )
  
  
  for (
    lag in seq_len(
      max_lag
    )
  ) {
    
    rho[lag] <- sample_autocorrelation(
      samples,
      lag
    )
  }
  
  
  # Simple positive-sequence truncation
  
  first_negative <- which(
    rho <
      0
  )
  
  
  if (
    length(
      first_negative
    ) >
    0
  ) {
    
    cutoff <- first_negative[1] -
      1
    
  } else {
    
    cutoff <- max_lag
  }
  
  
  if (
    cutoff <= 0
  ) {
    
    integrated_autocorrelation_time <- 1
    
  } else {
    
    integrated_autocorrelation_time <-
      1 +
      2 *
      sum(
        rho[
          seq_len(
            cutoff
          )
        ]
      )
  }
  
  
  ESS <- n /
    integrated_autocorrelation_time
  
  
  return(
    list(
      ESS = ESS,
      IACT = integrated_autocorrelation_time,
      autocorrelations = rho
    )
  )
}


ESS_result <- effective_sample_size(
  MH_post_burn,
  max_lag = 500
)


ESS_result$ESS


ESS_result$IACT


# ------------------------------------------------------------------------------
# 29. Monte Carlo Standard Error for MCMC Mean
# ------------------------------------------------------------------------------

# Independent-sample SE would incorrectly use:
#
# sd / sqrt(M)
#
# For MCMC, replace M with effective sample size.


MCMC_MCSE <-
  sd(
    MH_post_burn
  ) /
  sqrt(
    ESS_result$ESS
  )


MCMC_MCSE


# ------------------------------------------------------------------------------
# 30. Compare Naive and ESS-Corrected MCSE
# ------------------------------------------------------------------------------

naive_MCSE <-
  sd(
    MH_post_burn
  ) /
  sqrt(
    length(
      MH_post_burn
    )
  )


data.frame(
  Method = c(
    "Naive Independent-Sample MCSE",
    "ESS-Corrected MCMC MCSE"
  ),
  MCSE = c(
    naive_MCSE,
    MCMC_MCSE
  )
)


# ==============================================================================
# PART IV
#
# PROPOSAL-SCALE TUNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Compare Proposal Standard Deviations
# ------------------------------------------------------------------------------

proposal_values <- c(
  0.01,
  0.05,
  0.1,
  0.25,
  0.5,
  1,
  2,
  5
)


proposal_results <- data.frame(
  Proposal_SD = proposal_values,
  Acceptance_Rate = NA,
  ESS = NA,
  Posterior_Mean = NA,
  Posterior_SD = NA
)


proposal_chains <- vector(
  "list",
  length(
    proposal_values
  )
)


for (
  i in seq_along(
    proposal_values
  )
) {
  
  set.seed(
    100 +
      i
  )
  
  
  chain_i <- metropolis_hastings(
    log_target = log_posterior,
    initial_value = 0,
    iterations = 15000,
    proposal_sd = proposal_values[i],
    y = y
  )
  
  
  samples_i <- chain_i$samples[
    2001:15000
  ]
  
  
  ESS_i <- effective_sample_size(
    samples_i,
    max_lag = 500
  )$ESS
  
  
  proposal_results$Acceptance_Rate[i] <-
    chain_i$acceptance_rate
  
  
  proposal_results$ESS[i] <-
    ESS_i
  
  
  proposal_results$Posterior_Mean[i] <-
    mean(
      samples_i
    )
  
  
  proposal_results$Posterior_SD[i] <-
    sd(
      samples_i
    )
  
  
  proposal_chains[[i]] <- samples_i
}


proposal_results


# ------------------------------------------------------------------------------
# 32. Acceptance Rate vs Proposal Scale
# ------------------------------------------------------------------------------

plot(
  proposal_results$Proposal_SD,
  proposal_results$Acceptance_Rate,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Proposal Standard Deviation",
  ylab = "Acceptance Rate",
  main = "Proposal Scale and Acceptance"
)


# ------------------------------------------------------------------------------
# 33. ESS vs Proposal Scale
# ------------------------------------------------------------------------------

plot(
  proposal_results$Proposal_SD,
  proposal_results$ESS,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Proposal Standard Deviation",
  ylab = "Effective Sample Size",
  main = "Proposal Scale and MCMC Efficiency"
)


# ------------------------------------------------------------------------------
# 34. Too-Small Proposal
# ------------------------------------------------------------------------------

small_proposal_chain <- proposal_chains[[1]]


plot(
  small_proposal_chain[
    1:2000
  ],
  type = "l",
  xlab = "Iteration",
  ylab = expression(mu),
  main = "Proposal Too Small"
)


# ------------------------------------------------------------------------------
# 35. Too-Large Proposal
# ------------------------------------------------------------------------------

large_proposal_chain <- proposal_chains[[length(proposal_chains)]]


plot(
  large_proposal_chain[
    1:2000
  ],
  type = "l",
  xlab = "Iteration",
  ylab = expression(mu),
  main = "Proposal Too Large"
)


# ==============================================================================
# PART V
#
# MULTIPLE CHAINS
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Run Multiple Chains from Different Starting Values
# ------------------------------------------------------------------------------

starting_values <- c(
  -10,
  -3,
  0,
  6,
  12
)


multiple_chains <- matrix(
  NA,
  nrow = 10000,
  ncol = length(
    starting_values
  )
)


acceptance_rates <- numeric(
  length(
    starting_values
  )
)


for (
  chain in seq_along(
    starting_values
  )
) {
  
  set.seed(
    500 +
      chain
  )
  
  
  result <- metropolis_hastings(
    log_target = log_posterior,
    initial_value = starting_values[chain],
    iterations = 10000,
    proposal_sd = 0.5,
    y = y
  )
  
  
  multiple_chains[
    ,
    chain
  ] <- result$samples
  
  
  acceptance_rates[
    chain
  ] <- result$acceptance_rate
}


# ------------------------------------------------------------------------------
# 37. Plot Multiple Chains
# ------------------------------------------------------------------------------

matplot(
  multiple_chains[
    1:2000,
    ,
    drop = FALSE
  ],
  type = "l",
  lty = 1:length(
    starting_values
  ),
  xlab = "Iteration",
  ylab = expression(mu),
  main = "Multiple MCMC Chains"
)


abline(
  h = posterior_mean,
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 38. Compare Post-Burn-In Chain Means
# ------------------------------------------------------------------------------

multi_burn <- 2000


chain_summaries <- data.frame(
  Starting_Value = starting_values,
  Acceptance_Rate = acceptance_rates,
  Mean = NA,
  SD = NA
)


for (
  chain in seq_along(
    starting_values
  )
) {
  
  samples_chain <- multiple_chains[
    (multi_burn + 1):nrow(
      multiple_chains
    ),
    chain
  ]
  
  
  chain_summaries$Mean[chain] <-
    mean(
      samples_chain
    )
  
  
  chain_summaries$SD[chain] <-
    sd(
      samples_chain
    )
}


chain_summaries


# ------------------------------------------------------------------------------
# 39. Simple Gelman-Rubin R-hat
# ------------------------------------------------------------------------------

# Multiple chains should agree if they are exploring the same stationary
# distribution.
#
# This is a basic classical R-hat calculation.


rhat <- function(
    chains
) {
  
  chains <- as.matrix(
    chains
  )
  
  
  n <- nrow(
    chains
  )
  
  
  m <- ncol(
    chains
  )
  
  
  chain_means <- colMeans(
    chains
  )
  
  
  chain_variances <- apply(
    chains,
    2,
    var
  )
  
  
  W <- mean(
    chain_variances
  )
  
  
  B <- n *
    var(
      chain_means
    )
  
  
  variance_hat <-
    (
      n -
        1
    ) /
    n *
    W +
    B /
    n
  
  
  R_hat <- sqrt(
    variance_hat /
      W
  )
  
  
  return(
    R_hat
  )
}


post_burn_chains <- multiple_chains[
  (multi_burn + 1):nrow(
    multiple_chains
  ),
  ,
  drop = FALSE
]


R_hat <- rhat(
  post_burn_chains
)


R_hat


# ==============================================================================
# PART VI
#
# GIBBS SAMPLING
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Unknown Mean and Variance
# ------------------------------------------------------------------------------

# Now consider:
#
# y_i | mu, sigma^2 ~ N(mu, sigma^2)
#
#
# Prior:
#
# mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0)
#
# sigma^2 ~ Inverse-Gamma(alpha_0, beta_0)
#
#
# These priors are conjugate, so each full conditional has a standard form.


mu_0_gibbs <- 0

kappa_0 <- 0.01

alpha_0 <- 2

beta_0 <- 2


# ------------------------------------------------------------------------------
# 41. Inverse-Gamma Random Generator
# ------------------------------------------------------------------------------

# If:
#
# X ~ Gamma(alpha, rate = beta)
#
# then:
#
# 1/X ~ Inverse-Gamma(alpha, beta)


rinvgamma <- function(
    n,
    shape,
    rate
) {
  
  1 /
    rgamma(
      n,
      shape = shape,
      rate = rate
    )
}


# ------------------------------------------------------------------------------
# 42. Gibbs Sampler
# ------------------------------------------------------------------------------

gibbs_normal <- function(
    y,
    iterations = 20000,
    mu_0 = 0,
    kappa_0 = 0.01,
    alpha_0 = 2,
    beta_0 = 2,
    initial_mu = 0,
    initial_sigma2 = 1
) {
  
  n <- length(
    y
  )
  
  
  y_bar <- mean(
    y
  )
  
  
  mu_samples <- numeric(
    iterations
  )
  
  
  sigma2_samples <- numeric(
    iterations
  )
  
  
  mu_samples[1] <- initial_mu
  
  sigma2_samples[1] <- initial_sigma2
  
  
  for (
    iteration in 2:iterations
  ) {
    
    # ------------------------------------------------------------------------
    # Sample mu | sigma^2, y
    # ------------------------------------------------------------------------
    
    kappa_n <-
      kappa_0 +
      n
    
    
    mu_n <-
      (
        kappa_0 *
          mu_0 +
          n *
          y_bar
      ) /
      kappa_n
    
    
    mu_variance <-
      sigma2_samples[
        iteration - 1
      ] /
      kappa_n
    
    
    mu_samples[
      iteration
    ] <- rnorm(
      1,
      mean = mu_n,
      sd = sqrt(
        mu_variance
      )
    )
    
    
    # ------------------------------------------------------------------------
    # Sample sigma^2 | mu, y
    # ------------------------------------------------------------------------
    
    alpha_n <-
      alpha_0 +
      (
        n +
          1
      ) /
      2
    
    
    beta_n <-
      beta_0 +
      0.5 *
      sum(
        (
          y -
            mu_samples[
              iteration
            ]
        )^2
      ) +
      0.5 *
      kappa_0 *
      (
        mu_samples[
          iteration
        ] -
          mu_0
      )^2
    
    
    sigma2_samples[
      iteration
    ] <- rinvgamma(
      1,
      shape = alpha_n,
      rate = beta_n
    )
  }
  
  
  return(
    list(
      mu = mu_samples,
      sigma2 = sigma2_samples
    )
  )
}


# ------------------------------------------------------------------------------
# 43. Run Gibbs Sampler
# ------------------------------------------------------------------------------

set.seed(123)


gibbs_model <- gibbs_normal(
  y = y,
  iterations = 20000,
  mu_0 = mu_0_gibbs,
  kappa_0 = kappa_0,
  alpha_0 = alpha_0,
  beta_0 = beta_0,
  initial_mu = 0,
  initial_sigma2 = 4
)


# ------------------------------------------------------------------------------
# 44. Gibbs Trace Plots
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    2,
    1
  )
)


plot(
  gibbs_model$mu,
  type = "l",
  xlab = "Iteration",
  ylab = expression(mu),
  main = "Gibbs Sampler: Mean"
)


plot(
  gibbs_model$sigma2,
  type = "l",
  xlab = "Iteration",
  ylab = expression(sigma^2),
  main = "Gibbs Sampler: Variance"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 45. Remove Burn-In
# ------------------------------------------------------------------------------

gibbs_burn <- 2000


gibbs_mu <- gibbs_model$mu[
  (gibbs_burn + 1):length(
    gibbs_model$mu
  )
]


gibbs_sigma2 <- gibbs_model$sigma2[
  (gibbs_burn + 1):length(
    gibbs_model$sigma2
  )
]


# ------------------------------------------------------------------------------
# 46. Gibbs Posterior Summaries
# ------------------------------------------------------------------------------

data.frame(
  Parameter = c(
    "mu",
    "sigma^2"
  ),
  
  Posterior_Mean = c(
    mean(
      gibbs_mu
    ),
    mean(
      gibbs_sigma2
    )
  ),
  
  Posterior_SD = c(
    sd(
      gibbs_mu
    ),
    sd(
      gibbs_sigma2
    )
  ),
  
  Lower_95 = c(
    quantile(
      gibbs_mu,
      0.025
    ),
    quantile(
      gibbs_sigma2,
      0.025
    )
  ),
  
  Upper_95 = c(
    quantile(
      gibbs_mu,
      0.975
    ),
    quantile(
      gibbs_sigma2,
      0.975
    )
  )
)


# ------------------------------------------------------------------------------
# 47. Gibbs Marginal Posterior Histograms
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    2
  )
)


hist(
  gibbs_mu,
  breaks = 50,
  probability = TRUE,
  xlab = expression(mu),
  main = "Posterior of Mean"
)


abline(
  v = mu_true,
  lty = 2
)


hist(
  gibbs_sigma2,
  breaks = 50,
  probability = TRUE,
  xlab = expression(sigma^2),
  main = "Posterior of Variance"
)


abline(
  v = sigma^2,
  lty = 2
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 48. Joint Posterior Samples
# ------------------------------------------------------------------------------

plot(
  gibbs_mu,
  gibbs_sigma2,
  pch = 19,
  cex = 0.3,
  xlab = expression(mu),
  ylab = expression(sigma^2),
  main = "Joint Gibbs Samples"
)


# ------------------------------------------------------------------------------
# 49. Gibbs Autocorrelation
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    2
  )
)


acf(
  gibbs_mu,
  lag.max = 50,
  main = "ACF: mu"
)


acf(
  gibbs_sigma2,
  lag.max = 50,
  main = expression(
    paste(
      "ACF: ",
      sigma^2
    )
  )
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 50. Gibbs Effective Sample Sizes
# ------------------------------------------------------------------------------

ESS_mu <- effective_sample_size(
  gibbs_mu,
  max_lag = 500
)$ESS


ESS_sigma2 <- effective_sample_size(
  gibbs_sigma2,
  max_lag = 500
)$ESS


data.frame(
  Parameter = c(
    "mu",
    "sigma^2"
  ),
  Samples = c(
    length(
      gibbs_mu
    ),
    length(
      gibbs_sigma2
    )
  ),
  Effective_Sample_Size = c(
    ESS_mu,
    ESS_sigma2
  )
)


# ==============================================================================
# PART VII
#
# THINNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Thinning
# ------------------------------------------------------------------------------

# Thinning keeps every k-th draw.
#
# Historically this was often used to reduce autocorrelation.
#
# However, thinning usually throws away useful information.
#
# It is generally preferable to retain all draws unless storage or
# computational constraints provide a specific reason to thin.


thin_by <- 10


thinned_samples <- MH_post_burn[
  seq(
    1,
    length(
      MH_post_burn
    ),
    by = thin_by
  )
]


length(
  thinned_samples
)


# ------------------------------------------------------------------------------
# 52. Compare Full and Thinned Chains
# ------------------------------------------------------------------------------

ESS_full <- effective_sample_size(
  MH_post_burn,
  max_lag = 500
)$ESS


ESS_thinned <- effective_sample_size(
  thinned_samples,
  max_lag = 500
)$ESS


data.frame(
  Chain = c(
    "Full",
    "Thinned"
  ),
  Number_Samples = c(
    length(
      MH_post_burn
    ),
    length(
      thinned_samples
    )
  ),
  ESS = c(
    ESS_full,
    ESS_thinned
  )
)


# ==============================================================================
# PART VIII
#
# RUNNING ESTIMATES
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. Running Posterior Mean
# ------------------------------------------------------------------------------

running_mean <- cumsum(
  MH_post_burn
) /
  seq_along(
    MH_post_burn
  )


plot(
  running_mean,
  type = "l",
  xlab = "Iteration",
  ylab = "Running Posterior Mean",
  main = "Monte Carlo Convergence"
)


abline(
  h = posterior_mean,
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 54. Running Posterior Probability
# ------------------------------------------------------------------------------

running_probability <- cumsum(
  MH_post_burn >
    3
) /
  seq_along(
    MH_post_burn
  )


plot(
  running_probability,
  type = "l",
  xlab = "Iteration",
  ylab = expression(
    P(
      mu >
        3 ~ "|" ~ y
    )
  ),
  main = "Running Posterior Probability"
)


abline(
  h = exact_probability,
  lty = 2,
  lwd = 2
)


# ==============================================================================
# PART IX
#
# POSTERIOR PREDICTIVE DISTRIBUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Posterior Predictive Sampling
# ------------------------------------------------------------------------------

# If:
#
# y_new | mu ~ N(mu, sigma^2)
#
# then posterior predictive samples can be generated by:
#
# 1. Draw mu^(m) from posterior.
# 2. Draw y_new^(m) ~ N(mu^(m), sigma^2).


set.seed(123)


posterior_predictive <- rnorm(
  length(
    MH_post_burn
  ),
  mean = MH_post_burn,
  sd = sigma
)


# ------------------------------------------------------------------------------
# 56. Plot Posterior Predictive Distribution
# ------------------------------------------------------------------------------

hist(
  posterior_predictive,
  breaks = 50,
  probability = TRUE,
  xlab = expression(
    y[new]
  ),
  main = "Posterior Predictive Distribution"
)


# ------------------------------------------------------------------------------
# 57. Posterior Predictive Interval
# ------------------------------------------------------------------------------

predictive_interval <- quantile(
  posterior_predictive,
  c(
    0.025,
    0.975
  )
)


predictive_interval


# ------------------------------------------------------------------------------
# 58. Mean Uncertainty vs Predictive Uncertainty
# ------------------------------------------------------------------------------

parameter_interval <- quantile(
  MH_post_burn,
  c(
    0.025,
    0.975
  )
)


data.frame(
  Quantity = c(
    "Posterior Mean Parameter",
    "Future Observation"
  ),
  
  Lower = c(
    parameter_interval[
      1
    ],
    predictive_interval[
      1
    ]
  ),
  
  Upper = c(
    parameter_interval[
      2
    ],
    predictive_interval[
      2
    ]
  )
)


# ==============================================================================
# PART X
#
# FINAL COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 59. Compare Exact, Direct Monte Carlo, and MCMC
# ------------------------------------------------------------------------------

comparison <- data.frame(
  Method = c(
    "Exact Posterior",
    "Direct Independent Monte Carlo",
    "Metropolis-Hastings"
  ),
  
  Mean = c(
    posterior_mean,
    mean(
      direct_samples
    ),
    mean(
      MH_post_burn
    )
  ),
  
  SD = c(
    posterior_sd,
    sd(
      direct_samples
    ),
    sd(
      MH_post_burn
    )
  )
)


comparison


# ------------------------------------------------------------------------------
# 60. Final Summary
# ------------------------------------------------------------------------------

cat(
  "MCMC Sampling Summary\n"
)


cat(
  "---------------------\n"
)


cat(
  "True mu:",
  mu_true,
  "\n"
)


cat(
  "Exact posterior mean:",
  round(
    posterior_mean,
    4
  ),
  "\n"
)


cat(
  "Metropolis-Hastings posterior mean:",
  round(
    MH_mean,
    4
  ),
  "\n"
)


cat(
  "Exact posterior SD:",
  round(
    posterior_sd,
    4
  ),
  "\n"
)


cat(
  "Metropolis-Hastings posterior SD:",
  round(
    MH_sd,
    4
  ),
  "\n"
)


cat(
  "MH acceptance rate:",
  round(
    MH_model$acceptance_rate,
    4
  ),
  "\n"
)


cat(
  "MH effective sample size:",
  round(
    ESS_result$ESS,
    1
  ),
  "\n"
)


cat(
  "MCMC Monte Carlo standard error:",
  round(
    MCMC_MCSE,
    6
  ),
  "\n"
)


cat(
  "Multiple-chain R-hat:",
  round(
    R_hat,
    4
  ),
  "\n"
)


cat(
  "Gibbs posterior mean of mu:",
  round(
    mean(
      gibbs_mu
    ),
    4
  ),
  "\n"
)


cat(
  "Gibbs posterior mean of sigma^2:",
  round(
    mean(
      gibbs_sigma2
    ),
    4
  ),
  "\n"
)