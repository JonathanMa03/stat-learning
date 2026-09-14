# ==============================================================================
# Bootstrapping
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Nonparametric bootstrap
#   - Sampling with replacement
#   - Bootstrap sampling distribution
#   - Bootstrap standard error
#   - Bootstrap bias
#   - Percentile confidence interval
#   - Basic bootstrap confidence interval
#   - Bootstrap for regression coefficients
#   - Bootstrap for nonlinear statistics
#   - Monte Carlo stability as B increases
#
# Core idea:
#
#   Original sample:
#
#       Z = {z1, ..., zn}
#
#   Bootstrap sample:
#
#       Z* = sample(Z, n, replace = TRUE)
#
#   Repeat B times:
#
#       theta_hat^(1), ..., theta_hat^(B)
#
#   Use the empirical distribution of bootstrap estimates to approximate
#   the sampling distribution of theta_hat.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 150


x <- rnorm(
  n,
  mean = 0,
  sd = 1
)


epsilon <- rnorm(
  n,
  mean = 0,
  sd = 2
)


y <- 2 +
  3 * x +
  epsilon


data <- data.frame(
  x = x,
  y = y
)


# ------------------------------------------------------------------------------
# 2. Explore Data
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Bootstrap Regression Example"
)


abline(
  lm(
    y ~ x
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Fit Original Linear Regression Manually
# ------------------------------------------------------------------------------

X <- cbind(
  Intercept = 1,
  x = x
)


beta_hat <- solve(
  crossprod(
    X
  ),
  crossprod(
    X,
    y
  )
)


beta_hat <- as.vector(
  beta_hat
)


names(
  beta_hat
) <- c(
  "Intercept",
  "Slope"
)


beta_hat


# ------------------------------------------------------------------------------
# 4. Original Statistic of Interest
# ------------------------------------------------------------------------------

# We will start by bootstrapping the slope.


slope_hat <- beta_hat[
  "Slope"
]


slope_hat


# ------------------------------------------------------------------------------
# 5. One Bootstrap Sample
# ------------------------------------------------------------------------------

set.seed(123)


bootstrap_index <- sample(
  seq_len(n),
  size = n,
  replace = TRUE
)


bootstrap_index[
  1:20
]


# ------------------------------------------------------------------------------
# 6. Inspect Duplicates in Bootstrap Sample
# ------------------------------------------------------------------------------

bootstrap_counts <- table(
  bootstrap_index
)


head(
  bootstrap_counts
)


length(
  unique(
    bootstrap_index
  )
)


# ------------------------------------------------------------------------------
# 7. Fit One Bootstrap Regression
# ------------------------------------------------------------------------------

x_boot <- x[
  bootstrap_index
]


y_boot <- y[
  bootstrap_index
]


X_boot <- cbind(
  Intercept = 1,
  x = x_boot
)


beta_boot <- solve(
  crossprod(
    X_boot
  ),
  crossprod(
    X_boot,
    y_boot
  )
)


beta_boot <- as.vector(
  beta_boot
)


beta_boot


# ------------------------------------------------------------------------------
# 8. Bootstrap Function for Regression Coefficients
# ------------------------------------------------------------------------------

bootstrap_regression <- function(
    x,
    y,
    B = 1000,
    seed = NULL
) {
  
  if (
    !is.null(
      seed
    )
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  n <- length(
    y
  )
  
  
  estimates <- matrix(
    NA,
    nrow = B,
    ncol = 2
  )
  
  
  colnames(
    estimates
  ) <- c(
    "Intercept",
    "Slope"
  )
  
  
  for (
    b in seq_len(
      B
    )
  ) {
    
    index <- sample(
      seq_len(n),
      size = n,
      replace = TRUE
    )
    
    
    x_b <- x[
      index
    ]
    
    
    y_b <- y[
      index
    ]
    
    
    X_b <- cbind(
      1,
      x_b
    )
    
    
    beta_b <- qr.solve(
      X_b,
      y_b
    )
    
    
    estimates[
      b,
    ] <- beta_b
  }
  
  
  return(
    estimates
  )
}


# ------------------------------------------------------------------------------
# 9. Run Bootstrap
# ------------------------------------------------------------------------------

B <- 2000


bootstrap_estimates <- bootstrap_regression(
  x = x,
  y = y,
  B = B,
  seed = 123
)


head(
  bootstrap_estimates
)


# ------------------------------------------------------------------------------
# 10. Bootstrap Sampling Distribution of Slope
# ------------------------------------------------------------------------------

bootstrap_slopes <- bootstrap_estimates[
  ,
  "Slope"
]


hist(
  bootstrap_slopes,
  breaks = 40,
  xlab = "Bootstrap Slope",
  main = "Bootstrap Sampling Distribution of Slope"
)


abline(
  v = slope_hat,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 11. Bootstrap Standard Error
# ------------------------------------------------------------------------------

# Estimated standard error:
#
# SE_boot(theta_hat)
#
# =
# standard deviation of bootstrap estimates


bootstrap_SE_slope <- sd(
  bootstrap_slopes
)


bootstrap_SE_slope


# ------------------------------------------------------------------------------
# 12. Bootstrap Mean
# ------------------------------------------------------------------------------

bootstrap_mean_slope <- mean(
  bootstrap_slopes
)


bootstrap_mean_slope


# ------------------------------------------------------------------------------
# 13. Bootstrap Bias Estimate
# ------------------------------------------------------------------------------

# Estimated bias:
#
# Bias_boot =
#
# mean(theta_hat*) - theta_hat


bootstrap_bias_slope <-
  bootstrap_mean_slope -
  slope_hat


bootstrap_bias_slope


# ------------------------------------------------------------------------------
# 14. Bias-Corrected Point Estimate
# ------------------------------------------------------------------------------

# Simple bootstrap bias correction:
#
# theta_bc =
#
# theta_hat - Bias_boot
#
# =
#
# 2 theta_hat - mean(theta_hat*)


bias_corrected_slope <-
  slope_hat -
  bootstrap_bias_slope


bias_corrected_slope


# ------------------------------------------------------------------------------
# 15. Percentile Bootstrap Confidence Interval
# ------------------------------------------------------------------------------

# 95% percentile interval:
#
# empirical 2.5% and 97.5% quantiles of bootstrap estimates


percentile_CI <- quantile(
  bootstrap_slopes,
  probs = c(
    0.025,
    0.975
  )
)


percentile_CI


# ------------------------------------------------------------------------------
# 16. Basic Bootstrap Confidence Interval
# ------------------------------------------------------------------------------

# If:
#
# q_low  = 2.5% bootstrap quantile
# q_high = 97.5% bootstrap quantile
#
# basic CI:
#
# [2 theta_hat - q_high,
#  2 theta_hat - q_low]


basic_CI <- c(
  2 * slope_hat -
    percentile_CI[
      2
    ],
  
  2 * slope_hat -
    percentile_CI[
      1
    ]
)


names(
  basic_CI
) <- c(
  "2.5%",
  "97.5%"
)


basic_CI


# ------------------------------------------------------------------------------
# 17. Normal Bootstrap Confidence Interval
# ------------------------------------------------------------------------------

normal_bootstrap_CI <- c(
  slope_hat -
    qnorm(
      0.975
    ) *
    bootstrap_SE_slope,
  
  slope_hat +
    qnorm(
      0.975
    ) *
    bootstrap_SE_slope
)


normal_bootstrap_CI


# ------------------------------------------------------------------------------
# 18. Compare with Classical Regression Standard Error
# ------------------------------------------------------------------------------

model_builtin <- lm(
  y ~ x
)


summary(
  model_builtin
)


classical_SE_slope <- summary(
  model_builtin
)$coefficients[
  "x",
  "Std. Error"
]


classical_SE_slope


data.frame(
  Method = c(
    "Classical OLS",
    "Bootstrap"
  ),
  Standard_Error = c(
    classical_SE_slope,
    bootstrap_SE_slope
  )
)


# ------------------------------------------------------------------------------
# 19. Compare Confidence Intervals
# ------------------------------------------------------------------------------

classical_CI <- confint(
  model_builtin
)[
  "x",
]


CI_comparison <- data.frame(
  Method = c(
    "Classical",
    "Bootstrap Normal",
    "Bootstrap Percentile",
    "Bootstrap Basic"
  ),
  
  Lower = c(
    classical_CI[
      1
    ],
    normal_bootstrap_CI[
      1
    ],
    percentile_CI[
      1
    ],
    basic_CI[
      1
    ]
  ),
  
  Upper = c(
    classical_CI[
      2
    ],
    normal_bootstrap_CI[
      2
    ],
    percentile_CI[
      2
    ],
    basic_CI[
      2
    ]
  )
)


CI_comparison


# ------------------------------------------------------------------------------
# 20. Bootstrap Both Regression Coefficients
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    2
  )
)


hist(
  bootstrap_estimates[
    ,
    "Intercept"
  ],
  breaks = 40,
  main = "Bootstrap Intercept",
  xlab = "Intercept"
)


hist(
  bootstrap_estimates[
    ,
    "Slope"
  ],
  breaks = 40,
  main = "Bootstrap Slope",
  xlab = "Slope"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 21. Bootstrap Covariance Matrix
# ------------------------------------------------------------------------------

bootstrap_covariance <- cov(
  bootstrap_estimates
)


bootstrap_covariance


# ------------------------------------------------------------------------------
# 22. Bootstrap Correlation Between Estimates
# ------------------------------------------------------------------------------

cor(
  bootstrap_estimates
)


# ------------------------------------------------------------------------------
# 23. Scatterplot of Bootstrap Coefficients
# ------------------------------------------------------------------------------

plot(
  bootstrap_estimates[
    ,
    "Intercept"
  ],
  bootstrap_estimates[
    ,
    "Slope"
  ],
  pch = 19,
  cex = 0.5,
  xlab = "Intercept",
  ylab = "Slope",
  main = "Joint Bootstrap Distribution"
)


points(
  beta_hat[
    "Intercept"
  ],
  beta_hat[
    "Slope"
  ],
  pch = 19,
  cex = 1.5
)


# ------------------------------------------------------------------------------
# 24. Generic Nonparametric Bootstrap
# ------------------------------------------------------------------------------

# A reusable bootstrap function:
#
# data      = data object
# statistic = function that calculates theta_hat
# B         = number of bootstrap samples


bootstrap <- function(
    data,
    statistic,
    B = 1000,
    seed = NULL
) {
  
  if (
    !is.null(
      seed
    )
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  n <- nrow(
    data
  )
  
  
  original_estimate <- statistic(
    data
  )
  
  
  bootstrap_estimates <- replicate(
    B,
    {
      
      index <- sample(
        seq_len(n),
        size = n,
        replace = TRUE
      )
      
      
      bootstrap_sample <- data[
        index,
        ,
        drop = FALSE
      ]
      
      
      statistic(
        bootstrap_sample
      )
    }
  )
  
  
  return(
    list(
      original = original_estimate,
      estimates = bootstrap_estimates
    )
  )
}


# ------------------------------------------------------------------------------
# 25. Use Generic Bootstrap for Slope
# ------------------------------------------------------------------------------

slope_statistic <- function(
    data
) {
  
  model <- lm(
    y ~ x,
    data = data
  )
  
  
  coef(
    model
  )[
    "x"
  ]
}


generic_slope_bootstrap <- bootstrap(
  data = data,
  statistic = slope_statistic,
  B = 2000,
  seed = 123
)


generic_slope_bootstrap$original


sd(
  generic_slope_bootstrap$estimates
)


# ------------------------------------------------------------------------------
# 26. Bootstrap a Statistic Without a Simple Formula
# ------------------------------------------------------------------------------

# One major advantage of the bootstrap:
#
# the statistic does not need a convenient analytic standard error.
#
# Example:
#
# correlation between x and y.


correlation_statistic <- function(
    data
) {
  
  cor(
    data$x,
    data$y
  )
}


cor_bootstrap <- bootstrap(
  data = data,
  statistic = correlation_statistic,
  B = 2000,
  seed = 123
)


original_correlation <- cor_bootstrap$original


bootstrap_correlations <- cor_bootstrap$estimates


# ------------------------------------------------------------------------------
# 27. Bootstrap Distribution of Correlation
# ------------------------------------------------------------------------------

hist(
  bootstrap_correlations,
  breaks = 40,
  xlab = "Correlation",
  main = "Bootstrap Distribution of Correlation"
)


abline(
  v = original_correlation,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 28. Correlation Bootstrap SE and CI
# ------------------------------------------------------------------------------

correlation_SE <- sd(
  bootstrap_correlations
)


correlation_CI <- quantile(
  bootstrap_correlations,
  c(
    0.025,
    0.975
  )
)


data.frame(
  Estimate = original_correlation,
  Bootstrap_SE = correlation_SE,
  Lower = correlation_CI[
    1
  ],
  Upper = correlation_CI[
    2
  ]
)


# ------------------------------------------------------------------------------
# 29. Bootstrap the Median
# ------------------------------------------------------------------------------

median_statistic <- function(
    data
) {
  
  median(
    data$y
  )
}


median_bootstrap <- bootstrap(
  data = data,
  statistic = median_statistic,
  B = 2000,
  seed = 123
)


median_bootstrap_SE <- sd(
  median_bootstrap$estimates
)


median_bootstrap_CI <- quantile(
  median_bootstrap$estimates,
  c(
    0.025,
    0.975
  )
)


data.frame(
  Estimate = median_bootstrap$original,
  SE = median_bootstrap_SE,
  Lower = median_bootstrap_CI[
    1
  ],
  Upper = median_bootstrap_CI[
    2
  ]
)


# ------------------------------------------------------------------------------
# 30. Visualize Bootstrap Distribution of Median
# ------------------------------------------------------------------------------

hist(
  median_bootstrap$estimates,
  breaks = 40,
  xlab = "Bootstrap Median",
  main = "Bootstrap Distribution of Median"
)


abline(
  v = median_bootstrap$original,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 31. Number of Unique Observations in Bootstrap Samples
# ------------------------------------------------------------------------------

# A bootstrap sample contains n draws, but because sampling is with replacement,
# it usually contains fewer than n unique observations.


set.seed(123)


B_unique <- 1000


unique_counts <- numeric(
  B_unique
)


for (
  b in seq_len(
    B_unique
  )
) {
  
  index <- sample(
    seq_len(n),
    size = n,
    replace = TRUE
  )
  
  
  unique_counts[b] <- length(
    unique(
      index
    )
  )
}


mean(
  unique_counts
)


mean(
  unique_counts
) /
  n


# ------------------------------------------------------------------------------
# 32. Theoretical Fraction of Unique Observations
# ------------------------------------------------------------------------------

# Probability an observation is not selected in one bootstrap sample:
#
# (1 - 1/n)^n
#
# As n -> infinity:
#
# approximately exp(-1) = 0.3679
#
# Therefore approximately:
#
# 1 - exp(-1) = 0.6321
#
# of observations appear at least once.


theoretical_unique_fraction <-
  1 -
  (
    1 -
      1 /
      n
  )^n


theoretical_unique_fraction


1 -
  exp(
    -1
  )


# ------------------------------------------------------------------------------
# 33. Plot Number of Unique Observations
# ------------------------------------------------------------------------------

hist(
  unique_counts,
  breaks = 25,
  xlab = "Number of Unique Observations",
  main = "Unique Observations in Bootstrap Samples"
)


abline(
  v = n *
    (
      1 -
        exp(
          -1
        )
    ),
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 34. Out-of-Bag Observations
# ------------------------------------------------------------------------------

# Observations not selected in a bootstrap sample are called
# out-of-bag observations.


set.seed(123)


index <- sample(
  seq_len(n),
  n,
  replace = TRUE
)


OOB_index <- setdiff(
  seq_len(n),
  unique(
    index
  )
)


length(
  OOB_index
)


length(
  OOB_index
) /
  n


# ------------------------------------------------------------------------------
# 35. Bootstrap Stability as B Increases
# ------------------------------------------------------------------------------

# Bootstrap estimates themselves have Monte Carlo error.
#
# Increasing B stabilizes the estimate.


B_values <- c(
  50,
  100,
  250,
  500,
  1000,
  2000,
  5000
)


SE_by_B <- numeric(
  length(
    B_values
  )
)


for (
  i in seq_along(
    B_values
  )
) {
  
  estimates_i <- bootstrap_regression(
    x,
    y,
    B = B_values[i],
    seed = 123
  )
  
  
  SE_by_B[i] <- sd(
    estimates_i[
      ,
      "Slope"
    ]
  )
}


# ------------------------------------------------------------------------------
# 36. Plot Stability vs Number of Bootstrap Replications
# ------------------------------------------------------------------------------

plot(
  B_values,
  SE_by_B,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Number of Bootstrap Replications",
  ylab = "Estimated Bootstrap SE",
  main = "Bootstrap Monte Carlo Stability"
)


# ------------------------------------------------------------------------------
# 37. Cumulative Bootstrap SE
# ------------------------------------------------------------------------------

cumulative_SE <- numeric(
  B
)


for (
  b in seq_len(
    B
  )
) {
  
  if (
    b < 2
  ) {
    
    cumulative_SE[b] <- NA
    
  } else {
    
    cumulative_SE[b] <- sd(
      bootstrap_slopes[
        seq_len(
          b
        )
      ]
    )
  }
}


plot(
  seq_len(B),
  cumulative_SE,
  type = "l",
  xlab = "Bootstrap Replications",
  ylab = "Cumulative SE Estimate",
  main = "Convergence of Bootstrap Standard Error"
)


abline(
  h = bootstrap_SE_slope,
  lty = 2
)


# ------------------------------------------------------------------------------
# 38. Compare Bootstrap with Repeated Sampling
# ------------------------------------------------------------------------------

# In simulation, we know the actual population generating process.
#
# Therefore we can compare:
#
#   bootstrap approximation
#
# with:
#
#   the true sampling distribution generated from many independent datasets.


set.seed(999)


number_simulations <- 2000


simulation_slopes <- numeric(
  number_simulations
)


for (
  simulation in seq_len(
    number_simulations
  )
) {
  
  x_sim <- rnorm(
    n
  )
  
  
  y_sim <- 2 +
    3 *
    x_sim +
    rnorm(
      n,
      sd = sigma
    )
  
  
  X_sim <- cbind(
    1,
    x_sim
  )
  
  
  simulation_slopes[
    simulation
  ] <- qr.solve(
    X_sim,
    y_sim
  )[
    2
  ]
}


# ------------------------------------------------------------------------------
# 39. Compare True and Bootstrap Sampling Distributions
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    2
  )
)


hist(
  simulation_slopes,
  breaks = 40,
  xlab = "Slope",
  main = "True Sampling Distribution"
)


abline(
  v = 3,
  lwd = 2
)


hist(
  bootstrap_slopes,
  breaks = 40,
  xlab = "Slope",
  main = "Bootstrap Approximation"
)


abline(
  v = slope_hat,
  lwd = 2
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 40. Compare Standard Deviations
# ------------------------------------------------------------------------------

true_sampling_SE <- sd(
  simulation_slopes
)


data.frame(
  Quantity = c(
    "True Sampling SE",
    "Bootstrap SE",
    "Classical OLS SE"
  ),
  
  Value = c(
    true_sampling_SE,
    bootstrap_SE_slope,
    classical_SE_slope
  )
)


# ------------------------------------------------------------------------------
# 41. Residual Bootstrap
# ------------------------------------------------------------------------------

# For regression, another option is the residual bootstrap.
#
# Instead of resampling (x_i, y_i) pairs:
#
#   1. fit regression
#   2. keep X fixed
#   3. resample residuals
#   4. construct new y*
#
#       y* = y_hat + residual*
#
# This assumes the regression model is correctly specified and errors are
# exchangeable / approximately homoskedastic.


original_model <- lm(
  y ~ x
)


original_fitted <- fitted(
  original_model
)


original_residuals <- residuals(
  original_model
)


centered_residuals <- original_residuals -
  mean(
    original_residuals
  )


# ------------------------------------------------------------------------------
# 42. Residual Bootstrap Function
# ------------------------------------------------------------------------------

residual_bootstrap <- function(
    x,
    y,
    B = 1000,
    seed = NULL
) {
  
  if (
    !is.null(
      seed
    )
  ) {
    
    set.seed(
      seed
    )
  }
  
  
  original_model <- lm(
    y ~ x
  )
  
  
  fitted_values <- fitted(
    original_model
  )
  
  
  residual_values <- residuals(
    original_model
  )
  
  
  residual_values <- residual_values -
    mean(
      residual_values
    )
  
  
  estimates <- matrix(
    NA,
    nrow = B,
    ncol = 2
  )
  
  
  colnames(
    estimates
  ) <- c(
    "Intercept",
    "Slope"
  )
  
  
  for (
    b in seq_len(
      B
    )
  ) {
    
    sampled_residuals <- sample(
      residual_values,
      size = length(
        residual_values
      ),
      replace = TRUE
    )
    
    
    y_boot <- fitted_values +
      sampled_residuals
    
    
    model_boot <- lm(
      y_boot ~ x
    )
    
    
    estimates[
      b,
    ] <- coef(
      model_boot
    )
  }
  
  
  return(
    estimates
  )
}


# ------------------------------------------------------------------------------
# 43. Run Residual Bootstrap
# ------------------------------------------------------------------------------

residual_estimates <- residual_bootstrap(
  x,
  y,
  B = 2000,
  seed = 123
)


residual_bootstrap_SE <- sd(
  residual_estimates[
    ,
    "Slope"
  ]
)


data.frame(
  Method = c(
    "Pairs Bootstrap",
    "Residual Bootstrap",
    "Classical OLS"
  ),
  
  Slope_SE = c(
    bootstrap_SE_slope,
    residual_bootstrap_SE,
    classical_SE_slope
  )
)


# ------------------------------------------------------------------------------
# 44. Pairs Bootstrap vs Residual Bootstrap
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    1,
    2
  )
)


hist(
  bootstrap_slopes,
  breaks = 40,
  main = "Pairs Bootstrap",
  xlab = "Slope"
)


hist(
  residual_estimates[
    ,
    "Slope"
  ],
  breaks = 40,
  main = "Residual Bootstrap",
  xlab = "Slope"
)


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 45. Bootstrap Prediction at a New x
# ------------------------------------------------------------------------------

# Bootstrap can also quantify uncertainty in a fitted quantity.
#
# Example:
#
# estimated conditional mean at x0.


x0 <- 1


bootstrap_predictions <- bootstrap_estimates[
  ,
  "Intercept"
] +
  bootstrap_estimates[
    ,
    "Slope"
  ] *
  x0


prediction_estimate <- beta_hat[
  "Intercept"
] +
  beta_hat[
    "Slope"
  ] *
  x0


prediction_CI <- quantile(
  bootstrap_predictions,
  c(
    0.025,
    0.975
  )
)


data.frame(
  x0 = x0,
  Estimate = prediction_estimate,
  Lower = prediction_CI[
    1
  ],
  Upper = prediction_CI[
    2
  ]
)


# ------------------------------------------------------------------------------
# 46. Visualize Bootstrap Prediction Distribution
# ------------------------------------------------------------------------------

hist(
  bootstrap_predictions,
  breaks = 40,
  xlab = "Predicted Mean at x0",
  main = paste(
    "Bootstrap Prediction at x =",
    x0
  )
)


abline(
  v = prediction_estimate,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 47. Important Distinction: Mean Prediction vs Future Observation
# ------------------------------------------------------------------------------

# The distribution above reflects uncertainty in:
#
# E[Y | X = x0]
#
# It does NOT automatically include new observation noise.
#
# A prediction interval for a new Y requires adding uncertainty from the
# future error term as well.


# ------------------------------------------------------------------------------
# 48. Nonparametric Bootstrap Algorithm Summary
# ------------------------------------------------------------------------------

cat(
  "Nonparametric Bootstrap Algorithm\n"
)


cat(
  "---------------------------------\n"
)


cat(
  "1. Start with n observed data points.\n"
)


cat(
  "2. Draw n observations with replacement.\n"
)


cat(
  "3. Compute the statistic on the resampled data.\n"
)


cat(
  "4. Repeat B times.\n"
)


cat(
  "5. Use the empirical distribution of the B statistics.\n"
)


# ------------------------------------------------------------------------------
# 49. Final Summary
# ------------------------------------------------------------------------------

cat(
  "\nOriginal slope estimate:",
  round(
    slope_hat,
    4
  ),
  "\n"
)


cat(
  "Bootstrap mean slope:",
  round(
    bootstrap_mean_slope,
    4
  ),
  "\n"
)


cat(
  "Bootstrap slope bias:",
  round(
    bootstrap_bias_slope,
    4
  ),
  "\n"
)


cat(
  "Bootstrap slope SE:",
  round(
    bootstrap_SE_slope,
    4
  ),
  "\n"
)


cat(
  "Classical slope SE:",
  round(
    classical_SE_slope,
    4
  ),
  "\n"
)


cat(
  "Bootstrap percentile 95% CI:",
  round(
    percentile_CI[
      1
    ],
    4
  ),
  "to",
  round(
    percentile_CI[
      2
    ],
    4
  ),
  "\n"
)


cat(
  "Average fraction of unique observations per bootstrap sample:",
  round(
    mean(
      unique_counts
    ) /
      n,
    4
  ),
  "\n"
)