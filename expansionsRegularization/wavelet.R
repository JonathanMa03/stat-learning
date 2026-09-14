# ==============================================================================
# Wavelet Smoothing
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Wavelet basis expansion
#   - Multiresolution decomposition
#   - Haar wavelets
#   - Discrete Wavelet Transform (DWT)
#   - Inverse Discrete Wavelet Transform
#   - Hard thresholding
#   - Soft thresholding
#   - Universal threshold
#   - Noise estimation using MAD
#   - Bias-variance tradeoff
#
# Wavelet smoothing idea:
#
#   1. Transform noisy observations into wavelet coefficients.
#   2. Small coefficients are treated as mostly noise.
#   3. Shrink or remove those coefficients.
#   4. Apply the inverse transform.
#
# This script implements the Haar wavelet transform manually.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate a Noisy Signal
# ------------------------------------------------------------------------------

set.seed(123)

# Haar DWT is simplest when n is a power of 2.

n <- 256

x <- seq(
  0,
  1,
  length.out = n
)


# A signal with both smooth and abrupt local behavior.
#
# Wavelets are especially useful when the function contains
# localized features or discontinuities.

true_signal <- function(x) {
  
  2 * sin(
    4 * pi * x
  ) +
    ifelse(
      x > 0.30,
      1.5,
      0
    ) -
    ifelse(
      x > 0.65,
      2,
      0
    ) +
    1.5 *
    exp(
      -200 *
        (
          x - 0.8
        )^2
    )
}


f_true <- true_signal(
  x
)


sigma <- 0.8


y <- f_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
  )


# ------------------------------------------------------------------------------
# 2. Plot Noisy Signal
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  type = "p",
  pch = 19,
  cex = 0.6,
  xlab = "x",
  ylab = "Signal",
  main = "Noisy Signal"
)


lines(
  x,
  f_true,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Haar Transform for One Level
# ------------------------------------------------------------------------------

# For pairs:
#
# y_1, y_2
#
# compute:
#
# approximation = (y_1 + y_2) / sqrt(2)
#
# detail        = (y_1 - y_2) / sqrt(2)
#
#
# Approximation coefficients represent coarse-scale structure.
# Detail coefficients represent local differences.


haar_level <- function(
    values
) {
  
  n <- length(
    values
  )
  
  
  if (
    n %% 2 != 0
  ) {
    
    stop(
      "Input length must be even."
    )
  }
  
  
  odd <- values[
    seq(
      1,
      n,
      by = 2
    )
  ]
  
  
  even <- values[
    seq(
      2,
      n,
      by = 2
    )
  ]
  
  
  approximation <- (
    odd +
      even
  ) /
    sqrt(2)
  
  
  detail <- (
    odd -
      even
  ) /
    sqrt(2)
  
  
  return(
    list(
      approximation = approximation,
      detail = detail
    )
  )
}


# ------------------------------------------------------------------------------
# 4. Test One-Level Haar Transform
# ------------------------------------------------------------------------------

test_values <- c(
  4,
  2,
  6,
  2
)


haar_level(
  test_values
)


# ------------------------------------------------------------------------------
# 5. Full Discrete Haar Wavelet Transform
# ------------------------------------------------------------------------------

haar_dwt <- function(
    y
) {
  
  y <- as.numeric(
    y
  )
  
  
  n <- length(
    y
  )
  
  
  # Require power-of-two sample size.
  
  if (
    n <= 0 ||
    (
      log2(n) %%
      1
    ) != 0
  ) {
    
    stop(
      "Length of y must be a power of 2."
    )
  }
  
  
  current <- y
  
  
  details <- list()
  
  
  level <- 1
  
  
  while (
    length(
      current
    ) >
    1
  ) {
    
    decomposition <- haar_level(
      current
    )
    
    
    details[[level]] <-
      decomposition$detail
    
    
    current <-
      decomposition$approximation
    
    
    level <- level + 1
  }
  
  
  return(
    list(
      final_approximation = current,
      details = details,
      n = n
    )
  )
}


# ------------------------------------------------------------------------------
# 6. Apply Haar DWT
# ------------------------------------------------------------------------------

wavelet_transform <- haar_dwt(
  y
)


wavelet_transform$final_approximation


length(
  wavelet_transform$details
)


# ------------------------------------------------------------------------------
# 7. Inspect Coefficients by Scale
# ------------------------------------------------------------------------------

for (level in seq_along(
  wavelet_transform$details
)) {
  
  cat(
    "Level",
    level,
    ":",
    length(
      wavelet_transform$details[[level]]
    ),
    "detail coefficients\n"
  )
}


# ------------------------------------------------------------------------------
# 8. Plot Detail Coefficients
# ------------------------------------------------------------------------------

number_levels <- length(
  wavelet_transform$details
)


par(
  mfrow = c(
    4,
    2
  )
)


for (level in seq_len(
  min(
    number_levels,
    8
  )
)) {
  
  plot(
    wavelet_transform$details[[level]],
    type = "h",
    xlab = "Coefficient Index",
    ylab = "Detail",
    main = paste(
      "Wavelet Detail Level",
      level
    )
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 9. Inverse Haar Transform for One Level
# ------------------------------------------------------------------------------

# Given:
#
# approximation_i
# detail_i
#
# recover:
#
# y_(2i-1) = (approximation_i + detail_i) / sqrt(2)
#
# y_(2i)   = (approximation_i - detail_i) / sqrt(2)


haar_inverse_level <- function(
    approximation,
    detail
) {
  
  if (
    length(
      approximation
    ) !=
    length(
      detail
    )
  ) {
    
    stop(
      "Approximation and detail lengths must match."
    )
  }
  
  
  n <- length(
    approximation
  )
  
  
  reconstructed <- numeric(
    2 * n
  )
  
  
  reconstructed[
    seq(
      1,
      2 * n,
      by = 2
    )
  ] <- (
    approximation +
      detail
  ) /
    sqrt(2)
  
  
  reconstructed[
    seq(
      2,
      2 * n,
      by = 2
    )
  ] <- (
    approximation -
      detail
  ) /
    sqrt(2)
  
  
  return(
    reconstructed
  )
}


# ------------------------------------------------------------------------------
# 10. Full Inverse Haar Transform
# ------------------------------------------------------------------------------

haar_idwt <- function(
    transform
) {
  
  current <- transform$final_approximation
  
  
  details <- transform$details
  
  
  for (
    level in rev(
      seq_along(
        details
      )
    )
  ) {
    
    current <- haar_inverse_level(
      current,
      details[[level]]
    )
  }
  
  
  return(
    current
  )
}


# ------------------------------------------------------------------------------
# 11. Verify Perfect Reconstruction
# ------------------------------------------------------------------------------

y_reconstructed <- haar_idwt(
  wavelet_transform
)


max(
  abs(
    y -
      y_reconstructed
  )
)


# ------------------------------------------------------------------------------
# 12. Hard Thresholding
# ------------------------------------------------------------------------------

# Hard threshold:
#
# coefficient is retained if:
#
# |d| > lambda
#
# otherwise it is set to zero.


hard_threshold <- function(
    coefficients,
    threshold
) {
  
  ifelse(
    abs(
      coefficients
    ) >
      threshold,
    coefficients,
    0
  )
}


# ------------------------------------------------------------------------------
# 13. Soft Thresholding
# ------------------------------------------------------------------------------

# Soft threshold:
#
# sign(d) * max(|d| - lambda, 0)
#
# This both eliminates small coefficients and shrinks large ones.


soft_threshold <- function(
    coefficients,
    threshold
) {
  
  sign(
    coefficients
  ) *
    pmax(
      abs(
        coefficients
      ) -
        threshold,
      0
    )
}


# ------------------------------------------------------------------------------
# 14. Estimate Noise Level Using MAD
# ------------------------------------------------------------------------------

# For wavelet denoising, noise is commonly estimated from
# the finest-scale detail coefficients.
#
# For Gaussian noise:
#
# sigma_hat =
# median(|d - median(d)|) / 0.6745


mad_sigma <- function(
    coefficients
) {
  
  median(
    abs(
      coefficients -
        median(
          coefficients
        )
    )
  ) /
    0.6745
}


finest_details <- wavelet_transform$details[[1]]


sigma_hat <- mad_sigma(
  finest_details
)


sigma_hat


# ------------------------------------------------------------------------------
# 15. Universal Threshold
# ------------------------------------------------------------------------------

# Donoho-Johnstone universal threshold:
#
# lambda =
#
# sigma_hat * sqrt(2 log n)


universal_threshold <-
  sigma_hat *
  sqrt(
    2 *
      log(
        n
      )
  )


universal_threshold


# ------------------------------------------------------------------------------
# 16. Threshold Entire Wavelet Transform
# ------------------------------------------------------------------------------

threshold_wavelet <- function(
    transform,
    threshold,
    method = c(
      "soft",
      "hard"
    ),
    threshold_levels = NULL
) {
  
  method <- match.arg(
    method
  )
  
  
  output <- transform
  
  
  number_levels <- length(
    transform$details
  )
  
  
  if (
    is.null(
      threshold_levels
    )
  ) {
    
    threshold_levels <- seq_len(
      number_levels
    )
  }
  
  
  for (level in seq_len(
    number_levels
  )) {
    
    if (
      level %in%
      threshold_levels
    ) {
      
      if (
        method ==
        "soft"
      ) {
        
        output$details[[level]] <-
          soft_threshold(
            transform$details[[level]],
            threshold
          )
        
      } else {
        
        output$details[[level]] <-
          hard_threshold(
            transform$details[[level]],
            threshold
          )
      }
    }
  }
  
  
  return(
    output
  )
}


# ------------------------------------------------------------------------------
# 17. Soft-Threshold Wavelet Smoothing
# ------------------------------------------------------------------------------

soft_transform <- threshold_wavelet(
  wavelet_transform,
  threshold = universal_threshold,
  method = "soft"
)


soft_smoothed <- haar_idwt(
  soft_transform
)


# ------------------------------------------------------------------------------
# 18. Hard-Threshold Wavelet Smoothing
# ------------------------------------------------------------------------------

hard_transform <- threshold_wavelet(
  wavelet_transform,
  threshold = universal_threshold,
  method = "hard"
)


hard_smoothed <- haar_idwt(
  hard_transform
)


# ------------------------------------------------------------------------------
# 19. Plot Soft Threshold Fit
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x",
  ylab = "Signal",
  main = "Wavelet Smoothing: Soft Threshold"
)


lines(
  x,
  soft_smoothed,
  lwd = 2
)


lines(
  x,
  f_true,
  lty = 2,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Wavelet Estimate",
    "True Signal"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 20. Plot Hard Threshold Fit
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x",
  ylab = "Signal",
  main = "Wavelet Smoothing: Hard Threshold"
)


lines(
  x,
  hard_smoothed,
  lwd = 2
)


lines(
  x,
  f_true,
  lty = 2,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 21. Compare Hard and Soft Thresholding
# ------------------------------------------------------------------------------

plot(
  x,
  f_true,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "Signal",
  main = "Hard vs Soft Wavelet Thresholding"
)


lines(
  x,
  hard_smoothed,
  lty = 2,
  lwd = 2
)


lines(
  x,
  soft_smoothed,
  lty = 3,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "True Signal",
    "Hard Threshold",
    "Soft Threshold"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 22. Mean Squared Error
# ------------------------------------------------------------------------------

raw_MSE <- mean(
  (
    y -
      f_true
  )^2
)


hard_MSE <- mean(
  (
    hard_smoothed -
      f_true
  )^2
)


soft_MSE <- mean(
  (
    soft_smoothed -
      f_true
  )^2
)


data.frame(
  Method = c(
    "Noisy Data",
    "Hard Threshold",
    "Soft Threshold"
  ),
  MSE = c(
    raw_MSE,
    hard_MSE,
    soft_MSE
  )
)


# ------------------------------------------------------------------------------
# 23. Sparsity of Wavelet Representation
# ------------------------------------------------------------------------------

count_nonzero_details <- function(
    transform,
    tolerance = 1e-12
) {
  
  coefficients <- unlist(
    transform$details
  )
  
  
  sum(
    abs(
      coefficients
    ) >
      tolerance
  )
}


original_nonzero <- count_nonzero_details(
  wavelet_transform
)


hard_nonzero <- count_nonzero_details(
  hard_transform
)


soft_nonzero <- count_nonzero_details(
  soft_transform
)


data.frame(
  Model = c(
    "Original Transform",
    "Hard Threshold",
    "Soft Threshold"
  ),
  Nonzero_Detail_Coefficients = c(
    original_nonzero,
    hard_nonzero,
    soft_nonzero
  )
)


# ------------------------------------------------------------------------------
# 24. Fraction of Coefficients Retained
# ------------------------------------------------------------------------------

total_details <- length(
  unlist(
    wavelet_transform$details
  )
)


data.frame(
  Method = c(
    "Hard",
    "Soft"
  ),
  Fraction_Nonzero = c(
    hard_nonzero /
      total_details,
    soft_nonzero /
      total_details
  )
)


# ------------------------------------------------------------------------------
# 25. Explore Threshold Strength
# ------------------------------------------------------------------------------

threshold_multipliers <- seq(
  0,
  2,
  by = 0.1
)


threshold_results <- data.frame(
  Multiplier = threshold_multipliers,
  Threshold = NA,
  MSE = NA,
  Nonzero = NA
)


for (i in seq_along(
  threshold_multipliers
)) {
  
  multiplier <- threshold_multipliers[i]
  
  
  threshold_i <-
    multiplier *
    universal_threshold
  
  
  transformed_i <- threshold_wavelet(
    wavelet_transform,
    threshold = threshold_i,
    method = "soft"
  )
  
  
  fitted_i <- haar_idwt(
    transformed_i
  )
  
  
  threshold_results$Threshold[i] <-
    threshold_i
  
  
  threshold_results$MSE[i] <-
    mean(
      (
        fitted_i -
          f_true
      )^2
    )
  
  
  threshold_results$Nonzero[i] <-
    count_nonzero_details(
      transformed_i
    )
}


threshold_results


# ------------------------------------------------------------------------------
# 26. Plot Threshold vs MSE
# ------------------------------------------------------------------------------

plot(
  threshold_results$Multiplier,
  threshold_results$MSE,
  type = "b",
  pch = 19,
  xlab = "Universal Threshold Multiplier",
  ylab = "MSE",
  main = "Wavelet Threshold Strength"
)


# ------------------------------------------------------------------------------
# 27. Threshold vs Number of Nonzero Coefficients
# ------------------------------------------------------------------------------

plot(
  threshold_results$Multiplier,
  threshold_results$Nonzero,
  type = "b",
  pch = 19,
  xlab = "Universal Threshold Multiplier",
  ylab = "Nonzero Detail Coefficients",
  main = "Wavelet Sparsity"
)


# ------------------------------------------------------------------------------
# 28. Oracle Threshold for Simulation Comparison
# ------------------------------------------------------------------------------

# Since this is simulated data, we know f_true and can identify
# the threshold with the smallest actual MSE.
#
# In real data this quantity is not available.


best_index <- which.min(
  threshold_results$MSE
)


best_multiplier <- threshold_results$Multiplier[
  best_index
]


best_threshold <- threshold_results$Threshold[
  best_index
]


best_threshold


# ------------------------------------------------------------------------------
# 29. Fit Oracle Threshold Model
# ------------------------------------------------------------------------------

oracle_transform <- threshold_wavelet(
  wavelet_transform,
  threshold = best_threshold,
  method = "soft"
)


oracle_smoothed <- haar_idwt(
  oracle_transform
)


# ------------------------------------------------------------------------------
# 30. Plot Universal vs Oracle Threshold
# ------------------------------------------------------------------------------

plot(
  x,
  f_true,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "Signal",
  main = "Universal vs Oracle Threshold"
)


lines(
  x,
  soft_smoothed,
  lty = 2,
  lwd = 2
)


lines(
  x,
  oracle_smoothed,
  lty = 3,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "True",
    "Universal",
    "Oracle"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 31. Threshold Individual Levels Differently
# ------------------------------------------------------------------------------

# Noise often dominates the finest scales.
#
# We can threshold only the first few detail levels
# and leave coarse-scale information unchanged.


fine_levels <- c(
  1,
  2,
  3,
  4
)


fine_transform <- threshold_wavelet(
  wavelet_transform,
  threshold = universal_threshold,
  method = "soft",
  threshold_levels = fine_levels
)


fine_smoothed <- haar_idwt(
  fine_transform
)


# ------------------------------------------------------------------------------
# 32. Compare Global vs Fine-Scale Thresholding
# ------------------------------------------------------------------------------

plot(
  x,
  f_true,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "Signal",
  main = "Thresholding Different Wavelet Scales"
)


lines(
  x,
  soft_smoothed,
  lty = 2,
  lwd = 2
)


lines(
  x,
  fine_smoothed,
  lty = 3,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "True Signal",
    "All Detail Levels",
    "Fine Levels Only"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 33. MSE Comparison by Threshold Strategy
# ------------------------------------------------------------------------------

fine_MSE <- mean(
  (
    fine_smoothed -
      f_true
  )^2
)


data.frame(
  Method = c(
    "Noisy",
    "Hard Universal",
    "Soft Universal",
    "Fine-Level Soft",
    "Oracle Soft"
  ),
  MSE = c(
    raw_MSE,
    hard_MSE,
    soft_MSE,
    fine_MSE,
    mean(
      (
        oracle_smoothed -
          f_true
      )^2
    )
  )
)


# ------------------------------------------------------------------------------
# 34. Visualize Approximation at Different Resolutions
# ------------------------------------------------------------------------------

# To illustrate multiresolution analysis, set selected fine-scale
# detail coefficients equal to zero.


remove_fine_levels <- function(
    transform,
    number_remove
) {
  
  output <- transform
  
  
  if (
    number_remove >
    0
  ) {
    
    for (
      level in seq_len(
        min(
          number_remove,
          length(
            output$details
          )
        )
      )
    ) {
      
      output$details[[level]][] <- 0
    }
  }
  
  
  return(
    output
  )
}


resolution_levels <- c(
  0,
  1,
  2,
  3,
  4
)


par(
  mfrow = c(
    3,
    2
  )
)


for (
  number_remove in resolution_levels
) {
  
  transform_i <- remove_fine_levels(
    wavelet_transform,
    number_remove
  )
  
  
  signal_i <- haar_idwt(
    transform_i
  )
  
  
  plot(
    x,
    signal_i,
    type = "l",
    lwd = 2,
    xlab = "x",
    ylab = "Signal",
    main = paste(
      "Remove",
      number_remove,
      "Finest Levels"
    )
  )
  
  
  lines(
    x,
    f_true,
    lty = 2
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ------------------------------------------------------------------------------
# 35. Energy by Wavelet Scale
# ------------------------------------------------------------------------------

# Wavelet energy at a level:
#
# sum_j d_j^2


level_energy <- numeric(
  number_levels
)


for (
  level in seq_len(
    number_levels
  )
) {
  
  level_energy[level] <- sum(
    wavelet_transform$details[[level]]^2
  )
}


energy_table <- data.frame(
  Level = seq_len(
    number_levels
  ),
  Number_of_Coefficients = sapply(
    wavelet_transform$details,
    length
  ),
  Energy = level_energy
)


energy_table


# ------------------------------------------------------------------------------
# 36. Plot Energy by Scale
# ------------------------------------------------------------------------------

plot(
  energy_table$Level,
  energy_table$Energy,
  type = "b",
  pch = 19,
  xlab = "Wavelet Level",
  ylab = "Energy",
  main = "Signal Energy Across Scales"
)


# ------------------------------------------------------------------------------
# 37. Signal Compression Interpretation
# ------------------------------------------------------------------------------

# Thresholding can also be viewed as signal compression:
#
# many small wavelet coefficients are removed while preserving
# the major features of the signal.


sorted_coefficients <- sort(
  abs(
    unlist(
      wavelet_transform$details
    )
  ),
  decreasing = TRUE
)


cumulative_energy <- cumsum(
  sorted_coefficients^2
) /
  sum(
    sorted_coefficients^2
  )


plot(
  seq_along(
    cumulative_energy
  ),
  cumulative_energy,
  type = "l",
  xlab = "Number of Largest Coefficients",
  ylab = "Fraction of Wavelet Energy",
  main = "Sparse Wavelet Representation"
)


abline(
  h = 0.95,
  lty = 2
)


# ------------------------------------------------------------------------------
# 38. Compare Against a Moving Average
# ------------------------------------------------------------------------------

moving_average <- function(
    y,
    window = 9
) {
  
  n <- length(
    y
  )
  
  
  output <- numeric(
    n
  )
  
  
  half_window <- floor(
    window /
      2
  )
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    lower <- max(
      1,
      i -
        half_window
    )
    
    
    upper <- min(
      n,
      i +
        half_window
    )
    
    
    output[i] <- mean(
      y[
        lower:upper
      ]
    )
  }
  
  
  return(
    output
  )
}


moving_fit <- moving_average(
  y,
  window = 9
)


moving_MSE <- mean(
  (
    moving_fit -
      f_true
  )^2
)


# ------------------------------------------------------------------------------
# 39. Compare Wavelet and Moving Average
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  cex = 0.4,
  xlab = "x",
  ylab = "Signal",
  main = "Wavelet vs Moving-Average Smoothing"
)


lines(
  x,
  soft_smoothed,
  lwd = 2
)


lines(
  x,
  moving_fit,
  lty = 2,
  lwd = 2
)


lines(
  x,
  f_true,
  lty = 3,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Wavelet",
    "Moving Average",
    "True Signal"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


data.frame(
  Method = c(
    "Wavelet",
    "Moving Average"
  ),
  MSE = c(
    soft_MSE,
    moving_MSE
  )
)


# ------------------------------------------------------------------------------
# 40. Demonstrate Local Adaptivity
# ------------------------------------------------------------------------------

# Wavelets can preserve abrupt changes because basis functions
# are localized in both position and scale.
#
# Examine the region around the jump at x = 0.65.


local_index <- x >= 0.5 &
  x <= 0.8


plot(
  x[
    local_index
  ],
  y[
    local_index
  ],
  pch = 19,
  cex = 0.6,
  xlab = "x",
  ylab = "Signal",
  main = "Local Behavior Near a Discontinuity"
)


lines(
  x[
    local_index
  ],
  f_true[
    local_index
  ],
  lwd = 2
)


lines(
  x[
    local_index
  ],
  soft_smoothed[
    local_index
  ],
  lty = 2,
  lwd = 2
)


lines(
  x[
    local_index
  ],
  moving_fit[
    local_index
  ],
  lty = 3,
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "True",
    "Wavelet",
    "Moving Average"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 41. Parseval Energy Check
# ------------------------------------------------------------------------------

# Haar transform is orthonormal.
#
# Therefore:
#
# sum(y^2)
#
# should equal:
#
# final approximation^2 +
# sum(all detail coefficients^2)


original_energy <- sum(
  y^2
)


wavelet_energy <-
  sum(
    wavelet_transform$final_approximation^2
  ) +
  sum(
    unlist(
      wavelet_transform$details
    )^2
  )


data.frame(
  Original_Energy = original_energy,
  Wavelet_Energy = wavelet_energy,
  Difference = original_energy -
    wavelet_energy
)


# ------------------------------------------------------------------------------
# 42. Summary
# ------------------------------------------------------------------------------

cat(
  "True noise SD:",
  sigma,
  "\n"
)


cat(
  "Estimated noise SD:",
  round(
    sigma_hat,
    4
  ),
  "\n"
)


cat(
  "Universal threshold:",
  round(
    universal_threshold,
    4
  ),
  "\n"
)


cat(
  "Noisy signal MSE:",
  round(
    raw_MSE,
    4
  ),
  "\n"
)


cat(
  "Hard-threshold MSE:",
  round(
    hard_MSE,
    4
  ),
  "\n"
)


cat(
  "Soft-threshold MSE:",
  round(
    soft_MSE,
    4
  ),
  "\n"
)


cat(
  "Soft-threshold nonzero coefficients:",
  soft_nonzero,
  "of",
  total_details,
  "\n"
)


cat(
  "Best simulation threshold multiplier:",
  best_multiplier,
  "\n"
)