# ==============================================================================
# Stochastic Search: Bumping
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Bootstrap-based stochastic search
#   - Bumping
#   - Instability of flexible models
#   - Candidate-model generation
#   - Selection using original-sample loss
#   - Difference between bumping and bagging
#   - Bootstrap search over local minima / alternative fits
#   - Model instability
#
# Core bumping algorithm:
#
#   1. Fit model to original data.
#
#   2. Generate B bootstrap samples.
#
#   3. Fit the model separately to each bootstrap sample.
#
#   4. Evaluate every fitted candidate on the ORIGINAL dataset.
#
#   5. Select the candidate with the smallest original-sample loss.
#
#
# Unlike bagging:
#
#   Bagging:
#
#       average bootstrap predictions
#
#   Bumping:
#
#       choose one particularly good bootstrap-generated model
#
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Nonlinear Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 250


x <- sort(
  runif(
    n,
    min = -3,
    max = 3
  )
)


true_function <- function(x) {
  
  2 +
    2 * sin(
      1.5 * x
    ) +
    0.4 * x^2
}


y_true <- true_function(
  x
)


sigma <- 1.2


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
  )


data <- data.frame(
  x = x,
  y = y
)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Nonlinear Regression Data"
)


lines(
  x,
  y_true,
  lwd = 2
)


# ==============================================================================
# PART I
#
# AN INTENTIONALLY FLEXIBLE / UNSTABLE MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Polynomial Regression Helper
# ------------------------------------------------------------------------------

# Bumping is most interesting when the fitted model is unstable enough that
# small changes in the dataset can produce meaningfully different fits.
#
# A high-degree polynomial gives a simple package-free demonstration.


polynomial_design <- function(
    x,
    degree
) {
  
  X <- matrix(
    1,
    nrow = length(
      x
    ),
    ncol = degree + 1
  )
  
  
  if (
    degree >= 1
  ) {
    
    for (
      d in seq_len(
        degree
      )
    ) {
      
      X[
        ,
        d + 1
      ] <- x^d
    }
  }
  
  
  return(
    X
  )
}


# ------------------------------------------------------------------------------
# 4. Fit Polynomial Model
# ------------------------------------------------------------------------------

polynomial_fit <- function(
    x,
    y,
    degree
) {
  
  X <- polynomial_design(
    x,
    degree
  )
  
  
  beta_hat <- qr.solve(
    X,
    y
  )
  
  
  return(
    list(
      coefficients = as.vector(
        beta_hat
      ),
      degree = degree
    )
  )
}


# ------------------------------------------------------------------------------
# 5. Polynomial Prediction
# ------------------------------------------------------------------------------

polynomial_predict <- function(
    model,
    x_new
) {
  
  X_new <- polynomial_design(
    x_new,
    model$degree
  )
  
  
  as.vector(
    X_new %*%
      model$coefficients
  )
}


# ------------------------------------------------------------------------------
# 6. Original Model
# ------------------------------------------------------------------------------

degree <- 10


original_model <- polynomial_fit(
  x,
  y,
  degree = degree
)


original_prediction <- polynomial_predict(
  original_model,
  x
)


original_MSE <- mean(
  (
    y -
      original_prediction
  )^2
)


original_MSE


# ------------------------------------------------------------------------------
# 7. Plot Original Flexible Model
# ------------------------------------------------------------------------------

x_grid <- seq(
  min(x),
  max(x),
  length.out = 500
)


original_grid_prediction <- polynomial_predict(
  original_model,
  x_grid
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Original Flexible Model"
)


lines(
  x_grid,
  original_grid_prediction,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 2,
  lwd = 2
)


# ==============================================================================
# PART II
#
# BOOTSTRAP-GENERATED CANDIDATE MODELS
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Generate One Bootstrap Candidate
# ------------------------------------------------------------------------------

set.seed(123)


bootstrap_index <- sample(
  seq_len(n),
  size = n,
  replace = TRUE
)


x_boot <- x[
  bootstrap_index
]


y_boot <- y[
  bootstrap_index
]


bootstrap_model <- polynomial_fit(
  x_boot,
  y_boot,
  degree = degree
)


bootstrap_prediction_original <- polynomial_predict(
  bootstrap_model,
  x
)


bootstrap_original_MSE <- mean(
  (
    y -
      bootstrap_prediction_original
  )^2
)


bootstrap_original_MSE


# ------------------------------------------------------------------------------
# 9. Compare Original Fit and One Bootstrap Fit
# ------------------------------------------------------------------------------

bootstrap_grid_prediction <- polynomial_predict(
  bootstrap_model,
  x_grid
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Original Fit vs One Bootstrap Fit"
)


lines(
  x_grid,
  original_grid_prediction,
  lwd = 2
)


lines(
  x_grid,
  bootstrap_grid_prediction,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Original Model",
    "Bootstrap Model"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART III
#
# BUMPING ALGORITHM
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Manual Bumping
# ------------------------------------------------------------------------------

bumping <- function(
    x,
    y,
    degree,
    B = 200,
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
  
  
  # Candidate 0:
  #
  # model fitted on original sample
  
  original_model <- polynomial_fit(
    x,
    y,
    degree
  )
  
  
  original_prediction <- polynomial_predict(
    original_model,
    x
  )
  
  
  original_loss <- mean(
    (
      y -
        original_prediction
    )^2
  )
  
  
  models <- vector(
    "list",
    B + 1
  )
  
  
  models[[1]] <- original_model
  
  
  losses <- numeric(
    B + 1
  )
  
  
  losses[1] <- original_loss
  
  
  bootstrap_unique_fraction <- numeric(
    B
  )
  
  
  # --------------------------------------------------------------------------
  # Bootstrap candidates
  # --------------------------------------------------------------------------
  
  for (
    b in seq_len(B)
  ) {
    
    index <- sample(
      seq_len(n),
      size = n,
      replace = TRUE
    )
    
    
    bootstrap_unique_fraction[b] <-
      length(
        unique(
          index
        )
      ) /
      n
    
    
    model_b <- polynomial_fit(
      x[
        index
      ],
      y[
        index
      ],
      degree
    )
    
    
    # IMPORTANT:
    #
    # evaluate candidate model on ORIGINAL sample,
    # not on the bootstrap sample.
    
    prediction_b <- polynomial_predict(
      model_b,
      x
    )
    
    
    loss_b <- mean(
      (
        y -
          prediction_b
      )^2
    )
    
    
    models[[b + 1]] <- model_b
    
    
    losses[
      b + 1
    ] <- loss_b
  }
  
  
  # --------------------------------------------------------------------------
  # Select best candidate
  # --------------------------------------------------------------------------
  
  best_index <- which.min(
    losses
  )
  
  
  best_model <- models[[best_index]]
  
  
  return(
    list(
      best_model = best_model,
      best_index = best_index,
      losses = losses,
      models = models,
      original_loss = original_loss,
      bootstrap_unique_fraction =
        bootstrap_unique_fraction
    )
  )
}


# ------------------------------------------------------------------------------
# 11. Run Bumping
# ------------------------------------------------------------------------------

set.seed(123)


B <- 500


bump_result <- bumping(
  x = x,
  y = y,
  degree = degree,
  B = B,
  seed = 123
)


bump_result$best_index


bump_result$original_loss


min(
  bump_result$losses
)


# ------------------------------------------------------------------------------
# 12. Was Original Model Selected?
# ------------------------------------------------------------------------------

# Candidate 1 is the original-data model.
#
# Candidate 2 onward are bootstrap models.


if (
  bump_result$best_index ==
  1
) {
  
  cat(
    "Original model selected.\n"
  )
  
} else {
  
  cat(
    "Bootstrap candidate",
    bump_result$best_index - 1,
    "selected.\n"
  )
}


# ------------------------------------------------------------------------------
# 13. Selected Bumped Model
# ------------------------------------------------------------------------------

bumped_model <- bump_result$best_model


bumped_prediction <- polynomial_predict(
  bumped_model,
  x
)


bumped_MSE <- mean(
  (
    y -
      bumped_prediction
  )^2
)


bumped_MSE


# ------------------------------------------------------------------------------
# 14. Plot Candidate Losses
# ------------------------------------------------------------------------------

plot(
  seq_along(
    bump_result$losses
  ) -
    1,
  bump_result$losses,
  type = "h",
  xlab = "Candidate",
  ylab = "Original-Sample MSE",
  main = "Bumping Candidate Losses"
)


points(
  bump_result$best_index -
    1,
  bump_result$losses[
    bump_result$best_index
  ],
  pch = 19,
  cex = 1.5
)


abline(
  h = bump_result$original_loss,
  lty = 2
)


# ------------------------------------------------------------------------------
# 15. Distribution of Candidate Losses
# ------------------------------------------------------------------------------

hist(
  bump_result$losses,
  breaks = 40,
  xlab = "Original-Sample MSE",
  main = "Distribution of Bumping Candidate Losses"
)


abline(
  v = bump_result$original_loss,
  lty = 2,
  lwd = 2
)


abline(
  v = min(
    bump_result$losses
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 16. Plot Original and Bumped Fits
# ------------------------------------------------------------------------------

bumped_grid_prediction <- polynomial_predict(
  bumped_model,
  x_grid
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Original Fit vs Bumped Fit"
)


lines(
  x_grid,
  original_grid_prediction,
  lty = 2,
  lwd = 2
)


lines(
  x_grid,
  bumped_grid_prediction,
  lwd = 2
)


lines(
  x,
  y_true,
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Bumped",
    "Original",
    "True Function"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ==============================================================================
# PART IV
#
# VISUALIZE STOCHASTIC SEARCH
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Plot Several Bootstrap Candidate Fits
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  cex = 0.5,
  xlab = "x",
  ylab = "y",
  main = "Bootstrap Search Through Alternative Fits"
)


number_to_plot <- 20


for (
  b in seq_len(
    number_to_plot
  )
) {
  
  prediction_b <- polynomial_predict(bump_result$models[[b + 1]],x_grid)
  
  
  lines(
    x_grid,
    prediction_b,
    lty = 3
  )
}


lines(
  x_grid,
  bumped_grid_prediction,
  lwd = 3
)


# ------------------------------------------------------------------------------
# 18. Candidate Coefficients
# ------------------------------------------------------------------------------

coefficient_matrix <- do.call(
  rbind,
  lapply(
    bump_result$models,
    function(model) {
      
      model$coefficients
    }
  )
)


colnames(
  coefficient_matrix
) <- paste0(
  "beta_",
  0:degree
)


dim(
  coefficient_matrix
)


# ------------------------------------------------------------------------------
# 19. Coefficient Instability
# ------------------------------------------------------------------------------

coefficient_SD <- apply(
  coefficient_matrix,
  2,
  sd
)


coefficient_SD


barplot(
  coefficient_SD,
  names.arg = 0:degree,
  xlab = "Polynomial Degree Term",
  ylab = "SD Across Bootstrap Fits",
  main = "Coefficient Instability"
)


# ------------------------------------------------------------------------------
# 20. Coefficient Paths Across Candidates
# ------------------------------------------------------------------------------

matplot(
  seq_len(
    nrow(
      coefficient_matrix
    )
  ) -
    1,
  coefficient_matrix,
  type = "l",
  lty = 1,
  xlab = "Candidate",
  ylab = "Coefficient",
  main = "Bootstrap Candidate Coefficients"
)


# ==============================================================================
# PART V
#
# BUMPING VS BAGGING
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Bootstrap Predictions on Grid
# ------------------------------------------------------------------------------

bootstrap_grid_predictions <- matrix(
  NA,
  nrow = length(
    x_grid
  ),
  ncol = B
)


for (
  b in seq_len(B)
) {
  
  bootstrap_grid_predictions[
    ,
    b
  ] <- polynomial_predict(bump_result$models[[b + 1]], x_grid)
}


# ------------------------------------------------------------------------------
# 22. Bagged Prediction
# ------------------------------------------------------------------------------

# Bagging:
#
# average predictions across bootstrap models.


bagged_grid_prediction <- rowMeans(
  bootstrap_grid_predictions
)


# ------------------------------------------------------------------------------
# 23. Bagged Training Predictions
# ------------------------------------------------------------------------------

bootstrap_training_predictions <- matrix(
  NA,
  nrow = n,
  ncol = B
)


for (
  b in seq_len(B)
) {
  
  bootstrap_training_predictions[
    ,
    b
  ] <- polynomial_predict(bump_result$models[[b + 1]],x)
}


bagged_training_prediction <- rowMeans(
  bootstrap_training_predictions
)


bagged_training_MSE <- mean(
  (
    y -
      bagged_training_prediction
  )^2
)


bagged_training_MSE


# ------------------------------------------------------------------------------
# 24. Compare Bumping and Bagging
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Original Fit",
    "Bumping",
    "Bagging"
  ),
  
  Training_MSE = c(
    original_MSE,
    bumped_MSE,
    bagged_training_MSE
  )
)


# ------------------------------------------------------------------------------
# 25. Plot Bumping vs Bagging
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  cex = 0.6,
  xlab = "x",
  ylab = "y",
  main = "Bumping vs Bagging"
)


lines(
  x_grid,
  original_grid_prediction,
  lty = 3,
  lwd = 2
)


lines(
  x_grid,
  bumped_grid_prediction,
  lty = 1,
  lwd = 2
)


lines(
  x_grid,
  bagged_grid_prediction,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Bumping",
    "Bagging",
    "Original"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ==============================================================================
# PART VI
#
# TRAIN-TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(321)


train_index <- sample(
  seq_len(n),
  size = floor(
    0.7 *
      n
  )
)


test_index <- setdiff(
  seq_len(n),
  train_index
)


x_train <- x[
  train_index
]


y_train <- y[
  train_index
]


x_test <- x[
  test_index
]


y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 27. Original Training Model
# ------------------------------------------------------------------------------

original_train_model <- polynomial_fit(
  x_train,
  y_train,
  degree
)


original_test_prediction <- polynomial_predict(
  original_train_model,
  x_test
)


original_test_MSE <- mean(
  (
    y_test -
      original_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 28. Bumping on Training Data Only
# ------------------------------------------------------------------------------

set.seed(123)


bump_train <- bumping(
  x = x_train,
  y = y_train,
  degree = degree,
  B = 500,
  seed = 123
)


bumped_test_prediction <- polynomial_predict(
  bump_train$best_model,
  x_test
)


bumped_test_MSE <- mean(
  (
    y_test -
      bumped_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 29. Bagging on Training Data
# ------------------------------------------------------------------------------

B_train <- 500


bagged_test_predictions <- matrix(
  NA,
  nrow = length(
    x_test
  ),
  ncol = B_train
)


for (
  b in seq_len(
    B_train
  )
) {
  
  candidate <- bump_train$models[[b + 1]]
  
  
  bagged_test_predictions[
    ,
    b
  ] <- polynomial_predict(
    candidate,
    x_test
  )
}


bagged_test_prediction <- rowMeans(
  bagged_test_predictions
)


bagged_test_MSE <- mean(
  (
    y_test -
      bagged_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 30. Test Comparison
# ------------------------------------------------------------------------------

test_results <- data.frame(
  Method = c(
    "Original Model",
    "Bumping",
    "Bagging"
  ),
  
  Test_MSE = c(
    original_test_MSE,
    bumped_test_MSE,
    bagged_test_MSE
  )
)


test_results$Test_RMSE <- sqrt(
  test_results$Test_MSE
)


test_results


# ==============================================================================
# PART VII
#
# HOW BUMPING CHANGES WITH NUMBER OF BOOTSTRAP SEARCHES
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Search Size
# ------------------------------------------------------------------------------

B_values <- c(
  1,
  5,
  10,
  25,
  50,
  100,
  250,
  500
)


best_training_loss <- numeric(
  length(
    B_values
  )
)


best_test_loss <- numeric(
  length(
    B_values
  )
)


for (
  i in seq_along(
    B_values
  )
) {
  
  bump_i <- bumping(
    x = x_train,
    y = y_train,
    degree = degree,
    B = B_values[i],
    seed = 100 +
      i
  )
  
  
  best_training_loss[i] <- min(
    bump_i$losses
  )
  
  
  test_prediction_i <- polynomial_predict(
    bump_i$best_model,
    x_test
  )
  
  
  best_test_loss[i] <- mean(
    (
      y_test -
        test_prediction_i
    )^2
  )
}


# ------------------------------------------------------------------------------
# 32. Search Size Results
# ------------------------------------------------------------------------------

search_results <- data.frame(
  Bootstrap_Candidates = B_values,
  Best_Training_MSE = best_training_loss,
  Test_MSE = best_test_loss
)


search_results


# ------------------------------------------------------------------------------
# 33. Plot Search Size vs Training Loss
# ------------------------------------------------------------------------------

plot(
  B_values,
  best_training_loss,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Number of Bootstrap Candidates",
  ylab = "Best Original-Sample MSE",
  main = "Bumping Search Size"
)


# ------------------------------------------------------------------------------
# 34. Plot Search Size vs Test Loss
# ------------------------------------------------------------------------------

plot(
  B_values,
  best_test_loss,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Number of Bootstrap Candidates",
  ylab = "Test MSE",
  main = "Does More Bumping Improve Generalization?"
)


# ==============================================================================
# PART VIII
#
# OUT-OF-BAG INFORMATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Typical Unique Fraction
# ------------------------------------------------------------------------------

mean(
  bump_result$bootstrap_unique_fraction
)


# Typical bootstrap sample contains about:
#
# 1 - exp(-1)
#
# approximately 0.632 of the original observations at least once.


1 -
  exp(
    -1
  )


# ------------------------------------------------------------------------------
# 36. Distribution of Unique Fractions
# ------------------------------------------------------------------------------

hist(
  bump_result$bootstrap_unique_fraction,
  breaks = 30,
  xlab = "Fraction of Unique Observations",
  main = "Bootstrap Candidate Diversity"
)


abline(
  v = 1 -
    exp(
      -1
    ),
  lty = 2
)


# ==============================================================================
# PART IX
#
# BUMPING WITH VARIABLE SELECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Generate Multivariate Data
# ------------------------------------------------------------------------------

set.seed(456)


n2 <- 250

p <- 8


X2 <- matrix(
  rnorm(
    n2 *
      p
  ),
  nrow = n2,
  ncol = p
)


colnames(
  X2
) <- paste0(
  "x",
  1:p
)


beta_true <- c(
  3,
  -2.5,
  1.5,
  0,
  0,
  1,
  0,
  0
)


y2 <- 2 +
  X2 %*%
  beta_true +
  rnorm(
    n2,
    sd = 2
  )


y2 <- as.vector(
  y2
)


# ------------------------------------------------------------------------------
# 38. Best-Subset Helper
# ------------------------------------------------------------------------------

fit_subset <- function(
    X,
    y,
    variables
) {
  
  if (
    length(
      variables
    ) ==
    0
  ) {
    
    X_model <- matrix(
      1,
      nrow = nrow(
        X
      ),
      ncol = 1
    )
    
  } else {
    
    X_model <- cbind(
      1,
      X[
        ,
        variables,
        drop = FALSE
      ]
    )
  }
  
  
  beta <- qr.solve(
    X_model,
    y
  )
  
  
  prediction <- as.vector(
    X_model %*%
      beta
  )
  
  
  RSS <- sum(
    (
      y -
        prediction
    )^2
  )
  
  
  return(
    list(
      variables = variables,
      coefficients = beta,
      RSS = RSS
    )
  )
}


# ------------------------------------------------------------------------------
# 39. Exhaustive Best Subset of Fixed Size
# ------------------------------------------------------------------------------

best_subset_fixed_size <- function(
    X,
    y,
    size
) {
  
  p <- ncol(
    X
  )
  
  
  subsets <- combn(
    p,
    size,
    simplify = FALSE
  )
  
  
  models <- lapply(
    subsets,
    function(vars) {
      
      fit_subset(
        X,
        y,
        vars
      )
    }
  )
  
  
  RSS_values <- sapply(
    models,
    function(model) {
      
      model$RSS
    }
  )
  
  
  models[[which.min(RSS_values)]]
}


# ------------------------------------------------------------------------------
# 40. Bumping Variable Selection
# ------------------------------------------------------------------------------

# Here each bootstrap sample can produce a different selected subset.
#
# Bumping searches through those unstable subset selections.


subset_size <- 3

B_subset <- 200


original_subset_model <- best_subset_fixed_size(
  X2,
  y2,
  subset_size
)


original_subset_model$variables


# ------------------------------------------------------------------------------
# 41. Generate Bootstrap-Selected Subsets
# ------------------------------------------------------------------------------

set.seed(123)


subset_candidates <- vector(
  "list",
  B_subset + 1
)


subset_candidates[[1]] <-
  original_subset_model


original_subset_losses <- numeric(
  B_subset + 1
)


# Evaluation helper on original data

predict_subset <- function(
    model,
    X_new
) {
  
  if (
    length(
      model$variables
    ) ==
    0
  ) {
    
    X_model <- matrix(
      1,
      nrow = nrow(
        X_new
      ),
      ncol = 1
    )
    
  } else {
    
    X_model <- cbind(
      1,
      X_new[
        ,
        model$variables,
        drop = FALSE
      ]
    )
  }
  
  
  as.vector(
    X_model %*%
      model$coefficients
  )
}


original_prediction_subset <- predict_subset(
  original_subset_model,
  X2
)


original_subset_losses[1] <- mean(
  (
    y2 -
      original_prediction_subset
  )^2
)


for (
  b in seq_len(
    B_subset
  )
) {
  
  index <- sample(
    seq_len(
      n2
    ),
    size = n2,
    replace = TRUE
  )
  
  
  model_b <- best_subset_fixed_size(
    X2[
      index,
      ,
      drop = FALSE
    ],
    y2[
      index
    ],
    subset_size
  )
  
  
  # IMPORTANT:
  #
  # The coefficients above were fitted using the bootstrap sample.
  #
  # Evaluate those exact coefficients on original observations.
  
  prediction_b <- predict_subset(
    model_b,
    X2
  )
  
  
  loss_b <- mean(
    (
      y2 -
        prediction_b
    )^2
  )
  
  
  subset_candidates[[b + 1]] <- model_b
  
  
  original_subset_losses[
    b + 1
  ] <- loss_b
}


# ------------------------------------------------------------------------------
# 42. Select Bumped Subset Model
# ------------------------------------------------------------------------------

best_subset_candidate <- which.min(
  original_subset_losses
)


bumped_subset_model <- subset_candidates[[best_subset_candidate]]


bumped_subset_model$variables


# ------------------------------------------------------------------------------
# 43. Variable Selection Frequencies
# ------------------------------------------------------------------------------

selection_frequency <- numeric(
  p
)


for (
  b in 2:length(
    subset_candidates
  )
) {
  
  selection_frequency[
    subset_candidates[[b]]$variables] <-selection_frequency[subset_candidates[[ b]]$variables] + 1}


selection_frequency <- selection_frequency /
  B_subset


names(
  selection_frequency
) <- colnames(
  X2
)


selection_frequency


barplot(
  selection_frequency,
  ylab = "Bootstrap Selection Frequency",
  main = "Model Instability Across Bootstrap Samples"
)


# ------------------------------------------------------------------------------
# 44. Compare True and Selected Variables
# ------------------------------------------------------------------------------

true_variables <- which(
  beta_true != 0
)


cat(
  "True active variables:",
  true_variables,
  "\n"
)


cat(
  "Original selected variables:",
  original_subset_model$variables,
  "\n"
)


cat(
  "Bumped selected variables:",
  bumped_subset_model$variables,
  "\n"
)


# ==============================================================================
# PART X
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Improvement on Original Data
# ------------------------------------------------------------------------------

training_improvement <-
  original_MSE -
  bumped_MSE


# ------------------------------------------------------------------------------
# 46. Final Summary
# ------------------------------------------------------------------------------

cat(
  "Stochastic Search: Bumping\n"
)


cat(
  "---------------------------\n"
)


cat(
  "Number of bootstrap candidates:",
  B,
  "\n"
)


cat(
  "Original training MSE:",
  round(
    original_MSE,
    4
  ),
  "\n"
)


cat(
  "Bumped training MSE:",
  round(
    bumped_MSE,
    4
  ),
  "\n"
)


cat(
  "Training improvement:",
  round(
    training_improvement,
    4
  ),
  "\n"
)


cat(
  "Original test MSE:",
  round(
    original_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Bumped test MSE:",
  round(
    bumped_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Bagged test MSE:",
  round(
    bagged_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Selected candidate:",
  bump_result$best_index -
    1,
  "\n"
)


cat(
  "Candidate 0 corresponds to the original-data fit.\n"
)