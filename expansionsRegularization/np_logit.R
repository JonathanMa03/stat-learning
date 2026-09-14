# ==============================================================================
# Nonparametric Logistic Regression
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Logistic regression with a flexible nonlinear predictor
#   - B-spline basis expansion
#   - Nonlinear log-odds
#   - IRLS / Newton-Raphson
#   - Classification probabilities
#   - Decision boundaries
#   - Cross-validation for model complexity
#
# Model:
#
#   log[p(x) / (1 - p(x))] = f(x)
#
# where f(x) is represented using spline basis functions.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Nonlinear Binary Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 400

x <- runif(
  n,
  min = -4,
  max = 4
)


# Nonlinear true log-odds function

true_log_odds <- function(x) {
  
  -0.5 +
    1.2 * sin(
      1.5 * x
    ) +
    0.2 * x
}


eta_true <- true_log_odds(
  x
)


true_probability <- 1 /
  (
    1 +
      exp(
        -eta_true
      )
  )


y <- rbinom(
  n = n,
  size = 1,
  prob = true_probability
)


data <- data.frame(
  x = x,
  y = y
)


head(data)

table(y)


# ------------------------------------------------------------------------------
# 2. Plot Observed Binary Responses
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "y",
  main = "Nonlinear Binary Classification Data"
)


# Plot true probability curve

x_grid <- seq(
  min(x),
  max(x),
  length.out = 500
)


true_probability_grid <- 1 /
  (
    1 +
      exp(
        -true_log_odds(
          x_grid
        )
      )
  )


lines(
  x_grid,
  true_probability_grid,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 3. Stable Logistic Function
# ------------------------------------------------------------------------------

sigmoid <- function(z) {
  
  output <- numeric(
    length(z)
  )
  
  
  positive <- z >= 0
  
  
  output[positive] <-
    1 /
    (
      1 +
        exp(
          -z[positive]
        )
    )
  
  
  exp_z <- exp(
    z[
      !positive
    ]
  )
  
  
  output[
    !positive
  ] <-
    exp_z /
    (
      1 +
        exp_z
    )
  
  
  return(
    output
  )
}


# ------------------------------------------------------------------------------
# 4. Ordinary Linear Logistic Regression for Comparison
# ------------------------------------------------------------------------------

X_linear <- cbind(
  Intercept = 1,
  x = x
)


# ------------------------------------------------------------------------------
# 5. Logistic Log-Likelihood
# ------------------------------------------------------------------------------

logistic_loglik <- function(
    beta,
    X,
    y
) {
  
  eta <- as.vector(
    X %*%
      beta
  )
  
  
  probability <- sigmoid(
    eta
  )
  
  
  probability <- pmin(
    pmax(
      probability,
      1e-12
    ),
    1 - 1e-12
  )
  
  
  sum(
    y *
      log(
        probability
      ) +
      (
        1 - y
      ) *
      log(
        1 - probability
      )
  )
}


# ------------------------------------------------------------------------------
# 6. Manual Logistic Regression with IRLS
# ------------------------------------------------------------------------------

logistic_irls <- function(
    X,
    y,
    max_iter = 100,
    tol = 1e-8,
    ridge = 1e-8
) {
  
  X <- as.matrix(
    X
  )
  
  
  y <- as.numeric(
    y
  )
  
  
  p <- ncol(
    X
  )
  
  
  beta <- rep(
    0,
    p
  )
  
  
  loglik_history <- numeric(
    max_iter
  )
  
  
  for (iteration in seq_len(
    max_iter
  )) {
    
    eta <- as.vector(
      X %*%
        beta
    )
    
    
    probability <- sigmoid(
      eta
    )
    
    
    weights <- probability *
      (
        1 -
          probability
      )
    
    
    # Avoid zero weights
    
    weights <- pmax(
      weights,
      1e-8
    )
    
    
    working_response <-
      eta +
      (
        y -
          probability
      ) /
      weights
    
    
    X_weighted <- sweep(
      X,
      MARGIN = 1,
      STATS = weights,
      FUN = "*"
    )
    
    
    penalty <- ridge *
      diag(
        p
      )
    
    
    # Do not penalize intercept
    
    penalty[
      1,
      1
    ] <- 0
    
    
    beta_new <- solve(
      crossprod(
        X,
        X_weighted
      ) +
        penalty,
      crossprod(
        X,
        weights *
          working_response
      )
    )
    
    
    loglik_history[
      iteration
    ] <- logistic_loglik(
      beta_new,
      X,
      y
    )
    
    
    if (
      max(
        abs(
          beta_new -
          beta
        )
      ) <
      tol
    ) {
      
      beta <- beta_new
      
      break
    }
    
    
    beta <- beta_new
  }
  
  
  loglik_history <- loglik_history[
    seq_len(
      iteration
    )
  ]
  
  
  eta <- as.vector(
    X %*%
      beta
  )
  
  
  probability <- sigmoid(
    eta
  )
  
  
  return(
    list(
      coefficients = as.vector(
        beta
      ),
      fitted_probability = probability,
      linear_predictor = eta,
      iterations = iteration,
      loglik_history = loglik_history
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Fit Ordinary Logistic Regression
# ------------------------------------------------------------------------------

linear_model <- logistic_irls(
  X = X_linear,
  y = y
)


linear_model$coefficients


# ------------------------------------------------------------------------------
# 8. Linear Logistic Predictions
# ------------------------------------------------------------------------------

linear_probability <- linear_model$fitted_probability


linear_prediction <- ifelse(
  linear_probability >= 0.5,
  1,
  0
)


linear_accuracy <- mean(
  linear_prediction ==
    y
)


linear_accuracy


# ------------------------------------------------------------------------------
# 9. Construct Cubic B-Spline Basis
# ------------------------------------------------------------------------------

# We will model:
#
# logit[p(x)] =
#
# beta0 +
# beta1 B1(x) +
# ...
# betaK BK(x)


number_knots <- 5


interior_knots <- as.numeric(
  quantile(
    x,
    probs = seq(
      0,
      1,
      length.out = number_knots + 2
    )[
      -c(
        1,
        number_knots + 2
      )
    ]
  )
)


boundary_knots <- range(
  x
)


B <- splines::bs(
  x,
  knots = interior_knots,
  degree = 3,
  Boundary.knots = boundary_knots,
  intercept = FALSE
)


B <- as.matrix(
  B
)


dim(
  B
)


# ------------------------------------------------------------------------------
# 10. Construct Nonparametric Logistic Design Matrix
# ------------------------------------------------------------------------------

X_spline <- cbind(
  Intercept = 1,
  B
)


colnames(
  X_spline
) <- c(
  "Intercept",
  paste0(
    "B",
    seq_len(
      ncol(B)
    )
  )
)


dim(
  X_spline
)


# ------------------------------------------------------------------------------
# 11. Plot Spline Basis Functions
# ------------------------------------------------------------------------------

order_x <- order(
  x
)


matplot(
  x[
    order_x
  ],
  B[
    order_x,
    ,
    drop = FALSE
  ],
  type = "l",
  lty = 1,
  xlab = "x",
  ylab = "Basis Value",
  main = "Cubic B-Spline Basis"
)


abline(
  v = interior_knots,
  lty = 3
)


# ------------------------------------------------------------------------------
# 12. Fit Nonparametric Logistic Regression
# ------------------------------------------------------------------------------

nonparametric_model <- logistic_irls(
  X = X_spline,
  y = y
)


nonparametric_model$coefficients


# ------------------------------------------------------------------------------
# 13. Convergence
# ------------------------------------------------------------------------------

plot(
  seq_along(
    nonparametric_model$loglik_history
  ),
  nonparametric_model$loglik_history,
  type = "l",
  xlab = "Iteration",
  ylab = "Log-Likelihood",
  main = "IRLS Convergence"
)


# ------------------------------------------------------------------------------
# 14. Fitted Probabilities
# ------------------------------------------------------------------------------

fitted_probability <-
  nonparametric_model$fitted_probability


head(
  fitted_probability
)


# ------------------------------------------------------------------------------
# 15. Classification
# ------------------------------------------------------------------------------

threshold <- 0.5


prediction <- ifelse(
  fitted_probability >= threshold,
  1,
  0
)


accuracy <- mean(
  prediction ==
    y
)


accuracy


# ------------------------------------------------------------------------------
# 16. Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y,
  Predicted = prediction
)


# ------------------------------------------------------------------------------
# 17. Classification Metrics
# ------------------------------------------------------------------------------

TP <- sum(
  prediction == 1 &
    y == 1
)


TN <- sum(
  prediction == 0 &
    y == 0
)


FP <- sum(
  prediction == 1 &
    y == 0
)


FN <- sum(
  prediction == 0 &
    y == 1
)


sensitivity <- TP /
  (
    TP +
      FN
  )


specificity <- TN /
  (
    TN +
      FP
  )


precision <- TP /
  (
    TP +
      FP
  )


F1 <- 2 *
  precision *
  sensitivity /
  (
    precision +
      sensitivity
  )


data.frame(
  Accuracy = accuracy,
  Sensitivity = sensitivity,
  Specificity = specificity,
  Precision = precision,
  F1 = F1
)


# ------------------------------------------------------------------------------
# 18. Generic Spline Logistic Fit Function
# ------------------------------------------------------------------------------

spline_logistic_fit <- function(
    x,
    y,
    knots,
    degree = 3,
    boundary_knots = range(x)
) {
  
  B <- splines::bs(
    x,
    knots = knots,
    degree = degree,
    Boundary.knots = boundary_knots,
    intercept = FALSE
  )
  
  
  X <- cbind(
    Intercept = 1,
    B
  )
  
  
  fit <- logistic_irls(
    X,
    y
  )
  
  
  return(
    list(
      coefficients = fit$coefficients,
      knots = knots,
      degree = degree,
      boundary_knots = boundary_knots,
      loglik_history = fit$loglik_history
    )
  )
}


# ------------------------------------------------------------------------------
# 19. Generic Prediction Function
# ------------------------------------------------------------------------------

spline_logistic_predict <- function(
    x_new,
    model
) {
  
  B_new <- splines::bs(
    x_new,
    knots = model$knots,
    degree = model$degree,
    Boundary.knots = model$boundary_knots,
    intercept = FALSE
  )
  
  
  X_new <- cbind(
    Intercept = 1,
    B_new
  )
  
  
  eta <- as.vector(
    X_new %*%
      model$coefficients
  )
  
  
  probability <- sigmoid(
    eta
  )
  
  
  return(
    list(
      linear_predictor = eta,
      probability = probability
    )
  )
}


# ------------------------------------------------------------------------------
# 20. Fit Using Generic Function
# ------------------------------------------------------------------------------

spline_model <- spline_logistic_fit(
  x = x,
  y = y,
  knots = interior_knots,
  degree = 3,
  boundary_knots = boundary_knots
)


# ------------------------------------------------------------------------------
# 21. Probability Curve on Dense Grid
# ------------------------------------------------------------------------------

x_grid <- seq(
  min(x),
  max(x),
  length.out = 500
)


grid_prediction <- spline_logistic_predict(
  x_grid,
  spline_model
)


# ------------------------------------------------------------------------------
# 22. Plot Nonparametric Logistic Regression
# ------------------------------------------------------------------------------

plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "Probability / Response",
  main = "Nonparametric Logistic Regression"
)


lines(
  x_grid,
  grid_prediction$probability,
  lwd = 2
)


lines(
  x_grid,
  true_probability_grid,
  lty = 2,
  lwd = 2
)


abline(
  h = 0.5,
  lty = 3
)


legend(
  "topleft",
  legend = c(
    "Spline Logistic",
    "True Probability",
    "Classification Threshold"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = c(
    2,
    2,
    1
  )
)


# ------------------------------------------------------------------------------
# 23. Plot Estimated Log-Odds
# ------------------------------------------------------------------------------

plot(
  x_grid,
  grid_prediction$linear_predictor,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "Log-Odds",
  main = "Estimated Nonlinear Log-Odds"
)


lines(
  x_grid,
  true_log_odds(
    x_grid
  ),
  lty = 2,
  lwd = 2
)


abline(
  h = 0,
  lty = 3
)


legend(
  "topleft",
  legend = c(
    "Estimated",
    "True"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 24. Compare with Ordinary Logistic Regression
# ------------------------------------------------------------------------------

linear_glm <- glm(
  y ~ x,
  family = binomial(),
  data = data
)


linear_grid_probability <- predict(
  linear_glm,
  newdata = data.frame(
    x = x_grid
  ),
  type = "response"
)


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "Probability / Response",
  main = "Linear vs Nonparametric Logistic Regression"
)


lines(
  x_grid,
  linear_grid_probability,
  lty = 2,
  lwd = 2
)


lines(
  x_grid,
  grid_prediction$probability,
  lty = 1,
  lwd = 2
)


lines(
  x_grid,
  true_probability_grid,
  lty = 3,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "Nonparametric Logistic",
    "Linear Logistic",
    "True Probability"
  ),
  lty = c(
    1,
    2,
    3
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 25. Compare Training Accuracy
# ------------------------------------------------------------------------------

comparison <- data.frame(
  Model = c(
    "Linear Logistic",
    "Spline Logistic"
  ),
  Accuracy = c(
    linear_accuracy,
    accuracy
  )
)


comparison


# ------------------------------------------------------------------------------
# 26. Compare Log-Likelihood
# ------------------------------------------------------------------------------

linear_loglik <- logistic_loglik(
  linear_model$coefficients,
  X_linear,
  y
)


spline_loglik <- logistic_loglik(
  nonparametric_model$coefficients,
  X_spline,
  y
)


data.frame(
  Model = c(
    "Linear Logistic",
    "Spline Logistic"
  ),
  Log_Likelihood = c(
    linear_loglik,
    spline_loglik
  )
)


# ------------------------------------------------------------------------------
# 27. Deviance
# ------------------------------------------------------------------------------

linear_deviance <- -2 *
  linear_loglik


spline_deviance <- -2 *
  spline_loglik


data.frame(
  Model = c(
    "Linear Logistic",
    "Spline Logistic"
  ),
  Deviance = c(
    linear_deviance,
    spline_deviance
  )
)


# ------------------------------------------------------------------------------
# 28. Probability Calibration / Brier Score
# ------------------------------------------------------------------------------

linear_brier <- mean(
  (
    y -
      linear_probability
  )^2
)


spline_brier <- mean(
  (
    y -
      fitted_probability
  )^2
)


data.frame(
  Model = c(
    "Linear Logistic",
    "Spline Logistic"
  ),
  Brier_Score = c(
    linear_brier,
    spline_brier
  )
)


# ------------------------------------------------------------------------------
# 29. ROC Curve Function
# ------------------------------------------------------------------------------

roc_curve <- function(
    y,
    probability,
    thresholds = seq(
      0,
      1,
      length.out = 200
    )
) {
  
  TPR <- numeric(
    length(
      thresholds
    )
  )
  
  
  FPR <- numeric(
    length(
      thresholds
    )
  )
  
  
  for (i in seq_along(
    thresholds
  )) {
    
    threshold_i <- thresholds[i]
    
    
    prediction <- ifelse(
      probability >= threshold_i,
      1,
      0
    )
    
    
    TP <- sum(
      prediction == 1 &
        y == 1
    )
    
    
    TN <- sum(
      prediction == 0 &
        y == 0
    )
    
    
    FP <- sum(
      prediction == 1 &
        y == 0
    )
    
    
    FN <- sum(
      prediction == 0 &
        y == 1
    )
    
    
    TPR[i] <- ifelse(
      TP + FN > 0,
      TP /
        (
          TP +
            FN
        ),
      NA
    )
    
    
    FPR[i] <- ifelse(
      FP + TN > 0,
      FP /
        (
          FP +
            TN
        ),
      NA
    )
  }
  
  
  return(
    data.frame(
      Threshold = thresholds,
      TPR = TPR,
      FPR = FPR
    )
  )
}


# ------------------------------------------------------------------------------
# 30. Plot ROC Curves
# ------------------------------------------------------------------------------

roc_linear <- roc_curve(
  y,
  linear_probability
)


roc_spline <- roc_curve(
  y,
  fitted_probability
)


plot(
  roc_linear$FPR,
  roc_linear$TPR,
  type = "l",
  lty = 2,
  lwd = 2,
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
  main = "ROC Curves"
)


lines(
  roc_spline$FPR,
  roc_spline$TPR,
  lty = 1,
  lwd = 2
)


abline(
  a = 0,
  b = 1,
  lty = 3
)


legend(
  "bottomright",
  legend = c(
    "Spline Logistic",
    "Linear Logistic"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ------------------------------------------------------------------------------
# 31. Rank-Based AUC
# ------------------------------------------------------------------------------

# For binary outcomes, AUC can be computed from ranks of predicted scores.


auc_rank <- function(
    y,
    probability
) {
  
  n_positive <- sum(
    y == 1
  )
  
  
  n_negative <- sum(
    y == 0
  )
  
  
  ranks <- rank(
    probability,
    ties.method = "average"
  )
  
  
  positive_rank_sum <- sum(
    ranks[
      y == 1
    ]
  )
  
  
  AUC <- (
    positive_rank_sum -
      n_positive *
      (
        n_positive +
          1
      ) /
      2
  ) /
    (
      n_positive *
        n_negative
    )
  
  
  return(
    AUC
  )
}


linear_AUC <- auc_rank(
  y,
  linear_probability
)


spline_AUC <- auc_rank(
  y,
  fitted_probability
)


data.frame(
  Model = c(
    "Linear Logistic",
    "Spline Logistic"
  ),
  AUC = c(
    linear_AUC,
    spline_AUC
  )
)


# ------------------------------------------------------------------------------
# 32. Explore Number of Knots
# ------------------------------------------------------------------------------

knot_counts <- c(
  0,
  1,
  2,
  3,
  5,
  8,
  12
)


knot_results <- data.frame(
  Number_of_Knots = knot_counts,
  Parameters = NA,
  Log_Likelihood = NA,
  Accuracy = NA,
  Brier = NA
)


knot_models <- vector(
  "list",
  length(
    knot_counts
  )
)


for (i in seq_along(
  knot_counts
)) {
  
  number_knots_i <- knot_counts[i]
  
  
  if (
    number_knots_i == 0
  ) {
    
    knots_i <- numeric(
      0
    )
    
  } else {
    
    probabilities <- seq(
      0,
      1,
      length.out = number_knots_i + 2
    )
    
    
    probabilities <- probabilities[
      -c(
        1,
        length(
          probabilities
        )
      )
    ]
    
    
    knots_i <- as.numeric(
      quantile(
        x,
        probs = probabilities
      )
    )
  }
  
  
  model_i <- spline_logistic_fit(
    x = x,
    y = y,
    knots = knots_i,
    degree = 3,
    boundary_knots = range(x)
  )
  
  
  probability_i <- spline_logistic_predict(
    x,
    model_i
  )$probability
  
  
  prediction_i <- ifelse(
    probability_i >= 0.5,
    1,
    0
  )
  
  
  B_i <- splines::bs(
    x,
    knots = knots_i,
    degree = 3,
    Boundary.knots = range(x),
    intercept = FALSE
  )
  
  
  X_i <- cbind(
    Intercept = 1,
    B_i
  )
  
  
  knot_results$Parameters[i] <- ncol(
    X_i
  )
  
  
  knot_results$Log_Likelihood[i] <-
    logistic_loglik(
      model_i$coefficients,
      X_i,
      y
    )
  
  
  knot_results$Accuracy[i] <- mean(
    prediction_i ==
      y
  )
  
  
  knot_results$Brier[i] <- mean(
    (
      y -
        probability_i
    )^2
  )
  
  
  knot_models[[i]] <- model_i
}


knot_results


# ------------------------------------------------------------------------------
# 33. Plot Model Complexity
# ------------------------------------------------------------------------------

plot(
  knot_results$Number_of_Knots,
  knot_results$Log_Likelihood,
  type = "b",
  pch = 19,
  xlab = "Number of Interior Knots",
  ylab = "Log-Likelihood",
  main = "Spline Complexity"
)


# ------------------------------------------------------------------------------
# 34. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)


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
# 35. Fit Train Model
# ------------------------------------------------------------------------------

training_knots <- as.numeric(
  quantile(
    x_train,
    probs = seq(
      0,
      1,
      length.out = 7
    )[
      -c(
        1,
        7
      )
    ]
  )
)


train_model <- spline_logistic_fit(
  x = x_train,
  y = y_train,
  knots = training_knots,
  degree = 3,
  boundary_knots = range(
    x_train
  )
)


# ------------------------------------------------------------------------------
# 36. Predict Test Observations
# ------------------------------------------------------------------------------

# Keep test points within the training-domain boundary for this demo.


inside_training_domain <-
  x_test >= min(
    x_train
  ) &
  x_test <= max(
    x_train
  )


test_probability <- spline_logistic_predict(
  x_test[
    inside_training_domain
  ],
  train_model
)$probability


test_prediction <- ifelse(
  test_probability >= 0.5,
  1,
  0
)


test_accuracy <- mean(
  test_prediction ==
    y_test[
      inside_training_domain
    ]
)


test_brier <- mean(
  (
    y_test[
      inside_training_domain
    ] -
      test_probability
  )^2
)


test_AUC <- auc_rank(
  y_test[
    inside_training_domain
  ],
  test_probability
)


data.frame(
  Test_Accuracy = test_accuracy,
  Test_Brier = test_brier,
  Test_AUC = test_AUC
)


# ------------------------------------------------------------------------------
# 37. K-Fold Cross-Validation for Number of Knots
# ------------------------------------------------------------------------------

spline_logistic_cv <- function(
    x,
    y,
    knot_counts,
    degree = 3,
    k = 5
) {
  
  n <- length(
    y
  )
  
  
  # Stratified folds
  
  folds <- integer(
    n
  )
  
  
  class_0 <- which(
    y == 0
  )
  
  
  class_1 <- which(
    y == 1
  )
  
  
  folds[class_0] <- sample(
    rep(
      seq_len(k),
      length.out = length(
        class_0
      )
    )
  )
  
  
  folds[class_1] <- sample(
    rep(
      seq_len(k),
      length.out = length(
        class_1
      )
    )
  )
  
  
  CV_logloss <- matrix(
    NA,
    nrow = length(
      knot_counts
    ),
    ncol = k
  )
  
  
  CV_accuracy <- matrix(
    NA,
    nrow = length(
      knot_counts
    ),
    ncol = k
  )
  
  
  for (fold in seq_len(
    k
  )) {
    
    train_index <- which(
      folds != fold
    )
    
    
    validation_index <- which(
      folds == fold
    )
    
    
    x_train <- x[
      train_index
    ]
    
    
    y_train <- y[
      train_index
    ]
    
    
    x_validation <- x[
      validation_index
    ]
    
    
    y_validation <- y[
      validation_index
    ]
    
    
    boundaries <- range(
      x_train
    )
    
    
    valid_domain <-
      x_validation >= boundaries[1] &
      x_validation <= boundaries[2]
    
    
    x_validation_use <- x_validation[
      valid_domain
    ]
    
    
    y_validation_use <- y_validation[
      valid_domain
    ]
    
    
    for (i in seq_along(
      knot_counts
    )) {
      
      number_knots_i <- knot_counts[i]
      
      
      if (
        number_knots_i == 0
      ) {
        
        knots_i <- numeric(
          0
        )
        
      } else {
        
        probabilities <- seq(
          0,
          1,
          length.out =
            number_knots_i +
            2
        )
        
        
        probabilities <- probabilities[
          -c(
            1,
            length(
              probabilities
            )
          )
        ]
        
        
        knots_i <- as.numeric(
          quantile(
            x_train,
            probs = probabilities
          )
        )
      }
      
      
      model_i <- spline_logistic_fit(
        x = x_train,
        y = y_train,
        knots = knots_i,
        degree = degree,
        boundary_knots = boundaries
      )
      
      
      probability_i <- spline_logistic_predict(
        x_validation_use,
        model_i
      )$probability
      
      
      probability_i <- pmin(
        pmax(
          probability_i,
          1e-12
        ),
        1 - 1e-12
      )
      
      
      CV_logloss[
        i,
        fold
      ] <- -mean(
        y_validation_use *
          log(
            probability_i
          ) +
          (
            1 -
              y_validation_use
          ) *
          log(
            1 -
              probability_i
          )
      )
      
      
      prediction_i <- ifelse(
        probability_i >= 0.5,
        1,
        0
      )
      
      
      CV_accuracy[
        i,
        fold
      ] <- mean(
        prediction_i ==
          y_validation_use
      )
    }
  }
  
  
  return(
    list(
      Knot_Count = knot_counts,
      Fold_LogLoss = CV_logloss,
      Mean_LogLoss = rowMeans(
        CV_logloss
      ),
      Mean_Accuracy = rowMeans(
        CV_accuracy
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 38. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


CV_results <- spline_logistic_cv(
  x = x,
  y = y,
  knot_counts = knot_counts,
  degree = 3,
  k = 5
)


CV_table <- data.frame(
  Number_of_Knots =
    CV_results$Knot_Count,
  CV_LogLoss =
    CV_results$Mean_LogLoss,
  CV_Accuracy =
    CV_results$Mean_Accuracy
)


CV_table


# ------------------------------------------------------------------------------
# 39. Select Best Number of Knots
# ------------------------------------------------------------------------------

best_index <- which.min(
  CV_results$Mean_LogLoss
)


best_knot_count <-
  CV_results$Knot_Count[
    best_index
  ]


best_knot_count


# ------------------------------------------------------------------------------
# 40. Plot Cross-Validated Log-Loss
# ------------------------------------------------------------------------------

plot(
  CV_results$Knot_Count,
  CV_results$Mean_LogLoss,
  type = "b",
  pch = 19,
  xlab = "Number of Interior Knots",
  ylab = "Cross-Validated Log-Loss",
  main = "Choosing Nonparametric Complexity"
)


# ------------------------------------------------------------------------------
# 41. Construct Final Knots
# ------------------------------------------------------------------------------

if (
  best_knot_count == 0
) {
  
  final_knots <- numeric(
    0
  )
  
} else {
  
  probabilities <- seq(
    0,
    1,
    length.out =
      best_knot_count +
      2
  )
  
  
  probabilities <- probabilities[
    -c(
      1,
      length(
        probabilities
      )
    )
  ]
  
  
  final_knots <- as.numeric(
    quantile(
      x,
      probs = probabilities
    )
  )
}


final_knots


# ------------------------------------------------------------------------------
# 42. Fit Final Nonparametric Logistic Model
# ------------------------------------------------------------------------------

final_model <- spline_logistic_fit(
  x = x,
  y = y,
  knots = final_knots,
  degree = 3,
  boundary_knots = range(x)
)


final_probability <- spline_logistic_predict(
  x,
  final_model
)$probability


final_prediction <- ifelse(
  final_probability >= 0.5,
  1,
  0
)


# ------------------------------------------------------------------------------
# 43. Final Metrics
# ------------------------------------------------------------------------------

final_accuracy <- mean(
  final_prediction ==
    y
)


final_brier <- mean(
  (
    y -
      final_probability
  )^2
)


final_AUC <- auc_rank(
  y,
  final_probability
)


data.frame(
  Accuracy = final_accuracy,
  Brier_Score = final_brier,
  AUC = final_AUC
)


# ------------------------------------------------------------------------------
# 44. Plot Final Probability Function
# ------------------------------------------------------------------------------

final_grid_probability <- spline_logistic_predict(
  x_grid,
  final_model
)$probability


plot(
  x,
  y,
  pch = 19,
  xlab = "x",
  ylab = "Probability / Response",
  main = "Final Nonparametric Logistic Regression"
)


lines(
  x_grid,
  final_grid_probability,
  lwd = 2
)


lines(
  x_grid,
  true_probability_grid,
  lty = 2,
  lwd = 2
)


abline(
  h = 0.5,
  lty = 3
)


# ------------------------------------------------------------------------------
# 45. Verify Against glm() with B-Spline Basis
# ------------------------------------------------------------------------------

builtin_model <- glm(
  y ~ splines::bs(
    x,
    knots = final_knots,
    degree = 3,
    Boundary.knots = range(x)
  ),
  family = binomial()
)


builtin_probability <- predict(
  builtin_model,
  type = "response"
)


# ------------------------------------------------------------------------------
# 46. Compare Manual and Built-In Probabilities
# ------------------------------------------------------------------------------

max_probability_difference <- max(
  abs(
    final_probability -
      builtin_probability
  )
)


max_probability_difference


plot(
  final_probability,
  builtin_probability,
  pch = 19,
  xlab = "Manual IRLS Probability",
  ylab = "glm() Probability",
  main = "Manual vs Built-In Logistic Fit"
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# ------------------------------------------------------------------------------
# 47. Compare Coefficients
# ------------------------------------------------------------------------------

manual_coefficients <-
  final_model$coefficients


builtin_coefficients <-
  coef(
    builtin_model
  )


data.frame(
  Manual = manual_coefficients,
  Built_In = builtin_coefficients
)


# ------------------------------------------------------------------------------
# 48. Summary
# ------------------------------------------------------------------------------

cat(
  "Best number of interior knots:",
  best_knot_count,
  "\n"
)


cat(
  "Training accuracy:",
  round(
    final_accuracy,
    4
  ),
  "\n"
)


cat(
  "Brier score:",
  round(
    final_brier,
    4
  ),
  "\n"
)


cat(
  "AUC:",
  round(
    final_AUC,
    4
  ),
  "\n"
)


cat(
  "Maximum difference from glm():",
  max_probability_difference,
  "\n"
)