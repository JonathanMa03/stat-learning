# ==============================================================================
# Support Vector Classifier
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Maximum-margin classification
#   - Hard vs soft margins
#   - Hinge loss
#   - Regularization parameter C
#   - Support vectors
#   - Margin violations
#   - Linear decision boundary
#   - Cross-validation for C
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Binary Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 240


# Class -1

X_negative <- cbind(
  rnorm(
    n / 2,
    mean = -1.5,
    sd = 1.2
  ),
  rnorm(
    n / 2,
    mean = -1.0,
    sd = 1.2
  )
)


# Class +1

X_positive <- cbind(
  rnorm(
    n / 2,
    mean = 1.5,
    sd = 1.2
  ),
  rnorm(
    n / 2,
    mean = 1.0,
    sd = 1.2
  )
)


X <- rbind(
  X_negative,
  X_positive
)

colnames(X) <- c(
  "x1",
  "x2"
)


y <- c(
  rep(-1, n / 2),
  rep(1, n / 2)
)


data <- data.frame(
  x1 = X[, 1],
  x2 = X[, 2],
  y = y
)


head(data)

table(y)


# ------------------------------------------------------------------------------
# 2. Plot Data
# ------------------------------------------------------------------------------

plot(
  X[, 1],
  X[, 2],
  pch = ifelse(
    y == 1,
    19,
    1
  ),
  xlab = "x1",
  ylab = "x2",
  main = "Support Vector Classifier Data"
)

legend(
  "topleft",
  legend = c(
    "Class -1",
    "Class +1"
  ),
  pch = c(
    1,
    19
  )
)


# ------------------------------------------------------------------------------
# 3. Standardize Predictors
# ------------------------------------------------------------------------------

X_means <- colMeans(
  X
)

X_sds <- apply(
  X,
  2,
  sd
)


X_scaled <- scale(
  X,
  center = X_means,
  scale = X_sds
)

X_scaled <- as.matrix(
  X_scaled
)


# ------------------------------------------------------------------------------
# 4. Hinge Loss
# ------------------------------------------------------------------------------

hinge_loss <- function(
    margins
) {
  
  pmax(
    0,
    1 - margins
  )
}


# Example

hinge_loss(
  c(
    2,
    1,
    0.5,
    0,
    -1
  )
)


# ------------------------------------------------------------------------------
# 5. SVC Objective Function
# ------------------------------------------------------------------------------

# We use:
#
#   1/2 ||w||^2 + C * sum_i max(0, 1 - y_i f(x_i))
#
# where:
#
#   f(x_i) = w'x_i + b
#
#
# A large C penalizes margin violations heavily.
# A small C allows more violations in exchange for a wider margin.


svc_objective <- function(
    X,
    y,
    w,
    b,
    C
) {
  
  scores <- as.vector(
    X %*% w + b
  )
  
  
  margins <- y *
    scores
  
  
  loss <- hinge_loss(
    margins
  )
  
  
  objective <- 0.5 *
    sum(
      w^2
    ) +
    C *
    sum(loss)
  
  
  return(objective)
}


# ------------------------------------------------------------------------------
# 6. Linear SVC via Subgradient Descent
# ------------------------------------------------------------------------------

svc_fit <- function(
    X,
    y,
    C = 1,
    learning_rate = 0.01,
    max_iter = 10000,
    tol = 1e-8,
    decay = 1e-4
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  # Initialize parameters
  
  w <- rep(
    0,
    p
  )
  
  b <- 0
  
  
  objective_history <- numeric(
    max_iter
  )
  
  
  for (iteration in seq_len(
    max_iter
  )) {
    
    scores <- as.vector(
      X %*% w + b
    )
    
    
    margins <- y *
      scores
    
    
    # Observations inside the margin or misclassified
    
    violation <- margins < 1
    
    
    # ------------------------------------------------------------
    # Subgradient of:
    #
    # 1/2 ||w||^2
    #
    # +
    #
    # C sum max(0, 1 - y_i f_i)
    # ------------------------------------------------------------
    
    grad_w <- w
    
    grad_b <- 0
    
    
    if (
      any(violation)
    ) {
      
      grad_w <- grad_w -
        C *
        colSums(
          sweep(
            X[
              violation,
              ,
              drop = FALSE
            ],
            MARGIN = 1,
            STATS = y[violation],
            FUN = "*"
          )
        )
      
      
      grad_b <- -C *
        sum(
          y[violation]
        )
    }
    
    
    # Decaying learning rate
    
    eta <- learning_rate /
      (
        1 +
          decay *
          iteration
      )
    
    
    w_new <- w -
      eta *
      grad_w
    
    
    b_new <- b -
      eta *
      grad_b
    
    
    objective_history[iteration] <-
      svc_objective(
        X,
        y,
        w_new,
        b_new,
        C
      )
    
    
    parameter_change <- max(
      c(
        abs(
          w_new -
            w
        ),
        abs(
          b_new -
            b
        )
      )
    )
    
    
    w <- w_new
    
    b <- b_new
    
    
    if (
      parameter_change <
      tol
    ) {
      
      break
    }
  }
  
  
  objective_history <- objective_history[
    seq_len(iteration)
  ]
  
  
  scores <- as.vector(
    X %*% w + b
  )
  
  
  margins <- y *
    scores
  
  
  return(
    list(
      weights = w,
      intercept = b,
      iterations = iteration,
      objective_history = objective_history,
      margins = margins,
      C = C
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Fit Support Vector Classifier
# ------------------------------------------------------------------------------

set.seed(123)


svc_model <- svc_fit(
  X_scaled,
  y,
  C = 1,
  learning_rate = 0.001,
  max_iter = 20000,
  tol = 1e-8
)


svc_model$weights

svc_model$intercept

svc_model$iterations


# ------------------------------------------------------------------------------
# 8. Prediction Function
# ------------------------------------------------------------------------------

svc_predict <- function(
    X_new,
    model
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  scores <- as.vector(
    X_new %*%
      model$weights +
      model$intercept
  )
  
  
  predicted <- ifelse(
    scores >= 0,
    1,
    -1
  )
  
  
  return(
    list(
      class = predicted,
      scores = scores
    )
  )
}


# ------------------------------------------------------------------------------
# 9. Training Predictions
# ------------------------------------------------------------------------------

train_prediction <- svc_predict(
  X_scaled,
  svc_model
)


head(
  train_prediction$class
)


# ------------------------------------------------------------------------------
# 10. Training Accuracy
# ------------------------------------------------------------------------------

train_accuracy <- mean(
  train_prediction$class ==
    y
)


train_accuracy


# ------------------------------------------------------------------------------
# 11. Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y,
  Predicted = train_prediction$class
)


# ------------------------------------------------------------------------------
# 12. Functional Margins
# ------------------------------------------------------------------------------

margins <- y *
  train_prediction$scores


summary(
  margins
)


# Correct classification:
#
# margin > 0
#
# Correct and outside margin:
#
# margin >= 1
#
# Inside margin:
#
# 0 < margin < 1
#
# Misclassified:
#
# margin < 0


# ------------------------------------------------------------------------------
# 13. Identify Margin Violations
# ------------------------------------------------------------------------------

margin_violation <- margins < 1


sum(
  margin_violation
)


# ------------------------------------------------------------------------------
# 14. Identify Misclassified Observations
# ------------------------------------------------------------------------------

misclassified <- margins < 0


sum(
  misclassified
)


# ------------------------------------------------------------------------------
# 15. Approximate Support Vectors
# ------------------------------------------------------------------------------

# In the exact SVM solution, support vectors are observations
# with nonzero dual coefficients.
#
# In this primal subgradient implementation, observations with
# y_i f(x_i) <= 1 are the points influencing the hinge-loss term.


support_vector_index <- which(
  margins <= 1 + 1e-4
)


length(
  support_vector_index
)


support_vectors <- X_scaled[
  support_vector_index,
  ,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# 16. Plot Objective Function
# ------------------------------------------------------------------------------

plot(
  seq_along(
    svc_model$objective_history
  ),
  svc_model$objective_history,
  type = "l",
  xlab = "Iteration",
  ylab = "Objective",
  main = "SVC Optimization"
)


# ------------------------------------------------------------------------------
# 17. Geometric Margin
# ------------------------------------------------------------------------------

w_norm <- sqrt(
  sum(
    svc_model$weights^2
  )
)


geometric_margin <- 1 /
  w_norm


geometric_margin


# Distance between the two margin hyperplanes:

margin_width <- 2 /
  w_norm


margin_width


# ------------------------------------------------------------------------------
# 18. Decision Boundary in Standardized Coordinates
# ------------------------------------------------------------------------------

w <- svc_model$weights

b <- svc_model$intercept


plot(
  X_scaled[, 1],
  X_scaled[, 2],
  pch = ifelse(
    y == 1,
    19,
    1
  ),
  xlab = "Standardized x1",
  ylab = "Standardized x2",
  main = "Support Vector Classifier"
)


# Decision boundary:
#
# w1*x1 + w2*x2 + b = 0

if (
  abs(w[2]) >
  1e-12
) {
  
  abline(
    a = -b /
      w[2],
    b = -w[1] /
      w[2],
    lwd = 2
  )
  
  
  # Positive margin:
  #
  # w'x + b = +1
  
  abline(
    a = (1 - b) /
      w[2],
    b = -w[1] /
      w[2],
    lty = 2
  )
  
  
  # Negative margin:
  #
  # w'x + b = -1
  
  abline(
    a = (-1 - b) /
      w[2],
    b = -w[1] /
      w[2],
    lty = 2
  )
}


# Highlight approximate support vectors

points(
  support_vectors[, 1],
  support_vectors[, 2],
  circles = rep(
    0.08,
    nrow(
      support_vectors
    )
  ),
  inches = FALSE,
  add = TRUE
)


legend(
  "topleft",
  legend = c(
    "Class -1",
    "Class +1",
    "Decision Boundary",
    "Margins"
  ),
  pch = c(
    1,
    19,
    NA,
    NA
  ),
  lty = c(
    NA,
    NA,
    1,
    2
  )
)


# ------------------------------------------------------------------------------
# 19. Plot Functional Margins
# ------------------------------------------------------------------------------

plot(
  margins,
  pch = 19,
  xlab = "Observation",
  ylab = "y * f(x)",
  main = "SVC Functional Margins"
)


abline(
  h = 1,
  lty = 2
)


abline(
  h = 0,
  lty = 3
)


# ------------------------------------------------------------------------------
# 20. Hinge Loss by Observation
# ------------------------------------------------------------------------------

individual_hinge_loss <- hinge_loss(
  margins
)


plot(
  individual_hinge_loss,
  type = "h",
  xlab = "Observation",
  ylab = "Hinge Loss",
  main = "Hinge Loss by Observation"
)


# ------------------------------------------------------------------------------
# 21. Compare Different Values of C
# ------------------------------------------------------------------------------

C_grid <- c(
  0.01,
  0.1,
  1,
  10,
  100
)


C_results <- data.frame(
  C = C_grid,
  Accuracy = NA,
  Margin_Width = NA,
  Margin_Violations = NA,
  Misclassified = NA
)


C_models <- vector(
  "list",
  length(
    C_grid
  )
)


for (i in seq_along(
  C_grid
)) {
  
  C_i <- C_grid[i]
  
  
  set.seed(123)
  
  
  model_i <- svc_fit(
    X_scaled,
    y,
    C = C_i,
    learning_rate = 0.001,
    max_iter = 20000
  )
  
  
  prediction_i <- svc_predict(
    X_scaled,
    model_i
  )
  
  
  margins_i <- y *
    prediction_i$scores
  
  
  C_results$Accuracy[i] <- mean(
    prediction_i$class ==
      y
  )
  
  
  C_results$Margin_Width[i] <- 2 /
    sqrt(
      sum(
        model_i$weights^2
      )
    )
  
  
  C_results$Margin_Violations[i] <- sum(
    margins_i < 1
  )
  
  
  C_results$Misclassified[i] <- sum(
    margins_i < 0
  )
  
  
  C_models[[i]] <- model_i
}


C_results


# ------------------------------------------------------------------------------
# 22. Plot Accuracy vs C
# ------------------------------------------------------------------------------

plot(
  log10(
    C_results$C
  ),
  C_results$Accuracy,
  type = "b",
  pch = 19,
  xlab = "log10(C)",
  ylab = "Training Accuracy",
  main = "Effect of C on Accuracy"
)


# ------------------------------------------------------------------------------
# 23. Plot Margin Width vs C
# ------------------------------------------------------------------------------

plot(
  log10(
    C_results$C
  ),
  C_results$Margin_Width,
  type = "b",
  pch = 19,
  xlab = "log10(C)",
  ylab = "Margin Width",
  main = "Effect of C on Margin Width"
)


# ------------------------------------------------------------------------------
# 24. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)


train_index <- sample(
  seq_len(n),
  size = floor(
    0.7 * n
  )
)


test_index <- setdiff(
  seq_len(n),
  train_index
)


X_train_raw <- X[
  train_index,
  ,
  drop = FALSE
]

X_test_raw <- X[
  test_index,
  ,
  drop = FALSE
]


y_train <- y[
  train_index
]

y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 25. Standardize Using Training Data Only
# ------------------------------------------------------------------------------

train_means <- colMeans(
  X_train_raw
)


train_sds <- apply(
  X_train_raw,
  2,
  sd
)


X_train <- scale(
  X_train_raw,
  center = train_means,
  scale = train_sds
)


X_test <- scale(
  X_test_raw,
  center = train_means,
  scale = train_sds
)


X_train <- as.matrix(
  X_train
)

X_test <- as.matrix(
  X_test
)


# ------------------------------------------------------------------------------
# 26. Fit Training SVC
# ------------------------------------------------------------------------------

set.seed(123)


train_model <- svc_fit(
  X_train,
  y_train,
  C = 1,
  learning_rate = 0.001,
  max_iter = 20000
)


# ------------------------------------------------------------------------------
# 27. Test Predictions
# ------------------------------------------------------------------------------

test_prediction <- svc_predict(
  X_test,
  train_model
)


test_accuracy <- mean(
  test_prediction$class ==
    y_test
)


test_accuracy


table(
  Actual = y_test,
  Predicted = test_prediction$class
)


# ------------------------------------------------------------------------------
# 28. K-Fold Cross-Validation for C
# ------------------------------------------------------------------------------

svc_cv <- function(
    X,
    y,
    C_grid,
    k = 5,
    learning_rate = 0.001,
    max_iter = 10000
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  n <- nrow(X)
  
  
  folds <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  errors <- matrix(
    NA,
    nrow = length(
      C_grid
    ),
    ncol = k
  )
  
  
  rownames(errors) <- paste0(
    "C=",
    C_grid
  )
  
  
  for (fold in seq_len(k)) {
    
    train_index <- which(
      folds != fold
    )
    
    valid_index <- which(
      folds == fold
    )
    
    
    X_train_raw <- X[
      train_index,
      ,
      drop = FALSE
    ]
    
    X_valid_raw <- X[
      valid_index,
      ,
      drop = FALSE
    ]
    
    
    y_train <- y[
      train_index
    ]
    
    y_valid <- y[
      valid_index
    ]
    
    
    # Training-fold standardization
    
    means <- colMeans(
      X_train_raw
    )
    
    
    sds <- apply(
      X_train_raw,
      2,
      sd
    )
    
    
    X_train <- scale(
      X_train_raw,
      center = means,
      scale = sds
    )
    
    
    X_valid <- scale(
      X_valid_raw,
      center = means,
      scale = sds
    )
    
    
    X_train <- as.matrix(
      X_train
    )
    
    X_valid <- as.matrix(
      X_valid
    )
    
    
    for (i in seq_along(
      C_grid
    )) {
      
      model <- svc_fit(
        X_train,
        y_train,
        C = C_grid[i],
        learning_rate = learning_rate,
        max_iter = max_iter
      )
      
      
      prediction <- svc_predict(
        X_valid,
        model
      )
      
      
      errors[i, fold] <- mean(
        prediction$class !=
          y_valid
      )
    }
  }
  
  
  mean_error <- rowMeans(
    errors
  )
  
  
  return(
    list(
      C = C_grid,
      errors = errors,
      mean_error = mean_error
    )
  )
}


# ------------------------------------------------------------------------------
# 29. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


cv_results <- svc_cv(
  X,
  y,
  C_grid = C_grid,
  k = 5,
  learning_rate = 0.001,
  max_iter = 10000
)


cv_table <- data.frame(
  C = cv_results$C,
  CV_Error = cv_results$mean_error
)


cv_table


# ------------------------------------------------------------------------------
# 30. Select Best C
# ------------------------------------------------------------------------------

best_index <- which.min(
  cv_results$mean_error
)


best_C <- cv_results$C[
  best_index
]


best_C


# ------------------------------------------------------------------------------
# 31. Plot Cross-Validation Error
# ------------------------------------------------------------------------------

plot(
  log10(
    cv_results$C
  ),
  cv_results$mean_error,
  type = "b",
  pch = 19,
  xlab = "log10(C)",
  ylab = "Cross-Validated Error",
  main = "SVC Cross-Validation"
)


# ------------------------------------------------------------------------------
# 32. Fit Final Model
# ------------------------------------------------------------------------------

set.seed(123)


final_model <- svc_fit(
  X_scaled,
  y,
  C = best_C,
  learning_rate = 0.001,
  max_iter = 20000
)


final_prediction <- svc_predict(
  X_scaled,
  final_model
)


final_accuracy <- mean(
  final_prediction$class ==
    y
)


final_accuracy


# ------------------------------------------------------------------------------
# 33. Compare SVC with Logistic Regression
# ------------------------------------------------------------------------------

y_binary <- ifelse(
  y == 1,
  1,
  0
)


logistic_data <- data.frame(
  y = y_binary,
  x1 = X_scaled[, 1],
  x2 = X_scaled[, 2]
)


logistic_model <- glm(
  y ~ x1 + x2,
  data = logistic_data,
  family = binomial()
)


logistic_probability <- predict(
  logistic_model,
  type = "response"
)


logistic_prediction <- ifelse(
  logistic_probability >= 0.5,
  1,
  -1
)


logistic_accuracy <- mean(
  logistic_prediction ==
    y
)


# ------------------------------------------------------------------------------
# 34. Compare SVC with Perceptron
# ------------------------------------------------------------------------------

perceptron_fit <- function(
    X,
    y,
    learning_rate = 1,
    max_epochs = 1000
) {
  
  X <- cbind(
    Intercept = 1,
    X
  )
  
  
  weights <- rep(
    0,
    ncol(X)
  )
  
  
  for (epoch in seq_len(
    max_epochs
  )) {
    
    mistakes <- 0
    
    
    for (i in sample(
      seq_len(
        nrow(X)
      )
    )) {
      
      prediction <- ifelse(
        sum(
          weights *
            X[i, ]
        ) >= 0,
        1,
        -1
      )
      
      
      if (
        prediction !=
        y[i]
      ) {
        
        weights <- weights +
          learning_rate *
          y[i] *
          X[i, ]
        
        
        mistakes <- mistakes +
          1
      }
    }
    
    
    if (
      mistakes == 0
    ) {
      
      break
    }
  }
  
  
  return(weights)
}


set.seed(123)


perceptron_weights <- perceptron_fit(
  X_scaled,
  y
)


perceptron_scores <- as.vector(
  cbind(
    1,
    X_scaled
  ) %*%
    perceptron_weights
)


perceptron_prediction <- ifelse(
  perceptron_scores >= 0,
  1,
  -1
)


perceptron_accuracy <- mean(
  perceptron_prediction ==
    y
)


# ------------------------------------------------------------------------------
# 35. Compare Classification Methods
# ------------------------------------------------------------------------------

method_comparison <- data.frame(
  Method = c(
    "Support Vector Classifier",
    "Logistic Regression",
    "Perceptron"
  ),
  Accuracy = c(
    final_accuracy,
    logistic_accuracy,
    perceptron_accuracy
  )
)


method_comparison


# ------------------------------------------------------------------------------
# 36. Summary
# ------------------------------------------------------------------------------

final_scores <- final_prediction$scores


final_margins <- y *
  final_scores


cat(
  "Best C:",
  best_C,
  "\n"
)


cat(
  "Training Accuracy:",
  round(
    final_accuracy,
    4
  ),
  "\n"
)


cat(
  "Test Accuracy:",
  round(
    test_accuracy,
    4
  ),
  "\n"
)


cat(
  "Approximate Support Vectors:",
  sum(
    final_margins <=
      1 +
      1e-4
  ),
  "\n"
)


cat(
  "Margin Violations:",
  sum(
    final_margins <
      1
  ),
  "\n"
)


cat(
  "Misclassified Observations:",
  sum(
    final_margins <
      0
  ),
  "\n"
)