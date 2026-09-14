# ==============================================================================
# Perceptron Learning Algorithm
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Binary classification
#   - Linear decision boundary
#   - Perceptron update rule
#   - Misclassification-driven learning
#   - Convergence for linearly separable data
#   - Effect of learning rate
#   - Comparison with logistic regression
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Linearly Separable Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 200


# Class -1

x1_negative <- rnorm(
  n / 2,
  mean = -2,
  sd = 0.8
)

x2_negative <- rnorm(
  n / 2,
  mean = -2,
  sd = 0.8
)


# Class +1

x1_positive <- rnorm(
  n / 2,
  mean = 2,
  sd = 0.8
)

x2_positive <- rnorm(
  n / 2,
  mean = 2,
  sd = 0.8
)


X <- rbind(
  cbind(
    x1_negative,
    x2_negative
  ),
  cbind(
    x1_positive,
    x2_positive
  )
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
  main = "Linearly Separable Classification Data"
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
# 3. Add Intercept
# ------------------------------------------------------------------------------

X_design <- cbind(
  Intercept = 1,
  X
)


head(
  X_design
)


# ------------------------------------------------------------------------------
# 4. Perceptron Prediction Function
# ------------------------------------------------------------------------------

perceptron_score <- function(
    X,
    weights
) {
  
  as.vector(
    X %*%
      weights
  )
}


perceptron_predict <- function(
    X,
    weights
) {
  
  scores <- perceptron_score(
    X,
    weights
  )
  
  
  prediction <- ifelse(
    scores >= 0,
    1,
    -1
  )
  
  
  return(prediction)
}


# ------------------------------------------------------------------------------
# 5. Perceptron Learning Algorithm
# ------------------------------------------------------------------------------

perceptron_fit <- function(
    X,
    y,
    learning_rate = 1,
    max_epochs = 1000,
    shuffle = TRUE
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  # Initialize weights
  
  weights <- rep(
    0,
    p
  )
  
  
  mistake_history <- numeric(
    max_epochs
  )
  
  
  weight_history <- matrix(
    NA,
    nrow = max_epochs + 1,
    ncol = p
  )
  
  weight_history[1, ] <-
    weights
  
  
  converged <- FALSE
  
  
  for (epoch in seq_len(
    max_epochs
  )) {
    
    mistakes <- 0
    
    
    if (shuffle) {
      
      observation_order <- sample(
        seq_len(n)
      )
      
    } else {
      
      observation_order <- seq_len(n)
    }
    
    
    for (i in observation_order) {
      
      score <- sum(
        weights *
          X[i, ]
      )
      
      
      predicted <- ifelse(
        score >= 0,
        1,
        -1
      )
      
      
      # ------------------------------------------------------------
      # Perceptron Update
      #
      # If observation i is misclassified:
      #
      # w <- w + eta * y_i * x_i
      # ------------------------------------------------------------
      
      if (
        predicted !=
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
    
    
    mistake_history[epoch] <-
      mistakes
    
    
    weight_history[
      epoch + 1,
    ] <- weights
    
    
    # Stop if all observations classified correctly
    
    if (mistakes == 0) {
      
      converged <- TRUE
      
      break
    }
  }
  
  
  mistake_history <- mistake_history[
    seq_len(epoch)
  ]
  
  
  weight_history <- weight_history[
    seq_len(epoch + 1),
    ,
    drop = FALSE
  ]
  
  
  colnames(weight_history) <-
    colnames(X)
  
  
  return(
    list(
      weights = weights,
      epochs = epoch,
      converged = converged,
      mistakes = mistake_history,
      weight_history = weight_history
    )
  )
}


# ------------------------------------------------------------------------------
# 6. Fit Perceptron
# ------------------------------------------------------------------------------

set.seed(123)


perceptron_model <- perceptron_fit(
  X_design,
  y,
  learning_rate = 1,
  max_epochs = 1000
)


perceptron_model$weights

perceptron_model$epochs

perceptron_model$converged


# ------------------------------------------------------------------------------
# 7. Training Predictions
# ------------------------------------------------------------------------------

train_prediction <- perceptron_predict(
  X_design,
  perceptron_model$weights
)


head(
  train_prediction
)


# ------------------------------------------------------------------------------
# 8. Training Accuracy
# ------------------------------------------------------------------------------

train_accuracy <- mean(
  train_prediction ==
    y
)


train_accuracy


# ------------------------------------------------------------------------------
# 9. Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y,
  Predicted = train_prediction
)


# ------------------------------------------------------------------------------
# 10. Error Rate
# ------------------------------------------------------------------------------

train_error <- mean(
  train_prediction !=
    y
)


train_error


# ------------------------------------------------------------------------------
# 11. Plot Mistakes per Epoch
# ------------------------------------------------------------------------------

plot(
  seq_along(
    perceptron_model$mistakes
  ),
  perceptron_model$mistakes,
  type = "b",
  pch = 19,
  xlab = "Epoch",
  ylab = "Number of Mistakes",
  main = "Perceptron Learning"
)


# ------------------------------------------------------------------------------
# 12. Decision Boundary
# ------------------------------------------------------------------------------

# The decision rule is
#
# w0 + w1*x1 + w2*x2 = 0
#
# Therefore:
#
# x2 = -(w0 + w1*x1) / w2


w <- perceptron_model$weights


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
  main = "Perceptron Decision Boundary"
)


if (
  abs(w[3]) > 1e-12
) {
  
  abline(
    a = -w[1] /
      w[3],
    b = -w[2] /
      w[3],
    lwd = 2
  )
}


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
# 13. Inspect Weight Evolution
# ------------------------------------------------------------------------------

perceptron_model$weight_history


# ------------------------------------------------------------------------------
# 14. Plot Weight Evolution
# ------------------------------------------------------------------------------

matplot(
  0:perceptron_model$epochs,
  perceptron_model$weight_history,
  type = "l",
  lty = 1,
  xlab = "Epoch",
  ylab = "Weight",
  main = "Perceptron Weight Evolution"
)

legend(
  "topright",
  legend = colnames(
    perceptron_model$weight_history
  ),
  lty = 1
)


# ------------------------------------------------------------------------------
# 15. Functional Margin
# ------------------------------------------------------------------------------

# For observation i:
#
# margin_i = y_i * (w'x_i)
#
# Correctly classified:
# margin_i > 0
#
# Misclassified:
# margin_i < 0


scores <- perceptron_score(
  X_design,
  w
)


functional_margin <- y *
  scores


summary(
  functional_margin
)


min(
  functional_margin
)


# ------------------------------------------------------------------------------
# 16. Plot Margins
# ------------------------------------------------------------------------------

plot(
  functional_margin,
  pch = 19,
  xlab = "Observation",
  ylab = "Functional Margin",
  main = "Perceptron Functional Margins"
)

abline(
  h = 0,
  lty = 2
)


# ------------------------------------------------------------------------------
# 17. Learning Rate Comparison
# ------------------------------------------------------------------------------

learning_rates <- c(
  0.01,
  0.1,
  1,
  10
)


learning_rate_results <- data.frame(
  Learning_Rate = learning_rates,
  Epochs = NA,
  Accuracy = NA,
  Converged = NA
)


for (i in seq_along(
  learning_rates
)) {
  
  set.seed(123)
  
  
  model_i <- perceptron_fit(
    X_design,
    y,
    learning_rate = learning_rates[i],
    max_epochs = 1000
  )
  
  
  prediction_i <- perceptron_predict(
    X_design,
    model_i$weights
  )
  
  
  learning_rate_results$Epochs[i] <-
    model_i$epochs
  
  
  learning_rate_results$Accuracy[i] <-
    mean(
      prediction_i ==
        y
    )
  
  
  learning_rate_results$Converged[i] <-
    model_i$converged
}


learning_rate_results


# ------------------------------------------------------------------------------
# 18. Train-Test Split
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


X_train <- X_design[
  train_index,
  ,
  drop = FALSE
]

X_test <- X_design[
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
# 19. Fit Perceptron on Training Data
# ------------------------------------------------------------------------------

set.seed(123)


train_model <- perceptron_fit(
  X_train,
  y_train,
  learning_rate = 1,
  max_epochs = 1000
)


train_model$weights


# ------------------------------------------------------------------------------
# 20. Test Prediction
# ------------------------------------------------------------------------------

test_prediction <- perceptron_predict(
  X_test,
  train_model$weights
)


test_accuracy <- mean(
  test_prediction ==
    y_test
)


test_accuracy


table(
  Actual = y_test,
  Predicted = test_prediction
)


# ------------------------------------------------------------------------------
# 21. Compare with Logistic Regression
# ------------------------------------------------------------------------------

# Convert labels from {-1, +1} to {0, 1}

y_binary <- ifelse(
  y == 1,
  1,
  0
)


logistic_data <- data.frame(
  y = y_binary,
  x1 = X[, 1],
  x2 = X[, 2]
)


logistic_model <- glm(
  y ~ x1 + x2,
  data = logistic_data,
  family = binomial()
)


summary(
  logistic_model
)


# ------------------------------------------------------------------------------
# 22. Logistic Regression Predictions
# ------------------------------------------------------------------------------

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


logistic_accuracy


# ------------------------------------------------------------------------------
# 23. Compare Classification Accuracy
# ------------------------------------------------------------------------------

comparison <- data.frame(
  Method = c(
    "Perceptron",
    "Logistic Regression"
  ),
  Accuracy = c(
    train_accuracy,
    logistic_accuracy
  )
)


comparison


# ------------------------------------------------------------------------------
# 24. Compare Decision Boundaries
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
  main = "Perceptron vs Logistic Regression"
)


# Perceptron boundary

w <- perceptron_model$weights


if (
  abs(w[3]) > 1e-12
) {
  
  abline(
    a = -w[1] /
      w[3],
    b = -w[2] /
      w[3],
    lwd = 2
  )
}


# Logistic regression boundary

beta_logistic <- coef(
  logistic_model
)


if (
  abs(beta_logistic[3]) > 1e-12
) {
  
  abline(
    a = -beta_logistic[1] /
      beta_logistic[3],
    b = -beta_logistic[2] /
      beta_logistic[3],
    lty = 2,
    lwd = 2
  )
}


legend(
  "topleft",
  legend = c(
    "Perceptron",
    "Logistic Regression"
  ),
  lty = c(
    1,
    2
  )
)


# ------------------------------------------------------------------------------
# 25. Nonseparable Data
# ------------------------------------------------------------------------------

# The classic perceptron convergence theorem only applies when
# the classes are linearly separable.


set.seed(456)


n_nonsep <- 200


X0 <- cbind(
  rnorm(
    n_nonsep / 2,
    mean = -1,
    sd = 1.5
  ),
  rnorm(
    n_nonsep / 2,
    mean = -1,
    sd = 1.5
  )
)


X1 <- cbind(
  rnorm(
    n_nonsep / 2,
    mean = 1,
    sd = 1.5
  ),
  rnorm(
    n_nonsep / 2,
    mean = 1,
    sd = 1.5
  )
)


X_nonsep <- rbind(
  X0,
  X1
)


y_nonsep <- c(
  rep(-1, n_nonsep / 2),
  rep(1, n_nonsep / 2)
)


X_nonsep_design <- cbind(
  Intercept = 1,
  X_nonsep
)


# ------------------------------------------------------------------------------
# 26. Plot Nonseparable Data
# ------------------------------------------------------------------------------

plot(
  X_nonsep[, 1],
  X_nonsep[, 2],
  pch = ifelse(
    y_nonsep == 1,
    19,
    1
  ),
  xlab = "x1",
  ylab = "x2",
  main = "Nonseparable Classification Data"
)


# ------------------------------------------------------------------------------
# 27. Fit Perceptron to Nonseparable Data
# ------------------------------------------------------------------------------

set.seed(123)


nonsep_model <- perceptron_fit(
  X_nonsep_design,
  y_nonsep,
  learning_rate = 1,
  max_epochs = 100
)


nonsep_model$converged

nonsep_model$epochs


# ------------------------------------------------------------------------------
# 28. Mistakes for Nonseparable Data
# ------------------------------------------------------------------------------

plot(
  seq_along(
    nonsep_model$mistakes
  ),
  nonsep_model$mistakes,
  type = "l",
  xlab = "Epoch",
  ylab = "Mistakes",
  main = "Perceptron on Nonseparable Data"
)


# ------------------------------------------------------------------------------
# 29. Pocket Perceptron
# ------------------------------------------------------------------------------

# For nonseparable data, keep the best-performing set
# of weights encountered during training.


pocket_perceptron <- function(
    X,
    y,
    learning_rate = 1,
    max_epochs = 100
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  weights <- rep(
    0,
    p
  )
  
  
  best_weights <- weights
  
  best_error <- Inf
  
  
  error_history <- numeric(
    max_epochs
  )
  
  
  for (epoch in seq_len(
    max_epochs
  )) {
    
    observation_order <- sample(
      seq_len(n)
    )
    
    
    for (i in observation_order) {
      
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
        
        
        current_prediction <- perceptron_predict(
          X,
          weights
        )
        
        
        current_error <- mean(
          current_prediction !=
            y
        )
        
        
        if (
          current_error <
          best_error
        ) {
          
          best_error <- current_error
          
          best_weights <- weights
        }
      }
    }
    
    
    error_history[epoch] <-
      best_error
  }
  
  
  return(
    list(
      weights = best_weights,
      error = best_error,
      error_history = error_history
    )
  )
}


# ------------------------------------------------------------------------------
# 30. Fit Pocket Perceptron
# ------------------------------------------------------------------------------

set.seed(123)


pocket_model <- pocket_perceptron(
  X_nonsep_design,
  y_nonsep,
  learning_rate = 1,
  max_epochs = 100
)


pocket_model$error

pocket_model$weights


# ------------------------------------------------------------------------------
# 31. Pocket Predictions
# ------------------------------------------------------------------------------

pocket_prediction <- perceptron_predict(
  X_nonsep_design,
  pocket_model$weights
)


pocket_accuracy <- mean(
  pocket_prediction ==
    y_nonsep
)


pocket_accuracy


# ------------------------------------------------------------------------------
# 32. Plot Pocket Error
# ------------------------------------------------------------------------------

plot(
  seq_along(
    pocket_model$error_history
  ),
  pocket_model$error_history,
  type = "l",
  xlab = "Epoch",
  ylab = "Best Error Rate",
  main = "Pocket Perceptron"
)


# ------------------------------------------------------------------------------
# 33. Pocket Decision Boundary
# ------------------------------------------------------------------------------

plot(
  X_nonsep[, 1],
  X_nonsep[, 2],
  pch = ifelse(
    y_nonsep == 1,
    19,
    1
  ),
  xlab = "x1",
  ylab = "x2",
  main = "Pocket Perceptron Decision Boundary"
)


w_pocket <- pocket_model$weights


if (
  abs(w_pocket[3]) > 1e-12
) {
  
  abline(
    a = -w_pocket[1] /
      w_pocket[3],
    b = -w_pocket[2] /
      w_pocket[3],
    lwd = 2
  )
}


# ------------------------------------------------------------------------------
# 34. Summary
# ------------------------------------------------------------------------------

cat(
  "Perceptron converged:",
  perceptron_model$converged,
  "\n"
)

cat(
  "Epochs to convergence:",
  perceptron_model$epochs,
  "\n"
)

cat(
  "Training accuracy:",
  round(
    train_accuracy,
    4
  ),
  "\n"
)

cat(
  "Test accuracy:",
  round(
    test_accuracy,
    4
  ),
  "\n"
)

cat(
  "Nonseparable model converged:",
  nonsep_model$converged,
  "\n"
)

cat(
  "Pocket accuracy:",
  round(
    pocket_accuracy,
    4
  ),
  "\n"
)