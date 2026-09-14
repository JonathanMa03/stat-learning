# ==============================================================================
# Logistic Regression
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
#
# Main ideas:
#   - Binary classification
#   - Logistic / sigmoid function
#   - Log-odds
#   - Maximum likelihood estimation
#   - Newton-Raphson / IRLS
#   - Predicted probabilities
#   - Classification thresholds
#   - Confusion matrix
#   - ROC curve and AUC
#   - Verification with glm()
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Binary Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 300

x1 <- rnorm(n)
x2 <- rnorm(n)

beta_true <- c(
  -0.5,
  2.0,
  -1.5
)


# Linear predictor

eta <- beta_true[1] +
  beta_true[2] * x1 +
  beta_true[3] * x2


# ------------------------------------------------------------------------------
# 2. Logistic Function
# ------------------------------------------------------------------------------

sigmoid <- function(z) {
  
  1 / (
    1 +
      exp(-z)
  )
}


probability <- sigmoid(
  eta
)


# Generate binary outcomes

y <- rbinom(
  n = n,
  size = 1,
  prob = probability
)


X <- cbind(
  x1,
  x2
)

colnames(X) <- c(
  "x1",
  "x2"
)


data <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2
)


head(data)

table(y)


# ------------------------------------------------------------------------------
# 3. Explore the Data
# ------------------------------------------------------------------------------

plot(
  x1,
  x2,
  pch = ifelse(
    y == 1,
    19,
    1
  ),
  xlab = "x1",
  ylab = "x2",
  main = "Binary Classification Data"
)

legend(
  "topright",
  legend = c(
    "Class 0",
    "Class 1"
  ),
  pch = c(
    1,
    19
  )
)


# ------------------------------------------------------------------------------
# 4. Design Matrix
# ------------------------------------------------------------------------------

X_design <- cbind(
  Intercept = 1,
  x1 = x1,
  x2 = x2
)


head(
  X_design
)


# ------------------------------------------------------------------------------
# 5. Logistic Regression Model
# ------------------------------------------------------------------------------

# The model assumes:
#
# P(Y = 1 | X) = p(x)
#
# where
#
# p(x) = 1 / (1 + exp(-X beta))
#
#
# The log-odds are linear:
#
# log(p / (1 - p)) = X beta


# ------------------------------------------------------------------------------
# 6. Log-Likelihood
# ------------------------------------------------------------------------------

log_likelihood <- function(
    beta,
    X,
    y
) {
  
  eta <- as.vector(
    X %*% beta
  )
  
  p <- sigmoid(
    eta
  )
  
  # Avoid log(0)
  
  epsilon <- 1e-12
  
  p <- pmin(
    pmax(
      p,
      epsilon
    ),
    1 - epsilon
  )
  
  
  ll <- sum(
    y * log(p) +
      (1 - y) *
      log(1 - p)
  )
  
  
  return(ll)
}


# ------------------------------------------------------------------------------
# 7. Evaluate Log-Likelihood at Initial Coefficients
# ------------------------------------------------------------------------------

beta_initial <- rep(
  0,
  ncol(X_design)
)


log_likelihood(
  beta_initial,
  X_design,
  y
)


# ------------------------------------------------------------------------------
# 8. Score / Gradient
# ------------------------------------------------------------------------------

logistic_gradient <- function(
    beta,
    X,
    y
) {
  
  eta <- as.vector(
    X %*% beta
  )
  
  p <- sigmoid(
    eta
  )
  
  
  gradient <- crossprod(
    X,
    y - p
  )
  
  
  return(
    as.vector(
      gradient
    )
  )
}


# ------------------------------------------------------------------------------
# 9. Hessian
# ------------------------------------------------------------------------------

logistic_hessian <- function(
    beta,
    X
) {
  
  eta <- as.vector(
    X %*% beta
  )
  
  p <- sigmoid(
    eta
  )
  
  w <- p *
    (1 - p)
  
  
  X_weighted <- sweep(
    X,
    MARGIN = 1,
    STATS = w,
    FUN = "*"
  )
  
  
  H <- -crossprod(
    X,
    X_weighted
  )
  
  
  return(H)
}


# ------------------------------------------------------------------------------
# 10. Newton-Raphson Logistic Regression
# ------------------------------------------------------------------------------

logistic_fit_newton <- function(
    X,
    y,
    beta_init = NULL,
    max_iter = 100,
    tol = 1e-8
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  p <- ncol(X)
  
  
  if (is.null(beta_init)) {
    
    beta <- rep(
      0,
      p
    )
    
  } else {
    
    beta <- beta_init
  }
  
  
  loglik_history <- numeric(
    max_iter
  )
  
  
  for (iteration in seq_len(
    max_iter
  )) {
    
    gradient <- logistic_gradient(
      beta,
      X,
      y
    )
    
    
    H <- logistic_hessian(
      beta,
      X
    )
    
    
    # Newton-Raphson:
    #
    # beta_new =
    # beta - H^(-1) gradient
    
    step <- solve(
      H,
      gradient
    )
    
    
    beta_new <- beta -
      step
    
    
    loglik_history[iteration] <-
      log_likelihood(
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
      ) < tol
    ) {
      
      beta <- beta_new
      
      break
    }
    
    
    beta <- beta_new
  }
  
  
  loglik_history <-
    loglik_history[
      seq_len(iteration)
    ]
  
  
  return(
    list(
      coefficients = beta,
      iterations = iteration,
      loglik_history = loglik_history
    )
  )
}


# ------------------------------------------------------------------------------
# 11. Fit Logistic Regression Manually
# ------------------------------------------------------------------------------

manual_model <- logistic_fit_newton(
  X_design,
  y
)


manual_model$coefficients

manual_model$iterations


# ------------------------------------------------------------------------------
# 12. Compare with True Coefficients
# ------------------------------------------------------------------------------

coefficient_comparison <- data.frame(
  Parameter = c(
    "Intercept",
    "x1",
    "x2"
  ),
  True = beta_true,
  Estimated = manual_model$coefficients
)


coefficient_comparison


# ------------------------------------------------------------------------------
# 13. Plot Log-Likelihood During Optimization
# ------------------------------------------------------------------------------

plot(
  seq_along(
    manual_model$loglik_history
  ),
  manual_model$loglik_history,
  type = "b",
  pch = 19,
  xlab = "Iteration",
  ylab = "Log-Likelihood",
  main = "Newton-Raphson Convergence"
)


# ------------------------------------------------------------------------------
# 14. Predicted Probabilities
# ------------------------------------------------------------------------------

beta_hat <- manual_model$coefficients


eta_hat <- as.vector(
  X_design %*%
    beta_hat
)


probability_hat <- sigmoid(
  eta_hat
)


head(
  probability_hat
)


summary(
  probability_hat
)


# ------------------------------------------------------------------------------
# 15. Classification with Threshold = 0.5
# ------------------------------------------------------------------------------

threshold <- 0.5


predicted_class <- ifelse(
  probability_hat >= threshold,
  1,
  0
)


head(
  predicted_class
)


# ------------------------------------------------------------------------------
# 16. Confusion Matrix
# ------------------------------------------------------------------------------

confusion_matrix <- table(
  Actual = y,
  Predicted = predicted_class
)


confusion_matrix


# ------------------------------------------------------------------------------
# 17. Classification Accuracy
# ------------------------------------------------------------------------------

accuracy <- mean(
  predicted_class ==
    y
)


accuracy


# ------------------------------------------------------------------------------
# 18. Classification Error Rate
# ------------------------------------------------------------------------------

error_rate <- 1 -
  accuracy


error_rate


# ------------------------------------------------------------------------------
# 19. Sensitivity and Specificity
# ------------------------------------------------------------------------------

TP <- sum(
  predicted_class == 1 &
    y == 1
)

TN <- sum(
  predicted_class == 0 &
    y == 0
)

FP <- sum(
  predicted_class == 1 &
    y == 0
)

FN <- sum(
  predicted_class == 0 &
    y == 1
)


sensitivity <- TP /
  (TP + FN)


specificity <- TN /
  (TN + FP)


precision <- TP /
  (TP + FP)


recall <- sensitivity


F1 <- 2 *
  precision *
  recall /
  (
    precision +
      recall
  )


data.frame(
  Metric = c(
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1"
  ),
  Value = c(
    accuracy,
    sensitivity,
    specificity,
    precision,
    F1
  )
)


# ------------------------------------------------------------------------------
# 20. Probability Distribution by Class
# ------------------------------------------------------------------------------

boxplot(
  probability_hat ~ y,
  xlab = "Observed Class",
  ylab = "Predicted Probability",
  main = "Predicted Probabilities by Class"
)


abline(
  h = 0.5,
  lty = 2
)


# ------------------------------------------------------------------------------
# 21. Decision Boundary
# ------------------------------------------------------------------------------

# At threshold 0.5:
#
# beta0 + beta1*x1 + beta2*x2 = 0
#
# Therefore:
#
# x2 = -(beta0 + beta1*x1) / beta2


plot(
  x1,
  x2,
  pch = ifelse(
    y == 1,
    19,
    1
  ),
  xlab = "x1",
  ylab = "x2",
  main = "Logistic Regression Decision Boundary"
)


abline(
  a = -beta_hat[1] /
    beta_hat[3],
  b = -beta_hat[2] /
    beta_hat[3],
  lwd = 2
)


legend(
  "topright",
  legend = c(
    "Class 0",
    "Class 1"
  ),
  pch = c(
    1,
    19
  )
)


# ------------------------------------------------------------------------------
# 22. Odds
# ------------------------------------------------------------------------------

odds_hat <- probability_hat /
  (
    1 -
      probability_hat
  )


head(
  odds_hat
)


# ------------------------------------------------------------------------------
# 23. Log-Odds
# ------------------------------------------------------------------------------

log_odds_hat <- log(
  odds_hat
)


head(
  log_odds_hat
)


# Verify log-odds equal the linear predictor

max(
  abs(
    log_odds_hat -
      eta_hat
  )
)


# ------------------------------------------------------------------------------
# 24. Odds Ratios
# ------------------------------------------------------------------------------

odds_ratios <- exp(
  beta_hat
)


data.frame(
  Parameter = names(beta_hat),
  Coefficient = beta_hat,
  Odds_Ratio = odds_ratios
)


# ------------------------------------------------------------------------------
# 25. Standard Errors
# ------------------------------------------------------------------------------

H_hat <- logistic_hessian(
  beta_hat,
  X_design
)


variance_covariance <- solve(
  -H_hat
)


standard_errors <- sqrt(
  diag(
    variance_covariance
  )
)


standard_errors


# ------------------------------------------------------------------------------
# 26. Wald Statistics
# ------------------------------------------------------------------------------

wald_z <- beta_hat /
  standard_errors


wald_p <- 2 *
  (
    1 -
      pnorm(
        abs(wald_z)
      )
  )


wald_table <- data.frame(
  Estimate = beta_hat,
  Std_Error = standard_errors,
  Z = wald_z,
  P_Value = wald_p
)


wald_table


# ------------------------------------------------------------------------------
# 27. Confidence Intervals
# ------------------------------------------------------------------------------

critical_value <- qnorm(
  0.975
)


CI_lower <- beta_hat -
  critical_value *
  standard_errors


CI_upper <- beta_hat +
  critical_value *
  standard_errors


confidence_intervals <- data.frame(
  Estimate = beta_hat,
  Lower = CI_lower,
  Upper = CI_upper
)


confidence_intervals


# ------------------------------------------------------------------------------
# 28. Odds Ratio Confidence Intervals
# ------------------------------------------------------------------------------

odds_ratio_CI <- data.frame(
  Odds_Ratio = exp(beta_hat),
  Lower = exp(CI_lower),
  Upper = exp(CI_upper)
)


odds_ratio_CI


# ------------------------------------------------------------------------------
# 29. Deviance
# ------------------------------------------------------------------------------

model_loglik <- log_likelihood(
  beta_hat,
  X_design,
  y
)


model_deviance <- -2 *
  model_loglik


model_deviance


# ------------------------------------------------------------------------------
# 30. Null Model
# ------------------------------------------------------------------------------

null_probability <- mean(
  y
)


null_loglik <- sum(
  y *
    log(
      null_probability
    ) +
    (1 - y) *
    log(
      1 -
        null_probability
    )
)


null_deviance <- -2 *
  null_loglik


null_deviance


# ------------------------------------------------------------------------------
# 31. Likelihood Ratio Test
# ------------------------------------------------------------------------------

LR_statistic <- null_deviance -
  model_deviance


LR_df <- ncol(
  X_design
) - 1


LR_p_value <- 1 -
  pchisq(
    LR_statistic,
    df = LR_df
  )


data.frame(
  Statistic = LR_statistic,
  Degrees_Freedom = LR_df,
  P_Value = LR_p_value
)


# ------------------------------------------------------------------------------
# 32. McFadden Pseudo R-Squared
# ------------------------------------------------------------------------------

mcfadden_R2 <- 1 -
  model_loglik /
  null_loglik


mcfadden_R2


# ------------------------------------------------------------------------------
# 33. ROC Curve
# ------------------------------------------------------------------------------

threshold_grid <- seq(
  0,
  1,
  length.out = 500
)


TPR <- numeric(
  length(
    threshold_grid
  )
)


FPR <- numeric(
  length(
    threshold_grid
  )
)


for (i in seq_along(
  threshold_grid
)) {
  
  threshold_i <-
    threshold_grid[i]
  
  
  predicted_i <- ifelse(
    probability_hat >=
      threshold_i,
    1,
    0
  )
  
  
  TP_i <- sum(
    predicted_i == 1 &
      y == 1
  )
  
  
  FN_i <- sum(
    predicted_i == 0 &
      y == 1
  )
  
  
  FP_i <- sum(
    predicted_i == 1 &
      y == 0
  )
  
  
  TN_i <- sum(
    predicted_i == 0 &
      y == 0
  )
  
  
  TPR[i] <- TP_i /
    (
      TP_i +
        FN_i
    )
  
  
  FPR[i] <- FP_i /
    (
      FP_i +
        TN_i
    )
}


# ------------------------------------------------------------------------------
# 34. Plot ROC Curve
# ------------------------------------------------------------------------------

plot(
  FPR,
  TPR,
  type = "l",
  lwd = 2,
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
  main = "ROC Curve",
  xlim = c(
    0,
    1
  ),
  ylim = c(
    0,
    1
  )
)


abline(
  a = 0,
  b = 1,
  lty = 2
)


# ------------------------------------------------------------------------------
# 35. Calculate AUC Manually
# ------------------------------------------------------------------------------

roc_order <- order(
  FPR
)


FPR_sorted <- FPR[
  roc_order
]

TPR_sorted <- TPR[
  roc_order
]


AUC <- sum(
  diff(
    FPR_sorted
  ) *
    (
      head(
        TPR_sorted,
        -1
      ) +
        tail(
          TPR_sorted,
          -1
        )
    ) /
    2
)


AUC


# ------------------------------------------------------------------------------
# 36. Threshold Comparison
# ------------------------------------------------------------------------------

thresholds_to_compare <- c(
  0.3,
  0.5,
  0.7
)


threshold_results <- data.frame(
  Threshold = thresholds_to_compare,
  Accuracy = NA,
  Sensitivity = NA,
  Specificity = NA
)


for (i in seq_along(
  thresholds_to_compare
)) {
  
  t <- thresholds_to_compare[i]
  
  
  prediction <- ifelse(
    probability_hat >= t,
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
  
  
  threshold_results$Accuracy[i] <-
    mean(
      prediction ==
        y
    )
  
  
  threshold_results$Sensitivity[i] <-
    TP /
    (
      TP +
        FN
    )
  
  
  threshold_results$Specificity[i] <-
    TN /
    (
      TN +
        FP
    )
}


threshold_results


# ------------------------------------------------------------------------------
# 37. Train-Test Split
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
# 38. Fit Training Model
# ------------------------------------------------------------------------------

train_model <- logistic_fit_newton(
  X_train,
  y_train
)


train_model$coefficients


# ------------------------------------------------------------------------------
# 39. Test Probabilities
# ------------------------------------------------------------------------------

test_eta <- as.vector(
  X_test %*%
    train_model$coefficients
)


test_probability <- sigmoid(
  test_eta
)


# ------------------------------------------------------------------------------
# 40. Test Predictions
# ------------------------------------------------------------------------------

test_prediction <- ifelse(
  test_probability >=
    0.5,
  1,
  0
)


test_accuracy <- mean(
  test_prediction ==
    y_test
)


test_accuracy


# ------------------------------------------------------------------------------
# 41. Test Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y_test,
  Predicted = test_prediction
)


# ------------------------------------------------------------------------------
# 42. Verification with glm()
# ------------------------------------------------------------------------------

glm_model <- glm(
  y ~ x1 + x2,
  data = data,
  family = binomial(
    link = "logit"
  )
)


summary(
  glm_model
)


# ------------------------------------------------------------------------------
# 43. Compare Manual and glm() Coefficients
# ------------------------------------------------------------------------------

comparison <- data.frame(
  Parameter = names(
    coef(
      glm_model
    )
  ),
  Manual = manual_model$coefficients,
  GLM = as.numeric(
    coef(
      glm_model
    )
  )
)


comparison


# ------------------------------------------------------------------------------
# 44. Maximum Difference Between Estimates
# ------------------------------------------------------------------------------

max(
  abs(
    manual_model$coefficients -
      coef(glm_model)
  )
)


# ------------------------------------------------------------------------------
# 45. Compare Standard Errors
# ------------------------------------------------------------------------------

manual_SE <- standard_errors


glm_SE <- coef(
  summary(
    glm_model
  )
)[
  ,
  "Std. Error"
]


data.frame(
  Parameter = names(
    coef(glm_model)
  ),
  Manual_SE = manual_SE,
  GLM_SE = glm_SE
)


# ------------------------------------------------------------------------------
# 46. Compare Predicted Probabilities
# ------------------------------------------------------------------------------

glm_probability <- predict(
  glm_model,
  type = "response"
)


max(
  abs(
    probability_hat -
      glm_probability
  )
)


# ------------------------------------------------------------------------------
# 47. IRLS Interpretation
# ------------------------------------------------------------------------------

# Newton-Raphson for logistic regression is equivalent
# to Iteratively Reweighted Least Squares (IRLS).
#
# At each iteration:
#
# W_i = p_i (1 - p_i)
#
# z = X beta + (y - p) / W
#
# beta_new = (X' W X)^(-1) X' W z


logistic_fit_irls <- function(
    X,
    y,
    beta_init = NULL,
    max_iter = 100,
    tol = 1e-8
) {
  
  X <- as.matrix(X)
  
  y <- as.numeric(y)
  
  p <- ncol(X)
  
  
  if (is.null(beta_init)) {
    
    beta <- rep(
      0,
      p
    )
    
  } else {
    
    beta <- beta_init
  }
  
  
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
    
    
    # Prevent numerical instability
    
    weights <- pmax(
      weights,
      1e-8
    )
    
    
    working_response <- eta +
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
    
    
    beta_new <- solve(
      crossprod(
        X,
        X_weighted
      ),
      crossprod(
        X,
        weights *
          working_response
      )
    )
    
    
    beta_new <- as.vector(
      beta_new
    )
    
    
    if (
      max(
        abs(
          beta_new -
          beta
        )
      ) < tol
    ) {
      
      beta <- beta_new
      
      break
    }
    
    
    beta <- beta_new
  }
  
  
  return(
    list(
      coefficients = beta,
      iterations = iteration
    )
  )
}


# ------------------------------------------------------------------------------
# 48. Fit with IRLS
# ------------------------------------------------------------------------------

irls_model <- logistic_fit_irls(
  X_design,
  y
)


irls_model$coefficients


# ------------------------------------------------------------------------------
# 49. Compare Newton-Raphson, IRLS, and glm()
# ------------------------------------------------------------------------------

data.frame(
  Parameter = names(
    coef(glm_model)
  ),
  Newton_Raphson = manual_model$coefficients,
  IRLS = irls_model$coefficients,
  GLM = as.numeric(
    coef(glm_model)
  )
)


# ------------------------------------------------------------------------------
# 50. Summary
# ------------------------------------------------------------------------------

cat(
  "Iterations:",
  manual_model$iterations,
  "\n"
)

cat(
  "Training Accuracy:",
  round(
    accuracy,
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
  "AUC:",
  round(
    AUC,
    4
  ),
  "\n"
)

cat(
  "McFadden R-squared:",
  round(
    mcfadden_R2,
    4
  ),
  "\n"
)