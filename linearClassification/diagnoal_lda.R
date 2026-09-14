# ==============================================================================
# Diagonal Linear Discriminant Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - LDA with diagonal pooled covariance
#   - Ignores predictor correlations
#   - Uses class-specific means and shared feature variances
#   - Useful when p is large relative to n
#   - Compare with ordinary LDA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n1 <- 120
n2 <- 120

p <- 8


# Correlated predictors

Sigma_true <- matrix(
  0.5,
  nrow = p,
  ncol = p
)

diag(Sigma_true) <- 1


mu1 <- c(
  -1.5,
  -1.0,
  -0.5,
  0,
  0,
  0,
  0,
  0
)

mu2 <- c(
  1.5,
  1.0,
  0.5,
  0,
  0,
  0,
  0,
  0
)


# ------------------------------------------------------------------------------
# 2. Multivariate Normal Generator
# ------------------------------------------------------------------------------

rmvn <- function(
    n,
    mu,
    Sigma
) {
  
  p <- length(mu)
  
  Z <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )
  
  L <- chol(Sigma)
  
  X <- Z %*% L
  
  X <- sweep(
    X,
    MARGIN = 2,
    STATS = mu,
    FUN = "+"
  )
  
  return(X)
}


# ------------------------------------------------------------------------------
# 3. Generate Each Class
# ------------------------------------------------------------------------------

X1 <- rmvn(
  n = n1,
  mu = mu1,
  Sigma = Sigma_true
)

X2 <- rmvn(
  n = n2,
  mu = mu2,
  Sigma = Sigma_true
)


X <- rbind(
  X1,
  X2
)

colnames(X) <- paste0(
  "x",
  seq_len(p)
)


y <- factor(
  c(
    rep("Class1", n1),
    rep("Class2", n2)
  )
)


data <- data.frame(
  X,
  Class = y
)


head(data)

table(y)


# ------------------------------------------------------------------------------
# 4. Examine Predictor Correlations
# ------------------------------------------------------------------------------

round(
  cor(X),
  2
)


# ------------------------------------------------------------------------------
# 5. Plot First Two Predictors
# ------------------------------------------------------------------------------

plot(
  X[, 1],
  X[, 2],
  pch = as.numeric(y),
  xlab = "x1",
  ylab = "x2",
  main = "Diagonal LDA Data"
)

legend(
  "topright",
  legend = levels(y),
  pch = seq_along(levels(y))
)


# ------------------------------------------------------------------------------
# 6. Diagonal LDA Fitting Function
# ------------------------------------------------------------------------------

diag_lda_fit <- function(
    X,
    y,
    variance_floor = 1e-8
) {
  
  X <- as.matrix(X)
  
  y <- factor(y)
  
  classes <- levels(y)
  
  K <- length(classes)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  # ---------------------------------------------------------------------------
  # Class Priors
  # ---------------------------------------------------------------------------
  
  priors <- as.numeric(
    table(y) / n
  )
  
  names(priors) <- classes
  
  
  # ---------------------------------------------------------------------------
  # Class Means
  # ---------------------------------------------------------------------------
  
  class_means <- matrix(
    NA,
    nrow = K,
    ncol = p
  )
  
  rownames(class_means) <- classes
  
  colnames(class_means) <- colnames(X)
  
  
  for (k in seq_len(K)) {
    
    X_k <- X[
      y == classes[k],
      ,
      drop = FALSE
    ]
    
    class_means[k, ] <- colMeans(
      X_k
    )
  }
  
  
  # ---------------------------------------------------------------------------
  # Pooled Variances
  # ---------------------------------------------------------------------------
  
  pooled_variances <- rep(
    0,
    p
  )
  
  
  for (k in seq_len(K)) {
    
    X_k <- X[
      y == classes[k],
      ,
      drop = FALSE
    ]
    
    n_k <- nrow(X_k)
    
    pooled_variances <- pooled_variances +
      (n_k - 1) *
      apply(
        X_k,
        2,
        var
      )
  }
  
  
  pooled_variances <- pooled_variances /
    (n - K)
  
  
  # Avoid division by zero
  
  pooled_variances <- pmax(
    pooled_variances,
    variance_floor
  )
  
  
  names(pooled_variances) <- colnames(X)
  
  
  return(
    list(
      classes = classes,
      priors = priors,
      means = class_means,
      variances = pooled_variances
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Fit Diagonal LDA
# ------------------------------------------------------------------------------

diag_lda_model <- diag_lda_fit(
  X,
  y
)


diag_lda_model$priors

diag_lda_model$means

diag_lda_model$variances


# ------------------------------------------------------------------------------
# 8. Construct Diagonal Covariance Matrix
# ------------------------------------------------------------------------------

Sigma_diag <- diag(
  diag_lda_model$variances
)

Sigma_diag


# ------------------------------------------------------------------------------
# 9. Compare with Full Pooled Covariance
# ------------------------------------------------------------------------------

full_pooled_covariance <- matrix(
  0,
  nrow = p,
  ncol = p
)


classes <- levels(y)

K <- length(classes)

n <- nrow(X)


for (k in seq_len(K)) {
  
  X_k <- X[
    y == classes[k],
    ,
    drop = FALSE
  ]
  
  full_pooled_covariance <- full_pooled_covariance +
    (nrow(X_k) - 1) *
    cov(X_k)
}


full_pooled_covariance <- full_pooled_covariance /
  (n - K)


round(
  full_pooled_covariance,
  3
)


round(
  Sigma_diag,
  3
)


# ------------------------------------------------------------------------------
# 10. Diagonal LDA Discriminant Scores
# ------------------------------------------------------------------------------

# Ordinary LDA:
#
# delta_k(x) =
# x' Sigma^(-1) mu_k
# - 1/2 mu_k' Sigma^(-1) mu_k
# + log(pi_k)
#
#
# With diagonal Sigma:
#
# delta_k(x) =
# sum_j x_j * mu_kj / sigma_j^2
# - 1/2 sum_j mu_kj^2 / sigma_j^2
# + log(pi_k)


diag_lda_score <- function(
    X,
    mu,
    variances,
    prior
) {
  
  X <- as.matrix(X)
  
  linear_term <- rowSums(
    sweep(
      X,
      MARGIN = 2,
      STATS = mu / variances,
      FUN = "*"
    )
  )
  
  
  constant_term <- 0.5 *
    sum(
      mu^2 /
        variances
    )
  
  
  score <- linear_term -
    constant_term +
    log(prior)
  
  
  return(score)
}


# ------------------------------------------------------------------------------
# 11. Prediction Function
# ------------------------------------------------------------------------------

diag_lda_predict <- function(
    X_new,
    model
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  classes <- model$classes
  
  K <- length(classes)
  
  
  scores <- matrix(
    NA,
    nrow = nrow(X_new),
    ncol = K
  )
  
  colnames(scores) <- classes
  
  
  for (k in seq_len(K)) {
    
    scores[, k] <- diag_lda_score(
      X = X_new,
      mu = model$means[k, ],
      variances = model$variances,
      prior = model$priors[k]
    )
  }
  
  
  predicted_index <- max.col(
    scores,
    ties.method = "first"
  )
  
  
  predicted_class <- factor(
    classes[
      predicted_index
    ],
    levels = classes
  )
  
  
  return(
    list(
      class = predicted_class,
      scores = scores
    )
  )
}


# ------------------------------------------------------------------------------
# 12. Training Predictions
# ------------------------------------------------------------------------------

diag_predictions <- diag_lda_predict(
  X,
  diag_lda_model
)


head(
  diag_predictions$class
)


# ------------------------------------------------------------------------------
# 13. Confusion Matrix
# ------------------------------------------------------------------------------

diag_confusion <- table(
  Actual = y,
  Predicted = diag_predictions$class
)


diag_confusion


# ------------------------------------------------------------------------------
# 14. Training Accuracy
# ------------------------------------------------------------------------------

diag_accuracy <- mean(
  diag_predictions$class ==
    y
)


diag_accuracy


# ------------------------------------------------------------------------------
# 15. Posterior Probabilities
# ------------------------------------------------------------------------------

scores_to_posterior <- function(
    scores
) {
  
  scores <- as.matrix(
    scores
  )
  
  probabilities <- matrix(
    NA,
    nrow = nrow(scores),
    ncol = ncol(scores)
  )
  
  colnames(probabilities) <- colnames(scores)
  
  
  for (i in seq_len(
    nrow(scores)
  )) {
    
    m <- max(
      scores[i, ]
    )
    
    temp <- exp(
      scores[i, ] -
        m
    )
    
    probabilities[i, ] <- temp /
      sum(temp)
  }
  
  
  return(probabilities)
}


diag_posterior <- scores_to_posterior(
  diag_predictions$scores
)


head(
  diag_posterior
)


head(
  rowSums(
    diag_posterior
  )
)


# ------------------------------------------------------------------------------
# 16. Feature Contribution to the Discriminant Function
# ------------------------------------------------------------------------------

# For two classes, the decision rule can be written as:
#
# beta0 + beta1*x1 + ... + betap*xp
#
# where:
#
# beta_j =
# (mu_2j - mu_1j) / sigma_j^2


mu_class1 <- diag_lda_model$means[
  1,
]

mu_class2 <- diag_lda_model$means[
  2,
]


feature_weights <- (
  mu_class2 -
    mu_class1
) /
  diag_lda_model$variances


feature_weights


# ------------------------------------------------------------------------------
# 17. Rank Predictors by Absolute Discriminant Weight
# ------------------------------------------------------------------------------

feature_importance <- data.frame(
  Feature = colnames(X),
  Weight = feature_weights,
  Absolute_Weight = abs(
    feature_weights
  )
)


feature_importance <- feature_importance[
  order(
    feature_importance$Absolute_Weight,
    decreasing = TRUE
  ),
]


feature_importance


# ------------------------------------------------------------------------------
# 18. Plot Discriminant Weights
# ------------------------------------------------------------------------------

barplot(
  feature_weights,
  names.arg = colnames(X),
  xlab = "Predictor",
  ylab = "Discriminant Weight",
  main = "Diagonal LDA Feature Weights"
)


# ------------------------------------------------------------------------------
# 19. Ordinary LDA for Comparison
# ------------------------------------------------------------------------------

lda_fit <- function(
    X,
    y,
    ridge = 1e-8
) {
  
  X <- as.matrix(X)
  
  y <- factor(y)
  
  classes <- levels(y)
  
  K <- length(classes)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  priors <- as.numeric(
    table(y) / n
  )
  
  names(priors) <- classes
  
  
  means <- matrix(
    NA,
    nrow = K,
    ncol = p
  )
  
  rownames(means) <- classes
  
  colnames(means) <- colnames(X)
  
  
  pooled_covariance <- matrix(
    0,
    nrow = p,
    ncol = p
  )
  
  
  for (k in seq_len(K)) {
    
    X_k <- X[
      y == classes[k],
      ,
      drop = FALSE
    ]
    
    
    means[k, ] <- colMeans(
      X_k
    )
    
    
    pooled_covariance <- pooled_covariance +
      (nrow(X_k) - 1) *
      cov(X_k)
  }
  
  
  pooled_covariance <- pooled_covariance /
    (n - K)
  
  
  pooled_covariance <- pooled_covariance +
    ridge *
    diag(p)
  
  
  return(
    list(
      classes = classes,
      priors = priors,
      means = means,
      covariance = pooled_covariance
    )
  )
}


# ------------------------------------------------------------------------------
# 20. Ordinary LDA Prediction
# ------------------------------------------------------------------------------

lda_predict <- function(
    X_new,
    model
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  classes <- model$classes
  
  K <- length(classes)
  
  Sigma_inv <- solve(
    model$covariance
  )
  
  
  scores <- matrix(
    NA,
    nrow = nrow(X_new),
    ncol = K
  )
  
  colnames(scores) <- classes
  
  
  for (k in seq_len(K)) {
    
    mu_k <- as.numeric(
      model$means[k, ]
    )
    
    
    scores[, k] <- as.vector(
      X_new %*%
        Sigma_inv %*%
        mu_k
    ) -
      0.5 *
      as.numeric(
        t(mu_k) %*%
          Sigma_inv %*%
          mu_k
      ) +
      log(
        model$priors[k]
      )
  }
  
  
  predicted_index <- max.col(
    scores,
    ties.method = "first"
  )
  
  
  predicted_class <- factor(
    classes[
      predicted_index
    ],
    levels = classes
  )
  
  
  return(
    list(
      class = predicted_class,
      scores = scores
    )
  )
}


# ------------------------------------------------------------------------------
# 21. Fit Ordinary LDA
# ------------------------------------------------------------------------------

lda_model <- lda_fit(
  X,
  y
)


lda_predictions <- lda_predict(
  X,
  lda_model
)


lda_accuracy <- mean(
  lda_predictions$class ==
    y
)


lda_accuracy


# ------------------------------------------------------------------------------
# 22. Compare Training Accuracy
# ------------------------------------------------------------------------------

training_comparison <- data.frame(
  Method = c(
    "Ordinary LDA",
    "Diagonal LDA"
  ),
  Accuracy = c(
    lda_accuracy,
    diag_accuracy
  ),
  Error_Rate = c(
    1 - lda_accuracy,
    1 - diag_accuracy
  )
)


training_comparison


# ------------------------------------------------------------------------------
# 23. Train-Test Split
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


X_train <- X[
  train_index,
  ,
  drop = FALSE
]

X_test <- X[
  test_index,
  ,
  drop = FALSE
]


y_train <- droplevels(
  y[
    train_index
  ]
)

y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 24. Fit Diagonal LDA on Training Data
# ------------------------------------------------------------------------------

diag_train_model <- diag_lda_fit(
  X_train,
  y_train
)


diag_test_prediction <- diag_lda_predict(
  X_test,
  diag_train_model
)


diag_test_accuracy <- mean(
  diag_test_prediction$class ==
    y_test
)


diag_test_accuracy


# ------------------------------------------------------------------------------
# 25. Test Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y_test,
  Predicted = diag_test_prediction$class
)


# ------------------------------------------------------------------------------
# 26. Fit Ordinary LDA on Training Data
# ------------------------------------------------------------------------------

lda_train_model <- lda_fit(
  X_train,
  y_train
)


lda_test_prediction <- lda_predict(
  X_test,
  lda_train_model
)


lda_test_accuracy <- mean(
  lda_test_prediction$class ==
    y_test
)


lda_test_accuracy


# ------------------------------------------------------------------------------
# 27. Compare Test Performance
# ------------------------------------------------------------------------------

test_comparison <- data.frame(
  Method = c(
    "Ordinary LDA",
    "Diagonal LDA"
  ),
  Accuracy = c(
    lda_test_accuracy,
    diag_test_accuracy
  ),
  Error_Rate = c(
    1 - lda_test_accuracy,
    1 - diag_test_accuracy
  )
)


test_comparison


# ------------------------------------------------------------------------------
# 28. High-Dimensional Example
# ------------------------------------------------------------------------------

# Diagonal LDA becomes especially attractive when p is large
# relative to n because the full covariance matrix may become singular.


set.seed(456)

n_high <- 80

p_high <- 120


X_high <- matrix(
  rnorm(
    n_high *
      p_high
  ),
  nrow = n_high,
  ncol = p_high
)


y_high <- factor(
  rep(
    c(
      "Class1",
      "Class2"
    ),
    each = n_high / 2
  )
)


# Give the first 10 predictors class signal

X_high[
  y_high == "Class2",
  1:10
] <- X_high[
  y_high == "Class2",
  1:10
] + 1


colnames(X_high) <- paste0(
  "x",
  seq_len(p_high)
)


# ------------------------------------------------------------------------------
# 29. Fit Diagonal LDA in High Dimensions
# ------------------------------------------------------------------------------

diag_high_model <- diag_lda_fit(
  X_high,
  y_high
)


diag_high_prediction <- diag_lda_predict(
  X_high,
  diag_high_model
)


diag_high_accuracy <- mean(
  diag_high_prediction$class ==
    y_high
)


diag_high_accuracy


# ------------------------------------------------------------------------------
# 30. Full Covariance Rank
# ------------------------------------------------------------------------------

full_cov_high <- cov(
  X_high
)


dim(
  full_cov_high
)


qr(
  full_cov_high
)$rank


# Since p > n, the empirical covariance matrix cannot have full rank.


# ------------------------------------------------------------------------------
# 31. Cross-Validation Function
# ------------------------------------------------------------------------------

diag_lda_cv <- function(
    X,
    y,
    k = 5
) {
  
  X <- as.matrix(X)
  
  y <- factor(y)
  
  n <- nrow(X)
  
  
  folds <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  accuracy <- numeric(
    k
  )
  
  
  for (fold in seq_len(k)) {
    
    train_index <- which(
      folds != fold
    )
    
    valid_index <- which(
      folds == fold
    )
    
    
    X_train <- X[
      train_index,
      ,
      drop = FALSE
    ]
    
    X_valid <- X[
      valid_index,
      ,
      drop = FALSE
    ]
    
    
    y_train <- droplevels(
      y[
        train_index
      ]
    )
    
    y_valid <- y[
      valid_index
    ]
    
    
    model <- diag_lda_fit(
      X_train,
      y_train
    )
    
    
    prediction <- diag_lda_predict(
      X_valid,
      model
    )
    
    
    accuracy[fold] <- mean(
      prediction$class ==
        y_valid
    )
  }
  
  
  return(
    accuracy
  )
}


# ------------------------------------------------------------------------------
# 32. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


diag_cv_accuracy <- diag_lda_cv(
  X,
  y,
  k = 5
)


diag_cv_accuracy


mean(
  diag_cv_accuracy
)


# ------------------------------------------------------------------------------
# 33. Compare Number of Covariance Parameters
# ------------------------------------------------------------------------------

# Full LDA covariance:
#
# p(p + 1) / 2 parameters
#
# Diagonal LDA covariance:
#
# p parameters


full_cov_parameters <- p *
  (
    p + 1
  ) /
  2


diag_cov_parameters <- p


data.frame(
  Model = c(
    "Full LDA",
    "Diagonal LDA"
  ),
  Covariance_Parameters = c(
    full_cov_parameters,
    diag_cov_parameters
  )
)


# ------------------------------------------------------------------------------
# 34. Summary
# ------------------------------------------------------------------------------

cat(
  "Ordinary LDA Training Accuracy:",
  round(
    lda_accuracy,
    4
  ),
  "\n"
)

cat(
  "Diagonal LDA Training Accuracy:",
  round(
    diag_accuracy,
    4
  ),
  "\n"
)

cat(
  "Ordinary LDA Test Accuracy:",
  round(
    lda_test_accuracy,
    4
  ),
  "\n"
)

cat(
  "Diagonal LDA Test Accuracy:",
  round(
    diag_test_accuracy,
    4
  ),
  "\n"
)

cat(
  "Diagonal LDA Mean CV Accuracy:",
  round(
    mean(
      diag_cv_accuracy
    ),
    4
  ),
  "\n"
)