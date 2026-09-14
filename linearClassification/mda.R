# ==============================================================================
# Mixture Discriminant Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main idea:
#
#   LDA:
#       Class k ~ N(mu_k, Sigma)
#
#   MDA:
#       Class k ~ sum_j alpha_kj N(mu_kj, Sigma)
#
#   Each class may contain multiple Gaussian subclasses.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Multimodal Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n_per_cluster <- 80

Sigma_true <- matrix(
  c(
    0.55, 0.10,
    0.10, 0.55
  ),
  nrow = 2,
  byrow = TRUE
)


# ------------------------------------------------------------------------------
# 2. Multivariate Normal Generator
# ------------------------------------------------------------------------------

rmvn <- function(n, mu, Sigma) {
  
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
# 3. Generate Subclasses
# ------------------------------------------------------------------------------

X11 <- rmvn(
  n = n_per_cluster,
  mu = c(-3, 0),
  Sigma = Sigma_true
)

X12 <- rmvn(
  n = n_per_cluster,
  mu = c(3, 0),
  Sigma = Sigma_true
)

X21 <- rmvn(
  n = n_per_cluster,
  mu = c(0, -3),
  Sigma = Sigma_true
)

X22 <- rmvn(
  n = n_per_cluster,
  mu = c(0, 3),
  Sigma = Sigma_true
)


# ------------------------------------------------------------------------------
# 4. Combine Data
# ------------------------------------------------------------------------------

X <- rbind(
  X11,
  X12,
  X21,
  X22
)

colnames(X) <- c(
  "x1",
  "x2"
)

y <- factor(
  c(
    rep("Class1", 2 * n_per_cluster),
    rep("Class2", 2 * n_per_cluster)
  )
)

data <- data.frame(
  x1 = X[, 1],
  x2 = X[, 2],
  Class = y
)

head(data)

table(y)


# ------------------------------------------------------------------------------
# 5. Plot Data
# ------------------------------------------------------------------------------

plot(
  X[, 1],
  X[, 2],
  pch = as.numeric(y),
  xlab = "x1",
  ylab = "x2",
  main = "Multimodal Classification Data"
)

legend(
  "topright",
  legend = levels(y),
  pch = seq_along(levels(y))
)


# ------------------------------------------------------------------------------
# 6. Multivariate Normal Log-Density
# ------------------------------------------------------------------------------

log_mvnorm <- function(X, mu, Sigma) {
  
  X <- as.matrix(X)
  
  p <- ncol(X)
  
  centered <- sweep(
    X,
    MARGIN = 2,
    STATS = mu,
    FUN = "-"
  )
  
  Sigma_inv <- solve(Sigma)
  
  log_det <- as.numeric(
    determinant(
      Sigma,
      logarithm = TRUE
    )$modulus
  )
  
  quadratic <- rowSums(
    (centered %*% Sigma_inv) *
      centered
  )
  
  log_density <- -0.5 * (
    p * log(2 * pi) +
      log_det +
      quadratic
  )
  
  return(
    as.numeric(log_density)
  )
}


# ------------------------------------------------------------------------------
# 7. Log-Sum-Exp
# ------------------------------------------------------------------------------

log_sum_exp <- function(x) {
  
  m <- max(x)
  
  result <- m +
    log(
      sum(
        exp(x - m)
      )
    )
  
  return(
    as.numeric(result)
  )
}


# ------------------------------------------------------------------------------
# 8. Ordinary LDA for Comparison
# ------------------------------------------------------------------------------

lda_fit <- function(X, y) {
  
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
    
    means[k, ] <- colMeans(X_k)
    
    pooled_covariance <- pooled_covariance +
      (nrow(X_k) - 1) *
      cov(X_k)
  }
  
  pooled_covariance <- pooled_covariance /
    (n - K)
  
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
# 9. LDA Prediction
# ------------------------------------------------------------------------------

lda_predict <- function(X_new, model) {
  
  X_new <- as.matrix(X_new)
  
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
    classes[predicted_index],
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
# 10. Fit Ordinary LDA
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
  lda_predictions$class == y
)

lda_accuracy

table(
  Actual = y,
  Predicted = lda_predictions$class
)


# ------------------------------------------------------------------------------
# 11. MDA Fitting Function
# ------------------------------------------------------------------------------

mda_fit <- function(
    X,
    y,
    subclasses = 2,
    max_iter = 200,
    tol = 1e-6,
    ridge = 1e-6
) {
  
  X <- as.matrix(X)
  
  y <- factor(y)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  classes <- levels(y)
  
  K <- length(classes)
  
  
  # ---------------------------------------------------------------------------
  # Check Subclass Argument
  # ---------------------------------------------------------------------------
  
  if (subclasses < 1) {
    
    stop(
      "subclasses must be at least 1."
    )
  }
  
  
  # ---------------------------------------------------------------------------
  # Class Priors
  # ---------------------------------------------------------------------------
  
  priors <- as.numeric(
    table(y) / n
  )
  
  names(priors) <- classes
  
  
  # ---------------------------------------------------------------------------
  # Storage
  # ---------------------------------------------------------------------------
  
  mixture_weights <- vector(
    mode = "list",
    length = K
  )
  
  subclass_means <- vector(
    mode = "list",
    length = K
  )
  
  responsibilities <- vector(
    mode = "list",
    length = K
  )
  
  names(mixture_weights) <- classes
  
  names(subclass_means) <- classes
  
  names(responsibilities) <- classes
  
  
  # ---------------------------------------------------------------------------
  # Initialize Subclasses
  # ---------------------------------------------------------------------------
  
  for (k in seq_len(K)) {
    
    X_k <- X[
      y == classes[k],
      ,
      drop = FALSE
    ]
    
    n_k <- nrow(X_k)
    
    
    if (subclasses == 1) {
      
      cluster_assignment <- rep(
        1,
        n_k
      )
      
      means_k <- matrix(
        colMeans(X_k),
        nrow = 1
      )
      
      weights_k <- 1
      
    } else {
      
      km <- kmeans(
        X_k,
        centers = subclasses,
        nstart = 20
      )
      
      cluster_assignment <- km$cluster
      
      means_k <- km$centers
      
      counts_k <- tabulate(
        cluster_assignment,
        nbins = subclasses
      )
      
      weights_k <- counts_k /
        sum(counts_k)
    }
    
    
    R_k <- matrix(
      0,
      nrow = n_k,
      ncol = subclasses
    )
    
    R_k[
      cbind(
        seq_len(n_k),
        cluster_assignment
      )
    ] <- 1
    
    
    mixture_weights[[k]] <- as.numeric(
      weights_k
    )
    
    subclass_means[[k]] <- as.matrix(
      means_k
    )
    
    responsibilities[[k]] <- R_k
  }
  
  
  # ---------------------------------------------------------------------------
  # Initial Shared Covariance Matrix
  # ---------------------------------------------------------------------------
  
  Sigma <- matrix(
    0,
    nrow = p,
    ncol = p
  )
  
  total_weight <- 0
  
  
  for (k in seq_len(K)) {
    
    X_k <- X[
      y == classes[k],
      ,
      drop = FALSE
    ]
    
    R_k <- responsibilities[[k]]
    
    
    for (j in seq_len(subclasses)) {
      
      centered <- sweep(
        X_k,
        MARGIN = 2,
        STATS = subclass_means[[k]][j, ],
        FUN = "-"
      )
      
      weights_j <- R_k[, j]
      
      weighted_centered <- sweep(
        centered,
        MARGIN = 1,
        STATS = sqrt(weights_j),
        FUN = "*"
      )
      
      Sigma <- Sigma +
        crossprod(
          weighted_centered
        )
      
      total_weight <- total_weight +
        sum(weights_j)
    }
  }
  
  
  Sigma <- Sigma /
    total_weight
  
  Sigma <- Sigma +
    ridge *
    diag(p)
  
  
  # ---------------------------------------------------------------------------
  # EM Algorithm
  # ---------------------------------------------------------------------------
  
  log_likelihood_old <- -Inf
  
  log_likelihood_history <- numeric(0)
  
  
  for (iteration in seq_len(max_iter)) {
    
    # ==========================================================================
    # E-Step
    # ==========================================================================
    
    for (k in seq_len(K)) {
      
      X_k <- X[
        y == classes[k],
        ,
        drop = FALSE
      ]
      
      n_k <- nrow(X_k)
      
      
      log_probabilities <- matrix(
        NA,
        nrow = n_k,
        ncol = subclasses
      )
      
      
      for (j in seq_len(subclasses)) {
        
        log_probabilities[, j] <-
          log(
            mixture_weights[[k]][j]
          ) +
          log_mvnorm(
            X = X_k,
            mu = subclass_means[[k]][j, ],
            Sigma = Sigma
          )
      }
      
      
      R_k <- matrix(
        NA,
        nrow = n_k,
        ncol = subclasses
      )
      
      
      for (i in seq_len(n_k)) {
        
        denominator <- log_sum_exp(
          log_probabilities[i, ]
        )
        
        R_k[i, ] <- exp(
          log_probabilities[i, ] -
            denominator
        )
      }
      
      
      responsibilities[[k]] <- R_k
    }
    
    
    # ==========================================================================
    # M-Step: Mixture Weights and Means
    # ==========================================================================
    
    for (k in seq_len(K)) {
      
      X_k <- X[
        y == classes[k],
        ,
        drop = FALSE
      ]
      
      R_k <- responsibilities[[k]]
      
      n_k <- nrow(X_k)
      
      
      N_kj <- colSums(
        R_k
      )
      
      
      # Prevent division by zero
      
      N_kj <- pmax(
        N_kj,
        1e-12
      )
      
      
      mixture_weights[[k]] <- N_kj /
        sum(N_kj)
      
      
      means_k <- matrix(
        NA,
        nrow = subclasses,
        ncol = p
      )
      
      
      for (j in seq_len(subclasses)) {
        
        means_k[j, ] <- colSums(
          sweep(
            X_k,
            MARGIN = 1,
            STATS = R_k[, j],
            FUN = "*"
          )
        ) /
          N_kj[j]
      }
      
      
      subclass_means[[k]] <- means_k
    }
    
    
    # ==========================================================================
    # M-Step: Shared Covariance Matrix
    # ==========================================================================
    
    Sigma_new <- matrix(
      0,
      nrow = p,
      ncol = p
    )
    
    total_weight <- 0
    
    
    for (k in seq_len(K)) {
      
      X_k <- X[
        y == classes[k],
        ,
        drop = FALSE
      ]
      
      R_k <- responsibilities[[k]]
      
      
      for (j in seq_len(subclasses)) {
        
        centered <- sweep(
          X_k,
          MARGIN = 2,
          STATS = subclass_means[[k]][j, ],
          FUN = "-"
        )
        
        weights_j <- R_k[, j]
        
        
        weighted_centered <- sweep(
          centered,
          MARGIN = 1,
          STATS = sqrt(weights_j),
          FUN = "*"
        )
        
        
        Sigma_new <- Sigma_new +
          crossprod(
            weighted_centered
          )
        
        
        total_weight <- total_weight +
          sum(weights_j)
      }
    }
    
    
    Sigma <- Sigma_new /
      total_weight
    
    Sigma <- Sigma +
      ridge *
      diag(p)
    
    
    # ==========================================================================
    # Compute Log-Likelihood
    # ==========================================================================
    
    log_likelihood <- 0
    
    
    for (k in seq_len(K)) {
      
      X_k <- X[
        y == classes[k],
        ,
        drop = FALSE
      ]
      
      n_k <- nrow(X_k)
      
      
      for (i in seq_len(n_k)) {
        
        component_logs <- numeric(
          subclasses
        )
        
        
        for (j in seq_len(subclasses)) {
          
          component_logs[j] <-
            log(
              mixture_weights[[k]][j]
            ) +
            log_mvnorm(
              X = X_k[
                i,
                ,
                drop = FALSE
              ],
              mu = subclass_means[[k]][j, ],
              Sigma = Sigma
            )
        }
        
        
        log_likelihood <- log_likelihood +
          log_sum_exp(
            component_logs
          )
      }
    }
    
    
    log_likelihood_history <- c(
      log_likelihood_history,
      log_likelihood
    )
    
    
    # ==========================================================================
    # Check Convergence
    # ==========================================================================
    
    if (
      is.finite(log_likelihood_old) &&
      abs(
        log_likelihood -
        log_likelihood_old
      ) < tol
    ) {
      
      break
    }
    
    
    log_likelihood_old <- log_likelihood
  }
  
  
  return(
    list(
      classes = classes,
      priors = priors,
      weights = mixture_weights,
      means = subclass_means,
      covariance = Sigma,
      responsibilities = responsibilities,
      log_likelihood = log_likelihood,
      log_likelihood_history = log_likelihood_history,
      iterations = iteration,
      subclasses = subclasses
    )
  )
}


# ------------------------------------------------------------------------------
# 12. Fit MDA
# ------------------------------------------------------------------------------

set.seed(123)

mda_model <- mda_fit(
  X = X,
  y = y,
  subclasses = 2
)


mda_model$iterations

mda_model$log_likelihood

mda_model$weights

mda_model$means

mda_model$covariance


# ------------------------------------------------------------------------------
# 13. Plot EM Log-Likelihood
# ------------------------------------------------------------------------------

plot(
  seq_along(
    mda_model$log_likelihood_history
  ),
  mda_model$log_likelihood_history,
  type = "l",
  xlab = "EM Iteration",
  ylab = "Log-Likelihood",
  main = "MDA EM Convergence"
)


# ------------------------------------------------------------------------------
# 14. MDA Prediction Function
# ------------------------------------------------------------------------------

mda_predict <- function(
    X_new,
    model
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  classes <- model$classes
  
  K <- length(classes)
  
  J <- model$subclasses
  
  
  scores <- matrix(
    NA,
    nrow = nrow(X_new),
    ncol = K
  )
  
  colnames(scores) <- classes
  
  
  for (k in seq_len(K)) {
    
    for (i in seq_len(
      nrow(X_new)
    )) {
      
      component_logs <- numeric(
        J
      )
      
      
      for (j in seq_len(J)) {
        
        component_logs[j] <-
          log(
            model$weights[[k]][j]
          ) +
          log_mvnorm(
            X = X_new[
              i,
              ,
              drop = FALSE
            ],
            mu = model$means[[k]][j, ],
            Sigma = model$covariance
          )
      }
      
      
      scores[i, k] <-
        log(
          model$priors[k]
        ) +
        log_sum_exp(
          component_logs
        )
    }
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
# 15. Training Predictions
# ------------------------------------------------------------------------------

mda_predictions <- mda_predict(
  X,
  mda_model
)

head(
  mda_predictions$class
)


# ------------------------------------------------------------------------------
# 16. Training Confusion Matrix
# ------------------------------------------------------------------------------

mda_confusion <- table(
  Actual = y,
  Predicted = mda_predictions$class
)

mda_confusion


# ------------------------------------------------------------------------------
# 17. Training Accuracy
# ------------------------------------------------------------------------------

mda_accuracy <- mean(
  mda_predictions$class ==
    y
)

mda_accuracy


# ------------------------------------------------------------------------------
# 18. Compare LDA and MDA
# ------------------------------------------------------------------------------

comparison <- data.frame(
  Method = c(
    "LDA",
    "Mixture Discriminant Analysis"
  ),
  Accuracy = c(
    lda_accuracy,
    mda_accuracy
  ),
  Error_Rate = c(
    1 - lda_accuracy,
    1 - mda_accuracy
  )
)

comparison


# ------------------------------------------------------------------------------
# 19. Convert Scores to Posterior Probabilities
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
  
  colnames(probabilities) <-
    colnames(scores)
  
  
  for (i in seq_len(
    nrow(scores)
  )) {
    
    maximum <- max(
      scores[i, ]
    )
    
    values <- exp(
      scores[i, ] -
        maximum
    )
    
    probabilities[i, ] <-
      values /
      sum(values)
  }
  
  
  return(probabilities)
}


mda_posterior <- scores_to_posterior(
  mda_predictions$scores
)

head(
  mda_posterior
)

head(
  rowSums(
    mda_posterior
  )
)


# ------------------------------------------------------------------------------
# 20. Estimated Subclass Assignments
# ------------------------------------------------------------------------------

subclass_assignments <- integer(
  nrow(X)
)

start_index <- 1


for (k in seq_along(
  mda_model$classes
)) {
  
  class_index <- which(
    y ==
      mda_model$classes[k]
  )
  
  
  assignments <- max.col(
    mda_model$responsibilities[[k]],
    ties.method = "first"
  )
  
  
  subclass_assignments[
    class_index
  ] <- assignments
}


subclass_labels <- factor(
  paste(
    y,
    subclass_assignments,
    sep = "_Subclass"
  )
)


table(
  subclass_labels
)


# ------------------------------------------------------------------------------
# 21. Plot Estimated Subclasses
# ------------------------------------------------------------------------------

plot(
  X[, 1],
  X[, 2],
  pch = as.numeric(
    subclass_labels
  ),
  xlab = "x1",
  ylab = "x2",
  main = "Estimated MDA Subclasses"
)

legend(
  "topright",
  legend = levels(
    subclass_labels
  ),
  pch = seq_along(
    levels(
      subclass_labels
    )
  ),
  cex = 0.8
)


# ------------------------------------------------------------------------------
# 22. Plot Estimated Subclass Centers
# ------------------------------------------------------------------------------

plot(
  X[, 1],
  X[, 2],
  pch = as.numeric(y),
  xlab = "x1",
  ylab = "x2",
  main = "Estimated MDA Subclass Centers"
)


for (k in seq_along(
  mda_model$classes
)) {
  
  centers <- mda_model$means[[k]]
  
  points(
    centers[, 1],
    centers[, 2],
    pch = 8,
    cex = 2,
    lwd = 2
  )
}


# ------------------------------------------------------------------------------
# 23. Decision Boundary Grid
# ------------------------------------------------------------------------------

x1_grid <- seq(
  min(X[, 1]) - 1,
  max(X[, 1]) + 1,
  length.out = 200
)

x2_grid <- seq(
  min(X[, 2]) - 1,
  max(X[, 2]) + 1,
  length.out = 200
)


grid <- expand.grid(
  x1 = x1_grid,
  x2 = x2_grid
)


# ------------------------------------------------------------------------------
# 24. MDA Decision Regions
# ------------------------------------------------------------------------------

grid_prediction <- mda_predict(
  as.matrix(grid),
  mda_model
)


grid_class <- matrix(
  as.numeric(
    grid_prediction$class
  ),
  nrow = length(x1_grid),
  ncol = length(x2_grid)
)


image(
  x1_grid,
  x2_grid,
  grid_class,
  xlab = "x1",
  ylab = "x2",
  main = "MDA Decision Regions"
)


points(
  X[, 1],
  X[, 2],
  pch = as.numeric(y)
)


for (k in seq_along(
  mda_model$classes
)) {
  
  centers <- mda_model$means[[k]]
  
  points(
    centers[, 1],
    centers[, 2],
    pch = 8,
    cex = 2,
    lwd = 2
  )
}


# ------------------------------------------------------------------------------
# 25. LDA Decision Regions
# ------------------------------------------------------------------------------

grid_lda <- lda_predict(
  as.matrix(grid),
  lda_model
)


grid_class_lda <- matrix(
  as.numeric(
    grid_lda$class
  ),
  nrow = length(x1_grid),
  ncol = length(x2_grid)
)


image(
  x1_grid,
  x2_grid,
  grid_class_lda,
  xlab = "x1",
  ylab = "x2",
  main = "LDA Decision Regions"
)

points(
  X[, 1],
  X[, 2],
  pch = as.numeric(y)
)


# ------------------------------------------------------------------------------
# 26. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)

n <- nrow(X)

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


y_train <- y[
  train_index
]

y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 27. LDA Test Performance
# ------------------------------------------------------------------------------

lda_train <- lda_fit(
  X_train,
  y_train
)


lda_test_prediction <- lda_predict(
  X_test,
  lda_train
)


lda_test_accuracy <- mean(
  lda_test_prediction$class ==
    y_test
)


lda_test_accuracy


# ------------------------------------------------------------------------------
# 28. MDA Test Performance
# ------------------------------------------------------------------------------

set.seed(123)

mda_train <- mda_fit(
  X_train,
  y_train,
  subclasses = 2
)


mda_test_prediction <- mda_predict(
  X_test,
  mda_train
)


mda_test_accuracy <- mean(
  mda_test_prediction$class ==
    y_test
)


mda_test_accuracy


table(
  Actual = y_test,
  Predicted = mda_test_prediction$class
)


# ------------------------------------------------------------------------------
# 29. Compare Test Performance
# ------------------------------------------------------------------------------

test_comparison <- data.frame(
  Method = c(
    "LDA",
    "MDA"
  ),
  Accuracy = c(
    lda_test_accuracy,
    mda_test_accuracy
  ),
  Error_Rate = c(
    1 - lda_test_accuracy,
    1 - mda_test_accuracy
  )
)

test_comparison


# ------------------------------------------------------------------------------
# 30. Test Different Numbers of Subclasses
# ------------------------------------------------------------------------------

subclass_grid <- 1:4


subclass_accuracy <- numeric(
  length(subclass_grid)
)


for (m in seq_along(
  subclass_grid
)) {
  
  J <- subclass_grid[m]
  
  
  set.seed(123)
  
  
  model <- mda_fit(
    X_train,
    y_train,
    subclasses = J
  )
  
  
  prediction <- mda_predict(
    X_test,
    model
  )
  
  
  subclass_accuracy[m] <- mean(
    prediction$class ==
      y_test
  )
}


subclass_results <- data.frame(
  Subclasses = subclass_grid,
  Test_Accuracy = subclass_accuracy
)


subclass_results


# ------------------------------------------------------------------------------
# 31. Plot Number of Subclasses vs Test Accuracy
# ------------------------------------------------------------------------------

plot(
  subclass_results$Subclasses,
  subclass_results$Test_Accuracy,
  type = "b",
  pch = 19,
  xlab = "Subclasses per Class",
  ylab = "Test Accuracy",
  main = "MDA Model Complexity"
)


# ------------------------------------------------------------------------------
# 32. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

mda_cv <- function(
    X,
    y,
    subclass_grid = 1:4,
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
  
  
  errors <- matrix(
    NA,
    nrow = length(subclass_grid),
    ncol = k
  )
  
  
  rownames(errors) <- paste0(
    "J=",
    subclass_grid
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
      y[train_index]
    )
    
    y_valid <- y[
      valid_index
    ]
    
    
    for (m in seq_along(
      subclass_grid
    )) {
      
      J <- subclass_grid[m]
      
      
      model <- mda_fit(
        X_train,
        y_train,
        subclasses = J
      )
      
      
      prediction <- mda_predict(
        X_valid,
        model
      )
      
      
      errors[m, fold] <- mean(
        prediction$class !=
          y_valid
      )
    }
  }
  
  
  return(
    list(
      errors = errors,
      mean_error = rowMeans(
        errors
      ),
      subclasses = subclass_grid
    )
  )
}


# ------------------------------------------------------------------------------
# 33. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


cv_results <- mda_cv(
  X,
  y,
  subclass_grid = 1:4,
  k = 5
)


cv_table <- data.frame(
  Subclasses = cv_results$subclasses,
  CV_Error = cv_results$mean_error
)


cv_table


# ------------------------------------------------------------------------------
# 34. Select Best Number of Subclasses
# ------------------------------------------------------------------------------

best_index <- which.min(
  cv_results$mean_error
)


best_subclasses <- cv_results$subclasses[
  best_index
]


best_subclasses


# ------------------------------------------------------------------------------
# 35. Plot Cross-Validation Error
# ------------------------------------------------------------------------------

plot(
  cv_results$subclasses,
  cv_results$mean_error,
  type = "b",
  pch = 19,
  xlab = "Number of Subclasses per Class",
  ylab = "Cross-Validated Error",
  main = "MDA Cross-Validation"
)


# ------------------------------------------------------------------------------
# 36. Fit Final MDA Model
# ------------------------------------------------------------------------------

set.seed(123)


final_model <- mda_fit(
  X,
  y,
  subclasses = best_subclasses
)


final_model$weights

final_model$means

final_model$covariance


# ------------------------------------------------------------------------------
# 37. Final Predictions
# ------------------------------------------------------------------------------

final_prediction <- mda_predict(
  X,
  final_model
)


final_accuracy <- mean(
  final_prediction$class ==
    y
)


final_accuracy


# ------------------------------------------------------------------------------
# 38. Summary
# ------------------------------------------------------------------------------

cat(
  "LDA Training Accuracy:",
  round(
    lda_accuracy,
    4
  ),
  "\n"
)

cat(
  "MDA Training Accuracy:",
  round(
    mda_accuracy,
    4
  ),
  "\n"
)

cat(
  "MDA Test Accuracy:",
  round(
    mda_test_accuracy,
    4
  ),
  "\n"
)

cat(
  "Best Number of Subclasses:",
  best_subclasses,
  "\n"
)