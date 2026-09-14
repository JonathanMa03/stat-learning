# ==============================================================================
# Linear Discriminant Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n1 <- 100
n2 <- 100
n3 <- 100

p <- 2


# Common covariance matrix assumed by LDA

Sigma <- matrix(
  c(
    1.0, 0.5,
    0.5, 1.0
  ),
  nrow = 2,
  byrow = TRUE
)


# Class means

mu1 <- c(
  -2,
  0
)

mu2 <- c(
  2,
  0
)

mu3 <- c(
  0,
  3
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
    2,
    mu,
    "+"
  )
  
  return(X)
}


# ------------------------------------------------------------------------------
# 3. Generate Each Class
# ------------------------------------------------------------------------------

X1 <- rmvn(
  n1,
  mu1,
  Sigma
)

X2 <- rmvn(
  n2,
  mu2,
  Sigma
)

X3 <- rmvn(
  n3,
  mu3,
  Sigma
)


X <- rbind(
  X1,
  X2,
  X3
)

colnames(X) <- c(
  "x1",
  "x2"
)


y <- factor(
  c(
    rep("Class1", n1),
    rep("Class2", n2),
    rep("Class3", n3)
  )
)


data <- data.frame(
  X,
  Class = y
)

head(data)

table(data$Class)


# ------------------------------------------------------------------------------
# 4. Plot the Data
# ------------------------------------------------------------------------------

plot(
  X,
  pch = as.numeric(y),
  xlab = "x1",
  ylab = "x2",
  main = "Linear Discriminant Analysis Data"
)

legend(
  "topright",
  legend = levels(y),
  pch = 1:length(levels(y))
)


# ------------------------------------------------------------------------------
# 5. Class Priors
# ------------------------------------------------------------------------------

classes <- levels(y)

K <- length(classes)

n <- nrow(X)


priors <- table(y) /
  n

priors


# ------------------------------------------------------------------------------
# 6. Estimate Class Means
# ------------------------------------------------------------------------------

class_means <- matrix(
  NA,
  nrow = K,
  ncol = p
)

rownames(class_means) <- classes

colnames(class_means) <- colnames(X)


for (k in 1:K) {
  
  class_means[k, ] <- colMeans(
    X[
      y == classes[k],
      ,
      drop = FALSE
    ]
  )
}


class_means


# ------------------------------------------------------------------------------
# 7. Estimate Class-Specific Covariance Matrices
# ------------------------------------------------------------------------------

class_covariances <- vector(
  "list",
  K
)

names(class_covariances) <- classes


for (k in 1:K) {
  
  X_k <- X[
    y == classes[k],
    ,
    drop = FALSE
  ]
  
  class_covariances[[k]] <- cov(
    X_k
  )
}


class_covariances


# ------------------------------------------------------------------------------
# 8. Estimate Pooled Covariance Matrix
# ------------------------------------------------------------------------------

# LDA assumes all classes share the same covariance matrix.

pooled_covariance <- matrix(
  0,
  nrow = p,
  ncol = p
)


for (k in 1:K) {
  
  X_k <- X[
    y == classes[k],
    ,
    drop = FALSE
  ]
  
  n_k <- nrow(X_k)
  
  pooled_covariance <-
    pooled_covariance +
    (n_k - 1) *
    cov(X_k)
}


pooled_covariance <-
  pooled_covariance /
  (n - K)


pooled_covariance


# ------------------------------------------------------------------------------
# 9. Inverse Pooled Covariance
# ------------------------------------------------------------------------------

Sigma_inv <- solve(
  pooled_covariance
)

Sigma_inv


# ------------------------------------------------------------------------------
# 10. LDA Discriminant Function
# ------------------------------------------------------------------------------

# delta_k(x) =
#
# x' Sigma^(-1) mu_k
# - 1/2 mu_k' Sigma^(-1) mu_k
# + log(pi_k)


lda_score <- function(
    x,
    mu,
    Sigma_inv,
    prior
) {
  
  linear_term <-
    as.numeric(
      t(x) %*%
        Sigma_inv %*%
        mu
    )
  
  quadratic_mean_term <-
    0.5 *
    as.numeric(
      t(mu) %*%
        Sigma_inv %*%
        mu
    )
  
  score <-
    linear_term -
    quadratic_mean_term +
    log(prior)
  
  return(score)
}


# ------------------------------------------------------------------------------
# 11. Example Discriminant Scores
# ------------------------------------------------------------------------------

x_new <- c(
  1,
  1
)


scores <- numeric(
  K
)

names(scores) <- classes


for (k in 1:K) {
  
  scores[k] <- lda_score(
    x = x_new,
    mu = class_means[k, ],
    Sigma_inv = Sigma_inv,
    prior = priors[k]
  )
}


scores


# ------------------------------------------------------------------------------
# 12. Class Prediction
# ------------------------------------------------------------------------------

predicted_class <- names(
  which.max(scores)
)

predicted_class


# ------------------------------------------------------------------------------
# 13. Prediction Function
# ------------------------------------------------------------------------------

lda_predict <- function(
    X_new,
    class_means,
    Sigma,
    priors
) {
  
  classes <- rownames(
    class_means
  )
  
  K <- length(classes)
  
  Sigma_inv <- solve(
    Sigma
  )
  
  scores <- matrix(
    NA,
    nrow = nrow(X_new),
    ncol = K
  )
  
  colnames(scores) <- classes
  
  
  for (i in 1:nrow(X_new)) {
    
    for (k in 1:K) {
      
      scores[i, k] <- lda_score(
        x = X_new[i, ],
        mu = class_means[k, ],
        Sigma_inv = Sigma_inv,
        prior = priors[k]
      )
    }
  }
  
  
  predicted_index <- max.col(
    scores
  )
  
  
  predicted_class <- classes[
    predicted_index
  ]
  
  
  return(
    list(
      class = factor(
        predicted_class,
        levels = classes
      ),
      scores = scores
    )
  )
}


# ------------------------------------------------------------------------------
# 14. Predict Training Observations
# ------------------------------------------------------------------------------

train_predictions <- lda_predict(
  X,
  class_means,
  pooled_covariance,
  priors
)

head(
  train_predictions$class
)


# ------------------------------------------------------------------------------
# 15. Confusion Matrix
# ------------------------------------------------------------------------------

confusion_matrix <- table(
  Actual = y,
  Predicted = train_predictions$class
)

confusion_matrix


# ------------------------------------------------------------------------------
# 16. Classification Accuracy
# ------------------------------------------------------------------------------

accuracy <- mean(
  y ==
    train_predictions$class
)

accuracy


# ------------------------------------------------------------------------------
# 17. Classification Error Rate
# ------------------------------------------------------------------------------

error_rate <- 1 -
  accuracy

error_rate


# ------------------------------------------------------------------------------
# 18. Convert Discriminant Scores to Posterior Probabilities
# ------------------------------------------------------------------------------

# Posterior probabilities are proportional to exp(delta_k(x)).

score_to_probability <- function(
    scores
) {
  
  # Numerical stabilization
  
  scores_shifted <- scores -
    max(scores)
  
  probabilities <- exp(
    scores_shifted
  )
  
  probabilities <-
    probabilities /
    sum(probabilities)
  
  return(probabilities)
}


example_probabilities <- score_to_probability(
  scores
)

example_probabilities


# ------------------------------------------------------------------------------
# 19. Posterior Probability Function
# ------------------------------------------------------------------------------

lda_posterior <- function(
    score_matrix
) {
  
  probabilities <- matrix(
    NA,
    nrow = nrow(score_matrix),
    ncol = ncol(score_matrix)
  )
  
  colnames(probabilities) <-
    colnames(score_matrix)
  
  
  for (i in 1:nrow(score_matrix)) {
    
    probabilities[i, ] <-
      score_to_probability(
        score_matrix[i, ]
      )
  }
  
  
  return(probabilities)
}


posterior_probabilities <- lda_posterior(
  train_predictions$scores
)


head(
  posterior_probabilities
)


# ------------------------------------------------------------------------------
# 20. Verify Posterior Probabilities Sum to One
# ------------------------------------------------------------------------------

head(
  rowSums(
    posterior_probabilities
  )
)


# ------------------------------------------------------------------------------
# 21. Most Uncertain Observations
# ------------------------------------------------------------------------------

max_probability <- apply(
  posterior_probabilities,
  1,
  max
)


uncertain_index <- order(
  max_probability
)[1:10]


data[
  uncertain_index,
]


posterior_probabilities[
  uncertain_index,
]


# ------------------------------------------------------------------------------
# 22. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(123)

train_index <- sample(
  1:n,
  size = floor(
    0.7 * n
  )
)


test_index <- setdiff(
  1:n,
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
# 23. LDA Fitting Function
# ------------------------------------------------------------------------------

lda_fit <- function(
    X,
    y
) {
  
  y <- factor(y)
  
  classes <- levels(y)
  
  K <- length(classes)
  
  n <- nrow(X)
  
  p <- ncol(X)
  
  
  # Priors
  
  priors <- table(y) /
    n
  
  
  # Class means
  
  class_means <- matrix(
    NA,
    nrow = K,
    ncol = p
  )
  
  rownames(class_means) <-
    classes
  
  colnames(class_means) <-
    colnames(X)
  
  
  for (k in 1:K) {
    
    class_means[k, ] <-
      colMeans(
        X[
          y == classes[k],
          ,
          drop = FALSE
        ]
      )
  }
  
  
  # Pooled covariance
  
  pooled_covariance <- matrix(
    0,
    nrow = p,
    ncol = p
  )
  
  
  for (k in 1:K) {
    
    X_k <- X[
      y == classes[k],
      ,
      drop = FALSE
    ]
    
    n_k <- nrow(X_k)
    
    
    pooled_covariance <-
      pooled_covariance +
      (n_k - 1) *
      cov(X_k)
  }
  
  
  pooled_covariance <-
    pooled_covariance /
    (n - K)
  
  
  return(
    list(
      classes = classes,
      priors = priors,
      means = class_means,
      covariance = pooled_covariance
    )
  )
}


# ------------------------------------------------------------------------------
# 24. Fit LDA on Training Data
# ------------------------------------------------------------------------------

lda_model <- lda_fit(
  X_train,
  y_train
)

lda_model$priors

lda_model$means

lda_model$covariance


# ------------------------------------------------------------------------------
# 25. Predict Test Data
# ------------------------------------------------------------------------------

test_predictions <- lda_predict(
  X_test,
  lda_model$means,
  lda_model$covariance,
  lda_model$priors
)


# ------------------------------------------------------------------------------
# 26. Test Confusion Matrix
# ------------------------------------------------------------------------------

test_confusion <- table(
  Actual = y_test,
  Predicted = test_predictions$class
)

test_confusion


# ------------------------------------------------------------------------------
# 27. Test Accuracy
# ------------------------------------------------------------------------------

test_accuracy <- mean(
  y_test ==
    test_predictions$class
)

test_accuracy


# ------------------------------------------------------------------------------
# 28. Test Error Rate
# ------------------------------------------------------------------------------

test_error <- 1 -
  test_accuracy

test_error


# ------------------------------------------------------------------------------
# 29. Class-Specific Accuracy
# ------------------------------------------------------------------------------

class_accuracy <- numeric(
  K
)

names(class_accuracy) <-
  classes


for (k in 1:K) {
  
  class_index <- which(
    y_test ==
      classes[k]
  )
  
  
  class_accuracy[k] <- mean(
    test_predictions$class[
      class_index
    ] ==
      y_test[
        class_index
      ]
  )
}


class_accuracy


# ------------------------------------------------------------------------------
# 30. Decision Boundary Grid
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


grid_predictions <- lda_predict(
  as.matrix(grid),
  class_means,
  pooled_covariance,
  priors
)


grid_class <- matrix(
  as.numeric(
    grid_predictions$class
  ),
  nrow = length(x1_grid),
  ncol = length(x2_grid)
)


# ------------------------------------------------------------------------------
# 31. Plot Decision Regions
# ------------------------------------------------------------------------------

image(
  x1_grid,
  x2_grid,
  grid_class,
  xlab = "x1",
  ylab = "x2",
  main = "LDA Decision Regions"
)


points(
  X,
  pch = as.numeric(y)
)


legend(
  "topright",
  legend = levels(y),
  pch = 1:K
)


# ------------------------------------------------------------------------------
# 32. Pairwise Linear Decision Boundary
# ------------------------------------------------------------------------------

# The boundary between classes k and l occurs where
#
# delta_k(x) = delta_l(x).
#
# Because the covariance matrix is shared, this boundary is linear.


lda_boundary <- function(
    mean_1,
    mean_2,
    Sigma_inv,
    prior_1,
    prior_2
) {
  
  w <- Sigma_inv %*%
    (
      mean_1 -
        mean_2
    )
  
  
  b <-
    -0.5 *
    (
      t(mean_1) %*%
        Sigma_inv %*%
        mean_1 -
        t(mean_2) %*%
        Sigma_inv %*%
        mean_2
    ) +
    log(
      prior_1 /
        prior_2
    )
  
  
  return(
    list(
      w = as.vector(w),
      b = as.numeric(b)
    )
  )
}


# ------------------------------------------------------------------------------
# 33. Boundary Between Class 1 and Class 2
# ------------------------------------------------------------------------------

boundary_12 <- lda_boundary(
  class_means[1, ],
  class_means[2, ],
  Sigma_inv,
  priors[1],
  priors[2]
)

boundary_12


# ------------------------------------------------------------------------------
# 34. Plot Pairwise Decision Boundary
# ------------------------------------------------------------------------------

plot(
  X,
  pch = as.numeric(y),
  xlab = "x1",
  ylab = "x2",
  main = "LDA Linear Decision Boundary"
)


w <- boundary_12$w
b <- boundary_12$b


# w1*x1 + w2*x2 + b = 0
#
# x2 = -(w1*x1 + b) / w2

abline(
  a = -b / w[2],
  b = -w[1] / w[2],
  lty = 2
)


# ------------------------------------------------------------------------------
# 35. Within-Class Scatter Matrix
# ------------------------------------------------------------------------------

overall_mean <- colMeans(
  X
)


S_W <- matrix(
  0,
  nrow = p,
  ncol = p
)


for (k in 1:K) {
  
  X_k <- X[
    y == classes[k],
    ,
    drop = FALSE
  ]
  
  
  centered_k <- sweep(
    X_k,
    2,
    class_means[k, ],
    "-"
  )
  
  
  S_W <- S_W +
    crossprod(
      centered_k
    )
}


S_W


# ------------------------------------------------------------------------------
# 36. Between-Class Scatter Matrix
# ------------------------------------------------------------------------------

S_B <- matrix(
  0,
  nrow = p,
  ncol = p
)


for (k in 1:K) {
  
  n_k <- sum(
    y ==
      classes[k]
  )
  
  
  mean_difference <-
    class_means[k, ] -
    overall_mean
  
  
  S_B <- S_B +
    n_k *
    outer(
      mean_difference,
      mean_difference
    )
}


S_B


# ------------------------------------------------------------------------------
# 37. Fisher LDA Directions
# ------------------------------------------------------------------------------

# Fisher's formulation finds directions that maximize
#
# between-class variation / within-class variation.

eigen_result <- eigen(
  solve(S_W) %*%
    S_B
)


lda_directions <-
  Re(
    eigen_result$vectors
  )


lda_eigenvalues <-
  Re(
    eigen_result$values
  )


lda_eigenvalues

lda_directions


# ------------------------------------------------------------------------------
# 38. Number of Discriminant Directions
# ------------------------------------------------------------------------------

# At most min(p, K - 1) discriminant directions exist.

num_directions <- min(
  p,
  K - 1
)

num_directions


# ------------------------------------------------------------------------------
# 39. Project Observations onto LDA Directions
# ------------------------------------------------------------------------------

W <- lda_directions[
  ,
  1:num_directions,
  drop = FALSE
]


X_lda <- X %*%
  W


colnames(X_lda) <- paste0(
  "LD",
  1:num_directions
)


head(
  X_lda
)


# ------------------------------------------------------------------------------
# 40. Plot First Two Discriminant Coordinates
# ------------------------------------------------------------------------------

if (num_directions >= 2) {
  
  plot(
    X_lda[, 1],
    X_lda[, 2],
    pch = as.numeric(y),
    xlab = "LD1",
    ylab = "LD2",
    main = "Fisher LDA Projection"
  )
  
  
  legend(
    "topright",
    legend = levels(y),
    pch = 1:K
  )
}


# ------------------------------------------------------------------------------
# 41. Explained Discriminatory Information
# ------------------------------------------------------------------------------

positive_eigenvalues <- lda_eigenvalues[
  1:num_directions
]


discriminant_proportion <-
  positive_eigenvalues /
  sum(
    positive_eigenvalues
  )


data.frame(
  Direction = paste0(
    "LD",
    1:num_directions
  ),
  Eigenvalue = positive_eigenvalues,
  Proportion = discriminant_proportion
)


# ------------------------------------------------------------------------------
# 42. Verify with MASS::lda if MASS is Available
# ------------------------------------------------------------------------------

# MASS ships with most R installations.
# This section is only for verification of the manual implementation.

if (
  requireNamespace(
    "MASS",
    quietly = TRUE
  )
) {
  
  lda_builtin <- MASS::lda(
    Class ~ x1 + x2,
    data = data
  )
  
  
  lda_builtin
  
  
  builtin_predictions <- predict(
    lda_builtin,
    newdata = data
  )
  
  
  table(
    Manual = train_predictions$class,
    MASS = builtin_predictions$class
  )
  
  
  mean(
    train_predictions$class ==
      builtin_predictions$class
  )
}