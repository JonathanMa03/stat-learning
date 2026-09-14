# ==============================================================================
# Flexible Discriminant Analysis
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main idea:
#   Flexible Discriminant Analysis extends LDA by replacing the original
#   predictors with flexible nonlinear transformations.
#
#   Original predictors:
#
#       x1, x2
#
#   Expanded predictors:
#
#       x1, x1^2, ..., spline terms,
#       x2, x2^2, ..., spline terms
#
#   LDA is then performed in the expanded feature space.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Nonlinear Classification Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 450


x1 <- runif(
  n,
  -4,
  4
)

x2 <- runif(
  n,
  -4,
  4
)


# Nonlinear class structure

radius <- sqrt(
  x1^2 +
    x2^2
)


y <- ifelse(
  radius < 1.5,
  "Class1",
  ifelse(
    radius < 2.8,
    "Class2",
    "Class3"
  )
)


# Add some label noise

noise_index <- sample(
  1:n,
  size = round(
    0.05 * n
  )
)

y[
  noise_index
] <- sample(
  c(
    "Class1",
    "Class2",
    "Class3"
  ),
  length(noise_index),
  replace = TRUE
)


y <- factor(y)


X <- cbind(
  x1,
  x2
)

colnames(X) <- c(
  "x1",
  "x2"
)


data <- data.frame(
  x1 = x1,
  x2 = x2,
  Class = y
)

head(data)

table(y)


# ------------------------------------------------------------------------------
# 2. Plot the Data
# ------------------------------------------------------------------------------

plot(
  x1,
  x2,
  pch = as.numeric(y),
  xlab = "x1",
  ylab = "x2",
  main = "Nonlinear Classification Data"
)

legend(
  "topright",
  legend = levels(y),
  pch = 1:length(levels(y))
)


# ------------------------------------------------------------------------------
# 3. Ordinary LDA Fitting Function
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
  
  
  # Class priors
  
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
  
  
  # Pooled covariance matrix
  
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
# 4. LDA Prediction Function
# ------------------------------------------------------------------------------

lda_predict <- function(
    X_new,
    model
) {
  
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
  
  colnames(scores) <-
    classes
  
  
  for (k in 1:K) {
    
    mu_k <- model$means[
      k,
    ]
    
    
    scores[, k] <-
      as.vector(
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
    scores
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
# 5. Fit Ordinary LDA
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
# 6. Ordinary LDA Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y,
  Predicted = lda_predictions$class
)


# ------------------------------------------------------------------------------
# 7. Truncated Power Basis
# ------------------------------------------------------------------------------

# Flexible discriminant analysis needs flexible transformations.
#
# We construct a cubic spline-like basis manually.
#
# For one variable:
#
#   x
#   x^2
#   x^3
#   (x - knot_1)^3_+
#   (x - knot_2)^3_+
#   ...
#
# where
#
#   (z)_+ = max(z, 0)


positive_part <- function(x) {
  
  pmax(
    x,
    0
  )
}


# ------------------------------------------------------------------------------
# 8. One-Dimensional Spline Basis
# ------------------------------------------------------------------------------

spline_basis <- function(
    x,
    knots,
    degree = 3
) {
  
  basis <- matrix(
    NA,
    nrow = length(x),
    ncol = degree +
      length(knots)
  )
  
  
  # Polynomial terms
  
  for (d in 1:degree) {
    
    basis[, d] <-
      x^d
  }
  
  
  # Truncated power terms
  
  for (j in seq_along(knots)) {
    
    basis[
      ,
      degree + j
    ] <-
      positive_part(
        x -
          knots[j]
      )^degree
  }
  
  
  return(basis)
}


# ------------------------------------------------------------------------------
# 9. Select Knots
# ------------------------------------------------------------------------------

# Use predictor quantiles as knots.

knots_x1 <- as.numeric(
  quantile(
    x1,
    probs = c(
      0.25,
      0.50,
      0.75
    )
  )
)


knots_x2 <- as.numeric(
  quantile(
    x2,
    probs = c(
      0.25,
      0.50,
      0.75
    )
  )
)


knots_x1

knots_x2


# ------------------------------------------------------------------------------
# 10. Construct Flexible Feature Matrix
# ------------------------------------------------------------------------------

basis_x1 <- spline_basis(
  x1,
  knots = knots_x1
)

basis_x2 <- spline_basis(
  x2,
  knots = knots_x2
)


Z <- cbind(
  basis_x1,
  basis_x2
)


colnames(Z) <- c(
  "x1",
  "x1_2",
  "x1_3",
  "x1_knot1",
  "x1_knot2",
  "x1_knot3",
  "x2",
  "x2_2",
  "x2_3",
  "x2_knot1",
  "x2_knot2",
  "x2_knot3"
)


head(Z)


# ------------------------------------------------------------------------------
# 11. Standardize Expanded Features
# ------------------------------------------------------------------------------

Z_means <- colMeans(
  Z
)

Z_sds <- apply(
  Z,
  2,
  sd
)


Z_scaled <- scale(
  Z,
  center = Z_means,
  scale = Z_sds
)


# ------------------------------------------------------------------------------
# 12. Fit Flexible Discriminant Analysis
# ------------------------------------------------------------------------------

fda_model <- lda_fit(
  Z_scaled,
  y
)


# ------------------------------------------------------------------------------
# 13. Predict Training Data
# ------------------------------------------------------------------------------

fda_predictions <- lda_predict(
  Z_scaled,
  fda_model
)


# ------------------------------------------------------------------------------
# 14. Training Accuracy
# ------------------------------------------------------------------------------

fda_accuracy <- mean(
  fda_predictions$class ==
    y
)


fda_accuracy


# ------------------------------------------------------------------------------
# 15. Flexible Discriminant Analysis Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y,
  Predicted = fda_predictions$class
)


# ------------------------------------------------------------------------------
# 16. Compare Ordinary LDA and FDA
# ------------------------------------------------------------------------------

data.frame(
  Method = c(
    "Ordinary LDA",
    "Flexible Discriminant Analysis"
  ),
  Accuracy = c(
    lda_accuracy,
    fda_accuracy
  ),
  Error_Rate = c(
    1 - lda_accuracy,
    1 - fda_accuracy
  )
)


# ------------------------------------------------------------------------------
# 17. Flexible Transformation Function
# ------------------------------------------------------------------------------

# We need the exact same transformations for new observations.

fda_transform <- function(
    X_new,
    knots_x1,
    knots_x2,
    means = NULL,
    sds = NULL
) {
  
  basis_1 <- spline_basis(
    X_new[, 1],
    knots_x1
  )
  
  basis_2 <- spline_basis(
    X_new[, 2],
    knots_x2
  )
  
  
  Z_new <- cbind(
    basis_1,
    basis_2
  )
  
  
  if (
    !is.null(means) &&
    !is.null(sds)
  ) {
    
    Z_new <- scale(
      Z_new,
      center = means,
      scale = sds
    )
  }
  
  
  return(Z_new)
}


# ------------------------------------------------------------------------------
# 18. Decision Boundary Grid
# ------------------------------------------------------------------------------

x1_grid <- seq(
  min(x1) - 0.5,
  max(x1) + 0.5,
  length.out = 250
)

x2_grid <- seq(
  min(x2) - 0.5,
  max(x2) + 0.5,
  length.out = 250
)


grid <- expand.grid(
  x1 = x1_grid,
  x2 = x2_grid
)


# ------------------------------------------------------------------------------
# 19. Transform Grid
# ------------------------------------------------------------------------------

grid_Z <- fda_transform(
  as.matrix(grid),
  knots_x1 = knots_x1,
  knots_x2 = knots_x2,
  means = Z_means,
  sds = Z_sds
)


# ------------------------------------------------------------------------------
# 20. Predict FDA Classes on Grid
# ------------------------------------------------------------------------------

grid_predictions <- lda_predict(
  grid_Z,
  fda_model
)


grid_class <- matrix(
  as.numeric(
    grid_predictions$class
  ),
  nrow = length(x1_grid),
  ncol = length(x2_grid)
)


# ------------------------------------------------------------------------------
# 21. Plot FDA Decision Regions
# ------------------------------------------------------------------------------

image(
  x1_grid,
  x2_grid,
  grid_class,
  xlab = "x1",
  ylab = "x2",
  main = "Flexible Discriminant Analysis"
)


points(
  x1,
  x2,
  pch = as.numeric(y)
)


legend(
  "topright",
  legend = levels(y),
  pch = 1:length(levels(y))
)


# ------------------------------------------------------------------------------
# 22. Compare Ordinary LDA Decision Regions
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
  main = "Ordinary LDA Decision Regions"
)


points(
  x1,
  x2,
  pch = as.numeric(y)
)


# ------------------------------------------------------------------------------
# 23. Train-Test Split
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
# 24. Fit Ordinary LDA on Training Data
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
# 25. Determine Knots from Training Data Only
# ------------------------------------------------------------------------------

knots_train_x1 <- as.numeric(
  quantile(
    X_train[, 1],
    probs = c(
      0.25,
      0.50,
      0.75
    )
  )
)


knots_train_x2 <- as.numeric(
  quantile(
    X_train[, 2],
    probs = c(
      0.25,
      0.50,
      0.75
    )
  )
)


# ------------------------------------------------------------------------------
# 26. Create Training Basis
# ------------------------------------------------------------------------------

Z_train_raw <- fda_transform(
  X_train,
  knots_train_x1,
  knots_train_x2
)


# ------------------------------------------------------------------------------
# 27. Standardize Using Training Data
# ------------------------------------------------------------------------------

train_Z_means <- colMeans(
  Z_train_raw
)

train_Z_sds <- apply(
  Z_train_raw,
  2,
  sd
)


Z_train <- scale(
  Z_train_raw,
  center = train_Z_means,
  scale = train_Z_sds
)


# ------------------------------------------------------------------------------
# 28. Transform Test Data
# ------------------------------------------------------------------------------

Z_test <- fda_transform(
  X_test,
  knots_train_x1,
  knots_train_x2,
  means = train_Z_means,
  sds = train_Z_sds
)


# ------------------------------------------------------------------------------
# 29. Fit FDA on Training Data
# ------------------------------------------------------------------------------

fda_train_model <- lda_fit(
  Z_train,
  y_train
)


# ------------------------------------------------------------------------------
# 30. Predict Test Data
# ------------------------------------------------------------------------------

fda_test_prediction <- lda_predict(
  Z_test,
  fda_train_model
)


# ------------------------------------------------------------------------------
# 31. Test Accuracy
# ------------------------------------------------------------------------------

fda_test_accuracy <- mean(
  fda_test_prediction$class ==
    y_test
)


fda_test_accuracy


# ------------------------------------------------------------------------------
# 32. Test Confusion Matrix
# ------------------------------------------------------------------------------

table(
  Actual = y_test,
  Predicted = fda_test_prediction$class
)


# ------------------------------------------------------------------------------
# 33. Compare Test Performance
# ------------------------------------------------------------------------------

performance <- data.frame(
  Method = c(
    "Ordinary LDA",
    "Flexible Discriminant Analysis"
  ),
  Test_Accuracy = c(
    lda_test_accuracy,
    fda_test_accuracy
  ),
  Test_Error = c(
    1 - lda_test_accuracy,
    1 - fda_test_accuracy
  )
)


performance


# ------------------------------------------------------------------------------
# 34. K-Fold Cross-Validation
# ------------------------------------------------------------------------------

fda_cv <- function(
    X,
    y,
    k = 10
) {
  
  n <- nrow(X)
  
  
  folds <- sample(
    rep(
      1:k,
      length.out = n
    )
  )
  
  
  lda_accuracy <- numeric(k)
  
  fda_accuracy <- numeric(k)
  
  
  for (fold in 1:k) {
    
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
    
    
    y_train <- y[
      train_index
    ]
    
    y_valid <- y[
      valid_index
    ]
    
    
    # --------------------------------------------------------------------------
    # Ordinary LDA
    # --------------------------------------------------------------------------
    
    lda_model <- lda_fit(
      X_train,
      y_train
    )
    
    
    lda_prediction <- lda_predict(
      X_valid,
      lda_model
    )
    
    
    lda_accuracy[fold] <- mean(
      lda_prediction$class ==
        y_valid
    )
    
    
    # --------------------------------------------------------------------------
    # FDA Knots
    # --------------------------------------------------------------------------
    
    knots_x1 <- as.numeric(
      quantile(
        X_train[, 1],
        probs = c(
          0.25,
          0.50,
          0.75
        )
      )
    )
    
    
    knots_x2 <- as.numeric(
      quantile(
        X_train[, 2],
        probs = c(
          0.25,
          0.50,
          0.75
        )
      )
    )
    
    
    # --------------------------------------------------------------------------
    # Expanded Training Features
    # --------------------------------------------------------------------------
    
    Z_train_raw <- fda_transform(
      X_train,
      knots_x1,
      knots_x2
    )
    
    
    means <- colMeans(
      Z_train_raw
    )
    
    
    sds <- apply(
      Z_train_raw,
      2,
      sd
    )
    
    
    Z_train <- scale(
      Z_train_raw,
      center = means,
      scale = sds
    )
    
    
    # --------------------------------------------------------------------------
    # Expanded Validation Features
    # --------------------------------------------------------------------------
    
    Z_valid <- fda_transform(
      X_valid,
      knots_x1,
      knots_x2,
      means,
      sds
    )
    
    
    # --------------------------------------------------------------------------
    # FDA Model
    # --------------------------------------------------------------------------
    
    fda_model <- lda_fit(
      Z_train,
      y_train
    )
    
    
    fda_prediction <- lda_predict(
      Z_valid,
      fda_model
    )
    
    
    fda_accuracy[fold] <- mean(
      fda_prediction$class ==
        y_valid
    )
  }
  
  
  return(
    data.frame(
      Fold = 1:k,
      LDA = lda_accuracy,
      FDA = fda_accuracy
    )
  )
}


# ------------------------------------------------------------------------------
# 35. Run Cross-Validation
# ------------------------------------------------------------------------------

set.seed(123)


cv_results <- fda_cv(
  X,
  y,
  k = 10
)


cv_results


# ------------------------------------------------------------------------------
# 36. Mean Cross-Validated Accuracy
# ------------------------------------------------------------------------------

colMeans(
  cv_results[
    ,
    c(
      "LDA",
      "FDA"
    )
  ]
)


# ------------------------------------------------------------------------------
# 37. Compare CV Accuracy
# ------------------------------------------------------------------------------

boxplot(
  cv_results$LDA,
  cv_results$FDA,
  names = c(
    "LDA",
    "FDA"
  ),
  ylab = "Validation Accuracy",
  main = "LDA vs Flexible Discriminant Analysis"
)


# ------------------------------------------------------------------------------
# 38. Fisher Directions in Expanded Feature Space
# ------------------------------------------------------------------------------

# FDA can also be interpreted as performing discriminant analysis
# after mapping observations into a nonlinear feature space.


classes <- levels(y)

K <- length(classes)

q <- ncol(Z_scaled)


overall_mean <- colMeans(
  Z_scaled
)


S_W <- matrix(
  0,
  nrow = q,
  ncol = q
)


S_B <- matrix(
  0,
  nrow = q,
  ncol = q
)


for (k in 1:K) {
  
  Z_k <- Z_scaled[
    y == classes[k],
    ,
    drop = FALSE
  ]
  
  
  mu_k <- colMeans(
    Z_k
  )
  
  
  centered <- sweep(
    Z_k,
    2,
    mu_k,
    "-"
  )
  
  
  S_W <- S_W +
    crossprod(
      centered
    )
  
  
  mean_difference <-
    mu_k -
    overall_mean
  
  
  S_B <- S_B +
    nrow(Z_k) *
    outer(
      mean_difference,
      mean_difference
    )
}


# ------------------------------------------------------------------------------
# 39. Regularized Fisher Directions
# ------------------------------------------------------------------------------

# Expanded features can create near-singularity.
# Add a very small ridge term for numerical stability.

ridge <- 1e-6


fisher_matrix <- solve(
  S_W +
    ridge *
    diag(q)
) %*%
  S_B


eigen_result <- eigen(
  fisher_matrix
)


directions <- Re(
  eigen_result$vectors
)


eigenvalues <- Re(
  eigen_result$values
)


eigenvalues


# ------------------------------------------------------------------------------
# 40. FDA Discriminant Coordinates
# ------------------------------------------------------------------------------

num_directions <- min(
  K - 1,
  q
)


FDA_coordinates <-
  Z_scaled %*%
  directions[
    ,
    1:num_directions,
    drop = FALSE
  ]


colnames(FDA_coordinates) <-
  paste0(
    "FD",
    1:num_directions
  )


head(
  FDA_coordinates
)


# ------------------------------------------------------------------------------
# 41. Plot First Two Flexible Discriminant Coordinates
# ------------------------------------------------------------------------------

if (
  num_directions >= 2
) {
  
  plot(
    FDA_coordinates[, 1],
    FDA_coordinates[, 2],
    pch = as.numeric(y),
    xlab = "FD1",
    ylab = "FD2",
    main = "Flexible Discriminant Coordinates"
  )
  
  
  legend(
    "topright",
    legend = levels(y),
    pch = 1:K
  )
}