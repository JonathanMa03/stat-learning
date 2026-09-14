# ==============================================================================
# Multidimensional Splines
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Spline basis expansion in multiple dimensions
#   - Tensor-product spline basis
#   - Smooth nonlinear interaction surfaces
#   - Additive vs interaction models
#   - Regularization with difference penalties
#   - Cross-validation for smoothing strength
#
# Model:
#
#   f(x1, x2)
#   =
#   sum_j sum_k theta_jk B_j(x1) C_k(x2)
#
# where B_j and C_k are one-dimensional spline basis functions.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Two-Dimensional Nonlinear Regression Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 500

x1 <- runif(
  n,
  min = -3,
  max = 3
)

x2 <- runif(
  n,
  min = -3,
  max = 3
)


true_function <- function(
    x1,
    x2
) {
  
  2 +
    sin(x1) +
    cos(x2) +
    0.5 *
    sin(
      x1 * x2
    )
}


y_true <- true_function(
  x1,
  x2
)


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = 0.5
  )


data <- data.frame(
  x1 = x1,
  x2 = x2,
  y = y
)


head(data)


# ------------------------------------------------------------------------------
# 2. Basic Data Visualization
# ------------------------------------------------------------------------------

plot(
  x1,
  y,
  pch = 19,
  xlab = "x1",
  ylab = "y",
  main = "Response vs x1"
)


plot(
  x2,
  y,
  pch = 19,
  xlab = "x2",
  ylab = "y",
  main = "Response vs x2"
)


# ------------------------------------------------------------------------------
# 3. Build One-Dimensional B-Spline Bases
# ------------------------------------------------------------------------------

degree <- 3

number_knots <- 4


knots_x1 <- as.numeric(
  quantile(
    x1,
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


knots_x2 <- as.numeric(
  quantile(
    x2,
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


boundary_x1 <- range(
  x1
)

boundary_x2 <- range(
  x2
)


B1 <- splines::bs(
  x1,
  knots = knots_x1,
  degree = degree,
  Boundary.knots = boundary_x1,
  intercept = TRUE
)


B2 <- splines::bs(
  x2,
  knots = knots_x2,
  degree = degree,
  Boundary.knots = boundary_x2,
  intercept = TRUE
)


B1 <- as.matrix(
  B1
)

B2 <- as.matrix(
  B2
)


dim(
  B1
)

dim(
  B2
)


# ------------------------------------------------------------------------------
# 4. Plot One-Dimensional Basis Functions
# ------------------------------------------------------------------------------

order_x1 <- order(
  x1
)


matplot(
  x1[
    order_x1
  ],
  B1[
    order_x1,
    ,
    drop = FALSE
  ],
  type = "l",
  lty = 1,
  xlab = "x1",
  ylab = "Basis Value",
  main = "B-Spline Basis for x1"
)


order_x2 <- order(
  x2
)


matplot(
  x2[
    order_x2
  ],
  B2[
    order_x2,
    ,
    drop = FALSE
  ],
  type = "l",
  lty = 1,
  xlab = "x2",
  ylab = "Basis Value",
  main = "B-Spline Basis for x2"
)


# ------------------------------------------------------------------------------
# 5. Tensor-Product Basis
# ------------------------------------------------------------------------------

# If:
#
# B1 has K1 basis functions
# B2 has K2 basis functions
#
# then the tensor-product basis contains:
#
# K1 * K2
#
# basis functions.
#
# Each basis function is:
#
# B_j(x1) * C_k(x2)


tensor_product_basis <- function(
    B1,
    B2
) {
  
  B1 <- as.matrix(
    B1
  )
  
  B2 <- as.matrix(
    B2
  )
  
  
  n <- nrow(
    B1
  )
  
  
  K1 <- ncol(
    B1
  )
  
  K2 <- ncol(
    B2
  )
  
  
  X_tensor <- matrix(
    0,
    nrow = n,
    ncol = K1 * K2
  )
  
  
  column_index <- 1
  
  
  for (j in seq_len(
    K1
  )) {
    
    for (k in seq_len(
      K2
    )) {
      
      X_tensor[
        ,
        column_index
      ] <-
        B1[
          ,
          j
        ] *
        B2[
          ,
          k
        ]
      
      
      column_index <- column_index + 1
    }
  }
  
  
  colnames(
    X_tensor
  ) <- paste0(
    "B",
    rep(
      seq_len(
        K1
      ),
      each = K2
    ),
    "_C",
    rep(
      seq_len(
        K2
      ),
      times = K1
    )
  )
  
  
  return(
    X_tensor
  )
}


# ------------------------------------------------------------------------------
# 6. Construct Tensor-Product Design Matrix
# ------------------------------------------------------------------------------

X_tensor <- tensor_product_basis(
  B1,
  B2
)


dim(
  X_tensor
)


head(
  X_tensor[
    ,
    1:6
  ]
)


# ------------------------------------------------------------------------------
# 7. Fit Unpenalized Tensor-Product Spline
# ------------------------------------------------------------------------------

beta_tensor <- solve(
  crossprod(
    X_tensor
  ),
  crossprod(
    X_tensor,
    y
  )
)


# ------------------------------------------------------------------------------
# 8. Training Predictions
# ------------------------------------------------------------------------------

y_hat_tensor <- as.vector(
  X_tensor %*%
    beta_tensor
)


# ------------------------------------------------------------------------------
# 9. Training Error
# ------------------------------------------------------------------------------

tensor_MSE <- mean(
  (
    y -
      y_hat_tensor
  )^2
)


tensor_RMSE <- sqrt(
  tensor_MSE
)


data.frame(
  MSE = tensor_MSE,
  RMSE = tensor_RMSE
)


# ------------------------------------------------------------------------------
# 10. Compare with Linear Regression
# ------------------------------------------------------------------------------

linear_model <- lm(
  y ~ x1 + x2
)


linear_prediction <- predict(
  linear_model
)


linear_MSE <- mean(
  (
    y -
      linear_prediction
  )^2
)


data.frame(
  Model = c(
    "Linear Regression",
    "Tensor-Product Spline"
  ),
  MSE = c(
    linear_MSE,
    tensor_MSE
  )
)


# ------------------------------------------------------------------------------
# 11. Additive Spline Model
# ------------------------------------------------------------------------------

# Before using the full tensor-product interaction, compare with:
#
# f(x1, x2) = f1(x1) + f2(x2)
#
# This can represent nonlinear effects but not nonlinear interaction.


X_additive <- cbind(
  Intercept = 1,
  B1[
    ,
    -1,
    drop = FALSE
  ],
  B2[
    ,
    -1,
    drop = FALSE
  ]
)


beta_additive <- solve(
  crossprod(
    X_additive
  ),
  crossprod(
    X_additive,
    y
  )
)


y_hat_additive <- as.vector(
  X_additive %*%
    beta_additive
)


additive_MSE <- mean(
  (
    y -
      y_hat_additive
  )^2
)


data.frame(
  Model = c(
    "Linear",
    "Additive Splines",
    "Tensor-Product Splines"
  ),
  Training_MSE = c(
    linear_MSE,
    additive_MSE,
    tensor_MSE
  )
)


# ------------------------------------------------------------------------------
# 12. Why Tensor Products Matter
# ------------------------------------------------------------------------------

# Additive model:
#
# f(x1, x2) = f1(x1) + f2(x2)
#
#
# Tensor-product model:
#
# f(x1, x2) =
# sum_j sum_k theta_jk B_j(x1) C_k(x2)
#
#
# This allows the effect of x1 to depend on x2.


# ------------------------------------------------------------------------------
# 13. Generic Tensor-Spline Fit Function
# ------------------------------------------------------------------------------

tensor_spline_fit <- function(
    x1,
    x2,
    y,
    knots_x1,
    knots_x2,
    degree = 3,
    boundary_x1 = range(x1),
    boundary_x2 = range(x2),
    ridge = 1e-8
) {
  
  B1 <- splines::bs(
    x1,
    knots = knots_x1,
    degree = degree,
    Boundary.knots = boundary_x1,
    intercept = TRUE
  )
  
  
  B2 <- splines::bs(
    x2,
    knots = knots_x2,
    degree = degree,
    Boundary.knots = boundary_x2,
    intercept = TRUE
  )
  
  
  B1 <- as.matrix(
    B1
  )
  
  B2 <- as.matrix(
    B2
  )
  
  
  X <- tensor_product_basis(
    B1,
    B2
  )
  
  
  p <- ncol(
    X
  )
  
  
  beta <- solve(
    crossprod(
      X
    ) +
      ridge *
      diag(
        p
      ),
    crossprod(
      X,
      y
    )
  )
  
  
  return(
    list(
      coefficients = as.vector(
        beta
      ),
      knots_x1 = knots_x1,
      knots_x2 = knots_x2,
      degree = degree,
      boundary_x1 = boundary_x1,
      boundary_x2 = boundary_x2,
      K1 = ncol(
        B1
      ),
      K2 = ncol(
        B2
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 14. Generic Tensor-Spline Prediction
# ------------------------------------------------------------------------------

tensor_spline_predict <- function(
    x1_new,
    x2_new,
    model
) {
  
  B1_new <- splines::bs(
    x1_new,
    knots = model$knots_x1,
    degree = model$degree,
    Boundary.knots = model$boundary_x1,
    intercept = TRUE
  )
  
  
  B2_new <- splines::bs(
    x2_new,
    knots = model$knots_x2,
    degree = model$degree,
    Boundary.knots = model$boundary_x2,
    intercept = TRUE
  )
  
  
  X_new <- tensor_product_basis(
    B1_new,
    B2_new
  )
  
  
  prediction <- as.vector(
    X_new %*%
      model$coefficients
  )
  
  
  return(
    prediction
  )
}


# ------------------------------------------------------------------------------
# 15. Fit Generic Tensor Model
# ------------------------------------------------------------------------------

tensor_model <- tensor_spline_fit(
  x1 = x1,
  x2 = x2,
  y = y,
  knots_x1 = knots_x1,
  knots_x2 = knots_x2
)


generic_prediction <- tensor_spline_predict(
  x1,
  x2,
  tensor_model
)


max(
  abs(
    generic_prediction -
      y_hat_tensor
  )
)


# ------------------------------------------------------------------------------
# 16. Create Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 60


x1_grid <- seq(
  min(
    x1
  ),
  max(
    x1
  ),
  length.out = grid_size
)


x2_grid <- seq(
  min(
    x2
  ),
  max(
    x2
  ),
  length.out = grid_size
)


grid <- expand.grid(
  x1 = x1_grid,
  x2 = x2_grid
)


# ------------------------------------------------------------------------------
# 17. Predict Tensor Surface
# ------------------------------------------------------------------------------

grid_prediction <- tensor_spline_predict(
  grid$x1,
  grid$x2,
  tensor_model
)


prediction_surface <- matrix(
  grid_prediction,
  nrow = length(
    x1_grid
  ),
  ncol = length(
    x2_grid
  )
)


# ------------------------------------------------------------------------------
# 18. True Surface
# ------------------------------------------------------------------------------

true_surface <- outer(
  x1_grid,
  x2_grid,
  FUN = true_function
)


# ------------------------------------------------------------------------------
# 19. Perspective Plot
# ------------------------------------------------------------------------------

persp(
  x1_grid,
  x2_grid,
  prediction_surface,
  theta = 35,
  phi = 25,
  expand = 0.7,
  xlab = "x1",
  ylab = "x2",
  zlab = "Predicted y",
  main = "Tensor-Product Spline Surface"
)


# ------------------------------------------------------------------------------
# 20. True Surface
# ------------------------------------------------------------------------------

persp(
  x1_grid,
  x2_grid,
  true_surface,
  theta = 35,
  phi = 25,
  expand = 0.7,
  xlab = "x1",
  ylab = "x2",
  zlab = "True y",
  main = "True Regression Surface"
)


# ------------------------------------------------------------------------------
# 21. Contour Plot
# ------------------------------------------------------------------------------

contour(
  x1_grid,
  x2_grid,
  prediction_surface,
  xlab = "x1",
  ylab = "x2",
  main = "Tensor-Product Spline Contours"
)


points(
  x1,
  x2,
  pch = 19,
  cex = 0.4
)


# ------------------------------------------------------------------------------
# 22. Difference Between True and Estimated Surface
# ------------------------------------------------------------------------------

surface_error <- prediction_surface -
  true_surface


contour(
  x1_grid,
  x2_grid,
  surface_error,
  xlab = "x1",
  ylab = "x2",
  main = "Surface Estimation Error"
)


# ------------------------------------------------------------------------------
# 23. Coefficient Matrix
# ------------------------------------------------------------------------------

# Tensor coefficients can be arranged as a K1 x K2 matrix.


Theta <- matrix(
  tensor_model$coefficients,
  nrow = tensor_model$K1,
  ncol = tensor_model$K2,
  byrow = TRUE
)


Theta


# ------------------------------------------------------------------------------
# 24. Visualize Tensor Coefficients
# ------------------------------------------------------------------------------

image(
  seq_len(
    nrow(
      Theta
    )
  ),
  seq_len(
    ncol(
      Theta
    )
  ),
  Theta,
  xlab = "x1 Basis Index",
  ylab = "x2 Basis Index",
  main = "Tensor-Spline Coefficients"
)


# ------------------------------------------------------------------------------
# 25. Penalized Tensor-Product Splines
# ------------------------------------------------------------------------------

# With many tensor basis functions, overfitting becomes a major concern.
#
# We can penalize roughness along both dimensions.


K1 <- ncol(
  B1
)

K2 <- ncol(
  B2
)


# Second-difference penalty for x1 basis coefficients

D1 <- diff(
  diag(
    K1
  ),
  differences = 2
)


P1 <- crossprod(
  D1
)


# Second-difference penalty for x2 basis coefficients

D2 <- diff(
  diag(
    K2
  ),
  differences = 2
)


P2 <- crossprod(
  D2
)


# ------------------------------------------------------------------------------
# 26. Kronecker-Product Penalty
# ------------------------------------------------------------------------------

# Roughness along x1:
#
# P1 kron I
#
# Roughness along x2:
#
# I kron P2


Penalty_x1 <- kronecker(
  P1,
  diag(
    K2
  )
)


Penalty_x2 <- kronecker(
  diag(
    K1
  ),
  P2
)


Penalty <- Penalty_x1 +
  Penalty_x2


dim(
  Penalty
)


# ------------------------------------------------------------------------------
# 27. Penalized Tensor Fit
# ------------------------------------------------------------------------------

penalized_tensor_fit <- function(
    X,
    y,
    Penalty,
    lambda
) {
  
  p <- ncol(
    X
  )
  
  
  beta <- solve(
    crossprod(
      X
    ) +
      lambda *
      Penalty +
      1e-8 *
      diag(
        p
      ),
    crossprod(
      X,
      y
    )
  )
  
  
  fitted <- as.vector(
    X %*%
      beta
  )
  
  
  return(
    list(
      coefficients = as.vector(
        beta
      ),
      fitted = fitted,
      lambda = lambda
    )
  )
}


# ------------------------------------------------------------------------------
# 28. Fit One Penalized Model
# ------------------------------------------------------------------------------

lambda <- 1


penalized_model <- penalized_tensor_fit(
  X_tensor,
  y,
  Penalty,
  lambda
)


penalized_MSE <- mean(
  (
    y -
      penalized_model$fitted
  )^2
)


penalized_MSE


# ------------------------------------------------------------------------------
# 29. Smoother Matrix
# ------------------------------------------------------------------------------

# Penalized tensor-product splines are linear smoothers:
#
# y_hat = S_lambda y


tensor_smoother_matrix <- function(
    X,
    Penalty,
    lambda
) {
  
  solve_part <- solve(
    crossprod(
      X
    ) +
      lambda *
      Penalty +
      1e-8 *
      diag(
        ncol(
          X
        )
      )
  )
  
  
  X %*%
    solve_part %*%
    t(X)
}


# ------------------------------------------------------------------------------
# 30. Effective Degrees of Freedom
# ------------------------------------------------------------------------------

effective_df <- function(
    X,
    Penalty,
    lambda
) {
  
  S <- tensor_smoother_matrix(
    X,
    Penalty,
    lambda
  )
  
  
  sum(
    diag(
      S
    )
  )
}


effective_df(
  X_tensor,
  Penalty,
  lambda = 1
)


# ------------------------------------------------------------------------------
# 31. Regularization Path
# ------------------------------------------------------------------------------

lambda_grid <- 10^seq(
  -4,
  5,
  length.out = 100
)


training_MSE_path <- numeric(
  length(
    lambda_grid
  )
)


df_path <- numeric(
  length(
    lambda_grid
  )
)


roughness_path <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  model_i <- penalized_tensor_fit(
    X_tensor,
    y,
    Penalty,
    lambda_grid[i]
  )
  
  
  training_MSE_path[i] <- mean(
    (
      y -
        model_i$fitted
    )^2
  )
  
  
  df_path[i] <- effective_df(
    X_tensor,
    Penalty,
    lambda_grid[i]
  )
  
  
  roughness_path[i] <-
    as.numeric(
      t(
        model_i$coefficients
      ) %*%
        Penalty %*%
        model_i$coefficients
    )
}


# ------------------------------------------------------------------------------
# 32. Plot Effective Degrees of Freedom
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  df_path,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "Effective Degrees of Freedom",
  main = "Tensor-Spline Complexity"
)


# ------------------------------------------------------------------------------
# 33. Plot Training MSE
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  training_MSE_path,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "Training MSE",
  main = "Training Error vs Penalty"
)


# ------------------------------------------------------------------------------
# 34. Plot Roughness
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  roughness_path,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "Roughness",
  main = "Surface Roughness vs Penalty"
)


# ------------------------------------------------------------------------------
# 35. Generalized Cross-Validation
# ------------------------------------------------------------------------------

tensor_gcv <- function(
    X,
    y,
    Penalty,
    lambda
) {
  
  S <- tensor_smoother_matrix(
    X,
    Penalty,
    lambda
  )
  
  
  fitted <- as.vector(
    S %*%
      y
  )
  
  
  RSS <- sum(
    (
      y -
        fitted
    )^2
  )
  
  
  df <- sum(
    diag(
      S
    )
  )
  
  
  n <- length(
    y
  )
  
  
  GCV <- (
    RSS /
      n
  ) /
    (
      1 -
        df /
        n
    )^2
  
  
  return(
    GCV
  )
}


# ------------------------------------------------------------------------------
# 36. GCV Across Lambda
# ------------------------------------------------------------------------------

GCV_path <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  GCV_path[i] <- tensor_gcv(
    X_tensor,
    y,
    Penalty,
    lambda_grid[i]
  )
}


best_index <- which.min(
  GCV_path
)


best_lambda <- lambda_grid[
  best_index
]


best_lambda


# ------------------------------------------------------------------------------
# 37. Plot GCV
# ------------------------------------------------------------------------------

plot(
  log10(
    lambda_grid
  ),
  GCV_path,
  type = "l",
  xlab = "log10(lambda)",
  ylab = "GCV",
  main = "Tensor-Spline GCV"
)


abline(
  v = log10(
    best_lambda
  ),
  lty = 2
)


# ------------------------------------------------------------------------------
# 38. Fit Final Penalized Tensor Model
# ------------------------------------------------------------------------------

final_model <- penalized_tensor_fit(
  X_tensor,
  y,
  Penalty,
  best_lambda
)


final_MSE <- mean(
  (
    y -
      final_model$fitted
  )^2
)


final_df <- effective_df(
  X_tensor,
  Penalty,
  best_lambda
)


data.frame(
  Lambda = best_lambda,
  Training_MSE = final_MSE,
  Effective_DF = final_df
)


# ------------------------------------------------------------------------------
# 39. Predict Final Penalized Surface
# ------------------------------------------------------------------------------

B1_grid <- splines::bs(
  grid$x1,
  knots = knots_x1,
  degree = degree,
  Boundary.knots = boundary_x1,
  intercept = TRUE
)


B2_grid <- splines::bs(
  grid$x2,
  knots = knots_x2,
  degree = degree,
  Boundary.knots = boundary_x2,
  intercept = TRUE
)


X_grid_tensor <- tensor_product_basis(
  B1_grid,
  B2_grid
)


grid_penalized_prediction <- as.vector(
  X_grid_tensor %*%
    final_model$coefficients
)


penalized_surface <- matrix(
  grid_penalized_prediction,
  nrow = length(
    x1_grid
  ),
  ncol = length(
    x2_grid
  )
)


# ------------------------------------------------------------------------------
# 40. Plot Final Penalized Surface
# ------------------------------------------------------------------------------

persp(
  x1_grid,
  x2_grid,
  penalized_surface,
  theta = 35,
  phi = 25,
  expand = 0.7,
  xlab = "x1",
  ylab = "x2",
  zlab = "Predicted y",
  main = "Penalized Tensor-Product Spline"
)


# ------------------------------------------------------------------------------
# 41. Train-Test Split
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


x1_train <- x1[
  train_index
]

x2_train <- x2[
  train_index
]

y_train <- y[
  train_index
]


x1_test <- x1[
  test_index
]

x2_test <- x2[
  test_index
]

y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 42. Training Knots
# ------------------------------------------------------------------------------

train_knots_x1 <- as.numeric(
  quantile(
    x1_train,
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


train_knots_x2 <- as.numeric(
  quantile(
    x2_train,
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


train_boundary_x1 <- range(
  x1_train
)


train_boundary_x2 <- range(
  x2_train
)


# ------------------------------------------------------------------------------
# 43. Build Training Tensor Basis
# ------------------------------------------------------------------------------

B1_train <- splines::bs(
  x1_train,
  knots = train_knots_x1,
  degree = 3,
  Boundary.knots = train_boundary_x1,
  intercept = TRUE
)


B2_train <- splines::bs(
  x2_train,
  knots = train_knots_x2,
  degree = 3,
  Boundary.knots = train_boundary_x2,
  intercept = TRUE
)


X_train_tensor <- tensor_product_basis(
  B1_train,
  B2_train
)


# ------------------------------------------------------------------------------
# 44. Construct Training Penalty
# ------------------------------------------------------------------------------

K1_train <- ncol(
  B1_train
)


K2_train <- ncol(
  B2_train
)


D1_train <- diff(
  diag(
    K1_train
  ),
  differences = 2
)


D2_train <- diff(
  diag(
    K2_train
  ),
  differences = 2
)


P1_train <- crossprod(
  D1_train
)


P2_train <- crossprod(
  D2_train
)


Penalty_train <-
  kronecker(
    P1_train,
    diag(
      K2_train
    )
  ) +
  kronecker(
    diag(
      K1_train
    ),
    P2_train
  )


# ------------------------------------------------------------------------------
# 45. Choose Lambda Using Training GCV
# ------------------------------------------------------------------------------

training_GCV <- numeric(
  length(
    lambda_grid
  )
)


for (i in seq_along(
  lambda_grid
)) {
  
  training_GCV[i] <- tensor_gcv(
    X_train_tensor,
    y_train,
    Penalty_train,
    lambda_grid[i]
  )
}


best_train_lambda <- lambda_grid[
  which.min(
    training_GCV
  )
]


best_train_lambda


# ------------------------------------------------------------------------------
# 46. Fit Training Model
# ------------------------------------------------------------------------------

train_model <- penalized_tensor_fit(
  X_train_tensor,
  y_train,
  Penalty_train,
  best_train_lambda
)


# ------------------------------------------------------------------------------
# 47. Build Test Tensor Basis
# ------------------------------------------------------------------------------

# Use the training knots and boundaries.


inside_domain <-
  x1_test >= train_boundary_x1[1] &
  x1_test <= train_boundary_x1[2] &
  x2_test >= train_boundary_x2[1] &
  x2_test <= train_boundary_x2[2]


x1_test_use <- x1_test[
  inside_domain
]

x2_test_use <- x2_test[
  inside_domain
]

y_test_use <- y_test[
  inside_domain
]


B1_test <- splines::bs(
  x1_test_use,
  knots = train_knots_x1,
  degree = 3,
  Boundary.knots = train_boundary_x1,
  intercept = TRUE
)


B2_test <- splines::bs(
  x2_test_use,
  knots = train_knots_x2,
  degree = 3,
  Boundary.knots = train_boundary_x2,
  intercept = TRUE
)


X_test_tensor <- tensor_product_basis(
  B1_test,
  B2_test
)


# ------------------------------------------------------------------------------
# 48. Test Predictions
# ------------------------------------------------------------------------------

test_prediction <- as.vector(
  X_test_tensor %*%
    train_model$coefficients
)


test_MSE <- mean(
  (
    y_test_use -
      test_prediction
  )^2
)


test_RMSE <- sqrt(
  test_MSE
)


data.frame(
  Test_MSE = test_MSE,
  Test_RMSE = test_RMSE
)


# ------------------------------------------------------------------------------
# 49. Compare Additive and Interaction Structure
# ------------------------------------------------------------------------------

# True model includes:
#
# sin(x1 * x2)
#
# which is a nonlinear interaction.
#
# Additive splines cannot represent this exactly:
#
# f1(x1) + f2(x2)
#
# Tensor-product splines can.


cat(
  "Linear regression MSE:",
  round(
    linear_MSE,
    4
  ),
  "\n"
)


cat(
  "Additive spline MSE:",
  round(
    additive_MSE,
    4
  ),
  "\n"
)


cat(
  "Unpenalized tensor spline MSE:",
  round(
    tensor_MSE,
    4
  ),
  "\n"
)


cat(
  "Best tensor penalty lambda:",
  best_lambda,
  "\n"
)


cat(
  "Final effective degrees of freedom:",
  round(
    final_df,
    3
  ),
  "\n"
)


cat(
  "Final penalized tensor MSE:",
  round(
    final_MSE,
    4
  ),
  "\n"
)


cat(
  "Test RMSE:",
  round(
    test_RMSE,
    4
  ),
  "\n"
)