# ==============================================================================
# Neural Networks
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Feedforward neural networks
#   - Single hidden layer
#   - Hidden units
#   - Activation functions
#   - Forward propagation
#   - Backpropagation
#   - Gradient descent
#   - Weight initialization
#   - Feature scaling
#   - Learning rate
#   - Hidden-layer size
#   - Weight decay
#   - Early stopping
#   - Gradient checking
#
#
# Single-hidden-layer neural network:
#
#       z_m
#       =
#       alpha_0m + alpha_m' x
#
#
#       h_m
#       =
#       sigma(z_m)
#
#
#       f(x)
#       =
#       beta_0
#       +
#       sum_{m=1}^M beta_m h_m
#
#
# In matrix notation:
#
#       Z = X W1 + b1
#
#       H = sigma(Z)
#
#       y_hat = H W2 + b2
#
#
# For regression, the output layer is linear.
#
#
# This script implements the entire neural network manually using base R.
# No neural-network package is required.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR REGRESSION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Predictors
# ------------------------------------------------------------------------------

set.seed(123)

n <- 500

p <- 5


X <- matrix(
  rnorm(
    n * p
  ),
  nrow = n,
  ncol = p
)


colnames(X) <- paste0(
  "x",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 2. True Regression Function
# ------------------------------------------------------------------------------

true_function <- function(
    x1,
    x2,
    x3,
    x4,
    x5
) {
  
  2 +
    2.5 * sin(
      x1 + 0.75 * x2
    ) +
    1.5 * tanh(
      x3 - x4
    ) +
    0.5 * x5^2
}


y_true <- true_function(
  X[, 1],
  X[, 2],
  X[, 3],
  X[, 4],
  X[, 5]
)


sigma <- 0.75


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
  )


# ------------------------------------------------------------------------------
# 3. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data.frame(
    y = y,
    X
  ),
  main = "Neural Network Regression Data"
)


# ==============================================================================
# PART II
#
# TRAIN / VALIDATION / TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Split Data
# ------------------------------------------------------------------------------

set.seed(456)


indices <- sample(
  seq_len(n)
)


n_train <- floor(
  0.60 * n
)


n_validation <- floor(
  0.20 * n
)


train_index <- indices[
  seq_len(
    n_train
  )
]


validation_index <- indices[
  (
    n_train + 1
  ):(
    n_train +
      n_validation
  )
]


test_index <- indices[
  (
    n_train +
      n_validation +
      1
  ):n
]


X_train <- X[
  train_index,
  ,
  drop = FALSE
]


y_train <- y[
  train_index
]


X_validation <- X[
  validation_index,
  ,
  drop = FALSE
]


y_validation <- y[
  validation_index
]


X_test <- X[
  test_index,
  ,
  drop = FALSE
]


y_test <- y[
  test_index
]


# ==============================================================================
# PART III
#
# STANDARDIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Predictor Scaling
# ------------------------------------------------------------------------------

X_mean <- colMeans(
  X_train
)


X_sd <- apply(
  X_train,
  2,
  sd
)


X_train_scaled <- sweep(
  X_train,
  2,
  X_mean,
  "-"
)


X_train_scaled <- sweep(
  X_train_scaled,
  2,
  X_sd,
  "/"
)


X_validation_scaled <- sweep(
  X_validation,
  2,
  X_mean,
  "-"
)


X_validation_scaled <- sweep(
  X_validation_scaled,
  2,
  X_sd,
  "/"
)


X_test_scaled <- sweep(
  X_test,
  2,
  X_mean,
  "-"
)


X_test_scaled <- sweep(
  X_test_scaled,
  2,
  X_sd,
  "/"
)


# ------------------------------------------------------------------------------
# 6. Response Scaling
# ------------------------------------------------------------------------------

y_mean <- mean(
  y_train
)


y_sd <- sd(
  y_train
)


y_train_scaled <- (
  y_train -
    y_mean
) /
  y_sd


y_validation_scaled <- (
  y_validation -
    y_mean
) /
  y_sd


y_test_scaled <- (
  y_test -
    y_mean
) /
  y_sd


# Neural networks are highly sensitive to scale.
#
# All scaling parameters come ONLY from the training set.


# ==============================================================================
# PART IV
#
# ACTIVATION FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Sigmoid
# ------------------------------------------------------------------------------

stable_sigmoid <- function(z) {
  
  result <- numeric(
    length(z)
  )
  
  
  positive <- z >= 0
  
  
  result[
    positive
  ] <- 1 /
    (
      1 +
        exp(
          -z[
            positive
          ]
        )
    )
  
  
  exp_z <- exp(
    z[
      !positive
    ]
  )
  
  
  result[
    !positive
  ] <- exp_z /
    (
      1 +
        exp_z
    )
  
  
  dim(
    result
  ) <- dim(z)
  
  
  result
}


# ------------------------------------------------------------------------------
# 8. Tanh Activation
# ------------------------------------------------------------------------------

tanh_activation <- function(z) {
  
  tanh(z)
}


# ------------------------------------------------------------------------------
# 9. ReLU
# ------------------------------------------------------------------------------

relu <- function(z) {
  
  pmax(
    0,
    z
  )
}


# ------------------------------------------------------------------------------
# 10. Activation Wrapper
# ------------------------------------------------------------------------------

activation_function <- function(
    z,
    activation
) {
  
  if (
    activation ==
    "sigmoid"
  ) {
    
    return(
      stable_sigmoid(z)
    )
  }
  
  
  if (
    activation ==
    "tanh"
  ) {
    
    return(
      tanh_activation(z)
    )
  }
  
  
  if (
    activation ==
    "relu"
  ) {
    
    return(
      relu(z)
    )
  }
  
  
  stop(
    "Unknown activation function."
  )
}


# ------------------------------------------------------------------------------
# 11. Activation Derivatives
# ------------------------------------------------------------------------------

activation_derivative <- function(
    z,
    activation
) {
  
  if (
    activation ==
    "sigmoid"
  ) {
    
    s <- stable_sigmoid(
      z
    )
    
    
    return(
      s *
        (
          1 -
            s
        )
    )
  }
  
  
  if (
    activation ==
    "tanh"
  ) {
    
    return(
      1 -
        tanh(z)^2
    )
  }
  
  
  if (
    activation ==
    "relu"
  ) {
    
    return(
      1 *
        (
          z > 0
        )
    )
  }
  
  
  stop(
    "Unknown activation function."
  )
}


# ==============================================================================
# PART V
#
# INITIALIZE THE NETWORK
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Weight Initialization
# ------------------------------------------------------------------------------

initialize_network <- function(
    input_size,
    hidden_size,
    activation = "tanh",
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
  
  
  # Xavier-style initialization works well for sigmoid/tanh.
  #
  # He-style initialization is preferable for ReLU.
  
  if (
    activation ==
    "relu"
  ) {
    
    sd_W1 <- sqrt(
      2 /
        input_size
    )
    
  } else {
    
    sd_W1 <- sqrt(
      2 /
        (
          input_size +
            hidden_size
        )
    )
  }
  
  
  W1 <- matrix(
    rnorm(
      input_size *
        hidden_size,
      mean = 0,
      sd = sd_W1
    ),
    nrow =
      input_size,
    ncol =
      hidden_size
  )
  
  
  b1 <- rep(
    0,
    hidden_size
  )
  
  
  W2 <- matrix(
    rnorm(
      hidden_size,
      mean = 0,
      sd = sqrt(
        1 /
          hidden_size
      )
    ),
    nrow =
      hidden_size,
    ncol = 1
  )
  
  
  b2 <- 0
  
  
  return(
    list(
      W1 = W1,
      b1 = b1,
      W2 = W2,
      b2 = b2
    )
  )
}


# ==============================================================================
# PART VI
#
# FORWARD PROPAGATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Forward Pass
# ------------------------------------------------------------------------------

forward_pass <- function(
    X,
    parameters,
    activation = "tanh"
) {
  
  X <- as.matrix(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Hidden pre-activation
  # --------------------------------------------------------------------------
  
  Z1 <- sweep(
    X %*%
      parameters$W1,
    2,
    parameters$b1,
    "+"
  )
  
  
  # --------------------------------------------------------------------------
  # Hidden activation
  # --------------------------------------------------------------------------
  
  H <- activation_function(
    Z1,
    activation
  )
  
  
  # --------------------------------------------------------------------------
  # Linear output layer
  # --------------------------------------------------------------------------
  
  y_hat <- as.vector(
    H %*%
      parameters$W2 +
      parameters$b2
  )
  
  
  return(
    list(
      Z1 = Z1,
      H = H,
      y_hat = y_hat
    )
  )
}


# ==============================================================================
# PART VII
#
# LOSS FUNCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Mean Squared Error Loss
# ------------------------------------------------------------------------------

mse_loss <- function(
    y,
    y_hat
) {
  
  mean(
    (
      y_hat -
        y
    )^2
  )
}


# ------------------------------------------------------------------------------
# 15. Regularized Objective
# ------------------------------------------------------------------------------

network_objective <- function(
    y,
    y_hat,
    parameters,
    lambda = 0
) {
  
  data_loss <- 0.5 *
    mean(
      (
        y_hat -
          y
      )^2
    )
  
  
  penalty <- 0.5 *
    lambda *
    (
      sum(
        parameters$W1^2
      ) +
        sum(
          parameters$W2^2
        )
    )
  
  
  data_loss +
    penalty
}


# Bias terms are deliberately not penalized.


# ==============================================================================
# PART VIII
#
# BACKPROPAGATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Compute Gradients
# ------------------------------------------------------------------------------

backward_pass <- function(
    X,
    y,
    parameters,
    cache,
    activation = "tanh",
    lambda = 0
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  # ==========================================================================
  # OUTPUT LAYER
  # ==========================================================================
  
  # Objective:
  #
  #       L
  #       =
  #       1/(2n)
  #       sum_i (y_hat_i - y_i)^2
  #
  #
  # Therefore:
  #
  #       dL / dy_hat
  #       =
  #       (y_hat - y) / n
  
  
  dY <- (
    cache$y_hat -
      y
  ) /
    n
  
  
  # --------------------------------------------------------------------------
  # Gradient for W2
  # --------------------------------------------------------------------------
  
  dW2 <- crossprod(
    cache$H,
    dY
  )
  
  
  dW2 <- matrix(
    dW2,
    ncol = 1
  )
  
  
  dW2 <- dW2 +
    lambda *
    parameters$W2
  
  
  # --------------------------------------------------------------------------
  # Gradient for b2
  # --------------------------------------------------------------------------
  
  db2 <- sum(
    dY
  )
  
  
  # ==========================================================================
  # HIDDEN LAYER
  # ==========================================================================
  
  # --------------------------------------------------------------------------
  # Propagate gradient into hidden activations
  # --------------------------------------------------------------------------
  
  dH <- outer(
    dY,
    as.vector(
      parameters$W2
    )
  )
  
  
  # --------------------------------------------------------------------------
  # Chain rule through activation
  # --------------------------------------------------------------------------
  
  dZ1 <- dH *
    activation_derivative(
      cache$Z1,
      activation
    )
  
  
  # --------------------------------------------------------------------------
  # Gradient for W1
  # --------------------------------------------------------------------------
  
  dW1 <- crossprod(
    X,
    dZ1
  )
  
  
  dW1 <- dW1 +
    lambda *
    parameters$W1
  
  
  # --------------------------------------------------------------------------
  # Gradient for b1
  # --------------------------------------------------------------------------
  
  db1 <- colSums(
    dZ1
  )
  
  
  return(
    list(
      dW1 = dW1,
      db1 = db1,
      dW2 = dW2,
      db2 = db2
    )
  )
}


# ==============================================================================
# PART IX
#
# GRADIENT DESCENT UPDATE
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Update Parameters
# ------------------------------------------------------------------------------

update_parameters <- function(
    parameters,
    gradients,
    learning_rate
) {
  
  parameters$W1 <-
    parameters$W1 -
    learning_rate *
    gradients$dW1
  
  
  parameters$b1 <-
    parameters$b1 -
    learning_rate *
    gradients$db1
  
  
  parameters$W2 <-
    parameters$W2 -
    learning_rate *
    gradients$dW2
  
  
  parameters$b2 <-
    parameters$b2 -
    learning_rate *
    gradients$db2
  
  
  parameters
}


# ==============================================================================
# PART X
#
# TRAIN THE NEURAL NETWORK
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Training Function
# ------------------------------------------------------------------------------

fit_neural_network <- function(
    X_train,
    y_train,
    X_validation = NULL,
    y_validation = NULL,
    hidden_size = 10,
    activation = "tanh",
    epochs = 3000,
    learning_rate = 0.05,
    lambda = 0,
    patience = Inf,
    seed = 123,
    verbose = FALSE
) {
  
  X_train <- as.matrix(
    X_train
  )
  
  
  input_size <- ncol(
    X_train
  )
  
  
  parameters <- initialize_network(
    input_size =
      input_size,
    hidden_size =
      hidden_size,
    activation =
      activation,
    seed =
      seed
  )
  
  
  training_loss <- numeric(
    epochs
  )
  
  
  validation_loss <- rep(
    NA_real_,
    epochs
  )
  
  
  best_parameters <- parameters
  
  best_validation_loss <- Inf
  
  best_epoch <- epochs
  
  epochs_without_improvement <- 0
  
  
  # ==========================================================================
  # TRAINING LOOP
  # ==========================================================================
  
  for (
    epoch in seq_len(
      epochs
    )
  ) {
    
    # ------------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------------
    
    cache <- forward_pass(
      X_train,
      parameters,
      activation =
        activation
    )
    
    
    # ------------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------------
    
    training_loss[epoch] <-
      network_objective(
        y = y_train,
        y_hat =
          cache$y_hat,
        parameters =
          parameters,
        lambda =
          lambda
      )
    
    
    # ------------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------------
    
    gradients <- backward_pass(
      X = X_train,
      y = y_train,
      parameters =
        parameters,
      cache =
        cache,
      activation =
        activation,
      lambda =
        lambda
    )
    
    
    # ------------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------------
    
    parameters <- update_parameters(
      parameters =
        parameters,
      gradients =
        gradients,
      learning_rate =
        learning_rate
    )
    
    
    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------
    
    if (
      !is.null(
        X_validation
      )
    ) {
      
      validation_cache <- forward_pass(
        X_validation,
        parameters,
        activation =
          activation
      )
      
      
      validation_loss[epoch] <-
        0.5 *
        mean(
          (
            validation_cache$y_hat -
              y_validation
          )^2
        )
      
      
      if (
        validation_loss[epoch] <
        best_validation_loss -
        1e-8
      ) {
        
        best_validation_loss <-
          validation_loss[epoch]
        
        best_parameters <-
          parameters
        
        best_epoch <-
          epoch
        
        epochs_without_improvement <-
          0
        
      } else {
        
        epochs_without_improvement <-
          epochs_without_improvement +
          1
      }
      
      
      if (
        epochs_without_improvement >=
        patience
      ) {
        
        training_loss <-
          training_loss[
            seq_len(epoch)
          ]
        
        
        validation_loss <-
          validation_loss[
            seq_len(epoch)
          ]
        
        
        break
      }
      
    } else {
      
      best_parameters <-
        parameters
      
      best_epoch <-
        epoch
    }
    
    
    if (
      verbose &&
      (
        epoch == 1 ||
        epoch %% 250 == 0
      )
    ) {
      
      cat(
        "Epoch:",
        epoch,
        "| Training Loss:",
        round(
          training_loss[epoch],
          6
        )
      )
      
      
      if (
        !is.null(
          X_validation
        )
      ) {
        
        cat(
          "| Validation Loss:",
          round(
            validation_loss[epoch],
            6
          )
        )
      }
      
      
      cat(
        "\n"
      )
    }
  }
  
  
  return(
    list(
      parameters =
        best_parameters,
      final_parameters =
        parameters,
      training_loss =
        training_loss,
      validation_loss =
        validation_loss,
      best_epoch =
        best_epoch,
      hidden_size =
        hidden_size,
      activation =
        activation,
      learning_rate =
        learning_rate,
      lambda =
        lambda
    )
  )
}


# ==============================================================================
# PART XI
#
# FIT THE NETWORK
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Train
# ------------------------------------------------------------------------------

NN_model <- fit_neural_network(
  X_train =
    X_train_scaled,
  y_train =
    y_train_scaled,
  X_validation =
    X_validation_scaled,
  y_validation =
    y_validation_scaled,
  hidden_size = 10,
  activation = "tanh",
  epochs = 5000,
  learning_rate = 0.05,
  lambda = 0.001,
  patience = 400,
  seed = 123,
  verbose = TRUE
)


NN_model$best_epoch


# ==============================================================================
# PART XII
#
# PREDICTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Prediction on Scaled Response
# ------------------------------------------------------------------------------

predict_neural_network_scaled <- function(
    model,
    X_new
) {
  
  forward_pass(
    X_new,
    model$parameters,
    activation =
      model$activation
  )$y_hat
}


# ------------------------------------------------------------------------------
# 21. Prediction on Original Response Scale
# ------------------------------------------------------------------------------

predict_neural_network <- function(
    model,
    X_new,
    y_mean,
    y_sd
) {
  
  prediction_scaled <-
    predict_neural_network_scaled(
      model,
      X_new
    )
  
  
  y_mean +
    y_sd *
    prediction_scaled
}


# ------------------------------------------------------------------------------
# 22. Training Prediction
# ------------------------------------------------------------------------------

train_prediction <- predict_neural_network(
  NN_model,
  X_train_scaled,
  y_mean,
  y_sd
)


# ------------------------------------------------------------------------------
# 23. Validation Prediction
# ------------------------------------------------------------------------------

validation_prediction <- predict_neural_network(
  NN_model,
  X_validation_scaled,
  y_mean,
  y_sd
)


# ------------------------------------------------------------------------------
# 24. Test Prediction
# ------------------------------------------------------------------------------

test_prediction <- predict_neural_network(
  NN_model,
  X_test_scaled,
  y_mean,
  y_sd
)


# ==============================================================================
# PART XIII
#
# PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Metrics
# ------------------------------------------------------------------------------

MSE <- function(
    actual,
    predicted
) {
  
  mean(
    (
      actual -
        predicted
    )^2
  )
}


RMSE <- function(
    actual,
    predicted
) {
  
  sqrt(
    MSE(
      actual,
      predicted
    )
  )
}


R_squared <- function(
    actual,
    predicted
) {
  
  1 -
    sum(
      (
        actual -
          predicted
      )^2
    ) /
    sum(
      (
        actual -
          mean(actual)
      )^2
    )
}


# ------------------------------------------------------------------------------
# 26. Performance Table
# ------------------------------------------------------------------------------

performance <- data.frame(
  Dataset = c(
    "Training",
    "Validation",
    "Test"
  ),
  MSE = c(
    MSE(
      y_train,
      train_prediction
    ),
    MSE(
      y_validation,
      validation_prediction
    ),
    MSE(
      y_test,
      test_prediction
    )
  ),
  RMSE = c(
    RMSE(
      y_train,
      train_prediction
    ),
    RMSE(
      y_validation,
      validation_prediction
    ),
    RMSE(
      y_test,
      test_prediction
    )
  ),
  R_squared = c(
    R_squared(
      y_train,
      train_prediction
    ),
    R_squared(
      y_validation,
      validation_prediction
    ),
    R_squared(
      y_test,
      test_prediction
    )
  )
)


performance


# ==============================================================================
# PART XIV
#
# LEARNING CURVES
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Training and Validation Loss
# ------------------------------------------------------------------------------

epochs_used <- seq_along(
  NN_model$training_loss
)


matplot(
  epochs_used,
  cbind(
    NN_model$training_loss,
    NN_model$validation_loss
  ),
  type = "l",
  lty = c(
    1,
    2
  ),
  lwd = 2,
  xlab = "Epoch",
  ylab = "Loss",
  main = "Neural Network Learning Curves"
)


legend(
  "topright",
  legend = c(
    "Training",
    "Validation"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


abline(
  v =
    NN_model$best_epoch,
  lty = 3
)


# ==============================================================================
# PART XV
#
# OBSERVED VS PREDICTED
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Test Predictions
# ------------------------------------------------------------------------------

plot(
  y_test,
  test_prediction,
  pch = 19,
  xlab = "Observed Response",
  ylab = "Predicted Response",
  main = "Neural Network: Observed vs Predicted"
)


abline(
  a = 0,
  b = 1,
  lwd = 2,
  lty = 2
)


# ------------------------------------------------------------------------------
# 29. Test Residuals
# ------------------------------------------------------------------------------

test_residuals <- y_test -
  test_prediction


plot(
  test_prediction,
  test_residuals,
  pch = 19,
  xlab = "Predicted Response",
  ylab = "Residual",
  main = "Neural Network Residuals"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART XVI
#
# UNDERSTANDING THE HIDDEN LAYER
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. First-Layer Weights
# ------------------------------------------------------------------------------

W1 <- NN_model$parameters$W1


rownames(
  W1
) <- colnames(X)


colnames(
  W1
) <- paste0(
  "Hidden_",
  seq_len(
    ncol(W1)
  )
)


round(
  W1,
  3
)


# Each column defines a projection:
#
#       z_m
#       =
#       b_m + alpha_m' x
#
#
# This is the connection with projection pursuit regression.


# ------------------------------------------------------------------------------
# 31. Hidden Biases
# ------------------------------------------------------------------------------

NN_model$parameters$b1


# ------------------------------------------------------------------------------
# 32. Output Weights
# ------------------------------------------------------------------------------

NN_model$parameters$W2


# ==============================================================================
# PART XVII
#
# NORMALIZED HIDDEN DIRECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Normalize First-Layer Weight Vectors
# ------------------------------------------------------------------------------

hidden_directions <- W1


for (
  j in seq_len(
    ncol(
      hidden_directions
    )
  )
) {
  
  norm_j <- sqrt(
    sum(
      hidden_directions[
        ,
        j
      ]^2
    )
  )
  
  
  if (
    norm_j >
    0
  ) {
    
    hidden_directions[
      ,
      j
    ] <-
      hidden_directions[
        ,
        j
      ] /
      norm_j
  }
}


round(
  hidden_directions,
  3
)


# ==============================================================================
# PART XVIII
#
# HIDDEN UNIT ACTIVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Hidden-Layer Values
# ------------------------------------------------------------------------------

hidden_cache <- forward_pass(
  X_train_scaled,
  NN_model$parameters,
  activation =
    NN_model$activation
)


hidden_activations <- hidden_cache$H


# ------------------------------------------------------------------------------
# 35. Plot First Four Hidden Units
# ------------------------------------------------------------------------------

number_to_plot <- min(
  4,
  ncol(
    hidden_activations
  )
)


par(
  mfrow = c(
    2,
    2
  )
)


for (
  j in seq_len(
    number_to_plot
  )
) {
  
  hidden_projection <-
    X_train_scaled %*%
    NN_model$parameters$W1[
      ,
      j
    ] +
    NN_model$parameters$b1[j]
  
  
  plot(
    hidden_projection,
    hidden_activations[
      ,
      j
    ],
    pch = 19,
    cex = 0.5,
    xlab = paste(
      "Pre-Activation:",
      j
    ),
    ylab = paste(
      "Hidden Activation:",
      j
    ),
    main = paste(
      "Hidden Unit",
      j
    )
  )
}


par(
  mfrow = c(
    1,
    1
  )
)


# ==============================================================================
# PART XIX
#
# MANUAL BACKPROPAGATION INSPECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. One Forward Pass
# ------------------------------------------------------------------------------

inspection_cache <- forward_pass(
  X_train_scaled,
  NN_model$parameters,
  activation =
    NN_model$activation
)


# ------------------------------------------------------------------------------
# 37. One Backward Pass
# ------------------------------------------------------------------------------

inspection_gradients <- backward_pass(
  X = X_train_scaled,
  y = y_train_scaled,
  parameters =
    NN_model$parameters,
  cache =
    inspection_cache,
  activation =
    NN_model$activation,
  lambda =
    NN_model$lambda
)


# ------------------------------------------------------------------------------
# 38. Gradient Norms
# ------------------------------------------------------------------------------

gradient_norms <- c(
  W1 =
    sqrt(
      sum(
        inspection_gradients$dW1^2
      )
    ),
  
  b1 =
    sqrt(
      sum(
        inspection_gradients$db1^2
      )
    ),
  
  W2 =
    sqrt(
      sum(
        inspection_gradients$dW2^2
      )
    ),
  
  b2 =
    abs(
      inspection_gradients$db2
    )
)


gradient_norms


# ==============================================================================
# PART XX
#
# NUMERICAL GRADIENT CHECKING
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Numerical Gradient for One Weight
# ------------------------------------------------------------------------------

numerical_gradient_W1 <- function(
    X,
    y,
    parameters,
    row,
    column,
    activation = "tanh",
    lambda = 0,
    epsilon = 1e-5
) {
  
  parameters_plus <- parameters
  
  parameters_minus <- parameters
  
  
  parameters_plus$W1[
    row,
    column
  ] <-
    parameters_plus$W1[
      row,
      column
    ] +
    epsilon
  
  
  parameters_minus$W1[
    row,
    column
  ] <-
    parameters_minus$W1[
      row,
      column
    ] -
    epsilon
  
  
  prediction_plus <- forward_pass(
    X,
    parameters_plus,
    activation
  )$y_hat
  
  
  prediction_minus <- forward_pass(
    X,
    parameters_minus,
    activation
  )$y_hat
  
  
  loss_plus <- network_objective(
    y,
    prediction_plus,
    parameters_plus,
    lambda
  )
  
  
  loss_minus <- network_objective(
    y,
    prediction_minus,
    parameters_minus,
    lambda
  )
  
  
  (
    loss_plus -
      loss_minus
  ) /
    (
      2 *
        epsilon
    )
}


# ------------------------------------------------------------------------------
# 40. Compare Analytical and Numerical Gradient
# ------------------------------------------------------------------------------

gradient_check_parameters <- initialize_network(
  input_size = p,
  hidden_size = 4,
  activation = "tanh",
  seed = 999
)


gradient_check_cache <- forward_pass(
  X_train_scaled,
  gradient_check_parameters,
  activation = "tanh"
)


gradient_check_analytical <- backward_pass(
  X = X_train_scaled,
  y = y_train_scaled,
  parameters =
    gradient_check_parameters,
  cache =
    gradient_check_cache,
  activation = "tanh",
  lambda = 0.001
)


analytical_gradient <-
  gradient_check_analytical$dW1[
    1,
    1
  ]


numerical_gradient <- numerical_gradient_W1(
  X = X_train_scaled,
  y = y_train_scaled,
  parameters =
    gradient_check_parameters,
  row = 1,
  column = 1,
  activation = "tanh",
  lambda = 0.001
)


gradient_check <- data.frame(
  Analytical =
    analytical_gradient,
  Numerical =
    numerical_gradient,
  Absolute_Difference =
    abs(
      analytical_gradient -
        numerical_gradient
    )
)


gradient_check


# ==============================================================================
# PART XXI
#
# EFFECT OF HIDDEN-LAYER SIZE
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Compare Network Width
# ------------------------------------------------------------------------------

hidden_sizes <- c(
  1,
  2,
  5,
  10,
  20
)


hidden_size_results <- data.frame(
  Hidden_Units =
    hidden_sizes,
  Validation_MSE =
    NA_real_,
  Test_MSE =
    NA_real_
)


hidden_size_models <- vector(
  "list",
  length(
    hidden_sizes
  )
)


for (
  i in seq_along(
    hidden_sizes
  )
) {
  
  model_i <- fit_neural_network(
    X_train =
      X_train_scaled,
    y_train =
      y_train_scaled,
    X_validation =
      X_validation_scaled,
    y_validation =
      y_validation_scaled,
    hidden_size =
      hidden_sizes[i],
    activation = "tanh",
    epochs = 3000,
    learning_rate = 0.05,
    lambda = 0.001,
    patience = 300,
    seed =
      100 + i,
    verbose = FALSE
  )
  
  
  hidden_size_models[[i]] <-
    model_i
  
  
  validation_prediction_i <-
    predict_neural_network(
      model_i,
      X_validation_scaled,
      y_mean,
      y_sd
    )
  
  
  test_prediction_i <-
    predict_neural_network(
      model_i,
      X_test_scaled,
      y_mean,
      y_sd
    )
  
  
  hidden_size_results$Validation_MSE[i] <-
    MSE(
      y_validation,
      validation_prediction_i
    )
  
  
  hidden_size_results$Test_MSE[i] <-
    MSE(
      y_test,
      test_prediction_i
    )
}


hidden_size_results


# ------------------------------------------------------------------------------
# 42. Plot Hidden-Layer Complexity
# ------------------------------------------------------------------------------

matplot(
  hidden_size_results$Hidden_Units,
  cbind(
    hidden_size_results$Validation_MSE,
    hidden_size_results$Test_MSE
  ),
  type = "b",
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2,
  xlab = "Number of Hidden Units",
  ylab = "MSE",
  main = "Neural Network Width"
)


legend(
  "topright",
  legend = c(
    "Validation",
    "Test"
  ),
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# Test error is shown only for illustration.
#
# Hyperparameters should be selected using the validation set, not the test set.


# ==============================================================================
# PART XXII
#
# WEIGHT DECAY
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Compare Regularization Strength
# ------------------------------------------------------------------------------

lambda_values <- c(
  0,
  0.0001,
  0.001,
  0.01,
  0.1
)


lambda_results <- data.frame(
  Lambda =
    lambda_values,
  Validation_MSE =
    NA_real_,
  Weight_Norm =
    NA_real_
)


lambda_models <- vector(
  "list",
  length(
    lambda_values
  )
)


for (
  i in seq_along(
    lambda_values
  )
) {
  
  model_i <- fit_neural_network(
    X_train =
      X_train_scaled,
    y_train =
      y_train_scaled,
    X_validation =
      X_validation_scaled,
    y_validation =
      y_validation_scaled,
    hidden_size = 10,
    activation = "tanh",
    epochs = 3000,
    learning_rate = 0.05,
    lambda =
      lambda_values[i],
    patience = 300,
    seed = 123,
    verbose = FALSE
  )
  
  
  lambda_models[[i]] <-
    model_i
  
  
  validation_prediction_i <-
    predict_neural_network(
      model_i,
      X_validation_scaled,
      y_mean,
      y_sd
    )
  
  
  lambda_results$Validation_MSE[i] <-
    MSE(
      y_validation,
      validation_prediction_i
    )
  
  
  lambda_results$Weight_Norm[i] <-
    sqrt(
      sum(
        model_i$parameters$W1^2
      ) +
        sum(
          model_i$parameters$W2^2
        )
    )
}


lambda_results


# ------------------------------------------------------------------------------
# 44. Regularization Path
# ------------------------------------------------------------------------------

plot(
  lambda_results$Lambda,
  lambda_results$Weight_Norm,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Lambda",
  ylab = "Total Weight Norm",
  main = "Neural Network Weight Decay"
)


# ==============================================================================
# PART XXIII
#
# ACTIVATION FUNCTION COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Compare Activations
# ------------------------------------------------------------------------------

activation_values <- c(
  "sigmoid",
  "tanh",
  "relu"
)


activation_results <- data.frame(
  Activation =
    activation_values,
  Validation_MSE =
    NA_real_,
  Test_MSE =
    NA_real_
)


for (
  i in seq_along(
    activation_values
  )
) {
  
  model_i <- fit_neural_network(
    X_train =
      X_train_scaled,
    y_train =
      y_train_scaled,
    X_validation =
      X_validation_scaled,
    y_validation =
      y_validation_scaled,
    hidden_size = 10,
    activation =
      activation_values[i],
    epochs = 3000,
    learning_rate =
      ifelse(
        activation_values[i] ==
          "relu",
        0.02,
        0.05
      ),
    lambda = 0.001,
    patience = 300,
    seed = 123,
    verbose = FALSE
  )
  
  
  validation_prediction_i <-
    predict_neural_network(
      model_i,
      X_validation_scaled,
      y_mean,
      y_sd
    )
  
  
  test_prediction_i <-
    predict_neural_network(
      model_i,
      X_test_scaled,
      y_mean,
      y_sd
    )
  
  
  activation_results$Validation_MSE[i] <-
    MSE(
      y_validation,
      validation_prediction_i
    )
  
  
  activation_results$Test_MSE[i] <-
    MSE(
      y_test,
      test_prediction_i
    )
}


activation_results


# ==============================================================================
# PART XXIV
#
# EFFECT OF LEARNING RATE
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Compare Learning Rates
# ------------------------------------------------------------------------------

learning_rates <- c(
  0.005,
  0.01,
  0.05,
  0.1
)


learning_rate_results <- data.frame(
  Learning_Rate =
    learning_rates,
  Validation_MSE =
    NA_real_,
  Best_Epoch =
    NA_integer_
)


for (
  i in seq_along(
    learning_rates
  )
) {
  
  model_i <- fit_neural_network(
    X_train =
      X_train_scaled,
    y_train =
      y_train_scaled,
    X_validation =
      X_validation_scaled,
    y_validation =
      y_validation_scaled,
    hidden_size = 10,
    activation = "tanh",
    epochs = 3000,
    learning_rate =
      learning_rates[i],
    lambda = 0.001,
    patience = 300,
    seed = 123,
    verbose = FALSE
  )
  
  
  prediction_i <- predict_neural_network(
    model_i,
    X_validation_scaled,
    y_mean,
    y_sd
  )
  
  
  learning_rate_results$Validation_MSE[i] <-
    MSE(
      y_validation,
      prediction_i
    )
  
  
  learning_rate_results$Best_Epoch[i] <-
    model_i$best_epoch
}


learning_rate_results


# ==============================================================================
# PART XXV
#
# MULTIPLE RANDOM INITIALIZATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Demonstrate Nonconvex Optimization
# ------------------------------------------------------------------------------

seeds <- 1:10


initialization_results <- data.frame(
  Seed =
    seeds,
  Validation_MSE =
    NA_real_
)


for (
  i in seq_along(
    seeds
  )
) {
  
  model_i <- fit_neural_network(
    X_train =
      X_train_scaled,
    y_train =
      y_train_scaled,
    X_validation =
      X_validation_scaled,
    y_validation =
      y_validation_scaled,
    hidden_size = 10,
    activation = "tanh",
    epochs = 2500,
    learning_rate = 0.05,
    lambda = 0.001,
    patience = 250,
    seed =
      seeds[i],
    verbose = FALSE
  )
  
  
  prediction_i <- predict_neural_network(
    model_i,
    X_validation_scaled,
    y_mean,
    y_sd
  )
  
  
  initialization_results$Validation_MSE[i] <-
    MSE(
      y_validation,
      prediction_i
    )
}


initialization_results


# ------------------------------------------------------------------------------
# 48. Initialization Variability
# ------------------------------------------------------------------------------

boxplot(
  initialization_results$Validation_MSE,
  ylab = "Validation MSE",
  main = "Sensitivity to Random Initialization"
)


# ==============================================================================
# PART XXVI
#
# TWO-DIMENSIONAL PREDICTION SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Construct Grid
# ------------------------------------------------------------------------------

grid_size <- 80


grid_x1 <- seq(
  -3,
  3,
  length.out =
    grid_size
)


grid_x2 <- seq(
  -3,
  3,
  length.out =
    grid_size
)


grid <- expand.grid(
  x1 =
    grid_x1,
  x2 =
    grid_x2
)


X_grid <- cbind(
  x1 =
    grid$x1,
  x2 =
    grid$x2,
  x3 = 0,
  x4 = 0,
  x5 = 0
)


# ------------------------------------------------------------------------------
# 50. Standardize Grid
# ------------------------------------------------------------------------------

X_grid_scaled <- sweep(
  X_grid,
  2,
  X_mean,
  "-"
)


X_grid_scaled <- sweep(
  X_grid_scaled,
  2,
  X_sd,
  "/"
)


# ------------------------------------------------------------------------------
# 51. Predict Grid
# ------------------------------------------------------------------------------

grid_prediction <- predict_neural_network(
  NN_model,
  X_grid_scaled,
  y_mean,
  y_sd
)


prediction_matrix <- matrix(
  grid_prediction,
  nrow =
    grid_size,
  ncol =
    grid_size
)


# ------------------------------------------------------------------------------
# 52. Prediction Surface
# ------------------------------------------------------------------------------

image(
  grid_x1,
  grid_x2,
  prediction_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "Neural Network Prediction Surface"
)


contour(
  grid_x1,
  grid_x2,
  prediction_matrix,
  add = TRUE
)


# ==============================================================================
# PART XXVII
#
# TRUE VS ESTIMATED RESPONSE SLICE
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. Vary x1
# ------------------------------------------------------------------------------

x1_grid <- seq(
  -3,
  3,
  length.out = 300
)


X_slice <- cbind(
  x1 = x1_grid,
  x2 = 0,
  x3 = 0,
  x4 = 0,
  x5 = 0
)


X_slice_scaled <- sweep(
  X_slice,
  2,
  X_mean,
  "-"
)


X_slice_scaled <- sweep(
  X_slice_scaled,
  2,
  X_sd,
  "/"
)


slice_prediction <- predict_neural_network(
  NN_model,
  X_slice_scaled,
  y_mean,
  y_sd
)


true_slice <- true_function(
  x1_grid,
  0,
  0,
  0,
  0
)


plot(
  x1_grid,
  true_slice,
  type = "l",
  lwd = 3,
  xlab = "x1",
  ylab = "Response",
  main = "True vs Neural Network Function"
)


lines(
  x1_grid,
  slice_prediction,
  lwd = 2,
  lty = 2
)


legend(
  "topleft",
  legend = c(
    "True",
    "Neural Network"
  ),
  lty = c(
    1,
    2
  ),
  lwd = c(
    3,
    2
  )
)


# ==============================================================================
# PART XXVIII
#
# LINEAR REGRESSION BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Fit Linear Model
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = y_train,
  X_train
)


test_data <- data.frame(
  X_test
)


linear_model <- lm(
  y ~ .,
  data =
    train_data
)


linear_prediction <- predict(
  linear_model,
  newdata =
    test_data
)


linear_MSE <- MSE(
  y_test,
  linear_prediction
)


# ==============================================================================
# PART XXIX
#
# POLYNOMIAL REGRESSION BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 55. Additive Polynomial Model
# ------------------------------------------------------------------------------

polynomial_model <- lm(
  y ~
    poly(x1, 3) +
    poly(x2, 3) +
    poly(x3, 3) +
    poly(x4, 3) +
    poly(x5, 3),
  data =
    train_data
)


polynomial_prediction <- predict(
  polynomial_model,
  newdata =
    test_data
)


polynomial_MSE <- MSE(
  y_test,
  polynomial_prediction
)


# ==============================================================================
# PART XXX
#
# MODEL COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Compare Test Performance
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Additive Cubic Regression",
    "Neural Network"
  ),
  Test_MSE = c(
    linear_MSE,
    polynomial_MSE,
    MSE(
      y_test,
      test_prediction
    )
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XXXI
#
# NUMBER OF PARAMETERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 57. Count Network Parameters
# ------------------------------------------------------------------------------

count_parameters <- function(
    input_size,
    hidden_size
) {
  
  # W1:
  #
  #       p * M
  #
  # b1:
  #
  #       M
  #
  # W2:
  #
  #       M
  #
  # b2:
  #
  #       1
  
  input_size *
    hidden_size +
    hidden_size +
    hidden_size +
    1
}


parameter_count <- count_parameters(
  input_size = p,
  hidden_size =
    NN_model$hidden_size
)


parameter_count


# ==============================================================================
# PART XXXII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 58. Summary
# ------------------------------------------------------------------------------

cat(
  "Neural Network Summary\n"
)


cat(
  "----------------------\n"
)


cat(
  "Input variables:",
  p,
  "\n"
)


cat(
  "Hidden units:",
  NN_model$hidden_size,
  "\n"
)


cat(
  "Activation:",
  NN_model$activation,
  "\n"
)


cat(
  "Learning rate:",
  NN_model$learning_rate,
  "\n"
)


cat(
  "Weight decay:",
  NN_model$lambda,
  "\n"
)


cat(
  "Number of parameters:",
  parameter_count,
  "\n"
)


cat(
  "Best epoch:",
  NN_model$best_epoch,
  "\n"
)


cat(
  "Training MSE:",
  round(
    MSE(
      y_train,
      train_prediction
    ),
    4
  ),
  "\n"
)


cat(
  "Validation MSE:",
  round(
    MSE(
      y_validation,
      validation_prediction
    ),
    4
  ),
  "\n"
)


cat(
  "Test MSE:",
  round(
    MSE(
      y_test,
      test_prediction
    ),
    4
  ),
  "\n"
)


cat(
  "Test R-squared:",
  round(
    R_squared(
      y_test,
      test_prediction
    ),
    4
  ),
  "\n"
)