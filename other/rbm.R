# ==============================================================================
# Restricted Boltzmann Machines
# Jonathan Ma
#
# Package-free binary-binary Restricted Boltzmann Machine.
#
# Main ideas:
#   - Energy-based models
#   - Visible and hidden units
#   - Bipartite graphical models
#   - Bernoulli conditional distributions
#   - Gibbs sampling
#   - Positive phase
#   - Negative phase
#   - Contrastive Divergence CD-k
#   - Reconstruction error
#   - Hidden representations
#   - Generative sampling
#   - Free energy
#
#
# Binary RBM:
#
#       visible units:
#
#           v in {0,1}^D
#
#       hidden units:
#
#           h in {0,1}^H
#
#
# Energy:
#
#       E(v,h)
#
#       =
#
#       -a'v
#       -b'h
#       -v'Wh
#
#
# Joint distribution:
#
#       p(v,h)
#
#       =
#
#       exp(-E(v,h)) / Z
#
#
# where Z is the partition function:
#
#       Z = sum_v sum_h exp(-E(v,h))
#
#
# Because the graph is bipartite:
#
#       p(h_j = 1 | v)
#
#       =
#
#       sigmoid(
#           b_j + sum_i W_ij v_i
#       )
#
#
# and:
#
#       p(v_i = 1 | h)
#
#       =
#
#       sigmoid(
#           a_i + sum_j W_ij h_j
#       )
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE STRUCTURED BINARY DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# We create simple 6 x 6 binary images.
#
# There are four latent pattern types:
#
#   1. vertical bar on left
#   2. vertical bar on right
#   3. horizontal bar near top
#   4. horizontal bar near bottom
#
# Noise randomly flips pixels.
#
# The RBM is never given the pattern labels.
# ------------------------------------------------------------------------------

set.seed(123)


image_height <- 6

image_width <- 6

number_visible <- image_height *
  image_width


# ------------------------------------------------------------------------------
# Helper: Convert Matrix to Vector
# ------------------------------------------------------------------------------

matrix_to_vector <- function(x) {
  
  as.numeric(
    t(x)
  )
}


# ------------------------------------------------------------------------------
# Generate Prototype Patterns
# ------------------------------------------------------------------------------

make_prototype <- function(type) {
  
  image <- matrix(
    0,
    nrow = image_height,
    ncol = image_width
  )
  
  
  if (
    type ==
    1
  ) {
    
    image[
      ,
      2
    ] <- 1
    
    
  } else if (
    type ==
    2
  ) {
    
    image[
      ,
      5
    ] <- 1
    
    
  } else if (
    type ==
    3
  ) {
    
    image[
      2,
    ] <- 1
    
    
  } else if (
    type ==
    4
  ) {
    
    image[
      5,
    ] <- 1
  }
  
  
  matrix_to_vector(
    image
  )
}


prototypes <- t(
  sapply(
    1:4,
    make_prototype
  )
)


dim(
  prototypes
)


# ------------------------------------------------------------------------------
# Plot Prototype
# ------------------------------------------------------------------------------

plot_binary_image <- function(
    vector,
    height = image_height,
    width = image_width,
    main = ""
) {
  
  image_matrix <- matrix(
    vector,
    nrow = height,
    ncol = width,
    byrow = TRUE
  )
  
  
  image(
    t(
      image_matrix[
        nrow(image_matrix):1,
        ,
        drop = FALSE
      ]
    ),
    axes = FALSE,
    main = main
  )
}


par(
  mfrow = c(2, 2)
)


for (
  k in 1:4
) {
  
  plot_binary_image(
    prototypes[
      k,
    ],
    main = paste(
      "Prototype",
      k
    )
  )
}


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART II
#
# SAMPLE NOISY OBSERVATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# Flip each pixel independently with small probability.
# ------------------------------------------------------------------------------

generate_noisy_pattern <- function(
    prototype,
    flip_probability = 0.08
) {
  
  flips <- rbinom(
    length(
      prototype
    ),
    size = 1,
    prob =
      flip_probability
  )
  
  
  ifelse(
    flips ==
      1,
    1 -
      prototype,
    prototype
  )
}


# ------------------------------------------------------------------------------
# Data Set
# ------------------------------------------------------------------------------

number_samples <- 800


latent_class <- sample(
  1:4,
  size =
    number_samples,
  replace = TRUE
)


X <- matrix(
  0,
  nrow =
    number_samples,
  ncol =
    number_visible
)


for (
  i in seq_len(
    number_samples
  )
) {
  
  X[
    i,
  ] <- generate_noisy_pattern(
    prototypes[
      latent_class[i],
    ],
    flip_probability = 0.08
  )
}


colnames(
  X
) <- paste0(
  "Pixel_",
  seq_len(
    number_visible
  )
)


# ------------------------------------------------------------------------------
# View Random Training Examples
# ------------------------------------------------------------------------------

set.seed(456)


example_indices <- sample(
  seq_len(
    number_samples
  ),
  9
)


par(
  mfrow = c(3, 3)
)


for (
  index in example_indices
) {
  
  plot_binary_image(
    X[
      index,
    ],
    main = paste(
      "Class",
      latent_class[
        index
      ]
    )
  )
}


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART III
#
# TRAIN / TEST SPLIT
# ==============================================================================


set.seed(789)


train_indices <- sample(
  seq_len(
    number_samples
  ),
  size =
    floor(
      0.80 *
        number_samples
    )
)


test_indices <- setdiff(
  seq_len(
    number_samples
  ),
  train_indices
)


X_train <- X[
  train_indices,
  ,
  drop = FALSE
]


X_test <- X[
  test_indices,
  ,
  drop = FALSE
]


class_train <- latent_class[
  train_indices
]


class_test <- latent_class[
  test_indices
]


# ==============================================================================
# PART IV
#
# NUMERICAL UTILITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# Stable Sigmoid
# ------------------------------------------------------------------------------

sigmoid <- function(x) {
  
  result <- numeric(
    length(x)
  )
  
  
  positive <- x >=
    0
  
  
  result[
    positive
  ] <- 1 /
    (
      1 +
        exp(
          -x[
            positive
          ]
        )
    )
  
  
  exponential <- exp(
    x[
      !positive
    ]
  )
  
  
  result[
    !positive
  ] <- exponential /
    (
      1 +
        exponential
    )
  
  
  result
}


# ------------------------------------------------------------------------------
# Bernoulli Sampling
# ------------------------------------------------------------------------------

sample_bernoulli <- function(
    probability
) {
  
  matrix(
    rbinom(
      length(
        probability
      ),
      size = 1,
      prob =
        as.numeric(
          probability
        )
    ),
    nrow =
      nrow(
        probability
      ),
    ncol =
      ncol(
        probability
      )
  )
}


# ==============================================================================
# PART V
#
# RBM INITIALIZATION
# ==============================================================================


initialize_rbm <- function(
    number_visible,
    number_hidden,
    weight_sd = 0.01,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  W <- matrix(
    rnorm(
      number_visible *
        number_hidden,
      sd =
        weight_sd
    ),
    nrow =
      number_visible,
    ncol =
      number_hidden
  )
  
  
  visible_bias <- rep(
    0,
    number_visible
  )
  
  
  hidden_bias <- rep(
    0,
    number_hidden
  )
  
  
  list(
    W =
      W,
    visible_bias =
      visible_bias,
    hidden_bias =
      hidden_bias
  )
}


# ==============================================================================
# PART VI
#
# CONDITIONAL DISTRIBUTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# p(h = 1 | v)
#
#       sigmoid(
#           b + vW
#       )
# ------------------------------------------------------------------------------

hidden_probabilities <- function(
    visible,
    rbm
) {
  
  visible <- as.matrix(
    visible
  )
  
  
  linear_predictor <- sweep(
    visible %*%
      rbm$W,
    2,
    rbm$hidden_bias,
    "+"
  )
  
  
  matrix(
    sigmoid(
      as.numeric(
        linear_predictor
      )
    ),
    nrow =
      nrow(
        linear_predictor
      ),
    ncol =
      ncol(
        linear_predictor
      )
  )
}


# ------------------------------------------------------------------------------
# p(v = 1 | h)
#
#       sigmoid(
#           a + h W'
#       )
# ------------------------------------------------------------------------------

visible_probabilities <- function(
    hidden,
    rbm
) {
  
  hidden <- as.matrix(
    hidden
  )
  
  
  linear_predictor <- sweep(
    hidden %*%
      t(
        rbm$W
      ),
    2,
    rbm$visible_bias,
    "+"
  )
  
  
  matrix(
    sigmoid(
      as.numeric(
        linear_predictor
      )
    ),
    nrow =
      nrow(
        linear_predictor
      ),
    ncol =
      ncol(
        linear_predictor
      )
  )
}


# ==============================================================================
# PART VII
#
# ONE BLOCK GIBBS STEP
# ==============================================================================


gibbs_step <- function(
    visible,
    rbm
) {
  
  hidden_probability <- hidden_probabilities(
    visible,
    rbm
  )
  
  
  hidden_sample <- sample_bernoulli(
    hidden_probability
  )
  
  
  visible_probability <- visible_probabilities(
    hidden_sample,
    rbm
  )
  
  
  visible_sample <- sample_bernoulli(
    visible_probability
  )
  
  
  list(
    visible_probability =
      visible_probability,
    visible_sample =
      visible_sample,
    hidden_probability =
      hidden_probability,
    hidden_sample =
      hidden_sample
  )
}


# ==============================================================================
# PART VIII
#
# RBM ENERGY
# ==============================================================================


# ------------------------------------------------------------------------------
# Energy:
#
#       E(v,h)
#
#       =
#
#       -a'v
#       -b'h
#       -v'Wh
# ------------------------------------------------------------------------------

rbm_energy <- function(
    visible,
    hidden,
    rbm
) {
  
  visible <- as.numeric(
    visible
  )
  
  
  hidden <- as.numeric(
    hidden
  )
  
  
  -sum(
    rbm$visible_bias *
      visible
  ) -
    sum(
      rbm$hidden_bias *
        hidden
    ) -
    as.numeric(
      t(
        visible
      ) %*%
        rbm$W %*%
        hidden
    )
}


# ==============================================================================
# PART IX
#
# FREE ENERGY
# ==============================================================================


# ------------------------------------------------------------------------------
# Visible-vector free energy:
#
#       F(v)
#
#       =
#
#       -a'v
#
#       -
#
#       sum_j log(
#
#           1 + exp(
#
#               b_j + W_j'v
#
#           )
#
#       )
#
#
# Lower free energy means the RBM assigns relatively greater unnormalized
# probability to that visible vector.
# ------------------------------------------------------------------------------

softplus <- function(x) {
  
  pmax(
    x,
    0
  ) +
    log1p(
      exp(
        -abs(
          x
        )
      )
    )
}


free_energy <- function(
    visible,
    rbm
) {
  
  visible <- as.matrix(
    visible
  )
  
  
  hidden_linear <- sweep(
    visible %*%
      rbm$W,
    2,
    rbm$hidden_bias,
    "+"
  )
  
  
  visible_bias_term <- as.numeric(
    visible %*%
      rbm$visible_bias
  )
  
  
  hidden_term <- rowSums(
    matrix(
      softplus(
        as.numeric(
          hidden_linear
        )
      ),
      nrow =
        nrow(
          hidden_linear
        ),
      ncol =
        ncol(
          hidden_linear
        )
    )
  )
  
  
  -visible_bias_term -
    hidden_term
}


# ==============================================================================
# PART X
#
# RECONSTRUCTION
# ==============================================================================


# ------------------------------------------------------------------------------
# Deterministic one-step reconstruction probabilities:
#
#       v
#       ->
#       E[h|v]
#       ->
#       E[v|h]
# ------------------------------------------------------------------------------

reconstruct_rbm <- function(
    visible,
    rbm
) {
  
  hidden_probability <- hidden_probabilities(
    visible,
    rbm
  )
  
  
  visible_probability <- visible_probabilities(
    hidden_probability,
    rbm
  )
  
  
  visible_probability
}


# ------------------------------------------------------------------------------
# Reconstruction Cross-Entropy
# ------------------------------------------------------------------------------

binary_cross_entropy <- function(
    observed,
    probability,
    epsilon = 1e-10
) {
  
  probability <- pmin(
    pmax(
      probability,
      epsilon
    ),
    1 -
      epsilon
  )
  
  
  -mean(
    observed *
      log(
        probability
      ) +
      (
        1 -
          observed
      ) *
      log(
        1 -
          probability
      )
  )
}


# ==============================================================================
# PART XI
#
# CONTRASTIVE DIVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# Exact maximum-likelihood gradient:
#
#       d log p(v) / dW
#
#       =
#
#       E_data[v h']
#
#       -
#
#       E_model[v h']
#
#
# Positive phase:
#
#       correlations under observed data.
#
# Negative phase:
#
#       correlations under model distribution.
#
#
# The negative phase is expensive because exact model expectations require
# sampling from the equilibrium distribution.
#
# Contrastive Divergence approximates it by starting a short Gibbs chain at the
# observed data.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XII
#
# TRAIN RBM WITH MINI-BATCH CD-k
# ==============================================================================


train_rbm <- function(
    X,
    number_hidden = 8,
    epochs = 150,
    batch_size = 50,
    learning_rate = 0.05,
    cd_steps = 1,
    weight_decay = 0.0001,
    momentum = 0.5,
    seed = 123,
    verbose = TRUE
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  number_visible <- ncol(
    X
  )
  
  
  rbm <- initialize_rbm(
    number_visible =
      number_visible,
    number_hidden =
      number_hidden,
    seed =
      seed
  )
  
  
  velocity_W <- matrix(
    0,
    nrow =
      number_visible,
    ncol =
      number_hidden
  )
  
  
  velocity_visible_bias <- rep(
    0,
    number_visible
  )
  
  
  velocity_hidden_bias <- rep(
    0,
    number_hidden
  )
  
  
  reconstruction_history <- numeric(
    epochs
  )
  
  
  free_energy_history <- numeric(
    epochs
  )
  
  
  set.seed(
    seed
  )
  
  
  for (
    epoch in seq_len(
      epochs
    )
  ) {
    
    shuffled_indices <- sample(
      seq_len(
        n
      )
    )
    
    
    batches <- split(
      shuffled_indices,
      ceiling(
        seq_along(
          shuffled_indices
        ) /
          batch_size
      )
    )
    
    
    for (
      batch_indices in batches
    ) {
      
      v0 <- X[
        batch_indices,
        ,
        drop = FALSE
      ]
      
      
      actual_batch_size <- nrow(
        v0
      )
      
      
      # ======================================================================
      # POSITIVE PHASE
      # ======================================================================
      
      ph0 <- hidden_probabilities(
        v0,
        rbm
      )
      
      
      positive_association <- t(
        v0
      ) %*%
        ph0 /
        actual_batch_size
      
      
      # ======================================================================
      # START GIBBS CHAIN
      #
      # Sample h0 ~ p(h|v0).
      # ======================================================================
      
      h_sample <- sample_bernoulli(
        ph0
      )
      
      
      vk <- v0
      
      
      phk <- ph0
      
      
      # ======================================================================
      # CD-k GIBBS STEPS
      # ======================================================================
      
      for (
        step in seq_len(
          cd_steps
        )
      ) {
        
        pvk <- visible_probabilities(
          h_sample,
          rbm
        )
        
        
        vk <- sample_bernoulli(
          pvk
        )
        
        
        phk <- hidden_probabilities(
          vk,
          rbm
        )
        
        
        if (
          step <
          cd_steps
        ) {
          
          h_sample <- sample_bernoulli(
            phk
          )
        }
      }
      
      
      # ======================================================================
      # NEGATIVE PHASE
      # ======================================================================
      
      negative_association <- t(
        vk
      ) %*%
        phk /
        actual_batch_size
      
      
      # ======================================================================
      # GRADIENTS
      # ======================================================================
      
      gradient_W <- positive_association -
        negative_association -
        weight_decay *
        rbm$W
      
      
      gradient_visible_bias <- colMeans(
        v0 -
          vk
      )
      
      
      gradient_hidden_bias <- colMeans(
        ph0 -
          phk
      )
      
      
      # ======================================================================
      # MOMENTUM UPDATE
      # ======================================================================
      
      velocity_W <- momentum *
        velocity_W +
        learning_rate *
        gradient_W
      
      
      velocity_visible_bias <- momentum *
        velocity_visible_bias +
        learning_rate *
        gradient_visible_bias
      
      
      velocity_hidden_bias <- momentum *
        velocity_hidden_bias +
        learning_rate *
        gradient_hidden_bias
      
      
      rbm$W <- rbm$W +
        velocity_W
      
      
      rbm$visible_bias <- rbm$visible_bias +
        velocity_visible_bias
      
      
      rbm$hidden_bias <- rbm$hidden_bias +
        velocity_hidden_bias
    }
    
    
    # ========================================================================
    # TRAINING DIAGNOSTICS
    # ========================================================================
    
    reconstruction <- reconstruct_rbm(
      X,
      rbm
    )
    
    
    reconstruction_history[
      epoch
    ] <- binary_cross_entropy(
      X,
      reconstruction
    )
    
    
    free_energy_history[
      epoch
    ] <- mean(
      free_energy(
        X,
        rbm
      )
    )
    
    
    if (
      verbose &&
      (
        epoch ==
        1 ||
        epoch %% 10 ==
        0
      )
    ) {
      
      cat(
        "Epoch:",
        epoch,
        " Reconstruction CE:",
        round(
          reconstruction_history[
            epoch
          ],
          5
        ),
        " Mean free energy:",
        round(
          free_energy_history[
            epoch
          ],
          5
        ),
        "\n"
      )
    }
  }
  
  
  rbm$reconstruction_history <-
    reconstruction_history
  
  
  rbm$free_energy_history <-
    free_energy_history
  
  
  rbm$number_hidden <-
    number_hidden
  
  
  rbm
}


# ==============================================================================
# PART XIII
#
# TRAIN RBM
# ==============================================================================


rbm <- train_rbm(
  X_train,
  number_hidden = 8,
  epochs = 150,
  batch_size = 50,
  learning_rate = 0.05,
  cd_steps = 1,
  weight_decay = 0.0001,
  momentum = 0.5,
  seed = 123,
  verbose = TRUE
)


# ==============================================================================
# PART XIV
#
# TRAINING DIAGNOSTICS
# ==============================================================================


# ------------------------------------------------------------------------------
# Reconstruction Error
# ------------------------------------------------------------------------------

plot(
  rbm$reconstruction_history,
  type = "l",
  lwd = 2,
  xlab = "Epoch",
  ylab = "Binary Cross-Entropy",
  main = "RBM Reconstruction Error"
)


# ------------------------------------------------------------------------------
# Mean Free Energy
# ------------------------------------------------------------------------------

plot(
  rbm$free_energy_history,
  type = "l",
  lwd = 2,
  xlab = "Epoch",
  ylab = "Mean Free Energy",
  main = "Training-Data Free Energy"
)


# Important:
#
# Reconstruction error is useful diagnostically, but RBMs are probabilistic
# generative models. Lower reconstruction error is NOT identical to higher
# likelihood.


# ==============================================================================
# PART XV
#
# TRAIN AND TEST RECONSTRUCTION
# ==============================================================================


train_reconstruction <- reconstruct_rbm(
  X_train,
  rbm
)


test_reconstruction <- reconstruct_rbm(
  X_test,
  rbm
)


train_cross_entropy <- binary_cross_entropy(
  X_train,
  train_reconstruction
)


test_cross_entropy <- binary_cross_entropy(
  X_test,
  test_reconstruction
)


data.frame(
  Dataset = c(
    "Train",
    "Test"
  ),
  Reconstruction_Cross_Entropy = c(
    train_cross_entropy,
    test_cross_entropy
  )
)


# ==============================================================================
# PART XVI
#
# VISUALIZE RECONSTRUCTIONS
# ==============================================================================


set.seed(2024)


reconstruction_indices <- sample(
  seq_len(
    nrow(
      X_test
    )
  ),
  5
)


par(
  mfrow = c(
    2,
    5
  )
)


for (
  index in reconstruction_indices
) {
  
  plot_binary_image(
    X_test[
      index,
    ],
    main = "Observed"
  )
}


for (
  index in reconstruction_indices
) {
  
  plot_binary_image(
    test_reconstruction[
      index,
    ],
    main = "Reconstructed"
  )
}


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XVII
#
# HIDDEN REPRESENTATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# Hidden activation probabilities:
#
#       phi(x)
#
#       =
#
#       p(h=1 | x)
#
# These can be used as learned nonlinear features.
# ------------------------------------------------------------------------------

H_train <- hidden_probabilities(
  X_train,
  rbm
)


H_test <- hidden_probabilities(
  X_test,
  rbm
)


dim(
  H_train
)


summary(
  as.numeric(
    H_train
  )
)


# ==============================================================================
# PART XVIII
#
# VISUALIZE HIDDEN ACTIVATIONS
# ==============================================================================


image(
  t(
    H_train[
      1:100,
      ,
      drop = FALSE
    ]
  ),
  axes = FALSE,
  xlab = "Observation",
  ylab = "Hidden Unit",
  main = "Hidden Activation Probabilities"
)


# ==============================================================================
# PART XIX
#
# HIDDEN UNIT WEIGHTS AS RECEPTIVE FIELDS
# ==============================================================================


# ------------------------------------------------------------------------------
# Each hidden unit has a weight vector connecting it to visible pixels.
#
# We can visualize those weights as a 6x6 pattern.
# ------------------------------------------------------------------------------

plot_weight_image <- function(
    weight_vector,
    height = image_height,
    width = image_width,
    main = ""
) {
  
  weight_matrix <- matrix(
    weight_vector,
    nrow =
      height,
    ncol =
      width,
    byrow = TRUE
  )
  
  
  maximum_absolute <- max(
    abs(
      weight_matrix
    )
  )
  
  
  image(
    t(
      weight_matrix[
        nrow(
          weight_matrix
        ):1,
        ,
        drop = FALSE
      ]
    ),
    zlim = c(
      -maximum_absolute,
      maximum_absolute
    ),
    axes = FALSE,
    main = main
  )
}


par(
  mfrow = c(
    2,
    4
  )
)


for (
  hidden_unit in seq_len(
    rbm$number_hidden
  )
) {
  
  plot_weight_image(
    rbm$W[
      ,
      hidden_unit
    ],
    main = paste(
      "Hidden",
      hidden_unit
    )
  )
}


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XX
#
# DO HIDDEN FEATURES CAPTURE LATENT STRUCTURE?
# ==============================================================================


# ------------------------------------------------------------------------------
# The RBM never saw class labels.
#
# We now inspect average hidden activation by true simulation class.
# ------------------------------------------------------------------------------

hidden_means_by_class <- matrix(
  NA_real_,
  nrow = 4,
  ncol =
    rbm$number_hidden
)


for (
  class_value in 1:4
) {
  
  hidden_means_by_class[
    class_value,
  ] <- colMeans(
    H_train[
      class_train ==
        class_value,
      ,
      drop = FALSE
    ]
  )
}


rownames(
  hidden_means_by_class
) <- paste0(
  "Class_",
  1:4
)


colnames(
  hidden_means_by_class
) <- paste0(
  "Hidden_",
  seq_len(
    rbm$number_hidden
  )
)


round(
  hidden_means_by_class,
  3
)


image(
  t(
    hidden_means_by_class
  ),
  axes = FALSE,
  main = "Mean Hidden Activation by True Pattern"
)


axis(
  1,
  at = seq(
    0,
    1,
    length.out = 4
  ),
  labels =
    rownames(
      hidden_means_by_class
    )
)


axis(
  2,
  at = seq(
    0,
    1,
    length.out =
      rbm$number_hidden
  ),
  labels =
    colnames(
      hidden_means_by_class
    ),
  las = 2
)


# ==============================================================================
# PART XXI
#
# UNSUPERVISED FEATURES + SIMPLE CLASSIFIER
# ==============================================================================


# ------------------------------------------------------------------------------
# This is NOT part of RBM training.
#
# It illustrates how the learned representation can be used downstream.
#
# We use nearest-centroid classification in hidden space.
# ------------------------------------------------------------------------------

hidden_centroids <- matrix(
  NA_real_,
  nrow = 4,
  ncol =
    rbm$number_hidden
)


for (
  class_value in 1:4
) {
  
  hidden_centroids[
    class_value,
  ] <- colMeans(
    H_train[
      class_train ==
        class_value,
      ,
      drop = FALSE
    ]
  )
}


nearest_centroid_predict <- function(
    X,
    centroids
) {
  
  predictions <- integer(
    nrow(X)
  )
  
  
  for (
    i in seq_len(
      nrow(X)
    )
  ) {
    
    distances <- rowSums(
      (
        centroids -
          matrix(
            X[
              i,
            ],
            nrow =
              nrow(
                centroids
              ),
            ncol =
              ncol(
                centroids
              ),
            byrow = TRUE
          )
      )^2
    )
    
    
    predictions[i] <- which.min(
      distances
    )
  }
  
  
  predictions
}


hidden_class_prediction <- nearest_centroid_predict(
  H_test,
  hidden_centroids
)


mean(
  hidden_class_prediction ==
    class_test
)


table(
  Truth =
    class_test,
  Prediction =
    hidden_class_prediction
)


# ==============================================================================
# PART XXII
#
# COMPARE RAW PIXEL FEATURES
# ==============================================================================


raw_centroids <- matrix(
  NA_real_,
  nrow = 4,
  ncol =
    number_visible
)


for (
  class_value in 1:4
) {
  
  raw_centroids[
    class_value,
  ] <- colMeans(
    X_train[
      class_train ==
        class_value,
      ,
      drop = FALSE
    ]
  )
}


raw_class_prediction <- nearest_centroid_predict(
  X_test,
  raw_centroids
)


data.frame(
  Feature_Space = c(
    "Raw Pixels",
    "RBM Hidden Representation"
  ),
  Accuracy = c(
    mean(
      raw_class_prediction ==
        class_test
    ),
    mean(
      hidden_class_prediction ==
        class_test
    )
  )
)


# This comparison is only illustrative.
#
# The RBM was not optimized for classification.


# ==============================================================================
# PART XXIII
#
# GENERATIVE SAMPLING
# ==============================================================================


# ------------------------------------------------------------------------------
# To generate from an RBM:
#
#   1. initialize visible vector;
#   2. alternate h ~ p(h|v), v ~ p(v|h);
#   3. after sufficiently many Gibbs steps, treat v as approximate model sample.
#
#
# A short chain is not guaranteed to reach equilibrium.
# ------------------------------------------------------------------------------

sample_from_rbm <- function(
    rbm,
    number_samples = 16,
    burn_in = 500,
    steps_between_samples = 50,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  number_visible <- nrow(
    rbm$W
  )
  
  
  visible <- matrix(
    rbinom(
      number_visible,
      size = 1,
      prob = 0.5
    ),
    nrow = 1
  )
  
  
  # --------------------------------------------------------------------------
  # Burn-In
  # --------------------------------------------------------------------------
  
  for (
    step in seq_len(
      burn_in
    )
  ) {
    
    state <- gibbs_step(
      visible,
      rbm
    )
    
    
    visible <- state$visible_sample
  }
  
  
  samples <- matrix(
    0,
    nrow =
      number_samples,
    ncol =
      number_visible
  )
  
  
  # --------------------------------------------------------------------------
  # Collect Samples
  # --------------------------------------------------------------------------
  
  for (
    sample_index in seq_len(
      number_samples
    )
  ) {
    
    for (
      step in seq_len(
        steps_between_samples
      )
    ) {
      
      state <- gibbs_step(
        visible,
        rbm
      )
      
      
      visible <- state$visible_sample
    }
    
    
    samples[
      sample_index,
    ] <- visible
  }
  
  
  samples
}


# ------------------------------------------------------------------------------
# Generate
# ------------------------------------------------------------------------------

generated_samples <- sample_from_rbm(
  rbm,
  number_samples = 16,
  burn_in = 1000,
  steps_between_samples = 100,
  seed = 2025
)


# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

par(
  mfrow = c(
    4,
    4
  )
)


for (
  i in seq_len(
    nrow(
      generated_samples
    )
  )
) {
  
  plot_binary_image(
    generated_samples[
      i,
    ],
    main = paste(
      "Sample",
      i
    )
  )
}


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XXIV
#
# GIBBS CHAIN EVOLUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# Watch one chain evolve from random noise.
# ------------------------------------------------------------------------------

set.seed(77)


visible <- matrix(
  rbinom(
    number_visible,
    size = 1,
    prob = 0.5
  ),
  nrow = 1
)


chain_steps <- c(
  0,
  1,
  5,
  20,
  100,
  500
)


chain_states <- matrix(
  NA_real_,
  nrow =
    length(
      chain_steps
    ),
  ncol =
    number_visible
)


chain_states[
  1,
] <- visible


current_step <- 0


for (
  target_index in 2:length(
    chain_steps
  )
) {
  
  target_step <- chain_steps[
    target_index
  ]
  
  
  while (
    current_step <
    target_step
  ) {
    
    state <- gibbs_step(
      visible,
      rbm
    )
    
    
    visible <- state$visible_sample
    
    
    current_step <- current_step +
      1
  }
  
  
  chain_states[
    target_index,
  ] <- visible
}


par(
  mfrow = c(
    2,
    3
  )
)


for (
  i in seq_along(
    chain_steps
  )
) {
  
  plot_binary_image(
    chain_states[
      i,
    ],
    main = paste(
      "Step",
      chain_steps[i]
    )
  )
}


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XXV
#
# FREE ENERGY OF DATA VS RANDOM NOISE
# ==============================================================================


# ------------------------------------------------------------------------------
# Well-trained RBMs should often assign relatively lower free energy to
# structured data than to arbitrary random binary vectors.
# ------------------------------------------------------------------------------

set.seed(888)


random_binary_data <- matrix(
  rbinom(
    nrow(
      X_test
    ) *
      number_visible,
    size = 1,
    prob = 0.5
  ),
  nrow =
    nrow(
      X_test
    ),
  ncol =
    number_visible
)


data_free_energy <- free_energy(
  X_test,
  rbm
)


random_free_energy <- free_energy(
  random_binary_data,
  rbm
)


data.frame(
  Dataset = c(
    "Structured Test Data",
    "Random Binary Noise"
  ),
  Mean_Free_Energy = c(
    mean(
      data_free_energy
    ),
    mean(
      random_free_energy
    )
  )
)


boxplot(
  data_free_energy,
  random_free_energy,
  names = c(
    "Data",
    "Noise"
  ),
  ylab = "Free Energy",
  main = "RBM Free Energy"
)


# ==============================================================================
# PART XXVI
#
# NUMBER OF HIDDEN UNITS
# ==============================================================================


# ------------------------------------------------------------------------------
# Compare model capacity.
#
# More hidden units can represent richer dependencies but also add parameters.
# ------------------------------------------------------------------------------

hidden_values <- c(
  2,
  4,
  8,
  16
)


hidden_results <- data.frame(
  Hidden_Units =
    hidden_values,
  Train_CE =
    NA_real_,
  Test_CE =
    NA_real_
)


hidden_models <- vector(
  "list",
  length(
    hidden_values
  )
)


for (
  index in seq_along(
    hidden_values
  )
) {
  
  fit <- train_rbm(
    X_train,
    number_hidden =
      hidden_values[
        index
      ],
    epochs = 80,
    batch_size = 50,
    learning_rate = 0.05,
    cd_steps = 1,
    weight_decay = 0.0001,
    momentum = 0.5,
    seed =
      100 +
      index,
    verbose = FALSE
  )
  
  
  hidden_models[[index]] <- fit
  
  
  train_reconstruction_i <- reconstruct_rbm(
    X_train,
    fit
  )
  
  
  test_reconstruction_i <- reconstruct_rbm(
    X_test,
    fit
  )
  
  
  hidden_results$Train_CE[index] <-
    binary_cross_entropy(
      X_train,
      train_reconstruction_i
    )
  
  
  hidden_results$Test_CE[index] <-
    binary_cross_entropy(
      X_test,
      test_reconstruction_i
    )
}


hidden_results


plot(
  hidden_results$Hidden_Units,
  hidden_results$Train_CE,
  type = "b",
  pch = 19,
  ylim = range(
    c(
      hidden_results$Train_CE,
      hidden_results$Test_CE
    )
  ),
  xlab = "Hidden Units",
  ylab = "Reconstruction Cross-Entropy",
  main = "RBM Capacity"
)


lines(
  hidden_results$Hidden_Units,
  hidden_results$Test_CE,
  type = "b",
  pch = 19,
  lty = 2
)


legend(
  "topright",
  legend = c(
    "Train",
    "Test"
  ),
  lty = c(
    1,
    2
  ),
  pch = 19
)


# ==============================================================================
# PART XXVII
#
# CD-1 VS CD-k
# ==============================================================================


# ------------------------------------------------------------------------------
# Larger k runs the negative-phase chain longer.
#
# In principle this moves the negative sample closer to the model distribution.
#
# But computation becomes more expensive.
# ------------------------------------------------------------------------------

cd_values <- c(
  1,
  3,
  5
)


cd_results <- data.frame(
  CD_Steps =
    cd_values,
  Train_CE =
    NA_real_,
  Test_CE =
    NA_real_
)


for (
  index in seq_along(
    cd_values
  )
) {
  
  fit <- train_rbm(
    X_train,
    number_hidden = 8,
    epochs = 80,
    batch_size = 50,
    learning_rate = 0.05,
    cd_steps =
      cd_values[
        index
      ],
    weight_decay = 0.0001,
    momentum = 0.5,
    seed =
      200 +
      index,
    verbose = FALSE
  )
  
  
  train_probability <- reconstruct_rbm(
    X_train,
    fit
  )
  
  
  test_probability <- reconstruct_rbm(
    X_test,
    fit
  )
  
  
  cd_results$Train_CE[index] <-
    binary_cross_entropy(
      X_train,
      train_probability
    )
  
  
  cd_results$Test_CE[index] <-
    binary_cross_entropy(
      X_test,
      test_probability
    )
}


cd_results


# ==============================================================================
# PART XXVIII
#
# LEARNING RATE COMPARISON
# ==============================================================================


learning_rates <- c(
  0.005,
  0.02,
  0.05,
  0.10
)


learning_rate_results <- data.frame(
  Learning_Rate =
    learning_rates,
  Final_Train_CE =
    NA_real_
)


for (
  index in seq_along(
    learning_rates
  )
) {
  
  fit <- train_rbm(
    X_train,
    number_hidden = 8,
    epochs = 60,
    batch_size = 50,
    learning_rate =
      learning_rates[
        index
      ],
    cd_steps = 1,
    weight_decay = 0.0001,
    momentum = 0.5,
    seed = 500,
    verbose = FALSE
  )
  
  
  learning_rate_results$Final_Train_CE[
    index
  ] <- tail(
    fit$reconstruction_history,
    1
  )
}


learning_rate_results


# ==============================================================================
# PART XXIX
#
# EXACT PARTITION FUNCTION FOR A TINY RBM
# ==============================================================================


# ------------------------------------------------------------------------------
# This is only feasible for VERY small visible dimension.
#
# For D visible units there are:
#
#       2^D
#
# possible visible states.
#
# For a normal image RBM this is impossible.
#
# We demonstrate exact normalization with D = 4.
# ------------------------------------------------------------------------------

enumerate_binary_states <- function(
    dimension
) {
  
  integers <- 0:(
    2^dimension -
      1
  )
  
  
  states <- matrix(
    0,
    nrow =
      length(
        integers
      ),
    ncol =
      dimension
  )
  
  
  for (
    column in seq_len(
      dimension
    )
  ) {
    
    states[
      ,
      column
    ] <- bitwAnd(
      bitwShiftR(
        integers,
        column -
          1
      ),
      1
    )
  }
  
  
  states
}


# ------------------------------------------------------------------------------
# Tiny Random RBM
# ------------------------------------------------------------------------------

tiny_rbm <- initialize_rbm(
  number_visible = 4,
  number_hidden = 2,
  seed = 999
)


tiny_visible_states <- enumerate_binary_states(
  4
)


tiny_hidden_states <- enumerate_binary_states(
  2
)


# ------------------------------------------------------------------------------
# Exact Partition Function
# ------------------------------------------------------------------------------

compute_exact_partition_function <- function(
    rbm
) {
  
  visible_states <- enumerate_binary_states(
    nrow(
      rbm$W
    )
  )
  
  
  hidden_states <- enumerate_binary_states(
    ncol(
      rbm$W
    )
  )
  
  
  total <- 0
  
  
  for (
    visible_index in seq_len(
      nrow(
        visible_states
      )
    )
  ) {
    
    for (
      hidden_index in seq_len(
        nrow(
          hidden_states
        )
      )
    ) {
      
      energy <- rbm_energy(
        visible_states[
          visible_index,
        ],
        hidden_states[
          hidden_index,
        ],
        rbm
      )
      
      
      total <- total +
        exp(
          -energy
        )
    }
  }
  
  
  total
}


tiny_Z <- compute_exact_partition_function(
  tiny_rbm
)


tiny_Z


# ==============================================================================
# PART XXX
#
# EXACT VISIBLE PROBABILITIES FOR TINY RBM
# ==============================================================================


exact_visible_probability <- function(
    visible,
    rbm,
    partition_function
) {
  
  hidden_states <- enumerate_binary_states(
    ncol(
      rbm$W
    )
  )
  
  
  numerator <- 0
  
  
  for (
    hidden_index in seq_len(
      nrow(
        hidden_states
      )
    )
  ) {
    
    energy <- rbm_energy(
      visible,
      hidden_states[
        hidden_index,
      ],
      rbm
    )
    
    
    numerator <- numerator +
      exp(
        -energy
      )
  }
  
  
  numerator /
    partition_function
}


tiny_visible_probabilities <- numeric(
  nrow(
    tiny_visible_states
  )
)


for (
  i in seq_len(
    nrow(
      tiny_visible_states
    )
  )
) {
  
  tiny_visible_probabilities[i] <-
    exact_visible_probability(
      tiny_visible_states[
        i,
      ],
      tiny_rbm,
      tiny_Z
    )
}


sum(
  tiny_visible_probabilities
)


# This should be numerically equal to one.


# ==============================================================================
# PART XXXI
#
# VERIFY FREE-ENERGY PROBABILITY RELATION
# ==============================================================================


# ------------------------------------------------------------------------------
# For visible state v:
#
#       p(v)
#
#       =
#
#       exp(-F(v)) / Z
#
# ------------------------------------------------------------------------------

free_energy_probability <- exp(
  -free_energy(
    tiny_visible_states,
    tiny_rbm
  )
) /
  tiny_Z


max(
  abs(
    tiny_visible_probabilities -
      free_energy_probability
  )
)


# ==============================================================================
# PART XXXII
#
# RECONSTRUCTION IS NOT LIKELIHOOD
# ==============================================================================


# ------------------------------------------------------------------------------
# Important conceptual point:
#
# An RBM is trained as a generative probabilistic model.
#
# Reconstruction error is convenient but it is NOT:
#
#       - the log likelihood;
#       - the partition function;
#       - an exact measure of model probability.
#
#
# Exact log likelihood would require:
#
#       log p(v)
#
#       =
#
#       -F(v)
#       -
#       log Z
#
#
# Computing Z exactly scales exponentially with the number of units.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XXXIII
#
# ANOMALY-SCORE ILLUSTRATION
# ==============================================================================


# ------------------------------------------------------------------------------
# Free energy can be used as a relative scoring function:
#
#       lower free energy
#       ->
#       generally greater model preference.
#
# This is NOT automatically a calibrated anomaly probability.
# ------------------------------------------------------------------------------

test_scores <- free_energy(
  X_test,
  rbm
)


noise_scores <- free_energy(
  random_binary_data,
  rbm
)


threshold <- quantile(
  test_scores,
  0.95
)


mean(
  test_scores >
    threshold
)


mean(
  noise_scores >
    threshold
)


# ==============================================================================
# PART XXXIV
#
# PERSISTENT CHAIN CONCEPT
# ==============================================================================


# ------------------------------------------------------------------------------
# Contrastive Divergence:
#
#       initialize negative chain at DATA each update.
#
#
# Persistent Contrastive Divergence:
#
#       maintain model chains across optimization updates.
#
#
# PCD can approximate model equilibrium expectations better than very short
# CD chains in some settings.
#
# We do not implement full PCD here because the main goal is the standard
# CD-k RBM mechanism.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XXXV
#
# RBM VS AUTOENCODER
# ==============================================================================


# ------------------------------------------------------------------------------
# Both can learn hidden representations and reconstruct inputs.
#
# But their probabilistic meanings differ:
#
#
# AUTOENCODER
#
#       x -> encoder -> z -> decoder -> x_hat
#
#       directly minimizes reconstruction loss.
#
#
# RBM
#
#       defines joint probability:
#
#           p(v,h) proportional to exp(-E(v,h))
#
#       and aims to increase likelihood of observed data.
#
#
# Reconstruction arises from conditional distributions but is not the defining
# objective of the RBM.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XXXVI
#
# RBM VS PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# PCA:
#
#       deterministic linear latent representation.
#
#
# RBM:
#
#       stochastic nonlinear latent representation.
#
#
# PCA score:
#
#       z = Xv
#
#
# RBM hidden probability:
#
#       p(h_j=1|v)
#
#       =
#
#       sigmoid(
#           b_j + W_j'v
#       )
#
#
# Thus hidden features are nonlinear sigmoidal functions of the visible data.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XXXVII
#
# RBM VS NEURAL NETWORK
# ==============================================================================


# ------------------------------------------------------------------------------
# A standard feedforward neural network has directed computations:
#
#       input -> hidden -> output
#
#
# An RBM is an undirected graphical model:
#
#       visible <-> hidden
#
#
# The same weights participate in both:
#
#       p(h|v)
#
# and
#
#       p(v|h).
#
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XXXVIII
#
# DEEP BELIEF NETWORK CONNECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# Historically RBMs were used as building blocks for Deep Belief Networks.
#
# One could:
#
#       1. train RBM on raw inputs;
#       2. use hidden activations as new data;
#       3. train another RBM on those activations;
#       4. stack layers;
#       5. optionally fine-tune supervised network.
#
#
# This unsupervised layer-wise pretraining was especially important before
# modern initialization, normalization, optimizers, and very large labeled
# datasets made deep feedforward training easier.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART XXXIX
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nRestricted Boltzmann Machine Summary\n"
)


cat(
  "------------------------------------\n"
)


cat(
  "Training observations:",
  nrow(
    X_train
  ),
  "\n"
)


cat(
  "Visible units:",
  ncol(
    X_train
  ),
  "\n"
)


cat(
  "Hidden units:",
  rbm$number_hidden,
  "\n"
)


cat(
  "Train reconstruction CE:",
  round(
    train_cross_entropy,
    5
  ),
  "\n"
)


cat(
  "Test reconstruction CE:",
  round(
    test_cross_entropy,
    5
  ),
  "\n"
)


cat(
  "Mean test free energy:",
  round(
    mean(
      data_free_energy
    ),
    5
  ),
  "\n"
)


cat(
  "Mean random-noise free energy:",
  round(
    mean(
      random_free_energy
    ),
    5
  ),
  "\n"
)


cat(
  "Hidden-feature centroid accuracy:",
  round(
    mean(
      hidden_class_prediction ==
        class_test
    ),
    4
  ),
  "\n"
)