# ==============================================================================
# Supervised Principal Components
# Jonathan Ma
#
# Package-free pedagogical implementation.
#
# Main ideas:
#   - Principal Components
#   - Supervised dimension reduction
#   - Marginal screening
#   - Threshold selection
#   - PCA after supervised screening
#   - Regression on supervised PC scores
#   - Cross-validation
#   - High-dimensional p > n settings
#   - Ordinary PCA vs supervised PCA
#   - Variable selection vs dimension reduction
#   - Prediction
#
#
# Ordinary PCA:
#
#       X
#       ->
#       directions explaining predictor variance
#
#
# Supervised PCA:
#
#       X, y
#       ->
#       screen variables using association with y
#       ->
#       PCA on selected predictors
#       ->
#       regression using supervised PC scores
#
#
# A simple marginal score:
#
#       s_j = |cor(X_j, y)|
#
#
# Select:
#
#       S(theta)
#
#       =
#
#       {j : s_j >= theta}
#
#
# Then perform PCA on:
#
#       X_S
#
#
# and fit:
#
#       y
#
#       =
#
#       beta_0
#
#       +
#
#       gamma_1 PC_1
#
#       +
#
#       ...
#
#       +
#
#       gamma_M PC_M
#
#       +
#
#       epsilon
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE HIGH-DIMENSIONAL DATA
# ==============================================================================


set.seed(123)


n <- 250

p <- 120


# ------------------------------------------------------------------------------
# We deliberately construct several latent factors.
#
# Some factors are highly variable but unrelated to y.
#
# Another lower-variance factor is predictive of y.
#
# This creates a setting where ordinary PCA can focus on the wrong directions.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 1. Latent Factors
# ------------------------------------------------------------------------------

z_noise_1 <- rnorm(
  n,
  sd = 4
)


z_noise_2 <- rnorm(
  n,
  sd = 3
)


z_signal_1 <- rnorm(
  n,
  sd = 1
)


z_signal_2 <- rnorm(
  n,
  sd = 0.8
)


# ==============================================================================
# PART II
#
# BUILD PREDICTORS
# ==============================================================================


X <- matrix(
  rnorm(
    n * p,
    sd = 0.5
  ),
  nrow = n,
  ncol = p
)


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# Variables 1:30:
#
# High-variance nuisance factor.
# ------------------------------------------------------------------------------

for (
  j in 1:30
) {
  
  X[
    ,
    j
  ] <- 1.2 *
    z_noise_1 +
    rnorm(
      n,
      sd = 0.5
    )
}


# ------------------------------------------------------------------------------
# Variables 31:60:
#
# Another high-variance nuisance factor.
# ------------------------------------------------------------------------------

for (
  j in 31:60
) {
  
  X[
    ,
    j
  ] <- 1.0 *
    z_noise_2 +
    rnorm(
      n,
      sd = 0.5
    )
}


# ------------------------------------------------------------------------------
# Variables 61:70:
#
# Predictive factor 1.
# ------------------------------------------------------------------------------

for (
  j in 61:70
) {
  
  X[
    ,
    j
  ] <- z_signal_1 +
    rnorm(
      n,
      sd = 0.5
    )
}


# ------------------------------------------------------------------------------
# Variables 71:80:
#
# Predictive factor 2.
# ------------------------------------------------------------------------------

for (
  j in 71:80
) {
  
  X[
    ,
    j
  ] <- z_signal_2 +
    rnorm(
      n,
      sd = 0.5
    )
}


# ------------------------------------------------------------------------------
# Variables 81:120:
#
# Independent noise.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART III
#
# RESPONSE
# ==============================================================================


beta_0_true <- 2


y <- beta_0_true +
  3 *
  z_signal_1 -
  2.5 *
  z_signal_2 +
  rnorm(
    n,
    sd = 1.5
  )


# ------------------------------------------------------------------------------
# True signal variables.
# ------------------------------------------------------------------------------

true_signal_variables <- 61:80


# ==============================================================================
# PART IV
#
# TRAIN / TEST SPLIT
# ==============================================================================


set.seed(456)


train_indices <- sample(
  seq_len(n),
  size = floor(
    0.70 *
      n
  )
)


test_indices <- setdiff(
  seq_len(n),
  train_indices
)


X_train_raw <- X[
  train_indices,
  ,
  drop = FALSE
]


X_test_raw <- X[
  test_indices,
  ,
  drop = FALSE
]


y_train <- y[
  train_indices
]


y_test <- y[
  test_indices
]


n_train <- nrow(
  X_train_raw
)


# ==============================================================================
# PART V
#
# STANDARDIZE USING TRAINING DATA
# ==============================================================================


x_means <- colMeans(
  X_train_raw
)


x_sds <- apply(
  X_train_raw,
  2,
  sd
)


x_sds[
  x_sds <
    1e-12
] <- 1


X_train <- sweep(
  X_train_raw,
  2,
  x_means,
  "-"
)


X_train <- sweep(
  X_train,
  2,
  x_sds,
  "/"
)


X_test <- sweep(
  X_test_raw,
  2,
  x_means,
  "-"
)


X_test <- sweep(
  X_test,
  2,
  x_sds,
  "/"
)


y_mean <- mean(
  y_train
)


y_train_centered <- y_train -
  y_mean


# ==============================================================================
# PART VI
#
# REGRESSION METRICS
# ==============================================================================


regression_metrics <- function(
    observed,
    predicted
) {
  
  mse <- mean(
    (
      observed -
        predicted
    )^2
  )
  
  
  rmse <- sqrt(
    mse
  )
  
  
  mae <- mean(
    abs(
      observed -
        predicted
    )
  )
  
  
  r_squared <- 1 -
    sum(
      (
        observed -
          predicted
      )^2
    ) /
    sum(
      (
        observed -
          mean(
            observed
          )
      )^2
    )
  
  
  c(
    MSE = mse,
    RMSE = rmse,
    MAE = mae,
    R2 = r_squared
  )
}


# ==============================================================================
# PART VII
#
# ORDINARY PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# PCA ignores y entirely.
# ------------------------------------------------------------------------------

ordinary_svd <- svd(
  X_train
)


ordinary_scores_train <- ordinary_svd$u %*%
  diag(
    ordinary_svd$d
  )


ordinary_loadings <- ordinary_svd$v


ordinary_scores_test <- X_test %*%
  ordinary_loadings


# ------------------------------------------------------------------------------
# Explained Variance
# ------------------------------------------------------------------------------

ordinary_eigenvalues <- ordinary_svd$d^2 /
  (
    n_train -
      1
  )


ordinary_variance_proportion <- ordinary_eigenvalues /
  sum(
    ordinary_eigenvalues
  )


ordinary_cumulative_variance <- cumsum(
  ordinary_variance_proportion
)


# ==============================================================================
# PART VIII
#
# ORDINARY PCA SCREE PLOT
# ==============================================================================


plot(
  ordinary_variance_proportion[
    1:20
  ],
  type = "b",
  pch = 19,
  xlab = "Principal Component",
  ylab = "Proportion Variance Explained",
  main = "Ordinary PCA"
)


# ==============================================================================
# PART IX
#
# ARE THE FIRST PCs PREDICTIVE?
# ==============================================================================


# ------------------------------------------------------------------------------
# Correlation between PC scores and response.
# ------------------------------------------------------------------------------

ordinary_pc_response_correlation <- sapply(
  seq_len(
    min(
      20,
      ncol(
        ordinary_scores_train
      )
    )
  ),
  function(component) {
    
    cor(
      ordinary_scores_train[
        ,
        component
      ],
      y_train
    )
  }
)


data.frame(
  PC =
    seq_along(
      ordinary_pc_response_correlation
    ),
  Variance_Explained =
    ordinary_variance_proportion[
      seq_along(
        ordinary_pc_response_correlation
      )
    ],
  Correlation_With_Y =
    ordinary_pc_response_correlation
)


# ==============================================================================
# PART X
#
# MARGINAL SUPERVISION SCORES
# ==============================================================================


# ------------------------------------------------------------------------------
# Simplest supervised score:
#
#       s_j = |cor(X_j, y)|
#
# Because X is standardized, this is closely related to the univariate
# regression score statistic.
# ------------------------------------------------------------------------------

supervision_scores <- numeric(
  p
)


for (
  j in seq_len(p)
) {
  
  supervision_scores[j] <- abs(
    cor(
      X_train[
        ,
        j
      ],
      y_train
    )
  )
}


# ------------------------------------------------------------------------------
# Rank Variables
# ------------------------------------------------------------------------------

supervision_order <- order(
  supervision_scores,
  decreasing = TRUE
)


head(
  data.frame(
    Variable =
      colnames(
        X_train
      )[
        supervision_order
      ],
    Score =
      supervision_scores[
        supervision_order
      ]
  ),
  30
)


# ==============================================================================
# PART XI
#
# VISUALIZE SUPERVISION SCORES
# ==============================================================================


plot(
  seq_len(p),
  supervision_scores,
  type = "h",
  lwd = 2,
  xlab = "Predictor",
  ylab = "|Correlation with Y|",
  main = "Supervised Screening Scores"
)


abline(
  v = c(
    60.5,
    80.5
  ),
  lty = 2
)


# True signal lies in variables 61:80.


# ==============================================================================
# PART XII
#
# DEFINE SUPERVISED PCA FIT
# ==============================================================================


fit_supervised_pca <- function(
    X,
    y,
    threshold,
    number_components = 1
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  # --------------------------------------------------------------------------
  # Marginal supervision scores.
  # --------------------------------------------------------------------------
  
  scores <- numeric(
    p
  )
  
  
  for (
    j in seq_len(p)
  ) {
    
    if (
      sd(
        X[
          ,
          j
        ]
      ) <
      1e-12
    ) {
      
      scores[j] <- 0
      
    } else {
      
      scores[j] <- abs(
        cor(
          X[
            ,
            j
          ],
          y
        )
      )
    }
  }
  
  
  selected <- which(
    scores >=
      threshold
  )
  
  
  # --------------------------------------------------------------------------
  # Ensure at least one predictor.
  # --------------------------------------------------------------------------
  
  if (
    length(
      selected
    ) ==
    0
  ) {
    
    selected <- which.max(
      scores
    )
  }
  
  
  X_selected <- X[
    ,
    selected,
    drop = FALSE
  ]
  
  
  # --------------------------------------------------------------------------
  # PCA via SVD.
  # --------------------------------------------------------------------------
  
  decomposition <- svd(
    X_selected
  )
  
  
  maximum_components <- min(
    number_components,
    ncol(
      decomposition$v
    )
  )
  
  
  loadings <- decomposition$v[
    ,
    seq_len(
      maximum_components
    ),
    drop = FALSE
  ]
  
  
  pc_scores <- X_selected %*%
    loadings
  
  
  # --------------------------------------------------------------------------
  # Regress centered y on selected PC scores.
  # --------------------------------------------------------------------------
  
  y_mean <- mean(
    y
  )
  
  
  y_centered <- y -
    y_mean
  
  
  pc_coefficients <- solve(
    crossprod(
      pc_scores
    ),
    crossprod(
      pc_scores,
      y_centered
    )
  )
  
  
  list(
    selected =
      selected,
    supervision_scores =
      scores,
    loadings =
      loadings,
    pc_coefficients =
      as.numeric(
        pc_coefficients
      ),
    y_mean =
      y_mean,
    number_components =
      maximum_components
  )
}


# ==============================================================================
# PART XIII
#
# PREDICTION FUNCTION
# ==============================================================================


predict_supervised_pca <- function(
    fit,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  X_selected <- X_new[
    ,
    fit$selected,
    drop = FALSE
  ]
  
  
  pc_scores <- X_selected %*%
    fit$loadings
  
  
  fit$y_mean +
    as.numeric(
      pc_scores %*%
        fit$pc_coefficients
    )
}


# ==============================================================================
# PART XIV
#
# INITIAL SUPERVISED PCA FIT
# ==============================================================================


initial_threshold <- 0.15


initial_spc <- fit_supervised_pca(
  X_train,
  y_train,
  threshold =
    initial_threshold,
  number_components = 2
)


length(
  initial_spc$selected
)


colnames(
  X_train
)[
  initial_spc$selected
]


# ==============================================================================
# PART XV
#
# INITIAL PREDICTION
# ==============================================================================


initial_spc_prediction <- predict_supervised_pca(
  initial_spc,
  X_test
)


regression_metrics(
  y_test,
  initial_spc_prediction
)


# ==============================================================================
# PART XVI
#
# VISUALIZE SUPERVISED PC SCORES
# ==============================================================================


initial_selected_train <- X_train[
  ,
  initial_spc$selected,
  drop = FALSE
]


initial_score_matrix <- initial_selected_train %*%
  initial_spc$loadings


if (
  ncol(
    initial_score_matrix
  ) >=
  2
) {
  
  plot(
    initial_score_matrix[
      ,
      1
    ],
    initial_score_matrix[
      ,
      2
    ],
    pch = 19,
    cex = 0.7,
    xlab = "Supervised PC1",
    ylab = "Supervised PC2",
    main = "Supervised Principal Component Scores"
  )
}


# ==============================================================================
# PART XVII
#
# CORRELATION OF SUPERVISED PCs WITH Y
# ==============================================================================


spc_response_correlations <- apply(
  initial_score_matrix,
  2,
  function(score) {
    
    cor(
      score,
      y_train
    )
  }
)


spc_response_correlations


# ==============================================================================
# PART XVIII
#
# SUPERVISED LOADINGS
# ==============================================================================


# ------------------------------------------------------------------------------
# Map supervised loadings back into full p-dimensional space.
# ------------------------------------------------------------------------------

full_supervised_loadings <- matrix(
  0,
  nrow = p,
  ncol =
    initial_spc$number_components
)


full_supervised_loadings[
  initial_spc$selected,
] <- initial_spc$loadings


matplot(
  seq_len(p),
  full_supervised_loadings,
  type = "h",
  lty = 1,
  lwd = 2,
  xlab = "Predictor",
  ylab = "Loading",
  main = "Supervised Principal Component Loadings"
)


# ==============================================================================
# PART XIX
#
# ORDINARY PCR
# ==============================================================================


fit_pcr <- function(
    X,
    y,
    number_components
) {
  
  decomposition <- svd(
    X
  )
  
  
  number_components <- min(
    number_components,
    ncol(
      decomposition$v
    )
  )
  
  
  loadings <- decomposition$v[
    ,
    seq_len(
      number_components
    ),
    drop = FALSE
  ]
  
  
  scores <- X %*%
    loadings
  
  
  y_mean <- mean(
    y
  )
  
  
  coefficients <- solve(
    crossprod(
      scores
    ),
    crossprod(
      scores,
      y -
        y_mean
    )
  )
  
  
  list(
    loadings =
      loadings,
    coefficients =
      as.numeric(
        coefficients
      ),
    y_mean =
      y_mean
  )
}


predict_pcr <- function(
    fit,
    X
) {
  
  scores <- X %*%
    fit$loadings
  
  
  fit$y_mean +
    as.numeric(
      scores %*%
        fit$coefficients
    )
}


# ==============================================================================
# PART XX
#
# ORDINARY PCR PERFORMANCE
# ==============================================================================


pcr_2 <- fit_pcr(
  X_train,
  y_train,
  number_components = 2
)


pcr_2_prediction <- predict_pcr(
  pcr_2,
  X_test
)


regression_metrics(
  y_test,
  pcr_2_prediction
)


# ==============================================================================
# PART XXI
#
# WHY ORDINARY PCA CAN FAIL
# ==============================================================================


# ------------------------------------------------------------------------------
# Examine first ordinary PC loadings.
#
# If the large-variance nuisance factors dominate, the first PCs should place
# most of their loading weight among X1:X60.
# ------------------------------------------------------------------------------

ordinary_pc1_loading_squared <- ordinary_loadings[
  ,
  1
]^2


loading_region_summary <- data.frame(
  Region = c(
    "High-Variance Noise 1: X1-X30",
    "High-Variance Noise 2: X31-X60",
    "Predictive Variables: X61-X80",
    "Independent Noise: X81-X120"
  ),
  PC1_Loading_Mass = c(
    sum(
      ordinary_pc1_loading_squared[
        1:30
      ]
    ),
    sum(
      ordinary_pc1_loading_squared[
        31:60
      ]
    ),
    sum(
      ordinary_pc1_loading_squared[
        61:80
      ]
    ),
    sum(
      ordinary_pc1_loading_squared[
        81:120
      ]
    )
  )
)


loading_region_summary


# ==============================================================================
# PART XXII
#
# SUPERVISED PC LOADING MASS
# ==============================================================================


spc_pc1_squared <- full_supervised_loadings[
  ,
  1
]^2


data.frame(
  Region = c(
    "High-Variance Noise 1",
    "High-Variance Noise 2",
    "Predictive Variables",
    "Independent Noise"
  ),
  SPC1_Loading_Mass = c(
    sum(
      spc_pc1_squared[
        1:30
      ]
    ),
    sum(
      spc_pc1_squared[
        31:60
      ]
    ),
    sum(
      spc_pc1_squared[
        61:80
      ]
    ),
    sum(
      spc_pc1_squared[
        81:120
      ]
    )
  )
)


# ==============================================================================
# PART XXIII
#
# THRESHOLD PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# Threshold controls how aggressively we screen variables.
#
# Higher threshold:
#
#       fewer variables.
#
# Lower threshold:
#
#       more variables.
# ------------------------------------------------------------------------------

threshold_grid <- seq(
  0,
  max(
    supervision_scores
  ) *
    0.95,
  length.out = 40
)


number_selected_path <- integer(
  length(
    threshold_grid
  )
)


for (
  i in seq_along(
    threshold_grid
  )
) {
  
  number_selected_path[i] <- sum(
    supervision_scores >=
      threshold_grid[i]
  )
}


plot(
  threshold_grid,
  number_selected_path,
  type = "s",
  xlab = "Screening Threshold",
  ylab = "Number of Selected Predictors",
  main = "SPC Screening Path"
)


# ==============================================================================
# PART XXIV
#
# TEST ERROR VS THRESHOLD
# ==============================================================================


# ------------------------------------------------------------------------------
# This uses the test set only as a diagnostic visualization.
#
# We will use CV later for actual tuning.
# ------------------------------------------------------------------------------

threshold_test_mse <- numeric(
  length(
    threshold_grid
  )
)


for (
  i in seq_along(
    threshold_grid
  )
) {
  
  fit_i <- fit_supervised_pca(
    X_train,
    y_train,
    threshold =
      threshold_grid[i],
    number_components = 2
  )
  
  
  prediction_i <- predict_supervised_pca(
    fit_i,
    X_test
  )
  
  
  threshold_test_mse[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


plot(
  threshold_grid,
  threshold_test_mse,
  type = "l",
  lwd = 2,
  xlab = "Screening Threshold",
  ylab = "Test MSE",
  main = "SPC Error vs Screening Threshold"
)


# ==============================================================================
# PART XXV
#
# NUMBER OF COMPONENTS
# ==============================================================================


component_grid <- 1:10


component_test_mse <- numeric(
  length(
    component_grid
  )
)


for (
  i in seq_along(
    component_grid
  )
) {
  
  fit_i <- fit_supervised_pca(
    X_train,
    y_train,
    threshold =
      initial_threshold,
    number_components =
      component_grid[i]
  )
  
  
  prediction_i <- predict_supervised_pca(
    fit_i,
    X_test
  )
  
  
  component_test_mse[i] <- mean(
    (
      y_test -
        prediction_i
    )^2
  )
}


plot(
  component_grid,
  component_test_mse,
  type = "b",
  pch = 19,
  xlab = "Number of Supervised PCs",
  ylab = "Test MSE",
  main = "SPC Model Complexity"
)


# ==============================================================================
# PART XXVI
#
# MANUAL K-FOLD CROSS-VALIDATION
# ==============================================================================


make_folds <- function(
    n,
    number_folds = 5,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  sample(
    rep(
      seq_len(
        number_folds
      ),
      length.out = n
    )
  )
}


# ------------------------------------------------------------------------------
# IMPORTANT:
#
# Screening MUST happen inside every CV training fold.
#
# If we screen using all of the training observations before CV, then the
# validation fold has already influenced which variables are selected.
#
# That is information leakage.
# ------------------------------------------------------------------------------


cross_validate_spc <- function(
    X_raw,
    y,
    threshold_grid,
    component_grid,
    number_folds = 5,
    seed = 123
) {
  
  X_raw <- as.matrix(
    X_raw
  )
  
  
  n <- nrow(
    X_raw
  )
  
  
  folds <- make_folds(
    n,
    number_folds =
      number_folds,
    seed =
      seed
  )
  
  
  results <- expand.grid(
    Threshold =
      threshold_grid,
    Components =
      component_grid
  )
  
  
  results$CV_MSE <- NA_real_
  
  results$CV_SE <- NA_real_
  
  results$Mean_Selected <- NA_real_
  
  
  for (
    combination in seq_len(
      nrow(
        results
      )
    )
  ) {
    
    threshold <- results$Threshold[
      combination
    ]
    
    
    components <- results$Components[
      combination
    ]
    
    
    fold_mse <- numeric(
      number_folds
    )
    
    
    selected_counts <- numeric(
      number_folds
    )
    
    
    for (
      fold in seq_len(
        number_folds
      )
    ) {
      
      validation_indices <- which(
        folds ==
          fold
      )
      
      
      training_indices <- which(
        folds !=
          fold
      )
      
      
      X_fold_train_raw <- X_raw[
        training_indices,
        ,
        drop = FALSE
      ]
      
      
      X_fold_validation_raw <- X_raw[
        validation_indices,
        ,
        drop = FALSE
      ]
      
      
      y_fold_train <- y[
        training_indices
      ]
      
      
      y_fold_validation <- y[
        validation_indices
      ]
      
      
      # ======================================================================
      # FOLD-SPECIFIC STANDARDIZATION
      # ======================================================================
      
      fold_means <- colMeans(
        X_fold_train_raw
      )
      
      
      fold_sds <- apply(
        X_fold_train_raw,
        2,
        sd
      )
      
      
      fold_sds[
        fold_sds <
          1e-12
      ] <- 1
      
      
      X_fold_train <- sweep(
        X_fold_train_raw,
        2,
        fold_means,
        "-"
      )
      
      
      X_fold_train <- sweep(
        X_fold_train,
        2,
        fold_sds,
        "/"
      )
      
      
      X_fold_validation <- sweep(
        X_fold_validation_raw,
        2,
        fold_means,
        "-"
      )
      
      
      X_fold_validation <- sweep(
        X_fold_validation,
        2,
        fold_sds,
        "/"
      )
      
      
      # ======================================================================
      # SCREEN AND FIT INSIDE TRAINING FOLD
      # ======================================================================
      
      fit <- fit_supervised_pca(
        X_fold_train,
        y_fold_train,
        threshold =
          threshold,
        number_components =
          components
      )
      
      
      selected_counts[
        fold
      ] <- length(
        fit$selected
      )
      
      
      prediction <- predict_supervised_pca(
        fit,
        X_fold_validation
      )
      
      
      fold_mse[
        fold
      ] <- mean(
        (
          y_fold_validation -
            prediction
        )^2
      )
    }
    
    
    results$CV_MSE[
      combination
    ] <- mean(
      fold_mse
    )
    
    
    results$CV_SE[
      combination
    ] <- sd(
      fold_mse
    ) /
      sqrt(
        number_folds
      )
    
    
    results$Mean_Selected[
      combination
    ] <- mean(
      selected_counts
    )
  }
  
  
  results
}


# ==============================================================================
# PART XXVII
#
# CV GRIDS
# ==============================================================================


# ------------------------------------------------------------------------------
# Instead of using every possible threshold, use quantiles of the observed
# supervision scores.
# ------------------------------------------------------------------------------

threshold_cv_grid <- unique(
  as.numeric(
    quantile(
      supervision_scores,
      probs = seq(
        0.40,
        0.95,
        length.out = 12
      ),
      names = FALSE
    )
  )
)


component_cv_grid <- 1:5


# ==============================================================================
# PART XXVIII
#
# RUN CROSS-VALIDATION
# ==============================================================================


cv_results <- cross_validate_spc(
  X_train_raw,
  y_train,
  threshold_grid =
    threshold_cv_grid,
  component_grid =
    component_cv_grid,
  number_folds = 5,
  seed = 123
)


head(
  cv_results
)


# ==============================================================================
# PART XXIX
#
# SELECT MINIMUM-CV MODEL
# ==============================================================================


best_index <- which.min(
  cv_results$CV_MSE
)


best_threshold <- cv_results$Threshold[
  best_index
]


best_components <- cv_results$Components[
  best_index
]


best_threshold


best_components


# ==============================================================================
# PART XXX
#
# ONE-STANDARD-ERROR RULE
# ==============================================================================


minimum_error <- cv_results$CV_MSE[
  best_index
]


minimum_se <- cv_results$CV_SE[
  best_index
]


one_se_threshold <- minimum_error +
  minimum_se


eligible_models <- which(
  cv_results$CV_MSE <=
    one_se_threshold
)


# ------------------------------------------------------------------------------
# Define "simpler" as:
#
#   1. fewer selected variables on average;
#   2. then fewer PCs;
#   3. then higher screening threshold.
# ------------------------------------------------------------------------------

eligible_table <- cv_results[
  eligible_models,
]


eligible_order <- order(
  eligible_table$Mean_Selected,
  eligible_table$Components,
  -eligible_table$Threshold
)


one_se_model <- eligible_table[
  eligible_order[1],
]


one_se_model


# ==============================================================================
# PART XXXI
#
# CV VISUALIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# Plot best CV MSE for each threshold after optimizing over PC count.
# ------------------------------------------------------------------------------

best_mse_by_threshold <- numeric(
  length(
    threshold_cv_grid
  )
)


for (
  i in seq_along(
    threshold_cv_grid
  )
) {
  
  subset_rows <- cv_results$Threshold ==
    threshold_cv_grid[i]
  
  
  best_mse_by_threshold[i] <- min(
    cv_results$CV_MSE[
      subset_rows
    ]
  )
}


plot(
  threshold_cv_grid,
  best_mse_by_threshold,
  type = "b",
  pch = 19,
  xlab = "Screening Threshold",
  ylab = "Best CV MSE",
  main = "Supervised PCA Cross-Validation"
)


# ==============================================================================
# PART XXXII
#
# FINAL SPC MODEL
# ==============================================================================


final_spc <- fit_supervised_pca(
  X_train,
  y_train,
  threshold =
    best_threshold,
  number_components =
    best_components
)


final_prediction <- predict_supervised_pca(
  final_spc,
  X_test
)


final_metrics <- regression_metrics(
  y_test,
  final_prediction
)


final_metrics


# ==============================================================================
# PART XXXIII
#
# SELECTED VARIABLES
# ==============================================================================


selected_variables <- final_spc$selected


selected_names <- colnames(
  X_train
)[
  selected_variables
]


selected_names


length(
  selected_variables
)


# ==============================================================================
# PART XXXIV
#
# SUPPORT RECOVERY
# ==============================================================================


selected_indicator <- seq_len(p) %in%
  selected_variables


truth_indicator <- seq_len(p) %in%
  true_signal_variables


TP <- sum(
  selected_indicator &
    truth_indicator
)


FP <- sum(
  selected_indicator &
    !truth_indicator
)


FN <- sum(
  !selected_indicator &
    truth_indicator
)


TN <- sum(
  !selected_indicator &
    !truth_indicator
)


precision <- TP /
  max(
    TP + FP,
    1
  )


recall <- TP /
  (
    TP +
      FN
  )


data.frame(
  TP = TP,
  FP = FP,
  TN = TN,
  FN = FN,
  Precision =
    precision,
  Recall =
    recall
)


# ==============================================================================
# PART XXXV
#
# FINAL SUPERVISED PC LOADINGS
# ==============================================================================


final_full_loadings <- matrix(
  0,
  nrow = p,
  ncol =
    final_spc$number_components
)


final_full_loadings[
  final_spc$selected,
] <- final_spc$loadings


matplot(
  seq_len(p),
  final_full_loadings,
  type = "h",
  lty = 1,
  lwd = 2,
  xlab = "Predictor",
  ylab = "Loading",
  main = "Final Supervised PC Loadings"
)


abline(
  v = c(
    60.5,
    80.5
  ),
  lty = 2
)


# ==============================================================================
# PART XXXVI
#
# EFFECTIVE REGRESSION COEFFICIENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# Final prediction:
#
#       y_hat
#
#       =
#
#       y_bar
#
#       +
#
#       X_selected V gamma
#
#
# Therefore effective standardized-variable coefficient vector is:
#
#       beta_SPC
#
#       =
#
#       V gamma
# ------------------------------------------------------------------------------

selected_effective_beta <- as.numeric(
  final_spc$loadings %*%
    final_spc$pc_coefficients
)


effective_beta <- rep(
  0,
  p
)


effective_beta[
  final_spc$selected
] <- selected_effective_beta


plot(
  seq_len(p),
  effective_beta,
  type = "h",
  lwd = 2,
  xlab = "Predictor",
  ylab = "Effective SPC Coefficient",
  main = "Effective Supervised-PC Regression Coefficients"
)


abline(
  h = 0,
  lty = 2
)


# ==============================================================================
# PART XXXVII
#
# BACK-TRANSFORM EFFECTIVE COEFFICIENTS
# ==============================================================================


beta_original_scale <- effective_beta /
  x_sds


intercept_original_scale <- y_mean -
  sum(
    beta_original_scale *
      x_means
  )


c(
  Intercept =
    intercept_original_scale,
  beta_original_scale
)[
  1:25
]


# ==============================================================================
# PART XXXVIII
#
# ORDINARY PCR CROSS-VALIDATION
# ==============================================================================


cross_validate_pcr <- function(
    X_raw,
    y,
    component_grid,
    number_folds = 5,
    seed = 123
) {
  
  X_raw <- as.matrix(
    X_raw
  )
  
  
  n <- nrow(
    X_raw
  )
  
  
  folds <- make_folds(
    n,
    number_folds,
    seed
  )
  
  
  fold_mse <- matrix(
    NA_real_,
    nrow =
      number_folds,
    ncol =
      length(
        component_grid
      )
  )
  
  
  for (
    fold in seq_len(
      number_folds
    )
  ) {
    
    validation_indices <- which(
      folds ==
        fold
    )
    
    
    training_indices <- which(
      folds !=
        fold
    )
    
    
    X_fold_train_raw <- X_raw[
      training_indices,
      ,
      drop = FALSE
    ]
    
    
    X_fold_validation_raw <- X_raw[
      validation_indices,
      ,
      drop = FALSE
    ]
    
    
    y_fold_train <- y[
      training_indices
    ]
    
    
    y_fold_validation <- y[
      validation_indices
    ]
    
    
    fold_means <- colMeans(
      X_fold_train_raw
    )
    
    
    fold_sds <- apply(
      X_fold_train_raw,
      2,
      sd
    )
    
    
    fold_sds[
      fold_sds <
        1e-12
    ] <- 1
    
    
    X_fold_train <- sweep(
      X_fold_train_raw,
      2,
      fold_means,
      "-"
    )
    
    
    X_fold_train <- sweep(
      X_fold_train,
      2,
      fold_sds,
      "/"
    )
    
    
    X_fold_validation <- sweep(
      X_fold_validation_raw,
      2,
      fold_means,
      "-"
    )
    
    
    X_fold_validation <- sweep(
      X_fold_validation,
      2,
      fold_sds,
      "/"
    )
    
    
    for (
      i in seq_along(
        component_grid
      )
    ) {
      
      fit <- fit_pcr(
        X_fold_train,
        y_fold_train,
        number_components =
          component_grid[i]
      )
      
      
      prediction <- predict_pcr(
        fit,
        X_fold_validation
      )
      
      
      fold_mse[
        fold,
        i
      ] <- mean(
        (
          y_fold_validation -
            prediction
        )^2
      )
    }
  }
  
  
  list(
    mean_mse =
      colMeans(
        fold_mse
      ),
    se =
      apply(
        fold_mse,
        2,
        sd
      ) /
      sqrt(
        number_folds
      ),
    component_grid =
      component_grid
  )
}


# ==============================================================================
# PART XXXIX
#
# TUNE ORDINARY PCR
# ==============================================================================


pcr_component_grid <- 1:20


pcr_cv <- cross_validate_pcr(
  X_train_raw,
  y_train,
  component_grid =
    pcr_component_grid,
  number_folds = 5,
  seed = 123
)


pcr_best_index <- which.min(
  pcr_cv$mean_mse
)


pcr_best_components <- pcr_component_grid[
  pcr_best_index
]


pcr_best_components


# ==============================================================================
# PART XL
#
# FINAL ORDINARY PCR
# ==============================================================================


final_pcr <- fit_pcr(
  X_train,
  y_train,
  number_components =
    pcr_best_components
)


final_pcr_prediction <- predict_pcr(
  final_pcr,
  X_test
)


pcr_metrics <- regression_metrics(
  y_test,
  final_pcr_prediction
)


pcr_metrics


# ==============================================================================
# PART XLI
#
# LINEAR REGRESSION BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# p < n in the main simulation, so OLS is possible.
# ------------------------------------------------------------------------------

ols_fit <- lm.fit(
  x = cbind(
    1,
    X_train
  ),
  y = y_train
)


ols_coefficients <- coef(
  ols_fit
)


ols_prediction <- as.numeric(
  cbind(
    1,
    X_test
  ) %*%
    ols_coefficients
)


ols_metrics <- regression_metrics(
  y_test,
  ols_prediction
)


ols_metrics


# ==============================================================================
# PART XLII
#
# SCREENED REGRESSION WITHOUT PCA
# ==============================================================================


# ------------------------------------------------------------------------------
# This comparison isolates what PCA contributes after screening.
# ------------------------------------------------------------------------------

screened_X_train <- X_train[
  ,
  final_spc$selected,
  drop = FALSE
]


screened_X_test <- X_test[
  ,
  final_spc$selected,
  drop = FALSE
]


screened_lm <- lm.fit(
  x = cbind(
    1,
    screened_X_train
  ),
  y = y_train
)


screened_coefficients <- coef(
  screened_lm
)


screened_prediction <- as.numeric(
  cbind(
    1,
    screened_X_test
  ) %*%
    screened_coefficients
)


screened_metrics <- regression_metrics(
  y_test,
  screened_prediction
)


screened_metrics


# ==============================================================================
# PART XLIII
#
# MODEL COMPARISON
# ==============================================================================


comparison <- data.frame(
  Model = c(
    "OLS",
    "Ordinary PCR",
    "Marginal Screening + OLS",
    "Supervised PCA"
  ),
  MSE = c(
    ols_metrics["MSE"],
    pcr_metrics["MSE"],
    screened_metrics["MSE"],
    final_metrics["MSE"]
  ),
  RMSE = c(
    ols_metrics["RMSE"],
    pcr_metrics["RMSE"],
    screened_metrics["RMSE"],
    final_metrics["RMSE"]
  ),
  R2 = c(
    ols_metrics["R2"],
    pcr_metrics["R2"],
    screened_metrics["R2"],
    final_metrics["R2"]
  )
)


comparison


# ==============================================================================
# PART XLIV
#
# PREDICTED VS OBSERVED
# ==============================================================================


plot(
  y_test,
  final_prediction,
  pch = 19,
  xlab = "Observed Y",
  ylab = "Predicted Y",
  main = "Supervised Principal Components"
)


abline(
  0,
  1,
  lwd = 2,
  lty = 2
)


# ==============================================================================
# PART XLV
#
# RESIDUAL DIAGNOSTICS
# ==============================================================================


spc_residuals <- y_test -
  final_prediction


par(
  mfrow = c(1, 3)
)


plot(
  final_prediction,
  spc_residuals,
  pch = 19,
  xlab = "Predicted",
  ylab = "Residual",
  main = "Residuals vs Predicted"
)


abline(
  h = 0,
  lty = 2
)


hist(
  spc_residuals,
  breaks = 20,
  main = "Residual Distribution",
  xlab = "Residual"
)


qqnorm(
  spc_residuals,
  main = "Residual Q-Q Plot"
)


qqline(
  spc_residuals
)


par(
  mfrow = c(1, 1)
)


# ==============================================================================
# PART XLVI
#
# STABILITY OF VARIABLE SCREENING
# ==============================================================================


# ------------------------------------------------------------------------------
# Bootstrap the training observations and repeat marginal screening.
#
# This shows how stable the supervised selection stage is.
# ------------------------------------------------------------------------------

number_bootstraps <- 200


selection_frequency <- numeric(
  p
)


set.seed(2025)


for (
  bootstrap_index in seq_len(
    number_bootstraps
  )
) {
  
  indices <- sample(
    seq_len(
      n_train
    ),
    size =
      n_train,
    replace = TRUE
  )
  
  
  X_bootstrap <- X_train[
    indices,
    ,
    drop = FALSE
  ]
  
  
  y_bootstrap <- y_train[
    indices
  ]
  
  
  bootstrap_scores <- numeric(
    p
  )
  
  
  for (
    j in seq_len(p)
  ) {
    
    bootstrap_scores[j] <- abs(
      cor(
        X_bootstrap[
          ,
          j
        ],
        y_bootstrap
      )
    )
  }
  
  
  bootstrap_selected <- bootstrap_scores >=
    best_threshold
  
  
  selection_frequency <- selection_frequency +
    bootstrap_selected
}


selection_frequency <- selection_frequency /
  number_bootstraps


# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

plot(
  seq_len(p),
  selection_frequency,
  type = "h",
  lwd = 2,
  xlab = "Predictor",
  ylab = "Bootstrap Selection Frequency",
  main = "SPC Screening Stability"
)


abline(
  v = c(
    60.5,
    80.5
  ),
  lty = 2
)


# ==============================================================================
# PART XLVII
#
# IMPORTANT FAILURE MODE:
#
# PURE INTERACTION SIGNAL
# ==============================================================================


# ------------------------------------------------------------------------------
# Marginal screening can fail when individual predictors have little marginal
# association with y but matter jointly.
#
# Example:
#
#       y = X1 * X2 + epsilon
#
# with independent zero-mean X1 and X2.
#
# Then:
#
#       cor(X1, y) approximately 0
#       cor(X2, y) approximately 0
#
# even though both variables are crucial.
# ------------------------------------------------------------------------------

set.seed(999)


n_interaction <- 1000


x1_interaction <- rnorm(
  n_interaction
)


x2_interaction <- rnorm(
  n_interaction
)


y_interaction <- x1_interaction *
  x2_interaction +
  rnorm(
    n_interaction,
    sd = 0.2
  )


cor(
  x1_interaction,
  y_interaction
)


cor(
  x2_interaction,
  y_interaction
)


# Both marginal correlations should be close to zero.
#
# Therefore simple SPC screening could discard both predictors.


# ==============================================================================
# PART XLVIII
#
# HIGH-DIMENSIONAL p > n EXAMPLE
# ==============================================================================


set.seed(888)


n_hd <- 80

p_hd <- 500


X_hd <- matrix(
  rnorm(
    n_hd *
      p_hd
  ),
  nrow =
    n_hd,
  ncol =
    p_hd
)


# ------------------------------------------------------------------------------
# Create a shared predictive factor among 15 variables.
# ------------------------------------------------------------------------------

signal_factor_hd <- rnorm(
  n_hd
)


for (
  j in 1:15
) {
  
  X_hd[
    ,
    j
  ] <- signal_factor_hd +
    rnorm(
      n_hd,
      sd = 0.5
    )
}


y_hd <- 4 *
  signal_factor_hd +
  rnorm(
    n_hd,
    sd = 1
  )


# ------------------------------------------------------------------------------
# Standardize
# ------------------------------------------------------------------------------

X_hd <- scale(
  X_hd
)


# ------------------------------------------------------------------------------
# Marginal Scores
# ------------------------------------------------------------------------------

scores_hd <- abs(
  apply(
    X_hd,
    2,
    cor,
    y =
      y_hd
  )
)


# ------------------------------------------------------------------------------
# Select Top 25 Predictors
#
# Instead of a fixed threshold, screening can equivalently be parameterized by
# the number of selected variables.
# ------------------------------------------------------------------------------

selected_hd <- order(
  scores_hd,
  decreasing = TRUE
)[
  1:25
]


X_hd_selected <- X_hd[
  ,
  selected_hd,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# PCA after screening.
# ------------------------------------------------------------------------------

svd_hd <- svd(
  X_hd_selected
)


score_hd <- X_hd_selected %*%
  svd_hd$v[
    ,
    1,
    drop = FALSE
  ]


cor(
  score_hd[
    ,
    1
  ],
  y_hd
)


# ------------------------------------------------------------------------------
# How many true variables appeared among top 25?
# ------------------------------------------------------------------------------

sum(
  selected_hd <=
    15
)


# SPC remains computationally attractive when:
#
#       p >> n
#
# because screening can dramatically reduce the dimension before PCA/regression.


# ==============================================================================
# PART XLIX
#
# SCREENING BY TOP-K RATHER THAN THRESHOLD
# ==============================================================================


# ------------------------------------------------------------------------------
# Sometimes it is easier to tune:
#
#       keep top K predictors
#
# rather than:
#
#       keep predictors with score >= threshold.
# ------------------------------------------------------------------------------

fit_supervised_pca_top_k <- function(
    X,
    y,
    top_k,
    number_components = 1
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  top_k <- min(
    top_k,
    p
  )
  
  
  scores <- abs(
    sapply(
      seq_len(p),
      function(j) {
        
        cor(
          X[
            ,
            j
          ],
          y
        )
      }
    )
  )
  
  
  selected <- order(
    scores,
    decreasing = TRUE
  )[
    seq_len(
      top_k
    )
  ]
  
  
  X_selected <- X[
    ,
    selected,
    drop = FALSE
  ]
  
  
  decomposition <- svd(
    X_selected
  )
  
  
  number_components <- min(
    number_components,
    top_k
  )
  
  
  loadings <- decomposition$v[
    ,
    seq_len(
      number_components
    ),
    drop = FALSE
  ]
  
  
  pc_scores <- X_selected %*%
    loadings
  
  
  y_mean <- mean(
    y
  )
  
  
  coefficients <- solve(
    crossprod(
      pc_scores
    ),
    crossprod(
      pc_scores,
      y -
        y_mean
    )
  )
  
  
  list(
    selected =
      selected,
    scores =
      scores,
    loadings =
      loadings,
    pc_coefficients =
      as.numeric(
        coefficients
      ),
    y_mean =
      y_mean
  )
}


# ==============================================================================
# PART L
#
# TOP-K EXAMPLE
# ==============================================================================


top_k_fit <- fit_supervised_pca_top_k(
  X_train,
  y_train,
  top_k = 25,
  number_components = 2
)


top_k_prediction <- predict_supervised_pca(
  top_k_fit,
  X_test
)


regression_metrics(
  y_test,
  top_k_prediction
)


# ==============================================================================
# PART LI
#
# SUPERVISED PCA VS LASSO CONCEPT
# ==============================================================================


# ------------------------------------------------------------------------------
# Lasso:
#
#       directly estimates sparse coefficients:
#
#           beta
#
#
# SPC:
#
#       screens variables first,
#       then constructs latent combinations of selected variables.
#
#
# Thus SPC is not purely variable selection.
#
# It is:
#
#       supervised screening
#
#       +
#
#       unsupervised latent dimension reduction
#
#       +
#
#       regression.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART LII
#
# SUPERVISED PCA VS PLS CONCEPT
# ==============================================================================


# ------------------------------------------------------------------------------
# Supervised PCA:
#
#       use y to SCREEN variables;
#       PCA itself still maximizes variance in selected X.
#
#
# Partial Least Squares:
#
#       constructs latent directions using X and y jointly.
#
#
# Thus after screening, SPC's PCA step is still ordinary PCA.
# ------------------------------------------------------------------------------


# ==============================================================================
# PART LIII
#
# OPTIONAL PACKAGE VERIFICATION
# ==============================================================================


# There are several R packages implementing supervised principal-component
# methods in particular application areas, historically including genomic /
# survival-analysis workflows.
#
# Exact formulations vary:
#
#   - regression score statistic vs correlation;
#   - Cox scores for survival outcomes;
#   - number of supervised components;
#   - threshold tuning;
#   - shrinkage of final coefficients.
#
# The purpose of this script is to expose the core algorithm directly rather
# than rely on one package implementation.


# ==============================================================================
# PART LIV
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nSupervised Principal Components Summary\n"
)


cat(
  "---------------------------------------\n"
)


cat(
  "Training observations:",
  n_train,
  "\n"
)


cat(
  "Predictors:",
  p,
  "\n"
)


cat(
  "True signal variables:",
  length(
    true_signal_variables
  ),
  "\n"
)


cat(
  "Selected threshold:",
  round(
    best_threshold,
    5
  ),
  "\n"
)


cat(
  "Selected supervised PCs:",
  best_components,
  "\n"
)


cat(
  "Selected predictors:",
  length(
    selected_variables
  ),
  "\n"
)


cat(
  "Selection precision:",
  round(
    precision,
    4
  ),
  "\n"
)


cat(
  "Selection recall:",
  round(
    recall,
    4
  ),
  "\n"
)


cat(
  "SPC test MSE:",
  round(
    final_metrics["MSE"],
    5
  ),
  "\n"
)


cat(
  "SPC test R2:",
  round(
    final_metrics["R2"],
    5
  ),
  "\n"
)


cat(
  "\nModel comparison:\n"
)


print(
  comparison
)