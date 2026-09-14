# ==============================================================================
# Exploratory Projection Pursuit
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Unsupervised projection pursuit
#   - Interesting low-dimensional projections
#   - Projection indices
#   - Non-Gaussianity
#   - Kurtosis
#   - Skewness
#   - Approximate negentropy
#   - Random projection search
#   - Numerical optimization
#   - Orthogonal sequential projections
#   - Comparison with PCA
#   - Connection to ICA
#
#
# Projection pursuit searches for directions:
#
#       a
#
# such that the projection:
#
#       z = X a
#
# is "interesting".
#
#
# A general one-dimensional problem is:
#
#       max_a I(X a)
#
# subject to:
#
#       ||a||_2 = 1
#
#
# where:
#
#       I(.) = projection index
#
#
# PCA corresponds to one special choice:
#
#       I(Xa) = Var(Xa)
#
#
# Exploratory projection pursuit instead often looks for departures from
# Gaussianity or other low-dimensional structure.
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE DATA WITH HIDDEN NON-GAUSSIAN STRUCTURE
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulation
# ------------------------------------------------------------------------------

set.seed(123)

n <- 800

p <- 6


# --------------------------------------------------------------------------
# Latent directions
#
# Most observed variation will be ordinary Gaussian noise.
# Two hidden directions contain more interesting structure:
#
#   latent_1 = bimodal
#   latent_2 = skewed
# --------------------------------------------------------------------------

latent_1 <- c(
  rnorm(
    n / 2,
    mean = -2,
    sd = 0.5
  ),
  rnorm(
    n / 2,
    mean = 2,
    sd = 0.5
  )
)


latent_1 <- sample(
  latent_1
)


latent_2 <- rexp(
  n,
  rate = 1
)


latent_2 <- latent_2 -
  mean(
    latent_2
  )


# Gaussian nuisance dimensions.

noise_1 <- rnorm(
  n
)

noise_2 <- rnorm(
  n
)

noise_3 <- rnorm(
  n
)

noise_4 <- rnorm(
  n
)


latent_matrix <- cbind(
  latent_1,
  latent_2,
  noise_1,
  noise_2,
  noise_3,
  noise_4
)


# Standardize latent columns.

latent_matrix <- scale(
  latent_matrix
)


# ==============================================================================
# PART II
#
# MIX LATENT VARIABLES
# ==============================================================================


# ------------------------------------------------------------------------------
# 2. Random Orthogonal Mixing Matrix
# ------------------------------------------------------------------------------

set.seed(456)


random_matrix <- matrix(
  rnorm(
    p * p
  ),
  nrow = p
)


QR <- qr(
  random_matrix
)


mixing_matrix <- qr.Q(
  QR
)


# ------------------------------------------------------------------------------
# 3. Observed Data
# ------------------------------------------------------------------------------

X <- latent_matrix %*%
  t(
    mixing_matrix
  )


colnames(
  X
) <- paste0(
  "X",
  seq_len(p)
)


# ------------------------------------------------------------------------------
# 4. Inspect
# ------------------------------------------------------------------------------

head(
  X
)


round(
  cor(
    X
  ),
  2
)


# The interesting directions are now hidden as linear combinations of the
# observed variables.


# ==============================================================================
# PART III
#
# STANDARDIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Standardize Observed Variables
# ------------------------------------------------------------------------------

feature_means <- colMeans(
  X
)


feature_sds <- apply(
  X,
  2,
  sd
)


X_standardized <- sweep(
  X,
  2,
  feature_means,
  "-"
)


X_standardized <- sweep(
  X_standardized,
  2,
  feature_sds,
  "/"
)


# ==============================================================================
# PART IV
#
# PCA BASELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 6. Ordinary PCA
# ------------------------------------------------------------------------------

PCA <- svd(
  X_standardized
)


PCA_loadings <- PCA$v


PCA_scores <- X_standardized %*%
  PCA_loadings


# ------------------------------------------------------------------------------
# 7. PCA Variances
# ------------------------------------------------------------------------------

apply(
  PCA_scores,
  2,
  var
)


# Because the observed variables were standardized and the mixing is close to
# orthogonal, the variance structure alone may not reveal the most interesting
# non-Gaussian directions.


# ==============================================================================
# PART V
#
# PROJECTION INDEX UTILITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. Standardize a Projection
# ------------------------------------------------------------------------------

standardize_vector <- function(
    z
) {
  
  z <- z -
    mean(z)
  
  
  standard_deviation <- sd(
    z
  )
  
  
  if (
    standard_deviation <=
    .Machine$double.eps
  ) {
    
    return(
      rep(
        0,
        length(z)
      )
    )
  }
  
  
  z /
    standard_deviation
}


# ------------------------------------------------------------------------------
# 9. Skewness
# ------------------------------------------------------------------------------

projection_skewness <- function(
    z
) {
  
  z <- standardize_vector(
    z
  )
  
  
  mean(
    z^3
  )
}


# ------------------------------------------------------------------------------
# 10. Excess Kurtosis
# ------------------------------------------------------------------------------

projection_kurtosis <- function(
    z
) {
  
  z <- standardize_vector(
    z
  )
  
  
  mean(
    z^4
  ) -
    3
}


# ------------------------------------------------------------------------------
# 11. Absolute Kurtosis Projection Index
# ------------------------------------------------------------------------------

index_absolute_kurtosis <- function(
    z
) {
  
  abs(
    projection_kurtosis(
      z
    )
  )
}


# ------------------------------------------------------------------------------
# 12. Absolute Skewness Projection Index
# ------------------------------------------------------------------------------

index_absolute_skewness <- function(
    z
) {
  
  abs(
    projection_skewness(
      z
    )
  )
}


# ==============================================================================
# PART VI
#
# APPROXIMATE NEGENTROPY
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Numerically Stable log(cosh(x))
# ------------------------------------------------------------------------------

log_cosh_stable <- function(
    x
) {
  
  abs(x) +
    log1p(
      exp(
        -2 *
          abs(x)
      )
    ) -
    log(2)
}


# ------------------------------------------------------------------------------
# 14. Gaussian Reference Expectation
# ------------------------------------------------------------------------------

set.seed(999)


gaussian_reference <- rnorm(
  100000
)


gaussian_logcosh_mean <- mean(
  log_cosh_stable(
    gaussian_reference
  )
)


# ------------------------------------------------------------------------------
# 15. Approximate Negentropy Projection Index
# ------------------------------------------------------------------------------

index_negentropy <- function(
    z
) {
  
  z <- standardize_vector(
    z
  )
  
  
  difference <- mean(
    log_cosh_stable(
      z
    )
  ) -
    gaussian_logcosh_mean
  
  
  difference^2
}


# ==============================================================================
# PART VII
#
# EVALUATE KNOWN DISTRIBUTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Gaussian
# ------------------------------------------------------------------------------

set.seed(100)


example_gaussian <- rnorm(
  5000
)


# ------------------------------------------------------------------------------
# 17. Skewed
# ------------------------------------------------------------------------------

example_skewed <- rexp(
  5000
)


# ------------------------------------------------------------------------------
# 18. Bimodal
# ------------------------------------------------------------------------------

example_bimodal <- c(
  rnorm(
    2500,
    -2,
    0.5
  ),
  rnorm(
    2500,
    2,
    0.5
  )
)


# ------------------------------------------------------------------------------
# 19. Heavy-Tailed
# ------------------------------------------------------------------------------

example_heavy <- rt(
  5000,
  df = 3
)


distribution_index_table <- data.frame(
  Distribution = c(
    "Gaussian",
    "Skewed",
    "Bimodal",
    "Heavy-Tailed"
  ),
  Skewness = c(
    projection_skewness(
      example_gaussian
    ),
    projection_skewness(
      example_skewed
    ),
    projection_skewness(
      example_bimodal
    ),
    projection_skewness(
      example_heavy
    )
  ),
  Kurtosis = c(
    projection_kurtosis(
      example_gaussian
    ),
    projection_kurtosis(
      example_skewed
    ),
    projection_kurtosis(
      example_bimodal
    ),
    projection_kurtosis(
      example_heavy
    )
  ),
  Negentropy = c(
    index_negentropy(
      example_gaussian
    ),
    index_negentropy(
      example_skewed
    ),
    index_negentropy(
      example_bimodal
    ),
    index_negentropy(
      example_heavy
    )
  )
)


distribution_index_table


# ==============================================================================
# PART VIII
#
# NORMALIZE PROJECTION DIRECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Unit-Norm Direction
# ------------------------------------------------------------------------------

normalize_direction <- function(
    a
) {
  
  norm_a <- sqrt(
    sum(
      a^2
    )
  )
  
  
  if (
    norm_a <=
    .Machine$double.eps
  ) {
    
    return(
      rep(
        1 /
          sqrt(
            length(a)
          ),
        length(a)
      )
    )
  }
  
  
  a /
    norm_a
}


# ==============================================================================
# PART IX
#
# SCORE A PROJECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Generic Projection Index
# ------------------------------------------------------------------------------

projection_index <- function(
    X,
    direction,
    index_function
) {
  
  direction <- normalize_direction(
    direction
  )
  
  
  z <- as.numeric(
    X %*%
      direction
  )
  
  
  index_function(
    z
  )
}


# ==============================================================================
# PART X
#
# RANDOM PROJECTION SEARCH
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Random Search
# ------------------------------------------------------------------------------

random_projection_search <- function(
    X,
    index_function,
    number_directions = 5000,
    seed = 123
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  set.seed(
    seed
  )
  
  
  directions <- matrix(
    NA_real_,
    nrow =
      number_directions,
    ncol =
      p
  )
  
  
  index_values <- numeric(
    number_directions
  )
  
  
  for (
    i in seq_len(
      number_directions
    )
  ) {
    
    direction <- rnorm(
      p
    )
    
    
    direction <- normalize_direction(
      direction
    )
    
    
    directions[
      i,
    ] <- direction
    
    
    index_values[i] <- projection_index(
      X,
      direction,
      index_function
    )
  }
  
  
  best_index <- which.max(
    index_values
  )
  
  
  list(
    direction =
      directions[
        best_index,
      ],
    index_value =
      index_values[
        best_index
      ],
    directions =
      directions,
    index_values =
      index_values
  )
}


# ==============================================================================
# PART XI
#
# SEARCH FOR NON-GAUSSIAN DIRECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Kurtosis Search
# ------------------------------------------------------------------------------

random_kurtosis_search <- random_projection_search(
  X_standardized,
  index_absolute_kurtosis,
  number_directions = 5000,
  seed = 123
)


random_kurtosis_search$index_value


# ------------------------------------------------------------------------------
# 24. Negentropy Search
# ------------------------------------------------------------------------------

random_negentropy_search <- random_projection_search(
  X_standardized,
  index_negentropy,
  number_directions = 5000,
  seed = 456
)


random_negentropy_search$index_value


# ------------------------------------------------------------------------------
# 25. Distribution of Random Projection Indices
# ------------------------------------------------------------------------------

hist(
  random_negentropy_search$index_values,
  breaks = 40,
  xlab = "Approximate Negentropy",
  main = "Interestingness of Random Projections"
)


abline(
  v =
    random_negentropy_search$index_value,
  lwd = 2
)


# ==============================================================================
# PART XII
#
# NUMERICAL OPTIMIZATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Optimization Objective
#
# optim() minimizes, so return negative projection index.
# ------------------------------------------------------------------------------

projection_objective <- function(
    parameters,
    X,
    index_function
) {
  
  direction <- normalize_direction(
    parameters
  )
  
  
  -projection_index(
    X,
    direction,
    index_function
  )
}


# ------------------------------------------------------------------------------
# 27. Optimize One Direction
# ------------------------------------------------------------------------------

optimize_projection <- function(
    X,
    index_function,
    initial_direction = NULL,
    number_starts = 10,
    seed = 123,
    maxit = 1000
) {
  
  X <- as.matrix(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  set.seed(
    seed
  )
  
  
  starting_directions <- vector(
    "list",
    number_starts
  )
  
  
  if (
    !is.null(
      initial_direction
    )
  ) {
    
    starting_directions[[1]] <-
      normalize_direction(
        initial_direction
      )
    
    
    if (
      number_starts >
      1
    ) {
      
      for (
        start in 2:number_starts
      ) {
        
        starting_directions[[start]] <-
          normalize_direction(
            rnorm(
              p
            )
          )
      }
    }
    
  } else {
    
    for (
      start in seq_len(
        number_starts
      )
    ) {
      
      starting_directions[[start]] <-
        normalize_direction(
          rnorm(
            p
          )
        )
    }
  }
  
  
  best_value <- -Inf
  
  best_direction <- NULL
  
  best_result <- NULL
  
  
  for (
    start in seq_len(
      number_starts
    )
  ) {
    
    result <- optim(
      par =
        starting_directions[[start]],
      fn =
        projection_objective,
      X =
        X,
      index_function =
        index_function,
      method = "BFGS",
      control = list(
        maxit =
          maxit
      )
    )
    
    
    direction <- normalize_direction(
      result$par
    )
    
    
    value <- projection_index(
      X,
      direction,
      index_function
    )
    
    
    if (
      value >
      best_value
    ) {
      
      best_value <-
        value
      
      
      best_direction <-
        direction
      
      
      best_result <-
        result
    }
  }
  
  
  list(
    direction =
      best_direction,
    index_value =
      best_value,
    optim_result =
      best_result,
    scores =
      as.numeric(
        X %*%
          best_direction
      )
  )
}


# ==============================================================================
# PART XIII
#
# OPTIMIZED EXPLORATORY PROJECTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Use Random Search Winner as One Initialization
# ------------------------------------------------------------------------------

EPP_negentropy <- optimize_projection(
  X_standardized,
  index_function =
    index_negentropy,
  initial_direction =
    random_negentropy_search$direction,
  number_starts = 15,
  seed = 123
)


EPP_negentropy$direction


EPP_negentropy$index_value


# ------------------------------------------------------------------------------
# 29. Kurtosis Direction
# ------------------------------------------------------------------------------

EPP_kurtosis <- optimize_projection(
  X_standardized,
  index_function =
    index_absolute_kurtosis,
  initial_direction =
    random_kurtosis_search$direction,
  number_starts = 15,
  seed = 456
)


# ------------------------------------------------------------------------------
# 30. Skewness Direction
# ------------------------------------------------------------------------------

EPP_skewness <- optimize_projection(
  X_standardized,
  index_function =
    index_absolute_skewness,
  number_starts = 15,
  seed = 789
)


# ==============================================================================
# PART XIV
#
# PLOT DISCOVERED PROJECTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Negentropy Projection
# ------------------------------------------------------------------------------

hist(
  EPP_negentropy$scores,
  breaks = 40,
  xlab = "Projection Score",
  main = "EPP Projection: Approximate Negentropy"
)


# ------------------------------------------------------------------------------
# 32. Kurtosis Projection
# ------------------------------------------------------------------------------

hist(
  EPP_kurtosis$scores,
  breaks = 40,
  xlab = "Projection Score",
  main = "EPP Projection: Absolute Kurtosis"
)


# ------------------------------------------------------------------------------
# 33. Skewness Projection
# ------------------------------------------------------------------------------

hist(
  EPP_skewness$scores,
  breaks = 40,
  xlab = "Projection Score",
  main = "EPP Projection: Absolute Skewness"
)


# ==============================================================================
# PART XV
#
# COMPARE PROJECTION INDICES
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Index Table
# ------------------------------------------------------------------------------

projection_comparison <- data.frame(
  Projection = c(
    "PCA PC1",
    "EPP Negentropy",
    "EPP Kurtosis",
    "EPP Skewness"
  ),
  Variance = c(
    var(
      PCA_scores[
        ,
        1
      ]
    ),
    var(
      EPP_negentropy$scores
    ),
    var(
      EPP_kurtosis$scores
    ),
    var(
      EPP_skewness$scores
    )
  ),
  Absolute_Skewness = c(
    index_absolute_skewness(
      PCA_scores[
        ,
        1
      ]
    ),
    index_absolute_skewness(
      EPP_negentropy$scores
    ),
    index_absolute_skewness(
      EPP_kurtosis$scores
    ),
    index_absolute_skewness(
      EPP_skewness$scores
    )
  ),
  Absolute_Kurtosis = c(
    index_absolute_kurtosis(
      PCA_scores[
        ,
        1
      ]
    ),
    index_absolute_kurtosis(
      EPP_negentropy$scores
    ),
    index_absolute_kurtosis(
      EPP_kurtosis$scores
    ),
    index_absolute_kurtosis(
      EPP_skewness$scores
    )
  ),
  Negentropy = c(
    index_negentropy(
      PCA_scores[
        ,
        1
      ]
    ),
    index_negentropy(
      EPP_negentropy$scores
    ),
    index_negentropy(
      EPP_kurtosis$scores
    ),
    index_negentropy(
      EPP_skewness$scores
    )
  )
)


projection_comparison


# ==============================================================================
# PART XVI
#
# COMPARE TO TRUE HIDDEN LATENT VARIABLES
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Correlations with Latent Variables
# ------------------------------------------------------------------------------

discovered_scores <- cbind(
  PCA1 =
    PCA_scores[
      ,
      1
    ],
  EPP_Negentropy =
    EPP_negentropy$scores,
  EPP_Kurtosis =
    EPP_kurtosis$scores,
  EPP_Skewness =
    EPP_skewness$scores
)


latent_correlations <- cor(
  latent_matrix,
  discovered_scores
)


round(
  latent_correlations,
  3
)


# Because this is simulated data, we can inspect whether the EPP directions
# recovered the hidden bimodal or skewed factors.
#
# In real exploratory data analysis, these latent variables are unknown.


# ==============================================================================
# PART XVII
#
# DIRECTION SIMILARITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. True Observed-Space Directions
#
# Since:
#
#       X = latent_matrix %*% t(mixing_matrix)
#
# the columns of mixing_matrix correspond to observed-space directions of
# the latent factors under the orthogonal simulation.
# ------------------------------------------------------------------------------

true_direction_1 <- mixing_matrix[
  ,
  1
]


true_direction_2 <- mixing_matrix[
  ,
  2
]


# Standardization slightly changes the exact observed-space geometry, so this
# comparison is approximate.


direction_similarity <- function(
    a,
    b
) {
  
  a <- normalize_direction(
    a
  )
  
  
  b <- normalize_direction(
    b
  )
  
  
  abs(
    sum(
      a *
        b
    )
  )
}


data.frame(
  EPP_Method = c(
    "Negentropy",
    "Kurtosis",
    "Skewness"
  ),
  Similarity_to_Bimodal_Direction = c(
    direction_similarity(
      EPP_negentropy$direction,
      true_direction_1
    ),
    direction_similarity(
      EPP_kurtosis$direction,
      true_direction_1
    ),
    direction_similarity(
      EPP_skewness$direction,
      true_direction_1
    )
  ),
  Similarity_to_Skewed_Direction = c(
    direction_similarity(
      EPP_negentropy$direction,
      true_direction_2
    ),
    direction_similarity(
      EPP_kurtosis$direction,
      true_direction_2
    ),
    direction_similarity(
      EPP_skewness$direction,
      true_direction_2
    )
  )
)


# ==============================================================================
# PART XVIII
#
# SEQUENTIAL PROJECTION PURSUIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 37. Orthogonalize Candidate Direction
# ------------------------------------------------------------------------------

orthogonalize_direction <- function(
    direction,
    previous_directions
) 
  
  direction <- as.numeric(
    direction
  )
  
  
  if (
    is.null(
      previous_directions
    )
  ) {
    
    return(
      normalize_direction(
        direction
      )
    )
    
    
    for (
      j in seq_len(
        ncol(
          previous_directions
        )
      )
    ) {
      
      previous_direction <- previous_directions[
        ,
        j
      ]
      
      
      direction <- direction -
        sum(
          direction *
            previous_direction
        ) *
        previous_direction
    }
    
    
    norm_direction <- sqrt(
      sum(
        direction^2
      )
    )
    
    
    if (
      norm_direction <=
      1e-10
    ) {
      
      return(
        NULL
      )
    }
    
    
    direction /
      norm_direction
  }
  
  
  # ------------------------------------------------------------------------------
  # 38. Objective with Orthogonality Constraints
  # ------------------------------------------------------------------------------
  
  projection_objective_orthogonal <- function(
    parameters,
    X,
    index_function,
    previous_directions = NULL
  ) {
    
    direction <- orthogonalize_direction(
      parameters,
      previous_directions
    )
    
    
    if (
      is.null(
        direction
      )
    ) {
      
      return(
        1e6
      )
    }
    
    
    -projection_index(
      X,
      direction,
      index_function
    )
  }
  
  
  # ------------------------------------------------------------------------------
  # 39. Sequential EPP
  # ------------------------------------------------------------------------------
  
  fit_sequential_epp <- function(
    X,
    number_components = 3,
    index_function = index_negentropy,
    number_starts = 10,
    seed = 123,
    maxit = 1000
  ) {
    
    X <- as.matrix(
      X
    )
    
    
    p <- ncol(
      X
    )
    
    
    directions <- matrix(
      0,
      nrow =
        p,
      ncol =
        number_components
    )
    
    
    scores <- matrix(
      0,
      nrow =
        nrow(X),
      ncol =
        number_components
    )
    
    
    index_values <- numeric(
      number_components
    )
    
    
    set.seed(
      seed
    )
    
    
    for (
      component in seq_len(
        number_components
      )
    ) {
      
      previous_directions <- NULL
      
      
      if (
        component >
        1
      ) {
        
        previous_directions <- directions[
          ,
          seq_len(
            component - 1
          ),
          drop = FALSE
        ]
      }
      
      
      best_value <- -Inf
      
      best_direction <- NULL
      
      
      for (
        start in seq_len(
          number_starts
        )
      ) {
        
        initial <- rnorm(
          p
        )
        
        
        result <- optim(
          par =
            initial,
          fn =
            projection_objective_orthogonal,
          X =
            X,
          index_function =
            index_function,
          previous_directions =
            previous_directions,
          method = "BFGS",
          control = list(
            maxit =
              maxit
          )
        )
        
        
        direction <- orthogonalize_direction(
          result$par,
          previous_directions
        )
        
        
        if (
          is.null(
            direction
          )
        ) {
          
          next
        }
        
        
        value <- projection_index(
          X,
          direction,
          index_function
        )
        
        
        if (
          value >
          best_value
        ) {
          
          best_value <-
            value
          
          
          best_direction <-
            direction
        }
      }
      
      
      if (
        is.null(
          best_direction
        )
      ) {
        
        stop(
          "Could not find a valid orthogonal projection direction."
        )
      }
      
      
      directions[
        ,
        component
      ] <-
        best_direction
      
      
      scores[
        ,
        component
      ] <-
        X %*%
        best_direction
      
      
      index_values[
        component
      ] <-
        best_value
    }
    
    
    colnames(
      directions
    ) <- paste0(
      "EPP",
      seq_len(
        number_components
      )
    )
    
    
    rownames(
      directions
    ) <- colnames(
      X
    )
    
    
    colnames(
      scores
    ) <- paste0(
      "EPP",
      seq_len(
        number_components
      )
    )
    
    
    list(
      directions =
        directions,
      scores =
        scores,
      index_values =
        index_values
    )
  }
  
  
  # ==============================================================================
  # PART XIX
  #
  # FIT MULTIPLE EXPLORATORY DIRECTIONS
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 40. Sequential Negentropy EPP
  # ------------------------------------------------------------------------------
  
  EPP_sequence <- fit_sequential_epp(
    X_standardized,
    number_components = 3,
    index_function =
      index_negentropy,
    number_starts = 15,
    seed = 123
  )
  
  
  EPP_sequence$directions
  
  
  EPP_sequence$index_values
  
  
  # ------------------------------------------------------------------------------
  # 41. Verify Orthogonality
  # ------------------------------------------------------------------------------
  
  round(
    t(
      EPP_sequence$directions
    ) %*%
      EPP_sequence$directions,
    5
  )
  
  
  # ==============================================================================
  # PART XX
  #
  # VISUALIZE MULTIPLE EPP COMPONENTS
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 42. Histograms
  # ------------------------------------------------------------------------------
  
  par(
    mfrow = c(
      1,
      3
    )
  )
  
  
  for (
    component in 1:3
  ) {
    
    hist(
      EPP_sequence$scores[
        ,
        component
      ],
      breaks = 40,
      xlab = paste0(
        "EPP",
        component
      ),
      main = paste(
        "Projection",
        component
      )
    )
  }
  
  
  par(
    mfrow = c(
      1,
      1
    )
  )
  
  
  # ------------------------------------------------------------------------------
  # 43. Two-Dimensional EPP View
  # ------------------------------------------------------------------------------
  
  plot(
    EPP_sequence$scores[
      ,
      1
    ],
    EPP_sequence$scores[
      ,
      2
    ],
    pch = 19,
    cex = 0.4,
    xlab = "EPP1",
    ylab = "EPP2",
    main = "Exploratory Projection Pursuit Space"
  )
  
  
  # ==============================================================================
  # PART XXI
  #
  # COMPARE PCA AND EPP TWO-DIMENSIONAL VIEWS
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 44. PCA
  # ------------------------------------------------------------------------------
  
  plot(
    PCA_scores[
      ,
      1
    ],
    PCA_scores[
      ,
      2
    ],
    pch = 19,
    cex = 0.4,
    xlab = "PC1",
    ylab = "PC2",
    main = "PCA Projection"
  )
  
  
  # ------------------------------------------------------------------------------
  # 45. EPP
  # ------------------------------------------------------------------------------
  
  plot(
    EPP_sequence$scores[
      ,
      1
    ],
    EPP_sequence$scores[
      ,
      2
    ],
    pch = 19,
    cex = 0.4,
    xlab = "EPP1",
    ylab = "EPP2",
    main = "Projection Pursuit Projection"
  )
  
  
  # ==============================================================================
  # PART XXII
  #
  # A SIMPLE BIMODALITY INDEX
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 46. Histogram Contrast Index
  #
  # This is a pedagogical multimodality index, not a standard projection-pursuit
  # index.
  #
  # It compares density concentration away from the central region.
  # ------------------------------------------------------------------------------
  
  index_tail_separation <- function(
    z
  ) {
    
    z <- standardize_vector(
      z
    )
    
    
    central_fraction <- mean(
      abs(z) <
        0.5
    )
    
    
    outer_fraction <- mean(
      abs(z) >
        1
    )
    
    
    outer_fraction -
      central_fraction
  }
  
  
  # ------------------------------------------------------------------------------
  # 47. Search
  # ------------------------------------------------------------------------------
  
  EPP_bimodal <- optimize_projection(
    X_standardized,
    index_function =
      index_tail_separation,
    number_starts = 20,
    seed = 321
  )
  
  
  hist(
    EPP_bimodal$scores,
    breaks = 40,
    xlab = "Projection Score",
    main = "Projection Pursuit for Bimodal Structure"
  )
  
  
  # ==============================================================================
  # PART XXIII
  #
  # RANDOM PROJECTION BASELINE
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 48. Compare Optimized EPP to Random Directions
  # ------------------------------------------------------------------------------
  
  set.seed(111)
  
  
  number_random <- 2000
  
  
  random_indices <- numeric(
    number_random
  )
  
  
  for (
    i in seq_len(
      number_random
    )
  ) {
    
    random_direction <- normalize_direction(
      rnorm(
        p
      )
    )
    
    
    random_indices[i] <- projection_index(
      X_standardized,
      random_direction,
      index_negentropy
    )
  }
  
  
  quantile(
    random_indices,
    probs = c(
      0.50,
      0.90,
      0.95,
      0.99
    )
  )
  
  
  EPP_negentropy$index_value
  
  
  hist(
    random_indices,
    breaks = 40,
    xlab = "Negentropy Index",
    main = "Optimized EPP vs Random Projections"
  )
  
  
  abline(
    v =
      EPP_negentropy$index_value,
    lwd = 2
  )
  
  
  # ==============================================================================
  # PART XXIV
  #
  # STABILITY ACROSS RANDOM STARTS
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 49. Refit Multiple Times
  # ------------------------------------------------------------------------------
  
  number_refits <- 20
  
  
  refit_directions <- matrix(
    NA_real_,
    nrow =
      number_refits,
    ncol =
      p
  )
  
  
  refit_indices <- numeric(
    number_refits
  )
  
  
  for (
    i in seq_len(
      number_refits
    )
  ) {
    
    fit_i <- optimize_projection(
      X_standardized,
      index_function =
        index_negentropy,
      number_starts = 5,
      seed =
        1000 +
        i
    )
    
    
    refit_directions[
      i,
    ] <- fit_i$direction
    
    
    refit_indices[i] <-
      fit_i$index_value
  }
  
  
  # ------------------------------------------------------------------------------
  # 50. Direction Similarity to Best Fit
  # ------------------------------------------------------------------------------
  
  stability_similarity <- apply(
    refit_directions,
    1,
    function(direction) {
      
      direction_similarity(
        direction,
        EPP_negentropy$direction
      )
    }
  )
  
  
  data.frame(
    Refit =
      1:number_refits,
    Index =
      refit_indices,
    Similarity_to_Reference =
      stability_similarity
  )
  
  
  # ==============================================================================
  # PART XXV
  #
  # OUTLIER SENSITIVITY OF KURTOSIS
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 51. Add a Few Extreme Observations
  # ------------------------------------------------------------------------------
  
  set.seed(222)
  
  
  X_outlier <- X_standardized
  
  
  outlier_rows <- sample(
    seq_len(n),
    5
  )
  
  
  X_outlier[
    outlier_rows,
    1
  ] <- X_outlier[
    outlier_rows,
    1
  ] +
    15
  
  
  # ------------------------------------------------------------------------------
  # 52. Refit Kurtosis Projection
  # ------------------------------------------------------------------------------
  
  EPP_kurtosis_outlier <- optimize_projection(
    X_outlier,
    index_function =
      index_absolute_kurtosis,
    number_starts = 15,
    seed = 222
  )
  
  
  # ------------------------------------------------------------------------------
  # 53. Compare Directions
  # ------------------------------------------------------------------------------
  
  direction_similarity(
    EPP_kurtosis$direction,
    EPP_kurtosis_outlier$direction
  )
  
  
  # Kurtosis can be highly sensitive to a few extreme observations.
  
  
  # ==============================================================================
  # PART XXVI
  #
  # NEGENTROPY ROBUSTNESS COMPARISON
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 54. Negentropy with Outliers
  # ------------------------------------------------------------------------------
  
  EPP_negentropy_outlier <- optimize_projection(
    X_outlier,
    index_function =
      index_negentropy,
    number_starts = 15,
    seed = 222
  )
  
  
  direction_similarity(
    EPP_negentropy$direction,
    EPP_negentropy_outlier$direction
  )
  
  
  # log-cosh-based criteria are generally less extreme-value-sensitive than
  # fourth-moment kurtosis criteria.
  
  
  # ==============================================================================
  # PART XXVII
  #
  # GAUSSIAN DATA
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 55. Pure Gaussian Example
  # ------------------------------------------------------------------------------
  
  set.seed(333)
  
  
  X_gaussian <- matrix(
    rnorm(
      1000 *
        6
    ),
    nrow = 1000,
    ncol = 6
  )
  
  
  X_gaussian <- scale(
    X_gaussian
  )
  
  
  # ------------------------------------------------------------------------------
  # 56. EPP on Gaussian Data
  # ------------------------------------------------------------------------------
  
  gaussian_EPP <- optimize_projection(
    X_gaussian,
    index_function =
      index_negentropy,
    number_starts = 15,
    seed = 333
  )
  
  
  gaussian_EPP$index_value
  
  
  EPP_negentropy$index_value
  
  
  # On approximately Gaussian data, no projection should show strong systematic
  # non-Gaussian structure beyond finite-sample fluctuations.
  
  
  # ==============================================================================
  # PART XXVIII
  #
  # PERMUTATION / NULL CALIBRATION
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 57. Destroy Multivariate Structure
  #
  # Independently permute each feature. This preserves marginal distributions
  # while removing dependence between variables.
  # ------------------------------------------------------------------------------
  
  set.seed(444)
  
  
  X_permuted <- X_standardized
  
  
  for (
    j in seq_len(
      ncol(
        X_permuted
      )
    )
  ) {
    
    X_permuted[
      ,
      j
    ] <- sample(
      X_permuted[
        ,
        j
      ]
    )
  }
  
  
  permuted_EPP <- optimize_projection(
    X_permuted,
    index_function =
      index_negentropy,
    number_starts = 10,
    seed = 444
  )
  
  
  data.frame(
    Dataset = c(
      "Observed",
      "Independently Permuted Features"
    ),
    Best_Negentropy_Index = c(
      EPP_negentropy$index_value,
      permuted_EPP$index_value
    )
  )
  
  
  # ==============================================================================
  # PART XXIX
  #
  # PCA VS PROJECTION PURSUIT OBJECTIVES
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 58. Evaluate PCA Directions with EPP Index
  # ------------------------------------------------------------------------------
  
  PCA_negentropy <- numeric(
    p
  )
  
  
  for (
    j in seq_len(p)
  ) {
    
    PCA_negentropy[j] <- index_negentropy(
      PCA_scores[
        ,
        j
      ]
    )
  }
  
  
  data.frame(
    Component =
      paste0(
        "PC",
        seq_len(p)
      ),
    Variance =
      apply(
        PCA_scores,
        2,
        var
      ),
    Negentropy =
      PCA_negentropy
  )
  
  
  # A direction can explain relatively modest variance while being highly
  # interesting under a non-Gaussian projection index.
  
  
  # ==============================================================================
  # PART XXX
  #
  # CONNECTION TO ICA
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 59. Conceptual Comparison
  # ------------------------------------------------------------------------------
  
  # Projection Pursuit:
  #
  #       Search for interesting projections.
  #
  #
  # ICA:
  #
  #       Search for a set of projections that are approximately statistically
  #       independent.
  #
  #
  # Non-Gaussianity is often the projection index used in ICA.
  #
  # Therefore ICA can be viewed as a structured form of projection pursuit.
  
  
  # ==============================================================================
  # PART XXXI
  #
  # OPTIONAL PACKAGE NOTE
  # ==============================================================================
  
  
  # ------------------------------------------------------------------------------
  # 60. tourr / classifly / projection-pursuit packages
  # ------------------------------------------------------------------------------
  
  # There are R packages for dynamic projection pursuit and grand tours, but the
  # purpose of this script is to expose the search problem manually.
  #
  # A production exploratory workflow may combine:
  #
  #   - projection pursuit indices
  #   - optimization
  #   - grand tours
  #   - interactive visualization
  #
  # rather than rely on a single static optimum.
  
  
  # ==============================================================================
  # PART XXXII
  #
  # FINAL SUMMARY
  # ==============================================================================
  
  
  cat(
    "\nExploratory Projection Pursuit Summary\n"
  )
  
  
  cat(
    "--------------------------------------\n"
  )
  
  
  cat(
    "Observations:",
    n,
    "\n"
  )
  
  
  cat(
    "Variables:",
    p,
    "\n"
  )
  
  
  cat(
    "Best negentropy index:",
    round(
      EPP_negentropy$index_value,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Best absolute kurtosis index:",
    round(
      EPP_kurtosis$index_value,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Best absolute skewness index:",
    round(
      EPP_skewness$index_value,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Sequential EPP indices:",
    round(
      EPP_sequence$index_values,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Gaussian-data negentropy optimum:",
    round(
      gaussian_EPP$index_value,
      6
    ),
    "\n"
  )