# ==============================================================================
# MARS: Multivariate Adaptive Regression Splines
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Multivariate Adaptive Regression Splines (MARS)
#   - Hinge functions
#   - Adaptive knot selection
#   - Forward basis construction
#   - Tensor-product interactions
#   - Backward pruning
#   - Generalized Cross Validation (GCV)
#   - Piecewise-linear regression
#
# Basic MARS model:
#
#   f(x) = beta_0 + sum_{m=1}^M beta_m h_m(x)
#
# where h_m(x) is a hinge basis function or a product of hinge functions.
#
# Basic hinge pair:
#
#   h_+(x; t) = max(0, x - t)
#   h_-(x; t) = max(0, t - x)
#
# Forward MARS:
#
#   1. Start with intercept.
#   2. Search predictors and knots.
#   3. Add hinge-function pairs.
#   4. Allow products with existing basis functions.
#   5. Continue until maximum model size.
#
# Backward MARS:
#
#   1. Start with large forward model.
#   2. Remove basis functions.
#   3. Evaluate model complexity using GCV.
#   4. Select best pruned model.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE NONLINEAR DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 400


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


x3 <- runif(
  n,
  min = -3,
  max = 3
)


# True model contains:
#
#   - piecewise-linear effect of x1
#   - piecewise-linear effect of x2
#   - interaction between x1 and x2
#
# x3 is irrelevant.


hinge_positive <- function(
    x,
    knot
) {
  
  pmax(
    0,
    x - knot
  )
}


hinge_negative <- function(
    x,
    knot
) {
  
  pmax(
    0,
    knot - x
  )
}


true_function <- function(
    x1,
    x2,
    x3
) {
  
  2 +
    2.5 *
    hinge_positive(
      x1,
      0
    ) -
    1.5 *
    hinge_negative(
      x1,
      0
    ) +
    2 *
    hinge_positive(
      x2,
      0.5
    ) +
    1.5 *
    hinge_positive(
      x1,
      0
    ) *
    hinge_negative(
      x2,
      0.5
    )
}


y_true <- true_function(
  x1,
  x2,
  x3
)


sigma <- 1


y <- y_true +
  rnorm(
    n,
    mean = 0,
    sd = sigma
  )


X <- cbind(
  x1 = x1,
  x2 = x2,
  x3 = x3
)


data <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  x3 = x3
)


# ------------------------------------------------------------------------------
# 2. Explore Data
# ------------------------------------------------------------------------------

pairs(
  data,
  main = "MARS Simulation Data"
)


# ==============================================================================
# PART II
#
# HINGE FUNCTIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Visualize Hinge Functions
# ------------------------------------------------------------------------------

x_grid <- seq(
  -3,
  3,
  length.out = 500
)


knot_demo <- 0


positive_demo <- hinge_positive(
  x_grid,
  knot_demo
)


negative_demo <- hinge_negative(
  x_grid,
  knot_demo
)


plot(
  x_grid,
  positive_demo,
  type = "l",
  lwd = 2,
  xlab = "x",
  ylab = "Hinge Value",
  main = "MARS Hinge Functions"
)


lines(
  x_grid,
  negative_demo,
  lwd = 2,
  lty = 2
)


abline(
  v = knot_demo,
  lty = 3
)


legend(
  "topleft",
  legend = c(
    "max(0, x - t)",
    "max(0, t - x)"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART III
#
# BASIS-FUNCTION REPRESENTATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Create Intercept Basis
# ------------------------------------------------------------------------------

make_intercept_basis <- function(
    n
) {
  
  list(
    values = rep(
      1,
      n
    ),
    parent = 0,
    variable = 0,
    knot = NA_real_,
    direction = 0,
    degree = 0,
    description = "1"
  )
}


# ------------------------------------------------------------------------------
# 5. Evaluate One Hinge
# ------------------------------------------------------------------------------

evaluate_hinge <- function(
    x,
    knot,
    direction
) {
  
  if (direction == 1) {
    
    return(
      pmax(
        0,
        x - knot
      )
    )
  }
  
  
  if (direction == -1) {
    
    return(
      pmax(
        0,
        knot - x
      )
    )
  }
  
  
  stop(
    "direction must be +1 or -1"
  )
}


# ------------------------------------------------------------------------------
# 6. Hinge Description
# ------------------------------------------------------------------------------

hinge_description <- function(
    variable_name,
    knot,
    direction
) {
  
  if (direction == 1) {
    
    paste0(
      "max(0, ",
      variable_name,
      " - ",
      round(
        knot,
        3
      ),
      ")"
    )
    
  } else {
    
    paste0(
      "max(0, ",
      round(
        knot,
        3
      ),
      " - ",
      variable_name,
      ")"
    )
  }
}


# ==============================================================================
# PART IV
#
# LINEAR MODEL HELPERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Fit Least Squares Safely
# ------------------------------------------------------------------------------

least_squares_fit <- function(
    B,
    y
) {
  
  B <- as.matrix(
    B
  )
  
  
  fit <- lm.fit(
    x = B,
    y = y
  )
  
  
  coefficients <- fit$coefficients
  
  
  # Rank-deficient candidate designs can occur.
  #
  # Reject them rather than allowing NA coefficients into the search.
  
  if (
    fit$rank <
    ncol(B) ||
    anyNA(coefficients)
  ) {
    
    return(
      list(
        coefficients = coefficients,
        fitted = rep(
          NA_real_,
          length(y)
        ),
        residuals = rep(
          NA_real_,
          length(y)
        ),
        RSS = Inf,
        rank = fit$rank
      )
    )
  }
  
  
  fitted <- as.vector(
    B %*%
      coefficients
  )
  
  
  residuals <- y -
    fitted
  
  
  RSS <- sum(
    residuals^2
  )
  
  
  return(
    list(
      coefficients = coefficients,
      fitted = fitted,
      residuals = residuals,
      RSS = RSS,
      rank = fit$rank
    )
  )
}


# ------------------------------------------------------------------------------
# 8. Build Basis Matrix
# ------------------------------------------------------------------------------

basis_matrix <- function(
    basis_list
) {
  
  do.call(
    cbind,
    lapply(
      basis_list,
      function(basis) {
        
        basis$values
      }
    )
  )
}


# ==============================================================================
# PART V
#
# CANDIDATE KNOTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Candidate Knots
# ------------------------------------------------------------------------------

# Searching every observed x-value is possible but expensive.
#
# For this educational implementation, use interior quantiles.
#
# This preserves the adaptive knot-search idea while keeping runtime reasonable.


candidate_knots <- function(
    x,
    number_knots = 15
) {
  
  probabilities <- seq(
    0.05,
    0.95,
    length.out = number_knots
  )
  
  
  knots <- as.numeric(
    quantile(
      x,
      probs = probabilities,
      names = FALSE,
      type = 7
    )
  )
  
  
  unique(
    knots
  )
}


# ==============================================================================
# PART VI
#
# FORWARD MARS SEARCH
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Check Whether Variable Already Appears in Basis
# ------------------------------------------------------------------------------

# To keep interaction degree interpretable, we track the variables appearing
# in each basis term.


basis_variables <- function(
    basis,
    basis_list
) {
  
  variables <- integer(0)
  
  
  current <- basis
  
  
  while (
    current$parent != 0
  ) {
    
    variables <- c(
      variables,
      current$variable
    )
    
    
    current <- basis_list[[current$parent]]
  }
  
  
  if (basis$variable != 0) {
    
    variables <- c(
      variables,
      basis$variable
    )
  }
  
  
  unique(
    variables
  )
}


# ------------------------------------------------------------------------------
# 11. Forward MARS
# ------------------------------------------------------------------------------

mars_forward <- function(
    X,
    y,
    max_terms = 15,
    max_degree = 2,
    number_knots = 15,
    min_improvement = 1e-8,
    verbose = FALSE
) {
  
  X <- as.matrix(
    X
  )
  
  
  n <- nrow(
    X
  )
  
  
  p <- ncol(
    X
  )
  
  
  feature_names <- colnames(
    X
  )
  
  
  if (is.null(feature_names)) {
    
    feature_names <- paste0(
      "X",
      seq_len(p)
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Start with intercept
  # --------------------------------------------------------------------------
  
  basis_list <- list(
    make_intercept_basis(
      n
    )
  )
  
  
  B <- basis_matrix(
    basis_list
  )
  
  
  current_fit <- least_squares_fit(
    B,
    y
  )
  
  
  current_RSS <- current_fit$RSS
  
  
  history <- data.frame(
    Step = 0,
    Terms = 1,
    RSS = current_RSS,
    Improvement = NA_real_,
    Parent = NA_integer_,
    Variable = NA_character_,
    Knot = NA_real_
  )
  
  
  # --------------------------------------------------------------------------
  # Forward search
  # --------------------------------------------------------------------------
  
  step <- 0
  
  
  while (
    length(basis_list) + 2 <=
    max_terms
  ) {
    
    step <- step + 1
    
    
    best_RSS <- Inf
    
    best_candidate <- NULL
    
    
    # ------------------------------------------------------------------------
    # Search over parent basis functions
    # ------------------------------------------------------------------------
    
    for (
      parent_index in seq_along(
        basis_list
      )
    ) {
      
      parent_basis <- basis_list[[parent_index]]
        
      
      
      # A new hinge multiplies the parent.
      #
      # Interaction degree increases by one.
      
      new_degree <- parent_basis$degree +
        1
      
      
      if (
        new_degree >
        max_degree
      ) {
        
        next
      }
      
      
      # ----------------------------------------------------------------------
      # Search over predictors
      # ----------------------------------------------------------------------
      
      for (
        j in seq_len(p)
      ) {
        
        knots_j <- candidate_knots(
          X[, j],
          number_knots = number_knots
        )
        
        
        for (
          knot in knots_j
        ) {
          
          positive_hinge <- evaluate_hinge(
            X[, j],
            knot,
            direction = 1
          )
          
          
          negative_hinge <- evaluate_hinge(
            X[, j],
            knot,
            direction = -1
          )
          
          
          positive_values <-
            parent_basis$values *
            positive_hinge
          
          
          negative_values <-
            parent_basis$values *
            negative_hinge
          
          
          # Reject useless basis functions.
          
          if (
            all(
              positive_values == 0
            ) ||
            all(
              negative_values == 0
            )
          ) {
            
            next
          }
          
          
          candidate_B <- cbind(
            B,
            positive_values,
            negative_values
          )
          
          
          candidate_fit <- least_squares_fit(
            candidate_B,
            y
          )
          
          
          candidate_RSS <- candidate_fit$RSS
          
          
          if (
            candidate_RSS <
            best_RSS
          ) {
            
            best_RSS <- candidate_RSS
            
            
            best_candidate <- list(
              parent = parent_index,
              variable = j,
              knot = knot,
              degree = new_degree,
              positive_values = positive_values,
              negative_values = negative_values
            )
          }
        }
      }
    }
    
    
    # ------------------------------------------------------------------------
    # Stop if no candidate exists
    # ------------------------------------------------------------------------
    
    if (
      is.null(
        best_candidate
      )
    ) {
      
      break
    }
    
    
    improvement <- current_RSS -
      best_RSS
    
    
    if (
      improvement <=
      min_improvement
    ) {
      
      break
    }
    
    
    # ------------------------------------------------------------------------
    # Add best hinge pair
    # ------------------------------------------------------------------------
    
    parent_description <-
      basis_list[[best_candidate$parent]]$description
    
    
    variable_name <- feature_names[
      best_candidate$variable
    ]
    
    
    positive_description <- hinge_description(
      variable_name,
      best_candidate$knot,
      1
    )
    
    
    negative_description <- hinge_description(
      variable_name,
      best_candidate$knot,
      -1
    )
    
    
    if (
      best_candidate$parent != 1
    ) {
      
      positive_description <- paste0(
        "(",
        parent_description,
        ") * ",
        positive_description
      )
      
      
      negative_description <- paste0(
        "(",
        parent_description,
        ") * ",
        negative_description
      )
    }
    
    
    positive_basis <- list(
      values = best_candidate$positive_values,
      parent = best_candidate$parent,
      variable = best_candidate$variable,
      knot = best_candidate$knot,
      direction = 1,
      degree = best_candidate$degree,
      description = positive_description
    )
    
    
    negative_basis <- list(
      values = best_candidate$negative_values,
      parent = best_candidate$parent,
      variable = best_candidate$variable,
      knot = best_candidate$knot,
      direction = -1,
      degree = best_candidate$degree,
      description = negative_description
    )
    
    
    basis_list[[length(basis_list) + 1]] <- positive_basis
    
    
    basis_list[[length(basis_list) + 1]] <- negative_basis
    
    
    B <- basis_matrix(
      basis_list
    )
    
    
    current_fit <- least_squares_fit(
      B,
      y
    )
    
    
    current_RSS <- current_fit$RSS
    
    
    history <- rbind(
      history,
      data.frame(
        Step = step,
        Terms = length(
          basis_list
        ),
        RSS = current_RSS,
        Improvement = improvement,
        Parent = best_candidate$parent,
        Variable = variable_name,
        Knot = best_candidate$knot
      )
    )
    
    
    if (verbose) {
      
      cat(
        "Step:",
        step,
        "| Terms:",
        length(basis_list),
        "| Variable:",
        variable_name,
        "| Knot:",
        round(
          best_candidate$knot,
          3
        ),
        "| RSS:",
        round(
          current_RSS,
          3
        ),
        "\n"
      )
    }
  }
  
  
  final_fit <- least_squares_fit(
    B,
    y
  )
  
  
  return(
    list(
      basis = basis_list,
      basis_matrix = B,
      coefficients = final_fit$coefficients,
      fitted = final_fit$fitted,
      residuals = final_fit$residuals,
      RSS = final_fit$RSS,
      history = history,
      feature_names = feature_names,
      max_degree = max_degree
    )
  )
}


# ------------------------------------------------------------------------------
# 12. Fit Forward MARS Model
# ------------------------------------------------------------------------------

forward_model <- mars_forward(
  X = X,
  y = y,
  max_terms = 15,
  max_degree = 2,
  number_knots = 15,
  verbose = TRUE
)


# ------------------------------------------------------------------------------
# 13. Forward Search History
# ------------------------------------------------------------------------------

forward_model$history


# ------------------------------------------------------------------------------
# 14. Inspect Basis Functions
# ------------------------------------------------------------------------------

basis_descriptions <- sapply(
  forward_model$basis,
  function(basis) {
    
    basis$description
  }
)


basis_descriptions


# ------------------------------------------------------------------------------
# 15. Forward Coefficients
# ------------------------------------------------------------------------------

forward_coefficients <- data.frame(
  Basis = basis_descriptions,
  Coefficient = forward_model$coefficients
)


forward_coefficients


# ==============================================================================
# PART VII
#
# FORWARD MODEL PERFORMANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. Training Error
# ------------------------------------------------------------------------------

forward_training_MSE <- mean(
  forward_model$residuals^2
)


forward_training_RMSE <- sqrt(
  forward_training_MSE
)


forward_training_MSE


forward_training_RMSE


# ------------------------------------------------------------------------------
# 17. Plot Forward RSS
# ------------------------------------------------------------------------------

plot(
  forward_model$history$Terms,
  forward_model$history$RSS,
  type = "b",
  pch = 19,
  xlab = "Number of Basis Functions",
  ylab = "Training RSS",
  main = "MARS Forward Search"
)


# ==============================================================================
# PART VIII
#
# EVALUATING BASIS FUNCTIONS ON NEW DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Evaluate One Stored Basis Function
# ------------------------------------------------------------------------------

evaluate_basis_new <- function(
    basis_index,
    basis_list,
    X_new
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  basis <- basis_list[[basis_index]]
  
  
  # Intercept
  
  if (
    basis_index == 1
  ) {
    
    return(
      rep(
        1,
        nrow(X_new)
      )
    )
  }
  
  
  parent_values <- evaluate_basis_new(
    basis$parent,
    basis_list,
    X_new
  )
  
  
  hinge_values <- evaluate_hinge(
    X_new[
      ,
      basis$variable
    ],
    basis$knot,
    basis$direction
  )
  
  
  parent_values *
    hinge_values
}


# ------------------------------------------------------------------------------
# 19. Construct New Basis Matrix
# ------------------------------------------------------------------------------

mars_basis_new <- function(
    model,
    X_new,
    selected_terms = seq_along(
      model$basis
    )
) {
  
  X_new <- as.matrix(
    X_new
  )
  
  
  B_new <- sapply(
    selected_terms,
    function(index) {
      
      evaluate_basis_new(
        index,
        model$basis,
        X_new
      )
    }
  )
  
  
  if (
    is.null(
      dim(B_new)
    )
  ) {
    
    B_new <- matrix(
      B_new,
      ncol = 1
    )
  }
  
  
  return(
    B_new
  )
}


# ------------------------------------------------------------------------------
# 20. Prediction Helper
# ------------------------------------------------------------------------------

predict_mars <- function(
    model,
    X_new,
    selected_terms = NULL,
    coefficients = NULL
) {
  
  if (
    is.null(
      selected_terms
    )
  ) {
    
    selected_terms <- seq_along(
      model$basis
    )
  }
  
  
  B_new <- mars_basis_new(
    model,
    X_new,
    selected_terms
  )
  
  
  if (
    is.null(
      coefficients
    )
  ) {
    
    coefficients <- model$coefficients
  }
  
  
  as.vector(
    B_new %*%
      coefficients
  )
}


# ------------------------------------------------------------------------------
# 21. Verify Training Predictions
# ------------------------------------------------------------------------------

training_prediction_check <- predict_mars(
  forward_model,
  X
)


max(
  abs(
    training_prediction_check -
      forward_model$fitted
  )
)


# ==============================================================================
# PART IX
#
# BACKWARD PRUNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. GCV Criterion
# ------------------------------------------------------------------------------

# A common MARS model-selection criterion is generalized cross validation.
#
# Simplified educational form:
#
#        RSS / n
# GCV = ----------------
#       (1 - C(M)/n)^2
#
# where C(M) penalizes model complexity.
#
# We use:
#
#   C(M) = M + penalty * (M - 1) / 2
#
# M includes the intercept.
#
# The extra term penalizes adaptive basis-function construction.


mars_gcv <- function(
    RSS,
    n,
    number_terms,
    penalty = 2
) {
  
  effective_complexity <-
    number_terms +
    penalty *
    (
      number_terms -
        1
    ) /
    2
  
  
  denominator <-
    1 -
    effective_complexity /
    n
  
  
  if (
    denominator <= 0
  ) {
    
    return(Inf)
  }
  
  
  (
    RSS /
      n
  ) /
    denominator^2
}


# ------------------------------------------------------------------------------
# 23. Fit a Selected Set of Basis Functions
# ------------------------------------------------------------------------------

fit_selected_basis <- function(
    full_model,
    y,
    selected_terms
) {
  
  B <- full_model$basis_matrix[
    ,
    selected_terms,
    drop = FALSE
  ]
  
  
  fit <- least_squares_fit(
    B,
    y
  )
  
  
  return(
    list(
      selected_terms = selected_terms,
      coefficients = fit$coefficients,
      fitted = fit$fitted,
      residuals = fit$residuals,
      RSS = fit$RSS
    )
  )
}


# ------------------------------------------------------------------------------
# 24. Backward Pruning
# ------------------------------------------------------------------------------

mars_backward_prune <- function(
    forward_model,
    y,
    penalty = 2
) {
  
  n <- length(
    y
  )
  
  
  selected_terms <- seq_along(
    forward_model$basis
  )
  
  
  models <- list()
  
  
  full_fit <- fit_selected_basis(
    forward_model,
    y,
    selected_terms
  )
  
  
  models[[1]] <- list(
    selected_terms = selected_terms,
    coefficients = full_fit$coefficients,
    RSS = full_fit$RSS,
    GCV = mars_gcv(
      RSS = full_fit$RSS,
      n = n,
      number_terms = length(
        selected_terms
      ),
      penalty = penalty
    )
  )
  
  
  # --------------------------------------------------------------------------
  # Remove one non-intercept term at a time
  # --------------------------------------------------------------------------
  
  while (
    length(selected_terms) >
    1
  ) {
    
    removable_terms <- selected_terms[
      selected_terms != 1
    ]
    
    
    best_RSS <- Inf
    
    best_terms <- NULL
    
    best_fit <- NULL
    
    
    for (
      term in removable_terms
    ) {
      
      candidate_terms <- selected_terms[
        selected_terms != term
      ]
      
      
      candidate_fit <- fit_selected_basis(
        forward_model,
        y,
        candidate_terms
      )
      
      
      if (
        candidate_fit$RSS <
        best_RSS
      ) {
        
        best_RSS <- candidate_fit$RSS
        
        best_terms <- candidate_terms
        
        best_fit <- candidate_fit
      }
    }
    
    
    selected_terms <- best_terms
    
    
    model_GCV <- mars_gcv(
      RSS = best_fit$RSS,
      n = n,
      number_terms = length(
        selected_terms
      ),
      penalty = penalty
    )
    
    
    models[[length(models) + 1]] <- list(
      selected_terms = selected_terms,
      coefficients = best_fit$coefficients,
      RSS = best_fit$RSS,
      GCV = model_GCV
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Select minimum-GCV subtree/model
  # --------------------------------------------------------------------------
  
  GCV_values <- sapply(
    models,
    function(model) {
      
      model$GCV
    }
  )
  
  
  best_index <- which.min(
    GCV_values
  )
  
  
  best_model <- models[[best_index]]
   
  
  
  return(
    list(
      models = models,
      GCV = GCV_values,
      best_index = best_index,
      best_model = best_model
    )
  )
}


# ------------------------------------------------------------------------------
# 25. Run Backward Pruning
# ------------------------------------------------------------------------------

pruning_result <- mars_backward_prune(
  forward_model = forward_model,
  y = y,
  penalty = 2
)


# ------------------------------------------------------------------------------
# 26. Pruning Results
# ------------------------------------------------------------------------------

pruning_table <- data.frame(
  Step = seq_along(
    pruning_result$models
  ),
  Terms = sapply(
    pruning_result$models,
    function(model) {
      
      length(
        model$selected_terms
      )
    }
  ),
  RSS = sapply(
    pruning_result$models,
    function(model) {
      
      model$RSS
    }
  ),
  GCV = pruning_result$GCV
)


pruning_table


# ------------------------------------------------------------------------------
# 27. Plot GCV
# ------------------------------------------------------------------------------

plot(
  pruning_table$Terms,
  pruning_table$GCV,
  type = "b",
  pch = 19,
  xlab = "Number of Basis Functions",
  ylab = "GCV",
  main = "MARS Backward Pruning"
)


# ==============================================================================
# PART X
#
# FINAL PRUNED MODEL
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Selected Terms
# ------------------------------------------------------------------------------

best_pruned <- pruning_result$best_model


best_terms <- best_pruned$selected_terms


best_basis_descriptions <- basis_descriptions[
  best_terms
]


best_basis_descriptions


# ------------------------------------------------------------------------------
# 29. Final Coefficients
# ------------------------------------------------------------------------------

final_coefficients <- data.frame(
  Basis = best_basis_descriptions,
  Coefficient = best_pruned$coefficients
)


final_coefficients


# ------------------------------------------------------------------------------
# 30. Final Training Predictions
# ------------------------------------------------------------------------------

pruned_training_prediction <- predict_mars(
  model = forward_model,
  X_new = X,
  selected_terms = best_terms,
  coefficients = best_pruned$coefficients
)


pruned_training_MSE <- mean(
  (
    y -
      pruned_training_prediction
  )^2
)


pruned_training_MSE


# ==============================================================================
# PART XI
#
# VISUALIZE PIECEWISE-LINEAR FIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. One-Dimensional Slice Through x1
# ------------------------------------------------------------------------------

x1_grid <- seq(
  -3,
  3,
  length.out = 500
)


slice_x1 <- cbind(
  x1 = x1_grid,
  x2 = 0,
  x3 = 0
)


mars_slice_x1 <- predict_mars(
  model = forward_model,
  X_new = slice_x1,
  selected_terms = best_terms,
  coefficients = best_pruned$coefficients
)


true_slice_x1 <- true_function(
  x1_grid,
  0,
  0
)


plot(
  x1_grid,
  mars_slice_x1,
  type = "l",
  lwd = 2,
  xlab = "x1",
  ylab = "Predicted y",
  main = "MARS Piecewise-Linear Prediction"
)


lines(
  x1_grid,
  true_slice_x1,
  lty = 2,
  lwd = 2
)


legend(
  "topleft",
  legend = c(
    "MARS",
    "True Function"
  ),
  lty = c(
    1,
    2
  ),
  lwd = 2
)


# ==============================================================================
# PART XII
#
# TWO-DIMENSIONAL RESPONSE SURFACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Prediction Grid
# ------------------------------------------------------------------------------

grid_size <- 75


grid_x1 <- seq(
  -3,
  3,
  length.out = grid_size
)


grid_x2 <- seq(
  -3,
  3,
  length.out = grid_size
)


prediction_grid <- expand.grid(
  x1 = grid_x1,
  x2 = grid_x2
)


X_grid <- cbind(
  x1 = prediction_grid$x1,
  x2 = prediction_grid$x2,
  x3 = 0
)


grid_prediction <- predict_mars(
  model = forward_model,
  X_new = X_grid,
  selected_terms = best_terms,
  coefficients = best_pruned$coefficients
)


prediction_matrix <- matrix(
  grid_prediction,
  nrow = grid_size,
  ncol = grid_size
)


# ------------------------------------------------------------------------------
# 33. Plot MARS Surface
# ------------------------------------------------------------------------------

image(
  grid_x1,
  grid_x2,
  prediction_matrix,
  xlab = "x1",
  ylab = "x2",
  main = "MARS Prediction Surface"
)


contour(
  grid_x1,
  grid_x2,
  prediction_matrix,
  add = TRUE
)


# ==============================================================================
# PART XIII
#
# TRAIN-TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Train-Test Split
# ------------------------------------------------------------------------------

set.seed(456)


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


X_train <- X[
  train_index,
  ,
  drop = FALSE
]


y_train <- y[
  train_index
]


X_test <- X[
  test_index,
  ,
  drop = FALSE
]


y_test <- y[
  test_index
]


# ------------------------------------------------------------------------------
# 35. Forward Search on Training Data Only
# ------------------------------------------------------------------------------

forward_train <- mars_forward(
  X = X_train,
  y = y_train,
  max_terms = 15,
  max_degree = 2,
  number_knots = 15
)


# ------------------------------------------------------------------------------
# 36. Prune Training Model
# ------------------------------------------------------------------------------

pruning_train <- mars_backward_prune(
  forward_model = forward_train,
  y = y_train,
  penalty = 2
)


selected_train_model <- pruning_train$best_model


# ------------------------------------------------------------------------------
# 37. Test Predictions
# ------------------------------------------------------------------------------

mars_test_prediction <- predict_mars(
  model = forward_train,
  X_new = X_test,
  selected_terms =
    selected_train_model$selected_terms,
  coefficients =
    selected_train_model$coefficients
)


mars_test_MSE <- mean(
  (
    y_test -
      mars_test_prediction
  )^2
)


mars_test_RMSE <- sqrt(
  mars_test_MSE
)


mars_test_MSE


mars_test_RMSE


# ==============================================================================
# PART XIV
#
# COMPARE INTERACTION DEGREES
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Additive MARS
# ------------------------------------------------------------------------------

# max_degree = 1:
#
# only univariate hinge functions
#
# no products / interactions.


mars_additive <- mars_forward(
  X = X_train,
  y = y_train,
  max_terms = 15,
  max_degree = 1,
  number_knots = 15
)


mars_additive_pruned <- mars_backward_prune(
  mars_additive,
  y_train,
  penalty = 2
)


additive_selected <-
  mars_additive_pruned$best_model


additive_test_prediction <- predict_mars(
  model = mars_additive,
  X_new = X_test,
  selected_terms =
    additive_selected$selected_terms,
  coefficients =
    additive_selected$coefficients
)


additive_test_MSE <- mean(
  (
    y_test -
      additive_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 39. Interaction MARS
# ------------------------------------------------------------------------------

# max_degree = 2 allows two-variable interactions.


mars_interaction <- mars_forward(
  X = X_train,
  y = y_train,
  max_terms = 15,
  max_degree = 2,
  number_knots = 15
)


mars_interaction_pruned <- mars_backward_prune(
  mars_interaction,
  y_train,
  penalty = 2
)


interaction_selected <-
  mars_interaction_pruned$best_model


interaction_test_prediction <- predict_mars(
  model = mars_interaction,
  X_new = X_test,
  selected_terms =
    interaction_selected$selected_terms,
  coefficients =
    interaction_selected$coefficients
)


interaction_test_MSE <- mean(
  (
    y_test -
      interaction_test_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 40. Compare Interaction Complexity
# ------------------------------------------------------------------------------

data.frame(
  Model = c(
    "Additive MARS",
    "Interaction MARS"
  ),
  Test_MSE = c(
    additive_test_MSE,
    interaction_test_MSE
  )
)


# ==============================================================================
# PART XV
#
# VARIABLE IMPORTANCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Basis-Based Variable Importance
# ------------------------------------------------------------------------------

# Simple educational importance:
#
# For every retained basis function, allocate the absolute coefficient
# magnitude to each variable appearing in that basis.
#
# This is NOT the exact importance measure used by production MARS software.


mars_variable_importance <- function(
    model,
    selected_terms,
    coefficients
) {
  
  p <- length(
    model$feature_names
  )
  
  
  importance <- numeric(
    p
  )
  
  
  for (
    k in seq_along(
      selected_terms
    )
  ) {
    
    term_index <- selected_terms[k]
    
    
    if (
      term_index == 1
    ) {
      
      next
    }
    
    
    variables <- basis_variables(model$basis[[term_index]],model$basis)
    
    
    importance[
      variables
    ] <-
      importance[
        variables
      ] +
      abs(
        coefficients[k]
      )
  }
  
  
  names(
    importance
  ) <- model$feature_names
  
  
  if (
    sum(importance) >
    0
  ) {
    
    importance <-
      importance /
      sum(
        importance
      )
  }
  
  
  return(
    importance
  )
}


# ------------------------------------------------------------------------------
# 42. Calculate Importance
# ------------------------------------------------------------------------------

variable_importance <- mars_variable_importance(
  model = forward_train,
  selected_terms =
    selected_train_model$selected_terms,
  coefficients =
    selected_train_model$coefficients
)


variable_importance


barplot(
  variable_importance,
  ylab = "Relative Importance",
  main = "MARS Variable Importance"
)


# ==============================================================================
# PART XVI
#
# COMPARE WITH LINEAR REGRESSION
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Linear Regression Baseline
# ------------------------------------------------------------------------------

train_data <- data.frame(
  y = y_train,
  x1 = X_train[
    ,
    "x1"
  ],
  x2 = X_train[
    ,
    "x2"
  ],
  x3 = X_train[
    ,
    "x3"
  ]
)


test_data <- data.frame(
  x1 = X_test[
    ,
    "x1"
  ],
  x2 = X_test[
    ,
    "x2"
  ],
  x3 = X_test[
    ,
    "x3"
  ]
)


linear_model <- lm(
  y ~ x1 + x2 + x3,
  data = train_data
)


linear_prediction <- predict(
  linear_model,
  newdata = test_data
)


linear_test_MSE <- mean(
  (
    y_test -
      linear_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 44. Linear Model with Explicit Interaction
# ------------------------------------------------------------------------------

linear_interaction_model <- lm(
  y ~
    x1 *
    x2 +
    x3,
  data = train_data
)


linear_interaction_prediction <- predict(
  linear_interaction_model,
  newdata = test_data
)


linear_interaction_test_MSE <- mean(
  (
    y_test -
      linear_interaction_prediction
  )^2
)


# ------------------------------------------------------------------------------
# 45. Model Comparison
# ------------------------------------------------------------------------------

model_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Linear + Interaction",
    "Additive MARS",
    "Interaction MARS"
  ),
  
  Test_MSE = c(
    linear_test_MSE,
    linear_interaction_test_MSE,
    additive_test_MSE,
    interaction_test_MSE
  )
)


model_comparison$Test_RMSE <- sqrt(
  model_comparison$Test_MSE
)


model_comparison


# ==============================================================================
# PART XVII
#
# MANUAL K-FOLD CROSS-VALIDATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Cross-Validate Maximum Interaction Degree
# ------------------------------------------------------------------------------

mars_cv_degree <- function(
    X,
    y,
    degrees = 1:2,
    k = 5,
    max_terms = 15,
    number_knots = 10,
    penalty = 2,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  n <- nrow(
    X
  )
  
  
  fold_id <- sample(
    rep(
      seq_len(k),
      length.out = n
    )
  )
  
  
  fold_error <- matrix(
    NA_real_,
    nrow = length(
      degrees
    ),
    ncol = k
  )
  
  
  for (
    d in seq_along(
      degrees
    )
  ) {
    
    for (
      fold in seq_len(k)
    ) {
      
      training_indices <- which(
        fold_id != fold
      )
      
      
      validation_indices <- which(
        fold_id == fold
      )
      
      
      forward_fold <- mars_forward(
        X = X[
          training_indices,
          ,
          drop = FALSE
        ],
        y = y[
          training_indices
        ],
        max_terms = max_terms,
        max_degree = degrees[d],
        number_knots = number_knots
      )
      
      
      pruning_fold <- mars_backward_prune(
        forward_model = forward_fold,
        y = y[
          training_indices
        ],
        penalty = penalty
      )
      
      
      selected_fold <-
        pruning_fold$best_model
      
      
      validation_prediction <- predict_mars(
        model = forward_fold,
        X_new = X[
          validation_indices,
          ,
          drop = FALSE
        ],
        selected_terms =
          selected_fold$selected_terms,
        coefficients =
          selected_fold$coefficients
      )
      
      
      fold_error[
        d,
        fold
      ] <- mean(
        (
          y[
            validation_indices
          ] -
            validation_prediction
        )^2
      )
    }
  }
  
  
  mean_MSE <- rowMeans(
    fold_error
  )
  
  
  SE_MSE <- apply(
    fold_error,
    1,
    sd
  ) /
    sqrt(k)
  
  
  return(
    data.frame(
      Degree = degrees,
      CV_MSE = mean_MSE,
      CV_SE = SE_MSE
    )
  )
}


# ------------------------------------------------------------------------------
# 47. Run Cross Validation
# ------------------------------------------------------------------------------

CV_results <- mars_cv_degree(
  X = X_train,
  y = y_train,
  degrees = 1:2,
  k = 5,
  max_terms = 15,
  number_knots = 10,
  penalty = 2,
  seed = 123
)


CV_results


# ==============================================================================
# PART XVIII
#
# OPTIONAL VERIFICATION WITH EARTH
# ==============================================================================


# ------------------------------------------------------------------------------
# 48. Compare with Production MARS Implementation
# ------------------------------------------------------------------------------

# The earth package implements MARS.
#
# It is NOT required for the manual demo.


if (
  requireNamespace(
    "earth",
    quietly = TRUE
  )
) {
  
  earth_model <- earth::earth(
    y ~ .,
    data = train_data,
    degree = 2
  )
  
  
  earth_prediction <- predict(
    earth_model,
    newdata = test_data
  )
  
  
  earth_test_MSE <- mean(
    (
      y_test -
        earth_prediction
    )^2
  )
  
  
  cat(
    "Manual MARS test MSE:",
    round(
      interaction_test_MSE,
      4
    ),
    "\n"
  )
  
  
  cat(
    "earth MARS test MSE:",
    round(
      earth_test_MSE,
      4
    ),
    "\n"
  )
  
  
  print(
    summary(
      earth_model
    )
  )
}


# ==============================================================================
# PART XIX
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Print Final Model
# ------------------------------------------------------------------------------

cat(
  "MARS Summary\n"
)


cat(
  "------------\n"
)


cat(
  "Forward model terms:",
  length(
    forward_train$basis
  ),
  "\n"
)


cat(
  "Selected model terms:",
  length(
    selected_train_model$selected_terms
  ),
  "\n"
)


cat(
  "Selected interaction degree:",
  CV_results$Degree[
    which.min(
      CV_results$CV_MSE
    )
  ],
  "\n"
)


cat(
  "Linear regression test MSE:",
  round(
    linear_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Additive MARS test MSE:",
  round(
    additive_test_MSE,
    4
  ),
  "\n"
)


cat(
  "Interaction MARS test MSE:",
  round(
    interaction_test_MSE,
    4
  ),
  "\n"
)


cat(
  "\nSelected MARS basis functions:\n"
)


selected_descriptions <- sapply(
  selected_train_model$selected_terms,
  function(index) {
    
    forward_train$basis[[index]]$description
  }
)


print(
  data.frame(
    Basis = selected_descriptions,
    Coefficient =
      selected_train_model$coefficients
  )
)