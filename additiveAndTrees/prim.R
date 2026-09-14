# ==============================================================================
# PRIM: Bump Hunting
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Patient Rule Induction Method (PRIM)
#   - Bump hunting
#   - High-response regions
#   - Rectangular boxes
#   - Peeling
#   - Pasting
#   - Box support
#   - Box mean
#   - Peeling trajectory
#   - Cross-validation / validation selection
#   - Sequential bump extraction
#
# Core idea:
#
# PRIM searches for a rectangular region:
#
#   B = [a1, b1] x [a2, b2] x ... x [ap, bp]
#
# having an unusually large response mean:
#
#   mean(y | X in B)
#
# subject to the box containing enough observations.
#
# Unlike a regression tree, PRIM does not attempt to partition all of
# predictor space. It deliberately searches for "interesting" high-response
# regions.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE DATA WITH A HIGH-RESPONSE BUMP
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 600


x1 <- runif(
  n,
  min = 0,
  max = 1
)


x2 <- runif(
  n,
  min = 0,
  max = 1
)


x3 <- runif(
  n,
  min = 0,
  max = 1
)


# True high-response region:
#
#   0.55 <= x1 <= 0.90
#   0.20 <= x2 <= 0.55
#
# x3 is irrelevant.


inside_true_bump <-
  x1 >= 0.55 &
  x1 <= 0.90 &
  x2 >= 0.20 &
  x2 <= 0.55


baseline <- 2


bump_effect <- 6


y_true <-
  baseline +
  bump_effect *
  inside_true_bump


sigma <- 1.2


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
# 2. Inspect Response Distribution
# ------------------------------------------------------------------------------

summary(
  y
)


mean(
  y
)


mean(
  y[
    inside_true_bump
  ]
)


mean(
  y[
    !inside_true_bump
  ]
)


# ------------------------------------------------------------------------------
# 3. Plot the True Bump
# ------------------------------------------------------------------------------

plot(
  x1,
  x2,
  pch = 19,
  cex = 0.6,
  xlab = "x1",
  ylab = "x2",
  main = "Synthetic Bump-Hunting Data"
)


points(
  x1[
    inside_true_bump
  ],
  x2[
    inside_true_bump
  ],
  pch = 19,
  cex = 0.8
)


rect(
  xleft = 0.55,
  ybottom = 0.20,
  xright = 0.90,
  ytop = 0.55,
  lwd = 2
)


# ==============================================================================
# PART II
#
# REPRESENTING A PRIM BOX
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Create Initial Box
# ------------------------------------------------------------------------------

# The initial PRIM box contains the entire training dataset.


initial_box <- function(X) {
  
  X <- as.matrix(
    X
  )
  
  
  lower <- apply(
    X,
    2,
    min
  )
  
  
  upper <- apply(
    X,
    2,
    max
  )
  
  
  return(
    list(
      lower = lower,
      upper = upper
    )
  )
}


# ------------------------------------------------------------------------------
# 5. Determine Which Observations Are Inside a Box
# ------------------------------------------------------------------------------

inside_box <- function(
    X,
    box
) {
  
  X <- as.matrix(
    X
  )
  
  
  lower_check <- sweep(
    X,
    2,
    box$lower,
    FUN = ">="
  )
  
  
  upper_check <- sweep(
    X,
    2,
    box$upper,
    FUN = "<="
  )
  
  
  apply(
    lower_check &
      upper_check,
    1,
    all
  )
}


# ------------------------------------------------------------------------------
# 6. Box Statistics
# ------------------------------------------------------------------------------

box_statistics <- function(
    X,
    y,
    box
) {
  
  inside <- inside_box(
    X,
    box
  )
  
  
  number_inside <- sum(
    inside
  )
  
  
  if (number_inside == 0) {
    
    return(
      list(
        n = 0,
        support = 0,
        mean = NA_real_,
        median = NA_real_
      )
    )
  }
  
  
  return(
    list(
      n = number_inside,
      support = number_inside / nrow(X),
      mean = mean(
        y[
          inside
        ]
      ),
      median = median(
        y[
          inside
        ]
      )
    )
  )
}


# ------------------------------------------------------------------------------
# 7. Inspect Initial Box
# ------------------------------------------------------------------------------

box_0 <- initial_box(
  X
)


box_statistics(
  X,
  y,
  box_0
)


# ==============================================================================
# PART III
#
# PEELING
# ==============================================================================


# ------------------------------------------------------------------------------
# 8. One Peeling Step
# ------------------------------------------------------------------------------

# At each PRIM iteration, examine each variable and consider:
#
#   lower peel:
#       remove the lowest alpha fraction
#
#   upper peel:
#       remove the highest alpha fraction
#
# Select the peel that produces the largest mean response among the
# remaining observations.


prim_best_peel <- function(
    X,
    y,
    box,
    peel_alpha = 0.05,
    min_support = 0.10
) {
  
  X <- as.matrix(
    X
  )
  
  
  current_inside <- inside_box(
    X,
    box
  )
  
  
  current_indices <- which(
    current_inside
  )
  
  
  X_current <- X[
    current_indices,
    ,
    drop = FALSE
  ]
  
  
  p <- ncol(
    X
  )
  
  
  best_box <- NULL
  
  best_mean <- -Inf
  
  best_variable <- NA_integer_
  
  best_direction <- NA_character_
  
  best_cut <- NA_real_
  
  best_support <- NA_real_
  
  
  for (j in seq_len(p)) {
    
    xj <- X_current[
      ,
      j
    ]
    
    
    # ------------------------------------------------------------------------
    # Lower peel
    # ------------------------------------------------------------------------
    
    lower_cut <- as.numeric(
      quantile(
        xj,
        probs = peel_alpha,
        names = FALSE,
        type = 7
      )
    )
    
    
    lower_candidate <- box
    
    lower_candidate$lower[j] <-
      max(
        box$lower[j],
        lower_cut
      )
    
    
    lower_inside <- inside_box(
      X,
      lower_candidate
    )
    
    
    lower_support <- mean(
      lower_inside
    )
    
    
    if (
      sum(lower_inside) > 0 &&
      lower_support >= min_support
    ) {
      
      lower_mean <- mean(
        y[
          lower_inside
        ]
      )
      
      
      if (lower_mean > best_mean) {
        
        best_mean <- lower_mean
        
        best_box <- lower_candidate
        
        best_variable <- j
        
        best_direction <- "lower"
        
        best_cut <- lower_candidate$lower[j]
        
        best_support <- lower_support
      }
    }
    
    
    # ------------------------------------------------------------------------
    # Upper peel
    # ------------------------------------------------------------------------
    
    upper_cut <- as.numeric(
      quantile(
        xj,
        probs = 1 - peel_alpha,
        names = FALSE,
        type = 7
      )
    )
    
    
    upper_candidate <- box
    
    upper_candidate$upper[j] <-
      min(
        box$upper[j],
        upper_cut
      )
    
    
    upper_inside <- inside_box(
      X,
      upper_candidate
    )
    
    
    upper_support <- mean(
      upper_inside
    )
    
    
    if (
      sum(upper_inside) > 0 &&
      upper_support >= min_support
    ) {
      
      upper_mean <- mean(
        y[
          upper_inside
        ]
      )
      
      
      if (upper_mean > best_mean) {
        
        best_mean <- upper_mean
        
        best_box <- upper_candidate
        
        best_variable <- j
        
        best_direction <- "upper"
        
        best_cut <- upper_candidate$upper[j]
        
        best_support <- upper_support
      }
    }
  }
  
  
  if (is.null(best_box)) {
    
    return(NULL)
  }
  
  
  return(
    list(
      box = best_box,
      mean = best_mean,
      variable = best_variable,
      direction = best_direction,
      cut = best_cut,
      support = best_support
    )
  )
}


# ------------------------------------------------------------------------------
# 9. Inspect One Peel
# ------------------------------------------------------------------------------

first_peel <- prim_best_peel(
  X = X,
  y = y,
  box = box_0,
  peel_alpha = 0.05,
  min_support = 0.10
)


first_peel


colnames(X)[
  first_peel$variable
]


# ==============================================================================
# PART IV
#
# FULL PEELING TRAJECTORY
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. PRIM Peeling Algorithm
# ------------------------------------------------------------------------------

prim_peel <- function(
    X,
    y,
    peel_alpha = 0.05,
    min_support = 0.10,
    max_steps = 100
) {
  
  X <- as.matrix(
    X
  )
  
  
  current_box <- initial_box(
    X
  )
  
  
  boxes <- list(
    current_box
  )
  
  
  initial_stats <- box_statistics(
    X,
    y,
    current_box
  )
  
  
  trajectory <- data.frame(
    Step = 0,
    Variable = NA_character_,
    Direction = NA_character_,
    Cut = NA_real_,
    Support = initial_stats$support,
    Mean = initial_stats$mean
  )
  
  
  for (step in seq_len(max_steps)) {
    
    candidate <- prim_best_peel(
      X = X,
      y = y,
      box = current_box,
      peel_alpha = peel_alpha,
      min_support = min_support
    )
    
    
    if (is.null(candidate)) {
      break
    }
    
    
    current_stats <- box_statistics(
      X,
      y,
      current_box
    )
    
    
    # Stop if no improvement occurs.
    
    if (
      candidate$mean <=
      current_stats$mean
    ) {
      
      break
    }
    
    
    current_box <- candidate$box
    
    
    boxes[[length(boxes) + 1]] <- current_box
    
    
    trajectory <- rbind(
      trajectory,
      data.frame(
        Step = step,
        Variable = colnames(X)[
          candidate$variable
        ],
        Direction = candidate$direction,
        Cut = candidate$cut,
        Support = candidate$support,
        Mean = candidate$mean
      )
    )
    
    
    if (
      candidate$support <=
      min_support
    ) {
      
      break
    }
  }
  
  
  return(
    list(
      boxes = boxes,
      trajectory = trajectory
    )
  )
}


# ------------------------------------------------------------------------------
# 11. Run Peeling
# ------------------------------------------------------------------------------

peel_result <- prim_peel(
  X = X,
  y = y,
  peel_alpha = 0.05,
  min_support = 0.10,
  max_steps = 100
)


peel_result$trajectory


# ==============================================================================
# PART V
#
# PEELING PATH
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Plot Box Mean vs Peeling Step
# ------------------------------------------------------------------------------

plot(
  peel_result$trajectory$Step,
  peel_result$trajectory$Mean,
  type = "b",
  pch = 19,
  xlab = "Peeling Step",
  ylab = "Mean Response in Box",
  main = "PRIM Peeling Trajectory"
)


abline(
  h = mean(y),
  lty = 2
)


# ------------------------------------------------------------------------------
# 13. Plot Support vs Peeling Step
# ------------------------------------------------------------------------------

plot(
  peel_result$trajectory$Step,
  peel_result$trajectory$Support,
  type = "b",
  pch = 19,
  xlab = "Peeling Step",
  ylab = "Box Support",
  main = "PRIM Box Support"
)


# ------------------------------------------------------------------------------
# 14. Mean vs Support
# ------------------------------------------------------------------------------

plot(
  peel_result$trajectory$Support,
  peel_result$trajectory$Mean,
  type = "b",
  pch = 19,
  xlab = "Box Support",
  ylab = "Mean Response",
  main = "PRIM Mean-Support Tradeoff"
)


# ==============================================================================
# PART VI
#
# SELECT A BOX
# ==============================================================================


# ------------------------------------------------------------------------------
# 15. Choose a Box by Minimum Support
# ------------------------------------------------------------------------------

# For illustration, choose the box along the trajectory with the highest
# training mean.
#
# Later we will choose the peeling step using validation data.


best_training_step <- which.max(
  peel_result$trajectory$Mean
)


best_training_box <- peel_result$boxes[[best_training_step]]
  


best_training_statistics <- box_statistics(
  X,
  y,
  best_training_box
)


best_training_statistics


# Note:
#
# trajectory row 1 corresponds to Step = 0
# boxes[[1]] is also Step = 0
#
# Therefore which.max() on trajectory rows aligns directly with boxes.


# ------------------------------------------------------------------------------
# 16. Print Selected Box
# ------------------------------------------------------------------------------

print_box <- function(
    box,
    feature_names
) {
  
  for (j in seq_along(feature_names)) {
    
    cat(
      feature_names[j],
      ":",
      round(
        box$lower[j],
        4
      ),
      "to",
      round(
        box$upper[j],
        4
      ),
      "\n"
    )
  }
  
  
  invisible(NULL)
}


print_box(
  best_training_box,
  colnames(X)
)


# ==============================================================================
# PART VII
#
# VISUALIZE THE LEARNED BOX
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Observations Inside Learned Box
# ------------------------------------------------------------------------------

selected_inside <- inside_box(
  X,
  best_training_box
)


sum(
  selected_inside
)


mean(
  selected_inside
)


mean(
  y[
    selected_inside
  ]
)


# ------------------------------------------------------------------------------
# 18. Plot Learned Region in x1-x2 Space
# ------------------------------------------------------------------------------

plot(
  x1,
  x2,
  pch = 19,
  cex = 0.5,
  xlab = "x1",
  ylab = "x2",
  main = "PRIM Selected Box"
)


points(
  x1[
    selected_inside
  ],
  x2[
    selected_inside
  ],
  pch = 19,
  cex = 0.8
)


rect(
  xleft = best_training_box$lower[
    "x1"
  ],
  ybottom = best_training_box$lower[
    "x2"
  ],
  xright = best_training_box$upper[
    "x1"
  ],
  ytop = best_training_box$upper[
    "x2"
  ],
  lwd = 3
)


# ------------------------------------------------------------------------------
# 19. Compare Learned Region with True Region
# ------------------------------------------------------------------------------

true_positive_region <- inside_true_bump


predicted_region <- selected_inside


region_confusion <- table(
  True_Bump = true_positive_region,
  PRIM_Box = predicted_region
)


region_confusion


# ==============================================================================
# PART VIII
#
# REGION METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Region Precision and Recall
# ------------------------------------------------------------------------------

TP <- sum(
  predicted_region &
    true_positive_region
)


FP <- sum(
  predicted_region &
    !true_positive_region
)


FN <- sum(
  !predicted_region &
    true_positive_region
)


precision <- TP /
  (
    TP +
      FP
  )


recall <- TP /
  (
    TP +
      FN
  )


precision


recall


# ------------------------------------------------------------------------------
# 21. Jaccard Overlap
# ------------------------------------------------------------------------------

jaccard <- sum(
  predicted_region &
    true_positive_region
) /
  sum(
    predicted_region |
      true_positive_region
  )


jaccard


# ==============================================================================
# PART IX
#
# PASTING
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. One Pasting Step
# ------------------------------------------------------------------------------

# Peeling removes observations.
#
# Pasting attempts to add observations back if doing so improves the box
# according to the chosen criterion.
#
# Here, for each restricted boundary, we move it outward by a small amount
# determined by the original data and retain the best improving candidate.


prim_best_paste <- function(
    X,
    y,
    box,
    reference_box,
    paste_alpha = 0.05
) {
  
  X <- as.matrix(
    X
  )
  
  
  current_stats <- box_statistics(
    X,
    y,
    box
  )
  
  
  best_box <- NULL
  
  best_mean <- current_stats$mean
  
  best_variable <- NA_integer_
  
  best_direction <- NA_character_
  
  
  p <- ncol(
    X
  )
  
  
  for (j in seq_len(p)) {
    
    # ------------------------------------------------------------------------
    # Expand lower boundary downward
    # ------------------------------------------------------------------------
    
    if (
      box$lower[j] >
      reference_box$lower[j]
    ) {
      
      available <- X[
        X[, j] <
          box$lower[j] &
          X[, j] >=
          reference_box$lower[j],
        j
      ]
      
      
      if (length(available) > 0) {
        
        candidate <- box
        
        
        new_lower <- as.numeric(
          quantile(
            available,
            probs = max(
              0,
              1 - paste_alpha
            ),
            names = FALSE
          )
        )
        
        
        candidate$lower[j] <-
          max(
            reference_box$lower[j],
            new_lower
          )
        
        
        candidate_stats <- box_statistics(
          X,
          y,
          candidate
        )
        
        
        if (
          !is.na(candidate_stats$mean) &&
          candidate_stats$mean >
          best_mean
        ) {
          
          best_mean <- candidate_stats$mean
          
          best_box <- candidate
          
          best_variable <- j
          
          best_direction <- "lower"
        }
      }
    }
    
    
    # ------------------------------------------------------------------------
    # Expand upper boundary upward
    # ------------------------------------------------------------------------
    
    if (
      box$upper[j] <
      reference_box$upper[j]
    ) {
      
      available <- X[
        X[, j] >
          box$upper[j] &
          X[, j] <=
          reference_box$upper[j],
        j
      ]
      
      
      if (length(available) > 0) {
        
        candidate <- box
        
        
        new_upper <- as.numeric(
          quantile(
            available,
            probs = min(
              1,
              paste_alpha
            ),
            names = FALSE
          )
        )
        
        
        candidate$upper[j] <-
          min(
            reference_box$upper[j],
            new_upper
          )
        
        
        candidate_stats <- box_statistics(
          X,
          y,
          candidate
        )
        
        
        if (
          !is.na(candidate_stats$mean) &&
          candidate_stats$mean >
          best_mean
        ) {
          
          best_mean <- candidate_stats$mean
          
          best_box <- candidate
          
          best_variable <- j
          
          best_direction <- "upper"
        }
      }
    }
  }
  
  
  if (is.null(best_box)) {
    return(NULL)
  }
  
  
  return(
    list(
      box = best_box,
      mean = best_mean,
      variable = best_variable,
      direction = best_direction
    )
  )
}


# ------------------------------------------------------------------------------
# 23. Pasting Algorithm
# ------------------------------------------------------------------------------

prim_paste <- function(
    X,
    y,
    box,
    reference_box,
    paste_alpha = 0.05,
    max_steps = 50
) {
  
  current_box <- box
  
  
  history <- data.frame(
    Step = 0,
    Mean = box_statistics(
      X,
      y,
      current_box
    )$mean
  )
  
  
  for (step in seq_len(max_steps)) {
    
    candidate <- prim_best_paste(
      X = X,
      y = y,
      box = current_box,
      reference_box = reference_box,
      paste_alpha = paste_alpha
    )
    
    
    if (is.null(candidate)) {
      break
    }
    
    
    current_box <- candidate$box
    
    
    history <- rbind(
      history,
      data.frame(
        Step = step,
        Mean = candidate$mean
      )
    )
  }
  
  
  return(
    list(
      box = current_box,
      history = history
    )
  )
}


# ------------------------------------------------------------------------------
# 24. Run Pasting
# ------------------------------------------------------------------------------

paste_result <- prim_paste(
  X = X,
  y = y,
  box = best_training_box,
  reference_box = box_0,
  paste_alpha = 0.05,
  max_steps = 50
)


pasted_box <- paste_result$box


print_box(
  pasted_box,
  colnames(X)
)


box_statistics(
  X,
  y,
  pasted_box
)


# ==============================================================================
# PART X
#
# TRAIN-VALIDATION-TEST SPLIT
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. Split Data
# ------------------------------------------------------------------------------

set.seed(456)


all_indices <- seq_len(
  n
)


train_indices <- sample(
  all_indices,
  size = floor(
    0.60 * n
  )
)


remaining_indices <- setdiff(
  all_indices,
  train_indices
)


validation_indices <- sample(
  remaining_indices,
  size = floor(
    0.50 *
      length(
        remaining_indices
      )
  )
)


test_indices <- setdiff(
  remaining_indices,
  validation_indices
)


X_train <- X[
  train_indices,
  ,
  drop = FALSE
]


y_train <- y[
  train_indices
]


X_validation <- X[
  validation_indices,
  ,
  drop = FALSE
]


y_validation <- y[
  validation_indices
]


X_test <- X[
  test_indices,
  ,
  drop = FALSE
]


y_test <- y[
  test_indices
]


# ==============================================================================
# PART XI
#
# VALIDATION-BASED SELECTION OF PEELING STEP
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Fit Peeling Sequence on Training Data
# ------------------------------------------------------------------------------

train_peel <- prim_peel(
  X = X_train,
  y = y_train,
  peel_alpha = 0.05,
  min_support = 0.10,
  max_steps = 100
)


# ------------------------------------------------------------------------------
# 27. Evaluate Every Training Box on Validation Data
# ------------------------------------------------------------------------------

validation_results <- data.frame(
  Step = seq_along(
    train_peel$boxes
  ) - 1,
  Train_Support = NA_real_,
  Train_Mean = NA_real_,
  Validation_Support = NA_real_,
  Validation_Mean = NA_real_
)


for (
  i in seq_along(
    train_peel$boxes
  )
) {
  
  box_i <- train_peel$boxes[[i]]
    
  
  
  train_stats <- box_statistics(
    X_train,
    y_train,
    box_i
  )
  
  
  validation_stats <- box_statistics(
    X_validation,
    y_validation,
    box_i
  )
  
  
  validation_results$Train_Support[i] <-
    train_stats$support
  
  
  validation_results$Train_Mean[i] <-
    train_stats$mean
  
  
  validation_results$Validation_Support[i] <-
    validation_stats$support
  
  
  validation_results$Validation_Mean[i] <-
    validation_stats$mean
}


validation_results


# ------------------------------------------------------------------------------
# 28. Choose Best Validation Box
# ------------------------------------------------------------------------------

# Ignore candidate boxes containing no validation observations.


valid_candidates <- which(
  !is.na(
    validation_results$Validation_Mean
  ) &
    validation_results$Validation_Support >
    0
)


best_validation_index <- valid_candidates[
  which.max(
    validation_results$Validation_Mean[
      valid_candidates
    ]
  )
]


selected_box <- train_peel$boxes[[best_validation_index]]
  

validation_results[
  best_validation_index,
]


# ------------------------------------------------------------------------------
# 29. Plot Training and Validation Mean Along Path
# ------------------------------------------------------------------------------

matplot(
  validation_results$Step,
  cbind(
    validation_results$Train_Mean,
    validation_results$Validation_Mean
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
  xlab = "Peeling Step",
  ylab = "Box Mean",
  main = "Selecting PRIM Complexity"
)


legend(
  "bottomright",
  legend = c(
    "Training",
    "Validation"
  ),
  pch = c(
    19,
    17
  ),
  lty = c(
    1,
    2
  )
)


# ==============================================================================
# PART XII
#
# TEST EVALUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Selected Box on Test Data
# ------------------------------------------------------------------------------

test_stats <- box_statistics(
  X_test,
  y_test,
  selected_box
)


test_stats


# ------------------------------------------------------------------------------
# 31. Compare Overall and Box Mean
# ------------------------------------------------------------------------------

data.frame(
  Region = c(
    "Entire Test Set",
    "PRIM Box"
  ),
  Mean_Response = c(
    mean(
      y_test
    ),
    test_stats$mean
  ),
  Support = c(
    1,
    test_stats$support
  )
)


# ==============================================================================
# PART XIII
#
# PRIM VERSUS A SIMPLE THRESHOLD SEARCH
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Best Single-Predictor Threshold
# ------------------------------------------------------------------------------

# PRIM is multivariate and rectangular.
#
# Compare it with a much simpler search over one-sided rules.


best_single_threshold <- function(
    X,
    y,
    min_support = 0.10
) {
  
  X <- as.matrix(
    X
  )
  
  
  best_mean <- -Inf
  
  best_variable <- NA_integer_
  
  best_cut <- NA_real_
  
  best_direction <- NA_character_
  
  
  for (
    j in seq_len(
      ncol(X)
    )
  ) {
    
    cuts <- sort(
      unique(
        X[, j]
      )
    )
    
    
    for (cut in cuts) {
      
      lower_rule <- X[, j] <= cut
      
      if (
        mean(lower_rule) >= min_support
      ) {
        
        response_mean <- mean(
          y[
            lower_rule
          ]
        )
        
        
        if (
          response_mean >
          best_mean
        ) {
          
          best_mean <- response_mean
          
          best_variable <- j
          
          best_cut <- cut
          
          best_direction <- "<="
        }
      }
      
      
      upper_rule <- X[, j] >= cut
      
      
      if (
        mean(upper_rule) >= min_support
      ) {
        
        response_mean <- mean(
          y[
            upper_rule
          ]
        )
        
        
        if (
          response_mean >
          best_mean
        ) {
          
          best_mean <- response_mean
          
          best_variable <- j
          
          best_cut <- cut
          
          best_direction <- ">="
        }
      }
    }
  }
  
  
  return(
    list(
      variable = best_variable,
      cut = best_cut,
      direction = best_direction,
      mean = best_mean
    )
  )
}


single_threshold <- best_single_threshold(
  X_train,
  y_train,
  min_support = 0.10
)


single_threshold


colnames(X)[
  single_threshold$variable
]


# ==============================================================================
# PART XIV
#
# VARIABLE USAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Count Peels by Predictor
# ------------------------------------------------------------------------------

peel_variables <- train_peel$trajectory$Variable


peel_variables <- peel_variables[
  !is.na(
    peel_variables
  )
]


peel_frequency <- table(
  factor(
    peel_variables,
    levels = colnames(X)
  )
)


peel_frequency


barplot(
  peel_frequency,
  ylab = "Number of Peels",
  main = "PRIM Variable Usage"
)


# ==============================================================================
# PART XV
#
# PEELING PARAMETER
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Compare Peel Fractions
# ------------------------------------------------------------------------------

peel_values <- c(
  0.02,
  0.05,
  0.10,
  0.15
)


peel_parameter_results <- data.frame(
  Peel_Alpha = peel_values,
  Number_Steps = NA_integer_,
  Best_Training_Mean = NA_real_,
  Final_Support = NA_real_
)


for (
  i in seq_along(
    peel_values
  )
) {
  
  result_i <- prim_peel(
    X = X_train,
    y = y_train,
    peel_alpha = peel_values[i],
    min_support = 0.10,
    max_steps = 100
  )
  
  
  peel_parameter_results$Number_Steps[i] <-
    nrow(
      result_i$trajectory
    ) - 1
  
  
  peel_parameter_results$Best_Training_Mean[i] <-
    max(
      result_i$trajectory$Mean
    )
  
  
  peel_parameter_results$Final_Support[i] <-
    tail(
      result_i$trajectory$Support,
      1
    )
}


peel_parameter_results


# ==============================================================================
# PART XVI
#
# MULTIPLE BUMPS
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Generate Data with Two High-Response Regions
# ------------------------------------------------------------------------------

set.seed(789)


n2 <- 800


z1 <- runif(
  n2
)


z2 <- runif(
  n2
)


z3 <- runif(
  n2
)


bump_1 <-
  z1 >= 0.10 &
  z1 <= 0.35 &
  z2 >= 0.60 &
  z2 <= 0.90


bump_2 <-
  z1 >= 0.65 &
  z1 <= 0.90 &
  z2 >= 0.10 &
  z2 <= 0.40


y2 <-
  2 +
  7 *
  bump_1 +
  5 *
  bump_2 +
  rnorm(
    n2,
    sd = 1
  )


X2 <- cbind(
  z1 = z1,
  z2 = z2,
  z3 = z3
)


# ------------------------------------------------------------------------------
# 36. Find First Bump
# ------------------------------------------------------------------------------

first_bump_fit <- prim_peel(
  X = X2,
  y = y2,
  peel_alpha = 0.05,
  min_support = 0.05,
  max_steps = 100
)


first_bump_index <- which.max(
  first_bump_fit$trajectory$Mean
)


first_bump_box <- first_bump_fit$boxes[[first_bump_index]]
    


first_bump_membership <- inside_box(
  X2,
  first_bump_box
)


# ------------------------------------------------------------------------------
# 37. Remove First Bump and Search Again
# ------------------------------------------------------------------------------

remaining <- !first_bump_membership


second_bump_fit <- prim_peel(
  X = X2[
    remaining,
    ,
    drop = FALSE
  ],
  y = y2[
    remaining
  ],
  peel_alpha = 0.05,
  min_support = 0.05,
  max_steps = 100
)


second_bump_index <- which.max(
  second_bump_fit$trajectory$Mean
)


second_bump_box <- second_bump_fit$boxes[[second_bump_index]]


# ------------------------------------------------------------------------------
# 38. Inspect the Two Boxes
# ------------------------------------------------------------------------------

cat(
  "First bump:\n"
)


print_box(
  first_bump_box,
  colnames(X2)
)


cat(
  "\nSecond bump:\n"
)


print_box(
  second_bump_box,
  colnames(X2)
)


# ==============================================================================
# PART XVII
#
# VISUALIZE MULTIPLE BUMPS
# ==============================================================================


# ------------------------------------------------------------------------------
# 39. Plot True Multiple-Bump Data
# ------------------------------------------------------------------------------

plot(
  z1,
  z2,
  pch = 19,
  cex = 0.4,
  xlab = "z1",
  ylab = "z2",
  main = "Sequential PRIM Bump Hunting"
)


rect(
  xleft = first_bump_box$lower[
    "z1"
  ],
  ybottom = first_bump_box$lower[
    "z2"
  ],
  xright = first_bump_box$upper[
    "z1"
  ],
  ytop = first_bump_box$upper[
    "z2"
  ],
  lwd = 2
)


# Second bump was estimated using the reduced dataset,
# but its bounds remain in the original predictor coordinate system.

rect(
  xleft = second_bump_box$lower[
    "z1"
  ],
  ybottom = second_bump_box$lower[
    "z2"
  ],
  xright = second_bump_box$upper[
    "z1"
  ],
  ytop = second_bump_box$upper[
    "z2"
  ],
  lwd = 2,
  lty = 2
)


# ==============================================================================
# PART XVIII
#
# SIMPLE RULE EXTRACTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Convert Box into Human-Readable Rules
# ------------------------------------------------------------------------------

box_rules <- function(
    box,
    feature_names,
    digits = 3
) {
  
  rules <- character(
    length(
      feature_names
    )
  )
  
  
  for (
    j in seq_along(
      feature_names
    )
  ) {
    
    rules[j] <- paste0(
      round(
        box$lower[j],
        digits
      ),
      " <= ",
      feature_names[j],
      " <= ",
      round(
        box$upper[j],
        digits
      )
    )
  }
  
  
  return(
    rules
  )
}


selected_rules <- box_rules(
  selected_box,
  colnames(X)
)


selected_rules


cat(
  paste(
    selected_rules,
    collapse = "\nAND\n"
  ),
  "\n"
)


# ==============================================================================
# PART XIX
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Final Output
# ------------------------------------------------------------------------------

selected_train_stats <- box_statistics(
  X_train,
  y_train,
  selected_box
)


selected_validation_stats <- box_statistics(
  X_validation,
  y_validation,
  selected_box
)


selected_test_stats <- box_statistics(
  X_test,
  y_test,
  selected_box
)


cat(
  "PRIM Bump Hunting Summary\n"
)


cat(
  "-------------------------\n"
)


cat(
  "Overall training response mean:",
  round(
    mean(
      y_train
    ),
    4
  ),
  "\n"
)


cat(
  "Selected training box mean:",
  round(
    selected_train_stats$mean,
    4
  ),
  "\n"
)


cat(
  "Selected training box support:",
  round(
    selected_train_stats$support,
    4
  ),
  "\n"
)


cat(
  "Validation box mean:",
  round(
    selected_validation_stats$mean,
    4
  ),
  "\n"
)


cat(
  "Validation box support:",
  round(
    selected_validation_stats$support,
    4
  ),
  "\n"
)


cat(
  "Test box mean:",
  round(
    selected_test_stats$mean,
    4
  ),
  "\n"
)


cat(
  "Test box support:",
  round(
    selected_test_stats$support,
    4
  ),
  "\n"
)


cat(
  "\nSelected rules:\n"
)


cat(
  paste(
    selected_rules,
    collapse = "\n"
  ),
  "\n"
)