# ==============================================================================
# Subset Selection
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#   An Introduction to Statistical Learning
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Generate Data
# ------------------------------------------------------------------------------

set.seed(123)

n <- 200

x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
x4 <- rnorm(n)
x5 <- rnorm(n)
x6 <- rnorm(n)

# Only x1, x2, and x4 truly affect y

beta_0 <- 2
beta_1 <- 3
beta_2 <- -2
beta_4 <- 1.5

sigma <- 2

epsilon <- rnorm(n, mean = 0, sd = sigma)

y <- beta_0 +
  beta_1 * x1 +
  beta_2 * x2 +
  beta_4 * x4 +
  epsilon

data <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  x3 = x3,
  x4 = x4,
  x5 = x5,
  x6 = x6
)

head(data)
summary(data)


# ------------------------------------------------------------------------------
# 2. Explore the Data
# ------------------------------------------------------------------------------

pairs(
  data,
  pch = 19,
  main = "Subset Selection Data"
)

cor(data)


# ------------------------------------------------------------------------------
# 3. Candidate Predictors
# ------------------------------------------------------------------------------

predictors <- setdiff(
  names(data),
  "y"
)

predictors

p <- length(predictors)


# ------------------------------------------------------------------------------
# 4. Fit a Model from a Predictor Subset
# ------------------------------------------------------------------------------

fit_subset <- function(selected_predictors, data) {
  
  if (length(selected_predictors) == 0) {
    
    formula <- y ~ 1
    
  } else {
    
    formula <- as.formula(
      paste(
        "y ~",
        paste(selected_predictors, collapse = " + ")
      )
    )
  }
  
  model <- lm(
    formula,
    data = data
  )
  
  residuals <- resid(model)
  
  RSS <- sum(residuals^2)
  
  n <- nrow(data)
  
  k <- length(coef(model))
  
  TSS <- sum(
    (data$y - mean(data$y))^2
  )
  
  R2 <- 1 - RSS / TSS
  
  adjusted_R2 <- 1 -
    (RSS / (n - k)) /
    (TSS / (n - 1))
  
  sigma2_mle <- RSS / n
  
  AIC_manual <- n * log(sigma2_mle) + 2 * k
  
  BIC_manual <- n * log(sigma2_mle) + log(n) * k
  
  return(
    list(
      model = model,
      predictors = selected_predictors,
      size = length(selected_predictors),
      RSS = RSS,
      R2 = R2,
      adjusted_R2 = adjusted_R2,
      AIC = AIC_manual,
      BIC = BIC_manual
    )
  )
}


# ------------------------------------------------------------------------------
# 5. Intercept-Only Model
# ------------------------------------------------------------------------------

null_fit <- fit_subset(
  selected_predictors = character(0),
  data = data
)

null_fit$RSS
null_fit$R2


# ------------------------------------------------------------------------------
# 6. Exhaustive Best Subset Selection
# ------------------------------------------------------------------------------

all_results <- list()

result_index <- 1

for (subset_size in 0:p) {
  
  if (subset_size == 0) {
    
    fit <- fit_subset(
      selected_predictors = character(0),
      data = data
    )
    
    all_results[[result_index]] <- fit
    
    result_index <- result_index + 1
    
  } else {
    
    subsets <- combn(
      predictors,
      subset_size,
      simplify = FALSE
    )
    
    for (selected_predictors in subsets) {
      
      fit <- fit_subset(
        selected_predictors,
        data
      )
      
      all_results[[result_index]] <- fit
      
      result_index <- result_index + 1
    }
  }
}


# ------------------------------------------------------------------------------
# 7. Convert Results to a Data Frame
# ------------------------------------------------------------------------------

results_table <- data.frame(
  Size = sapply(
    all_results,
    function(x) x$size
  ),
  
  Predictors = sapply(
    all_results,
    function(x) {
      if (length(x$predictors) == 0) {
        "(Intercept only)"
      } else {
        paste(
          x$predictors,
          collapse = ", "
        )
      }
    }
  ),
  
  RSS = sapply(
    all_results,
    function(x) x$RSS
  ),
  
  R2 = sapply(
    all_results,
    function(x) x$R2
  ),
  
  Adjusted_R2 = sapply(
    all_results,
    function(x) x$adjusted_R2
  ),
  
  AIC = sapply(
    all_results,
    function(x) x$AIC
  ),
  
  BIC = sapply(
    all_results,
    function(x) x$BIC
  )
)

head(results_table)

nrow(results_table)


# ------------------------------------------------------------------------------
# 8. Best Model of Each Size
# ------------------------------------------------------------------------------

best_by_size <- do.call(
  rbind,
  lapply(
    0:p,
    function(size) {
      
      candidates <- results_table[
        results_table$Size == size,
      ]
      
      candidates[
        which.min(candidates$RSS),
      ]
    }
  )
)

rownames(best_by_size) <- NULL

best_by_size


# ------------------------------------------------------------------------------
# 9. Plot RSS by Model Size
# ------------------------------------------------------------------------------

plot(
  best_by_size$Size,
  best_by_size$RSS,
  type = "b",
  pch = 19,
  xlab = "Number of Predictors",
  ylab = "RSS",
  main = "Best Subset Selection: RSS"
)


# ------------------------------------------------------------------------------
# 10. Plot R-Squared
# ------------------------------------------------------------------------------

plot(
  best_by_size$Size,
  best_by_size$R2,
  type = "b",
  pch = 19,
  xlab = "Number of Predictors",
  ylab = "R-Squared",
  main = "Best Subset Selection: R-Squared"
)


# ------------------------------------------------------------------------------
# 11. Plot Adjusted R-Squared
# ------------------------------------------------------------------------------

plot(
  best_by_size$Size,
  best_by_size$Adjusted_R2,
  type = "b",
  pch = 19,
  xlab = "Number of Predictors",
  ylab = "Adjusted R-Squared",
  main = "Best Subset Selection: Adjusted R-Squared"
)


# ------------------------------------------------------------------------------
# 12. Plot AIC
# ------------------------------------------------------------------------------

plot(
  best_by_size$Size,
  best_by_size$AIC,
  type = "b",
  pch = 19,
  xlab = "Number of Predictors",
  ylab = "AIC",
  main = "Best Subset Selection: AIC"
)


# ------------------------------------------------------------------------------
# 13. Plot BIC
# ------------------------------------------------------------------------------

plot(
  best_by_size$Size,
  best_by_size$BIC,
  type = "b",
  pch = 19,
  xlab = "Number of Predictors",
  ylab = "BIC",
  main = "Best Subset Selection: BIC"
)


# ------------------------------------------------------------------------------
# 14. Select Best Model by Adjusted R-Squared
# ------------------------------------------------------------------------------

best_adjusted_R2 <- results_table[
  which.max(
    results_table$Adjusted_R2
  ),
]

best_adjusted_R2


# ------------------------------------------------------------------------------
# 15. Select Best Model by AIC
# ------------------------------------------------------------------------------

best_AIC <- results_table[
  which.min(
    results_table$AIC
  ),
]

best_AIC


# ------------------------------------------------------------------------------
# 16. Select Best Model by BIC
# ------------------------------------------------------------------------------

best_BIC <- results_table[
  which.min(
    results_table$BIC
  ),
]

best_BIC


# ------------------------------------------------------------------------------
# 17. Retrieve the Best BIC Model
# ------------------------------------------------------------------------------

best_BIC_index <- which.min(
  sapply(
    all_results,
    function(x) x$BIC
  )
)

best_BIC_fit <- all_results[[best_BIC_index]]

best_BIC_fit$predictors

summary(best_BIC_fit$model)


# ------------------------------------------------------------------------------
# 18. Forward Stepwise Selection
# ------------------------------------------------------------------------------

remaining_predictors <- predictors

selected_predictors <- character(0)

forward_results <- list()

for (step in 1:p) {
  
  candidate_fits <- list()
  
  for (candidate in remaining_predictors) {
    
    candidate_set <- c(
      selected_predictors,
      candidate
    )
    
    candidate_fits[[candidate]] <- fit_subset(
      candidate_set,
      data
    )
  }
  
  candidate_RSS <- sapply(
    candidate_fits,
    function(x) x$RSS
  )
  
  best_candidate <- names(
    which.min(candidate_RSS)
  )
  
  selected_predictors <- c(
    selected_predictors,
    best_candidate
  )
  
  remaining_predictors <- setdiff(
    remaining_predictors,
    best_candidate
  )
  
  forward_results[[step]] <- candidate_fits[[best_candidate]]
}


# ------------------------------------------------------------------------------
# 19. Forward Selection Results
# ------------------------------------------------------------------------------

forward_table <- data.frame(
  Step = 1:p,
  
  Predictors = sapply(
    forward_results,
    function(x) {
      paste(
        x$predictors,
        collapse = ", "
      )
    }
  ),
  
  RSS = sapply(
    forward_results,
    function(x) x$RSS
  ),
  
  Adjusted_R2 = sapply(
    forward_results,
    function(x) x$adjusted_R2
  ),
  
  AIC = sapply(
    forward_results,
    function(x) x$AIC
  ),
  
  BIC = sapply(
    forward_results,
    function(x) x$BIC
  )
)

forward_table


# ------------------------------------------------------------------------------
# 20. Backward Stepwise Selection
# ------------------------------------------------------------------------------

selected_predictors <- predictors

backward_results <- list()

step <- 1

while (length(selected_predictors) > 0) {
  
  current_fit <- fit_subset(
    selected_predictors,
    data
  )
  
  backward_results[[step]] <- current_fit
  
  if (length(selected_predictors) == 1) {
    break
  }
  
  candidate_fits <- list()
  
  for (candidate in selected_predictors) {
    
    candidate_set <- setdiff(
      selected_predictors,
      candidate
    )
    
    candidate_fits[[candidate]] <- fit_subset(
      candidate_set,
      data
    )
  }
  
  candidate_RSS <- sapply(
    candidate_fits,
    function(x) x$RSS
  )
  
  removed_predictor <- names(
    which.min(candidate_RSS)
  )
  
  selected_predictors <- setdiff(
    selected_predictors,
    removed_predictor
  )
  
  step <- step + 1
}


# ------------------------------------------------------------------------------
# 21. Backward Selection Results
# ------------------------------------------------------------------------------

backward_table <- data.frame(
  Size = sapply(
    backward_results,
    function(x) x$size
  ),
  
  Predictors = sapply(
    backward_results,
    function(x) {
      paste(
        x$predictors,
        collapse = ", "
      )
    }
  ),
  
  RSS = sapply(
    backward_results,
    function(x) x$RSS
  ),
  
  Adjusted_R2 = sapply(
    backward_results,
    function(x) x$adjusted_R2
  ),
  
  AIC = sapply(
    backward_results,
    function(x) x$AIC
  ),
  
  BIC = sapply(
    backward_results,
    function(x) x$BIC
  )
)

backward_table


# ------------------------------------------------------------------------------
# 22. Compare Selection Methods
# ------------------------------------------------------------------------------

best_by_size

forward_table

backward_table


# ------------------------------------------------------------------------------
# 23. Compare Selected Model with True Model
# ------------------------------------------------------------------------------

true_model <- lm(
  y ~ x1 + x2 + x4,
  data = data
)

summary(true_model)

summary(
  best_BIC_fit$model
)


# ------------------------------------------------------------------------------
# 24. Prediction with Selected Model
# ------------------------------------------------------------------------------

new_data <- data.frame(
  x1 = 0.5,
  x2 = -1,
  x3 = 0,
  x4 = 1,
  x5 = 0,
  x6 = 0
)

predict(
  best_BIC_fit$model,
  newdata = new_data
)