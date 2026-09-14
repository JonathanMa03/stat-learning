# ==============================================================================
# Multiple Linear Regression
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

x1 <- rnorm(n, mean = 0, sd = 1)
x2 <- runif(n, min = -2, max = 2)
x3 <- rnorm(n, mean = 2, sd = 1.5)

beta_0 <- 2
beta_1 <- 3
beta_2 <- -1.5
beta_3 <- 0.75

sigma <- 2

epsilon <- rnorm(n, mean = 0, sd = sigma)

y <- beta_0 +
  beta_1 * x1 +
  beta_2 * x2 +
  beta_3 * x3 +
  epsilon

data <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  x3 = x3
)

head(data)
summary(data)


# ------------------------------------------------------------------------------
# 2. Explore the Data
# ------------------------------------------------------------------------------

pairs(
  data,
  pch = 19,
  main = "Multiple Linear Regression Data"
)

cor(data)


# ------------------------------------------------------------------------------
# 3. Construct the Design Matrix
# ------------------------------------------------------------------------------

# First column corresponds to the intercept

X <- cbind(
  Intercept = 1,
  x1 = x1,
  x2 = x2,
  x3 = x3
)

head(X)

dim(X)

p <- ncol(X)


# ------------------------------------------------------------------------------
# 4. Estimate Regression Coefficients
# ------------------------------------------------------------------------------

# Least-squares estimator:
#
# beta_hat = (X'X)^(-1) X'y

XtX <- t(X) %*% X
Xty <- t(X) %*% y

beta_hat <- solve(XtX, Xty)

beta_hat


# ------------------------------------------------------------------------------
# 5. Fitted Values
# ------------------------------------------------------------------------------

y_hat <- X %*% beta_hat

head(y_hat)

plot(
  y,
  y_hat,
  pch = 19,
  xlab = "Observed Values",
  ylab = "Fitted Values",
  main = "Observed vs Fitted"
)

abline(
  a = 0,
  b = 1,
  lwd = 2,
  lty = 2
)


# ------------------------------------------------------------------------------
# 6. Residuals
# ------------------------------------------------------------------------------

residuals <- y - y_hat

head(residuals)

RSS <- sum(residuals^2)

y_bar <- mean(y)

TSS <- sum((y - y_bar)^2)

ESS <- sum((y_hat - y_bar)^2)

RSS
TSS
ESS

# Verify decomposition
TSS
ESS + RSS


# ------------------------------------------------------------------------------
# 7. Estimate Error Variance
# ------------------------------------------------------------------------------

df_residual <- n - p

sigma2_hat <- RSS / df_residual
sigma_hat <- sqrt(sigma2_hat)

sigma2_hat
sigma_hat


# ------------------------------------------------------------------------------
# 8. Variance-Covariance Matrix
# ------------------------------------------------------------------------------

# Var(beta_hat) = sigma^2 (X'X)^(-1)

vcov_beta <- sigma2_hat * solve(XtX)

vcov_beta


# ------------------------------------------------------------------------------
# 9. Standard Errors
# ------------------------------------------------------------------------------

standard_errors <- sqrt(diag(vcov_beta))

standard_errors


# ------------------------------------------------------------------------------
# 10. t-Statistics
# ------------------------------------------------------------------------------

# Test:
#
# H0: beta_j = 0
# H1: beta_j != 0

t_statistics <- beta_hat / standard_errors

t_statistics


# ------------------------------------------------------------------------------
# 11. p-Values
# ------------------------------------------------------------------------------

p_values <- 2 * pt(
  -abs(t_statistics),
  df = df_residual
)

p_values


# ------------------------------------------------------------------------------
# 12. Confidence Intervals
# ------------------------------------------------------------------------------

alpha <- 0.05

t_critical <- qt(
  1 - alpha / 2,
  df = df_residual
)

lower <- beta_hat - t_critical * standard_errors
upper <- beta_hat + t_critical * standard_errors

confidence_intervals <- cbind(
  Lower = lower,
  Estimate = beta_hat,
  Upper = upper
)

confidence_intervals


# ------------------------------------------------------------------------------
# 13. Coefficient Table
# ------------------------------------------------------------------------------

coefficient_table <- data.frame(
  Estimate = as.vector(beta_hat),
  Std_Error = standard_errors,
  t_value = as.vector(t_statistics),
  p_value = as.vector(p_values)
)

rownames(coefficient_table) <- colnames(X)

coefficient_table


# ------------------------------------------------------------------------------
# 14. R-Squared
# ------------------------------------------------------------------------------

R2 <- 1 - RSS / TSS

R2


# ------------------------------------------------------------------------------
# 15. Adjusted R-Squared
# ------------------------------------------------------------------------------

adjusted_R2 <- 1 -
  (RSS / (n - p)) /
  (TSS / (n - 1))

adjusted_R2


# ------------------------------------------------------------------------------
# 16. Overall F-Test
# ------------------------------------------------------------------------------

# H0: beta_1 = beta_2 = ... = beta_(p-1) = 0
#
# H1: At least one slope coefficient is nonzero

df_regression <- p - 1

MSR <- ESS / df_regression
MSE <- RSS / df_residual

F_statistic <- MSR / MSE

F_p_value <- pf(
  F_statistic,
  df1 = df_regression,
  df2 = df_residual,
  lower.tail = FALSE
)

F_statistic
F_p_value


# ------------------------------------------------------------------------------
# 17. Hat Matrix
# ------------------------------------------------------------------------------

# H = X (X'X)^(-1) X'

H <- X %*% solve(XtX) %*% t(X)

dim(H)

# Verify fitted values
y_hat_H <- H %*% y

max(abs(y_hat - y_hat_H))

# Leverage values
leverage <- diag(H)

summary(leverage)


# ------------------------------------------------------------------------------
# 18. Residual Diagnostics
# ------------------------------------------------------------------------------

plot(
  y_hat,
  residuals,
  pch = 19,
  xlab = "Fitted Values",
  ylab = "Residuals",
  main = "Residuals vs Fitted"
)

abline(
  h = 0,
  lty = 2
)

hist(
  residuals,
  breaks = 20,
  main = "Residual Distribution",
  xlab = "Residual"
)

qqnorm(
  residuals,
  pch = 19,
  main = "Normal Q-Q Plot"
)

qqline(residuals)


# ------------------------------------------------------------------------------
# 19. Leverage Plot
# ------------------------------------------------------------------------------

plot(
  leverage,
  type = "h",
  xlab = "Observation",
  ylab = "Leverage",
  main = "Leverage Values"
)

abline(
  h = 2 * p / n,
  lty = 2
)


# ------------------------------------------------------------------------------
# 20. Prediction
# ------------------------------------------------------------------------------

# New observation:
# x1 = 0.5
# x2 = -1
# x3 = 2

x_new <- c(
  1,
  0.5,
  -1,
  2
)

y_new_hat <- as.numeric(
  x_new %*% beta_hat
)

y_new_hat


# ------------------------------------------------------------------------------
# 21. Prediction Standard Error
# ------------------------------------------------------------------------------

# Standard error for estimated conditional mean

se_mean <- sqrt(
  sigma2_hat *
    as.numeric(
      t(x_new) %*%
        solve(XtX) %*%
        x_new
    )
)

# Standard error for a new observation

se_prediction <- sqrt(
  sigma2_hat *
    (
      1 +
        as.numeric(
          t(x_new) %*%
            solve(XtX) %*%
            x_new
        )
    )
)

se_mean
se_prediction


# ------------------------------------------------------------------------------
# 22. Confidence and Prediction Intervals
# ------------------------------------------------------------------------------

confidence_interval <- y_new_hat +
  c(-1, 1) * t_critical * se_mean

prediction_interval <- y_new_hat +
  c(-1, 1) * t_critical * se_prediction

confidence_interval
prediction_interval


# ------------------------------------------------------------------------------
# 23. Verify Orthogonality Conditions
# ------------------------------------------------------------------------------

# Least-squares residuals should be orthogonal to every column of X:
#
# X'e = 0

t(X) %*% residuals

# Residuals should sum to zero when an intercept is included

sum(residuals)


# ------------------------------------------------------------------------------
# 24. Verify Using lm()
# ------------------------------------------------------------------------------

model <- lm(
  y ~ x1 + x2 + x3,
  data = data
)

summary(model)

coef(model)

confint(model)


# Compare coefficients

cbind(
  Manual = beta_hat,
  lm = coef(model)
)


# Compare fitted values

max(
  abs(
    as.vector(y_hat) -
      fitted(model)
  )
)


# Compare residuals

max(
  abs(
    as.vector(residuals) -
      resid(model)
  )
)


# ------------------------------------------------------------------------------
# 25. Built-In Diagnostic Plots
# ------------------------------------------------------------------------------

par(mfrow = c(2, 2))

plot(model)

par(mfrow = c(1, 1))