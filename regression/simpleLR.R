# ==============================================================================
# Simple Linear Regression
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

n <- 100

x <- runif(n, min = 0, max = 10)

beta_0 <- 2
beta_1 <- 3
sigma <- 2

epsilon <- rnorm(n, mean = 0, sd = sigma)

y <- beta_0 + beta_1 * x + epsilon


# ------------------------------------------------------------------------------
# 2. Explore the Data
# ------------------------------------------------------------------------------

head(data.frame(x, y))

summary(x)
summary(y)

plot(
  x, y,
  pch = 19,
  xlab = "X",
  ylab = "Y",
  main = "Simulated Data"
)


# ------------------------------------------------------------------------------
# 3. Estimate the Regression Coefficients
# ------------------------------------------------------------------------------

x_bar <- mean(x)
y_bar <- mean(y)

# Slope
beta_1_hat <- sum((x - x_bar) * (y - y_bar)) /
  sum((x - x_bar)^2)

# Intercept
beta_0_hat <- y_bar - beta_1_hat * x_bar

beta_0_hat
beta_1_hat


# ------------------------------------------------------------------------------
# 4. Fitted Values
# ------------------------------------------------------------------------------

y_hat <- beta_0_hat + beta_1_hat * x

head(y_hat)

plot(
  x, y,
  pch = 19,
  xlab = "X",
  ylab = "Y",
  main = "Simple Linear Regression"
)

abline(
  a = beta_0_hat,
  b = beta_1_hat,
  lwd = 2
)


# ------------------------------------------------------------------------------
# 5. Residuals
# ------------------------------------------------------------------------------

residuals <- y - y_hat

head(residuals)

# Residual sum of squares
RSS <- sum(residuals^2)

# Total sum of squares
TSS <- sum((y - y_bar)^2)

# Regression sum of squares
ESS <- sum((y_hat - y_bar)^2)

RSS
TSS
ESS

# Verify decomposition
TSS
ESS + RSS


# ------------------------------------------------------------------------------
# 6. Estimate Error Variance
# ------------------------------------------------------------------------------

p <- 2

sigma2_hat <- RSS / (n - p)
sigma_hat <- sqrt(sigma2_hat)

sigma2_hat
sigma_hat


# ------------------------------------------------------------------------------
# 7. Standard Errors
# ------------------------------------------------------------------------------

Sxx <- sum((x - x_bar)^2)

se_beta_1 <- sqrt(sigma2_hat / Sxx)

se_beta_0 <- sqrt(
  sigma2_hat * (1 / n + x_bar^2 / Sxx)
)

se_beta_0
se_beta_1


# ------------------------------------------------------------------------------
# 8. Hypothesis Test for the Slope
# ------------------------------------------------------------------------------

# H0: beta_1 = 0
# H1: beta_1 != 0

t_stat <- beta_1_hat / se_beta_1

df <- n - p

p_value <- 2 * pt(
  -abs(t_stat),
  df = df
)

t_stat
p_value


# ------------------------------------------------------------------------------
# 9. Confidence Intervals
# ------------------------------------------------------------------------------

alpha <- 0.05

t_critical <- qt(
  1 - alpha / 2,
  df = df
)

beta_0_ci <- beta_0_hat +
  c(-1, 1) * t_critical * se_beta_0

beta_1_ci <- beta_1_hat +
  c(-1, 1) * t_critical * se_beta_1

beta_0_ci
beta_1_ci


# ------------------------------------------------------------------------------
# 10. R-Squared
# ------------------------------------------------------------------------------

R2 <- 1 - RSS / TSS

R2


# ------------------------------------------------------------------------------
# 11. Residual Diagnostics
# ------------------------------------------------------------------------------

plot(
  y_hat,
  residuals,
  pch = 19,
  xlab = "Fitted Values",
  ylab = "Residuals",
  main = "Residuals vs Fitted"
)

abline(h = 0, lty = 2)

hist(
  residuals,
  breaks = 15,
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
# 12. Prediction
# ------------------------------------------------------------------------------

x_new <- 5

y_new_hat <- beta_0_hat + beta_1_hat * x_new

y_new_hat


# ------------------------------------------------------------------------------
# 13. Verify Using lm()
# ------------------------------------------------------------------------------

model <- lm(y ~ x)

summary(model)

coef(model)

confint(model)

# Compare manual and lm() estimates
c(
  manual_intercept = beta_0_hat,
  lm_intercept = coef(model)[1]
)

c(
  manual_slope = beta_1_hat,
  lm_slope = coef(model)[2]
)


# ------------------------------------------------------------------------------
# 14. Built-In Diagnostic Plots
# ------------------------------------------------------------------------------

par(mfrow = c(2, 2))

plot(model)

par(mfrow = c(1, 1))
