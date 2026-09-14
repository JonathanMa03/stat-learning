# ==============================================================================
# Google's PageRank Algorithm
# Jonathan Ma
#
# Main ideas:
#   - Directed graphs
#   - Adjacency matrices
#   - Markov chains
#   - Random-surfer interpretation
#   - Stationary distributions
#   - Eigenvectors
#   - Power iteration
#   - Dangling nodes
#   - Spider traps / rank sinks
#   - Teleportation
#   - Damping factor
#   - Google matrix
#   - Personalized PageRank
#
#
# Core PageRank equation:
#
#       r = alpha * P^T r + (1 - alpha) * v
#
# where:
#
#       r       = PageRank vector
#       P       = row-stochastic transition matrix
#       alpha   = damping factor
#       v       = teleportation distribution
#
#
# Uniform PageRank:
#
#       v = (1/n) * 1
#
#
# The standard random-surfer interpretation:
#
#   With probability alpha:
#       follow an outgoing hyperlink.
#
#   With probability 1-alpha:
#       teleport according to v.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# BUILD A SMALL DIRECTED WEB GRAPH
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Page Names
# ------------------------------------------------------------------------------

pages <- c(
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H"
)


n <- length(
  pages
)


# ------------------------------------------------------------------------------
# 2. Directed Links
#
# A -> B, C
# B -> C, D
# C -> A
# D -> C
# E -> C, D, F
# F -> E
# G -> C, F
# H -> nothing
#
# H is a dangling page.
# ------------------------------------------------------------------------------

links <- list(
  A = c("B", "C"),
  B = c("C", "D"),
  C = c("A"),
  D = c("C"),
  E = c("C", "D", "F"),
  F = c("E"),
  G = c("C", "F"),
  H = character(0)
)


# ==============================================================================
# PART II
#
# ADJACENCY MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Construct Directed Adjacency Matrix
#
# Convention:
#
#       A[i,j] = 1
#
# means page i links TO page j.
# ------------------------------------------------------------------------------

build_adjacency_matrix <- function(
    pages,
    links
) {
  
  n <- length(
    pages
  )
  
  
  A <- matrix(
    0,
    nrow = n,
    ncol = n,
    dimnames = list(
      pages,
      pages
    )
  )
  
  
  for (
    page in pages
  ) {
    
    destinations <- links[[page]]
    
    
    if (
      length(
        destinations
      ) >
      0
    ) {
      
      A[
        page,
        destinations
      ] <- 1
    }
  }
  
  
  A
}


# ------------------------------------------------------------------------------
# 4. Build Matrix
# ------------------------------------------------------------------------------

A <- build_adjacency_matrix(
  pages,
  links
)


A


# ==============================================================================
# PART III
#
# IN-DEGREE AND OUT-DEGREE
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Graph Degrees
# ------------------------------------------------------------------------------

out_degree <- rowSums(
  A
)


in_degree <- colSums(
  A
)


degree_table <- data.frame(
  Page =
    pages,
  In_Degree =
    in_degree,
  Out_Degree =
    out_degree
)


degree_table


# ------------------------------------------------------------------------------
# 6. In-Degree Ranking
# ------------------------------------------------------------------------------

degree_table[
  order(
    -degree_table$In_Degree
  ),
]


# PageRank is NOT simply in-degree.
#
# A link from an important page should generally matter more than a link from
# an unimportant page.


# ==============================================================================
# PART IV
#
# SIMPLE RANDOM-WALK TRANSITION MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Convert Adjacency Matrix into Transition Probabilities
#
# If page i has d_i outgoing links:
#
#       P[i,j] = A[i,j] / d_i
#
# Each non-dangling row therefore sums to one.
# ------------------------------------------------------------------------------

adjacency_to_transition <- function(
    A
) {
  
  A <- as.matrix(
    A
  )
  
  
  n <- nrow(
    A
  )
  
  
  P <- matrix(
    0,
    nrow = n,
    ncol = n,
    dimnames = dimnames(
      A
    )
  )
  
  
  out_degree <- rowSums(
    A
  )
  
  
  for (
    i in seq_len(
      n
    )
  ) {
    
    if (
      out_degree[i] >
      0
    ) {
      
      P[
        i,
      ] <- A[
        i,
      ] /
        out_degree[i]
    }
  }
  
  
  P
}


# ------------------------------------------------------------------------------
# 8. Raw Transition Matrix
# ------------------------------------------------------------------------------

P_raw <- adjacency_to_transition(
  A
)


round(
  P_raw,
  3
)


rowSums(
  P_raw
)


# Notice that the dangling page H has row sum zero.
#
# This is not yet a valid Markov transition matrix.


# ==============================================================================
# PART V
#
# DANGLING NODES
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Identify Dangling Pages
# ------------------------------------------------------------------------------

dangling_nodes <- which(
  rowSums(
    A
  ) ==
    0
)


pages[
  dangling_nodes
]


# ------------------------------------------------------------------------------
# 10. Repair Dangling Nodes
#
# Standard PageRank treatment:
#
# A page with no outgoing links behaves as if it links according to the
# teleportation distribution.
#
# Under uniform PageRank:
#
#       P[i,j] = 1/n
#
# for every j if i is dangling.
# ------------------------------------------------------------------------------

repair_dangling_nodes <- function(
    P,
    teleportation = NULL
) {
  
  P <- as.matrix(
    P
  )
  
  
  n <- nrow(
    P
  )
  
  
  if (
    is.null(
      teleportation
    )
  ) {
    
    teleportation <- rep(
      1 / n,
      n
    )
  }
  
  
  teleportation <- teleportation /
    sum(
      teleportation
    )
  
  
  dangling <- which(
    rowSums(
      P
    ) ==
      0
  )
  
  
  if (
    length(
      dangling
    ) >
    0
  ) {
    
    for (
      i in dangling
    ) {
      
      P[
        i,
      ] <- teleportation
    }
  }
  
  
  P
}


# ------------------------------------------------------------------------------
# 11. Repaired Matrix
# ------------------------------------------------------------------------------

uniform_teleportation <- rep(
  1 / n,
  n
)


P <- repair_dangling_nodes(
  P_raw,
  teleportation =
    uniform_teleportation
)


round(
  P,
  3
)


rowSums(
  P
)


# ==============================================================================
# PART VI
#
# RANDOM WALK WITHOUT TELEPORTATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Power Iteration for an Ordinary Markov Chain
#
# If r is a column vector:
#
#       r_new = P^T r
# ------------------------------------------------------------------------------

random_walk_power <- function(
    P,
    initial = NULL,
    tolerance = 1e-12,
    maximum_iterations = 10000
) {
  
  n <- nrow(
    P
  )
  
  
  if (
    is.null(
      initial
    )
  ) {
    
    r <- rep(
      1 / n,
      n
    )
    
  } else {
    
    r <- initial /
      sum(
        initial
      )
  }
  
  
  history <- matrix(
    NA_real_,
    nrow =
      maximum_iterations +
      1,
    ncol = n
  )
  
  
  history[
    1,
  ] <- r
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      maximum_iterations
    )
  ) {
    
    r_new <- as.numeric(
      t(
        P
      ) %*%
        r
    )
    
    
    difference <- sum(
      abs(
        r_new -
          r
      )
    )
    
    
    history[
      iteration +
        1,
    ] <- r_new
    
    
    if (
      difference <
      tolerance
    ) {
      
      r <- r_new
      
      converged <- TRUE
      
      break
    }
    
    
    r <- r_new
  }
  
  
  used_rows <- seq_len(
    iteration +
      1
  )
  
  
  list(
    stationary =
      r,
    iterations =
      iteration,
    converged =
      converged,
    history =
      history[
        used_rows,
        ,
        drop = FALSE
      ]
  )
}


# ------------------------------------------------------------------------------
# 13. Ordinary Random-Walk Stationary Distribution
# ------------------------------------------------------------------------------

random_walk_fit <- random_walk_power(
  P
)


data.frame(
  Page =
    pages,
  Stationary_Probability =
    random_walk_fit$stationary
)


# ==============================================================================
# PART VII
#
# THE GOOGLE MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Construct Google Matrix
#
#       G = alpha P + (1-alpha) 1 v^T
#
# Every row of G sums to one.
# ------------------------------------------------------------------------------

google_matrix <- function(
    P,
    alpha = 0.85,
    teleportation = NULL
) {
  
  P <- as.matrix(
    P
  )
  
  
  n <- nrow(
    P
  )
  
  
  if (
    is.null(
      teleportation
    )
  ) {
    
    teleportation <- rep(
      1 / n,
      n
    )
  }
  
  
  teleportation <- teleportation /
    sum(
      teleportation
    )
  
  
  P <- repair_dangling_nodes(
    P,
    teleportation =
      teleportation
  )
  
  
  teleportation_matrix <- matrix(
    teleportation,
    nrow = n,
    ncol = n,
    byrow = TRUE
  )
  
  
  G <- alpha *
    P +
    (
      1 -
        alpha
    ) *
    teleportation_matrix
  
  
  G
}


# ------------------------------------------------------------------------------
# 15. Build Google Matrix
# ------------------------------------------------------------------------------

alpha <- 0.85


G <- google_matrix(
  P_raw,
  alpha =
    alpha,
  teleportation =
    uniform_teleportation
)


round(
  G,
  3
)


rowSums(
  G
)


# ==============================================================================
# PART VIII
#
# MANUAL PAGERANK VIA POWER ITERATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 16. PageRank
#
#       r_(t+1)
#
#       =
#
#       alpha P^T r_t
#
#       +
#
#       (1-alpha) v
# ------------------------------------------------------------------------------

pagerank_power <- function(
    A,
    alpha = 0.85,
    teleportation = NULL,
    initial = NULL,
    tolerance = 1e-12,
    maximum_iterations = 10000
) {
  
  A <- as.matrix(
    A
  )
  
  
  n <- nrow(
    A
  )
  
  
  if (
    is.null(
      teleportation
    )
  ) {
    
    teleportation <- rep(
      1 / n,
      n
    )
  }
  
  
  teleportation <- teleportation /
    sum(
      teleportation
    )
  
  
  P_raw <- adjacency_to_transition(
    A
  )
  
  
  P <- repair_dangling_nodes(
    P_raw,
    teleportation =
      teleportation
  )
  
  
  if (
    is.null(
      initial
    )
  ) {
    
    rank <- rep(
      1 / n,
      n
    )
    
  } else {
    
    rank <- initial /
      sum(
        initial
      )
  }
  
  
  rank_history <- matrix(
    NA_real_,
    nrow =
      maximum_iterations +
      1,
    ncol = n
  )
  
  
  difference_history <- numeric(
    maximum_iterations
  )
  
  
  rank_history[
    1,
  ] <- rank
  
  
  converged <- FALSE
  
  
  for (
    iteration in seq_len(
      maximum_iterations
    )
  ) {
    
    new_rank <- alpha *
      as.numeric(
        t(
          P
        ) %*%
          rank
      ) +
      (
        1 -
          alpha
      ) *
      teleportation
    
    
    # Normalize to protect against tiny floating-point drift.
    
    new_rank <- new_rank /
      sum(
        new_rank
      )
    
    
    difference <- sum(
      abs(
        new_rank -
          rank
      )
    )
    
    
    difference_history[
      iteration
    ] <- difference
    
    
    rank_history[
      iteration +
        1,
    ] <- new_rank
    
    
    if (
      difference <
      tolerance
    ) {
      
      rank <- new_rank
      
      converged <- TRUE
      
      break
    }
    
    
    rank <- new_rank
  }
  
  
  list(
    rank =
      rank,
    transition =
      P,
    alpha =
      alpha,
    teleportation =
      teleportation,
    iterations =
      iteration,
    converged =
      converged,
    rank_history =
      rank_history[
        seq_len(
          iteration +
            1
        ),
        ,
        drop = FALSE
      ],
    difference_history =
      difference_history[
        seq_len(
          iteration
        )
      ]
  )
}


# ------------------------------------------------------------------------------
# 17. Fit PageRank
# ------------------------------------------------------------------------------

pagerank_fit <- pagerank_power(
  A,
  alpha = 0.85,
  tolerance = 1e-12
)


pagerank_fit$converged


pagerank_fit$iterations


# ------------------------------------------------------------------------------
# 18. Rankings
# ------------------------------------------------------------------------------

pagerank_table <- data.frame(
  Page =
    pages,
  PageRank =
    pagerank_fit$rank,
  In_Degree =
    in_degree,
  Out_Degree =
    out_degree
)


pagerank_table <- pagerank_table[
  order(
    -pagerank_table$PageRank
  ),
]


pagerank_table


sum(
  pagerank_table$PageRank
)


# ==============================================================================
# PART IX
#
# VISUALIZE PAGERANK
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. PageRank Bar Plot
# ------------------------------------------------------------------------------

barplot(
  pagerank_table$PageRank,
  names.arg =
    pagerank_table$Page,
  xlab = "Page",
  ylab = "PageRank",
  main = "Google PageRank Scores"
)


# ==============================================================================
# PART X
#
# PAGERANK CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 20. Error vs Iteration
# ------------------------------------------------------------------------------

plot(
  pagerank_fit$difference_history,
  type = "l",
  log = "y",
  xlab = "Iteration",
  ylab = "L1 Change",
  main = "PageRank Power-Iteration Convergence"
)


abline(
  h = 1e-12,
  lty = 2
)


# ------------------------------------------------------------------------------
# 21. Individual Rank Trajectories
# ------------------------------------------------------------------------------

matplot(
  pagerank_fit$rank_history,
  type = "l",
  lty = 1,
  xlab = "Iteration",
  ylab = "PageRank",
  main = "PageRank Trajectories"
)


legend(
  "right",
  legend =
    pages,
  lty = 1,
  cex = 0.8
)


# ==============================================================================
# PART XI
#
# VERIFY FIXED-POINT EQUATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. PageRank Should Satisfy
#
#       r = alpha P^T r + (1-alpha)v
# ------------------------------------------------------------------------------

r <- pagerank_fit$rank


fixed_point_rhs <- alpha *
  as.numeric(
    t(
      pagerank_fit$transition
    ) %*%
      r
  ) +
  (
    1 -
      alpha
  ) *
  pagerank_fit$teleportation


max(
  abs(
    r -
      fixed_point_rhs
  )
)


# ==============================================================================
# PART XII
#
# PAGERANK AS AN EIGENVECTOR
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Eigenvector of G^T
#
# At stationarity:
#
#       G^T r = r
#
# so PageRank is an eigenvector associated with eigenvalue 1.
# ------------------------------------------------------------------------------

G <- google_matrix(
  P_raw,
  alpha = alpha
)


eigen_G <- eigen(
  t(
    G
  )
)


# ------------------------------------------------------------------------------
# 24. Find Eigenvalue Closest to One
# ------------------------------------------------------------------------------

index_one <- which.min(
  abs(
    eigen_G$values -
      1
  )
)


eigenvalue_one <- eigen_G$values[
  index_one
]


eigenvector_rank <- Re(
  eigen_G$vectors[
    ,
    index_one
  ]
)


# Orient positively.

if (
  sum(
    eigenvector_rank
  ) <
  0
) {
  
  eigenvector_rank <- -eigenvector_rank
}


eigenvector_rank <- eigenvector_rank /
  sum(
    eigenvector_rank
  )


eigenvalue_one


# ------------------------------------------------------------------------------
# 25. Compare
# ------------------------------------------------------------------------------

data.frame(
  Page =
    pages,
  Power_Iteration =
    pagerank_fit$rank,
  Eigenvector =
    eigenvector_rank,
  Difference =
    pagerank_fit$rank -
    eigenvector_rank
)


# ==============================================================================
# PART XIII
#
# DIRECT LINEAR-SYSTEM SOLUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Solve PageRank Directly
#
# Starting from:
#
#       r = alpha P^T r + (1-alpha)v
#
# Rearrange:
#
#       (I - alpha P^T)r = (1-alpha)v
#
# Therefore:
#
#       r = (1-alpha)(I-alpha P^T)^(-1)v
#
# We use solve(A,b), not an explicit matrix inverse.
# ------------------------------------------------------------------------------

pagerank_direct <- function(
    A,
    alpha = 0.85,
    teleportation = NULL
) {
  
  A <- as.matrix(
    A
  )
  
  
  n <- nrow(
    A
  )
  
  
  if (
    is.null(
      teleportation
    )
  ) {
    
    teleportation <- rep(
      1 / n,
      n
    )
  }
  
  
  teleportation <- teleportation /
    sum(
      teleportation
    )
  
  
  P <- adjacency_to_transition(
    A
  )
  
  
  P <- repair_dangling_nodes(
    P,
    teleportation
  )
  
  
  rank <- solve(
    diag(
      n
    ) -
      alpha *
      t(
        P
      ),
    (
      1 -
        alpha
    ) *
      teleportation
  )
  
  
  rank <- as.numeric(
    rank
  )
  
  
  rank /
    sum(
      rank
    )
}


# ------------------------------------------------------------------------------
# 27. Compare Direct and Iterative Solutions
# ------------------------------------------------------------------------------

rank_direct <- pagerank_direct(
  A,
  alpha = 0.85
)


data.frame(
  Page =
    pages,
  Power =
    pagerank_fit$rank,
  Direct =
    rank_direct,
  Difference =
    pagerank_fit$rank -
    rank_direct
)


# ==============================================================================
# PART XIV
#
# WHY IN-DEGREE IS NOT PAGERANK
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Compare Rankings
# ------------------------------------------------------------------------------

comparison <- data.frame(
  Page =
    pages,
  In_Degree =
    in_degree,
  PageRank =
    pagerank_fit$rank
)


comparison[
  order(
    -comparison$PageRank
  ),
]


cor(
  comparison$In_Degree,
  comparison$PageRank,
  method = "spearman"
)


# In-degree and PageRank can correlate, but they measure different things.


# ==============================================================================
# PART XV
#
# SIMPLE RECURSIVE RANK INTUITION
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Incoming PageRank Contributions
#
# Contribution from page j to page i:
#
#       alpha * r_j / outdegree_j
#
# when j links to i.
# ------------------------------------------------------------------------------

pagerank_contributions <- function(
    A,
    rank,
    alpha = 0.85
) {
  
  n <- nrow(
    A
  )
  
  
  contributions <- matrix(
    0,
    nrow = n,
    ncol = n,
    dimnames = dimnames(
      A
    )
  )
  
  
  out_degree <- rowSums(
    A
  )
  
  
  for (
    source in seq_len(
      n
    )
  ) {
    
    if (
      out_degree[source] >
      0
    ) {
      
      destinations <- which(
        A[
          source,
        ] >
          0
      )
      
      
      contributions[
        source,
        destinations
      ] <- alpha *
        rank[source] /
        out_degree[source]
    }
  }
  
  
  contributions
}


# ------------------------------------------------------------------------------
# 30. Contribution Matrix
# ------------------------------------------------------------------------------

contributions <- pagerank_contributions(
  A,
  pagerank_fit$rank,
  alpha = 0.85
)


round(
  contributions,
  4
)


# Each row shows how much rank a source page passes to linked pages.


# ==============================================================================
# PART XVI
#
# DAMPING FACTOR
# ==============================================================================


# ------------------------------------------------------------------------------
# 31. Compare Different Alpha Values
# ------------------------------------------------------------------------------

alpha_values <- c(
  0,
  0.25,
  0.50,
  0.75,
  0.85,
  0.95,
  0.99
)


alpha_results <- matrix(
  NA_real_,
  nrow =
    length(
      alpha_values
    ),
  ncol = n
)


alpha_iterations <- numeric(
  length(
    alpha_values
  )
)


for (
  i in seq_along(
    alpha_values
  )
) {
  
  fit_i <- pagerank_power(
    A,
    alpha =
      alpha_values[i],
    tolerance = 1e-12
  )
  
  
  alpha_results[
    i,
  ] <- fit_i$rank
  
  
  alpha_iterations[i] <-
    fit_i$iterations
}


colnames(
  alpha_results
) <- pages


rownames(
  alpha_results
) <- alpha_values


round(
  alpha_results,
  4
)


# ------------------------------------------------------------------------------
# 32. Rank Paths
# ------------------------------------------------------------------------------

matplot(
  alpha_values,
  alpha_results,
  type = "b",
  pch = 19,
  lty = 1,
  xlab = "Damping Factor Alpha",
  ylab = "PageRank",
  main = "PageRank vs Damping Factor"
)


legend(
  "topright",
  legend =
    pages,
  lty = 1,
  pch = 19,
  cex = 0.7
)


# ------------------------------------------------------------------------------
# 33. Convergence Speed
# ------------------------------------------------------------------------------

plot(
  alpha_values,
  alpha_iterations,
  type = "b",
  pch = 19,
  xlab = "Damping Factor Alpha",
  ylab = "Iterations",
  main = "Damping Factor and Convergence"
)


# As alpha approaches one, the link structure dominates and convergence can
# become slower.


# ==============================================================================
# PART XVII
#
# ALPHA = ZERO
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Pure Teleportation
#
# If alpha = 0:
#
#       r = v
# ------------------------------------------------------------------------------

pagerank_alpha_zero <- pagerank_power(
  A,
  alpha = 0
)


data.frame(
  Page =
    pages,
  PageRank =
    pagerank_alpha_zero$rank
)


# Under uniform teleportation, every page has equal PageRank.


# ==============================================================================
# PART XVIII
#
# ALPHA CLOSE TO ONE
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Almost Pure Link Following
# ------------------------------------------------------------------------------

pagerank_alpha_high <- pagerank_power(
  A,
  alpha = 0.99
)


data.frame(
  Page =
    pages,
  PageRank =
    pagerank_alpha_high$rank
)


# ==============================================================================
# PART XIX
#
# SPIDER TRAPS / RANK SINKS
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Construct a Graph with a Trap
#
# Pages D and E link only to one another.
#
# Without teleportation, once a random surfer enters this component,
# it cannot leave.
# ------------------------------------------------------------------------------

trap_pages <- c(
  "A",
  "B",
  "C",
  "D",
  "E"
)


trap_links <- list(
  A = c("B"),
  B = c("C", "D"),
  C = c("A"),
  D = c("E"),
  E = c("D")
)


A_trap <- build_adjacency_matrix(
  trap_pages,
  trap_links
)


A_trap


# ------------------------------------------------------------------------------
# 37. Transition Matrix
# ------------------------------------------------------------------------------

P_trap <- adjacency_to_transition(
  A_trap
)


# ------------------------------------------------------------------------------
# 38. Random Walk without Teleportation
# ------------------------------------------------------------------------------

trap_random_walk <- random_walk_power(
  P_trap
)


data.frame(
  Page =
    trap_pages,
  Probability =
    trap_random_walk$stationary
)


# Rank tends to become trapped in the closed D-E component.


# ------------------------------------------------------------------------------
# 39. PageRank with Teleportation
# ------------------------------------------------------------------------------

trap_pagerank <- pagerank_power(
  A_trap,
  alpha = 0.85
)


data.frame(
  Page =
    trap_pages,
  PageRank =
    trap_pagerank$rank
)


# Teleportation allows probability mass to escape the trap.


# ==============================================================================
# PART XX
#
# PERIODICITY PROBLEM
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Two-Page Cycle
#
# A -> B
# B -> A
#
# Pure random-walk iteration can oscillate depending on initialization.
# ------------------------------------------------------------------------------

cycle_A <- matrix(
  c(
    0, 1,
    1, 0
  ),
  nrow = 2,
  byrow = TRUE
)


rownames(
  cycle_A
) <- c(
  "A",
  "B"
)


colnames(
  cycle_A
) <- c(
  "A",
  "B"
)


P_cycle <- adjacency_to_transition(
  cycle_A
)


# ------------------------------------------------------------------------------
# 41. Start Entirely at A
# ------------------------------------------------------------------------------

cycle_initial <- c(
  1,
  0
)


cycle_rank <- cycle_initial


cycle_history <- matrix(
  NA_real_,
  nrow = 11,
  ncol = 2
)


cycle_history[
  1,
] <- cycle_rank


for (
  iteration in 1:10
) {
  
  cycle_rank <- as.numeric(
    t(
      P_cycle
    ) %*%
      cycle_rank
  )
  
  
  cycle_history[
    iteration +
      1,
  ] <- cycle_rank
}


cycle_history


# ------------------------------------------------------------------------------
# 42. Teleportation Removes Periodicity
# ------------------------------------------------------------------------------

cycle_pagerank <- pagerank_power(
  cycle_A,
  alpha = 0.85,
  initial =
    cycle_initial
)


cycle_pagerank$rank


# ==============================================================================
# PART XXI
#
# PERSONALIZED PAGERANK
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Personalized Teleportation Distribution
#
# Suppose the user is particularly interested in pages E and F.
# ------------------------------------------------------------------------------

personalized_v <- c(
  A = 0.02,
  B = 0.02,
  C = 0.02,
  D = 0.02,
  E = 0.45,
  F = 0.43,
  G = 0.02,
  H = 0.02
)


personalized_v <- personalized_v /
  sum(
    personalized_v
  )


personalized_v


# ------------------------------------------------------------------------------
# 44. Personalized PageRank
# ------------------------------------------------------------------------------

personalized_fit <- pagerank_power(
  A,
  alpha = 0.85,
  teleportation =
    personalized_v
)


personalized_table <- data.frame(
  Page =
    pages,
  Standard =
    pagerank_fit$rank,
  Personalized =
    personalized_fit$rank
)


personalized_table[
  order(
    -personalized_table$Personalized
  ),
]


# ==============================================================================
# PART XXII
#
# STANDARD VS PERSONALIZED PAGERANK
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Plot
# ------------------------------------------------------------------------------

barplot(
  rbind(
    Standard =
      pagerank_fit$rank,
    Personalized =
      personalized_fit$rank
  ),
  beside = TRUE,
  names.arg =
    pages,
  xlab = "Page",
  ylab = "PageRank",
  main = "Standard vs Personalized PageRank"
)


legend(
  "topright",
  legend = c(
    "Standard",
    "Personalized"
  ),
  fill = c(
    "gray",
    "black"
  )
)


# ==============================================================================
# PART XXIII
#
# EFFECT OF ADDING A LINK
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Add a Link from an Important Page
#
# Compare adding C -> G.
# ------------------------------------------------------------------------------

A_modified <- A


A_modified[
  "C",
  "G"
] <- 1


modified_fit <- pagerank_power(
  A_modified,
  alpha = 0.85
)


link_effect <- data.frame(
  Page =
    pages,
  Original =
    pagerank_fit$rank,
  Modified =
    modified_fit$rank
)


link_effect$Change <- link_effect$Modified -
  link_effect$Original


link_effect[
  order(
    -link_effect$Change
  ),
]


# ==============================================================================
# PART XXIV
#
# A LINK FROM AN IMPORTANT PAGE VS UNIMPORTANT PAGE
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Identify High- and Low-Rank Sources
# ------------------------------------------------------------------------------

ordered_pages <- pages[
  order(
    -pagerank_fit$rank
  )
]


high_rank_page <- ordered_pages[1]


low_rank_page <- ordered_pages[
  length(
    ordered_pages
  )
]


high_rank_page


low_rank_page


# ------------------------------------------------------------------------------
# 48. Add a Link to G from Each Source
# ------------------------------------------------------------------------------

A_high_link <- A


A_low_link <- A


A_high_link[
  high_rank_page,
  "G"
] <- 1


A_low_link[
  low_rank_page,
  "G"
] <- 1


rank_high_link <- pagerank_power(
  A_high_link,
  alpha = 0.85
)$rank


rank_low_link <- pagerank_power(
  A_low_link,
  alpha = 0.85
)$rank


data.frame(
  Scenario = c(
    "Original",
    paste(
      high_rank_page,
      "links to G"
    ),
    paste(
      low_rank_page,
      "links to G"
    )
  ),
  G_PageRank = c(
    pagerank_fit$rank[
      which(
        pages ==
          "G"
      )
    ],
    rank_high_link[
      which(
        pages ==
          "G"
      )
    ],
    rank_low_link[
      which(
        pages ==
          "G"
      )
    ]
  )
)


# ==============================================================================
# PART XXV
#
# OUTGOING LINKS DILUTE CONTRIBUTION
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Contribution from a Page is Split Across Its Outgoing Links
#
# If page i has rank r_i and d_i outgoing links, each receives roughly:
#
#       alpha * r_i / d_i
# ------------------------------------------------------------------------------

source_page <- "C"


source_index <- which(
  pages ==
    source_page
)


pagerank_fit$rank[
  source_index
]


out_degree[
  source_index
]


# Add several outgoing links from C.

A_many_links <- A


A_many_links[
  "C",
  c(
    "D",
    "E",
    "F",
    "G"
  )
] <- 1


many_links_fit <- pagerank_power(
  A_many_links,
  alpha = 0.85
)


data.frame(
  Page =
    pages,
  Original =
    pagerank_fit$rank,
  More_Outlinks_From_C =
    many_links_fit$rank
)


# ==============================================================================
# PART XXVI
#
# MONTE CARLO RANDOM-SURFER SIMULATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Simulate Random Surfer
# ------------------------------------------------------------------------------

simulate_random_surfer <- function(
    A,
    alpha = 0.85,
    teleportation = NULL,
    steps = 100000,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  A <- as.matrix(
    A
  )
  
  
  n <- nrow(
    A
  )
  
  
  if (
    is.null(
      teleportation
    )
  ) {
    
    teleportation <- rep(
      1 / n,
      n
    )
  }
  
  
  teleportation <- teleportation /
    sum(
      teleportation
    )
  
  
  current <- sample(
    seq_len(
      n
    ),
    size = 1,
    prob =
      teleportation
  )
  
  
  visits <- numeric(
    n
  )
  
  
  for (
    step in seq_len(
      steps
    )
  ) {
    
    visits[current] <- visits[current] +
      1
    
    
    follow_link <- runif(
      1
    ) <
      alpha
    
    
    outgoing <- which(
      A[
        current,
      ] >
        0
    )
    
    
    if (
      follow_link &&
      length(
        outgoing
      ) >
      0
    ) {
      
      current <- sample(
        outgoing,
        size = 1
      )
      
    } else {
      
      current <- sample(
        seq_len(
          n
        ),
        size = 1,
        prob =
          teleportation
      )
    }
  }
  
  
  visits /
    sum(
      visits
    )
}


# ------------------------------------------------------------------------------
# 51. Simulate
# ------------------------------------------------------------------------------

monte_carlo_rank <- simulate_random_surfer(
  A,
  alpha = 0.85,
  steps = 200000,
  seed = 123
)


# ------------------------------------------------------------------------------
# 52. Compare
# ------------------------------------------------------------------------------

data.frame(
  Page =
    pages,
  PageRank =
    pagerank_fit$rank,
  Monte_Carlo =
    monte_carlo_rank,
  Difference =
    monte_carlo_rank -
    pagerank_fit$rank
)


# ==============================================================================
# PART XXVII
#
# MONTE CARLO ACCURACY
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. Different Simulation Lengths
# ------------------------------------------------------------------------------

simulation_lengths <- c(
  100,
  1000,
  10000,
  100000
)


simulation_error <- numeric(
  length(
    simulation_lengths
  )
)


for (
  i in seq_along(
    simulation_lengths
  )
) {
  
  simulated <- simulate_random_surfer(
    A,
    alpha = 0.85,
    steps =
      simulation_lengths[i],
    seed = 123
  )
  
  
  simulation_error[i] <- sum(
    abs(
      simulated -
        pagerank_fit$rank
    )
  )
}


data.frame(
  Steps =
    simulation_lengths,
  L1_Error =
    simulation_error
)


plot(
  simulation_lengths,
  simulation_error,
  type = "b",
  pch = 19,
  log = "x",
  xlab = "Random-Surfer Steps",
  ylab = "L1 Error",
  main = "Monte Carlo Approximation to PageRank"
)


# ==============================================================================
# PART XXVIII
#
# SECOND EIGENVALUE AND CONVERGENCE
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Google Matrix Spectrum
# ------------------------------------------------------------------------------

google_eigenvalues <- eigen(
  t(
    G
  ),
  only.values = TRUE
)$values


eigenvalue_magnitudes <- sort(
  Mod(
    google_eigenvalues
  ),
  decreasing = TRUE
)


head(
  eigenvalue_magnitudes
)


# The dominant eigenvalue is one.
#
# The magnitude of the next eigenvalue helps determine asymptotic power-method
# convergence speed.


# ------------------------------------------------------------------------------
# 55. Spectral Gap
# ------------------------------------------------------------------------------

spectral_gap <- 1 -
  eigenvalue_magnitudes[2]


spectral_gap


# ==============================================================================
# PART XXIX
#
# RANDOM WEB GRAPH
# ==============================================================================


# ------------------------------------------------------------------------------
# 56. Generate Random Directed Graph
# ------------------------------------------------------------------------------

generate_random_web <- function(
    number_pages = 100,
    link_probability = 0.04,
    seed = 123
) {
  
  set.seed(
    seed
  )
  
  
  A <- matrix(
    rbinom(
      number_pages^2,
      size = 1,
      prob =
        link_probability
    ),
    nrow =
      number_pages,
    ncol =
      number_pages
  )
  
  
  diag(
    A
  ) <- 0
  
  
  page_names <- paste0(
    "Page_",
    seq_len(
      number_pages
    )
  )
  
  
  rownames(
    A
  ) <- page_names
  
  
  colnames(
    A
  ) <- page_names
  
  
  A
}


# ------------------------------------------------------------------------------
# 57. Random Web
# ------------------------------------------------------------------------------

A_random <- generate_random_web(
  number_pages = 100,
  link_probability = 0.04,
  seed = 123
)


random_pagerank <- pagerank_power(
  A_random,
  alpha = 0.85
)


# ------------------------------------------------------------------------------
# 58. Top Pages
# ------------------------------------------------------------------------------

random_table <- data.frame(
  Page =
    rownames(
      A_random
    ),
  PageRank =
    random_pagerank$rank,
  In_Degree =
    colSums(
      A_random
    ),
  Out_Degree =
    rowSums(
      A_random
    )
)


head(
  random_table[
    order(
      -random_table$PageRank
    ),
  ],
  10
)


# ==============================================================================
# PART XXX
#
# PAGERANK VS IN-DEGREE ON RANDOM GRAPH
# ==============================================================================


# ------------------------------------------------------------------------------
# 59. Scatterplot
# ------------------------------------------------------------------------------

plot(
  random_table$In_Degree,
  random_table$PageRank,
  pch = 19,
  xlab = "In-Degree",
  ylab = "PageRank",
  main = "PageRank vs In-Degree"
)


cor(
  random_table$In_Degree,
  random_table$PageRank,
  method = "spearman"
)


# ==============================================================================
# PART XXXI
#
# LARGE SPARSE-LIKE EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 60. Larger Random Graph
#
# Base matrices are dense, so this is only a moderate-size demonstration.
# Real web-scale PageRank requires sparse matrix representations.
# ------------------------------------------------------------------------------

A_large <- generate_random_web(
  number_pages = 500,
  link_probability = 0.01,
  seed = 999
)


large_fit <- pagerank_power(
  A_large,
  alpha = 0.85,
  tolerance = 1e-10
)


large_fit$iterations


sum(
  large_fit$rank
)


# ==============================================================================
# PART XXXII
#
# PAGERANK WITH NONUNIFORM TELEPORTATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 61. Topic-Specific PageRank on Random Web
# ------------------------------------------------------------------------------

topic_v <- rep(
  0,
  100
)


# Pretend pages 1:10 belong to a topic of interest.

topic_v[
  1:10
] <- 1


topic_v <- topic_v /
  sum(
    topic_v
  )


topic_fit <- pagerank_power(
  A_random,
  alpha = 0.85,
  teleportation =
    topic_v
)


topic_table <- data.frame(
  Page =
    rownames(
      A_random
    ),
  Global_PageRank =
    random_pagerank$rank,
  Topic_PageRank =
    topic_fit$rank
)


head(
  topic_table[
    order(
      -topic_table$Topic_PageRank
    ),
  ],
  15
)


# ==============================================================================
# PART XXXIII
#
# VERIFY PROBABILITY PROPERTIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 62. PageRank is a Probability Distribution
# ------------------------------------------------------------------------------

sum(
  pagerank_fit$rank
)


min(
  pagerank_fit$rank
)


all(
  pagerank_fit$rank >=
    0
)


# ==============================================================================
# PART XXXIV
#
# PERRON-FROBENIUS INTUITION
# ==============================================================================


# ------------------------------------------------------------------------------
# 63. Positive Google Matrix
#
# With:
#
#       0 < alpha < 1
#
# and strictly positive teleportation probabilities,
# every entry of G is positive.
# ------------------------------------------------------------------------------

min(
  G
)


all(
  G >
    0
)


# This strong positivity gives the Google matrix particularly convenient
# Perron-Frobenius properties:
#
#   - dominant eigenvalue 1
#   - unique positive stationary distribution
#   - power iteration converges from arbitrary probability starts


# ==============================================================================
# PART XXXV
#
# FINAL SUMMARY
# ==============================================================================


cat(
  "\nGoogle PageRank Summary\n"
)


cat(
  "-----------------------\n"
)


cat(
  "Pages:",
  n,
  "\n"
)


cat(
  "Directed links:",
  sum(
    A
  ),
  "\n"
)


cat(
  "Dangling pages:",
  length(
    dangling_nodes
  ),
  "\n"
)


cat(
  "Damping factor:",
  pagerank_fit$alpha,
  "\n"
)


cat(
  "Teleportation probability:",
  1 -
    pagerank_fit$alpha,
  "\n"
)


cat(
  "Power iteration converged:",
  pagerank_fit$converged,
  "\n"
)


cat(
  "Iterations:",
  pagerank_fit$iterations,
  "\n"
)


cat(
  "PageRank sum:",
  round(
    sum(
      pagerank_fit$rank
    ),
    12
  ),
  "\n"
)


cat(
  "Top-ranked page:",
  pagerank_table$Page[1],
  "\n"
)


cat(
  "Top PageRank:",
  round(
    pagerank_table$PageRank[1],
    6
  ),
  "\n"
)


cat(
  "Maximum power-vs-direct difference:",
  format(
    max(
      abs(
        pagerank_fit$rank -
          rank_direct
      )
    ),
    scientific = TRUE
  ),
  "\n"
)