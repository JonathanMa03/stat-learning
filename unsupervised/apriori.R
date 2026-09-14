# ==============================================================================
# Apriori Algorithm
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Market basket analysis
#   - Transaction data
#   - Itemsets
#   - Support
#   - Frequent itemsets
#   - Apriori property
#   - Candidate generation
#   - Candidate pruning
#   - Association rules
#   - Confidence
#   - Lift
#   - Leverage
#   - Conviction
#
# Apriori property:
#
#   If an itemset is frequent, every subset of that itemset must also
#   be frequent.
#
# Contrapositive:
#
#   If an itemset is infrequent, every superset containing that itemset
#   must also be infrequent.
#
# ==============================================================================


# ==============================================================================
# PART I
#
# CREATE TRANSACTION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Simulate Market Basket Transactions
# ------------------------------------------------------------------------------

set.seed(123)

number_transactions <- 1000


items <- c(
  "Bread",
  "Milk",
  "Eggs",
  "Butter",
  "Cheese",
  "Coffee",
  "Tea",
  "Cereal",
  "Bananas",
  "Apples",
  "Yogurt",
  "Juice"
)


transactions <- vector(
  "list",
  number_transactions
)


for (i in seq_len(number_transactions)) {
  
  basket <- character(0)
  
  
  # --------------------------------------------------------------------------
  # Base item probabilities
  # --------------------------------------------------------------------------
  
  if (runif(1) < 0.45) {
    basket <- c(basket, "Bread")
  }
  
  if (runif(1) < 0.50) {
    basket <- c(basket, "Milk")
  }
  
  if (runif(1) < 0.30) {
    basket <- c(basket, "Eggs")
  }
  
  if (runif(1) < 0.18) {
    basket <- c(basket, "Butter")
  }
  
  if (runif(1) < 0.22) {
    basket <- c(basket, "Cheese")
  }
  
  if (runif(1) < 0.28) {
    basket <- c(basket, "Coffee")
  }
  
  if (runif(1) < 0.20) {
    basket <- c(basket, "Tea")
  }
  
  if (runif(1) < 0.25) {
    basket <- c(basket, "Cereal")
  }
  
  if (runif(1) < 0.30) {
    basket <- c(basket, "Bananas")
  }
  
  if (runif(1) < 0.25) {
    basket <- c(basket, "Apples")
  }
  
  if (runif(1) < 0.22) {
    basket <- c(basket, "Yogurt")
  }
  
  if (runif(1) < 0.18) {
    basket <- c(basket, "Juice")
  }
  
  
  # --------------------------------------------------------------------------
  # Introduce Association Structure
  # --------------------------------------------------------------------------
  
  # Bread -> Butter
  
  if (
    "Bread" %in% basket &&
    runif(1) < 0.45
  ) {
    
    basket <- c(
      basket,
      "Butter"
    )
  }
  
  
  # Cereal -> Milk
  
  if (
    "Cereal" %in% basket &&
    runif(1) < 0.65
  ) {
    
    basket <- c(
      basket,
      "Milk"
    )
  }
  
  
  # Coffee -> Milk
  
  if (
    "Coffee" %in% basket &&
    runif(1) < 0.35
  ) {
    
    basket <- c(
      basket,
      "Milk"
    )
  }
  
  
  # Apples -> Yogurt
  
  if (
    "Apples" %in% basket &&
    runif(1) < 0.40
  ) {
    
    basket <- c(
      basket,
      "Yogurt"
    )
  }
  
  
  # Eggs + Bread -> Cheese
  
  if (
    all(
      c(
        "Eggs",
        "Bread"
      ) %in% basket
    ) &&
    runif(1) < 0.55
  ) {
    
    basket <- c(
      basket,
      "Cheese"
    )
  }
  
  
  transactions[[i]] <- sort(
    unique(
      basket
    )
  )
}


# ------------------------------------------------------------------------------
# 2. Inspect Transactions
# ------------------------------------------------------------------------------

transactions[1:10]


# ------------------------------------------------------------------------------
# 3. Basket Sizes
# ------------------------------------------------------------------------------

basket_sizes <- lengths(
  transactions
)


summary(
  basket_sizes
)


hist(
  basket_sizes,
  breaks = seq(
    -0.5,
    max(basket_sizes) + 0.5,
    by = 1
  ),
  xlab = "Number of Items",
  main = "Transaction Basket Sizes"
)


# ==============================================================================
# PART II
#
# TRANSACTION MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Convert Transactions to Binary Matrix
# ------------------------------------------------------------------------------

transaction_matrix <- matrix(
  0L,
  nrow = number_transactions,
  ncol = length(items)
)


colnames(
  transaction_matrix
) <- items


for (i in seq_len(number_transactions)) {
  
  transaction_matrix[
    i,
    transactions[[i]]
  ] <- 1L
}


head(
  transaction_matrix
)


# ------------------------------------------------------------------------------
# 5. Item Frequencies
# ------------------------------------------------------------------------------

item_counts <- colSums(
  transaction_matrix
)


item_support <- item_counts /
  number_transactions


item_frequency_table <- data.frame(
  Item = names(item_counts),
  Count = as.integer(item_counts),
  Support = as.numeric(item_support)
)


item_frequency_table <- item_frequency_table[
  order(
    item_frequency_table$Support,
    decreasing = TRUE
  ),
]


item_frequency_table


# ------------------------------------------------------------------------------
# 6. Plot Item Support
# ------------------------------------------------------------------------------

barplot(
  item_frequency_table$Support,
  names.arg = item_frequency_table$Item,
  las = 2,
  ylab = "Support",
  main = "Individual Item Support"
)


# ==============================================================================
# PART III
#
# ITEMSET REPRESENTATION
# ==============================================================================


# ------------------------------------------------------------------------------
# 7. Canonical Itemset Key
# ------------------------------------------------------------------------------

itemset_key <- function(itemset) {
  
  paste(
    sort(
      unique(
        itemset
      )
    ),
    collapse = "|"
  )
}


# ------------------------------------------------------------------------------
# 8. Convert Key Back to Items
# ------------------------------------------------------------------------------

key_to_itemset <- function(key) {
  
  if (
    is.na(key) ||
    nchar(key) == 0
  ) {
    
    return(
      character(0)
    )
  }
  
  
  strsplit(
    key,
    split = "|",
    fixed = TRUE
  )[[1]]
}


# ------------------------------------------------------------------------------
# 9. Examples
# ------------------------------------------------------------------------------

itemset_key(
  c(
    "Milk",
    "Bread"
  )
)


key_to_itemset(
  "Bread|Milk"
)


# ==============================================================================
# PART IV
#
# SUPPORT
# ==============================================================================


# ------------------------------------------------------------------------------
# 10. Count Transactions Containing an Itemset
# ------------------------------------------------------------------------------

itemset_count <- function(
    itemset,
    transaction_matrix
) {
  
  itemset <- unique(
    itemset
  )
  
  
  if (
    length(itemset) == 0
  ) {
    
    return(
      nrow(
        transaction_matrix
      )
    )
  }
  
  
  rowSums(
    transaction_matrix[
      ,
      itemset,
      drop = FALSE
    ]
  ) |>
    {
      sum(
        . == length(itemset)
      )
    }()
}


# ------------------------------------------------------------------------------
# 11. Itemset Support
# ------------------------------------------------------------------------------

itemset_support <- function(
    itemset,
    transaction_matrix
) {
  
  itemset_count(
    itemset,
    transaction_matrix
  ) /
    nrow(
      transaction_matrix
    )
}


# ------------------------------------------------------------------------------
# 12. Examples
# ------------------------------------------------------------------------------

itemset_support(
  "Bread",
  transaction_matrix
)


itemset_support(
  c(
    "Bread",
    "Butter"
  ),
  transaction_matrix
)


itemset_support(
  c(
    "Cereal",
    "Milk"
  ),
  transaction_matrix
)


itemset_support(
  c(
    "Bread",
    "Eggs",
    "Cheese"
  ),
  transaction_matrix
)


# ==============================================================================
# PART V
#
# APRIORI PROPERTY
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Demonstrate Downward Closure
# ------------------------------------------------------------------------------

triple <- c(
  "Bread",
  "Eggs",
  "Cheese"
)


triple_support <- itemset_support(
  triple,
  transaction_matrix
)


pair_subsets <- combn(
  triple,
  2,
  simplify = FALSE
)


pair_supports <- sapply(
  pair_subsets,
  itemset_support,
  transaction_matrix = transaction_matrix
)


data.frame(
  Itemset = c(
    itemset_key(triple),
    sapply(
      pair_subsets,
      itemset_key
    )
  ),
  Support = c(
    triple_support,
    pair_supports
  )
)


# Every subset must have support at least as large as the complete itemset.


# ==============================================================================
# PART VI
#
# FREQUENT ONE-ITEMSETS
# ==============================================================================


# ------------------------------------------------------------------------------
# 14. Minimum Support
# ------------------------------------------------------------------------------

minimum_support <- 0.08


# ------------------------------------------------------------------------------
# 15. Find Frequent Single Items
# ------------------------------------------------------------------------------

frequent_single_items <- names(
  item_support[
    item_support >= minimum_support
  ]
)


frequent_single_items


# ------------------------------------------------------------------------------
# 16. Store Frequent One-Itemsets
# ------------------------------------------------------------------------------

L1 <- lapply(
  frequent_single_items,
  function(item) {
    
    c(item)
  }
)


L1


# ==============================================================================
# PART VII
#
# GENERATE CANDIDATES
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Generate Candidate k-Itemsets
# ------------------------------------------------------------------------------

generate_candidates <- function(
    previous_frequent,
    k
) {
  
  if (
    length(previous_frequent) < 2
  ) {
    
    return(
      list()
    )
  }
  
  
  previous_frequent <- lapply(
    previous_frequent,
    sort
  )
  
  
  candidates <- list()
  
  
  candidate_index <- 1
  
  
  for (
    i in seq_len(
      length(previous_frequent) - 1
    )
  ) {
    
    for (
      j in (
        i + 1
      ):length(previous_frequent)
    ) {
      
      union_set <- sort(
        unique(
          c(
            previous_frequent[[i]],
            previous_frequent[[j]]
          )
        )
      )
      
      
      if (
        length(union_set) == k
      ) {
        
        candidates[[candidate_index]] <-
          union_set
        
        
        candidate_index <-
          candidate_index + 1
      }
    }
  }
  
  
  if (
    length(candidates) == 0
  ) {
    
    return(
      list()
    )
  }
  
  
  keys <- sapply(
    candidates,
    itemset_key
  )
  
  
  candidates[
    !duplicated(keys)
  ]
}


# ------------------------------------------------------------------------------
# 18. Generate Candidate Pairs
# ------------------------------------------------------------------------------

C2 <- generate_candidates(
  L1,
  k = 2
)


head(
  C2,
  10
)


length(
  C2
)


# ==============================================================================
# PART VIII
#
# APRIORI PRUNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Test Whether All (k - 1)-Subsets Are Frequent
# ------------------------------------------------------------------------------

all_subsets_frequent <- function(
    candidate,
    previous_frequent
) {
  
  k <- length(
    candidate
  )
  
  
  if (
    k <= 1
  ) {
    
    return(
      TRUE
    )
  }
  
  
  previous_keys <- sapply(
    previous_frequent,
    itemset_key
  )
  
  
  subsets <- combn(
    candidate,
    k - 1,
    simplify = FALSE
  )
  
  
  subset_keys <- sapply(
    subsets,
    itemset_key
  )
  
  
  all(
    subset_keys %in%
      previous_keys
  )
}


# ------------------------------------------------------------------------------
# 20. Prune Candidate Itemsets
# ------------------------------------------------------------------------------

prune_candidates <- function(
    candidates,
    previous_frequent
) {
  
  if (
    length(candidates) == 0
  ) {
    
    return(
      list()
    )
  }
  
  
  keep <- sapply(
    candidates,
    all_subsets_frequent,
    previous_frequent =
      previous_frequent
  )
  
  
  candidates[
    keep
  ]
}


# ==============================================================================
# PART IX
#
# SUPPORT COUNTING FOR CANDIDATES
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Evaluate Candidate Itemsets
# ------------------------------------------------------------------------------

evaluate_candidates <- function(
    candidates,
    transaction_matrix,
    minimum_support
) {
  
  if (
    length(candidates) == 0
  ) {
    
    return(
      list(
        frequent = list(),
        table = data.frame()
      )
    )
  }
  
  
  supports <- sapply(
    candidates,
    itemset_support,
    transaction_matrix =
      transaction_matrix
  )
  
  
  frequent <- candidates[
    supports >=
      minimum_support
  ]
  
  
  result_table <- data.frame(
    Itemset = sapply(
      candidates,
      itemset_key
    ),
    Size = lengths(
      candidates
    ),
    Support = supports,
    Frequent = supports >=
      minimum_support
  )
  
  
  list(
    frequent =
      frequent,
    table =
      result_table
  )
}


# ==============================================================================
# PART X
#
# MANUAL APRIORI ALGORITHM
# ==============================================================================


# ------------------------------------------------------------------------------
# 22. Apriori
# ------------------------------------------------------------------------------

apriori <- function(
    transaction_matrix,
    minimum_support = 0.05,
    maximum_size = NULL,
    verbose = TRUE
) {
  
  items <- colnames(
    transaction_matrix
  )
  
  
  p <- length(
    items
  )
  
  
  if (
    is.null(
      maximum_size
    )
  ) {
    
    maximum_size <- p
  }
  
  
  # --------------------------------------------------------------------------
  # Frequent 1-itemsets
  # --------------------------------------------------------------------------
  
  single_supports <- colMeans(
    transaction_matrix
  )
  
  
  frequent_items <- names(
    single_supports[
      single_supports >=
        minimum_support
    ]
  )
  
  
  current_frequent <- lapply(
    frequent_items,
    function(item) {
      
      c(item)
    }
  )
  
  
  all_frequent <- list()
  
  
  support_lookup <- numeric(0)
  
  
  if (
    length(current_frequent) > 0
  ) {
    
    for (
      itemset in current_frequent
    ) {
      
      key <- itemset_key(
        itemset
      )
      
      
      support_lookup[key] <-
        itemset_support(
          itemset,
          transaction_matrix
        )
    }
  }
  
  
  all_frequent[[1]] <-
    current_frequent
  
  
  candidate_counts <- integer(
    maximum_size
  )
  
  
  frequent_counts <- integer(
    maximum_size
  )
  
  
  candidate_counts[1] <-
    p
  
  
  frequent_counts[1] <-
    length(
      current_frequent
    )
  
  
  if (
    verbose
  ) {
    
    cat(
      "k = 1",
      "| Candidates =",
      p,
      "| Frequent =",
      length(
        current_frequent
      ),
      "\n"
    )
  }
  
  
  # --------------------------------------------------------------------------
  # k >= 2
  # --------------------------------------------------------------------------
  
  if (
    maximum_size >= 2
  ) {
    
    for (
      k in 2:maximum_size
    ) {
      
      if (
        length(current_frequent) <
        2
      ) {
        
        break
      }
      
      
      # ----------------------------------------------------------------------
      # Join step
      # ----------------------------------------------------------------------
      
      candidates <- generate_candidates(
        current_frequent,
        k
      )
      
      
      # ----------------------------------------------------------------------
      # Prune step
      # ----------------------------------------------------------------------
      
      candidates <- prune_candidates(
        candidates,
        current_frequent
      )
      
      
      candidate_counts[k] <-
        length(
          candidates
        )
      
      
      if (
        length(candidates) == 0
      ) {
        
        break
      }
      
      
      # ----------------------------------------------------------------------
      # Support counting
      # ----------------------------------------------------------------------
      
      evaluation <- evaluate_candidates(
        candidates,
        transaction_matrix,
        minimum_support
      )
      
      
      current_frequent <-
        evaluation$frequent
      
      
      frequent_counts[k] <-
        length(
          current_frequent
        )
      
      
      if (
        length(current_frequent) >
        0
      ) {
        
        for (
          itemset in current_frequent
        ) {
          
          key <- itemset_key(
            itemset
          )
          
          
          support_lookup[key] <-
            itemset_support(
              itemset,
              transaction_matrix
            )
        }
      }
      
      
      all_frequent[[k]] <-
        current_frequent
      
      
      if (
        verbose
      ) {
        
        cat(
          "k =",
          k,
          "| Candidates =",
          length(
            candidates
          ),
          "| Frequent =",
          length(
            current_frequent
          ),
          "\n"
        )
      }
      
      
      if (
        length(current_frequent) ==
        0
      ) {
        
        break
      }
    }
  }
  
  
  # --------------------------------------------------------------------------
  # Flatten frequent itemsets
  # --------------------------------------------------------------------------
  
  frequent_flat <- unlist(
    all_frequent,
    recursive = FALSE
  )
  
  
  if (
    length(frequent_flat) >
    0
  ) {
    
    frequent_table <- data.frame(
      Itemset = sapply(
        frequent_flat,
        itemset_key
      ),
      Size = lengths(
        frequent_flat
      ),
      Support = sapply(
        frequent_flat,
        function(itemset) {
          
          support_lookup[
            itemset_key(
              itemset
            )
          ]
        }
      )
    )
    
    
    rownames(
      frequent_table
    ) <- NULL
    
    
    frequent_table <- frequent_table[
      order(
        frequent_table$Size,
        -frequent_table$Support
      ),
    ]
    
  } else {
    
    frequent_table <- data.frame(
      Itemset = character(0),
      Size = integer(0),
      Support = numeric(0)
    )
  }
  
  
  list(
    frequent_by_size =
      all_frequent,
    frequent_itemsets =
      frequent_flat,
    frequent_table =
      frequent_table,
    support_lookup =
      support_lookup,
    candidate_counts =
      candidate_counts,
    frequent_counts =
      frequent_counts,
    minimum_support =
      minimum_support
  )
}


# ==============================================================================
# PART XI
#
# RUN APRIORI
# ==============================================================================


# ------------------------------------------------------------------------------
# 23. Fit
# ------------------------------------------------------------------------------

apriori_model <- apriori(
  transaction_matrix,
  minimum_support = 0.08,
  maximum_size = 5,
  verbose = TRUE
)


# ------------------------------------------------------------------------------
# 24. Frequent Itemsets
# ------------------------------------------------------------------------------

apriori_model$frequent_table


# ------------------------------------------------------------------------------
# 25. Most Frequent Itemsets
# ------------------------------------------------------------------------------

frequent_sorted <- apriori_model$frequent_table[
  order(
    apriori_model$frequent_table$Support,
    decreasing = TRUE
  ),
]


head(
  frequent_sorted,
  20
)


# ==============================================================================
# PART XII
#
# CANDIDATE PRUNING
# ==============================================================================


# ------------------------------------------------------------------------------
# 26. Candidate Counts by Level
# ------------------------------------------------------------------------------

levels_used <- which(
  apriori_model$candidate_counts >
    0
)


candidate_summary <- data.frame(
  Itemset_Size =
    levels_used,
  Candidates =
    apriori_model$candidate_counts[
      levels_used
    ],
  Frequent =
    apriori_model$frequent_counts[
      levels_used
    ]
)


candidate_summary


# ------------------------------------------------------------------------------
# 27. Plot Search Reduction
# ------------------------------------------------------------------------------

matplot(
  candidate_summary$Itemset_Size,
  cbind(
    candidate_summary$Candidates,
    candidate_summary$Frequent
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
  xlab = "Itemset Size",
  ylab = "Number of Itemsets",
  main = "Apriori Candidate Pruning"
)


legend(
  "topright",
  legend = c(
    "Candidates",
    "Frequent"
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
# PART XIII
#
# BRUTE-FORCE COMPARISON
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Number of Possible Itemsets
# ------------------------------------------------------------------------------

number_items <- ncol(
  transaction_matrix
)


all_possible_itemsets <- 2^number_items -
  1


all_possible_itemsets


# ------------------------------------------------------------------------------
# 29. Number of Possible Itemsets by Size
# ------------------------------------------------------------------------------

possible_by_size <- sapply(
  seq_len(
    number_items
  ),
  function(k) {
    
    choose(
      number_items,
      k
    )
  }
)


data.frame(
  Size =
    seq_len(
      number_items
    ),
  Possible =
    possible_by_size
)


# ==============================================================================
# PART XIV
#
# SUPPORT THRESHOLD EFFECT
# ==============================================================================


# ------------------------------------------------------------------------------
# 30. Compare Minimum Support Values
# ------------------------------------------------------------------------------

support_values <- c(
  0.03,
  0.05,
  0.08,
  0.10,
  0.15,
  0.20
)


support_results <- data.frame(
  Minimum_Support =
    support_values,
  Frequent_Itemsets =
    NA_integer_,
  Maximum_Size =
    NA_integer_
)


for (
  i in seq_along(
    support_values
  )
) {
  
  model_i <- apriori(
    transaction_matrix,
    minimum_support =
      support_values[i],
    maximum_size = 6,
    verbose = FALSE
  )
  
  
  support_results$Frequent_Itemsets[i] <-
    nrow(
      model_i$frequent_table
    )
  
  
  if (
    nrow(
      model_i$frequent_table
    ) >
    0
  ) {
    
    support_results$Maximum_Size[i] <-
      max(
        model_i$frequent_table$Size
      )
    
  } else {
    
    support_results$Maximum_Size[i] <-
      0
  }
}


support_results


# ------------------------------------------------------------------------------
# 31. Plot Complexity
# ------------------------------------------------------------------------------

plot(
  support_results$Minimum_Support,
  support_results$Frequent_Itemsets,
  type = "b",
  pch = 19,
  xlab = "Minimum Support",
  ylab = "Number of Frequent Itemsets",
  main = "Effect of Minimum Support"
)


# ==============================================================================
# PART XV
#
# ASSOCIATION RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Generate Nonempty Proper Subsets
# ------------------------------------------------------------------------------

nonempty_proper_subsets <- function(
    itemset
) {
  
  m <- length(
    itemset
  )
  
  
  if (
    m < 2
  ) {
    
    return(
      list()
    )
  }
  
  
  subsets <- list()
  
  
  index <- 1
  
  
  for (
    size in seq_len(
      m - 1
    )
  ) {
    
    combinations <- combn(
      itemset,
      size,
      simplify = FALSE
    )
    
    
    for (
      combination in combinations
    ) {
      
      subsets[[index]] <-
        combination
      
      
      index <- index + 1
    }
  }
  
  
  subsets
}


# ------------------------------------------------------------------------------
# 33. Rule Metrics
# ------------------------------------------------------------------------------

calculate_rule_metrics <- function(
    antecedent,
    consequent,
    transaction_matrix
) {
  
  union_set <- union(
    antecedent,
    consequent
  )
  
  
  support_X <- itemset_support(
    antecedent,
    transaction_matrix
  )
  
  
  support_Y <- itemset_support(
    consequent,
    transaction_matrix
  )
  
  
  support_XY <- itemset_support(
    union_set,
    transaction_matrix
  )
  
  
  confidence <- if (
    support_X >
    0
  ) {
    
    support_XY /
      support_X
    
  } else {
    
    NA_real_
  }
  
  
  lift <- if (
    support_Y >
    0
  ) {
    
    confidence /
      support_Y
    
  } else {
    
    NA_real_
  }
  
  
  leverage <- support_XY -
    support_X *
    support_Y
  
  
  conviction <- if (
    is.finite(
      confidence
    ) &&
    confidence <
    1
  ) {
    
    (
      1 -
        support_Y
    ) /
      (
        1 -
          confidence
      )
    
  } else if (
    confidence ==
    1
  ) {
    
    Inf
    
  } else {
    
    NA_real_
  }
  
  
  list(
    support =
      support_XY,
    confidence =
      confidence,
    lift =
      lift,
    leverage =
      leverage,
    conviction =
      conviction
  )
}


# ==============================================================================
# PART XVI
#
# GENERATE ASSOCIATION RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Generate Rules from Frequent Itemsets
# ------------------------------------------------------------------------------

generate_association_rules <- function(
    apriori_model,
    transaction_matrix,
    minimum_confidence = 0.5
) {
  
  frequent_itemsets <-
    apriori_model$frequent_itemsets
  
  
  rules <- list()
  
  
  rule_index <- 1
  
  
  for (
    itemset in frequent_itemsets
  ) {
    
    if (
      length(itemset) <
      2
    ) {
      
      next
    }
    
    
    antecedents <- nonempty_proper_subsets(
      itemset
    )
    
    
    for (
      antecedent in antecedents
    ) {
      
      consequent <- setdiff(
        itemset,
        antecedent
      )
      
      
      metrics <- calculate_rule_metrics(
        antecedent,
        consequent,
        transaction_matrix
      )
      
      
      if (
        metrics$confidence >=
        minimum_confidence
      ) {
        
        rules[[rule_index]] <- list(
          antecedent =
            antecedent,
          consequent =
            consequent,
          support =
            metrics$support,
          confidence =
            metrics$confidence,
          lift =
            metrics$lift,
          leverage =
            metrics$leverage,
          conviction =
            metrics$conviction
        )
        
        
        rule_index <- rule_index + 1
      }
    }
  }
  
  
  if (
    length(rules) == 0
  ) {
    
    return(
      data.frame()
    )
  }
  
  
  rule_table <- data.frame(
    Antecedent = sapply(
      rules,
      function(rule) {
        
        itemset_key(
          rule$antecedent
        )
      }
    ),
    Consequent = sapply(
      rules,
      function(rule) {
        
        itemset_key(
          rule$consequent
        )
      }
    ),
    Support = sapply(
      rules,
      function(rule) {
        
        rule$support
      }
    ),
    Confidence = sapply(
      rules,
      function(rule) {
        
        rule$confidence
      }
    ),
    Lift = sapply(
      rules,
      function(rule) {
        
        rule$lift
      }
    ),
    Leverage = sapply(
      rules,
      function(rule) {
        
        rule$leverage
      }
    ),
    Conviction = sapply(
      rules,
      function(rule) {
        
        rule$conviction
      }
    )
  )
  
  
  rule_table[
    order(
      rule_table$Lift,
      decreasing = TRUE
    ),
  ]
}


# ------------------------------------------------------------------------------
# 35. Generate Rules
# ------------------------------------------------------------------------------

association_rules <- generate_association_rules(
  apriori_model,
  transaction_matrix,
  minimum_confidence = 0.50
)


head(
  association_rules,
  20
)


# ==============================================================================
# PART XVII
#
# INTERPRETING ASSOCIATION RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 36. Bread -> Butter
# ------------------------------------------------------------------------------

bread_butter <- calculate_rule_metrics(
  antecedent =
    "Bread",
  consequent =
    "Butter",
  transaction_matrix =
    transaction_matrix
)


bread_butter


# ------------------------------------------------------------------------------
# 37. Cereal -> Milk
# ------------------------------------------------------------------------------

cereal_milk <- calculate_rule_metrics(
  antecedent =
    "Cereal",
  consequent =
    "Milk",
  transaction_matrix =
    transaction_matrix
)


cereal_milk


# ------------------------------------------------------------------------------
# 38. Coffee -> Milk
# ------------------------------------------------------------------------------

coffee_milk <- calculate_rule_metrics(
  antecedent =
    "Coffee",
  consequent =
    "Milk",
  transaction_matrix =
    transaction_matrix
)


coffee_milk


# ------------------------------------------------------------------------------
# 39. Apples -> Yogurt
# ------------------------------------------------------------------------------

apple_yogurt <- calculate_rule_metrics(
  antecedent =
    "Apples",
  consequent =
    "Yogurt",
  transaction_matrix =
    transaction_matrix
)


apple_yogurt


# ==============================================================================
# PART XVIII
#
# SUPPORT VS CONFIDENCE VS LIFT
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Plot Association Rule Metrics
# ------------------------------------------------------------------------------

if (
  nrow(
    association_rules
  ) >
  0
) {
  
  plot(
    association_rules$Support,
    association_rules$Confidence,
    pch = 19,
    cex =
      0.5 +
      association_rules$Lift,
    xlab = "Support",
    ylab = "Confidence",
    main = "Association Rules"
  )
  
  
  abline(
    h = 0.5,
    lty = 2
  )
}


# ==============================================================================
# PART XIX
#
# WHY CONFIDENCE CAN BE MISLEADING
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. High-Prevalence Consequent
# ------------------------------------------------------------------------------

milk_support <- itemset_support(
  "Milk",
  transaction_matrix
)


milk_support


# Suppose a rule has:
#
#       confidence(X -> Milk) = 0.65
#
# If Milk already occurs in 0.65 of all baskets, this rule provides essentially
# no new information.
#
# This motivates lift:
#
#       lift(X -> Y)
#       =
#       confidence(X -> Y)
#       ------------------
#       support(Y)


# ==============================================================================
# PART XX
#
# INDEPENDENCE AND LIFT
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Empirical Independence Comparison
# ------------------------------------------------------------------------------

independence_table <- data.frame(
  Rule = c(
    "Bread -> Butter",
    "Cereal -> Milk",
    "Coffee -> Milk",
    "Apples -> Yogurt"
  ),
  Lift = c(
    bread_butter$lift,
    cereal_milk$lift,
    coffee_milk$lift,
    apple_yogurt$lift
  )
)


independence_table


# Interpretation:
#
#       Lift > 1   positive association
#       Lift = 1   approximate independence
#       Lift < 1   negative association


# ==============================================================================
# PART XXI
#
# RULES WITH SINGLE-ITEM CONSEQUENTS
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Restrict Rules to One Consequent Item
# ------------------------------------------------------------------------------

if (
  nrow(
    association_rules
  ) >
  0
) {
  
  consequent_sizes <- sapply(
    association_rules$Consequent,
    function(key) {
      
      length(
        key_to_itemset(
          key
        )
      )
    }
  )
  
  
  single_consequent_rules <-
    association_rules[
      consequent_sizes == 1,
      ,
      drop = FALSE
    ]
  
  
  head(
    single_consequent_rules,
    20
  )
}


# ==============================================================================
# PART XXII
#
# TOP RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 44. Highest Confidence
# ------------------------------------------------------------------------------

if (
  nrow(
    association_rules
  ) >
  0
) {
  
  top_confidence <- association_rules[
    order(
      association_rules$Confidence,
      decreasing = TRUE
    ),
  ]
  
  
  head(
    top_confidence,
    10
  )
}


# ------------------------------------------------------------------------------
# 45. Highest Lift
# ------------------------------------------------------------------------------

if (
  nrow(
    association_rules
  ) >
  0
) {
  
  top_lift <- association_rules[
    order(
      association_rules$Lift,
      decreasing = TRUE
    ),
  ]
  
  
  head(
    top_lift,
    10
  )
}


# ------------------------------------------------------------------------------
# 46. Highest Leverage
# ------------------------------------------------------------------------------

if (
  nrow(
    association_rules
  ) >
  0
) {
  
  top_leverage <- association_rules[
    order(
      association_rules$Leverage,
      decreasing = TRUE
    ),
  ]
  
  
  head(
    top_leverage,
    10
  )
}


# ==============================================================================
# PART XXIII
#
# FREQUENT CLOSED ITEMSETS
# ==============================================================================


# ------------------------------------------------------------------------------
# 47. Identify Closed Frequent Itemsets
# ------------------------------------------------------------------------------

is_closed_itemset <- function(
    itemset,
    frequent_itemsets,
    transaction_matrix
) {
  
  support_itemset <- itemset_support(
    itemset,
    transaction_matrix
  )
  
  
  for (
    other in frequent_itemsets
  ) {
    
    if (
      length(other) >
      length(itemset) &&
      all(
        itemset %in%
        other
      )
    ) {
      
      support_other <- itemset_support(
        other,
        transaction_matrix
      )
      
      
      if (
        abs(
          support_other -
          support_itemset
        ) <
        1e-12
      ) {
        
        return(
          FALSE
        )
      }
    }
  }
  
  
  TRUE
}


# ------------------------------------------------------------------------------
# 48. Find Closed Itemsets
# ------------------------------------------------------------------------------

closed_indicator <- sapply(
  apriori_model$frequent_itemsets,
  is_closed_itemset,
  frequent_itemsets =
    apriori_model$frequent_itemsets,
  transaction_matrix =
    transaction_matrix
)


closed_itemsets <- apriori_model$frequent_itemsets[
  closed_indicator
]


closed_table <- data.frame(
  Itemset = sapply(
    closed_itemsets,
    itemset_key
  ),
  Support = sapply(
    closed_itemsets,
    itemset_support,
    transaction_matrix =
      transaction_matrix
  )
)


head(
  closed_table,
  20
)


# ==============================================================================
# PART XXIV
#
# MAXIMAL FREQUENT ITEMSETS
# ==============================================================================


# ------------------------------------------------------------------------------
# 49. Identify Maximal Frequent Itemsets
# ------------------------------------------------------------------------------

is_maximal_itemset <- function(
    itemset,
    frequent_itemsets
) {
  
  for (
    other in frequent_itemsets
  ) {
    
    if (
      length(other) >
      length(itemset) &&
      all(
        itemset %in%
        other
      )
    ) {
      
      return(
        FALSE
      )
    }
  }
  
  
  TRUE
}


# ------------------------------------------------------------------------------
# 50. Find Maximal Frequent Itemsets
# ------------------------------------------------------------------------------

maximal_indicator <- sapply(
  apriori_model$frequent_itemsets,
  is_maximal_itemset,
  frequent_itemsets =
    apriori_model$frequent_itemsets
)


maximal_itemsets <- apriori_model$frequent_itemsets[
  maximal_indicator
]


maximal_table <- data.frame(
  Itemset = sapply(
    maximal_itemsets,
    itemset_key
  ),
  Support = sapply(
    maximal_itemsets,
    itemset_support,
    transaction_matrix =
      transaction_matrix
  )
)


maximal_table


# ==============================================================================
# PART XXV
#
# APRIORI PRUNING EXAMPLE
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Explicit Pruning Demonstration
# ------------------------------------------------------------------------------

example_previous_frequent <- list(
  c(
    "Bread",
    "Milk"
  ),
  c(
    "Bread",
    "Eggs"
  ),
  c(
    "Milk",
    "Eggs"
  )
)


example_candidate <- c(
  "Bread",
  "Milk",
  "Eggs"
)


all_subsets_frequent(
  example_candidate,
  example_previous_frequent
)


# Now remove Milk + Eggs.

example_previous_incomplete <- list(
  c(
    "Bread",
    "Milk"
  ),
  c(
    "Bread",
    "Eggs"
  )
)


all_subsets_frequent(
  example_candidate,
  example_previous_incomplete
)


# Because {Milk, Eggs} is not frequent, Apriori can discard:
#
#       {Bread, Milk, Eggs}
#
# without scanning the transaction database to calculate its support.


# ==============================================================================
# PART XXVI
#
# SCALING OF ITEMSET SPACE
# ==============================================================================


# ------------------------------------------------------------------------------
# 52. Exponential Growth
# ------------------------------------------------------------------------------

p_values <- 1:30


possible_itemsets <- 2^p_values -
  1


plot(
  p_values,
  possible_itemsets,
  type = "l",
  log = "y",
  xlab = "Number of Items p",
  ylab = "Possible Nonempty Itemsets (log scale)",
  main = "Combinatorial Growth of Itemsets"
)


# ==============================================================================
# PART XXVII
#
# OPTIONAL VERIFICATION WITH arules
# ==============================================================================


# ------------------------------------------------------------------------------
# 53. Verify if arules is Installed
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "arules",
    quietly = TRUE
  )
) {
  
  transaction_list <- lapply(
    transactions,
    unique
  )
  
  
  arules_transactions <- as(
    transaction_list,
    "transactions"
  )
  
  
  arules_model <- arules::apriori(
    arules_transactions,
    parameter = list(
      support = minimum_support,
      confidence = 0.50,
      target = "rules"
    )
  )
  
  
  cat(
    "\nNumber of manual rules:",
    nrow(
      association_rules
    ),
    "\n"
  )
  
  
  cat(
    "Number of arules rules:",
    length(
      arules_model
    ),
    "\n"
  )
  
  
  print(
    arules::inspect(
      head(
        arules::sort(
          arules_model,
          by = "lift"
        ),
        10
      )
    )
  )
}


# ==============================================================================
# PART XXVIII
#
# FINAL SUMMARY
# ==============================================================================


# ------------------------------------------------------------------------------
# 54. Output
# ------------------------------------------------------------------------------

cat(
  "Apriori Algorithm Summary\n"
)


cat(
  "-------------------------\n"
)


cat(
  "Transactions:",
  number_transactions,
  "\n"
)


cat(
  "Unique items:",
  length(items),
  "\n"
)


cat(
  "Possible nonempty itemsets:",
  all_possible_itemsets,
  "\n"
)


cat(
  "Minimum support:",
  apriori_model$minimum_support,
  "\n"
)


cat(
  "Frequent itemsets:",
  nrow(
    apriori_model$frequent_table
  ),
  "\n"
)


if (
  nrow(
    apriori_model$frequent_table
  ) >
  0
) {
  
  cat(
    "Largest frequent itemset:",
    max(
      apriori_model$frequent_table$Size
    ),
    "items\n"
  )
}


cat(
  "Association rules:",
  nrow(
    association_rules
  ),
  "\n"
)


cat(
  "Closed frequent itemsets:",
  length(
    closed_itemsets
  ),
  "\n"
)


cat(
  "Maximal frequent itemsets:",
  length(
    maximal_itemsets
  ),
  "\n"
)