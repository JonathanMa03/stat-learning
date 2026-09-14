# ==============================================================================
# Association Rules
# Jonathan Ma
#
# References:
#   The Elements of Statistical Learning
#
# Main ideas:
#   - Transaction data
#   - Frequent itemsets
#   - Association rules
#   - Antecedent and consequent
#   - Support
#   - Confidence
#   - Lift
#   - Leverage
#   - Conviction
#   - Rule redundancy
#   - Directionality
#   - Rare-item effects
#   - Rule filtering
#
#
# Association rule:
#
#       A -> B
#
# where:
#
#       A = antecedent
#       B = consequent
#       A intersect B = empty set
#
#
# Main metrics:
#
#       support(A -> B)
#       =
#       support(A union B)
#
#
#       confidence(A -> B)
#       =
#       support(A union B) / support(A)
#
#
#       lift(A -> B)
#       =
#       confidence(A -> B) / support(B)
#
#
#       leverage(A -> B)
#       =
#       support(A union B) - support(A) support(B)
#
#
#       conviction(A -> B)
#       =
#       (1 - support(B)) / (1 - confidence(A -> B))
#
# ==============================================================================


# ==============================================================================
# PART I
#
# GENERATE TRANSACTION DATA
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Items
# ------------------------------------------------------------------------------

set.seed(123)

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


number_transactions <- 1000


# ------------------------------------------------------------------------------
# 2. Simulate Transactions
# ------------------------------------------------------------------------------

transactions <- vector(
  "list",
  number_transactions
)


for (
  i in seq_len(
    number_transactions
  )
) {
  
  basket <- character(0)
  
  
  if (
    runif(1) < 0.45
  ) {
    
    basket <- c(
      basket,
      "Bread"
    )
  }
  
  
  if (
    runif(1) < 0.50
  ) {
    
    basket <- c(
      basket,
      "Milk"
    )
  }
  
  
  if (
    runif(1) < 0.30
  ) {
    
    basket <- c(
      basket,
      "Eggs"
    )
  }
  
  
  if (
    runif(1) < 0.18
  ) {
    
    basket <- c(
      basket,
      "Butter"
    )
  }
  
  
  if (
    runif(1) < 0.22
  ) {
    
    basket <- c(
      basket,
      "Cheese"
    )
  }
  
  
  if (
    runif(1) < 0.28
  ) {
    
    basket <- c(
      basket,
      "Coffee"
    )
  }
  
  
  if (
    runif(1) < 0.20
  ) {
    
    basket <- c(
      basket,
      "Tea"
    )
  }
  
  
  if (
    runif(1) < 0.25
  ) {
    
    basket <- c(
      basket,
      "Cereal"
    )
  }
  
  
  if (
    runif(1) < 0.30
  ) {
    
    basket <- c(
      basket,
      "Bananas"
    )
  }
  
  
  if (
    runif(1) < 0.25
  ) {
    
    basket <- c(
      basket,
      "Apples"
    )
  }
  
  
  if (
    runif(1) < 0.22
  ) {
    
    basket <- c(
      basket,
      "Yogurt"
    )
  }
  
  
  if (
    runif(1) < 0.18
  ) {
    
    basket <- c(
      basket,
      "Juice"
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Inject Association Structure
  # --------------------------------------------------------------------------
  
  if (
    "Bread" %in% basket &&
    runif(1) < 0.50
  ) {
    
    basket <- c(
      basket,
      "Butter"
    )
  }
  
  
  if (
    "Cereal" %in% basket &&
    runif(1) < 0.70
  ) {
    
    basket <- c(
      basket,
      "Milk"
    )
  }
  
  
  if (
    "Coffee" %in% basket &&
    runif(1) < 0.40
  ) {
    
    basket <- c(
      basket,
      "Milk"
    )
  }
  
  
  if (
    "Apples" %in% basket &&
    runif(1) < 0.45
  ) {
    
    basket <- c(
      basket,
      "Yogurt"
    )
  }
  
  
  if (
    all(
      c(
        "Bread",
        "Eggs"
      ) %in%
      basket
    ) &&
    runif(1) < 0.60
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
# 3. Inspect
# ------------------------------------------------------------------------------

transactions[1:10]


# ==============================================================================
# PART II
#
# TRANSACTION MATRIX
# ==============================================================================


# ------------------------------------------------------------------------------
# 4. Binary Encoding
# ------------------------------------------------------------------------------

transaction_matrix <- matrix(
  0L,
  nrow =
    number_transactions,
  ncol =
    length(items)
)


colnames(
  transaction_matrix
) <- items


for (
  i in seq_len(
    number_transactions
  )
) {
  
  transaction_matrix[
    i,
    transactions[[i]]
  ] <- 1L
}


head(
  transaction_matrix
)


# ==============================================================================
# PART III
#
# BASIC ITEMSET UTILITIES
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Canonical Itemset Key
# ------------------------------------------------------------------------------

itemset_key <- function(
    itemset
) {
  
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
# 6. Convert Key Back to Itemset
# ------------------------------------------------------------------------------

key_to_itemset <- function(
    key
) {
  
  if (
    length(key) == 0 ||
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
# 7. Itemset Count
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
  
  
  indicator <- rowSums(
    transaction_matrix[
      ,
      itemset,
      drop = FALSE
    ]
  ) ==
    length(itemset)
  
  
  sum(
    indicator
  )
}


# ------------------------------------------------------------------------------
# 8. Itemset Support
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


# ==============================================================================
# PART IV
#
# FREQUENT ITEMSETS
# ==============================================================================


# ------------------------------------------------------------------------------
# 9. Brute-Force Frequent Itemsets for Small Item Universe
# ------------------------------------------------------------------------------

find_frequent_itemsets <- function(
    transaction_matrix,
    minimum_support = 0.05,
    maximum_size = 4
) {
  
  items <- colnames(
    transaction_matrix
  )
  
  
  maximum_size <- min(
    maximum_size,
    length(items)
  )
  
  
  frequent_itemsets <- list()
  
  support_values <- numeric(0)
  
  counter <- 1
  
  
  for (
    size in seq_len(
      maximum_size
    )
  ) {
    
    candidates <- combn(
      items,
      size,
      simplify = FALSE
    )
    
    
    for (
      candidate in candidates
    ) {
      
      support_candidate <- itemset_support(
        candidate,
        transaction_matrix
      )
      
      
      if (
        support_candidate >=
        minimum_support
      ) {
        
        frequent_itemsets[[counter]] <-
          candidate
        
        
        support_values[counter] <-
          support_candidate
        
        
        counter <- counter + 1
      }
    }
  }
  
  
  list(
    itemsets =
      frequent_itemsets,
    support =
      support_values
  )
}


# ------------------------------------------------------------------------------
# 10. Find Frequent Itemsets
# ------------------------------------------------------------------------------

minimum_support <- 0.05


frequent_result <- find_frequent_itemsets(
  transaction_matrix,
  minimum_support =
    minimum_support,
  maximum_size = 4
)


length(
  frequent_result$itemsets
)


# ------------------------------------------------------------------------------
# 11. Frequent Itemset Table
# ------------------------------------------------------------------------------

frequent_table <- data.frame(
  Itemset = sapply(
    frequent_result$itemsets,
    itemset_key
  ),
  Size = lengths(
    frequent_result$itemsets
  ),
  Support =
    frequent_result$support
)


frequent_table <- frequent_table[
  order(
    frequent_table$Size,
    -frequent_table$Support
  ),
]


head(
  frequent_table,
  30
)


# ==============================================================================
# PART V
#
# ASSOCIATION RULE METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 12. Calculate Rule Metrics
# ------------------------------------------------------------------------------

calculate_rule_metrics <- function(
    antecedent,
    consequent,
    transaction_matrix
) {
  
  antecedent <- unique(
    antecedent
  )
  
  
  consequent <- unique(
    consequent
  )
  
  
  if (
    length(
      intersect(
        antecedent,
        consequent
      )
    ) >
    0
  ) {
    
    stop(
      "Antecedent and consequent must be disjoint."
    )
  }
  
  
  union_itemset <- union(
    antecedent,
    consequent
  )
  
  
  support_A <- itemset_support(
    antecedent,
    transaction_matrix
  )
  
  
  support_B <- itemset_support(
    consequent,
    transaction_matrix
  )
  
  
  support_AB <- itemset_support(
    union_itemset,
    transaction_matrix
  )
  
  
  confidence <- if (
    support_A >
    0
  ) {
    
    support_AB /
      support_A
    
  } else {
    
    NA_real_
  }
  
  
  lift <- if (
    support_B >
    0
  ) {
    
    confidence /
      support_B
    
  } else {
    
    NA_real_
  }
  
  
  leverage <- support_AB -
    support_A *
    support_B
  
  
  conviction <- if (
    is.na(
      confidence
    )
  ) {
    
    NA_real_
    
  } else if (
    confidence >=
    1 -
    1e-12
  ) {
    
    Inf
    
  } else {
    
    (
      1 -
        support_B
    ) /
      (
        1 -
          confidence
      )
  }
  
  
  cosine <- if (
    support_A >
    0 &&
    support_B >
    0
  ) {
    
    support_AB /
      sqrt(
        support_A *
          support_B
      )
    
  } else {
    
    NA_real_
  }
  
  
  jaccard <- if (
    support_A +
    support_B -
    support_AB >
    0
  ) {
    
    support_AB /
      (
        support_A +
          support_B -
          support_AB
      )
    
  } else {
    
    NA_real_
  }
  
  
  confidence_reverse <- if (
    support_B >
    0
  ) {
    
    support_AB /
      support_B
    
  } else {
    
    NA_real_
  }
  
  
  list(
    support_A =
      support_A,
    support_B =
      support_B,
    support =
      support_AB,
    confidence =
      confidence,
    reverse_confidence =
      confidence_reverse,
    lift =
      lift,
    leverage =
      leverage,
    conviction =
      conviction,
    cosine =
      cosine,
    jaccard =
      jaccard
  )
}


# ==============================================================================
# PART VI
#
# EXAMPLE RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 13. Bread -> Butter
# ------------------------------------------------------------------------------

bread_to_butter <- calculate_rule_metrics(
  antecedent =
    "Bread",
  consequent =
    "Butter",
  transaction_matrix =
    transaction_matrix
)


bread_to_butter


# ------------------------------------------------------------------------------
# 14. Butter -> Bread
# ------------------------------------------------------------------------------

butter_to_bread <- calculate_rule_metrics(
  antecedent =
    "Butter",
  consequent =
    "Bread",
  transaction_matrix =
    transaction_matrix
)


butter_to_bread


# ------------------------------------------------------------------------------
# 15. Cereal -> Milk
# ------------------------------------------------------------------------------

cereal_to_milk <- calculate_rule_metrics(
  antecedent =
    "Cereal",
  consequent =
    "Milk",
  transaction_matrix =
    transaction_matrix
)


cereal_to_milk


# ------------------------------------------------------------------------------
# 16. Bread + Eggs -> Cheese
# ------------------------------------------------------------------------------

bread_eggs_to_cheese <- calculate_rule_metrics(
  antecedent = c(
    "Bread",
    "Eggs"
  ),
  consequent =
    "Cheese",
  transaction_matrix =
    transaction_matrix
)


bread_eggs_to_cheese


# ==============================================================================
# PART VII
#
# RULE DIRECTION MATTERS
# ==============================================================================


# ------------------------------------------------------------------------------
# 17. Compare A -> B and B -> A
# ------------------------------------------------------------------------------

direction_comparison <- data.frame(
  Rule = c(
    "Bread -> Butter",
    "Butter -> Bread"
  ),
  Support = c(
    bread_to_butter$support,
    butter_to_bread$support
  ),
  Confidence = c(
    bread_to_butter$confidence,
    butter_to_bread$confidence
  ),
  Lift = c(
    bread_to_butter$lift,
    butter_to_bread$lift
  ),
  Leverage = c(
    bread_to_butter$leverage,
    butter_to_bread$leverage
  )
)


direction_comparison


# Support and lift are symmetric with respect to A and B.
#
# Confidence is directional:
#
#       P(B | A)
#
# is generally not equal to:
#
#       P(A | B)


# ==============================================================================
# PART VIII
#
# GENERATE SUBSETS
# ==============================================================================


# ------------------------------------------------------------------------------
# 18. Nonempty Proper Subsets
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
    subset_size in seq_len(
      m - 1
    )
  ) {
    
    combinations <- combn(
      itemset,
      subset_size,
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


# ==============================================================================
# PART IX
#
# GENERATE ALL ASSOCIATION RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 19. Generate Rules
# ------------------------------------------------------------------------------

generate_rules <- function(
    frequent_itemsets,
    transaction_matrix,
    minimum_support = 0,
    minimum_confidence = 0,
    minimum_lift = 0
) {
  
  rule_list <- list()
  
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
        metrics$support >=
        minimum_support &&
        metrics$confidence >=
        minimum_confidence &&
        metrics$lift >=
        minimum_lift
      ) {
        
        rule_list[[rule_index]] <- list(
          antecedent =
            antecedent,
          consequent =
            consequent,
          support_A =
            metrics$support_A,
          support_B =
            metrics$support_B,
          support =
            metrics$support,
          confidence =
            metrics$confidence,
          reverse_confidence =
            metrics$reverse_confidence,
          lift =
            metrics$lift,
          leverage =
            metrics$leverage,
          conviction =
            metrics$conviction,
          cosine =
            metrics$cosine,
          jaccard =
            metrics$jaccard
        )
        
        
        rule_index <- rule_index + 1
      }
    }
  }
  
  
  if (
    length(rule_list) ==
    0
  ) {
    
    return(
      data.frame()
    )
  }
  
  
  rule_table <- data.frame(
    Antecedent = sapply(
      rule_list,
      function(rule) {
        
        itemset_key(
          rule$antecedent
        )
      }
    ),
    Consequent = sapply(
      rule_list,
      function(rule) {
        
        itemset_key(
          rule$consequent
        )
      }
    ),
    Antecedent_Support = sapply(
      rule_list,
      function(rule) {
        
        rule$support_A
      }
    ),
    Consequent_Support = sapply(
      rule_list,
      function(rule) {
        
        rule$support_B
      }
    ),
    Support = sapply(
      rule_list,
      function(rule) {
        
        rule$support
      }
    ),
    Confidence = sapply(
      rule_list,
      function(rule) {
        
        rule$confidence
      }
    ),
    Reverse_Confidence = sapply(
      rule_list,
      function(rule) {
        
        rule$reverse_confidence
      }
    ),
    Lift = sapply(
      rule_list,
      function(rule) {
        
        rule$lift
      }
    ),
    Leverage = sapply(
      rule_list,
      function(rule) {
        
        rule$leverage
      }
    ),
    Conviction = sapply(
      rule_list,
      function(rule) {
        
        rule$conviction
      }
    ),
    Cosine = sapply(
      rule_list,
      function(rule) {
        
        rule$cosine
      }
    ),
    Jaccard = sapply(
      rule_list,
      function(rule) {
        
        rule$jaccard
      }
    )
  )
  
  
  rownames(
    rule_table
  ) <- NULL
  
  
  rule_table
}


# ------------------------------------------------------------------------------
# 20. Generate Rules
# ------------------------------------------------------------------------------

rules <- generate_rules(
  frequent_result$itemsets,
  transaction_matrix,
  minimum_support = 0.05,
  minimum_confidence = 0.40,
  minimum_lift = 1.00
)


nrow(
  rules
)


head(
  rules,
  20
)


# ==============================================================================
# PART X
#
# TOP RULES BY DIFFERENT METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 21. Highest Confidence
# ------------------------------------------------------------------------------

top_confidence <- rules[
  order(
    rules$Confidence,
    decreasing = TRUE
  ),
]


head(
  top_confidence,
  10
)


# ------------------------------------------------------------------------------
# 22. Highest Lift
# ------------------------------------------------------------------------------

top_lift <- rules[
  order(
    rules$Lift,
    decreasing = TRUE
  ),
]


head(
  top_lift,
  10
)


# ------------------------------------------------------------------------------
# 23. Highest Leverage
# ------------------------------------------------------------------------------

top_leverage <- rules[
  order(
    rules$Leverage,
    decreasing = TRUE
  ),
]


head(
  top_leverage,
  10
)


# ------------------------------------------------------------------------------
# 24. Highest Conviction
# ------------------------------------------------------------------------------

top_conviction <- rules[
  order(
    rules$Conviction,
    decreasing = TRUE,
    na.last = TRUE
  ),
]


head(
  top_conviction,
  10
)


# ==============================================================================
# PART XI
#
# WHY CONFIDENCE CAN BE MISLEADING
# ==============================================================================


# ------------------------------------------------------------------------------
# 25. High-Prevalence Consequent
# ------------------------------------------------------------------------------

milk_support <- itemset_support(
  "Milk",
  transaction_matrix
)


milk_support


# ------------------------------------------------------------------------------
# 26. Rules Predicting Milk
# ------------------------------------------------------------------------------

milk_rules <- rules[
  rules$Consequent ==
    "Milk",
  ,
  drop = FALSE
]


milk_rules <- milk_rules[
  order(
    milk_rules$Confidence,
    decreasing = TRUE
  ),
]


head(
  milk_rules,
  20
)


# A rule can have high confidence simply because Milk is already common.
#
# Lift adjusts for this baseline prevalence.


# ==============================================================================
# PART XII
#
# HIGH CONFIDENCE BUT LOW LIFT
# ==============================================================================


# ------------------------------------------------------------------------------
# 27. Search for Potentially Misleading Rules
# ------------------------------------------------------------------------------

misleading_rules <- rules[
  rules$Confidence >=
    0.60 &
    rules$Lift <=
    1.10,
  ,
  drop = FALSE
]


misleading_rules[
  order(
    misleading_rules$Confidence,
    decreasing = TRUE
  ),
]


# ==============================================================================
# PART XIII
#
# HIGH LIFT BUT LOW SUPPORT
# ==============================================================================


# ------------------------------------------------------------------------------
# 28. Rare High-Lift Rules
# ------------------------------------------------------------------------------

rare_high_lift <- rules[
  rules$Lift >=
    1.5,
  ,
  drop = FALSE
]


rare_high_lift <- rare_high_lift[
  order(
    rare_high_lift$Support,
    decreasing = FALSE
  ),
]


head(
  rare_high_lift,
  20
)


# High lift alone can also be misleading when based on very rare combinations.
#
# A useful rule generally needs both:
#
#       sufficient prevalence
#
# and
#
#       meaningful association.


# ==============================================================================
# PART XIV
#
# VISUALIZE RULE METRICS
# ==============================================================================


# ------------------------------------------------------------------------------
# 29. Support vs Confidence
# ------------------------------------------------------------------------------

if (
  nrow(rules) >
  0
) {
  
  plot(
    rules$Support,
    rules$Confidence,
    pch = 19,
    cex =
      0.5 +
      pmin(
        rules$Lift,
        3
      ) /
      2,
    xlab = "Support",
    ylab = "Confidence",
    main = "Association Rule Support vs Confidence"
  )
}


# ------------------------------------------------------------------------------
# 30. Support vs Lift
# ------------------------------------------------------------------------------

if (
  nrow(rules) >
  0
) {
  
  plot(
    rules$Support,
    rules$Lift,
    pch = 19,
    xlab = "Support",
    ylab = "Lift",
    main = "Association Rule Support vs Lift"
  )
  
  
  abline(
    h = 1,
    lty = 2
  )
}


# ------------------------------------------------------------------------------
# 31. Confidence vs Lift
# ------------------------------------------------------------------------------

if (
  nrow(rules) >
  0
) {
  
  plot(
    rules$Confidence,
    rules$Lift,
    pch = 19,
    xlab = "Confidence",
    ylab = "Lift",
    main = "Association Rule Confidence vs Lift"
  )
  
  
  abline(
    h = 1,
    lty = 2
  )
}


# ==============================================================================
# PART XV
#
# SINGLE-CONSEQUENT RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 32. Consequent Size
# ------------------------------------------------------------------------------

consequent_size <- sapply(
  rules$Consequent,
  function(key) {
    
    length(
      key_to_itemset(
        key
      )
    )
  }
)


single_consequent_rules <- rules[
  consequent_size == 1,
  ,
  drop = FALSE
]


head(
  single_consequent_rules,
  20
)


# These are often easiest to interpret:
#
#       A -> one item


# ==============================================================================
# PART XVI
#
# RULES FOR A SPECIFIC CONSEQUENT
# ==============================================================================


# ------------------------------------------------------------------------------
# 33. Rules Predicting Cheese
# ------------------------------------------------------------------------------

cheese_rules <- single_consequent_rules[
  single_consequent_rules$Consequent ==
    "Cheese",
  ,
  drop = FALSE
]


cheese_rules <- cheese_rules[
  order(
    cheese_rules$Lift,
    decreasing = TRUE
  ),
]


cheese_rules


# ==============================================================================
# PART XVII
#
# RULES CONTAINING A SPECIFIC ANTECEDENT ITEM
# ==============================================================================


# ------------------------------------------------------------------------------
# 34. Rules Whose Antecedent Contains Bread
# ------------------------------------------------------------------------------

contains_item <- function(
    key,
    item
) {
  
  item %in%
    key_to_itemset(
      key
    )
}


bread_rule_indicator <- sapply(
  rules$Antecedent,
  contains_item,
  item = "Bread"
)


bread_rules <- rules[
  bread_rule_indicator,
  ,
  drop = FALSE
]


bread_rules <- bread_rules[
  order(
    bread_rules$Lift,
    decreasing = TRUE
  ),
]


head(
  bread_rules,
  20
)


# ==============================================================================
# PART XVIII
#
# RULE REDUNDANCY
# ==============================================================================


# ------------------------------------------------------------------------------
# 35. Antecedent Subset Check
# ------------------------------------------------------------------------------

is_subset_key <- function(
    smaller_key,
    larger_key
) {
  
  smaller <- key_to_itemset(
    smaller_key
  )
  
  
  larger <- key_to_itemset(
    larger_key
  )
  
  
  all(
    smaller %in%
      larger
  )
}


# ------------------------------------------------------------------------------
# 36. Simple Redundancy Criterion
# ------------------------------------------------------------------------------

is_rule_redundant <- function(
    rule_index,
    rule_table,
    tolerance = 1e-12
) {
  
  target_rule <- rule_table[
    rule_index,
    ,
    drop = FALSE
  ]
  
  
  same_consequent <- which(
    rule_table$Consequent ==
      target_rule$Consequent
  )
  
  
  same_consequent <- setdiff(
    same_consequent,
    rule_index
  )
  
  
  if (
    length(
      same_consequent
    ) ==
    0
  ) {
    
    return(
      FALSE
    )
  }
  
  
  for (
    j in same_consequent
  ) {
    
    smaller_antecedent <- is_subset_key(
      rule_table$Antecedent[j],
      target_rule$Antecedent
    )
    
    
    strictly_smaller <- length(
      key_to_itemset(
        rule_table$Antecedent[j]
      )
    ) <
      length(
        key_to_itemset(
          target_rule$Antecedent
        )
      )
    


no_worse_confidence <-
  rule_table$Confidence[j] >=
  target_rule$Confidence -
  tolerance


if (
  smaller_antecedent &&
  strictly_smaller &&
  no_worse_confidence
) {
  
  return(
    TRUE
  )
}
  }
  
  
  FALSE
}


# ------------------------------------------------------------------------------
# 37. Identify Redundant Rules
# ------------------------------------------------------------------------------

if (
  nrow(rules) >
  0
) {
  
  redundancy_indicator <- sapply(
    seq_len(
      nrow(rules)
    ),
    is_rule_redundant,
    rule_table =
      rules
  )
  
  
  table(
    Redundant =
      redundancy_indicator
  )
  
  
  nonredundant_rules <- rules[
    !redundancy_indicator,
    ,
    drop = FALSE
  ]
  
  
  nonredundant_rules <- nonredundant_rules[
    order(
      nonredundant_rules$Lift,
      decreasing = TRUE
    ),
  ]
  
  
  head(
    nonredundant_rules,
    20
  )
}


# ==============================================================================
# PART XIX
#
# IMPROVEMENT OVER SIMPLER RULE
# ==============================================================================


# ------------------------------------------------------------------------------
# 38. Confidence Improvement
# ------------------------------------------------------------------------------

rule_confidence_improvement <- function(
    rule_index,
    rule_table
) {
  
  target <- rule_table[
    rule_index,
    ,
    drop = FALSE
  ]
  
  
  target_antecedent <- key_to_itemset(
    target$Antecedent
  )
  
  
  if (
    length(
      target_antecedent
    ) <=
    1
  ) {
    
    return(
      NA_real_
    )
  }
  
  
  candidates <- which(
    rule_table$Consequent ==
      target$Consequent
  )
  
  
  candidate_confidences <- numeric(0)
  
  
  for (
    j in candidates
  ) {
    
    candidate_antecedent <- key_to_itemset(
      rule_table$Antecedent[j]
    )
    
    
    if (
      length(
        candidate_antecedent
      ) <
      length(
        target_antecedent
      ) &&
      all(
        candidate_antecedent %in%
        target_antecedent
      )
    ) {
      
      candidate_confidences <- c(
        candidate_confidences,
        rule_table$Confidence[j]
      )
    }
  }
  
  
  if (
    length(
      candidate_confidences
    ) ==
    0
  ) {
    
    return(
      NA_real_
    )
  }
  
  
  target$Confidence -
    max(
      candidate_confidences
    )
}


# ------------------------------------------------------------------------------
# 39. Compute Rule Improvement
# ------------------------------------------------------------------------------

if (
  nrow(rules) >
  0
) {
  
  rules$Confidence_Improvement <- sapply(
    seq_len(
      nrow(rules)
    ),
    rule_confidence_improvement,
    rule_table =
      rules
  )
  
  
  improved_rules <- rules[
    !is.na(
      rules$Confidence_Improvement
    ),
    ,
    drop = FALSE
  ]
  
  
  improved_rules <- improved_rules[
    order(
      improved_rules$Confidence_Improvement,
      decreasing = TRUE
    ),
  ]
  
  
  head(
    improved_rules,
    20
  )
}


# ==============================================================================
# PART XX
#
# LIFT IS SYMMETRIC
# ==============================================================================


# ------------------------------------------------------------------------------
# 40. Demonstrate Symmetry
# ------------------------------------------------------------------------------

bread_butter_lift <- calculate_rule_metrics(
  "Bread",
  "Butter",
  transaction_matrix
)$lift


butter_bread_lift <- calculate_rule_metrics(
  "Butter",
  "Bread",
  transaction_matrix
)$lift


bread_butter_lift


butter_bread_lift


# Lift(A -> B) = Lift(B -> A)
#
# because:
#
#       lift
#       =
#       P(A,B) / [P(A) P(B)]


# ==============================================================================
# PART XXI
#
# LEVERAGE IS ALSO SYMMETRIC
# ==============================================================================


# ------------------------------------------------------------------------------
# 41. Compare Leverage
# ------------------------------------------------------------------------------

calculate_rule_metrics(
  "Bread",
  "Butter",
  transaction_matrix
)$leverage


calculate_rule_metrics(
  "Butter",
  "Bread",
  transaction_matrix
)$leverage


# ==============================================================================
# PART XXII
#
# NEGATIVE ASSOCIATIONS
# ==============================================================================


# ------------------------------------------------------------------------------
# 42. Find Rules with Lift < 1
# ------------------------------------------------------------------------------

all_rules <- generate_rules(
  frequent_result$itemsets,
  transaction_matrix,
  minimum_support = 0.05,
  minimum_confidence = 0,
  minimum_lift = 0
)


negative_rules <- all_rules[
  all_rules$Lift <
    1,
  ,
  drop = FALSE
]


negative_rules <- negative_rules[
  order(
    negative_rules$Lift,
    decreasing = FALSE
  ),
]


head(
  negative_rules,
  20
)


# Lift below 1 suggests that the consequent occurs less often in the presence of
# the antecedent than its baseline prevalence would suggest.


# ==============================================================================
# PART XXIII
#
# RULE FILTERING PIPELINE
# ==============================================================================


# ------------------------------------------------------------------------------
# 43. Practical Filtering Function
# ------------------------------------------------------------------------------

filter_rules <- function(
    rules,
    minimum_support = 0.05,
    minimum_confidence = 0.50,
    minimum_lift = 1.20,
    minimum_leverage = 0
) {
  
  rules[
    rules$Support >=
      minimum_support &
      rules$Confidence >=
      minimum_confidence &
      rules$Lift >=
      minimum_lift &
      rules$Leverage >=
      minimum_leverage,
    ,
    drop = FALSE
  ]
}


# ------------------------------------------------------------------------------
# 44. Filter
# ------------------------------------------------------------------------------

filtered_rules <- filter_rules(
  all_rules,
  minimum_support = 0.05,
  minimum_confidence = 0.50,
  minimum_lift = 1.20,
  minimum_leverage = 0.01
)


filtered_rules <- filtered_rules[
  order(
    filtered_rules$Lift,
    decreasing = TRUE
  ),
]


filtered_rules


# ==============================================================================
# PART XXIV
#
# SUPPORT-CONFIDENCE TRADEOFF
# ==============================================================================


# ------------------------------------------------------------------------------
# 45. Compare Thresholds
# ------------------------------------------------------------------------------

support_thresholds <- c(
  0.02,
  0.05,
  0.08,
  0.10
)


confidence_thresholds <- c(
  0.30,
  0.50,
  0.70
)


threshold_grid <- expand.grid(
  Minimum_Support =
    support_thresholds,
  Minimum_Confidence =
    confidence_thresholds
)


threshold_grid$Number_Rules <- NA_integer_


for (
  i in seq_len(
    nrow(
      threshold_grid
    )
  )
) {
  
  frequent_i <- find_frequent_itemsets(
    transaction_matrix,
    minimum_support =
      threshold_grid$Minimum_Support[i],
    maximum_size = 4
  )
  
  
  rules_i <- generate_rules(
    frequent_i$itemsets,
    transaction_matrix,
    minimum_support =
      threshold_grid$Minimum_Support[i],
    minimum_confidence =
      threshold_grid$Minimum_Confidence[i],
    minimum_lift = 0
  )
  
  
  threshold_grid$Number_Rules[i] <-
    nrow(
      rules_i
    )
}


threshold_grid


# ==============================================================================
# PART XXV
#
# HOLDOUT VALIDATION OF RULES
# ==============================================================================


# ------------------------------------------------------------------------------
# 46. Split Transactions
# ------------------------------------------------------------------------------

set.seed(999)


training_indices <- sample(
  seq_len(
    number_transactions
  ),
  size = floor(
    0.7 *
      number_transactions
  )
)


test_indices <- setdiff(
  seq_len(
    number_transactions
  ),
  training_indices
)


transaction_train <- transaction_matrix[
  training_indices,
  ,
  drop = FALSE
]


transaction_test <- transaction_matrix[
  test_indices,
  ,
  drop = FALSE
]


# ------------------------------------------------------------------------------
# 47. Learn Rules on Training Data
# ------------------------------------------------------------------------------

frequent_train <- find_frequent_itemsets(
  transaction_train,
  minimum_support = 0.05,
  maximum_size = 4
)


train_rules <- generate_rules(
  frequent_train$itemsets,
  transaction_train,
  minimum_support = 0.05,
  minimum_confidence = 0.50,
  minimum_lift = 1.10
)


# ------------------------------------------------------------------------------
# 48. Evaluate Same Rules on Test Data
# ------------------------------------------------------------------------------

evaluate_rule_table <- function(
    rule_table,
    transaction_matrix
) {
  
  if (
    nrow(
      rule_table
    ) ==
    0
  ) {
    
    return(
      data.frame()
    )
  }
  
  
  result <- rule_table[
    ,
    c(
      "Antecedent",
      "Consequent"
    ),
    drop = FALSE
  ]
  
  
  result$Support <- NA_real_
  
  result$Confidence <- NA_real_
  
  result$Lift <- NA_real_
  
  
  for (
    i in seq_len(
      nrow(
        result
      )
    )
  ) {
    
    metrics <- calculate_rule_metrics(
      key_to_itemset(
        result$Antecedent[i]
      ),
      key_to_itemset(
        result$Consequent[i]
      ),
      transaction_matrix
    )
    
    
    result$Support[i] <-
      metrics$support
    
    
    result$Confidence[i] <-
      metrics$confidence
    
    
    result$Lift[i] <-
      metrics$lift
  }
  
  
  result
}


test_rule_metrics <- evaluate_rule_table(
  train_rules,
  transaction_test
)


# ------------------------------------------------------------------------------
# 49. Combine Train/Test Rule Metrics
# ------------------------------------------------------------------------------

if (
  nrow(
    train_rules
  ) >
  0
) {
  
  rule_stability <- data.frame(
    Antecedent =
      train_rules$Antecedent,
    Consequent =
      train_rules$Consequent,
    Train_Support =
      train_rules$Support,
    Test_Support =
      test_rule_metrics$Support,
    Train_Confidence =
      train_rules$Confidence,
    Test_Confidence =
      test_rule_metrics$Confidence,
    Train_Lift =
      train_rules$Lift,
    Test_Lift =
      test_rule_metrics$Lift
  )
  
  
  head(
    rule_stability[
      order(
        rule_stability$Train_Lift,
        decreasing = TRUE
      ),
    ],
    20
  )
}


# ==============================================================================
# PART XXVI
#
# RULE STABILITY
# ==============================================================================


# ------------------------------------------------------------------------------
# 50. Train vs Test Lift
# ------------------------------------------------------------------------------

if (
  exists(
    "rule_stability"
  )
) {
  
  plot(
    rule_stability$Train_Lift,
    rule_stability$Test_Lift,
    pch = 19,
    xlab = "Training Lift",
    ylab = "Test Lift",
    main = "Association Rule Stability"
  )
  
  
  abline(
    0,
    1,
    lty = 2
  )
}


# Rules discovered through exploratory search can look much stronger in the data
# used to discover them than in new data.


# ==============================================================================
# PART XXVII
#
# OPTIONAL VERIFICATION WITH arules
# ==============================================================================


# ------------------------------------------------------------------------------
# 51. Verify if arules is Installed
# ------------------------------------------------------------------------------

if (
  requireNamespace(
    "arules",
    quietly = TRUE
  )
) {
  
  arules_transactions <- as(
    transactions,
    "transactions"
  )
  
  
  arules_rules <- arules::apriori(
    arules_transactions,
    parameter = list(
      support = 0.05,
      confidence = 0.50,
      minlen = 2,
      target = "rules"
    )
  )
  
  
  cat(
    "Manual rules:",
    nrow(
      train_rules
    ),
    "\n"
  )
  
  
  cat(
    "arules rules:",
    length(
      arules_rules
    ),
    "\n"
  )
  
  
  print(
    arules::inspect(
      head(
        arules::sort(
          arules_rules,
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
# 52. Summary
# ------------------------------------------------------------------------------

cat(
  "Association Rules Summary\n"
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
  "Items:",
  length(items),
  "\n"
)


cat(
  "Frequent itemsets:",
  length(
    frequent_result$itemsets
  ),
  "\n"
)


cat(
  "Rules satisfying initial filters:",
  nrow(rules),
  "\n"
)


cat(
  "Rules satisfying stricter practical filters:",
  nrow(filtered_rules),
  "\n"
)


if (
  nrow(rules) >
  0
) {
  
  cat(
    "Maximum confidence:",
    round(
      max(
        rules$Confidence
      ),
      4
    ),
    "\n"
  )
  
  
  cat(
    "Maximum lift:",
    round(
      max(
        rules$Lift
      ),
      4
    ),
    "\n"
  )
  
  
  cat(
    "Maximum leverage:",
    round(
      max(
        rules$Leverage
      ),
      4
    ),
    "\n"
  )
}