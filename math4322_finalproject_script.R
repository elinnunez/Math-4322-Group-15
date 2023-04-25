#read data from csv file
set.seed(1)
base_adult <- read.csv("adult.csv", stringsAsFactors = TRUE)
adult <- read.csv("adult.csv")
summary(base_adult)

# Clean data by removing rows with missing values
unknown_val_cols <- c("occupation", "workclass", "country")
base_adult <- base_adult[rowSums(base_adult[, unknown_val_cols] == " ?") == 0, ]
base_adult$income <- as.integer(base_adult$income != "<=50K") # 0 for <50k, 1 for >= 50k
base_adult$income <- as.factor(base_adult$income)
base_adult$country <- as.integer(base_adult$country != " United-States") # 0 for US, 1 everything else
base_adult$country <- as.factor(base_adult$country)
base_adult <- na.omit(base_adult)
summary(base_adult)

# Split data into training set (80%) and testing set (20%)
#This code uses the runif() function to generate a vector of random values between 0 and 1 of length equal to the number of rows in data. 
#The < operator used to compare each value vector to 0.8, which produces a logical vector where TRUE values correspond to the rows selected for training set and FALSE values correspond to rows selected for test set. 
train_rows <- runif(nrow(base_adult)) < 0.8
train <- base_adult[train_rows, ]
test <- base_adult[!train_rows, ]
print(nrow(train))
print(nrow(test))

#check correlations between the numeric values
is_numeric <- sapply(adult, is.numeric)
pairs(adult[, is_numeric])
cor(adult[, is_numeric])
#note: not really much correlation between the variables, no clear pattern in scatterplots output

##Logistic Regression Method

# Fit a logistic regression model
lr.glm <- glm(income ~ ., data = train, family = "binomial")
#summary(lr.glm)

# Make predictions on the test set
lr.pred <- predict.glm(lr.glm, test, type = "response")
pred_income <- ifelse(lr.pred > 0.5, TRUE, FALSE)

# Create a confusion matrix to evaluate the performance of the model and compute accuracy metrics
confusion_matrix <- table(test$income, pred_income)
confusion_matrix

# Calculate misclassification rate
misclassification_rate <- (confusion_matrix[2,1] + confusion_matrix[1,2]) / sum(confusion_matrix)
misclassification_rate # .3486127 = 35%, higher than we'd like, means our model predicted only about 65% of the data correctly
# note: The misclassification rate is simply the proportion of misclassified cases out of the total number of cases in the test data. 
#It is a measure of the model's overall accuracy, with lower values indicating better performance. 

#we should find variables that aren't as important in predicting response and remove them from our model for better results
summary(lr.glm)

#--------
# Extract p-values of coefficients
p_values <- summary(lr.glm)$coef[, 4]

# Identify significant variables based on p-values
sig_vars <- names(p_values[p_values < 0.05]) # pick based off this
sig_vars
#-------

# Subset train and test data using significant variables
sig <- c("age","workclass", "education","occupation","relationship","sex","capital_gain","capital_loss","hours_per_week","income")
train_sig <- train[, sig]
test_sig <- test[, sig]

# Refit logistic regression model using significant variables
lr_sig <- glm(income ~ ., train_sig, family = "binomial")

# Make predictions on test data
glm.pred <- predict(lr_sig, test_sig, type = "response")
y_hat <- ifelse(glm.pred > 0.5, 1, 0)

# Create a confusion matrix to evaluate the performance of the model and compute accuracy metrics
conf_matrix <- table(test_sig$income, y_hat)
conf_matrix

# Calculate misclassification rate
misclass_rate <- mean(y_hat != test_sig$income)
misclass_rate #.3473918, practically the same as original missclassification rate after removing non-significant variables.
# thus these are the most important variables in how income is affected

#todo: move logic into functions to evaluate in trial runs for models, 10 times


# Classification Decision Tree

# Load library
library(tree)

# Build the tree
btree <- tree(income ~ ., data = train)

# Plot tree
plot(btree)
text(btree)

# Get summary of base tree
base_tree <- summary(btree)
base_tree # current terminal nodes: 5, variables used: relationship, capital_gain, education

# Optimize by pruning

# Define a function to find the pruning parameter with the lowest deviance
best_pruning <- function(cv_results) {
  min_dev <- min(cv_results$dev)
  min_dev_idx <- which(cv_results$dev == min_dev)
  return(cv_results$size[min_dev_idx])
}

# Cross-validate and find the best pruning parameter
class_cv <- cv.tree(btree, FUN = prune.misclass)
best_param <- best_pruning(class_cv)

# Plot the cross-validation results
plot(class_cv$size, class_cv$dev, type = "b", xlab = "# of terminal nodes", ylab = "Deviance")

# Prune the tree using the best parameter
prunedTree <- prune.misclass(btree, best = best_param)

# Print the summary of the pruned tree
summary(prunedTree) # variables: relationship, capital_gain

# Plot the pruned tree and label the nodes
plot(prunedTree)
text(prunedTree)

# summary, still the same misclassfication error rate: .3627 after pruning

# Trials Run
evaluate_trials <- function() {
  trials <- 10
  
  # Evaluate misclassification rate over multiple trials
  logreg_misclass_rates <- numeric(trials)
  tree_misclass_rates <- numeric(trials)
  
  for (i in 1:trials) {
    set.seed(i)
    
    # Split data into train and test sets
    train_rows_x <- runif(nrow(base_adult)) < 0.8
    train_x <- base_adult[train_rows_x, ]
    test_x <- base_adult[!train_rows_x, ]
    summary(train_x)
    
    # Subset train and test data using significant variables
    train_sig <- train_x[, c("age","workclass", "education","occupation","relationship","sex","capital_gain","capital_loss","hours_per_week","income")]
    test_sig <- test_x[, c("age", "workclass", "education","occupation","relationship","sex","capital_gain","capital_loss","hours_per_week","income")]
    
    # Refit logistic regression model using significant variables
    lr_sig <- glm(income ~ ., train_sig, family = "binomial")
    
    # Make predictions on test data
    glm.pred <- predict(lr_sig, test_sig, type = "response")
    y_hat <- ifelse(glm.pred > 0.5, 1, 0)
    
    # Calculate misclassification rate for logistic regression
    logreg_misclass_rate <- mean(y_hat != test_sig$income)
    logreg_misclass_rates[i] = logreg_misclass_rate
    
    # Build the tree
    ba_tree <- tree(income ~ ., train_x)
    
    # Optimize by pruning
    class_cv <- cv.tree(ba_tree, FUN = prune.misclass)
    best_param <- best_pruning(class_cv)
    
    # Prune the tree using the best parameter
    pruned_tree <- prune.misclass(ba_tree, best = best_param)
    
    # Calculate misclassification rate for classification decision tree
    tree_sum <- summary(pruned_tree)
    tree_misclass_rate <- tree_sum$misclass[1]/tree_sum$misclass[2]
    tree_misclass_rates[i] <- misclass_rate
  }
  
  # Return the average misclassification rates for both
  return (list(mean(logreg_misclass_rates), mean(tree_misclass_rates)))
}

misclass_rate_vals = evaluate_trials()

logreg_misclass_avg <- misclass_rate_vals[[1]]
tree_misclass_avg <- misclass_rate_vals[[2]]

logreg_misclass_avg
tree_misclass_avg

# Based on the evaluation of the logistic regression and decision tree models using the adult dataset from the 1994 census, it appears that the decision tree model performs slightly better in predicting whether an individual's income exceeds $50,000. The average misclassification rate for the logistic regression model over 10 trials was around 35.05%, while the average misclassification rate for the decision tree model was around 34.74%.
# However, it's important to note that the difference in performance between the two models is relatively small. It's also possible that different variable selection, model tuning, or evaluation methods could yield different results
