# Analytics Vidhya - McKinsey Analytics Online Hackathon 2018
# Here is the link for Hackathon - https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon/
# Dataset has several health, demographic and lifestyle details about patients. Hackathon is to predict the probability of stroke to patients.
# 
# https://satya-are.blogspot.com/
##############################################################

# Cleaning up environment
setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Satya/Data Science/ML/Analytics Vidhya/McKinsey")
rm(list=ls())

# Loading training data
train <- read.csv("train_ajEneEa.csv")
test <- read.csv("test_v2akXPA.csv")

##############################################################
# Data Understanding
##############################################################

str(train)

ifelse(length(unique(train$id))==nrow(train), "id is unique column", "id is NOT unique column")
# id is unique column

cat("There are", length(unique(train$id)), "unique IDs")

sapply(train, function(x) table(x))

sapply(train, function(x) sum(is.na(x)))
# Only bmi has NULL values

ifelse(sum(duplicated(train))==0, "There are NO duplicate records", "There are duplicate records")
# There are NO duplicate records


# Feature standardisation
# Normalising continuous features 
train$age <- scale(train$age)
train$avg_glucose_level <- scale(train$avg_glucose_level)
train$bmi <- scale(train$bmi)

# Checking factors
train_fact <- data.frame(sapply(train[, c(2,6:8,11)], function(x) factor(x)))
str(train_fact)

# Creating dummy variables for factor attributes
dummies <- data.frame(sapply(train_fact, function(x) data.frame(model.matrix(~x-1, data=train_fact))[,-1]))
str(dummies)

# Final dataset
train_final <- cbind(train[, c(3,4,5,9,10,12)], dummies) 
View(train_final)

str(train_final)

##############################################################
# Logistic Regression
##############################################################
# Initial model
model_1 = glm(stroke ~ ., data = train_final, family = "binomial")
summary(model_1)

# Stepwise selection
library("MASS")
model_2 <- stepAIC(model_1, direction="both")
summary(model_2)

# Removing multicollinearity through VIF check
library(car)
vif(model_2)


model_3 <- glm(formula = stroke ~ age + hypertension + heart_disease + avg_glucose_level + 
                bmi + gender.xMale + smoking_status.xformerly.smoked + smoking_status.xnever.smoked + 
                smoking_status.xsmokes, family = "binomial", data = train_final) 

summary(model_3)

sort(vif(model_3))

model_4 <- glm(formula = stroke ~ age + hypertension + heart_disease + avg_glucose_level + 
                 smoking_status.xformerly.smoked + smoking_status.xnever.smoked + 
                smoking_status.xsmokes, family = "binomial", data = train_final) 

summary(model_4)

model_5 <- glm(formula = stroke ~ age + hypertension + heart_disease + avg_glucose_level + 
                 smoking_status.xsmokes, family = "binomial", data = train_final) 

summary(model_5)

model_6 <- glm(formula = stroke ~ age + hypertension + heart_disease + avg_glucose_level, family = "binomial", data = train_final) 

summary(model_6)

final_model <- model_6

# Predicted probabilities of stroke
str(test)
test$age <- scale(test$age)
test$avg_glucose_level <- scale(test$avg_glucose_level)

test_pred = predict(final_model, type = "response", newdata = test)


# See the summary 

summary(test_pred)

test$prob <- test_pred
View(test)
summary(test$prob)
sort(test$prob)

# Use the probability cutoff of 30%.
test$stroke <- factor(ifelse(test_pred > 0.3, "1", "0"))

test$stroke
table(test$stroke)


# Creating output file for submission

# submission_data <- read.csv("sample_submission.csv")
# str(submission_data)

submission <- test[, c("id", "stroke")]
table(submission$stroke)

write.csv(submission, "sample_submission.csv", row.names = FALSE)

