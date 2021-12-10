library("data.table")
library("mltools")
library("ggplot2")
library("corrr")
library("dplyr")
library("tidyverse")
library("lattice")
library("caTools")
pacman::p_load(psych)
library("plotly")
library("corrplot")
library("Metrics")
library("DAAG")
library("caret")
library("GGally")
library("MASS")
library("olsrr")
library("caret")
library("e1071")
library("kernlab")

df_unclean = read.csv("merged_df2.csv")
head(df_unclean)
dim(df_unclean)
sapply(df_unclean,function (x) sum(is.na(x))) # No Null values
summary(df_unclean)
# The Mean of the values are around the same range except for feature 11
features_list_v1 = list('ENGINE_COOLANT_TEMP','ENGINE_LOAD','ENGINE_RPM','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED','THROTTLE_POS','TROUBLE_CODES','TROUBLE_CODES_converted')


#df$TROUBLE_CODES_converted[df$TROUBLE_CODES==""] <- 0
#df_14[c("TROUBLE_CODES_converted")][is.na(df_14[c("TROUBLE_CODES_converted")])] <- 1


table(df_unclean['TROUBLE_CODES'], useNA = "ifany")
table(df_unclean['TROUBLE_CODES_converted'], useNA = "ifany")

df_unclean['ENGINE_LOAD'] <- sapply(df_unclean['ENGINE_LOAD'],function(x) {x <- gsub("%","",x)})
df_unclean['ENGINE_LOAD'] <- sapply(df_unclean['ENGINE_LOAD'],function(x) {x <- gsub(",",".",x)})

df_unclean['THROTTLE_POS'] <- sapply(df_unclean['THROTTLE_POS'],function(x) {x <- gsub("%","",x)})
df_unclean['THROTTLE_POS'] <- sapply(df_unclean['THROTTLE_POS'],function(x) {x <- gsub(",",".",x)})


df_unclean['FUEL_LEVEL'] <- sapply(df_unclean['FUEL_LEVEL'],function(x) {x <- gsub(",",".",x)})

df_unclean <- as.data.frame(apply(df_unclean, 2, as.numeric))

table(df_unclean['FUEL_LEVEL'], useNA = "ifany")
sapply(df_unclean,function (x) sum(is.na(x))) # No Null values
###################################

new_df = df_unclean[,c('ENGINE_COOLANT_TEMP','ENGINE_LOAD','ENGINE_RPM','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED','THROTTLE_POS','TROUBLE_CODES_converted')]    
sapply(new_df,function (x) sum(is.na(x))) # No Null values
new_df<- na.omit(new_df)
sapply(new_df,function (x) sum(is.na(x))) # No Null values
new_df$TROUBLE_CODES_converted = as.factor(new_df$TROUBLE_CODES_converted)
str(new_df)

table(new_df['TROUBLE_CODES_converted'], useNA = "ifany")
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(TROUBLE_CODES_converted~., data = new_df, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
corr<-cor(new_df[1:7])
corrplot(corr)

###################################
df = df_unclean[,c('ENGINE_COOLANT_TEMP','ENGINE_LOAD','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED','TROUBLE_CODES_converted')] 
############################
#Handle missing values
df<- na.omit(df)
dim(df)
sapply(df,function (x) sum(is.na(x)))
###################
str(df)
corr<-cor(df)
corrplot(corr)
#throttle pos is collinear
table(df['TROUBLE_CODES_converted'], useNA = "ifany")


#ggcorrplot(corr)
index <- findCorrelation(abs(corr), 0.75,exact=FALSE)

#the name of the columns chosen above
#to_be_removed <- colnames(corr)[index]

#now go back to df and use to_be_removed to subset the original df
#datats=df[,c(to_be_removed):=NULL]
#data=df[!names(df) %in% to_be_removed]
##########################################
dim(df)
dim(data)
corrplot(cor(data))
table(data['TROUBLE_CODES_converted'], useNA = "ifany")
##############################################
#Split the data
set.seed(42)
training_index = createDataPartition(df$TROUBLE_CODES_converted, p=.8, list=FALSE)
training_data = df[training_index, ] 
testing_data = df[-training_index, ]

#Normalize the data
train_stats = preProcess(training_data[1:5], method = "range")
normalized_train_data = predict(train_stats, training_data)
normalized_test_data = predict(train_stats, testing_data)

normalized_train_data$TROUBLE_CODES_converted = as.factor(normalized_train_data$TROUBLE_CODES_converted)
normalized_test_data$TROUBLE_CODES_converted = as.factor(normalized_test_data$TROUBLE_CODES_converted)
###########################################


training_set = normalized_train_data[,c('ENGINE_COOLANT_TEMP','ENGINE_LOAD','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED','TROUBLE_CODES_converted')]
testing_set  = normalized_test_data[,c('ENGINE_COOLANT_TEMP','ENGINE_LOAD','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED','TROUBLE_CODES_converted')]

trctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions=TRUE)
############################
#Logistic Regression

logit_model <- glm(TROUBLE_CODES_converted ~.,family=binomial(link='logit'),data=training_set)

logit_pred = predict(logit_model, testing_set[,-6])
logit_pred <- ifelse(logit_pred > 0.5,1,0)
confusionMatrix(as.factor(logit_pred),testing_set$TROUBLE_CODES_converted)


#############################################################
#CART

forest_train = train(TROUBLE_CODES_converted ~ ., 
                     data=training_set, 
                     method="rpart")


pred_forest = predict(forest_train, testing_set[,-6])
confusionMatrix(pred_forest,testing_set$TROUBLE_CODES_converted)

#summary(forest_train$finalModel)
#library("rattle")
#fancyRpartPlot(forest_train$finalModel)
###################
#BAGGING
bagging_model = train(TROUBLE_CODES_converted ~ ., 
                      data=training_set, 
                      method="treebag")


pred_bagging = predict(bagging_model, testing_set[,-6])
confusionMatrix(pred_bagging,testing_set$TROUBLE_CODES_converted)
#library("rattle")
#fancyRpartPlot(bagging_model$finalModel)
##################################
#Random Forest

rf_model = train(TROUBLE_CODES_converted ~ ., 
                 data=training_set, trControl = trctrl,
                 method="rf",prox = TRUE)


pred_rf = predict(rf_model, testing_set[,-6])
confusionMatrix(pred_rf,testing_set$TROUBLE_CODES_converted)


######################
#Boosting
boosting_model = train(TROUBLE_CODES_converted ~ ., 
                       data=training_set, trControl = trctrl,
                       method="gbm",verbose = FALSE)


pred_boosting = predict(boosting_model, testing_set[,-6])
confusionMatrix(pred_boosting,testing_set$TROUBLE_CODES_converted)


########################################
#Naive Bayes
bayes_model = train(TROUBLE_CODES_converted ~ ., 
                    data=training_set, 
                    method="nb")


pred_bayes = predict(bayes_model, testing_set[,-6])
confusionMatrix(pred_bayes,testing_set$TROUBLE_CODES_converted)

#########################################
model_knn = train(TROUBLE_CODES_converted ~ ., 
                      data=training_set, 
                      method="knn")
pred_knn = predict(model_knn, testing_set[,-6])
confusionMatrix(pred_knn,testing_set$TROUBLE_CODES_converted)
##################################################

install.packages('randomForest')
library(Boruta)
library(mlbench)
library(randomForest)
boruta <- Boruta(x = df[,1:7],y = df[,8], doTrace = 1, maxRuns = 500)
print(boruta)
plot(boruta, las = 2, cex.axis = 0.7)



library(ROCR)
pr <- prediction(pred_boosting,testing_set$TROUBLE_CODES_converted)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
