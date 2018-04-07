main_data[is.na(main_data)]<-0
table(main_data$target_var)#   0       1 #1472441   11785 
library(caTools)
library(caret)
library(xgboost)
library(stringr)
library(e1071)
library(randomForest)
library(mlbench)
library(dplyr)
library(plyr)
set.seed(123876)
split = sample.split(main_data$target_var, SplitRatio = 0.7)
train_new = subset(main_data, split == TRUE)
test_new = subset(main_data, split == FALSE) 
table(train_new$target_var)/nrow(train_new)
table(test_new$target_var)/nrow(test_new)
train_new$target_var <- as.factor(train_new$target_var)
test_new$target_var <- as.factor(test_new$target_var)
class(train_new$target_var)
drops <- c("cust_num","weeknum")
train2<- train_new[,!names(train_new) %in% drops]
test2 <- test_new[,!names(test_new) %in% drops]
df_train = train2 %>%  na.omit() %>%                                                                # listwise deletion   mutate(target_var = factor(target_var,labels = c("Failure", "Success")))
  #..................................... Grid Search of Xgboost ....................................................................
  xgb_grid_1 = expand.grid(  nrounds = c(500),  max_depth = c( 4, 6, 8),  eta = c(0.1, 0.01),  gamma = c(1, 2),  colsample_bytree = c(0.5, 0.8),  subsample=0.7,  min_child_weight=1)
# pack the training control parametersxgb_trcontrol_1 = trainControl(  method = "repeatedcv",  number = 5,  repeats = 2,  verboseIter = TRUE,  returnData = FALSE,  returnResamp = "all",                                                        # save losses across all models  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed  summaryFunction = twoClassSummary,  allowParallel = TRUE)
# train the model for each parameter combination in the grid, #   using CV to evaluatexgb_train_1 = train(  x = as.matrix(df_train %>%                  select(-target_var)),  y = as.factor(df_train$target_var),  trControl = xgb_trcontrol_1,  tuneGrid = xgb_grid_1,  objective = 'binary:logistic',  eval_metric = 'auc',  method = "xgbTree")
y<-as.matrix(train2$target_var)
ytest<-as.matrix(test2$target_var)
dropevent<- names(train2) %in% c('target_var') 
x <- train2[!dropevent]
xtest <- test2[!dropevent]
x1<-as.matrix(x)
x1test<-as.matrix(xtest)
dtrain<- xgb.DMatrix(data.matrix(x),label=y)
param <- list("objective" = "binary:logistic",               "eval_metric" = "error",                "nthread" = 10,               "max_depth" =4 ,               "eta" = 0.1,              "gamma"=2,              "subsample"=0.7,              "min_child_weight" = 1,              "colsample_bytree"=1) 
set.seed(1234)#k-fold cross validation
#nround.cv=200
bst.cv <-xgb.cv(param=param,data=dtrain,nfold=10,nround=nround.cv,prediction = T,verbose = T)#index of maximum AUC
min.error.idx <-which.min(bst.cv$evaluation_log[,test_error_mean])
bst <- xgboost(param=param, data=x1, label=y,nrounds=min.error.idx, verbose=1,prediction=T)
names<- dimnames(x1)[[2]]
importance=xgb.importance(names,model=bst)
gp=xgb.plot.importance(importance,10)
write.csv(importance, file = "importance.csv")
pred <- predict(bst, x1) 
prediction_train <- as.factor(as.numeric(pred >0.01))
pred_datf_train<-data.frame(prediction_train)
pred_test <- predict(bst, x1test)
prediction_test <- as.factor(as.numeric(pred_test >0.01))
pred_datf_test<-data.frame(prediction_test)
output_train<-cbind(train2,pred_datf_train)
output_test<-cbind(test2,pred_datf_test)
#confusion matrix
confusionMatrix(data=output_train$prediction_train, 
                reference=output_train$target_var,
                positive='1')
#Reference#Prediction      0      1#         0 883286   1233#         1 147423   7017
#Accuracy : 0.8569  #Sensitivity : 0.850545        #Specificity : 0.856969 
confusionMatrix(data=output_test$prediction_test, 
                reference=output_test$target_var, 
                positive='1')
#Confusion Matrix and Statistics
#              Reference#Prediction      0      1#          0 377876    924#          1  63856   2611
#Accuracy : 0.8545 #Sensitivity : 0.738614        #Specificity : 0.855442 
pred_train1<-data.frame(pred)
train_loan <- cbind(train2,pred_train1)
AUC <- function(pred,depvar){require(ROCR)  p <- prediction(as.numeric(pred),depvar)  auc<- performance(p,"auc")  auc <- unlist(slot(auc,"y.values"))  return (auc)}
KS <- function(pred,depvar){require(ROCR)  p <- prediction(as.numeric(pred),depvar)  perf <- performance(p,"tpr","fpr")  ks <- max(attr(perf,"y.values")[[1]]-(attr(perf,"x.values")[[1]]))  return(ks)}
KS(train_loan$pred,train_loan$target_var)#0.7084075
AUC(train_loan$pred,train_loan$target_var)#0.9320322
pred_test1<-data.frame(pred_test)
test_loan <- cbind(test2,pred_test1)
KS(test_loan$pred_test,test_loan$target_var)#0.60
AUC(test_loan$pred_test,test_loan$target_var)#0.885135
lift <- function(depvar, predcol, groups=20) {  if(!require(dplyr)){    install.packages("dplyr")    library(dplyr)}  if(is.factor(depvar)) depvar <- as.integer(as.character(depvar))  if(is.factor(predcol)) predcol <- as.integer(as.character(predcol))  helper = data.frame(cbind(depvar, predcol))  helper[,"bucket"] = ntile(helper[,"predcol"], groups)  gaintable = helper %>% group_by(bucket)  %>%    summarise_at(vars(depvar), funs(total = n(),                                    totalresp=sum(., na.rm = TRUE))) %>%    mutate(Cumresp = cumsum(totalresp),           Gain=Cumresp/sum(totalresp)*100,           Cumlift=Gain/(bucket*(100/groups)))  return(gaintable)}
train_loan <- train_loan[order(train_loan$pred),]
train_10 = lift(train_loan$target_var, train_loan$pred, groups = 10)