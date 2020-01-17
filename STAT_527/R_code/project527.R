# project
setwd('C:/Users/46541/Desktop/myhw/527/project')
# package
library(rpart)
library(rpart.plot)

# Problem 2
spam_all = read.csv('spam.txt', header = FALSE, sep = ' ')
indicator = read.csv('spam_traintest.txt', header = FALSE, sep = ' ')
# divide data by indicator
spam_train <- spam_all[which(indicator==0),]
spam_test <- spam_all[which(indicator==1),]

#a
spam_train$V58<- factor(spam_train$V58)   #,levels = 0:1,labels = c("Non","S"))
dtree<-rpart(V58~.,data=spam_train,method="class", parms=list(split="gini"),cp=0)
printcp(dtree)
rpart.plot::rpart.plot(dtree)


spam_fitted <- predict(dtree,newdata=spam_train,type = 'class')
err_train <- 1- sum(spam_fitted == spam_train$V58)/length(spam_fitted)
paste("The error rate on the training set is:",err_train)

spam_test_val <-predict(dtree, newdata = spam_test,type='class')
err_test <- 1- sum(spam_test_val == spam_test$V58)/length(spam_test_val)
paste("The error rate on the test set is:",err_test)



#b
cp <- which.min(dtree$cptable[,"xstd"])
paste("The optimal tunning parameter is:",dtree$cptable[,"CP"][cp])
tree_pruned<-prune(dtree,cp=dtree$cptable[which.min(dtree$cptable[,"xstd"]),"CP"])
printcp(tree_pruned)
rpart.plot::rpart.plot(tree_pruned)

pruned_fitted <- predict(tree_pruned, newdata = spam_train,type='class')
err_train_pruned <- 1- sum(pruned_fitted== spam_train$V58)/length(pruned_fitted)

paste("The error rate on the training set is:",err_train_pruned)

pruned_test_val <-predict(tree_pruned, newdata = spam_test,type='class')
err_test_pruned <- 1- sum(pruned_test_val == spam_test$V58)/length(pruned_test_val)
paste("The error rate on the test set is:",err_test_pruned)



# Problem 3
library(tidyverse)
library(broom)
library(glmnet)
# reload the data

spam_all = read.csv('spam.txt', header = FALSE, sep = ' ')
indicator = read.csv('spam_traintest.txt', header = FALSE, sep = ' ')
spam_train <- spam_all[which(indicator==0),]
spam_test <- spam_all[which(indicator==1),]
trainx <- spam_train[,-58]
trainx <- as.matrix(trainx)
trainy <- spam_train[,58]
trainy <- as.matrix(trainy)

lambdas = seq(0, 1, by = 0.001)
set.seed(123)
cv_fit = cv.glmnet(trainx, trainy, alpha = 0, standardize =TRUE,lambda = lambdas,nfolds = 10)
plot(cv_fit)
opt_lambda = cv_fit$lambda.min
paste("The opyimal lambda is:",opt_lambda)
#coef
fit_ridge = glmnet(trainx, trainy, alpha = 0, standardize =TRUE,lambda =opt_lambda)
coef(fit_ridge)


# pick optimal c
spam_train <- spam_all[which(indicator==0),]
train_y_val <- spam_train[,58]
train_y_val <- as.matrix(train_y_val)

spam_test <- spam_all[which(indicator==1),]
testx <- spam_test[,-58]
testx <- as.matrix(testx)
testy <-spam_test[,58]
testy <- as.matrix(testy)

fitted_ridge <- predict(fit_ridge, newx = trainx )
c <- seq(-1,1,by=0.01)

pick_c <- function (c,fit_val, true_class){
    class1 <- which(fit_val > c)
    fit_val[class1] <- 1
    fit_val[-class1] <- 0
    err =1- sum(fit_val == true_class)/length(fit_val)
    return (err)
}

opt_c <-function(c){
  pick_c(c, fitted_ridge,train_y_val )
}

c_all <- sapply(c, opt_c)
plot(c,c_all,xlab="c", ylab = "error rate",main="Ridge regression classification with different c" )
c_opt <- which(min(c_all)==c_all)
paste("The optimal c is:", c[c_opt])


# resubstitution error
paste("The resubstitution error for ridge regression is:", min(c_all))
# prediction error
predict_ridge <- predict(fit_ridge, newx = testx )
predict_ridge[predict_ridge>c[c_opt]] <- 1
predict_ridge[predict_ridge<=c[c_opt]] <- 0
predict_err <- 1-(sum(predict_ridge == testy))/length(predict_ridge)
paste("The prediction error for ridge regression is:",predict_err )


# compared with linear regression

fit_ols <- lm(V58~.,data = spam_train)
fitted_ols <- predict(fit_ols, newdata = spam_train) 

opt_c_ols <-function(c){
  pick_c(c, fitted_ols,spam_train$V58 )
}
c_ols <- seq(0,1,by=0.01)

c_ols_all <-sapply(c_ols,opt_c_ols)
plot(c_ols,c_ols_all,xlab="c",ylab="error rate",main="linear regression classification with different c")
c_ols_opt <- c_ols[which(min(c_ols_all) == c_ols_all)]
paste("The optimal c for linear regression ",c_ols_opt)

# resubstitution error
paste("The resubstitution error for linear regression is:",min(c_ols_all))

predict_ols <- predict(fit_ols, newdata = spam_test )
predict_ols[predict_ols>c_ols_opt] <- 1
predict_ols[predict_ols<=c_ols_opt] <- 0
predict_err_ols <- 1 - sum(predict_ols == spam_test$V58)/length(predict_ols)
paste("The prediction error for linear regression is:",predict_err_ols )




# Problem 4
library(MASS)
library(gam)
library( ISLR)
library(leaps)
data("College")
dt <- College
set.seed(123)
id_sample <- sample(1:777,500,replace=FALSE)
training <- dt[id_sample,]
training$Private<-ifelse(training$Private=="Yes",1,0)
testing <- dt[-id_sample,]
testing$Private<-ifelse(testing$Private=="Yes",1,0)

# 5-fold
fold1<-training[-seq(1,100,1),]
out1 <-training[seq(1,100,1),]
train2<-training[-seq(101,200,1),]
out2 <-training[seq(101,200,1),]
train3<-training[-seq(201,300,1),]
out3 <-training[seq(201,300,1),]
train4<-training[-seq(301,400,1),]
out4 <-training[seq(301,400,1),]
train5<-training[-seq(401,500,1),]
out5 <-training[seq(401,500,1),]


null_model = lm(Outstate~1, data = training)
full_model = lm(Outstate~., data = training)



fit_forward = stepAIC(null_model, scope=list(upper=full_model,lower=null_model),
                      direction = "forward", trace = FALSE)



forward_fit <- regsubsets(Outstate ~ ., data = training, nvmax = 17, method = "forward")
forward_fit_summary <-summary(forward_fit)


# BIC
plot(1:17,forward_fit_summary$bic, xlab = "The number of predictors",ylab ="BIC")
which(min(forward_fit_summary$bic)==forward_fit_summary$bic)

# cp
plot(1:17,forward_fit_summary$cp, xlab = "The number of predictors",ylab ="Cp")
which(min(forward_fit_summary$cp)==forward_fit_summary$cp)

# AIC
null_model = lm(Outstate~1, data = training)
full_model = lm(Outstate~., data = training)
fit_forward = stepAIC(null_model, scope=list(upper=full_model,lower=null_model),direction = "forward")#, trace = FALSE)
# predictors I chose
names(coef(fit_forward ))[-1]






predict_val <- predict(fit_forward, newdata = testing)
err_forward <- mean((testing$Outstate-predict_val)^2)


# b

library(gam)
library(mgcv)

gam_fit <- gam(Outstate ~ Private+s(Expend)+s(Terminal)+
                 s(Top10perc)+s(Accept)+s(Apps)+s(Grad.Rate)
               +s(perc.alumni) +s(F.Undergrad) +s(Room.Board), data =training)
par(mfrow = c(3, 3))
plot(gam_fit,se = T)


# c
gam_pred_val <- predict(gam_fit, testing)
gam.err <- mean((testing$Outstate - gam_pred_val)^2)
gam.err

paste("The predicted RSS by forward stepwise is:",err_forward)
paste("The predicted RSS by GAM is:",gam.err)

# explanation: 






# Problem 6
train_zip = read.csv('zip-train.txt', header = FALSE, sep = ' ')
train_zip_x <- train_zip[,-1]
train_zip_x$V258 <- c()
train_zip_y <- factor(train_zip[,1])

test_zip = read.csv('zip-test.txt', header = FALSE, sep = ' ')
test_zip_x <- test_zip[,-1]
test_zip_x$V258 <- c()
test_zip_y <- factor(test_zip[,1])
# a
library(caret)
library(class)

knn.classifier<- function(X.train, y.train, X.test, k.try=1 , pi=rep(1/K,K),CV=F){

  if (CV==FALSE){
    pred_class = sapply(k.try, function(ne){knn(train = X.train, test = X.test, y.train ,k=ne)} )
    
  }
  else {
    pred_class = sapply(k.try, function(ne){knn.cv(train = X.train,  y.train ,k=ne)} )
  }
  
  return (pred_class)
}

data("iris")
iris$Species <- as.integer(iris$Species)
X.train_iris <- iris[,-5]
y.train_iris <- iris[,5]


# use cv
res1 <- knn.classifier(X.train_iris,y.train_iris,X.train_iris,k.try = 5,pi=rep(1/3,3),CV = T)
paste("The number of classification with 'CV=T' is:",sum(res1 != y.train_iris ))
# not use cv
res2 <- knn.classifier(X.train_iris,y.train_iris,X.train_iris,k.try = 5,pi=rep(1/3,3),CV =F)
paste("The number of classification with 'CV=F' is:",sum(res2 != y.train_iris ))


# (c)

res3 <- knn.classifier(train_zip_x,train_zip_y,test_zip_x,
               k.try = c(1, 3, 7, 11, 15, 21, 27, 35, 43),pi=rep(1/10,10),CV = T)

err <- rep(0,9)
for (i in 1:9){
  err[i] <- 1- sum(res3[,i] == train_zip_y )/length(train_zip_y )

}

plot(c(1, 3, 7, 11, 15, 21, 27, 35, 43),err, xlab = "k",ylab='error rate',ylim=c(0,0.15))

paste("The optimal k is:",which(min(err)==err))
paste("The corresponding error rate is:", min(err))


# (d)
pred_knn <-  knn(train_zip_x,test_zip_x , train_zip_y ,k=1)
err_predict <- 1- sum(pred_knn == test_zip_y)/length(test_zip_y)
paste("The predict error rate for test data is:",err_predict)

