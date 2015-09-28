# Introduction : What is the question?
# Data from a sample of 6 people using accelerometers on the belt, forearm, arm, and dumbell are used to understand how well
# they are performing compared to a 'witness' group of professionnals store in A group
# From collected observations, the aim is to buld a predictive model that will assess how well the exercise is done and rank
# in the right group, from A to E
#

## LOAD DATA and Libraries

# load relevant libraries
library(caret)
library(ISLR); library(ggplot2);

# Create 1 datasets reading the CSV file
# We will Use this dataset both for training(70%) and testing(30%) purpose.
# 
inDatasetHAR<-read.csv(file="pml-training.csv",head=TRUE,sep=",")
inDatasetValidHAR<-read.csv(file="pml-testing.csv",head=TRUE,sep=",")
#
#This is the data file that Velloso, E.; Bulling, A.; Gellersen, H.; 
#Ugulino, W.; Fuks, H. use in thei paper "Qualitative Activity Recognition of 
# Weight Lifting Exercises. Proceedings of 4th International Conference in 
#Cooperation with SIGCHI (Augmented Human '13) . 
##Stuttgart, Germany: ACM SIGCHI, 2013.
#
##Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3mvTxcXYG
#inDatasetRefHAR<-read.csv(file="dataset-har.csv",head=TRUE,sep=";")
#
#
#

inTrainTest <- createDataPartition(y=inDatasetHAR$classe,p=0.7, list=FALSE)
#70% of data for training and tests
#Now Split in training and testing dataset (70% 30%)
trainingHAR<-inDatasetHAR[inTrainTest,]
testingHAR<-inDatasetHAR[-inTrainTest,]

##
## DATA ANALYSIS
##
#
# First 7 columns of the datasets can be skipped as they are not relevant for the analysis (name, window, ID...)
#
d<-dim(trainingHAR)
trainingHAR<-trainingHAR[,8:d[2]]
testingHAR<-testingHAR[,8:d[2]]
#
# Analysis of training dataset : a lot of variable with "MISSING values", correlated and without variance
# we can consider these as "poor" predictors
#

#
#
# Eliminates zero Variance predictors..
nsv<-nearZeroVar(trainingHAR)
trainingHAR<- trainingHAR[, -nsv]
testingHAR<-testingHAR[,-nsv]
##
## Then eliminates predictors with imprtant percentage of missing values (>90%)
##
##
na_count <-sapply(trainingHAR, function(y) sum(length(which(is.na(y)))))
na_perc=na_count/dim(trainingHAR)
reducedTrainingHAR<-trainingHAR[, na_perc <= 0.9]
reducedTestingHAR<-testingHAR[, na_perc <= 0.9]
#
# So 53 selected Predictors of 160 that we will use for building the prediction model  
#





#
# The Algorythm
#
#Random Forest used with cross validation seems to be a good approach, because the class of alghorithm fit
# well this kind of classification problem, bootstrap and cross validation are easy to use.
# We can estimate how
# the number of predictors will influence the error of the model using the rfcv function in caret
# set seed value and parallel options
set.seed(125)
doMC::registerDoMC(cores=4)
library(MASS)
library(randomForest)
library(class)

##FitModel0 <- rfcv(reducedTrainingHAR[,1:(dim(reducedTrainingHAR)[2]-1)],reducedTrainingHAR[,dim(reducedTrainingHAR)[2]],cv.fold=5,mtry=function(p) max(1, floor(sqrt(p))), recursive=TRUE)
#
# So the model will work fine with 26 out of the 52 predictors. the varimp function will 
# tell us which are these predictors
# Alternatively, we can use this
#FitModel <- randomForest(reducedTrainingHAR[,1:(dim(reducedTrainingHAR)[2]-1)],reducedTrainingHAR[,dim(reducedTrainingHAR)[2]],cv.fold=5, importance=TRUE)
FitModel2 <- train(classe ~ ., method="rf",data=reducedTrainingHAR,trControl=trainControl(method="cv",number=5),prox=TRUE,importance=TRUE,allowParallel=TRUE)
plot(FitModel2)
plot(FitModel)
plot(FitModel$importance)
varImpPlot(FitModel2)



resu<-confusionMatrix(reducedTestingHAR$classe, predict(FitModel2, reducedTestingHAR))
resu
z<-as.table(resu)
colnames(z) = c("A","B","C","D","E")
rownames(z)=colnames(z)
image(z[,ncol(z):1], axes=FALSE)
heatmap(t(z)[ncol(z):1,], Rowv=NA,Colv=NA, col = heat.colors(256))
outdata20<-inDatasetValidHAR[,8:d[2]]
outdata20<-outdata20[,-nsv]
outdata20<-outdata20[, na_perc <= 0.9]
predict20=predict(FitModel,outdata20)
predict20
