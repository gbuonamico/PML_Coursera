---
title: "Assignment Practical Machine Learning Course"
output: html_document
---

Assignment for the Practical Machine Learning Course
Published on 27/09/2015 By Lupo Argentato (Facebook Pseudo) alias Gianluca BUONAMICO


STEP 1: What is the question?

Data from a sample of 6 people using accelerometers on the belt, forearm, arm, and dumbell are used to understand how well they are performing compared to a 'witness' group of professionnals store in A group

From collected observations, the aim is to buld a predictive model that will assess how well the exercise is done and rank in the right group, from A to E
#


STEP 2 : LOAD AND ANALYSE DATA

```r
## 

# load relevant libraries
library(caret)
library(ISLR); library(ggplot2);
# Create 1 datasets reading the CSV file
# We will Use this dataset both for training(70%) and testing(30%) purpose.
# 
inDatasetHAR<-read.csv(file="pml-training.csv",head=TRUE,sep=",")
inDatasetValidHAR<-read.csv(file="pml-testing.csv",head=TRUE,sep=",")
#
```
This is the data file that Velloso, E.; Bulling, A.; Gellersen, H.; 
Ugulino, W.; Fuks, H. use in thei paper "Qualitative Activity Recognition of 
Weight Lifting Exercises. Proceedings of 4th International Conference in 
Cooperation with SIGCHI (Augmented Human '13) . 
Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3mvTxcXYG

Now Load data and store in two different datasets, for training and testing purpose

```r
inTrainTest <- createDataPartition(y=inDatasetHAR$classe,p=0.7, list=FALSE)
#70% of data for training and tests
#Now Split in training and testing dataset (70% 30%)
trainingHAR<-inDatasetHAR[inTrainTest,]
testingHAR<-inDatasetHAR[-inTrainTest,]
```
So we choose to have a training SET for 70% of data and 30% for testing. The dataset contains 19622 observations and 160 variables.
First 7 variables are probably not relevant (name, ID line, Window...). So we throw them out



```r
d<-dim(trainingHAR)
trainingHAR<-trainingHAR[,8:d[2]]
testingHAR<-testingHAR[,8:d[2]]
```

Analysis of training dataset : a lot of variable with "MISSING values", and variables with zero variance
we can consider these as "poor" predictors

```r
# Eliminates zero Variance predictors..
nearZeroVar(trainingHAR,saveMetrics=TRUE)
```

```
##                            freqRatio percentUnique zeroVar   nzv
## roll_belt                   1.143791    8.09492611   FALSE FALSE
## pitch_belt                  1.030534   12.25158332   FALSE FALSE
## yaw_belt                    1.120787   13.05962000   FALSE FALSE
## total_accel_belt            1.062917    0.21110868   FALSE FALSE
## kurtosis_roll_belt       3366.500000    1.95093543   FALSE  TRUE
## kurtosis_picth_belt       792.117647    1.63791221   FALSE  TRUE
## kurtosis_yaw_belt          49.690037    0.01455922   FALSE  TRUE
## skewness_roll_belt       3366.500000    1.92909660   FALSE  TRUE
## skewness_roll_belt.1      792.117647    1.74710636   FALSE  TRUE
## skewness_yaw_belt          49.690037    0.01455922   FALSE  TRUE
## max_roll_belt               1.000000    1.06282303   FALSE FALSE
## max_picth_belt              1.666667    0.14559220   FALSE FALSE
## max_yaw_belt              673.300000    0.42949698   FALSE  TRUE
## min_roll_belt               1.222222    1.00458615   FALSE FALSE
## min_pitch_belt              2.228571    0.10919415   FALSE FALSE
## min_yaw_belt              673.300000    0.42949698   FALSE  TRUE
## amplitude_roll_belt         1.041667    0.80075708   FALSE FALSE
## amplitude_pitch_belt        3.266667    0.09463493   FALSE FALSE
## amplitude_yaw_belt         51.992278    0.02911844   FALSE  TRUE
## var_total_accel_belt        1.263158    0.40037854   FALSE FALSE
## avg_roll_belt               1.000000    1.01186576   FALSE FALSE
## stddev_roll_belt            1.000000    0.40037854   FALSE FALSE
## var_roll_belt               1.705882    0.51685230   FALSE FALSE
## avg_pitch_belt              1.333333    1.18657640   FALSE FALSE
## stddev_pitch_belt           1.175000    0.28390478   FALSE FALSE
## var_pitch_belt              1.333333    0.37126010   FALSE FALSE
## avg_yaw_belt                1.125000    1.32488899   FALSE FALSE
## stddev_yaw_belt             1.750000    0.34942127   FALSE FALSE
## var_yaw_belt                1.433333    0.79347747   FALSE FALSE
## gyros_belt_x                1.070760    0.93179006   FALSE FALSE
## gyros_belt_y                1.155655    0.47317464   FALSE FALSE
## gyros_belt_z                1.022763    1.19385601   FALSE FALSE
## accel_belt_x                1.059590    1.17201718   FALSE FALSE
## accel_belt_y                1.137089    0.99730654   FALSE FALSE
## accel_belt_z                1.057971    2.09652763   FALSE FALSE
## magnet_belt_x               1.051383    2.18388294   FALSE FALSE
## magnet_belt_y               1.030369    2.07468880   FALSE FALSE
## magnet_belt_z               1.084112    3.18118949   FALSE FALSE
## roll_arm                   52.652174   17.54385965   FALSE FALSE
## pitch_arm                  80.733333   20.13540074   FALSE FALSE
## yaw_arm                    30.658228   19.23272913   FALSE FALSE
## total_accel_arm             1.018957    0.48045425   FALSE FALSE
## var_accel_arm               5.000000    1.94365582   FALSE FALSE
## avg_roll_arm               52.000000    1.60151416   FALSE  TRUE
## stddev_roll_arm            52.000000    1.60151416   FALSE  TRUE
## var_roll_arm               52.000000    1.60151416   FALSE  TRUE
## avg_pitch_arm              52.000000    1.60151416   FALSE  TRUE
## stddev_pitch_arm           52.000000    1.60151416   FALSE  TRUE
## var_pitch_arm              52.000000    1.60151416   FALSE  TRUE
## avg_yaw_arm                52.000000    1.60151416   FALSE  TRUE
## stddev_yaw_arm             54.000000    1.58695494   FALSE  TRUE
## var_yaw_arm                54.000000    1.58695494   FALSE  TRUE
## gyros_arm_x                 1.100592    4.60071340   FALSE FALSE
## gyros_arm_y                 1.414634    2.67161680   FALSE FALSE
## gyros_arm_z                 1.112022    1.68886948   FALSE FALSE
## accel_arm_x                 1.070312    5.53250346   FALSE FALSE
## accel_arm_y                 1.255172    3.80723593   FALSE FALSE
## accel_arm_z                 1.232558    5.52522385   FALSE FALSE
## magnet_arm_x                1.140351    9.61636456   FALSE FALSE
## magnet_arm_y                1.015625    6.20950717   FALSE FALSE
## magnet_arm_z                1.080000    9.11407149   FALSE FALSE
## kurtosis_roll_arm         258.961538    1.60879377   FALSE  TRUE
## kurtosis_picth_arm        249.370370    1.59423455   FALSE  TRUE
## kurtosis_yaw_arm         2693.200000    1.94365582   FALSE  TRUE
## skewness_roll_arm         258.961538    1.60879377   FALSE  TRUE
## skewness_pitch_arm        249.370370    1.59423455   FALSE  TRUE
## skewness_yaw_arm         2693.200000    1.94365582   FALSE  TRUE
## max_roll_arm               17.333333    1.50687923   FALSE FALSE
## max_picth_arm              10.400000    1.38312586   FALSE FALSE
## max_yaw_arm                 1.066667    0.36398049   FALSE FALSE
## min_roll_arm               13.000000    1.44864235   FALSE FALSE
## min_pitch_arm              17.333333    1.42680352   FALSE FALSE
## min_yaw_arm                 1.166667    0.27662517   FALSE FALSE
## amplitude_roll_arm         26.000000    1.52143845   FALSE  TRUE
## amplitude_pitch_arm        18.000000    1.49232001   FALSE FALSE
## amplitude_yaw_arm           1.000000    0.36398049   FALSE FALSE
## roll_dumbbell               1.065934   86.72199170   FALSE FALSE
## pitch_dumbbell              2.175258   84.50171071   FALSE FALSE
## yaw_dumbbell                1.227848   86.08866565   FALSE FALSE
## kurtosis_roll_dumbbell   4488.666667    1.94365582   FALSE  TRUE
## kurtosis_picth_dumbbell  6733.000000    1.96549465   FALSE  TRUE
## kurtosis_yaw_dumbbell      49.690037    0.01455922   FALSE  TRUE
## skewness_roll_dumbbell   4488.666667    1.94365582   FALSE  TRUE
## skewness_pitch_dumbbell  6733.000000    1.96549465   FALSE  TRUE
## skewness_yaw_dumbbell      49.690037    0.01455922   FALSE  TRUE
## max_roll_dumbbell           1.333333    1.73982675   FALSE FALSE
## max_picth_dumbbell          1.333333    1.72526753   FALSE FALSE
## max_yaw_dumbbell          897.733333    0.42949698   FALSE  TRUE
## min_roll_dumbbell           1.333333    1.71070831   FALSE FALSE
## min_pitch_dumbbell          1.333333    1.75438596   FALSE FALSE
## min_yaw_dumbbell          897.733333    0.42949698   FALSE  TRUE
## amplitude_roll_dumbbell     6.000000    1.86358011   FALSE FALSE
## amplitude_pitch_dumbbell    6.000000    1.85630050   FALSE FALSE
## amplitude_yaw_dumbbell     50.246269    0.02183883   FALSE  TRUE
## total_accel_dumbbell        1.095643    0.30574361   FALSE FALSE
## var_accel_dumbbell         13.000000    1.88541894   FALSE FALSE
## avg_roll_dumbbell           1.000000    1.93637621   FALSE FALSE
## stddev_roll_dumbbell       12.000000    1.89269855   FALSE FALSE
## var_roll_dumbbell          12.000000    1.89269855   FALSE FALSE
## avg_pitch_dumbbell          1.000000    1.93637621   FALSE FALSE
## stddev_pitch_dumbbell      12.000000    1.89269855   FALSE FALSE
## var_pitch_dumbbell         12.000000    1.89269855   FALSE FALSE
## avg_yaw_dumbbell            1.000000    1.93637621   FALSE FALSE
## stddev_yaw_dumbbell        12.000000    1.89269855   FALSE FALSE
## var_yaw_dumbbell           12.000000    1.89269855   FALSE FALSE
## gyros_dumbbell_x            1.036866    1.69614909   FALSE FALSE
## gyros_dumbbell_y            1.288509    1.98005387   FALSE FALSE
## gyros_dumbbell_z            1.021226    1.43408313   FALSE FALSE
## accel_dumbbell_x            1.058036    2.98464002   FALSE FALSE
## accel_dumbbell_y            1.047904    3.29038364   FALSE FALSE
## accel_dumbbell_z            1.103448    2.91184393   FALSE FALSE
## magnet_dumbbell_x           1.085470    7.84741938   FALSE FALSE
## magnet_dumbbell_y           1.089552    6.02023732   FALSE FALSE
## magnet_dumbbell_z           1.072581    4.81910170   FALSE FALSE
## roll_forearm               10.802372   13.47455776   FALSE FALSE
## pitch_forearm              62.090909   18.96338356   FALSE FALSE
## yaw_forearm                15.429379   13.03778117   FALSE FALSE
## kurtosis_roll_forearm     240.464286    1.57239572   FALSE  TRUE
## kurtosis_picth_forearm    236.245614    1.57239572   FALSE  TRUE
## kurtosis_yaw_forearm       49.690037    0.01455922   FALSE  TRUE
## skewness_roll_forearm     240.464286    1.57967533   FALSE  TRUE
## skewness_pitch_forearm    236.245614    1.55055689   FALSE  TRUE
## skewness_yaw_forearm       49.690037    0.01455922   FALSE  TRUE
## max_roll_forearm           18.666667    1.38312586   FALSE FALSE
## max_picth_forearm           3.500000    0.87355318   FALSE FALSE
## max_yaw_forearm           240.464286    0.28390478   FALSE  TRUE
## min_roll_forearm           18.666667    1.42680352   FALSE FALSE
## min_pitch_forearm           3.294118    0.88811240   FALSE FALSE
## min_yaw_forearm           240.464286    0.28390478   FALSE  TRUE
## amplitude_roll_forearm     18.666667    1.47776079   FALSE FALSE
## amplitude_pitch_forearm     3.800000    0.98274732   FALSE FALSE
## amplitude_yaw_forearm      62.632558    0.02183883   FALSE  TRUE
## total_accel_forearm         1.123864    0.48773386   FALSE FALSE
## var_accel_forearm           2.000000    1.94365582   FALSE FALSE
## avg_roll_forearm           28.000000    1.56511611   FALSE  TRUE
## stddev_roll_forearm        58.000000    1.55783650   FALSE  TRUE
## var_roll_forearm           58.000000    1.55783650   FALSE  TRUE
## avg_pitch_forearm          56.000000    1.57239572   FALSE  TRUE
## stddev_pitch_forearm       56.000000    1.57239572   FALSE  TRUE
## var_pitch_forearm          56.000000    1.57239572   FALSE  TRUE
## avg_yaw_forearm            56.000000    1.57239572   FALSE  TRUE
## stddev_yaw_forearm         57.000000    1.56511611   FALSE  TRUE
## var_yaw_forearm            57.000000    1.56511611   FALSE  TRUE
## gyros_forearm_x             1.063361    2.05284997   FALSE FALSE
## gyros_forearm_y             1.058140    5.27043750   FALSE FALSE
## gyros_forearm_z             1.097701    2.14020528   FALSE FALSE
## accel_forearm_x             1.015625    5.66353643   FALSE FALSE
## accel_forearm_y             1.000000    7.08306035   FALSE FALSE
## accel_forearm_z             1.018349    4.06930189   FALSE FALSE
## magnet_forearm_x            1.083333   10.54087501   FALSE FALSE
## magnet_forearm_y            1.192982   13.32168596   FALSE FALSE
## magnet_forearm_z            1.023256   11.76384946   FALSE FALSE
## classe                      1.469526    0.03639805   FALSE FALSE
```

```r
nsv<-nearZeroVar(trainingHAR)
trainingHAR<- trainingHAR[, -nsv]
testingHAR<-testingHAR[,-nsv]
```
Variable with missing values are also frequent. We measur the quantity of these NA values and put a threshold to 90%. Then we will keep only variables with less than 90% of NA values

```r
##
## Then eliminates predictors with imprtant percentage of missing values (>90%)
##
##
na_count <-sapply(trainingHAR, function(y) sum(length(which(is.na(y)))))
na_perc=na_count/dim(trainingHAR)
```

```
## Warning in na_count/dim(trainingHAR): la taille d'un objet plus long n'est
## pas multiple de la taille d'un objet plus court
```

```r
reducedTrainingHAR<-trainingHAR[, na_perc <= 0.9]
reducedTestingHAR<-testingHAR[, na_perc <= 0.9]
```

So 53 selected Predictors of 160 remains for building the prediction model  

STEP 3 The Algorythm

Random Forest used with cross validation seems to be a good approach, because the class of alghorithm fit well this kind of classification problem, bootstrap and cross validation are also easy to use.
We can estimate how the number of predictors will influence the error of the model using the rfcv,randomForest or train function in caret. the differences are in execution time and output variables, and the use with the varImp output to estimate importances of predictors and errors outputs.
The train function is the most time consuming.

```r
# set seed value and parallel options
set.seed(125)
doMC::registerDoMC(cores=4)
library(MASS)
library(randomForest)
library(class)

#FitModel <- rfcv(reducedTrainingHAR[,1:(dim(reducedTrainingHAR)[2]-1)],reducedTrainingHAR[#,dim(reducedTrainingHAR)[2]],cv.fold=5,mtry=function(p) max(1, floor(sqrt(p))), #recursive=TRUE)

#
# Alternatively, we can use this
#FitModel0 <- randomForest(reducedTrainingHAR[,1:(dim(reducedTrainingHAR)[2]-1)],reducedTrainingHAR[,dim(reducedTrainingHAR)[2]],cv.fold=5, importance=TRUE)
#FitModel2 <- train(classe ~ ., #method="rf",data=reducedTrainingHAR,trControl=trainControl(method="cv",number=5),prox=TRUE#,importance=TRUE,allowParallel=TRUE)
#
```
Plotting FitModel2 shows that the error decrease is not very important after 26 predictors chosen by the alghorythm
FitModel (obtained with rfcv function)shows also how errors decrease with n° trees for each class


```r
plot(varImp(FitModel2))
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png) 

```r
#
#
#
plot(FitModel0)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-2.png) 

```r
# print the nb of selected predictors and the accuracy

FitModel$error.cv
```

```
##          52          26          13           6           3           1 
## 0.006988425 0.008517143 0.080366892 0.117638495 0.148504040 0.474776152
```

```r
# also print the most important predictors
varImpPlot(FitModel0,type=2)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-3.png) 

```r
#varImp(FitModel0)
```


The model result shows a high accuracy in the training set.

```r
FitModel2
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10991, 10989, 10991, 10988 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9910469  0.9886739  0.002960099  0.003745062
##   27    0.9913375  0.9890407  0.001824816  0.002309775
##   52    0.9855129  0.9816717  0.003170484  0.004012370
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

STEP4 EVALUATION

Lets try it on the testing set and use confusion matrix to assess the score


```r
resu<-confusionMatrix(reducedTestingHAR$classe, predict(FitModel2, reducedTestingHAR))
resu
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    1 1136    2    0    0
##          C    0    2 1024    0    0
##          D    0    0    5  959    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9981          
##                  95% CI : (0.9967, 0.9991)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9976          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9982   0.9932   0.9990   1.0000
## Specificity            1.0000   0.9994   0.9996   0.9990   0.9998
## Pos Pred Value         1.0000   0.9974   0.9981   0.9948   0.9991
## Neg Pred Value         0.9998   0.9996   0.9986   0.9998   1.0000
## Prevalence             0.2846   0.1934   0.1752   0.1631   0.1837
## Detection Rate         0.2845   0.1930   0.1740   0.1630   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9988   0.9964   0.9990   0.9999
```

```r
z<-as.table(resu)
z
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    1 1136    2    0    0
##          C    0    2 1024    0    0
##          D    0    0    5  959    0
##          E    0    0    0    1 1081
```

```r
# try also a heatmap but not really nice
# colnames(z) = c("A","B","C","D","E")
# rownames(z)=colnames(z)
# image(z[,ncol(z):1], axes=FALSE)
# heatmap(t(z)[ncol(z):1,], Rowv=NA,Colv=NA, col = heat.colors(256))
#
```
Balanced accuracy is really nice.

Let's finish with the 20 observations loaded in the file (let's apply same reduction to variables)

```r
outdata20<-inDatasetValidHAR[,8:d[2]]
outdata20<-outdata20[,-nsv]
outdata20<-outdata20[, na_perc <= 0.9]
#predict20=predict(FitModel,outdata20)
#predict20
```
