\#Peer-graded Assignment: Prediction Assignment Writeup

This report is for the prediction assigment writeup. The goal is to
predict the classe using the data from
<a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" class="uri">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a>.

\#Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" class="uri">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a>
(see the section on the Weight Lifting Exercise Dataset).

Data

The training data for this project are available here:

<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv</a>

The test data are available here:

<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv</a>

The data for this project come from this source:
<a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" class="uri">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a>.
If you use the document you create for this class for any purpose please
cite them as they have been very generous in allowing their data to be
used for this kind of assignment.

\#\#Preprocessing

``` r
setwd("C:\\Users\\daatas\\Desktop")
#load packages
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(Hmisc)
```

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: Formula

    ## 
    ## Attaching package: 'Hmisc'

    ## The following objects are masked from 'package:base':
    ## 
    ##     format.pval, units

``` r
library(corrplot)
```

    ## corrplot 0.84 loaded

``` r
library(rattle)
```

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
library(rpart.plot)
```

    ## Loading required package: rpart

    ## 
    ## Attaching package: 'rpart'

    ## The following object is masked from 'package:survival':
    ## 
    ##     solder

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(gbm)
```

    ## Loaded gbm 2.1.5

``` r
#load data
training=read.csv("pml-training.csv", header=TRUE)
testing=read.csv("pml-testing.csv", header=TRUE)

#remove na
dim(training)
```

    ## [1] 19622   160

``` r
training= training[, colSums(is.na(training)) == 0]
dim(training)
```

    ## [1] 19622    93

``` r
dim(testing)
```

    ## [1]  20 160

``` r
testing= testing[, colSums(is.na(testing)) == 0]
dim(testing)
```

    ## [1] 20 60

``` r
#remove empty columns
dim(training)
```

    ## [1] 19622    93

``` r
training=training[!sapply(training, function(x) any(x == ""))]
dim(training)
```

    ## [1] 19622    60

``` r
dim(testing)
```

    ## [1] 20 60

``` r
testing=testing[!sapply(testing, function(x) any(x == ""))]
dim(testing)
```

    ## [1] 20 60

``` r
#remove unimportant columns
training=training[,-c(1,2,3,4,5)]
testing=testing[,-c(1,2,3,4,5)]

#remove near zero variance variables

n <- nearZeroVar(training)
trainingn <- training[, -n]
testn  <- testing[,-n]
dim(trainingn)
```

    ## [1] 19622    54

``` r
dim(testn)
```

    ## [1] 20 54

``` r
# correlations

c=cor(trainingn[,-c(1,54)])
#c
#rcorr(as.matrix(trainingn[,-c(1,54)])) #we can see that we have a small number of insignificant probabilities

corrplot(c, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
```

![](Proj_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
#create data for cross validation

set.seed(123456)
training = data.frame(trainingn)
inTrain <- createDataPartition(training$classe, p=0.70, list=F)
train <- training[inTrain, ]
cross_validation <- training[-inTrain, ]
```

\#\#Prediction models

``` r
#Decisional trees


modFit1=train(classe~.,method="rpart",data=train)
print(modFit1$finalModel)
```

    ## n= 13737 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
    ##    2) roll_belt< 130.5 12584 8686 A (0.31 0.21 0.19 0.18 0.11)  
    ##      4) pitch_forearm< -33.95 1090    4 A (1 0.0037 0 0 0) *
    ##      5) pitch_forearm>=-33.95 11494 8682 A (0.24 0.23 0.21 0.2 0.12)  
    ##       10) magnet_dumbbell_y< 426.5 9607 6880 A (0.28 0.18 0.24 0.19 0.11)  
    ##         20) roll_forearm< 123.5 5961 3517 A (0.41 0.18 0.18 0.17 0.058) *
    ##         21) roll_forearm>=123.5 3646 2430 C (0.078 0.18 0.33 0.22 0.19) *
    ##       11) magnet_dumbbell_y>=426.5 1887  955 B (0.045 0.49 0.042 0.23 0.19) *
    ##    3) roll_belt>=130.5 1153    8 E (0.0069 0 0 0 0.99) *

``` r
#plot the tree
plot(modFit1$finalModel, uniform=TRUE, main="Classification Tree")
text(modFit1$finalModel, use.n=TRUE, all=TRUE, cex=.8)
```

![](Proj_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
fancyRpartPlot(modFit1$finalModel) 
```

![](Proj_files/figure-markdown_github/unnamed-chunk-2-2.png)

``` r
#Confusion Matrix
confusionMatrix(predict(modFit1,newdata=cross_validation), cross_validation$classe) #using cross_validation data
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1517  456  468  429  152
    ##          B   38  416   50  175  180
    ##          C  113  267  508  360  264
    ##          D    0    0    0    0    0
    ##          E    6    0    0    0  486
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4974          
    ##                  95% CI : (0.4845, 0.5102)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3434          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9062  0.36523  0.49513   0.0000  0.44917
    ## Specificity            0.6426  0.90666  0.79337   1.0000  0.99875
    ## Pos Pred Value         0.5020  0.48428  0.33598      NaN  0.98780
    ## Neg Pred Value         0.9452  0.85615  0.88155   0.8362  0.88949
    ## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
    ## Detection Rate         0.2578  0.07069  0.08632   0.0000  0.08258
    ## Detection Prevalence   0.5135  0.14596  0.25692   0.0000  0.08360
    ## Balanced Accuracy      0.7744  0.63595  0.64425   0.5000  0.72396

``` r
#Values predicted for testing data
predict(modFit1,newdata=testing)
```

    ##  [1] C A C A A C C A A A C C C A C A A A A C
    ## Levels: A B C D E

``` r
### Random forest

modFit<-randomForest(classe~., data=train,ntree=200)
varImp(modFit)# importance of variables
```

    ##                        Overall
    ## num_window           965.82489
    ## roll_belt            838.38421
    ## pitch_belt           441.29402
    ## yaw_belt             564.56344
    ## total_accel_belt     140.85784
    ## gyros_belt_x          59.54466
    ## gyros_belt_y          70.63266
    ## gyros_belt_z         178.91950
    ## accel_belt_x          83.48096
    ## accel_belt_y          88.97697
    ## accel_belt_z         261.66900
    ## magnet_belt_x        163.93657
    ## magnet_belt_y        260.79270
    ## magnet_belt_z        244.77226
    ## roll_arm             216.01408
    ## pitch_arm            108.05907
    ## yaw_arm              148.24949
    ## total_accel_arm       54.26832
    ## gyros_arm_x           70.09609
    ## gyros_arm_y           80.83241
    ## gyros_arm_z           34.88951
    ## accel_arm_x          158.17518
    ## accel_arm_y           95.68672
    ## accel_arm_z           76.91507
    ## magnet_arm_x         163.69816
    ## magnet_arm_y         152.67296
    ## magnet_arm_z         113.29712
    ## roll_dumbbell        269.93517
    ## pitch_dumbbell       121.88695
    ## yaw_dumbbell         175.24452
    ## total_accel_dumbbell 172.52873
    ## gyros_dumbbell_x      72.04009
    ## gyros_dumbbell_y     153.05866
    ## gyros_dumbbell_z      53.89657
    ## accel_dumbbell_x     149.91083
    ## accel_dumbbell_y     286.17146
    ## accel_dumbbell_z     218.17490
    ## magnet_dumbbell_x    316.26994
    ## magnet_dumbbell_y    464.70505
    ## magnet_dumbbell_z    497.89520
    ## roll_forearm         359.42835
    ## pitch_forearm        533.36411
    ## yaw_forearm           98.73386
    ## total_accel_forearm   61.85610
    ## gyros_forearm_x       43.91731
    ## gyros_forearm_y       70.32173
    ## gyros_forearm_z       45.82861
    ## accel_forearm_x      188.62204
    ## accel_forearm_y       82.22762
    ## accel_forearm_z      146.72561
    ## magnet_forearm_x     135.26638
    ## magnet_forearm_y     136.67143
    ## magnet_forearm_z     170.18755

``` r
#Confusion Matrix
confusionMatrix(predict(modFit,newdata=cross_validation), cross_validation$classe) #using cross_validation data
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    5    0    0    0
    ##          B    0 1134    8    0    0
    ##          C    0    0 1018    5    0
    ##          D    0    0    0  959    0
    ##          E    0    0    0    0 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9969          
    ##                  95% CI : (0.9952, 0.9982)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9961          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9956   0.9922   0.9948   1.0000
    ## Specificity            0.9988   0.9983   0.9990   1.0000   1.0000
    ## Pos Pred Value         0.9970   0.9930   0.9951   1.0000   1.0000
    ## Neg Pred Value         1.0000   0.9989   0.9984   0.9990   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1927   0.1730   0.1630   0.1839
    ## Detection Prevalence   0.2853   0.1941   0.1738   0.1630   0.1839
    ## Balanced Accuracy      0.9994   0.9970   0.9956   0.9974   1.0000

``` r
#Values predicted for testing data
predict(modFit,newdata=testing)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

``` r
### Boosted trees


modFit <- train(classe ~ ., method="gbm", data=train,trControl=trainControl(method = "repeatedcv", number = 5, repeats = 1),verbose=FALSE)
#Confusion Matrix
confusionMatrix(predict(modFit,newdata=cross_validation), cross_validation$classe) #using cross_validation data
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1666   19    0    0    0
    ##          B    7 1115   14    1    4
    ##          C    0    4 1007   15    2
    ##          D    1    1    5  947    5
    ##          E    0    0    0    1 1071
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9866          
    ##                  95% CI : (0.9833, 0.9894)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.983           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9952   0.9789   0.9815   0.9824   0.9898
    ## Specificity            0.9955   0.9945   0.9957   0.9976   0.9998
    ## Pos Pred Value         0.9887   0.9772   0.9796   0.9875   0.9991
    ## Neg Pred Value         0.9981   0.9949   0.9961   0.9965   0.9977
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2831   0.1895   0.1711   0.1609   0.1820
    ## Detection Prevalence   0.2863   0.1939   0.1747   0.1630   0.1822
    ## Balanced Accuracy      0.9954   0.9867   0.9886   0.9900   0.9948

``` r
#Values predicted for testing data
predict(modFit,newdata=testing)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
