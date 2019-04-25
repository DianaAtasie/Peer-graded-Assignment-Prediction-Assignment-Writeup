
# Preprocessing  ----------------------------------------------------------


#load packages
library(caret)
library(Hmisc)
library(corrplot)
library(rattle)
library(rpart.plot)
library(randomForest)
library(gbm)
library(rmarkdown)
#load data
training=read.csv("pml-training.csv", header=TRUE)
testing=read.csv("pml-testing.csv", header=TRUE)

#remove na
dim(training)
training= training[, colSums(is.na(training)) == 0]
dim(training)

dim(testing)
testing= testing[, colSums(is.na(testing)) == 0]
dim(testing)

#remove empty columns
dim(training)
training=training[!sapply(training, function(x) any(x == ""))]
dim(training)

dim(testing)
testing=testing[!sapply(testing, function(x) any(x == ""))]
dim(testing)

#remove unimportant columns
training=training[,-c(1,2,3,4,5)]
testing=testing[,-c(1,2,3,4,5)]

#remove near zero variance variables

n <- nearZeroVar(training)
trainingn <- training[, -n]
testn  <- testing[,-n]
dim(trainingn)
dim(testn)

# correlations

c=cor(trainingn[,-c(1,54)])
c

rcorr(as.matrix(trainingn[,-c(1,54)])) #we can see that we have a small number of insignificant probabilities

corrplot(c, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#create data for cross validation

set.seed(123456)
training = data.frame(trainingn)
inTrain <- createDataPartition(training$classe, p=0.70, list=F)
train <- training[inTrain, ]
cross_validation <- training[-inTrain, ]


# Prediction models -------------------------------------------------------

###Decisional trees


modFit1=train(classe~.,method="rpart",data=train)
print(modFit1$finalModel)
#plot the tree
plot(modFit1$finalModel, uniform=TRUE, main="Classification Tree")
text(modFit1$finalModel, use.n=TRUE, all=TRUE, cex=.8)
fancyRpartPlot(modFit1$finalModel) 
#Confusion Matrix
confusionMatrix(predict(modFit1,newdata=cross_validation), cross_validation$classe) #using cross_validation data
#Values predicted for testing data
predict(modFit1,newdata=testing)


### Random forest

modFit<-randomForest(classe~., data=train,ntree=200)
varImp(modFit)# importance of variables
getTree(modFit,k=1) #extract a single tree from forest
#Confusion Matrix
confusionMatrix(predict(modFit,newdata=cross_validation), cross_validation$classe) #using cross_validation data
#Values predicted for testing data
predict(modFit,newdata=testing)

### Boosted trees


modFit <- train(classe ~ ., method="gbm", data=train,trControl=trainControl(method = "repeatedcv", number = 5, repeats = 1),verbose=FALSE)
#Confusion Matrix
confusionMatrix(predict(modFit,newdata=cross_validation), cross_validation$classe) #using cross_validation data
#Values predicted for testing data
predict(modFit,newdata=testing)
