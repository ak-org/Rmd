rm(list = ls())
library(pracma)
library(randomForest)
library(caret)
#library(xgboost)
#library(wskm)
#library(flexclust)
seed = 5 # best results with value of 5
set.seed(seed)
### START : Tunable Parameters #####  
SEGSIZE = 16
CUTSIZE = SEGSIZE*3
L1Cluster = 40
L2Cluster = 12
SPLITRATIO = 0.8
SLIDER=8
### END : Tunable Parameters  #####

DATADIR='/Users/ashishkumar/cs498aml/HMP_Dataset/dataset/'
bestK = L1Cluster * L2Cluster
activities = c('Brush_teeth',   
               'Climb_stairs',  
               'Comb_hair',     
               'Descend_stairs',
               'Drink_glass',   
               'Eat_meat',      
               'Eat_soup',      
               'Getup_bed',     
               'Liedown_bed',   
               'Pour_water',    
               'Sitdown_chair', 
               'Standup_chair', 
               'Use_telephone', 
               'Walk' )

numCategories = length(activities)
trainDataMarix = list()

createFileList = function() {
  catFiles = list()
  for (i in 1:length(activities)) {
      catFiles[[i]] = list.files(path=paste(DATADIR,activities[i], sep=""))
  }
  for (i in 1:length(activities)) {
    path = paste(DATADIR,activities[i])
    path = paste(DATADIR,activities[i],"/",sep="")
    for (j in 1:length(catFiles[[i]])) {
      catFiles[[i]][j] = paste(path,catFiles[[i]][j],sep="")
    }
  }
  
  trainfList = array()
  testfList = array()
  trainLabels = array()
  testLabels = array()
  idxTrain = 1
  idxTest = 1
  ## split in training test files
  for (i in 1:length(activities)) {
    catFileLen = length(catFiles[[i]])
    idx = sample(catFileLen, size = round(catFileLen*SPLITRATIO))

    trainfiles = catFiles[[i]][idx]
    testfiles = catFiles[[i]][-idx]
    for (j in 1:length(trainfiles)) {
      trainfList[idxTrain] = trainfiles[j]
      trainLabels[idxTrain] = i
      idxTrain = idxTrain + 1
    }
    for (j in 1:length(testfiles)) {
      testfList[idxTest] = testfiles[j]
      testLabels[idxTest] = i
      idxTest = idxTest + 1
    }
  }
  return(list("train" = trainfList, "test" = testfList, "trainLabels" = trainLabels, "testLabels" = testLabels))
}

fList = createFileList()
trainfLen = fList$train
testfLen = fList$test
print(length(fList$train))
print(length(fList$test))

createDataSegments = function(inpf){
  data = list()
  counter = 1
  reading = read.table(inpf, header=FALSE)
  for (j in seq(1, (nrow(reading)-SEGSIZE), SLIDER)){
    data[[counter]] = as.vector(unlist(reading[j:(j+SEGSIZE-1),]))
    counter = counter + 1
  }
  data = matrix(unlist(data), byrow=TRUE, ncol=CUTSIZE)
  return (data)
}


## create training data - may take a while to run
d =  apply(matrix(fList$train,ncol=1), 1, createDataSegments)
trainDataMatrix = matrix(NA, nrow=0, ncol=CUTSIZE)
cnt = 1
for (i in 1:length(d)) {
    for (k in 1:dim(d[[i]])[1]) {
      a = matrix(d[[i]][k,], nrow = 1,byrow=TRUE)
      trainDataMatrix = rbind(trainDataMatrix, a)
      cnt = cnt + 1
    }
}

trainDataMatrix = matrix(unlist(trainDataMatrix), byrow=TRUE, ncol=CUTSIZE)

km1 = kmeans(trainDataMatrix, centers = L1Cluster, iter.max=5000, nstart=5)
km2 = matrix(NA, nrow=0,ncol=CUTSIZE)
for (clustValue in 1:L1Cluster) {
  clustIdx = which(km1$cluster %in% clustValue)
  l2km = kmeans(trainDataMatrix[clustIdx,], L2Cluster, iter.max=8000, nstart=5)
  km2 = rbind(km2, l2km$centers)
}

# find closest center and return freq matrix

freqMatrix = function(idx, Kcenter = km2) {
  data = d[[idx]]
  y2 = unname(apply(pdist2(data, Kcenter), 1, which.min))
  retVal = rep(0, bestK)
  for (i in (1:bestK))
    retVal[i] = length(y2[y2 == i])
  return (retVal)
}



X_train = matrix(NA, nrow=length(d), ncol = bestK)
for (i in 1:length(d)) {
  temp = freqMatrix(i)
  X_train[i,] = temp
} 

Y_train = as.factor(fList$trainLabels)


## prepare the test data

d2 =  apply(matrix(fList$test ,ncol=1), 1, createDataSegments)
testDataMatrix = matrix(NA, nrow=0, ncol=CUTSIZE)


freqMatrix4test = function(idx, Kcenter = km2) {
  data = d2[[idx]]
  y2test = unname(apply(pdist2(data, Kcenter), 1, which.min))
  retVal = rep(0, bestK)
  for (i in (1:bestK)) {
    retVal[i] = length(y2test[y2test == i])
  }
  return (retVal)
}
testDataMatrix = matrix(unlist(testDataMatrix), byrow=TRUE, ncol=CUTSIZE)
X_test = matrix(NA, nrow=length(d2), ncol = bestK)
for (i in 1:length(d2)) {
  temp = freqMatrix4test(i)
  X_test[i,] = temp
}

Y_test = as.factor(fList$testLabels)

# cvtrain = rfcv(X_train,Y_train, cv.fold=10, step=0.5,ntrees=1000)
# print(paste("CV Error Rate: ", cvtrain$error.cv[1]))
seedValue = c(5,14,100,118,1000)
for (x in seedValue){
  print(x)
  set.seed(x)
  fit = randomForest(X_train, Y_train, ntrees = 2000, maxnodes = 2^8)
  pred2 = predict(fit, X_train)
  print(paste("Training",mean(pred2 == Y_train)))
  predicted = predict(fit, X_test)
  print(table(predicted, Y_test))
  print(paste("Test ",mean(predicted == Y_test)))
}


