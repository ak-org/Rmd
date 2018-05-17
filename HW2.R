
set.seed(0517)
setwd("~/cs498aml")
options(scipen=999)
library(pracma)
library(e1071)
library(caret)
train_data = "/Users/ashishkumar/cs498aml/adult.data"
test_data = "/Users/ashishkumar/cs498aml/adult.test"
EPOCHS = 50
STEPS = 300
EVAL_STEPS = 30
VALIDATION_SET_IN_EPOCH = 1000


adulttrain = read.csv(train_data,header = FALSE,stringsAsFactors=FALSE,na.strings = " ?" )
adulttest = read.csv(test_data,header = FALSE ,stringsAsFactors=FALSE,na.strings = " ?")
adulttrain<-na.omit(adulttrain)
adulttest<-na.omit(adulttest)
# Set <=50K as -1 and >50K as 1


dataset = rbind(adulttrain, adulttest, make.row.names=FALSE)
dataset$V1 = scale(as.numeric(as.matrix(dataset$V1)))
dataset$V3 = scale(as.numeric(as.matrix(dataset$V3)))
dataset$V5 = scale(as.numeric(as.matrix(dataset$V5)))
dataset$V11 = scale(as.numeric(as.matrix(dataset$V11)))
dataset$V12 = scale(as.numeric(as.matrix(dataset$V12)))
dataset$V13 = scale(as.numeric(as.matrix(dataset$V13)))

## convert labels into binary class : -1 and +1 
## following text book / lectures 

dataset$V15[(dataset$V15 == " <=50K") | (dataset$V15 == " <=50K.") ] = -1
dataset$V15[(dataset$V15 == " >50K.") | (dataset$V15 == " >50K")] = 1
dataset$V15 = as.factor(dataset$V15)
## continous variables are age, fnlwgt, education-num, capital-gain, capital-loss


train = dataset[,c(1,3,5,11,12,13,15)]

trainIndex = createDataPartition(train$V15, p = 0.8, list=FALSE, times = 1)
trainSet = train[trainIndex,]
testSplit = train[-trainIndex,]
valIndex =  createDataPartition(testSplit$V15, p = 0.5, list=FALSE, times = 1)
testSet = testSplit[valIndex,]
valSet = testSplit[-valIndex,]
print(dim(trainSet))
print(dim(testSet))
print(dim(valSet))

## Standard Library package
## Use for validation only
##
#svmpackage()
## train a support vector machine on this data using stochastic gradient descent.
##
##

svmpackage = function() {
  lambda2 = c(1e-4,1e-3, 1e-2, 0.1, 1)
  for (i in c(1:5)) {
    print(lambda2[i])
  }
  model = svm(V15 ~ V1 + V3 + V5 + V11 + V12 + V13, data=trainSet, cost = 1, gamma = .5)
  predi = predict(model, valSet[,-c(7)])
  #confusionMatrix(predi, valSet[,c(7)])
  print(table(predi, valSet[,c(7)]))
}


## search for an appropriate value of the regularization constant, trying at least the values 
## [1e-3, 1e-2, 1e-1, 1]. Use the validation set for this search. You should use at least 50 epochs of at 
## least 300 steps each. In each epoch, you should separate out 50 training examples at random for 
## evaluation (call this the set held out for the epoch). 
## compute the accuracy of the current classifier on the set held out for the epoch every 30 steps.

labelInt = function(label) {
  yy = 0
  if (label == -1) {
      yy = -1
  } 
  else {
    yy = 1
  }
  return(yy)
}

predLabel = function(val) {
  if (val > 0) {
    return(1)
  }
  else {
    return(-1)
  }
}

cost = function(X,y,aparam, bparam,lambda) {
  rows = dim(X)
  localy = labelInt(y[1])
  
  sum = 0
  for (i in 1:6) {
    xi = X[,i]
    at = aparam[i]
    sum = sum + xi * at
  }
  
  yg = (sum[1] + bparam)
  hLoss = yg * localy
  #hLoss2 = (max(0,1 - hLoss))
  return(hLoss)

}

regu = function(lambda = 0.001, a) {
  locala = as.vector(a)
  bb = (t(locala) %*% locala)/2
  return (lambda* bb)
}

calc_testset_accuracy = function(feat, labels, a, b, lambda) {
  ## for each element in feature array
  ## calculate cost 
  ## hard code feature length to 50 
  right = 0
  wrong = 0
  for (i in 1:length(labels)) {
    yy = labelInt(labels[i]) 
    valCost = cost(feat[i,], labels[i], a, b, lambda)
    if (valCost >= 0) {
      right = right + 1
    } 
    else {
      wrong = wrong + 1
    }
    
  }
  accuracy = right/(right+wrong)
  
  ## compare with actual label
  ## determine accuracy
  return(accuracy)
}

calc_validationset_accuracy = function(feat, labels, a, b, lambda) {
     ## for each element in feature array
     ## calculate cost 
     ## hard code feature length to 50 
     right = 0
     wrong = 0
     for (i in 1:VALIDATION_SET_IN_EPOCH) {
         yy = labelInt(labels[i]) 
         valCost = cost(feat[i,], labels[i], a, b, lambda)
         if (valCost >= 0) {
             right = right + 1
         } 
         else {
             wrong = wrong + 1
         }

     }
     accuracy = right/(right+wrong)

     ## compare with actual label
     ## determine accuracy
     return(accuracy)
}


plotAccuracy_flat = function(l, m, epochs, lambda, fname,idx) {
  colors = rainbow(7)
  xrange = range(500)
  yrange = range(4.0)
  x = c(1:500)
  y = as.vector(t(m))
  png(filename=fname)
  plot(x,y,main=paste("Accuray plot for lambda ",toString(lambda)),col=colors[idx],
       xlim=c(-10,520),ylim=c(0.0,1.2),type = "l",xlab="Checkpoint",ylab="Accuracy")


  dev.off()
  return(y)
}


plotAccuracy = function(m, fname,title,ylim) {
   numLines = dim(m)[1]
   colors = rainbow(numLines)
   xrange = range(1000)
   yrange = range(ylim)
   x = c(1:500)
   png(filename=fname)
   for (i in 1:numLines) {
     y = m[i,]
     if (i == 1) {
       plot(x,y, main=paste(title ," Plot"),col=colors[i],
            type = "l",xlab="Checkpoint",
            ylab=title,ylim=c(0.0,ylim),xlim=c(0,700))
    }
     lines(x,y,type = "l",lty=1, lwd=2, col=colors[i])
   }
   legend(600, ylim, legend=c("1e-4","1e-3","1e-2","0.1","1"),
          title="Lambda (λ)",
          col=rainbow(numLines), lty=1:1, cex=0.8)
   dev.off()
}

plotMgn = function(l, m, epochs,lambda, fname) {
  colors = rainbow(epochs)
  xrange = range(10)
  yrange = range(1.2)
  x = c(1:10)
  png(filename=fname)
  for (i in 1:epochs) {
    y = m[i,]
    if (i == 1) {
      plot(x,y, main=paste("Vector Magnitude plot, lambda ",toString(lambda)),
           col=colors[i],xlab="Checkpoint",ylab="Mangnitude",ylim=c(1.2,2.6),xlim=c(0,12))
    }
    lines(x,y,type = "o",lty=1, lwd=2, col=colors[i])
  }
  dev.off()
  
}

plotMgn_flat = function(l, m, epochs,lambda, fname,idx) {
  colors = rainbow(7)
  xrange = range(10)
  yrange = range(6.2)
  x = c(1:500)
  y = as.vector(t(m))
  png(filename=fname)
  plot(x,y, main=paste("Vector Magnitude plot, lambda ",toString(lambda)),type="l",
           col=colors[idx],xlab="Checkpoint",ylab="Mangnitude",ylim=c(0,6.2),xlim=c(0,510))
  dev.off()
  return(y)
  
}



#mySVMAlgo = function(lambda) {
mySVMAlgo = function() {
  ## input matrix is column 6 wide
  ## initialize a and b to some value
  lambdalist = c(1e-4,1e-3,1e-2,0.1,1)
  accLambda = matrix(list(), nrow=length(lambdalist), ncol = EPOCHS * STEPS/EVAL_STEPS)
  mgnLambda = matrix(list(), nrow=length(lambdalist), ncol = EPOCHS * STEPS/EVAL_STEPS)

  for (l in c(1:length(lambdalist))) {
    lambda = lambdalist[l]
    print(paste("Lambda is ",lambda))
    a = c(0.5,0.5,0.5,0.5,0.5,0.5)
    #a = c(2,2,2,2,2,2)
    b = 1
    
    accuracy_list = c()
    plus = 0
    minus = 0
    steplength_a <- .01
    steplength_b <- 50
    ## create two dimensional array to store accuracy for 10 checkpoint for 50 epochs
    accuracyMatrix = matrix(list(), nrow=EPOCHS, ncol = STEPS/EVAL_STEPS)
    mgnMatrix = matrix(list(), nrow=EPOCHS, ncol = STEPS/EVAL_STEPS)

    for (epoch in 1:EPOCHS){
      ## randomly chose 50 training samples for evaluation
      epoch_idx = sample(1:dim(trainSet)[1],VALIDATION_SET_IN_EPOCH)
      epoch_val_features = trainSet[epoch_idx,-c(7)]
      epoch_val_labels = trainSet[epoch_idx,c(7)]
      epoch_train_features = trainSet[-epoch_idx,-c(7)]
      epoch_train_labels = trainSet[-epoch_idx,c(7)]
      epoch_train_count = dim(epoch_train_features)
      epoch_val_count = dim(epoch_val_features)   

      rnd_idx = sample(1:epoch_train_count[1],STEPS, replace = T) 
      eta = 1/(0.01*epoch + 50)
      #eta = 0.1*exp(-0.075*epoch) # .6, ,65, .7,.75
      #eta = 1/((0.0001 * epoch) + 50)
  
      checkpoint = 1
      for (step in 1:STEPS){ #STEPS
        ## calculate the cost
        ## pick a random sample from training data
        ## we will do it 300 times (1 for each step) and for 50 epochs
        totalCost =  cost(epoch_train_features[rnd_idx[step],], epoch_train_labels[rnd_idx[step]], a,b, lambda)
        yy = labelInt(epoch_train_labels[rnd_idx[step]])
        #print(paste("totalcost ",totalCost))
        if (totalCost >= 1.0) {
          b = b - (eta * 0)
          for (i in 1:6) {
            a[i] = a[i] - (eta * lambda * a[i])
          }
          
        }
        else {
          b = b - (eta * -yy)
          for (i in 1:6) {
            xx = epoch_train_features[rnd_idx[step],i]
            a[i] = a[i] - (eta * ((lambda * a[i]) - (xx *yy)))
          }
        }
        ## if steps is multiple of 30, calculate accuracy
        if (step %% EVAL_STEPS == 0) {
          accuracyMatrix[epoch, checkpoint] = calc_validationset_accuracy(epoch_val_features,epoch_val_labels,a,b,lambda)
          #mgnMatrix[epoch, checkpoint] = norm(a, type='2')
          mgnMatrix[epoch, checkpoint] = sum(abs(a))
          #mgnMatrix[epoch, checkpoint] = norm(as.matrix(a))
          checkpoint = checkpoint + 1
        }
        
      }

      regVal = regu(lambda,a)

    }
    
    fname = paste("plotAccuracy2",toString(lambda),".png")
    print(fname)
    accLambda[l,] = plotAccuracy_flat(lambda, accuracyMatrix, EPOCHS, lambda, fname,l) 
    
    fname = paste("mgnPlot2",toString(lambda),".png")
    print(fname)
    mgnLambda[l,] = plotMgn_flat(lambda, mgnMatrix, EPOCHS,lambda, fname,l)
    
    sum = 0
    for (i in 1:EPOCHS) {
      sum = sum + as.numeric(accuracyMatrix[i,10])
    }
    overallAcc = sum/EPOCHS
    print(paste("Overall Validation Accuracy ",overallAcc))
    vld_features = valSet[,-c(7)]
    vld_labels = valSet[,c(7)]
    vldAccuracy = calc_testset_accuracy(vld_features,vld_labels,a,b,lambda)
    print(paste("Accuracy on VALIDATION dataset is",vldAccuracy,dim(valSet)[1],lambda))
    
    tst_features = testSet[,-c(7)]
    tst_labels = testSet[,c(7)]
    tstAccuracy = calc_testset_accuracy(tst_features,tst_labels,a,b,lambda)
    print(paste("Accuracy on TEST dataset is",tstAccuracy,dim(testSet)[1]))
    
  }
  ## print all weights and accuracy in one plot
  print(dim(mgnLambda))
  plotAccuracy(accLambda,"acc.png","Accuracy",1.1)
  plotAccuracy(mgnLambda,"mgn.png","Magnitude",6.0)
}

mySVMAlgo()


