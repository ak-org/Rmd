library(caret)
library(klaR)


set.seed(3456)
adult1 = read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
                 header=FALSE, stringsAsFactors = FALSE, na.strings=c(" ?", "?", "NA"))
adult2 = read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", 
                 header=FALSE, stringsAsFactors = FALSE, na.strings=c(" ?", "?","NA"))
adult = rbind(adult1, adult2)

#Trim leading and trailing spaces and remove missing entries
adult = data.frame(lapply(adult, trimws), stringsAsFactors = FALSE)
adult = na.omit(adult)

# select only columns with continuous values.
adult = subset(adult, select = c("V1", "V3", "V5", "V11", "V12", "V13", "V15"))
# add column names
names(adult) = c("age", "fntwgt", "edu_num", "cap_gain", "cap_loss", "hrsperweek", "wages")

# scale all continuous features to a mean of 0 and SD of 1
# for (i in 1:6) 
#   adult[i] = scale(as.numeric(as.matrix(adult[i])))

# Set <=50K as -1 and >50K as 1
adult$wages[(adult$wages == "<=50K") | (adult$wages == "<=50K.") ] = -1
adult$wages[(adult$wages == ">50K.") | (adult$wages == ">50K")] = 1
adult$wages = as.factor(adult$wages)

# 80% of data for training
trindex = createDataPartition(adult$wages, p = .8, list = FALSE)
adult_train = adult[trindex,]
adult_rem = adult[-trindex,]

# 10% for Validation; 10% for test
index = createDataPartition(adult_rem$wages, p = .5, list = FALSE)
adult_val = adult_rem[index,]
adult_test = adult_rem[-index,]

# scale all continuous features to a mean of 0 and SD of 1
for (i in 1:6) {
  adult_train[i] <- scale(as.numeric(as.matrix(adult_train[i])))
  adult_val[i] <- scale(as.numeric(as.matrix(adult_val[i])))
  adult_test[i] <- scale(as.numeric(as.matrix(adult_test[i])))
}

nepochs = 50
nsteps = 300
lambdas = c(0.001, 0.01, 0.1, 1)
steps_till_eval = 30
steplength_a = 0.01
steplength_b = 50
A_vector = rep(0, nepochs)

# array of Validation and Test accuracies
train_acc = c()
validation_acc = c()
test_acc = c()

x_train = adult_train[,-7]
y_train = adult_train[,7]

x_val = adult_val[,-7]
y_val = adult_val[,7]

x_test = adult_test[,-7]
y_test = adult_test[,7]

# Evaluate a^T*x + b 
evaluate = function(x, a, b) {
  new_x = as.numeric(as.matrix(x))
  return (t(a) %*% new_x + b)
}

convertPred = function(val) {
  if (val >= 0) return(1)
  else return(-1)
}

# Calculate accuracy 
calcAccuracy = function (x, y, a, b) {
  correct = 0
  wrong = 0
  for (i in 1:length(y)) {
    pred = evaluate(x[i,], a, b)
    pred = convertPred(pred)
    
    if (y[i] == pred) 
      correct = correct +1 
    else 
      wrong = wrong + 1
  }
  
  return (c((correct/(correct+wrong)), correct, wrong))
}

# loop over all lambda values
for (lambda in lambdas) {
  a = c(0,0,0,0,0,0)
  b = 0
  accuracies = c()
  # pos=0
  # neg = 0
  for (epoch in 1:nepochs) {
    valIndex = sample(1:nrow(x_train), 50)
    val_data = x_train[valIndex, ]
    val_labels = y_train[valIndex]
    train_data = x_train[-valIndex, ]
    train_labels = y_train[-valIndex]
   
    for (step in 1:nsteps) {
      if(step %% steps_till_eval ==0) {
        acc = calcAccuracy(val_data, val_labels, a, b)
        accuracies = c(accuracies, acc[1])
        
        # jpeg(file="a1.jpg")
        # title <- paste("Lambda = ", toString(a[1]), " Accuracies Graph")
        # plot(1:length(a) , a, type="o", col="dodgerblue", xlab ="Steps", ylab ="coeff vector", main = title)
        # dev.off()
        # 
        jpeg(file=paste(toString(lambda),".jpg", sep=""))
        title <- paste("Lambda = ", toString(lambda), " Accuracies Graph")
        plot(1:length(accuracies) , accuracies, type="o", col="dodgerblue", xlab ="Steps", ylab ="Accuracy", main = title)
        dev.off()
      }
      
      k = sample(1:length(train_labels) , 1)
      x = as.numeric(as.matrix(train_data[k,]))
      y = as.numeric(train_labels[k]) 
      if (y == 2)  y = 1 else y = -1
      pred = evaluate(x, a, b)
      steplength = 1/((steplength_a * epoch) + steplength_b)
      
      if (y*pred >= 1) {
        val1 = lambda * a
        val2 = 0
      } else {
        val1 = (lambda*a) - (y*x)
        val2 = -y
      }
      a = a - (steplength * val1)
      b = b - (steplength * val2)
    }
  }
  train_acc = c(train_acc, mean(accuracies))
  valacc = calcAccuracy(x_val, y_val, a, b)
  validation_acc = c(validation_acc,valacc[1])
  print(validation_acc)
  
  testacc = calcAccuracy(x_test, y_test, a, b)
  test_acc = c(test_acc,testacc[1])
  
  #jpeg(file=paste(toString(lambda),".jpg") )
  title = paste("Lambda = ", toString(lambda), " Accuracies Graph")
  plot(1:length(accuracies) , accuracies, type="o", col="blue", xlab ="Epochs", ylab ="Accuracy", main = title)
}

mydf = data.frame(c(train_acc), c(validation_acc),c(test_acc))
colnames(mydf) =  c("Train Accuracy", "Validation Accuracy","Test Accuracy")
rownames(mydf) =  c("lambda = 0.001","lambda = 0.01",
                    "lambda = 0.1", "lambda = 1")
knitr::kable(mydf)



max_index = 1
for(i in 1:length(validation_acc)){
  if (validation_acc[i] >= validation_acc[max_index]){
    max_index = i
  }
}
max_lambda = lambdas[max_index]
max_lambda


# accuracy of the best classifier on test data
test_acc[max_index]



# library("e1071")
# 
# svm_model = svm(wages ~ ., data=adult_train)
# summary(svm_model)
# print(svm_model)
# 
# pred2 = fitted(svm_model)
# 
# # Check accuracy:
# table(pred2, adult_train$wages)
# 
# pred_train = predict(svm_model,adult_train)
# tr_acc = mean(pred_train==adult_train$wages)
# 
# pred_val = predict(svm_model,adult_val)
# val_acc = mean(pred_val==adult_val$wages)
# table(pred_val, adult_val$wages)
# 
# pred_test = predict(svm_model,adult_test)
# test_acc = mean(pred_test==adult_test$wages)
# table(pred_test, adult_test$wages)
# mydf1 = data.frame(c(tr_acc), c(val_acc),c(test_acc))
# colnames(mydf1) =  c("Train Accuracy", "Validation Accuracy","Test Accuracy")
# 
# knitr::kable(mydf1)