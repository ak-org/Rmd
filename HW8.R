rm(list = ls())
library(stats)
library(matrixStats)
library(knitr)
library(jpeg)
library(OpenImageR)
library(raster)
set.seed(1492)

# helper function for visualization 
#  gray(12:1 / 12) to reverse the image

show_digit = function(arr784, col = gray(1:12 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}


# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# Part 1 - Obtain dataset load images and data files

#path = "~/Desktop/MCSDS/AppliedMachineLearning/week10/"
path = "~/cs498aml/"

trainFile = paste(path, "mnist/train-images-idx3-ubyte", sep='')
noiseFile = paste(path, "SupplementaryAndSampleData/NoiseCoordinates.csv", sep='')
updCoodFile = paste(path, "SupplementaryAndSampleData/UpdateOrderCoordinates.csv", sep='')
inpParamFile = paste(path, "SupplementaryAndSampleData/InitialParametersModel.csv", sep='')
sampleDesnoisedFile = paste(path, "SupplementaryAndSampleData/SampleDenoised.csv", sep='')

IMAGES = 20

train = load_image_file(trainFile)
noiseCoods = read.csv(noiseFile, header=TRUE)
updCoods = read.csv(updCoodFile, header=TRUE)
origInpParam = read.csv(inpParamFile, header=FALSE)
sampleDenoised = read.csv(sampleDesnoisedFile, header = FALSE)
denoisedImages = matrix(0, nrow=IMAGES, ncol=28 * 28)
## binarize the first 20 images by mapping any value below .5 to -1
## and any value above to 1

train = train[1:IMAGES,1:784]
binTrain = train/255.0

for (i in 1:nrow(binTrain)) {
  for (j in 1:ncol(binTrain)) {
    if (binTrain[i,j] < 0.5) {
      binTrain[i,j] = -1
    }
    else {
      binTrain[i,j] = 1
    }
  }
}

## Part 2 - Add predetermined noise to the images by flipping some pixels

flippedBinTrain = binTrain

## both values are same
## all(binTrain[1,] == flippedBinTrain[1,]) 
## will return true

## csv has 2 rows per image
for (img in seq(1,2 * IMAGES ,2)) { 
  bits = ncol(noiseCoods)
  for (i in 2:bits) { # 2:15
    idx = (noiseCoods[img,i] * 28) + (noiseCoods[img+1,i] + 1)
    trIdx = (img+1)/2
    #print(paste(idx, trIdx))
    flippedBinTrain[trIdx,idx] = -1.0 * flippedBinTrain[trIdx,idx]
  }
}


retEng = matrix(0, ncol = 11, nrow = 20)


vfe = function(idx, inpPI, theta1_ij = 0.8, theta2_ij = 2) {
  image <- as.numeric(flippedBinTrain[idx, ])
  img = matrix(image, nrow = 28, ncol = 28)
  img = t(img)
  epsilon = 1e-10
  pi = inpPI
  eqlogq = 0.0
  for (x in 1:28) {
    for (y in 1:28) {
      eqlogq = eqlogq + pi[x, y] * log(pi[x, y] + epsilon) + (1 - pi[x, y]) * log((1 -
                                                                                     pi[x, y]) + epsilon)
    }
  }
  
  eqlogp = 0.0
  for (x in 1:28) {
    for (y in 1:28) {
      if (x > 1) {
        eqlogp = eqlogp + theta1_ij * (2 * pi[x, y] - 1) * (2 * pi[x - 1, y] - 1)
      }
      if (x < 28) {
        eqlogp = eqlogp + theta1_ij * (2 * pi[x, y] - 1) * (2 * pi[x + 1, y] - 1)
      }
      if (y > 1) {
        eqlogp = eqlogp + theta1_ij * (2 * pi[x, y] - 1) * (2 * pi[x, y - 1] - 1)
      }
      if (y < 28) {
        eqlogp = eqlogp + theta1_ij * (2 * pi[x, y] - 1) * (2 * pi[x, y + 1] - 1)
      }
      eqlogp = eqlogp + theta2_ij * (2 * pi[x, y] - 1) * img[x, y]
    }
  }
  energy = eqlogq - eqlogp
  return(energy)
  
}

findnewPi = function(idx, inpParam, theta1_ij = 0.8, theta2_ij = 2) {
  image<-as.numeric(flippedBinTrain[idx,])
  img = matrix(image, nrow = 28, ncol = 28)
  img = t(img)
  pi = inpParam
  xlist = as.numeric(updCoods[idx * 2 - 1, 2:785]) + 1
  ylist = as.numeric(updCoods[idx * 2, 2:785]) + 1  
  
  for (i in 1:784) {
    x = xlist[i]
    y = ylist[i]
    z = 0.0
    if (x > 1) {
      z = z + theta1_ij  * (2 * pi[x - 1, y] - 1)
    }
    if (x < 28) {
      z = z + theta1_ij  * (2 * pi[x + 1, y] - 1)
    }
    if (y > 1) {
      z = z + theta1_ij * (2 * pi[x, y - 1] - 1)
    }
    if (y < 28) {
      z = z + theta1_ij * (2 * pi[x, y + 1] - 1)
    }
    z = z + theta2_ij * img[x,y]
    pi[x,y] = exp(z)/(exp(z) + exp(-z))
  }
  return(pi)
}

## energy 20 x 11 matrix
for (img in 1:IMAGES) {
  inpParam = origInpParam
  retEng[img,1] = vfe(img, inpParam)
  for (ip in 1:10) {
    inpParam = findnewPi(img, inpParam, theta1_ij = 0.8, theta2_ij = 2)
    retEng[img,ip+1] = vfe(img, inpParam)  
  }
}


##################################### reconstruct images and create image matrix ##############################


## part 3 - Building a Boltzman Machine for denoising the images and using Mean-Field Inference

theta1_ij = 0.8 #(Hi,Hj terms)
theta2_ij = 2 #(Hi,Xj terms)
STOP_THRESH = 1e-3
HEIGHT = 28
WIDTH = 28
denoisedImages = matrix(0, nrow=IMAGES, ncol=28 * 28)


reconstructImage = function(imgIdx, image, theta1, theta2) {
  image = flippedBinTrain[imgIdx,]
  image<-as.numeric(image)
  img = matrix(image, nrow = 28, ncol = 28)
  img = t(img)
  # every image has two entries in update coords file
  # use imageIdx to identify relevant rows 
  coodIdx = imgIdx * 2 
  pi_mat = inpParam
  
  for (iteration in 1:10) {
    for (j in 2:785) {
      x = updCoods[coodIdx - 1, j]
      y = updCoods[coodIdx, j]
      x = x + 1 # values are zero index based, adjust it for R
      y = y + 1
      # print(paste(x,y))
      z = 0.0
      if (x > 1) {
        z = z + (theta1 * (2 * pi_mat[x - 1, y] - 1) )
      }
      if (x < 28) {
        z = z + (theta1 * (2 * pi_mat[x + 1, y] - 1) )
      }
      if (y > 1) {
        z = z + (theta1 * (2 * pi_mat[x, y - 1] - 1) )
      }
      if (y < 28) {
        z = z + (theta1 * (2 * pi_mat[x, y + 1] - 1) )
      }
      z = z +  theta2 * img[x,y]
      
      pi_mat[x,y] = (exp(z) / (exp(-z) + exp(z)))
    }
    
  } # end of 10 iterations
  #Use pi to reconstruct image, returns 784 column wide array
  
  return(matrix(unlist(pi_mat), nrow=1))
  
}


## create denoised 20 images

for (i in  1:IMAGES) {
  denoisedImages[i,] = reconstructImage(i, flippedBinTrain[i,], theta1_ij, theta2_ij)
}


# reconstruct the denoised image 
# change 1:1 to 1:IMAGES when code is ready
for(i in 1:IMAGES) {
  for (j in 1:ncol(denoisedImages)) {
    if (denoisedImages[i,j] < 0.5) {
      denoisedImages[i,j] = 0
    }
    else {
      denoisedImages[i,j] = 1
    }
  }
}



for (i in 1:IMAGES) {
  par(mfrow = c(1, 3))
  ## verify that bit flipping and denoising happened
  show_digit(binTrain[i, ])
  show_digit(flippedBinTrain[i, ])
  show_digit(denoisedImages[i, ])
}


## create a 28 x 280 matrix outlining the 1-10 images by reshaping the denoisedImages matrix above
denoisedMatrix = matrix(nrow = 28, ncol = 0)
for (i in 1:10) {
  denoisedImg = matrix(as.numeric(denoisedImages[i,]), nrow =1 )
  denoi = matrix(denoisedImg, nrow = 28, ncol = 28)
  denoisedMatrix = cbind(denoisedMatrix, denoi)
}

# Match with the sample denoised images given
which(denoisedMatrix - sampleDenoised != 0)

## create a 28 x 280 matrix outlining the 11-20 images by reshaping the denoisedImages matrix above
denoisedMatrix = matrix(nrow = 28, ncol = 0)
for (i in 11:20) {
  denoisedImg = matrix(as.numeric(denoisedImages[i,]), nrow =1 )
  denoi = matrix(denoisedImg, nrow = 28, ncol = 28)
  denoisedMatrix = cbind(denoisedMatrix, denoi)
}



############# PART 6 Construction of an ROC curve ##########################
c = c(5, 0.6, 0.4, 0.35, 0.3, 0.1)

for (j in length(c)) {
  theta1_ij = c[j]
  FPR =  matrix(nrow = 20, ncol = 6)
  TPR =  matrix(nrow = 20, ncol = 6)
  for (i in  1:IMAGES) {
    denoisedImages[i, ] = reconstructImage(i, flippedBinTrain[i, ], theta1_ij, theta2_ij)
    onesInDenoisedImage = sum(denoisedImages[i, ] == 1)
    onesInOriginalImage = sum(binTrain[i,] == 1)
    minusOnesInDenoisedImage = sum(denoisedImages[i, ] == -1)
    minusOnesInOriginalImage = sum(binTrain[i, ] == -1)
  }
}

FPR =  matrix(-1, nrow = 10, ncol = 6)
TPR =  matrix(-1, nrow = 10, ncol = 6)
c = c(5, 0.6, 0.4, 0.35, 0.3, 0.1)

for (j in 1:length(c)) {
  theta1_ij = c[j]
  for (i in  11:IMAGES) {
    o1p1 = 0
    o1p0 = 0
    o0p0 = 0
    o0p1 = 0
    denoisedImages[i, ] = reconstructImage(i, flippedBinTrain[i, ], theta1_ij, theta2_ij)
    for (k in 1:ncol(denoisedImages)) {
      if (denoisedImages[i,k] < 0.5) {
        denoisedImages[i,k] = 0
      }
      else {
        denoisedImages[i,k] = 1
      }
    }
    for (px in 1:784) {
      if (denoisedImages[i, px] == 1 & binTrain[i, px] == 1) {
        o1p1 = o1p1 + 1
      }
      if (denoisedImages[i, px] == 0 & binTrain[i, px] == 1) {
        o1p0 = o1p0 + 1
      }
      if (denoisedImages[i, px] == 1 & binTrain[i, px] == -1) {
        o0p1 = o0p1 + 1
      }
      if (denoisedImages[i, px] == 0 & binTrain[i, px] == -1) {
        o0p0 = o0p0 + 1
      }
    }
    #print(paste(o1p1, o1p0,o0p1,o0p0))
    TPR[i - 10, j] = o1p1/(o1p1 + o1p0)
    FPR[i - 10, j] = o0p1/(o0p0 + o0p1)
  }
}

dim(retEng)
write.table(retEng[11:12,1:2], file = paste(path,"energy.csv",sep=''),row.names=FALSE,col.names=FALSE,sep=',')
dim(denoisedMatrix)
write.table(denoisedMatrix, file = paste(path,"denoised.csv",sep=''),row.names=FALSE,col.names=FALSE,sep=',')

