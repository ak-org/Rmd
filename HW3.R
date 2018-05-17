# Read binary file and convert to integer vectors
# [Necessary because reading directly as integer()
# reads first bit as signed otherwise]
#
# File format is 10000 records following the pattern:
# [label x 1][red x 1024][green x 1024][blue x 1024]
# NOT broken into rows, so need to be careful with "size" and "n"
#
# (See http://www.cs.toronto.edu/~kriz/cifar.html)

## Catgegories 1:10
## (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
pwd = "/Users/ashishkumar/cs498aml/cifar-10-batches-py/"
labels = read.table("/Users/ashishkumar/cs498aml/cifar-10-batches-bin/batches.meta.txt")
images.rgb = list()
images.lab = list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory
totalLabels = 10
imgH = 32
imgW = 32
totalImages = num.images * 5
totalCat = 10
maxImgInCat = 5000
idxImgLabel = array(rep(0, totalLabels * maxImgInCat) , c(totalLabels, maxImgInCat))

# Cycle through all 5 binary files
for (f in 1:5) {
  to.read = file(paste("/Users/ashishkumar/cs498aml/cifar-10-batches-bin/data_batch_", f, ".bin", sep=""), "rb")
  for(i in 1:num.images) {
    l = readBin(to.read, integer(), size=1, n=1, endian="big")
    r = as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    g = as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    b = as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    index = num.images * (f-1) + i
    images.rgb[[index]] = data.frame(r, g, b)
    images.lab[[index]] = l+1
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}


# function to run sanity check on photos & labels import
drawImage = function(index) {

  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img = images.rgb[[index]]
  print(index)
  print(images.lab[[index]])
  img.r.mat = matrix(img$r, ncol=32, byrow = TRUE)
  img.g.mat = matrix(img$g, ncol=32, byrow = TRUE)
  img.b.mat = matrix(img$b, ncol=32, byrow = TRUE)
  img.col.mat = rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) = dim(img.r.mat)
  
  # Plot and output label
  library(grid)
  grid.raster(img.col.mat, interpolate=FALSE)

  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  print(labels)
  labels[[1]][images.lab[[index]]]
}

imgDataPerLabel = function() {
  for (lbl in 1:totalCat) { 
    l = as.integer(lbl)
    idxList = which(images.lab %in% c(l))
    for (k in 1:length(idxList)) {
      idxImgLabel[l,k] = idxList[k]
    }
    print(length(idxList))
    print(l)
  }
}

# function to run sanity check on photos & labels import
meanImage = function(imageCount) {
  sumImages = array(rep(0, totalLabels * imgW * imgH * 3) , c(totalLabels, imgW * imgH , 3))
    print(dim(sumImages))
  lblCount = c(0,0,0,0,0,0,0,0,0,0)
  # go through the images 
  # each label has 10000 images
  # combine into an rgb object, and display as a plot
  for (img_num in 1:imageCount) { 
    img = images.rgb[[img_num]]
    imgLabel = images.lab[[img_num]]
    lblCount[imgLabel] = lblCount[imgLabel] + 1
    #print(paste("img, label",dim(img),imgLabel))
    for (i in 1:(imgW * imgH)) {
      for (j in 1:3) {
        sumImages[imgLabel,i,j] = sumImages[imgLabel,i,j] + img[i,j]
      }
    }
    if ((img_num %% 500) == 0) {
      print(img_num)  
    }

    #for (i in 1:10) {
    #  for (j in 1:3) {
    #    print(paste(i,j,img[i,j],sumImages[imgLabel,i,j]))
    #  }
    #}
  }
  print("dim of sum of all images")
  print(lblCount)
  #print(sumImages[1:10,1,1:3])
  for (l in 1:totalCat) {
    sumImages[l, 1:(imgW * imgH) ,1:3]  = sumImages[l, 1:(imgW * imgH) ,1:3]/lblCount[l] 
  }
  return(sumImages[1:totalCat, 1:(imgW * imgH) ,1:3])
}


displMeanImage = function(l) {
  imgMean.r.mat = matrix(meanofImages[l, 1:(imgW * imgH) ,1], ncol=imgW, byrow = TRUE)
  imgMean.g.mat = matrix(meanofImages[l, 1:(imgW * imgH) ,2], ncol=imgW, byrow = TRUE)
  imgMean.b.mat = matrix(meanofImages[l, 1:(imgW * imgH) ,3], ncol=imgW, byrow = TRUE)
  imgMean.col.mat = rgb(imgMean.r.mat, imgMean.g.mat, imgMean.b.mat, maxColorValue = 255)
  dim(imgMean.col.mat) = dim(imgMean.r.mat)
  #print(imgMean.col.mat)
  # Plot and output label
  library(grid)
  grid.raster(imgMean.col.mat, interpolate=FALSE)
}


#drawImage(sample(1:(num.images*5), size=1))
meanofImages = meanImage(5000)
print(dim(meanofImages))

for (l in 1:totalCat) {
  for (j in 1:3) {
    meanofImages[l,1:1024,j] = as.integer(meanofImages[l,1:1024,j])
  }
}


displMeanImage(1) 
dd=matrix(data=meanofImages[7,1:1024,1], nrow=32, ncol=32, byrow=TRUE)
pc = prcomp(dd, rank. = 20)
fviz_eig(pc,ncp=20)


