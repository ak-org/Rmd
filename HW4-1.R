library(cluster)
library(ape)

##
## Source https://www.statmethods.net/advstats/cluster.html
##
rm(list = ls())
setwd('/Users/ashishkumar/cs498aml/jobs')
jobsdata = read.csv("/Users/ashishkumar/cs498aml/jobs/data.csv", header=TRUE)
countries = as.vector(jobsdata[,1])

## map the data in 2-D space to make sense of data
d=dist(jobsdata[,2:10])
fit = cmdscale(d, eig=TRUE, k = 2)
plot(fit$points[,1], fit$points[,2],xlim=c(-25,65), ylim=c(-15,15),  
     main="European Countries Job Data", pch=16,
     xlab = "Countries", ylab="Countries", xaxt='n',yaxt='n')
text(fit$points[,1], fit$points[,2], labels=countries, cex=0.9, pos=3)

## part 1a.

hClusterResults1 = hclust(dist(jobsdata[,-c(1)]), method="complete")


plot(hClusterResults1,  labels = countries, check = TRUE,
     axes = TRUE, frame.plot = FALSE, ann = TRUE,
     main = "Cluster Dendrogram - Complete Link",
     sub = NULL, xlab = "Countries", ylab = "Height",hang = -1)
rect.hclust(hClusterResults1, k = 4, border = 1:6)

hClusterResults2 = hclust(dist(jobsdata[,-c(1)]), method="single")

plot(hClusterResults2,  labels = countries,check = TRUE,
     axes = TRUE, frame.plot = FALSE, ann = TRUE,
     main = "Cluster Dendrogram - Single Link",
     sub = NULL, xlab = "Countries", ylab = "Height",hang = -1)
rect.hclust(hClusterResults2, k = 3, border = 1:3)

hClusterResults3 = hclust(dist(jobsdata[,-c(1)]), method="average")

plot(hClusterResults3, labels = countries, check = TRUE,
     axes = TRUE, frame.plot = FALSE, ann = TRUE,
     main = "Cluster Dendrogram - Average Link",
     sub = NULL, xlab = "Countries", ylab = "Height",hang = -1)
rect.hclust(hClusterResults3, k = 3,border = 2:4)

## part 1b.

withinss = rep(1, times=9)
X = 2:10
for (k in 2:10) {
  kmCluster = kmeans(jobsdata[,-c(1)], k, iter.max = 20, nstart = 1)
  withinss[k-1] = kmCluster$tot.withinss
  #print(paste("Number of clusters", k))
  #print(paste("Size = ",kmCluster$size))
  print(paste("Total Withiness ", kmCluster$tot.withinss, " Betweeness ",kmCluster$betweenss))
  #print("======")
}
print(X)
print(withinss)
plot(X, withinss, type='b',xlab="Clusters", ylab="Distance within cluster", 
     col="blue", xlim=c(2,10), main="Cluster Count vs. Intra Cluster distance ")


