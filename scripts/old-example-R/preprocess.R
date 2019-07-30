library(caret)

fileT <- "0h-4h"

dataset <- read.csv(paste0(fileT, ".csv"), na.strings = "", row.names = 1)

methods <- c("zv", "nzv", "center","scale","YeoJohnson", "bagImpute")

preProcValues <- preProcess(dataset, method = methods)

datasetTransformed <- predict(preProcValues, dataset)

#detecting inconsistent or dupplicated examples
datasetTransformed <- datasetTransformed[!duplicated(datasetTransformed[,1:(ncol(datasetTransformed)-1)]),]

write.table(datasetTransformed, file= paste(fileT, "-preproc.csv",sep = ""),
            quote = FALSE, sep="," , row.names = TRUE, col.names = TRUE, na = "")
