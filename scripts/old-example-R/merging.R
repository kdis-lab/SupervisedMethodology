library(caret)

file0 <- "0h"
file1 <- "4h"

dataset0 <- read.csv(paste0(file0, ".csv"), na.strings = "", row.names = 1)
dataset0$Diebetes <- NULL

dataset1 <- read.csv(paste0(file1, ".csv"), na.strings = "", row.names = 1)

mergedDataset <- merge(dataset0, dataset1, by = "row.names", sort = F)

rownames(mergedDataset) <- mergedDataset$Row.names
mergedDataset$Row.names <- NULL

write.table(mergedDataset, file= paste(file0, "-", file1, ".csv",sep = ""),
            quote = FALSE, sep="," , row.names = TRUE, col.names = TRUE, na = "")
