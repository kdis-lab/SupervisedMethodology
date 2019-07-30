library(ggpubr)

#https://www.r-bloggers.com/add-p-values-and-significance-levels-to-ggplots/

# load the csv
output <- "results/boxplots/"

fileName <- "4h"

outputT <-paste0(output, fileName, "/")

if(!dir.exists(outputT)){
  dir.create(outputT, recursive = T)
}

#load the dataset
dataset <-read.csv(paste0(fileName, ".csv"), sep = ",", na.strings = "", row.names = 1)

colnamesT <- colnames(dataset)[-ncol(dataset)]

classes <- levels(dataset[, ncol(dataset)])

colClassName <- colnames(dataset)[ncol(dataset)]

#for each column
for(nameC in colnamesT){
  
  formula <- as.formula(paste0(nameC," ~ Diabetes"))
  
  a<-compare_means(formula, data = dataset)
  
  p <- ggboxplot(dataset, x = "Diabetes", y = nameC,
                 color = "Diabetes", palette = "jco")
  #  Add p-value
  p + stat_compare_means()
  
  # To change to t-test
  #p + stat_compare_means(method = "t.test")
  
  ggsave(width = 5, height = 5, filename = paste(outputT,nameC,".png",sep = ""), bg = "transparent")
}