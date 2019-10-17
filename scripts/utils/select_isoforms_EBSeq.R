library(EBSeq)
library(stringr)

path <- "../../datasets/"

files <- list.files(path)

for(fileName in files){
  
  if (endsWith(fileName, "-preproc-filter.csv")){
    
    header <- read.csv(paste0(path, fileName), sep = ",", row.names = 1, header = FALSE, stringsAsFactors= FALSE)
    
    isoform_names <- c()
    gene_names <- c()
    
    original_names <- header[1,1:(ncol(header)-1)]
    
    for(name in original_names){
      
      names <- strsplit(name, ",")[[1]]
      
      if(length(names) == 2){
        
        gene_names <- c(gene_names, names[1]) #
        isoform_names <- c(isoform_names, names[2]) #names[2]
        
      } else {
        gene_names <- c(gene_names, names[1]) #names[1]
        isoform_names <- c(isoform_names, names[1]) #names[1] # The same name of the isoform is assigned as gene name
      }
    }
    
    conditions <- as.factor(header[2: nrow(header), ncol(header)])
    
    matrix <- t(data.matrix(header[2:nrow(header), -ncol(header)]))
    
    IsoSizes <- MedianNorm(matrix)
    
    NgList <- GetNg(isoform_names, gene_names)
    
    IsoNgTrun <- NgList$IsoformNgTrun
    
    IsoEBOut <- EBTest(Data= matrix, NgVector= IsoNgTrun, Conditions= conditions, 
                       sizeFactors= IsoSizes, maxround= 5)
    
    IsoEBDERes <- GetDEResults(IsoEBOut, FDR=0.05)
    
    indexes <- which(IsoEBDERes$PPMat[,2] >= 0.95)
    
    dataset <- header[2: nrow(header), c(indexes, ncol(header))]
    
    colnames(dataset) <- c(original_names[indexes], "Class")
    
    write.csv(dataset, file = paste0(path, substr(files[2], 1, nchar(files[2])-4), "-isoforms-filter.csv"), sep = ",", row.names = TRUE)
  }
}