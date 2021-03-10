library(tidyverse) 
library(tximport) # package for getting Kallisto results into R
library(rtracklayer)
library(edgeR) # well known package for differential expression analysis, but we only use for the DGEList object and for normalization methods
library(matrixStats) # let's us easily calculate stats on rows or columns of a data matrix

# set working directory to path containing the output of kallisto i.e. the abundance.tsv for each sample
setwd("/Users/aqib/Desktop/UCSB/Research/BCCL/koopman-sensor-placement/malathion-fluorescens-rnaseq/results-kallisto-trimmed")
targets <- read_tsv("studydesign.txt")
path <- file.path(targets$sample, "abundance.tsv") # set file paths to your mapped data
all(file.exists(path)) 

Tx <- rtracklayer::import("GCA_000009225.1_ASM922v1_genomic.gtf")
Tx <- as_tibble(Tx)
Tx <- dplyr::select (Tx,-c(source,type,score,phase))
Tx <- distinct(Tx, transcript_id,gene_id, .keep_all= TRUE)
# rename column which contains unique transcript identifier to target_id . 
# rename column which contains unique locus tag to gene_name
Tx <- dplyr::rename(Tx, target_id = protein_id, gene_name = locus_tag)
# just select those columns, don't really need the other for this script. 
Tx <- dplyr::select(Tx, "target_id", "gene_name")
# drop rows that contain NA in any column 
Tx <- na.omit(Tx)

Txi_gene <- tximport(path, 
                     type = "kallisto", 
                     tx2gene = Tx, 
                     txOut = TRUE, 
                     countsFromAbundance = "lengthScaledTPM",
                     ignoreTxVersion = TRUE)

myTPM <- Txi_gene$abundance
myCounts <- Txi_gene$counts
geneLengths <- Txi_gene$length
sampleLabels <- targets$sample

# Generate summary stats for your data 
myTPM.stats <- transform(myTPM, 
                         SD=rowSds(myTPM), 
                         AVG=rowMeans(myTPM),
                         MED=rowMedians(myTPM))

# create a DGEList using the edgeR package ----
DGEList.counts <- DGEList(myCounts)
cpm <- cpm(DGEList.counts) 

# Filter data
cutoff = 5
keepers <- rowSums(cpm>cutoff)>=(length(sampleLabels) - 2) # rowSums(cpm>cutoff)>=(length(sampleLabels)/2)
DGEList.counts.filtered <- DGEList.counts[keepers,]
myTPM.filtered <- myTPM[keepers,]
dim(DGEList.counts.filtered)

myTPM.filtered.df <- as_tibble(myTPM.filtered, rownames = "geneID")
colnames(myTPM.filtered.df) <- c("geneID", sampleLabels)



write.table(myTPM.filtered.df,file="tpm_removed_low_count_genes.csv",sep=',') # TPM
# write.table(rpk.filtered.norm.df,file="getmm.csv",sep=',') # gene length corrected TMM
# write.table(log2.cpm.filtered.norm.df,file='../cpm_tmm.csv',sep=',')
# write.table(rpk.filtered.norm.df,file='../rpk_tmm_nolog.csv',sep=',')


