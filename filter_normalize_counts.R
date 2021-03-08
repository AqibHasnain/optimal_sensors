library(tidyverse) 
library(tximport) # package for getting Kallisto results into R
library(rtracklayer)
library(edgeR) # well known package for differential expression analysis, but we only use for the DGEList object and for normalization methods
library(matrixStats) # let's us easily calculate stats on rows or columns of a data matrix
library(cowplot) # allows you to combine multiple plots in one figure
library(ggplot2)

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
cutoff = 10
keepers <- rowSums(cpm>cutoff)>=(length(sampleLabels) - 2) # rowSums(cpm>cutoff)>=(length(sampleLabels)/2)
DGEList.counts.filtered <- DGEList.counts[keepers,]
myTPM.filtered <- myTPM[keepers,]
dim(DGEList.counts.filtered)

myTPM.filtered.df <- as_tibble(myTPM.filtered, rownames = "geneID")
colnames(myTPM.filtered.df) <- c("geneID", sampleLabels)

# Normalize data
DGEList.counts.filtered.norm <- calcNormFactors(DGEList.counts.filtered, method = "TMM")

# use the 'cpm' function from EdgeR to get counts per million from your normalized data
log2.cpm.filtered.norm <- cpm(DGEList.counts.filtered.norm, log=TRUE)
log2.cpm.filtered.norm.df <- as_tibble(log2.cpm.filtered.norm, rownames = "geneID")
colnames(log2.cpm.filtered.norm.df) <- c("geneID", sampleLabels)
# pivot this NORMALIZED data, just as you did earlier
log2.cpm.filtered.norm.df.pivot <- pivot_longer(log2.cpm.filtered.norm.df, # dataframe to be pivoted
                                                cols = '1-Rep1_S1':'12M-Rep3_S36', # column names to be stored as a SINGLE variable
                                                names_to = "samples", # name of that new variable (column)
                                                values_to = "expression") # name of new variable (column) storing all the values (data)

fig1 <- ggplot(log2.cpm.filtered.norm.df.pivot) +
  aes(x=samples, y=expression, fill=samples) +
  geom_violin(trim = FALSE, show.legend = FALSE) +
  stat_summary(fun = "median", 
               geom = "point", 
               shape = 95, 
               size = 10, 
               color = "black", 
               show.legend = FALSE) +
  labs(y="log2 expression", x = "sample",
       title="Log2 Counts per Million (CPM)",
       subtitle="filtered, TMM normalized") +
  theme_bw() +
  theme(axis.text.x = element_text(angle=90))


# Now do the same process again of creating a DGEList, filtering, and normalizing, however the starting input will be RPK's instead of counts. ----
# RPK's are calculated as 
# RPK <- counts / gene_length
rpk <- myCounts / geneLengths

DGEList.rpk <- DGEList(rpk)
DGEList.rpk.filtered <- DGEList.rpk[keepers,]
DGEList.rpk.filtered.norm <- calcNormFactors(DGEList.rpk.filtered, method = "TMM")

rpk.filtered.norm <- cpm(DGEList.rpk.filtered.norm, log=FALSE)
log2.rpk.filtered.norm <- cpm(DGEList.rpk.filtered.norm, log=TRUE)
rpk.filtered.norm.df <- as_tibble(rpk.filtered.norm, rownames = "geneID")
log2.rpk.filtered.norm.df <- as_tibble(log2.rpk.filtered.norm, rownames = "geneID")
colnames(rpk.filtered.norm.df) <- c("geneID", sampleLabels)
colnames(log2.rpk.filtered.norm.df) <- c("geneID", sampleLabels)
# pivot this NORMALIZED data, just as you did earlier
log2.rpk.filtered.norm.df.pivot <- pivot_longer(log2.rpk.filtered.norm.df, # dataframe to be pivoted
                                                cols = '1-Rep1_S1':'12M-Rep3_S36', # column names to be stored as a SINGLE variable
                                                names_to = "samples", # name of that new variable (column)
                                                values_to = "expression") # name of new variable (column) storing all the values (data)

rpk.filtered.norm <- cpm(DGEList.rpk.filtered.norm)
rpk.filtered.norm.df <- as_tibble(rpk.filtered.norm, rownames = "geneID")
colnames(rpk.filtered.norm.df) <- c("geneID", sampleLabels)

fig2 <- ggplot(log2.rpk.filtered.norm.df.pivot) +
  aes(x=samples, y=expression, fill=samples) +
  geom_violin(trim = FALSE, show.legend = FALSE) +
  stat_summary(fun = "median", 
               geom = "point", 
               shape = 95, 
               size = 10, 
               color = "black", 
               show.legend = FALSE) +
  labs(y="log2 expression", x = "sample",
       title="Log2 Fragments per Kilobase per Million (FPKM)",
       subtitle="filtered, TMM normalized") +
  theme_bw() +
  theme(axis.text.x = element_text(angle=90))

plot_grid(fig1, fig2, labels = c('A', 'B'), label_size = 12, nrow = 2)

# write.table(myTPM.filtered.df,file="tpm.csv",sep=',') # TPM
# write.table(rpk.filtered.norm.df,file="getmm.csv",sep=',') # gene length corrected TMM
# write.table(log2.cpm.filtered.norm.df,file='../cpm_tmm.csv',sep=',')
# write.table(rpk.filtered.norm.df,file='../rpk_tmm_nolog.csv',sep=',')


