# Learning Koopman operators to design reporters from two-condition, time-series gene expression

Important: Gene expression time-series should be sampled homogeneously in time. If you are looking to apply a similar approach to heterogeneously sampled data, let me know and I may be able to help with replacing the Koopman operator model with a Koopman generator model, allowing for downstream observability maximization to be applied.

## main.py
Required input: data directory (variable datadir) containing a compressed pickle (.gz) file. The file contains a list of the following: [rank 3 tensor of background subtracted gene expression dynamics (i.e. $maximum(treatment - control, 0.0)$) where 1st dim-sample, 2nd dim -time, 3rd dim-replicates, gene identifiers of dataset, row indices corresponding to dataset (this is here because the input dataset may not be the whole dataset), list of timepoints as integers (e.g. if the input datset consists of only the 2nd and 4th timepoints then it will be [1,3])].

There are flags in the script for saving the results, for selecting between two approaches for parameter inference/model selection (dynamic mode decomposition (exact or reduced), and deep dynamic mode decomposition (feedforward neural network with learned observables)), and for inducing model sparsity. 

## preprocess.py
After receiving as input the raw two-condition gene expression dynamics, this script does the following: 0) wrangle data (very specific to one dataset unfortunately), 1) downselect replicates and timepoints, 2) apply an autocorrelation based noise filter which removes genes comprised mostly of noise in any replicate, 3) computes dynamic time warping distance between genes' time series to determine if the dynamics are consistent across replicates, and 4) smooths the time series using a Savitsky-Golay filter. Most hyperparameters are hard-coded in script.

The result is a compressed pickle (.gz) file containing the necessary objects for main.py (i.e. the list of mentioned above in the main.py section). That means that the raw data file input here should also contain the geneIDs as well. 

## run.sh

Bash script that calls preprocess.py (if desired) and main.py (if desired). There are arguments that can be specified from the command line. Check out the comments in the script. 

Note: If not calling run.sh with arguments to run preprocess.py or main.py, be sure to update those variables in the respective scripts e.g. tps_to_keep in preprocess.py to specify which timepoints are being selected for modeling and optimization.

## postprocessing.ipynb

Take the scores given by the observabilty maximization algorithm in observability.py and produce figures. 

Dependent on regions.py to find the intergenic regions for all genes of interest. Further processing is still needed to grab putative promoters since some intergenic regions are extremely long compared to the average bacterial promoter. 

Also queries uniprot to grab function annotations for genes of interest. 