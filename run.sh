#!/bin/bash

# run one of or both preprocess.py and main.py and write outputs to specified directory

dir='X1'
mkdir run-outputs/${dir}

# save output of preprocess to directory specified in first argument
python3 preprocess.py run-outputs/${dir} | tee run-outputs/${dir}/preprocess_log.txt

# arguments for main.py are: 
# 1) replicates to use
# 2) when saving the output (compressed) pickle, what should be the unique tag over the various runs
# 3) the directory to save the outputs to 
# 4) the final time, Tf, for the optimization
# 5) the initial time, IC, for optimization

python3 main.py [0,1,2] run1 ${dir} | tee run-outputs/${dir}/log.txt 
# python3 main.py [0,1,2] run2 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run3 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run4 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run5 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run6 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run7 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run8 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run9 ${dir} | tee run-outputs/${dir}/log.txt
# python3 main.py [0,1,2] run10 ${dir} | tee run-outputs/${dir}/log.txt 

# example using all arguments
# python3 main.py [0] runX run-outputs/${dir} 2 0 | tee -a run-outputs/${dir}/log.txt

