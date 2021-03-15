#!/bin/bash

# run one of or both preprocess.py and main.py and write outputs to specified directory

dir1='X1'
dir2='X2'
dir3='X3'
dir4='X4'
dir5='X5'
mkdir run-outputs/${dir1}
mkdir run-outputs/${dir2}
mkdir run-outputs/${dir3}
mkdir run-outputs/${dir4}
mkdir run-outputs/${dir5}


preprocess=true
main=true

if [ "$preprocess" = true ] ; then
	# save output of preprocess to directory specified in first argument
	# arguments: 
	# 1) directory to save outputs to 
	# 2) timepoints to keep as a list
	python3 preprocess.py run-outputs/${dir1} [2,4,6,8,10] | tee run-outputs/${dir1}/preprocess_log.txt
	python3 preprocess.py run-outputs/${dir2} [2,5,8] | tee run-outputs/${dir2}/preprocess_log.txt
	python3 preprocess.py run-outputs/${dir3} [3,5,7,9] | tee run-outputs/${dir3}/preprocess_log.txt
	python3 preprocess.py run-outputs/${dir4} [3,6,9] | tee run-outputs/${dir4}/preprocess_log.txt
	python3 preprocess.py run-outputs/${dir5} [2,6,10] | tee run-outputs/${dir5}/preprocess_log.txt

fi

if [ "$main" = true ] ; then
	# arguments for main.py are: 
	# 1) replicates to use
	# 2) when saving the output (compressed) pickle, what should be the unique tag over the various runs
	# 3) the directory to save the outputs to 
	# 4) the final time, Tf, for the optimization
	# 5) the initial time, IC, for optimization

	python3 main.py [0,1,2] run1 ${dir1} | tee -a run-outputs/${dir1}/log.txt 
	python3 main.py [0,1,2] run2 ${dir1} | tee -a run-outputs/${dir1}/log.txt
	python3 main.py [0,1,2] run3 ${dir1} | tee -a run-outputs/${dir1}/log.txt
	python3 main.py [0,1,2] run1 ${dir2} | tee -a run-outputs/${dir2}/log.txt 
	python3 main.py [0,1,2] run2 ${dir2} | tee -a run-outputs/${dir2}/log.txt
	python3 main.py [0,1,2] run3 ${dir2} | tee -a run-outputs/${dir2}/log.txt
	python3 main.py [0,1,2] run1 ${dir3} | tee -a run-outputs/${dir3}/log.txt 
	python3 main.py [0,1,2] run2 ${dir3} | tee -a run-outputs/${dir3}/log.txt
	python3 main.py [0,1,2] run3 ${dir3} | tee -a run-outputs/${dir3}/log.txt
	python3 main.py [0,1,2] run1 ${dir4} | tee -a run-outputs/${dir4}/log.txt 
	python3 main.py [0,1,2] run2 ${dir4} | tee -a run-outputs/${dir4}/log.txt
	python3 main.py [0,1,2] run3 ${dir4} | tee -a run-outputs/${dir4}/log.txt
	python3 main.py [0,1,2] run1 ${dir5} | tee -a run-outputs/${dir5}/log.txt 
	python3 main.py [0,1,2] run2 ${dir5} | tee -a run-outputs/${dir5}/log.txt
	python3 main.py [0,1,2] run3 ${dir5} | tee -a run-outputs/${dir5}/log.txt
	
	# example using all arguments
	# python3 main.py [0] runX run-outputs/${dir} 2 0 | tee -a run-outputs/${dir}/log.txt
fi






