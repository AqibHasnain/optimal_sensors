#!/bin/bash

rm log.txt
# the inputs to main.py are the replicates to run the optimizations with. 
python3 main.py [0] | tee -a log.txt
python3 main.py [1] | tee -a log.txt
python3 main.py [2] | tee -a log.txt
python3 main.py [0,1] | tee -a log.txt
python3 main.py [0,2] | tee -a log.txt
python3 main.py [1,2] | tee -a log.txt
python3 main.py [0,1,2] | tee -a log.txt