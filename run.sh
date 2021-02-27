#!/bin/bash

rm log.txt

# the inputs to main.py are the replicates to run the optimizations with. 
# python3 main.py [0,1,2] run0 #| tee -a log.txt
python3 main.py [0,1,2] run1 | tee -a log.txt
python3 main.py [0,1,2] run2 | tee -a log.txt
python3 main.py [0,1,2] run3 | tee -a log.txt
python3 main.py [0,1,2] run4 | tee -a log.txt
python3 main.py [0,1,2] run5 | tee -a log.txt
python3 main.py [0,1,2] run6 | tee -a log.txt
python3 main.py [0,1,2] run7 | tee -a log.txt
python3 main.py [0,1,2] run8 | tee -a log.txt
python3 main.py [0,1,2] run9 | tee -a log.txt
python3 main.py [0,1,2] run10 | tee -a log.txt 
python3 main.py [0,1,2] run11 | tee - log.txt

# first argument is the reps to use and the second is the horizon for optimization 
# python3 main.py [0] 2 0
# python3 main.py [0] 3 0
# python3 main.py [0] 4 0
# python3 main.py [0] 5 0
# python3 main.py [0] 6 0
# python3 main.py [0] 7 0
# python3 main.py [0] 8 0
# python3 main.py [0] 9 0
# python3 main.py [0] 4 2
# python3 main.py [0] 6 4
# python3 main.py [0] 9 6
