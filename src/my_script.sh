#!/bin/bash

i=-1
alpha=0
number=20
del=$(( 1000/number ))  # alpha scaled from 0 to 1000
#del=50
#number=1

# integers 0 to 100 are used instead between 0 to 1

while [ $i -lt $number ]
do 
    make all ALPHA=$alpha
    echo $alpha
    # echo $del
    (( alpha += del ))
    (( i++ ))
done

#number=20

python3 plot_optimal_front.py $number #(number argument for number of divisions )
