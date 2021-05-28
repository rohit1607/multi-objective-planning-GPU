#!/bin/bash

i=-1
alpha=0
number=10
del=$(( 100/number ))

# integers 0 to 100 are used instead between 0 to 1

while [ $i -lt $number ]
do 
    make all ALPHA=$alpha
    echo $alpha
    # echo $del
    (( alpha += del ))
    (( i++ ))
done

python3 plot_optimal_front.py $number #(number argument for number of divisions )
