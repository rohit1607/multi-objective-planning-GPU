#!/bin/bash

i=-1
alpha=550
number=9
#del=$(( 1000/number ))
del=50
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

number=20

python3 plot_optimal_front.py $number #(number argument for number of divisions )
