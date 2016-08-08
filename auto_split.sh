#!/bin/bash

# split datafile into 10 parts
file=$1
N=$2
out_name="${file%.*}"

head -n 1 $file > header.csv

for (( i=0; i<$N; i++))
do
    echo "split " $i
    cat header.csv > $out_name.$i.csv
    awk -v ii=$i -v nn=$N 'NR>1 && NR%nn==ii {print}' $file >> $out_name.$i.csv
done

