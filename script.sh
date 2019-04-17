#!/bin/bash
IFS='_' read -ra sep1 <<< "$1"

model=${sep1[0]}
params=${sep1[1]}
tp=${sep1[2]}
numbers="1 3 5"

for i in $numbers
do
echo "Number: $i"
done	

echo $model
echo $params

echo "Hello World"


echo $tp
