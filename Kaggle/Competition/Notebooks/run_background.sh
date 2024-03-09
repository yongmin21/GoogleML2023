#!/bin/bash

if [ -z $1 ];
then
    echo "No parameter passed"
else
    echo "Run $1 Background."
    nohup jupyter nbconvert --to notebook --execute $1 > output.log 2>&1 &

fi