#!/usr/bin/env bash

# download data and convert to .json format

if [ ! -f "data/all_data/data.json" ]; then
	echo "Please download the adult dataset : https://www.kaggle.com/wenruliu/adult-income-dataset , then put adult.csv in data/adult/data and run the adult_to_json.py script to generate data.json."
	exit 1
fi


NAME="adult" # name of the dataset, equivalent to directory name
 

cd ../utils

./preprocess.sh --name $NAME $@

cd ../$NAME