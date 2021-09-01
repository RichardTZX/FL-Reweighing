#!/usr/bin/env bash

# download data and convert to .json format

if [ ! -f "data/all_data/data.json" ]; then
	echo "Please download the COMPAS dataset : https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv , then put compas-scores-two-years.csv in data/compas/data and run the compas_to_json.py script to generate data.json."
	exit 1
fi


NAME="compas" # name of the dataset, equivalent to directory name
 

cd ../utils

./preprocess.sh --name $NAME $@

cd ../$NAME

if [ -f "data/all_data/gan.json" ]; then
	echo "GAN Setup"
	python gan_data.py
fi
