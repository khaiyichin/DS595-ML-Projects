#!/bin/bash

for d in $(find . -maxdepth 6 -type d)
do
  # Check if the config file exists
  FILE=$d/turing_train_config.yaml

  if test -f "$FILE"; then

    # Copy and rename the file
    cp $FILE $d/turing_evaluation_config.yaml

    # Create array to store all model filenames
    readarray arr < <(find $d -type f -name "*pth" -print0 | sort -z | xargs -0 basename -a)

    m1=$(echo ${arr[0]})
    m2=$(echo ${arr[1]})
    m3=$(echo ${arr[2]})
    m4=$(echo ${arr[3]})

    # Add lines to newly created evaluation config file
    sed -i "s|paths:|paths:\n  modelInputFolder: \"./\"|" $d/turing_evaluation_config.yaml
    sed -i "s|  inputFolder: \"/home/kchin/DS595-ML-Projects/final_project/data/\"|  dataInputFolder: \"/home/kchin/DS595-ML-Projects/final_project/data/\"|g" $d/turing_evaluation_config.yaml
    sed -i "s|    - \"041322_1706_test_4-fold-4.csv\"|    - \"041322_1706_test_4-fold-4.csv\"\n  models:|" $d/turing_evaluation_config.yaml
    sed -i "s|  models:|  models:\n    - \"$m1\"|" $d/turing_evaluation_config.yaml
    sed -i "s|    - \"$m1\"|    - \"$m1\"\n    - \"$m2\"|" $d/turing_evaluation_config.yaml
    sed -i "s|    - \"$m2\"|    - \"$m2\"\n    - \"$m3\"|" $d/turing_evaluation_config.yaml
    sed -i "s|    - \"$m3\"|    - \"$m3\"\n    - \"$m4\"|" $d/turing_evaluation_config.yaml
    sed -i "s|  analyticsSaveName: \"train_analytics.csv\"|  analyticsSaveName: \"eval_analytics.csv\"|g" $d/turing_evaluation_config.yaml

  fi
done
