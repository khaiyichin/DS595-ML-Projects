#!/bin/bash

for d in $(find . -maxdepth 6 -type d)
do
  # Check if the config file exists
  FILE=$d/turing_evaluation_config.yaml

  if test -f "$FILE"; then

    # Copy and rename the file
    cp $FILE $d/turing_validation_config.yaml

    # Remove training and testing data information
    sed -i "/trainingData:/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_train_4-fold-1.csv\"/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_train_4-fold-2.csv\"/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_train_4-fold-3.csv\"/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_train_4-fold-4.csv\"/d" $d/turing_validation_config.yaml
    sed -i "/testingData:/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_test_4-fold-1.csv\"/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_test_4-fold-2.csv\"/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_test_4-fold-3.csv\"/d" $d/turing_validation_config.yaml
    sed -i "/- \"041822_1124_test_4-fold-4.csv\"/d" $d/turing_validation_config.yaml

    # Remove save names for losses and model files
    sed -i "/modelSaveName: \"model.pth\"/d" $d/turing_validation_config.yaml
    sed -i "/lossesSaveName: \"losses.csv\"/d" $d/turing_validation_config.yaml
    
        # Insert line for data files information
    sed -i "s|  outputFolder: \"./\"|  outputFolder: \"./\"\n  dataFiles: \"041822_1124_validation_data_enc.csv\"|" $d/turing_validation_config.yaml

  fi
done