#!/bin/bash
  
for d in $(find . -maxdepth 6 -type d)
do
  # Check if the config file exists
  FILE=$d/turing_train_config.yaml
  MODEL=$d/*pth

  if test -f "$FILE"; then
    if ! ls $d/*.pth &>/dev/null; then
        echo "No model files detected here at $d"
    fi

    # Add new lines
  fi
done


