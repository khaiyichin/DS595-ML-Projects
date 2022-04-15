#!/bin/bash

for d in $(find . -maxdepth 6 -type d)
do
  # Check if the config file exists
  FILE=$d/turing_evaluation_config.yaml

  if test -f "$FILE"; then
    echo -e "Copying sbatch_run_evaluation.sh to $d"
    cp sbatch_run_evaluation.sh $d/
    
    # Add new lines
    echo ""
  fi
done

