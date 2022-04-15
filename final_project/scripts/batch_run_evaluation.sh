#!/bin/bash

for d in $(find . -maxdepth 6 -type d)
do
  # Check if the config file exists
  FILE=$d/turing_evaluation_config.yaml

  if test -f "$FILE"; then
    pushd $d
    echo "Executing sbatch script."
    sbatch sbatch_run_evaluation.sh
    popd

    # Add new lines
    echo ""
  fi
done

