import util_modules as utm
import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics

"""
This script isn't well written because of time constraints.

What it does here is to use a validation data file (whose filename is passed as an argument in calling this python
Python script) and create ensemble neural networks to make predictions on it. The ensemble NNs live in multiple

"""

if __name__ == "__main__":

    args = utm.parse_arguments()

    os.chdir(args.working_dir) # change to desired working directory

    working_dir = os.getcwd()

    # Load validation file
    csv_file = args.v

    validation_df = pd.read_csv(csv_file)

    # Separate into inputs and outputs
    feature_validation_df = validation_df.iloc[:,:-5].copy()
    label_validation_df = validation_df.iloc[:,-5:].copy()

    # Obtain actual labels
    actual_y = np.argmax(label_validation_df.values, axis=1).tolist()

    # Create multicolumn and multirow dataframes
    row_arr = [["e1k", "e5k"], ["lr5", "lr1"]]
    col_arr = [["sgd", "adam"], ["b50", "b100"]]

    row_ind = pd.MultiIndex.from_product(row_arr)
    col_ind = pd.MultiIndex.from_product(col_arr)

    validation_results_df = pd.DataFrame(np.zeros([4, 4]), index=row_ind, columns=col_ind)

    validation_results_df = validation_results_df.sort_index()

    # Iterate through each subdirectory until you find a validation config file
    for root, dirs, files in os.walk(working_dir):

        if "turing_validation_config.yaml" in files:

            # Change to directory
            os.chdir(root)

            # Modify the args variable to point to the config file (i.e., create ensembles)
            args.config = "turing_validation_config.yaml"

            nn_ensemble = utm.NNEnsemble(args)

            # Generate predictions using loaded ensemble model
            predicted_y = nn_ensemble.make_prediction(feature_validation_df)
        
            # Determine attributes for storing into dataframe
            # Identify epoch number
            if "/e5000" in root: e = "e5k"
            else: e = "e1k"
            
            # Identify learning rate
            if "/lr005" in root: lr = "lr5"
            else: lr = "lr1"

            # Identify optimizer
            if "/adam" in root: opt = "adam"
            else: opt = "sgd"

            # Identify batch number
            if "/b50" in root: b = "b50"
            else: b = "b100"

            # Store accuracy score
            validation_results_df.loc[(e,lr), (opt,b)] = skmetrics.accuracy_score(actual_y, predicted_y)

    # Write the data to CSV
    os.chdir(args.working_dir) # change to back to desired working directory

    validation_results_df.to_csv("validation_analytics.csv")
