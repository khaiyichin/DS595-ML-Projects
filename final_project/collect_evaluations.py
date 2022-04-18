import util_modules as utm
import os
import pandas as pd
import numpy as np

if __name__ == "__main__":

    args = utm.parse_arguments()

    os.chdir(args.working_dir) # change to desired working directory

    # Iterate each subdirectory until you find a eval_analytics*.csv
    eval_csv_files = []

    for root, dirs, files in os.walk(os.getcwd()):

        # get all the paths to the csv files
        for name in files:
            if name.endswith(".csv") and name[:14] == "eval_analytics":
                eval_csv_files.append(os.path.join(root, name))

    eval_csv_files_organized = {}

    # Read through each CSV file's accuracy and its attributes
    for csv_file in eval_csv_files:
        attributes = []
        
        # Identify epoch number
        if '/e5000' in csv_file: attributes.append('e5k')
        else: attributes.append('e1k')
        
        # Identify learning rate
        if '/lr005' in csv_file: attributes.append('lr5')
        else: attributes.append('lr1')

        # Identify optimizer
        if '/adam' in csv_file: attributes.append('adam')
        else: attributes.append('sgd')

        # Identify batch number
        if '/b50' in csv_file: attributes.append('b50')
        else: attributes.append('b100')

        df = pd.read_csv(csv_file)

        eval_csv_files_organized[csv_file] = {"attributes": attributes,
                                              "acc_train": df['acc_train'].values,
                                              "acc_test": df['acc_test'].values}

    # Create multicolumn and multirow dataframes
    row_arr = [["e5k", "e1k"], ["lr5", "lr1"], ["k" + str(i+1) for i in range(4)]]
    col_arr = [["sgd", "adam"], ["b50", "b100"]]

    row_ind = pd.MultiIndex.from_product(row_arr)
    col_ind = pd.MultiIndex.from_product(col_arr)

    df_train = pd.DataFrame(np.zeros([16, 4]), index=row_ind, columns=col_ind)
    df_test = pd.DataFrame(np.zeros([16, 4]), index=row_ind, columns=col_ind)

    df_train = df_train.sort_index()
    df_test = df_test.sort_index()

    # Find specific row
    for e in row_arr[0]:
        for lr in row_arr[1]:

            # Find specific column
            for opt in col_arr[0]:
                for b in col_arr[1]:

                    # Identify the key-value pair based on those rows and columns
                    key = [ k for k,v in eval_csv_files_organized.items() if v['attributes'] == [e,lr,opt,b] ][0] # key containing the attribute that matches the rows and cols
                    
                    df_train.loc[(e,lr), (opt,b)] = eval_csv_files_organized[key]["acc_train"]
                    df_test.loc[(e,lr), (opt,b)] = eval_csv_files_organized[key]["acc_test"]

    # Write the data to CSV
    df_train.to_csv('train_eval.csv')
    df_test.to_csv('test_eval.csv')