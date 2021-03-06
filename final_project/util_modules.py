import pandas as pd
import numpy as np
import sklearn
import yaml
import torch
import os
import sys
import sklearn.metrics as skmetrics
from datetime import datetime
from enum import Enum
import argparse

def parse_arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(description="DS 595-ML Final Project")
    parser.add_argument('working_dir', type=str, help='working directory to run training/evaluation/validation')
    parser.add_argument('--config', type=str, help='path to configuration file (required for training and evaluation)')
    parser.add_argument('--no_save', action="store_true", help='path to configuration file')
    parser.add_argument('-v', type=str, help='run validation flag with the desired validation CSV file')

    return parser.parse_args()

def normalize(pandas_df, norm_type):
    """Normalize dataframe values.

    Args:
        pandas_df: Pandas dataframe.
        norm_type: 'full' for normalization using all the value in the dataframe,
                   'freq' for normalization along each column.

    Returns:
        Pandas dataframe with normalized values.
    """

    if norm_type == 'full': # across the entire dataset
        min_val = pandas_df.min().min()
        max_val = pandas_df.max().max()
        
        return (pandas_df - min_val) / (max_val - min_val)

    elif norm_type == 'freq': # across each frequency (feature)
        min_vals = pandas_df.min()
        max_vals = pandas_df.max()
        diff = max_vals - min_vals

        return pandas_df.sub(min_vals.values, axis=1).div(diff.values, axis=1)

    else:
        raise Exception('Unknown normalization type string.')

def standardize(pandas_df):
    """Standardize dataframe values using Z-score standardization.

    Args:
        pandas_df: Pandas dataframe.

    Returns:
        Pandas dataframe with standardized values.
    """
    return pandas_df.sub(pandas_df.mean(0), axis=1) / pandas_df.std(0)

class DynamicNetwork(torch.nn.Module):
    """Dynamic neural network class.

    Adapted from: https://github.com/jesus-333/Dynamic-PyTorch-Net
    """

    def __init__(self, parameters, verbose=True):

        super().__init__()

        self.verbose = verbose

        # Assign fully connected layers
        if ("num_fc_layers" in parameters.keys()):
            self.num_fc_layers = int(parameters["num_fc_layers"])

            if (self.verbose): print("Number of fully connected layers:\t\t{}".format(self.num_fc_layers))

            # Assign neurons in fully connected layers
            if ("neuron_num_lst" in parameters.keys()):
                neuron_num_lst = parameters["neuron_num_lst"]

                if (len(neuron_num_lst) != self.num_fc_layers + 1): raise Exception("The number of neurons doesn't match the number of fully connected layers + 1.")

                # Break neuron numbers into list of tuples
                if (self.num_fc_layers > 1): self.neuron_num_lst = self.convert_arr_to_tuple_lst(neuron_num_lst)

                if verbose:
                    print("Paired number of neurons:\t\t\t{}".format(self.neuron_num_lst))
                    sys.stdout.flush()

            else:
                raise Exception("neuron_num_lst parameter is not set!")

        else:
            raise Exception("num_fc_layers parameter is not set!")

        # Assign activation functions
        if ("activation_str_lst" in parameters.keys()):

            self.activation_str_lst = parameters["activation_str_lst"]

            if len(self.activation_str_lst) != self.num_fc_layers - 1:
                raise Exception("The number of activations doesn't match the number of fully connected layers - 1.")

            self.activations = self.get_activations(parameters["activation_str_lst"])

            if (self.verbose):
                print("Activation functions:\t\t\t\t{}".format(str(self.activation_str_lst)))
                sys.stdout.flush()

        else:
            raise Exception("activation_str_lst parameter is not set!")

        # Assign batch normalization layers (optional)
        if ("batch_norm_lst" in parameters.keys()):

            self.batch_norm_lst = parameters["batch_norm_lst"]

            if len(self.batch_norm_lst) != self.num_fc_layers - 1:
                raise Exception("The number of batch normalizations doesn't match the number of fully connected layers - 1.")

            if (self.verbose):
                print("Batch Normalizations:\t\t\t\t{}".format(self.batch_norm_lst))
                sys.stdout.flush()

        else:
            self.batch_norm_lst = []

        # Assign dropout layers (optional)
        if ("dropout_rate_lst" in parameters.keys()):

            self.dropout_rate_lst = parameters["dropout_rate_lst"]

            if len(self.dropout_rate_lst) != self.num_fc_layers - 1:
                raise Exception("The number of dropout rates doesn't match the number of fully connected layers - 1.")

            if (self.verbose):
                print("Dropout rates:\t\t\t\t{}".format(self.dropout_rate_lst))
                sys.stdout.flush()

        else:
            self.dropout_rate_lst = []

        self.setup_network()

        if self.verbose:
            print("Network summary:\n{}".format(self))
            sys.stdout.flush()

    def setup_network(self):

        temp_lst = []

        for ind in range(self.num_fc_layers - 1):

            # Create fully connected layer
            temp_linear_layer = torch.nn.Linear(self.neuron_num_lst[ind][0], self.neuron_num_lst[ind][1])
            temp_lst.append(temp_linear_layer)

            # Create batch normalization layer
            if self.batch_norm_lst:
                temp_lst.append(torch.nn.BatchNorm1d(self.neuron_num_lst[ind][1]))

            # Create dropout layer
            if self.dropout_rate_lst:
                temp_lst.append(torch.nn.Dropout(self.dropout_rate_lst[ind]))

            # Create activation layer
            if self.activations:
                temp_lst.append(self.activations[ind])

        # Create final output layer
        temp_lst.append(torch.nn.Linear(self.neuron_num_lst[-1][0], self.neuron_num_lst[-1][1]))

        # Finalize network setup
        self.network = torch.nn.Sequential(*temp_lst)

    def convert_arr_to_tuple_lst(self, arr):
        """
        Convert an array (or a list) of element in a list of tuple where each element is a tuple with two sequential element of the original arr/list
        Parameters
        ----------
        arr : numpy array/list
        Returns
        -------
        tuple_lst. List of tuple
            Given the input arr = [a, b, c, d ...] the tuple_lst will be [(a, b), (b, c), (c, d) ...]
        """

        tuple_lst = []

        for i in range(len(arr) - 1):
            tmp_tuple = (arr[i], arr[i + 1])

            tuple_lst.append(tmp_tuple)

        return tuple_lst

    def get_activations(self, activation_lst_str):

        activations_lst = []

        for act_str in activation_lst_str:

            if act_str.lower() == "sigmoid":
                activations_lst.append(torch.nn.Sigmoid())

            elif act_str.lower() == "relu":
                activations_lst.append(torch.nn.ReLU())

            elif act_str.lower() == "leakyrelu":
                activations_lst.append(torch.nn.LeakyReLU())

            elif act_str.lower() == "softmax":
                activations_lst.append(torch.nn.Softmax())

            else:
                raise Exception("The activation " + act_str + " is not supported currently.")

        return activations_lst

    def forward(self, x):

        return self.network(x)

class NNUtils():

    class UtilType(Enum):
        TRAINER = 1
        EVALUATOR = 2
        ENSEMBLE = 3

    def __init__(self, args, verbose=True):

        # Initialize class variables
        self.verbose = verbose

        self.no_save = args.no_save

        self.networks = []
        self.optimizers = []

        self.preprocessing_params = []

        self.train_loaders = []

        self.training_data = {"filenames": [], "input": [], "output": []}
        self.testing_data = {"filenames": [], "input": [], "output": []}

    def standardize(self, pandas_df, params=None):
        """Standardize dataframe values using Z-score standardization.

        Args:
            pandas_df: Pandas dataframe.
            params: Precomputed parameters.

        Returns:
            Tuple of pandas dataframe with standardized values and (mean, standard deviation) tuple.
        """
        if params:
            return pandas_df.sub(params[0], axis=1) / params[1], (params[0], params[1])
        else:
            return pandas_df.sub(pandas_df.mean(0), axis=1) / pandas_df.std(0), (pandas_df.mean(0), pandas_df.std(0) )

    def normalize(self, pandas_df, params=None):
        """Normalize dataframe values.

        Args:
            pandas_df: Pandas dataframe.
            norm_type: 'full' for normalization using all the value in the dataframe,
                    'freq' for normalization along each column.
            params: Precomputed parameters.

        Returns:
            Tuple of pandas dataframe with normalized values and (min, max) tuple.
        """

        if self.parameters["normalization_type"] == 'full': # across the entire dataset

            if params:
                min_val, max_val = params
            else:
                min_val = pandas_df.min().min()
                max_val = pandas_df.max().max()

            return (pandas_df - min_val) / (max_val - min_val), (min_val, max_val)

        elif self.parameters["normalization_type"] == 'freq': # across each frequency (feature)

            if params:
                min_vals, max_vals = params
            else:
                min_vals = pandas_df.min()
                max_vals = pandas_df.max()

            diff = max_vals - min_vals

            return pandas_df.sub(min_vals.values, axis=1).div(diff.values, axis=1), (min_vals, max_vals)

        else:
            raise Exception('Unknown normalization type string.')

    def setup_device(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            if torch.cuda.is_available():
                print("Device:\t\t\t{}".format(self.device), end='')
                print('\t\t\t', torch.cuda.get_device_name(0))

            else: print("Device:\t\t\t{}".format(self.device))

            sys.stdout.flush()

    def setup_data_loaders(self, inputs, outputs):

        dataset = torch.utils.data.TensorDataset(inputs, outputs)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.parameters["batch_size"],
                                                  shuffle=True)

        return data_loader

    def read_yaml_config(self, config_file):
        yaml_config = None

        # Read YAML configuration file
        with open(config_file, 'r') as fopen:
            try:
                yaml_config = yaml.safe_load(fopen)
            except yaml.YAMLError as exception:
                print(exception)

        # Display input arguments
        print('\n\t' + '='*50 + ' Inputs ' + '='*50 + '\n')
        print('\tConfig file\t: ' + os.path.abspath(config_file))
        print('\n\t' + '='*48 + ' End Inputs ' + '='*48  + '\n')

        # Displayed processed arguments
        print('\n\t\t\t\t\t' + '='*15 + ' Processed Config ' + '='*15 + '\n')
        print('\t\t\t\t\t\t', end='')
        for line in yaml.dump(yaml_config, indent=4, default_flow_style=False):
            if line == '\n':
                print(line, '\t\t\t\t\t\t',  end='')
            else:
                print(line, end='')
        print('\r', end='') # reset the cursor for print
        print('\n\t\t\t\t\t' + '='*13 + ' End Processed Config ' + '='*13  + '\n')

        self.parameters = {}

        # Assign common parameters

        # Hyperparameters
        self.parameters["k_fold_cv"] = int(yaml_config["kFoldCrossValidation"])
        self.parameters["num_fc_layers"] = yaml_config["numberOfFullyConnectedLayers"]
        self.parameters["neuron_num_lst"] = yaml_config["neuronNumberList"]
        self.parameters["activation_str_lst"] = yaml_config["activationFunctionStringList"]
        self.parameters["batch_size"] = int(yaml_config["batchSize"])
        self.parameters["learning_rate"] = float(yaml_config["learningRate"])

        # Preprocessors
        self.parameters["standardize"] = yaml_config["standardize"]
        self.parameters["normalize"] = yaml_config["normalize"]["bool"]
        self.parameters["normalization_type"] = yaml_config["normalize"]["type"]

        if self.verbose:
            if self.parameters["standardize"]:
                print("Data will be z-score standardized (mean and standard deviation).")

            elif self.parameters["normalize"]:
                print("Data will be normalized (minimum and maximum values).")

        if self.parameters["standardize"] and self.parameters["normalize"]:
            raise Exception("Only one type of preprocessor can be used at a time.")

        # Path parameters
        self.parameters["data_input_folder"] = yaml_config["paths"]["dataInputFolder"]
        self.parameters["output_folder"] = yaml_config["paths"]["outputFolder"]

        # Assign specific parameters
        if self.util_type == self.UtilType.TRAINER: # trainer params

            # Assign target number of epochs to run
            self.parameters["target_epochs"] = int(yaml_config["targetEpochs"])

            # Assign optimizer
            self.parameters["optimizer_str"] = yaml_config["optimizerString"]

            self.parameters["model_save_name"] = yaml_config["paths"]["modelSaveName"]
            self.parameters["losses_save_name"] = yaml_config["paths"]["lossesSaveName"]

            self.parameters["training_data"] = yaml_config["paths"]["trainingData"]
            self.parameters["analytics_save_name"] = yaml_config["paths"]["analyticsSaveName"]

            # Assign optional parameters
            try:
                self.parameters["batch_norm_lst"] = yaml_config["optional"]["batchNormalizationBooleanList"]
            except Exception as e:
                if self.verbose: print("Not setting batch normalization layers.")

            try:
                self.parameters["dropout_rate_lst"] = yaml_config["optional"]["dropoutRateList"]
            except Exception as e:
                if self.verbose: print("Not setting dropout layers.")

        elif self.util_type == self.UtilType.EVALUATOR: # evaluator params
            self.parameters["model_input_folder"] = yaml_config["paths"]["modelInputFolder"]
            self.parameters["training_data"] = yaml_config["paths"]["trainingData"]
            self.parameters["testing_data"] = yaml_config["paths"]["testingData"]
            self.parameters["models"] = yaml_config["paths"]["models"]
            self.parameters["analytics_save_name"] = yaml_config["paths"]["analyticsSaveName"]

        elif self.util_type == self.UtilType.ENSEMBLE: # ensemble params
            self.parameters["model_input_folder"] = yaml_config["paths"]["modelInputFolder"]
            self.parameters["data_files"] = yaml_config["paths"]["dataFiles"]
            self.parameters["models"] = yaml_config["paths"]["models"]

        else:
            raise Exception("Unknown caller type for read_yaml_config function.")

        sys.stdout.flush() # flush print statements

    def setup_networks(self, loaded_state_dicts=None):

        # Setup for k-fold cross validation models
        for i in range(self.parameters["k_fold_cv"]):

            if self.verbose:
                print("\nSetting up {0} out of {1} networks:\n".format(i+1, self.parameters["k_fold_cv"]))
                sys.stdout.flush()

            nn = DynamicNetwork(self.parameters, self.verbose).to(self.device)

            if self.util_type == self.UtilType.TRAINER:
                opt = self.get_optimizer(self.parameters["optimizer_str"], nn)
                self.optimizers.append(opt)

                if self.verbose:
                    print("Optimizer:\n{}\n".format(opt))
                    sys.stdout.flush()

            elif self.util_type == self.UtilType.EVALUATOR or self.util_type == self.UtilType.ENSEMBLE:
                nn.load_state_dict(loaded_state_dicts[i]["model_state_dict"])
                self.preprocessing_params = loaded_state_dicts[i]["preprocessing_params"]
                nn.eval()

            else:
                raise Exception("Unknown caller type for the setup_networks function.")

            self.networks.append(nn)

    def import_models(self):

        loaded_state_dicts = []

        # Iterate through all k models
        for i in range(self.parameters["k_fold_cv"]):

            loaded_state_dicts.append(
                torch.load( os.path.join(self.parameters["model_input_folder"],
                                         self.parameters["models"][i]), map_location=self.device ) )

        self.setup_networks(loaded_state_dicts)

    def get_optimizer(self, opt_str, network):

        # TODO: remove opt_str argument (since we can just use self.parameters)

        if opt_str.lower() == "sgd":
            return torch.optim.SGD(network.parameters(), lr=self.parameters["learning_rate"]) # using no momentum

        elif opt_str.lower() == "adam":
            return torch.optim.Adam(network.parameters(), lr=self.parameters["learning_rate"])

        else:
            raise Exception("The optimizer " + opt_str + " is not supported currently.")

    def read_and_store_data(self):

        # Store training data and set up data loaders
        if self.util_type == self.UtilType.ENSEMBLE:

            pass

        else: # both trainers and evaluators

            # Store training data
            for ind, csv_file in enumerate(self.parameters["training_data"]):
                pd_data = pd.read_csv(os.path.join(self.parameters["data_input_folder"], csv_file))

                # Check whether to preprocess data
                if self.parameters["standardize"]:

                    if self.util_type == self.UtilType.EVALUATOR:
                        pd_data_processed, _ = self.standardize(pd_data.drop(columns=pd_data.columns[-5:]),
                                                                self.preprocessing_params[ind])
                    else:
                        pd_data_processed, params = self.standardize(pd_data.drop(columns=pd_data.columns[-5:]))
                        self.preprocessing_params.append(params)

                elif self.parameters["normalize"]:

                    if self.util_type == self.UtilType.EVALUATOR:
                        pd_data_processed, _ = self.normalize(pd_data.drop(columns=pd_data.columns[-5:]),
                                                                    self.preprocessing_params[ind])
                    else:
                        pd_data_processed, params = self.normalize(pd_data.drop(columns=pd_data.columns[-5:]))
                        self.preprocessing_params.append(params)

                # Split 5 classes with one hot encoding => 5 columns at the end of the dataframe (if not preprocessed already)
                if self.preprocessing_params:
                    input = torch.tensor(pd_data_processed.values, dtype=torch.float32)
                else:
                    input = torch.tensor(pd_data.iloc[:, :-5].values, dtype=torch.float32)

                output = torch.tensor(pd_data.iloc[:,-5:].values, dtype=torch.float32)

                self.train_loaders.append(self.setup_data_loaders(input, output))

                self.training_data["filenames"].append(csv_file)
                self.training_data["input"].append(input)
                self.training_data["output"].append(output)

            # Store testing data (only applicable to evaluators)
            if self.util_type == self.UtilType.EVALUATOR:

                for ind, csv_file in enumerate(self.parameters["testing_data"]):
                    pd_data = pd.read_csv(os.path.join(self.parameters["data_input_folder"], csv_file))

                    # Apply preprocessing from training data
                    if self.parameters["standardize"]:
                        pd_data_processed, _ = self.standardize(pd_data.drop(columns=pd_data.columns[-5:]),
                                                                self.preprocessing_params[ind])

                    elif self.parameters["normalize"]:
                        pd_data_processed, _ = self.normalize(pd_data.drop(columns=pd_data.columns[-5:]),
                                                                self.preprocessing_params[ind])

                    # Split 5 classes with one hot encoding => 5 columns at the end of the dataframe
                    if self.preprocessing_params:
                        input = torch.tensor(pd_data_processed.values, dtype=torch.float32)
                    else:
                        input = torch.tensor(pd_data.iloc[:, :-5].values, dtype=torch.float32)

                    output = torch.tensor(pd_data.iloc[:,-5:].values, dtype=torch.float32)

                    self.testing_data["filenames"].append(csv_file)
                    self.testing_data["input"].append(input)
                    self.testing_data["output"].append(output)

class NNTrainer(NNUtils):

    def __init__(self, args, verbose=True):

        super().__init__(args, verbose)

        self.util_type = self.UtilType.TRAINER # assign identity to this class

        # Parse configuration file
        self.read_yaml_config(args.config)

        # Store data
        self.read_and_store_data()

        # Setup device
        self.setup_device()

        # Instantiate k neural networks and optimizers based on parameters (k-fold cross validation)
        self.setup_networks()

        # Assign loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Set up loggers
        self.losses_log = []
        self.analytics = []

    def run(self):

        # Run k-fold cross validation
        self.k_fold_cross_validation()

        # Save network and optimizer and analytics
        save_time = datetime.now()

        if not self.no_save:
            self.save_models(save_time)
            self.save_analytics(save_time)

    def k_fold_cross_validation(self):

        for i in range(self.parameters["k_fold_cv"]):
            self.train(self.train_loaders[i], self.networks[i], self.optimizers[i])

    def train(self, train_loader, network, optimizer):

        losses = []
        accuracy = []
        mac_precision = []
        mic_precision = []
        mac_recall = []
        mic_recall = []
        mac_f1 = []
        mic_f1 = []

        for i in range(1, self.parameters["target_epochs"]+1):
            y_pred_lst = []
            y_act_lst = []

            for X, y in train_loader:

                # Make predictions
                predictions = network(X.to(self.device))

                # Find the loss and then backpropagate it
                optimizer.zero_grad() # set gradients to zero
                loss = self.criterion(predictions, y.to(self.device).argmax(dim=1))
                loss.backward()

                # Update weights
                optimizer.step()

                # Obtain predicted and actual labels
                _, y_pred_ind = torch.max(torch.log_softmax(predictions.detach(), dim=1), dim=1)
                _, y_act_ind = torch.max(y, dim=1)

                y_pred_lst.extend(y_pred_ind.tolist())
                y_act_lst.extend(y_act_ind.tolist())

            # Store analytics for current epoch
            losses.append( float(loss.item())/self.parameters["batch_size"] )

            if self.verbose and i % int(self.parameters["target_epochs"]/10) == 0:
                print("\tEpoch {0}: Loss = {1}".format(i, losses[-1]))
                sys.stdout.flush()

            accuracy.append( skmetrics.accuracy_score(y_act_lst, y_pred_lst) )

            mac_precision.append( skmetrics.precision_score(y_act_lst, y_pred_lst, average='macro', zero_division=0) )
            mic_precision.append( skmetrics.precision_score(y_act_lst, y_pred_lst, average='micro', zero_division=0) )

            mac_recall.append( skmetrics.recall_score(y_act_lst, y_pred_lst, average='macro') )
            mic_recall.append( skmetrics.recall_score(y_act_lst, y_pred_lst, average='micro') )

            mac_f1.append( skmetrics.f1_score(y_act_lst, y_pred_lst, average='macro') )
            mic_f1.append( skmetrics.f1_score(y_act_lst, y_pred_lst, average='micro') )

        analytics_df = pd.DataFrame([accuracy,
                                     mac_precision,
                                     mic_precision,
                                     mac_recall,
                                     mic_recall,
                                     mac_f1,
                                     mic_f1]).transpose()
        analytics_df.columns = ['acc', 'mac_prec', 'mic_prec', 'mac_recall', 'mic_recall', 'mac_f1', 'mic_f1']

        self.losses_log.append(losses)
        self.analytics.append(analytics_df)

    def save_models(self, curr_time):
        """Save neural network models.
        """

        # Check if output directory exists
        if not os.path.exists(self.parameters["output_folder"]): os.makedirs(self.parameters["output_folder"])

        # Get the date of the training data used to train the models
        prefix = self.parameters["training_data"][0][:11]

        split_path = os.path.splitext(os.path.join(self.parameters["output_folder"], self.parameters["model_save_name"]))

        for i in range(self.parameters["k_fold_cv"]):

            model_save_path = split_path[0] + str(i+1) + "_data" + prefix + \
                              "_model" + curr_time.strftime("%m%d%y_%H%M") + split_path[1]

            # TODO: 
            # - instead of saving 4 files for 4 models, save all 4 state_dict in a list so that we get only 1 .pth file
            # - save self.parameters so that yaml config file is much shorter
            torch.save({
                'model_state_dict': self.networks[i].state_dict(),
                'optimizer_state_dict': self.optimizers[i].state_dict(),
                'loss': self.criterion,
                'preprocessing_params': self.preprocessing_params
            }, model_save_path)

    def save_analytics(self, curr_time):

        # Get the date of the training data used to train the models
        prefix = self.parameters["training_data"][0][:11]

        # Write losses to file
        split_losses_path = os.path.splitext(os.path.join(self.parameters["output_folder"], self.parameters["losses_save_name"]))

        for ind, losses in enumerate(self.losses_log):

            losses_file_path = split_losses_path[0] + str(ind+1) + "_data" + prefix + \
                              "_losses" + curr_time.strftime("%m%d%y_%H%M") + split_losses_path[1]

            with open (losses_file_path, "w") as l_file:
                str_lst = [str(i) + "\n" for i in losses]
                l_file.writelines(str_lst)

        # Write metrics to file
        split_analytics_path = os.path.splitext(os.path.join(self.parameters["output_folder"], self.parameters["analytics_save_name"]))

        for ind, df in enumerate(self.analytics):

            analytics_file_path = split_analytics_path[0] + str(ind+1) + "_data" + prefix + \
                              "_analytics" + curr_time.strftime("%m%d%y_%H%M") + split_analytics_path[1]

            df.to_csv(analytics_file_path, index=False)

class NNEvaluator(NNUtils):

    def __init__(self, args, verbose=True):

        super().__init__(args, verbose)

        self.util_type = self.UtilType.EVALUATOR

        self.read_yaml_config(args.config)

        self.setup_device()

        self.import_models()

        self.read_and_store_data()

    def run(self):

        self.evaluate_models()

        curr_time = datetime.now()

        self.save_metrics(curr_time)

    def evaluate_models(self):

        accuracy = {"training": [], "testing": []}
        mac_precision = {"training": [], "testing": []}
        mic_precision = {"training": [], "testing": []}
        mac_recall = {"training": [], "testing": []}
        mic_recall = {"training": [], "testing": []}
        mac_f1 = {"training": [], "testing": []}
        mic_f1 = {"training": [], "testing": []}

        for ind, model in enumerate(self.networks):

            with torch.no_grad():

                model.eval() # activate evaluation mode

                training_predictions = model(self.training_data["input"][ind].to(self.device))
                testing_predictions = model(self.testing_data["input"][ind].to(self.device))

                # Convert one-hot to indices
                _, train_pred_label = torch.max(torch.log_softmax(training_predictions.detach(), dim=1), dim=1)
                _, test_pred_label = torch.max(torch.log_softmax(testing_predictions.detach(), dim=1), dim=1)

                _, train_act_label = torch.max(self.training_data["output"][ind].to(self.device), dim=1)
                _, test_act_label = torch.max(self.testing_data["output"][ind].to(self.device), dim=1)

                # Convert tensor to python lists
                train_pred_label = train_pred_label.tolist()
                test_pred_label = test_pred_label.tolist()

                train_act_label = train_act_label.tolist()
                test_act_label = test_act_label.tolist()

                # Store results
                accuracy["training"].append( skmetrics.accuracy_score(train_act_label, train_pred_label) )
                accuracy["testing"].append( skmetrics.accuracy_score(test_act_label, test_pred_label) )

                mac_precision["training"].append( skmetrics.precision_score(train_act_label, train_pred_label, average='macro') )
                mac_precision["testing"].append( skmetrics.precision_score(test_act_label, test_pred_label, average='macro') )

                mic_precision["training"].append( skmetrics.precision_score(train_act_label, train_pred_label, average='micro') )
                mic_precision["testing"].append( skmetrics.precision_score(test_act_label, test_pred_label, average='micro') )

                mac_recall["training"].append( skmetrics.recall_score(train_act_label, train_pred_label, average='macro') )
                mac_recall["testing"].append( skmetrics.recall_score(test_act_label, test_pred_label, average='macro') )

                mic_recall["training"].append( skmetrics.recall_score(train_act_label, train_pred_label, average='micro') )
                mic_recall["testing"].append( skmetrics.recall_score(test_act_label, test_pred_label, average='micro') )

                mac_f1["training"].append( skmetrics.f1_score(train_act_label, train_pred_label, average='macro') )
                mac_f1["testing"].append( skmetrics.f1_score(test_act_label, test_pred_label, average='macro') )

                mic_f1["training"].append( skmetrics.f1_score(train_act_label, train_pred_label, average='micro') )
                mic_f1["testing"].append( skmetrics.f1_score(test_act_label, test_pred_label, average='micro') )
        
        # Store
        analytics_df = pd.DataFrame([accuracy["training"],
                                     accuracy["testing"],
                                     mac_precision["training"],
                                     mac_precision["testing"],
                                     mic_precision["training"],
                                     mic_precision["testing"],
                                     mac_recall["training"],
                                     mac_recall["testing"],
                                     mic_recall["training"],
                                     mic_recall["testing"],
                                     mac_f1["training"],
                                     mac_f1["testing"],
                                     mic_f1["training"],
                                     mic_f1["testing"]]).transpose()

        analytics_df.columns = ['acc_train',
                                'acc_test',
                                'mac_prec_train',
                                'mac_prec_test',
                                'mic_prec_train',
                                'mic_prec_test',
                                'mac_recall_train',
                                'mac_recall_test',
                                'mic_recall_train',
                                'mic_recall_test',
                                'mac_f1_train',
                                'mac_f1_test',
                                'mic_f1_train',
                                'mic_f1_test']

        self.analytics_df = analytics_df

    def save_metrics(self, curr_time):

        # Get the date of the training data used to train the models
        prefix = self.parameters["models"][0][-15:-4]

        # Write metrics to file
        split_analytics_path = os.path.splitext(os.path.join(self.parameters["output_folder"], self.parameters["analytics_save_name"]))

        analytics_file_path = split_analytics_path[0] + "_model" + prefix + "_metrics" + split_analytics_path[1]

        self.analytics_df.to_csv(analytics_file_path, index=False)

class NNEnsemble(NNUtils):

    def __init__(self, args, verbose=True):

        super().__init__(args, verbose)

        self.util_type = self.UtilType.ENSEMBLE

        self.read_yaml_config(args.config)

        self.setup_device()

        self.import_models()

    def make_prediction(self, pd_input):
        """Make predictions based on given inputs.

        The ensemble adds up each of the networks raw predictions, and then applies a softmax function to
        classify.

        Args:
            input: A Pandas dataframe with rows of feature values.

        Returns:
            The predicted class index integer(s) in a python list.
        """

        pred = torch.zeros(self.networks[0].network[-1].out_features).to(self.device)

        for ind, model in enumerate(self.networks):

            # Preprocess input
            if self.parameters["standardize"]:
                proc_pd_input, _ = self.standardize(pd_input, self.preprocessing_params[ind])

            elif self.parameters["normalize"]:
                proc_pd_input, _ = self.normalize(pd_input, self.preprocessing_params[ind])

            else:
                proc_pd_input = pd_input

            # Make prediction
            with torch.no_grad():
                model.eval()

                pred = torch.add(pred, model(torch.tensor(proc_pd_input.values, dtype=torch.float32).to(self.device)))

        _, pred_class_ind = torch.max(torch.log_softmax(pred.detach(), dim=1), dim=1)

        return [*map(int, pred_class_ind)]

def evaluate_predictions(act_output, pred_output):

    print(skmetrics.classification_report(act_output, pred_output))

    return skmetrics.confusion_matrix(act_output, pred_output)