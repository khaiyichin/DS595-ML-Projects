import util_modules as utm
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="DS 595-ML Final Project")
parser.add_argument('config', type=str, help='path to configuration file')
parser.add_argument('working_dir', type=str, help='working directory to run the training')
parser.add_argument('--no_save', action="store_true", help='path to configuration file')

args = parser.parse_args()

if __name__ == "__main__":

    os.chdir(args.working_dir) # change to desired working directory

    nn_trainer = utm.NNTrainer(args)

    # Run trainer
    nn_trainer.run()