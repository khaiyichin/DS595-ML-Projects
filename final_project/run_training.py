import util_modules as utm
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="DS 595-ML Final Project")
parser.add_argument('config', type=str, help='path to configuration file')
parser.add_argument('--no_save', action="store_true", help='path to configuration file')

args = parser.parse_args()

if __name__ == "__main__":
    nn_trainer = utm.NNTrainer(args)

    # Run trainer
    nn_trainer.run()