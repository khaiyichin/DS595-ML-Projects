import util_modules as utm
import os

if __name__ == "__main__":

    args = utm.parse_arguments()

    os.chdir(args.working_dir) # change to desired working directory

    nn_trainer = utm.NNTrainer(args)

    # Run trainer
    nn_trainer.run()