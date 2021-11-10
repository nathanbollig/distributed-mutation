# A Distributed Implementation of Model-Guided Adversarial Mutation Algorithms for Viral Sequences

## Overview

This project is broken down into three phases.

* Phase 1: Generate a large synthetic data set
* Phase 2: Train a classifier (locally) on some of this
data
* Phase 3: Deploy a distributed model-guided mutation
application


## Phases 1 and 2 - ON LOCAL MACHINE

Steps to generate the synthetic data set and train a classifier, on a local machine.

1. Clone this GitHub repository: ***
2. Apply requirements.txt to the runtime environment (ensure Python and necessary packages are installed, as listed).
3. View the script `phases_1_and_2.sh`. This calls a Python script with several parameters as defined in `phases_1_and_2.py`. These parameters can be modified as desired.
4. Run `phases_1_and_2.sh`.
5. Optionally, play through the notebook `explore_data.ipynb` to explore the size, shape, and example content from the output of this program.

The output of these steps is a pickle file `phase_1_2_results.pkl` saved to local disk, which is a tuple of the items as defined in `phases_1_and_2.py`. This file includes the model object, model validation results, data, and an instance of the custom HMMGenerator class used to generate the sequences.



... notes ... 

phases_1_and_2.sh
python phases_1_2.py --class_signal 1 --n_generated 10000 --p 0.5 --model_type LR --n_epochs 10

HMM_generator_motif.py
seq_model.py
