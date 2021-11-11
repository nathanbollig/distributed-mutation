# A Distributed Implementation of Model-Guided Adversarial Mutation Algorithms for Viral Sequences

## Overview

This project is broken down into three phases.

* Phase 1: Generate a large synthetic data set
* Phase 2: Train a classifier (locally) on some of this
data
* Phase 3: Deploy a distributed model-guided mutation
application


## Phases 1 and 2 - ON LOCAL MACHINE

Steps to generate the synthetic data set and train a classifier, on a local machine. The code for this section was developed and tested using Python 3.6.

1. Clone this GitHub repository: `git clone https://github.com/nathanbollig/distributed-mutation`
2. Apply `requirements.txt` to the runtime environment: `pip install -r requirements.txt`
3. (Optional) View the script `phases_1_and_2.sh`. This calls a Python script with several parameters as defined in `phases_1_and_2.py`. These parameters can be modified as desired.
4. Run `phases_1_and_2.sh`.
5. (Optional) Play through the notebook `explore_data.ipynb` to explore the size, shape, and example content from the output of this program.

The output of these steps is a pickle file `phase_1_2_results.pkl` saved to local disk, which is a tuple of the items as defined in `phases_1_and_2.py`. This file includes the model object, a dictionary of model validation results, data, and an instance of the custom `HMMGenerator` class used to generate the sequences.
