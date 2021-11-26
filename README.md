# A Distributed Implementation of Model-Guided Adversarial Mutation Algorithms for Viral Sequences

## Overview

This project is broken down into three phases.

* Phase 1: Generate a large synthetic data set
* Phase 2: Train a classifier (locally) on some of this
data
* Phase 3: Deploy a distributed model-guided mutation
application


## Phases 1 and 2

Steps to generate the synthetic data set and train a classifier, on a local machine. The code for this section was developed and tested using Python 3.6.

1. Clone this GitHub repository: `git clone https://github.com/nathanbollig/distributed-mutation`
2. Apply `requirements.txt` to the runtime environment: `pip install -r requirements.txt`
3. (Optional) View the script `phases_1_and_2.sh`. This calls a Python script with several parameters as defined in `phases_1_and_2.py`. These parameters can be modified as desired.
4. Run `phases_1_and_2.sh`.
5. (Optional) Play through the notebook `explore_data.ipynb` to explore the size, shape, and example content from the output of this program.

The output of these steps is a set of files saved to local disk, as defined in the header of `phases_1_and_2.py`. These files include the model object, a dictionary of model validation results, data, and an instance of the custom `HMMGenerator` class used to generate the sequences.

## Phase 3

The model-guided mutation algorithm (MGM) is implemented as a Spark application in the file `mgm.py`. 

Note: The Spark application was tested locally using a small subset of data. The local version of the app is `mgm_local.py` and can be executed using `MGM_spark_local.ipynb`.

The following are steps for running Phase 3 on AWS EMR.

1.

