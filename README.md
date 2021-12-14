# A Distributed Implementation of Model-Guided Adversarial Mutation Algorithms for Viral Sequences

## Overview

This project is broken down into three phases.

* Phase 1: Generate a large synthetic data set
* Phase 2: Train a classifier (locally) on some of this
data
* Phase 3: Deploy a distributed model-guided mutation
application in Spark
* Phase 4: Evaluate the system performance of the Spark application, compared to a single-core application


## Phases 1 and 2

Steps to generate the synthetic data set and train a classifier, on a local machine. The code for this section was developed and tested using Python 3.6.

1. Clone this GitHub repository: `git clone https://github.com/nathanbollig/distributed-mutation`
2. Apply `requirements.txt` to the runtime environment: `pip install -r requirements.txt`
3. (Optional) View the script `phases_1_and_2.sh`. This calls a Python script with several parameters as defined in `phases_1_and_2.py`. These parameters can be modified as desired.
4. Run `phases_1_and_2.sh`.
5. (Optional) Play through the notebook `explore_data.ipynb` to explore the size, shape, and example content from the output of this program.

The output of these steps is a set of files saved to local disk, as defined in the header of `phases_1_and_2.py`. These files include the model object, a dictionary of model validation results, data, and an instance of the custom `HMMGenerator` class used to generate the sequences.

## Phase 3 - Spark Application

The model-guided mutation algorithm (MGM) is implemented as a Spark application in the file `mgm.py`. 

Note: The Spark application was tested locally using a small subset of data. The local version of the app is `mgm_local.py` and can be executed using `MGM_spark_local.ipynb`.

The following are steps for running Phase 3 on AWS EMR.

1. Create a AWS S3 bucket, upload the script `libraries.sh`.
2. Create a cluster on the AWS EMR platform. Go to Advanced Options and set the following software configuration:
 - Hadoop 2.10.1
 - Spark 2.4.7
 - TensorFlow 2.4.1
 - Ganglia 3.7.2
3. Set the hardware to m5.xlarge, 3 instances.
4. Set up a custom bootstrap action: `s3://<bucket_ID>/libraries.sh`
5. Add a SSH key
6. Upload the required files from Phases 1 & 2 to the S3 bucket:
 - `mgm.py`
 - `data_test.txt`
 - `model.tf`
7. Add a step for the Spark application:
 - Location: `s3://<bucket_ID>/mgm.py`
 - Arguments:
```
--data_path s3://<bucket_ID>/data_test.txt 
--model_file_path s3://<bucket_ID>/model.tf  
--output_path s3://<bucket_ID>/sequences
```
 - Config parameters:
```
--conf "spark.yarn.heterogeneousExecutors.enabled=false" 
--conf "spark.maximizeResourceAllocation=true"
```
8. Output is deposited in `s3://<bucket_ID>/sequences`

## Phase 3 - Single-Core Application

The single-core implementation of MGM is available from my `mgm` package. The steps to execute this are below.

1. SSH into the desired machine.
2. Run the following:
```
sudo yum update -y
sudo yum install git -y
sudo pip3 install git+https://github.com/nathanbollig/mgm
sudo pip3 install numpy==1.19.2
sudo git clone https://github.com/nathanbollig/mgm
export PYTHONPATH=${PYTHONPATH}:${HOME}/mgm
nohup python3 mgm/tests/test_744_experiment.py
```
3. Output files are saved into the working directory.

## Phase 4

System performance was evaluated using several methods.

1. Spark UI: on AWS EMR, this is accessible by link from the GUI console
2. Ganglia: the UI is accessible by the following steps
 - SSH into master machine, set up dynamic IP tunnel on port 8157
 - Get the private IP address for the machine
 - Set up `localhost:8157` proxy in browser
 - Go to URL: `<IP>/ganglia/`
3. Timing summaries of the Spark apps were collected from .csv outputs. Place them into a directory with a copy of `compute_metrics.py`, and run this script to generate a summary for the given .csv files.
