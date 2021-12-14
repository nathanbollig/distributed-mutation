"""
Script for computing average time metrics from the output of the Spark application. Run this in a directory containing
all the csv files that need to be merged.
"""


import os
import pandas as pd
from io import StringIO

directory = os.path.dirname(__file__)
dataframes = []

cols = ['seq', 'start_time', 'finish_time', 'init_pred', 'current_pred', 'working', 'total_changes', 'change_log']

change_log_cols = ['total_changes', 'i', 'a', 'b', 'new_conf', 'time_s']

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, names=cols)
        dataframes.append(df)

dataframes = pd.concat(dataframes)

# # Get per trajectory metric
# dataframes['traj_time'] = dataframes['finish_time'] - dataframes['start_time']
# print("Average time per traj: %.3f min" % (dataframes['traj_time'].mean()/60,))
print("Average number of mutations per traj: %.3f" % (dataframes['total_changes'].mean(),))
print("Maximum number of mutations per traj: %.3f" % (dataframes['total_changes'].max(),))

# Get per mutation metric
change_logs = dataframes['change_log'].tolist()

change_log_full = ""
for substring in change_logs:
    change_log_full += substring + '\n'

change_log_io = StringIO(change_log_full)
change_log = pd.read_csv(change_log_io, sep=',', names=change_log_cols)

print("Average time per char mutation: %.3f s" % (change_log['time_s'].mean(),))
print("")
