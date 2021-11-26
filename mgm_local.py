################################################################################
# Setup
################################################################################

# Set up pyspark to run locally
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import collect_list, lit, udf, explode, array, col, monotonically_increasing_id

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType, ArrayType, BooleanType

spark = (SparkSession
	.builder
	.appName("MGM")
	.master("local")
	.getOrCreate())

# General imports
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--conf_thresh", type=float, default=0.9)
parser.add_argument("--data_path", type=str, default='data_test.txt')
parser.add_argument("--log_file_path", type=str, default='mutation_log.csv')
args = parser.parse_args()

CONF_THRESHOLD = args.conf_thresh
data_path = args.data_path
log_file_path = args.log_file_path

if __name__ == "__main__":

    ################################################################################
    # Load Data
    ################################################################################

    # Specify schema of input file
    schema = StructType([
        StructField("seq", StringType()),
        StructField("label", IntegerType())
    ])

    # Row processing code
    def row_transform(row):
        input = row['value']
        items = input.lower().split(',')
        seq = ','.join(str(e) for e in items[0:-1])
        return Row(seq=seq, label=int(items[-1]))

    # Read in sequence data
    file_lines = spark.read.text(data_path).rdd.map(row_transform).filter(lambda row: row['label']==0)
    sequences = spark.createDataFrame(file_lines, schema).drop('label')

    # Add index column
    sequences = sequences.select("*").withColumn("id", monotonically_increasing_id())

    # Cache in memory
    sequences = sequences.cache()
    sequences.show() # TODO: can remove

    ################################################################################
    # Load Model
    ################################################################################

    model = keras.models.load_model('model.tf')

    ################################################################################
    # Setup Sequences Dataframe
    ################################################################################

    # Add `start_time` column with current POSIX timestamp
    ts = str(datetime.datetime.timestamp(datetime.datetime.now()))
    sequences = sequences.select("*").withColumn("start_time", lit(ts))

    # Add `finish_time` column with current POSIX timestamp
    sequences = sequences.select("*").withColumn("finish_time", lit(''))

    # Code for running model on sequence read from a text file
    '''
    Convert string representation of sequence in `sequences` to integer-encoded list
    '''
    def seq_str_to_list(s):
        seq = s.split(',')
        return list(map(int, seq))

    '''
    Model confidence. Takes x as the string in `sequences`.
    '''
    def f(x):
        x = seq_str_to_list(x)
        assert(len(x) == 60)
        one_hot_x = []
        for i in range(60):
            vec = np.zeros((20,))
            vec[x[i]] = 1
            one_hot_x.append(vec)
        one_hot_x = np.array(one_hot_x)
        output = model.predict(one_hot_x.reshape(1,60,20)).item()
        return output
        
    # Convert python function to PySpark user-defined function (UDF)
    f_udf = udf(f)

    # Add column for initial model prediction to sequences dataframe
    sequences = sequences.withColumn("init_pred", f_udf(sequences['seq']).cast(DoubleType()))

    # Add `current_pred` column, originally with same value of `init_pred`
    sequences = sequences.select("*").withColumn("current_pred", sequences.init_pred)

    # Add boolean flag with initial value of working=True
    sequences = sequences.select("*").withColumn("working", lit(True))

    # Add counter column with initial values total_changes=0
    sequences = sequences.select("*").withColumn("total_changes", lit(0))

    # Add string column with initial vlues last_change_log=''
    sequences = sequences.select("*").withColumn("last_change_log", lit(''))
        
    ################################################################################
    # Model-guided mutation code
    ################################################################################

    import tensorflow as tf
    from tensorflow.keras import backend as K

    '''
    Compute gradient of model loss with respect to inputs. Takes x as integer-encoded sequence.
    '''
    def compute_gradient(x, model):
        # Convert x to one-hot encoding
        one_hot_x = []
        for i in range(60):
            vec = np.zeros((20,))
            vec[x[i]] = 1
            one_hot_x.append(vec)
        one_hot_x = np.array(one_hot_x)
        x = one_hot_x

        # Compute gradients
        input = tf.Variable(one_hot_x.reshape(1,60,20))
        with tf.GradientTape() as tape:
            prediction = model(input, training=False)  # Logits for this minibatch
            target = tf.constant(0.0)
            target = tf.reshape(target, [1, 1])
            loss_value = K.binary_crossentropy(target, prediction)
            grads = tape.gradient(loss_value, input)

        # return 60 x 20 output
        return grads.numpy()[0]

    '''
    Select the i,b values best for a given integer-encoded sequence x.
    '''
    def select_substitution(x, grads):
        max_loss_increase = 0
        pos_to_change = None
        current_char_idx = None
        new_char_idx = None

        for i in range(60):
            a = int(x[i])
            for b in range(20):
                loss_b = grads[i][b]
                loss_a = grads[i][a]
                loss_increase = loss_b - loss_a
                if loss_increase > max_loss_increase:
                    max_loss_increase = loss_increase
                    pos_to_change = i
                    current_char_idx = a
                    new_char_idx = b
          
        return pos_to_change, current_char_idx, new_char_idx

    '''
    Perform a new substitution and return the new row to be added to the sequences dataframe
    '''
    def make_substitution(row, f, model):
        # Record initial time
        sub_start_time = datetime.datetime.now()
        
        # Extract data from row
        x = seq_str_to_list(row['seq'])
        id = row['id']
        total_changes = row['total_changes']
        working = row['working']
        start_time = row['start_time']
        finish_time = row['finish_time']

        # If done working, do not proceed
        if working == False:
            return row

        # Compute gradient
        grads = compute_gradient(x, model)

        # Find optimal i and b, and corresponding current character a
        i, a, b = select_substitution(x, grads)

        # Make the substitution
        x[i] = b

        # Produce new sequence
        new_seq_str = ','.join(str(e) for e in x)

        # Compute model score of new sequence
        new_conf = f(new_seq_str)

        # Measure substitution time
        end_time = datetime.datetime.now()
        time_s = (end_time - sub_start_time).total_seconds()

        # Create mutation log string
        log = "%i,%i,%i,%i,%i,%.8f,%.4f\n" % (id, total_changes, i, a, b, new_conf, time_s)

        # Determine if done working on this sample
        total_changes += 1
        if total_changes >= 60 or new_conf >= CONF_THRESHOLD:
            working = False
            finish_time = str(datetime.datetime.timestamp(datetime.datetime.now()))

        # Add data to new row in the order required by the existing `sequences` schema
        new_row = []
        new_row.append(new_seq_str)
        new_row.append(id)
        new_row.append(start_time)
        new_row.append(finish_time)
        new_row.append(row['init_pred'])
        new_row.append(new_conf)
        new_row.append(working)
        new_row.append(total_changes)
        new_row.append(log)
        return tuple(new_row)
        
    '''
    Run one iteration of MGM, then return updated dataframe.
    '''
    def one_iteration(sequences, f, model, log_file_path):
        # Update the sequences table
        sequences = sequences.rdd.map(lambda x: make_substitution(x, f, model)).toDF(sequences.schema.names)

        # Extract the logs of changes just made
        data = sequences.select('last_change_log').collect()
        output = ""
        for item in data:
            output += item['last_change_log']

        # Append to log
        with open(log_file_path, "a") as text_file:
            text_file.write(output)
        
        return sequences
        
    ################################################################################
    # Main
    ################################################################################

    # Create log file for mutations
    header = "seq_id,change_num,position,prev_char,new_char,new_pred,time_s\n"
    with open(log_file_path, "w") as text_file:
        text_file.write(header)

    # Set the initial working count
    working_count = sequences.filter(sequences['working'] == True).count()

    # Iterate until done working
    while working_count > 0:
        sequences = one_iteration(sequences, f, model, log_file_path)
        working_count = sequences.filter(sequences['working'] == True).count()

    # Save sequences table to disk
    sequences.toPandas().to_csv('sequences.csv')
