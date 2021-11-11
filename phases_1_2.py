"""
Runs Phases 1 and 2 of my CS 744 project. The following command-line parameters are accepted:
    class_signal: A constant that designates how different the positive class is from the negative class (as described
                        in report)
    n_generated: The total number of sequences to generate.
    p: The positive class prevalence in the generated data.
    model_type: Can be "LR" or "LSTM" (architectures described in report)
    n_epochs: Number of epochs to train the model.

The following is a summary of what this program does:
1. Generates the dataset of size `n_generated`, with class prevalence `p`, according to the designated `class_signal`.
2. Split the data into train, validation, test (80/10/10).
3. Train the designated `model_type` on the training set, report performance on the validation set.
4. Store output

The following objects are created:
    model: Keras model object
    result: dictionary of model performance statistics during training and validation
    X_train, X_val, X_test: one-hot-encoding synthetic sequences
    y_train, y_val, y_test: integer labels (0 or 1)
    gen: the HMMGenerator object used to generate the dataset
    aa_vocab: the ordered amino acid character alphabet used for the sequence encoding

The objects created by this program are saved to disk into the following pickle files:
    model.pkl: tuple (model, result)
    aa_vocab.pkl: aa_vocab
    generator.pkl: gen
    data_train.pkl: tuple (X_train, y_train)
    data_val.pkl: tuple (X_val, y_val)
    data_test.pkl: tuple (X_test, y_test)
"""

from seq_model import big_bang
import argparse
import pickle

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_signal", type=int, default=10)
    parser.add_argument("--n_generated", type=int, default=10)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--model_type", default="LSTM")
    parser.add_argument("--n_epochs", type=int, default=10)
    args = parser.parse_args()

    # Generate data and train a model
    big_bang_result_tuple = big_bang(class_signal=args.class_signal,
                                      num_instances=args.n_generated,
                                      p=args.p,
                                      model_type=args.model_type,
                                      n_epochs=args.n_epochs)

    # Unpack the results tuple
    model, result, X_list, y_list, gen, aa_vocab = big_bang_result_tuple
    X_train, X_val, X_test = X_list
    y_train, y_val, y_test = y_list

    # Save results to disk
    with open("model.pkl", 'wb') as pfile:
        pickle.dump((model, result), pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open("aa_vocab.pkl", 'wb') as pfile:
        pickle.dump(aa_vocab, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open("generator.pkl", 'wb') as pfile:
        pickle.dump(gen, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open("data_train.pkl", 'wb') as pfile:
        pickle.dump((X_train, y_train), pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open("data_val.pkl", 'wb') as pfile:
        pickle.dump((X_val, y_val), pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open("data_test.pkl", 'wb') as pfile:
        pickle.dump((X_test, y_test), pfile, protocol=pickle.HIGHEST_PROTOCOL)
