import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.SVM_Predictor_Tester import SVM_Predictor_Tester

TRAINING_INDEX_END = 4000
TESTING_INDEX_END = None
C = 10
MAX_ITERATIONS = 10
WARMUP_ITERATIONS = 5


# np.dot(x, z) : C=10, max=50, warmup=1
# np.exp(-20*np.dot(x-z, x-z)) : C=10, max=5, warmup=1
def kernel(x, z):
    return np.exp(-20 * np.dot(x - z, x - z))


def read_data():
    df = pd.read_csv('datasets/hmeq/hmeq_prepared.csv').sample(frac=1)

    x_data = df.drop('BAD', axis=1).values
    y_data = df['BAD'].values

    inputs_train = x_data[: TRAINING_INDEX_END, ]
    outputs_train = y_data[: TRAINING_INDEX_END, ]

    inputs_test = x_data[TRAINING_INDEX_END: TESTING_INDEX_END, ]
    outputs_test = y_data[TRAINING_INDEX_END: TESTING_INDEX_END, ]

    return inputs_train, outputs_train, inputs_test, outputs_test


if __name__ == '__main__':
    inputs_train, outputs_train, inputs_test, outputs_test = read_data()
    T = SVM_Predictor_Tester(inputs_train, outputs_train, inputs_test, outputs_test, kernel, C=C,
                             max_iterations=MAX_ITERATIONS,
                             warmup_iterations=WARMUP_ITERATIONS)
    T.print_parameters()
    T.run()
