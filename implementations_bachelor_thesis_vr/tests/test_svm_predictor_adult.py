import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.SVM_Predictor_Tester import SVM_Predictor_Tester

TRAINING_SET_SIZE = 4000
TESTING_SET_SIZE = 5000
C = 10
MAX_ITERATIONS = 10
WARMUP_ITERATIONS = 1


def kernel0(x, z):
    return np.exp(-np.dot((x - z)/100, (x - z)/100))


def kernel1(x, z):
    return np.exp(-0.000001 * np.dot(x - z, x - z))


def kernel2(x, z):
    return np.exp(-0.0000001 * np.dot(x - z, x - z))


if __name__ == '__main__':
    df = pd.read_csv('datasets/adult/adult_prepared.csv').sample(frac=1)
    x_data = df.drop('income_>50K', axis=1).values
    y_data = df['income_>50K'].values

    for kernel in [kernel0, kernel1, kernel2, kernel3]:
        T = SVM_Predictor_Tester(x_data, y_data, TRAINING_SET_SIZE,
                                 TESTING_SET_SIZE, kernel, C=C, max_iterations=MAX_ITERATIONS,
                                 warmup_iterations=WARMUP_ITERATIONS)
        T.print_parameters()
        T.run()
