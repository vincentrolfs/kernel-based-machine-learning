import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.SVM_Predictor_Tester import SVM_Predictor_Tester

C = 10
MAX_ITERATIONS = 10
WARMUP_ITERATIONS = 0


def read_datasets():
    datasets = {}

    for data_type in ['train', 'validation', 'test']:
        df = pd.read_csv('datasets/hmeq/hmeq_' + data_type + '.csv')

        datasets[data_type] = {
            'x': df.drop('GOOD', axis=1).values,
            'y': df['GOOD'].values
        }

    return datasets


def kernel_1(x, z):
    return np.exp(-0.01 * np.dot(x - z, x - z))


def kernel_2(x, z):
    return np.exp(-0.1 * np.dot(x - z, x - z))


def kernel_3(x, z):
    return np.exp(-1 * np.dot(x - z, x - z))


def kernel_4(x, z):
    return np.exp(-10 * np.dot(x - z, x - z))


def kernel_5(x, z):
    return np.exp(-20 * np.dot(x - z, x - z))


def kernel_6(x, z):
    return np.exp(-30 * np.dot(x - z, x - z))


def kernel_7(x, z):
    return np.exp(-40 * np.dot(x - z, x - z))


def kernel_8(x, z):
    return np.exp(-60 * np.dot(x - z, x - z))


def kernel_9(x, z):
    return np.exp(-80 * np.dot(x - z, x - z))


if __name__ == '__main__':
    datasets = read_datasets()

    for i in range(1, 10):
        kernel = globals()['kernel_' + str(i)]
        Tester = SVM_Predictor_Tester()
        Tester.calculate_predictor(datasets['train']['x'], datasets['train']['y'], kernel, C, MAX_ITERATIONS,
                                   WARMUP_ITERATIONS)
        Tester.perform_test(datasets['validation']['x'], datasets['validation']['y'],
                            label_names={1: 'GOOD', -1: 'BAD'})
