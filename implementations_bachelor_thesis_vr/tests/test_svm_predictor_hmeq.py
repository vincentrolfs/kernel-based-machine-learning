import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.SVM_Predictor_Tester import SVM_Predictor_Tester

C = 10
MAX_ITERATIONS = 10
WARMUP_ITERATIONS = 2


# np.dot(x, z) : C=10, max=50, warmup=1
# np.exp(-20*np.dot(x-z, x-z)) : C=10, max=5, warmup=1
def kernel(x, z):
    return np.exp(-20 * np.dot(x - z, x - z))


def read_data():
    data = {}
    for data_type in ['train', 'test']:
        df = pd.read_csv('datasets/hmeq/hmeq_' + data_type + '.csv')

        data[data_type] = {
            'x': df.drop('GOOD', axis=1).values,
            'y': df['GOOD'].values
        }

    return data


if __name__ == '__main__':
    data = read_data()
    T = SVM_Predictor_Tester(data['train']['x'], data['train']['y'], data['test']['x'], data['test']['y'], kernel, C,
                             MAX_ITERATIONS, WARMUP_ITERATIONS, label_names={1: 'GOOD', -1: 'BAD'})
    T.print_parameters()
    T.run()
