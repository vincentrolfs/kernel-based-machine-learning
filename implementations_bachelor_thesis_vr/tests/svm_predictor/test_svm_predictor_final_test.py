import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.svm_predictor.SVM_Predictor_Tester import SVM_Predictor_Tester

C = 14
MAX_ITERATIONS = 5


def kernel(x, z):
    a = x - z
    return np.exp(-12.5 * np.dot(a, a))


def read_datasets():
    datasets = {}

    for data_type in ['train', 'validation', 'test']:
        df = pd.read_csv('../datasets/hmeq/hmeq_' + data_type + '.csv')

        datasets[data_type] = {
            'x': df.drop('GOOD', axis=1).values,
            'y': df['GOOD'].values
        }

    return datasets


testers = []
matthews_coefficients = []

datasets = read_datasets()

for i in range(5):
    print('-' * 70, 'Predictor #', i + 1)

    Tester = SVM_Predictor_Tester()
    Tester.calculate_predictor(datasets['train']['x'], datasets['train']['y'], kernel, C,
                               MAX_ITERATIONS)
    validation_result = Tester.perform_test(datasets['validation']['x'], datasets['validation']['y'],
                                            label_names={1: 'GOOD', -1: 'BAD'})

    matthews_coefficients.append(validation_result[-1])
    testers.append(Tester)

best_index = np.argmax(matthews_coefficients)
tester_of_best_model = testers[best_index]

print('>> Best model was #', best_index + 1)
print('>> Now performing test on test set.')

tester_of_best_model.perform_test(datasets['test']['x'], datasets['test']['y'],
                                  label_names={1: 'GOOD', -1: 'BAD'})
