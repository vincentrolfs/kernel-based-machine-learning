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


def shuffle_unison(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]


datasets = read_datasets()
all_validation_results = []
total_amount_training_data = len(datasets['train']['x'])

for training_set_fraction in [0.2, 0.4, 0.6, 0.8, 1]:
    print('-' * 70, 'training_set_fraction = ', training_set_fraction)
    selection_size = int(training_set_fraction * total_amount_training_data)

    validation_result_group = []
    all_validation_results.append(validation_result_group)

    for j in range(3):
        print('-' * 55, 'Try #', j + 1)
        indices = np.random.choice(total_amount_training_data, selection_size, replace=False)
        training_set_selection = {'x': datasets['train']['x'][indices],
                                  'y': datasets['train']['y'][indices]}

        Tester = SVM_Predictor_Tester()
        Tester.calculate_predictor(training_set_selection['x'], training_set_selection['y'], kernel, C, MAX_ITERATIONS)

        one_result = Tester.perform_test(datasets['validation']['x'], datasets['validation']['y'],
                                         label_names={1: 'GOOD', -1: 'BAD'})
        validation_result_group.append(one_result)

averages = []

for kernel_results in all_validation_results:
    averages.append([sum(result_list) / len(result_list) for result_list in zip(*kernel_results)])

print('=' * 80)
print('=' * 80)

print('Averages (sensitivity, specificity, ppv, npv, accuracy, Matthew\'s coefficient):')
for i in range(len(averages)):
    averages[i] = list(map(lambda x: "{0:.3f}".format(x), averages[i]))
    print('#', i, ':', ', '.join(averages[i]))

print('Latex code:')
for i in range(len(averages)):
    print('$', str(i + 1), '$ & ?? & $', '$ & $'.join(averages[i]), '$ \\\\ \\hline')