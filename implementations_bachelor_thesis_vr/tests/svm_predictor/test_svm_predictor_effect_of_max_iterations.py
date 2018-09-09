import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.svm_predictor.SVM_Predictor_Tester import SVM_Predictor_Tester

C = 14
max_iterations = [1, 5, 10, 15, 20, 30, 50, 100, 200, 400]


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


datasets = read_datasets()
all_validation_results = []

for i in range(len(max_iterations)):
    print('-' * 70, 'max_iterations = ', max_iterations[i])
    all_validation_results.append([])

    for j in [1, 2, 3]:
        print('-' * 55, 'Try #', j)
        Tester = SVM_Predictor_Tester()
        training_time = Tester.calculate_predictor(datasets['train']['x'], datasets['train']['y'], kernel, C,
                                                   max_iterations[i])
        validation_result = Tester.perform_test(datasets['validation']['x'], datasets['validation']['y'],
                                                label_names={1: 'GOOD', -1: 'BAD'})
        one_result = [training_time] + list(validation_result)

        all_validation_results[i].append(one_result)

averages = []

for kernel_results in all_validation_results:
    averages.append([sum(result_list) / len(result_list) for result_list in zip(*kernel_results)])

print('=' * 80)
print('=' * 80)

print('Averages (training time, sensitivity, specificity, ppv, npv, accuracy, Matthew\'s coefficient):')
for i in range(len(averages)):
    averages[i] = list(map(lambda x: "{0:.3f}".format(x), averages[i]))
    print('#', i, ':', ', '.join(averages[i]))

print('Latex code:')
for i in range(len(averages)):
    print('$', str(i + 1), '$ & ?? & $', '$ & $'.join(averages[i]), '$ \\\\ \\hline')
