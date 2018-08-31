import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.svm_predictor.SVM_Predictor_Tester import SVM_Predictor_Tester

MAX_ITERATIONS = 10
C = 14


def read_datasets():
    datasets = {}

    for data_type in ['train', 'validation', 'test']:
        df = pd.read_csv('../datasets/hmeq/hmeq_' + data_type + '.csv')

        datasets[data_type] = {
            'x': df.drop('GOOD', axis=1).values,
            'y': df['GOOD'].values
        }

    return datasets


def kernel(x, z):
    return np.exp(-12.5 * np.dot(x - z, x - z))


warmup_iterations = [0, 1, 2, 3]

if __name__ == '__main__':
    datasets = read_datasets()
    all_validation_results = []

    for i in range(len(warmup_iterations)):
        print('-' * 70, 'C = ', warmup_iterations[i])
        all_validation_results.append([])

        for j in [1, 2, 3]:
            print('-' * 55, 'Try #', j)
            Tester = SVM_Predictor_Tester()
            training_time = Tester.calculate_predictor(datasets['train']['x'], datasets['train']['y'], kernel, C,
                                                       MAX_ITERATIONS,
                                                       warmup_iterations[i])
            one_result = Tester.perform_test(datasets['validation']['x'], datasets['validation']['y'],
                                                               label_names={1: 'GOOD', -1: 'BAD'})
            one_result = [training_time] + list(one_result)

            all_validation_results[i].append(one_result)

    averages = []

    for kernel_results in all_validation_results:
        averages.append([sum(result_list) / len(result_list) for result_list in zip(*kernel_results)])

    print('=' * 80)
    print('=' * 80)

    print('Averages (training_time, sensitivity, specificity, ppv, npv, accuracy, Matthew\'s coefficient):')
    for i in range(len(averages)):
        averages[i] = list(map(lambda x: "{0:.3f}".format(x), averages[i]))
        print('#', i, ':', ', '.join(averages[i]))

    print('Latex code:')
    for i in range(len(averages)):
        print('$', str(i + 1), '$ & ?? & $', '$ & $'.join(averages[i]), '$ \\\\ \\hline')
