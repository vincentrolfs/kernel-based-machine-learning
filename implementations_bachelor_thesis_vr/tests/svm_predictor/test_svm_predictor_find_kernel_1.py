import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.tests.svm_predictor.SVM_Predictor_Tester import SVM_Predictor_Tester

C = 10
MAX_ITERATIONS = 10


def kernel_0(x, z):
    return np.exp(-0.01 * np.dot(x - z, x - z))


def kernel_1(x, z):
    return np.exp(-0.1 * np.dot(x - z, x - z))


def kernel_2(x, z):
    return np.exp(-1 * np.dot(x - z, x - z))


def kernel_3(x, z):
    return np.exp(-10 * np.dot(x - z, x - z))


def kernel_4(x, z):
    return np.exp(-20 * np.dot(x - z, x - z))


def kernel_5(x, z):
    return np.exp(-30 * np.dot(x - z, x - z))


def kernel_6(x, z):
    return np.exp(-40 * np.dot(x - z, x - z))


def kernel_7(x, z):
    return np.exp(-50 * np.dot(x - z, x - z))


def kernel_8(x, z):
    return np.dot(x, z)


def kernel_9(x, z):
    return np.dot(0.1 * x, 0.1 * z)


def kernel_10(x, z):
    return (1 + np.dot(0.1 * x, 0.1 * z)) ** 2


def kernel_11(x, z):
    return (1 + np.dot(0.1 * x, 0.1 * z)) ** 3


def kernel_12(x, z):
    return (1 + np.dot(0.1 * x, 0.1 * z)) ** 4


def kernel_13(x, z):
    return (1 + np.dot(0.1 * x, 0.1 * z)) ** 5


def kernel_14(x, z):
    a = 0.1 * (x - z)
    return (1 + np.dot(a, a)) ** (-3)


def kernel_15(x, z):
    a = 0.1 * (x - z)
    return (1 + np.dot(a, a)) ** (-4)


def kernel_16(x, z):
    a = 0.1 * (x - z)
    return (1 + np.dot(a, a)) ** (-5)


def kernel_17(x, z):
    a = 0.1 * (x - z)
    return (1 + np.dot(a, a)) ** (-6)


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

for i in range(0, 18):
    print('-' * 70, 'Kernel', i)
    all_validation_results.append([])

    for j in [1, 2, 3]:
        print('-' * 55, 'Try #', j)
        kernel = globals()['kernel_' + str(i)]
        Tester = SVM_Predictor_Tester()
        Tester.calculate_predictor(datasets['train']['x'], datasets['train']['y'], kernel, C, MAX_ITERATIONS)
        one_result = Tester.perform_test(datasets['validation']['x'], datasets['validation']['y'],
                                         label_names={1: 'GOOD', -1: 'BAD'})
        all_validation_results[i].append(one_result)

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
