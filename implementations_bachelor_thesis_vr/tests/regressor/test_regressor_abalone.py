import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.Regressor import Regressor

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)

M = 27
m = 1


def kernel(x, z):
    a = x - z
    return np.exp(-0.5 * np.dot(a, a))


MU = 0.15


def read_datasets():
    datasets = {}

    for data_type in ['train', 'validation', 'test']:
        df = pd.read_csv('../datasets/abalone/abalone_' + data_type + '.csv')

        datasets[data_type] = {
            'x': df.drop('Rings', axis=1).values,
            'y': df['Rings'].values
        }

    return datasets


datasets = read_datasets()

print('>> Amount training data:', len(datasets['train']['x']))
print('>> Amount validation data:', len(datasets['validation']['x']))
print('>> Amount testing data:', len(datasets['test']['x']))

true_ring_sizes_validation = m + (datasets['validation']['y'] + 1) * (M - m) / 2
true_ring_sizes_test = m + (datasets['test']['y'] + 1) * (M - m) / 2

all_avg_errors_validation = []
all_smallest_errors_validation = []
all_errors_test = []

basis_sizes = [1, 10, 50, 100, 300, 500, 1000, 2000, 2923]

for basis_size in basis_sizes:
    print('-' * 80, 'basis_size =', basis_size)

    regressors = []
    avg_error_validation = 0
    smallest_error_validation = np.inf
    best_regressor_index = -1

    for i in range(3):
        print('-' * 55, 'Try #', i)
        print('>> Training...')
        regressor = Regressor(datasets['train']['x'], datasets['train']['y'])
        regressor.train(kernel, mu=MU, basis_size=basis_size)
        regressors.append(regressor)

        print('>> Validating...')
        predictions_validation = np.apply_along_axis(lambda z: regressor.predict(z), 1, datasets['validation']['x'])
        predictions_validation_denormalized = np.around(m + (predictions_validation + 1) * (M - m) / 2)
        error_validation = np.mean(np.abs(predictions_validation_denormalized - true_ring_sizes_validation))

        print('>> Validation mean absolute error:', error_validation)
        avg_error_validation += error_validation

        if error_validation < smallest_error_validation:
            smallest_error_validation = error_validation
            best_regressor_index = i

    print('-' * 55)

    avg_error_validation /= 3
    print('>> Validation average mean absolute error:', avg_error_validation)
    all_avg_errors_validation.append(avg_error_validation)

    best_regressor = regressors[best_regressor_index]
    print('>> Best regressor was #', best_regressor_index + 1)
    print('>> Smallest validation error achieved:', smallest_error_validation)
    all_smallest_errors_validation.append(smallest_error_validation)

    print('>> Evaluating on test set...')
    predictions_test = np.apply_along_axis(lambda z: regressor.predict(z), 1, datasets['test']['x'])
    predictions_test_denormalized = np.around(m + (predictions_test + 1) * (M - m) / 2)
    error_test = np.mean(np.abs(predictions_test_denormalized - true_ring_sizes_test))

    print('>> Test set mean absolute error:', error_test)
    all_errors_test.append(error_test)

print('>> Summary (basis size, average validation error, smallest validation error, test error):')
for i in range(len(basis_sizes)):
    print([basis_sizes[i], all_avg_errors_validation[i], all_smallest_errors_validation[i], all_errors_test[i]])
print('>> Latex code:')

for i in range(len(basis_sizes)):
    print('$', '$ & $'.join(map(str, [basis_sizes[i], all_avg_errors_validation[i], all_smallest_errors_validation[i],
                                      all_errors_test[i]])), '$')
