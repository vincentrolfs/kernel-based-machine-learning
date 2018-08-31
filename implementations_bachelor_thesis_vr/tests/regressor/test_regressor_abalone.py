import numpy as np
import pandas as pd

from implementations_bachelor_thesis_vr.Regressor import Regressor

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)

M = 30.5000
m = 2.5000


def read_datasets():
    datasets = {}

    for data_type in ['train', 'validation', 'test']:
        df = pd.read_csv('../datasets/abalone/abalone_' + data_type + '.csv')

        datasets[data_type] = {
            'x': df.drop('Age', axis=1).values,
            'y': df['Age'].values
        }

    return datasets


datasets = read_datasets()
r = Regressor(datasets['train']['x'], datasets['train']['y'])

print('>> Amount training data:', len(datasets['train']['x']))
print('>> Amount validation data:', len(datasets['validation']['x']))
print('>> Amount testing data:', len(datasets['test']['x']))


# Good: mu=0.15, basis_size=2923, kernel(x, z) = np.exp(-1.5 * np.dot(x-z, x-z))


def kernel(x, z):
    a = x - z
    return np.exp(-0.125 * np.dot(a, a))

# 1.6
for C in [0.5, 1, 1.5, 2, 2.5, 3]:
    print('-'*80, 'C =', C)
    avg_mean_absolute_error = 0

    for _ in range(3):
        print('>> Training...')
        r.train(kernel, mu=C, basis_size=500)
        print('>> Validating...')
        predictions = np.apply_along_axis(lambda z: r.predict(z), 1, datasets['validation']['x'])

        predictions_transformed = m + (predictions + 1) * (M - m) / 2
        truth_transformed = m + (datasets['validation']['y'] + 1) * (M - m) / 2

        mean_absolute_error = np.mean(np.abs(predictions_transformed - truth_transformed))
        mean_squared_error = np.sqrt(np.mean((predictions_transformed - truth_transformed) ** 2))

        print('>> Mean absolute error:', mean_absolute_error)
        print('>> Mean squared error:', mean_squared_error)

        avg_mean_absolute_error += mean_absolute_error

    avg_mean_absolute_error /= 3
    print('>> avg_mean_absolute_error =', avg_mean_absolute_error)
