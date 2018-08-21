import pandas as pd

TRAINING_SET_FRACTION = 0.7
VALIDATION_SET_FRACTION = 0.15

df_original = pd.read_csv('hmeq_original.csv')

print('>> Replacing BAD with GOOD:')
df_original['BAD'] = 1-df_original['BAD']
df_original.rename(columns={'BAD': 'GOOD'}, inplace=True)

print('>> HMEQ dataset description:')
print(df_original.describe(include='all').T.sort_values('count'))

print('>> Number of rows with missing values:')
print(len(df_original) - len(df_original.dropna()))

print('>> Some data points:')
print(df_original.iloc[[27, 200, 1230]])

print('>> Creating dummies...')
df_temp = pd.get_dummies(df_original)

print('>> The same data points with dummies:')
print(df_temp.iloc[[27, 200, 1230]])

print('>> Creating training, validation and test set...')
df_temp = df_temp.sample(frac=1)
cutoff_train = int(TRAINING_SET_FRACTION * len(df_temp))
cutoff_validation = int((TRAINING_SET_FRACTION + VALIDATION_SET_FRACTION) * len(df_temp))

df_train = df_temp[:cutoff_train].copy()
df_validation = df_temp[cutoff_train:cutoff_validation].copy()
df_test = df_temp[cutoff_validation:].copy()

print('>> Imputing...')
mean = df_train.mean()
df_train.fillna(mean, inplace=True)
df_validation.fillna(mean, inplace=True)
df_test.fillna(mean, inplace=True)

print('>> Normalizing...')
M = df_train.max()
m = df_train.min()
df_train = -1 + 2 * ((df_train - m) / (M - m))
df_validation = -1 + 2 * ((df_validation - m) / (M - m))
df_test = -1 + 2 * ((df_test - m) / (M - m))

print('>> Saving...')
df_train.to_csv('hmeq_train.csv', index=False)
df_validation.to_csv('hmeq_validation.csv', index=False)
df_test.to_csv('hmeq_test.csv', index=False)

print('>> Done.')
