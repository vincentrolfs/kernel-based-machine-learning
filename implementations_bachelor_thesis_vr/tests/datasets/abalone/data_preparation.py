import pandas as pd

TRAINING_SET_FRACTION = 0.7
VALIDATION_SET_FRACTION = 0.15

df_original = pd.read_csv('abalone_original.csv')

print('>> Replacing Rings with Age:')
df_original['Rings'] = 1.5 + df_original['Rings']
df_original.rename(columns={'Rings': 'Age'}, inplace=True)

print('>> Abalone dataset description:')
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

print('>> Normalizing...')
M = df_train.max()
m = df_train.min()
print('>> M =', M)
print('>> m =', m)
df_train = -1 + 2 * ((df_train - m) / (M - m))
df_validation = -1 + 2 * ((df_validation - m) / (M - m))
df_test = -1 + 2 * ((df_test - m) / (M - m))

print('>> Saving...')
df_train.to_csv('abalone_train.csv', index=False)
df_validation.to_csv('abalone_validation.csv', index=False)
df_test.to_csv('abalone_test.csv', index=False)

print('>> Done.')
