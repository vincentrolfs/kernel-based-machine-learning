import pandas as pd

df = pd.read_csv('adult_original.csv',
                 sep=', ',
                 names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income'])
df = pd.get_dummies(df, drop_first=True)
for i in (0, 1):
    df_i = df[df["income_>50K"] == i]
    df[df["income_>50K"] == i] = df_i.fillna(df_i.mean())

df_nm = -1 + 2 * ((df - df.min()) / (df.max() - df.min()))

df_nm.to_csv('adult_prepared.csv', index=False)