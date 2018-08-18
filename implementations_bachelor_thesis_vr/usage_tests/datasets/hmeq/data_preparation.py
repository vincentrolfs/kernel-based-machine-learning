import pandas as pd

df = pd.read_csv('hmeq_original.csv')
df = pd.get_dummies(df)

for i in (0, 1):
    df_i = df[df["BAD"] == i]
    df[df["BAD"] == i] = df_i.fillna(df_i.mean())

df_nm = -1 + 2 * ((df - df.min()) / (df.max() - df.min()))

df_nm.to_csv('hmeq_prepared.csv', index=False)
