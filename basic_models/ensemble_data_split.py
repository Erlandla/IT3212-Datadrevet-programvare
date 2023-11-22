import pandas

df = pandas.read_csv("preprocessed_data.csv")

df = df.sample(frac=1, random_state=1)

df_25_percent = df.sample(frac=0.25, random_state=1)
df_75_percent = df.drop(df_25_percent.index)


print(df)
print(df_25_percent)
print(df_75_percent)

df_75_percent.to_csv('preprocessed_data.csv', index=False)
df_25_percent.to_csv('preprocessed_data_test.csv', index=False)
print('SAVED')