import pandas as pd

# test with mock data
# df = pd.read_csv('D:\TommyChan\school\datadrevet\IT3212-Datadrevet-programvare\preprocessing\mockData\mock.csv')

normalized_df=(df-df.min())/(df.max()-df.min())
print(normalized_df)