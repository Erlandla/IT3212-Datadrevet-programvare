import pandas as pd

# test with mock data
# df = pd.read_csv('D:\TommyChan\school\datadrevet\IT3212-Datadrevet-programvare\preprocessing\mockData\mock.csv')
colomnNames = ['Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption']

df = pd.read_csv('D:\TommyChan\school\datadrevet\IT3212-Datadrevet-programvare\Tetuan City power consumption.csv', usecols=colomnNames)

print(df)

normalized_df=(df-df.min())/(df.max()-df.min())
print(normalized_df)