import pandas as pd


#columnNames = ['Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption']

columnNames = ['DateTime']

df = pd.read_csv('D:\TommyChan\school\datadrevet\IT3212-Datadrevet-programvare\Tetuan City power consumption.csv', usecols=columnNames)

print(df)

# Fill missing data
#df.interpolate(method='linear')


print(df)