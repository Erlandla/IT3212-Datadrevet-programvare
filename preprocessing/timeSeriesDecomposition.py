import pandas as pd
from datetime import datetime


#columnNames = ['Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows','Zone 1 Power Consumption','Zone 2  Power Consumption','Zone 3  Power Consumption']

columnNames = ['DateTime']

df = pd.read_csv('D:\TommyChan\school\datadrevet\IT3212-Datadrevet-programvare\Tetuan City power consumption.csv', usecols=columnNames)

print(df['DateTime'].iloc[0])

# Fill missing data
#df.interpolate(method='linear')

datetime_obj = datetime.strptime(df['DateTime'].iloc[0], '%m/%d/%Y %H:%M')



print(datetime_obj)