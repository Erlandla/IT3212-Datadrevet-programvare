import pandas as pd
from datetime import datetime


columnNames = ['DateTime', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows',
               'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']

# columnNames = [dateTime]


df = pd.read_csv('Tetuan City power consumption.csv', usecols=columnNames)

# print(df[dateTime].iloc[0])

# Fill missing data
# df.interpolate(method='linear')

# datetime_obj = datetime.strptime(df[dateTime].iloc[0], '%m/%d/%Y %H:%M')


def decomposeDateTime(df: pd.DataFrame):
    dateTime = 'datetime'
    if dateTime in df:
        dateTime_df = df.copy()
        # time_df['day'] = dateTime_df.map(lambda x: x + 'sdhjfls')
        dateTime_df['day'] = pd.DatetimeIndex(dateTime_df[dateTime]).dayofyear
        dateTime_df['hour'] = pd.DatetimeIndex(dateTime_df[dateTime]).hour
        dateTime_df['minute'] = pd.DatetimeIndex(dateTime_df[dateTime]).minute
        dateTime_df['10minuteofday'] = (
            dateTime_df['hour'] * 6) + (dateTime_df['minute'] / 10)
        # print(dateTime_df['day'], dateTime_df['10minuteofday'])
        dateTime_df = dateTime_df.drop('hour', axis=1).drop('minute', axis=1)
        df = df.drop(dateTime, axis=1)
        df = pd.concat([df, dateTime_df], axis=1)
        print(df)
        return df
    else:
        print('No dateTime in this dataframe')


decomposeDateTime(df)
# print(datetime_obj)
