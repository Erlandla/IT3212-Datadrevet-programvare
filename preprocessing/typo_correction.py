import pandas as pd

columnNames = ['DateTime', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows',
               'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']

# columnNames = [dateTime]


df = pd.read_csv('Tetuan City power consumption.csv', usecols=columnNames)


def typo_correction(df):
    df.columns = [x.lower() for x in df.columns]
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('__', '_')
    return df


print(df)
print(typo_correction(df))
print("hei :)")
