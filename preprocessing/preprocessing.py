import pandas as pd
import threading
import asyncio
from sklearn.model_selection import train_test_split
from typo_correction import typo_correction
from basic_moving_average import moving_average
from timeSeriesDecomposition import decomposeDateTime
from outlier_detection_test import outlier_detection

df = pd.read_csv('Tetuan City power consumption.csv')


def aggregate_zone_consumptions(df: pd.DataFrame):
    """
    Combines the values of the three zones in the dataset and returns a 
    dataframe with the zone consumptions replaced with an aggregated consumption.
    """
    zone1 = "zone_1_power_consumption"
    zone2 = "zone_2_power_consumption"
    zone3 = "zone_3_power_consumption"

    aggregated_consumption = []
    for i in range(len(df[zone1])):
        aggregated_consumption.append(
            df[zone1][i] + df[zone2][i] + df[zone3][i])

    df["aggregated_consumption"] = aggregated_consumption

    df = df.drop(zone1, axis=1)
    df = df.drop(zone2, axis=1)
    df = df.drop(zone3, axis=1)
    return df


"""
Preprocessing rekkefølge:
 - fix typos
 - fill
 - aggregate zone consumptions to one number
 - outlier-detection, fill if needed
 - moving average
 - standardize
 - pca?
 - Feature selection? (RFE?)
 - normalize
"""
print('============ Preprocessing - Start ============')
# Fix column names
df = typo_correction(df)

# Drop null values, fill using a linear method
df.dropna(inplace=True)
df.interpolate(method="linear")

# Aggregate the zone consumptions to a single column
df = aggregate_zone_consumptions(df)

# Outlier-detection, fill if needed
# TODO: fix outlier detection
"""
async def tasks():
    for column in df.columns[1:]:
        print(column)
        #task = asyncio.create_task(outlier_detection(df, df[column]))
        outlier_detection(df, df[column])

        #await task
asyncio.run(tasks())
"""

# Moving average
# TODO: moving average
# moving_average()

# Data extraction
df = decomposeDateTime(df)

# Split input and output dataframe

output_df = df["aggregated_consumption"]
input_df = df.drop("aggregated_consumption", axis=1)
# Standardize

# PCA
# TODO:

# Normalize
# TODO:
# input_df = (float(input_df)-float(input_df.min()))/(float(input_df.max())-float(input_df.min()))

print('============ Preprocessing - End ============')
print('--- Input ---')
print(input_df)
print(input_df.columns)
print('--- Output ---')
print(output_df)
