import pandas as pd
import threading
import asyncio
from aggregate_zone_consumption import aggregate_zone_consumptions
from basic_moving_average import moving_average
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from outlier_detection_test import outlier_detection
from typo_correction import typo_correction
from timeSeriesDecomposition import decomposeDateTime
from rfe import rfe
from pca import pca_components

df = pd.read_csv('Tetuan City power consumption.csv')

"""
Preprocessing rekkefølge:
 - fix typos
 - fill
 - aggregate zone consumptions to one number
 - outlier-detection, fill if needed
 - standardize
 - pca?
 - Feature selection? (RFE?)
 - normalize
"""
print('============ Preprocessing - Start ============')
# Fix column names
print("==== Typo correction... ====")
df = typo_correction(df)
print("Done.", end="\n")

# Dropping diffuse and general diffuse flows
df = df.drop("diffuse_flows", axis=1)
df = df.drop("general_diffuse_flows", axis=1)

# Drop null values, fill using a linear method
print("==== Dropping null-equivalents... ====")
df.dropna(inplace=True)
df.interpolate(method="linear")
print("Done.", end="\n")

# Aggregate the zone consumptions to a single column
print("==== Aggregating zone consumptions... ====")
df = aggregate_zone_consumptions(df)
print("Done.", end="\n")

# Outlier-detection, fill if needed
print("==== Scanning for outliers... ====")
for column in df.columns:
    if column != "datetime":
        print(column)
        df[column] = outlier_detection(df, df[column])
print("Done.", end="\n")

# Split datetime into two features: day and 10minuteofday
print("==== Decompose datetime ====")
df = decomposeDateTime(df)
print("Done.")

# Split input and output dataframe
print("==== Splitting input/output df... ====")
output_df = df["aggregated_consumption"]
input_df = df.drop("aggregated_consumption", axis=1)
print("Done.", end="\n")

# Normalize
print("==== Data normalization... ====")
print("Before normalize")
print(input_df)
columns = input_df.columns
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
min_max_scaler.fit(input_df)
input_df = min_max_scaler.transform(input_df)
# input_df = normalize(X=input_df, axis=0)
input_df = pd.DataFrame(input_df, columns=columns)
print("After normalize")
print(input_df)
print("Done.")

# Standardize and PCA
print("==== PCA ====")
input_df = pca_components(input_df)
# input_df = rfe(input_df, output_df)
print("Done.", end="\n")


print('============ Preprocessing - End ============')
print('--- Input ---')
print(input_df)
print(input_df.columns)
print('--- Output ---')
print(output_df)

path = "./basic_models/preprocessed_data.csv"
preprocessed_data = pd.concat(objs=[input_df, output_df], axis=1)
preprocessed_data.to_csv(path)
