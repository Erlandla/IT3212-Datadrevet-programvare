import pandas as pd


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
