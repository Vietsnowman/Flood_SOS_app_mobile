
def build_lag_rolling_features(df, group_col="GID_3", date_col="date_local"):
    df = df.sort_values([group_col, date_col]).copy()

    # precip sum
    for k in [1,2,3,7]:
        df[f"tp_lag{k}"] = df.groupby(group_col)["precipitation_sum"].shift(k)

    for w in [3,7,14]:
        df[f"tp_sum_{w}d"] = df.groupby(group_col)["precipitation_sum"].rolling(w).sum().reset_index(0,drop=True)
        df[f"tp_max_{w}d"] = df.groupby(group_col)["precipitation_sum"].rolling(w).max().reset_index(0,drop=True)

    rain_flag = (df["precipitation_sum"] > 1).astype(int)
    for w in [3,7,14]:
        df[f"tp_rain_days_{w}d"] = rain_flag.groupby(df[group_col]).rolling(w).sum().reset_index(0,drop=True)

    # max 1h
    for w in [3,7,14]:
        df[f"tp_max1h_max_{w}d"] = df.groupby(group_col)["precipitation_max_1h"].rolling(w).max().reset_index(0,drop=True)

    # rain hours
    for w in [3,7,14]:
        df[f"rain_hours_sum_{w}d"] = df.groupby(group_col)["precipitation_hours"].rolling(w).sum().reset_index(0,drop=True)

    return df

def attach_static_features(df_forecast, static_row):
    X = df_forecast.copy()
    for col in static_row.index:
        if col not in X.columns and col not in ["date_local"]:
            X[col] = static_row[col]
    return X

def add_time_features(df):
    df = df.copy()
    df["month"] = df["date_local"].dt.month
    df["dayofyear"] = df["date_local"].dt.dayofyear
    return df