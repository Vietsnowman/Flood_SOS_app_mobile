import pandas as pd
import os

FILE_2023 = "data/NgheAn_weather32full_flood_merge_by_commune_time_2023_NO_NaN.csv"
FILE_2024 = "data/NgheAn_weather32full_flood_merge_by_commune_time_2024_NO_NaN.csv"
FILE_2025 = "data/NgheAn_weather32full_flood_merge_by_commune_time_2025_NO_NaN.csv"

OUTPUT = "cache/merged.csv"

def main():
    os.makedirs("cache", exist_ok=True)

    df1 = pd.read_csv(FILE_2023)
    df2 = pd.read_csv(FILE_2024)
    df3 = pd.read_csv(FILE_2025)

    for df in [df1, df2, df3]:
        df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce") #trả về NaT nếu gtri ngày ko parse đc

    common = sorted(list(set(df1.columns) & set(df2.columns) & set(df3.columns)))
    df = pd.concat([df1[common], df2[common], df3[common]], ignore_index=True)
    df = df.sort_values(["date_local", "GID_3"]).reset_index(drop=True)

    df.to_csv(OUTPUT, index=False)
    print("✅ merged saved:", OUTPUT, df.shape)

if __name__ == "__main__":
    main()