"""Ingestion agent: validates and cleans the selected well's data, reports data-quality issues."""
import pandas as pd


def run(df_well, well):
    issues = []
    df = df_well.copy()
    n0 = len(df)

    dup = int(df.duplicated(subset="date").sum())
    if dup:
        df = df.drop_duplicates(subset="date", keep="last")
        issues.append(f"Removed {dup} duplicate date rows")

    for col in ["oil", "water", "gas"]:
        if col in df:
            neg = int((df[col] < 0).sum())
            if neg:
                df.loc[df[col] < 0, col] = float("nan")
                issues.append(f"Set {neg} negative {col} values to missing")

    miss = int(df["oil"].isna().sum())
    if miss:
        df = df.dropna(subset=["oil"])
        issues.append(f"Dropped {miss} rows with missing oil volume")

    if "hours" not in df:
        df["hours"] = 24.0
    df["hours"] = df["hours"].fillna(24.0)

    df = df.sort_values("date").reset_index(drop=True)
    span = (df["date"].max() - df["date"].min()).days + 1
    gaps = span - len(df)
    if gaps > 0:
        issues.append(f"{gaps} calendar days have no record (gaps in reporting)")

    return df, {"well": well, "rows_in": n0, "rows_out": len(df),
                "start": df["date"].min(), "end": df["date"].max(),
                "columns": [c for c in ["oil", "water", "gas", "whp", "bhp", "choke", "hours"] if c in df and df[c].notna().any()],
                "issues": issues or ["No data-quality problems found"]}
