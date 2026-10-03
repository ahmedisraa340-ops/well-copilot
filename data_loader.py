"""Load production data (Equinor Volve Excel/CSV or generic CSV) and generate synthetic demo wells."""
import numpy as np
import pandas as pd

# Volve column names -> our standard names
COLUMN_MAP = {
    "DATEPRD": "date", "DATE": "date",
    "NPD_WELL_BORE_NAME": "well", "WELL": "well", "WELL_NAME": "well",
    "ON_STREAM_HRS": "hours",
    "AVG_DOWNHOLE_PRESSURE": "bhp", "AVG_WHP_P": "whp", "AVG_CHOKE_SIZE_P": "choke",
    "BORE_OIL_VOL": "oil", "BORE_GAS_VOL": "gas", "BORE_WAT_VOL": "water",
    "OIL": "oil", "GAS": "gas", "WATER": "water",
}


def read_any(file):
    """Read an uploaded file-like object or path (xlsx / xls / csv)."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(file)
        sheet = next((s for s in xl.sheet_names if "daily" in s.lower()), xl.sheet_names[0])
        return xl.parse(sheet)
    return pd.read_csv(file)


def standardize(raw):
    """Rename columns, keep producers, sort by date, compute water cut and GOR."""
    df = raw.rename(columns={c: COLUMN_MAP[c.upper()] for c in raw.columns if c.upper() in COLUMN_MAP})
    if "date" not in df or "oil" not in df:
        raise ValueError("Data needs at least a date column and an oil volume column "
                         "(Volve: DATEPRD and BORE_OIL_VOL).")
    if "well" not in df:
        df["well"] = "WELL-1"
    if "FLOW_KIND" in raw.columns:  # Volve: keep production rows only
        df = df[raw["FLOW_KIND"].astype(str).str.lower() == "production"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["oil", "gas", "water", "hours", "bhp", "whp", "choke"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "water" not in df:
        df["water"] = 0.0
    if "gas" not in df:
        df["gas"] = np.nan
    liquid = df["oil"].fillna(0) + df["water"].fillna(0)
    df["wc"] = np.where(liquid > 0, df["water"].fillna(0) / liquid, np.nan)
    df["gor"] = np.where(df["oil"] > 0, df["gas"] / df["oil"], np.nan)
    return df.sort_values(["well", "date"]).reset_index(drop=True)


def load_production(file):
    return standardize(read_any(file))


def list_wells(df):
    return sorted(df["well"].dropna().unique().tolist())


def get_well(df, well):
    return df[df["well"] == well].sort_values("date").reset_index(drop=True)


def generate_synthetic_well(seed=7, days=900, name="DEMO-WELL-A"):
    """A realistic-looking demo well with three planted events:
       1) water breakthrough (day ~400), 2) sudden rate drop with WHP fall (day ~640, ESP-type failure),
       3) a short shut-in (day ~780)."""
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    qi, di, b = 2200.0, 0.006, 0.6
    oil = qi / np.power(1 + b * di * t, 1 / b)
    wc = 0.05 + 0.02 * t / days
    wc = wc + np.clip((t - 400) / 250, 0, 1) * 0.35          # water breakthrough
    hours = np.full(days, 24.0)
    whp = 55 - 0.008 * t + rng.normal(0, 0.6, days)
    choke = np.full(days, 48.0)
    choke[(t >= 250)] = 52.0
    liquid = oil / (1 - wc)
    oil = oil * (1 - np.clip((t - 400) / 250, 0, 1) * 0.10)  # water reduces oil
    oil[t >= 640] *= 0.55                                    # ESP-type failure
    whp[t >= 640] -= 9
    oil[780:784] = 0
    hours[780:784] = 0
    whp[780:784] = np.nan
    oil = oil * (1 + rng.normal(0, 0.015, days))
    water = oil * wc / (1 - wc)
    gas = oil * (150 + 0.03 * t) * (1 + rng.normal(0, 0.02, days))
    dates = pd.date_range("2022-01-01", periods=days)
    raw = pd.DataFrame({"DATEPRD": dates, "NPD_WELL_BORE_NAME": name, "ON_STREAM_HRS": hours,
                        "AVG_WHP_P": whp, "AVG_CHOKE_SIZE_P": choke, "BORE_OIL_VOL": oil.clip(0),
                        "BORE_GAS_VOL": gas * (hours > 0), "BORE_WAT_VOL": water * (hours > 0),
                        "FLOW_KIND": "production"})
    return raw


def generate_demo_field():
    """Two wells: one with problems, one healthy (for showing generalization)."""
    a = generate_synthetic_well(7, 900, "DEMO-WELL-A")
    rng = np.random.default_rng(3)
    t = np.arange(900)
    healthy = a.copy()
    healthy["NPD_WELL_BORE_NAME"] = "DEMO-WELL-B"
    q = 1500 / np.power(1 + 0.5 * 0.004 * t, 1 / 0.5) * (1 + rng.normal(0, 0.01, 900))
    healthy["BORE_OIL_VOL"] = q
    healthy["BORE_WAT_VOL"] = q * 0.08 / 0.92
    healthy["BORE_GAS_VOL"] = q * 140
    healthy["ON_STREAM_HRS"] = 24.0
    healthy["AVG_WHP_P"] = 60 - 0.006 * t + rng.normal(0, 0.5, 900)
    healthy["AVG_CHOKE_SIZE_P"] = 48.0
    return pd.concat([a, healthy], ignore_index=True)


if __name__ == "__main__":
    generate_demo_field().to_csv("demo_field_production.csv", index=False)
    print("Wrote data/demo_field_production.csv")
