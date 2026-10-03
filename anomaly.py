"""Rule-based anomaly detection on daily production data."""
import numpy as np


def _merge_events(events, gap_days=3):
    """Merge same-type events that are close in time into single events."""
    events = sorted(events, key=lambda e: (e["type"], e["start"]))
    merged = []
    for e in events:
        if merged and merged[-1]["type"] == e["type"] and (e["start"] - merged[-1]["end"]).days <= gap_days:
            m = merged[-1]
            m["end"] = max(m["end"], e["end"])
            if e["magnitude"] > m["magnitude"]:
                m["magnitude"], m["detail"] = e["magnitude"], e["detail"]
        else:
            merged.append(dict(e))
    return sorted(merged, key=lambda e: e["start"])


def detect_rate_drops(df, window=14, threshold=0.20, persist=3):
    """Oil rate stays > threshold below its prior rolling mean for `persist` days."""
    q = df["oil"].astype(float)
    base = q.rolling(window).mean().shift(1)
    drop = (base - q) / base
    on = (df["hours"] > 12) if "hours" in df else True
    sustained = ((drop > threshold) & on).rolling(persist).sum() >= persist
    events = []
    for i in np.where(sustained.fillna(False))[0]:
        events.append({"type": "rate_drop", "start": df["date"].iloc[i - persist + 1], "end": df["date"].iloc[i],
                       "magnitude": float(drop.iloc[i]),
                       "detail": f"Oil rate {drop.iloc[i]*100:.0f}% below prior {window}-day average"})
    return _merge_events(events)


def detect_watercut_rise(df, window=30, slope_per_month=0.03):
    """Water cut rising faster than `slope_per_month` (fraction per 30 days)."""
    if "wc" not in df:
        return []
    wc = df["wc"].astype(float)
    slope = (wc - wc.shift(window)) / window * 30
    events = []
    for i in np.where((slope > slope_per_month).fillna(False))[0]:
        events.append({"type": "watercut_rise", "start": df["date"].iloc[max(i - window + 1, 0)],
                       "end": df["date"].iloc[i], "magnitude": float(slope.iloc[i]),
                       "detail": f"Water cut rising ~{slope.iloc[i]*100:.1f} pts/month (now {wc.iloc[i]*100:.0f}%)"})
    return _merge_events(events, gap_days=10)


def detect_shutins(df, hours_min=2):
    if "hours" not in df:
        return []
    events = [{"type": "shut_in", "start": r["date"], "end": r["date"], "magnitude": 1.0,
               "detail": "Well shut-in / not on stream"} for _, r in df[df["hours"] < hours_min].iterrows()]
    return _merge_events(events, gap_days=1)


def detect_pressure_drop(df, col="whp", window=14, threshold=0.15, persist=3):
    if col not in df or df[col].notna().sum() < window * 2:
        return []
    p = df[col].astype(float)
    base = p.rolling(window).mean().shift(1)
    drop = (base - p) / base
    sustained = (drop > threshold).rolling(persist).sum() >= persist
    events = []
    for i in np.where(sustained.fillna(False))[0]:
        events.append({"type": "pressure_drop", "start": df["date"].iloc[i - persist + 1], "end": df["date"].iloc[i],
                       "magnitude": float(drop.iloc[i]),
                       "detail": f"Wellhead pressure {drop.iloc[i]*100:.0f}% below prior average"})
    return _merge_events(events)


def detect_choke_change(df, threshold=0.05):
    if "choke" not in df or df["choke"].notna().sum() < 10:
        return []
    c = df["choke"].astype(float)
    chg = (c - c.shift(1)).abs() / c.shift(1).replace(0, np.nan)
    events = []
    for i in np.where((chg > threshold).fillna(False))[0]:
        events.append({"type": "choke_change", "start": df["date"].iloc[i], "end": df["date"].iloc[i],
                       "magnitude": float(chg.iloc[i]), "detail": f"Choke changed {c.iloc[i-1]:.1f} -> {c.iloc[i]:.1f}"})
    return _merge_events(events, gap_days=2)


def detect_gor_rise(df, window=30, factor=1.5):
    if "gor" not in df or df["gor"].notna().sum() < window * 2:
        return []
    g = df["gor"].astype(float)
    base = g.rolling(window).mean().shift(window)
    events = []
    for i in np.where(((g / base) > factor).fillna(False))[0]:
        events.append({"type": "gor_rise", "start": df["date"].iloc[i], "end": df["date"].iloc[i],
                       "magnitude": float(g.iloc[i] / base.iloc[i]),
                       "detail": f"GOR {g.iloc[i]/base.iloc[i]:.1f}x its earlier level"})
    return _merge_events(events, gap_days=10)


def run_all(df):
    events = []
    for fn in (detect_rate_drops, detect_watercut_rise, detect_shutins,
               detect_pressure_drop, detect_choke_change, detect_gor_rise):
        events += fn(df)
    return sorted(events, key=lambda e: e["start"])
