"""Analysis agent: deterministic engineering calculations (no LLM)."""
import numpy as np
import decline, anomaly


def run(df, q_limit=10.0):
    on = df[df["hours"] > 12]
    last = on.tail(7)
    peak_idx = df["oil"].idxmax()
    cum_oil = float(df["oil"].sum())
    cum_water = float(df["water"].fillna(0).sum())

    m = {
        "start": df["date"].min(), "end": df["date"].max(), "days": len(df),
        "peak_rate": float(df["oil"].max()), "peak_date": df.loc[peak_idx, "date"],
        "current_rate": float(last["oil"].mean()) if len(last) else 0.0,
        "current_wc": float(last["wc"].mean()) if len(last) else float("nan"),
        "cum_oil": cum_oil, "cum_water": cum_water,
        "current_whp": float(last["whp"].mean()) if "whp" in last and last["whp"].notna().any() else None,
        "q_limit": q_limit,
    }

    t, q = decline.select_decline_segment(df["date"].values, df["oil"].values)
    m["decline"] = None
    if t is not None:
        # Fit only on data after the last detected step-change so the fit reflects current behaviour
        params = decline.fit_decline(t, q)
        if params:
            t_now = float(t[-1])
            rem, days_left = decline.remaining_reserves(params, t_now, q_limit)
            m["decline"] = {**params, "seg_start": df["date"].iloc[-len(t):].iloc[0] if len(t) <= len(df) else df["date"].iloc[0],
                            "t": t, "q": q, "remaining": rem, "days_to_limit": days_left,
                            "eur": cum_oil + rem,
                            "annual_decline_pct": float((1 - np.exp(-params["di"] * 365)) * 100)}

    events = anomaly.run_all(df)
    return m, events
