"""Decline curve analysis (Arps). All numbers in this project come from here, not from the LLM."""
import numpy as np
from scipy.optimize import curve_fit


def arps(t, qi, di, b):
    """Arps rate at time t (days). di in 1/day."""
    t = np.asarray(t, dtype=float)
    b = max(b, 1e-6)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def select_decline_segment(dates, q, min_points=30, window_days=180):
    """Use the most recent `window_days` of positive-rate data as the decline segment."""
    dates = np.asarray(dates, dtype="datetime64[ns]")
    q = np.asarray(q, dtype=float)
    mask = q > 0
    d, qq = dates[mask], q[mask]
    if len(qq) < min_points:
        return None, None
    d, qq = d[-window_days:], qq[-window_days:]
    t = (d - d[0]).astype("timedelta64[D]").astype(float)
    return t, qq


def fit_decline(t, q):
    """Fit Arps hyperbolic decline. Returns dict with qi, di, b, r2 or None if it fails."""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(q) < 10:
        return None
    p0 = [q[0], 0.005, 0.5]
    bounds = ([q.max() * 0.2, 1e-6, 0.01], [q.max() * 3, 1.0, 1.5])
    try:
        popt, _ = curve_fit(arps, t, q, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    pred = arps(t, *popt)
    ss_res = float(np.sum((q - pred) ** 2))
    ss_tot = float(np.sum((q - q.mean()) ** 2)) or 1.0
    return {"qi": float(popt[0]), "di": float(popt[1]), "b": float(popt[2]), "r2": 1 - ss_res / ss_tot}


def forecast(params, days=365):
    """Forecast from the start of the fitted segment; returns (t, q)."""
    t = np.arange(0, days)
    return t, arps(t, params["qi"], params["di"], params["b"])


def remaining_reserves(params, t_now, q_limit=10.0, max_days=365 * 30):
    """Integrate the fitted decline from t_now to the economic limit rate.
    Returns (remaining volume, days to limit)."""
    t = np.arange(t_now, t_now + max_days)
    q = arps(t, params["qi"], params["di"], params["b"])
    below = np.where(q < q_limit)[0]
    end = below[0] if len(below) else max_days
    if end < 2:
        return 0.0, 0
    return float(np.sum(q[:end])), int(end)
