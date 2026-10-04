"""Ask-your-well agent: Gemini decides which engineering tools to call, Python runs them.
The model never calculates numbers itself; it reads tool results and explains them."""
import numpy as np
import pandas as pd

import decline
import llm

MAX_STEPS = 6

SYSTEM = """You are an assistant for a petroleum production engineer, answering questions about ONE well.
Rules:
- Use the tools to get facts. Never guess or calculate numbers yourself; quote numbers exactly as the tools return them.
- Call several tools if needed. For 'why' questions, look at the anomalies, the diagnosis and the period stats around the event.
- Give dates and units when you know them. Say clearly when the data cannot answer the question.
- Be concise: at most about 120 words, plain text, no headings."""

TOOLS = [
    {"name": "get_well_summary",
     "description": "Overall well metrics: data period, current and peak oil rate, current water cut, cumulative oil, "
                    "Arps decline parameters, EUR and remaining reserves."},
    {"name": "list_anomalies",
     "description": "All detected anomalies (rate drop, water cut rise, shut-in, pressure drop, choke change, GOR rise) "
                    "with start and end dates and details."},
    {"name": "get_diagnosis",
     "description": "Ranked likely causes for each anomaly with confidence and suggested checks."},
    {"name": "get_period_stats",
     "description": "Average oil rate, water cut, wellhead pressure, choke and on-stream hours between two dates (inclusive).",
     "parameters": {"type": "object", "properties": {
         "start_date": {"type": "string", "description": "YYYY-MM-DD"},
         "end_date": {"type": "string", "description": "YYYY-MM-DD"}},
         "required": ["start_date", "end_date"]}},
    {"name": "get_daily_values",
     "description": "Daily values (oil, water, water cut, wellhead pressure, choke, hours) for the record nearest to a date.",
     "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                    "required": ["date"]}},
    {"name": "forecast_reserves",
     "description": "Remaining reserves and years left from the fitted decline curve for a chosen economic limit rate.",
     "parameters": {"type": "object", "properties": {
         "economic_limit_rate": {"type": "number", "description": "Rate below which production stops being economic"}},
         "required": ["economic_limit_rate"]}},
]


def _clean(x):
    """Make values JSON-safe."""
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


class WellTools:
    def __init__(self, result, units="Sm3/d"):
        self.df = result["df"]
        self.m = result["metrics"]
        self.events = result["events"]
        self.findings = result["findings"]
        self.units = units

    def call(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return _clean(fn(**(args or {})))
        except Exception as e:
            return {"error": str(e)}

    # ---- tools ----
    def t_get_well_summary(self):
        m, d = self.m, self.m.get("decline")
        out = {"units": self.units, "data_start": m["start"], "data_end": m["end"], "days": m["days"],
               "current_oil_rate_7d_avg": m["current_rate"], "peak_oil_rate": m["peak_rate"], "peak_date": m["peak_date"],
               "current_water_cut_pct": m["current_wc"] * 100, "cumulative_oil": m["cum_oil"],
               "current_wellhead_pressure": m["current_whp"]}
        if d:
            out["decline"] = {"qi": d["qi"], "Di_per_day": d["di"], "b": d["b"], "r2": d["r2"],
                              "annual_decline_pct": d["annual_decline_pct"], "remaining_reserves": d["remaining"],
                              "years_to_limit": d["days_to_limit"] / 365, "eur": d["eur"],
                              "economic_limit_used": m["q_limit"]}
        return out

    def t_list_anomalies(self):
        return [{"type": e["type"], "start": e["start"], "end": e["end"], "detail": e["detail"]} for e in self.events] \
            or "No anomalies were detected."

    def t_get_diagnosis(self):
        return [{"event": f["event"]["type"], "start": f["event"]["start"], "end": f["event"]["end"],
                 "likely_causes": [{"cause": c["label"], "confidence": c["confidence"]} for c in f["candidates"][:3]],
                 "first_checks": f["candidates"][0]["checks"][:3]} for f in self.findings]

    def t_get_period_stats(self, start_date, end_date):
        d = self.df
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
        p = d[(d["date"] >= s) & (d["date"] <= e)]
        if p.empty:
            return {"error": f"no data between {start_date} and {end_date}",
                    "data_range": [str(d["date"].min().date()), str(d["date"].max().date())]}
        on = p[p["hours"] > 12]
        out = {"days": len(p), "avg_oil_rate": on["oil"].mean() if len(on) else 0.0,
               "avg_water_cut_pct": p["wc"].mean() * 100, "shut_in_days": int((p["hours"] < 2).sum())}
        for col, key in (("whp", "avg_wellhead_pressure"), ("choke", "avg_choke")):
            if col in p and p[col].notna().any():
                out[key] = p[col].mean()
        return out

    def t_get_daily_values(self, date):
        d = self.df
        t = pd.to_datetime(date)
        row = d.iloc[(d["date"] - t).abs().argmin()]
        out = {"date": row["date"], "oil": row["oil"], "water": row.get("water"), "water_cut_pct": row["wc"] * 100,
               "hours": row["hours"]}
        for col in ("whp", "choke", "gas"):
            if col in row.index and pd.notna(row[col]):
                out[col] = row[col]
        return out

    def t_forecast_reserves(self, economic_limit_rate):
        d = self.m.get("decline")
        if not d:
            return {"error": "decline curve could not be fitted for this well"}
        rem, days = decline.remaining_reserves(d, float(d["t"][-1]), float(economic_limit_rate))
        return {"economic_limit_rate": economic_limit_rate, "remaining_reserves": rem, "years_to_limit": days / 365,
                "eur": self.m["cum_oil"] + rem}


def answer(question, history, tools, well="", units=""):
    """Run the agent loop. history: Gemini contents list from earlier turns.
    Returns (text, new_history, steps) where steps lists every tool call made."""
    context = (f"[Well: {well}. Rate units: {units}. Data covers {tools.m['start'].date()} to {tools.m['end'].date()}.]\n"
               f"{question}")
    contents = list(history) + [{"role": "user", "parts": [{"text": context}]}]
    steps = []
    try:
        for _ in range(MAX_STEPS):
            content = llm.generate(contents, SYSTEM, TOOLS)
            parts = content.get("parts", [])
            contents.append({"role": "model", "parts": parts})  # keep as received (includes any thought signatures)
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not calls:
                text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                return (text or "I could not produce an answer."), contents, steps
            replies = []
            for c in calls:
                out = tools.call(c["name"], c.get("args", {}))
                steps.append({"tool": c["name"], "args": c.get("args", {}), "result": out})
                replies.append({"functionResponse": {"name": c["name"], "response": {"result": out}}})
            contents.append({"role": "user", "parts": replies})
        return "I needed too many steps for this question. Try a more specific one.", contents, steps
    except Exception as e:
        return f"The AI call failed: {e}", list(history), steps
