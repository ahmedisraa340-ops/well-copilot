"""Report agent: builds the chart and the management-ready daily/weekly report."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import llm
import decline

COLORS = {"rate_drop": "#d62728", "watercut_rise": "#1f77b4", "shut_in": "#7f7f7f",
          "pressure_drop": "#ff7f0e", "choke_change": "#9467bd", "gor_rise": "#2ca02c"}

SYSTEM = """You are a petroleum engineer writing a well surveillance report for management.
Use ONLY the numbers in the JSON provided; never invent or recompute numbers. Write in Markdown with these sections:
## Executive summary (3-4 sentences), ## Key metrics (bullet list), ## Findings and likely causes,
## Recommended actions (numbered, prioritised), ## Data quality notes. Be concise and specific. Mark uncertainty honestly."""


def make_chart(df, metrics, events, well, units="Sm³/d"):
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
    ax[0].plot(df["date"], df["oil"], color="#2b7a3d", lw=1.2, label="Oil rate")
    d = metrics.get("decline")
    if d:
        seg_dates = df["date"].iloc[-len(d["t"]):]
        fit = decline.arps(d["t"], d["qi"], d["di"], d["b"])
        ax[0].plot(seg_dates, fit, "k--", lw=1.5, label=f"Arps fit (b={d['b']:.2f}, R²={d['r2']:.2f})")
        fdays = np.arange(1, 366)
        fq = decline.arps(d["t"][-1] + fdays, d["qi"], d["di"], d["b"])
        ax[0].plot(df["date"].iloc[-1] + pd.to_timedelta(fdays, "D"), fq, color="k", ls=":", label="12-month forecast")
    ax[0].set_ylabel(f"Oil rate ({units})")
    ax[0].set_title(f"{well}: production surveillance")
    ax[1].plot(df["date"], df["wc"] * 100, color="#1f77b4")
    ax[1].set_ylabel("Water cut (%)")
    if "whp" in df and df["whp"].notna().any():
        ax[2].plot(df["date"], df["whp"], color="#ff7f0e")
    ax[2].set_ylabel("WHP (bar)")
    seen = set()
    for e in events:
        for a in ax:
            a.axvspan(e["start"] - pd.Timedelta(days=1), e["end"] + pd.Timedelta(days=1),
                      color=COLORS.get(e["type"], "grey"), alpha=0.25,
                      label=e["type"].replace("_", " ") if (a is ax[0] and e["type"] not in seen) else None)
        seen.add(e["type"])
    ax[0].legend(loc="upper right", fontsize=8, ncol=2)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _fmt(x, nd=0):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def _template(well, metrics, findings, quality, units):
    d = metrics.get("decline")
    lines = [f"# Well Surveillance Report: {well}",
             f"*Data period: {metrics['start'].date()} to {metrics['end'].date()}*", "",
             "## Executive summary",
             f"{well} is producing about {_fmt(metrics['current_rate'])} {units} of oil at {_fmt(metrics['current_wc']*100)}% water cut "
             f"(peak {_fmt(metrics['peak_rate'])} {units} on {metrics['peak_date'].date()}). "
             f"{len([f for f in findings if f['event']['type'] != 'none'])} anomalies were detected; see findings below.", "",
             "## Key metrics",
             f"- Current oil rate (7-day avg): {_fmt(metrics['current_rate'])} {units}",
             f"- Current water cut: {_fmt(metrics['current_wc']*100, 1)} %",
             f"- Cumulative oil: {_fmt(metrics['cum_oil'])}"]
    if d:
        lines += [f"- Arps decline: qi={_fmt(d['qi'])}, Di={d['di']:.4f}/day, b={d['b']:.2f}, R²={d['r2']:.2f} "
                  f"(~{d['annual_decline_pct']:.0f}%/yr effective)",
                  f"- Remaining reserves to {metrics['q_limit']:.0f} {units} limit: {_fmt(d['remaining'])} "
                  f"(~{d['days_to_limit']/365:.1f} years); EUR: {_fmt(d['eur'])}"]
    lines += ["", "## Findings and likely causes"]
    for f in findings:
        e = f["event"]
        lines.append(f"**{e['type'].replace('_',' ').title()}** ({e['start'].date()} to {e['end'].date()}): {e['detail']}")
        for c in f["candidates"][:2]:
            lines.append(f"- {c['label']} (confidence {c['confidence']*100:.0f}%)")
        lines.append("")
    lines += ["## Recommended actions"]
    n = 1
    seen = set()
    for f in findings:
        for c in f["candidates"][:1]:
            for chk in c["checks"][:2]:
                if chk not in seen:
                    lines.append(f"{n}. {chk}")
                    seen.add(chk)
                    n += 1
    lines += ["", "## Data quality notes"] + [f"- {i}" for i in quality["issues"]]
    return "\n".join(lines)


def run(well, metrics, findings, narrative, quality, units="Sm³/d"):
    d = metrics.get("decline") or {}
    payload = {
        "well": well, "period": f"{metrics['start'].date()} to {metrics['end'].date()}", "units": units,
        "current_oil_rate": round(metrics["current_rate"], 1), "peak_rate": round(metrics["peak_rate"], 1),
        "peak_date": str(metrics["peak_date"].date()), "current_water_cut_pct": round(metrics["current_wc"] * 100, 1),
        "cum_oil": round(metrics["cum_oil"]), "decline": {k: (round(v, 4) if isinstance(v, float) else v)
                                                        for k, v in d.items() if k in ("qi", "di", "b", "r2", "remaining", "eur", "days_to_limit", "annual_decline_pct")},
        "findings": [{"type": f["event"]["type"], "start": str(f["event"]["start"].date()), "end": str(f["event"]["end"].date()),
                      "detail": f["event"]["detail"],
                      "candidates": [{"cause": c["label"], "confidence": c["confidence"], "checks": c["checks"]} for c in f["candidates"]]}
                     for f in findings],
        "engineer_interpretation": narrative, "data_quality": quality["issues"],
    }
    text = llm.ask(SYSTEM, json.dumps(payload, default=str), max_tokens=1800)
    used_llm = text is not None
    return (text or _template(well, metrics, findings, quality, units)), used_llm
