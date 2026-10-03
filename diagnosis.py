"""Diagnosis agent: rule-based scoring against the knowledge base, optionally explained by the language model."""
import json
import pandas as pd
import llm
from knowledge import CAUSES

SYSTEM = """You are a senior production/reservoir engineer reviewing well surveillance results.
You are given detected anomalies with ranked candidate causes computed by deterministic rules.
Write a concise engineering interpretation for each anomaly: what most likely happened, why (cite only the
numbers provided), what you are NOT sure about, and the first checks to do. NEVER invent numbers, dates or
values that are not in the input. Plain text, short paragraphs, no markdown headers."""


def _near(events, etype, date, days=7):
    return [e for e in events if e["type"] == etype and abs((e["start"] - date).days) <= days]


def _wc_shift(df, date, span=14):
    before = df[(df["date"] < date) & (df["date"] >= date - pd.Timedelta(days=span))]["wc"].mean()
    after = df[(df["date"] >= date) & (df["date"] < date + pd.Timedelta(days=span))]["wc"].mean()
    return (after - before) if pd.notna(before) and pd.notna(after) else 0.0


def _cand(key, conf):
    c = CAUSES[key]
    return {"cause": key, "label": c["label"], "confidence": round(conf, 2), "checks": c["checks"]}


def rule_based(df, events):
    findings = []
    for e in events:
        cands = []
        t = e["type"]
        if t == "shut_in":
            cands = [_cand("shut_in", 0.9)]
        elif t == "watercut_rise":
            whp_drop = _near(events, "pressure_drop", e["end"], 30)
            cands = [_cand("water_breakthrough", 0.55 if whp_drop else 0.8),
                     _cand("wellbore_issue", 0.1)]
        elif t == "gor_rise":
            cands = [_cand("gas_breakthrough", 0.75), _cand("reservoir_depletion", 0.2)]
        elif t == "choke_change":
            rd = _near(events, "rate_drop", e["start"], 5)
            cands = [_cand("choke_change", 0.7 if rd else 0.4)]
        elif t == "rate_drop":
            pd_ev = _near(events, "pressure_drop", e["start"], 5)
            ch_ev = _near(events, "choke_change", e["start"], 5)
            wc_up = _wc_shift(df, e["start"]) > 0.05
            if ch_ev:
                cands = [_cand("choke_change", 0.75), _cand("artificial_lift_failure", 0.2)]
            elif pd_ev and not wc_up:
                cands = [_cand("artificial_lift_failure", 0.75), _cand("wellbore_issue", 0.2)]
            elif wc_up:
                cands = [_cand("water_breakthrough", 0.65), _cand("artificial_lift_failure", 0.2)]
            else:
                cands = [_cand("wellbore_issue", 0.45), _cand("reservoir_depletion", 0.3)]
        elif t == "pressure_drop":
            cands = [_cand("artificial_lift_failure", 0.5), _cand("wellbore_issue", 0.3)]
        cands.sort(key=lambda c: -c["confidence"])
        findings.append({"event": e, "candidates": cands})
    if not findings:
        findings.append({"event": {"type": "none", "start": df["date"].min(), "end": df["date"].max(),
                                   "magnitude": 0, "detail": "No anomalies detected"},
                         "candidates": [_cand("reservoir_depletion", 0.9)]})
    return findings


def run(df, events, metrics):
    findings = rule_based(df, events)
    payload = [{"type": f["event"]["type"], "start": str(f["event"]["start"].date()),
                "end": str(f["event"]["end"].date()), "detail": f["event"]["detail"],
                "candidates": [{"cause": c["label"], "confidence": c["confidence"], "signature": CAUSES[c["cause"]]["signature"]}
                               for c in f["candidates"]]} for f in findings]
    context = {"current_rate": round(metrics["current_rate"], 1), "current_water_cut_pct": round(metrics["current_wc"] * 100, 1),
               "anomalies": payload}
    narrative = llm.ask(SYSTEM, json.dumps(context, default=str))
    return findings, narrative
