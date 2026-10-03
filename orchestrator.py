"""Orchestrator: runs the agents in order and emits trace events for the UI."""
import time
from datetime import datetime
from pathlib import Path
import ingestion, analysis, diagnosis, report, llm


def run_pipeline(df_well, well, q_limit=10.0, units="Sm³/d", on_event=None):
    trace = []

    def emit(agent, msg):
        trace.append({"agent": agent, "msg": msg, "t": time.strftime("%H:%M:%S")})
        if on_event:
            on_event(agent, msg)

    emit("Ingestion Agent", f"Validating {len(df_well)} rows for {well}...")
    df, quality = ingestion.run(df_well, well)
    emit("Ingestion Agent", f"Clean dataset: {quality['rows_out']} rows, {quality['start'].date()} to {quality['end'].date()}. "
                            f"Issues: {'; '.join(quality['issues'])}")

    emit("Analysis Agent", "Running decline curve fit, reserves and anomaly detection (Python tools)...")
    metrics, events = analysis.run(df, q_limit)
    d = metrics["decline"]
    emit("Analysis Agent", f"Current rate {metrics['current_rate']:.0f} {units}, WC {metrics['current_wc']*100:.0f}%. "
                           + (f"Arps b={d['b']:.2f}, Di={d['di']:.4f}/d, EUR {d['eur']:,.0f}. " if d else "Decline fit failed. ")
                           + f"{len(events)} anomalies flagged.")

    emit("Diagnosis Agent", "Scoring candidate causes against the engineering knowledge base"
                            + (" and asking the language model for interpretation..." if llm.available() else " (no API key: rule-based mode)..."))
    findings, narrative = diagnosis.run(df, events, metrics)
    top = [f"{f['event']['type']} -> {f['candidates'][0]['label']} ({f['candidates'][0]['confidence']*100:.0f}%)" for f in findings]
    emit("Diagnosis Agent", "; ".join(top))

    emit("Report Agent", "Building charts and drafting the surveillance report...")
    fig = report.make_chart(df, metrics, events, well, units)
    text, used_llm = report.run(well, metrics, findings, narrative, quality, units)
    emit("Report Agent", "Report drafted" + (" by the language model." if used_llm else " from template (no API key)."))
    emit("Human Approval", "Waiting for engineer review and approval.")

    return {"df": df, "quality": quality, "metrics": metrics, "events": events, "findings": findings,
            "narrative": narrative, "report": text, "fig": fig, "trace": trace, "used_llm": used_llm}


def save_approved(result, well, approver="Engineer", out_dir="reports"):
    """Human-in-the-loop step: only approved reports are written out."""
    Path(out_dir).mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    md = Path(out_dir) / f"{well}_{stamp}.md"
    png = Path(out_dir) / f"{well}_{stamp}.png"
    md.write_text(result["report"] + f"\n\n---\n*Approved by {approver} on {datetime.now():%Y-%m-%d %H:%M}*\n", encoding="utf-8")
    result["fig"].savefig(png, dpi=150)
    return md, png
