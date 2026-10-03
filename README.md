# WellCopilot: Multi-Agent Well Surveillance & Formation Evaluation

Agentic AI copilot for petroleum engineers with two modules:

1. **Well surveillance**: agents ingest production data, run decline and anomaly analysis, diagnose likely
   causes and draft a management-ready report with **human approval**.
2. **Formation evaluation**: upload a LAS log, get Vsh, porosity, Archie Sw, net pay zones, a log plot and a summary.

**Design rule:** every number comes from Python engineering tools (Arps fit, EUR, water cut, anomaly rules,
Archie). The language model (Gemini) only interprets and writes, so it cannot invent engineering values.

## Architecture
```
Upload data → Ingestion Agent → Analysis Agent (Python tools)
            → Diagnosis Agent (rules + knowledge base + LLM) → Report Agent (LLM)
            → Human approval → saved report (.md + chart .png)

Upload LAS  → Formation Agent (Vsh, porosity, Sw, net pay) → log plot + summary
```

| Agent | Job | Files |
|---|---|---|
| Ingestion | Validate/clean data, report quality issues | `./ingestion.py` |
| Analysis | Arps decline, EUR, water cut, anomaly detection | `./analysis.py`, `./` |
| Diagnosis | Score causes against the knowledge base, LLM explains | `./diagnosis.py`, `./knowledge.py` |
| Report | Chart + report writing | `./report.py` |
| Formation | LAS interpretation, pay zones, log plot | `./formation.py`, `./petrophysics.py` |
| Orchestrator | Runs surveillance agents in order, emits trace | `orchestrator.py` |

## Setup
```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # paste your Google AI Studio key as GOOGLE_API_KEY (optional)
streamlit run app.py
```
Without a key everything still works; the text is generated from templates instead of the LLM.
Use **Test AI connection** in the sidebar to check the key. Model is set by `GEMINI_MODEL` (default `gemini-2.5-flash`).

## Using real data
- **Production:** download the Volve production Excel, pick **Upload Volve / CSV / Excel** and choose a well.
  Columns `DATEPRD, BORE_OIL_VOL, BORE_WAT_VOL, BORE_GAS_VOL, ON_STREAM_HRS, AVG_WHP_P, AVG_CHOKE_SIZE_P` are mapped automatically.
- **Logs:** choose **Upload LAS file** in the Formation evaluation module (LAS 2.0, one line per depth).
  Curve names are matched from lists in `./petrophysics.py` (`CURVE_NAMES`); add yours if needed.

## Demo script (3 minutes)
1. Problem: engineers spend hours pulling data, checking logs and writing reports.
2. Surveillance: run the demo well, show the agent trace live.
3. Chart: water breakthrough, then the sudden rate drop with the WHP fall.
4. Diagnosis: artificial lift failure, with the checks an engineer would do.
5. Edit one line in the report, press **Approve**, show the saved file.
6. Run the healthy well: no false alarms.
7. Switch to Formation evaluation: run the demo log, show the pay zones and the water leg excluded.
8. Close with the metric: "hours to minutes".

## Tuning
Anomaly thresholds: `./anomaly.py`. Failure modes and checks: `./knowledge.py`.
Petrophysical parameters (Rw, cutoffs) are in the sidebar; Archie a, m, n are in `./petrophysics.py`.
