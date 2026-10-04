"""WellCopilot: multi-agent well surveillance and formation evaluation. Run with:  streamlit run app.py"""
import io

import pandas as pd
import streamlit as st

import formation, llm
from orchestrator import run_pipeline, save_approved
import data_loader as dl
from las_reader import read_las

st.set_page_config(page_title="WellCopilot", page_icon="🛢️", layout="wide")
st.title("🛢️ WellCopilot")
st.caption("Multi-agent copilot for petroleum engineers. All numbers come from Python engineering tools; "
           "the language model only interprets and writes.")

# ---------------- sidebar ----------------
with st.sidebar:
    module = st.radio("Module", ["Well surveillance", "Formation evaluation"])
    st.divider()
    ai_on = llm.available()
    st.write("AI writer: " + ("🟢 Gemini connected" if ai_on else "🟡 no API key (template text)"))
    if ai_on and st.button("Test AI connection", use_container_width=True):
        reply = llm.ask("Reply with one short sentence.", "Say that the connection works.", max_tokens=50)
        if reply:
            st.success(reply)
        else:
            st.error(f"AI call failed, template text will be used. {llm.last_error()}")
    st.divider()


# =====================================================================
# MODULE 1: WELL SURVEILLANCE
# =====================================================================
def surveillance():
    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", ["Demo field (synthetic)", "Upload Volve / CSV / Excel"])
        raw = None
        if source.startswith("Demo"):
            raw = dl.load_production("demo_field_production.csv")
        else:
            up = st.file_uploader("Production data (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
            if up:
                try:
                    raw = dl.load_production(up)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
        well = None
        if raw is not None and len(raw):
            well = st.selectbox("Well", dl.list_wells(raw))
        st.header("Settings")
        units = st.text_input("Rate units", "Sm³/d")
        q_limit = st.number_input("Economic limit rate", min_value=1.0, value=10.0, step=1.0)
        run = st.button("▶ Run agents", type="primary", disabled=well is None, use_container_width=True)

    if run and well:
        with st.status("Agents working...", expanded=True) as status:
            def show(agent, msg):
                st.write(f"**{agent}**: {msg}")
            result = run_pipeline(dl.get_well(raw, well), well, q_limit=q_limit, units=units, on_event=show)
            status.update(label="Agents finished. Awaiting your approval.", state="complete")
        st.session_state["result"] = result
        st.session_state["well"] = well
        st.session_state["units"] = units
        st.session_state["report_text"] = result["report"]
        st.session_state.pop("saved", None)

    result = st.session_state.get("result")
    if result is None:
        st.info("Pick a well in the sidebar and press **Run agents**. The demo field has one well with "
                "planted problems and one healthy well.")
        return

    m, well, units = result["metrics"], st.session_state["well"], st.session_state.get("units", units)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current oil rate", f"{m['current_rate']:,.0f} {units}",
              delta=f"{(m['current_rate']/m['peak_rate']-1)*100:.0f}% vs peak")
    c2.metric("Water cut", f"{m['current_wc']*100:.0f} %")
    c3.metric("Cumulative oil", f"{m['cum_oil']:,.0f}")
    d = m["decline"]
    c4.metric("EUR (Arps)", f"{d['eur']:,.0f}" if d else "n/a",
              delta=f"b = {d['b']:.2f}" if d else None, delta_color="off")

    tab_chart, tab_events, tab_diag, tab_report, tab_trace = st.tabs(
        ["📈 Chart", "🚨 Anomalies", "🩺 Diagnosis", "📝 Report & approval", "🤖 Agent trace"])

    with tab_chart:
        st.pyplot(result["fig"])

    with tab_events:
        if result["events"]:
            st.dataframe(pd.DataFrame([{"Type": e["type"].replace("_", " "), "Start": e["start"].date(),
                                        "End": e["end"].date(), "Detail": e["detail"]} for e in result["events"]]),
                         use_container_width=True, hide_index=True)
        else:
            st.success("No anomalies detected.")

    with tab_diag:
        for f in result["findings"]:
            e = f["event"]
            with st.expander(f"{e['type'].replace('_', ' ').title()}  ({e['start'].date()} → {e['end'].date()})",
                             expanded=True):
                st.write(e["detail"])
                for cnd in f["candidates"]:
                    st.progress(min(cnd["confidence"], 1.0), text=f"{cnd['label']}: {cnd['confidence']*100:.0f}%")
                st.markdown("**Suggested checks**\n" + "\n".join(f"- {x}" for x in f["candidates"][0]["checks"]))
        if result["narrative"]:
            st.subheader("Engineering interpretation")
            st.write(result["narrative"])

    with tab_report:
        st.caption("Edit the draft if needed, then approve. Nothing is saved or sent until you approve.")
        st.session_state["report_text"] = st.text_area("Draft report (Markdown)", st.session_state["report_text"],
                                                       height=420)
        a, b = st.columns([1, 2])
        approver = a.text_input("Approved by", "Petroleum Engineer")
        if b.button("✅ Approve & save", type="primary"):
            result["report"] = st.session_state["report_text"]
            st.session_state["saved"] = save_approved(result, well, approver)
        if "saved" in st.session_state:
            st.success(f"Saved: {st.session_state['saved'][0]}")
        st.download_button("⬇ Download report (.md)", st.session_state["report_text"], file_name=f"{well}_report.md")
        st.markdown("---")
        st.markdown(st.session_state["report_text"])

    with tab_trace:
        for t in result["trace"]:
            st.write(f"`{t['t']}` **{t['agent']}**: {t['msg']}")


# =====================================================================
# MODULE 2: FORMATION EVALUATION (well logs)
# =====================================================================
def formation_eval():
    with st.sidebar:
        st.header("Well log")
        src = st.radio("Log source", ["Demo log (synthetic)", "Upload LAS file"])
        st.header("Parameters")
        rw = st.number_input("Rw (ohm.m)", min_value=0.005, max_value=2.0, value=0.05, step=0.005, format="%.3f")
        vsh_cut = st.slider("Vsh cutoff", 0.1, 0.8, 0.40, 0.05)
        phi_cut = st.slider("Porosity cutoff", 0.02, 0.20, 0.08, 0.01)
        sw_cut = st.slider("Sw cutoff", 0.2, 0.9, 0.60, 0.05)
        go = st.button("▶ Evaluate formation", type="primary", use_container_width=True)

    df, info, name = None, {}, "DEMO-WELL-LOG-1"
    if src.startswith("Demo"):
        df, info = read_las("demo_well.las")
    else:
        up = st.sidebar.file_uploader("LAS file (.las)", type=["las", "LAS", "txt"])
        if up:
            try:
                df, info = read_las(io.StringIO(up.getvalue().decode("utf-8", errors="ignore")))
                name = info.get("WELL") or up.name
            except Exception as e:
                st.error(f"Could not read LAS file: {e}")

    if df is None:
        st.info("Upload a LAS 2.0 file (unwrapped) or use the demo log, then press **Evaluate formation**.")
        return
    if go:
        with st.status("Formation evaluation agent working...", expanded=False) as status:
            try:
                out = formation.run(df, rw=rw, vsh_cut=vsh_cut, phi_cut=phi_cut, sw_cut=sw_cut, well=name)
            except KeyError as e:
                st.error(str(e).strip("'"))
                return
            status.update(label="Done", state="complete")
        st.session_state["fe"] = (name,) + out

    fe = st.session_state.get("fe")
    if not fe:
        st.write(f"Loaded **{name}**: {len(df)} depth points, curves: {', '.join(df.columns)}.")
        st.info("Press **Evaluate formation** in the sidebar.")
        return
    name, res, s, fig, text, used = fe
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross interval", f"{s['gross']:.0f} m")
    c2.metric("Net pay", f"{s['net_pay']:.1f} m", delta=f"N/G {s['net_to_gross']*100:.0f}%", delta_color="off")
    c3.metric("Avg porosity (pay)", f"{s['avg_phi_pay']*100:.1f} %" if s["avg_phi_pay"] is not None else "n/a")
    c4.metric("Avg Sw (pay)", f"{s['avg_sw_pay']*100:.0f} %" if s["avg_sw_pay"] is not None else "n/a")

    t1, t2, t3, t4 = st.tabs(["📉 Log plot", "🎯 Pay zones", "📝 Summary", "📋 Data"])
    with t1:
        st.pyplot(fig)
    with t2:
        if s["zones"]:
            st.dataframe(pd.DataFrame([{"Top (m)": round(z["top"], 1), "Base (m)": round(z["base"], 1),
                                        "Thickness (m)": round(z["thickness"], 1), "Porosity (%)": round(z["phi"] * 100, 1),
                                        "Sw (%)": round(z["sw"] * 100, 0), "Vsh (%)": round(z["vsh"] * 100, 0)}
                                       for z in s["zones"]]), use_container_width=True, hide_index=True)
        else:
            st.warning("No zones passed the cutoffs.")
        st.caption("Parameters: " + ", ".join(f"{k} = {v}" for k, v in s["params"].items()))
    with t3:
        st.write(text)
        st.caption("Written by the AI writer from the computed numbers." if used else "Template summary (no AI key).")
    with t4:
        st.dataframe(res.round(3), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download results (.csv)", res.to_csv(index=False), file_name=f"{name}_petrophysics.csv")


if module == "Well surveillance":
    surveillance()
else:
    formation_eval()
