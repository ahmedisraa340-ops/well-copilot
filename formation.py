"""Formation evaluation agent: log interpretation (Python tools) + written summary."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import llm
import petrophysics

SYSTEM = """You are a senior petrophysicist writing a short formation evaluation summary.
Use ONLY the numbers in the JSON provided; never invent values. Write plain text in 3 short paragraphs:
1) overall picture, 2) the pay zones and how good they look, 3) risks, uncertainties and recommended next steps
(for example core calibration of Rw, checking bad-hole intervals, testing the best zone)."""


def make_log_plot(res, summary, well="Well"):
    d = res["DEPT"].values
    fig, ax = plt.subplots(1, 4, figsize=(11, 9), sharey=True)
    for a in ax:
        for z in summary["zones"]:
            a.axhspan(z["top"], z["base"], color="#2ca02c", alpha=0.18, lw=0)
        a.grid(alpha=0.3)

    ax[0].plot(res["GR"], d, color="#2b7a3d", lw=0.9)
    ax[0].set_xlim(0, 150)
    ax[0].set_xlabel("GR (API)")
    ax[0].fill_betweenx(d, 0, res["VSH"] * 150, color="#8c6d31", alpha=0.25)

    ax[1].plot(res["RT"], d, color="#d62728", lw=0.9)
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.2, 2000)
    ax[1].set_xlabel("Deep resistivity (ohm.m)")

    ax[2].plot(res["RHOB"], d, color="#1f77b4", lw=0.9, label="RHOB")
    ax[2].set_xlim(1.95, 2.95)
    ax[2].set_xlabel("RHOB (g/cc)")
    if "NPHI" in res:
        tw = ax[2].twiny()
        tw.plot(res["NPHI"], d, color="#ff7f0e", lw=0.9, ls="--")
        tw.set_xlim(0.45, -0.15)  # standard reversed neutron scale so gas shows crossover
        tw.set_xlabel("NPHI (v/v)")

    ax[3].plot(res["PHI"], d, color="#9467bd", lw=1, label="Porosity")
    ax[3].plot(res["SW"], d, color="#17becf", lw=1, label="Sw")
    ax[3].plot(res["VSH"], d, color="#8c564b", lw=0.8, ls=":", label="Vsh")
    ax[3].set_xlim(0, 1)
    ax[3].set_xlabel("Phi / Sw / Vsh (v/v)")
    ax[3].legend(fontsize=8, loc="lower right")

    ax[0].set_ylim(d.max(), d.min())
    ax[0].set_ylabel("Depth (m)")
    fig.suptitle(f"{well}: formation evaluation (green bands = net pay)", y=0.995)
    fig.tight_layout()
    return fig


def _fallback_text(summary):
    z = summary["zones"]
    if not z:
        return ("No interval passed the pay cutoffs. Review the cutoffs and Rw, or check whether the logged "
                "section really contains a reservoir.")
    best = max(z, key=lambda x: x["thickness"])
    return (f"Logged interval {summary['top']:.0f} to {summary['base']:.0f} m. {summary['n_zones']} pay zone(s) were "
            f"identified with {summary['net_pay']:.1f} m net pay (net-to-gross {summary['net_to_gross']*100:.0f}%). "
            f"Pay-weighted porosity is {summary['avg_phi_pay']*100:.1f}% and water saturation {summary['avg_sw_pay']*100:.0f}%.\n\n"
            f"The thickest zone is {best['top']:.1f} to {best['base']:.1f} m ({best['thickness']:.1f} m, "
            f"porosity {best['phi']*100:.1f}%, Sw {best['sw']*100:.0f}%).\n\n"
            "Sw depends on the assumed Rw and Archie parameters, so calibrate against core or water analysis before "
            "using these volumes. Check bad-hole intervals on the density log, and consider a test of the best zone.")


def run(df, rw=0.05, vsh_cut=0.4, phi_cut=0.08, sw_cut=0.6, well="Well"):
    res, summary = petrophysics.evaluate(df, rw=rw, vsh_cut=vsh_cut, phi_cut=phi_cut, sw_cut=sw_cut)
    fig = make_log_plot(res, summary, well)
    payload = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k != "zones"}
    payload["zones"] = [{k: round(v, 3) for k, v in z.items()} for z in summary["zones"]]
    text = llm.ask(SYSTEM, json.dumps(payload, default=float), max_tokens=900)
    return res, summary, fig, (text or _fallback_text(summary)), text is not None
