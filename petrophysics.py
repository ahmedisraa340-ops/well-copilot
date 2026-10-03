"""Basic formation evaluation from open-hole logs: Vsh, porosity, Archie Sw, net pay and zones."""
import numpy as np
import pandas as pd

# Common mnemonics for each curve (first match wins)
CURVE_NAMES = {
    "gr": ["GR", "GR_1", "GRC", "CGR", "SGR"],
    "rt": ["RT", "ILD", "RDEEP", "LLD", "RES_DEEP", "RILD", "AT90", "RLA5"],
    "rhob": ["RHOB", "DEN", "RHOZ", "RHOB_1", "DENS"],
    "nphi": ["NPHI", "NPOR", "TNPH", "CNL", "PHIN"],
}


def vshale_gr(gr, gr_clean, gr_shale):
    return np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0, 1)


def porosity_density(rhob, rho_ma=2.65, rho_fl=1.0):
    return np.clip((rho_ma - rhob) / (rho_ma - rho_fl), 0, 0.5)


def archie_sw(rt, phi, rw=0.05, a=1.0, m=2.0, n=2.0):
    phi = np.where(np.asarray(phi) <= 0, np.nan, phi)
    rt = np.where(np.asarray(rt) <= 0, np.nan, rt)
    return np.clip(((a * rw) / (np.power(phi, m) * rt)) ** (1.0 / n), 0, 1)


def _pick(df, key):
    for name in CURVE_NAMES[key]:
        if name in df.columns:
            return df[name].values.astype(float)
    return None


def evaluate(df, rw=0.05, rho_ma=2.65, rho_fl=1.0, a=1.0, m=2.0, n=2.0,
             vsh_cut=0.4, phi_cut=0.08, sw_cut=0.6, min_thickness=1.0):
    """df: depth-indexed log table. Returns (result_df, summary_dict)."""
    gr, rt, rhob = _pick(df, "gr"), _pick(df, "rt"), _pick(df, "rhob")
    nphi = _pick(df, "nphi")
    missing = [k.upper() for k, v in (("gr", gr), ("rt", rt), ("rhob", rhob)) if v is None]
    if missing:
        raise KeyError(f"Missing curves: {', '.join(missing)}. Found: {list(df.columns)}. "
                       f"Add your mnemonics to CURVE_NAMES in tools/petrophysics.py")
    depth = df.index.values.astype(float)
    vsh = vshale_gr(gr, np.nanpercentile(gr, 5), np.nanpercentile(gr, 95))
    phi = porosity_density(rhob, rho_ma, rho_fl)
    sw = archie_sw(rt, phi, rw, a, m, n)
    pay = (vsh < vsh_cut) & (phi > phi_cut) & (sw < sw_cut)
    out = pd.DataFrame({"DEPT": depth, "GR": gr, "RT": rt, "RHOB": rhob, "VSH": vsh,
                        "PHI": phi, "SW": sw, "PAY": pay})
    if nphi is not None:
        out["NPHI"] = nphi
    step = float(np.nanmedian(np.diff(depth)))
    zones = _zones(out, step, min_thickness)
    net = float(sum(z["thickness"] for z in zones))
    gross = float(depth.max() - depth.min())
    summary = {
        "top": float(depth.min()), "base": float(depth.max()), "step": step,
        "gross": gross, "net_pay": net, "net_to_gross": net / gross if gross else 0.0,
        "n_zones": len(zones), "zones": zones,
        "avg_phi_pay": float(np.average([z["phi"] for z in zones], weights=[z["thickness"] for z in zones])) if zones else None,
        "avg_sw_pay": float(np.average([z["sw"] for z in zones], weights=[z["thickness"] for z in zones])) if zones else None,
        "params": {"Rw": rw, "a": a, "m": m, "n": n, "rho_ma": rho_ma, "rho_fl": rho_fl,
                   "Vsh cutoff": vsh_cut, "Phi cutoff": phi_cut, "Sw cutoff": sw_cut},
    }
    return out, summary


def _zones(res, step, min_thickness):
    zones, start = [], None
    pay = res["PAY"].values
    for i in range(len(res) + 1):
        on = i < len(res) and pay[i]
        if on and start is None:
            start = i
        if not on and start is not None:
            seg = res.iloc[start:i]
            thick = len(seg) * step
            if thick >= min_thickness:
                zones.append({"top": float(seg["DEPT"].iloc[0]), "base": float(seg["DEPT"].iloc[-1] + step),
                              "thickness": float(thick), "phi": float(seg["PHI"].mean()),
                              "sw": float(seg["SW"].mean()), "vsh": float(seg["VSH"].mean())})
            start = None
    return zones
