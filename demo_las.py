"""Writes a synthetic LAS 2.0 file with shale, a gas/oil sand, a water sand and a tight streak.
Run:  python tools/demo_las.py"""
import numpy as np


def build(path="demo_well.las", seed=11):
    rng = np.random.default_rng(seed)
    depth = np.arange(2500.0, 2700.0, 0.5)
    n = len(depth)
    gr = np.full(n, 105.0)
    rhob = np.full(n, 2.55)
    rt = np.full(n, 2.0)
    nphi = np.full(n, 0.32)

    def layer(top, base, g, rho, res, nph):
        i = (depth >= top) & (depth < base)
        gr[i], rhob[i], rt[i], nphi[i] = g, rho, res, nph

    layer(2540, 2562, 35, 2.28, 45.0, 0.20)    # clean hydrocarbon sand
    layer(2562, 2575, 38, 2.30, 1.2, 0.24)     # same sand, water-bearing
    layer(2598, 2606, 60, 2.45, 12.0, 0.14)    # shaly / tight streak
    layer(2630, 2660, 30, 2.25, 80.0, 0.22)    # second clean hydrocarbon sand
    layer(2660, 2672, 33, 2.27, 1.0, 0.25)     # water leg
    gr += rng.normal(0, 3, n)
    rhob += rng.normal(0, 0.012, n)
    rt *= np.exp(rng.normal(0, 0.06, n))
    nphi += rng.normal(0, 0.008, n)
    # a few bad-hole spikes and a null section (as in real logs)
    rhob[np.arange(0, n, 97)] += 0.12
    null = -999.25
    gr[:4] = rt[:4] = rhob[:4] = nphi[:4] = null

    head = f"""~Version Information
VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP.   NO  : ONE LINE PER DEPTH STEP
~Well Information
STRT.M  {depth[0]:.1f} : START DEPTH
STOP.M  {depth[-1]:.1f} : STOP DEPTH
STEP.M  0.5 : STEP
NULL.   {null} : NULL VALUE
COMP.   DEMO ENERGY : COMPANY
WELL.   DEMO-WELL-LOG-1 : WELL
FLD.    DEMO FIELD : FIELD
~Curve Information
DEPT.M     : Depth
GR  .GAPI  : Gamma Ray
RT  .OHMM  : Deep Resistivity
RHOB.G/CC  : Bulk Density
NPHI.V/V   : Neutron Porosity
~Parameter Information
RW  .OHMM 0.05 : Formation water resistivity
~A  DEPT GR RT RHOB NPHI
"""
    rows = "\n".join(f"{d:.1f} {a:.2f} {b:.3f} {c:.4f} {e:.4f}" for d, a, b, c, e in zip(depth, gr, rt, rhob, nphi))
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + rows + "\n")
    return path


if __name__ == "__main__":
    print("Wrote", build())
