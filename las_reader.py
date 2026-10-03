"""Minimal LAS 2.0 reader (unwrapped files). Returns a DataFrame indexed by depth."""
import io
import re

import numpy as np
import pandas as pd


def _text(source):
    if hasattr(source, "read"):
        raw = source.read()
        return raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
    with open(source, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_las(source):
    """source: path or file-like object. Returns (df, info) where df index is depth."""
    lines = _text(source).splitlines()
    section, null, curves, rows, info = "", -999.25, [], [], {}
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            section = s[1:2].upper()
            continue
        if section in ("W", "P"):
            m = re.match(r"^([^.\s]+)\s*\.[^\s]*\s+([^:]*):", s)
            if m:
                key, val = m.group(1).upper(), m.group(2).strip()
                info[key] = val
                if key == "NULL":
                    try:
                        null = float(val)
                    except ValueError:
                        pass
        elif section == "C":
            m = re.match(r"^([^.\s]+)\s*\.", s)
            if m:
                curves.append(m.group(1).upper())
        elif section == "A":
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                continue
    if not curves or not rows:
        raise ValueError("Could not find curve names (~C) and data (~A) in the LAS file.")
    n = min(len(curves), min(len(r) for r in rows))
    data = np.array([r[:n] for r in rows], dtype=float)
    data[data == null] = np.nan
    df = pd.DataFrame(data, columns=curves[:n])
    df = df.set_index(curves[0])
    df.index.name = "DEPT"
    return df, info
