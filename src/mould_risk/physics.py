"""Building physics: surface humidity (ISO 13788 style) and the VTT mould growth model.

VTT: Hukka & Viitanen (1999), sensitivity classes from Ojanen et al. (2010).
"""

import math

import numpy as np

SURFACE_RH_LIMIT = 80.0

VTT_CLASSES = {
    "very_sensitive": {"k1": (1.0, 2.0), "A": 1.0, "B": 7.0, "C": 2.0, "rh_min": 80.0},
    "sensitive": {"k1": (0.578, 0.386), "A": 0.3, "B": 6.0, "C": 1.0, "rh_min": 80.0},
}


def psat(t_c):
    """Saturation vapour pressure in hPa (Magnus)."""
    return 6.112 * np.exp(17.62 * np.asarray(t_c) / (243.12 + np.asarray(t_c)))


def vapour_pressure(t_c, rh):
    return np.asarray(rh) / 100.0 * psat(t_c)


def surface_temperature(t_in, t_out, f_rsi):
    return np.asarray(t_out) + f_rsi * (np.asarray(t_in) - np.asarray(t_out))


def surface_rh(t_in, vp_in, t_out, f_rsi):
    """RH at the coldest internal surface, given indoor vapour pressure."""
    t_si = surface_temperature(t_in, t_out, f_rsi)
    return np.clip(100.0 * np.asarray(vp_in) / psat(t_si), 0.0, 100.0)


def rh_crit(t_c, rh_min=80.0):
    poly = -0.00267 * t_c**3 + 0.160 * t_c**2 - 3.13 * t_c + 100.0
    return max(poly, rh_min) if t_c <= 20 else rh_min


def vtt_index(t, rh, sensitivity="very_sensitive", decline=1.0):
    """Hourly mould index (0-6). NaN hours hold the index and do not count as dry time."""
    p = VTT_CLASSES[sensitivity]
    m = 0.0
    dry_hours = 0
    out = np.empty(len(t))
    for i, (tc, r) in enumerate(zip(np.asarray(t, float), np.asarray(rh, float))):
        if math.isnan(tc) or math.isnan(r):
            out[i] = m
            continue
        crit = rh_crit(tc, p["rh_min"])
        if 0.0 < tc < 50.0 and r >= crit:
            dry_hours = 0
            k1 = p["k1"][0] if m < 1 else p["k1"][1]
            ratio = (crit - r) / (crit - 100.0) if crit < 100 else 1.0
            m_max = p["A"] + p["B"] * ratio - p["C"] * ratio**2
            k2 = max(1.0 - math.exp(2.3 * (m - m_max)), 0.0)
            per_day = k1 * k2 / (7.0 * math.exp(-0.68 * math.log(tc) - 13.9 * math.log(r) + 66.02))
            m += per_day / 24.0
        else:
            dry_hours += 1
            if dry_hours <= 6:
                m -= decline * 0.00133
            elif dry_hours > 24:
                m -= decline * 0.000667
            m = max(m, 0.0)
        out[i] = m
    return out
