"""
Simple cosmology utilities that may be useful when plugging GoP into
cosmological codes. These are intentionally minimal and non-opinionated.
"""

from __future__ import annotations

import numpy as np

from .gop_constants import C_LIGHT, G_NEWTON


def critical_density(H: float) -> float:
    """
    Compute the critical density for a given Hubble parameter H.

    Parameters
    ----------
    H : float
        Hubble parameter in s^-1.

    Returns
    -------
    rho_crit : float
        Critical density in g/cm^3.
    """
    # 3 H^2 / (8 pi G)
    rho_crit = 3.0 * H**2 / (8.0 * np.pi * G_NEWTON)
    return rho_crit
