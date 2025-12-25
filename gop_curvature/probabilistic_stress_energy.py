"""
Probabilistic stress-energy tensor T^{prob}_{mu nu} for the GoP framework.

This module provides simple, usable functions that map:
    (E, rho_b, z) -> T_prob(mu,nu)

under a set of phenomenological assumptions consistent with existing GoP work.

IMPORTANT CONVENTION:
- The canonical bell-curve kernel gamma_bell_curve(E) already includes κA by default.
  Therefore this module MUST NOT multiply by κA again (to avoid double counting).
"""

from __future__ import annotations

import numpy as np

from .gop_constants import (
    C_LIGHT,
    F_ENT,
)
from .bell_curve_decoherence_kernel import gamma_bell_curve


def rho_psi_effective(
    rho_b: float | np.ndarray,
    z: float = 0.0,
    f_ent: float = F_ENT,
) -> float | np.ndarray:
    """
    Compute an effective probabilistic/entanglement density rho_Psi
    for a given baryonic mass density rho_b and redshift z.

    Parameters
    ----------
    rho_b : float or numpy.ndarray
        Baryonic mass density in g/cm^3 (or a consistent proxy).
    z : float, optional
        Cosmological redshift. Default is 0.0.
    f_ent : float, optional
        Entanglement fraction: the fraction of baryonic density that
        effectively participates in the probabilistic scalar field.
        Default is the global F_ENT.

    Returns
    -------
    rho_psi : float or numpy.ndarray
        Effective "probabilistic" density in g/cm^3 (or proxy units).

    Notes
    -----
    This is a phenomenological mapping. In a full GoP implementation,
    rho_Psi may depend on entropy, temperature, environment, etc.
    Here we assume, as a first-order approximation:

        rho_Psi ≈ f_ent * rho_b * (1 + z)^3

    so that it scales with cosmological dilution similarly to matter.
    """
    rho_b_arr = np.asarray(rho_b, dtype=float)
    rho_psi = f_ent * rho_b_arr * (1.0 + z) ** 3
    return rho_psi


def compute_tmunu_prob(
    E: float | np.ndarray,
    rho_b: float | np.ndarray,
    z: float = 0.0,
    f_ent: float = F_ENT,
    eos_w: float = 0.0,
) -> np.ndarray:
    """
    Compute a simple diagonal T^{prob}_{mu nu} for a homogeneous fluid
    in its rest frame, using the GoP probabilistic curvature ansatz.

    Parameters
    ----------
    E : float or numpy.ndarray
        Characteristic energy scale (erg or consistent proxy).
    rho_b : float or numpy.ndarray
        Baryonic mass density in g/cm^3 (or consistent proxy).
    z : float, optional
        Cosmological redshift. Default is 0.0.
    f_ent : float, optional
        Entanglement fraction. Defaults to F_ENT.
    eos_w : float, optional
        Effective equation-of-state parameter w for the probabilistic
        fluid: p = w * u. Default is 0 (dust-like).

    Returns
    -------
    T_prob : numpy.ndarray
        A 4x4 numpy array representing T^{prob}_{mu nu} in cgs units,
        with energy density in erg/cm^3 and pressures in erg/cm^3.

        The returned tensor is diagonal:
            diag(T00, T11, T22, T33)
        with signature (+, -, -, -) assumed.

    Notes
    -----
    Convention (canonical):

        Γ(E)          = gamma_bell_curve(E)            [includes κA by default]
        rho_Psi       = rho_psi_effective(rho_b, z)
        rho_eff_prob  = Γ(E) * rho_Psi
        u_eff_prob    = rho_eff_prob * c^2

    and

        T00 = u_eff_prob
        Tii = - w * u_eff_prob   (for i = 1, 2, 3)
    """
    # Ensure numpy arrays for vectorized operations
    E_arr = np.asarray(E, dtype=float)
    rho_b_arr = np.asarray(rho_b, dtype=float)

    # Canonical decoherence kernel Γ(E) (already includes κA)
    Gamma = gamma_bell_curve(E_arr)

    # Effective probabilistic density rho_Psi
    rho_psi = rho_psi_effective(rho_b_arr, z=z, f_ent=f_ent)

    # Effective probabilistic mass density (do NOT multiply by κA again)
    rho_eff_prob = Gamma * rho_psi

    # Convert to energy density (erg/cm^3)
    u_eff_prob = rho_eff_prob * (C_LIGHT ** 2)

    # Build diagonal T^{prob}_{mu nu}
    T = np.zeros((4, 4), dtype=float)
    T[0, 0] = u_eff_prob

    # Isotropic pressures: p = w * u
    p_eff = eos_w * u_eff_prob
    T[1, 1] = -p_eff
    T[2, 2] = -p_eff
    T[3, 3] = -p_eff

    return T
