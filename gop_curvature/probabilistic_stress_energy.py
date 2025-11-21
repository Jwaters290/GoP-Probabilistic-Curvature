"""
Probabilistic stress-energy tensor T^{prob}_{mu nu} for the GoP framework.

This module provides simple, usable functions that map:
    (E, rho_b, z) -> T_prob(mu,nu)

under a set of phenomenological assumptions consistent with existing GoP work.
"""

from __future__ import annotations

import numpy as np

from .gop_constants import (
    C_LIGHT,
    KAPPA_A,
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
        Baryonic mass density in g/cm^3.
    z : float, optional
        Cosmological redshift. For many local/galactic applications,
        z ≈ 0 is appropriate. Default is 0.0.
    f_ent : float, optional
        Entanglement fraction: the fraction of baryonic density that
        effectively participates in the probabilistic scalar field.
        Default is the global F_ENT.

    Returns
    -------
    rho_psi : float or numpy.ndarray
        Effective "probabilistic" density in g/cm^3.

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
    kappa_A: float = KAPPA_A,
    f_ent: float = F_ENT,
    eos_w: float = 0.0,
) -> np.ndarray:
    """
    Compute a simple diagonal T^{prob}_{mu nu} for a homogeneous fluid
    in its rest frame, using the GoP probabilistic curvature ansatz.

    Parameters
    ----------
    E : float or numpy.ndarray
        Characteristic energy scale in erg (e.g. thermal energy, CR energy).
    rho_b : float or numpy.ndarray
        Baryonic mass density in g/cm^3.
    z : float, optional
        Cosmological redshift. Default is 0.0.
    kappa_A : float, optional
        Global GoP amplitude parameter. Defaults to KAPPA_A.
    f_ent : float, optional
        Entanglement fraction. Defaults to F_ENT.
    eos_w : float, optional
        Effective equation-of-state parameter w for the probabilistic
        fluid: p = w * rho c^2. Default is 0 (dust-like).

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
    The implemented mapping is:

        Gamma(E)      = bell-curve decoherence kernel (dimensionless)
        rho_Psi       = rho_psi_effective(rho_b, z, f_ent)
        rho_eff_prob  = kappa_A * Gamma(E) * rho_Psi        [g/cm^3]
        u_eff_prob    = rho_eff_prob * c^2                  [erg/cm^3]

    and

        T00 = u_eff_prob
        Tii = - w * u_eff_prob   (for i = 1, 2, 3)

    This is intentionally simple and is meant as an interface layer:
    cosmology codes can plug u_eff_prob into modified Friedmann or
    growth equations as an effective additional component.
    """
    # Ensure numpy arrays for vectorized operations
    E_arr = np.asarray(E, dtype=float)
    rho_b_arr = np.asarray(rho_b, dtype=float)

    # Decoherence kernel
    Gamma = gamma_bell_curve(E_arr)

    # Effective probabilistic density
    rho_psi = rho_psi_effective(rho_b_arr, z=z, f_ent=f_ent)

    # Effective probabilistic mass density
    rho_eff_prob = kappa_A * Gamma * rho_psi

    # Convert to energy density (erg/cm^3)
    u_eff_prob = rho_eff_prob * C_LIGHT**2

    # Build diagonal T^{prob}_{mu nu}
    # For simplicity we assume a fluid at rest in this frame.
    T = np.zeros((4, 4), dtype=float)
    T[0, 0] = u_eff_prob

    # Isotropic pressures: p = w * u
    p_eff = eos_w * u_eff_prob
    T[1, 1] = -p_eff
    T[2, 2] = -p_eff
    T[3, 3] = -p_eff

    return T
