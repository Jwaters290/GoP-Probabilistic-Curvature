#!/usr/bin/env python3
"""
gop_lss_earlytest.py

Early-phase DESI Lyα / LSS test for the Gravity of Probability (GoP) framework.

Goal:
    - Ingest DESI power spectrum VAC products (P(k) vs k).
    - Compute the GoP-predicted multiplicative modifier f_gop(k).
    - Report ΔP/P ≈ f_gop(k) - 1 around k ~ 0.1 h/Mpc and plot the result.

This is a template harness. You need to:
    - Adjust `load_desi_pk` to the final VAC schema (FITS column names, extensions).
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on the Python path so gop_* packages can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Data loading stubs
# ----------------------------------------------------------------------

def load_desi_pk(path: str | Path):
    """
    Load DESI VAC power spectrum file.

    Returns:
        k   : array of k [h/Mpc]
        pk  : array of P(k) (currently unused in modifier-mode ΔP/P computation)

    Supports:
        - FITS files (typical for DESI VACs) if astropy is installed
        - ASCII 2-column text files as a fallback

    NOTE:
        You will likely need to adjust the FITS column names ('K', 'PK')
        to match the DESI VAC schema once public.
    """
    path = Path(path)

    # FITS case (DESI VAC style)
    if path.suffix.lower() in {".fits", ".fit", ".fz"}:
        try:
            from astropy.io import fits
        except ImportError as e:
            raise ImportError(
                "Reading FITS requires astropy. Install with `pip install astropy` "
                "or provide an ASCII (k, P(k)) file."
            ) from e

        with fits.open(path) as hdul:
            # This assumes the power spectrum is in the first extension (hdul[1])
            data = hdul[1].data

            # Adjust to actual VAC column names when known
            k = np.array(data["K"])
            pk = np.array(data["PK"])
        return k, pk

    # ASCII fallback: 2-column plain text (k, P(k))
    data = np.loadtxt(path)
    k = data[:, 0]
    pk = data[:, 1]
    return k, pk


# ----------------------------------------------------------------------
# 2. GoP prediction hook
# ----------------------------------------------------------------------

def gop_predict_modifier(k_array: np.ndarray, cosmo_params: dict | None = None) -> np.ndarray:
    """
    Compute GoP-predicted multiplicative modifier f_gop(k) on the same k grid.

    Contract:
        Returns f_gop(k) such that:
            P_GoP(k) = f_gop(k) * P_LCDM(k)

    This calls your canonical implementation in gop_core.gop_cosmology.
    """
    from gop_core.gop_cosmology import compute_pk_gop
    return compute_pk_gop(k_array, **(cosmo_params or {}))


# ----------------------------------------------------------------------
# 3. Comparison and plotting
# ----------------------------------------------------------------------

def compute_delta_pk_over_pk(f_gop: np.ndarray) -> np.ndarray:
    """
    Modifier-mode ΔP/P:
        ΔP/P ≈ f_gop(k) - 1
    """
    return f_gop - 1.0


def summarize_delta(k: np.ndarray, delta_over_pk: np.ndarray, k_target: float = 0.10, window: float = 0.02):
    """
    Print a numerical summary of ΔP/P near k_target.
    """
    mask = (k >= (k_target - window)) & (k <= (k_target + window))
    if not np.any(mask):
        print("No k-modes found in the target window.")
        return

    mean_delta = float(np.mean(delta_over_pk[mask]))
    std_delta = float(np.std(delta_over_pk[mask]))

    print("--------------------------------------------------")
    print(f" GoP early-phase ΔP/P summary around k ~ {k_target:.3f} h/Mpc")
    print(f" Window: [{k_target - window:.3f}, {k_target + window:.3f}] h/Mpc")
    print(f" N modes: {int(np.sum(mask))}")
    print(f" Mean ΔP/P: {mean_delta:.3%}")
    print(f" Std  ΔP/P: {std_delta:.3%}")
    print("--------------------------------------------------")
    print("Prediction band expectation (pre-registered): ~2–4% near k ~ 0.1 h/Mpc.")
    print("--------------------------------------------------")


def plot_delta(k: np.ndarray, delta_over_pk: np.ndarray, outpath: str | None = None):
    """
    Plot ΔP/P vs k and optionally save to file.
    """
    plt.figure()
    plt.axhline(0.0, linestyle="--")
    plt.plot(k, delta_over_pk)
    plt.xlabel(r"$k \; [h/\mathrm{Mpc}]$")
    plt.ylabel(r"$\Delta P / P$")
    plt.title("GoP Early-Phase Prediction: ΔP/P vs k")
    plt.xlim(0.01, 0.3)
    plt.grid(True)

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
        print(f"Saved plot to: {outpath}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# 4. Main CLI
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Early DESI Lyα / LSS ΔP/P test for Gravity of Probability."
    )
    parser.add_argument(
        "--pk-file",
        required=True,
        help="Path to DESI VAC power spectrum file (ASCII, FITS, etc.; adjust loader as needed).",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="Optional output path for the ΔP/P plot (e.g., plots/delta_pk_gop.png).",
    )

    args = parser.parse_args()

    # 1) Load DESI P(k) (currently unused for modifier-mode ΔP/P)
    k, _pk_data = load_desi_pk(args.pk_file)

    # 2) Compute GoP multiplicative modifier f_gop(k)
    f_gop = gop_predict_modifier(k)

    # 3) Compute ΔP/P
    delta_over_pk = compute_delta_pk_over_pk(f_gop)

    # 4) Summarize around k ~ 0.1 h/Mpc
    summarize_delta(k, delta_over_pk, k_target=0.10, window=0.02)

    # 5) Plot
    plot_delta(k, delta_over_pk, outpath=args.plot_out)


if __name__ == "__main__":
    main()
