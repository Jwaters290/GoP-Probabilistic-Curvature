#!/usr/bin/env python3
"""
gop_lss_earlytest.py

Early-phase DESI Lyα / LSS test for the Gravity of Probability (GoP) framework.

Goal:
    - Ingest DESI power spectrum VAC products (P(k) vs k).
    - Compute the GoP-predicted P(k) using your existing model.
    - Compare ΔP/P around k ~ 0.1 h/Mpc and print/plot the result.

This is a template. You need to:
    - Implement `load_desi_pk` for the actual VAC format.
    - Plug in your GoP prediction where indicated.
"""

import sys
import os

# Ensure repo root is on the Python path so gop_curvature can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path


# ----------------------------------------------------------------------
# 1. Data loading stubs
# ----------------------------------------------------------------------

def load_desi_pk(path):
    """
    Load DESI VAC power spectrum file.

    Returns:
        k   : array of k [h/Mpc]
        pk  : array of P(k) [units consistent across LCDM/GoP comparison]

    Supports:
        - FITS files (typical for DESI VACs)
        - ASCII 2-column text files as a fallback

    NOTE:
        You will likely need to adjust the FITS column names ('K', 'PK')
        to match the actual DESI VAC schema once it's public.
    """
    path = Path(path)

    # FITS case (DESI VAC style)
    if path.suffix.lower() in {".fits", ".fit", ".fz"}:
        with fits.open(path) as hdul:
            # This assumes the power spectrum is in the first extension (hdul[1])
            data = hdul[1].data

            # You may need to change these to the actual column names later
            # e.g., 'K' -> 'k', 'PK' -> 'power' depending on DESI schema
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

def gop_predict_pk(k_array, cosmo_params=None):
    """
    Compute GoP-predicted P(k) on the same k grid as the data.

    This function is intentionally minimal and acts as a bridge to your
    existing GoP cosmology / P(k) code.

    IMPORTANT:
        This now connects directly to the real GoP implementation
        in gop_curvature.gop_cosmology. Update that module to change
        the physics behavior.
    """
    # --- REAL GoP MODEL IMPLEMENTATION ---
    from gop_curvature.gop_cosmology import compute_pk_gop
    return compute_pk_gop(k_array, **(cosmo_params or {}))


# ----------------------------------------------------------------------
# 3. Comparison and plotting
# ----------------------------------------------------------------------

def compute_delta_pk_over_pk(k, pk_data, pk_gop_modifier):
    """
    Given:
        k              : array of k [h/Mpc]
        pk_data        : DESI-measured P(k) (approx ΛCDM+noise)
        pk_gop_modifier: multiplicative factor f_gop(k) such that
                         P_GoP(k) = f_gop(k) * P_LCDM(k)

    We approximate:
        ΔP/P ≈ f_gop(k) - 1
    assuming DESI data is near ΛCDM in the regime of interest.
    """
    delta_over_pk = pk_gop_modifier - 1.0
    return delta_over_pk


def summarize_delta(k, delta_over_pk, k_target=0.10, window=0.02):
    """
    Print a simple numerical summary of ΔP/P near k_target.

    Parameters:
        k            : array of k [h/Mpc]
        delta_over_pk: array of ΔP/P
        k_target     : central k for the test (default 0.1 h/Mpc)
        window       : half-width of the window around k_target
    """
    mask = (k >= (k_target - window)) & (k <= (k_target + window))
    if not np.any(mask):
        print("No k-modes found in the target window.")
        return

    mean_delta = np.mean(delta_over_pk[mask])
    std_delta = np.std(delta_over_pk[mask])

    print("--------------------------------------------------")
    print(" GoP early-phase ΔP/P summary around k ~ {:.3f} h/Mpc".format(k_target))
    print(" Window: [{:.3f}, {:.3f}] h/Mpc".format(k_target - window, k_target + window))
    print(" N modes: {}".format(np.sum(mask)))
    print(" Mean ΔP/P: {:.3%}".format(mean_delta))
    print(" Std  ΔP/P: {:.3%}".format(std_delta))
    print("--------------------------------------------------")
    print("The prediction band is ~2–4%. Compare the above with that range.")
    print("Replace the toy GoP model with your real P(k) implementation.")
    print("--------------------------------------------------")


def plot_delta(k, delta_over_pk, outpath=None):
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

    # 1. Load DESI P(k)
    k, pk_data = load_desi_pk(args.pk_file)

    # 2. Compute GoP multiplicative factor (placeholder or real)
    gop_modifier = gop_predict_pk(k)

    # 3. Compute ΔP/P
    delta_over_pk = compute_delta_pk_over_pk(k, pk_data, gop_modifier)

    # 4. Summarize around k ~ 0.1 h/Mpc
    summarize_delta(k, delta_over_pk, k_target=0.10, window=0.02)

    # 5. Plot
    plot_delta(k, delta_over_pk, outpath=args.plot_out)


if __name__ == "__main__":
    main()
