# Keywords
gravity-of-probability, decoherence, probabilistic-curvature, alternative-gravity,
S8-tension, DESI-DR2, warm-void-core, void-redshift-peak-0.55, S8-0.76-0.79,
dipole-amplification, BAO-smoothing, DESI, DESI-DR1, Lyman-alpha, LyA-forest,
large-scale-structure, LSS, Pk-tilt, matter-power-spectrum, BAO-smearing,
cosmic-voids, void-stacking, SPARC-rotation-curves, falsifiable-gravity-model,
pre-registered-predictions, DESI-void-test, ACP-CP-violation

---

# GoP-Probabilistic-Curvature

This repository provides the reference implementation of the Gravity of Probability (GoP)
probabilistic curvature term `T_prob(mu,nu)`.

Includes:

- decoherence kernel `Gamma(E)`
- global GoP constants
- probabilistic stress-energy tools
- warm-core and void predictions
- rotation curve and lensing utilities
- DESI / Euclid test scripts

This is the official codebase used for GoP predictions in cosmology and large-scale structure.

---

# New (Pinned): DESI Warm-Core Prediction Script

### File: `gop_warm_core_desipipeline.py`

This script implements the GoP warm-core prediction using the fixed July 2025 parameter set:

- kappaA = 1.5e-15  
- E0_erg = 1e12  
- f_ent = 0.20  
- A_CP = 0.0245  

It computes:

- baryonic density profile rho_b(r)
- probabilistic density rho_prob(r)
- effective density rho_eff(r)
- inner density slope (cusp vs. core)
- warm-core enhancement factor
- diagnostic plots compatible with DESI stacked void maps

Run the script:

```bash
python gop_warm_core_desipipeline.py
```
Expected (pre-registered):

- inner slope changes from approx. -1 (cusp) to approx. 0 (core)
- warm inner profile
- radial warm-core structure matching DESI DR2 VAC predictions

---

# GoP Predictions for DESI DR2

- Warm-core cosmic void temperature imprint
- Void redshift peak at z ≈ 0.55
- S8 suppressed to 0.76 – 0.79
- Dipole amplification factor 1.2x – 1.5x
- BAO peak broadening / smoothing
- H(z) crossover near z = 0.45 – 0.55
- Tracer independence (BGS, LRG, ELG, QSO voids)

All are based on a single fixed global parameter set (no tuning).

---

# Citations

- Jordan Waters
"The Gravity of Probability: Replicating Dark Matter Effects Through Quantum Decoherence Curvature"
DOI: 10.6084/m9.figshare.29815934
https://figshare.com/articles/thesis/The_Gravity_of_Probability/29815934

- Jordan Waters
"DESI DR2 VACs Predictions"
DOI: 10.6084/m9.figshare.30593876
https://figshare.com/articles/preprint/DESI_DR2_VACs_Predictions/30593876

- Jordan Waters
"Foundations of the Gravity of Probability"
DOI: 10.6084/m9.figshare.30662603
https://figshare.com/articles/preprint/Foundations_of_the_Gravity_of_Probability/30662603

ORCID
https://orcid.org/0009-0009-0793-8089

Figshare Profile
https://figshare.com/authors/Jordan_Waters/21620558

---

# Background

The GoP framework modifies the total stress-energy tensor:

T_total(mu,nu) = T_classical(mu,nu) + T_prob(mu,nu)

Where:

- T_classical is normal matter and radiation
- T_prob is a decoherence-weighted curvature term arising from unrealized quantum amplitudes

Approximate form: T_prob(mu,nu) ≈ kappaA * Gamma(E) * rho_psi

Where:

- kappaA sets amplitude
- Gamma(E) is the GoP decoherence kernel
- rho_psi is the entanglement-weighted baryonic density

The global parameters are fixed prior to DESI DR2:

kappaA = 1.5e-15
E0_erg = 1e12
f_ent = 0.20
A_CP = 0.0245

This repository provides a clean Python API for cosmology and LSS pipelines.

---

# Innstallation

```bash
git clone https://github.com/Jwaters290/GoP-Probabilistic-Curvature.git
cd GoP-Probabilistic-Curvature
pip install -r requirements.txt
```

---

# Qucikstart (Core Prediction)

```bash
git clone https://github.com/Jwaters290/GoP-Probabilistic-Curvature.git
cd GoP-Probabilistic-Curvature
pip install -r requirements.txt
jupyter notebook examples/DESI_void_overlay.ipynb
```
This notebook reproduces the predicted warm-core void shape and redshift ramp used in the preregistered DESI predictions.

---

# VAC Phasing & Falsifiability Roadmap

Phase I — Lyman-Alpha / LSS Power Spectrum (2026)

Prediction:
DeltaP/P ≈ 0.02 – 0.04 at k ≈ 0.1 h/Mpc

Run:
```bash
python scripts/gop_lss_earlytest.py --pk-file desi_pk.dat
```
Phase II — Void Stacking (Primary Test)

DESI stacked voids should show:

- warm central region
- redshift peak at ~0.55
- curvature residual matching GoP predictions

Run:
```bash
python gop_warm_core_desipipeline.py
```

---

# Developer Note: Switching to Full GoP P(k)
Change the toy function in gop_lss_earlytest.py to:

```
from gop_cosmology import compute_pk_gop

def gop_predict_pk(k_array, cosmo=None):
    return compute_pk_gop(k_array, **(cosmo or {}))
```
This upgrades the script to the full GoP cosmology.
