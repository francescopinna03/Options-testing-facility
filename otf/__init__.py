"""Options Testing Facility.

Research infrastructure for calibrating the SFV model (affine jump-diffusion
prior + restricted Schrödinger-bridge correction on the variance channel) to
real option surfaces and benchmarking it, out of sample, against standard
alternatives (Black-Scholes, SVI, Heston).

Layout:
    otf.models       pricing models (BS, Heston CF, SVI, SFV path engine)
    otf.calibration  inverse problems (Heston fit, bridge fit, surface fit)
    otf.data         chain loading/cleaning, realized-measure estimators
    otf.evaluation   forecast-comparison statistics (DM test, bootstrap CIs)
    otf.experiments  runnable studies (surface calibration, OOS comparison)

The numerical core is deliberately pure-stdlib (portable, dependency-free);
the LSEG collectors under scripts/ are the only modules that need the
Workspace SDK.
"""

__version__ = "0.1.0"
