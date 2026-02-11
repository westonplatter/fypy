"""
SABR Synthetic Volatility Surface Example
==========================================

Generates synthetic equity option market data across 4 tenors (7-45 DTE),
calibrates the SABR model both globally and per-tenor, and compares the
fitted surface against "market" to identify residual mispricings.

Inspired by: https://tr8dr.github.io/Volatility-Surfaces/

The synthetic vol smile includes extra curvature (cubic/quartic terms) and
noise that SABR's 4-parameter model cannot perfectly capture. The residuals
(model vol - market vol) represent what a trader might interpret as
supply/demand imbalances or relative value opportunities.

Two calibration approaches are demonstrated:
  1. Global: single set of [v_0, alpha, beta, rho] across all tenors
  2. Per-tenor: independent calibration per expiry (better fit, captures
     term structure of SABR params)
"""

import numpy as np

from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from fypy.market.MarketSlice import MarketSlice
from fypy.market.MarketSurface import MarketSurface
from fypy.model.slv.Sabr import Sabr
from fypy.calibrate.SabrModelCalibrator import SabrModelCalibrator
from fypy.pricing.analytical.black_scholes import black76_price_strikes
from fypy.fit.Minimizer import LeastSquares

# ============================================================================
# Constants
# ============================================================================
S0 = 100.0         # Spot price (normalized, e.g. an ETF)
r = 0.045          # risk-free rate
q = 0.015          # dividend yield
DTE_LIST = [7, 14, 30, 45]

np.random.seed(42)


# ============================================================================
# Synthetic vol smile generator
# ============================================================================
def synthetic_vol_smile(x: np.ndarray, T: float) -> np.ndarray:
    """
    Generate a synthetic implied vol smile that is NOT perfectly SABR-consistent.

    The smile is a polynomial in log-moneyness with tenor-dependent coefficients.
    The cubic and quartic terms create curvature that SABR cannot fully capture,
    producing meaningful residuals when SABR is calibrated to this data.

    :param x: log-moneyness = log(K/F), array
    :param T: time to maturity in years
    :return: array of implied vols
    """
    # ATM vol: higher for shorter tenors (inverted term structure, common in
    # elevated vol regimes). Short-dated ~28%, longer-dated ~22%.
    atm_vol = 0.20 + 0.06 * np.exp(-5.0 * T)

    # Skew: negative (equity put skew), steeper for shorter tenors
    skew = -0.10 / np.sqrt(T + 0.02)

    # Convexity (smile): more pronounced for shorter tenors
    convexity = 0.35 / (T + 0.05)

    # Cubic: asymmetry beyond SABR's reach (steeper ITM put wing)
    cubic = 0.15 / (T + 0.08)

    # Quartic: extra wing curvature that SABR cannot fit
    quartic = 0.3 / (T + 0.15)

    vol = atm_vol + skew * x + convexity * x**2 + cubic * x**3 + quartic * x**4

    # Floor to avoid negative or unrealistically low vols
    return np.maximum(vol, 0.05)


def make_strikes(F: float, T: float, n_strikes: int = 17) -> np.ndarray:
    """
    Generate a strike grid appropriate for the tenor.
    Width scales with sqrt(T) so shorter tenors have narrower ranges.
    Strikes are rounded to $1 increments.
    """
    width = max(0.06, 0.25 * np.sqrt(T))
    k_min = F * np.exp(-2.5 * width)
    k_max = F * np.exp(2.5 * width)

    strikes = np.linspace(k_min, k_max, n_strikes)
    strikes = np.round(strikes)  # round to $1 increments
    strikes = np.unique(strikes)
    return strikes


# ============================================================================
# Build market surface
# ============================================================================
def build_synthetic_surface() -> MarketSurface:
    """Build a MarketSurface with synthetic option data across all tenors."""
    disc_curve = DiscountCurve_ConstRate(rate=r)
    div_disc = DiscountCurve_ConstRate(rate=q)
    fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

    surface = MarketSurface(forward_curve=fwd, discount_curve=disc_curve)

    for dte in DTE_LIST:
        T = dte / 365.0
        F = fwd.fwd_T(T)
        disc = disc_curve.discount_T(T)

        strikes = make_strikes(F, T)
        x = np.log(strikes / F)  # log-moneyness

        # Synthetic "market" vols with noise
        clean_vols = synthetic_vol_smile(x, T)
        noise = np.random.normal(0, 0.003, size=len(strikes))
        market_vols = np.maximum(clean_vols + noise, 0.05)

        # OTM convention: puts below forward, calls above
        is_calls = (strikes >= F).astype(int)

        # Price options from the synthetic vols (needed by MarketSlice constructor)
        prices = black76_price_strikes(
            F=F, K=strikes, is_calls=is_calls, vol=market_vols, disc=disc, T=T
        )

        # Build the market slice
        market_slice = MarketSlice(
            T=T, F=F, disc=disc,
            strikes=strikes, is_calls=is_calls,
            mid_prices=prices
        )
        # Set vols directly (avoids round-trip through IV calculator)
        market_slice.set_vols(mid_vols=market_vols)

        surface.add_slice(T, market_slice)

    return surface


# ============================================================================
# Calibration
# ============================================================================
def calibrate_global(surface: MarketSurface) -> dict:
    """Calibrate a single SABR model to the entire surface."""
    fwd = surface.forward_curve
    disc_curve = surface.discount_curve

    sabr = Sabr(
        forwardCurve=fwd, discountCurve=disc_curve,
        v_0=0.04, alpha=0.3, beta=0.5, rho=-0.4
    )
    minimizer = LeastSquares(max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=0)
    calibrator = SabrModelCalibrator(surface=surface, minimizer=minimizer)
    result, pricer, _, target_vols = calibrator.calibrate(model=sabr)

    # Compute residuals per tenor
    residuals = {}
    for ttm, market_slice in surface.slices.items():
        F = fwd.fwd_T(ttm)
        model_vols = np.array([
            sabr.implied_vol(K=k, T=ttm, fwd=F)
            for k in market_slice.strikes
        ])
        residuals[ttm] = model_vols - market_slice.mid_vols

    return {
        'model': sabr,
        'params': sabr.get_params().copy(),
        'residuals': residuals,
        'result': result,
    }


def calibrate_per_tenor(surface: MarketSurface) -> dict:
    """Calibrate a separate SABR model for each tenor independently."""
    fwd = surface.forward_curve
    disc_curve = surface.discount_curve

    all_params = {}
    all_residuals = {}
    all_models = {}

    for ttm, market_slice in surface.slices.items():
        # Single-slice surface for this tenor
        single_surface = MarketSurface(forward_curve=fwd, discount_curve=disc_curve)
        single_surface.add_slice(ttm, market_slice)

        sabr = Sabr(
            forwardCurve=fwd, discountCurve=disc_curve,
            v_0=0.04, alpha=0.3, beta=0.5, rho=-0.4
        )
        minimizer = LeastSquares(max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=0)
        cal = SabrModelCalibrator(surface=single_surface, minimizer=minimizer)
        cal.calibrate(model=sabr)

        all_params[ttm] = sabr.get_params().copy()
        all_models[ttm] = sabr

        F = fwd.fwd_T(ttm)
        model_vols = np.array([
            sabr.implied_vol(K=k, T=ttm, fwd=F)
            for k in market_slice.strikes
        ])
        all_residuals[ttm] = model_vols - market_slice.mid_vols

    return {
        'models': all_models,
        'params': all_params,
        'residuals': all_residuals,
    }


# ============================================================================
# Display results
# ============================================================================
def print_results(surface: MarketSurface, global_result: dict, tenor_result: dict):
    """Print calibration results as formatted tables."""
    fwd = surface.forward_curve

    # --- Global calibration summary ---
    gp = global_result['params']
    print("=" * 72)
    print("GLOBAL SABR CALIBRATION (single param set across all tenors)")
    print("=" * 72)
    print(f"  v_0 (vol-of-vol) = {gp[0]:.6f}")
    print(f"  alpha            = {gp[1]:.6f}")
    print(f"  beta  (backbone) = {gp[2]:.6f}")
    print(f"  rho (spot-vol corr) = {gp[3]:.6f}")

    all_global_res = np.concatenate(list(global_result['residuals'].values()))
    print(f"  Global RMSE: {np.sqrt(np.mean(all_global_res**2)) * 10000:.1f} bps")
    print()

    # --- Per-tenor calibration summary ---
    print("=" * 72)
    print("PER-TENOR SABR CALIBRATION")
    print("=" * 72)
    hdr = (f"  {'DTE':>4s} | {'TTM':>8s} | {'v_0':>8s} | {'alpha':>8s} "
           f"| {'beta':>6s} | {'rho':>7s} | {'RMSE':>8s}")
    print(hdr)
    print(f"  {'─'*4:s}─┼─{'─'*8:s}─┼─{'─'*8:s}─┼─{'─'*8:s}─┼─{'─'*6:s}─┼─{'─'*7:s}─┼─{'─'*8:s}")

    for ttm in sorted(surface.slices.keys()):
        dte = int(round(ttm * 365))
        p = tenor_result['params'][ttm]
        res = tenor_result['residuals'][ttm]
        rmse = np.sqrt(np.mean(res**2)) * 10000
        print(f"  {dte:4d} | {ttm:8.5f} | {p[0]:8.4f} | {p[1]:8.4f} "
              f"| {p[2]:6.4f} | {p[3]:+7.4f} | {rmse:6.1f} bp")

    all_tenor_res = np.concatenate(list(tenor_result['residuals'].values()))
    print(f"\n  Per-tenor avg RMSE: {np.sqrt(np.mean(all_tenor_res**2)) * 10000:.1f} bps")
    print()

    # --- Per-strike detail ---
    print("=" * 72)
    print("RESIDUALS BY STRIKE (Model Vol - Market Vol, in bps)")
    print("  Positive = model overestimates (market option is 'cheap')")
    print("  Negative = model underestimates (market option is 'rich')")
    print("=" * 72)

    for ttm in sorted(surface.slices.keys()):
        dte = int(round(ttm * 365))
        ms = surface.slices[ttm]
        F = fwd.fwd_T(ttm)

        glb_rmse = np.sqrt(np.mean(global_result['residuals'][ttm]**2)) * 10000
        tnr_rmse = np.sqrt(np.mean(tenor_result['residuals'][ttm]**2)) * 10000

        print(f"\n--- {dte} DTE  (T={ttm:.5f}, F={F:.2f})"
              f"  Global RMSE={glb_rmse:.0f}bp  Per-Tenor RMSE={tnr_rmse:.0f}bp ---")
        print(f"  {'Strike':>7s} {'K/F':>6s} {'MktVol':>7s} {'GlbFit':>7s} {'TnrFit':>7s}"
              f" {'GlbRes':>7s} {'TnrRes':>7s}")
        print(f"  {'─'*7:s} {'─'*6:s} {'─'*7:s} {'─'*7:s} {'─'*7:s} {'─'*7:s} {'─'*7:s}")

        for i, K in enumerate(ms.strikes):
            mkt_vol = ms.mid_vols[i]
            glb_res = global_result['residuals'][ttm][i]
            tnr_res = tenor_result['residuals'][ttm][i]
            glb_fit = mkt_vol + glb_res
            tnr_fit = mkt_vol + tnr_res

            print(f"  {K:7.0f} {K/F:6.3f} {mkt_vol*100:6.2f}% {glb_fit*100:6.2f}%"
                  f" {tnr_fit*100:6.2f}% {glb_res*10000:+6.0f}bp {tnr_res*10000:+6.0f}bp")


def plot_results(surface: MarketSurface, global_result: dict, tenor_result: dict):
    """Optional matplotlib visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    fwd = surface.forward_curve
    sorted_ttms = sorted(surface.slices.keys())

    # --- Figure 1: Vol smiles ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SABR Synthetic Surface: Market vs Calibrated Smiles', fontsize=14)

    for idx, ttm in enumerate(sorted_ttms):
        ax = axes[idx // 2, idx % 2]
        dte = int(round(ttm * 365))
        ms = surface.slices[ttm]
        F = fwd.fwd_T(ttm)
        rel_k = ms.strikes / F

        ax.plot(rel_k, ms.mid_vols * 100, 'ko', ms=5, label='Market', zorder=3)

        glb_fit = ms.mid_vols + global_result['residuals'][ttm]
        ax.plot(rel_k, glb_fit * 100, 'b-', lw=2, label='Global SABR')

        tnr_fit = ms.mid_vols + tenor_result['residuals'][ttm]
        ax.plot(rel_k, tnr_fit * 100, 'r--', lw=2, label='Per-Tenor SABR')

        ax.set_title(f'{dte} DTE', fontsize=12)
        ax.set_xlabel('K / F (moneyness)')
        ax.set_ylabel('Implied Vol (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/fypy/examples/sabr_vol_smiles.png', dpi=150)
    print("\nSaved vol smile plot to examples/sabr_vol_smiles.png")

    # --- Figure 2: Residuals ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('SABR Residuals (Model - Market) in bps', fontsize=14)

    for idx, ttm in enumerate(sorted_ttms):
        ax = axes2[idx // 2, idx % 2]
        dte = int(round(ttm * 365))
        ms = surface.slices[ttm]
        F = fwd.fwd_T(ttm)
        rel_k = ms.strikes / F

        width = 0.004
        ax.bar(rel_k - width/2, global_result['residuals'][ttm] * 10000,
               width=width, alpha=0.7, label='Global', color='steelblue')
        ax.bar(rel_k + width/2, tenor_result['residuals'][ttm] * 10000,
               width=width, alpha=0.7, label='Per-Tenor', color='indianred')

        ax.axhline(y=0, color='k', lw=0.5)
        ax.set_title(f'{dte} DTE', fontsize=12)
        ax.set_xlabel('K / F (moneyness)')
        ax.set_ylabel('Residual (bps)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/fypy/examples/sabr_residuals.png', dpi=150)
    print("Saved residuals plot to examples/sabr_residuals.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Building synthetic market surface...")
    surface = build_synthetic_surface()

    print(f"Surface has {surface.num_slices} slices: "
          f"{[int(round(t*365)) for t in sorted(surface.ttms)]} DTE\n")

    print("--- Global SABR calibration ---")
    global_result = calibrate_global(surface)

    print("\n--- Per-tenor SABR calibration ---")
    tenor_result = calibrate_per_tenor(surface)

    print("\n")
    print_results(surface, global_result, tenor_result)
    plot_results(surface, global_result, tenor_result)
