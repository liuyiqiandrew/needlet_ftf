# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weighted foreground template fitting for Healpix Q/U polarization maps. A lightweight alternative to a full inverse-covariance pipeline: smooth maps to a common beam, optionally apply harmonic (`ell`/`m`) filtering, build difference templates (e.g. `353 - 217`), and solve for template amplitudes via diagonal-weight normal equations. Includes Monte Carlo bootstrap for amplitude uncertainty.

## Commands

```bash
# Run full test suite
python -m pytest -q

# Run a single test
python -m pytest -q tests/test_fg_weighted_template_fit.py::test_name

# Run only tests that don't require healpy (fast, pure-numpy tests)
python -m pytest -q -m "not skipif"
```

There is no build step, linter, or formatter configured. The package is imported directly from the source tree (`import fg_weighted_template_fit as ftf`).

## Dependencies

- `numpy` (required)
- `healpy` (required for smoothing/harmonic filtering; algebraic fitting works without it)
- `tqdm` (optional, for `show_progress=True` in bootstrap)
- `pytest` (test suite)

## Architecture

The package is a single flat module under `fg_weighted_template_fit/` with no subpackages. All public symbols are re-exported through `__init__.py`.

### Module data flow

The pipeline has two layers: a **harmonic preprocessing** layer and a **pixel-space fitting** layer.

1. **`_types.py`** - Frozen dataclasses: `HarmonicFilter`, `DifferenceTemplateInput`, `WeightedFitResult`, `MultiMaskFitResult`, `BootstrapFitResult`, `MultiMaskBootstrapResult`. `FloatArray` type alias lives here. `DifferenceTemplateInput` carries optional `beam_window_a`/`beam_window_b` custom input beams.

2. **`_arrays.py`** - Shape normalization (`as_qu_map`, `as_template_stack`, `as_weight_map`, `as_covariance`) that accepts `(2, npix)` or `(npix, 2)` and always returns `(2, npix)`. Also contains `weighted_inner_product` and `coerce_rng`. No healpy dependency.

3. **`_filters.py`** - Healpy-dependent harmonic operations. `smooth_and_filter_qu_map` is the single-map workhorse: it prepends a zero-T row, does one `map2alm`/`almxfl`/`alm2map` round-trip combining beam matching and `ell`-filter, then applies `m`-filter. A custom input beam may be supplied via `beam_window_in`; it is a real, positive, axisymmetric alm-amplitude `B_ell` applied equally to E and B, and the beam transfer becomes Gaussian-output / custom-input instead of Gaussian-on-Gaussian. `construct_difference_template` and `build_template_stack` call it for each input map. `build_ell_filter`/`build_m_filter` are public helpers that produce reusable high-pass windows; `_build_apodized_highpass` implements the NaMaster-style C1/C2 taper shared by both `ell` and `m` cutoffs.

4. **`_fit.py`** - Pixel-space fitting. `weighted_template_gls` builds the cross normal matrix `d_1^T W d_2` (left × right templates) and solves `(d_1^T W d_2)^-1 d_3^T W m` via `np.linalg.solve` (falls back to `pinv` if singular). `fit_foreground_templates` is the high-level entry: preprocess target + templates, then call `weighted_template_gls`. `fit_foreground_templates_multi_mask` smooths/filters the target + templates **once** under a shared `master_mask`, then runs an independent GLS solve for each named weight map in `weight_maps`, returning a `MultiMaskFitResult`.

5. **`_noise.py`** - `realize_qu_noise` does per-pixel 2x2 Cholesky-style draws from `(QQ, UU, QU)` covariance. `bootstrap_template_amplitudes` runs `n_mc` noisy refits through the full pipeline, storing every amplitude draw. `bootstrap_template_amplitudes_multi_mask` does the same for the shared-preprocessing multi-mask fit, returning a `MultiMaskBootstrapResult` with samples of shape `(n_mc, n_fit_mask, n_template)`.

### Key design patterns

- **Cross-estimator**: Separate left/right template stacks (`template_inputs` vs `template_inputs_rhs`) avoid noise bias in the normal matrix auto-term. When `template_inputs_rhs` is omitted, both sides use the same stack.
- **Data-projection template**: An optional third stack (`template_inputs_data` / `templates_data_qu`, the `d_3` in `d_3^T W m`) enters only the right-hand vector, not the normal matrix. When omitted, `d_3` defaults to the left-hand stack.
- **Multi-mask shared preprocessing**: `fit_foreground_templates_multi_mask` runs the expensive harmonic smoothing/filtering once under a `master_mask`, restricts to binary master support, then reuses those processed maps across every named weight map. Support is applied *after* filtering so an apodized mask is not multiplied in twice.
- **Pre-harmonic masking**: When `mask` is supplied, it is applied in pixel space *before* the harmonic transform (to apodize edges), then again in the pixel-space fit weights.
- **Shared harmonic plan**: One operation resolves the unique explicit `lmax`, or native `3 * nside - 1` when none is set. Conflicting values and short beam/filter arrays raise. If any target or template operand needs harmonic work, every operand uses the same SHT bandlimit.
- **healpy is optional at import**: `_filters.py` catches `ImportError` so the fitting algebra can be used without healpy installed. Tests that need healpy use `@pytest.mark.skipif`.

## Data Conventions

- Q/U maps: `(2, npix)` canonical, `(npix, 2)` accepted and transposed automatically.
- Per-pixel covariance order: `QQ, UU, QU` with shape `(3, npix)`.
- All beam FWHM values are in **radians**.
- FWHMs used for Gaussian beams are finite and nonnegative; the common output
  FWHM is always validated this way.
- Custom beam arrays are real, finite, strictly positive, axisymmetric
  alm-amplitude `B_ell`, not `B_ell**2`, and apply equally to E and B. Separate
  E/B, asymmetric or `m`-dependent, and cross-polar beams are unsupported.
- Healpix default ordering is RING; pass `nest=True` for NEST.
