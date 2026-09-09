# Repository Guidelines

## Project Structure & Module Organization

This repository is a small, source-tree Python package for weighted foreground
template fitting of Healpix Q/U maps. There is no packaging metadata, build
step, CI configuration, linter, or formatter configuration; run imports and
tests from the repository root.

- `fg_weighted_template_fit/_types.py`: frozen configuration and result
  dataclasses plus the `FloatArray` alias.
- `fg_weighted_template_fit/_arrays.py`: `float64` shape normalization,
  diagonal-weight helpers, and RNG coercion. It has no Healpy dependency.
- `fg_weighted_template_fit/_filters.py`: common-beam matching, Healpix
  smoothing, `ell`/`m` filtering, and difference-template construction.
- `fg_weighted_template_fit/_fit.py`: single-mask and shared-preprocessing
  multi-mask weighted solves.
- `fg_weighted_template_fit/_noise.py`: Q/U noise draws and serial or threaded
  Monte Carlo amplitude bootstraps.
- `fg_weighted_template_fit/__init__.py`: public imports and `__all__`. Keep both
  synchronized whenever the public API changes.
- `tests/`: `test_filters.py`, `test_fit.py`, and `test_noise.py` mirror the
  source responsibilities.
- `docs/api.md`, `docs/multi_mask.md`, and `docs/parallel_bootstrap.md`: detailed
  API, mask, and threading/RNG behavior. `README.md` holds the user overview and
  examples.

## Numerical & API Invariants

- Code uses Python 3.10+ syntax, `from __future__ import annotations`, type
  hints, and NumPy `float64` arrays for numerical paths.
- Canonical Q/U shape is `(2, npix)`; `(npix, 2)` is accepted. Canonical
  template shape is `(n_template, 2, npix)`. Covariance shape is `(3, npix)` in
  `QQ, UU, QU` order. Reuse `_arrays.py` coercion helpers instead of duplicating
  shape logic.
- Weights are diagonal pixel-space weights and may be scalar, `(npix,)`, or a
  Q/U layout. This package does not implement a dense inverse covariance.
- Beam FWHM values are radians. RING ordering is the default; `nest=True`
  enables NEST input/output around harmonic transforms.
- Preserve the three-stack estimator: the normal matrix is `d_1^T W d_2`, and
  the right-hand vector is `d_3^T W m`. The right and data-projection stacks
  default independently to the left stack. Shared non-finite samples are
  removed, singular normal matrices fall back to `pinv`, and the residual model
  uses the right-hand (`d_2`) templates.
- Mask arguments are not interchangeable. `weighted_template_gls(mask=...)`
  converts finite nonzero values to binary fit support. The high-level
  `fit_foreground_templates(mask=...)` uses its mask only for pre-harmonic
  apodization, not as another GLS weight. Multi-mask fits preprocess once under
  `master_mask`, apply binary master support once after filtering, then solve
  with each named `weight_map` in insertion order; do not multiply apodized
  mask values into the fit a second time.
- Without a custom input beam, require finite, nonnegative FWHMs and
  `fwhm_out >= fwhm_in`; `fwhm_out` is always finite and nonnegative. A custom
  input window is a real, finite, strictly positive, axisymmetric alm-amplitude
  `B_ell` applied equally to E and B, not `B_ell**2`; separate E/B, asymmetric,
  and cross-polar beams are unsupported. Each complete preprocessing operation
  uses the unique explicit `lmax`, or native `3 * nside - 1` if none is set;
  conflicts and short beam/filter arrays raise. If any operand needs harmonic
  work, all operands use the same SHT/`lmax`. Validate completed transfers as
  finite before transforming; only an all-identity operation uses the
  pure-NumPy path.
- Bootstrap noise is drawn on native-resolution target/template maps before the
  full preprocessing pipeline. Template noise is included only when the
  corresponding `noise_cov_a`/`noise_cov_b` is present. Multi-mask amplitudes
  remain paired within each draw. Serial `n_jobs=1` preserves the serial RNG
  path; threaded results must be reproducible for the same seed and worker
  count but need not match serial samples.

## Build, Test & Dependency Commands

- `python -m pytest -q`: run the full suite.
- `python -m pytest -q tests/test_fit.py::test_weighted_template_gls_recovers_known_amplitudes`:
  run one concrete test.
- `python -m pytest -q -m "not skipif"`: exclude tests marked `skipif`
  (currently the tests requiring a real Healpy installation).

`numpy` is the runtime requirement and `pytest` is test-only. `healpy` is
optional at import but required for smoothing or harmonic transforms. `tqdm`
is optional and needed only for `show_progress=True`. The `pymaster` snippet in
`docs/multi_mask.md` is an example mask-building recipe, not a package
dependency.

## Coding & Documentation Style

Use snake_case for functions, variables, and tests, and PascalCase for
dataclasses such as `HarmonicFilter`. Keep modules flat and focused. Prefer
concise NumPy-style docstrings that state accepted shapes, units, mask/support
semantics, return shapes, and failure modes. The dataclasses are frozen; create
replacements instead of mutating configuration objects. Follow the existing
formatting because no formatter is enforced.

Public behavior changes should update the import and `__all__` lists in
`__init__.py`, relevant examples in `README.md`, and `docs/api.md`. Put
multi-mask support or weighting changes in `docs/multi_mask.md`, and bootstrap
threading or RNG changes in `docs/parallel_bootstrap.md`.

## Testing Guidelines

Name tests `test_<behavior>` and add `-> None` plus a short behavioral docstring.
Use deterministic arrays or seeded generators, `numpy.testing` for numerical
comparisons, and `pytest.raises(..., match=...)` or `pytest.warns` for failure
and warning contracts. Keep fixtures and monkeypatch fakes local unless reuse
justifies `conftest.py`.

Mark tests that require a real Healpy installation with the established form
`@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")`.
Pure algebra tests should remain runnable without Healpy; monkeypatch the module
boundary when testing argument forwarding rather than a spherical-harmonic
integration.

## Commit & Pull Request Guidelines

Use short, focused, imperative, lowercase commit summaries consistent with
history, such as `add custom beam support` or `fix filter bug, lowpass to
highpass`. Pull requests should describe the scientific and API impact, list
tests run, and call out changes to shapes, units, estimator stack roles, mask or
support behavior, RNG/threading behavior, or optional dependencies.
