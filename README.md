# fg_weighted_template_fit

Weighted foreground template fitting for Healpix Q/U maps.

This repository provides a lightweight alternative to a full inverse-covariance
pipeline when the fitting problem is already expressed through a user-supplied
weight map. The core workflow is:

1. If a mask is supplied, apodize the target and template-construction maps in
   pixel space before any harmonic preprocessing.
2. Match target and template-construction maps to a common Gaussian output beam.
3. Optionally apply harmonic filtering in `ell` and `m`.
4. Build foreground templates from map differences such as `353 - 217`.
5. Estimate template amplitudes with a weighted normal equation.
6. Bootstrap amplitude uncertainty with Monte Carlo noise realizations built
   from per-pixel `QQ`, `UU`, and `QU` covariances.

The main package is [`fg_weighted_template_fit/`](./fg_weighted_template_fit).

## Method

For a target polarization map `m` and left/right template stacks
`T_left`, `T_right`, the fitted amplitudes are

```text
a_hat = (T_left^T W T_right)^(-1) T_left^T W m
```

where:

- `m` is the target map with Q and U pixels stacked into one data vector
- `T_left` and `T_right` are matrices of stacked template maps
- `W` is a diagonal weight map supplied by the user

Using separate left and right template realizations is useful when you want to
avoid noise bias from a template auto-term in the normal matrix. If no separate
right-hand stack is supplied, the package falls back to the same-template solve

```text
a_hat = (T^T W T)^(-1) T^T W m
```

The data-projection vector `T_left^T W m` can also be decoupled from the
normal-matrix factors by supplying a third template stack `d_3`
(`templates_data_qu` / `template_inputs_data`). With three distinct stacks
`d_1 = T_left`, `d_2 = T_right`, and `d_3`, the estimator becomes

```text
a_hat = (d_1^T W d_2)^(-1) d_3^T W m
```

where finite `d_3` samples enter only the right-hand vector and not the normal
matrix. When `d_3` is omitted it defaults to `d_1`, recovering the forms above;
supplying only `template_inputs` makes all three stacks equal. Non-finite samples
in any of `d_1`, `d_2`, `d_3`, `m`, or `W` are removed from the shared fit
support.

This is still not the fully optimal generalized least-squares estimator you
would get from a full inverse covariance matrix, but it is often a useful fast
estimator when a scalar or diagonal weight definition is already available and
you want a swift amplitude estimate.

## Repository Layout

```text
fg_weighted_template_fit/
├── README.md
├── docs/
│   ├── api.md
│   ├── multi_mask.md
│   └── parallel_bootstrap.md
├── fg_weighted_template_fit/
│   ├── __init__.py
│   ├── _arrays.py
│   ├── _filters.py
│   ├── _fit.py
│   ├── _noise.py
│   └── _types.py
└── tests/
    ├── test_filters.py
    ├── test_fit.py
    └── test_noise.py
```

Module responsibilities:

- `_types.py`: dataclasses for filters, template definitions, and fit results
- `_filters.py`: Healpix smoothing, `ell`/`m` filtering, and template building
- `_fit.py`: weighted template solving
- `_noise.py`: Q/U noise realizations and Monte Carlo uncertainty estimation
- `_arrays.py`: array-shape normalization helpers

## Main Features

- Difference-template construction for split maps such as dust or synchrotron
- Common-beam matching from input `fwhm_in` to output `fwhm_out`
- Optional custom input beam windows `B_ell` that replace Gaussian `fwhm_in`
- Optional pre-harmonic masking for apodized sky cuts
- Optional harmonic filtering in both `ell` and `m`
- Public helpers for reusable explicit `ell` and `m` filter arrays
- Smooth `ell` and `m` cutoffs with `C1` or `C2` apodized edges
- Weighted diagonal GLS-like solve for template amplitudes
- Independent left, right, and data-projection template stacks for cross fits
- Multi-mask fits with shared harmonic preprocessing and per-region weights
- Bootstrap uncertainty estimation from per-pixel `QQ`, `UU`, `QU` covariance
- Storage of the recovered amplitude from every Monte Carlo realization

## Key Assumptions

- Maps are Healpix Q/U polarization maps.
- Beam widths are given in radians.
- The output FWHM, and every input FWHM used for Gaussian beam matching, must
  be finite and nonnegative.
- Custom beam windows are real, finite, strictly positive, axisymmetric
  alm-amplitude transfer functions indexed by `ell`. They are applied equally
  to E and B, are not power-spectrum windows `B_ell**2`, and are used as
  supplied without automatic normalization.
- Separate E/B responses, asymmetric or `m`-dependent beams, and cross-polar
  response are not supported by this first custom-beam interface.
- Every map in one preprocessing operation uses one common harmonic `lmax`: the
  unique explicit `HarmonicFilter.lmax`, or `3 * nside - 1` when none is set.
  Beam and explicit filter arrays must cover that support.
- Template maps are built as a difference of two Q/U maps after smoothing to a
  common resolution.
- Noise covariance is provided per pixel in the order `QQ`, `UU`, `QU`.
- The fit weight is diagonal in pixel space. This package does not implement a
  dense inverse covariance solve.

## Dependencies

- `numpy`
- `healpy` for smoothing and harmonic filtering
- `pytest` for the test suite

The purely algebraic fitting functions can still be imported without `healpy`,
but smoothing and harmonic filtering require it.

## Quick Start

```python
import numpy as np
import fg_weighted_template_fit as ftf

# Example target map and diagonal weights.
target_qu = np.random.standard_normal((2, 12 * 8**2))
weight_map = np.ones(target_qu.shape[1])

dust_split_a = ftf.DifferenceTemplateInput(
    map_a_qu=planck_353_qu,
    map_b_qu=planck_217_qu,
    fwhm_in_a=fwhm_353_rad,
    fwhm_in_b=fwhm_217_rad,
    noise_cov_a=planck_353_cov,
    noise_cov_b=planck_217_cov,
    name="dust",
)

dust_split_b = ftf.DifferenceTemplateInput(
    map_a_qu=planck_353_split_b_qu,
    map_b_qu=planck_217_split_b_qu,
    fwhm_in_a=fwhm_353_rad,
    fwhm_in_b=fwhm_217_rad,
    noise_cov_a=planck_353_split_b_cov,
    noise_cov_b=planck_217_split_b_cov,
    name="dust",
)

sync_split_a = ftf.DifferenceTemplateInput(
    map_a_qu=wmap_k_qu,
    map_b_qu=wmap_ka_qu,
    fwhm_in_a=fwhm_wmap_k_rad,
    fwhm_in_b=fwhm_wmap_ka_rad,
    noise_cov_a=wmap_k_cov,
    noise_cov_b=wmap_ka_cov,
    name="sync",
)

sync_split_b = ftf.DifferenceTemplateInput(
    map_a_qu=wmap_k_split_b_qu,
    map_b_qu=wmap_ka_split_b_qu,
    fwhm_in_a=fwhm_wmap_k_rad,
    fwhm_in_b=fwhm_wmap_ka_rad,
    noise_cov_a=wmap_k_split_b_cov,
    noise_cov_b=wmap_ka_split_b_cov,
    name="sync",
)

filter_config = ftf.HarmonicFilter(
    ell_cutoff=180.0,
    ell_halfwidth=20.0,
    m_cutoff=64.0,
    m_halfwidth=8.0,
    transition_type="C2",
)

result = ftf.fit_foreground_templates(
    target_qu=target_qu,
    target_fwhm_in=target_fwhm_rad,
    # Optional: target_beam_window=target_beam_b_ell,
    template_inputs=[dust_split_a, sync_split_a],
    template_inputs_rhs=[dust_split_b, sync_split_b],
    weight_map=weight_map,
    fwhm_out=common_fwhm_rad,
    target_filter=filter_config,
)

print(result.template_names)
print(result.amplitudes)
```

## Monte Carlo Uncertainty Example

The Monte Carlo routine can propagate uncertainty from both the target map and
the maps used to build the templates. Template uncertainty is included when
each `DifferenceTemplateInput` carries `noise_cov_a` and `noise_cov_b`.

```python
bootstrap = ftf.bootstrap_template_amplitudes(
    target_qu=target_qu,
    target_noise_cov=target_noise_cov,
    target_fwhm_in=target_fwhm_rad,
    template_inputs=[dust_split_a, sync_split_a],
    template_inputs_rhs=[dust_split_b, sync_split_b],
    weight_map=weight_map,
    fwhm_out=common_fwhm_rad,
    n_mc=200,
    target_filter=filter_config,
    rng=1234,
    show_progress=True,
    n_jobs=4,
)

print(bootstrap.amplitude_mean)
print(bootstrap.amplitude_std)
print(bootstrap.amplitude_samples.shape)
```

Pass `show_progress=True` to show a standard `tqdm` progress bar in notebooks
or terminals while the Monte Carlo draws are running, without relying on
ipywidgets. Pass `n_jobs > 1` to run independent Monte Carlo draws in worker
threads; this can be called directly from JupyterLab.

See [`docs/parallel_bootstrap.md`](./docs/parallel_bootstrap.md) for the
parallel execution and random-number design.

In the example above, the reported `bootstrap.amplitude_std` includes:

- target-map noise from `target_noise_cov`
- template-map noise from `noise_cov_a` and `noise_cov_b` on each template input
- the effect of rebuilding the templates after adding those noise realizations

If template noise covariances are omitted, the Monte Carlo spread will only
reflect target-map noise and will therefore underestimate the total uncertainty
associated with noisy templates.

## Multi-mask Fits

`fit_foreground_templates_multi_mask` fits the same target and templates
under several named weight maps in a single call. The target and templates
are smoothed and filtered once with a shared `master_mask`, then each named
weight map drives an independent weighted GLS solve. This keeps the harmonic
preprocessing identical across all fitted regions, so per-region amplitude
differences come from the per-region weighting and not from boundary-driven
leakage or ringing.

```python
master_mask = build_apodized_union(mask1, mask2)  # any apodized (npix,) float

result = ftf.fit_foreground_templates_multi_mask(
    target_qu=target_qu,
    target_fwhm_in=target_fwhm_rad,
    template_inputs=[dust_split_a],
    template_inputs_rhs=[dust_split_b],
    weight_maps={"low": mask1, "high": mask2},
    master_mask=master_mask,
    fwhm_out=common_fwhm_rad,
    target_filter=filter_config,
)

print(result.fit_names)
for name in result.fit_names:
    print(name, result.fit_results[name].amplitudes)
```

`bootstrap_template_amplitudes_multi_mask` runs the same pattern through
Monte Carlo draws, sharing one noisy realization across all named fits per
draw. The amplitudes for different regions are therefore paired, which is
the right default when differencing them.

```python
bootstrap = ftf.bootstrap_template_amplitudes_multi_mask(
    target_qu=target_qu,
    target_noise_cov=target_noise_cov,
    target_fwhm_in=target_fwhm_rad,
    template_inputs=[dust_split_a],
    template_inputs_rhs=[dust_split_b],
    weight_maps={"low": mask1, "high": mask2},
    master_mask=master_mask,
    fwhm_out=common_fwhm_rad,
    n_mc=200,
    target_filter=filter_config,
    rng=1234,
    show_progress=True,
    n_jobs=4,
)

# amplitude_samples has shape (n_mc, n_fit_mask, n_template).
print(bootstrap.amplitude_samples.shape)
print(bootstrap.amplitude_mean)
print(bootstrap.amplitude_std)
```

See [`docs/multi_mask.md`](./docs/multi_mask.md) for how to construct the
master mask, how it composes multiplicatively with each per-fit
`weight_maps[r]`, the `master_support_threshold` and `master_support_mask`
knobs, and the paired Monte Carlo design.

## Filtering Options

`HarmonicFilter` supports two styles of harmonic filtering:

- Explicit transfer arrays via `ell_filter` and `m_filter`
- Smooth high-pass cutoffs via `ell_cutoff`, `ell_halfwidth`, `m_cutoff`,
  `m_halfwidth`, and `transition_type`

For cutoff-based filters:

- modes below `cutoff - halfwidth` are set to zero
- modes above `cutoff + halfwidth` pass unchanged
- the transition band uses a NaMaster-style `C1` or `C2` edge
- the default transition type is `C2`

If you want reusable explicit filter arrays, the package also exposes:

- `build_ell_filter`
- `build_m_filter`

These helpers build the same high-pass taper used internally by the cutoff
options, but return the window explicitly so you can reuse it across maps.

```python
lmax = 3 * nside - 1

filter_config = ftf.HarmonicFilter(
    ell_filter=ftf.build_ell_filter(
        lmax=lmax,
        cutoff=40.0,
        halfwidth=10.0,
        transition_type="C2",
    ),
    m_filter=ftf.build_m_filter(
        lmax=lmax,
        cutoff=20.0,
        halfwidth=4.0,
        transition_type="C1",
    ),
)
```

## Public API

Most users will interact with:

- `HarmonicFilter`
- `DifferenceTemplateInput`
- `build_ell_filter`
- `build_m_filter`
- `fit_foreground_templates`
- `fit_foreground_templates_multi_mask`
- `bootstrap_template_amplitudes`
- `bootstrap_template_amplitudes_multi_mask`
- `construct_difference_template`
- `smooth_and_filter_qu_map`

If you want the cross-template estimator specifically, pass the left-hand split
through `template_inputs` and the independent right-hand split through
`template_inputs_rhs`. To additionally decouple the data-projection vector
`d_3^T W m`, pass a third split through `template_inputs_data`; when omitted it
defaults to the left-hand stack.

If an input map has a non-Gaussian beam, pass its beam transfer function through
`target_beam_window` for the target or `beam_window_a` / `beam_window_b` on the
corresponding `DifferenceTemplateInput`. The supplied `B_ell` replaces that
map's Gaussian `fwhm_in`; the fitted maps still share the Gaussian output beam
defined by `fwhm_out`. This v1 window is a real, positive, axisymmetric
alm-amplitude response applied identically to E and B, not a power-spectrum
window `B_ell**2`.

One operation resolves one common `lmax` across the target and every left,
right, and data-projection template input. Conflicting explicit values raise an
error; without an explicit value, the map-native `3 * nside - 1` is used.
Custom beams and explicit `ell`/`m` filters shorter than `lmax + 1` are rejected
rather than silently lowering the bandlimit. If any participating map requires
beam matching, filtering, or an explicit bandlimit, every participating map is
sent through the same SHT bandlimit so that pixel-space differences do not mix
different harmonic operators. Non-finite combined beam/filter transfers are
rejected before transforming rather than allowed to produce invalid maps.

A more detailed API reference is available in [`docs/api.md`](./docs/api.md).

## Testing

Run the tests from the repository root:

```bash
python -m pytest -q
```

The current test suite covers:

- recovery of known template amplitudes
- Q/U noise realization from requested covariance
- Monte Carlo sample storage and nonzero uncertainty
- cross-template and data-projection weighted solves
- multi-mask fitting under shared preprocessing
- public `ell`/`m` filter helper construction
- smoothing and `m`-filter integration
- explicit filter-array integration through `smooth_and_filter_qu_map`
- `C1` and `C2` taper behavior for smooth cutoffs

## Notes

- Importing `healpy` may trigger local Matplotlib cache warnings in restricted
  environments. Those do not affect the numerical routines.
- The repository currently focuses on the fitting utilities themselves rather
  than file I/O. You are expected to load maps and noise covariances upstream.
