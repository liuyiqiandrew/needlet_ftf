# API Reference

This document summarizes the main public objects exposed by
`fg_weighted_template_fit`.

## Core Dataclasses

### `HarmonicFilter`

Configuration for optional harmonic-domain filtering.

Fields:

- `ell_filter`: explicit multiplicative transfer function indexed by `ell`
- `m_filter`: explicit multiplicative transfer function indexed by `m`
- `ell_cutoff`: optional high-pass multipole cutoff
- `ell_halfwidth`: half-width of the smooth `ell` transition
- `m_cutoff`: optional high-pass azimuthal cutoff
- `m_halfwidth`: half-width of the smooth `m` transition
- `transition_type`: `"C1"` or `"C2"` smooth edge, default `"C2"`
- `lmax`: optional explicit harmonic truncation
- `iter`: number of `healpy.map2alm` iterations

Usage notes:

- If both an explicit filter and a cutoff are provided, both are applied.
- `build_ell_filter` and `build_m_filter` return explicit windows that can be
  passed to `ell_filter` and `m_filter`.
- `ell_cutoff` and `m_cutoff` suppress low modes but do not lower the transform
  `lmax`.
- A complete preprocessing operation uses the unique explicit `lmax` found
  across its effective target and template filters. Conflicting explicit
  values raise `ValueError`; if none is set, the map-native
  `3 * nside - 1` is used.
- Explicit `ell_filter` and `m_filter` arrays must contain at least
  `lmax + 1` values. Array length never lowers the resolved `lmax`.
- All beam widths used with filtering are in radians.

### `DifferenceTemplateInput`

Definition of one template built from a difference of two Q/U maps.

Fields:

- `map_a_qu`
- `map_b_qu`
- `fwhm_in_a`
- `fwhm_in_b`
- `noise_cov_a`
- `noise_cov_b`
- `filter_config`
- `name`
- `beam_window_a`
- `beam_window_b`

Usage notes:

- The template is built as `processed(map_a_qu) - processed(map_b_qu)`.
- Noise covariances are optional, but they are needed for Monte Carlo
  uncertainty propagation through template construction.
- `beam_window_a` and `beam_window_b` are optional input beam transfer functions
  indexed by `ell`; when supplied, they replace the Gaussian beam implied by
  `fwhm_in_a` or `fwhm_in_b`.
- Custom beam windows are real, finite, strictly positive, axisymmetric
  alm-amplitude responses applied equally to E and B. They are not
  power-spectrum windows `B_ell**2`.

### `WeightedFitResult`

Container returned by the weighted fit.

Fields:

- `amplitudes`
- `normal_matrix`
- `normal_matrix_inverse`
- `rhs`
- `residual_qu`
- `processed_target_qu`
- `processed_templates_qu`
- `processed_templates_rhs_qu`
- `processed_templates_data_qu`
- `template_names`
- `solver`

### `BootstrapFitResult`

Container returned by the Monte Carlo uncertainty routine.

Fields:

- `reference_fit`
- `amplitude_samples`
- `amplitude_mean`
- `amplitude_std`
- `template_names`

### `MultiMaskFitResult`

Container returned by the multi-mask weighted fit.

Fields:

- `fit_names`
- `fit_results`
- `template_names`
- `processed_target_qu`
- `processed_templates_qu`
- `processed_templates_rhs_qu`
- `processed_templates_data_qu`

### `MultiMaskBootstrapResult`

Container returned by the multi-mask Monte Carlo uncertainty routine.

Fields:

- `reference_fit`
- `amplitude_samples`
- `amplitude_mean`
- `amplitude_std`
- `fit_names`
- `template_names`

## Main Functions

### `build_ell_filter`

```python
build_ell_filter(
    lmax,
    *,
    cutoff,
    halfwidth=0.0,
    transition_type="C2",
)
```

Builds a reusable high-pass `ell`-space transfer function with shape
`(lmax + 1,)`.

Important behavior:

- returns the same taper shape used by `HarmonicFilter.ell_cutoff`
- uses a hard cutoff when `halfwidth == 0`
- supports `C1` and `C2` smooth edges
- is intended for `HarmonicFilter(ell_filter=...)`

### `build_m_filter`

```python
build_m_filter(
    lmax,
    *,
    cutoff,
    halfwidth=0.0,
    transition_type="C2",
)
```

Builds a reusable high-pass `m`-space transfer function with shape
`(lmax + 1,)`.

Important behavior:

- returns the same taper shape used by `HarmonicFilter.m_cutoff`
- uses a hard cutoff when `halfwidth == 0`
- supports `C1` and `C2` smooth edges
- is intended for `HarmonicFilter(m_filter=...)`

### `smooth_and_filter_qu_map`

```python
smooth_and_filter_qu_map(
    qu_map,
    fwhm_in,
    fwhm_out,
    *,
    beam_window_in=None,
    filter_config=None,
    mask=None,
    nest=False,
)
```

Applies beam matching and optional harmonic filtering to a Q/U Healpix map.

Important behavior:

- accepts shape `(2, npix)` or `(npix, 2)`
- requires finite, nonnegative `fwhm_out`; `fwhm_in` must be finite and
  nonnegative when it supplies the Gaussian input beam
- raises if `fwhm_out < fwhm_in` when no custom input beam is supplied
- when `beam_window_in` is supplied, uses `B_out_ell / beam_window_in[ell]`
  instead of Gaussian input-beam matching; `B_out_ell` is the Gaussian output
  beam from `fwhm_out`
- custom beam windows must be one-dimensional, real, finite, and strictly
  positive, and must contain at least `lmax + 1` values
- the custom input window and Gaussian output window use a scalar,
  axisymmetric alm-amplitude `B_ell` applied equally to E and B; `B_ell**2`,
  separate E/B responses, asymmetric beams, and cross-polar response are not
  supported
- explicit `ell`/`m` filters must also contain at least `lmax + 1` values;
  short arrays raise instead of implicitly truncating harmonic support
- custom beam matching can deconvolve modes where `B_out_ell / B_in_ell > 1`,
  so choose `fwhm_out`, explicit `lmax`, and harmonic filters to avoid
  unstable high-`ell` amplification
- the complete ell and m transfers are validated as finite before `map2alm`;
  an overflow or invalid transfer raises instead of producing an `inf`/`nan`
  map
- an explicit `lmax` requests a harmonic bandlimit even when the beam and
  filters would otherwise be identities
- when a mask is supplied, applies it in pixel space before the harmonic
  transform used for smoothing/filtering
- uses a single alm pass to combine smoothing and filtering
- supports both RING and NEST ordering

### `construct_difference_template`

```python
construct_difference_template(
    map_a_qu,
    map_b_qu,
    fwhm_in_a,
    fwhm_in_b,
    fwhm_out,
    *,
    beam_window_a=None,
    beam_window_b=None,
    filter_config=None,
    mask=None,
    nest=False,
)
```

Builds a foreground template from two Q/U maps after matching both to the same
output resolution and filter definition.

When `beam_window_a` or `beam_window_b` is supplied, that map's custom `B_ell`
replaces the Gaussian input beam implied by its `fwhm_in_*` value. Both maps
must have the same Healpix resolution and use one common `lmax`. If either map
needs harmonic work, both are transformed with that `lmax`; this includes
projecting an otherwise identity operand through the same bandlimit.

### `build_template_stack`

```python
build_template_stack(
    template_inputs,
    *,
    fwhm_out,
    default_filter=None,
    mask=None,
    nest=False,
)
```

Constructs a stack of processed difference templates and returns
`(templates, template_names)`. Every map in the stack must have the same
Healpix resolution. The stack resolves one common `lmax`, and if any operand
requires harmonic work, every operand uses the same SHT bandlimit.

### `weighted_template_gls`

```python
weighted_template_gls(
    target_qu,
    templates_qu,
    weight_map,
    *,
    templates_rhs_qu=None,
    templates_data_qu=None,
    mask=None,
    template_names=None,
)
```

Solves the weighted normal equations

```text
(d_1^T W d_2) a = d_3^T W m
```

with Q/U pixels stacked into one data vector, where `d_1 = templates_qu`,
`d_2 = templates_rhs_qu`, and `d_3 = templates_data_qu`.

Accepted weight shapes:

- `(npix,)`
- `(2, npix)`
- `(npix, 2)`
- scalar

Important behavior:

- non-finite target, template, and weight entries are automatically removed
- if a mask is supplied, the solver converts it to binary support instead of
  using apodized mask values as extra weights
- `templates_qu` is the left-hand template stack `d_1`
- `templates_rhs_qu` is the right-hand template stack `d_2`
- if `templates_rhs_qu` is omitted, the routine falls back to the
  same-template normal matrix
- `templates_data_qu` is the data-projection stack `d_3` used only in the
  right-hand vector `d_3^T W m`; if omitted, it defaults to `d_1`
- non-finite values in any stack, including `d_3`, remove that Q/U sample from
  the shared support used by both the normal matrix and right-hand vector
- if the normal matrix is singular, the routine falls back to a pseudoinverse

### `fit_foreground_templates`

```python
fit_foreground_templates(
    target_qu,
    target_fwhm_in,
    template_inputs,
    weight_map,
    fwhm_out,
    *,
    target_beam_window=None,
    template_inputs_rhs=None,
    template_inputs_data=None,
    target_filter=None,
    mask=None,
    nest=False,
)
```

High-level entry point that:

- smooths and filters the target map
- constructs the left-hand template stack
- optionally constructs an independent right-hand template stack
- optionally constructs an independent data-projection stack for the right-hand
  vector
- solves for the weighted template amplitudes

Procedure overview:

1. The target and every map in the effective left, right, and data-projection
   template stacks are checked for one common Healpix resolution and harmonic
   plan. The unique explicit `HarmonicFilter.lmax` is used when present;
   conflicting values raise, and otherwise `3 * nside - 1` is used.
2. If any operand requires beam matching, filtering, or an explicit `lmax`, all
   operands use an SHT with the common `lmax`. If all operands are harmonic
   identities, the operation retains the pure-NumPy path.
3. During map processing, any NEST input is reordered to RING in
   pixel space before harmonic transforms.
4. If `mask` is supplied, that same mask is first applied in pixel space to
   the target map before any harmonic transform so an apodized edge can reduce
   truncation ringing.
5. Beam matching and any `ell`-space filter are combined into one multiplicative
   transfer function in harmonic space. By default this matches the Gaussian
   input beam `target_fwhm_in` to `fwhm_out`; if `target_beam_window` is
   supplied, it uses the custom target `B_ell` instead of `target_fwhm_in`.
6. Any `m`-space filter is then applied in harmonic space after the `ell`-space
   transfer and before transforming back to map space.
7. The filtered target is transformed back to Q/U pixel space, and converted
   back to NEST ordering if requested.
8. The left-hand template stack is built next. For each
   `DifferenceTemplateInput`, `map_a_qu` and `map_b_qu` are each processed
   through the same masked smooth-and-filter pipeline, using the entry's own
   `filter_config` when present or `target_filter` otherwise.
9. Each template difference `processed(map_a_qu) - processed(map_b_qu)` is
   formed in pixel space after both maps have already been beam-matched and
   harmonically filtered.
10. If `template_inputs_rhs` is omitted, the right-hand stack reuses the
   already-built left-hand templates. If it is supplied, the right-hand stack is
   built afterward with the same per-template ordering. The same rule applies to
   `template_inputs_data`: when omitted the data-projection stack `d_3` defaults
   to the left-hand templates, otherwise it is built independently.
11. Only after all target and template preprocessing is complete does the
   routine enter `weighted_template_gls`, where the user-supplied weights are
   applied in pixel space, non-finite samples are zero-weighted, the weighted
   normal matrix and right-hand side are accumulated, and the amplitudes are
   solved.

Space/order notes:

- Harmonic-space work is limited to the per-map preprocessing inside
  `smooth_and_filter_qu_map`.
- Pixel-space work covers NEST reordering, template differencing, pre-harmonic
  apodization, weight application, normal-matrix accumulation, model
  construction, and residual formation.
- The code never subtracts `map_a_qu - map_b_qu` in harmonic space, and it does
  not apply the fit weights in harmonic space.

### `fit_foreground_templates_multi_mask`

```python
fit_foreground_templates_multi_mask(
    target_qu,
    target_fwhm_in,
    template_inputs,
    weight_maps,
    fwhm_out,
    *,
    master_mask,
    target_beam_window=None,
    template_inputs_rhs=None,
    template_inputs_data=None,
    target_filter=None,
    master_support_mask=None,
    master_support_threshold=0.0,
    nest=False,
)
```

High-level entry point for fitting multiple named regions after one common
harmonic preprocessing pass.

Important behavior:

- `weight_maps` is a mapping such as `{"m1": mask1, "m2": mask2}`; insertion
  order defines the `fit_names` order
- `master_mask` is applied before smoothing/filtering for the target and all
  template input maps
- `target_beam_window` and template-entry `beam_window_a` / `beam_window_b`
  follow the same custom input-beam rules as the single-mask fit
- the target and every effective template operand share one resolved `lmax`;
  if any operand requires harmonic work, all operands use that SHT bandlimit
- after smoothing/filtering, processed maps are multiplied by binary master
  support, not by the apodized mask values again
- default support is `isfinite(master_mask) & (master_mask >
  master_support_threshold)`
- if `master_support_mask` is supplied, finite nonzero pixels define support
  and override the threshold-derived support
- each named fit calls `weighted_template_gls` with its own `weight_map` and no
  GLS `mask`, so fitting masks act as weights only
- `template_inputs_data` follows the same ordering and length requirements as
  `template_inputs`; when omitted, the data-projection stack defaults to the
  left-hand stack

### `realize_qu_noise`

```python
realize_qu_noise(
    pixel_cov_qu,
    *,
    rng=None,
)
```

Draws a Q/U noise realization from per-pixel covariance in the order
`QQ`, `UU`, `QU`.

Accepted covariance shapes:

- `(3, npix)`
- `(npix, 3)`

### `bootstrap_template_amplitudes`

```python
bootstrap_template_amplitudes(
    target_qu,
    target_noise_cov,
    target_fwhm_in,
    template_inputs,
    weight_map,
    fwhm_out,
    *,
    n_mc,
    target_beam_window=None,
    template_inputs_rhs=None,
    template_inputs_data=None,
    target_filter=None,
    mask=None,
    nest=False,
    rng=None,
    show_progress=False,
    n_jobs=1,
)
```

Runs Monte Carlo amplitude estimation by:

1. realizing noise for the target map
2. realizing noise for the maps used to construct each left-hand template
3. optionally realizing noise for the independent right-hand template stack and
   the independent data-projection stack
4. rebuilding templates at the target output resolution
5. reapplying harmonic filtering
6. refitting amplitudes
7. storing the recovered amplitudes from every draw

The output spread is reported through `amplitude_std`, and every draw is kept in
`amplitude_samples`.

Usage note:

- `amplitude_std` includes template-map uncertainty when the corresponding
  `DifferenceTemplateInput` entries provide `noise_cov_a` and `noise_cov_b`
- target and template custom beam windows are preserved for the reference fit
  and every Monte Carlo draw
- if those template noise covariances are omitted, the Monte Carlo spread only
  reflects target-map noise and any uncertainty induced by the supplied target
  covariance
- `show_progress=True` displays a standard `tqdm` progress bar over the Monte
  Carlo draws and does not depend on ipywidgets
- `n_jobs > 1` runs independent Monte Carlo draws with worker threads, which can
  be used from JupyterLab without multiprocessing setup
- threaded runs are reproducible for the same `rng` and `n_jobs`, but are not
  required to match the exact serial sample sequence

The parallel execution design is documented in
[`parallel_bootstrap.md`](./parallel_bootstrap.md).

### `bootstrap_template_amplitudes_multi_mask`

```python
bootstrap_template_amplitudes_multi_mask(
    target_qu,
    target_noise_cov,
    target_fwhm_in,
    template_inputs,
    weight_maps,
    fwhm_out,
    *,
    n_mc,
    master_mask,
    target_beam_window=None,
    template_inputs_rhs=None,
    template_inputs_data=None,
    target_filter=None,
    master_support_mask=None,
    master_support_threshold=0.0,
    nest=False,
    rng=None,
    show_progress=False,
    n_jobs=1,
)
```

Runs paired Monte Carlo amplitude estimation for every named weight map.

Important behavior:

- each draw realizes one noisy target/template set
- that draw is smoothed/filtered once under `master_mask`
- every named weight map is fit against the same processed noisy realization
- `amplitude_samples` has shape `(n_mc, n_fit_mask, n_template)`
- `fit_names` and `template_names` label the second and third sample axes
- progress and threaded execution follow the same rules as
  `bootstrap_template_amplitudes`
- custom target and template input beams follow the same rules as
  `fit_foreground_templates_multi_mask`

## Data Conventions

- Q/U maps may be passed as `(2, npix)` or `(npix, 2)`.
- Template stacks may be passed as `(n_template, 2, npix)` or
  `(n_template, npix, 2)`.
- Custom beam windows are one-dimensional, real, finite, strictly positive
  axisymmetric alm-amplitude arrays indexed by `ell`, applied equally to E and
  B. They are not `B_ell**2` power-spectrum windows.
- Per-pixel covariance must be ordered as `QQ`, `UU`, `QU`.
- FWHM values are always in radians. `fwhm_out` is finite and nonnegative;
  each `fwhm_in` used for Gaussian input-beam matching is also finite and
  nonnegative.

## Internal Helpers

The package re-exports `_build_apodized_highpass` for testing and inspection.
It is not intended to be the main user-facing entry point, but it can be useful
when validating filter shapes.
