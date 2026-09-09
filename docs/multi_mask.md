# Multi-mask Fits

`fit_foreground_templates_multi_mask` and
`bootstrap_template_amplitudes_multi_mask` fit the same target and template
stacks under several named weight maps in one call. The target and templates
are smoothed and filtered once under a shared `master_mask`, then each named
weight map drives an independent weighted GLS solve. The Monte Carlo routine
shares one noise realization across all named fits per draw, which keeps the
amplitudes from different masks paired.

## When to Use It

Use this entry point when:

- you want to fit the same templates against the same target under several
  region definitions (for example an SNR-low and an SNR-high split, or a
  galactic-latitude scan), and
- you care about comparing or differencing the recovered amplitudes across
  those regions.

If you only have one weight map, use `fit_foreground_templates` and
`bootstrap_template_amplitudes` directly.

## The Master Mask

### Purpose

`master_mask` defines the single common analysis footprint that the harmonic
preprocessing sees. Every named region in `weight_maps` smooths and filters
the same target and templates against the same apodization. This matters for
two reasons:

- **Systematics consistency.** If each per-region weight (`mask1`, `mask2`,
  ...) were used to apodize its own harmonic transform, every region would
  acquire its own leakage and ringing pattern from a different boundary. The
  recovered amplitudes would then differ partly because of preprocessing
  artifacts rather than sky content. Sharing one `master_mask` makes the
  harmonic preprocessing identical across all fits, so per-region differences
  are driven by the per-region weighting, not by boundary effects.
- **One shared SHT.** The expensive `map2alm`/`almxfl`/`alm2map` round trip
  runs once per target and once per template, regardless of how many named
  fits you request.

### What It Has to Be

A pixel-domain mask of shape `(npix,)` (or any shape `as_weight_map`
accepts) that covers every pixel where you want any of the per-region
`weight_maps[r]` to contribute. Pixels with zero `master_support` are zeroed
in the processed maps, so `weight_maps[r]` values outside `master_support`
are silently ignored regardless of how large they are. Smooth or apodized
edges are strongly recommended; a hard-edged binary mask passed through
`smooth_and_filter_qu_map` will introduce harmonic ringing.

The master mask **does not have to be the union** of the per-region masks.
A larger footprint — for example a single galactic-latitude cut that
contains every region you might fit — is often a better choice:

- The apodization design becomes independent of the per-region mask
  geometry. You can pick the apodization purely to suppress harmonic
  ringing, without distorting it to match the boundary of every small
  analysis region.
- The same preprocessing run can be reused if you later add or change a
  per-region mask, as long as the new mask still falls inside the master
  footprint.
- A smoother, simpler master boundary tends to give cleaner harmonic
  behavior than a tightly fitted union of small disjoint regions.

The union-of-per-region-masks recipe is fine when the per-region masks
already cover the full sky area you care about (the reference notebook does
this). Pick it when you want the apodization band to live just outside your
analysis regions; pick a larger superset when you want to decouple the
apodization geometry from the per-region geometry.

### How to Build One

The package does not require any specific apodization library. A few common
recipes:

```python
# NaMaster: 10 arcmin C2 apodization of the binary union.
import pymaster as nmt
master_mask = ((mask1 + mask2) > 0).astype(float)
master_mask = nmt.mask_apodization(master_mask, 10.0, "C2")
```

```python
# Healpy-only: smooth the binary union with a Gaussian taper.
import healpy as hp
union = ((mask1 + mask2) > 0).astype(float)
master_mask = hp.smoothing(union, fwhm=np.radians(10.0 / 60.0))
master_mask = np.clip(master_mask, 0.0, 1.0)
```

```python
# Bring-your-own apodization. Anything that returns a (npix,) float array
# in [0, 1] supported by every region you plan to fit will work.
master_mask = my_distance_transform_apodization(union, taper_width_deg=0.2)
```

The apodization scale should be wide enough to suppress ringing at the
highest `ell` you care about, but no wider than necessary — the apodized band
is excluded from the binary support (next section) and so does not contribute
to the fit at full weight.

### How `master_mask` and `weight_maps[r]` Combine

For a target pixel `p` and a fit region `r`, the multiplicative chain that
produces the value entering region `r`'s normal matrix is:

1. **Pre-SHT** — every input map is multiplied by `master_mask(p)` in pixel
   space.
2. **Harmonic round-trip** — `map2alm` → beam match × `ell`-filter `almxfl` →
   `alm2map`, followed by the `m`-filter. Beam matching uses Gaussian
   `fwhm_in` values by default, or a supplied custom input `B_ell` for maps that
   define `target_beam_window` or `beam_window_a` / `beam_window_b`. The target
   and every effective template map share the unique explicit
   `HarmonicFilter.lmax`, or map-native `3 * nside - 1` when none is set.
   Conflicting explicit values and beam or filter arrays shorter than
   `lmax + 1` raise errors. If any operand needs harmonic work, all operands go
   through the same SHT bandlimit. The processed map carries the imprint of
   `master_mask`'s apodization profile through the SHT, but no further
   multiplication is applied in harmonic space.
3. **Post-SHT** — the processed map is multiplied by the **binary**
   `master_support(p)` (0 or 1). This zeroes pixels outside the analysis
   footprint without re-applying the apodization profile a second time.
4. **GLS for region `r`** — the processed map is weighted by
   `weight_maps[r](p)` in the normal-matrix accumulation
   `T_left^T W T_right` and the right-hand side `T_left^T W m` (or
   `d_3^T W m` when an independent `template_inputs_data` stack is supplied).

The effective per-pixel weight that ends up in the normal matrix is therefore

```text
W_eff(p, r) = master_support(p) * weight_maps[r](p)
```

with the smoothed signal in `T_left`, `T_right`, and `m` already reflecting
the earlier `master_mask` apodization through the SHT. Two consequences:

- The apodization values in `master_mask` are **not** a multiplicative GLS
  weight. They shape the pre-SHT input only.
- If you want a boundary taper inside the GLS — for example to down-weight
  the apodized transition region — bake that taper into `weight_maps[r]`
  itself. The post-filter cut is intentionally binary so the master
  apodization is not applied a second time.

## Fit Example

```python
import numpy as np
import fg_weighted_template_fit as ftf

# Reuse the dust template setup from the README's Quick Start.
weight_maps = {
    "low": mask1,   # SNR-low region
    "high": mask2,  # SNR-high region
}

result = ftf.fit_foreground_templates_multi_mask(
    target_qu=target_qu,
    target_fwhm_in=target_fwhm_rad,
    template_inputs=[dust_split_a],
    template_inputs_rhs=[dust_split_b],
    weight_maps=weight_maps,
    master_mask=master_mask,
    fwhm_out=common_fwhm_rad,
    target_filter=filter_config,
)

print(result.fit_names)        # ('low', 'high')
print(result.template_names)   # ('dust',)

for fit_name in result.fit_names:
    fit = result.fit_results[fit_name]
    print(fit_name, fit.amplitudes)

# Processed maps are shared across all fits — useful for diagnostics.
shared_target = result.processed_target_qu
```

`MultiMaskFitResult.fit_results[name]` is a normal `WeightedFitResult`, so
the per-region residual map, normal matrix, and amplitude vector are all
available the same way as in the single-mask path.

To decouple the data-projection vector from the normal-matrix factors, pass a
third stack through `template_inputs_data`. The per-region solve then becomes
`(d_1^T W d_2)^-1 d_3^T W m`, where `d_1 = template_inputs`,
`d_2 = template_inputs_rhs`, and finite `d_3 = template_inputs_data` samples
enter only the right-hand vector. When `template_inputs_data` is omitted, `d_3`
defaults to the left-hand stack `d_1`, reproducing the existing behavior. The
same keyword is accepted by `bootstrap_template_amplitudes_multi_mask`, which
realizes an independent noise draw on the `d_3` maps when they carry
covariances. Non-finite samples in any template stack are removed from the shared
fit support before the per-region solves.

## Bootstrap Example

```python
bootstrap = ftf.bootstrap_template_amplitudes_multi_mask(
    target_qu=target_qu,
    target_noise_cov=target_noise_cov,
    target_fwhm_in=target_fwhm_rad,
    template_inputs=[dust_split_a],
    template_inputs_rhs=[dust_split_b],
    weight_maps=weight_maps,
    master_mask=master_mask,
    fwhm_out=common_fwhm_rad,
    n_mc=200,
    target_filter=filter_config,
    rng=1234,
    show_progress=True,
    n_jobs=4,
)

# Shape is (n_mc, n_fit_mask, n_template).
print(bootstrap.amplitude_samples.shape)
print(bootstrap.amplitude_mean)   # shape (n_fit_mask, n_template)
print(bootstrap.amplitude_std)    # shape (n_fit_mask, n_template)

# Paired difference between the two regions for the first template.
low_idx = bootstrap.fit_names.index("low")
high_idx = bootstrap.fit_names.index("high")
delta = (
    bootstrap.amplitude_samples[:, low_idx, 0]
    - bootstrap.amplitude_samples[:, high_idx, 0]
)
print(delta.mean(), delta.std(ddof=1))
```

## Support Knob

`master_support` is the binary post-filter factor in the multiplicative
chain above. Two ways to set it:

- Default — derived from `master_mask` and `master_support_threshold`.
  Pixels survive when `master_mask` is finite and strictly greater than the
  threshold. The default `master_support_threshold=0.0` keeps every pixel
  with positive apodization, i.e. the full apodized footprint.
- Override — pass `master_support_mask=...` directly. Any finite nonzero
  pixel keeps its sample. The threshold is bypassed entirely. Useful when
  the support comes from a separate analysis cut (a hit-count map, a
  scan-coverage mask, or a hand-drawn region).

The threshold is the right knob if you want to drop the very edge of the
apodization band; the override is the right knob if your support is defined
independently of the apodization at all.

## Paired Monte Carlo Draws

`bootstrap_template_amplitudes_multi_mask` realizes one noisy target and one
noisy set of template inputs per Monte Carlo draw, then fits every named
weight map against that same noisy realization. As a result,
`bootstrap.amplitude_samples[:, i, :]` and
`bootstrap.amplitude_samples[:, j, :]` are correlated — the noise component
of their difference partially cancels.

This is the right default when comparing or differencing per-region
amplitudes (`amplitude_samples[:, i, :] - amplitude_samples[:, j, :]`):
shared noise drops out, and the reported `delta.std` reflects how much the
sky and template-construction noise actually move the per-region difference.

If you instead need statistically independent per-region bootstraps (for
example to combine with another analysis that assumes independent draws),
run `bootstrap_template_amplitudes` separately for each weight map with
different `rng` seeds. The threaded execution and progress conventions are
identical to `bootstrap_template_amplitudes` and follow the design in
[`parallel_bootstrap.md`](./parallel_bootstrap.md).

## Scientific Considerations

A few choices in this pipeline interact in ways that are easy to get wrong.

### Apodization Width

The pre-SHT apodization in `master_mask` controls how much mode-mixing
leaks into the smoothed maps near the boundary. As a rule of thumb, the
apodization width should comfortably exceed `pi / ell_max` for the highest
`ell` you actually fit — i.e. the apodization should transition over more
than one mode at your finest scale. A 10 arcmin C2 taper is more than wide
enough for an `ell` high-pass at `~180`. Going wider buys diminishing
returns and shrinks the usable footprint; going narrower lets harmonic
ringing into the analysis region.

### Weight-Map Design

Inside the master support, the effective per-pixel weight on region `r`
is just `weight_maps[r](p)`. Choices:

- **Inverse-variance weights** (`1 / sigma_p^2`) are the natural choice
  when per-pixel noise variance is well measured and noise is roughly
  diagonal in pixel space. This recovers a near-optimal diagonal-GLS
  estimator on the analysis region.
- **Binary region masks** give an unweighted least-squares fit on the
  region. This is the right choice when you want the amplitude to reflect
  the average sky inside the region without weighting by noise structure.
- **Products** (e.g. `region_mask * inverse_variance`) combine the two
  and are common in practice.

The estimator is invariant to a global scaling of any single
`weight_maps[r]`, so you do not need to normalize the weights to compare
amplitudes across regions.

### Cross-Template Estimator Removes Auto-Bias

Pass independent template realizations through `template_inputs` and
`template_inputs_rhs` (e.g. half-mission half-A and half-B of Planck 353
and 217). With the same realization on both sides, noise in the templates
appears in both factors of the normal matrix `T_left^T W T_right` and
biases the recovered amplitudes. The cross estimator is especially
important when comparing amplitudes across regions, because the
auto-bias depends on the per-region weighting and can therefore differ
from region to region.

### What the Bootstrap Captures and What It Doesn't

`bootstrap_template_amplitudes_multi_mask` propagates:

- target-map noise from `target_noise_cov`,
- template-map noise from `noise_cov_a` / `noise_cov_b` on each
  `DifferenceTemplateInput` in the left, right-hand, and data-projection stacks,
- the effect of rebuilding the templates (smoothing, filtering, masking)
  after each noise draw.

It does **not** capture:

- pixel–pixel noise correlations in either the target or the templates.
  The fit weight is diagonal in pixel space and the noise realizer treats
  pixels independently, so any residual stripes, atmospheric correlations,
  or scan-induced covariance are not modeled. The reported
  `amplitude_std` will underestimate the true uncertainty when those
  correlations are significant.
- E↔B leakage from masking. The pipeline operates in Q/U directly, so
  leakage induced by the boundary stays in the residuals. Because the
  same `master_mask` is used for the target and every template, the
  leakage acts on both sides of `T_left^T W T_right` and largely
  divides out for amplitude recovery, but it does not vanish entirely.
- modeling errors in the templates themselves (frequency dependence,
  spatial variation of the dust SED, etc.). The bootstrap quantifies
  noise propagation, not model adequacy.

### Beam Matching and Filter Choice

For maps without custom beam windows, `fwhm_out` should be at least as large as
the largest `fwhm_in`; smoothing to a finer beam acts as deconvolution and
amplifies high-`ell` noise. For maps with custom input beam windows, the
transfer is the Gaussian output beam divided by the supplied `B_ell`, so callers
should choose `fwhm_out` and harmonic cutoffs with the same deconvolution risk in
mind. A custom window is a real, finite, strictly positive, axisymmetric
alm-amplitude `B_ell`, applied equally to E and B rather than supplied as
`B_ell**2`. Separate E/B responses, asymmetric or `m`-dependent beams, and
cross-polar response are not supported. `fwhm_out` and every Gaussian input
FWHM must be finite and nonnegative, and the completed harmonic transfer must
remain finite. The `ell` high-pass and `m` cutoffs in `HarmonicFilter` are the
standard place to suppress whatever modes are dominated by signal or noise
components you do not want in the fit (large-scale CMB, scan-aligned 1/f,
etc.) and should be chosen jointly with the apodization width above.
