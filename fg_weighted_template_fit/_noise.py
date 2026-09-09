from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Iterable, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from ._arrays import as_covariance, as_qu_map, coerce_rng
from ._fit import fit_foreground_templates, fit_foreground_templates_multi_mask
from ._types import (
    BootstrapFitResult,
    DifferenceTemplateInput,
    FloatArray,
    HarmonicFilter,
    MultiMaskBootstrapResult,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is unavailable.
    _tqdm = None

_ProgressItem = TypeVar("_ProgressItem")


def bootstrap_template_amplitudes(
    target_qu: npt.ArrayLike,
    target_noise_cov: npt.ArrayLike,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_map: npt.ArrayLike,
    fwhm_out: float,
    *,
    n_mc: int,
    target_beam_window: npt.ArrayLike | None = None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None = None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None = None,
    target_filter: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
    rng: np.random.Generator | int | None = None,
    show_progress: bool = False,
    n_jobs: int = 1,
) -> BootstrapFitResult:
    """Estimate template amplitude uncertainty with Monte Carlo noise draws.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    target_noise_cov
        Per-pixel target-map covariance in the order ``QQ, UU, QU`` with shape
        ``(3, npix)`` or ``(npix, 3)``.
    target_fwhm_in
        Beam FWHM of the target map in radians.
    template_inputs
        Sequence of left-hand template definitions used in the fit.
    weight_map
        Diagonal pixel weight map used in the fit.
    fwhm_out
        Common output beam FWHM in radians.
    n_mc
        Number of Monte Carlo realizations.
    target_beam_window
        Optional input beam transfer function ``B_ell`` for the target map. When
        supplied, it replaces the Gaussian beam implied by ``target_fwhm_in``.
    template_inputs_rhs
        Optional sequence of right-hand template definitions for the cross
        normal matrix.
    template_inputs_data
        Optional sequence of data-projection template definitions ``d_3`` used
        only in the right-hand vector ``d_3^T W m``. Each Monte Carlo draw
        realizes its own noise on these maps when covariances are provided.
    target_filter
        Optional harmonic filter applied to the target map. Template entries may
        override this with their own ``filter_config`` values.
    mask
        Optional binary or floating fit mask. When provided, each Monte Carlo
        draw applies the same mask before any harmonic preprocessing so the
        apodized edge treatment matches the reference fit.
    nest
        If ``True``, maps are treated as NEST ordered during harmonic
        transforms.
    rng
        Existing random generator, integer seed, or ``None``.
    show_progress
        If ``True``, display a standard ``tqdm`` progress bar over the Monte
        Carlo draws. This avoids relying on notebook widget frontends.
    n_jobs
        Number of worker threads used for Monte Carlo draws. The default
        ``1`` preserves the serial execution path. Values greater than one use
        independent per-draw random generators.

    Returns
    -------
    BootstrapFitResult
        Reference fit together with the amplitude recovered from each Monte
        Carlo realization.

    Raises
    ------
    ValueError
        If ``n_mc`` or ``n_jobs`` is not positive.
    ImportError
        If ``show_progress`` is ``True`` but ``tqdm`` is not installed.

    Notes
    -----
    Threaded execution uses a ``ThreadPoolExecutor`` so the function can be
    called directly from JupyterLab cells without multiprocessing setup. In
    threaded mode, one deterministic child seed is generated per draw before
    work is submitted, and each worker owns its local random generator.
    """

    if n_mc <= 0:
        raise ValueError("n_mc must be a positive integer.")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer.")

    reference_fit = fit_foreground_templates(
        target_qu=target_qu,
        target_fwhm_in=target_fwhm_in,
        template_inputs=template_inputs,
        weight_map=weight_map,
        fwhm_out=fwhm_out,
        target_beam_window=target_beam_window,
        template_inputs_rhs=template_inputs_rhs,
        template_inputs_data=template_inputs_data,
        target_filter=target_filter,
        mask=mask,
        nest=nest,
    )

    rng_obj = coerce_rng(rng)
    target = as_qu_map(target_qu, name="target_qu")
    target_noise_cov_qu = as_covariance(
        target_noise_cov,
        npix=reference_fit.processed_target_qu.shape[1],
        name="target_noise_cov",
    )

    samples = np.zeros((n_mc, len(reference_fit.template_names)), dtype=np.float64)
    if n_jobs == 1:
        draw_indices = _wrap_progress(
            range(n_mc),
            show_progress=show_progress,
            total=n_mc,
            desc="Bootstrap MC",
        )
        for draw_index in draw_indices:
            draw_index, amplitudes = _fit_bootstrap_draw(
                draw_index=draw_index,
                target_qu=target,
                target_noise_cov_qu=target_noise_cov_qu,
                target_fwhm_in=target_fwhm_in,
                template_inputs=template_inputs,
                weight_map=weight_map,
                fwhm_out=fwhm_out,
                target_beam_window=target_beam_window,
                template_inputs_rhs=template_inputs_rhs,
                template_inputs_data=template_inputs_data,
                target_filter=target_filter,
                mask=mask,
                nest=nest,
                rng=rng_obj,
            )
            samples[draw_index] = amplitudes
    else:
        if show_progress:
            _require_tqdm()
        # Derive all child seeds before starting worker threads. This keeps the
        # parent RNG single-threaded and gives each draw an independent stream.
        draw_seeds: npt.NDArray[np.int64] = rng_obj.integers(
            0,
            np.iinfo(np.int64).max,
            size=n_mc,
            dtype=np.int64,
        )
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _fit_bootstrap_draw,
                    draw_index=draw_index,
                    target_qu=target,
                    target_noise_cov_qu=target_noise_cov_qu,
                    target_fwhm_in=target_fwhm_in,
                    template_inputs=template_inputs,
                    weight_map=weight_map,
                    fwhm_out=fwhm_out,
                    target_beam_window=target_beam_window,
                    template_inputs_rhs=template_inputs_rhs,
                    template_inputs_data=template_inputs_data,
                    target_filter=target_filter,
                    mask=mask,
                    nest=nest,
                    rng=int(draw_seed),
                )
                for draw_index, draw_seed in enumerate(draw_seeds)
            ]
            completed_draws = _wrap_progress(
                as_completed(futures),
                show_progress=show_progress,
                total=n_mc,
                desc="Bootstrap MC",
            )
            for future in completed_draws:
                # Futures complete out of order, so each worker returns the row
                # where its amplitudes should be stored.
                draw_index, amplitudes = future.result()
                samples[draw_index] = amplitudes

    ddof = 1 if n_mc > 1 else 0
    return BootstrapFitResult(
        reference_fit=reference_fit,
        amplitude_samples=samples,
        amplitude_mean=np.mean(samples, axis=0),
        amplitude_std=np.std(samples, axis=0, ddof=ddof),
        template_names=reference_fit.template_names,
    )


def bootstrap_template_amplitudes_multi_mask(
    target_qu: npt.ArrayLike,
    target_noise_cov: npt.ArrayLike,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_maps: Mapping[str, npt.ArrayLike],
    fwhm_out: float,
    *,
    n_mc: int,
    master_mask: npt.ArrayLike,
    target_beam_window: npt.ArrayLike | None = None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None = None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None = None,
    target_filter: HarmonicFilter | None = None,
    master_support_mask: npt.ArrayLike | None = None,
    master_support_threshold: float = 0.0,
    nest: bool = False,
    rng: np.random.Generator | int | None = None,
    show_progress: bool = False,
    n_jobs: int = 1,
) -> MultiMaskBootstrapResult:
    """Estimate amplitude uncertainty for multiple weights with paired draws.

    Each Monte Carlo draw realizes one noisy target and one noisy set of
    template inputs, smooths/filters them once under ``master_mask``, and then
    fits every named weight map against that same realization. The shared draw
    convention preserves covariance between fitted regions, which is important
    for derived quantities such as the difference between two masks.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    target_noise_cov
        Per-pixel target-map covariance in the order ``QQ, UU, QU`` with shape
        ``(3, npix)`` or ``(npix, 3)``.
    target_fwhm_in
        Beam FWHM of the target map in radians.
    template_inputs
        Sequence of left-hand template definitions used in the fit.
    weight_maps
        Mapping from fit-region name to diagonal pixel weight map. Insertion
        order defines the fit-mask axis in the returned arrays.
    fwhm_out
        Common output beam FWHM in radians.
    n_mc
        Number of Monte Carlo realizations.
    master_mask
        Binary or apodized mask applied to every target/template input before
        harmonic smoothing/filtering.
    target_beam_window
        Optional input beam transfer function ``B_ell`` for the target map. When
        supplied, it replaces the Gaussian beam implied by ``target_fwhm_in``.
    template_inputs_rhs
        Optional sequence of right-hand template definitions for the cross
        normal matrix.
    template_inputs_data
        Optional sequence of data-projection template definitions ``d_3`` used
        only in the right-hand vector ``d_3^T W m``. Each Monte Carlo draw
        realizes its own noise on these maps when covariances are provided.
    target_filter
        Optional harmonic filter applied to the target map. Template entries may
        override this with their own ``filter_config`` values.
    master_support_mask
        Optional explicit post-filter support mask. Finite nonzero samples are
        kept. When omitted, support is derived from ``master_mask`` and
        ``master_support_threshold``.
    master_support_threshold
        Threshold used for support derived from ``master_mask``.
    nest
        If ``True``, maps are treated as NEST ordered during harmonic
        transforms.
    rng
        Existing random generator, integer seed, or ``None``.
    show_progress
        If ``True``, display a standard ``tqdm`` progress bar over the Monte
        Carlo draws.
    n_jobs
        Number of worker threads used for Monte Carlo draws.

    Returns
    -------
    MultiMaskBootstrapResult
        Reference fit and amplitude samples with shape
        ``(n_mc, n_fit_mask, n_template)``.

    Raises
    ------
    ValueError
        If ``n_mc`` or ``n_jobs`` is not positive, or if any forwarded
        multi-mask fit input is invalid.
    ImportError
        If ``show_progress`` is ``True`` but ``tqdm`` is not installed.
    """

    if n_mc <= 0:
        raise ValueError("n_mc must be a positive integer.")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer.")

    reference_fit = fit_foreground_templates_multi_mask(
        target_qu=target_qu,
        target_fwhm_in=target_fwhm_in,
        template_inputs=template_inputs,
        weight_maps=weight_maps,
        fwhm_out=fwhm_out,
        master_mask=master_mask,
        target_beam_window=target_beam_window,
        template_inputs_rhs=template_inputs_rhs,
        template_inputs_data=template_inputs_data,
        target_filter=target_filter,
        master_support_mask=master_support_mask,
        master_support_threshold=master_support_threshold,
        nest=nest,
    )

    rng_obj = coerce_rng(rng)
    target = as_qu_map(target_qu, name="target_qu")
    target_noise_cov_qu = as_covariance(
        target_noise_cov,
        npix=reference_fit.processed_target_qu.shape[1],
        name="target_noise_cov",
    )

    samples = np.zeros(
        (n_mc, len(reference_fit.fit_names), len(reference_fit.template_names)),
        dtype=np.float64,
    )
    if n_jobs == 1:
        draw_indices = _wrap_progress(
            range(n_mc),
            show_progress=show_progress,
            total=n_mc,
            desc="Bootstrap MC",
        )
        for draw_index in draw_indices:
            draw_index, amplitudes = _fit_bootstrap_draw_multi_mask(
                draw_index=draw_index,
                target_qu=target,
                target_noise_cov_qu=target_noise_cov_qu,
                target_fwhm_in=target_fwhm_in,
                template_inputs=template_inputs,
                weight_maps=weight_maps,
                fwhm_out=fwhm_out,
                master_mask=master_mask,
                target_beam_window=target_beam_window,
                template_inputs_rhs=template_inputs_rhs,
                template_inputs_data=template_inputs_data,
                target_filter=target_filter,
                master_support_mask=master_support_mask,
                master_support_threshold=master_support_threshold,
                nest=nest,
                rng=rng_obj,
            )
            samples[draw_index] = amplitudes
    else:
        if show_progress:
            _require_tqdm()
        # All masks for a given row share the same child seed and therefore the
        # same noisy realization. Different rows still use independent streams.
        draw_seeds: npt.NDArray[np.int64] = rng_obj.integers(
            0,
            np.iinfo(np.int64).max,
            size=n_mc,
            dtype=np.int64,
        )
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _fit_bootstrap_draw_multi_mask,
                    draw_index=draw_index,
                    target_qu=target,
                    target_noise_cov_qu=target_noise_cov_qu,
                    target_fwhm_in=target_fwhm_in,
                    template_inputs=template_inputs,
                    weight_maps=weight_maps,
                    fwhm_out=fwhm_out,
                    master_mask=master_mask,
                    target_beam_window=target_beam_window,
                    template_inputs_rhs=template_inputs_rhs,
                    template_inputs_data=template_inputs_data,
                    target_filter=target_filter,
                    master_support_mask=master_support_mask,
                    master_support_threshold=master_support_threshold,
                    nest=nest,
                    rng=int(draw_seed),
                )
                for draw_index, draw_seed in enumerate(draw_seeds)
            ]
            completed_draws = _wrap_progress(
                as_completed(futures),
                show_progress=show_progress,
                total=n_mc,
                desc="Bootstrap MC",
            )
            for future in completed_draws:
                draw_index, amplitudes = future.result()
                samples[draw_index] = amplitudes

    ddof = 1 if n_mc > 1 else 0
    return MultiMaskBootstrapResult(
        reference_fit=reference_fit,
        amplitude_samples=samples,
        amplitude_mean=np.mean(samples, axis=0),
        amplitude_std=np.std(samples, axis=0, ddof=ddof),
        fit_names=reference_fit.fit_names,
        template_names=reference_fit.template_names,
    )


def _fit_bootstrap_draw(
    *,
    draw_index: int,
    target_qu: FloatArray,
    target_noise_cov_qu: FloatArray,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_map: npt.ArrayLike,
    fwhm_out: float,
    target_beam_window: npt.ArrayLike | None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None,
    target_filter: HarmonicFilter | None,
    mask: npt.ArrayLike | None,
    nest: bool,
    rng: np.random.Generator | int,
) -> tuple[int, FloatArray]:
    """Run one noisy bootstrap refit.

    Parameters
    ----------
    draw_index
        Row index for this realization in the output sample matrix.
    target_qu
        Native-resolution target Q/U map with shape ``(2, npix)``.
    target_noise_cov_qu
        Native-resolution target covariance with shape ``(3, npix)`` in
        ``QQ, UU, QU`` order.
    target_fwhm_in
        Beam FWHM of ``target_qu`` in radians.
    template_inputs
        Left-hand template definitions used for the fit.
    weight_map
        Diagonal pixel weight map passed through to ``fit_foreground_templates``.
    fwhm_out
        Common output beam FWHM in radians.
    target_beam_window
        Optional input beam transfer function for the target map.
    template_inputs_rhs
        Optional right-hand template definitions for the cross normal matrix.
    template_inputs_data
        Optional data-projection template definitions for the right-hand vector.
    target_filter
        Optional harmonic filter applied to the target map and default
        template preprocessing.
    mask
        Optional preprocessing mask.
    nest
        If ``True``, maps are treated as NEST ordered during harmonic
        transforms.
    rng
        Per-draw random generator or seed. Threaded callers pass independent
        seeds; serial callers pass the shared serial generator.

    Returns
    -------
    tuple of int and ndarray
        ``(draw_index, amplitudes)`` where ``amplitudes`` has shape
        ``(n_template,)``.
    """

    rng_obj = coerce_rng(rng)

    # Realize noise on the native-resolution inputs first so every draw goes
    # through the same smoothing/filtering/template-construction pipeline.
    noisy_target = target_qu + realize_qu_noise(
        target_noise_cov_qu,
        rng=rng_obj,
    )
    noisy_templates = tuple(
        _realize_noisy_template_input(template_input, rng_obj)
        for template_input in template_inputs
    )
    if template_inputs_rhs is None:
        noisy_templates_rhs = None
    else:
        noisy_templates_rhs = tuple(
            _realize_noisy_template_input(template_input, rng_obj)
            for template_input in template_inputs_rhs
        )
    if template_inputs_data is None:
        noisy_templates_data = None
    else:
        noisy_templates_data = tuple(
            _realize_noisy_template_input(template_input, rng_obj)
            for template_input in template_inputs_data
        )

    draw_fit = fit_foreground_templates(
        target_qu=noisy_target,
        target_fwhm_in=target_fwhm_in,
        template_inputs=noisy_templates,
        weight_map=weight_map,
        fwhm_out=fwhm_out,
        target_beam_window=target_beam_window,
        template_inputs_rhs=noisy_templates_rhs,
        template_inputs_data=noisy_templates_data,
        target_filter=target_filter,
        mask=mask,
        nest=nest,
    )
    return draw_index, draw_fit.amplitudes


def _fit_bootstrap_draw_multi_mask(
    *,
    draw_index: int,
    target_qu: FloatArray,
    target_noise_cov_qu: FloatArray,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_maps: Mapping[str, npt.ArrayLike],
    fwhm_out: float,
    master_mask: npt.ArrayLike,
    target_beam_window: npt.ArrayLike | None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None,
    target_filter: HarmonicFilter | None,
    master_support_mask: npt.ArrayLike | None,
    master_support_threshold: float,
    nest: bool,
    rng: np.random.Generator | int,
) -> tuple[int, FloatArray]:
    """Run one noisy bootstrap refit for every named weight map.

    Returns
    -------
    tuple of int and ndarray
        ``(draw_index, amplitudes)`` where ``amplitudes`` has shape
        ``(n_fit_mask, n_template)``.
    """

    rng_obj = coerce_rng(rng)
    # Draw noise once, then pass the same noisy maps to the multi-mask fitter so
    # per-mask amplitudes from this draw remain paired.
    noisy_target = target_qu + realize_qu_noise(
        target_noise_cov_qu,
        rng=rng_obj,
    )
    noisy_templates = tuple(
        _realize_noisy_template_input(template_input, rng_obj)
        for template_input in template_inputs
    )
    if template_inputs_rhs is None:
        noisy_templates_rhs = None
    else:
        noisy_templates_rhs = tuple(
            _realize_noisy_template_input(template_input, rng_obj)
            for template_input in template_inputs_rhs
        )
    if template_inputs_data is None:
        noisy_templates_data = None
    else:
        noisy_templates_data = tuple(
            _realize_noisy_template_input(template_input, rng_obj)
            for template_input in template_inputs_data
        )

    draw_fit = fit_foreground_templates_multi_mask(
        target_qu=noisy_target,
        target_fwhm_in=target_fwhm_in,
        template_inputs=noisy_templates,
        weight_maps=weight_maps,
        fwhm_out=fwhm_out,
        master_mask=master_mask,
        target_beam_window=target_beam_window,
        template_inputs_rhs=noisy_templates_rhs,
        template_inputs_data=noisy_templates_data,
        target_filter=target_filter,
        master_support_mask=master_support_mask,
        master_support_threshold=master_support_threshold,
        nest=nest,
    )
    amplitudes = np.stack(
        [draw_fit.fit_results[fit_name].amplitudes for fit_name in draw_fit.fit_names],
        axis=0,
    )
    return draw_index, amplitudes


def realize_qu_noise(
    pixel_cov_qu: npt.ArrayLike,
    *,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Realize a Q/U noise map from per-pixel ``QQ, UU, QU`` covariance.

    Parameters
    ----------
    pixel_cov_qu
        Covariance array with shape ``(3, npix)`` or ``(npix, 3)`` in the
        order ``QQ, UU, QU``.
    rng
        Existing random generator, integer seed, or ``None``.

    Returns
    -------
    numpy.ndarray
        Noise realization with shape ``(2, npix)``.

    Raises
    ------
    ValueError
        If the covariance is not positive semi-definite on a per-pixel basis.
    """

    covariance = as_covariance(pixel_cov_qu, name="pixel_cov_qu")
    qq, uu, qu = covariance
    npix = qq.size

    cov_matrices = np.zeros((npix, 2, 2), dtype=np.float64)
    cov_matrices[:, 0, 0] = qq
    cov_matrices[:, 1, 1] = uu
    cov_matrices[:, 0, 1] = qu
    cov_matrices[:, 1, 0] = qu

    eigvals, eigvecs = np.linalg.eigh(cov_matrices)
    scale = max(np.max(np.abs(cov_matrices)), 1.0)
    tolerance = 1e-12 * scale
    if np.min(eigvals) < -tolerance:
        raise ValueError("pixel_cov_qu must be positive semi-definite per pixel.")

    # Each 2x2 pixel covariance is diagonalized, sampled with independent
    # standard normals, and rotated back into the native Q/U basis.
    eigvals = np.clip(eigvals, 0.0, None)
    rng_obj = coerce_rng(rng)
    standard_draw = rng_obj.standard_normal((npix, 2))
    scaled_draw = np.sqrt(eigvals) * standard_draw
    realized = np.einsum("pij,pj->pi", eigvecs, scaled_draw)
    return realized.T


def _realize_noisy_template_input(
    template_input: DifferenceTemplateInput,
    rng: np.random.Generator,
) -> DifferenceTemplateInput:
    """Add one noise realization to the two maps defining a template input.

    Parameters
    ----------
    template_input
        Template definition containing the input maps and optional per-pixel
        noise covariances.
    rng
        Random generator used for the noise draw.

    Returns
    -------
    DifferenceTemplateInput
        Copy of ``template_input`` with realized noise added to ``map_a_qu`` and
        ``map_b_qu`` when the corresponding covariance arrays are available.
    """

    map_a_qu = as_qu_map(template_input.map_a_qu, name="template_input.map_a_qu")
    map_b_qu = as_qu_map(template_input.map_b_qu, name="template_input.map_b_qu")

    if template_input.noise_cov_a is not None:
        map_a_qu = map_a_qu + realize_qu_noise(template_input.noise_cov_a, rng=rng)
    if template_input.noise_cov_b is not None:
        map_b_qu = map_b_qu + realize_qu_noise(template_input.noise_cov_b, rng=rng)

    return replace(
        template_input,
        map_a_qu=map_a_qu,
        map_b_qu=map_b_qu,
    )


def _wrap_progress(
    iterable: Iterable[_ProgressItem],
    *,
    show_progress: bool,
    total: int,
    desc: str,
) -> Iterable[_ProgressItem]:
    """Return an iterable, optionally wrapped in a tqdm progress bar."""

    if not show_progress:
        return iterable
    _require_tqdm()
    return _tqdm(iterable, total=total, desc=desc, unit="draw")


def _require_tqdm() -> None:
    """Raise the standard progress error when tqdm is unavailable."""

    if _tqdm is None:
        raise ImportError(
            "show_progress=True requires tqdm. Install tqdm or pass "
            "show_progress=False."
        )
