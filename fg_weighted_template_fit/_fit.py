from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._arrays import as_qu_map, as_template_stack, as_weight_map, weighted_inner_product
from ._filters import (
    _apply_harmonic_preprocessing,
    _build_template_stack_with_lmax,
    _resolve_fit_harmonic_lmax,
)
from ._types import (
    DifferenceTemplateInput,
    FloatArray,
    HarmonicFilter,
    MultiMaskFitResult,
    WeightedFitResult,
)


def weighted_template_gls(
    target_qu: npt.ArrayLike,
    templates_qu: npt.ArrayLike,
    weight_map: npt.ArrayLike,
    *,
    templates_rhs_qu: npt.ArrayLike | None = None,
    templates_data_qu: npt.ArrayLike | None = None,
    mask: npt.ArrayLike | None = None,
    template_names: Sequence[str] | None = None,
) -> WeightedFitResult:
    """Fit template amplitudes with a weighted normal equation.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    templates_qu
        Left-hand template stack with shape ``(n_template, 2, npix)`` or
        ``(n_template, npix, 2)``.
    weight_map
        Diagonal pixel weight map. Accepted shapes are scalar, ``(npix,)``,
        ``(2, npix)``, and ``(npix, 2)``.
    templates_rhs_qu
        Optional right-hand template stack. When supplied, the estimator
        becomes ``(T_left^T W T_right)^-1 T_left^T W m``. When omitted, the
        routine falls back to the same-template solve
        ``(T^T W T)^-1 T^T W m``.
    templates_data_qu
        Optional data-projection template stack ``d_3``. When supplied, the
        estimator becomes ``(d_1^T W d_2)^-1 d_3^T W m`` where ``d_1`` is
        ``templates_qu`` and ``d_2`` is ``templates_rhs_qu`` (or ``d_1`` when the
        latter is omitted). The data stack enters only the right-hand vector
        ``d_3^T W m`` and not the normal matrix. When omitted, ``d_3`` defaults to
        ``d_1``, reproducing the ``d_1^T W m`` projection.
    mask
        Optional pixel-domain exclusion mask with the same accepted shapes as
        ``weight_map``. The mask is converted to binary support before entering
        the weighted solve, so any finite nonzero value keeps a sample and zero
        removes it.
    template_names
        Optional names associated with the templates.

    Returns
    -------
    WeightedFitResult
        Fitted amplitudes, normal matrix, processed maps, and residual map.

    Raises
    ------
    ValueError
        If the target map and template stacks do not have compatible shapes.

    Notes
    -----
    Q and U pixels are stacked into a single data vector, but the computation is
    carried out in map form for clarity.
    """

    target = as_qu_map(target_qu, name="target_qu")
    templates_lhs = as_template_stack(templates_qu, name="templates_qu")
    if templates_rhs_qu is None:
        templates_rhs = templates_lhs.copy()
    else:
        templates_rhs = as_template_stack(
            templates_rhs_qu,
            name="templates_rhs_qu",
        )
    if templates_data_qu is None:
        templates_data = templates_lhs.copy()
    else:
        templates_data = as_template_stack(
            templates_data_qu,
            name="templates_data_qu",
        )

    if templates_lhs.shape[1:] != target.shape:
        raise ValueError(
            "Each left-hand template must have the same Q/U shape as the target map."
        )
    if templates_rhs.shape[1:] != target.shape:
        raise ValueError(
            "Each right-hand template must have the same Q/U shape as the target map."
        )
    if templates_data.shape[1:] != target.shape:
        raise ValueError(
            "Each data-projection template must have the same Q/U shape as the target map."
        )
    if templates_lhs.shape[0] != templates_rhs.shape[0]:
        raise ValueError(
            "Left and right template stacks must contain the same number of templates."
        )
    if templates_lhs.shape[0] != templates_data.shape[0]:
        raise ValueError(
            "Left and data-projection template stacks must contain the same number of templates."
        )

    weights = as_weight_map(weight_map, npix=target.shape[1], name="weight_map")
    if mask is not None:
        weights = weights * _as_binary_fit_mask(mask, npix=target.shape[1])

    # Non-finite samples are dropped by zeroing their weights and the matching
    # left/right/data template and target entries, so the normal equations only
    # see valid samples on every side of the cross-estimator.
    valid = np.isfinite(target)
    valid &= np.isfinite(weights)
    valid &= np.isfinite(templates_lhs).all(axis=0)
    valid &= np.isfinite(templates_rhs).all(axis=0)
    valid &= np.isfinite(templates_data).all(axis=0)

    weights = np.where(valid, weights, 0.0)
    target = np.where(valid, target, 0.0)
    templates_lhs = np.where(valid[None, :, :], templates_lhs, 0.0)
    templates_rhs = np.where(valid[None, :, :], templates_rhs, 0.0)
    templates_data = np.where(valid[None, :, :], templates_data, 0.0)

    n_template = templates_lhs.shape[0]
    normal_matrix = np.zeros((n_template, n_template), dtype=np.float64)
    rhs = np.zeros(n_template, dtype=np.float64)

    # Build the cross normal matrix one template pair at a time so the algebra
    # stays close to the estimator definition d_1^T W d_2 for the matrix and
    # d_3^T W m for the right-hand vector.
    for i in range(n_template):
        rhs[i] = weighted_inner_product(templates_data[i], target, weights)
        for j in range(n_template):
            value = weighted_inner_product(
                templates_lhs[i],
                templates_rhs[j],
                weights,
            )
            normal_matrix[i, j] = value

    try:
        amplitudes = np.linalg.solve(normal_matrix, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        amplitudes = np.linalg.pinv(normal_matrix) @ rhs
        solver = "pinv"

    normal_matrix_inverse = np.linalg.pinv(normal_matrix)
    model_qu = np.tensordot(amplitudes, templates_rhs, axes=(0, 0))
    residual_qu = target - model_qu

    if template_names is None:
        names = tuple(f"template_{index}" for index in range(n_template))
    else:
        if len(template_names) != n_template:
            raise ValueError(
                "template_names must match the number of templates in templates_qu."
            )
        names = tuple(template_names)

    return WeightedFitResult(
        amplitudes=amplitudes,
        normal_matrix=normal_matrix,
        normal_matrix_inverse=normal_matrix_inverse,
        rhs=rhs,
        residual_qu=residual_qu,
        processed_target_qu=target,
        processed_templates_qu=templates_lhs,
        processed_templates_rhs_qu=templates_rhs,
        processed_templates_data_qu=templates_data,
        template_names=names,
        solver=solver,
    )


def fit_foreground_templates(
    target_qu: npt.ArrayLike,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_map: npt.ArrayLike,
    fwhm_out: float,
    *,
    target_beam_window: npt.ArrayLike | None = None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None = None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None = None,
    target_filter: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
) -> WeightedFitResult:
    """Construct difference templates and fit their amplitudes to a target map.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    target_fwhm_in
        Beam FWHM of the target map in radians.
    template_inputs
        Sequence of left-hand template definitions.
    weight_map
        Diagonal pixel weight map used in the fit.
    fwhm_out
        Common output beam FWHM in radians applied to target and templates.
    target_beam_window
        Optional real, finite, strictly positive 1D transfer ``B_ell`` for a
        scalar-valued, axisymmetric target beam. It applies equally to E and B,
        must cover the shared ``lmax``, and replaces the Gaussian beam implied
        by ``target_fwhm_in``.
    template_inputs_rhs
        Optional sequence of right-hand template definitions. When supplied,
        the fit uses separate left- and right-hand template stacks in the
        normal matrix, ``(T_left^T W T_right)^-1 T_left^T W m``.
    template_inputs_data
        Optional sequence of data-projection template definitions ``d_3``. When
        supplied, the fit becomes ``(d_1^T W d_2)^-1 d_3^T W m`` where ``d_3``
        enters only the right-hand vector. When omitted, ``d_3`` defaults to the
        left-hand stack ``d_1``.
    target_filter
        Optional harmonic filter applied to the target map. Template entries may
        override this with their own ``filter_config`` values.
    mask
        Optional binary or floating mask applied once to the target and
        template input maps before any harmonic smoothing or filtering. This is
        intended for pre-apodizing map edges before the harmonic operations,
        not for modifying the final GLS weights.
    nest
        If ``True``, maps are treated as NEST ordered during harmonic
        transforms.

    Returns
    -------
    WeightedFitResult
        Weighted template-fit result for the processed target and template
        stacks.

    Raises
    ------
    ValueError
        If the template lists have incompatible lengths or any map, beam,
        filter, or harmonic support is invalid.
    ImportError
        If harmonic preprocessing is required but Healpy is unavailable.
    """

    (
        processed_target,
        processed_templates,
        processed_templates_rhs,
        processed_templates_data,
        template_names,
    ) = _preprocess_fit_inputs(
        target_qu=target_qu,
        target_fwhm_in=target_fwhm_in,
        target_beam_window=target_beam_window,
        template_inputs=template_inputs,
        template_inputs_rhs=template_inputs_rhs,
        template_inputs_data=template_inputs_data,
        fwhm_out=fwhm_out,
        target_filter=target_filter,
        mask=mask,
        nest=nest,
    )
    return weighted_template_gls(
        target_qu=processed_target,
        templates_qu=processed_templates,
        templates_rhs_qu=processed_templates_rhs,
        templates_data_qu=processed_templates_data,
        weight_map=weight_map,
        template_names=template_names,
    )


def fit_foreground_templates_multi_mask(
    target_qu: npt.ArrayLike,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_maps: Mapping[str, npt.ArrayLike],
    fwhm_out: float,
    *,
    master_mask: npt.ArrayLike,
    target_beam_window: npt.ArrayLike | None = None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None = None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None = None,
    target_filter: HarmonicFilter | None = None,
    master_support_mask: npt.ArrayLike | None = None,
    master_support_threshold: float = 0.0,
    nest: bool = False,
) -> MultiMaskFitResult:
    """Fit template amplitudes for multiple weights after one preprocessing pass.

    The target and templates are smoothed and filtered once with ``master_mask``
    as the harmonic preprocessing mask. The processed maps are then restricted
    to binary master support before each named weight map is used in an
    independent GLS solve. This keeps harmonic preprocessing homogeneous across
    all fitted regions while allowing the final weighted solve to use
    region-specific masks or inverse-variance weights.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    target_fwhm_in
        Beam FWHM of the target map in radians.
    template_inputs
        Sequence of left-hand template definitions.
    weight_maps
        Mapping from fit-region name to diagonal pixel weight map. Each weight
        accepts the same shapes as ``weighted_template_gls``: scalar,
        ``(npix,)``, ``(2, npix)``, or ``(npix, 2)``. Insertion order is
        preserved in the returned ``fit_names``.
    fwhm_out
        Common output beam FWHM in radians applied to target and templates.
    master_mask
        Binary or apodized mask applied to every input map before harmonic
        smoothing/filtering.
    target_beam_window
        Optional real, finite, strictly positive 1D transfer ``B_ell`` for a
        scalar-valued, axisymmetric target beam. It applies equally to E and B,
        must cover the shared ``lmax``, and replaces the Gaussian beam implied
        by ``target_fwhm_in``.
    template_inputs_rhs
        Optional sequence of right-hand template definitions for the cross
        normal matrix.
    template_inputs_data
        Optional sequence of data-projection template definitions ``d_3``. When
        supplied, each per-mask solve becomes ``(d_1^T W d_2)^-1 d_3^T W m``.
        When omitted, ``d_3`` defaults to the left-hand stack ``d_1``.
    target_filter
        Optional harmonic filter applied to the target map. Template entries may
        override this with their own ``filter_config`` values.
    master_support_mask
        Optional explicit post-filter support mask. Finite nonzero samples are
        kept. When omitted, support is derived from ``master_mask`` and
        ``master_support_threshold``.
    master_support_threshold
        Threshold used for support derived from ``master_mask``. Samples are
        kept where ``master_mask`` is finite and greater than this value.
    nest
        If ``True``, maps are treated as NEST ordered during harmonic
        transforms.

    Returns
    -------
    MultiMaskFitResult
        Shared processed maps and one weighted fit result per named weight.

    Raises
    ------
    ValueError
        If ``weight_maps`` is empty, a mask or weight has an incompatible
        shape, the support threshold is not finite, template lists have
        incompatible lengths, or harmonic preprocessing inputs are invalid.
    ImportError
        If harmonic preprocessing is required but Healpy is unavailable.
    """

    target = as_qu_map(target_qu, name="target_qu")
    npix = target.shape[1]
    # Validate all cheap pixel-domain inputs before invoking any Healpix
    # smoothing/filtering, which is the expensive part of the workflow.
    weights_by_name = _validate_weight_maps(weight_maps, npix=npix)
    master_mask_map = as_weight_map(master_mask, npix=npix, name="master_mask")
    master_support = _build_master_support(
        master_mask_map=master_mask_map,
        master_support_mask=master_support_mask,
        master_support_threshold=master_support_threshold,
        npix=npix,
    )

    (
        processed_target,
        processed_templates,
        processed_templates_rhs,
        processed_templates_data,
        template_names,
    ) = _preprocess_under_master_mask(
        target_qu=target,
        target_fwhm_in=target_fwhm_in,
        template_inputs=template_inputs,
        fwhm_out=fwhm_out,
        master_mask_map=master_mask_map,
        master_support=master_support,
        target_beam_window=target_beam_window,
        template_inputs_rhs=template_inputs_rhs,
        template_inputs_data=template_inputs_data,
        target_filter=target_filter,
        nest=nest,
    )

    fit_results: dict[str, WeightedFitResult] = {
        fit_name: weighted_template_gls(
            target_qu=processed_target,
            templates_qu=processed_templates,
            templates_rhs_qu=processed_templates_rhs,
            templates_data_qu=processed_templates_data,
            weight_map=weight_map,
            template_names=template_names,
        )
        for fit_name, weight_map in weights_by_name.items()
    }

    return MultiMaskFitResult(
        fit_names=tuple(weights_by_name),
        fit_results=fit_results,
        template_names=template_names,
        processed_target_qu=processed_target,
        processed_templates_qu=processed_templates,
        processed_templates_rhs_qu=processed_templates_rhs,
        processed_templates_data_qu=processed_templates_data,
    )


def _as_binary_fit_mask(mask: npt.ArrayLike, *, npix: int) -> npt.NDArray[np.float64]:
    """Convert a general mask definition into binary support for the GLS step."""

    mask_map = as_weight_map(mask, npix=npix, name="mask")
    return np.where(np.isfinite(mask_map) & (mask_map != 0.0), 1.0, 0.0)


def _validate_weight_maps(
    weight_maps: Mapping[str, npt.ArrayLike],
    *,
    npix: int,
) -> dict[str, FloatArray]:
    """Normalize named fit weights before any harmonic preprocessing.

    Parameters
    ----------
    weight_maps
        Mapping from fit-region name to scalar, pixel, or Q/U weight map.
    npix
        Expected number of Healpix pixels.

    Returns
    -------
    dict of str to ndarray
        Weight maps normalized to shape ``(2, npix)`` in input order.

    Raises
    ------
    ValueError
        If the mapping is empty, a key is not a non-empty string, or any weight
        map has an unsupported shape.
    """

    if not weight_maps:
        raise ValueError("weight_maps must contain at least one named weight map.")

    validated: dict[str, FloatArray] = {}
    for fit_name, weight_map in weight_maps.items():
        if not isinstance(fit_name, str) or fit_name == "":
            raise ValueError("weight_maps keys must be non-empty strings.")
        validated[fit_name] = as_weight_map(
            weight_map,
            npix=npix,
            name=f"weight_maps[{fit_name!r}]",
        )
    return validated


def _build_master_support(
    *,
    master_mask_map: FloatArray,
    master_support_mask: npt.ArrayLike | None,
    master_support_threshold: float,
    npix: int,
) -> FloatArray:
    """Build binary post-filter support from an explicit or thresholded mask.

    The returned support has shape ``(2, npix)`` so it can broadcast directly
    over Q/U maps and stacked template arrays.
    """

    threshold = float(master_support_threshold)
    if not np.isfinite(threshold):
        raise ValueError("master_support_threshold must be finite.")

    if master_support_mask is None:
        # The master mask controls the harmonic edge treatment. Post-filtering
        # only enforces support so apodized mask values are not applied twice.
        support_source = master_mask_map
        keep = np.isfinite(support_source) & (support_source > threshold)
    else:
        support_source = as_weight_map(
            master_support_mask,
            npix=npix,
            name="master_support_mask",
        )
        keep = np.isfinite(support_source) & (support_source != 0.0)
    return keep.astype(np.float64)


def _preprocess_fit_inputs(
    *,
    target_qu: npt.ArrayLike,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    fwhm_out: float,
    target_beam_window: npt.ArrayLike | None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None,
    target_filter: HarmonicFilter | None,
    mask: npt.ArrayLike | None,
    nest: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, tuple[str, ...]]:
    """Preprocess every fit operand under one validated harmonic support.

    Omitted RHS or data groups reuse the processed LHS stack. The return value
    contains the target, LHS, RHS, and data arrays followed by template names.
    """

    lhs_inputs = tuple(template_inputs)
    if not lhs_inputs:
        raise ValueError("template_inputs must contain at least one template.")

    rhs_inputs = None if template_inputs_rhs is None else tuple(template_inputs_rhs)
    data_inputs = None if template_inputs_data is None else tuple(template_inputs_data)
    for name, inputs in (
        ("template_inputs_rhs", rhs_inputs),
        ("template_inputs_data", data_inputs),
    ):
        if inputs is not None and len(inputs) != len(lhs_inputs):
            raise ValueError(
                f"{name} must contain the same number of templates as "
                "template_inputs."
            )

    template_input_groups = tuple(
        inputs for inputs in (lhs_inputs, rhs_inputs, data_inputs) if inputs is not None
    )
    harmonic_lmax = _resolve_fit_harmonic_lmax(
        target_qu=target_qu,
        target_fwhm_in=target_fwhm_in,
        target_beam_window=target_beam_window,
        template_input_groups=template_input_groups,
        fwhm_out=fwhm_out,
        target_filter=target_filter,
    )

    effective_target_filter = target_filter or HarmonicFilter()
    processed_target = _apply_harmonic_preprocessing(
        target_qu,
        fwhm_in=target_fwhm_in,
        fwhm_out=fwhm_out,
        beam_window_in=target_beam_window,
        filter_config=effective_target_filter,
        mask=mask,
        nest=nest,
        harmonic_lmax=harmonic_lmax,
    )
    processed_lhs, template_names = _build_template_stack_with_lmax(
        lhs_inputs,
        fwhm_out=fwhm_out,
        default_filter=target_filter,
        mask=mask,
        nest=nest,
        harmonic_lmax=harmonic_lmax,
    )

    if rhs_inputs is None:
        processed_rhs = processed_lhs
    else:
        processed_rhs, _ = _build_template_stack_with_lmax(
            rhs_inputs,
            fwhm_out=fwhm_out,
            default_filter=target_filter,
            mask=mask,
            nest=nest,
            harmonic_lmax=harmonic_lmax,
        )

    if data_inputs is None:
        processed_data = processed_lhs
    else:
        processed_data, _ = _build_template_stack_with_lmax(
            data_inputs,
            fwhm_out=fwhm_out,
            default_filter=target_filter,
            mask=mask,
            nest=nest,
            harmonic_lmax=harmonic_lmax,
        )

    return (
        processed_target,
        processed_lhs,
        processed_rhs,
        processed_data,
        template_names,
    )


def _preprocess_under_master_mask(
    *,
    target_qu: FloatArray,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    fwhm_out: float,
    master_mask_map: FloatArray,
    master_support: FloatArray,
    target_beam_window: npt.ArrayLike | None,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None,
    template_inputs_data: Sequence[DifferenceTemplateInput] | None,
    target_filter: HarmonicFilter | None,
    nest: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, tuple[str, ...]]:
    """Smooth/filter all fit inputs once and apply binary master support.

    Parameters
    ----------
    target_qu
        Target Q/U map already normalized to shape ``(2, npix)``.
    target_fwhm_in
        Beam FWHM of ``target_qu`` in radians.
    template_inputs
        Left-hand template definitions.
    fwhm_out
        Common output beam FWHM in radians.
    master_mask_map
        Harmonic preprocessing mask normalized to shape ``(2, npix)``.
    master_support
        Binary support map with shape ``(2, npix)`` applied after filtering.
    target_beam_window
        Optional input beam transfer function for ``target_qu``. When omitted,
        the target input beam is treated as Gaussian with ``target_fwhm_in``.
    template_inputs_rhs
        Optional right-hand template definitions.
    template_inputs_data
        Optional data-projection template definitions ``d_3``. When omitted, the
        returned data stack falls back to the left-hand stack.
    target_filter
        Optional default harmonic filter.
    nest
        Whether input maps are in NEST ordering.

    Returns
    -------
    tuple
        ``(target, templates_lhs, templates_rhs, templates_data, template_names)``
        where the target has shape ``(2, npix)`` and template stacks have shape
        ``(n_template, 2, npix)``.
    """

    (
        processed_target,
        processed_templates,
        processed_templates_rhs,
        processed_templates_data,
        template_names,
    ) = _preprocess_fit_inputs(
        target_qu=target_qu,
        target_fwhm_in=target_fwhm_in,
        target_beam_window=target_beam_window,
        template_inputs=template_inputs,
        template_inputs_rhs=template_inputs_rhs,
        template_inputs_data=template_inputs_data,
        fwhm_out=fwhm_out,
        target_filter=target_filter,
        mask=master_mask_map,
        nest=nest,
    )

    # Apply support after the harmonic operation to zero pixels outside the
    # shared analysis region without multiplying by the apodization profile a
    # second time. Multiplication is non-mutating, so the data/rhs stacks that
    # alias the left-hand stack in their default branches are each masked once.
    processed_target = processed_target * master_support
    processed_templates_rhs = processed_templates_rhs * master_support[None, :, :]
    processed_templates_data = processed_templates_data * master_support[None, :, :]
    processed_templates = processed_templates * master_support[None, :, :]
    return (
        processed_target,
        processed_templates,
        processed_templates_rhs,
        processed_templates_data,
        template_names,
    )
