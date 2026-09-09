from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._arrays import as_qu_map, as_weight_map
from ._types import DifferenceTemplateInput, FloatArray, HarmonicFilter

try:
    import healpy as hp
except ImportError:  # pragma: no cover - exercised only when healpy is unavailable.
    hp = None


# Numerically identity beam ratios can stay on the transform-free NumPy path.
_UNITY_TRANSFER_RTOL = 1.0e-12


@dataclass(frozen=True)
class _HarmonicInput:
    """One map and its effective harmonic preprocessing configuration."""

    qu_map: npt.ArrayLike
    fwhm_in: float
    beam_window_in: npt.ArrayLike | None
    filter_config: HarmonicFilter


def build_ell_filter(
    lmax: int,
    *,
    cutoff: float,
    halfwidth: float = 0.0,
    transition_type: str = "C2",
) -> FloatArray:
    """Build a reusable high-pass ``ell``-space filter window.

    Parameters
    ----------
    lmax
        Maximum multipole index included in the returned transfer function.
    cutoff
        Center of the high-pass transition band in multipole ``ell``.
    halfwidth
        Half-width of the transition band. A value of zero gives a hard
        cutoff.
    transition_type
        Smooth edge type. Supported values are ``"C1"`` and ``"C2"``.

    Returns
    -------
    numpy.ndarray
        Multiplicative ``ell``-space transfer function with shape
        ``(lmax + 1,)``.

    Notes
    -----
    The returned array is suitable for ``HarmonicFilter(ell_filter=...)``.
    """

    return _build_apodized_highpass(
        num_modes=_num_modes_from_lmax(lmax),
        cutoff=cutoff,
        halfwidth=halfwidth,
        transition_type=transition_type,
    )


def build_m_filter(
    lmax: int,
    *,
    cutoff: float,
    halfwidth: float = 0.0,
    transition_type: str = "C2",
) -> FloatArray:
    """Build a reusable high-pass ``m``-space filter window.

    Parameters
    ----------
    lmax
        Maximum azimuthal mode index included in the returned transfer
        function.
    cutoff
        Center of the high-pass transition band in azimuthal mode ``m``.
    halfwidth
        Half-width of the transition band. A value of zero gives a hard
        cutoff.
    transition_type
        Smooth edge type. Supported values are ``"C1"`` and ``"C2"``.

    Returns
    -------
    numpy.ndarray
        Multiplicative ``m``-space transfer function with shape
        ``(lmax + 1,)``.

    Notes
    -----
    The returned array is suitable for ``HarmonicFilter(m_filter=...)``.
    """

    return _build_apodized_highpass(
        num_modes=_num_modes_from_lmax(lmax),
        cutoff=cutoff,
        halfwidth=halfwidth,
        transition_type=transition_type,
    )


def smooth_and_filter_qu_map(
    qu_map: npt.ArrayLike,
    fwhm_in: float,
    fwhm_out: float,
    *,
    beam_window_in: npt.ArrayLike | None = None,
    filter_config: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
) -> FloatArray:
    """Smooth and optionally filter a Healpix Q/U map.

    Parameters
    ----------
    qu_map
        Input Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    fwhm_in
        Beam FWHM of the input map in radians.
    fwhm_out
        Target Gaussian beam FWHM in radians. When ``beam_window_in`` is
        omitted, the routine applies additional Gaussian smoothing, so
        ``fwhm_out`` must be at least ``fwhm_in``.
    beam_window_in
        Optional scalar-valued, axisymmetric input beam transfer ``B_ell``. It
        must be a real, finite, strictly positive 1D array covering the selected
        ``lmax`` and is applied equally to E and B modes. When supplied, it
        replaces the Gaussian input beam implied by ``fwhm_in``; ``fwhm_in`` is
        then ignored.
    filter_config
        Optional harmonic filter configuration. Both the beam matching and the
        harmonic filters are applied in a single alm-domain pass.
    mask
        Optional binary or floating mask applied in pixel space before any
        harmonic transform. This is useful for apodizing map edges before beam
        smoothing or ``ell``/``m`` filtering.
    nest
        If ``True``, the map is assumed to be in NEST ordering and is converted
        to RING before harmonic transforms, then converted back on output.

    Returns
    -------
    numpy.ndarray
        Smoothed and filtered Q/U map with shape ``(2, npix)``.

    Raises
    ------
    ValueError
        If map shapes, beam widths, harmonic support, or transfer windows are
        invalid.
    ImportError
        If harmonic preprocessing is required but Healpy is unavailable.

    Notes
    -----
    All beam widths are expressed in radians.
    """

    effective_filter = filter_config or HarmonicFilter()
    harmonic_lmax = _resolve_harmonic_lmax(
        (
            _HarmonicInput(
                qu_map=qu_map,
                fwhm_in=fwhm_in,
                beam_window_in=beam_window_in,
                filter_config=effective_filter,
            ),
        ),
        fwhm_out=fwhm_out,
    )
    return _apply_harmonic_preprocessing(
        qu_map,
        fwhm_in=fwhm_in,
        fwhm_out=fwhm_out,
        beam_window_in=beam_window_in,
        filter_config=effective_filter,
        mask=mask,
        nest=nest,
        harmonic_lmax=harmonic_lmax,
    )


def construct_difference_template(
    map_a_qu: npt.ArrayLike,
    map_b_qu: npt.ArrayLike,
    fwhm_in_a: float,
    fwhm_in_b: float,
    fwhm_out: float,
    *,
    beam_window_a: npt.ArrayLike | None = None,
    beam_window_b: npt.ArrayLike | None = None,
    filter_config: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
) -> FloatArray:
    """Construct a foreground template from the difference of two Q/U maps.

    Parameters
    ----------
    map_a_qu
        First Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    map_b_qu
        Second Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    fwhm_in_a
        Beam FWHM of ``map_a_qu`` in radians.
    fwhm_in_b
        Beam FWHM of ``map_b_qu`` in radians.
    fwhm_out
        Common output beam FWHM in radians.
    beam_window_a
        Optional scalar-valued, axisymmetric input beam transfer ``B_ell`` for
        ``map_a_qu``. It must be real, finite, strictly positive, and cover the
        shared ``lmax``.
    beam_window_b
        Optional input beam transfer ``B_ell`` for ``map_b_qu`` with the same
        requirements as ``beam_window_a``.
    filter_config
        Optional harmonic filter applied after beam matching.
    mask
        Optional binary or floating mask applied to both input maps before any
        harmonic transform.
    nest
        If ``True``, input maps are interpreted as NEST ordered during the
        harmonic transform stage.

    Returns
    -------
    numpy.ndarray
        Difference template ``processed(map_a_qu) - processed(map_b_qu)`` with
        shape ``(2, npix)``.

    Raises
    ------
    ValueError
        If the maps cannot share one harmonic support or a beam/filter transfer
        is invalid.
    ImportError
        If harmonic preprocessing is required but Healpy is unavailable.
    """

    effective_filter = filter_config or HarmonicFilter()
    harmonic_lmax = _resolve_harmonic_lmax(
        (
            _HarmonicInput(
                qu_map=map_a_qu,
                fwhm_in=fwhm_in_a,
                beam_window_in=beam_window_a,
                filter_config=effective_filter,
            ),
            _HarmonicInput(
                qu_map=map_b_qu,
                fwhm_in=fwhm_in_b,
                beam_window_in=beam_window_b,
                filter_config=effective_filter,
            ),
        ),
        fwhm_out=fwhm_out,
    )
    return _construct_difference_template_with_lmax(
        map_a_qu=map_a_qu,
        map_b_qu=map_b_qu,
        fwhm_in_a=fwhm_in_a,
        fwhm_in_b=fwhm_in_b,
        fwhm_out=fwhm_out,
        beam_window_a=beam_window_a,
        beam_window_b=beam_window_b,
        filter_config=effective_filter,
        mask=mask,
        nest=nest,
        harmonic_lmax=harmonic_lmax,
    )


def build_template_stack(
    template_inputs: Sequence[DifferenceTemplateInput],
    *,
    fwhm_out: float,
    default_filter: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Construct a stack of difference templates ready for fitting.

    Parameters
    ----------
    template_inputs
        Sequence of template definitions. Each entry specifies the two maps
        used to build one difference template.
    fwhm_out
        Common output beam FWHM in radians for all templates.
    default_filter
        Harmonic filter used for template entries that do not define their own
        ``filter_config``.
    mask
        Optional binary or floating mask applied to every input map before any
        harmonic transform.
    nest
        If ``True``, input maps are interpreted as NEST ordered during the
        harmonic transform stage.

    Returns
    -------
    tuple
        Tuple ``(templates, template_names)`` where ``templates`` has shape
        ``(n_template, 2, npix)``.

    Raises
    ------
    ValueError
        If ``template_inputs`` is empty, its maps cannot share one harmonic
        support, or a beam/filter transfer is invalid.
    ImportError
        If harmonic preprocessing is required but Healpy is unavailable.
    """

    template_inputs = tuple(template_inputs)
    if not template_inputs:
        raise ValueError("template_inputs must contain at least one template.")

    harmonic_lmax = _resolve_harmonic_lmax(
        _template_harmonic_inputs(
            template_inputs,
            default_filter=default_filter,
        ),
        fwhm_out=fwhm_out,
    )
    return _build_template_stack_with_lmax(
        template_inputs,
        fwhm_out=fwhm_out,
        default_filter=default_filter,
        mask=mask,
        nest=nest,
        harmonic_lmax=harmonic_lmax,
    )


def _apply_harmonic_preprocessing(
    qu_map: npt.ArrayLike,
    fwhm_in: float,
    fwhm_out: float,
    *,
    beam_window_in: npt.ArrayLike | None,
    filter_config: HarmonicFilter,
    mask: npt.ArrayLike | None,
    nest: bool,
    harmonic_lmax: int | None,
) -> FloatArray:
    """Apply a previously validated harmonic support to one Q/U map.

    ``harmonic_lmax=None`` denotes an identity operation and returns a copy.
    """

    qu = as_qu_map(qu_map, name="qu_map")
    if harmonic_lmax is None:
        return qu.copy()

    _require_healpy()
    map_for_transform = qu
    if mask is not None:
        # Zero masked pixels before the harmonic transform so an apodized mask
        # can suppress ringing from the finite lmax truncation.
        map_for_transform = _apply_harmonic_preprocessing_mask(
            qu=map_for_transform,
            mask=mask,
        )
    if nest:
        map_for_transform = np.asarray(
            [hp.reorder(component, n2r=True) for component in map_for_transform],
            dtype=np.float64,
        )

    npix = map_for_transform.shape[1]
    nside = hp.npix2nside(npix)
    ell_transfer = _build_ell_transfer(
        lmax=harmonic_lmax,
        fwhm_in=fwhm_in,
        fwhm_out=fwhm_out,
        filter_config=filter_config,
        beam_window_in=beam_window_in,
    )

    # Healpy's polarized transform works on T/Q/U. Prepending a zero
    # temperature map lets one transform produce both polarization alms.
    tqu = np.vstack([np.zeros(npix, dtype=np.float64), map_for_transform])
    alm_t, alm_e, alm_b = hp.map2alm(
        tqu,
        lmax=harmonic_lmax,
        iter=filter_config.iter,
        pol=True,
    )
    alm_t[...] = 0.0
    alm_e = hp.almxfl(alm_e, ell_transfer, inplace=False)
    alm_b = hp.almxfl(alm_b, ell_transfer, inplace=False)

    if filter_config.m_filter is not None or filter_config.m_cutoff is not None:
        m_transfer = _build_m_transfer(
            filter_config=filter_config,
            lmax=harmonic_lmax,
        )
        _, m_indices = hp.Alm.getlm(harmonic_lmax)
        packed_m_transfer = m_transfer[m_indices]
        alm_e *= packed_m_transfer
        alm_b *= packed_m_transfer

    filtered_tqu = hp.alm2map(
        [alm_t, alm_e, alm_b],
        nside=nside,
        lmax=harmonic_lmax,
        pol=True,
    )
    filtered_qu = np.asarray(filtered_tqu[1:], dtype=np.float64)

    if nest:
        filtered_qu = np.asarray(
            [hp.reorder(component, r2n=True) for component in filtered_qu],
            dtype=np.float64,
        )
    return filtered_qu


def _construct_difference_template_with_lmax(
    map_a_qu: npt.ArrayLike,
    map_b_qu: npt.ArrayLike,
    fwhm_in_a: float,
    fwhm_in_b: float,
    fwhm_out: float,
    *,
    beam_window_a: npt.ArrayLike | None,
    beam_window_b: npt.ArrayLike | None,
    filter_config: HarmonicFilter,
    mask: npt.ArrayLike | None,
    nest: bool,
    harmonic_lmax: int | None,
) -> FloatArray:
    """Build one difference template using a validated shared support.

    ``harmonic_lmax=None`` keeps both operands on the identity copy path.
    """

    map_a_processed = _apply_harmonic_preprocessing(
        map_a_qu,
        fwhm_in=fwhm_in_a,
        fwhm_out=fwhm_out,
        beam_window_in=beam_window_a,
        filter_config=filter_config,
        mask=mask,
        nest=nest,
        harmonic_lmax=harmonic_lmax,
    )
    map_b_processed = _apply_harmonic_preprocessing(
        map_b_qu,
        fwhm_in=fwhm_in_b,
        fwhm_out=fwhm_out,
        beam_window_in=beam_window_b,
        filter_config=filter_config,
        mask=mask,
        nest=nest,
        harmonic_lmax=harmonic_lmax,
    )
    return map_a_processed - map_b_processed


def _build_template_stack_with_lmax(
    template_inputs: Sequence[DifferenceTemplateInput],
    *,
    fwhm_out: float,
    default_filter: HarmonicFilter | None,
    mask: npt.ArrayLike | None,
    nest: bool,
    harmonic_lmax: int | None,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Build a template stack using a validated shared support.

    ``harmonic_lmax=None`` keeps every operand on the identity copy path.
    """

    if not template_inputs:
        raise ValueError("template_inputs must contain at least one template.")

    template_maps: list[FloatArray] = []
    template_names: list[str] = []
    for index, template_input in enumerate(template_inputs):
        effective_filter = (
            template_input.filter_config or default_filter or HarmonicFilter()
        )
        template_maps.append(
            _construct_difference_template_with_lmax(
                map_a_qu=template_input.map_a_qu,
                map_b_qu=template_input.map_b_qu,
                fwhm_in_a=template_input.fwhm_in_a,
                fwhm_in_b=template_input.fwhm_in_b,
                fwhm_out=fwhm_out,
                beam_window_a=template_input.beam_window_a,
                beam_window_b=template_input.beam_window_b,
                filter_config=effective_filter,
                mask=mask,
                nest=nest,
                harmonic_lmax=harmonic_lmax,
            )
        )
        template_names.append(template_input.name or f"template_{index}")

    return np.stack(template_maps, axis=0), tuple(template_names)


def _apply_harmonic_preprocessing_mask(
    *,
    qu: FloatArray,
    mask: npt.ArrayLike,
) -> FloatArray:
    """Apply a pixel-space mask before any harmonic-domain preprocessing."""

    mask_map = as_weight_map(mask, npix=qu.shape[1], name="mask")
    safe_mask = np.where(np.isfinite(mask_map), mask_map, 0.0)
    safe_qu = np.where(np.isfinite(qu), qu, 0.0)
    return safe_qu * safe_mask


def _resolve_harmonic_lmax(
    harmonic_inputs: Sequence[_HarmonicInput],
    *,
    fwhm_out: float,
) -> int | None:
    """Validate a map collection and return its shared transform support.

    Every map must have the same pixel count and use one explicit ``lmax`` or
    the map-native support. All transfer windows are validated before any SHT.
    ``None`` denotes a validated identity operation that can stay in NumPy.
    """

    if not harmonic_inputs:
        raise ValueError("harmonic_inputs must contain at least one map.")

    output_fwhm = _validate_fwhm(fwhm_out, name="fwhm_out")
    npix_values = {
        as_qu_map(item.qu_map, name="qu_map").shape[1] for item in harmonic_inputs
    }
    if len(npix_values) != 1:
        raise ValueError("All harmonically processed maps must have the same npix.")

    needs_transfer_evaluation = False
    for item in harmonic_inputs:
        if item.beam_window_in is None:
            input_fwhm = _validate_fwhm(item.fwhm_in, name="fwhm_in")
            if output_fwhm < input_fwhm and not np.isclose(
                output_fwhm,
                input_fwhm,
            ):
                raise ValueError("fwhm_out must be greater than or equal to fwhm_in.")
            needs_transfer_evaluation = needs_transfer_evaluation or not np.isclose(
                input_fwhm,
                output_fwhm,
            )
        else:
            # A custom beam fully replaces fwhm_in, including its validation.
            needs_transfer_evaluation = True
        needs_transfer_evaluation = (
            needs_transfer_evaluation or _filter_requests_harmonic(item.filter_config)
        )

    if not needs_transfer_evaluation:
        return None

    npix = next(iter(npix_values))
    nside = _nside_from_npix(npix)

    explicit_lmax_values = {
        int(item.filter_config.lmax)
        for item in harmonic_inputs
        if item.filter_config.lmax is not None
    }
    if len(explicit_lmax_values) > 1:
        raise ValueError(
            "All harmonic filter configurations must use the same explicit lmax."
        )
    explicit_lmax = next(iter(explicit_lmax_values)) if explicit_lmax_values else None
    common_lmax = _resolve_lmax(nside=nside, explicit_lmax=explicit_lmax)

    requires_transform = False
    for item in harmonic_inputs:
        ell_transfer = _build_ell_transfer(
            lmax=common_lmax,
            fwhm_in=item.fwhm_in,
            fwhm_out=output_fwhm,
            filter_config=item.filter_config,
            beam_window_in=item.beam_window_in,
        )
        if (
            item.filter_config.m_filter is not None
            or item.filter_config.m_cutoff is not None
        ):
            _build_m_transfer(
                filter_config=item.filter_config,
                lmax=common_lmax,
            )
        requires_transform = (
            requires_transform
            or _filter_requests_harmonic(item.filter_config)
            or not np.allclose(
                ell_transfer,
                1.0,
                rtol=_UNITY_TRANSFER_RTOL,
                atol=0.0,
            )
        )

    return common_lmax if requires_transform else None


def _filter_requests_harmonic(filter_config: HarmonicFilter) -> bool:
    """Return whether a filter configuration explicitly requests harmonic work."""

    return any(
        value is not None
        for value in (
            filter_config.ell_filter,
            filter_config.m_filter,
            filter_config.ell_cutoff,
            filter_config.m_cutoff,
            filter_config.lmax,
        )
    )


def _template_harmonic_inputs(
    template_inputs: Sequence[DifferenceTemplateInput],
    *,
    default_filter: HarmonicFilter | None,
) -> tuple[_HarmonicInput, ...]:
    """Expand template definitions into their per-map harmonic inputs."""

    harmonic_inputs: list[_HarmonicInput] = []
    for template_input in template_inputs:
        effective_filter = (
            template_input.filter_config or default_filter or HarmonicFilter()
        )
        harmonic_inputs.extend(
            (
                _HarmonicInput(
                    qu_map=template_input.map_a_qu,
                    fwhm_in=template_input.fwhm_in_a,
                    beam_window_in=template_input.beam_window_a,
                    filter_config=effective_filter,
                ),
                _HarmonicInput(
                    qu_map=template_input.map_b_qu,
                    fwhm_in=template_input.fwhm_in_b,
                    beam_window_in=template_input.beam_window_b,
                    filter_config=effective_filter,
                ),
            )
        )
    return tuple(harmonic_inputs)


def _resolve_fit_harmonic_lmax(
    *,
    target_qu: npt.ArrayLike,
    target_fwhm_in: float,
    target_beam_window: npt.ArrayLike | None,
    template_input_groups: Sequence[Sequence[DifferenceTemplateInput]],
    fwhm_out: float,
    target_filter: HarmonicFilter | None,
) -> int | None:
    """Resolve one shared harmonic support for every operand in a fit."""

    effective_target_filter = target_filter or HarmonicFilter()
    harmonic_inputs: list[_HarmonicInput] = [
        _HarmonicInput(
            qu_map=target_qu,
            fwhm_in=target_fwhm_in,
            beam_window_in=target_beam_window,
            filter_config=effective_target_filter,
        )
    ]
    for group in template_input_groups:
        harmonic_inputs.extend(
            _template_harmonic_inputs(
                group,
                default_filter=target_filter,
            )
        )

    return _resolve_harmonic_lmax(
        harmonic_inputs,
        fwhm_out=fwhm_out,
    )


def _validate_fwhm(fwhm: float, *, name: str) -> float:
    """Validate and return a finite, nonnegative beam FWHM."""

    try:
        value = float(fwhm)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite, nonnegative number.") from error
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _nside_from_npix(npix: int) -> int:
    """Infer Healpix nside without importing Healpy on identity-only paths."""

    nside = isqrt(npix // 12)
    if nside == 0 or 12 * nside**2 != npix:
        raise ValueError(f"npix={npix} is not a valid Healpix pixel count.")
    return nside


def _require_healpy() -> None:
    """Raise an informative error when Healpy-dependent paths are requested.

    Raises
    ------
    ImportError
        If ``healpy`` is not available in the current Python environment.
    """

    if hp is None:
        raise ImportError(
            "healpy is required for beam smoothing or harmonic filtering. "
            "Install healpy, or run with matching fwhm and no l/m filter."
        )


def _resolve_lmax(
    nside: int,
    explicit_lmax: int | None,
) -> int:
    """Resolve a map-native or explicit harmonic truncation.

    Parameters
    ----------
    nside
        Healpix ``nside`` of the working map.
    explicit_lmax
        Caller-selected maximum multipole, or ``None`` for map-native support.

    Returns
    -------
    int
        Maximum multipole used in the harmonic transform.

    Raises
    ------
    ValueError
        If ``explicit_lmax`` exceeds the map-native support or resolves below 2.
    """

    native_lmax = 3 * nside - 1
    lmax = native_lmax if explicit_lmax is None else int(explicit_lmax)
    if lmax < 2:
        raise ValueError("Resolved lmax must be at least 2.")
    if lmax > native_lmax:
        raise ValueError("Explicit lmax exceeds the map-native harmonic support.")
    return lmax


def _build_ell_transfer(
    *,
    lmax: int,
    fwhm_in: float,
    fwhm_out: float,
    filter_config: HarmonicFilter,
    beam_window_in: npt.ArrayLike | None = None,
) -> FloatArray:
    """Assemble the full multipole transfer function for beam/filter matching.

    Parameters
    ----------
    lmax
        Maximum multipole included in the transfer function.
    fwhm_in
        Input beam FWHM in radians.
    fwhm_out
        Output beam FWHM in radians.
    filter_config
        Harmonic filter configuration.
    beam_window_in
        Optional custom input beam transfer function. When supplied, the beam
        transfer is the Gaussian output beam divided by this window.

    Returns
    -------
    numpy.ndarray
        Multiplicative transfer function indexed by ``ell`` with length
        ``lmax + 1``.
    """

    output_fwhm = _validate_fwhm(fwhm_out, name="fwhm_out")
    ells = np.arange(lmax + 1, dtype=np.float64)
    ell_factor = ells * (ells + 1.0)
    sigma_out = _fwhm_to_sigma(output_fwhm)
    if beam_window_in is None:
        input_fwhm = _validate_fwhm(fwhm_in, name="fwhm_in")
        if output_fwhm < input_fwhm and not np.isclose(
            output_fwhm,
            input_fwhm,
        ):
            raise ValueError("fwhm_out must be greater than or equal to fwhm_in.")
        sigma_in = _fwhm_to_sigma(input_fwhm)
        # ``isclose`` permits tiny roundoff-level inversions in the requested
        # widths, so clip the additional smoothing variance at zero.
        sigma_extra_sq = np.maximum(sigma_out**2 - sigma_in**2, 0.0)
        transfer_numerator = np.exp(-0.5 * ell_factor * sigma_extra_sq)
    else:
        transfer_numerator = np.exp(-0.5 * ell_factor * sigma_out**2)

    if filter_config.ell_filter is not None:
        ell_filter_window = _as_transfer_window(
            filter_config.ell_filter,
            lmax=lmax,
            name="ell_filter",
        )
        transfer_numerator *= ell_filter_window

    if filter_config.ell_cutoff is not None:
        transfer_numerator *= _build_apodized_highpass(
            num_modes=lmax + 1,
            cutoff=filter_config.ell_cutoff,
            halfwidth=filter_config.ell_halfwidth,
            transition_type=filter_config.transition_type,
        )

    if beam_window_in is None:
        transfer = transfer_numerator
    else:
        input_beam = _as_beam_window(
            beam_window_in,
            lmax=lmax,
            name="beam_window_in",
        )
        # Apply filter zeros before deconvolution. Suppressed modes then remain
        # exactly zero even when the corresponding input beam is extremely small.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            transfer = transfer_numerator / input_beam

    if not np.isfinite(transfer).all():
        raise ValueError(
            "The assembled ell transfer is non-finite; reduce lmax or "
            "deconvolution gain."
        )
    return transfer


def _as_transfer_window(
    transfer: npt.ArrayLike,
    *,
    lmax: int,
    name: str,
    kind: str = "transfer function",
) -> FloatArray:
    """Validate and return a real transfer window through ``lmax``."""

    raw_array = np.asarray(transfer)
    if np.iscomplexobj(raw_array):
        raise ValueError(f"{name} must be a real-valued {kind}.")
    array = np.asarray(raw_array, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D {kind}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")
    if array.shape[0] < lmax + 1:
        raise ValueError(f"{name} must have length at least lmax + 1.")
    return array[: lmax + 1]


def _as_beam_window(
    beam_window: npt.ArrayLike,
    *,
    lmax: int,
    name: str,
) -> FloatArray:
    """Validate a custom input beam window through ``lmax``."""

    window = _as_transfer_window(
        beam_window,
        lmax=lmax,
        name=name,
        kind="beam transfer function",
    )
    if np.any(window <= 0.0):
        raise ValueError(f"{name} must be strictly positive through lmax.")
    return window


def _build_m_transfer(
    *,
    filter_config: HarmonicFilter,
    lmax: int,
) -> FloatArray:
    """Build and validate the configured azimuthal transfer function."""

    transfer = np.ones(lmax + 1, dtype=np.float64)
    if filter_config.m_filter is not None:
        m_filter_window = _as_transfer_window(
            filter_config.m_filter,
            lmax=lmax,
            name="m_filter",
        )
        transfer *= m_filter_window

    if filter_config.m_cutoff is not None:
        transfer *= _build_apodized_highpass(
            num_modes=lmax + 1,
            cutoff=filter_config.m_cutoff,
            halfwidth=filter_config.m_halfwidth,
            transition_type=filter_config.transition_type,
        )

    if not np.isfinite(transfer).all():
        raise ValueError("The assembled m transfer must be finite.")
    return transfer


def _build_apodized_highpass(
    *,
    num_modes: int,
    cutoff: float,
    halfwidth: float,
    transition_type: str,
) -> FloatArray:
    """Build a high-pass taper with a NaMaster-style smooth edge.

    Parameters
    ----------
    num_modes
        Number of discrete modes in the output transfer function.
    cutoff
        Center of the high-pass transition band.
    halfwidth
        Half-width of the transition band. A value of zero gives a hard cutoff.
    transition_type
        Smooth edge type. Supported values are ``"C1"`` and ``"C2"``.

    Returns
    -------
    numpy.ndarray
        High-pass transfer function with length ``num_modes``.

    Raises
    ------
    ValueError
        If ``num_modes`` is not positive, or if ``cutoff``/``halfwidth`` are
        negative.
    """

    if num_modes <= 0:
        raise ValueError("num_modes must be positive.")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative.")
    if halfwidth < 0:
        raise ValueError("halfwidth must be non-negative.")

    modes = np.arange(num_modes, dtype=np.float64)
    if halfwidth == 0:
        return np.where(modes >= cutoff, 1.0, 0.0)

    # Treat the cutoff as the center of the apodized band so the edge is fully
    # specified by the user-provided half-width.
    left = cutoff - halfwidth
    right = cutoff + halfwidth

    response = np.zeros(num_modes, dtype=np.float64)
    response[modes >= right] = 1.0

    transition = (modes > left) & (modes < right)
    if np.any(transition):
        x = (modes[transition] - left) / (right - left)
        response[transition] = _namaster_transition_profile(
            x=x,
            transition_type=transition_type,
        )

    return response


def _num_modes_from_lmax(lmax: int) -> int:
    """Validate ``lmax`` and return the corresponding number of modes.

    Parameters
    ----------
    lmax
        Maximum harmonic index included in the returned filter.

    Returns
    -------
    int
        Number of discrete modes, equal to ``lmax + 1``.

    Raises
    ------
    ValueError
        If ``lmax`` is negative.
    """

    if lmax < 0:
        raise ValueError("lmax must be non-negative.")
    return lmax + 1


def _namaster_transition_profile(
    *,
    x: FloatArray,
    transition_type: str,
) -> FloatArray:
    """Evaluate the normalized NaMaster-style edge profile.

    Parameters
    ----------
    x
        Normalized transition coordinate in the interval ``[0, 1]``.
    transition_type
        Smooth edge type. Supported values are ``"C1"`` and ``"C2"``.

    Returns
    -------
    numpy.ndarray
        Edge profile evaluated at ``x``.

    Raises
    ------
    ValueError
        If ``transition_type`` is not one of the supported options.
    """

    transition = transition_type.upper()
    if transition == "C1":
        return x - np.sin(2.0 * np.pi * x) / (2.0 * np.pi)
    if transition == "C2":
        return 0.5 * (1.0 - np.cos(np.pi * x))
    raise ValueError("transition_type must be either 'C1' or 'C2'.")


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert a Gaussian FWHM to its standard deviation.

    Parameters
    ----------
    fwhm
        Full width at half maximum in radians.

    Returns
    -------
    float
        Corresponding Gaussian standard deviation in radians.
    """

    return float(fwhm) / np.sqrt(8.0 * np.log(2.0))
