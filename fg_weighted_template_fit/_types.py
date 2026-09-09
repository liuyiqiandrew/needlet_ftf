from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HarmonicFilter:
    """Configuration for optional harmonic-domain filtering.

    Parameters
    ----------
    ell_filter
        Optional real, finite 1D transfer indexed by multipole ``ell``. It must
        cover ``lmax``. Arrays from ``build_ell_filter`` can be passed directly.
    m_filter
        Optional real, finite 1D transfer indexed by azimuthal mode ``m``. It
        must cover ``lmax``. Arrays from ``build_m_filter`` can be passed
        directly.
    ell_cutoff
        Optional high-pass cutoff in multipole. The response is zero below
        ``ell_cutoff - ell_halfwidth`` and unity above
        ``ell_cutoff + ell_halfwidth``.
    ell_halfwidth
        Half-width of the smooth transition around ``ell_cutoff``.
    m_cutoff
        Optional high-pass cutoff in azimuthal mode. The response is zero below
        ``m_cutoff - m_halfwidth`` and unity above ``m_cutoff + m_halfwidth``.
    m_halfwidth
        Half-width of the smooth transition around ``m_cutoff``.
    transition_type
        Edge shape used for the optional ``ell`` and ``m`` cutoffs. Supported
        values are ``"C1"`` and ``"C2"``. The default ``"C2"`` follows the
        NaMaster convention for a smooth edge on a normalized transition
        coordinate.
    lmax
        Shared maximum multipole used for the spherical-harmonic transform. If
        omitted, the implementation uses ``3 * nside - 1``. Every explicit
        filter and custom beam window must cover the selected support.
    iter
        Number of map-to-alm Jacobi iterations used by ``healpy.map2alm``.
    """

    ell_filter: npt.ArrayLike | None = None
    m_filter: npt.ArrayLike | None = None
    ell_cutoff: float | None = None
    ell_halfwidth: float = 0.0
    m_cutoff: float | None = None
    m_halfwidth: float = 0.0
    transition_type: str = "C2"
    lmax: int | None = None
    iter: int = 3


@dataclass(frozen=True)
class DifferenceTemplateInput:
    """Inputs needed to build a foreground template from two Q/U maps.

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
    noise_cov_a
        Optional per-pixel ``QQ, UU, QU`` covariance for ``map_a_qu`` with
        shape ``(3, npix)`` or ``(npix, 3)``.
    noise_cov_b
        Optional per-pixel ``QQ, UU, QU`` covariance for ``map_b_qu`` with
        shape ``(3, npix)`` or ``(npix, 3)``.
    filter_config
        Optional harmonic filter applied while constructing the template. If
        omitted, the default filter passed to the fit routine is used.
    name
        Human-readable name for the template.
    beam_window_a
        Optional real, finite, strictly positive 1D transfer ``B_ell`` for a
        scalar-valued, axisymmetric input beam on ``map_a_qu``. It applies
        equally to E and B and must cover the shared ``lmax``. When supplied,
        it replaces the Gaussian beam implied by ``fwhm_in_a``.
    beam_window_b
        Optional scalar input beam ``B_ell`` for ``map_b_qu`` with the same
        requirements as ``beam_window_a``.
    """

    map_a_qu: npt.ArrayLike
    map_b_qu: npt.ArrayLike
    fwhm_in_a: float
    fwhm_in_b: float
    noise_cov_a: npt.ArrayLike | None = None
    noise_cov_b: npt.ArrayLike | None = None
    filter_config: HarmonicFilter | None = None
    name: str = "template"
    beam_window_a: npt.ArrayLike | None = None
    beam_window_b: npt.ArrayLike | None = None


@dataclass(frozen=True)
class WeightedFitResult:
    """Result of the weighted template fit."""

    amplitudes: FloatArray
    normal_matrix: FloatArray
    normal_matrix_inverse: FloatArray
    rhs: FloatArray
    residual_qu: FloatArray
    processed_target_qu: FloatArray
    processed_templates_qu: FloatArray
    processed_templates_rhs_qu: FloatArray
    processed_templates_data_qu: FloatArray
    template_names: tuple[str, ...]
    solver: str


@dataclass(frozen=True)
class MultiMaskFitResult:
    """Result of fitting one processed map/template set with multiple weights.

    Attributes
    ----------
    fit_names
        Names of the fitted regions in the order supplied by the input weight
        mapping.
    fit_results
        Per-region weighted fit results keyed by ``fit_names``.
    template_names
        Names associated with the template axis.
    processed_target_qu
        Master-mask-processed target Q/U map with shape ``(2, npix)``.
    processed_templates_qu
        Master-mask-processed left-hand template stack with shape
        ``(n_template, 2, npix)``.
    processed_templates_rhs_qu
        Master-mask-processed right-hand template stack with shape
        ``(n_template, 2, npix)``.
    processed_templates_data_qu
        Master-mask-processed data-projection template stack (the ``d_3`` used in
        the RHS vector ``d_3^T W m``) with shape ``(n_template, 2, npix)``.
    """

    fit_names: tuple[str, ...]
    fit_results: dict[str, WeightedFitResult]
    template_names: tuple[str, ...]
    processed_target_qu: FloatArray
    processed_templates_qu: FloatArray
    processed_templates_rhs_qu: FloatArray
    processed_templates_data_qu: FloatArray


@dataclass(frozen=True)
class BootstrapFitResult:
    """Monte Carlo uncertainty estimate for weighted template amplitudes."""

    reference_fit: WeightedFitResult
    amplitude_samples: FloatArray
    amplitude_mean: FloatArray
    amplitude_std: FloatArray
    template_names: tuple[str, ...]


@dataclass(frozen=True)
class MultiMaskBootstrapResult:
    """Monte Carlo uncertainty estimate for multiple weighted fits.

    Attributes
    ----------
    reference_fit
        Deterministic multi-mask fit to the input maps before noise draws.
    amplitude_samples
        Fitted amplitudes with shape ``(n_mc, n_fit_mask, n_template)``.
    amplitude_mean
        Mean amplitudes over Monte Carlo draws with shape
        ``(n_fit_mask, n_template)``.
    amplitude_std
        Sample standard deviation over Monte Carlo draws with shape
        ``(n_fit_mask, n_template)``.
    fit_names
        Names associated with the fit-mask axis.
    template_names
        Names associated with the template axis.
    """

    reference_fit: MultiMaskFitResult
    amplitude_samples: FloatArray
    amplitude_mean: FloatArray
    amplitude_std: FloatArray
    fit_names: tuple[str, ...]
    template_names: tuple[str, ...]
