from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._filters as filters_mod


def test_build_ell_filter_matches_c1_and_c2_profiles() -> None:
    """Build ell-space tapers that match the expected C1 and C2 profiles."""

    c2_window = ftf.build_ell_filter(
        lmax=9,
        cutoff=5.0,
        halfwidth=2.0,
        transition_type="C2",
    )
    c1_window = ftf.build_ell_filter(
        lmax=9,
        cutoff=5.0,
        halfwidth=2.0,
        transition_type="C1",
    )

    np.testing.assert_allclose(c2_window[:4], 0.0)
    np.testing.assert_allclose(c1_window[:4], 0.0)
    np.testing.assert_allclose(c2_window[7:], 1.0)
    np.testing.assert_allclose(c1_window[7:], 1.0)

    transition_x = np.array([0.25, 0.5, 0.75])
    expected_c2 = 0.5 * (1.0 - np.cos(np.pi * transition_x))
    expected_c1 = transition_x - np.sin(2.0 * np.pi * transition_x) / (2.0 * np.pi)

    np.testing.assert_allclose(c2_window[4:7], expected_c2)
    np.testing.assert_allclose(c1_window[4:7], expected_c1)


def test_build_m_filter_matches_ell_filter_profile() -> None:
    """Match the m-space helper output to the ell-space taper profile."""

    ell_window = ftf.build_ell_filter(
        lmax=9,
        cutoff=4.0,
        halfwidth=1.0,
        transition_type="C2",
    )
    m_window = ftf.build_m_filter(
        lmax=9,
        cutoff=4.0,
        halfwidth=1.0,
        transition_type="C2",
    )

    np.testing.assert_allclose(m_window, ell_window)


def test_build_ell_filter_rejects_negative_lmax() -> None:
    """Reject negative harmonic truncation when building ell filters."""

    with pytest.raises(ValueError, match="lmax must be non-negative"):
        ftf.build_ell_filter(
            lmax=-1,
            cutoff=3.0,
        )


def test_resolve_lmax_does_not_truncate_highpass_ell_cutoff() -> None:
    """Keep the native transform support when only an ell cutoff is requested."""

    assert filters_mod._resolve_lmax(nside=8, explicit_lmax=None) == 23


@pytest.mark.parametrize(
    ("filter_config", "beam_window_in", "array_name"),
    [
        (ftf.HarmonicFilter(), np.ones(11), "beam_window_in"),
        (ftf.HarmonicFilter(ell_filter=np.ones(11)), None, "ell_filter"),
        (ftf.HarmonicFilter(m_filter=np.ones(11)), None, "m_filter"),
    ],
)
def test_resolve_lmax_rejects_implicit_transfer_truncation(
    filter_config: ftf.HarmonicFilter,
    beam_window_in: np.ndarray | None,
    array_name: str,
) -> None:
    """Reject short transfers instead of silently dropping high-ell content."""

    with pytest.raises(
        ValueError,
        match=rf"{array_name}.*length at least lmax \+ 1",
    ):
        ftf.smooth_and_filter_qu_map(
            qu_map=np.zeros((2, 12 * 8**2), dtype=np.float64),
            fwhm_in=0.0,
            fwhm_out=0.0,
            filter_config=filter_config,
            beam_window_in=beam_window_in,
        )


def test_resolve_lmax_accepts_explicit_supported_lmax() -> None:
    """Accept a caller-selected lmax within the map-native support."""

    assert filters_mod._resolve_lmax(nside=8, explicit_lmax=10) == 10


def test_build_ell_transfer_uses_custom_input_beam() -> None:
    """Use B_out / B_in when a custom input beam window is supplied."""

    lmax = 6
    fwhm_out = 0.05
    beam_window = np.linspace(1.0, 1.6, lmax + 1)

    transfer = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        filter_config=ftf.HarmonicFilter(),
        beam_window_in=beam_window,
    )

    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_out = filters_mod._fwhm_to_sigma(fwhm_out)
    expected = np.exp(-0.5 * ells * (ells + 1.0) * sigma_out**2) / beam_window
    np.testing.assert_allclose(transfer, expected)


def test_build_ell_transfer_custom_gaussian_matches_default() -> None:
    """Match the legacy Gaussian transfer when B_in is the same Gaussian beam."""

    lmax = 8
    fwhm_in = 0.03
    fwhm_out = 0.08
    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_in = filters_mod._fwhm_to_sigma(fwhm_in)
    beam_window = np.exp(-0.5 * ells * (ells + 1.0) * sigma_in**2)

    default = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=fwhm_in,
        fwhm_out=fwhm_out,
        filter_config=ftf.HarmonicFilter(),
    )
    custom = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        filter_config=ftf.HarmonicFilter(),
        beam_window_in=beam_window,
    )

    np.testing.assert_allclose(custom, default)


def test_build_ell_transfer_custom_beam_composes_with_ell_filters() -> None:
    """Apply custom beam matching and ell filters in one transfer."""

    lmax = 8
    fwhm_out = 0.04
    beam_window = np.linspace(1.0, 1.4, lmax + 1)
    ell_filter = np.linspace(0.5, 1.0, lmax + 1)
    filter_config = ftf.HarmonicFilter(
        ell_filter=ell_filter,
        ell_cutoff=4.0,
        ell_halfwidth=1.0,
        transition_type="C2",
    )

    transfer = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        filter_config=filter_config,
        beam_window_in=beam_window,
    )

    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_out = filters_mod._fwhm_to_sigma(fwhm_out)
    output_beam = np.exp(-0.5 * ells * (ells + 1.0) * sigma_out**2)
    highpass = filters_mod._build_apodized_highpass(
        num_modes=lmax + 1,
        cutoff=4.0,
        halfwidth=1.0,
        transition_type="C2",
    )
    expected = output_beam / beam_window * ell_filter * highpass

    np.testing.assert_allclose(transfer, expected)


@pytest.mark.parametrize(
    ("beam_window", "match"),
    [
        (np.ones((2, 4)), "1D"),
        (np.array([1.0, 1.0]), "length"),
        (np.ones(3, dtype=np.complex128), "real|complex"),
        (np.array([1.0, 0.0, 1.0]), "strictly positive"),
        (np.array([1.0, -0.5, 1.0]), "strictly positive"),
        (np.array([1.0, np.nan, 1.0]), "finite"),
        (np.array([1.0, np.inf, 1.0]), "finite"),
    ],
)
def test_build_ell_transfer_rejects_invalid_custom_beam_window(
    beam_window,
    match,
) -> None:
    """Reject custom input beams that would make deconvolution ill-defined."""

    with pytest.raises(ValueError, match=match):
        filters_mod._build_ell_transfer(
            lmax=2,
            fwhm_in=0.0,
            fwhm_out=0.0,
            filter_config=ftf.HarmonicFilter(),
            beam_window_in=beam_window,
        )


@pytest.mark.parametrize(
    ("fwhm_in", "fwhm_out", "beam_window_in", "parameter_name"),
    [
        (-0.1, 0.0, None, "fwhm_in"),
        (np.nan, 0.0, None, "fwhm_in"),
        (np.inf, 0.0, None, "fwhm_in"),
        (0.0, -0.1, None, "fwhm_out"),
        (0.0, np.nan, None, "fwhm_out"),
        (0.0, np.inf, None, "fwhm_out"),
        (np.nan, -0.1, np.ones(3), "fwhm_out"),
    ],
)
def test_smooth_and_filter_qu_map_rejects_invalid_fwhm(
    fwhm_in: float,
    fwhm_out: float,
    beam_window_in: np.ndarray | None,
    parameter_name: str,
) -> None:
    """Reject non-finite or negative beam widths before harmonic work."""

    with pytest.raises(ValueError, match=parameter_name):
        ftf.smooth_and_filter_qu_map(
            qu_map=np.ones((2, 12), dtype=np.float64),
            fwhm_in=fwhm_in,
            fwhm_out=fwhm_out,
            beam_window_in=beam_window_in,
        )


def test_build_ell_transfer_rejects_nonfinite_custom_beam_gain() -> None:
    """Reject custom-beam deconvolution whose final gain overflows."""

    beam_window = np.array([1.0, 1.0e-320, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="finite|deconvolution|gain"):
        filters_mod._build_ell_transfer(
            lmax=2,
            fwhm_in=0.0,
            fwhm_out=0.0,
            filter_config=ftf.HarmonicFilter(),
            beam_window_in=beam_window,
        )


def test_build_ell_transfer_filters_before_custom_beam_division() -> None:
    """Keep a zero filtered mode finite despite a tiny custom input beam."""

    beam_window = np.array([1.0, 1.0e-320, 1.0], dtype=np.float64)
    transfer = filters_mod._build_ell_transfer(
        lmax=2,
        fwhm_in=0.0,
        fwhm_out=0.0,
        filter_config=ftf.HarmonicFilter(ell_filter=np.array([1.0, 0.0, 1.0])),
        beam_window_in=beam_window,
    )

    assert np.all(np.isfinite(transfer))
    assert transfer[1] == 0.0


def test_smooth_and_filter_qu_map_unity_custom_beam_is_healpy_free(
    monkeypatch,
) -> None:
    """Keep exact custom-beam matching on the pure-NumPy identity path."""

    monkeypatch.setattr(filters_mod, "hp", None)
    qu_map = np.arange(24, dtype=np.float64).reshape(2, 12)

    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=np.nan,
        fwhm_out=0.0,
        beam_window_in=np.ones(3, dtype=np.float64),
    )

    np.testing.assert_allclose(filtered, qu_map)
    assert filtered is not qu_map


def test_smooth_and_filter_qu_map_custom_beam_overrides_fwhm_validation(
    monkeypatch,
) -> None:
    """Let a custom input beam replace the Gaussian input FWHM."""

    class FakeHealpy:
        class Alm:
            @staticmethod
            def getlm(lmax):
                nalm = (lmax + 1) * (lmax + 2) // 2
                return np.zeros(nalm, dtype=np.int64), np.zeros(nalm, dtype=np.int64)

        def __init__(self) -> None:
            self.transfers: list[np.ndarray] = []
            self.last_tqu: np.ndarray | None = None

        def npix2nside(self, npix: int) -> int:
            assert npix == 12
            return 1

        def map2alm(self, tqu, lmax, iter, pol):
            del iter, pol
            self.last_tqu = np.asarray(tqu, dtype=np.float64).copy()
            nalm = (lmax + 1) * (lmax + 2) // 2
            zeros = np.zeros(nalm, dtype=np.complex128)
            return zeros.copy(), zeros.copy(), zeros.copy()

        def almxfl(self, alm, transfer, inplace=False):
            del inplace
            self.transfers.append(np.asarray(transfer, dtype=np.float64).copy())
            return np.asarray(alm, dtype=np.complex128).copy()

        def alm2map(self, alms, nside, lmax, pol):
            del alms, nside, lmax, pol
            assert self.last_tqu is not None
            return self.last_tqu.copy()

    fake_hp = FakeHealpy()
    monkeypatch.setattr(filters_mod, "hp", fake_hp)

    beam_window = np.array([1.0, 1.1, 1.2])
    qu_map = np.ones((2, 12), dtype=np.float64)
    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=np.nan,
        fwhm_out=0.0,
        beam_window_in=beam_window,
    )

    np.testing.assert_allclose(filtered, qu_map)
    assert len(fake_hp.transfers) == 2
    np.testing.assert_allclose(fake_hp.transfers[0], 1.0 / beam_window)


def test_smooth_and_filter_qu_map_gaussian_beam_window_matches_fwhm(
    monkeypatch,
) -> None:
    """Match public preprocessing when B_ell is the same Gaussian input beam."""

    class FakeHealpy:
        class Alm:
            @staticmethod
            def getlm(lmax):
                nalm = (lmax + 1) * (lmax + 2) // 2
                return np.zeros(nalm, dtype=np.int64), np.zeros(nalm, dtype=np.int64)

        def __init__(self) -> None:
            self.transfers: list[np.ndarray] = []

        def npix2nside(self, npix: int) -> int:
            assert npix == 12
            return 1

        def map2alm(self, tqu, lmax, iter, pol):
            del tqu, iter, pol
            nalm = (lmax + 1) * (lmax + 2) // 2
            zeros = np.zeros(nalm, dtype=np.complex128)
            return zeros.copy(), zeros.copy(), zeros.copy()

        def almxfl(self, alm, transfer, inplace=False):
            del alm, inplace
            self.transfers.append(np.asarray(transfer, dtype=np.float64).copy())
            return np.asarray(transfer, dtype=np.complex128)

        def alm2map(self, alms, nside, lmax, pol):
            del lmax, pol
            npix = 12 * nside**2
            q_value = float(np.sum(alms[1].real))
            u_value = float(np.sum(alms[2].real))
            return np.vstack(
                [
                    np.zeros(npix, dtype=np.float64),
                    np.full(npix, q_value, dtype=np.float64),
                    np.full(npix, u_value, dtype=np.float64),
                ]
            )

    fake_hp = FakeHealpy()
    monkeypatch.setattr(filters_mod, "hp", fake_hp)

    lmax = 2
    fwhm_in = 0.03
    fwhm_out = 0.08
    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_in = filters_mod._fwhm_to_sigma(fwhm_in)
    gaussian_input_beam = np.exp(-0.5 * ells * (ells + 1.0) * sigma_in**2)
    qu_map = np.arange(24, dtype=np.float64).reshape(2, 12)

    gaussian_fwhm_result = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=fwhm_in,
        fwhm_out=fwhm_out,
    )
    custom_beam_result = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        beam_window_in=gaussian_input_beam,
    )

    np.testing.assert_allclose(custom_beam_result, gaussian_fwhm_result)
    assert len(fake_hp.transfers) == 4
    np.testing.assert_allclose(fake_hp.transfers[2], fake_hp.transfers[0])
    np.testing.assert_allclose(fake_hp.transfers[3], fake_hp.transfers[1])


def test_build_template_stack_threads_custom_beam_windows(monkeypatch) -> None:
    """Pass per-map custom beam windows from DifferenceTemplateInput."""

    calls: list[dict[str, object]] = []

    def fake_construct_difference_template(**kwargs):
        calls.append(kwargs)
        return np.ones((2, 12), dtype=np.float64)

    monkeypatch.setattr(
        filters_mod,
        "_construct_difference_template_with_lmax",
        fake_construct_difference_template,
    )

    beam_a = np.ones(5, dtype=np.float64)
    beam_b = np.linspace(1.0, 1.2, 5)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=np.ones((2, 12), dtype=np.float64),
        map_b_qu=np.zeros((2, 12), dtype=np.float64),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
        beam_window_a=beam_a,
        beam_window_b=beam_b,
    )

    templates, names = ftf.build_template_stack(
        (template_input,),
        fwhm_out=0.0,
    )

    assert names == ("dust",)
    assert templates.shape == (1, 2, 12)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["beam_window_a"], beam_a)
    np.testing.assert_allclose(calls[0]["beam_window_b"], beam_b)


def test_construct_difference_template_rejects_unequal_pixel_counts() -> None:
    """Reject operands that cannot share one harmonic preprocessing plan."""

    with pytest.raises(ValueError, match="same npix|same number of pixels"):
        ftf.construct_difference_template(
            map_a_qu=np.ones((2, 12), dtype=np.float64),
            map_b_qu=np.ones((2, 48), dtype=np.float64),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            fwhm_out=0.0,
        )


def test_build_template_stack_rejects_conflicting_explicit_lmax() -> None:
    """Reject incompatible band limits within one template stack."""

    qu_map = np.ones((2, 48), dtype=np.float64)
    template_inputs = tuple(
        ftf.DifferenceTemplateInput(
            map_a_qu=qu_map,
            map_b_qu=np.zeros_like(qu_map),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            filter_config=ftf.HarmonicFilter(lmax=lmax),
            name=f"template_{lmax}",
        )
        for lmax in (3, 4)
    )

    with pytest.raises(ValueError, match="same explicit lmax|conflicting.*lmax"):
        ftf.build_template_stack(template_inputs, fwhm_out=0.0)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_smoothing_reduces_variance() -> None:
    """Lower map variance when Gaussian smoothing is applied."""

    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(42)
    qu_map = rng.standard_normal((2, npix))

    smoothed = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=np.radians(1.0),
    )

    assert smoothed.shape == qu_map.shape
    assert np.var(smoothed) < np.var(qu_map)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_explicit_lmax_projects_high_ell_mode() -> None:
    """Apply an explicit band limit even when Gaussian beams already match."""

    hp = filters_mod.hp
    assert hp is not None

    nside = 16
    input_lmax = 16
    output_lmax = 5
    alm_e = np.zeros(hp.Alm.getsize(input_lmax), dtype=np.complex128)
    alm_b = np.zeros_like(alm_e)
    alm_e[hp.Alm.getidx(input_lmax, 12, 5)] = 1.0
    input_qu = np.asarray(
        hp.alm2map(
            [np.zeros_like(alm_e), alm_e, alm_b],
            nside=nside,
            lmax=input_lmax,
            pol=True,
        )[1:],
        dtype=np.float64,
    )

    projected = ftf.smooth_and_filter_qu_map(
        qu_map=input_qu,
        fwhm_in=0.0,
        fwhm_out=0.0,
        filter_config=ftf.HarmonicFilter(lmax=output_lmax, iter=5),
    )

    relative_rms = np.linalg.norm(projected) / np.linalg.norm(input_qu)
    assert relative_rms < 1.0e-3


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_unity_custom_beam_matches_gaussian_path() -> None:
    """Match unity custom-beam and zero-width Gaussian preprocessing."""

    nside = 8
    npix = 12 * nside**2
    lmax = 12
    rng = np.random.default_rng(481)
    qu_map = rng.standard_normal((2, npix))
    filter_config = ftf.HarmonicFilter(lmax=lmax)

    gaussian = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=0.0,
        filter_config=filter_config,
    )
    custom = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=np.nan,
        fwhm_out=0.0,
        beam_window_in=np.ones(lmax + 1, dtype=np.float64),
        filter_config=filter_config,
    )

    np.testing.assert_allclose(custom, gaussian, rtol=0.0, atol=1.0e-12)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_construct_difference_template_uses_common_mixed_beam_operator() -> None:
    """Use the same SHT support for custom- and Gaussian-beam operands."""

    nside = 8
    npix = 12 * nside**2
    lmax = 12
    rng = np.random.default_rng(982)
    qu_map = rng.standard_normal((2, npix))

    difference = ftf.construct_difference_template(
        map_a_qu=qu_map,
        map_b_qu=qu_map,
        fwhm_in_a=np.nan,
        fwhm_in_b=0.0,
        fwhm_out=0.0,
        beam_window_a=np.ones(lmax + 1, dtype=np.float64),
        filter_config=ftf.HarmonicFilter(lmax=lmax),
    )

    np.testing.assert_allclose(difference, 0.0, rtol=0.0, atol=1.0e-12)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_custom_beam_matches_direct_synthesis() -> None:
    """Recover direct output-beam synthesis from a custom-beam input sky."""

    hp = filters_mod.hp
    assert hp is not None

    nside = 16
    lmax = 12
    rng = np.random.default_rng(91)
    alm_size = hp.Alm.getsize(lmax)
    ell, emm = hp.Alm.getlm(lmax)
    alm_e = rng.standard_normal(alm_size) + 1j * rng.standard_normal(alm_size)
    alm_b = rng.standard_normal(alm_size) + 1j * rng.standard_normal(alm_size)
    alm_e[ell < 2] = 0.0
    alm_b[ell < 2] = 0.0
    alm_e[emm == 0] = alm_e[emm == 0].real
    alm_b[emm == 0] = alm_b[emm == 0].real

    ells = np.arange(lmax + 1, dtype=np.float64)
    input_beam = np.exp(-0.002 * ells * (ells + 1.0)) * (
        1.0 + 0.04 * np.cos(0.7 * ells)
    )
    input_beam /= input_beam[0]
    fwhm_out = 0.08
    sigma_out = filters_mod._fwhm_to_sigma(fwhm_out)
    output_beam = np.exp(-0.5 * ells * (ells + 1.0) * sigma_out**2)
    zero_alm = np.zeros(alm_size, dtype=np.complex128)

    input_qu = np.asarray(
        hp.alm2map(
            [
                zero_alm,
                hp.almxfl(alm_e, input_beam),
                hp.almxfl(alm_b, input_beam),
            ],
            nside=nside,
            lmax=lmax,
            pol=True,
        )[1:],
        dtype=np.float64,
    )
    expected_qu = np.asarray(
        hp.alm2map(
            [
                zero_alm,
                hp.almxfl(alm_e, output_beam),
                hp.almxfl(alm_b, output_beam),
            ],
            nside=nside,
            lmax=lmax,
            pol=True,
        )[1:],
        dtype=np.float64,
    )

    processed_qu = ftf.smooth_and_filter_qu_map(
        qu_map=input_qu,
        fwhm_in=np.nan,
        fwhm_out=fwhm_out,
        beam_window_in=input_beam,
        filter_config=ftf.HarmonicFilter(lmax=lmax, iter=5),
    )

    relative_rms = np.linalg.norm(processed_qu - expected_qu) / np.linalg.norm(
        expected_qu
    )
    assert relative_rms < 1.0e-9


def test_smooth_and_filter_qu_map_applies_mask_before_transform(
    monkeypatch,
) -> None:
    """Apply the preprocessing mask before calling the harmonic transform."""

    class FakeHealpy:
        class Alm:
            @staticmethod
            def getlm(lmax):
                nalm = (lmax + 1) * (lmax + 2) // 2
                return np.zeros(nalm, dtype=np.int64), np.zeros(nalm, dtype=np.int64)

        def __init__(self) -> None:
            self.last_tqu: np.ndarray | None = None

        def npix2nside(self, npix: int) -> int:
            assert npix == 12
            return 1

        def map2alm(self, tqu, lmax, iter, pol):
            del iter, pol
            self.last_tqu = np.asarray(tqu, dtype=np.float64).copy()
            nalm = (lmax + 1) * (lmax + 2) // 2
            zeros = np.zeros(nalm, dtype=np.complex128)
            return zeros.copy(), zeros.copy(), zeros.copy()

        def almxfl(self, alm, transfer, inplace=False):
            del transfer, inplace
            return np.asarray(alm, dtype=np.complex128).copy()

        def alm2map(self, alms, nside, lmax, pol):
            del alms, nside, lmax, pol
            assert self.last_tqu is not None
            return self.last_tqu.copy()

    fake_hp = FakeHealpy()
    monkeypatch.setattr(filters_mod, "hp", fake_hp)

    qu_map = np.arange(24, dtype=np.float64).reshape(2, 12)
    mask = np.linspace(0.0, 1.0, 12)

    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=np.radians(1.0),
        mask=mask,
    )

    expected = qu_map * np.vstack([mask, mask])
    assert fake_hp.last_tqu is not None
    np.testing.assert_allclose(fake_hp.last_tqu[0], 0.0)
    np.testing.assert_allclose(fake_hp.last_tqu[1:], expected)
    np.testing.assert_allclose(filtered, expected)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_accepts_m_cutoff_with_smooth_edge() -> None:
    """Accept smooth-edge m cutoffs and return a finite filtered map."""

    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(123)
    qu_map = rng.standard_normal((2, npix))

    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=0.0,
        filter_config=ftf.HarmonicFilter(
            m_cutoff=3.0,
            m_halfwidth=2.0,
            transition_type="C1",
        ),
    )

    assert filtered.shape == qu_map.shape
    assert np.all(np.isfinite(filtered))
    assert np.var(filtered) < np.var(qu_map)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_accepts_public_explicit_filters() -> None:
    """Accept explicit public ell and m filter arrays."""

    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(456)
    qu_map = rng.standard_normal((2, npix))

    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=0.0,
        filter_config=ftf.HarmonicFilter(
            ell_filter=ftf.build_ell_filter(
                lmax=3 * nside - 1,
                cutoff=6.0,
                halfwidth=2.0,
                transition_type="C2",
            ),
            m_filter=ftf.build_m_filter(
                lmax=3 * nside - 1,
                cutoff=4.0,
                halfwidth=1.0,
                transition_type="C1",
            ),
        ),
    )

    assert filtered.shape == qu_map.shape
    assert np.all(np.isfinite(filtered))
    assert not np.allclose(filtered, qu_map)
