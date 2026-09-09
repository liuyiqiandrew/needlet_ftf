from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._filters as filters_mod
import fg_weighted_template_fit._noise as noise_mod


def _make_bootstrap_concurrency_problem() -> dict[str, object]:
    """Build a tiny bootstrap problem that avoids healpy-dependent paths.

    Returns
    -------
    dict
        Keyword arguments accepted by ``bootstrap_template_amplitudes``. The
        target and templates are native-resolution Q/U maps with zero beam
        widths, so the fit stays on the pure NumPy path.
    """

    npix = 8
    dust = np.array(
        [
            [1.0, 0.4, -0.3, 0.8, -0.1, 0.6, 0.2, -0.5],
            [0.2, -0.6, 0.7, 0.1, 0.5, -0.4, 0.3, 0.9],
        ],
        dtype=np.float64,
    )
    sync = np.array(
        [
            [-0.2, 0.7, 0.5, -0.4, 0.9, 0.1, -0.6, 0.3],
            [0.8, 0.1, -0.5, 0.6, -0.3, 0.4, 0.2, -0.7],
        ],
        dtype=np.float64,
    )
    target = 1.35 * dust - 0.45 * sync
    target_noise_cov = np.repeat(
        np.array([[0.01], [0.015], [0.002]], dtype=np.float64),
        npix,
        axis=1,
    )
    template_noise_cov = np.repeat(
        np.array([[0.004], [0.006], [0.001]], dtype=np.float64),
        npix,
        axis=1,
    )
    template_inputs = (
        ftf.DifferenceTemplateInput(
            map_a_qu=dust,
            map_b_qu=np.zeros_like(dust),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_noise_cov,
            noise_cov_b=template_noise_cov,
            name="dust",
        ),
        ftf.DifferenceTemplateInput(
            map_a_qu=sync,
            map_b_qu=np.zeros_like(sync),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_noise_cov,
            noise_cov_b=template_noise_cov,
            name="sync",
        ),
    )

    return {
        "target_qu": target,
        "target_noise_cov": target_noise_cov,
        "target_fwhm_in": 0.0,
        "template_inputs": template_inputs,
        "weight_map": np.ones(npix),
        "fwhm_out": 0.0,
    }


def test_realize_qu_noise_matches_requested_covariance() -> None:
    """Draw Q/U noise realizations with the requested pixel covariance."""

    npix = 20_000
    covariance = np.array([[1.5], [0.9], [0.3]])
    covariance = np.repeat(covariance, npix, axis=1)

    noise = ftf.realize_qu_noise(covariance, rng=0)
    sample_cov = np.cov(noise)
    target_cov = np.array([[1.5, 0.3], [0.3, 0.9]])

    np.testing.assert_allclose(sample_cov, target_cov, atol=0.035)


def test_fit_and_bootstrap_store_mc_amplitudes() -> None:
    """Keep Monte Carlo amplitude draws and summarize their spread."""

    npix = 12
    dust_map_a = np.array(
        [
            [1.0, 1.2, 0.8, 0.5, -0.2, 0.4, 1.1, 0.7, -0.1, 0.2, 0.9, 0.6],
            [0.1, 0.2, -0.3, 0.4, 0.5, -0.4, 0.2, -0.2, 0.3, 0.8, 0.1, -0.1],
        ]
    )
    dust_map_b = np.array(
        [
            [0.2, 0.3, 0.1, 0.0, -0.1, 0.2, 0.4, 0.2, -0.1, 0.1, 0.3, 0.1],
            [0.0, 0.1, -0.1, 0.2, 0.1, -0.2, 0.0, -0.1, 0.1, 0.3, 0.0, -0.1],
        ]
    )
    sync_map_a = np.array(
        [
            [0.5, -0.2, 0.1, -0.4, 0.2, 0.3, -0.1, 0.4, 0.6, -0.3, 0.2, 0.1],
            [-0.3, 0.6, 0.2, -0.1, 0.4, -0.2, 0.5, -0.4, 0.1, 0.3, -0.5, 0.2],
        ]
    )
    sync_map_b = np.array(
        [
            [0.1, -0.1, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.2, -0.1, 0.0, 0.0],
            [-0.1, 0.2, 0.1, 0.0, 0.1, -0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.1],
        ]
    )

    dust_template = dust_map_a - dust_map_b
    sync_template = sync_map_a - sync_map_b
    amplitudes_true = np.array([1.4, -0.6])
    target = amplitudes_true[0] * dust_template + amplitudes_true[1] * sync_template

    noise_cov = np.repeat(np.array([[0.01], [0.01], [0.002]]), npix, axis=1)
    template_cov = np.repeat(np.array([[0.005], [0.005], [0.001]]), npix, axis=1)

    templates = (
        ftf.DifferenceTemplateInput(
            map_a_qu=dust_map_a,
            map_b_qu=dust_map_b,
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_cov,
            noise_cov_b=template_cov,
            name="dust",
        ),
        ftf.DifferenceTemplateInput(
            map_a_qu=sync_map_a,
            map_b_qu=sync_map_b,
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_cov,
            noise_cov_b=template_cov,
            name="sync",
        ),
    )

    reference = ftf.fit_foreground_templates(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=templates,
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        template_inputs_rhs=templates,
    )
    np.testing.assert_allclose(reference.amplitudes, amplitudes_true, atol=1e-12)

    bootstrap = ftf.bootstrap_template_amplitudes(
        target_qu=target,
        target_noise_cov=noise_cov,
        target_fwhm_in=0.0,
        template_inputs=templates,
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        n_mc=8,
        template_inputs_rhs=templates,
        rng=1234,
    )

    assert bootstrap.template_names == ("dust", "sync")
    assert bootstrap.amplitude_samples.shape == (8, 2)
    assert np.all(np.isfinite(bootstrap.amplitude_samples))
    assert np.all(bootstrap.amplitude_std > 0.0)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_bootstrap_template_amplitudes_runs_real_custom_beam_pipeline() -> None:
    """Run reference and Monte Carlo fits through real custom-beam transforms."""

    nside = 1
    npix = 12 * nside**2
    amplitude = 1.4
    rng = np.random.default_rng(811)
    template_map = rng.standard_normal((2, npix))
    zero_covariance = np.zeros((3, npix), dtype=np.float64)
    beam_window = np.array([1.0, 0.95, 0.85])
    filter_config = ftf.HarmonicFilter(lmax=2, iter=5)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template_map,
        map_b_qu=np.zeros_like(template_map),
        fwhm_in_a=np.nan,
        fwhm_in_b=0.0,
        noise_cov_a=zero_covariance,
        noise_cov_b=zero_covariance,
        filter_config=filter_config,
        name="dust",
        beam_window_a=beam_window,
    )

    result = ftf.bootstrap_template_amplitudes(
        target_qu=amplitude * template_map,
        target_noise_cov=zero_covariance,
        target_fwhm_in=np.nan,
        target_beam_window=beam_window,
        template_inputs=(template_input,),
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        n_mc=2,
        target_filter=filter_config,
        rng=123,
    )

    np.testing.assert_allclose(
        result.reference_fit.amplitudes,
        [amplitude],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        result.amplitude_samples,
        amplitude,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(result.amplitude_std, 0.0, atol=1.0e-12)


def test_bootstrap_template_amplitudes_multi_mask_uses_shared_draws() -> None:
    """Reuse each noisy realization across every named fitting mask."""

    npix = 6
    template = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ],
        dtype=np.float64,
    )
    target = 1.7 * template
    target_noise_cov = np.repeat(
        np.array([[0.01], [0.015], [0.002]], dtype=np.float64),
        npix,
        axis=1,
    )
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )
    weight_maps = {
        "m1": np.ones(npix),
        "m2": np.ones(npix),
    }

    first = ftf.bootstrap_template_amplitudes_multi_mask(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps=weight_maps,
        fwhm_out=0.0,
        n_mc=5,
        master_mask=np.array([1.0, 0.8, 1.0, 0.0, 1.0, 1.0]),
        rng=2468,
        n_jobs=2,
    )
    second = ftf.bootstrap_template_amplitudes_multi_mask(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps=weight_maps,
        fwhm_out=0.0,
        n_mc=5,
        master_mask=np.array([1.0, 0.8, 1.0, 0.0, 1.0, 1.0]),
        rng=2468,
        n_jobs=2,
    )

    assert first.fit_names == ("m1", "m2")
    assert first.template_names == ("dust",)
    assert first.amplitude_samples.shape == (5, 2, 1)
    assert np.all(np.isfinite(first.amplitude_samples))
    np.testing.assert_allclose(
        first.amplitude_samples[:, 0, :],
        first.amplitude_samples[:, 1, :],
    )
    np.testing.assert_allclose(first.amplitude_samples, second.amplitude_samples)
    np.testing.assert_allclose(first.amplitude_mean, second.amplitude_mean)
    np.testing.assert_allclose(first.amplitude_std, second.amplitude_std)


def test_bootstrap_template_amplitudes_multi_mask_threads_custom_beams(
    monkeypatch,
) -> None:
    """Forward custom beam windows through paired multi-mask bootstrap draws."""

    npix = 5
    target = np.ones((2, npix), dtype=np.float64)
    target_noise_cov = np.zeros((3, npix), dtype=np.float64)
    target_beam = np.linspace(1.0, 1.2, 4)
    template_beam = np.linspace(1.0, 1.3, 4)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=np.zeros_like(target),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        noise_cov_a=np.zeros((3, npix), dtype=np.float64),
        noise_cov_b=np.zeros((3, npix), dtype=np.float64),
        name="dust",
        beam_window_a=template_beam,
    )
    calls: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    def _weighted_fit(processed_target, processed_template):
        return ftf.WeightedFitResult(
            amplitudes=np.array([1.0], dtype=np.float64),
            normal_matrix=np.ones((1, 1), dtype=np.float64),
            normal_matrix_inverse=np.ones((1, 1), dtype=np.float64),
            rhs=np.ones(1, dtype=np.float64),
            residual_qu=np.zeros_like(processed_target),
            processed_target_qu=processed_target,
            processed_templates_qu=processed_template[None, :, :],
            processed_templates_rhs_qu=processed_template[None, :, :],
            processed_templates_data_qu=processed_template[None, :, :],
            template_names=("dust",),
            solver="solve",
        )

    def fake_fit_foreground_templates_multi_mask(**kwargs):
        calls.append(
            (
                None
                if kwargs["target_beam_window"] is None
                else np.asarray(kwargs["target_beam_window"]),
                None
                if kwargs["template_inputs"][0].beam_window_a is None
                else np.asarray(kwargs["template_inputs"][0].beam_window_a),
            )
        )
        processed_target = np.asarray(kwargs["target_qu"], dtype=np.float64)
        processed_template = np.asarray(
            kwargs["template_inputs"][0].map_a_qu,
            dtype=np.float64,
        )
        fit_results = {
            "m1": _weighted_fit(processed_target, processed_template),
            "m2": _weighted_fit(processed_target, processed_template),
        }
        return ftf.MultiMaskFitResult(
            fit_names=("m1", "m2"),
            fit_results=fit_results,
            template_names=("dust",),
            processed_target_qu=processed_target,
            processed_templates_qu=processed_template[None, :, :],
            processed_templates_rhs_qu=processed_template[None, :, :],
            processed_templates_data_qu=processed_template[None, :, :],
        )

    monkeypatch.setattr(
        noise_mod,
        "fit_foreground_templates_multi_mask",
        fake_fit_foreground_templates_multi_mask,
    )

    result = ftf.bootstrap_template_amplitudes_multi_mask(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=10.0,
        target_beam_window=target_beam,
        template_inputs=(template_input,),
        weight_maps={"m1": np.ones(npix), "m2": np.ones(npix)},
        fwhm_out=0.0,
        n_mc=3,
        master_mask=np.ones(npix),
        rng=1234,
        n_jobs=2,
    )

    assert result.amplitude_samples.shape == (3, 2, 1)
    assert len(calls) == 4
    for target_beam_seen, template_beam_seen in calls:
        np.testing.assert_allclose(target_beam_seen, target_beam)
        np.testing.assert_allclose(template_beam_seen, template_beam)


def test_bootstrap_template_amplitudes_show_progress_uses_tqdm(monkeypatch) -> None:
    """Wrap bootstrap draws in tqdm when progress reporting is enabled."""

    npix = 6
    template = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    target = 1.7 * template
    target_noise_cov = np.zeros((3, npix), dtype=np.float64)

    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(noise_mod, "_tqdm", fake_tqdm)

    result = ftf.bootstrap_template_amplitudes(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        n_mc=3,
        rng=1234,
        show_progress=True,
    )

    assert result.amplitude_samples.shape == (3, 1)
    assert calls == [{"total": 3, "desc": "Bootstrap MC", "unit": "draw"}]


def test_bootstrap_template_amplitudes_n_jobs_one_matches_default_serial() -> None:
    """Keep explicit single-worker bootstrap equivalent to the default path."""

    problem = _make_bootstrap_concurrency_problem()

    default = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=5,
        rng=9876,
    )
    explicit = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=5,
        rng=9876,
        n_jobs=1,
    )

    np.testing.assert_allclose(explicit.amplitude_samples, default.amplitude_samples)
    np.testing.assert_allclose(explicit.amplitude_mean, default.amplitude_mean)
    np.testing.assert_allclose(explicit.amplitude_std, default.amplitude_std)


def test_bootstrap_template_amplitudes_threaded_is_reproducible_for_seed() -> None:
    """Use independent per-draw RNG streams so threaded output is reproducible."""

    problem = _make_bootstrap_concurrency_problem()

    first = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=6,
        rng=12345,
        n_jobs=2,
    )
    second = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=6,
        rng=12345,
        n_jobs=2,
    )

    np.testing.assert_allclose(first.amplitude_samples, second.amplitude_samples)
    np.testing.assert_allclose(first.amplitude_mean, second.amplitude_mean)
    np.testing.assert_allclose(first.amplitude_std, second.amplitude_std)


def test_bootstrap_template_amplitudes_threaded_returns_valid_samples() -> None:
    """Return finite MC samples with the expected threaded output shape."""

    problem = _make_bootstrap_concurrency_problem()

    result = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=6,
        rng=4321,
        n_jobs=2,
    )

    assert result.amplitude_samples.shape == (6, 2)
    assert np.all(np.isfinite(result.amplitude_samples))
    assert np.all(np.isfinite(result.amplitude_mean))
    assert np.all(np.isfinite(result.amplitude_std))


def test_bootstrap_template_amplitudes_threads_custom_beams(monkeypatch) -> None:
    """Forward target and template beam windows through reference and draw fits."""

    npix = 5
    target = np.ones((2, npix), dtype=np.float64)
    target_noise_cov = np.zeros((3, npix), dtype=np.float64)
    target_beam = np.linspace(1.0, 1.2, 4)
    template_beam = np.linspace(1.0, 1.3, 4)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=np.zeros_like(target),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        noise_cov_a=np.zeros((3, npix), dtype=np.float64),
        noise_cov_b=np.zeros((3, npix), dtype=np.float64),
        name="dust",
        beam_window_a=template_beam,
    )
    calls: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    def fake_fit_foreground_templates(**kwargs):
        calls.append(
            (
                None
                if kwargs["target_beam_window"] is None
                else np.asarray(kwargs["target_beam_window"]),
                None
                if kwargs["template_inputs"][0].beam_window_a is None
                else np.asarray(kwargs["template_inputs"][0].beam_window_a),
            )
        )
        processed_target = np.asarray(kwargs["target_qu"], dtype=np.float64)
        processed_template = np.asarray(
            kwargs["template_inputs"][0].map_a_qu,
            dtype=np.float64,
        )
        return ftf.WeightedFitResult(
            amplitudes=np.array([1.0], dtype=np.float64),
            normal_matrix=np.ones((1, 1), dtype=np.float64),
            normal_matrix_inverse=np.ones((1, 1), dtype=np.float64),
            rhs=np.ones(1, dtype=np.float64),
            residual_qu=np.zeros_like(processed_target),
            processed_target_qu=processed_target,
            processed_templates_qu=processed_template[None, :, :],
            processed_templates_rhs_qu=processed_template[None, :, :],
            processed_templates_data_qu=processed_template[None, :, :],
            template_names=("dust",),
            solver="solve",
        )

    monkeypatch.setattr(
        noise_mod,
        "fit_foreground_templates",
        fake_fit_foreground_templates,
    )

    result = ftf.bootstrap_template_amplitudes(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=10.0,
        target_beam_window=target_beam,
        template_inputs=(template_input,),
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        n_mc=3,
        rng=1234,
        n_jobs=2,
    )

    assert result.amplitude_samples.shape == (3, 1)
    assert len(calls) == 4
    for target_beam_seen, template_beam_seen in calls:
        np.testing.assert_allclose(target_beam_seen, target_beam)
        np.testing.assert_allclose(template_beam_seen, template_beam)


def test_bootstrap_template_amplitudes_rejects_nonpositive_n_jobs() -> None:
    """Reject invalid worker counts before starting a bootstrap run."""

    problem = _make_bootstrap_concurrency_problem()

    for n_jobs in (0, -1):
        with pytest.raises(ValueError, match="n_jobs"):
            ftf.bootstrap_template_amplitudes(
                **problem,
                n_mc=1,
                rng=2468,
                n_jobs=n_jobs,
            )


def test_bootstrap_template_amplitudes_threaded_show_progress_uses_tqdm(
    monkeypatch,
) -> None:
    """Wrap completed threaded draws in tqdm when progress reporting is enabled."""

    problem = _make_bootstrap_concurrency_problem()
    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(noise_mod, "_tqdm", fake_tqdm)

    result = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=3,
        rng=1357,
        show_progress=True,
        n_jobs=2,
    )

    assert result.amplitude_samples.shape == (3, 2)
    assert calls == [{"total": 3, "desc": "Bootstrap MC", "unit": "draw"}]


def test_bootstrap_template_amplitudes_threaded_show_progress_requires_tqdm(
    monkeypatch,
) -> None:
    """Raise the standard progress error before threaded progress reporting."""

    problem = _make_bootstrap_concurrency_problem()
    monkeypatch.setattr(noise_mod, "_tqdm", None)

    with pytest.raises(ImportError, match="requires tqdm"):
        ftf.bootstrap_template_amplitudes(
            **problem,
            n_mc=2,
            rng=9753,
            show_progress=True,
            n_jobs=2,
        )


def test_bootstrap_template_amplitudes_show_progress_requires_tqdm(
    monkeypatch,
) -> None:
    """Raise an informative error when tqdm-backed progress is unavailable."""

    npix = 6
    template = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    target_noise_cov = np.zeros((3, npix), dtype=np.float64)

    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    monkeypatch.setattr(noise_mod, "_tqdm", None)

    with pytest.raises(ImportError, match="requires tqdm"):
        ftf.bootstrap_template_amplitudes(
            target_qu=template,
            target_noise_cov=target_noise_cov,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_map=np.ones(npix),
            fwhm_out=0.0,
            n_mc=2,
            show_progress=True,
        )


def test_bootstrap_template_amplitudes_accepts_data_projection_templates() -> None:
    """Realize noise on the third stack and return well-shaped samples."""

    npix = 8
    dust = np.array(
        [
            [1.0, 0.4, -0.3, 0.8, -0.1, 0.6, 0.2, -0.5],
            [0.2, -0.6, 0.7, 0.1, 0.5, -0.4, 0.3, 0.9],
        ],
        dtype=np.float64,
    )
    dust_lhs = dust
    dust_rhs = dust + 0.05
    dust_data = dust - 0.05
    target = 1.35 * dust
    target_noise_cov = np.repeat(
        np.array([[0.01], [0.015], [0.002]], dtype=np.float64), npix, axis=1
    )
    template_cov = np.repeat(
        np.array([[0.004], [0.006], [0.001]], dtype=np.float64), npix, axis=1
    )

    def _input(map_a):
        return ftf.DifferenceTemplateInput(
            map_a_qu=map_a,
            map_b_qu=np.zeros_like(map_a),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_cov,
            noise_cov_b=template_cov,
            name="dust",
        )

    bootstrap = ftf.bootstrap_template_amplitudes(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=0.0,
        template_inputs=(_input(dust_lhs),),
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        n_mc=6,
        template_inputs_rhs=(_input(dust_rhs),),
        template_inputs_data=(_input(dust_data),),
        rng=2024,
    )

    assert bootstrap.template_names == ("dust",)
    assert bootstrap.amplitude_samples.shape == (6, 1)
    assert np.all(np.isfinite(bootstrap.amplitude_samples))
    assert np.all(bootstrap.amplitude_std > 0.0)
    np.testing.assert_allclose(
        bootstrap.reference_fit.processed_templates_data_qu[0],
        dust_data,
        atol=1e-12,
    )
