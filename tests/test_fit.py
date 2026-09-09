from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._filters as filters_mod
import fg_weighted_template_fit._fit as fit_mod


def test_weighted_template_gls_recovers_known_amplitudes() -> None:
    """Recover exact amplitudes for the standard weighted template solve."""

    npix = 6
    dust = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    sync = np.array(
        [
            [0.4, -0.6, 0.1, 0.9, -0.1, 0.3],
            [-0.2, 0.7, 0.5, -0.4, 0.6, -0.8],
        ]
    )
    amplitudes_true = np.array([1.7, -0.35])
    target = amplitudes_true[0] * dust + amplitudes_true[1] * sync
    weight_map = np.array([1.0, 3.0, 0.5, 2.0, 4.0, 1.5])

    result = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=np.stack([dust, sync], axis=0),
        weight_map=weight_map,
        template_names=("dust", "sync"),
    )

    np.testing.assert_allclose(result.amplitudes, amplitudes_true, atol=1e-12)
    np.testing.assert_allclose(result.residual_qu, 0.0, atol=1e-12)
    assert result.template_names == ("dust", "sync")


def test_weighted_template_gls_supports_cross_normal_matrix() -> None:
    """Recover amplitudes when left- and right-hand template stacks differ."""

    npix = 6
    dust_lhs = np.array(
        [
            [1.0, 0.3, -0.2, 0.6, 0.1, -0.4],
            [0.4, -0.2, 0.5, 0.1, -0.3, 0.7],
        ]
    )
    sync_lhs = np.array(
        [
            [0.2, -0.4, 0.8, -0.1, 0.5, 0.3],
            [-0.5, 0.6, 0.1, -0.2, 0.4, -0.3],
        ]
    )
    dust_rhs = np.array(
        [
            [0.8, 0.2, -0.1, 0.5, 0.0, -0.2],
            [0.3, -0.1, 0.4, 0.2, -0.2, 0.5],
        ]
    )
    sync_rhs = np.array(
        [
            [0.1, -0.2, 0.7, 0.0, 0.3, 0.2],
            [-0.4, 0.5, 0.2, -0.1, 0.2, -0.2],
        ]
    )
    amplitudes_true = np.array([1.25, -0.55])
    target = amplitudes_true[0] * dust_rhs + amplitudes_true[1] * sync_rhs

    result = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=np.stack([dust_lhs, sync_lhs], axis=0),
        templates_rhs_qu=np.stack([dust_rhs, sync_rhs], axis=0),
        weight_map=np.ones(npix),
        template_names=("dust", "sync"),
    )

    np.testing.assert_allclose(result.amplitudes, amplitudes_true, atol=1e-12)
    np.testing.assert_allclose(
        result.processed_templates_rhs_qu,
        np.stack([dust_rhs, sync_rhs], axis=0),
        atol=1e-12,
    )


def test_weighted_template_gls_treats_mask_as_binary_support() -> None:
    """Convert any nonzero fit mask value into unit support inside the solver."""

    template = np.ones((1, 2, 3), dtype=np.float64)
    target = np.ones((2, 3), dtype=np.float64)
    mask = np.array([1.0, 0.5, 0.0])

    result = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=template,
        weight_map=np.ones(3),
        mask=mask,
    )

    np.testing.assert_allclose(result.normal_matrix, [[4.0]])
    np.testing.assert_allclose(result.rhs, [4.0])


def test_fit_foreground_templates_passes_mask_to_preprocessing_helpers(
    monkeypatch,
) -> None:
    """Use the fit mask in preprocessing only, not as a second GLS weight."""

    target = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    zero = np.zeros_like(target)
    mask = np.array([1.0, 0.9, 0.6, 0.4, 0.2, 0.0])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=zero,
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    target_masks: list[np.ndarray | None] = []
    template_masks: list[np.ndarray | None] = []
    gls_masks: list[np.ndarray | None] = []

    def fake_apply_harmonic_preprocessing(
        qu_map,
        fwhm_in,
        fwhm_out,
        *,
        beam_window_in,
        filter_config,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_in, fwhm_out, beam_window_in, filter_config, nest, harmonic_lmax
        target_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        return np.asarray(qu_map, dtype=np.float64)

    def fake_build_template_stack_with_lmax(
        template_inputs,
        *,
        fwhm_out,
        default_filter,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_out, default_filter, nest, harmonic_lmax
        template_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        templates = np.stack(
            [
                np.asarray(template_input.map_a_qu, dtype=np.float64)
                - np.asarray(template_input.map_b_qu, dtype=np.float64)
                for template_input in template_inputs
            ],
            axis=0,
        )
        template_names = tuple(
            template_input.name or f"template_{index}"
            for index, template_input in enumerate(template_inputs)
        )
        return templates, template_names

    original_weighted_template_gls = fit_mod.weighted_template_gls

    def fake_weighted_template_gls(
        target_qu,
        templates_qu,
        weight_map,
        *,
        templates_rhs_qu=None,
        templates_data_qu=None,
        mask=None,
        template_names=None,
    ):
        gls_masks.append(None if mask is None else np.asarray(mask, dtype=np.float64))
        return original_weighted_template_gls(
            target_qu=target_qu,
            templates_qu=templates_qu,
            templates_rhs_qu=templates_rhs_qu,
            templates_data_qu=templates_data_qu,
            weight_map=weight_map,
            mask=mask,
            template_names=template_names,
        )

    monkeypatch.setattr(
        fit_mod, "_apply_harmonic_preprocessing", fake_apply_harmonic_preprocessing
    )
    monkeypatch.setattr(
        fit_mod,
        "_build_template_stack_with_lmax",
        fake_build_template_stack_with_lmax,
    )
    monkeypatch.setattr(fit_mod, "weighted_template_gls", fake_weighted_template_gls)

    result = ftf.fit_foreground_templates(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        template_inputs_rhs=(template_input,),
        weight_map=np.ones(target.shape[1]),
        fwhm_out=0.0,
        mask=mask,
    )

    np.testing.assert_allclose(result.amplitudes, [1.0], atol=1e-12)
    assert len(target_masks) == 1
    np.testing.assert_allclose(target_masks[0], mask)
    assert len(template_masks) == 2
    np.testing.assert_allclose(template_masks[0], mask)
    np.testing.assert_allclose(template_masks[1], mask)
    assert gls_masks == [None]


def test_fit_foreground_templates_threads_target_beam_window(monkeypatch) -> None:
    """Pass a custom target beam window into target preprocessing."""

    target = np.tile(
        np.array(
            [
                [1.0, 0.5, -0.2, 0.3],
                [0.2, -0.3, 0.6, 1.2],
            ],
            dtype=np.float64,
        ),
        (1, 3),
    )
    beam_window = np.linspace(1.0, 1.3, 5)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=np.zeros_like(target),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )
    target_beams: list[np.ndarray | None] = []

    def fake_apply_harmonic_preprocessing(
        qu_map,
        fwhm_in,
        fwhm_out,
        *,
        beam_window_in,
        filter_config,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_in, fwhm_out, filter_config, mask, nest, harmonic_lmax
        target_beams.append(
            None if beam_window_in is None else np.asarray(beam_window_in)
        )
        return np.asarray(qu_map, dtype=np.float64)

    def fake_build_template_stack_with_lmax(
        template_inputs,
        *,
        fwhm_out,
        default_filter,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_out, default_filter, mask, nest, harmonic_lmax
        templates = np.stack(
            [
                np.asarray(template_input.map_a_qu, dtype=np.float64)
                - np.asarray(template_input.map_b_qu, dtype=np.float64)
                for template_input in template_inputs
            ],
            axis=0,
        )
        return templates, tuple(
            template_input.name for template_input in template_inputs
        )

    monkeypatch.setattr(
        fit_mod, "_apply_harmonic_preprocessing", fake_apply_harmonic_preprocessing
    )
    monkeypatch.setattr(
        fit_mod,
        "_build_template_stack_with_lmax",
        fake_build_template_stack_with_lmax,
    )

    result = ftf.fit_foreground_templates(
        target_qu=target,
        target_fwhm_in=10.0,
        template_inputs=(template_input,),
        weight_map=np.ones(target.shape[1]),
        fwhm_out=0.0,
        target_beam_window=beam_window,
    )

    np.testing.assert_allclose(result.amplitudes, [1.0], atol=1e-12)
    assert len(target_beams) == 1
    np.testing.assert_allclose(target_beams[0], beam_window)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_fit_foreground_templates_recovers_amplitude_with_custom_beams() -> None:
    """Recover an exact amplitude through real custom-beam preprocessing."""

    nside = 4
    npix = 12 * nside**2
    lmax = 5
    amplitude = 1.7
    rng = np.random.default_rng(283)
    template_map = rng.standard_normal((2, npix))
    beam_window = np.linspace(1.0, 0.8, lmax + 1)
    filter_config = ftf.HarmonicFilter(lmax=lmax, iter=5)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template_map,
        map_b_qu=np.zeros_like(template_map),
        fwhm_in_a=np.nan,
        fwhm_in_b=0.0,
        filter_config=filter_config,
        name="dust",
        beam_window_a=beam_window,
    )

    result = ftf.fit_foreground_templates(
        target_qu=amplitude * template_map,
        target_fwhm_in=np.nan,
        target_beam_window=beam_window,
        template_inputs=(template_input,),
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        target_filter=filter_config,
    )

    np.testing.assert_allclose(result.amplitudes, [amplitude], atol=1.0e-10)
    np.testing.assert_allclose(result.residual_qu, 0.0, atol=1.0e-10)


@pytest.mark.parametrize(
    "optional_group_name",
    ["template_inputs_rhs", "template_inputs_data"],
)
def test_preprocess_fit_inputs_shares_optional_group_lmax(
    monkeypatch,
    optional_group_name: str,
) -> None:
    """Propagate an optional group's explicit lmax to every fit operand."""

    qu_map = np.ones((2, 12), dtype=np.float64)
    lhs_input = ftf.DifferenceTemplateInput(
        map_a_qu=qu_map,
        map_b_qu=np.zeros_like(qu_map),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="lhs",
    )
    harmonic_input = ftf.DifferenceTemplateInput(
        map_a_qu=qu_map,
        map_b_qu=np.zeros_like(qu_map),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        filter_config=ftf.HarmonicFilter(lmax=2),
        name="optional",
    )

    seen_lmax: list[int | None] = []

    def fake_apply_harmonic_preprocessing(
        qu_map,
        fwhm_in,
        fwhm_out,
        *,
        harmonic_lmax,
        **kwargs,
    ):
        del fwhm_in, fwhm_out, kwargs
        seen_lmax.append(harmonic_lmax)
        return np.asarray(qu_map, dtype=np.float64)

    monkeypatch.setattr(
        fit_mod,
        "_apply_harmonic_preprocessing",
        fake_apply_harmonic_preprocessing,
    )
    monkeypatch.setattr(
        filters_mod,
        "_apply_harmonic_preprocessing",
        fake_apply_harmonic_preprocessing,
    )

    optional_groups: dict[
        str,
        tuple[ftf.DifferenceTemplateInput, ...] | None,
    ] = {
        "template_inputs_rhs": None,
        "template_inputs_data": None,
    }
    optional_groups[optional_group_name] = (harmonic_input,)
    fit_mod._preprocess_fit_inputs(
        target_qu=qu_map,
        target_fwhm_in=0.0,
        template_inputs=(lhs_input,),
        fwhm_out=0.0,
        target_beam_window=None,
        target_filter=None,
        mask=None,
        nest=False,
        **optional_groups,
    )

    assert seen_lmax == [2] * 5


def test_fit_foreground_templates_multi_mask_applies_binary_master_support() -> None:
    """Post-filter with binary support instead of applying apodization twice."""

    template = np.array(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.5, -0.3, 0.7, -0.2],
        ],
        dtype=np.float64,
    )
    target = 2.5 * template
    master_mask = np.array([1.0, 0.25, 0.0, np.nan])
    support = np.array([1.0, 1.0, 0.0, 0.0])
    support_qu = np.vstack([support, support])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps={
            "m1": np.ones(template.shape[1]),
            "m2": np.array([1.0, 3.0, 1.0, 1.0]),
        },
        fwhm_out=0.0,
        master_mask=master_mask,
    )

    assert result.fit_names == ("m1", "m2")
    np.testing.assert_allclose(result.processed_target_qu, target * support_qu)
    np.testing.assert_allclose(
        result.processed_templates_qu[0],
        template * support_qu,
    )
    np.testing.assert_allclose(
        result.processed_templates_rhs_qu[0],
        template * support_qu,
    )
    np.testing.assert_allclose(
        result.processed_templates_data_qu[0],
        template * support_qu,
    )
    np.testing.assert_allclose(result.fit_results["m1"].amplitudes, [2.5])
    np.testing.assert_allclose(result.fit_results["m2"].amplitudes, [2.5])


def test_multi_mask_exact_difference_template_recovers_same_amplitude() -> None:
    """Recover the same amplitude on every mask for an exact 353-217 template."""

    planck_353 = np.array(
        [
            [5.0, -2.0, 1.5, 0.8, -1.2, 2.4, 3.1, -0.6],
            [1.0, 3.0, -2.5, 0.4, 1.7, -1.1, 0.9, 2.2],
        ],
        dtype=np.float64,
    )
    planck_217 = np.array(
        [
            [1.0, -0.5, 0.2, 0.1, -0.4, 0.8, 1.0, -0.2],
            [0.2, 0.7, -0.4, 0.1, 0.5, -0.3, 0.2, 0.6],
        ],
        dtype=np.float64,
    )
    amplitude_true = 0.037
    target = amplitude_true * (planck_353 - planck_217)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=planck_353,
        map_b_qu=planck_217,
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps={
            "low": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
            "high": np.array([0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0]),
            "master": np.ones(planck_353.shape[1]),
        },
        fwhm_out=0.0,
        master_mask=np.ones(planck_353.shape[1]),
    )

    for fit_name in result.fit_names:
        np.testing.assert_allclose(
            result.fit_results[fit_name].amplitudes,
            [amplitude_true],
            rtol=0.0,
            atol=1e-14,
        )


def test_fit_foreground_templates_multi_mask_accepts_explicit_master_support() -> None:
    """Let callers override threshold-derived post-filter support."""

    template = np.array(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.5, -0.3, 0.7, -0.2],
        ],
        dtype=np.float64,
    )
    target = 1.75 * template
    support = np.array([0.0, 1.0, 0.0, 1.0])
    support_qu = np.vstack([support, support])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps={"m1": np.ones(template.shape[1])},
        fwhm_out=0.0,
        master_mask=np.ones(template.shape[1]),
        master_support_mask=support,
        master_support_threshold=2.0,
    )

    np.testing.assert_allclose(result.processed_target_qu, target * support_qu)
    np.testing.assert_allclose(result.fit_results["m1"].amplitudes, [1.75])


def test_fit_foreground_templates_multi_mask_validates_named_masks() -> None:
    """Reject empty or shape-incompatible multi-mask inputs early."""

    npix = 4
    template = np.ones((2, npix), dtype=np.float64)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    with pytest.raises(ValueError, match="weight_maps"):
        ftf.fit_foreground_templates_multi_mask(
            target_qu=template,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_maps={},
            fwhm_out=0.0,
            master_mask=np.ones(npix),
        )

    with pytest.raises(ValueError, match="weight_maps"):
        ftf.fit_foreground_templates_multi_mask(
            target_qu=template,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_maps={"m1": np.ones(npix + 1)},
            fwhm_out=0.0,
            master_mask=np.ones(npix),
        )

    with pytest.raises(ValueError, match="master_support_mask"):
        ftf.fit_foreground_templates_multi_mask(
            target_qu=template,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_maps={"m1": np.ones(npix)},
            fwhm_out=0.0,
            master_mask=np.ones(npix),
            master_support_mask=np.ones(npix + 1),
        )


def test_fit_foreground_templates_multi_mask_uses_master_mask_for_preprocessing(
    monkeypatch,
) -> None:
    """Keep fitting weights out of the harmonic preprocessing path."""

    target = np.array(
        [
            [1.0, 0.5, -0.2, 0.3],
            [0.2, -0.3, 0.6, 1.2],
        ],
        dtype=np.float64,
    )
    zero = np.zeros_like(target)
    master_mask = np.array([1.0, 0.9, 0.6, 0.4])
    weight_m1 = np.array([1.0, 0.0, 1.0, 0.0])
    weight_m2 = np.array([0.0, 1.0, 0.0, 1.0])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=zero,
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    target_masks: list[np.ndarray | None] = []
    template_masks: list[np.ndarray | None] = []
    gls_masks: list[np.ndarray | None] = []
    gls_weights: list[np.ndarray] = []

    def fake_apply_harmonic_preprocessing(
        qu_map,
        fwhm_in,
        fwhm_out,
        *,
        beam_window_in,
        filter_config,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_in, fwhm_out, beam_window_in, filter_config, nest, harmonic_lmax
        target_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        return np.asarray(qu_map, dtype=np.float64)

    def fake_build_template_stack_with_lmax(
        template_inputs,
        *,
        fwhm_out,
        default_filter,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_out, default_filter, nest, harmonic_lmax
        template_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        templates = np.stack(
            [
                np.asarray(template_input.map_a_qu, dtype=np.float64)
                - np.asarray(template_input.map_b_qu, dtype=np.float64)
                for template_input in template_inputs
            ],
            axis=0,
        )
        template_names = tuple(
            template_input.name or f"template_{index}"
            for index, template_input in enumerate(template_inputs)
        )
        return templates, template_names

    original_weighted_template_gls = fit_mod.weighted_template_gls

    def fake_weighted_template_gls(
        target_qu,
        templates_qu,
        weight_map,
        *,
        templates_rhs_qu=None,
        templates_data_qu=None,
        mask=None,
        template_names=None,
    ):
        gls_masks.append(None if mask is None else np.asarray(mask, dtype=np.float64))
        gls_weights.append(np.asarray(weight_map, dtype=np.float64))
        return original_weighted_template_gls(
            target_qu=target_qu,
            templates_qu=templates_qu,
            templates_rhs_qu=templates_rhs_qu,
            templates_data_qu=templates_data_qu,
            weight_map=weight_map,
            mask=mask,
            template_names=template_names,
        )

    monkeypatch.setattr(
        fit_mod, "_apply_harmonic_preprocessing", fake_apply_harmonic_preprocessing
    )
    monkeypatch.setattr(
        fit_mod,
        "_build_template_stack_with_lmax",
        fake_build_template_stack_with_lmax,
    )
    monkeypatch.setattr(fit_mod, "weighted_template_gls", fake_weighted_template_gls)

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        template_inputs_rhs=(template_input,),
        weight_maps={"m1": weight_m1, "m2": weight_m2},
        fwhm_out=0.0,
        master_mask=master_mask,
    )

    expected_master = np.vstack([master_mask, master_mask])
    assert result.fit_names == ("m1", "m2")
    assert len(target_masks) == 1
    np.testing.assert_allclose(target_masks[0], expected_master)
    assert len(template_masks) == 2
    np.testing.assert_allclose(template_masks[0], expected_master)
    np.testing.assert_allclose(template_masks[1], expected_master)
    assert gls_masks == [None, None]
    np.testing.assert_allclose(gls_weights[0], np.vstack([weight_m1, weight_m1]))
    np.testing.assert_allclose(gls_weights[1], np.vstack([weight_m2, weight_m2]))


def test_fit_foreground_templates_multi_mask_threads_target_beam_window(
    monkeypatch,
) -> None:
    """Pass a custom target beam through shared multi-mask preprocessing."""

    target = np.tile(
        np.array(
            [
                [1.0, 0.5, -0.2, 0.3],
                [0.2, -0.3, 0.6, 1.2],
            ],
            dtype=np.float64,
        ),
        (1, 3),
    )
    beam_window = np.linspace(1.0, 1.4, 5)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=np.zeros_like(target),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )
    target_beams: list[np.ndarray | None] = []

    def fake_apply_harmonic_preprocessing(
        qu_map,
        fwhm_in,
        fwhm_out,
        *,
        beam_window_in,
        filter_config,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_in, fwhm_out, filter_config, mask, nest, harmonic_lmax
        target_beams.append(
            None if beam_window_in is None else np.asarray(beam_window_in)
        )
        return np.asarray(qu_map, dtype=np.float64)

    def fake_build_template_stack_with_lmax(
        template_inputs,
        *,
        fwhm_out,
        default_filter,
        mask,
        nest,
        harmonic_lmax,
    ):
        del fwhm_out, default_filter, mask, nest, harmonic_lmax
        templates = np.stack(
            [
                np.asarray(template_input.map_a_qu, dtype=np.float64)
                - np.asarray(template_input.map_b_qu, dtype=np.float64)
                for template_input in template_inputs
            ],
            axis=0,
        )
        return templates, tuple(
            template_input.name for template_input in template_inputs
        )

    monkeypatch.setattr(
        fit_mod, "_apply_harmonic_preprocessing", fake_apply_harmonic_preprocessing
    )
    monkeypatch.setattr(
        fit_mod,
        "_build_template_stack_with_lmax",
        fake_build_template_stack_with_lmax,
    )

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=10.0,
        template_inputs=(template_input,),
        weight_maps={"m1": np.ones(target.shape[1])},
        fwhm_out=0.0,
        master_mask=np.ones(target.shape[1]),
        target_beam_window=beam_window,
    )

    np.testing.assert_allclose(result.fit_results["m1"].amplitudes, [1.0], atol=1e-12)
    assert len(target_beams) == 1
    np.testing.assert_allclose(target_beams[0], beam_window)


def test_weighted_template_gls_supports_data_projection_vector() -> None:
    """Use a third stack ``d_3`` only in the right-hand vector ``d_3^T W m``."""

    npix = 6
    dust_lhs = np.array(
        [
            [1.0, 0.3, -0.2, 0.6, 0.1, -0.4],
            [0.4, -0.2, 0.5, 0.1, -0.3, 0.7],
        ]
    )
    sync_lhs = np.array(
        [
            [0.2, -0.4, 0.8, -0.1, 0.5, 0.3],
            [-0.5, 0.6, 0.1, -0.2, 0.4, -0.3],
        ]
    )
    dust_rhs = np.array(
        [
            [0.8, 0.2, -0.1, 0.5, 0.0, -0.2],
            [0.3, -0.1, 0.4, 0.2, -0.2, 0.5],
        ]
    )
    sync_rhs = np.array(
        [
            [0.1, -0.2, 0.7, 0.0, 0.3, 0.2],
            [-0.4, 0.5, 0.2, -0.1, 0.2, -0.2],
        ]
    )
    dust_data = np.array(
        [
            [0.6, -0.3, 0.2, 0.4, -0.5, 0.1],
            [0.2, 0.5, -0.4, 0.3, 0.1, -0.6],
        ]
    )
    sync_data = np.array(
        [
            [-0.1, 0.4, 0.3, -0.2, 0.6, -0.5],
            [0.5, -0.3, 0.1, 0.2, -0.4, 0.3],
        ]
    )
    weight = np.array([1.0, 3.0, 0.5, 2.0, 4.0, 1.5])
    target = np.array(
        [
            [0.7, -0.4, 0.2, 0.9, -0.1, 0.5],
            [0.3, 0.6, -0.5, 0.1, 0.4, -0.2],
        ]
    )

    lhs = np.stack([dust_lhs, sync_lhs], axis=0)
    rhs = np.stack([dust_rhs, sync_rhs], axis=0)
    data = np.stack([dust_data, sync_data], axis=0)

    result = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=lhs,
        templates_rhs_qu=rhs,
        templates_data_qu=data,
        weight_map=weight,
        template_names=("dust", "sync"),
    )

    # Reproduce the estimator (d_1^T W d_2)^-1 d_3^T W m independently. The
    # weight map is broadcast across Q/U exactly as as_weight_map does.
    weight_qu = np.vstack([weight, weight])
    normal_matrix = np.array(
        [[np.sum(lhs[i] * weight_qu * rhs[j]) for j in range(2)] for i in range(2)]
    )
    rhs_vector = np.array([np.sum(data[i] * weight_qu * target) for i in range(2)])
    expected = np.linalg.solve(normal_matrix, rhs_vector)

    np.testing.assert_allclose(result.amplitudes, expected, atol=1e-12)
    np.testing.assert_allclose(result.normal_matrix, normal_matrix, atol=1e-12)
    np.testing.assert_allclose(result.rhs, rhs_vector, atol=1e-12)
    np.testing.assert_allclose(result.processed_templates_data_qu, data, atol=1e-12)


def test_weighted_template_gls_data_vector_defaults_to_left_stack() -> None:
    """Omitting ``templates_data_qu`` reuses the left stack for the vector."""

    npix = 6
    dust_lhs = np.array(
        [
            [1.0, 0.3, -0.2, 0.6, 0.1, -0.4],
            [0.4, -0.2, 0.5, 0.1, -0.3, 0.7],
        ]
    )
    sync_lhs = np.array(
        [
            [0.2, -0.4, 0.8, -0.1, 0.5, 0.3],
            [-0.5, 0.6, 0.1, -0.2, 0.4, -0.3],
        ]
    )
    dust_rhs = np.array(
        [
            [0.8, 0.2, -0.1, 0.5, 0.0, -0.2],
            [0.3, -0.1, 0.4, 0.2, -0.2, 0.5],
        ]
    )
    sync_rhs = np.array(
        [
            [0.1, -0.2, 0.7, 0.0, 0.3, 0.2],
            [-0.4, 0.5, 0.2, -0.1, 0.2, -0.2],
        ]
    )
    weight = np.linspace(0.5, 2.5, npix)
    target = np.array(
        [
            [0.7, -0.4, 0.2, 0.9, -0.1, 0.5],
            [0.3, 0.6, -0.5, 0.1, 0.4, -0.2],
        ]
    )
    lhs = np.stack([dust_lhs, sync_lhs], axis=0)
    rhs = np.stack([dust_rhs, sync_rhs], axis=0)

    default = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=lhs,
        templates_rhs_qu=rhs,
        weight_map=weight,
    )
    explicit = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=lhs,
        templates_rhs_qu=rhs,
        templates_data_qu=lhs,
        weight_map=weight,
    )

    np.testing.assert_allclose(default.amplitudes, explicit.amplitudes, atol=1e-12)
    np.testing.assert_allclose(default.rhs, explicit.rhs, atol=1e-12)
    np.testing.assert_allclose(default.processed_templates_data_qu, lhs, atol=1e-12)


def test_fit_foreground_templates_threads_data_projection_templates() -> None:
    """Build ``d_3`` from its own template inputs for the right-hand vector."""

    npix = 6
    weight = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.8])
    d1_a = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    d1_b = np.zeros_like(d1_a)
    d2_a = np.array(
        [
            [0.9, 0.4, -0.1, 0.4, 0.7, -0.3],
            [0.3, -0.2, 0.5, 1.0, -0.4, 0.2],
        ]
    )
    d2_b = np.zeros_like(d2_a)
    d3_a = np.array(
        [
            [0.6, -0.3, 0.2, 0.4, -0.5, 0.1],
            [0.2, 0.5, -0.4, 0.3, 0.1, -0.6],
        ]
    )
    d3_b = np.zeros_like(d3_a)
    target = np.array(
        [
            [0.7, -0.4, 0.2, 0.9, -0.1, 0.5],
            [0.3, 0.6, -0.5, 0.1, 0.4, -0.2],
        ]
    )

    def _input(map_a, map_b):
        return ftf.DifferenceTemplateInput(
            map_a_qu=map_a,
            map_b_qu=map_b,
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            name="dust",
        )

    result = ftf.fit_foreground_templates(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(_input(d1_a, d1_b),),
        template_inputs_rhs=(_input(d2_a, d2_b),),
        template_inputs_data=(_input(d3_a, d3_b),),
        weight_map=weight,
        fwhm_out=0.0,
    )

    weight_qu = np.vstack([weight, weight])
    d1, d2, d3 = d1_a - d1_b, d2_a - d2_b, d3_a - d3_b
    expected = np.sum(d3 * weight_qu * target) / np.sum(d1 * weight_qu * d2)

    np.testing.assert_allclose(result.amplitudes, [expected], atol=1e-12)
    np.testing.assert_allclose(result.processed_templates_data_qu[0], d3, atol=1e-12)


def test_fit_foreground_templates_multi_mask_threads_data_projection_templates() -> (
    None
):
    """Apply the third stack in every per-mask solve of the multi-mask fit."""

    npix = 6
    d1_a = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    d2_a = np.array(
        [
            [0.9, 0.4, -0.1, 0.4, 0.7, -0.3],
            [0.3, -0.2, 0.5, 1.0, -0.4, 0.2],
        ]
    )
    d3_a = np.array(
        [
            [0.6, -0.3, 0.2, 0.4, -0.5, 0.1],
            [0.2, 0.5, -0.4, 0.3, 0.1, -0.6],
        ]
    )
    target = np.array(
        [
            [0.7, -0.4, 0.2, 0.9, -0.1, 0.5],
            [0.3, 0.6, -0.5, 0.1, 0.4, -0.2],
        ]
    )

    def _input(map_a):
        return ftf.DifferenceTemplateInput(
            map_a_qu=map_a,
            map_b_qu=np.zeros_like(map_a),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            name="dust",
        )

    weight_maps = {
        "low": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.5]),
        "high": np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
    }

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(_input(d1_a),),
        template_inputs_rhs=(_input(d2_a),),
        template_inputs_data=(_input(d3_a),),
        weight_maps=weight_maps,
        fwhm_out=0.0,
        master_mask=np.ones(npix),
    )

    for fit_name, weight in weight_maps.items():
        weight_qu = np.vstack([weight, weight])
        expected = np.sum(d3_a * weight_qu * target) / np.sum(d1_a * weight_qu * d2_a)
        np.testing.assert_allclose(
            result.fit_results[fit_name].amplitudes, [expected], atol=1e-12
        )
    np.testing.assert_allclose(result.processed_templates_data_qu[0], d3_a, atol=1e-12)
