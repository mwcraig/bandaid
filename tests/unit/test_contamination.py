"""Unit tests for ``min_separation_fwhm`` and neighbor-contamination flags."""

import numpy as np
import pytest
from _helpers import SEED
from astropy.coordinates import SkyCoord

from bandaid.config import ApertureConfig, PhotometryConfig
from bandaid.photometry import (
    CONTAMINATION_TOLERANCE,
    MOFFAT_BETA,
    min_separation_fwhm,
    neighbor_contamination_flag,
    neighbor_contamination_flag_sky,
)


def test_min_separation_fwhm():
    """Check a few extreme cases for a reasonable minimum separation between sources."""
    tenk_flux_ratio = 10
    # first check for a target with a much, much dimmer companion.
    # In that case the minimum separation should be roughly zero.
    assert min_separation_fwhm(-tenk_flux_ratio, tolerance=0.01) == pytest.approx(0)

    # Now assume the neighbor is much brighter than the target. Then the minimum
    # separation should be large.
    assert min_separation_fwhm(tenk_flux_ratio, tolerance=0.01) == pytest.approx(
        11.558,
        rel=0.01,
    )

    # Now a case where the neighbor is the same brightness as the target.
    assert min_separation_fwhm(0, tolerance=0.01) == pytest.approx(2.299, rel=0.01)


def _min_sep_fwhm_general(delta_mag, r_ap_fwhm, tolerance, beta):
    """
    Independently re-derive the minimum separation for a general aperture radius.

    Models the neighbor (total flux ``F_n``) as a Moffat profile of index
    ``beta`` and approximates its intensity as constant across the target
    aperture of radius ``R = r_ap_fwhm * FWHM`` (the same approximation the
    shipped model uses), then solves ``f_spill = tolerance * E * F_target``
    for the separation, where ``E = 1 - (1 + (R/alpha)^2)^(1 - beta)`` is the
    fraction of the target's Moffat flux the aperture encloses (~0.76 at
    R = 1 FWHM for beta = 3): the tolerance bounds contamination of the
    *measured* (aperture-enclosed) target flux, not its total flux. Reference
    implementation from https://github.com/mwcraig/bandaid/issues/53,
    including point 3 of its proposed fix (the enclosed-flux normalization).

    Parameters
    ----------
    delta_mag : float or array-like
        How many magnitudes brighter the neighbor is than the target.
    r_ap_fwhm : float
        Aperture radius in units of the FWHM.
    tolerance : float
        Maximum tolerated fractional flux contamination.
    beta : float
        Moffat wing index.

    Returns
    -------
    ndarray
        Required separation in units of FWHM.
    """
    a = 4.0 * (2.0 ** (1.0 / beta) - 1.0)
    r_alpha_sq = r_ap_fwhm**2 * a
    enclosed = 1.0 - (1.0 + r_alpha_sq) ** (1.0 - beta)
    flux_ratio = 10.0 ** (0.4 * np.asarray(delta_mag))
    rhs = ((beta - 1.0) * flux_ratio * r_alpha_sq / (tolerance * enclosed)) ** (
        1.0 / beta
    )
    return np.sqrt(np.maximum((rhs - 1.0) / a, 0.0))


def test_min_separation_fwhm_matches_general_radius_derivation():
    """
    ``aperture_radius_fwhm`` reproduces the general-radius Moffat derivation.

    The default must stay the historical ``R = 1 FWHM`` special case, and any
    other radius must match the independent re-derivation: spillover into an
    aperture scales with its area, so the required separation grows with the
    radius. Regression test for
    https://github.com/mwcraig/bandaid/issues/53.
    """
    dm = np.array([0.0, 2.5, 5.0, 10.0])
    # The default is the historical R = 1 FWHM special case.
    np.testing.assert_allclose(
        min_separation_fwhm(dm),
        _min_sep_fwhm_general(dm, 1.0, CONTAMINATION_TOLERANCE, MOFFAT_BETA),
    )
    # Any other radius matches the general derivation, not the R = 1 case.
    for r_ap in (0.5, 2.0):
        np.testing.assert_allclose(
            min_separation_fwhm(dm, aperture_radius_fwhm=r_ap),
            _min_sep_fwhm_general(dm, r_ap, CONTAMINATION_TOLERANCE, MOFFAT_BETA),
        )
    # A bigger aperture sweeps up more of the neighbor -> larger separation.
    assert min_separation_fwhm(5.0, aperture_radius_fwhm=2.0) > min_separation_fwhm(5.0)


def test_min_separation_fwhm_tuning_params_are_keyword_only():
    """``tolerance``/``beta``/``aperture_radius_fwhm`` cannot be passed positionally."""
    with pytest.raises(TypeError):
        min_separation_fwhm(0.0, CONTAMINATION_TOLERANCE)


class TestNeighborContaminationFlag:
    """Unit tests for the bright-neighbor flag ``neighbor_contamination_flag``."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_stars_never_flagged(self, n):
        """With <2 stars no pair can exist, so the early return is all-False."""
        coords = np.zeros((n, 2))
        mags = np.zeros(n)
        flag = neighbor_contamination_flag(coords, mags, fwhm=2.0)
        assert flag.shape == (n,)
        assert flag.dtype == bool
        assert not flag.any()

    def test_equal_brightness_pair_flagged_inside_threshold(self):
        """Equal-mag neighbors flag both stars inside ~2.30 FWHM, neither outside."""
        fwhm = 2.0
        # min_separation_fwhm(0) ~ 2.30 FWHM -> ~4.60 px at fwhm=2.
        threshold_px = min_separation_fwhm(0.0) * fwhm

        close = np.array([[0.0, 0.0], [threshold_px - 0.5, 0.0]])
        far = np.array([[0.0, 0.0], [threshold_px + 0.5, 0.0]])
        mags = np.array([12.0, 12.0])

        np.testing.assert_array_equal(
            neighbor_contamination_flag(close, mags, fwhm=fwhm),
            [True, True],
        )
        assert not neighbor_contamination_flag(far, mags, fwhm=fwhm).any()

    def test_flag_is_asymmetric_for_unequal_brightness(self):
        """A faint star is flagged by a bright neighbor that it does not flag back."""
        fwhm = 2.0
        # delta_mag=6 needs ~6.2 FWHM (~12.4 px); delta_mag=-6 needs 0. At 6 px the
        # faint star (bright neighbor) is flagged; the bright star is not.
        mags = np.array([8.0, 14.0])  # star 0 bright, star 1 faint
        coords = np.array([[0.0, 0.0], [6.0, 0.0]])

        flag = neighbor_contamination_flag(coords, mags, fwhm=fwhm)
        assert not flag[0]  # bright star: faint neighbor spills negligibly
        assert flag[1]  # faint star: bright neighbor contaminates it

    def test_non_finite_magnitude_contributes_no_contamination(self):
        """A NaN-magnitude star neither flags nor is flagged."""
        fwhm = 2.0
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])  # essentially on top of one another
        mags = np.array([10.0, np.nan])
        flag = neighbor_contamination_flag(coords, mags, fwhm=fwhm)
        assert not flag.any()

    def test_tuning_params_are_keyword_only(self):
        """The tuning params cannot be passed positionally (issue #61)."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        mags = np.array([10.0, 10.0])
        with pytest.raises(TypeError):
            neighbor_contamination_flag(coords, mags, 2.0, CONTAMINATION_TOLERANCE)

    def test_aperture_radius_widens_the_flagged_region(self):
        """
        A pair clean for the default 1-FWHM aperture is flagged at 2 FWHM.

        The pair sits between the equal-magnitude thresholds for the two radii,
        so it is clean at the default ``aperture_radius_fwhm=1.0`` but flagged
        once the aperture (into which the neighbor spills) doubles. Regression
        test for https://github.com/mwcraig/bandaid/issues/53.
        """
        fwhm = 2.0
        r_ap = 2.0
        sep_r1 = min_separation_fwhm(0.0) * fwhm
        sep_r2 = min_separation_fwhm(0.0, aperture_radius_fwhm=r_ap) * fwhm
        coords = np.array([[0.0, 0.0], [0.5 * (sep_r1 + sep_r2), 0.0]])
        mags = np.array([12.0, 12.0])

        assert not neighbor_contamination_flag(coords, mags, fwhm=fwhm).any()
        np.testing.assert_array_equal(
            neighbor_contamination_flag(
                coords, mags, fwhm=fwhm, aperture_radius_fwhm=r_ap
            ),
            [True, True],
        )


class TestNeighborContaminationFlagSky:
    """Unit tests for the sky-space flag ``neighbor_contamination_flag_sky``."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_stars_never_flagged(self, n):
        """With <2 stars no pair can exist, so the early return is all-False."""
        radecs = np.zeros((n, 2))
        mags = np.zeros(n)
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=2.0)
        assert flag.shape == (n,)
        assert flag.dtype == bool
        assert not flag.any()

    def test_non_finite_magnitude_contributes_no_contamination(self):
        """A NaN-magnitude star neither flags nor is flagged."""
        # The two stars are ~0.5 arcsec apart -- essentially on top of one another.
        radecs = np.array([[10.0, 0.0], [10.0 + 0.5 / 3600.0, 0.0]])
        mags = np.array([10.0, np.nan])
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=2.0)
        assert not flag.any()

    def test_tuning_params_are_keyword_only(self):
        """The tuning params cannot be passed positionally (PR #70 review, #61)."""
        radecs = np.array([[10.0, 0.0], [10.0 + 0.5 / 3600.0, 0.0]])
        mags = np.array([10.0, 10.0])
        with pytest.raises(TypeError):
            neighbor_contamination_flag_sky(radecs, mags, 2.0, CONTAMINATION_TOLERANCE)

    def test_matches_pixel_front_end_on_equator(self):
        """
        Sky and pixel front ends agree star-for-star.

        Stars are placed along the celestial equator, where the great-circle
        separation between ``(ra, 0)`` points is exactly the RA difference. The
        equivalent pixel layout is those angular separations divided by the plate
        scale, and ``fwhm_arcsec == fwhm_pix * pixscale``; the two front ends scale
        identically, so they must flag the same stars.
        """
        pixscale = 2.4  # arcsec / pixel
        fwhm_pix = 2.0
        fwhm_arcsec = fwhm_pix * pixscale

        ra0 = 10.0
        offsets_arcsec = np.array([0.0, 3.0, 7.0, 30.0])
        ras = ra0 + offsets_arcsec / 3600.0
        decs = np.zeros_like(ras)
        radecs = np.column_stack([ras, decs])
        mags = np.array([12.0, 12.0, 9.0, 13.0])

        coords_pix = np.column_stack(
            [offsets_arcsec / pixscale, np.zeros_like(ras)],
        )

        sky_flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)
        pix_flag = neighbor_contamination_flag(coords_pix, mags, fwhm_pix)
        np.testing.assert_array_equal(sky_flag, pix_flag)
        # Guard: the case is non-trivial -- some flagged, some not.
        assert sky_flag.any()
        assert not sky_flag.all()

    def test_all_non_finite_magnitudes_never_flagged(self):
        """With no finite magnitude there is no valid pair, so nothing is flagged."""
        radecs = np.array([[10.0, 0.0], [10.0 + 0.5 / 3600.0, 0.0]])
        mags = np.array([np.nan, np.nan])
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=2.0)
        assert flag.shape == (2,)
        assert not flag.any()

    def test_zero_fwhm_never_flags(self):
        """A zero FWHM makes every required separation zero, so nothing is flagged."""
        radecs = np.array([[10.0, 0.0], [10.0 + 0.5 / 3600.0, 0.0]])
        mags = np.array([10.0, 12.0])
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=0.0)
        assert not flag.any()

    def test_matches_dense_all_pairs_reference_on_random_field(self):
        """
        The sky flag reproduces a brute-force all-pairs reference.

        The reference applies the documented contamination rule directly to an
        explicit N x N great-circle separation matrix: target i is flagged when
        any other star j with finite magnitudes sits closer than
        ``min_separation_fwhm(mag_i - mag_j) * fwhm``. This pins the sky front
        end (whatever its internal pair search) to the dense model on a
        realistic random field, including NaN magnitudes and an exact-duplicate
        position (a zero-separation pair, which the rule flags).
        """
        # A dense random field in a small RA/Dec patch, with some NaN
        # magnitudes and one exact-duplicate position to exercise edge cases.
        rng = np.random.default_rng(SEED)
        n = 800
        radecs = np.column_stack(
            [rng.uniform(10.0, 10.5, n), rng.uniform(19.75, 20.25, n)],
        )
        mags = rng.uniform(8.0, 18.0, n)
        mags[rng.choice(n, 20, replace=False)] = np.nan
        radecs[5] = radecs[4]
        fwhm_arcsec = 5.0

        # Brute-force reference: full N x N great-circle separations.
        coords = SkyCoord(radecs[:, 0], radecs[:, 1], unit="deg")
        sep_arcsec = coords[:, None].separation(coords[None, :]).arcsec
        # Minimum allowed separation for each pair from the magnitude difference.
        required = min_separation_fwhm(mags[:, None] - mags[None, :]) * fwhm_arcsec
        # Only pairs with two finite magnitudes count, and never compare a star
        # with itself (the diagonal).
        finite = np.isfinite(mags)
        valid = finite[:, None] & finite[None, :]
        np.fill_diagonal(valid, val=False)
        # A target is flagged if any valid neighbor is closer than required.
        expected = (valid & (sep_arcsec < required)).any(axis=1)

        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)

        np.testing.assert_array_equal(flag, expected)
        # Guard: the case is non-trivial -- some flagged, some not.
        assert expected.any()
        assert not expected.all()

    def test_target_mask_restricts_which_stars_can_be_flagged(self):
        """
        ``target_mask`` makes flagging asymmetric: only masked-in stars are victims.

        A bright star B (mag 10) and a faint star F (mag 13) sit ~3 arcsec apart --
        close enough that, symmetrically, each is flagged by the other (B because F
        spills into its aperture, F because the much brighter B spills into its,
        which needs a far larger separation). A third star sits well away.

        With ``target_mask`` selecting only B, B is still flagged (F contaminates
        it) but F is *not* -- F serves purely as a contaminator, never a victim.
        Without the mask the result is the symmetric default, flagging both.
        """
        ra0 = 10.0
        offsets_arcsec = np.array([0.0, 3.0, 50.0])
        ras = ra0 + offsets_arcsec / 3600.0
        radecs = np.column_stack([ras, np.zeros_like(ras)])
        mags = np.array([10.0, 13.0, 12.0])
        fwhm_arcsec = 5.0

        symmetric = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)
        np.testing.assert_array_equal(symmetric, [True, True, False])

        target_mask = np.array([True, False, False])
        asymmetric = neighbor_contamination_flag_sky(
            radecs, mags, fwhm_arcsec, target_mask=target_mask
        )
        np.testing.assert_array_equal(asymmetric, [True, False, False])

    def test_flagging_scales_with_the_configured_aperture_radius(self):
        """
        A pair contaminated for a configured 2-FWHM aperture is flagged as such.

        Adapted from the repro in https://github.com/mwcraig/bandaid/issues/53:
        the pair sits between the required separations for a 1-FWHM and a
        2-FWHM aperture, so it is clean at the default radius but must be
        flagged when the configured ``max(config.apertures.radii)`` is passed
        through ``aperture_radius_fwhm``.
        """
        config = PhotometryConfig(apertures=ApertureConfig(radii=(2.0,)))
        r_ap = max(config.apertures.radii)
        tol = config.instrument.contamination_tolerance
        beta = config.instrument.moffat_beta
        dm = 5.0  # neighbor 5 mag brighter than the target

        sep_r1 = float(min_separation_fwhm(dm, tolerance=tol, beta=beta))
        sep_r2 = float(
            min_separation_fwhm(dm, tolerance=tol, beta=beta, aperture_radius_fwhm=r_ap)
        )
        assert sep_r2 > sep_r1

        # Put the pair between the two thresholds: clean for R_ap = 1 FWHM,
        # contaminated for the configured R_ap = 2 FWHM.
        d_fwhm = 0.5 * (sep_r1 + sep_r2)
        fwhm_arcsec = 8.0
        radecs = np.array([[10.0, 0.0], [10.0 + d_fwhm * fwhm_arcsec / 3600.0, 0.0]])
        mags = np.array([15.0, 15.0 - dm])

        clean = neighbor_contamination_flag_sky(
            radecs, mags, fwhm_arcsec, tolerance=tol, beta=beta
        )
        assert not clean[0]

        flagged = neighbor_contamination_flag_sky(
            radecs,
            mags,
            fwhm_arcsec,
            tolerance=tol,
            beta=beta,
            aperture_radius_fwhm=r_ap,
        )
        assert flagged[0]
