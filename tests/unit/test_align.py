"""Unit tests for ``align`` and WCS solve/validation."""

import astropy.units as u
import numpy as np
import pytest
from _helpers import SEESTAR_PIXSCALE, _make_tan_wcs, align_coords
from astropy.coordinates import SkyCoord

from bandaid.exceptions import (
    WCSPointingError,
    WCSScaleError,
    WCSSolveError,
)
from bandaid.instruments import load_instrument
from bandaid.photometry import (
    N_GAIA_STARS_ALIGN,
    N_GAIA_STARS_ALIGN_RETRY,
    N_IMAGE_STARS_ALIGN,
    WCS_MATCH_TOLERANCE,
    _solve_pool_near,
    align,
)

# The Seestar50 profile's scale tolerance; the class default is looser.
TIGHT_SCALE_TOLERANCE = 0.005


class TestAlign:
    """Unit tests for the WCS-solve/projection helper ``align``."""

    def test_projects_photometry_coords_through_supplied_wcs(self):
        """photometry_coords are projected to pixels via the provided WCS."""
        wcs = _make_tan_wcs(crval=(10.0, 20.0))
        sky = SkyCoord(ra=[10.0, 10.01] * u.deg, dec=[20.0, 20.01] * u.deg)
        coords = np.array([[250.0, 250.0], [260.0, 260.0]])

        aligned, returned_wcs, _ = align(
            coords, radecs=None, photometry_coords=sky, wcs=wcs
        )

        assert returned_wcs is wcs
        expected = np.array(wcs.world_to_pixel(sky)).T
        np.testing.assert_allclose(aligned, expected)
        assert aligned.shape == (2, 2)

    def test_solves_wcs_from_detections_when_none_supplied(self, mocker):
        """
        With wcs=None, align slices image and Gaia coords *independently*.

        Detections are capped at N_IMAGE_STARS_ALIGN and Gaia references at
        N_GAIA_STARS_ALIGN -- the two counts are decoupled so the matcher can be
        fed more references than detections. The constants are patched to
        distinct values here to prove the slices are independent rather than a
        single shared cap. compute_wcs (twirl's slow, stochastic asterism solver)
        is stubbed with a sentinel WCS; the unit under test is align's slicing,
        not twirl's matching.
        """
        n_image = 4
        n_gaia = 7
        mocker.patch("bandaid.photometry.N_IMAGE_STARS_ALIGN", n_image)
        mocker.patch("bandaid.photometry.N_GAIA_STARS_ALIGN", n_gaia)
        sentinel_wcs = _make_tan_wcs()

        compute_wcs = mocker.patch(
            "bandaid.photometry.compute_wcs", return_value=sentinel_wcs
        )

        n_detected = 12  # more than either cap
        coords = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)
        radecs = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)

        aligned, returned_wcs, _ = align(coords, radecs, photometry_coords=None)

        assert returned_wcs is sentinel_wcs
        # The two lists are sliced by their own caps, independently.
        recorded_coords, recorded_radecs = compute_wcs.call_args.args
        assert len(recorded_coords) == n_image
        assert len(recorded_radecs) == n_gaia
        # align passes the tolerance constant through to twirl.
        assert compute_wcs.call_args.kwargs["tolerance"] == WCS_MATCH_TOLERANCE
        # With no photometry_coords, aligned coords are the detections themselves.
        np.testing.assert_array_equal(aligned, coords)

    def test_suppresses_compute_wcs_stdout(self, mocker, capsys):
        """
        Swallow the stdout twirl's asterism matcher prints.

        The matcher prints diagnostics (e.g. "Match took ... us") straight to
        stdout; align must swallow that noise so callers/notebooks stay clean.
        The WCS return value is unaffected.
        """
        sentinel_wcs = _make_tan_wcs()

        def noisy_compute_wcs(*args: object, **kwargs: object):  # noqa: ARG001
            print("Match took 12345.000 us")  # noqa: T201
            print(7)  # noqa: T201
            return sentinel_wcs

        mocker.patch("bandaid.photometry.compute_wcs", side_effect=noisy_compute_wcs)

        coords = align_coords(N_IMAGE_STARS_ALIGN)
        radecs = coords.copy()

        _, returned_wcs, _ = align(coords, radecs, photometry_coords=None)

        assert returned_wcs is sentinel_wcs
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "twirl_error",
        [
            # The original SS Leo failure: too few matched points reach
            # fit_wcs_from_points, so scipy's least-squares fitter raises.
            ValueError("Initial guess is outside of provided bounds"),
            # The shallower exit: cross_match finds zero pairs and the empty
            # float index array fails when used to index.
            IndexError("arrays used as indices must be of integer type"),
        ],
        ids=["fit_wcs_from_points-ValueError", "cross_match-IndexError"],
    )
    def test_twirl_raising_becomes_wcs_solve_error(self, mocker, twirl_error):
        """A too-few-stars raise from twirl surfaces as a recoverable WCSSolveError."""
        mocker.patch("bandaid.photometry.compute_wcs", side_effect=twirl_error)

        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(WCSSolveError, match="twirl raised") as excinfo:
            align(coords, coords.copy(), photometry_coords=None)
        # The original twirl error is preserved on the chain for the log.
        assert excinfo.value.__cause__ is twirl_error

    def test_twirl_returning_none_becomes_wcs_solve_error(self, mocker):
        """compute_wcs returning None (no match) surfaces as WCSSolveError."""
        mocker.patch("bandaid.photometry.compute_wcs", return_value=None)

        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(WCSSolveError, match="no acceptable WCS"):
            align(coords, coords.copy(), photometry_coords=None)

    def test_unexpected_twirl_error_propagates(self, mocker):
        """A non too-few-stars error is a bug and is left to propagate, not masked."""
        bug = TypeError("genuine bug, not a bad frame")
        mocker.patch("bandaid.photometry.compute_wcs", side_effect=bug)

        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(TypeError, match="genuine bug"):
            align(coords, coords.copy(), photometry_coords=None)

    def test_retries_with_deeper_gaia_pool_on_failure(self, mocker):
        """
        A shallow-pool match failure retries once at the deeper retry pool.

        The cheap match at N_GAIA_STARS_ALIGN is attempted first; only when it
        fails does align widen the Gaia reference pool to
        N_GAIA_STARS_ALIGN_RETRY, so the common case (which solves immediately)
        never pays the larger, slower asterism search.
        """
        sentinel_wcs = _make_tan_wcs()
        shallow_failure = ValueError("Initial guess is outside of provided bounds")

        def fake_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            # Fail at the shallow pool, succeed once the pool is deepened.
            if len(radecs) <= N_GAIA_STARS_ALIGN:
                raise shallow_failure
            return sentinel_wcs

        compute_wcs = mocker.patch(
            "bandaid.photometry.compute_wcs", side_effect=fake_compute_wcs
        )

        n_detected = N_GAIA_STARS_ALIGN_RETRY + 5  # more than either pool
        coords = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)
        radecs = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)

        _, returned_wcs, _ = align(coords, radecs, photometry_coords=None)

        assert returned_wcs is sentinel_wcs
        # Shallow pool tried first, then the deeper retry pool -- in that order.
        pool_sizes = [len(call.args[1]) for call in compute_wcs.call_args_list]
        assert pool_sizes == [N_GAIA_STARS_ALIGN, N_GAIA_STARS_ALIGN_RETRY]

    @pytest.mark.parametrize(
        ("pixscale", "expected_pixscale", "raises"),
        [
            (2.4, 2.4, None),
            (4.2, 2.4, WCSScaleError),
            (4.2, None, None),
            (SEESTAR_PIXSCALE * (1 - 0.007), SEESTAR_PIXSCALE, WCSScaleError),
            (SEESTAR_PIXSCALE * (1 + 0.003), SEESTAR_PIXSCALE, None),
            (SEESTAR_PIXSCALE * (1 - 0.002), SEESTAR_PIXSCALE, None),
        ],
        ids=[
            "matching-scale-accepted",
            "wrong-scale-rejected",
            "no-expected-scale-skips-check",
            "degraded-solve-0.7-percent-off-rejected",
            "real-spread-0.3-percent-high-accepted",
            "real-spread-0.2-percent-low-accepted",
        ],
    )
    def test_scale_check_gates_on_expected_pixscale(
        self, mocker, pixscale, expected_pixscale, raises
    ):
        """
        The plate-scale check accepts, rejects, or is skipped per expected_pixscale.

        A matching scale is accepted; a scale far from the expectation (the
        twirl-returns-a-self-consistent-but-wrong-scale case, ~4.2 vs the true
        ~2.4 arcsec/px) raises WCSScaleError rather than photometering at the
        wrong pixel positions; and expected_pixscale=None skips the check
        entirely (back-compat), trusting even a wrong-scale WCS. With a 0.5%
        ``scale_tolerance`` a degraded solve 0.7% off the profile scale is
        rejected, while the 0.2-0.3% spread of good solves is accepted.
        """
        solved_wcs = _make_tan_wcs(pixscale=pixscale)
        mocker.patch("bandaid.photometry.compute_wcs", return_value=solved_wcs)
        coords = align_coords(N_IMAGE_STARS_ALIGN)

        if raises is not None:
            with pytest.raises(raises, match="scale"):
                align(
                    coords,
                    coords.copy(),
                    photometry_coords=None,
                    expected_pixscale=expected_pixscale,
                    scale_tolerance=TIGHT_SCALE_TOLERANCE,
                )
            return

        _, returned_wcs, _ = align(
            coords,
            coords.copy(),
            photometry_coords=None,
            expected_pixscale=expected_pixscale,
            scale_tolerance=TIGHT_SCALE_TOLERANCE,
        )
        assert returned_wcs is solved_wcs

    @pytest.mark.parametrize(
        "failure_mode",
        ["scale", "center"],
    )
    def test_retries_deeper_pool_on_bad_first_solve(self, mocker, failure_mode):
        """
        A wrong-scale or mispointed shallow solve retries at the deeper Gaia pool.

        A bad-scale WCS and an off-frame-center WCS are both failures to retry
        just like a None/raise: align widens the reference pool and accepts the
        deeper pool's correct solve. The two rejection reasons share one retry
        path, exercised here keyed on scale-vs-center.
        """
        if failure_mode == "scale":
            good_wcs = _make_tan_wcs(pixscale=2.4)
            bad_wcs = _make_tan_wcs(pixscale=4.2)
            align_kwargs = {"expected_pixscale": 2.4}
        else:
            good_wcs = _make_tan_wcs(crval=(10.0, 20.0))
            bad_wcs = _make_tan_wcs(crval=(15.0, 20.0))
            align_kwargs = {
                "expected_center": SkyCoord(10.0, 20.0, unit="deg"),
                "shape": (500, 500),
            }

        def fake_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            return bad_wcs if len(radecs) <= N_GAIA_STARS_ALIGN else good_wcs

        compute_wcs = mocker.patch(
            "bandaid.photometry.compute_wcs", side_effect=fake_compute_wcs
        )

        n_detected = N_GAIA_STARS_ALIGN_RETRY + 5
        coords = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)
        radecs = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)

        _, returned_wcs, _ = align(
            coords, radecs, photometry_coords=None, **align_kwargs
        )

        assert returned_wcs is good_wcs
        pool_sizes = [len(call.args[1]) for call in compute_wcs.call_args_list]
        assert pool_sizes == [N_GAIA_STARS_ALIGN, N_GAIA_STARS_ALIGN_RETRY]

    def test_supplied_wcs_scale_not_checked(self):
        """A caller-supplied WCS is trusted and not scale-checked."""
        bad_wcs = _make_tan_wcs(pixscale=4.2)
        coords = np.array([[250.0, 250.0], [260.0, 260.0]])

        _, returned_wcs, _ = align(
            coords, radecs=None, wcs=bad_wcs, expected_pixscale=2.4
        )

        assert returned_wcs is bad_wcs

    def test_scale_tolerance_param_controls_the_check(self, mocker):
        """
        The ``scale_tolerance`` argument gates the check, not the module default.

        A WCS 10% off the expected scale is accepted under a loose 20% tolerance
        but rejected under a tight 5% tolerance, so a per-instrument tolerance
        threaded in from the config actually drives the decision.
        """
        wcs_10pct_off = _make_tan_wcs(pixscale=2.4 * 1.10)
        mocker.patch("bandaid.photometry.compute_wcs", return_value=wcs_10pct_off)
        coords = align_coords(N_IMAGE_STARS_ALIGN)

        _, returned_wcs, _ = align(
            coords,
            coords.copy(),
            photometry_coords=None,
            expected_pixscale=2.4,
            scale_tolerance=0.20,
        )
        assert returned_wcs is wcs_10pct_off

        with pytest.raises(WCSScaleError, match="scale"):
            align(
                coords,
                coords.copy(),
                photometry_coords=None,
                expected_pixscale=2.4,
                scale_tolerance=0.05,
            )

    @pytest.mark.parametrize(
        ("tolerance", "expected_text"),
        [(0.0025, "> 0.25% off"), (0.0004, "> 0.04% off"), (0.05, "> 5% off")],
    )
    def test_scale_error_states_the_tolerance_exactly(
        self, mocker, tolerance, expected_text
    ):
        """The wrong-scale message prints the tolerance without rounding it away."""
        mocker.patch(
            "bandaid.photometry.compute_wcs", return_value=_make_tan_wcs(pixscale=4.2)
        )
        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(WCSScaleError, match=expected_text) as excinfo:
            align(
                coords,
                coords.copy(),
                photometry_coords=None,
                expected_pixscale=2.4,
                scale_tolerance=tolerance,
            )
        assert excinfo.value.measured_scale == pytest.approx(4.2)

    def test_returns_the_measured_scale_and_center_offset(self, mocker):
        """
        ``align`` returns the plate scale and center offset of the accepted WCS.

        The values are the ones the validation compared against its limits, so
        the QA manifest cannot disagree with the gate. The offset is None when
        no expected center was given, and a supplied WCS is measured too.
        """
        solved_wcs = _make_tan_wcs(pixscale=2.38, crval=(10.0, 20.0))
        mocker.patch("bandaid.photometry.compute_wcs", return_value=solved_wcs)
        coords = align_coords(N_IMAGE_STARS_ALIGN)
        kwargs = {
            "photometry_coords": None,
            "expected_center": SkyCoord(10.0, 20.1, unit="deg"),
            "shape": (500, 500),
        }

        _, _, measured = align(coords, coords.copy(), **kwargs)
        assert measured.pixscale == pytest.approx(2.38, rel=1e-4)
        assert measured.offset_deg == pytest.approx(0.1, abs=1e-3)

        _, _, no_center = align(coords, coords.copy(), photometry_coords=None)
        assert no_center.offset_deg is None

        _, _, supplied = align(coords, None, wcs=solved_wcs, **kwargs)
        assert supplied == measured

    def test_wcs_scale_error_is_wcs_solve_error(self):
        """WCSScaleError is a WCSSolveError so the batch loop still skips the frame."""
        assert issubclass(WCSScaleError, WCSSolveError)

    @pytest.mark.parametrize(
        ("crval", "expected_center", "raises"),
        [
            ((10.0, 20.0), (10.0, 20.0), None),
            ((15.0, 20.0), (10.0, 20.0), WCSPointingError),
            ((10.0, 20.0), (10.0, 20.2), None),
            ((15.0, 20.0), None, None),
        ],
        ids=[
            "center-in-frame-accepted",
            "far-from-center-rejected",
            "slightly-off-frame-accepted",
            "no-expected-center-skips-check",
        ],
    )
    def test_center_check_gates_on_expected_center(
        self, mocker, crval, expected_center, raises
    ):
        """
        The in-frame check accepts, rejects, or is skipped per expected_center.

        A WCS placing the queried field center on-frame is accepted. A WCS whose
        frame center is more than one field radius (half-diagonal) from the
        queried pointing is a mispointed (false-asterism) solve and raises
        WCSPointingError -- the Gaia catalog, queried at the header pointing,
        would barely overlap such a frame. A center just past the frame edge is
        NOT mispointed (regression for #83, SS Leo 20260418: the header target
        can legitimately sit at/drift a few arcmin past the edge -- 0.2 deg is
        outside a 500-px frame's ~0.17 deg half-width but inside its ~0.24 deg
        half-diagonal). expected_center=None skips the check (back-compat).
        """
        solved_wcs = _make_tan_wcs(crval=crval)
        mocker.patch("bandaid.photometry.compute_wcs", return_value=solved_wcs)
        coords = align_coords(N_IMAGE_STARS_ALIGN)
        expected = (
            SkyCoord(*expected_center, unit="deg")
            if expected_center is not None
            else None
        )

        if raises is not None:
            with pytest.raises(raises, match="center"):
                align(
                    coords,
                    coords.copy(),
                    photometry_coords=None,
                    expected_center=expected,
                    shape=(500, 500),
                )
            return

        _, returned_wcs, _ = align(
            coords,
            coords.copy(),
            photometry_coords=None,
            expected_center=expected,
            shape=(500, 500),
        )
        assert returned_wcs is solved_wcs

    def test_pointing_tolerance_param_controls_the_check(self, mocker):
        """
        ``pointing_tolerance`` (degrees) sets how far the solved center may sit.

        A solve 0.2 deg from the header center passes the default (field-radius)
        limit but is rejected as mispointed once the tolerance drops to 0.1 deg,
        and the error says which limit was exceeded.
        """
        solved_wcs = _make_tan_wcs(crval=(10.0, 20.0))
        mocker.patch("bandaid.photometry.compute_wcs", return_value=solved_wcs)
        coords = align_coords(N_IMAGE_STARS_ALIGN)
        kwargs = {
            "photometry_coords": None,
            "expected_center": SkyCoord(10.0, 20.2, unit="deg"),
            "shape": (500, 500),
        }

        _, returned_wcs, _ = align(coords, coords.copy(), **kwargs)
        assert returned_wcs is solved_wcs

        with pytest.raises(WCSPointingError, match=r"0\.1 deg"):
            align(coords, coords.copy(), pointing_tolerance=0.1, **kwargs)

    def test_pointing_tolerance_is_independent_of_frame_size(self, mocker):
        """
        A fixed ``pointing_tolerance`` replaces the field-radius limit.

        A solve 0.4 deg from the header center sits inside the half-diagonal of
        a 1000-px frame at 2.376 arcsec/px (about 0.47 deg), so the default
        accepts it; a fixed 0.3 deg tolerance rejects it regardless of frame size.
        """
        mocker.patch(
            "bandaid.photometry.compute_wcs",
            return_value=_make_tan_wcs((1000, 1000), crval=(10.0, 20.0)),
        )
        coords = align_coords(N_IMAGE_STARS_ALIGN)
        kwargs = {
            "photometry_coords": None,
            "expected_center": SkyCoord(10.0, 20.4, unit="deg"),
            "shape": (1000, 1000),
        }

        align(coords, coords.copy(), **kwargs)

        with pytest.raises(WCSPointingError, match="center"):
            align(coords, coords.copy(), pointing_tolerance=0.30, **kwargs)

    def test_seestar_tolerance_accepts_largest_measured_offset(self, mocker):
        """
        The Seestar50 pointing tolerance accepts a 0.29 deg re-acquisition offset.

        0.29 deg is the largest legitimate header-to-solve offset seen across
        the six Seestar S50 fields the tolerance was tuned on. It is wider
        than a 500-px frame's half-diagonal (about 0.23 deg), so the
        field-radius default rejects it while the bundled profile's fixed
        tolerance accepts it.
        """
        solved_wcs = _make_tan_wcs(crval=(10.0, 20.0))
        mocker.patch("bandaid.photometry.compute_wcs", return_value=solved_wcs)
        coords = align_coords(N_IMAGE_STARS_ALIGN)
        kwargs = {
            "photometry_coords": None,
            "expected_center": SkyCoord(10.0, 20.29, unit="deg"),
            "shape": (500, 500),
        }
        seestar_tolerance = load_instrument("Seestar50").wcs_pointing_tolerance

        with pytest.raises(WCSPointingError, match="center"):
            align(coords, coords.copy(), **kwargs)

        _, returned_wcs, _ = align(
            coords, coords.copy(), pointing_tolerance=seestar_tolerance, **kwargs
        )
        assert returned_wcs is solved_wcs

    def test_supplied_wcs_center_not_checked(self):
        """A caller-supplied WCS is trusted and not center-checked."""
        mispointed_wcs = _make_tan_wcs(crval=(15.0, 20.0))
        coords = np.array([[250.0, 250.0], [260.0, 260.0]])

        _, returned_wcs, _ = align(
            coords,
            radecs=None,
            wcs=mispointed_wcs,
            expected_center=SkyCoord(10.0, 20.0, unit="deg"),
            shape=(500, 500),
        )

        assert returned_wcs is mispointed_wcs

    def test_wrong_scale_wins_over_bad_center(self, mocker):
        """
        A WCS failing both checks is reported as a scale error, not pointing.

        The scale check runs first: a wrong-scale solve is the known twirl
        failure mode and should not be masked as a pointing failure just because
        the bogus scale also throws the projected center off-frame.
        """
        doubly_bad_wcs = _make_tan_wcs(pixscale=4.2, crval=(15.0, 20.0))
        mocker.patch("bandaid.photometry.compute_wcs", return_value=doubly_bad_wcs)
        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(WCSScaleError, match="scale"):
            align(
                coords,
                coords.copy(),
                photometry_coords=None,
                expected_pixscale=2.4,
                expected_center=SkyCoord(10.0, 20.0, unit="deg"),
                shape=(500, 500),
            )

    def test_wcs_pointing_error_is_wcs_solve_error(self):
        """WCSPointingError is a WCSSolveError so the batch loop still skips."""
        assert issubclass(WCSPointingError, WCSSolveError)


class TestSolvePoolNear:
    """Unit tests for the per-frame solve-pool cone mask."""

    def test_keeps_stars_within_radius_in_order(self):
        """Stars inside the radius are kept, in their input order."""
        radecs = np.array([[10.0, 25.0], [10.0, 20.5], [10.0, 16.0], [10.0, 19.8]])
        mask = _solve_pool_near(radecs, 10.0, 20.0, 0.7)

        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, [False, True, False, True])
        np.testing.assert_array_equal(radecs[mask], radecs[[1, 3]])

    def test_empty_input(self):
        """An empty (0, 2) catalog gives an empty boolean mask."""
        mask = _solve_pool_near(np.empty((0, 2)), 10.0, 20.0, 1.0)
        assert mask.dtype == bool
        assert mask.shape == (0,)

    def test_zero_radius_keeps_only_coincident_star(self):
        """Radius zero keeps only an exactly coincident star."""
        radecs = np.array([[10.0, 20.0], [10.0, 20.001]])
        np.testing.assert_array_equal(
            _solve_pool_near(radecs, 10.0, 20.0, 0.0), [True, False]
        )
