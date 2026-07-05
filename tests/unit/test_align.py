"""Unit tests for ``align`` and WCS solve/validation."""

import astropy.units as u
import numpy as np
import pytest
from _helpers import _make_tan_wcs, align_coords
from astropy.coordinates import SkyCoord

from bandaid.exceptions import (
    WCSPointingError,
    WCSScaleError,
    WCSSolveError,
)
from bandaid.photometry import (
    N_GAIA_STARS_ALIGN,
    N_GAIA_STARS_ALIGN_RETRY,
    N_IMAGE_STARS_ALIGN,
    WCS_MATCH_TOLERANCE,
    align,
)


class TestAlign:
    """Unit tests for the WCS-solve/projection helper ``align``."""

    def test_projects_photometry_coords_through_supplied_wcs(self):
        """photometry_coords are projected to pixels via the provided WCS."""
        wcs = _make_tan_wcs(crval=(10.0, 20.0))
        sky = SkyCoord(ra=[10.0, 10.01] * u.deg, dec=[20.0, 20.01] * u.deg)
        coords = np.array([[250.0, 250.0], [260.0, 260.0]])

        aligned, returned_wcs = align(
            coords, radecs=None, photometry_coords=sky, wcs=wcs
        )

        assert returned_wcs is wcs
        expected = np.array(wcs.world_to_pixel(sky)).T
        np.testing.assert_allclose(aligned, expected)
        assert aligned.shape == (2, 2)

    def test_solves_wcs_from_detections_when_none_supplied(self, monkeypatch):
        """
        With wcs=None, align slices image and Gaia coords *independently*.

        Detections are capped at N_IMAGE_STARS_ALIGN and Gaia references at
        N_GAIA_STARS_ALIGN -- the two counts are decoupled so the matcher can be
        fed more references than detections. The constants are monkeypatched to
        distinct values here to prove the slices are independent rather than a
        single shared cap. compute_wcs (twirl's slow, stochastic asterism solver)
        is stubbed with a sentinel WCS; the unit under test is align's slicing,
        not twirl's matching.
        """
        n_image = 4
        n_gaia = 7
        monkeypatch.setattr("bandaid.photometry.N_IMAGE_STARS_ALIGN", n_image)
        monkeypatch.setattr("bandaid.photometry.N_GAIA_STARS_ALIGN", n_gaia)
        sentinel_wcs = _make_tan_wcs()
        calls = {}

        def fake_compute_wcs(coords, radecs, tolerance):
            calls["coords"] = coords
            calls["radecs"] = radecs
            calls["tolerance"] = tolerance
            return sentinel_wcs

        monkeypatch.setattr("bandaid.photometry.compute_wcs", fake_compute_wcs)

        n_detected = 12  # more than either cap
        coords = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)
        radecs = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)

        aligned, returned_wcs = align(coords, radecs, photometry_coords=None)

        assert returned_wcs is sentinel_wcs
        # The two lists are sliced by their own caps, independently.
        assert len(calls["coords"]) == n_image
        assert len(calls["radecs"]) == n_gaia
        # align passes the tolerance constant through to twirl.
        assert calls["tolerance"] == WCS_MATCH_TOLERANCE
        # With no photometry_coords, aligned coords are the detections themselves.
        np.testing.assert_array_equal(aligned, coords)

    def test_suppresses_compute_wcs_stdout(self, monkeypatch, capsys):
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

        monkeypatch.setattr("bandaid.photometry.compute_wcs", noisy_compute_wcs)

        coords = align_coords(N_IMAGE_STARS_ALIGN)
        radecs = coords.copy()

        _, returned_wcs = align(coords, radecs, photometry_coords=None)

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
    def test_twirl_raising_becomes_wcs_solve_error(self, monkeypatch, twirl_error):
        """A too-few-stars raise from twirl surfaces as a recoverable WCSSolveError."""

        def failing_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            raise twirl_error

        monkeypatch.setattr("bandaid.photometry.compute_wcs", failing_compute_wcs)

        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(WCSSolveError, match="twirl raised") as excinfo:
            align(coords, coords.copy(), photometry_coords=None)
        # The original twirl error is preserved on the chain for the log.
        assert excinfo.value.__cause__ is twirl_error

    def test_twirl_returning_none_becomes_wcs_solve_error(self, monkeypatch):
        """compute_wcs returning None (no match) surfaces as WCSSolveError."""

        def none_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            return None

        monkeypatch.setattr("bandaid.photometry.compute_wcs", none_compute_wcs)

        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(WCSSolveError, match="no acceptable WCS"):
            align(coords, coords.copy(), photometry_coords=None)

    def test_unexpected_twirl_error_propagates(self, monkeypatch):
        """A non too-few-stars error is a bug and is left to propagate, not masked."""
        bug = TypeError("genuine bug, not a bad frame")

        def buggy_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            raise bug

        monkeypatch.setattr("bandaid.photometry.compute_wcs", buggy_compute_wcs)

        coords = align_coords(N_IMAGE_STARS_ALIGN)

        with pytest.raises(TypeError, match="genuine bug"):
            align(coords, coords.copy(), photometry_coords=None)

    def test_retries_with_deeper_gaia_pool_on_failure(self, monkeypatch):
        """
        A shallow-pool match failure retries once at the deeper retry pool.

        The cheap match at N_GAIA_STARS_ALIGN is attempted first; only when it
        fails does align widen the Gaia reference pool to
        N_GAIA_STARS_ALIGN_RETRY, so the common case (which solves immediately)
        never pays the larger, slower asterism search.
        """
        sentinel_wcs = _make_tan_wcs()
        pool_sizes = []
        shallow_failure = ValueError("Initial guess is outside of provided bounds")

        def fake_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            pool_sizes.append(len(radecs))
            # Fail at the shallow pool, succeed once the pool is deepened.
            if len(radecs) <= N_GAIA_STARS_ALIGN:
                raise shallow_failure
            return sentinel_wcs

        monkeypatch.setattr("bandaid.photometry.compute_wcs", fake_compute_wcs)

        n_detected = N_GAIA_STARS_ALIGN_RETRY + 5  # more than either pool
        coords = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)
        radecs = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)

        _, returned_wcs = align(coords, radecs, photometry_coords=None)

        assert returned_wcs is sentinel_wcs
        # Shallow pool tried first, then the deeper retry pool -- in that order.
        assert pool_sizes == [N_GAIA_STARS_ALIGN, N_GAIA_STARS_ALIGN_RETRY]

    @pytest.mark.parametrize(
        ("pixscale", "expected_pixscale", "raises"),
        [
            (2.4, 2.4, None),
            (4.2, 2.4, WCSScaleError),
            (4.2, None, None),
        ],
        ids=[
            "matching-scale-accepted",
            "wrong-scale-rejected",
            "no-expected-scale-skips-check",
        ],
    )
    def test_scale_check_gates_on_expected_pixscale(
        self, monkeypatch, pixscale, expected_pixscale, raises
    ):
        """
        The plate-scale check accepts, rejects, or is skipped per expected_pixscale.

        A matching scale is accepted; a scale far from the expectation (the
        twirl-returns-a-self-consistent-but-wrong-scale case, ~4.2 vs the true
        ~2.4 arcsec/px) raises WCSScaleError rather than photometering at the
        wrong pixel positions; and expected_pixscale=None skips the check
        entirely (back-compat), trusting even a wrong-scale WCS.
        """
        solved_wcs = _make_tan_wcs(pixscale=pixscale)
        monkeypatch.setattr(
            "bandaid.photometry.compute_wcs",
            lambda coords, radecs, tolerance: solved_wcs,
        )
        coords = align_coords(N_IMAGE_STARS_ALIGN)

        if raises is not None:
            with pytest.raises(raises, match="scale"):
                align(
                    coords,
                    coords.copy(),
                    photometry_coords=None,
                    expected_pixscale=expected_pixscale,
                )
            return

        _, returned_wcs = align(
            coords,
            coords.copy(),
            photometry_coords=None,
            expected_pixscale=expected_pixscale,
        )
        assert returned_wcs is solved_wcs

    @pytest.mark.parametrize(
        "failure_mode",
        ["scale", "center"],
    )
    def test_retries_deeper_pool_on_bad_first_solve(self, monkeypatch, failure_mode):
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
        pool_sizes = []

        def fake_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            pool_sizes.append(len(radecs))
            return bad_wcs if len(radecs) <= N_GAIA_STARS_ALIGN else good_wcs

        monkeypatch.setattr("bandaid.photometry.compute_wcs", fake_compute_wcs)

        n_detected = N_GAIA_STARS_ALIGN_RETRY + 5
        coords = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)
        radecs = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)

        _, returned_wcs = align(coords, radecs, photometry_coords=None, **align_kwargs)

        assert returned_wcs is good_wcs
        assert pool_sizes == [N_GAIA_STARS_ALIGN, N_GAIA_STARS_ALIGN_RETRY]

    def test_supplied_wcs_scale_not_checked(self):
        """A caller-supplied WCS is trusted and not scale-checked."""
        bad_wcs = _make_tan_wcs(pixscale=4.2)
        coords = np.array([[250.0, 250.0], [260.0, 260.0]])

        _, returned_wcs = align(coords, radecs=None, wcs=bad_wcs, expected_pixscale=2.4)

        assert returned_wcs is bad_wcs

    def test_scale_tolerance_param_controls_the_check(self, monkeypatch):
        """
        The ``scale_tolerance`` argument gates the check, not the module default.

        A WCS 10% off the expected scale is accepted under a loose 20% tolerance
        but rejected under a tight 5% tolerance, so a per-instrument tolerance
        threaded in from the config actually drives the decision.
        """
        wcs_10pct_off = _make_tan_wcs(pixscale=2.4 * 1.10)
        monkeypatch.setattr(
            "bandaid.photometry.compute_wcs",
            lambda coords, radecs, tolerance: wcs_10pct_off,
        )
        coords = align_coords(N_IMAGE_STARS_ALIGN)

        _, returned_wcs = align(
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
        self, monkeypatch, crval, expected_center, raises
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
        monkeypatch.setattr(
            "bandaid.photometry.compute_wcs",
            lambda coords, radecs, tolerance: solved_wcs,
        )
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

        _, returned_wcs = align(
            coords,
            coords.copy(),
            photometry_coords=None,
            expected_center=expected,
            shape=(500, 500),
        )
        assert returned_wcs is solved_wcs

    def test_supplied_wcs_center_not_checked(self):
        """A caller-supplied WCS is trusted and not center-checked."""
        mispointed_wcs = _make_tan_wcs(crval=(15.0, 20.0))
        coords = np.array([[250.0, 250.0], [260.0, 260.0]])

        _, returned_wcs = align(
            coords,
            radecs=None,
            wcs=mispointed_wcs,
            expected_center=SkyCoord(10.0, 20.0, unit="deg"),
            shape=(500, 500),
        )

        assert returned_wcs is mispointed_wcs

    def test_wrong_scale_wins_over_bad_center(self, monkeypatch):
        """
        A WCS failing both checks is reported as a scale error, not pointing.

        The scale check runs first: a wrong-scale solve is the known twirl
        failure mode and should not be masked as a pointing failure just because
        the bogus scale also throws the projected center off-frame.
        """
        doubly_bad_wcs = _make_tan_wcs(pixscale=4.2, crval=(15.0, 20.0))
        monkeypatch.setattr(
            "bandaid.photometry.compute_wcs",
            lambda coords, radecs, tolerance: doubly_bad_wcs,
        )
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
