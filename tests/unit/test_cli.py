"""
Unit tests for the ``bandaid`` command-line interface in :mod:`bandaid.cli`.

The CLI is a thin dressing over :func:`bandaid.scripts.photometer_frames`: it turns
command-line flags into a `PhotometryConfig` and a metadata dict, then delegates
the file-expansion + ``prepare_batch`` → ``process_batch`` flow to that function.
These tests patch ``photometer_frames`` out and assert the flag-to-argument
wiring and the clean-error handling; the engine itself is covered in
``test_scripts.py``. The instrument/config commands run against the real bundled
``Seestar50`` profile and the real ``PhotometryConfig`` (both offline).
"""

import json
import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from click.testing import CliRunner

from bandaid import cli
from bandaid.config import InstrumentProfile, PhotometryConfig, SourceSelectionConfig
from bandaid.instruments import _REGISTERED, register_instrument
from bandaid.writers import write_starlist_set


@pytest.fixture
def runner():
    """Return a Click ``CliRunner`` for invoking the CLI in-process."""
    return CliRunner()


@pytest.fixture
def extra_instrument():
    """Register a second instrument so an override can be told from the default."""
    profile = InstrumentProfile(name="TestScope")
    register_instrument(profile)
    yield profile
    _REGISTERED.pop("TestScope", None)


@pytest.fixture
def patched_photometer(mocker):
    """
    Patch ``cli.photometer_frames`` and return the mock so calls can be inspected.

    The fake returns a ``(frames, results)`` pair with a deliberate frame/result
    count mismatch so the summary line is testable. Tests inspect what the CLI
    forwarded via ``patched_photometer.call_args`` -- ``.args[0]`` for the
    positional ``files`` argument, ``.kwargs[...]`` for everything else.
    """
    return mocker.patch(
        "bandaid.cli.photometer_frames",
        return_value=(["frame1", "frame2"], {"frame1": "frame1.star"}),
    )


@pytest.fixture
def fully_failed_photometer(mocker):
    """
    Patch ``cli.photometer_frames`` to simulate every frame in the batch failing.

    Mirrors what `bandaid.scripts.process_batch` does for a skipped/errored frame
    (a ``bandaid.scripts``-logger WARNING, per scripts.py:725/739) and returns 0
    results for 2 frames -- a fully failed batch, per issue #58.
    """

    def fake_photometer(_files, **_kwargs: object):
        scripts_logger = logging.getLogger("bandaid.scripts")
        scripts_logger.warning("skipping a.fit: not a FITS file")
        scripts_logger.warning("skipping b.fit: not a FITS file")
        return ["a.fit", "b.fit"], {}

    mocker.patch("bandaid.cli.photometer_frames", side_effect=fake_photometer)


def test_process_forwards_every_flag(runner, patched_photometer, tmp_path):
    """All process flags reach ``photometer_frames`` with the right values."""
    frame_dir = tmp_path / "night"
    frame_dir.mkdir()
    (frame_dir / "a.fit").write_bytes(b"")

    weights = tmp_path / "w.npz"
    weights.write_bytes(b"weights")

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"observer": "MWC"}))

    out_dir = tmp_path / "out"

    forced = tmp_path / "forced.csv"
    forced.write_text("name,ra,dec\nSN2024a,123.456,-10.0\n")

    result = runner.invoke(
        cli.main,
        [
            "process",
            str(frame_dir),
            "--weights",
            str(weights),
            "--user-metadata",
            str(meta),
            "--output-dir",
            str(out_dir),
            "--no-append-l4",
            "--fail-fast",
            "--output-format",
            "starlist",
            "--output-suffix",
            ".starlist",
            "--no-qa-manifest",
            "--forced-targets",
            str(forced),
        ],
    )

    assert result.exit_code == 0, result.output
    call_kwargs = patched_photometer.call_args.kwargs
    # The raw argument is forwarded; photometer_frames does the expansion.
    assert patched_photometer.call_args.args[0] == (str(frame_dir),)
    assert call_kwargs["weights"] == str(weights)
    assert call_kwargs["user_specific_metadata"] == {"observer": "MWC"}
    assert call_kwargs["output_dir"] == str(out_dir)
    assert call_kwargs["append_l4"] is False
    assert call_kwargs["fail_fast"] is True
    # --output-format resolves to the registered writer callable, not the name.
    assert call_kwargs["write_frame"] is write_starlist_set
    assert call_kwargs["output_suffix"] == ".starlist"
    assert call_kwargs["write_qa_manifest"] is False
    # The config carries the default (Seestar50) instrument.
    config = call_kwargs["config"]
    assert isinstance(config, PhotometryConfig)
    assert config.instrument.name == "Seestar50"
    # The summary reflects the returned (results, frames) counts.
    assert "Processed 1 of 2 frames" in result.output
    forced_targets = call_kwargs["forced_targets"]
    np.testing.assert_allclose(forced_targets.ra.deg, [123.456])
    np.testing.assert_allclose(forced_targets.dec.deg, [-10.0])


@pytest.mark.usefixtures("fully_failed_photometer")
def test_process_reports_frame_failures_to_stderr_by_default(runner, tmp_path):
    """
    Per-frame skip/error warnings reach the terminal even with no ``-v`` (#58).

    Before the fix the ``bandaid`` logger carried only a `logging.NullHandler`
    until ``-v`` was given, so every skip/error record (logged by
    `bandaid.scripts.process_batch`) vanished silently by default.
    """
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert result.stderr
    assert "skipping" in result.stderr


@pytest.mark.usefixtures("fully_failed_photometer")
def test_process_exit_code_reflects_a_fully_failed_batch(runner, tmp_path):
    """
    0 of N frames succeeding exits non-zero, not silent success (#58).

    Before the fix, a night where every frame failed still printed
    "Processed 0 of N frames" and exited 0 -- indistinguishable from success
    for a script or cron job.
    """
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert "Processed 0 of 2 frames" in result.output
    assert result.exit_code != 0


@pytest.mark.usefixtures("patched_photometer")
def test_process_partial_failure_still_exits_zero(runner, tmp_path):
    """A partially failed batch (some results) is normal robust-mode operation."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert "Processed 1 of 2 frames" in result.output
    assert result.exit_code == 0


@pytest.fixture
def spy_configure_logging(mocker):
    """Return the mock recording how ``cli.configure_logging`` was called."""
    return mocker.patch("bandaid.cli.configure_logging")


@pytest.mark.usefixtures("patched_photometer")
def test_process_quiet_by_default_still_logs_warnings(
    runner, spy_configure_logging, tmp_path
):
    """Without --verbose, WARNING+ (skip/error records) still reach stderr."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert result.exit_code == 0, result.output
    spy_configure_logging.assert_called_once_with(level=logging.WARNING, logfile=None)


@pytest.mark.usefixtures("patched_photometer")
def test_process_verbose_enables_info_logging(runner, spy_configure_logging, tmp_path):
    """``-v`` routes bandaid records to the terminal at INFO."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame), "-v"])

    assert result.exit_code == 0, result.output
    spy_configure_logging.assert_called_once_with(level=logging.INFO, logfile=None)


@pytest.mark.usefixtures("patched_photometer")
def test_process_double_verbose_enables_debug_logging(
    runner, spy_configure_logging, tmp_path
):
    """``-vv`` drops to DEBUG for extra detail."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame), "-vv"])

    assert result.exit_code == 0, result.output
    spy_configure_logging.assert_called_once_with(level=logging.DEBUG, logfile=None)


@pytest.mark.usefixtures("patched_photometer")
def test_process_log_file_forwards_to_configure_logging(
    runner, spy_configure_logging, tmp_path
):
    """``--log-file PATH`` is forwarded to ``configure_logging`` as ``logfile``."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    log_file = tmp_path / "run.log"

    result = runner.invoke(
        cli.main, ["process", str(frame), "--log-file", str(log_file)]
    )

    assert result.exit_code == 0, result.output
    spy_configure_logging.assert_called_once_with(
        level=logging.WARNING, logfile=str(log_file)
    )


def test_process_uses_robust_defaults(runner, patched_photometer, tmp_path):
    """Omitting options downloads weights, appends L4, and uses robust defaults."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert result.exit_code == 0, result.output
    call_kwargs = patched_photometer.call_args.kwargs
    assert call_kwargs["weights"] is None
    assert call_kwargs["user_specific_metadata"] == {}
    # append_l4 now defaults ON.
    assert call_kwargs["append_l4"] is True
    assert call_kwargs["fail_fast"] is False
    assert call_kwargs["write_qa_manifest"] is True


def test_process_omitted_forced_targets_defaults_to_none(
    runner, patched_photometer, tmp_path
):
    """Omitting ``--forced-targets`` forwards ``None`` (no forced targets)."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert result.exit_code == 0, result.output
    assert patched_photometer.call_args.kwargs["forced_targets"] is None


def test_process_forwards_multiple_directories(runner, patched_photometer, tmp_path):
    """Several directory arguments are all forwarded for expansion."""
    n1 = tmp_path / "n1"
    n2 = tmp_path / "n2"
    n1.mkdir()
    n2.mkdir()
    (n1 / "img.fit").write_bytes(b"")
    (n2 / "img.fit").write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(n1), str(n2)])

    assert result.exit_code == 0, result.output
    assert patched_photometer.call_args.args[0] == (str(n1), str(n2))


def test_process_instrument_override(
    runner, patched_photometer, extra_instrument, tmp_path
):
    """``--instrument`` selects a NON-default profile, proving the override took."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--instrument", extra_instrument.name]
    )

    assert result.exit_code == 0, result.output
    config = patched_photometer.call_args.kwargs["config"]
    assert config.instrument.name == extra_instrument.name


def test_process_profile_file_override(runner, patched_photometer, tmp_path):
    """``--profile FILE`` loads an unbundled profile onto the config."""
    profile = InstrumentProfile(name="MyScope")
    profile_file = tmp_path / "scope.json"
    profile.to_file(profile_file)

    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--profile", str(profile_file)]
    )

    assert result.exit_code == 0, result.output
    config = patched_photometer.call_args.kwargs["config"]
    assert config.instrument.name == "MyScope"


# CLI-override values for the source_selection override tests below: distinct
# from both the SourceSelectionConfig defaults and the config-file values in
# test_process_gaia_mag_limit_and_min_snr_win_over_config_file, so a passing
# assertion actually proves the CLI flag (not some other default) won.
_OVERRIDE_GAIA_MAG_LIMIT = 13.0
_OVERRIDE_MIN_SNR = 3.0
_CONFIG_FILE_GAIA_MAG_LIMIT = 10.0
_CONFIG_FILE_MIN_SNR = 1.0
_CONFIG_FILE_CONTAMINANT_OFFSET = 2.5


def test_process_gaia_mag_limit_override(runner, patched_photometer, tmp_path):
    """``--gaia-mag-limit`` overrides the source-selection default."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(
        cli.main,
        ["process", str(frame), "--gaia-mag-limit", str(_OVERRIDE_GAIA_MAG_LIMIT)],
    )

    assert result.exit_code == 0, result.output
    source_selection = patched_photometer.call_args.kwargs["config"].source_selection
    assert source_selection.gaia_mag_limit == _OVERRIDE_GAIA_MAG_LIMIT


def test_process_min_snr_override(runner, patched_photometer, tmp_path):
    """``--min-snr`` overrides the source-selection default."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--min-snr", str(_OVERRIDE_MIN_SNR)]
    )

    assert result.exit_code == 0, result.output
    source_selection = patched_photometer.call_args.kwargs["config"].source_selection
    assert source_selection.min_snr == _OVERRIDE_MIN_SNR


def test_process_gaia_mag_limit_and_min_snr_win_over_config_file(
    runner, patched_photometer, tmp_path
):
    """CLI flags override a ``--config`` file's source_selection, field for field."""
    base_config = PhotometryConfig(
        source_selection=SourceSelectionConfig(
            gaia_mag_limit=_CONFIG_FILE_GAIA_MAG_LIMIT,
            min_snr=_CONFIG_FILE_MIN_SNR,
            contaminant_mag_offset=_CONFIG_FILE_CONTAMINANT_OFFSET,
        )
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(base_config.model_dump_json())

    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(
        cli.main,
        [
            "process",
            str(frame),
            "--config",
            str(config_file),
            "--gaia-mag-limit",
            str(_OVERRIDE_GAIA_MAG_LIMIT),
            "--min-snr",
            str(_OVERRIDE_MIN_SNR),
        ],
    )

    assert result.exit_code == 0, result.output
    source_selection = patched_photometer.call_args.kwargs["config"].source_selection
    assert source_selection.gaia_mag_limit == _OVERRIDE_GAIA_MAG_LIMIT
    assert source_selection.min_snr == _OVERRIDE_MIN_SNR
    # The config file's other source_selection field is preserved, not reset.
    assert source_selection.contaminant_mag_offset == _CONFIG_FILE_CONTAMINANT_OFFSET


def test_process_non_finite_min_snr_is_clean_error(runner, tmp_path):
    """A non-finite ``--min-snr`` fails validation as a clean Click error."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame), "--min-snr", "inf"])

    assert result.exit_code == 1
    assert result.output.strip() != ""


def test_process_no_files_errors(runner, tmp_path):
    """A directory with no FITS frames is a clean error (exit 1), not a crash."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = runner.invoke(cli.main, ["process", str(empty)])

    assert result.exit_code == 1
    assert "no fits" in result.output.lower()


@pytest.mark.usefixtures("patched_photometer")
def test_process_unknown_output_format_is_clean_error(runner, tmp_path):
    """An unregistered ``--output-format`` is a clean Click error, not a crash."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--output-format", "no-such-format"]
    )

    assert result.exit_code == 1
    assert "no-such-format" in result.output


def test_process_bad_config_is_clean_error(runner, tmp_path):
    """A malformed ``--config`` file is a clean Click error, not a traceback."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    bad = tmp_path / "bad.json"
    bad.write_text('{"apertures": {"radii": "not-a-list"}}')

    result = runner.invoke(cli.main, ["process", str(frame), "--config", str(bad)])

    assert result.exit_code == 1
    assert result.output.strip() != ""


def test_process_bad_metadata_is_clean_error(runner, tmp_path):
    """Malformed ``--user-metadata`` JSON fails fast with a clear message."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    meta = tmp_path / "meta.json"
    meta.write_text("{not json")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--user-metadata", str(meta)]
    )

    assert result.exit_code == 1
    assert "json" in result.output.lower()


def test_process_non_object_metadata_is_clean_error(runner, tmp_path):
    """``--user-metadata`` that is valid JSON but not an object is rejected."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    meta = tmp_path / "meta.json"
    meta.write_text("[1, 2, 3]")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--user-metadata", str(meta)]
    )

    assert result.exit_code == 1
    assert "object" in result.output.lower()


def test_process_forced_targets_missing_column_is_clean_error(runner, tmp_path):
    """A forced-targets file missing a required column names it in the error."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("name,ra\nSN2024a,123.456\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert "dec" in result.output.lower()


def test_process_forced_targets_no_rows_is_clean_error(runner, tmp_path):
    """A header-only forced-targets file is rejected as having no rows."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("name,ra,dec\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert "no rows" in result.output.lower()


def test_process_forced_targets_non_numeric_ra_is_clean_error(runner, tmp_path):
    """Non-numeric ``ra`` values in the forced-targets file are a clean error."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("name,ra,dec\nSN2024a,not-a-number,-10.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert result.output.strip() != ""


def test_process_forced_targets_out_of_range_dec_is_clean_error(runner, tmp_path):
    """An out-of-range ``dec`` in the forced-targets file is a clean error."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("name,ra,dec\nSN2024a,123.456,100.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert result.output.strip() != ""


def test_process_forced_targets_uppercase_columns_are_accepted(
    runner, patched_photometer, tmp_path
):
    """Column names are matched case-insensitively (``RA``, ``Dec``, ``Name``)."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("Name,RA,Dec\nSN2024a,123.456,-10.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 0, result.output
    forced_targets = patched_photometer.call_args.kwargs["forced_targets"]
    np.testing.assert_allclose(forced_targets.ra.deg, [123.456])
    np.testing.assert_allclose(forced_targets.dec.deg, [-10.0])


def test_process_forced_targets_case_collision_is_clean_error(runner, tmp_path):
    """Columns colliding case-insensitively (both ``RA`` and ``ra``) are rejected."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("RA,ra,dec\n1.0,2.0,3.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert "collide" in result.output.lower()
    assert "RA" in result.output
    assert "ra" in result.output


def test_process_forced_targets_name_column_is_optional(
    runner, patched_photometer, tmp_path
):
    """A forced-targets file with only ``ra``/``dec`` (no ``name``) is accepted."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("ra,dec\n123.456,-10.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 0, result.output
    forced_targets = patched_photometer.call_args.kwargs["forced_targets"]
    np.testing.assert_allclose(forced_targets.ra.deg, [123.456])
    np.testing.assert_allclose(forced_targets.dec.deg, [-10.0])


def test_process_forced_targets_missing_both_ra_and_dec_lists_both(runner, tmp_path):
    """A file missing both required columns names both in a single error."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("name,mag\nSN2024a,12.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert "ra" in result.output.lower()
    assert "dec" in result.output.lower()


def test_process_forced_targets_blank_ra_cell_is_clean_error(runner, tmp_path):
    """A blank/masked ``ra`` cell is a clean error, not silently read as 0.0."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("name,ra,dec\nSN2024a,,-10.0\nSN2024b,50.0,10.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert "row" in result.output.lower()
    assert "1" in result.output


def test_process_forced_targets_literal_nan_value_is_clean_error(runner, tmp_path):
    """A literal ``nan`` typed in the file is rejected, not silently propagated."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("name,ra,dec\nSN2024a,nan,-10.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert "row" in result.output.lower()


@pytest.mark.parametrize("bad_ra", [400.0, -30.0])
def test_process_forced_targets_out_of_range_ra_is_clean_error(
    runner, tmp_path, bad_ra
):
    """A ``ra`` outside ``[0, 360)`` degrees is a clean error, not silent wrapping."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text(f"name,ra,dec\nSN2024a,{bad_ra},-10.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert result.output.strip() != ""


def test_process_forced_targets_hourangle_unit_is_converted(
    runner, patched_photometer, tmp_path
):
    """An ECSV ``ra`` column in hourangle units converts correctly to degrees."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    table = Table()
    table["ra"] = [10.0] * u.hourangle
    table["dec"] = [20.0] * u.deg
    forced = tmp_path / "forced.ecsv"
    table.write(forced, format="ascii.ecsv")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 0, result.output
    forced_targets = patched_photometer.call_args.kwargs["forced_targets"]
    np.testing.assert_allclose(forced_targets.ra.deg, [150.0])
    np.testing.assert_allclose(forced_targets.dec.deg, [20.0])


def test_process_forced_targets_non_angular_unit_is_clean_error(runner, tmp_path):
    """A ``ra``/``dec`` column carrying a non-angular unit is a clean error."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    table = Table()
    table["ra"] = [10.0] * u.m
    table["dec"] = [20.0] * u.deg
    forced = tmp_path / "forced.ecsv"
    table.write(forced, format="ascii.ecsv")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 1
    assert "unit" in result.output.lower()


def test_process_forced_targets_frame_is_icrs(runner, patched_photometer, tmp_path):
    """The returned forced-targets ``SkyCoord`` uses the ICRS frame explicitly."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")
    forced = tmp_path / "forced.csv"
    forced.write_text("ra,dec\n10.0,20.0\n")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--forced-targets", str(forced)]
    )

    assert result.exit_code == 0, result.output
    forced_targets = patched_photometer.call_args.kwargs["forced_targets"]
    assert forced_targets.frame.name == "icrs"


def test_instrument_list(runner):
    """``instrument list`` prints the resolvable profile names."""
    result = runner.invoke(cli.main, ["instrument", "list"])

    assert result.exit_code == 0, result.output
    assert "Seestar50" in result.output


def test_instrument_show(runner):
    """``instrument show NAME`` emits valid profile JSON."""
    result = runner.invoke(cli.main, ["instrument", "show", "Seestar50"])

    assert result.exit_code == 0, result.output
    parsed = InstrumentProfile.model_validate_json(result.output)
    assert parsed.name == "Seestar50"


def test_instrument_show_unknown(runner):
    """An unknown instrument name exits 1 (an application error) with a message."""
    result = runner.invoke(cli.main, ["instrument", "show", "NoSuchScope"])

    assert result.exit_code == 1
    assert "NoSuchScope" in result.output


def test_config_init_stdout(runner):
    """``config init`` to stdout round-trips back into a ``PhotometryConfig``."""
    result = runner.invoke(cli.main, ["config", "init"])

    assert result.exit_code == 0, result.output
    config = PhotometryConfig.model_validate_json(result.output)
    assert config.instrument.name == "Seestar50"


def test_config_init_file(runner, tmp_path):
    """``config init -o FILE`` writes a round-trippable config file."""
    out = tmp_path / "config.json"
    result = runner.invoke(cli.main, ["config", "init", "-o", str(out)])

    assert result.exit_code == 0, result.output
    config = PhotometryConfig.model_validate_json(out.read_text())
    assert config.instrument.name == "Seestar50"


def test_config_init_unwritable_is_clean_error(runner, tmp_path):
    """``config init -o`` to an unwritable path fails as a clean CLI error."""
    out = tmp_path / "nonexistent" / "config.json"  # parent dir does not exist

    result = runner.invoke(cli.main, ["config", "init", "-o", str(out)])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert str(out) in result.output


def test_config_validate_good(runner, tmp_path):
    """``config validate`` accepts a valid config file with exit 0."""
    good = tmp_path / "good.json"
    good.write_text(PhotometryConfig().model_dump_json(indent=2))

    result = runner.invoke(cli.main, ["config", "validate", str(good)])

    assert result.exit_code == 0, result.output


def test_config_validate_bad(runner, tmp_path):
    """``config validate`` rejects a malformed config with a message."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"apertures": {"radii": "not-a-list"}}')

    result = runner.invoke(cli.main, ["config", "validate", str(bad)])

    assert result.exit_code == 1
    assert result.output.strip() != ""


def test_weights_print(runner, mocker, tmp_path):
    """Bare ``weights`` prints the cached default weights path."""
    cached = tmp_path / "centroid_15x15.npz"
    cached.write_bytes(b"npzdata")
    mocker.patch("bandaid.cli.download_weights", return_value=str(cached))

    result = runner.invoke(cli.main, ["weights"])

    assert result.exit_code == 0, result.output
    assert str(cached) in result.output


def test_weights_copy(runner, mocker, tmp_path):
    """``weights -o DEST`` copies the cached ``.npz`` and prints the destination."""
    cached = tmp_path / "centroid_15x15.npz"
    cached.write_bytes(b"npzdata")
    mocker.patch("bandaid.cli.download_weights", return_value=str(cached))

    dest = tmp_path / "copied.npz"
    result = runner.invoke(cli.main, ["weights", "-o", str(dest)])

    assert result.exit_code == 0, result.output
    assert dest.read_bytes() == b"npzdata"
    assert str(dest) in result.output


def test_weights_copy_unwritable_is_clean_error(runner, mocker, tmp_path):
    """``weights -o`` to an unwritable destination fails as a clean CLI error."""
    cached = tmp_path / "centroid_15x15.npz"
    cached.write_bytes(b"npzdata")
    mocker.patch("bandaid.cli.download_weights", return_value=str(cached))

    dest = tmp_path / "nonexistent" / "copied.npz"  # parent dir does not exist
    result = runner.invoke(cli.main, ["weights", "-o", str(dest)])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert str(dest) in result.output


def test_main_help_lists_commands(runner):
    """``bandaid --help`` lists all four top-level commands."""
    result = runner.invoke(cli.main, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("process", "instrument", "config", "weights"):
        assert command in result.output
