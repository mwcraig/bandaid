"""
Unit tests for the ``bandaid`` command-line interface in :mod:`bandaid.cli`.

The CLI is a thin dressing over :func:`bandaid.scripts.photometer_frames`: it turns
command-line flags into a `PhotometryConfig` and a metadata dict, then delegates
the file-expansion + ``prepare_batch`` → ``process_batch`` flow to that function.
These tests monkeypatch ``photometer_frames`` out and assert the flag-to-argument
wiring and the clean-error handling; the engine itself is covered in
``test_scripts.py``. The instrument/config commands run against the real bundled
``Seestar50`` profile and the real ``PhotometryConfig`` (both offline).
"""

import json
import logging

import pytest
from click.testing import CliRunner

from bandaid import cli
from bandaid.config import InstrumentProfile, PhotometryConfig
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
def patched_photometer(monkeypatch):
    """
    Patch ``cli.photometer_frames`` and record how the CLI called it.

    Returns a dict the test can inspect: every keyword the CLI forwarded plus the
    positional ``files`` argument. The fake returns a ``(frames, results)`` pair
    with a deliberate frame/result count mismatch so the summary line is testable.
    """
    calls = {}

    def fake_photometer(files, **kwargs: object):
        calls["files"] = list(files)
        calls.update(kwargs)
        return ["frame1", "frame2"], {"frame1": "frame1.star"}

    monkeypatch.setattr(cli, "photometer_frames", fake_photometer)
    return calls


@pytest.fixture
def fully_failed_photometer(monkeypatch):
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

    monkeypatch.setattr(cli, "photometer_frames", fake_photometer)


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
        ],
    )

    assert result.exit_code == 0, result.output
    # The raw argument is forwarded; photometer_frames does the expansion.
    assert patched_photometer["files"] == [str(frame_dir)]
    assert patched_photometer["weights"] == str(weights)
    assert patched_photometer["user_specific_metadata"] == {"observer": "MWC"}
    assert patched_photometer["output_dir"] == str(out_dir)
    assert patched_photometer["append_l4"] is False
    assert patched_photometer["fail_fast"] is True
    # --output-format resolves to the registered writer callable, not the name.
    assert patched_photometer["write_frame"] is write_starlist_set
    assert patched_photometer["output_suffix"] == ".starlist"
    assert patched_photometer["write_qa_manifest"] is False
    # The config carries the default (Seestar50) instrument.
    config = patched_photometer["config"]
    assert isinstance(config, PhotometryConfig)
    assert config.instrument.name == "Seestar50"
    # The summary reflects the returned (results, frames) counts.
    assert "Processed 1 of 2 frames" in result.output


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
def spy_configure_logging(monkeypatch):
    """Record the level ``cli.configure_logging`` is called with (if at all)."""
    calls = []
    monkeypatch.setattr(cli, "configure_logging", lambda **kwargs: calls.append(kwargs))
    return calls


@pytest.mark.usefixtures("patched_photometer")
def test_process_quiet_by_default_still_logs_warnings(
    runner, spy_configure_logging, tmp_path
):
    """Without --verbose, WARNING+ (skip/error records) still reach stderr."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert result.exit_code == 0, result.output
    assert spy_configure_logging == [{"level": logging.WARNING}]


@pytest.mark.usefixtures("patched_photometer")
def test_process_verbose_enables_info_logging(runner, spy_configure_logging, tmp_path):
    """``-v`` routes bandaid records to the terminal at INFO."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame), "-v"])

    assert result.exit_code == 0, result.output
    assert spy_configure_logging == [{"level": logging.INFO}]


@pytest.mark.usefixtures("patched_photometer")
def test_process_double_verbose_enables_debug_logging(
    runner, spy_configure_logging, tmp_path
):
    """``-vv`` drops to DEBUG for extra detail."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame), "-vv"])

    assert result.exit_code == 0, result.output
    assert spy_configure_logging == [{"level": logging.DEBUG}]


def test_process_uses_robust_defaults(runner, patched_photometer, tmp_path):
    """Omitting options downloads weights, appends L4, and uses robust defaults."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert result.exit_code == 0, result.output
    assert patched_photometer["weights"] is None
    assert patched_photometer["user_specific_metadata"] == {}
    # append_l4 now defaults ON.
    assert patched_photometer["append_l4"] is True
    assert patched_photometer["fail_fast"] is False
    assert patched_photometer["write_qa_manifest"] is True


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
    assert patched_photometer["files"] == [str(n1), str(n2)]


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
    config = patched_photometer["config"]
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
    config = patched_photometer["config"]
    assert config.instrument.name == "MyScope"


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


def test_weights_print(runner, monkeypatch, tmp_path):
    """Bare ``weights`` prints the cached default weights path."""
    cached = tmp_path / "centroid_15x15.npz"
    cached.write_bytes(b"npzdata")
    monkeypatch.setattr(cli, "download_weights", lambda: str(cached))

    result = runner.invoke(cli.main, ["weights"])

    assert result.exit_code == 0, result.output
    assert str(cached) in result.output


def test_weights_copy(runner, monkeypatch, tmp_path):
    """``weights -o DEST`` copies the cached ``.npz`` and prints the destination."""
    cached = tmp_path / "centroid_15x15.npz"
    cached.write_bytes(b"npzdata")
    monkeypatch.setattr(cli, "download_weights", lambda: str(cached))

    dest = tmp_path / "copied.npz"
    result = runner.invoke(cli.main, ["weights", "-o", str(dest)])

    assert result.exit_code == 0, result.output
    assert dest.read_bytes() == b"npzdata"
    assert str(dest) in result.output


def test_weights_copy_unwritable_is_clean_error(runner, monkeypatch, tmp_path):
    """``weights -o`` to an unwritable destination fails as a clean CLI error."""
    cached = tmp_path / "centroid_15x15.npz"
    cached.write_bytes(b"npzdata")
    monkeypatch.setattr(cli, "download_weights", lambda: str(cached))

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
