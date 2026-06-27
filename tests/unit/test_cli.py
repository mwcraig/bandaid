"""
Unit tests for the ``bandaid`` command-line interface in :mod:`bandaid.cli`.

The CLI is a thin glue layer over the existing photometry functions, so these
tests exercise the argument parsing and I/O wiring while monkeypatching the
heavy/network dependencies (``Ballet``, ``prepare_batch``, ``process_batch``,
``download_weights``) out. The instrument/config commands run against the real
bundled ``Seestar50`` profile and the real ``PhotometryConfig`` (both offline).
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bandaid import cli
from bandaid.config import InstrumentProfile, PhotometryConfig


@pytest.fixture
def runner():
    """Return a Click ``CliRunner`` for invoking the CLI in-process."""
    return CliRunner()


def _make_frames(directory, names):
    """Create empty FITS files in ``directory`` and return their paths."""
    paths = []
    for name in names:
        path = Path(directory) / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


@pytest.fixture
def patched_batch(monkeypatch):
    """
    Patch the heavy ``process`` dependencies and record how they were called.

    Returns a dict the test can inspect: ``ballet`` (the model_file passed to
    ``Ballet``), ``prepare`` (the ``prepare_batch`` call kwargs + first file),
    and ``process`` (the ``process_batch`` call args/kwargs).
    """
    calls = {}

    cnn_sentinel = object()
    prep_sentinel = object()

    def fake_ballet(model_file=None):
        calls["ballet"] = model_file
        return cnn_sentinel

    def fake_prepare(first_file, *, cnn, config=None, append_l4=False):
        calls["prepare"] = {
            "first_file": first_file,
            "cnn": cnn,
            "config": config,
            "append_l4": append_l4,
        }
        return prep_sentinel

    def fake_process(files, prep, **kwargs: object):
        files = list(files)
        calls["process"] = {"files": files, "prep": prep, "kwargs": kwargs}
        return {f: Path(f).with_suffix(".star") for f in files}

    monkeypatch.setattr(cli, "Ballet", fake_ballet)
    monkeypatch.setattr(cli, "prepare_batch", fake_prepare)
    monkeypatch.setattr(cli, "process_batch", fake_process)
    calls["_cnn"] = cnn_sentinel
    calls["_prep"] = prep_sentinel
    return calls


def test_process_happy_path(runner, patched_batch, tmp_path):
    """A directory of frames is expanded, sorted, and wired through both steps."""
    frame_dir = tmp_path / "night"
    frame_dir.mkdir()
    # Deliberately out-of-order names; the CLI must sort them deterministically.
    _make_frames(frame_dir, ["c.fit", "a.fit", "b.fit"])

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
            "--metadata",
            str(meta),
            "--output-dir",
            str(out_dir),
            "--append-l4",
            "--fail-fast",
            "--output-suffix",
            ".star",
            "--no-qa-manifest",
        ],
    )

    assert result.exit_code == 0, result.output

    expected = sorted(str(p) for p in frame_dir.glob("*.fit"))
    # First file feeds prepare_batch; the full sorted list feeds process_batch.
    assert patched_batch["prepare"]["first_file"] == expected[0]
    assert patched_batch["process"]["files"] == expected

    # The Ballet model_file is the --weights path; the prep carries that CNN.
    assert patched_batch["ballet"] == str(weights)
    assert patched_batch["prepare"]["cnn"] is patched_batch["_cnn"]
    assert patched_batch["process"]["prep"] is patched_batch["_prep"]

    # The config carries the selected (default Seestar50) instrument.
    config = patched_batch["prepare"]["config"]
    assert isinstance(config, PhotometryConfig)
    assert config.instrument.name == "Seestar50"

    # append_l4 reaches prepare_batch; the rest reach process_batch.
    assert patched_batch["prepare"]["append_l4"] is True
    kwargs = patched_batch["process"]["kwargs"]
    assert kwargs["user_specific_metadata"] == {"observer": "MWC"}
    assert Path(kwargs["output_dir"]) == out_dir
    assert kwargs["fail_fast"] is True
    assert kwargs["output_suffix"] == ".star"
    assert kwargs["write_qa_manifest"] is False


def test_process_default_weights_downloads(runner, patched_batch, tmp_path):
    """Omitting ``--weights`` builds ``Ballet()`` with no model_file (downloads)."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(cli.main, ["process", str(frame)])

    assert result.exit_code == 0, result.output
    assert patched_batch["ballet"] is None
    # Defaults: no metadata -> {}, fail_fast off, qa manifest on, output dir ".".
    kwargs = patched_batch["process"]["kwargs"]
    assert kwargs["user_specific_metadata"] == {}
    assert kwargs["fail_fast"] is False
    assert kwargs["write_qa_manifest"] is True
    assert patched_batch["prepare"]["append_l4"] is False


def test_process_instrument_override(runner, patched_batch, tmp_path):
    """``--instrument`` selects the profile carried on the config."""
    frame = tmp_path / "a.fit"
    frame.write_bytes(b"")

    result = runner.invoke(
        cli.main, ["process", str(frame), "--instrument", "Seestar50"]
    )

    assert result.exit_code == 0, result.output
    config = patched_batch["prepare"]["config"]
    assert config.instrument.name == "Seestar50"


def test_process_profile_file_override(runner, patched_batch, tmp_path):
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
    config = patched_batch["prepare"]["config"]
    assert config.instrument.name == "MyScope"


def test_process_no_files_errors(runner, tmp_path):
    """A directory with no FITS frames is a clean error, not a crash."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = runner.invoke(cli.main, ["process", str(empty)])

    assert result.exit_code != 0
    assert "no fits" in result.output.lower()


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
    """An unknown instrument name exits non-zero with a message."""
    result = runner.invoke(cli.main, ["instrument", "show", "NoSuchScope"])

    assert result.exit_code != 0
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

    assert result.exit_code != 0
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


def test_main_help_lists_commands(runner):
    """``bandaid --help`` lists all four top-level commands."""
    result = runner.invoke(cli.main, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("process", "instrument", "config", "weights"):
        assert command in result.output
