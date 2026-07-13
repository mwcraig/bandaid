"""
The ``bandaid`` command-line interface.

A thin glue layer over the existing photometry functions so an observer can
photometer a night of frames and inspect instruments/config without writing
Python.
The heavy lifting lives in :mod:`bandaid.scripts`, :mod:`bandaid.instruments`,
and :mod:`bandaid.config`; this module only parses arguments and handles I/O.

The five command groups are:

* ``bandaid process`` -- photometer a batch of frames (the main command).
* ``bandaid stream`` -- photometer frames straight from an rclone remote.
* ``bandaid instrument list`` / ``show`` -- inspect instrument profiles.
* ``bandaid config init`` / ``validate`` -- create and check a photometry config.
* ``bandaid weights`` -- fetch/print the default Ballet centroider weights.

The names ``photometer_frames``, ``stream_frames``, ``download_weights``,
``available_instruments``, ``load_instrument``, ``InstrumentProfile``, and
``PhotometryConfig`` are imported into this module's namespace so the
network/heavy ones can be monkeypatched in tests. The file-expansion and
``prepare_batch`` -> ``process_batch`` flow lives in
:func:`bandaid.scripts.photometer_frames` (and its streaming counterpart
:func:`bandaid.streaming.stream_frames`); this module only turns flags into a
config + metadata and delegates to it.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

import click
from eloy.ballet.model import download_weights
from pydantic import ValidationError

from .config import InstrumentProfile, PhotometryConfig
from .exceptions import BatchPrepError, RemoteFetchError
from .instruments import available_instruments, load_instrument
from .logging_setup import configure_logging
from .scripts import QA_MANIFEST_FILENAME, _quiet_hf_xet, photometer_frames
from .streaming import stream_frames
from .writers import get_writer

__all__ = ["main"]

#: ``-v`` count at or above which ``process`` logs at DEBUG rather than INFO.
_DEBUG_VERBOSITY = 2


def _build_config(instrument, profile, config_file):
    """
    Build the `PhotometryConfig` for a run from the instrument/config options.

    ``--config`` supplies the full config; ``--instrument`` or ``--profile`` then
    override only its instrument (the frozen config is copied, not mutated). With
    no options a default `PhotometryConfig` (Seestar50) is returned.

    Parameters
    ----------
    instrument : str or None
        Name passed to ``--instrument``, resolved with
        :func:`~bandaid.instruments.load_instrument`.
    profile : str or None
        Path passed to ``--profile``, loaded with
        :meth:`~bandaid.config.InstrumentProfile.from_file`.
    config_file : str or None
        Path passed to ``--config``, loaded as a full `PhotometryConfig`.

    Returns
    -------
    PhotometryConfig
        The configuration to carry through the batch.

    Raises
    ------
    click.ClickException
        If ``--instrument`` and ``--profile`` are given together, a named
        instrument cannot be resolved, or a ``--config``/``--profile`` file fails
        validation.
    """
    if instrument is not None and profile is not None:
        msg = "use only one of --instrument and --profile, not both"
        raise click.ClickException(msg)

    if config_file is not None:
        # A malformed/invalid config should read as a clean CLI error, matching
        # ``config validate``, not a raw pydantic traceback.
        try:
            config = PhotometryConfig.model_validate_json(Path(config_file).read_text())
        except ValidationError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        config = PhotometryConfig()

    override = None
    if profile is not None:
        try:
            override = InstrumentProfile.from_file(profile)
        except ValidationError as exc:
            raise click.ClickException(str(exc)) from exc
    elif instrument is not None:
        try:
            override = load_instrument(instrument)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    if override is not None:
        config = config.model_copy(update={"instrument": override})
    return config


def _load_metadata(metadata_file):
    """
    Load the per-frame user-specific metadata for the batch.

    This is the ``user_specific_metadata`` recorded with every frame's output
    (the observer-identity layer threaded through `process_batch`).

    Parameters
    ----------
    metadata_file : str or None
        Path passed to ``--user-metadata`` holding a JSON object, or None.

    Returns
    -------
    dict
        The parsed metadata, or an empty dict when ``--user-metadata`` is omitted.

    Raises
    ------
    click.ClickException
        If the file is not valid JSON, or is valid JSON that is not an object
        (which would later break when the metadata is merged via ``dict.update``).
    """
    if metadata_file is None:
        return {}
    try:
        data = json.loads(Path(metadata_file).read_text())
    except json.JSONDecodeError as exc:
        msg = f"--user-metadata is not valid JSON: {exc}"
        raise click.ClickException(msg) from exc
    if not isinstance(data, dict):
        msg = "--user-metadata must be a JSON object"
        raise click.ClickException(msg)
    return data


def _resolve_writer(output_format):
    """
    Resolve ``--output-format`` to its registered writer callable.

    Resolved up front so an unknown name fails before any (expensive) frame
    processing, as a clean CLI error rather than a traceback.

    Parameters
    ----------
    output_format : str
        Name passed to ``--output-format``, resolved with
        :func:`~bandaid.writers.get_writer`.

    Returns
    -------
    collections.abc.Callable
        The registered per-frame writer.

    Raises
    ------
    click.ClickException
        If ``output_format`` is not a registered writer name.
    """
    try:
        return get_writer(output_format)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _configure_verbosity(verbose):
    """
    Route bandaid's log records to stderr at the level ``-v`` asks for.

    Always configured -- even with no ``-v`` -- so per-frame skip/error
    warnings are never silently lost: WARNING+ shows by default, ``-v`` adds
    INFO per-frame progress, ``-vv`` adds DEBUG detail.

    Parameters
    ----------
    verbose : int
        The ``-v`` count from the command line.
    """
    if verbose >= _DEBUG_VERBOSITY:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    configure_logging(level=level)


def _report_batch_outcome(frames, results, output_dir, qa_manifest):
    """
    Print the end-of-batch summary and enforce the shared exit contract.

    Parameters
    ----------
    frames : list
        Every frame the batch attempted.
    results : dict
        The successfully processed frames (see
        :func:`bandaid.scripts.process_batch`).
    output_dir : str
        Where the outputs were written, for the summary line.
    qa_manifest : bool
        Whether a QA manifest was written (and so should be pointed at).

    Raises
    ------
    click.ClickException
        If every frame in a non-empty batch failed: 0 of N succeeding must
        not exit 0, or an unattended/cron run is indistinguishable from
        success. A partial failure is normal robust-mode operation and still
        exits 0; see the per-frame warnings on stderr for what was skipped.
    """
    click.echo(f"Processed {len(results)} of {len(frames)} frames into {output_dir}")
    if qa_manifest:
        click.echo(f"QA manifest: {Path(output_dir) / QA_MANIFEST_FILENAME}")
    if frames and not results:
        msg = f"all {len(frames)} frames failed; see the QA manifest for details"
        raise click.ClickException(msg)


@click.group()
@click.version_option(package_name="bandaid")
def main():
    """Photometer Smart Telescope frames and inspect instruments/config."""


#: The batch-processing options ``process`` and ``stream`` share. The two
#: commands differ only in where the frames come from (local paths vs. an
#: rclone remote), so everything downstream of the frame source -- config,
#: metadata, output layout, verbosity -- is declared once here.
_PROCESSING_OPTIONS = (
    click.option(
        "-o",
        "--output-dir",
        default=".",
        type=click.Path(file_okay=False),
        show_default=True,
        help="Directory to write the .star files (and QA manifest) into.",
    ),
    click.option(
        "--instrument",
        default=None,
        help="Name of a bundled/registered instrument profile (e.g. Seestar50).",
    ),
    click.option(
        "--profile",
        default=None,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to an instrument-profile JSON file (alternative to --instrument).",
    ),
    click.option(
        "--config",
        "config_file",
        default=None,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to a full PhotometryConfig JSON file.",
    ),
    click.option(
        "--weights",
        default=None,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to Ballet centroider weights; omit to download the defaults.",
    ),
    click.option(
        "--user-metadata",
        "metadata_file",
        default=None,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to a JSON object of per-frame user-specific metadata.",
    ),
    click.option(
        "--append-l4/--no-append-l4",
        default=True,
        show_default=True,
        help="Add a full-frame L4 luminance channel to the Bayer masks.",
    ),
    click.option(
        "--fail-fast/--no-fail-fast",
        default=False,
        show_default=True,
        help="Re-raise unexpected per-frame errors instead of skipping the frame.",
    ),
    click.option(
        "--output-format",
        default="starlist",
        show_default=True,
        help="Name of a registered output writer (e.g. starlist).",
    ),
    click.option(
        "--output-suffix",
        default=".star",
        show_default=True,
        help="Suffix for the per-frame output files.",
    ),
    click.option(
        "--qa-manifest/--no-qa-manifest",
        default=True,
        show_default=True,
        help="Write a per-frame QA manifest alongside the per-frame output files.",
    ),
    click.option(
        "-v",
        "--verbose",
        count=True,
        help="Show per-frame progress in the terminal (-vv for debug detail).",
    ),
)


def _processing_options(command):
    """
    Apply the shared batch-processing options to a command.

    Parameters
    ----------
    command : collections.abc.Callable
        The command function being decorated.

    Returns
    -------
    collections.abc.Callable
        The command with every option in `_PROCESSING_OPTIONS` attached, in
        the same ``--help`` order as if they had been stacked as decorators.
    """
    # Decorators apply bottom-up, so reversed() reproduces the stacked-
    # decorator parameter order (click re-reverses at command creation).
    for option in reversed(_PROCESSING_OPTIONS):
        command = option(command)
    return command


@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path())
@_processing_options
def process(
    files,
    output_dir,
    instrument,
    profile,
    config_file,
    weights,
    metadata_file,
    append_l4,
    fail_fast,
    output_format,
    output_suffix,
    qa_manifest,
    verbose,
):
    """
    Photometer a batch of FITS frames into per-frame .star photometry files.

    FILES may be directories (expanded to their FITS frames), glob patterns, or
    individual frame paths. The first frame seeds the once-per-batch preparation
    and every frame is then photometered against it.
    \f

    Parameters
    ----------
    files : tuple of str
        Positional FITS files, globs, and/or directories to photometer.
    output_dir : str
        Directory to write the per-frame ``.star`` files (and QA manifest) into.
    instrument : str or None
        Name of a bundled/registered instrument profile to use.
    profile : str or None
        Path to an instrument-profile JSON file (alternative to ``instrument``).
    config_file : str or None
        Path to a full `PhotometryConfig` JSON file.
    weights : str or None
        Path to Ballet centroider weights; None downloads the defaults.
    metadata_file : str or None
        Path to a JSON object of per-frame user-specific metadata.
    append_l4 : bool
        Whether to add a full-frame L4 luminance channel to the Bayer masks.
    fail_fast : bool
        Whether to re-raise unexpected per-frame errors instead of skipping.
    output_format : str
        Name of a registered output writer to record each frame with.
    output_suffix : str
        Suffix for the per-frame output files.
    qa_manifest : bool
        Whether to write a per-frame QA manifest alongside the ``.star`` files.
    verbose : int
        Verbosity count from ``-v``: 0 logs only WARNING+ (skips/errors) to
        stderr, 1 adds per-frame progress at INFO, 2+ adds DEBUG detail.

    Raises
    ------
    click.ClickException
        If the arguments expand to no FITS frames, a path argument is missing or
        not a FITS frame, a config/profile/metadata file fails validation, the
        once-per-batch preparation fails, or every frame in the batch fails.
    """  # noqa: D301 -- the \f is click's marker truncating --help here.
    _configure_verbosity(verbose)

    config = _build_config(instrument, profile, config_file)
    metadata = _load_metadata(metadata_file)
    write_frame = _resolve_writer(output_format)

    # The file expansion + prepare/process flow lives in
    # scripts.photometer_frames; surface its argument errors (no frames, bad
    # path) and a fatal first-frame prep failure as clean CLI errors.
    try:
        frames, results = photometer_frames(
            files,
            config=config,
            weights=weights,
            user_specific_metadata=metadata,
            append_l4=append_l4,
            output_dir=output_dir,
            output_suffix=output_suffix,
            write_frame=write_frame,
            fail_fast=fail_fast,
            write_qa_manifest=qa_manifest,
        )
    except (ValueError, FileNotFoundError, BatchPrepError) as exc:
        raise click.ClickException(str(exc)) from exc

    _report_batch_outcome(frames, results, output_dir, qa_manifest)


@main.command()
@click.argument("remote")
@_processing_options
@click.option(
    "--incoming",
    "incoming_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Local staging directory for the downloads; omit for a temp dir.",
)
@click.option(
    "--keep/--no-keep",
    "keep_local",
    default=False,
    show_default=True,
    help=(
        "Keep each frame's local copy instead of deleting it after processing. "
        "Pair with --incoming to choose where the kept frames land."
    ),
)
@click.option(
    "--download-workers",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of concurrent rclone downloads.",
)
def stream(
    remote,
    output_dir,
    instrument,
    profile,
    config_file,
    weights,
    metadata_file,
    append_l4,
    fail_fast,
    output_format,
    output_suffix,
    qa_manifest,
    verbose,
    incoming_dir,
    keep_local,
    download_workers,
):
    """
    Photometer FITS frames straight from an rclone remote.

    REMOTE is an rclone remote path, e.g. "gdrive:My Frames". Each frame is
    downloaded just before it is processed and deleted afterwards, so a batch
    far larger than the local disk can be photometered; the remote is never
    modified. Requires rclone (https://rclone.org) with a configured remote.
    \f

    Parameters
    ----------
    remote : str
        The rclone remote path holding the frames.
    output_dir : str
        Directory to write the per-frame ``.star`` files (and QA manifest) into.
    instrument : str or None
        Name of a bundled/registered instrument profile to use.
    profile : str or None
        Path to an instrument-profile JSON file (alternative to ``instrument``).
    config_file : str or None
        Path to a full `PhotometryConfig` JSON file.
    weights : str or None
        Path to Ballet centroider weights; None downloads the defaults.
    metadata_file : str or None
        Path to a JSON object of per-frame user-specific metadata.
    append_l4 : bool
        Whether to add a full-frame L4 luminance channel to the Bayer masks.
    fail_fast : bool
        Whether to re-raise unexpected per-frame errors instead of skipping.
    output_format : str
        Name of a registered output writer to record each frame with.
    output_suffix : str
        Suffix for the per-frame output files.
    qa_manifest : bool
        Whether to write a per-frame QA manifest alongside the ``.star`` files.
    verbose : int
        Verbosity count from ``-v``, as for ``process``.
    incoming_dir : str or None
        Local staging directory for the downloads; None uses a temp dir.
    keep_local : bool
        Whether to keep each frame's local copy after its outcome is decided.
    download_workers : int
        Number of concurrent rclone downloads.

    Raises
    ------
    click.ClickException
        If rclone is not installed, the ``--incoming`` directory cannot be
        created, the remote cannot be listed or holds no FITS frames, the
        first frame cannot be fetched, the once-per-batch preparation fails, a
        config/profile/metadata file fails validation, or every frame in the
        batch fails.
    """  # noqa: D301 -- the \f is click's marker truncating --help here.
    _configure_verbosity(verbose)

    # Fail the missing-tool case up front, before any config parsing or
    # (expensive) weights download, with a pointer instead of a cryptic
    # FileNotFoundError from subprocess.
    if shutil.which("rclone") is None:
        msg = (
            "rclone not found on PATH; install it and configure a remote "
            "(https://rclone.org) to stream frames"
        )
        raise click.ClickException(msg)

    # Likewise fail a bad staging directory (an existing file, an unwritable
    # parent) up front as a clean CLI error, not a raw OSError traceback from
    # deep inside stream_frames.
    if incoming_dir is not None:
        try:
            Path(incoming_dir).mkdir(parents=True, exist_ok=True)
        except OSError as err:
            msg = f"could not create the --incoming directory {incoming_dir}: {err}"
            raise click.ClickException(msg) from err

    config = _build_config(instrument, profile, config_file)
    metadata = _load_metadata(metadata_file)
    write_frame = _resolve_writer(output_format)

    try:
        names, results = stream_frames(
            remote,
            config=config,
            weights=weights,
            user_specific_metadata=metadata,
            append_l4=append_l4,
            output_dir=output_dir,
            output_suffix=output_suffix,
            write_frame=write_frame,
            fail_fast=fail_fast,
            write_qa_manifest=qa_manifest,
            incoming_dir=incoming_dir,
            keep_local=keep_local,
            download_workers=download_workers,
        )
    except subprocess.CalledProcessError as exc:
        # rclone itself failed (unknown remote, expired auth): surface its
        # stderr, which names the actual problem, as the CLI error.
        msg = f"rclone failed: {(exc.stderr or '').strip() or exc}"
        raise click.ClickException(msg) from exc
    except (ValueError, RemoteFetchError, BatchPrepError) as exc:
        # An empty remote, a fatal first-frame fetch, or a failed batch prep;
        # all three messages already name the remote/frame/cause.
        raise click.ClickException(str(exc)) from exc

    _report_batch_outcome(names, results, output_dir, qa_manifest)


@main.group()
def instrument():
    """Inspect the instrument profiles the pipeline can resolve."""


@instrument.command(name="list")
def instrument_list():
    """List the resolvable instrument-profile names."""
    for name in available_instruments():
        click.echo(name)


@instrument.command(name="show")
@click.argument("name")
def instrument_show(name):
    """
    Print one instrument profile's settings as JSON.

    Parameters
    ----------
    name : str
        The instrument-profile name to show.

    Raises
    ------
    click.ClickException
        If ``name`` is not a resolvable instrument.
    """
    try:
        profile = load_instrument(name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(profile.model_dump_json(indent=2))


@main.group()
def config():
    """Create and validate photometry configuration files."""


@config.command(name="init")
@click.option(
    "-o",
    "--output",
    "output_file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Write the config here instead of to standard output.",
)
def config_init(output_file):
    """
    Write a default `PhotometryConfig` for the user to edit.

    Parameters
    ----------
    output_file : str or None
        Destination path; None prints the config to standard output.

    Raises
    ------
    click.ClickException
        If the config cannot be written to ``output_file``.
    """
    text = PhotometryConfig().model_dump_json(indent=2)
    if output_file is None:
        click.echo(text)
    else:
        # A bad destination (unwritable path, missing parent) should read as a
        # clean CLI error, not a raw OSError traceback.
        try:
            Path(output_file).write_text(text)
        except OSError as exc:
            msg = f"could not write config to {output_file}: {exc}"
            raise click.ClickException(msg) from exc
        click.echo(f"Wrote default config to {output_file}")


@config.command(name="validate")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
def config_validate(config_file):
    """
    Parse and validate a photometry config file.

    Parameters
    ----------
    config_file : str
        Path to the `PhotometryConfig` JSON file to validate.

    Raises
    ------
    click.ClickException
        If the file fails `PhotometryConfig` validation.
    """
    try:
        PhotometryConfig.model_validate_json(Path(config_file).read_text())
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{config_file} is a valid PhotometryConfig")


@main.command()
@click.option(
    "-o",
    "--output",
    "output_file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Copy the weights here in addition to printing their path.",
)
def weights(output_file):
    """
    Print (and optionally copy) the default Ballet centroider weights.

    The weights are downloaded from HuggingFace on first use; caching is handled
    by the HuggingFace hub cache (under ``HF_HOME``/``~/.cache/huggingface``), not
    by bandaid, so this simply prints the cached ``.npz`` path for reuse with
    ``bandaid process --weights``.

    Parameters
    ----------
    output_file : str or None
        Destination to copy the weights to, in addition to printing the path.

    Raises
    ------
    click.ClickException
        If the weights cannot be copied to ``output_file``.
    """
    _quiet_hf_xet()
    cached = download_weights()
    if output_file is not None:
        # A bad destination (unwritable path, missing parent, full disk) should
        # read as a clean CLI error, not a raw OSError traceback.
        try:
            shutil.copy(cached, output_file)
        except OSError as exc:
            msg = f"could not copy weights to {output_file}: {exc}"
            raise click.ClickException(msg) from exc
        click.echo(f"Copied default weights to {output_file}")
    else:
        click.echo(str(cached))
