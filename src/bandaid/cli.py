"""
The ``bandaid`` command-line interface.

A thin glue layer over the existing photometry functions so an observer can
reduce a night of frames and inspect instruments/config without writing Python.
The heavy lifting lives in :mod:`bandaid.scripts`, :mod:`bandaid.instruments`,
and :mod:`bandaid.config`; this module only parses arguments and handles I/O.

The four command groups are:

* ``bandaid process`` -- reduce a batch of frames (the main command).
* ``bandaid instrument list`` / ``show`` -- inspect instrument profiles.
* ``bandaid config init`` / ``validate`` -- create and check a photometry config.
* ``bandaid weights`` -- fetch/print the default Ballet centroider weights.

The names ``Ballet``, ``download_weights``, ``prepare_batch``,
``process_batch``, ``available_instruments``, ``load_instrument``,
``InstrumentProfile``, and ``PhotometryConfig`` are imported into this module's
namespace so the network/heavy ones can be monkeypatched in tests.
"""

import glob
import json
import shutil
from pathlib import Path

import click
from eloy.ballet.model import Ballet, download_weights
from pydantic import ValidationError

from .config import InstrumentProfile, PhotometryConfig
from .instruments import available_instruments, load_instrument
from .scripts import QA_MANIFEST_FILENAME, prepare_batch, process_batch

# FITS extensions used when expanding a directory argument to its frames. Seestar
# writes ``.fit``; the others are accepted for telescopes that use them.
_FITS_SUFFIXES = frozenset({".fit", ".fits", ".fts"})


def _expand_files(paths):
    """
    Expand the positional ``FILES`` arguments to a sorted list of frame paths.

    Each argument may be a directory (expanded to the FITS frames it contains), a
    glob pattern (expanded against the filesystem), or a literal path (kept as
    is). The combined result is de-duplicated and sorted so a batch is processed
    in a deterministic order regardless of how the shell passed the arguments.

    Parameters
    ----------
    paths : collections.abc.Iterable of str
        The raw positional arguments: directories, globs, and/or file paths.

    Returns
    -------
    list of str
        The expanded, de-duplicated, lexically sorted frame paths.
    """
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                str(child)
                for child in path.iterdir()
                if child.suffix.lower() in _FITS_SUFFIXES
            )
        elif glob.has_magic(raw):
            files.extend(glob.glob(raw))  # noqa: PTH207 -- need glob-pattern support
        else:
            files.append(raw)
    return sorted(set(files))


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
        If ``--instrument`` and ``--profile`` are given together, or a named
        instrument cannot be resolved.
    """
    if instrument is not None and profile is not None:
        msg = "use only one of --instrument and --profile, not both"
        raise click.ClickException(msg)

    if config_file is not None:
        config = PhotometryConfig.model_validate_json(Path(config_file).read_text())
    else:
        config = PhotometryConfig()

    override = None
    if profile is not None:
        override = InstrumentProfile.from_file(profile)
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
    Load the per-frame user metadata for ``process_batch``.

    Parameters
    ----------
    metadata_file : str or None
        Path passed to ``--metadata`` holding a JSON object, or None.

    Returns
    -------
    dict
        The parsed metadata, or an empty dict when ``--metadata`` is omitted.
    """
    if metadata_file is None:
        return {}
    return json.loads(Path(metadata_file).read_text())


@click.group()
@click.version_option(package_name="bandaid")
def main():
    """Reduce Smart Telescope frames and inspect instruments/config."""


@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path())
@click.option(
    "-o",
    "--output-dir",
    default=".",
    type=click.Path(file_okay=False),
    show_default=True,
    help="Directory to write the .star files (and QA manifest) into.",
)
@click.option(
    "--instrument",
    default=None,
    help="Name of a bundled/registered instrument profile (e.g. Seestar50).",
)
@click.option(
    "--profile",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an instrument-profile JSON file (alternative to --instrument).",
)
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a full PhotometryConfig JSON file.",
)
@click.option(
    "--weights",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to Ballet centroider weights; omit to download the defaults.",
)
@click.option(
    "--metadata",
    "metadata_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON object of per-frame user metadata.",
)
@click.option(
    "--append-l4/--no-append-l4",
    default=False,
    show_default=True,
    help="Add a full-frame L4 luminance channel to the Bayer masks.",
)
@click.option(
    "--fail-fast/--no-fail-fast",
    default=False,
    show_default=True,
    help="Re-raise unexpected per-frame errors instead of skipping the frame.",
)
@click.option(
    "--output-suffix",
    default=".star",
    show_default=True,
    help="Suffix for the per-frame output files.",
)
@click.option(
    "--qa-manifest/--no-qa-manifest",
    default=True,
    show_default=True,
    help="Write a per-frame QA manifest alongside the .star files.",
)
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
    output_suffix,
    qa_manifest,
):
    """
    Reduce a batch of FITS frames into per-frame .star photometry files.

    FILES may be directories (expanded to their FITS frames), glob patterns, or
    individual frame paths. The first frame seeds the once-per-batch preparation
    and every frame is then photometered against it.

    Parameters
    ----------
    files : tuple of str
        Positional FITS files, globs, and/or directories to reduce.
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
        Path to a JSON object of per-frame user metadata.
    append_l4 : bool
        Whether to add a full-frame L4 luminance channel to the Bayer masks.
    fail_fast : bool
        Whether to re-raise unexpected per-frame errors instead of skipping.
    output_suffix : str
        Suffix for the per-frame output files.
    qa_manifest : bool
        Whether to write a per-frame QA manifest alongside the ``.star`` files.

    Raises
    ------
    click.ClickException
        If the arguments expand to no FITS frames.
    """
    frames = _expand_files(files)
    if not frames:
        msg = "no FITS frames found in the given files/directories"
        raise click.ClickException(msg)

    config = _build_config(instrument, profile, config_file)
    cnn = Ballet(model_file=weights)
    metadata = _load_metadata(metadata_file)

    prep = prepare_batch(frames[0], cnn=cnn, config=config, append_l4=append_l4)
    results = process_batch(
        frames,
        prep,
        user_specific_metadata=metadata,
        output_dir=output_dir,
        output_suffix=output_suffix,
        fail_fast=fail_fast,
        write_qa_manifest=qa_manifest,
    )

    click.echo(f"Processed {len(results)} of {len(frames)} frames into {output_dir}")
    if qa_manifest:
        click.echo(f"QA manifest: {Path(output_dir) / QA_MANIFEST_FILENAME}")


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
    """
    text = PhotometryConfig().model_dump_json(indent=2)
    if output_file is None:
        click.echo(text)
    else:
        Path(output_file).write_text(text)
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

    The weights are downloaded from HuggingFace on first use and cached
    thereafter; this prints the cached ``.npz`` path so it can be reused with
    ``bandaid process --weights``.

    Parameters
    ----------
    output_file : str or None
        Destination to copy the weights to, in addition to printing the path.
    """
    cached = download_weights()
    if output_file is not None:
        shutil.copy(cached, output_file)
        click.echo(f"Copied default weights to {output_file}")
    else:
        click.echo(str(cached))
