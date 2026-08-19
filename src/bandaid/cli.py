"""
The ``bandaid`` command-line interface.

A thin glue layer over the existing photometry functions so an observer can
photometer a night of frames and inspect instruments/config without writing
Python.
The heavy lifting lives in :mod:`bandaid.scripts`, :mod:`bandaid.instruments`,
and :mod:`bandaid.config`; this module only parses arguments and handles I/O.

The four command groups are:

* ``bandaid process`` -- photometer a batch of frames (the main command).
* ``bandaid instrument list`` / ``show`` -- inspect instrument profiles.
* ``bandaid config init`` / ``validate`` -- create and check a photometry config.
* ``bandaid weights`` -- fetch/print the default Ballet centroider weights.

The names ``photometer_frames``, ``download_weights``, ``available_instruments``,
``load_instrument``, ``InstrumentProfile``, and ``PhotometryConfig`` are imported
into this module's namespace so the network/heavy ones can be monkeypatched in
tests. The file-expansion and ``prepare_batch`` -> ``process_batch`` flow lives
in :func:`bandaid.scripts.photometer_frames`; this module only turns flags into a
config + metadata and delegates to it.
"""

import json
import logging
import shutil
from pathlib import Path

import astropy.units as u
import click
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pydantic import ValidationError

from .ballet import download_weights
from .config import InstrumentProfile, PhotometryConfig, SourceSelectionConfig
from .instruments import available_instruments, load_instrument
from .logging_setup import configure_logging
from .scripts import QA_MANIFEST_FILENAME, photometer_frames
from .writers import get_writer

__all__ = ["main"]

#: ``-v`` count at or above which ``process`` logs at DEBUG rather than INFO.
_DEBUG_VERBOSITY = 2


def _override_source_selection(config, *, gaia_mag_limit, min_snr):
    """
    Apply ``--gaia-mag-limit``/``--min-snr`` overrides to ``config.source_selection``.

    Rebuilds the sub-config from *all* of its current fields (robust if more are
    added later) with just the given overrides applied, so an unset field --
    e.g. a ``--config`` file's ``contaminant_mag_offset`` -- carries forward
    unchanged. A no-op (returns ``config`` unchanged) when both overrides are None.

    Parameters
    ----------
    config : PhotometryConfig
        The config to layer the override onto.
    gaia_mag_limit : float or None
        Value passed to ``--gaia-mag-limit``; None leaves the field untouched.
    min_snr : float or None
        Value passed to ``--min-snr``; None leaves the field untouched.

    Returns
    -------
    PhotometryConfig
        ``config``, or a copy with ``source_selection`` replaced.

    Raises
    ------
    click.ClickException
        If the resulting `~bandaid.config.SourceSelectionConfig` fails validation.
    """
    overrides = {
        k: v
        for k, v in {"gaia_mag_limit": gaia_mag_limit, "min_snr": min_snr}.items()
        if v is not None
    }
    if not overrides:
        return config

    base = config.source_selection.model_dump()
    try:
        new_source_selection = SourceSelectionConfig(**{**base, **overrides})
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    return config.model_copy(update={"source_selection": new_source_selection})


def _build_config(
    instrument, profile, config_file, *, gaia_mag_limit=None, min_snr=None
):
    """
    Build the `PhotometryConfig` for a run from the instrument/config options.

    ``--config`` supplies the full config; ``--instrument`` or ``--profile`` then
    override only its instrument (the frozen config is copied, not mutated).
    ``--gaia-mag-limit``/``--min-snr`` then override only those fields of the
    resulting ``source_selection``, carrying its other fields (e.g. a
    ``--config`` file's ``contaminant_mag_offset``) forward unchanged. With no
    options a default `PhotometryConfig` (Seestar50) is returned.

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
    gaia_mag_limit : float or None, optional
        Value passed to ``--gaia-mag-limit``; None (default) leaves
        ``source_selection.gaia_mag_limit`` untouched.
    min_snr : float or None, optional
        Value passed to ``--min-snr``; None (default) leaves
        ``source_selection.min_snr`` untouched.

    Returns
    -------
    PhotometryConfig
        The configuration to carry through the batch.

    Raises
    ------
    click.ClickException
        If ``--instrument`` and ``--profile`` are given together, a named
        instrument cannot be resolved, or a ``--config``/``--profile``/source-
        selection override fails validation.
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

    return _override_source_selection(
        config, gaia_mag_limit=gaia_mag_limit, min_snr=min_snr
    )


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


#: Columns every forced-targets table must carry (after lowercasing). ``name``
#: is accepted too but is optional -- see `_load_forced_targets`.
_FORCED_TARGET_REQUIRED_COLUMNS = ("ra", "dec")

#: Upper (exclusive) bound of a valid forced-targets ``ra`` in degrees.
_RA_DEGREES_UPPER_BOUND = 360


def _forced_target_column_degrees(table, column_name, path):
    """
    Return one forced-targets column (``ra`` or ``dec``) as degrees, float64.

    Parameters
    ----------
    table : astropy.table.Table
        The forced-targets table, with column names already lowercased.
    column_name : str
        Column to read -- ``"ra"`` or ``"dec"``.
    path : str
        Original ``--forced-targets`` path, used only in error messages.

    Returns
    -------
    numpy.ndarray
        The column's values in degrees, one per row; a formerly-masked cell
        reads as NaN.

    Raises
    ------
    click.ClickException
        If the column has a unit that is not convertible to an angle, or its
        values cannot be read as numbers at all.

    Notes
    -----
    A masked/blank cell is filled with NaN first -- reading a `MaskedColumn`
    straight through ``np.asarray(..., dtype=float)`` silently drops the mask
    and reads a blank cell as 0.0, which this must not do. A column carrying
    an explicit angular unit (e.g. an ECSV column in ``hourangle``) is then
    converted through ``Quantity.to_value``; a unitless column keeps the
    documented "ICRS degrees" interpretation.
    """
    column = table[column_name]
    if getattr(column, "mask", None) is not None and np.any(column.mask):
        column = column.astype(float).filled(np.nan)

    if column.unit is not None:
        try:
            return np.asarray(column.quantity.to_value(u.deg), dtype=float)
        except u.UnitConversionError as exc:
            msg = (
                f"--forced-targets {path!r} column {column_name!r} has a unit "
                f"that is not an angle ({column.unit}): {exc}"
            )
            raise click.ClickException(msg) from exc

    try:
        return np.asarray(column, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = (
            f"--forced-targets {path!r} column {column_name!r} has "
            f"non-numeric values: {exc}"
        )
        raise click.ClickException(msg) from exc


def _load_forced_targets(path):
    """
    Load user-supplied forced-photometry targets for the batch.

    These are extra sky positions to photometer that are absent from the Gaia
    catalog (e.g. a nova or supernova), forwarded to
    `~bandaid.scripts.photometer_frames` as ``forced_targets``.

    Parameters
    ----------
    path : str or None
        Path passed to ``--forced-targets`` holding a CSV/ECSV table with
        required ``ra``/``dec`` columns (ICRS degrees, unless the column
        carries an explicit angular unit, e.g. an ECSV ``hourangle`` column)
        and an optional ``name`` column, or None.

    Returns
    -------
    astropy.coordinates.SkyCoord or None
        The forced-target sky positions in the ICRS frame, or None when
        ``--forced-targets`` is omitted.

    Raises
    ------
    click.ClickException
        If the file cannot be read as a table, its column names collide
        case-insensitively, it is missing a required column, it has no rows,
        a ``ra``/``dec`` column has a non-angular unit or non-numeric values,
        any row's ``ra``/``dec`` is non-finite, ``ra`` is outside
        ``[0, 360)`` degrees, or ``dec`` is outside the valid latitude range.

    Notes
    -----
    Column names are matched case-insensitively (``RA``, ``Ra``, and ``ra``
    are all the same column; a file defining more than one spelling of the
    same column is rejected as ambiguous rather than silently picking one).
    ``name`` is an optional, input-side self-documentation column -- the
    ``.star`` output schema has no name/ID field, so forced rows are
    identified in the output only by their ra/dec.
    """
    if path is None:
        return None
    try:
        table = Table.read(path)
    except Exception as exc:
        msg = f"could not read --forced-targets {path!r}: {exc}"
        raise click.ClickException(msg) from exc

    # Column names are matched case-insensitively; a file defining more than
    # one spelling of the same column (e.g. both "RA" and "ra") is rejected
    # rather than silently picking one of them.
    by_lower_name = {}
    for column in table.colnames:
        by_lower_name.setdefault(column.lower(), []).append(column)
    collided = sorted(
        original
        for originals in by_lower_name.values()
        if len(originals) > 1
        for original in originals
    )
    if collided:
        msg = (
            f"--forced-targets {path!r} has column names that collide "
            f"case-insensitively: {', '.join(collided)}"
        )
        raise click.ClickException(msg)
    table.rename_columns(table.colnames, [name.lower() for name in table.colnames])

    missing = [
        column
        for column in _FORCED_TARGET_REQUIRED_COLUMNS
        if column not in table.colnames
    ]
    if missing:
        msg = (
            f"--forced-targets {path!r} is missing required column(s): "
            f"{', '.join(missing)}"
        )
        raise click.ClickException(msg)

    if len(table) == 0:
        msg = f"--forced-targets {path!r} has no rows"
        raise click.ClickException(msg)

    ra = _forced_target_column_degrees(table, "ra", path)
    dec = _forced_target_column_degrees(table, "dec", path)

    non_finite_rows = np.flatnonzero(~np.isfinite(ra) | ~np.isfinite(dec)) + 1
    if len(non_finite_rows):
        rows = ", ".join(str(row) for row in non_finite_rows)
        msg = f"--forced-targets {path!r} has non-finite ra/dec at row(s): {rows}"
        raise click.ClickException(msg)

    out_of_range_rows = np.flatnonzero((ra < 0) | (ra >= _RA_DEGREES_UPPER_BOUND)) + 1
    if len(out_of_range_rows):
        rows = ", ".join(str(row) for row in out_of_range_rows)
        msg = (
            f"--forced-targets {path!r} has ra outside [0, 360) degrees at "
            f"row(s): {rows}"
        )
        raise click.ClickException(msg)

    # dec's valid range ([-90, 90]) is left to SkyCoord's own Latitude check.
    try:
        return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    except (TypeError, ValueError) as exc:
        msg = f"--forced-targets {path!r} has invalid ra/dec values: {exc}"
        raise click.ClickException(msg) from exc


@click.group()
@click.version_option(package_name="bandaid")
def main():
    """Photometer Smart Telescope frames and inspect instruments/config."""


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
    "--gaia-mag-limit",
    default=None,
    type=float,
    help="Magnitude limit for the photometry targets (overrides the config).",
)
@click.option(
    "--min-snr",
    default=None,
    type=float,
    help="Minimum SNR a star must have to reach the output (overrides the config).",
)
@click.option(
    "--weights",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to Ballet centroider weights; omit to download the defaults.",
)
@click.option(
    "--user-metadata",
    "metadata_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON object of per-frame user-specific metadata.",
)
@click.option(
    "--forced-targets",
    "forced_targets_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a CSV/ECSV table (required columns ra, dec in ICRS degrees; "
        "column names are case-insensitive; an optional name column is "
        "accepted and ignored) of extra targets to photometer that are "
        "absent from the Gaia catalog."
    ),
)
@click.option(
    "--append-l4/--no-append-l4",
    default=True,
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
    "--output-format",
    default="starlist",
    show_default=True,
    help="Name of a registered output writer (e.g. starlist).",
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
    help="Write a per-frame QA manifest alongside the per-frame output files.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Show per-frame progress in the terminal (-vv for debug detail).",
)
@click.option(
    "--log-file",
    "log_file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Also write log records to this file (same level as the terminal).",
)
def process(
    files,
    output_dir,
    instrument,
    profile,
    config_file,
    gaia_mag_limit,
    min_snr,
    weights,
    metadata_file,
    forced_targets_file,
    append_l4,
    fail_fast,
    output_format,
    output_suffix,
    qa_manifest,
    verbose,
    log_file,
):
    """
    Photometer a batch of FITS frames into per-frame .star photometry files.

    FILES may be directories (expanded to their FITS frames), glob patterns, or
    individual frame paths. The first frame seeds the once-per-batch preparation
    and every frame is then photometered against it.

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
    gaia_mag_limit : float or None
        Magnitude limit for the photometry targets; overrides
        ``source_selection.gaia_mag_limit`` when given.
    min_snr : float or None
        Minimum SNR a star must have to reach the output; overrides
        ``source_selection.min_snr`` when given.
    weights : str or None
        Path to Ballet centroider weights; None downloads the defaults.
    metadata_file : str or None
        Path to a JSON object of per-frame user-specific metadata.
    forced_targets_file : str or None
        Path to a CSV/ECSV table (columns ``name``, ``ra``, ``dec`` in
        degrees) of extra targets to photometer that are absent from the
        Gaia catalog.
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
    log_file : str or None
        Path to also write log records to, at the same level as the terminal.

    Raises
    ------
    click.ClickException
        If the arguments expand to no FITS frames, a path argument is missing or
        not a FITS frame, a config/profile/metadata/forced-targets file or a
        ``--gaia-mag-limit``/``--min-snr`` override fails validation, or every
        frame in the batch fails.
    """
    # Always route bandaid's records to stderr so per-frame skip/error warnings
    # are never silently lost: WARNING+ (skips, unexpected errors) shows even
    # with no -v. -v adds INFO per-frame progress; -vv adds DEBUG detail.
    if verbose >= _DEBUG_VERBOSITY:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    configure_logging(level=level, logfile=log_file)

    config = _build_config(
        instrument,
        profile,
        config_file,
        gaia_mag_limit=gaia_mag_limit,
        min_snr=min_snr,
    )
    metadata = _load_metadata(metadata_file)
    forced_targets = _load_forced_targets(forced_targets_file)
    # Resolve the output format up front so an unknown name fails before any
    # (expensive) frame processing, as a clean CLI error rather than a traceback.
    try:
        write_frame = get_writer(output_format)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # The file expansion + prepare/process flow lives in
    # scripts.photometer_frames; surface its argument errors (no frames, bad
    # path) as clean CLI errors.
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
            forced_targets=forced_targets,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Processed {len(results)} of {len(frames)} frames into {output_dir}")
    if qa_manifest:
        click.echo(f"QA manifest: {Path(output_dir) / QA_MANIFEST_FILENAME}")
    if frames and not results:
        # 0 of N succeeded: a fully failed batch must not exit 0, or an
        # unattended/cron run is indistinguishable from success. A partial
        # failure (some results) is normal robust-mode operation and still
        # exits 0; see the per-frame warnings on stderr for what was skipped.
        msg = f"all {len(frames)} frames failed; see the QA manifest for details"
        raise click.ClickException(msg)


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
