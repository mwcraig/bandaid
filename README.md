# Welcome to bandaid

|         |                                                                                                                                                                                                                                                                                                                                                                         |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Package | [![Latest PyPI Version](https://img.shields.io/pypi/v/bandaid.svg)](https://pypi.org/project/bandaid/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/bandaid.svg)](https://pypi.org/project/bandaid/) [![Documentation](https://readthedocs.org/projects/bandaid/badge/?version=latest)](https://bandaid.readthedocs.io/en/latest/?badge=latest) |
| Meta    | [![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)                                                                                                                                                                                                                                                 |

*TODO: the above badges that indicate python version and package version will only work if your package is on PyPI.
If you don't plan to publish to PyPI, you can remove them.*

bandaid is a project that (describe what it does here).

## Get started

You can install this package into your preferred Python environment using pip:

```bash
$ pip install bandaid
```

TODO: Add a brief example of how to use the package to this section

To use bandaid in your code:

```python
>>> import bandaid
>>> bandaid.hello_world()
```

## Command-line usage

Installing bandaid also provides a `bandaid` command for photometering a night of
frames without writing any Python:

```bash
# Photometer every FITS frame in a directory, writing .star files + a QA manifest
$ bandaid process night-of-2026-06-27/ -o out/

# Inspect instruments and configuration
$ bandaid instrument list
$ bandaid config init -o config.json

# Fetch the default Ballet centroider weights
$ bandaid weights
```

See the [command-line usage guide](docs/command_line.md) for every subcommand and
option.

## Data-quality flags

Photometry tables carry per-star boolean flags so untrustworthy measurements can
be identified downstream:

- `centroid_drift` — the star's measured centroid wandered too far from its
    aligned/expected position (bad WCS, too-faint star, or an obstruction). See the
    [centroid-drift check](docs/centroid_drift_check.md). Currently flag-only (no
    rows are dropped).
- `contaminated` — a bright neighbor's PSF wings spill into the aperture;
    `eloy_to_starlist` drops it when present. Not yet wired in: the helper
    `neighbor_contamination_flag` exists but is not called anywhere yet, so no
    `contaminated` column is currently written.

## Development

Code style is enforced with [ruff](https://docs.astral.sh/ruff/) and
[pydoclint](https://jsh9.github.io/pydoclint/), pinned in `.pre-commit-config.yaml`
and the `style` dependency group. The CI `pre-commit` job runs:

```bash
uvx pre-commit run --all-files
```

Run the same command locally before pushing, or `uvx pre-commit install` to run it
automatically on every commit. See the [code style guide](docs/code_style.md) for the
linting policy and the individual commands.

## Copyright

- Copyright © 2026 AAVSO.
- Free software distributed under the [MIT License](./LICENSE).
