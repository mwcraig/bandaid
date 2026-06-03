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
