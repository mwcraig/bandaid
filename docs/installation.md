# Installation

```bash
$ pip install bandaid
```

That single command is enough to get the `bandaid` command on your path and the
Python package importable. bandaid needs **Python ≥ 3.12**, so install it into a
recent virtual environment (a `venv`, conda env, or whatever you already use).

```pycon
>>> import bandaid
>>> bandaid.available_instruments()
['Seestar50']
```

## Developer / editable install

Contributing to bandaid (or running the tests and docs) means a checkout and an
editable install:

```bash
$ git clone https://github.com/mwcraig/bandaid
$ cd bandaid
$ pip install -e .
```

Day-to-day development uses [uv](https://docs.astral.sh/uv/) — `uv sync` builds
the dev environment, and the common workflows are named
[poe](https://poethepoet.natn.io/) tasks defined in `pyproject.toml`:

```bash
$ uv run poe test        # run the test suite
$ uv run poe lint        # run all the pre-commit checks
$ uv run poe docs        # serve the docs with live reload
```

See [Contributing](contributing.md) and the [code style guide](code_style.md)
for the developer toolchain.
