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

## Optional jax acceleration

The base install is numpy-only on purpose: bandaid stays importable in
environments where jax isn't available at all, such as the browser
(Pyodide). If you're running on a regular machine and want faster
centroiding, install the `jax` extra:

```bash
$ pip install bandaid[jax]
```

or, with uv:

```bash
$ uv add bandaid[jax]
```

This pulls in eloy's jax/flax Ballet model, which bandaid uses automatically
once both jax and flax are importable. Centroiding is roughly 2--3x faster
than the numpy path on CPU (measured on batches of 50--900 star cutouts), and
agrees with it to float32 round-off, so the choice is a speed one, not a
science one.

Set `BANDAID_BALLET_BACKEND=numpy` to force the numpy path even where jax is
installed, or `BANDAID_BALLET_BACKEND=jax` to make a missing jax an error
instead of a silent fallback. Either way, the backend actually chosen is
logged once at INFO when the centroider is built. See
[Training the Ballet centroider](training_the_ballet_centroider.md) for more
on how the centroider backend is chosen.

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
