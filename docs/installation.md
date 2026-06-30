# Installation

```bash
$ pip install bandaid
```

That single command is enough to get the `bandaid` command on your path and the
Python package importable. bandaid needs **Python ≥ 3.12**, so install it into a
recent virtual environment (a `venv`, conda env, or whatever you already use).

```python
>>> import bandaid
>>> bandaid.available_instruments()
['Seestar50']
```

## What gets pulled in automatically

Two of bandaid's dependencies are installed straight from public Git
repositories rather than from PyPI; pip handles them for you, but they explain
why the first install is not instant:

- **`eloy[jax]`** — the Ballet CNN centroider that refines star positions. The
    `[jax]` extra brings in JAX, which is the largest single download.
- **`aavso-starlist-schema`** — the schema behind the `.star` output files.

Both are public, so no credentials are required.

## The Ballet weights (first run)

The Ballet centroider needs trained weights. The **first** time you run a
photometry batch, bandaid downloads the default weights from HuggingFace and the
HuggingFace hub caches them, so subsequent runs reuse the cached copy with no
network access.

To pre-fetch the weights (handy before going offline, or to confirm the download
works) use the `weights` command, which prints the cached path:

```bash
$ bandaid weights
/Users/you/.cache/huggingface/.../ballet_weights.npz
```

If you trained or downloaded your own weights, point any run at them with
`--weights` (CLI) or `weights=` (Python) — see
[Training the Ballet centroider](training_the_ballet_centroider.md).

## Developer / editable install

Contributing to bandaid (or running the tests and docs) means a checkout and an
editable install:

```bash
$ git clone https://github.com/mwcraig/bandaid
$ cd bandaid
$ pip install -e .
```

The project also uses [Hatch](https://hatch.pypa.io/) environments for the test
and docs workflows; see [Contributing](contributing.md) and the
[code style guide](code_style.md) for the developer toolchain.
