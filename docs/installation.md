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
