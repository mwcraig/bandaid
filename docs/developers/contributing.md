# Contributing

## Set up

```bash
$ git clone https://github.com/mwcraig/bandaid
$ cd bandaid
$ uv sync
```

`uv sync` installs the package editable together with the `dev` dependency group
(tests, docs, style, types); git must be installed, since two dependencies are
fetched from GitHub. The `jax` and `photutils-dev` extras are opt-in:
`uv sync --extra jax` or `uv sync --extra photutils-dev` (the latter swaps in
the pinned photutils development commit; `photutils-release` is only the other
side of that conflict and never needs requesting).
Day-to-day tasks are [poe](https://poethepoet.natn.io/) tasks defined in
`pyproject.toml`:

| Task              | Runs                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `poe test`        | `pytest` — the fast suite, no network                                                     |
| `poe test-remote` | `pytest -m remote_data --remote-data=any` — downloads the Ballet weights from HuggingFace |
| `poe lint`        | `pre-commit run --all-files` — everything CI's lint job runs                              |
| `poe docs`        | `mkdocs serve` — live-reload docs at localhost:8000                                       |
| `poe docs-build`  | `mkdocs build --strict` — one-shot build, as the docs deployment does                     |

Run them as `uv run poe <task>`; `uv run pytest` is the same as `poe test` and is
what CI runs, on Python 3.12 and 3.13.

## Pre-commit

The `pre-commit` CI job must pass. `uv run poe lint` runs it with the project's
interpreter. To run it outside the environment, pin the interpreter, because the
pydoclint hook needs Python ≥ 3.10 and `uvx` may otherwise pick an older system
Python:

```bash
$ uvx --python 3.12 pre-commit run --all-files
$ uvx --python 3.12 pre-commit install    # run on every commit
```

The hooks:

- **pre-commit-hooks**: the standard file checks (JSON, TOML, YAML and Python
    syntax; merge conflicts, large files, whitespace, debug statements).
- **ruff**: `ruff-check` (with `--fix`) and `ruff-format`.
- **pydoclint**: docstring checks.
- **blacken-docs**: formats Python code blocks in the docs.
- **codespell**: spelling.
- **check-jsonschema**: `check-github-workflows` and `check-readthedocs`.
- **mdformat**: Markdown formatting, with the GFM, frontmatter, footnote and
    mkdocs plugins.

Tool versions are pinned in `.pre-commit-config.yaml` and the `style` dependency
group in `pyproject.toml`, so local runs match CI.

## Code style

- **ruff** selects every rule (`select = ["ALL"]`) and ignores a curated list under
    `[tool.ruff.lint]` in `pyproject.toml`.
- **Type annotations are optional.** The missing-annotation rules are ignored and
    pydoclint runs with `arg-type-hints-in-signature = false`; parameter types live
    in numpydoc docstrings. Run `uvx pydoclint --config=pyproject.toml src/ tests/`
    to check them.
- **The formatter owns quotes and trailing commas**, so the conflicting lint rules
    (`Q000`–`Q004`, `COM812`, `COM819`) are ignored.
- **Notebooks (`*.ipynb`) are not linted or formatted.** Commit them without
    outputs.
- **Standalone research and training scripts** carry per-file ignores in
    `pyproject.toml` for conventions normal there, such as progress `print`s.

## Tests

- `pytest` with no options is the whole offline suite: a test that reaches the
    network without the `remote_data` mark fails. Warnings are errors, and
    pytest-randomly shuffles the test order, so a test must not depend on another.
- Add tests to the existing test file for the module you are changing, reusing its
    fixtures, rather than creating a new file.
- Patch with `pytest-mock`'s `mocker`, not `monkeypatch`, and patch the name where
    it is *looked up*: `mocker.patch("bandaid.scripts.prepare_batch")`. Prefer
    `assert_called_once_with` or `.call_args` over a hand-rolled recorder.
- `monkeypatch` remains correct for environment variables and the working
    directory.
- Tests that need the network are marked `remote_data` and run only with
    `poe test-remote`.

## Bundling a telescope profile

To ship a profile with the package, follow
[Add a telescope](../howto/add_a_telescope.md).
