# Code style

`bandaid` enforces formatting and linting with
[ruff](https://docs.astral.sh/ruff/) and docstring checks with
[pydoclint](https://jsh9.github.io/pydoclint/). The exact versions are pinned in
both `.pre-commit-config.yaml` (the hook `rev`s) and the `style` dependency group in
`pyproject.toml`, so local runs match CI:

- ruff `0.15.15`
- pydoclint `0.8.6`

## Running the checks

The `pre-commit` CI job runs exactly this, and it must pass:

```bash
uvx pre-commit run --all-files
```

To run the checks automatically on every `git commit`:

```bash
uvx pre-commit install
```

The individual tools can also be run directly:

```bash
uvx ruff check .                 # lint
uvx ruff format .                # auto-format
uvx pydoclint --config=pyproject.toml src/ tests/
```

## Linting policy

- **ruff** selects the full rule set (`select = ["ALL"]`) and then ignores a
  curated list, configured under `[tool.ruff.lint]` in `pyproject.toml`.
- **Type annotations are optional.** The missing-annotation rules
  (`ANN001`, `ANN201`, `ANN202`, `ANN206`) are ignored, and pydoclint is configured
  with `arg-type-hints-in-signature = false`; parameter types live in the
  numpy-style docstrings instead.
- **The formatter owns quotes and trailing commas** (`Q000`–`Q004`, `COM812`,
  `COM819` are ignored), per the
  [ruff formatter guidance](https://docs.astral.sh/ruff/formatter/#conflicting-lint-rules).
- **Research notebooks (`*.ipynb`) are not linted or formatted** — they are excluded
  in `[tool.ruff]`.
- **Standalone research / CLI scripts** (the eloy ballet generator and training
  scripts, plus `eval_realistic_weights.py`, `full_pipeline_stwg_t_cr_bor.py`, and
  `image2sl_qt.py`) carry targeted per-file-ignores for conventions that are normal
  there: progress `print`s, lazy imports, and commented-out reference code.
