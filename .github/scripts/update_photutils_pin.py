"""
Move the photutils nightly pin in pyproject.toml.

Run from the repository root with ``unpin`` (turn ``==`` into ``>=`` so that
``uv lock --upgrade-package photutils`` can move it) or ``pin`` (write the
version that ended up in ``uv.lock`` back as an ``==`` pin). A local version
label (``+g<sha>``) is only valid with ``==``, so ``unpin`` drops it and
reports the full old version for ``pin --old-version``.
"""

import argparse
import os
import re
import sys
import tomllib
from pathlib import Path

PIN_RE = re.compile(
    r'^(?P<head>\s*"photutils)(?P<op>==|>=)(?P<ver>[^"\s]+)(?P<tail>".*)$'
)


def read_pin(lines, ops):
    """
    Locate the single photutils dependency line.

    Parameters
    ----------
    lines : list of str
        Lines of ``pyproject.toml``.
    ops : tuple of str
        Specifier operators the line is allowed to use.

    Returns
    -------
    tuple of (int, re.Match)
        Index of the line and its regex match.
    """
    found = [
        (i, m)
        for i, line in enumerate(lines)
        if (m := PIN_RE.match(line)) and m["op"] in ops
    ]
    if len(found) != 1:
        names = "/".join(ops)
        msg = f"Expected one photutils{names} line in pyproject.toml"
        sys.exit(f"{msg}, found {len(found)}")
    return found[0]


def rewrite(path, ops, new_op, new_version=None):
    """
    Rewrite the photutils dependency line in place.

    Parameters
    ----------
    path : pathlib.Path
        The ``pyproject.toml`` to edit.
    ops : tuple of str
        Specifier operators the existing line may use.
    new_op : str
        Operator to write.
    new_version : str, optional
        Version to write; the existing version, minus any ``+`` local label
        (invalid with ``>=``), is kept if not given.

    Returns
    -------
    str
        The version that was in the specifier before rewriting.
    """
    lines = path.read_text().splitlines(keepends=True)
    i, m = read_pin(lines, ops)
    version = new_version or m["ver"].split("+")[0]
    lines[i] = f"{m['head']}{new_op}{version}{m['tail']}\n"
    path.write_text("".join(lines))
    return m["ver"]


def short_sha(version):
    """
    Extract the short commit SHA from a nightly version.

    Parameters
    ----------
    version : str
        Version such as ``3.0.1.dev1404+ga6acb06a2``.

    Returns
    -------
    str
        The SHA, or an empty string if the version has no ``+g`` segment.
    """
    m = re.search(r"\+g([0-9a-f]+)", version)
    return m[1] if m else ""


def locked_version(lock_path):
    """
    Read the resolved photutils version from ``uv.lock``.

    Parameters
    ----------
    lock_path : pathlib.Path
        The ``uv.lock`` file.

    Returns
    -------
    str
        The locked version.
    """
    packages = tomllib.loads(lock_path.read_text())["package"]
    versions = [p["version"] for p in packages if p["name"] == "photutils"]
    if len(versions) != 1:
        sys.exit(f"Expected one photutils entry in uv.lock, found {len(versions)}")
    return versions[0]


def emit(outputs):
    """
    Print outputs and append them to ``$GITHUB_OUTPUT`` when it is set.

    Parameters
    ----------
    outputs : dict of str to str
        Output names and values.
    """
    text = "".join(f"{k}={v}\n" for k, v in outputs.items())
    sys.stdout.write(text)
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as f:
            f.write(text)


def main():
    """Parse arguments and run the requested subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["unpin", "pin"])
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument(
        "--old-version",
        help="Full version from before `unpin`; overrides the one in pyproject.toml.",
    )
    args = parser.parse_args()
    pyproject = args.root / "pyproject.toml"

    if args.command == "unpin":
        old = rewrite(pyproject, ("==",), ">=")
        emit({"old_version": old})
        return

    new = locked_version(args.root / "uv.lock")
    old = rewrite(pyproject, ("==", ">="), "==", new)
    old = args.old_version or old
    outputs = {
        "old_version": old,
        "new_version": new,
        "old_sha": short_sha(old),
        "new_sha": short_sha(new),
        "changed": str(old != new).lower(),
    }
    emit(outputs)


if __name__ == "__main__":
    main()
