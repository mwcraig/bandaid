# Working on bandaid

bandaid does photometry on smart-telescope frames: it plate-solves each frame
against Gaia, measures the Gaia stars, and writes AAVSO star lists. Setup,
tasks, pre-commit, code style and test conventions are in
`docs/developers/contributing.md`; read it before changing code. This file
covers what that page does not.

## Branches, forks and pull requests

- Never commit to `main`. Work on a feature branch.
- If the person you are working for cannot push to `mwcraig/bandaid`, work from
    their fork: `origin` is the fork, `upstream` is `mwcraig/bandaid`, and pull
    requests go to `upstream`'s `main`. If the checkout has no fork yet, set one
    up (`gh repo fork --remote`) before the first commit rather than asking the
    maintainer how to proceed.
- Stage files by name. Do not use `git add -A` or `git add .`; checkouts here
    often hold untracked data and scratch files.
- A user-visible change gets a short entry under `[Unreleased]` in
    `CHANGELOG.md`.
- Text you post to GitHub (issues, pull requests, review replies) says it was
    written by Claude at the direction of the person you are working for.

## Issue first, or pull request first

Open a pull request directly when the change needs nothing bandaid lacks: a new
bundled profile, a documentation fix, a bug fix with a test.

Open an issue first, and wait for the maintainer's answer, when the change needs
a design decision: a new profile field, a new configuration option, a new
processing step, or a change to what an existing field means. One issue per
question, so each can be settled on its own.

Do not ship a workaround as the supported path. If a telescope only runs after
overriding settings that are not profile fields, the bundled profile alone will
not work for other users. Say so in an issue, with the overrides and the
numbers that made them necessary, and leave them out of the documentation until
the maintainer decides.

## Reporting a problem or a result

Give the bandaid commit, the telescope and its settings (exposure, gain setting,
firmware), the number of frames, and the numbers you observed. Say which
numbers you measured and which you assumed.

## Adding a telescope

Follow `docs/howto/add_a_telescope.md`; the profile format is in
`docs/reference/instrument_profiles.md`.

Every value in a profile is measured, not copied from a specification sheet or
another telescope's profile, and the pull request states how each was measured
and on what data. A wrong `egain` produces plausible star lists with wrong
errors, so nothing downstream will flag it.

Before writing the profile, work through "Before you start" in the how-to on
real frames. The decisions behind that list:

- A profile covers one recommended camera gain setting and exposure, not a
    table of gains. `egain` is measured at that setting.
- Depth is what a profile author measures: the faintest Gaia magnitude at which
    photometry is usable at the recommended gain and exposure. Do not tune
    `gaia_row_limit` for a telescope.
- bandaid does not calibrate frames and will not: image calibration is out of
    scope, so do not add it or propose it. For a telescope that saves raw
    frames, the user calibrates them before bandaid runs.

The profile cannot yet record the timestamp convention, the faint magnitude
limit or `centroid.max_plane_rms_pix`; all three are planned as profile fields.
Until they exist, report the values you found, and set what you need for your
own runs in a file passed with `--config`. That file is not part of the
contribution.

A profile that needs a field bandaid does not have yet waits for the field.
The field is added first, in its own pull request, and the profile's pull
request is opened on top of it.

If anything else you find has no place in the profile, that is a design
question: open an issue.

## Code and tests

- Write the test first and watch it fail for the right reason before writing
    the fix.
- Do not put pull request or issue numbers in code or comments. They belong in
    the commit message and the changelog.
- In a numpydoc docstring, anything beyond the one-line summary and a short
    description goes in the `Notes` section.
- Do not add guards or tests for inputs that cannot occur in real data.
- Every `bandaid process` example in the documentation carries
    `--user-metadata`, because the output format requires the observer and site.
- Run pre-commit with `uv run poe lint`. Outside the project environment, pin
    the interpreter: `uvx --python 3.12 pre-commit run --all-files`.
