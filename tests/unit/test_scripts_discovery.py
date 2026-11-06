"""Unit tests for frame-path discovery and the photometer-frames entry point."""

import dataclasses
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from _helpers import _batch_metadata, _dummy_prep, _stub_load_frame
from astropy.coordinates import SkyCoord

from bandaid import scripts
from bandaid.config import (
    HeaderMatchRule,
    InstrumentProfile,
    PhotometryConfig,
)
from bandaid.exceptions import BatchPrepError
from bandaid.instruments import StarFlagger, load_instrument, register_star_flagger


class TestExpandFramePaths:
    """Unit tests for the ``expand_frame_paths`` file-name convenience."""

    def test_directory_expands_to_sorted_fits_only(self, tmp_path):
        """A directory yields its FITS frames, sorted, with non-FITS dropped."""
        night = tmp_path / "night"
        night.mkdir()
        for name in ["c.fit", "a.fit", "b.fit"]:
            (night / name).write_bytes(b"")
        (night / "notes.txt").write_text("not a frame")

        result = scripts.expand_frame_paths([str(night)])

        assert result == sorted(str(p.resolve()) for p in night.glob("*.fit"))

    def test_glob_matches_are_filtered_to_fits(self, tmp_path):
        """A wildcard that catches non-FITS files keeps only the FITS ones."""
        night = tmp_path / "night"
        night.mkdir()
        (night / "a.fit").write_bytes(b"")
        (night / "b.txt").write_text("not a frame")

        result = scripts.expand_frame_paths([str(night / "*")])

        assert result == [str((night / "a.fit").resolve())]

    def test_missing_literal_raises_file_not_found(self, tmp_path):
        """A non-existent literal path fails fast instead of reaching prepare_batch."""
        with pytest.raises(FileNotFoundError):
            scripts.expand_frame_paths([str(tmp_path / "nope.fits")])

    def test_non_fits_literal_raises_value_error(self, tmp_path):
        """An existing literal that is not a FITS frame is rejected with a message."""
        other = tmp_path / "foo.txt"
        other.write_text("not a frame")
        with pytest.raises(ValueError, match="FITS"):
            scripts.expand_frame_paths([str(other)])

    def test_same_name_in_different_dirs_kept_distinct(self, tmp_path):
        """Identically named frames in different directories are NOT collapsed."""
        n1 = tmp_path / "n1"
        n2 = tmp_path / "n2"
        n1.mkdir()
        n2.mkdir()
        (n1 / "img.fit").write_bytes(b"")
        (n2 / "img.fit").write_bytes(b"")

        result = scripts.expand_frame_paths([str(n1), str(n2)])

        expected = sorted(
            [str((n1 / "img.fit").resolve()), str((n2 / "img.fit").resolve())]
        )
        assert result == expected

    def test_same_file_referenced_two_ways_is_deduplicated(self, tmp_path):
        """The same file reached via a directory and an explicit path appears once."""
        night = tmp_path / "night"
        night.mkdir()
        frame = night / "a.fit"
        frame.write_bytes(b"")

        result = scripts.expand_frame_paths([str(night), str(frame)])

        assert result == [str(frame.resolve())]

    def test_compressed_fits_extension_accepted(self, tmp_path):
        """A compressed ``.fits.gz`` frame is recognised as a FITS frame."""
        frame = tmp_path / "a.fits.gz"
        frame.write_bytes(b"")

        assert scripts.expand_frame_paths([str(frame)]) == [str(frame.resolve())]

    def test_directory_with_fits_named_subdir_is_skipped(self, tmp_path):
        """A sub-directory whose name ends in a FITS suffix is not a frame."""
        night = tmp_path / "night"
        night.mkdir()
        (night / "a.fit").write_bytes(b"")
        (night / "bundle.fits").mkdir()  # a directory, not a frame

        result = scripts.expand_frame_paths([str(night)])

        assert result == [str((night / "a.fit").resolve())]

    def test_glob_matching_fits_named_dir_is_skipped(self, tmp_path):
        """A glob that catches a FITS-named directory keeps only real files."""
        night = tmp_path / "night"
        night.mkdir()
        (night / "a.fit").write_bytes(b"")
        (night / "bundle.fit").mkdir()

        result = scripts.expand_frame_paths([str(night / "*.fit")])

        assert result == [str((night / "a.fit").resolve())]

    def test_literal_non_regular_file_raises_value_error(self, tmp_path):
        """A literal FITS-named path that is not a regular file is rejected."""
        fifo = tmp_path / "pipe.fits"
        os.mkfifo(fifo)  # exists, ends in .fits, but is not a frame
        with pytest.raises(ValueError, match="FITS"):
            scripts.expand_frame_paths([str(fifo)])


class TestPhotometerFrames:
    """Unit tests for the high-level ``photometer_frames`` convenience entry point."""

    def test_expands_builds_cnn_and_wires_both_steps(self, mocker, tmp_path):
        """It expands the args, builds the CNN from weights, and threads both steps."""
        night = tmp_path / "night"
        night.mkdir()
        for name in ["b.fit", "a.fit"]:
            (night / name).write_bytes(b"")
        weights = tmp_path / "w.npz"
        weights.write_bytes(b"npz")

        cnn_sentinel = object()
        expected = sorted(str(p.resolve()) for p in night.glob("*.fit"))
        # A batch prepared for the bundled Seestar50, whose core-replaced
        # flagger builds its state from the first frames of the batch and
        # reads its saturation ceiling from the prep frame's metadata.
        prep_sentinel = dataclasses.replace(
            self._prep(
                PhotometryConfig(instrument=load_instrument("Seestar50")),
                auto_detected=True,
            ),
            prep_metadata={**_batch_metadata(), "height": 4, "width": 4},
        )
        # A header that detects to the bundled Seestar50.
        frame_sentinel = scripts.LoadedFrame(
            np.zeros((4, 4)), {"INSTRUME": "Seestar S50"}
        )

        def fake_process(files, _prep, **_kwargs: object):
            # `files` is already the plain list expand_frame_paths built
            # (scripts.py:1114,1124), so a mock reading it straight from
            # .call_args is fine -- no generator to materialize here.
            return {f: f + ".star" for f in files}

        load_frame = mocker.patch(
            "bandaid.scripts._load_frame", return_value=frame_sentinel
        )
        ballet = mocker.patch("bandaid.scripts.Ballet", return_value=cnn_sentinel)
        prepare_batch = mocker.patch(
            "bandaid.scripts.prepare_batch", return_value=prep_sentinel
        )
        process_batch = mocker.patch(
            "bandaid.scripts.process_batch", side_effect=fake_process
        )

        config = PhotometryConfig()
        frames, results = scripts.photometer_frames(
            [str(night)],
            config=config,
            weights=str(weights),
            user_specific_metadata={"observer": "MWC"},
            build_l4=True,
            output_dir=str(tmp_path / "out"),
            output_suffix=".sl",
            fail_fast=True,
            write_manifest=False,
        )

        assert frames == expected
        ballet.assert_called_once_with(model_file=str(weights))
        prepare_batch.assert_called_once_with(
            expected,
            cnn=cnn_sentinel,
            config=config,
            build_l4=True,
            forced_targets=None,
            frame=frame_sentinel,
            user_specific_metadata={"observer": "MWC"},
        )
        process_call = process_batch.call_args
        # process_batch gets the prep with the star-flag state built from both
        # frames of the batch; the first is the same load as the prep frame.
        assert process_call.args[0] == expected
        assert set(process_call.args[1].star_flag_state) == {"seestar_core_replaced"}
        assert process_call.args[1].radecs is prep_sentinel.radecs
        # The first frame is opened once and the same load handed to both
        # stages and the state; the second is read once for the state.
        assert [call.args[0] for call in load_frame.call_args_list] == expected
        kwargs = process_call.kwargs
        assert kwargs["first_frame"] is frame_sentinel
        assert kwargs["user_specific_metadata"] == {"observer": "MWC"}
        assert kwargs["output_dir"] == str(tmp_path / "out")
        assert kwargs["output_suffix"] == ".sl"
        assert kwargs["fail_fast"] is True
        assert kwargs["write_manifest"] is False
        assert results == {f: f + ".star" for f in expected}

    def test_no_frames_raises_value_error(self, tmp_path):
        """An argument set that expands to nothing is a clean ValueError."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no FITS"):
            scripts.photometer_frames([str(empty)])

    def test_defaults_download_weights_and_build_l4(self, mocker, tmp_path):
        """Omitting options downloads weights, appends L4, and uses robust defaults."""
        frame = tmp_path / "a.fit"
        frame.write_bytes(b"")

        _stub_load_frame(mocker)
        ballet = mocker.patch("bandaid.scripts.Ballet", return_value=object())
        prepare_batch = mocker.patch(
            "bandaid.scripts.prepare_batch", return_value=_dummy_prep()
        )
        process_batch = mocker.patch("bandaid.scripts.process_batch", return_value={})

        scripts.photometer_frames([str(frame)])

        ballet.assert_called_once_with(model_file=None)
        prepare_kwargs = prepare_batch.call_args.kwargs
        assert prepare_kwargs["build_l4"] is True
        assert isinstance(prepare_kwargs["config"], PhotometryConfig)
        kwargs = process_batch.call_args.kwargs
        assert kwargs["user_specific_metadata"] == {}
        assert kwargs["fail_fast"] is False
        assert kwargs["write_manifest"] is True
        assert kwargs["output_dir"] == "."
        assert kwargs["output_suffix"] == ".star"

    def test_batch_prep_error_is_reraised_with_a_batch_prefix(self, mocker, tmp_path):
        """
        A ``BatchPrepError`` from ``prepare_batch`` surfaces with a batch prefix.

        ``photometer_frames`` catches the fatal error and re-raises it as a
        clear, top-level message (chaining the original as the cause) instead
        of letting a bare ``BatchPrepError`` surface uncaught. The frame the
        failure concerns is named by ``prepare_batch``'s own message, since the
        batch may be prepared from a frame other than the first.
        """
        frame = tmp_path / "a.fit"
        frame.write_bytes(b"")

        _stub_load_frame(mocker)
        mocker.patch("bandaid.scripts.Ballet", return_value=object())
        original = BatchPrepError("no bundled/registered instrument matched")
        mocker.patch("bandaid.scripts.prepare_batch", side_effect=original)

        with pytest.raises(BatchPrepError) as exc_info:
            scripts.photometer_frames([str(frame)])

        assert str(exc_info.value) == (
            "could not prepare the batch: no bundled/registered instrument matched"
        )
        assert exc_info.value.__cause__ is original

    def test_first_frame_handed_to_process_batch_even_if_prep_used_a_later_one(
        self, mocker, tmp_path
    ):
        """
        The preloaded first frame reaches ``process_batch`` whatever seeded the prep.

        ``process_batch`` uses it for position 1 of the batch, so a walked-past
        first frame still reuses the load.
        """
        night = tmp_path / "night"
        night.mkdir()
        for name in ["a.fit", "b.fit"]:
            (night / name).write_bytes(b"")
        frame_sentinel = scripts.LoadedFrame(
            np.zeros((4, 4)), {"INSTRUME": "Seestar S50"}
        )
        mocker.patch("bandaid.scripts._load_frame", return_value=frame_sentinel)
        mocker.patch("bandaid.scripts.Ballet", return_value=object())
        expected = sorted(str(p.resolve()) for p in night.glob("*.fit"))
        prepare_batch = mocker.patch(
            "bandaid.scripts.prepare_batch", return_value=_dummy_prep()
        )
        process_batch = mocker.patch("bandaid.scripts.process_batch", return_value={})

        scripts.photometer_frames([str(night)])

        assert prepare_batch.call_args.args == (expected,)
        assert prepare_batch.call_args.kwargs["frame"] is frame_sentinel
        assert process_batch.call_args.kwargs["first_frame"] is frame_sentinel

    @staticmethod
    def _night_with_state_flagger(
        tmp_path, n_frames, n_state_frames, header_match=()
    ) -> tuple[list, PhotometryConfig]:
        """
        Write ``n_frames`` empty frames and register a stateful toy flagger.

        Returns the sorted frame paths and a config whose instrument names the
        flagger. The flagger's state is the list of frames it was given.
        """
        night = tmp_path / "night"
        night.mkdir()
        for index in range(n_frames):
            (night / f"f{index}.fit").write_bytes(b"")
        register_star_flagger(
            "toy",
            StarFlagger(
                column="core_replaced",
                flag=lambda inputs, _state: {},
                prepare=lambda frames, metadata: (frames, metadata),
                n_state_frames=n_state_frames,
            ),
        )
        instrument = InstrumentProfile(
            name="ToyScope", star_flaggers=("toy",), header_match=header_match
        )
        frames = sorted(str(p.resolve()) for p in night.glob("*.fit"))
        return frames, PhotometryConfig(instrument=instrument)

    @staticmethod
    def _prep(config, shape=(4, 4), *, auto_detected=False) -> scripts.BatchPrep:
        """A ``BatchPrep`` for ``config``'s instrument and a batch of ``shape``."""
        return dataclasses.replace(
            _dummy_prep(),
            config=config,
            shape=shape,
            instrument_auto_detected=auto_detected,
        )

    @staticmethod
    def _patch_steps(mocker, prep, events=None) -> tuple:
        """
        Patch ``Ballet``, ``prepare_batch`` (returning ``prep``) and ``process_batch``.

        Each ``prepare_batch`` call is appended to ``events`` when given.
        Returns the ``prepare_batch`` and ``process_batch`` mocks.
        """

        def prepare(*_args: object, **_kwargs: object):
            if events is not None:
                events.append("prepare_batch")
            return prep

        mocker.patch("bandaid.scripts.Ballet", return_value=object())
        prepare_batch = mocker.patch(
            "bandaid.scripts.prepare_batch", side_effect=prepare
        )
        process_batch = mocker.patch("bandaid.scripts.process_batch", return_value={})
        return prepare_batch, process_batch

    @staticmethod
    def _toy_state(process_batch) -> list:
        """The frames the toy flagger's ``prepare`` was given."""
        return process_batch.call_args.args[1].star_flag_state["toy"][0]

    def test_state_gets_the_prep_frames_metadata(self, mocker, tmp_path):
        """The state built after the preparation sees the prep frame's metadata."""
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=2, n_state_frames=2
        )
        mocker.patch(
            "bandaid.scripts._load_frame",
            return_value=scripts.LoadedFrame(np.zeros((4, 4)), {}),
        )
        metadata = {"height": 4, "width": 4, "largest_usable_adu_value": 50000}
        prep = dataclasses.replace(self._prep(config), prep_metadata=metadata)
        _, process_batch = self._patch_steps(mocker, prep)

        scripts.photometer_frames(frames, config=config)

        state = process_batch.call_args.args[1].star_flag_state["toy"]
        assert state[1] is metadata

    def test_loads_the_state_frames_after_the_preparation(self, mocker, tmp_path):
        """
        The first K frames' raw data build the state once the batch is prepared.

        ``prepare_batch`` gets no state frames; the later state frames are read
        only after it returns, so none is held while it runs, and the first
        frame's load is reused.
        """
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=4, n_state_frames=2
        )
        loaded = {
            name: scripts.LoadedFrame(np.full((4, 4), index), {})
            for index, name in enumerate(frames)
        }
        events = []

        def load(name):
            events.append(name)
            return loaded[name]

        mocker.patch("bandaid.scripts._load_frame", side_effect=load)
        prep = self._prep(config)
        prepare_batch, process_batch = self._patch_steps(mocker, prep, events)

        scripts.photometer_frames(frames, config=config)

        assert "state_frames" not in prepare_batch.call_args.kwargs
        assert events == [frames[0], "prepare_batch", frames[1]]
        state = self._toy_state(process_batch)
        assert len(state) == 2  # noqa: PLR2004
        assert state[0] is loaded[frames[0]].data
        assert state[1] is loaded[frames[1]].data
        assert process_batch.call_args.args[1].radecs is prep.radecs

    def test_state_frame_that_fails_to_load_is_skipped(self, mocker, tmp_path, caplog):
        """A state frame that cannot be read is left out, with a log line."""
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=3, n_state_frames=3
        )
        first = scripts.LoadedFrame(np.zeros((4, 4)), {})
        third = scripts.LoadedFrame(np.ones((4, 4)), {})

        def load(name):
            if name == frames[1]:
                msg = "Empty or corrupt FITS file"
                raise OSError(msg)
            return first if name == frames[0] else third

        mocker.patch("bandaid.scripts._load_frame", side_effect=load)
        _, process_batch = self._patch_steps(mocker, self._prep(config))

        with caplog.at_level(logging.WARNING, logger="bandaid.scripts"):
            scripts.photometer_frames(frames, config=config)

        state = self._toy_state(process_batch)
        assert len(state) == 2  # noqa: PLR2004
        assert state[0] is first.data
        assert state[1] is third.data
        assert frames[1] in caplog.text

    def test_truncated_state_frame_is_skipped(self, mocker, tmp_path, caplog):
        """
        A state frame whose load raises ``ValueError`` is left out, not fatal.

        A truncated uncompressed ``.fit`` makes astropy raise ``ValueError``;
        the batch goes on with the other state frames and no refill.
        """
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=4, n_state_frames=3
        )
        first = scripts.LoadedFrame(np.zeros((4, 4)), {})
        third = scripts.LoadedFrame(np.ones((4, 4)), {})

        def load(name):
            if name == frames[1]:
                msg = "buffer is too small for requested array"
                raise ValueError(msg)
            return first if name == frames[0] else third

        load_frame = mocker.patch("bandaid.scripts._load_frame", side_effect=load)
        _, process_batch = self._patch_steps(mocker, self._prep(config))

        with caplog.at_level(logging.WARNING, logger="bandaid.scripts"):
            scripts.photometer_frames(frames, config=config)

        state = self._toy_state(process_batch)
        assert len(state) == 2  # noqa: PLR2004
        assert state[0] is first.data
        assert state[1] is third.data
        assert [call.args[0] for call in load_frame.call_args_list] == frames[:3]
        assert frames[1] in caplog.text
        process_batch.assert_called_once()

    def test_state_frames_must_have_the_batch_shape(self, mocker, tmp_path, caplog):
        """
        Only frames of the prepared batch's shape feed the state.

        The batch can be prepared from a later frame whose shape differs from
        the first frame's; then the first frame, not the later ones, is left
        out, so the state matches the frames the batch processes.
        """
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=3, n_state_frames=3
        )
        loaded = {
            frames[0]: scripts.LoadedFrame(np.zeros((4, 4)), {}),
            frames[1]: scripts.LoadedFrame(np.zeros((8, 8)), {}),
            frames[2]: scripts.LoadedFrame(np.ones((8, 8)), {}),
        }
        mocker.patch("bandaid.scripts._load_frame", side_effect=loaded.__getitem__)
        _, process_batch = self._patch_steps(mocker, self._prep(config, shape=(8, 8)))

        with caplog.at_level(logging.WARNING, logger="bandaid.scripts"):
            scripts.photometer_frames(frames, config=config)

        state = self._toy_state(process_batch)
        assert len(state) == 2  # noqa: PLR2004
        assert state[0] is loaded[frames[1]].data
        assert state[1] is loaded[frames[2]].data
        assert frames[0] in caplog.text

    def test_state_frame_of_another_instrument_is_skipped(
        self, mocker, tmp_path, caplog
    ):
        """
        A frame that fails the batch's instrument check does not feed the state.

        Under an explicit selection a header matching no registered profile is
        admitted, and one that names another registered instrument is not.
        """
        frames, config = self._night_with_state_flagger(
            tmp_path,
            n_frames=3,
            n_state_frames=3,
            header_match=(HeaderMatchRule(keyword="INSTRUME", pattern="Toy"),),
        )
        loaded = {
            frames[0]: scripts.LoadedFrame(np.zeros((4, 4)), {"INSTRUME": "Toy"}),
            frames[1]: scripts.LoadedFrame(
                np.zeros((4, 4)), {"INSTRUME": "Seestar S50"}
            ),
            frames[2]: scripts.LoadedFrame(np.ones((4, 4)), {}),
        }
        mocker.patch("bandaid.scripts._load_frame", side_effect=loaded.__getitem__)
        _, process_batch = self._patch_steps(mocker, self._prep(config))

        with caplog.at_level(logging.WARNING, logger="bandaid.scripts"):
            scripts.photometer_frames(frames, config=config)

        state = self._toy_state(process_batch)
        assert len(state) == 2  # noqa: PLR2004
        assert state[0] is loaded[frames[0]].data
        assert state[1] is loaded[frames[2]].data
        assert frames[1] in caplog.text

    def test_unknown_flagger_fails_the_run(self, mocker, tmp_path):
        """A typo in ``star_flaggers`` fails ``photometer_frames`` with ValueError."""
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=2, n_state_frames=2
        )
        config = config.model_copy(
            update={
                "instrument": config.instrument.model_copy(
                    update={"star_flaggers": ("typo",)}
                )
            }
        )
        mocker.patch(
            "bandaid.scripts._load_frame",
            return_value=scripts.LoadedFrame(np.zeros((4, 4)), {}),
        )
        mocker.patch("bandaid.scripts.Ballet", return_value=object())

        with pytest.raises(ValueError, match="typo"):
            scripts.photometer_frames(frames, config=config)

    def test_failing_prepare_fails_the_run_before_any_frame(self, mocker, tmp_path):
        """A flagger ``prepare`` that raises fails the batch with the prep prefix."""
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=2, n_state_frames=2
        )
        cause = IndexError("index 9 is out of bounds")

        def prepare(_frames, _metadata):
            raise cause

        register_star_flagger(
            "toy",
            StarFlagger(
                column="core_replaced",
                flag=lambda inputs, _state: {},
                prepare=prepare,
                n_state_frames=2,
            ),
            replace=True,
        )
        mocker.patch(
            "bandaid.scripts._load_frame",
            return_value=scripts.LoadedFrame(np.zeros((4, 4)), {}),
        )
        _, process_batch = self._patch_steps(mocker, self._prep(config))

        with pytest.raises(
            BatchPrepError,
            match=(
                r"^could not prepare the batch: star flagger 'toy' failed to "
                r"build its state: index 9 is out of bounds$"
            ),
        ) as excinfo:
            scripts.photometer_frames(frames, config=config)

        assert excinfo.value.__cause__.__cause__ is cause
        process_batch.assert_not_called()

    def test_no_state_frames_loaded_without_a_stateful_flagger(self, mocker, tmp_path):
        """A profile whose flaggers need no state costs no extra frame loads."""
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=3, n_state_frames=0
        )
        load_frame = mocker.patch(
            "bandaid.scripts._load_frame",
            return_value=scripts.LoadedFrame(np.zeros((4, 4)), {}),
        )
        prep = self._prep(config)
        _, process_batch = self._patch_steps(mocker, prep)

        scripts.photometer_frames(frames, config=config)

        assert process_batch.call_args.args[1] is prep
        load_frame.assert_called_once_with(frames[0])

    def test_no_state_frames_loaded_for_a_flagger_without_prepare(
        self, mocker, tmp_path
    ):
        """A flagger asking for frames but with no ``prepare`` costs no loads."""
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=3, n_state_frames=0
        )
        register_star_flagger(
            "toy",
            StarFlagger(
                column="core_replaced", flag=lambda inputs, _state: {}, n_state_frames=3
            ),
            replace=True,
        )
        load_frame = mocker.patch(
            "bandaid.scripts._load_frame",
            return_value=scripts.LoadedFrame(np.zeros((4, 4)), {}),
        )
        prep = self._prep(config)
        _, process_batch = self._patch_steps(mocker, prep)

        scripts.photometer_frames(frames, config=config)

        assert process_batch.call_args.args[1] is prep
        load_frame.assert_called_once_with(frames[0])

    def test_switch_off_loads_no_state_frames(self, mocker, tmp_path):
        """With ``instrument_star_flags`` False only the first frame is loaded."""
        frames, config = self._night_with_state_flagger(
            tmp_path, n_frames=3, n_state_frames=3
        )
        config = config.model_copy(update={"instrument_star_flags": False})
        load_frame = mocker.patch(
            "bandaid.scripts._load_frame",
            return_value=scripts.LoadedFrame(np.zeros((4, 4)), {}),
        )
        prep = self._prep(config)
        _, process_batch = self._patch_steps(mocker, prep)

        scripts.photometer_frames(frames, config=config)

        assert process_batch.call_args.args[1] is prep
        load_frame.assert_called_once_with(frames[0])

    def test_forced_targets_forwarded_to_prepare_batch(self, mocker, tmp_path):
        """The ``forced_targets`` SkyCoord reaches ``prepare_batch`` unchanged."""
        frame = tmp_path / "a.fit"
        frame.write_bytes(b"")

        _stub_load_frame(mocker)
        mocker.patch("bandaid.scripts.Ballet", return_value=object())
        prepare_batch = mocker.patch(
            "bandaid.scripts.prepare_batch", return_value=_dummy_prep()
        )
        mocker.patch("bandaid.scripts.process_batch", return_value={})

        forced = SkyCoord([10.0], [5.0], unit="deg")
        scripts.photometer_frames([str(frame)], forced_targets=forced)

        assert prepare_batch.call_args.kwargs["forced_targets"] is forced

    @pytest.mark.usefixtures("_consistent_headers")
    def test_identical_names_write_distinct_starlists(
        self, mocker, tmp_path, by_filter
    ):
        """End-to-end: two same-named frames in different dirs give distinct outputs."""
        n1 = tmp_path / "n1"
        n2 = tmp_path / "n2"
        n1.mkdir()
        n2.mkdir()
        (n1 / "img.fit").write_bytes(b"")
        (n2 / "img.fit").write_bytes(b"")
        out = tmp_path / "out"
        inputs = [str(n1), str(n2)]

        mocker.patch("bandaid.scripts.Ballet", return_value=object())
        mocker.patch("bandaid.scripts.prepare_batch", return_value=_dummy_prep())
        # A fresh table per call, matching the two-frame batch: process_one_image
        # is called once per frame and each frame gets its own table instance.
        mocker.patch(
            "bandaid.scripts.process_one_image", side_effect=lambda *a, **k: by_filter()
        )

        frames, results = scripts.photometer_frames(inputs, output_dir=str(out))

        # Two source directories => mirrored tree, so each frame lands under its
        # own <dirname>/ subdir instead of overwriting a shared flat name.
        written = sorted(p.relative_to(out) for p in out.rglob("*.star"))
        assert len(frames) == len(inputs)
        assert written == [Path("n1/img.star"), Path("n2/img.star")]
        assert len(results) == len(frames)
        assert len({str(v) for v in results.values()}) == len(results)
