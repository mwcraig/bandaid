"""
Realistic-noise + bayer-aware synthetic data generator for the Ballet CNN.

This module owns the synthetic-data generation that bandaid's CNN-centroid training
scripts depend on. It subclasses eloy's base ``Moffat2D`` to add
``random_realistic_label`` (a realistic noise model -- sky pedestal + Poisson + read
noise, log-uniform SNR -- with optional bayer-residual augmentation that runs the real
``bandaid.bayer_balance_image`` on a padded frame) and its ``_moffat_frame`` helper.

The surrounding training utilities (``TrainState``, ``train_step``, ``eval_step``,
``get_batches``, ``params_to_flat_dict``, ``compute_loss``) are unchanged from eloy and
are re-exported here so callers have a single import surface and never touch
``eloy.ballet.training`` directly. eloy's jitted ``train_step``/``eval_step`` close over a
module-global ``model``; ``bind_model`` sets exactly that global, so re-exporting the
functions and binding through ``bind_model`` stay consistent.
"""

import eloy.ballet.training as _eloy_training
import numpy as np
from eloy.ballet.training import (
    Moffat2D as _EloyMoffat2D,
)
from eloy.ballet.training import (
    TrainState,
    compute_loss,
    eval_step,
    get_batches,
    params_to_flat_dict,
    train_step,
)

__all__ = [
    "Moffat2D",
    "TrainState",
    "bind_model",
    "compute_loss",
    "eval_step",
    "get_batches",
    "params_to_flat_dict",
    "train_step",
]


def bind_model(cnn):
    """
    Bind the model that eloy's jitted train/eval steps close over.

    eloy's ``train_step``/``eval_step``/``compute_loss`` reference a module-global
    ``model`` in ``eloy.ballet.training``. The training scripts must set it before
    training; this helper does so without the scripts reaching into eloy internals.

    Parameters
    ----------
    cnn : eloy.ballet.model.CNN
        The model instance to bind.

    Returns
    -------
    eloy.ballet.model.CNN
        The same ``cnn`` instance, for convenient one-line binding.
    """
    _eloy_training.model = cnn
    return cnn


class Moffat2D(_EloyMoffat2D):
    """
    Moffat 2D generator with a realistic-noise + bayer-aware sampler.

    Extends eloy's :class:`eloy.ballet.training.Moffat2D` with
    :meth:`random_realistic_label` (and the :meth:`_moffat_frame` helper it uses). The
    base clean generator (``random_model_label``), the Moffat evaluator
    (``moffat2D_model``), and ``sigma_to_fwhm`` are inherited unchanged.
    """

    def _moffat_frame(self, frame, amp, cx, cy, sx, sy, theta, b, beta):
        """
        Evaluate a Moffat on a frame x frame grid, matching ``moffat2D_model``'s
        coordinate convention exactly: ``self.x, self.y = np.indices(...)`` makes the
        first centre arg the *row* index and the second the *column* -- so ``cx`` here is
        the row coordinate, like ``x0`` in ``moffat2D_model``. (Getting this backwards
        swaps the label vs the image for bayer samples relative to the plain path.)
        """
        row, col = np.indices((frame, frame))
        dx_ = row - cx
        dy_ = col - cy
        dxr = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dyr = -dx_ * np.sin(theta) + dy_ * np.cos(theta)
        return b + amp / np.power(1 + (dxr / sx) ** 2 + (dyr / sy) ** 2, beta)

    def random_realistic_label(
        self,
        N=10000,
        snr_range=(1.5, 200.0),
        sky_range=(20.0, 500.0),
        read_noise_range=(2.0, 10.0),
        sigma=1.0,
        bayer_frac=0.0,
        balance_fn=None,
        channel_color_max=0.3,
        gen_pad=13,
        return_all=False,
    ):
        """
        Generate Moffat cutouts with a realistic noise model (+ optional bayer residual).

        Unlike ``random_model_label`` (amplitude fixed at 1, background 0, uniform
        noise <= 0.1, so every star has peak SNR >~ 10), this adds a sky pedestal,
        Poisson shot noise, and Gaussian read noise, with peak SNR sampled
        log-uniformly across ``snr_range``. That populates the faint, noise-dominated
        regime the clean generator never reaches.

        The CNN min-max normalizes each cutout internally (see ``model.CNN.__call__``),
        so absolute counts are irrelevant -- only the relative noise (sigma/peak) and
        PSF shape matter. SNR is therefore the controlling axis and is sampled in log
        space so faint and bright stars are represented evenly. Sky level and read
        noise are randomized per sample so the network sees a range of background
        regimes rather than one fixed pedestal.

        Bayer augmentation (``bayer_frac`` > 0): a fraction of samples reproduce the
        residual 2-px checkerboard a one-shot-colour sensor leaves after balancing,
        matching production. Per sample one per-CFA-channel gain (a colour latent: R up /
        B down, neutral included) multiplies the *whole* clean frame (Moffat + sky)
        together -- i.e. star and sky share the channel response, exactly the model the
        production/eval path uses; the residual then arises from `balance_fn`'s
        per-channel (sky-derived) noise normalization. The raw frame is built on a padded
        grid, `balance_fn` (e.g. `bandaid.bayer_balance_image`) is run on the *full* frame
        as in production, then the central cutout is cropped so balance statistics are not
        dominated by the star. The CFA corner phase is randomized per sample for
        invariance to the actual pattern. The neutral case (gains ~ 1, no residual) is
        included so white stars / clean data are not hurt. (The measured background
        imbalance on real frames is transient *sky colour*, not a fixed gain, so the gain
        amplitude is sampled, never hardcoded.)

        Parameters
        ----------
        N : int, optional
            Number of samples to generate.
        snr_range, sky_range, read_noise_range : tuple of float, optional
            (min, max) peak SNR (log-uniform), sky pedestal, and read noise (uniform).
        sigma : float, optional
            Std of the center coordinate distribution about the cutout center.
        bayer_frac : float, optional
            Fraction of samples that get the bayer-residual treatment (0 disables it,
            and leaves the noise-only path byte-for-byte unchanged).
        balance_fn : callable or None
            In-place background-balancing function applied to the full padded frame for
            bayer samples (inject ``bandaid.bayer_balance_image``). Required if
            ``bayer_frac`` > 0 -- approximating it defeats the purpose.
        channel_color_max : float, optional
            Max per-channel colour deviation (R up / B down by this fraction) of the
            single gain applied to star+sky together. ~0.3 spans the measured real
            spread; neutral (no residual) is always included.
        gen_pad : int, optional
            Padding added on each side for bayer samples so ``balance_fn`` sees a larger
            frame (cutout_size + 2*gen_pad) before the central crop.
        return_all : bool, optional
            If True, labels are (amp, x0, y0, sx, sy, theta, sky, beta, snr); else (x0, y0).

        Returns
        -------
        tuple
            (images, labels) with images of shape (N, cutout_size, cutout_size, 1).
        """
        if bayer_frac > 0 and balance_fn is None:
            raise ValueError(
                "bayer_frac > 0 requires balance_fn (e.g. bandaid.bayer_balance_image)",
            )
        cs = self.cutout_size
        x0, y0 = np.random.normal(cs / 2, sigma, (2, N))
        theta = np.random.uniform(0, np.pi / 8, size=N)
        beta = np.random.uniform(1, 8, size=N)
        sx = np.array(
            [np.random.uniform(1.5, 20.5) / self.sigma_to_fwhm(_beta) for _beta in beta],
        )
        sy = np.random.uniform(0.5, 1.5, size=N) * sx

        log_snr = np.random.uniform(
            np.log10(snr_range[0]), np.log10(snr_range[1]), size=N,
        )
        snr = 10.0**log_snr
        sky = np.random.uniform(sky_range[0], sky_range[1], size=N)
        read_noise = np.random.uniform(read_noise_range[0], read_noise_range[1], size=N)
        noise_floor = np.sqrt(sky + read_noise**2)  # background-limited noise std
        amp = snr * noise_floor

        # Bayer-augmentation draws are guarded so bayer_frac=0 leaves the RNG stream
        # (and therefore the noise-only output) identical to before.
        if bayer_frac > 0:
            do_bayer = np.random.random(N) < bayer_frac
            # one per-channel gain (R, G, B) per sample applied to star+sky together;
            # colour latent shifts R up / B down, G ~ 1; neutral (c~0) included.
            c = np.random.uniform(-1, 1, N) * channel_color_max
            g_g = 1.0 + np.random.uniform(-0.05, 0.05, N)
            chan = np.clip(np.stack([1 + c, g_g, 1 - c], axis=1), 0.3, None)  # (N, 3)
            phase = np.random.randint(0, 2, size=(N, 2))  # CFA corner phase per sample
            base_cidx = np.array([[0, 1], [1, 2]])  # RGGB colour index: R G / G B
        else:
            do_bayer = np.zeros(N, dtype=bool)

        images = np.empty((N, cs, cs))
        for i in range(N):
            if not do_bayer[i]:
                clean = self.moffat2D_model(
                    amp[i], x0[i], y0[i], sx[i], sy[i], theta[i], sky[i], beta[i],
                )
                images[i] = np.random.poisson(np.clip(clean, 0, None)).astype(
                    float,
                ) + np.random.normal(0.0, read_noise[i], (cs, cs))
                continue

            # Bayer sample: build (Moffat + sky) on a padded frame, apply one per-CFA-
            # channel gain to the whole frame (star+sky share the response, as in
            # production), add noise, balance the full frame, then crop the center.
            F = cs + 2 * gen_pad
            clean = self._moffat_frame(
                F, amp[i], x0[i] + gen_pad, y0[i] + gen_pad,
                sx[i], sy[i], theta[i], sky[i], beta[i],
            )
            cidx = np.roll(base_cidx, (phase[i, 0], phase[i, 1]), axis=(0, 1))
            tile = np.tile(cidx, (F // 2 + 1, F // 2 + 1))[:F, :F]
            raw = clean * chan[i][tile]
            noisy = np.random.poisson(np.clip(raw, 0, None)).astype(
                float,
            ) + np.random.normal(0.0, read_noise[i], (F, F))
            balance_fn(noisy)  # in place, on the full frame, exactly as production
            images[i] = noisy[gen_pad:gen_pad + cs, gen_pad:gen_pad + cs]
        images = images[:, :, :, None]

        if return_all:
            labels = np.array([amp, x0, y0, sx, sy, theta, sky, beta, snr]).T
        else:
            labels = np.array([x0, y0]).T
        return (np.array(images), np.array(labels))
