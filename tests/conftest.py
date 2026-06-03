import pytest
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_model_image, make_noise_image


@pytest.fixture
def make_test_image():
    """
    Factory fixture to create test images with Gaussian sources and optional noise.

    Returns
    -------
    callable
        A function that builds a test image; see its docstring for parameters.
    """

    def _make_test_image(
        image_size,
        source_properties,
        *,
        include_noise=True,
        noise_mean=0,
        noise_stddev=1,
        seed=None,
    ):
        """
        Create a test image with Gaussian sources and optional noise.

        Parameters
        ----------
        image_size : tuple
            Size of the test image (ny, nx).
        source_properties : astropy.table.Table
            Table containing properties of the Gaussian source (amplitude, x_mean,
            y_mean, x_stddev, y_stddev).
        include_noise : bool
            Whether to include Gaussian noise in the test image.
        noise_mean : float
            Mean of the Gaussian noise to be added to the image.
        noise_stddev : float
            Standard deviation of the Gaussian noise to be added to the image.
        seed : int, optional
            Random seed for reproducibility of the noise.

        Returns
        -------
        numpy.ndarray
            The generated test image.
        """
        # Create a Gaussian2D model for the source; photutils will scale
        # this appropriately based on source properties.
        model = Gaussian2D(
            x_stddev=1,
            y_stddev=1,
        )

        # Create an image of the Gaussian source
        source_image = make_model_image(
            image_size,
            model,
            source_properties,
            x_name="x_mean",
            y_name="y_mean",
        )

        if not include_noise:
            return source_image

        # Create a noise image
        noise_image = make_noise_image(
            image_size,
            mean=noise_mean,
            stddev=noise_stddev,
            seed=seed,
        )

        # Combine the source and noise to create the final test image
        return source_image + noise_image

    return _make_test_image
