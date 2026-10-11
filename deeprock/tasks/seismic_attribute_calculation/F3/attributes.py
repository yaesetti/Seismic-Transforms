import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def attribute_numpy_envelope(cube: np.ndarray) -> np.ndarray:
    """
    Envelope for 3D seismic cube (x, y, z).
    Computed along z-axis (time/depth).
    """

    cube = np.asarray(cube)
    assert cube.ndim == 3, "Input must be 3D (x, y, z)"

    fft_data = np.fft.fft(cube, axis=2)
    n = cube.shape[2]

    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    h = h.reshape(1, 1, n)

    analytic = np.fft.ifft(fft_data * h, axis=2)

    return np.abs(analytic)


def attribute_numpy_semblance(
    cube: np.ndarray,
    window_xy: int = 3,
    window_z: int = 9,
) -> np.ndarray:
    """
    Fast vectorized semblance for 3D seismic cube (x, y, z).

    Semablance is the V_rms = sqrt(sum(t_i) * V_i^2 / sum(t_i))

    Parameters
    ----------
    cube : np.ndarray
        Shape (x, y, z)
    window_xy : int
        Spatial window size
    window_z : int
        Time/depth window siz

    Returns
    -------
    np.ndarray
        Semblance cube (same shape as input)
    """

    cube = np.asarray(cube)
    assert cube.ndim == 3, "Input must be 3D (x, y, z)"
    assert window_xy % 2 == 1, "window_xy must be odd"
    assert window_z % 2 == 1, "window_z must be odd"

    pad_x = window_xy // 2
    pad_y = window_xy // 2
    pad_z = window_z // 2

    # Reflect padding (same as typical seismic processing)
    padded = np.pad(
        cube, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), mode="reflect"
    )

    # Create sliding windows
    # Result shape:
    # (nx, ny, nz, window_xy, window_xy, window_z)
    windows = sliding_window_view(padded, (window_xy, window_xy, window_z))
    # print(f"Windows shape: {windows.shape}")

    # Compute sums
    sum_vals = np.sum(windows, axis=(-3, -2, -1))
    sum_sq_vals = np.sum(windows**2, axis=(-3, -2, -1))

    # Number of samples in window
    N = window_xy * window_xy * window_z

    # Semblance formula
    semblance = (sum_vals**2) / (N * sum_sq_vals + 1e-12)

    return semblance.astype(np.float32)
