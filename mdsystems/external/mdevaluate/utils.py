"""
Collection of utility functions.
"""
import functools
from time import time as pytime
from subprocess import run
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.ndimage import uniform_filter1d

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from .logging import logger
from .functions import kww, kww_1e


def five_point_stencil(xdata: ArrayLike, ydata: ArrayLike) -> ArrayLike:
    """
    Calculate the derivative dy/dx with a five point stencil.
    This algorith is only valid for equally distributed x values.

    Args:
        xdata: x values of the data points
        ydata: y values of the data points

    Returns:
        Values where the derivative was estimated and the value of the derivative at
        these points.

    This algorithm is only valid for values on a regular grid, for unevenly distributed
    data it is only an approximation, albeit a quite good one.

    See: https://en.wikipedia.org/wiki/Five-point_stencil
    """
    return xdata[2:-2], (
        (-ydata[4:] + 8 * ydata[3:-1] - 8 * ydata[1:-3] + ydata[:-4])
        / (3 * (xdata[4:] - xdata[:-4]))
    )


def filon_fourier_transformation(
    time: NDArray,
    correlation: NDArray,
    frequencies: Optional[NDArray] = None,
    derivative: Union[str, NDArray] = "linear",
    imag: bool = True,
) -> tuple[NDArray, NDArray]:
    """
    Fourier-transformation for slow varying functions. The filon algorithm is
    described in detail in ref [Blochowicz]_, ch. 3.2.3.

    Args:
        time: List of times when the correlation function was sampled.
        correlation: Values of the correlation function.
        frequencies (opt.):
            List of frequencies where the fourier transformation will be calculated.
            If None the frequencies will be chosen based on the input times.
        derivative (opt.):
            Approximation algorithm for the derivative of the correlation function.
            Possible values are: 'linear', 'stencil' or a list of derivatives.
        imag (opt.): If imaginary part of the integral should be calculated.

    If frequencies are not explicitly given, they will be evenly placed on a log scale
    in the interval [1/tmax, 0.1/tmin] where tmin and tmax are the smallest respectively
    the biggest time (greater than 0) of the provided times. The frequencies are cut off
    at high values by one decade, since the fourier transformation deviates quite
    strongly in this regime.

    .. [Blochowicz]
      T. Blochowicz, Broadband dielectric spectroscopy in neat and binary
      molecular glass formers, Ph.D. thesis, Universität Bayreuth (2003)
    """
    if frequencies is None:
        f_min = 1 / max(time)
        f_max = 0.05 ** (1.2 - max(correlation)) / min(time[time > 0])
        frequencies = 2 * np.pi * np.logspace(np.log10(f_min), np.log10(f_max), num=60)
    frequencies.reshape(1, -1)

    if derivative == "linear":
        derivative = (np.diff(correlation) / np.diff(time)).reshape(-1, 1)
    elif derivative == "stencil":
        _, derivative = five_point_stencil(time, correlation)
        time = ((time[2:-1] * time[1:-2]) ** 0.5).reshape(-1, 1)
        derivative = derivative.reshape(-1, 1)
    elif isinstance(derivative, NDArray) and len(time) is len(derivative):
        derivative.reshape(-1, 1)
    else:
        raise NotImplementedError(
            'Invalid approximation method {}. Possible values are "linear", "stencil" '
            "or a list of values."
        )
    time = time.reshape(-1, 1)

    integral = (
        np.cos(frequencies * time[1:]) - np.cos(frequencies * time[:-1])
    ) / frequencies**2
    fourier = (derivative * integral).sum(axis=0)

    if imag:
        integral = (
            1j
            * (np.sin(frequencies * time[1:]) - np.sin(frequencies * time[:-1]))
            / frequencies**2
        )
        fourier = (
            fourier
            + (derivative * integral).sum(axis=0)
            + 1j * correlation[0] / frequencies
        )

    return frequencies.reshape(-1), fourier


def superpose(
    x1: NDArray, y1: NDArray, x2: NDArray, y2: NDArray, damping: float = 1.0
) -> tuple[NDArray, NDArray]:
    if x2[0] == 0:
        x2 = x2[1:]
        y2 = y2[1:]

    reg1 = x1 < x2[0]
    reg2 = x2 > x1[-1]
    x_ol = np.logspace(
        np.log10(np.max(x1[~reg1][0], x2[~reg2][0]) + 0.001),
        np.log10(np.min(x1[~reg1][-1], x2[~reg2][-1]) - 0.001),
        (np.sum(~reg1) + np.sum(~reg2)) / 2,
    )

    def w(x: NDArray) -> NDArray:
        A = x_ol.min()
        B = x_ol.max()
        return (np.log10(B / x) / np.log10(B / A)) ** damping

    xdata = np.concatenate((x1[reg1], x_ol, x2[reg2]))
    y1_interp = interp1d(x1[~reg1], y1[~reg1])
    y2_interp = interp1d(x2[~reg2], y2[~reg2])
    ydata = np.concatenate(
        (
            y1[x1 < x2.min()],
            w(x_ol) * y1_interp(x_ol) + (1 - w(x_ol)) * y2_interp(x_ol),
            y2[x2 > x1.max()],
        )
    )
    return xdata, ydata


def moving_average(data: NDArray, n: int = 3) -> NDArray:
    """
    Compute the running mean of an array.
    Uses the second axis if it is of higher dimensionality.

    Args:
        data: Input data of shape (N, )
        n: Number of points over which the data will be averaged

    Returns:
        Array of shape (N-(n-1), )

    Supports 2D-Arrays.
    """
    k1 = int(n / 2)
    k2 = int((n - 1) / 2)
    if k2 == 0:
        if data.ndim > 1:
            return uniform_filter1d(data, n)[:, k1:]
        return uniform_filter1d(data, n)[k1:]
    if data.ndim > 1:
        return uniform_filter1d(data, n)[:, k1:-k2]
    return uniform_filter1d(data, n)[k1:-k2]


def coherent_sum(
    func: Callable[[ArrayLike, ArrayLike], float],
    coord_a: ArrayLike,
    coord_b: ArrayLike,
) -> float:
    """
    Perform a coherent sum over two arrays :math:`A, B`.

    .. math::
      \\frac{1}{N_A N_B}\\sum_i\\sum_j f(A_i, B_j)

    For numpy arrays, this is equal to::

        N, d = x.shape
        M, d = y.shape
        coherent_sum(f, x, y) == f(x.reshape(N, 1, d), x.reshape(1, M, d)).sum()

    Args:
        func: The function is called for each two items in both arrays, this should
            return a scalar value.
        coord_a: First array.
        coord_b: Second array.

    """
    res = 0
    for i in range(len(coord_a)):
        for j in range(len(coord_b)):
            res += func(coord_a[i], coord_b[j])
    return res


def coherent_histogram(
    func: Callable[[ArrayLike, ArrayLike], float],
    coord_a: ArrayLike,
    coord_b: ArrayLike,
    bins: ArrayLike,
    distinct: bool = False,
) -> NDArray:
    """
    Compute a coherent histogram over two arrays, equivalent to coherent_sum.
    For numpy arrays, this is equal to::

        N, d = x.shape
        M, d = y.shape
        bins = np.arange(1, 5, 0.1)
        coherent_histogram(f, x, y, bins) == histogram(
            f(x.reshape(N, 1, d), x.reshape(1, M, d)), bins=bins
        )

    Args:
        func: The function is called for each two items in both arrays, this should
            return a scalar value.
        coord_a: First array.
        coord_b: Second array.
        bins: The bins used for the histogram must be distributed regularly on a linear
            scale.
        distinct: Only calculate distinct part.

    """
    assert np.isclose(
        np.diff(bins).mean(), np.diff(bins)
    ).all(), "A regular distribution of bins is required."
    hmin = bins[0]
    hmax = bins[-1]
    N = len(bins) - 1
    dh = (hmax - hmin) / N

    res = np.zeros((N,))
    for i in range(len(coord_a)):
        for j in range(len(coord_b)):
            if not (distinct and i == j):
                h = func(coord_a[i], coord_b[j])
                if hmin <= h < hmax:
                    res[int((h - hmin) / dh)] += 1
    return res


def Sq_from_gr(r: NDArray, gr: NDArray, q: NDArray, n: float) -> NDArray:
    r"""
    Compute the static structure factor as fourier transform of the pair correlation
    function. [Yarnell]_

    .. math::
        S(q)-1 = \\frac{4\\pi\\rho}{q}\\int\\limits_0^\\infty (g(r)-1)\\,r \\sin(qr) dr

    Args:
        r: Radii of the pair correlation function
        gr: Values of the pair correlation function
        q: List of q values
        n: Average number density

    .. [Yarnell]
      Yarnell, J. L., Katz, M. J., Wenzel, R. G., & Koenig, S. H. (1973). Physical
      Review A, 7(6), 2130–2144.
      http://doi.org/10.1017/CBO9781107415324.004

    """
    ydata = ((gr - 1) * r).reshape(-1, 1) * np.sin(r.reshape(-1, 1) * q.reshape(1, -1))
    return np.trapz(x=r, y=ydata, axis=0) * (4 * np.pi * n / q) + 1


def Fqt_from_Grt(
    data: Union[pd.DataFrame, ArrayLike], q: ArrayLike
) -> Union[pd.DataFrame, tuple[NDArray, NDArray]]:
    """
    Calculate the ISF from the van Hove function for a given q value by fourier
    transform.

    .. math::
      F_q(t) = \\int\\limits_0^\\infty dr \\; G(r, t) \\frac{\\sin(qr)}{qr}

    Args:
        data:
            Input data can be a pandas dataframe with columns 'r', 'time' and 'G'
            or an array of shape (N, 3), of tuples (r, t, G).
        q: Value of q.

    Returns:
        If input data was a dataframe the result will be returned as one too, else two
        arrays will be returned, which will contain times and values of Fq(t)
        respectively.

    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=["r", "time", "G"])
    df["isf"] = df["G"] * np.sinc(q / np.pi * df["r"])
    isf = df.groupby("time")["isf"].sum()
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame({"time": isf.index, "isf": isf.values, "q": q})
    else:
        return isf.index, isf.values


def singledispatchmethod(func: Callable) -> Callable:
    """
    A decorator to define a genric instance method, analogue to
    functools.singledispatch.
    """
    dispatcher = functools.singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    functools.update_wrapper(wrapper, func)
    return wrapper


def quick1etau(t: ArrayLike, C: ArrayLike, n: int = 7) -> float:
    """
    Estimate the time for a correlation function that goes from 1 to 0 to decay to 1/e.

    If successful, returns tau as fine interpolation with a kww fit.
    The data is reduce to points around 1/e to remove short and long times from the kww
    fit!
    t is the time
    C is C(t) the correlation function
    n is the minimum number of points around 1/e required
    """
    # first rough estimate, the closest time. This is returned if the interpolation fails!
    tau_est = t[np.argmin(np.fabs(C - np.exp(-1)))]
    # reduce the data to points around 1/e
    k = 0.1
    mask = (C < np.exp(-1) + k) & (C > np.exp(-1) - k)
    while np.sum(mask) < n:
        k += 0.01
        mask = (C < np.exp(-1) + k) & (C > np.exp(-1) - k)
        if k + np.exp(-1) > 1.0:
            break
    # if enough points are found, try a curve fit, else and in case of failing keep using the estimate
    if np.sum(mask) >= n:
        try:
            with np.errstate(invalid="ignore"):
                fit, _ = curve_fit(
                    kww, t[mask], C[mask], p0=[0.9, tau_est, 0.9], maxfev=100000
                )
                tau_est = kww_1e(*fit)
        except:
            pass
    return tau_est


def susceptibility(
    time: NDArray, correlation: NDArray, **kwargs
) -> tuple[NDArray, NDArray]:
    """
    Calculate the susceptibility of a correlation function.

    Args:
        time: Timesteps of the correlation data
        correlation: Value of the correlation function
    """
    frequencies, fourier = filon_fourier_transformation(
        time, correlation, imag=False, **kwargs
    )
    return frequencies, frequencies * fourier


def read_gro(file: str) -> tuple[pd.DataFrame, NDArray, str]:
    with open(file, "r") as f:
        lines = f.readlines()
        description = lines[0].splitlines()[0]
        boxsize = lines[-1]
        box = boxsize.split()

    if len(box) == 3:
        box = np.array([[box[0], 0, 0], [0, box[1], 0], [0, 0, box[2]]], dtype=float)
    else:
        box = np.array(
            [
                [box[0], box[3], box[4]],
                [box[5], box[1], box[6]],
                [box[7], box[8], box[2]],
            ],
            dtype=float,
        )

    atomdata = np.genfromtxt(
        file,
        delimiter=[5, 5, 5, 5, 8, 8, 8],
        dtype="i8,U5,U5,i8,f8,f8,f8",
        skip_header=2,
        skip_footer=1,
        unpack=True,
    )
    atoms_DF = pd.DataFrame(
        {
            "residue_id": atomdata[0],
            "residue_name": atomdata[1],
            "atom_name": atomdata[2],
            "atom_id": atomdata[3],
            "pos_x": atomdata[4],
            "pos_y": atomdata[5],
            "pos_z": atomdata[6],
        }
    )
    return atoms_DF, box, description


def write_gro(
    file: str, atoms_DF: pd.DataFrame, box: NDArray, description: str
) -> None:
    with open(file, "w") as f:
        f.write(f"{description} \n")
        f.write(f"{len(atoms_DF)}\n")
        for i, atom in atoms_DF.iterrows():
            f.write(
                f"{atom['residue_id']:>5}{atom['residue_name']:<5}"
                f"{atom['atom_name']:>5}{atom['atom_id']:>5}"
                f"{atom['pos_x']:8.3f}{atom['pos_y']:8.3f}"
                f"{atom['pos_z']:8.3f}\n"
            )
        f.write(
            f"{box[0,0]:10.5f}{box[1,1]:10.5f}{box[2,2]:10.5f}"
            f"{box[0,1]:10.5f}{box[0,2]:10.5f}{box[1,0]:10.5f}"
            f"{box[1,2]:10.5f}{box[2,0]:10.5f}{box[2,1]:10.5f}\n"
        )


def fibonacci_sphere(samples: int = 1000) -> NDArray:
    points = []
    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))

    return np.array(points)


def timing(function: Callable) -> Callable:
    @functools.wraps(function)
    def wrap(*args, **kw):
        start_time = pytime()
        result = function(*args, **kw)
        end_time = pytime()
        time_needed = end_time - start_time
        print(f"Finished in {int(time_needed // 60)} min " f"{int(time_needed % 60)} s")
        return result

    return wrap


def cleanup_h5(hdf5_file: str) -> None:
    hdf5_temp_file = f"{hdf5_file[:-3]}_temp.h5"
    run(
        [
            "ptrepack",
            "--chunkshape=auto",
            "--propindexes",
            "--complevel=9",
            "--complib=blosc",
            hdf5_file,
            hdf5_temp_file,
        ]
    )
    run(["mv", hdf5_temp_file, hdf5_file])
