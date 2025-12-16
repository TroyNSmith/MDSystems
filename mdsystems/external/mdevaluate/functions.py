import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gamma as spgamma
from scipy.integrate import quad as spquad


def kww(t: ArrayLike, A: float, tau: float, beta: float) -> ArrayLike:
    return A * np.exp(-((t / tau) ** beta))


def kww_1e(A: float, tau: float, beta: float) -> float:
    return tau * (-np.log(1 / (np.e * A))) ** (1 / beta)


def cole_davidson(omega: ArrayLike, A: float, beta: float, tau: float) -> ArrayLike:
    P = np.arctan(omega * tau)
    return A * np.cos(P) ** beta * np.sin(beta * P)


def cole_cole(omega: ArrayLike, A: float, beta: float, tau: float) -> ArrayLike:
    return (
        A
        * (omega * tau) ** beta
        * np.sin(np.pi * beta / 2)
        / (
            1
            + 2 * (omega * tau) ** beta * np.cos(np.pi * beta / 2)
            + (omega * tau) ** (2 * beta)
        )
    )


def havriliak_negami(
    omega: ArrayLike, A: float, beta: float, alpha: float, tau: float
) -> ArrayLike:
    r"""
    Imaginary part of the Havriliak-Negami function.

    .. math::
       \chi_{HN}(\omega) = \Im\left(\frac{A}{(1 + (i\omega\tau)^\alpha)^\beta}\right)
    """
    return -(A / (1 + (1j * omega * tau) ** alpha) ** beta).imag


# fits decay of correlation times, e.g. with distance to pore walls
def colen(d: ArrayLike, X: float, tau_pc: float, A: float) -> ArrayLike:
    return tau_pc * np.exp(A * np.exp(-d / X))


# fits decay of the plateau height of the overlap function,
# e.g. with distance to pore walls
def colenQ(d: ArrayLike, X: float, Qb: float, g: float) -> ArrayLike:
    return (1 - Qb) * np.exp(-((d / X) ** g)) + Qb


def vft(T: ArrayLike, tau_0: float, B: float, T_inf: float) -> ArrayLike:
    return tau_0 * np.exp(B / (T - T_inf))


def arrhenius(T: ArrayLike, tau_0: float, E_a: float) -> ArrayLike:
    return tau_0 * np.exp(E_a / T)


def MLfit(t: ArrayLike, tau: float, A: float, alpha: float) -> ArrayLike:
    def MLf(z: ArrayLike, a: float) -> ArrayLike:
        """Mittag-Leffler function"""
        z = np.atleast_1d(z)
        if a == 0:
            return 1 / (1 - z)
        elif a == 1:
            return np.exp(z)
        elif a > 1 or all(z > 0):
            k = np.arange(100)
            return np.polynomial.polynomial.polyval(z, 1 / spgamma(a * k + 1))

        # a helper for tricky case, from Gorenflo, Loutchko & Luchko
        def _MLf(z: float, a: float) -> ArrayLike:
            if z < 0:
                f = lambda x: (
                    np.exp(-x * (-z) ** (1 / a))
                    * x ** (a - 1)
                    * np.sin(np.pi * a)
                    / (x ** (2 * a) + 2 * x**a * np.cos(np.pi * a) + 1)
                )
                return 1 / np.pi * spquad(f, 0, np.inf)[0]
            elif z == 0:
                return 1
            else:
                return MLf(z, a)

        return np.vectorize(_MLf)(z, a)

    return A * MLf(-((t / tau) ** alpha), alpha)
