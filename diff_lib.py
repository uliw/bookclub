"""
Utility library for the /fastfd/ package.

THis libabry provides a small collection of
functions and a lightweight container class used throughout the Pyrite Burial
model.

The module contains:

- :class:=data_container= – a simple container that can be initialised from a
  space‑separated string of attribute names with optional default values, or
  from a dictionary mapping attribute names to values.

- :func:=diff_coeff= – computes the diffusion coefficient (m² s⁻¹) for a given
  temperature (°C), porosity (percent) and the linear parameters /m0/ and /m1/
  from Boudreau (1996).

- :func:=get_delta= – calculates the isotopic delta value (‰) from the total
  concentration of an isotope pair and a reference ratio.

- :func:=get_l_mass= – derives the concentration of the light isotope from a
  measured total concentration, a delta value and the reference ratio.

- :func:=relax_solution= – blends a current solution vector with a previous one,
  limiting the change to a specified fraction and enforcing non‑negative values.

These helpers are primarily intended for modelling isotope diffusion and
fractionation processes in geological simulations.
"""

import numpy as np


class data_container:
    """A simple container.

    Initialised from a space‑separated string of attribute names with optional default
    values, or from a dictionary mapping attribute names to values.
    """

    def __init__(self, names=None, defaults=None):
        if isinstance(names, str):
            names = names.split(" ")
            if isinstance(defaults, list):
                for i, name in enumerate(names):
                    if name != "":
                        setattr(self, name, defaults[i])
            else:
                for name in names:
                    if name != "":
                        setattr(self, name, defaults)

        elif isinstance(names, dict):
            for k, v in names.items():
                if k != "":
                    setattr(self, k, v)

    def update(self, d: dict):
        """Update Values."""
        for key, value in d.items():
            try:
                _n = getattr(self, key)
                setattr(self, key, value)
            except:
                raise ValueError(f"Umknown {key}, typo?")


def diff_coeff(T, m0, m1, phi):
    """Calculate the diffusion coeefficien in m^2/s.

    T: temperature in C
    phi: porosity in percent
    m0, m1: parameter as from table X in Boudreau 1996
    """
    return (m0 + m1 * T) * 1e-10 / (1 - np.log(phi**2))


def get_delta(c, li, r):
    """Calculate the delta from the mass of light and heavy isotope.

    :param li: light isotope mass/concentration
    :param h: heavy isotope mass/concentration
    :param r: reference ratio

    :return : delta

    """
    h = c - li

    return np.where(li < 0.001, float("nan"), 1000 * (h / li - r) / r)


def get_l_mass(m, d, r):
    """Derive the concentration of the light isotope.

    From a measured total concentration, a delta value and the reference ratio.
    :param m: mass or concentration
    :param d: delta value
    :param r: isotopic reference ratio

    return mass or concentration of the light isotopeb
    """
    return (1000.0 * m) / ((d + 1000.0) * r + 1000.0)


def relax_solution(curr_sol, last_sol, fraction):
    """Blend two solution vectors.

    In such away that they only chance by a given fraction
    """
    sol = last_sol * (1 - fraction) + curr_sol * fraction
    return sol * (sol >= 0)  # exclude negative solutions
