"""fastfd pyrite burial base model.

This module defines =model_setup= which builds and solves a one‑dimensional
diagenetic transport model using the /fastfd/ library.  It creates a vertical
grid, registers scalar variables for each chemical species, applies boundary
conditions, assembles the diffusion‑advection‑reaction equations, and iteratively
updates reaction rates and concentrations with relaxation until the solution
converges.
"""

from copy import copy

import fastfd as ffd
import numpy as np

from diff_lib import (
    data_container,
    relax_solution,
)


def model_setup(mp, k, bc, od, c, reaction_rates, diagenetic_reactions):
    """Set up the model."""
    ffd.sparse_lib("scipy")  # use scipy sparse matrix
    z = ffd.LinearAxis("z", start=0, stop=mp.max_depth, num=mp.grid_points)
    sc = data_container()
    var_list = []
    for v in vars(c):
        setattr(sc, v, ffd.Scalar(v, [z], accuracy=mp.accuracy))
        var_list.append(getattr(sc, v))

    model = ffd.FDModel(var_list)

    temp_dict = {}
    zvalue = ["z=0", "z=-1"]
    ivalue = [0, -1]
    for v in vars(c):
        for i, p in enumerate([0, 2]):
            if bc[v][p] == "concentration":
                temp_dict[f"{v} {zvalue[i]}"] = (
                    getattr(sc, v).i[ivalue[i]],
                    getattr(sc, v).i[ivalue[i]],
                    bc[v][p + 1],
                )
            elif bc[v][p] == "gradient":
                temp_dict[f"{v} {zvalue[i]}"] = (
                    getattr(sc, v).i[ivalue[i]],
                    getattr(sc, v).d("z")[ivalue[i]],
                    bc[v][p + 1],
                )
            else:
                raise ValueError(f"{bc[v][p]}is an unknown keyword - typo?")

    model.update_bocos(temp_dict)

    # build equations dictinaries
    equation_dict = {}
    update_dict = {}
    for species, v in bc.items():
        if v[4] == "dissolved":
            D = getattr(od, species)
        elif v[4] == "particulate":
            D = od.DB
        else:
            raise ValueError(f"{species} is an unknown keyword - typo?")

        phase = getattr(sc, species)
        f = getattr(reaction_rates, species) * v[5]

        equation_dict[species] = (D * phase.d("z", 2) - od.w * phase.d("z", 1), f)
        update_dict[species] = (None, f)

    model.update_equations(equation_dict)

    # run the model until solutions converge
    for i in range(mp.max_loops):
        p = copy(reaction_rates)  # save the old and get new reaction estimates
        # get new reaction rates
        reaction_rates, a = diagenetic_reactions(0, z, c, k, reaction_rates)
        for var_name in bc:  # relax reaction estimates
            new = getattr(reaction_rates, var_name)  # get new value for var name
            old = getattr(p, var_name)  # get old value for var name
            # make sure nan values don't interfere
            v = old * (1 - mp.relax) + mp.relax * new  # relax
            setattr(reaction_rates, var_name, v)  # set new value

        for species, v in bc.items():  # update equation matrix
            update_dict[species] = (None, getattr(reaction_rates, species) * v[5])

        model.update_equations(update_dict)
        results = model.solve()  # Solve the model

        for var_name in vars(c):  # relax concentrations
            new = getattr(c, var_name)
            v = relax_solution(results[var_name], new, mp.relax)
            setattr(c, var_name, v)

        # check residual
        residual = abs(np.sum(results["ch4"] - c.ch4) / mp.grid_points)
        if residual < mp.res_min and i > 4:
            break

    print(f"i = {i} residual = {residual:.2e}\n")

    return od, c, reaction_rates
