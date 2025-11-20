"""Define a simple methane diffusion model."""


def model(p, bc, k, diagenetic_reactions):
    """Model methane diffusion.

    As a function of organic matter availability, including isotopes
    Model units are meter/second, mmol/liter, and meter
    """
    import pathlib as pl

    import numpy as np
    import pandas as pd

    from base_model import model_setup
    from diff_lib import data_container

    np.seterr(divide="ignore", invalid="ignore")

    # --------------------------- Model parameters -------------- *
    mp = data_container({
        "max_depth": 30,  # meters
        "grid_points": 300,
        "temp": [4, 4],  # temp top, bottom, in C
        "phi": 0.65,  # porosity
        "w": 1.5e-11,  # sedimentation rate in m/s
        "CH4_concentration": 28,  # mmol/l
        "accuracy": 4,  # equation scheme
        "max_loops": 4000,  # max number of interactions
        "relax": 0.01,  # solution relaxation factor
        "res_min": 1e-5,  # convergence criterion
    })

    mp.update(p)
    fn = "methane_model.csv"

    # --------------------------- Initialize Datafields --------- #
    # use the same names as for boundary conditions!
    od = data_container({  # other vector data
        "w": np.zeros(mp.grid_points) + mp.w,  #
        "phi": np.zeros(mp.grid_points) + mp.phi,  # if porosity is constant
        "z": np.linspace(0, mp.max_depth, mp.grid_points),
        "T": np.linspace(mp.temp[0], mp.temp[1], mp.grid_points),
    })

    # get diffusion coefficient for CH4 with T dependence
    od.ch4 = 1e-9 * (1 - np.log(mp.phi**2))

    # setup initial concentration vectors
    var_list = " ".join(bc.keys())
    c = data_container(var_list, np.zeros(mp.grid_points))

    # pre populate concentration vectors. Best to loop over them!
    c.ch4 = np.linspace(bc["ch4"][1], bc["ch4"][3], mp.grid_points)  # initial CH4

    # setup initial reaction vectors
    f = data_container(var_list, np.zeros(mp.grid_points))

    # --------------------------- Run model and plot data ------- *
    od, c, reaction_rates = model_setup(
        mp,
        k,
        bc,
        od,
        c,
        f,
        diagenetic_reactions,
    )

    # limit display for very small concentrations

    # put everything on a dataframe
    df = pd.DataFrame(vars(c)).add_prefix("c_")
    df_f = pd.DataFrame(vars(reaction_rates)).add_prefix("f_")
    df_o = pd.DataFrame(vars(od))
    df = df.join(df_f).join(df_o)

    cwd: pl.Path = pl.Path.cwd()  # get the current working directory
    fqfn: pl.Path = pl.Path(f"{cwd}/{fn}")  # fully qualified file name

    df.to_csv(fqfn)
    return df


if __name__ == "__main__":
    """Run this interactively"""
    import matplotlib.pyplot as plt

    from diff_lib import data_container

    # a few parameters to play with
    p = {
        "max_depth": 0.35,  # meters
        "grid_points": 300,
        "phi": 0.65,  # porosity
        "w": 1.5e-9,  # sedimentation rate in m/s
    }

    bc = {
        "ch4": [  # species
            "concentration",  # upper bc type
            0,  # upper bc value
            "concentration",  # lower bc type
            30,  # lower bc value
            "dissolved",  # phase
            1,  # reaction type 1 = source, -1 = sink
        ],
    }

    k = data_container({"ch4": 0})

    # --------------------------- Define Reactions -------------- #
    def diagenetic_reactions(a, z, c, k, f):
        """Define microbial reactions."""
        f.ch4 = 0

        return f, 0

    df = model(p, bc, k, diagenetic_reactions)

    # plot data
    fig, ax = plt.subplots()
    ax.plot(df.c_ch4, df.z)
    ax.set_xlabel(r"CH$_4$ [mmol/l]")
    ax.set_ylabel("Depth [mbsf]")
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()
