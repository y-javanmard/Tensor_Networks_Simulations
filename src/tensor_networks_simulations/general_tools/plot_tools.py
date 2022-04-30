from pathlib import Path

import matplotlib.pyplot as plt


def get_repo_root():
    """
    Gives the project root path.
    """
    return Path(__file__).parent.parent.parent.parent


def figure_styling():
    # using aliases for color, linestyle and linewidth; gray, solid, thick
    plt.rc("grid", c=".25", ls=":", lw=0.5)
    plt.rc("lines", lw=1.2, color="g")
    # plt.rcParams['axes.labelsize'] = 16
    # plt.rcParams['axes.titlesize'] = 16

    # params = {'axes.labelsize': 16,
    #      'axes.titlesize': 16}
    # plt.rcParams.update(params)
    plt.rc("axes", linewidth=1.3, labelsize=23)
    # the axes attributes need to be set before the call to subplot
    # plt.rc('font', weight='bold')
    plt.rc("xtick.major", size=7, pad=7)
    plt.rc("ytick.major", size=7, pad=7)
    plt.rc("xtick.minor", size=4, pad=7)
    plt.rc("ytick.minor", size=4, pad=7)
    plt.rc("xtick", labelsize=19)
    plt.rc("ytick", labelsize=19)

    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    rc = {"font.family": "serif", "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


# plt.text(2.5, 1., "comic sans", family="Comic Sans MS")
