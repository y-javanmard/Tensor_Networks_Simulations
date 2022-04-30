import numpy as np
import qutip as qp

import matplotlib.pyplot as plt


def spin_operator(N):
    # pre-allocate operators
    si = qp.qeye(2)
    sx = qp.sigmax()/2
    sy = qp.sigmay()/2
    sz = qp.sigmaz()/2
    sp = qp.sigmap()
    sm = qp.sigmam()
    

    sx_list = []
    sy_list = []
    sz_list = []
    sp_list = []
    sm_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(qp.tensor(op_list))
            
        op_list[n] = sy
        sy_list.append(qp.tensor(op_list))
            
        op_list[n] = sz
        sz_list.append(qp.tensor(op_list))
        
        op_list[n] = sp
        sp_list.append(qp.tensor(op_list))
        
        op_list[n] = sm
        sm_list.append(qp.tensor(op_list))
            
            
    return sx_list, sy_list, sz_list, sp_list, sm_list







def figure_styling():
    # using aliases for color, linestyle and linewidth; gray, solid, thick
    plt.rc("grid", c=".25", ls=":", lw=0.5)
    plt.rc("lines", lw=1.2, color="g")
    # plt.rcParams['axes.labelsize'] = 16
    # plt.rcParams['axes.titlesize'] = 16

    # params = {'axes.labelsize': 16,
    #      'axes.titlesize': 16}
    # plt.rcParams.update(params)
    plt.rc("axes", linewidth=1.2, labelsize=25)
    # the axes attributes need to be set before the call to subplot
    # plt.rc('font', weight='bold')
    plt.rc("xtick.major", size=7, pad=7)
    plt.rc("ytick.major", size=7, pad=7)
    plt.rc("xtick.minor", size=4, pad=7)
    plt.rc("ytick.minor", size=4, pad=7)
    plt.rc("xtick", labelsize=22)
    plt.rc("ytick", labelsize=22)

    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    rc = {"font.family": "serif", "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]




#plt.text(2.5, 1., "comic sans", family="Comic Sans MS")
