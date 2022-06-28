from pathlib import Path
from re import A

import _pickle as cPickle
import matplotlib.pyplot as plt
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
import qutip as qt

from tensor_networks_simulations.general_tools import ED_tools as sf
from tensor_networks_simulations.general_tools.plot_tools import (
    figure_styling,
    get_repo_root,
)
from tensor_networks_simulations.mps.algorithms import (
    tebd_2nd_order_vidal_mpdo,
    tebd_2nd_order_vidal_mpdo_mpi4py,
    tebd_1st_order_vidal_mpdo_mpi4py,
    tebd_2nd_order_vidal_mpdo_mpi4py_efficient,
)
from tensor_networks_simulations.mps.models import HBond
from tensor_networks_simulations.mps.states import MPDO, MPS
from tensor_networks_simulations.mps.tools import (
    # apply_one_site_op_mix_state,
    # check_right_normalization,
    # correlation_one_site_mix,
    expectation_value_op,
    apply_one_site_op_mpo,
    overlap,
#     projection_Normalization_Mix,
#     right_normalization_mpdo,
#     right_normalization_mps,
#     T_spin_correl_mix,
)

figure_styling()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def expectation_value(rho, op, site, d=2):
    mlist = []
    for i in range(len(rho.Ms)):
        M = np.reshape(rho.Ms[i], (d, d, rho.Ms[i].shape[-2], rho.Ms[i].shape[-1]))
        mlist.append(M)
    new_rho = MPDO(mlist, rho.Ss, rho.bonds)
    val = expectation_value_op(new_rho, op, site)
    return val

def apply_loc_op(rho, op, site, d=2):
    mlist = []
    op = np.kron(op, np.eye(d))
    for i in range(len((rho.Ms))):
        if i != site:
            mlist.append(rho.Ms[i])
        else:
            mlist.append(np.tensordot(op, rho.Ms[i], axes=(1, 0)))
    
    rho_new = MPDO(mlist, rho.Ss, rho.bonds)
    return rho_new

def two_point_correlator(rho, op1, op2, site1, site2, d=2):
    """_summary_

    Args:
        rho (_type_): _description_
        op1 (_type_): _description_
        op2 (_type_): _description_
        site1 (_type_): _description_
        site2 (_type_): _description_
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    mlist = []
    #nlist = []
    for i in range(len(rho.Ms)):
        M = np.reshape(rho.Ms[i], (d, d, rho.Ms[i].shape[-2], rho.Ms[i].shape[-1]))
        #nlist.append(M)
        if i == site1:
            M = np.tensordot(op1, M, (1, 0))
        elif i == site2:
            M = np.tensordot(op2, M, (1, 0))
        else: M = M
        mlist.append(M)
    rho1 = MPDO(mlist, rho.Ss, rho.bonds)
    #rho2 = MPDO(nlist, rho.Ss, rho.bonds)
    #c = overlap(rho1=rho2, rho2=rho1)
    c = expectation_value(rho1, np.eye(2), 0)
    return c
    
def to_rho_verestrate(rho_vidal, d=2):
    """rho_vidal has the physical dimension 4 and each tensor is rank 3 while rho_verestrate has physical local dimesntion 2 and each tensor is rank 4.
    

    Args:
        rho_vidal (_type_): _description_
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    mlist = []
    for i in range(len(rho_vidal.Ms)):
        M = np.reshape(rho_vidal.Ms[i], (d, d, rho_vidal.Ms[i].shape[-2], rho_vidal.Ms[i].shape[-1]))
        mlist.append(M)
    rho_verestrate = MPDO(mlist, rho_vidal.Ss, rho_vidal.bonds)
    return rho_verestrate

def loschmidt_echo_square_norm(psi_in: MPS, rho_vidal: MPDO, d=2):
    rho_ver = to_rho_verestrate(rho_vidal=rho_vidal)
    C0 = np.tensordot(psi_in.Bs[0], rho_ver.Ms[0], axes=(0, 1)) # (a_i-1, a_i, s_i, a'_i-1, a'_i)
    C0 = np.tensordot(C0, np.conj(psi_in.Bs[0]), axes=(2, 0))             # (a_i-1, a_i, a'_i-1, a'_i, a_i-1, a_i)
    C0 = np.transpose(C0, (0, 2, 4, 1, 3, 5))                    # (a_i-1, a'_i-1, a_i-1, a_i, a'_i, a_i)
    for i in range(1, len(rho_vidal.Ms)):
        C = np.tensordot(psi_in.Bs[i], rho_ver.Ms[i], axes=(0, 1))
        C = np.tensordot(C, np.conj(psi_in.Bs[i]), axes=(2, 0))   # (a_i, a_i+1, a'_i, a'_i+1, a_i, a_i+1)
        C0 = np.tensordot(C0, C, axes=([3,4,5],[0, 2, 4]))  # (a_i-1, a'_i-1, a_i-1, a_i+1, a'_i+1, a_i+1)
    return np.squeeze(C0)
        



def run(chi_max=32, scale=1, gamma=0.05):

    L = 8
    # chi_max = 100
    epsilon = 1e-6
    dt = 0.01
    gap_measure= int(0.1/dt) 
    print(f"dt={dt}, gap_measurements={gap_measure}")
    final_time = 8
    # total_steps = int(final_time/(scale*dt))
    total_steps = int((scale * final_time) / (gap_measure*dt))
    print(f"total steps = {total_steps}")
    dt_list = np.array([dt,]* L)
    gamma = gamma
    model = "TFIChain"

    Jx = -0.1 * np.ones(L)#*0.5
    Jy = 0.0 * np.ones(L)#*0.5
    Jz = -1.0 * np.ones(L)#*0.5
    hz = -0.0 * np.ones(L)#*0.5
    hx = -0.1 * np.ones(L)#*0.5
    mu = 0.0 * np.ones(L)#*0.5

    Hb = HBond(L, Jxs=Jx, Jys=Jy, Jzs=Jz, Hxs=hx, Hzs=hz, mus=mu, d=4)

    Hs = [(1.0 / scale) * Hb.h_bond(i) 
          + (1./1.)*Hb.new_lindbladian(Hb.sx, i, gamma) 
          + (1./1.)*Hb.new_lindbladian(Hb.sy, i, gamma)
          + (1./1.)*Hb.new_lindbladian(Hb.sz, i, gamma) for i in range(L - 1)]
    # print(Hs)

    psi = MPS.af_product_state(L)
    rho = MPDO.from_mps_vidal(psi)

    psi_xf = MPS.GHZ_state(L)
    psi_f = MPS.initial_state_f(L)
    
    psi_0 = psi_xf
    rho = MPDO.from_mps_vidal(psi_0)

    ts = []
    mxs = []
    mzs = []
    rhos = []
    czz = two_point_correlator(rho, Hb.sz, Hb.sz, L/2, L/2 +1).real/(sum([expectation_value(rho, Hb.s0, i).real for i in range(L)])/ L)
    new_rho = apply_loc_op(rho, Hb.sz, int(L/2))
    czz = expectation_value(new_rho, Hb.sz, int(L/2)+1)
    norm = expectation_value(rho, Hb.s0, 0).real
    rate_fs = [-np.log(loschmidt_echo_square_norm(psi_in=psi_0, rho_vidal=rho)/(norm*norm))/L]
    czzs = [czz]
    czz1s=[0.0]
    czz2s=[0.0]
    tr_erros = [0.0]
    mxs.append(1.0)
    ts.append(0.0)
    mzs.append(0.0)
    
    
    path = Path(f"data_depolarizing_1st/subdata_{model}_{L}_{chi_max}")
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
        
    #dt_list = 1j*dt_list
    for i in range(total_steps):
        rho, err = tebd_2nd_order_vidal_mpdo_mpi4py_efficient(steps=int(gap_measure*scale), rho=rho, Hs=Hs, chi_max=chi_max, dt_list=dt_list, epsilon=epsilon, d=4)

        if rank == 0:
            #if np.mod(i, int((gap_measure) * scale)) == (gap_measure * scale) - 1:
                norm = expectation_value(rho, Hb.s0, 0).real
                czz = two_point_correlator(rho, Hb.sz, Hb.sz, L/2, L/2 +1).real/norm
                cz1 = expectation_value(rho, Hb.sz, int(L/2)).real/norm
                cz2 = expectation_value(rho, Hb.sz, int(L/2)+1).real/norm
                mz = sum([expectation_value(rho, Hb.sz, i).real for i in range(L)]) / L
                mx = sum([expectation_value(rho, Hb.sx, i).real for i in range(L)]) / L
                rate_f = -(L**-1.)*np.log(loschmidt_echo_square_norm(psi_in=psi_0, rho_vidal=rho).real/(norm))
                mxs.append(mx/ norm)
                mzs.append(mz/ norm)
                #rhos.append(rho)
                ts.append((i + 1) * gap_measure*dt)
                tr_erros.append(err)
                czzs.append(czz)
                czz1s.append(cz1)
                czz2s.append(cz2)
                rate_fs.append(rate_f)
                
                # check_right_normalization(rho.Ms)
                print(f"mx={mx:.5f}, mz={mz:.5f} t={(i+1)*gap_measure*dt}/{scale*final_time}")
                print(f"norm={norm:.5f},\
                          mx/nrom={mx/norm:.5f},\
                          mz/nrom={mz/norm:.5f},\
                          Czz = {czz:.5f},\
                          rate_f = {rate_f:.5f}")
                print(f"max-bond: {max(rho.bonds)}, scale={scale}, gamma={gamma}, chi_max={chi_max}, err={err}, epsilon={epsilon}")

                print("--------------------------")


                with open(path / f"mxs_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(mxs, fh)
                with open(path / f"mzs_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(mzs, fh)
                with open(path / f"ts_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(ts, fh)
                with open(path / f"czz_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(czzs, fh)
                with open(path / f"czz1_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(czz1s, fh)
                with open(path / f"czz2_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(czz2s, fh)
                with open(path / f"rate_f_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(rate_fs, fh)
                with open(path / f"trr_err_scale-{scale}_gamma-{gamma}.pickle", "wb") as fh:
                    cPickle.dump(tr_erros, fh)

    exact = 1
    if exact:
        L = 8
        solver = "me"  # use the ode solver
        # solver = "mc"   # use the monte-carlo solver

        # decoherence rate
        #gamma = 0.0
        gammas = gamma * np.ones(L)
        print(gammas)

        init_xf = [(qt.basis(2, 0) + qt.basis(2, 1)) / np.sqrt(2.0)] * L
        init_f = [qt.basis(2, 0), qt.basis(2, 0)] * int(L / 2)
        # print(init)

        psi0 = qt.tensor(init_xf)

        # tlist = np.linspace(0, 5, 100)

        expts = sf.integrate(L, hx, hz, Jx, Jy, Jz, psi0, ts, gammas, solver, noise_model="depolarizing")

        mx_ex = []
        mz_ex = []

        for i, t in enumerate(ts):
            mx_ex.append(np.sum([expts[j][i] for j in range(L)]) / L)
            mz_ex.append(np.sum([expts[j][i] for j in range(L, 2 * L)]) / L)
        czz_ex = expts[-2]
        rate_f_ex = -np.log(expts[-1])/L
    else:
        mz_ex = 0.0 * np.array(ts)
        mx_ex = 0.0 * np.array(ts)
        czz_ex = 0.0*np.array(ts)
    #print(czz_ex)

    return mzs, mxs, czzs, rate_fs, ts, mz_ex, mx_ex,  czz_ex, rate_f_ex, tr_erros


if __name__ == "__main__":
    plot_data = 1
    dta={}
    L = 8
    
    gammas = [1e-3]#[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:#[0.001, 0.0025, 0.005, 0.0075, 0.015, 0.02, 0.03 ]:#[0.01, 0.05, 0.1, 0.25, 0.5]
    scales = [1.0]#[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    for chi_max in [200]:
        
        for scale in scales:
            for gamma in gammas:
                mzs, mxs, czzs, rate_fs, ts, mz_ex, mx_ex, czz_ex, rate_f_ex, tr_erros = run(chi_max, scale, gamma)
                dta[f"{scale}-{gamma}"] = mzs, mxs, czzs, rate_fs, ts, mz_ex, mx_ex, czz_ex, rate_f_ex, tr_erros
                
    if plot_data:
        if rank ==0:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            csfont = {"fontname": "Comic Sans MS"}
            hfont = {"fontname": "Helvetica"}
            for scale in scales:
                for gamma in gammas:
            
                    #ax[0].plot(ts, np.array(mx_ex), label="exact $<S_x>$")
                    #x[0].plot(ts, np.array(mz_ex), label="exact $<S_z>$")
                    #ax[0].plot(ts, np.array(czz_ex), label="exact $<c_{{zz}}>$")
                    ax[0].plot(np.array(dta[f"{scale}-{gamma}"][4]), np.array(dta[f"{scale}-{gamma}"][8]), label="exact $\mathcal{{L}}(t)$")
                    
                    ax[1].plot(np.array(dta[f"{scale}-{gamma}"][4]), np.array(dta[f"{scale}-{gamma}"][-1]))
                    ax[1].set_yscale("log")
                    #ax[0].plot(ts,1 * np.array(mxs), marker="o", ms=4, ls=":", label=rf"$<S_x>~ TEBD;\gamma={gamma},~ \alpha={scale},~\chi_{{max}}={chi_max} $")
                    #ax[0].plot(ts, 1 * np.array(mzs), marker="s", ms=4, ls=":", label=rf"$<S_z>~ TEBD; \gamma={gamma},~ \alpha={scale},~\chi_{{max}}={chi_max} $")
                    #ax[0].plot(ts,1 * np.array(czzs), marker="o", ms=4, ls=":", label=rf"$C_{{zz}}~ TEBD;\gamma={gamma},~ \alpha={scale},~\chi_{{max}}={chi_max} $")
                    ax[0].plot(np.array(dta[f"{scale}-{gamma}"][4]), np.array(dta[f"{scale}-{gamma}"][3]), marker="v", ms=4, ls=":", label=rf"$\mathcal{{L}}(t)~ TEBD;\gamma={gamma},~ \alpha={scale},~\chi_{{max}}={chi_max} $")
                    ax[0].set_xlabel("time")
                    ax[1].set_xlabel("time")
                    ax[1].set_ylabel("truncation error")
                    ax[0].set_ylabel("txpectation values")
                    ax[0].legend(fontsize=10,frameon=False)
            plt.legend(fontsize=12, frameon=False)
            plt.tight_layout()
            plt.plot()
            plt.show()
            plt.savefig(f"exact_vs_mpdo-{L}.pdf")
