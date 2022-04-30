import copy

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import qutip as qp
from tqdm import tqdm

from tensor_networks_simulations.general_tools import ED_tools as sf
from tensor_networks_simulations.general_tools.plot_tools import figure_styling
from tensor_networks_simulations.mps.algorithms import (
    one_time_step_2nd_order,
    one_time_step_2nd_order_2approach,
    tebd_2nd_order_LPTN,
    tebd_alg_choi,
)
from tensor_networks_simulations.mps.models import (
    BondHamiltonian,
    H_bond_choi,
    HBondChoi,
)
from tensor_networks_simulations.mps.states import MPDO, MPS
from tensor_networks_simulations.mps.tools import (
    apply_one_site_op_mix_state,
    apply_one_site_op_mpo,
    correlation_one_site_mix,
    expectation_value,
    expectation_value_op,
    projection_Normalization_Mix,
    T_spin_correl_mix,
)

figure_styling()


def mpo_inf_temperature(L):
    Bs_inf = MPDO().infinite_temp_state(L)
    bonds_inf = MPDO().bond_vec_inf_temp(L)
    Bs_inf, Ss_inf = MPDO().schmidt_vals_from_mps(Bs_inf, bonds_inf)
    Ms_inf, Ss_inf = MPDO().mpo_from_purified_mps(Bs_inf, Ss_inf)
    bonds_inf = bonds_inf[::2]
    bonds_inf.append(1)
    mpo_inf = MPDO(Ms_inf, Ss_inf, bonds_inf)
    return mpo_inf


def run():
    L = 32
    hzs = 0.0 * np.ones(L)
    Jzs = -0.0 * np.ones(L)
    Jxs = [1.0] * L
    Jys = [1.0] * L
    hxs = mus = -0.0 * np.ones(L)
    gamma = 0.0
    chi_max = 60
    dt = 0.01
    dt_list = np.array(
        [
            1j * dt,
        ]
        * L
    )
    scale = 1
    tot_steps = int(16 / (scale * dt))

    Hb = BondHamiltonian(L, Jxs, Jys, Jzs, hxs, hzs, mus)
    Hs = [Hb.nn_term(i) for i in range(L - 1)]

    mpo_inf = mpo_inf_temperature(L)
    mpo_inf = MPDO.update_mpo_with_schmidt_vals(mpo_inf)

    phi_inf = copy.deepcopy(mpo_inf)
    phi_inf = apply_one_site_op_mpo(phi_inf, Hb.sz, int(L / 2))

    T_spin_correl_mix(
        phi_inf.Ms, Hb.s0, phi_inf.Ms, L, int(L / 2)
    ), projection_Normalization_Mix(mpo_inf.Ms, mpo_inf.Ms, L)

    mag_t = []
    SzSz_ts = []
    mx_ts_norms = []
    ts = []
    t = 0
    deltat = 0
    disc_err = []
    discerr = 0.0
    for i in tqdm(range(tot_steps)):
        mpo_inf, disc1 = tebd_2nd_order_LPTN(
            mpo_inf, Hs, chi_max, L, dt_list, epsilon=10 ** (-5), evolve_auxiliary=False
        )
        phi_inf, disc2 = tebd_2nd_order_LPTN(
            phi_inf, Hs, chi_max, L, dt_list, epsilon=10 ** (-5), evolve_auxiliary=False
        )

        t += dt
        discerr += disc1 + disc2
        if np.mod(i, 20) == 0:
            # rho = MPDO.from_mpo(mpo)
            SzSz = T_spin_correl_mix(mpo_inf.Ms, Hb.sz, phi_inf.Ms, L, int(L / 2))
            Mz = (
                np.sum(
                    [
                        T_spin_correl_mix(phi_inf.Ms, Hb.sz, phi_inf.Ms, L, i)
                        for i in range(L)
                    ]
                )
                / L
            )

            print(
                f"""
                  norm = {T_spin_correl_mix(mpo_inf.Ms, Hb.s0, mpo_inf.Ms, L, 0):.3f},
                  <Sz(t)Sz> = {SzSz/2:.3f}, Mz={Mz.real:.3f},
                  t={t:.2f}, disc = {discerr:.4f},
                  max_bond={max(mpo_inf.bonds)},{max(phi_inf.bonds)}
                  """
            )

            mag_t.append(Mz)
            SzSz_ts.append(SzSz / 2)

            disc_err.append(discerr)
            ts.append(t)

    with open(f"data_P.pickle", "wb") as fh:
        cPickle.dump(SzSz_ts, fh)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(
        ts,
        np.array(SzSz_ts),
        marker="o",
        ms=1.5,
        label="$<S^z_{L/2}(t)S^2_{L/2}(0)> TEBD$",
    )
    # ax.set_ylim(-0.04, 0.08)
    ax.set_xlim(0, 22)
    plt.legend()
    plt.plot()
    plt.show()


if __name__ == "__main__":
    run()
