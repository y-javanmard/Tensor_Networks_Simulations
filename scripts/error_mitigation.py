import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from tensor_networks_simulations.general_tools import ED_tools as sf
from tensor_networks_simulations.general_tools.plot_tools import figure_styling
from tensor_networks_simulations.mps.algorithms import (
    one_time_step_2nd_order,
    one_time_step_2nd_order_2approach,
    tebd_2nd_order_LPTN_thermal,
    tebd_alg_choi,
)
from tensor_networks_simulations.mps.models import HBond
from tensor_networks_simulations.mps.states import MPDO, MPS
from tensor_networks_simulations.mps.tools import (
    apply_one_site_op_mix_state,
    correlation_one_site_mix,
    expectation_value_op,
    projection_Normalization_Mix,
    right_normalization_mpdo,
    T_spin_correl_mix,
)

figure_styling()


def run():
    L = 4
    mag_t = []
    mx_ts = []
    mx_ts_norms = []
    ts = []
    t = 0
    deltat = 0
    disc_err = []
    discerr = 0.0

    mpdo_gen = MPDO()
    psi_ghz = MPS.GHZ_state(L)
    rho = MPDO.from_mps(psi_ghz)

    hzs = 0.7 * np.ones(L)
    Jzs = -0.5 * np.ones(L)
    Jxs = 0.5 * np.ones(L)
    Jys = 0.5 * np.ones(L)
    hxs = -0.0 * np.ones(L)
    mus = -0.0 * np.ones(L)
    gamma = 0.0

    chi_max = 128
    dt = 0.005
    dt_list = dt * np.ones(L)

    scale = 1

    Hb = HBond(L, Jxs, Jys, Jzs, hxs, hzs, mus)
    Hs = [Hb.nn_term(i) for i in range(L - 1)]

    # Hs_list = [scale * Hb.h_bond(i) + Hb.new_lindbladian(Hb.sz, gamma) for i in range(L - 1)]

    tot_steps = int(5 / (scale * dt))

    for i in range(tot_steps):
        # rho, disc = one_time_step_2nd_order(
        #    rho, Hs_list, chi_max, L, dt_list, epsilon=10 ** (-10)
        # )
        rho, disc = tebd_2nd_order_LPTN_thermal(
            rho, Hs, chi_max, L, dt_list, epsilon=10 ** (-10)
        )
        t += dt
        discerr += disc
        if np.mod(i, 10) == 0:
            Mz = np.sum([expectation_value_op(rho, 2 * Hb.sz, i) for i in range(L)]) / L
            Mx = np.sum([expectation_value_op(rho, 2 * Hb.sx, i) for i in range(L)]) / L
            print(
                f"norm = {expectation_value_op(rho, Hb.s0, 0):.5f}, Mx={Mx.real:.3f} and {Mx.real/np.sqrt(expectation_value_op(rho, Hb.s0, 2)):.3f}, Mz={Mz.real:.3f}, t={t}, disc_error = {discerr}"
            )
            # print([correlation_one_site_mix(rho.Ms, rho.Ss, rho.Ss, rho.Ms, Hb.sx, i) for i in range(L)])
            # Mz = op_tot(rho, Hb.sz)
            mag_t.append(Mz)
            mx_ts.append(Mx)
            mx_ts_norms.append(Mx.real / (expectation_value_op(rho, Hb.s0, 0)))
            disc_err.append(discerr)
            ts.append(t)

    # L = 6
    solver = "me"  # use the ode solver
    # solver = "mc"   # use the monte-carlo solver

    # decoherence rate
    gammas = gamma * np.ones(L)
    print(gammas)

    init_ghz = [(qt.basis(2, 0) + qt.basis(2, 1)) / np.sqrt(2.0)] * L
    # init = [qt.basis(2, 0), qt.basis(2, 0)]*int(L/2)
    # print(init)

    psi0 = qt.tensor(init_ghz)

    # tlist = np.linspace(0, 5, 100)

    expts = sf.integrate(L, hzs, Jxs, Jys, Jzs, psi0, ts, gammas, solver)

    mz_ex = []
    mx_ex = []
    for i, t in enumerate(ts):
        mz_ex.append(np.sum([expts[j][i] for j in range(L)]) / L)
        mx_ex.append(np.sum([expts[j][i] for j in range(L, 2 * L)]) / L)

    return ts, mag_t, mx_ts, mx_ts_norms, mz_ex, mx_ex, disc_err


if __name__ == "__main__":
    ts, mag_t, mx_ts, mx_ts_norms, mz_ex, mx_ex, disc_err = run()

    import matplotlib.pyplot as plt

    csfont = {"fontname": "Comic Sans MS"}
    hfont = {"fontname": "Helvetica"}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].plot(ts, 2 * np.array(mz_ex), label="exact $<S_z>$")
    ax[0].plot(ts, 2 * np.array(mx_ex), label="exact $<S_x>$")

    ax[1].plot(ts, disc_err)
    ax[1].set_yscale("log")
    ax[0].plot(ts, mag_t, marker="s", ms=1, label="$<S_z> TEBD$")
    ax[0].plot(ts, 1 * np.array(mx_ts), marker="s", ms=3, ls=":", label="$<S_x> TEBD$")
    ax[0].plot(
        ts,
        1 * np.array(mx_ts_norms),
        marker="s",
        ms=2.5,
        ls=":",
        label="$<S_x> TEBD~normalized$",
    )

    ax[0].set_xlabel("Time")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Truncation error")
    ax[0].set_ylabel("Expectation values")
    ax[0].legend(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
