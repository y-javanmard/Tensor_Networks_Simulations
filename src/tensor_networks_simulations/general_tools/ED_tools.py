import numpy as np
import qutip as qt


def spin_operator(N):
    # pre-allocate operators
    si = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    sp = qt.sigmap()
    sm = qt.sigmam()

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
        sx_list.append(qt.tensor(op_list))

        op_list[n] = sy
        sy_list.append(qt.tensor(op_list))

        op_list[n] = sz
        sz_list.append(qt.tensor(op_list))

        op_list[n] = sp
        sp_list.append(qt.tensor(op_list))

        op_list[n] = sm
        sm_list.append(qt.tensor(op_list))

    return sx_list, sy_list, sz_list, sp_list, sm_list


def integrate(L, hx, hz, Jx, Jy, Jz, psi0, tlist, gammas, solver):
    sx_list, sy_list, sz_list, sp_list, sm_list = spin_operator(L)
    # si = qeye(2)
    # sx = sigmax()
    # sy = sigmay()
    # sz = sigmaz()

    expect_list = []
    for i in range(L):
        expect_list.append(sx_list[i])
    for i in range(L):
        expect_list.append(sz_list[i])

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(L):
        H += hz[n] * sz_list[n]
        H += hx[n] * sx_list[n]

    # interaction terms
    for n in range(L - 1):
        H += Jx[n] * sx_list[n] * sx_list[n + 1]
        H += Jy[n] * sy_list[n] * sy_list[n + 1]
        H += Jz[n] * sz_list[n] * sz_list[n + 1]

    # collapse operators
    c_op_list = []

    # spin dephasing
    for n in range(L):
        if gammas[n] > 0.0:
            c_op_list.append(np.sqrt(gammas[n]) * sx_list[n])

    # evolve and calculate expectation values
    if solver == "me":
        result = qt.mesolve(H, psi0, tlist, c_op_list, expect_list)
    elif solver == "mc":
        ntraj = 500
        result = qt.mcsolve(H, psi0, tlist, c_op_list, expect_list, ntraj)
    # if solver == "me":
    #     return result
    # else:
    return result.expect
