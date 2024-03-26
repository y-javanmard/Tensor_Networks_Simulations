from re import U
from typing import List

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm

from tensor_networks_simulations.mps.models import BondHamiltonian, H_bond_choi
from tensor_networks_simulations.mps.states import MPDO
from tensor_networks_simulations.mps.tools import apply_mixed_gate_mpdo


class Operator(np.ndarray):
    @property
    def dag(self):
        return self.conj().T


def new_lindbladian(op, gamma=0.01, d=2):
    s0 = np.eye(d)

    iden = np.eye(d ** 2)
    op_1 = np.kron(op, s0)
    op_2 = np.kron(s0, op)
    op_list = [op_1.view(Operator), op_2.view(Operator)]

    lind = 0.0
    lind += 1 * gamma * np.kron(op_list[0], op_list[1].dag)
    lind += -1 * 0.5 * gamma * np.kron((op_list[0].dag @ op_list[0]), iden)
    lind += -1 * 0.5 * gamma * np.kron(iden, (op_list[1].dag @ op_list[1]))

    return lind


def kraus_op_svd(op, gamma, dt=0.01, d=2):
    idn = np.reshape(np.eye(d), (d * d, 1))
    op1 = np.reshape(op, (d * d, 1))
    op1 = np.kron(op1, op1.T)

    op2 = np.conj(op.T) @ op
    op2 = np.reshape(op2, (d * d, 1))
    op2 = np.kron(op2, idn.T)

    op3 = op.T @ np.conj(op)
    op3 = np.reshape(op3, (1, d * d))
    op3 = np.kron(idn, op3)

    op1 = np.kron(op, op.view(Operator).dag)
    op2 = np.kron((op.view(Operator).dag).dot(op), np.eye(d))
    op3 = np.kron(np.eye(d), op.T.dot(np.conj(op)))

    D_op = gamma * (op1 - 0.5 * op2 - 0.5 * op3)
    # D_op = new_lindbladian(op, gamma)

    eop = expm(dt * D_op)
    # print(f"D =\n {eop}, \n---\n {eop.T}")
    u, s, v = svd(eop)
    # s = s/sum(s**2)
    s = s ** (1 / 2)
    S = np.diag(s)  # **1/2
    # S = S/np.sum(S**2)
    B = u.dot(S)
    B_dag = S.dot(v)

    eop1 = B @ B.T
    assert eop1.all() == eop.all()
    B = np.reshape(B, (d, d, d * d))
    return B


def apply_noise_evolution(mpo: MPDO, gamma, op, dt_list, d=2, Kr=4):

    M_list = []
    for i in range(len(mpo.Ms)):
        B = kraus_op_svd(op, gamma, dt_list[i], d)
        M = np.tensordot(B, mpo.Ms[i], axes=([1], [0]))  #  (p, k, q, a_l, a_r)
        M = np.transpose(M, (1, 2, 0, 3, 4))  #  (k, q, p, a_l, a_r)
        k = M.shape[0]
        q = M.shape[1]
        a_l = M.shape[3]
        a_r = M.shape[4]
        M = np.reshape(M, (k * q, d * a_l * a_r))  #  (k*q, p*a_l*a_r)
        X, Y, Z = svd(M)  # (k*q, g), (g), (g, p*a_l*a_r)
        tmp = np.linalg.norm(Y[:Kr])
        S = Y[:Kr]  # / tmp
        Z = Z[:Kr, :]
        M_new = np.diag(S) @ Z
        # print(M_new.shape)
        M_new = np.reshape(M_new, (Kr, d, a_l, a_r))
        # print(M_new.shape)
        M_new = np.transpose(M_new, (1, 0, 2, 3))
        M_list.append(M_new)
    mpo.Ms = M_list
    return mpo


def tebd_alg_choi(
    rho: MPDO, Hs: List, chi_max, L, dt_list, epsilon=10 ** (-8), coeff=0.5, d=2
):
    """tebd algorithm based on Zowlack-Vidal paper.

    Args:
        rho (MPDO): _description_
        H_b (BondHamiltonian): _description_
        chi_max (_type_): _description_
        L (_type_): _description_
        dt_list (_type_): _description_
        epsilon (_type_, optional): _description_. Defaults to 10**(-8).
        coeff (float, optional): _description_. Defaults to 0.5.
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    j = 0
    k = 0
    discarded = 0.0
    site = 0

    ##  odd iterations: coeff = 0.5  &  even iterations: coeff = 1.0
    while 2 * j + int(coeff) < L - 1:

        p = 2 * j + int(coeff)
        q = p + 1
        site = p
        # print(site, p, q, j,  2*j+int(coeff))
        dt = dt_list[site]
        # print(dt)
        # print( "rank = {} , site = {} , dt = {}, coeff = {}, sub_L = {} ".format(rank, site, dt, coeff, sub_L))
        # H_bond = H.Hamiltonian(Jx_list, Jy_list, Delta_list, H_list, mu_list, L, site);

        H_bond = Hs[site]  # H_bond_choi(H_b, site)
        Ubond = np.reshape(
            expm(coeff * dt * H_bond), (d,) * 8
        )  # (p_i, p_i+1, p_i, p_i+1, q_i, q_i+1, q_i, q_i+1) ;or;  (q_i, q_i+1, q_i, q_i+1)

        theta = np.tensordot(
            rho.Ms[p], rho.Ms[q], axes=([3], [2])
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # if side == 'ket':
        theta = np.tensordot(
            Ubond, theta, axes=([4, 5, 6, 7], [0, 1, 3, 4])
        )  # (p_i, p_i+1, q_i, q_i+1, a_i, a_i+2)
        theta = np.transpose(
            theta, (0, 2, 4, 1, 3, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # elif side == 'bra':
        #    theta = np.tensordot(Uodd, theta, axes = ([2,3],[1,4]))  #(q_i, q_i+1, p_i, a_i, p_i+1, a_i+2)
        #    theta = np.transpose(theta, (2,0,3,4,1,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # theta = np.transpose(theta, (0,2,3,1,4,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)
        theta_bar = theta

        # print 'in the beginning=',theta.shape, np.diag(Llist[p]).shape, blist[p].shape
        theta = np.tensordot(
            np.diag(rho.Ss[p]), theta, axes=([1], [2])
        )  # (a_i, p_i, q_i, p_i+1, q_i+1, a_i+2)
        theta = np.transpose(
            theta, (1, 2, 0, 3, 4, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        a = d * d * rho.Ms[p].shape[2]
        b = d * d * rho.Ms[q].shape[3]
        # print j, site, coeff, a, b

        theta = np.reshape(theta, (a, b))
        # print theta_bar.shape

        # SVD:
        try:
            X, Y, Z = svd(theta, compute_uv=True, full_matrices=True)
        except np.linalg.linalg.LinAlgError:
            print("SVD did not converge, diagonalizing theta_dagger*theta")
            Y, Z = np.linalg.eigh(np.dot(theta.conj().T, theta))
            piv = np.argsort(Y)[::-1]
            Y = np.sqrt(np.abs(Y[piv]))
            Z = np.conj(Z[:, piv].T)

        rho.bonds[q] = np.min([np.sum(Y > epsilon), chi_max])
        # print(rho.bonds)
        discarded += np.sum(Y[rho.bonds[q] :] ** 2) / sum(Y ** 2)
        # print(discarded)
        tmp = np.linalg.norm(Y[: rho.bonds[q]])

        rho.Ss[q] = Y[: rho.bonds[q]] / tmp
        rho.Ms[q] = np.transpose(
            np.reshape(
                Z[: rho.bonds[q], :], (rho.bonds[q], d, d, int(Z.shape[1] / (d * d)))
            ),
            (1, 2, 0, 3),
        )  # (p_i+1, q_i+1, a_i+1, a_i+2)

        rho.Ms[p] = (
            np.tensordot(
                theta_bar, np.conjugate(rho.Ms[q]).T, axes=([3, 4, 5], [3, 2, 0])
            )
            / tmp
        )  # (p_i, q_i, a_i, a_i+1)
        # print blist[p].shape, blist[q].shape
        j += 1
        # print 'updated=',blist[q].shape, np.diag(Llist[p]).shape, blist[p].shape

    return rho, discarded


def tebd_mpdo(
    rho: MPDO, Hs: List, chi_max, dt_list, lattice_sites, epsilon=10 ** (-8), d=2
):
    discarded = 0.0

    ##  odd iterations: coeff = 0.5  &  even iterations: coeff = 1.0
    # if sites == "odd":
    #     lattice_sites = range(1, L - 1, 2)
    # else:
    #     lattice_sites = range(0, L, 2)

    for p in lattice_sites:

        q = p + 1
        site = p
        # print(site, p, q)
        dt = dt_list[site]
        # print(dt)
        # print( "rank = {} , site = {} , dt = {}, coeff = {}, sub_L = {} ".format(rank, site, dt, coeff, sub_L))
        # H_bond = H.Hamiltonian(Jx_list, Jy_list, Delta_list, H_list, mu_list, L, site);
        H_bond = Hs[site]  # H_bond_choi(H_b, site)
        # H_bond = np.kron( H_bond, np.eye(4)) + np.kron(np.eye(4), H_bond.T)

        Ubond = np.reshape(
            expm(-dt * H_bond), (d,) * 8
        )  # (p_i, p_i+1, p_i, p_i+1, q_i, q_i+1, q_i, q_i+1) ;or;  (q_i, q_i+1, q_i, q_i+1)
        # Ubond = np.transpose(Ubond, (4,5,6,7,0,1,2,3))
        # Ubond = 0.5*(np.reshape(expm(dt * H_bond), (d,) * 8) + np.transpose(np.reshape(expm(dt * H_bond), (d,) * 8)))

        theta = np.tensordot(
            rho.Ms[p], rho.Ms[q], axes=([3], [2])
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # if side == 'ket':
        theta = np.tensordot(
            Ubond, theta, axes=([4, 5, 6, 7], [0, 3, 1, 4])
        )  # (p_i, p_i+1, q_i, q_i+1, a_i, a_i+2)
        theta = np.transpose(
            theta, (0, 2, 4, 1, 3, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # elif side == 'bra':
        #    theta = np.tensordot(Uodd, theta, axes = ([2,3],[1,4]))  #(q_i, q_i+1, p_i, a_i, p_i+1, a_i+2)
        #    theta = np.transpose(theta, (2,0,3,4,1,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # theta = np.transpose(theta, (0,2,3,1,4,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)
        theta_bar = theta

        # print 'in the beginning=',theta.shape, np.diag(Llist[p]).shape, blist[p].shape

        ####
        theta = np.tensordot(
            np.diag(rho.Ss[p]), theta, axes=([1], [2])
        )  # (a_i, p_i, q_i, p_i+1, q_i+1, a_i+2)
        theta = np.transpose(
            theta, (1, 2, 0, 3, 4, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        a = d * d * rho.Ms[p].shape[2]
        b = d * d * rho.Ms[q].shape[3]
        # print j, site, coeff, a, b

        theta = np.reshape(theta, (a, b))
        # print theta_bar.shape

        # SVD:
        try:
            X, Y, Z = svd(theta, compute_uv=True, full_matrices=True)
        except np.linalg.linalg.LinAlgError:
            print("SVD did not converge, diagonalizing theta_dagger*theta")
            Y, Z = np.linalg.eigh(np.dot(theta.conj().T, theta))
            piv = np.argsort(Y)[::-1]
            Y = np.sqrt(np.abs(Y[piv]))
            Z = np.conj(Z[:, piv].T)

        rho.bonds[q] = np.min([np.sum(Y > epsilon), chi_max])
        # print(rho.bonds)
        discarded += np.sum(Y[rho.bonds[q] :]) / sum(Y ** 2)
        # print(discarded)
        tmp = np.linalg.norm(Y[: rho.bonds[q]])

        rho.Ss[q] = Y[: rho.bonds[q]] / tmp
        rho.Ms[q] = np.transpose(
            np.reshape(
                Z[: rho.bonds[q], :], (rho.bonds[q], d, d, int(Z.shape[1] / (d * d)))
            ),
            (1, 2, 0, 3),
        )  # (p_i+1, q_i+1, a_i+1, a_i+2)

        rho.Ms[p] = (
            np.tensordot(
                theta_bar, np.conjugate(rho.Ms[q]).T, axes=([3, 4, 5], [3, 2, 0])
            )
            / tmp
        )  # (p_i, q_i, a_i, a_i+1)
        # print blist[p].shape, blist[q].shape

        # print 'updated=',blist[q].shape, np.diag(Llist[p]).shape, blist[p].shape
    return rho, discarded


def tebd_mpdo_alternative(
    rho: MPDO, Hs: List, chi_max, dt_list, lattice_sites, epsilon=10 ** (-8), d=2
):
    discarded = 0.0

    ##  odd iterations: coeff = 0.5  &  even iterations: coeff = 1.0
    # if sites == "odd":
    #     lattice_sites = range(1, L - 1, 2)
    # else:
    #     lattice_sites = range(0, L, 2)

    for p in lattice_sites:

        q = p + 1
        site = p
        # print(site, p, q)
        dt = dt_list[site]
        # print(dt)
        # print( "rank = {} , site = {} , dt = {}, coeff = {}, sub_L = {} ".format(rank, site, dt, coeff, sub_L))
        # H_bond = H.Hamiltonian(Jx_list, Jy_list, Delta_list, H_list, mu_list, L, site);
        H_bond = Hs[site]  # H_bond_choi(H_b, site)
        Ubond = np.reshape(
            expm(dt * H_bond), (d,) * 8
        )  # (p_i, p_i+1, p_i, p_i+1, q_i, q_i+1, q_i, q_i+1) ;or;  (q_i, q_i+1, q_i, q_i+1)

        theta = np.tensordot(
            rho.Ms[p], rho.Ms[q], axes=([3], [2])
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # if side == 'ket':
        theta = np.tensordot(
            Ubond, theta, axes=([4, 5, 6, 7], [0, 3, 1, 4])
        )  # (p_i, p_i+1, q_i, q_i+1, a_i, a_i+2)
        theta = np.transpose(
            theta, (0, 2, 4, 1, 3, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # elif side == 'bra':
        #    theta = np.tensordot(Uodd, theta, axes = ([2,3],[1,4]))  #(q_i, q_i+1, p_i, a_i, p_i+1, a_i+2)
        #    theta = np.transpose(theta, (2,0,3,4,1,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # theta = np.transpose(theta, (0,2,3,1,4,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)
        theta_bar = theta

        # print 'in the beginning=',theta.shape, np.diag(Llist[p]).shape, blist[p].shape
        theta = np.tensordot(
            np.diag(rho.Ss[p]), theta, axes=([1], [2])
        )  # (a_i, p_i, q_i, p_i+1, q_i+1, a_i+2)
        theta = np.transpose(
            theta, (1, 2, 0, 3, 4, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        a = d * d * rho.Ms[p].shape[2]
        b = d * d * rho.Ms[q].shape[3]
        # print j, site, coeff, a, b

        theta = np.reshape(theta, (a, b))
        # print theta_bar.shape

        # SVD:
        try:
            X, Y, Z = svd(theta, compute_uv=True, full_matrices=True)
        except np.linalg.linalg.LinAlgError:
            print("SVD did not converge, diagonalizing theta_dagger*theta")
            Y, Z = np.linalg.eigh(np.dot(theta.conj().T, theta))
            piv = np.argsort(Y)[::-1]
            Y = np.sqrt(np.abs(Y[piv]))
            Z = np.conj(Z[:, piv].T)

        rho.bonds[q] = np.min([np.sum(Y > epsilon), chi_max])
        # print(rho.bonds)
        discarded += np.sum(Y[rho.bonds[q] :] ** 2) / sum(Y ** 2)
        # print(discarded)
        tmp = np.linalg.norm(Y[: rho.bonds[q]])

        rho.Ss[q] = Y[: rho.bonds[q]] / tmp
        rho.Ms[q] = np.transpose(
            np.reshape(
                Z[: rho.bonds[q], :], (rho.bonds[q], d, d, int(Z.shape[1] / (d * d)))
            ),
            (1, 2, 0, 3),
        )  # (p_i+1, q_i+1, a_i+1, a_i+2)

        rho.Ms[p] = (
            np.tensordot(
                theta_bar, np.conjugate(rho.Ms[q]).T, axes=([3, 4, 5], [3, 2, 0])
            )
            / tmp
        )  # (p_i, q_i, a_i, a_i+1)
        # print blist[p].shape, blist[q].shape

        # print 'updated=',blist[q].shape, np.diag(Llist[p]).shape, blist[p].shape
    return rho, discarded


def TEBD_alg_S(
    psi: MPDO, Hs: List, chi_max, dt_list, lattice_sites, epsilon=10 ** (-8), d=2
):
    """TEBD algorithm for locally puridied tensor to evolve only physical inices.
    Note that the lattice sites have following conditions.
    even sites: range(0, L, 2)
    odd sites: range(1, L-1, 2)

    Args:
        psi (MPDO): _description_
        Hs (List): _description_
        chi_max (_type_): _description_
        dt_list (_type_): _description_
        epsilon (_type_, optional): _description_. Defaults to 10**(-8).
        lattice_sites (_type_, optional): _description_. Defaults to range(0, 8, 2).
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    discarded = 0.0

    for p in lattice_sites:

        ad = psi.Ms[p].shape[1]
        site = p
        q = p + 1
        dt = dt_list[site]
        # print "rank = {} , site = {} , dt = {}, coeff = {}, sub_L = {} ".format(rank, site, dt, coeff, sub_L)
        H_bond = Hs[site]
        Uodd = np.reshape(
            expm(-1j * dt * H_bond), (d, d, d, d)
        )  # (p_i, p_i+1, p_i, p_i+1) ;or;  (q_i, q_i+1, q_i, q_i+1)

        theta = np.tensordot(
            psi.Ms[p], psi.Ms[q], axes=([3], [2])
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # if phys == True:
        theta = np.tensordot(
            Uodd, theta, axes=([2, 3], [0, 3])
        )  # (p_i, p_i+1, q_i, a_i, q_i+1, a_i+2)
        theta = np.transpose(
            theta, (0, 2, 3, 1, 4, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # elif phys == False:
        # theta = np.tensordot(Uodd, theta, axes = ([2,3],[1,4]))  #(q_i, q_i+1, p_i, a_i, p_i+1, a_i+2)
        # theta = np.transpose(theta, (2,0,3,4,1,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        # theta = np.transpose(theta, (0,2,3,1,4,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)
        theta_bar = theta

        # print 'in the beginning=',theta.shape, np.diag(Llist[p]).shape, blist[p].shape
        theta = np.tensordot(
            np.diag(psi.Ss[p]), theta, axes=([1], [2])
        )  # (a_i, p_i, q_i, p_i+1, q_i+1, a_i+2)
        theta = np.transpose(
            theta, (1, 2, 0, 3, 4, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        a = d * ad * psi.Ms[p].shape[2]
        b = d * ad * psi.Ms[q].shape[3]
        # print j, site, coeff, a, b

        theta = np.reshape(theta, (a, b))
        # print theta_bar.shape

        # SVD:
        try:
            X, Y, Z = svd(theta, compute_uv=True, full_matrices=True)
        except np.linalg.linalg.LinAlgError:
            print("SVD did not converge, diagonalizing theta_dagger*theta")
            Y, Z = np.linalg.eigh(np.dot(theta.conj().T, theta))
            piv = np.argsort(Y)[::-1]
            Y = np.sqrt(np.abs(Y[piv]))
            Z = np.conj(Z[:, piv].T)

        psi.bonds[q] = np.min([np.sum(Y > epsilon), chi_max])
        # print site, coeff, chi_vec[q]
        discarded = np.sum(Y[psi.bonds[q] :] ** 2) / sum(Y ** 2)
        tmp = np.linalg.norm(Y[: psi.bonds[q]])

        psi.Ss[q] = Y[: psi.bonds[q]] / tmp
        psi.Ms[q] = np.transpose(
            np.reshape(
                Z[: psi.bonds[q], :], (psi.bonds[q], d, ad, int(Z.shape[1] / (d * ad)))
            ),
            (1, 2, 0, 3),
        )  # (p_i+1, q_i+1, a_i+1, a_i+2)

        psi.Ms[p] = (
            np.tensordot(
                theta_bar, np.conjugate(psi.Ms[q]).T, axes=([3, 4, 5], [3, 2, 0])
            )
            / tmp
        )  # (p_i, q_i, a_i, a_i+1)
        # print blist[p].shape, blist[q].shape

        # print 'updated=',blist[q].shape, np.diag(Llist[p]).shape, blist[p].shape

    return psi, discarded


def TEBD_alg_S_A(
    rho: MPDO, Hs: List, chi_max, dt_list, lattice_sites, epsilon=10 ** (-8), d=2
):

    discarded = 0.0

    for p in lattice_sites:

        q = p + 1
        site = p
        dt = dt_list[site]

        H_bond = Hs[site]
        U_p = np.reshape(
            expm(-dt * H_bond), (2, 2, 2, 2)
        )  # (p_i, p_i+1, p_i, p_i+1) ;or;  (q_i, q_i+1, q_i, q_i+1)
        dt = -dt
        U_q = np.reshape(
            expm(-dt * H_bond.T), (2, 2, 2, 2)
        )  # (p_i, p_i+1, p_i, p_i+1) ;or;  (q_i, q_i+1, q_i, q_i+1)

        theta = np.tensordot(
            rho.Ms[p], rho.Ms[q], axes=([3], [2])
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        """evolve the physical indices"""
        theta = np.tensordot(
            U_p, theta, axes=([2, 3], [0, 3])
        )  # (p_i, p_i+1, q_i, a_i, q_i+1, a_i+2)
        # theta = np.transpose(theta, (0,2,3,1,4,5))               #(p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)
        """evolve the auxiliary indices"""
        theta = np.tensordot(
            U_q, theta, axes=([2, 3], [2, 4])
        )  # (q_i, q_i+1, p_i, p_i+1, a_i, a_i+2)
        theta = np.transpose(
            theta, (2, 0, 4, 3, 1, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        theta_bar = theta

        theta = np.tensordot(
            np.diag(rho.Ss[p]), theta, axes=([1], [2])
        )  # (a_i, p_i, q_i, p_i+1, q_i+1, a_i+2)
        theta = np.transpose(
            theta, (1, 2, 0, 3, 4, 5)
        )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)

        a = d * d * rho.Ms[p].shape[2]
        b = d * d * rho.Ms[q].shape[3]
        # print rank,j, p,q,site, coeff,'\n' #, a, b, theta.shape, blist[p].shape, blist[q].shape

        theta = np.reshape(theta, (a, b))
        # print "theta shape = ", theta.shape
        """SVD:"""
        try:
            X, Y, Z = svd(theta, compute_uv=True, full_matrices=True)
        except np.linalg.linalg.LinAlgError:
            print("SVD did not converge, diagonalizing theta_dagger*theta")
            Y, Z = np.linalg.eigh(np.dot(theta.conj().T, theta))
            piv = np.argsort(Y)[::-1]
            Y = np.sqrt(np.abs(Y[piv]))
            Z = np.conj(Z[:, piv].T)

        rho.bonds[q] = np.min([np.sum(Y > epsilon), chi_max])
        # print site, coeff, chi_vec[q]
        discarded = np.sum(Y[rho.bonds[q] :] ** 2) / sum(Y ** 2)
        tmp = np.linalg.norm(Y[: rho.bonds[q]])

        rho.Ss[q] = Y[: rho.bonds[q]] / tmp
        rho.Ms[q] = np.transpose(
            np.reshape(
                Z[: rho.bonds[q], :], (rho.bonds[q], d, d, int(Z.shape[1] / (d * d)))
            ),
            (1, 2, 0, 3),
        )  # (p_i+1, q_i+1, a_i+1, a_i+2)

        rho.Ms[p] = (
            np.tensordot(
                theta_bar, np.conjugate(rho.Ms[q]).T, axes=([3, 4, 5], [3, 2, 0])
            )
            / tmp
        )  # (p_i, q_i, a_i, a_i+1)
        # print blist[p].shape, blist[q].shape

    return rho, discarded


def tebd_vidal_mpdo(rho: MPDO, Hs: List, chi_max: int, dt_list, lattice_sites, epsilon, d=4):
    discarded = 0.0

    for p in lattice_sites:
        q = p + 1
        # print(p,q,list(lattice_sites))
        # constructing Hamiltonian
        H_b = Hs[p]
        H_b = np.reshape(H_b, (2,) * 8)
        H_b = np.transpose(H_b, (0, 2, 1, 3, 4, 6, 5, 7))
        H_b = np.reshape(H_b, (16, 16))
        # print coeff, site,'H_b parallel', dt, '\n'
        U_ev = np.reshape(expm(dt_list[p] * H_b), (d, d, d, d))
        # print(U_ev.shape)
        # print "p,q=",p, q , "length", len(Blist), len(llist), j, rank*(L/4)+p , rank  , site
        theta = np.tensordot(rho.Ms[p], rho.Ms[q], axes=(2, 1))  # (i, a_l, j, a_l+2)

        # theta_prime = np.tensordot(np.diag(llist[p]),theta, axes=(1,1))    #(a_l, i, j, a_l+2)

        # apply gate
        theta = np.tensordot(U_ev, theta, axes=([2, 3], [0, 2]))  # (k, l , a_l, a_l+2)

        # this is kept for later use
        theta_bar = theta
        # construct theta by applying lambda
        # theta shape --> (d, d, alpha_l, alpha_l+2)
        a = d * rho.bonds[p]
        b = d * rho.Ms[q].shape[2]
        # print 'TEBD',llist[p].shape, theta.shape, p , a
        theta = np.reshape(np.tensordot(np.diag(rho.Ss[p]), theta, axes=(1, 2)), (a, b))
        # now do svd on thet
        try:
            X, Y, Z = np.linalg.svd(theta, compute_uv=True, full_matrices=True)
        except np.linalg.linalg.LinAlgError:
            print("SVD did not converge, diagonalizing theta_dagger*theta")
            Y, Z = np.linalg.eigh(np.dot(theta.conj().T, theta))
            piv = np.argsort(Y)[::-1]
            Y = np.sqrt(np.abs(Y[piv]))
            Z = np.conj(Z[:, piv].T)

        rho.bonds[q] = np.min([np.sum(Y > epsilon), chi_max])
        # print ChiVec[q]
        # now new B next
        # Blist[q] = np.transpose(np.reshape(Z[:ChiVec[q], :], (ChiVec[q],d,Z.shape[1]/d)),(1,0,2))
        tmp = np.linalg.norm(Y[: rho.bonds[q]])
        rho.Ss[q] = Y[: rho.bonds[q]] / tmp
        discarded = np.sum(Y[rho.bonds[q] :] ** 2) / sum(Y ** 2)
        rho.Ms[q] = np.transpose(
            np.reshape(Z[: rho.bonds[q], :], (rho.bonds[q], d, int(Z.shape[1] / d))),
            (1, 0, 2),
        )
        # now B previous
        rho.Ms[p] = (
            np.tensordot(theta_bar, np.conjugate(rho.Ms[q]).T, axes=([1, 3], [2, 0]))
            / tmp
        )

    return rho, discarded


def tebd_2nd_order_vidal_mpdo(
    rho: MPDO, Hs: List, chi_max: int, dt_list: np.ndarray, epsilon=1e-6, d=4
):
    truncation_err = 0
    L = len(rho.Ms)
    rho, discarded = tebd_vidal_mpdo(
        rho=rho,
        Hs=Hs,
        chi_max=chi_max,
        dt_list=0.5 * dt_list,
        lattice_sites=range(0, L, 2),
        epsilon=epsilon,
        d=d,
    )
    truncation_err += discarded
    rho, discarded = tebd_vidal_mpdo(
        rho=rho,
        Hs=Hs,
        chi_max=chi_max,
        dt_list=1.0 * dt_list,
        lattice_sites=range(1, L - 1, 2),
        epsilon=epsilon,
        d=d,
    )
    truncation_err += discarded
    rho, discarded = tebd_vidal_mpdo(
        rho=rho,
        Hs=Hs,
        chi_max=chi_max,
        dt_list=0.5 * dt_list,
        lattice_sites=range(0, L, 2),
        epsilon=epsilon,
        d=d,
    )
    truncation_err += discarded
    return rho, truncation_err


def tebd_vidal_mpdo_mpi4py_even(
    rho: MPDO, Hs: List, chi_max: int, dt_list: np.ndarray, epsilon=1e-6, d=4
):
    """_summary_

    Args:
        rho (MPDO): _description_
        Hs (List): _description_
        chi_max (int): _description_
        dt_list (np.ndarray): _description_
        epsilon (_type_, optional): _description_. Defaults to 1e-6.
        d (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    L = len(rho.Ms)

    rank_L = int(L / size)
    assert np.mod(L, size) == 0
    truncation_err = 0
    data = {}
    error = {}

    data[f"{rank}"], error[f"{rank}"] = tebd_vidal_mpdo(
        rho,
        Hs,
        chi_max,
        dt_list,
        lattice_sites=range(rank * rank_L, rank * rank_L + rank_L, 2),
        epsilon=epsilon,
        d=d,
    )
    if rank != 0:
        comm.send((data[f"{rank}"], error[f"{rank}"]), tag=rank, dest=0)

    if rank == 0:
        rho.Ms[rank * rank_L : rank * rank_L + rank_L] = data[f"{rank}"].Ms[rank * rank_L : rank * rank_L + rank_L]
        rho.Ss[rank * rank_L : rank * rank_L + rank_L] = data[f"{rank}"].Ss[rank * rank_L : rank * rank_L + rank_L]
        rho.bonds[rank * rank_L : rank * rank_L + rank_L] = data[f"{rank}"].bonds[rank * rank_L : rank * rank_L + rank_L]
        truncation_err += error[f"{rank}"]
        for i in range(1, size):
            data[f"{i}"], error[f"{i}"] = comm.recv(source=i, tag=i)
            truncation_err += error[f"{i}"]
            rho.Ms[i * rank_L : i * rank_L + rank_L] = data[f"{i}"].Ms[i * rank_L : i * rank_L + rank_L]
            rho.Ss[i * rank_L : i * rank_L + rank_L] = data[f"{i}"].Ss[i * rank_L : i * rank_L + rank_L]
            rho.bonds[i * rank_L : i * rank_L + rank_L] = data[f"{i}"].bonds[i * rank_L : i * rank_L + rank_L]
    rho, truncation_err = comm.bcast((rho, truncation_err), root=0)
    return rho, truncation_err


def tebd_vidal_mpdo_mpi4py_odd(
    rho: MPDO, Hs: List, chi_max: int, dt_list: np.ndarray, epsilon=1e-6, d=4
):
    """_summary_

    Args:
        rho (MPDO): _description_
        Hs (List): _description_
        chi_max (int): _description_
        dt_list (np.ndarray): _description_
        epsilon (_type_, optional): _description_. Defaults to 1e-6.
        d (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    L = len(rho.Ms)

    rank_L = int(L / size)
    assert np.mod(L, size) == 0
    truncation_err = 0
    data = {}
    error = {}

    if rank != size - 1:
        # print("nnn",list(range(rank*rank_L+1, rank*rank_L+rank_L+1, 2)))
        data[f"{rank}"], error[f"{rank}"] = tebd_vidal_mpdo(
            rho,
            Hs,
            chi_max,
            dt_list,
            lattice_sites=range(rank * rank_L + 1, rank * rank_L + rank_L + 1, 2),
            epsilon=epsilon,
            d=d,
        )
        # comm.send((data[f"{rank}"], error[f"{rank}"]), tag=rank, dest=0)
    if rank == size - 1:
        # print("nnn",list(range(rank*rank_L+1, L-1, 2)))
        data[f"{rank}"], error[f"{rank}"] = tebd_vidal_mpdo(
            rho,
            Hs,
            chi_max,
            dt_list,
            lattice_sites=range(rank * rank_L + 1, L - 1, 2),
            epsilon=epsilon,
            d=d,
        )
        # comm.send((data[f"{rank}"], error[f"{rank}"]), tag=rank, dest=0)

    if rank != 0:
        comm.send((data[f"{rank}"], error[f"{rank}"]), tag=rank, dest=0)

    if rank == 0:
        rho.Ms[rank * rank_L + 1 : rank * rank_L + rank_L + 1] = data[f"{rank}"].Ms[rank * rank_L + 1 : rank * rank_L + rank_L + 1]
        rho.Ss[rank * rank_L + 1 : rank * rank_L + rank_L + 1] = data[f"{rank}"].Ss[rank * rank_L + 1 : rank * rank_L + rank_L + 1]
        rho.bonds[rank * rank_L + 1 : rank * rank_L + rank_L + 1] = data[f"{rank}"].bonds[rank * rank_L + 1 : rank * rank_L + rank_L + 1]
        truncation_err += error[f"{rank}"]
        for i in range(1, size):
            if i != size - 1:
                data[f"{i}"], error[f"{i}"] = comm.recv(source=i, tag=i)
                truncation_err += error[f"{i}"]
                rho.Ms[i * rank_L + 1 : i * rank_L + rank_L + 1] = data[f"{i}"].Ms[i * rank_L + 1 : i * rank_L + rank_L + 1]
                rho.Ss[i * rank_L + 1 : i * rank_L + rank_L + 1] = data[f"{i}"].Ss[i * rank_L + 1 : i * rank_L + rank_L + 1]
                rho.bonds[i * rank_L + 1 : i * rank_L + rank_L + 1] = data[f"{i}"].bonds[i * rank_L + 1 : i * rank_L + rank_L + 1]
            else:
                data[f"{i}"], error[f"{i}"] = comm.recv(source=i, tag=i)
                truncation_err += error[f"{i}"]
                rho.Ms[i * rank_L + 1 : L - 1] = data[f"{i}"].Ms[i * rank_L + 1 : L - 1]
                rho.Ss[i * rank_L + 1 : L - 1] = data[f"{i}"].Ss[i * rank_L + 1 : L - 1]
                rho.bonds[i * rank_L + 1 : L - 1] = data[f"{i}"].bonds[i * rank_L + 1 : L - 1]
    rho, truncation_err = comm.bcast((rho, truncation_err), root=0)
    return rho, truncation_err



def tebd_2nd_order_vidal_mpdo_mpi4py_efficient(steps, rho: MPDO, Hs: List, chi_max: int, dt_list: np.ndarray, epsilon=1e-6, d=4):
    truncation_err = 0
    rho, discarded = tebd_vidal_mpdo_mpi4py_even(rho, Hs, chi_max, dt_list=0.5 * dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    for _ in range(steps-1):
        rho, discarded = tebd_vidal_mpdo_mpi4py_odd(rho, Hs, chi_max, dt_list=dt_list, epsilon=epsilon, d=d)
        truncation_err += discarded
        rho, discarded = tebd_vidal_mpdo_mpi4py_even(rho, Hs, chi_max, dt_list=dt_list, epsilon=epsilon, d=d)
        truncation_err += discarded
    rho, discarded = tebd_vidal_mpdo_mpi4py_odd(rho, Hs, chi_max, dt_list=dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    rho, discarded = tebd_vidal_mpdo_mpi4py_even(rho, Hs, chi_max, dt_list=0.5 * dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    return rho, truncation_err



def tebd_2nd_order_vidal_mpdo_mpi4py(rho: MPDO, Hs: List, chi_max: int, dt_list: np.ndarray, epsilon=1e-6, d=4):
    truncation_err = 0
    rho, discarded = tebd_vidal_mpdo_mpi4py_even(rho, Hs, chi_max, dt_list=0.5 * dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    # print("------------------1")
    rho, discarded = tebd_vidal_mpdo_mpi4py_odd(rho, Hs, chi_max, dt_list=dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    # print("----------------------2")
    rho, discarded = tebd_vidal_mpdo_mpi4py_even(rho, Hs, chi_max, dt_list=0.5 * dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    return rho, truncation_err


def tebd_1st_order_vidal_mpdo_mpi4py(rho: MPDO, Hs: List, chi_max: int, dt_list: np.ndarray, epsilon=1e-6, d=4):
    truncation_err = 0
    rho, discarded = tebd_vidal_mpdo_mpi4py_even(rho, Hs, chi_max, dt_list=dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    # print("------------------1")
    rho, discarded = tebd_vidal_mpdo_mpi4py_odd(rho, Hs, chi_max, dt_list=dt_list, epsilon=epsilon, d=d)
    truncation_err += discarded
    # print("----------------------2")
    return rho, truncation_err



def one_time_step_2nd_order(rho, Hs, chi_max, L, dt_list, epsilon=10 ** (-8)):
    disc = 0
    rho, disc_err = tebd_mpdo(
        rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon
    )
    disc += disc_err
    rho, disc_err = tebd_mpdo(
        rho, Hs, chi_max, dt_list, lattice_sites=range(1, L - 1, 2), epsilon=epsilon
    )
    disc += disc_err
    rho, disc_err = tebd_mpdo(
        rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon
    )
    disc += disc_err
    return rho, disc


def one_time_step_2nd_order_2approach(
    rho, Hs, Gs, chi_max, L, dt_list, epsilon=10 ** (-8)
):
    disc = 0
    rho, disc_err = tebd_mpdo(
        rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon
    )
    disc += disc_err
    rho, disc_err = tebd_mpdo(rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(1, L - 1, 2), epsilon=epsilon)
    disc += disc_err

    rho = apply_mixed_gate_mpdo(rho, Gs)

    rho, disc_err = tebd_mpdo(rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon)
    disc += disc_err
    rho, disc_err = tebd_mpdo(rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(1, L - 1, 2), epsilon=epsilon)
    disc += disc_err
    return rho, disc_err


def tebd_2nd_order_LPTN_thermal(
    rho, Hs, chi_max, L, dt_list, epsilon=10 ** (-10), evolve_auxiliary=True
):

    if evolve_auxiliary:
        disc = 0
        rho, discard = TEBD_alg_S_A(rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon)
        disc += discard

        rho, discard = TEBD_alg_S_A(rho, Hs, chi_max, dt_list, lattice_sites=range(1, L - 1, 2), epsilon=epsilon)
        disc += discard

        rho, discard = TEBD_alg_S_A(rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon)
        disc += discard

    else:
        disc = 0
        rho, discard = TEBD_alg_S(rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon)
        disc += discard

        rho, discard = TEBD_alg_S(rho, Hs, chi_max, dt_list, lattice_sites=range(1, L - 1, 2), epsilon=epsilon)
        disc += discard

        rho, discard = TEBD_alg_S(rho, Hs, chi_max, 0.5 * dt_list, lattice_sites=range(0, L, 2), epsilon=epsilon)
        disc += discard

    return rho, disc


if __name__ == "__main__":

    class MyObj:
        def __init__(self, L):
            self.L = L
            self.rho = [np.array([0])] * L

    def manipulate(my_obj: MyObj, sites: List):
        for i in sites:
            my_obj.rho[i] = np.array([i])
        return my_obj

    def mpi4py_example():
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        L = 16

        rank_L = int(L / size)
        assert np.mod(L, size) == 0
        truncation_err = 0
        data = {}
        err = {}
        # if rank == 0:
        #    my_obj = MyObj(L)
        # else:
        #    my_obj = None
        # my_obj = comm.bcast(my_obj, root=0)
        my_obj = MyObj(L)
        if rank != size - 1:
            print(rank, list(range(rank * rank_L + 1, rank * rank_L + rank_L + 1, 2)))
            data[f"{rank}"] = manipulate(
                my_obj, range(rank * rank_L + 1, rank * rank_L + rank_L + 1)
            )
            err[f"{rank}"] = rank
        else:
            print(rank, list(range(rank * rank_L + 1, L - 1, 2)))
            data[f"{rank}"] = manipulate(my_obj, range(rank * rank_L + 1, L - 1, 2))
            err[f"{rank}"] = rank

        if rank != 0:
            comm.send((data[f"{rank}"], err[f"{rank}"]), tag=rank, dest=0)

        if rank == 0:
            my_obj.rho[rank * rank_L : rank * rank_L + rank_L] = data[f"{rank}"].rho[
                rank * rank_L : rank * rank_L + rank_L
            ]
            for i in range(1, size):
                data[f"{i}"], err[f"{i}"] = comm.recv(source=i, tag=i)
                print(err[f"{i}"])
                my_obj.rho[i * rank_L : i * rank_L + rank_L] = data[f"{i}"].rho[
                    i * rank_L : i * rank_L + rank_L
                ]
            print(
                f"size={size}, rank={rank}, \n my_list={my_obj.rho}",
                "here:",
                err[f"{rank}"],
            )

    mpi4py_example()
