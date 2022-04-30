from cmath import exp
from lib2to3.pgen2.token import OP

import numpy as np
from scipy.linalg import expm

from tensor_networks_simulations.general_tools.ED_tools import spin_operator


class Operator(np.ndarray):
    @property
    def dag(self):
        return self.conj().T


class BondHamiltonian:
    """Define the Hamiltonian for nearest-neighbour interactions."""

    def __init__(self, L, Jxs, Jys, Jzs, Hxs, Hzs, mus, d=2):
        self.L = L
        self.d = d
        self.Jxs = Jxs
        self.Jys = Jys
        self.Jzs = Jzs
        self.Hxs = Hxs
        self.Hzs = Hzs
        self.mus = mus
        # Pauli matrices
        self.s0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.sx = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.sy = np.array([[0.0, -1j], [1j, 0.0]])
        self.sz = np.array([[1.0, 0.0], [0.0, -1.0]])
        self.sp = (self.sx + 1j * self.sy) / 2
        self.sm = (self.sx - 1j * self.sy) / 2
        self.sn = np.array([[1.0, 0.0], [0.0, 0.0]])

    def nn_bonds(self, bond: int):
        sx, sy, sz, s0, sn = self.sx, self.sy, self.sz, self.s0, self.sn
        hzL = hzR = 0.5 * self.Hzs[bond]
        hxL = hxR = 0.5 * self.Hxs[bond]
        if bond == 0:
            hzL = self.Hzs[bond]
            hxL = self.Hxs[bond]
        if bond == self.L - 2:
            hzR = self.Hzs[bond]
            hxR = self.Hxs[bond]

        H = (
            self.Jxs[bond] * np.kron(sx, sx)
            + self.Jys[bond] * np.kron(sy, sy)
            + self.Jzs[bond] * np.kron(sz, sz)
            + hxL * np.kron(sx, s0)
            + hxR * np.kron(s0, sx)
            + hzL * np.kron(sz, s0)
            + hzR * np.kron(s0, sz)
            + self.mus[bond] * np.kron(sn, s0)
            + self.mus[bond + 1] * np.kron(s0, sn)
        )
        return H


def init_H_bonds(self):
    """Initialize `H_bonds` hamiltonian.

    Called by __init__().
    """
    sx, sz, id = self.sigmax, self.sigmaz, self.id
    d = self.d
    nbonds = self.L - 1 if self.bc == "finite" else self.L
    H_list = []
    for i in range(nbonds):
        gL = gR = 0.5 * self.g
        if self.bc == "finite":
            if i == 0:
                gL = self.g
            if i + 1 == self.L - 1:
                gR = self.g
        H_bond = -self.J * np.kron(sx, sx) - gL * np.kron(sz, id) - gR * np.kron(id, sz)
        # H_bond has legs ``i, j, i*, j*``
        H_list.append(np.reshape(H_bond, [d, d, d, d]))
    self.H_bonds = H_list


class HBond(BondHamiltonian):
    def __init__(self, L, Jxs, Jys, Jzs, Hxs, Hzs, mus, d=2):
        super().__init__(L, Jxs, Jys, Jzs, Hxs, Hzs, mus, d)

    def h_bond(self, bond):

        iden = np.eye(4)

        H = -1j * (
            np.kron(self.nn_bonds(bond), iden)
            - np.kron(iden, np.transpose(self.nn_bonds(bond)))
        )
        return H

    def lindbladian(self, op, gamma=0.01):
        iden = np.eye(4)
        op_i = np.kron(op, self.s0)
        op_j = np.kron(self.s0, op)
        op_op_dag = op @ op.view(Operator).dag
        op_dag_op = op.view(Operator).dag @ op
        op_ij = op_i + op_j
        op_op_dag_i = np.kron(op_op_dag, self.s0)
        op_op_dag_j = np.kron(self.s0, op_op_dag)
        op_dag_op_i = np.kron(op_dag_op, self.s0)
        op_dag_op_j = np.kron(self.s0, op_dag_op)

        h_noise = (
            np.kron(op_ij, np.conj(op_ij.T))
            - 0.5 * np.kron(iden, (op_op_dag_i + op_op_dag_j))
            - 0.5 * np.kron((op_dag_op_i + op_dag_op_j), iden)
        )
        return gamma * h_noise

    def new_lindbladian(self, op, bond, gamma=0.01):

        cL = cR = 0.5
        if bond == 0:
            cL = 1.0
        if bond == self.L - 2:
            cR = 1.0
        iden = np.eye(4)
        op_L = np.kron(op, self.s0)
        op_R = np.kron(self.s0, op)
        op_list = [op_L.view(Operator), op_R.view(Operator)]

        lind = 0.0
        for i in range(2):
            if i == 0:
                c = cL
            else:
                c = cR
            lind += c * gamma * np.kron(op_list[i], np.conjugate(op_list[i]))
            lind += -c * 0.5 * gamma * np.kron((op_list[i].dag @ op_list[i]), iden)
            lind += (
                -c
                * 0.5
                * gamma
                * np.kron(iden, (np.conjugate(op_list[i]) @ op_list[i].T))
            )

        return lind


def H_bond_choi(H_bond: BondHamiltonian, site, gamma=0.0, d=2):
    """Build bond hamiltonian.

    Args:
        H_bond (BondHamiltonian): _description_
        site (int): _description_
        gamma (float, optional): _description_. Defaults to 0.1.
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    iden = np.eye(d ** 2)
    sx = 2 * np.array([[0.0, 1 / 2.0], [1 / 2.0, 0.0]])
    H_lind = np.kron(sx, np.eye(2)) + np.kron(np.eye(2), sx)

    # h_ij = np.reshape(H_bond.nn_term(site), (d,d,d,d, 1,1,1,1))  # (p1, p2, p1', p2')
    # I_ij = np.reshape(np.eye(d**2), (d,d,d,d))   # (q1, q2, q1', q2')
    # H_1 = np.kron(h_ij, I_ij)
    # h_ij = np.reshape(H_bond.nn_term(site), (d,d,d,d))
    # I_ij = np.reshape(np.eye(d**2), (d,d,d,d,1,1,1,1))
    # H_2 = np.kron(I_ij, h_ij)
    # H = 1j*H_1 -1j*H_2
    H = (
        -1j * np.kron(H_bond.nn_term(site), iden)
        + 1j * np.kron(iden, np.conj(H_bond.nn_term(site)).T)
        + gamma * np.kron(H_lind, iden)
        + +gamma * np.kron(iden, H_lind)
        - gamma * np.eye(d ** 4)
    )
    return H


class TFIMPO(BondHamiltonian):
    def __init__(self, L, Jxs, Jys, Jzs, Hxs, Hzs, mus, d=4):
        super().__init__(L, Jxs, Jys, Jzs, Hxs, Hzs, mus, d)

    def get_Ws(self):
        W_list = []
        s0 = self.s0
        sx = self.sx
        sz = self.sz
        hx = self.Hxs
        Jz = self.Jzs
        for site in range(self.L):
            w = np.zeros((4, 4, self.d, self.d), dtype=float)
            w[0, 0] = w[3, 3] = np.kron(s0, s0)
            w[1, 0] = np.kron(sz, s0)
            w[2, 0] = np.kron(s0, sz.T)
            w[3, 0] = hx[site] * (np.kron(sx, s0) - np.kron(s0, sx.T))

            w[3, 1] = Jz[site] * np.kron(sz, s0)
            w[3, 2] = -Jz[site] * np.kron(s0, sz.T)
            W_list.append(w)
        return W_list

    def Ws(self):
        W_list = []
        s0 = self.s0
        sx = self.sx
        sz = self.sz
        hx = self.Hxs
        Jz = self.Jzs
        for site in range(self.L):
            w = np.zeros((self.d * self.d, self.d * self.d), dtype=float)
            w[0, 0] = w[3, 3] = np.kron(s0, s0)
            w[1, 0] = np.kron(sz, s0)
            w[2, 0] = np.kron(s0, sz.T)
            w[3, 0] = hx[site] * (np.kron(sx, s0) - np.kron(s0, sx.T))

            w[3, 1] = Jz[site] * np.kron(sz, s0)
            w[3, 2] = -Jz[site] * np.kron(s0, sz.T)
            W_list.append(w)
        return W_list


import numpy as np


class TFIModel:
    """Simple class generating the Hamiltonian of the transverse-field Ising model.

    The Hamiltonian reads
    .. math ::
        H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i

    Parameters
    ----------
    L : int
        Number of sites.
    J, g : float
        Coupling parameters of the above defined Hamiltonian.
    bc : 'infinite', 'finite'
        Boundary conditions.

    Attributes
    ----------
    L : int
        Number of sites.
    bc : 'infinite', 'finite'
        Boundary conditions.
    sigmax, sigmay, sigmaz, id :
        Local operators, namely the Pauli matrices and identity.
    H_bonds : list of np.Array[ndim=4]
        The Hamiltonian written in terms of local 2-site operators, ``H = sum_i H_bonds[i]``.
        Each ``H_bonds[i]`` has (physical) legs (i out, (i+1) out, i in, (i+1) in),
        in short ``i j i* j*``.
    H_mpo : lit of np.Array[ndim=4]
        The Hamiltonian written as an MPO.
        Each ``H_mpo[i]`` has legs (virtual left, virtual right, physical out, physical in),
        in short ``wL wR i i*``.
    """

    def __init__(self, L, J, g, bc="finite"):
        assert bc in ["finite", "infinite"]
        self.L, self.d, self.bc = L, 2, bc
        self.J, self.g = J, g
        self.sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.sigmay = np.array([[0.0, -1j], [1j, 0.0]])
        self.sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        self.id = np.eye(2)
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian.

        Called by __init__().
        """
        sx, sz, id = self.sigmax, self.sigmaz, self.id
        d = self.d
        nbonds = self.L - 1 if self.bc == "finite" else self.L
        H_list = []
        for i in range(nbonds):
            gL = gR = 0.5 * self.g
            if self.bc == "finite":
                if i == 0:
                    gL = self.g
                if i + 1 == self.L - 1:
                    gR = self.g
            H_bond = (
                -self.J * np.kron(sx, sx) - gL * np.kron(sz, id) - gR * np.kron(id, sz)
            )
            # H_bond has legs ``i, j, i*, j*``
            H_list.append(np.reshape(H_bond, [d, d, d, d]))
        self.H_bonds = H_list

    # (note: not required for TEBD)
    def init_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian.

        Called by __init__().
        """
        w_list = []
        for i in range(self.L):
            w = np.zeros((3, 3, self.d, self.d), dtype=float)
            w[0, 0] = w[2, 2] = self.id
            w[0, 1] = self.sigmax
            w[0, 2] = -self.g * self.sigmaz
            w[1, 2] = -self.J * self.sigmax
            w_list.append(w)
        self.H_mpo = w_list
