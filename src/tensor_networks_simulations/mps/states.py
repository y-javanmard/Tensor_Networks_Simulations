from matplotlib.pyplot import axes
import numpy as np


class MPS:
    def __init__(self, Bs=None, Ss=None, bonds=None, d=2):
        self.L = None if Bs is None else len(Bs)
        self.d = d
        self.Bs = Bs
        self.Ss = Ss
        self.bonds = bonds

    @classmethod
    def af_product_state(cls, L, d=2):
        """Create an antiferromagnetic product state."""
        B_list = []
        s_list = []
        bond_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=int)
            B[np.mod(i, 2), 0, 0] = 1.0
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            bond_vec.append(B_list[i].shape[1])
        s_list.append(s)
        bond_vec.append(1)
        return cls(B_list, s_list, bond_vec, d)

    @classmethod
    def GHZ_state(cls, L, d=2):
        """Create an GHZ product state."""
        Bs = []
        Ss = []
        bonds = []
        for i in range(L):
            B = np.zeros((d, 1, 1))
            B[0, 0, 0] = 1.0 / np.sqrt(2)
            B[1, 0, 0] = 1.0 / np.sqrt(2)
            s = np.zeros(1)
            s[0] = 1.0
            Bs.append(B)
            Ss.append(s)
            bonds.append(Bs[i].shape[1])
        Ss.append(s)
        bonds.append(1)
        return cls(Bs, Ss, bonds)

    ###############################################################
    @classmethod
    def initial_state_dw(cls, L, d=2, dtype=complex):
        "Create an antiferromagnetic product state."
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            if i < L / 2:
                B[0, 0, 0] = 1.0
            else:
                B[1, 0, 0] = 1.0
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)

    ################################################################

    @classmethod
    def initial_state_f(cls, L, d=2, dtype=complex):
        "Create an ferromagnetic product state."
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            B[np.mod(i, 1), 0, 0] = 1.0
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)

    @classmethod
    def initial_state_random(cls, L, d=2, dtype=complex):
        "Create an random product state."
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            B[np.randint(0, 1), 0, 0] = 1.0
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)

    @classmethod
    def initial_state_random_x_direction(cls, L, d=2, dtype=complex):
        "Create an random product state."
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            B[0, 0, 0] = 1.0 / np.sqrt(2.0)
            B[1, 0, 0] = np.random.choice([-1.0, 1.0]) / np.sqrt(2.0)
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)

    @classmethod
    def initial_state_x_f(cls, L, d=2, dtype=complex):
        "Create an ferromagnetic product state in the x-direction."
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            B[0, 0, 0] = 1.0 / np.sqrt(2.0)
            B[1, 0, 0] = 1.0 / np.sqrt(2.0)
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)

    @classmethod
    def initial_state_x_dw(cls, L, d=2, dtype=complex):
        "Create an ferromagnetic product state in the x-direction."
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            if i < L / 2:
                B[0, 0, 0] = 1.0 / np.sqrt(2.0)
                B[1, 0, 0] = 1.0 / np.sqrt(2.0)
            else:
                B[0, 0, 0] = 1.0 / np.sqrt(2.0)
                B[1, 0, 0] = -1.0 / np.sqrt(2.0)

            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)


class MPDO:
    """
    Create matrix product density operator (MPDO)
    """

    def __init__(self, Ms=None, Ss=None, bonds=None, d=2):
        self.L = None if Ms is None else len(Ms)
        self.d = d
        self.Ms = Ms
        self.Ss = Ss
        self.bonds = bonds

    @classmethod
    def af_product_state(cls, L, d=2):
        """Create an antiferromagnetc product state."""
        Ms = []
        Ss = []
        bonds = []
        for i in range(L):
            B = np.zeros((d, d, 1, 1), dtype=int)
            B[np.mod(i, 2), 0, 0, 0] = 1.0
            s = np.zeros(1)
            s[0] = 1.0
            Ms.append(B)
            Ss.append(s)
            bonds.append(Ms[i].shape[1])
        Ss.append(s)
        bonds.append(1)
        return cls(Ms, Ss, bonds, d)

    @classmethod
    def mix_state_mpdo(cls, L, d=2, rank_3=False):
        Ms = []
        Ss = []
        bonds = []
        for i in range(L):
            B = np.zeros((d, d, 1, 1))
            B[0, 0, 0, 0] = 1 / np.sqrt(2)
            B[1, 1, 0, 0] = 1 / np.sqrt(2)
            BB = np.tensordot(B, np.conj(B), axes=([1], [1]))
            BB = np.transpose(BB, (0, 3, 1, 2, 4, 5))
            if rank_3:
                BB = np.reshape(BB, (d ** 2, 1, 1))
            else:
                BB = np.reshape(BB, (d, d, 1, 1))
            s = np.zeros(1)
            s[0] = 1.0
            Ms.append(BB)
            Ss.append(s)
            bonds.append(Ms[i].shape[2])
        Ss.append(s)
        bonds.append(1)
        return cls(Ms, Ss, bonds, d)

    @classmethod
    def mix_state_mpo(cls, L, d=2):
        Ms = []
        Ss = []
        bonds = []
        for i in range(L):
            B = np.zeros((d, d, 1, 1))
            B[0, 0, 0, 0] = 1 / np.sqrt(2)
            B[1, 1, 0, 0] = 1 / np.sqrt(2)
            s = np.zeros(1)
            s[0] = 1.0
            Ms.append(B)
            Ss.append(s)
            bonds.append(Ms[i].shape[2])
        Ss.append(s)
        bonds.append(1)
        return cls(Ms, Ss, bonds, d)

    @classmethod
    def pure_state_mpo(cls, L, d=2):
        Ms = []
        Ss = []
        bonds = []
        for i in range(L):
            B = np.zeros((d, d, 1, 1))
            B[0, 0, 0, 0] = 1
            B[1, 0, 0, 0] = 1
            s = np.zeros(1)
            s[0] = 1.0
            Ms.append(B)
            Ss.append(s)
            bonds.append(Ms[i].shape[2])
        Ss.append(s)
        bonds.append(1)
        return cls(Ms, Ss, bonds, d)

    @staticmethod
    def update_mpo_with_schmidt_vals(mpo, d=2):
        """
        compute the schmidt values along the mpo chain.
        """

        Ss = []
        Ss.append(np.array([1.0]))
        for i in range(len(mpo.Ms) - 1):
            # print Blist[i].shape,Blist[i+1].shape
            theta = np.tensordot(
                mpo.Ms[i], mpo.Ms[i + 1], axes=(3, 2)
            )  # (p_i, q_i, a_i, p_i+1, q_i+1, a_i+2)
            theta_bar = theta
            a = d * d * mpo.bonds[i]
            b = d * d * mpo.Ms[i + 1].shape[3]
            # print(i, a, b, mpo.Ms[i + 1].shape[3], theta.shape, len(Ss[i]))
            theta = np.reshape(np.tensordot(np.diag(Ss[i]), theta, axes=(1, 2)), (a, b))
            X, Y, Z = np.linalg.svd(theta, compute_uv=True, full_matrices=True)
            mpo.Ms[i + 1] = np.transpose(
                np.reshape(
                    Z[: mpo.bonds[i + 1], :],
                    (
                        mpo.bonds[i + 1],
                        d,
                        d,
                        int(Z.shape[1] / (d * d)),
                    ),
                ),
                (1, 2, 0, 3),
            )
            tmp = np.linalg.norm(Y[: mpo.bonds[i + 1]])
            # print("here2",len(Y),mpo.bonds[i + 1] )
            Ss.append(Y[: mpo.bonds[i + 1]] / tmp)
            mpo.Ms[i] = (
                np.tensordot(
                    theta_bar,
                    np.conjugate(mpo.Ms[i + 1]).T,
                    axes=([3, 4, 5], [3, 2, 0]),
                )
                / tmp
            )
        Ss.append(np.array([1.0]))
        mpo.Ss = Ss
        return mpo

    @staticmethod
    def schmidt_vals_from_mps(Bs, chi_vec, d=2):
        Ss = []
        Ss.append(np.array([1.0]))
        LL = len(Bs)
        for i in range(LL - 1):
            # print Blist[i].shape,Blist[i+1].shape
            theta = np.tensordot(Bs[i], Bs[i + 1], axes=(2, 1))  # (i, a_l, j, a_l+2)
            theta_bar = theta

            a = d * chi_vec[i]
            b = d * Bs[i + 1].shape[2]

            # print i, theta.shape, a, b

            theta = np.reshape(np.tensordot(np.diag(Ss[i]), theta, axes=(1, 1)), (a, b))
            # print i, theta.shape, a, b

            X, Y, Z = np.linalg.svd(theta, compute_uv=True, full_matrices=True)

            # print X.shape, Y.shape, Z.shape

            Bs[i + 1] = np.transpose(
                np.reshape(
                    Z[: chi_vec[i + 1], :],
                    (chi_vec[i + 1], d, int(Z.shape[1] / d)),
                ),
                (1, 0, 2),
            )
            tmp = np.linalg.norm(Y[: chi_vec[i + 1]])
            Ss.append(Y[: chi_vec[i + 1]] / tmp)
            Bs[i] = (
                np.tensordot(
                    theta_bar, np.conjugate(Bs[i + 1]).T, axes=([2, 3], [2, 0])
                )
                / tmp
            )
        Ss.append(np.array([1.0]))
        return Bs, Ss

    @classmethod
    def from_mpo(cls, mpo, d=2):
        Ms = []
        bonds = []
        bonds.append(1)
        for i in range(len(mpo.Ms)):
            a_l = mpo.Ms[i].shape[2]
            a_r = mpo.Ms[i].shape[3]
            BB = np.tensordot(mpo.Ms[i], np.conj(mpo.Ms[i]), axes=([1], [1]))
            BB = np.transpose(BB, (0, 3, 1, 4, 2, 5))
            BB = np.reshape(BB, (d, d, a_l * a_l, a_r * a_r))
            Ms.append(BB)
            bonds.append(a_r * a_r)
        # print(bonds)
        rho = cls(Ms, mpo.Ss, bonds)
        # rho = cls.update_mpo_with_schmidt_vals(rho)
        return rho

    @classmethod
    def pure_state_mpdo(cls, L, d=2, rank_3=False):
        Ms = []
        Ss = []
        bonds = []
        for i in range(L):
            B = np.zeros((d, d, 1, 1))
            B[0, 0, 0, 0] = 1 / np.sqrt(2)
            B[1, 0, 0, 0] = 1 / np.sqrt(2)
            BB = np.tensordot(B, np.conj(B), axes=([1], [1]))
            BB = np.transpose(BB, (0, 3, 1, 2, 4, 5))
            if rank_3:
                BB = np.reshape(BB, (d ** 2, 1, 1))
            else:
                BB = np.reshape(BB, (d, d, 1, 1))
            s = np.zeros(1)
            s[0] = 1.0
            Ms.append(B)
            Ss.append(s)
            bonds.append(Ms[i].shape[2])
        Ss.append(s)
        bonds.append(1)
        return cls(Ms, Ss, bonds, d)

    def infinite_temp_state(self, L):
        """Create an singlet state product state.
        This function will produce the gamma list, Blist and schmidt values related to Gamma"""

        Gs = []
        Gp_f = np.zeros((self.d, 1, self.d))
        Gq_e = np.zeros((self.d, self.d, 1))

        Gp_f[0, 0, 0] = 1.0 / np.sqrt(1.0)
        Gp_f[0, 0, 1] = Gp_f[1, 0, 0] = 0.0
        Gp_f[1, 0, 1] = -1.0 / np.sqrt(1.0)
        Gq_e[0, 0, 0] = 0.0
        Gq_e[0, 1, 0] = 1.0 / np.sqrt(2.0)
        Gq_e[1, 0, 0] = 1.0 / np.sqrt(2.0)
        Gq_e[1, 1, 0] = 0.0
        for i in range(L):
            Gs.append(Gp_f)
            Gs.append(Gq_e)
        return Gs

    @classmethod
    def from_mps(cls, psi: MPS, d=2):
        Ms = []
        Ss = []
        bonds = []
        Ss.append(psi.Ss[0])
        for i in range(len(psi.Bs)):
            temp = np.kron(psi.Bs[i], np.conjugate(psi.Bs[i]))  # (p_i*p'_i, l*l', r*r')
            l = temp.shape[1]
            r = temp.shape[2]
            temp = np.reshape(temp, (d, d, l, r))
            Ms.append(temp)
            Ss.append(psi.Ss[i])
            bonds.append(1)
        bonds.append(1)
        return cls(Ms, Ss, bonds)

    @classmethod
    def from_mps_vidal(cls, psi: MPS, d=2):
        Mlist = []
        bonds = []
        for i in range(len(psi.Bs)):
            temp = np.kron(psi.Bs[i], np.conjugate(psi.Bs[i]))  # (p_i*p'_i, l*l', r*r')
            l = temp.shape[1]
            r = temp.shape[2]
            temp = np.reshape(temp, (d * d, l, r))
            Mlist.append(temp)
            bonds.append(temp.shape[-1])
        bonds.append(1)
        Ms, Ss = cls.schmidt_vals_from_mps(Mlist, bonds, d=d * d)
        rho = cls(Ms, Ss, bonds, d * d)

        return rho

    def mpo_from_purified_mps(self, Bs, Ls):
        """
        Create matrix product density operators (density matrix) from an MPS.
            p2
            |
        a1--M--a2
            |
            p1
        Remark: enumeration convention: 0: p1, 1: p2, 3: p3, 4: p4

        parameters:
        -----------

        Bs: Tensors for each site.
        Ls: Schmidt values at each bond.
        """
        Ms = []
        Ss = []
        Ss.append(Ls[0])
        for i in range(int(len(Bs) / 2)):
            temp = np.tensordot(Bs[2 * i], Bs[2 * i + 1], axes=(2, 1))
            temp = np.transpose(temp, (0, 2, 1, 3))
            Ms.append(temp)  # p1, p2, a1, a2
            Ss.append(Ls[2 * i + 2])
        return Ms, Ss

    @staticmethod
    def bond_vec(Bs):
        chi_vec = []
        chi_vec.append(Bs[0].shape[2])
        for i in range(len(Bs)):
            chi_vec.append(Bs[i].shape[3])
        return chi_vec

    def bond_vec_inf_temp(self, L):
        """
        Create maximum bond dimension : Chi max
        """
        vv = [1] * 2 * L
        for i in range(0, 2 * L - 1, 2):
            vv[i] = 1
            vv[i + 1] = 2
        return vv


def mpo_inf_temperature(L):
    Bs_inf = MPDO().infinite_temp_state(L)
    bonds_inf = MPDO().bond_vec_inf_temp(L)
    Bs_inf, Ss_inf = MPDO().schmidt_vals_from_mps(Bs_inf, bonds_inf)
    Ms_inf, Ss_inf = MPDO().mpo_from_purified_mps(Bs_inf, Ss_inf)
    bonds_inf = bonds_inf[::2]
    bonds_inf.append(1)
    mpo_inf = MPDO(Ms_inf, Ss_inf, bonds_inf)
    return mpo_inf
