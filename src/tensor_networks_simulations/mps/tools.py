import numpy as np
from numpy import ndarray

from tensor_networks_simulations.mps.models import Operator
from tensor_networks_simulations.mps.states import MPDO


def right_normalization_mps(M_list, chi_vec, d=2):
    """Checks if a tensor is right-normalized. For MPS only. It could be expanded for MPO.

    Args:
        M_list (_type_): _description_
        chi_vec (_type_): _description_
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    Blist = []
    L = len(M_list)
    # print(f"L={L}")
    # main loop for normalization
    for i in range(L):
        # index for list
        idx = L - 1 - i
        # shapes
        a = M_list[idx].shape[1]
        b = M_list[idx].shape[0]
        c = M_list[idx].shape[2]

        # reshape and transpose
        B_tmp = np.reshape(np.transpose(M_list[idx], (1, 0, 2)), (a, b * c))
        X, Y, Z = np.linalg.svd(B_tmp, full_matrices=True, compute_uv=True)
        # print(Z, chi_vec[idx], d, Z.shape[1]/d)
        # new B
        B_new = np.reshape(Z[: chi_vec[idx]], (chi_vec[idx], d, int(Z.shape[1] / d)))

        # add blist
        Blist.append(np.transpose(B_new, (1, 0, 2)))

        # for next B
        X = X[:, 0 : chi_vec[idx]]
        Y = Y[: chi_vec[idx]]
        # XY dot
        Umat = np.dot(X, np.diag(Y))

        # re multiply to M_list
        M_list[idx - 1] = np.tensordot(M_list[idx - 1], Umat, axes=(2, 0))

    return Blist[::-1]


def right_normalization_mpdo(rho, d=2):
    """Checks if a tensor is right-normalized. For MPS only. It could be expanded for MPO.

    Args:
        M_list (_type_): _description_
        chi_vec (_type_): _description_
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    Blist = []
    L = len(rho.Ms)
    # print(f"L={L}")
    # main loop for normalization
    for i in range(L):
        # index for list
        idx = L - 1 - i
        # shapes
        a = rho.Ms[idx].shape[2]
        b = rho.Ms[idx].shape[0]
        bb = rho.Ms[idx].shape[1]
        c = rho.Ms[idx].shape[3]

        # reshape and transpose
        B_tmp = np.reshape(np.transpose(rho.Ms[idx], (2, 0, 1, 3)), (a, b * bb * c))
        X, Y, Z = np.linalg.svd(B_tmp, full_matrices=True, compute_uv=True)
        # print(Z, chi_vec[idx], d, Z.shape[1]/d)
        # new B
        # print(rho.Ms[idx].shape, Z.shape,rho.bonds[idx] , "here",)
        new_ind = int(Z.shape[1] / (d * d))
        B_new = np.reshape(Z[: rho.bonds[idx]], (rho.bonds[idx], d, d, new_ind))
        # print(B_new.shape)
        # add blist
        Blist.append(np.transpose(B_new, (1, 2, 0, 3)))

        # for next B
        X = X[:, 0 : rho.bonds[idx]]
        Y = Y[: rho.bonds[idx]]
        # XY dot
        Umat = np.dot(X, np.diag(Y))

        # re multiply to M_list
        rho.Ms[idx - 1] = np.tensordot(rho.Ms[idx - 1], Umat, axes=(3, 0))
        normalized_rho = MPDO(Blist[::-1], rho.Ss, rho.bonds)

    return Blist[::-1]


def check_right_normalization(Bs, dd=0):
    """Checks the normaliztaion of the matrix.

    Args:
        Bs (_type_): _description_
        dd (int, optional): _description_. Defaults to 0.
    """
    L = len(Bs)
    for i in range(L):
        arr = np.diag(np.tensordot(Bs[i], np.conj(Bs[i].T), ([0, 2], [2, 0])), dd).real
        print(np.diag(np.tensordot(Bs[i], np.conj(Bs[i].T), ([0, 2], [2, 0])), dd).real)


def correlation_one_site_mix(Blist, llist, sllist, sBlist, Op, j):
    # print np.diag(llist[j]).shape
    lB = np.tensordot(Op, sBlist[j], axes=([1], [0]))
    lB = np.tensordot(np.diag(sllist[j]), lB, axes=([1], [2]))
    lB = np.transpose(lB, (1, 2, 0, 3))
    B = np.tensordot(np.diag(llist[j]), Blist[j], axes=([1], [2]))
    lB = np.tensordot(lB, np.conj(B), axes=([0, 1, 2, 3], [1, 2, 0, 3]))
    return np.squeeze(lB)


def T_spin_correl_mix(blist, o, Sblist, L, j, d=2):
    i = 0
    # sz=np.array([[1., 0.],[0., -1.]])
    # sz =  np.identity(2)
    T1 = np.tensordot(o, Sblist[j], ([1], [0]))  # (p_j, q_j, a_j, a_j+1)
    T2 = np.tensordot(
        T1, np.conj(blist[j].T), ([0, 1], [3, 2])
    )  # (a_i, a_i+1, b_i+1, b_i)
    if i == j:
        C = T2
        i = i + 1
    else:
        BB = np.tensordot(
            Sblist[i], np.conj(blist[i].T), ([0, 1], [3, 2])
        )  # (a_i+1, a_i+2, b_i+2 , b_i+1)
        C = BB
        i = i + 1
    while i < L:
        if i == j:
            BB = T2
            C = np.tensordot(C, BB, ([1, 2], [0, 3]))  # (a_i, b_i, a_i+2, b_i+2)
            C = np.transpose(C, (0, 2, 3, 1))  # (a_i, a_i+2, b_i, b_i+2)
            i = i + 1
        elif i != j:
            BB = np.tensordot(
                Sblist[i], np.conj(blist[i].T), ([0, 1], [3, 2])
            )  # (a_i+1, a_i+2, b_i+1 , b_i+2)
            C = np.tensordot(C, BB, ([1, 2], [0, 3]))  # (a_i, b_i, a_i+2, b_i+2)
            C = np.transpose(C, (0, 2, 3, 1))  # (a_i, a_i+2, b_i, b_i+2)

            i = i + 1
    return np.squeeze(C)


def expectation_value(rho, op, j):
    i = 0
    T1 = np.tensordot(op, rho.Ms[j], ([1], [0]))  # (p_j, q_j, a_j, a_j+1)
    T2 = np.tensordot(
        T1, np.conj(rho.Ms[j].T), ([0, 1], [3, 2])
    )  # (a_i, a_i+1, b_i+1, b_i)
    if i == j:
        C = T2
        i = i + 1
    else:
        BB = np.tensordot(
            Sblist[i], np.conj(blist[i].T), ([0, 1], [3, 2])
        )  # (a_i+1, a_i+2, b_i+2 , b_i+1)
        C = BB
        i = i + 1
    while i < L:
        if i == j:
            BB = T2
            C = np.tensordot(C, BB, ([1, 2], [0, 3]))  # (a_i, b_i, a_i+2, b_i+2)
            C = np.transpose(C, (0, 2, 3, 1))  # (a_i, a_i+2, b_i, b_i+2)
            i = i + 1
        elif i != j:
            BB = np.tensordot(
                Sblist[i], np.conj(blist[i].T), ([0, 1], [3, 2])
            )  # (a_i+1, a_i+2, b_i+1 , b_i+2)
            C = np.tensordot(C, BB, ([1, 2], [0, 3]))  # (a_i, b_i, a_i+2, b_i+2)
            C = np.transpose(C, (0, 2, 3, 1))  # (a_i, a_i+2, b_i, b_i+2)

            i = i + 1
    return np.squeeze(C)


def projection_Normalization_Mix(Blist, SBlist, L, d=2):
    B_dag = np.conj(SBlist[0])  # ( p_i, q_i, b_i, b_i+1 )
    BB = np.tensordot(
        Blist[0], B_dag, axes=([0, 1], [0, 1])
    )  # (a_i, a_i+1, b_i, b_i+1)
    # print BB.shape, Blist[0].shape, SBlist[0].shape
    i = 1
    while i < L:
        B_dag = np.conj(SBlist[i])  # ( p_i+1, q_i+1, b_i+1, b_i+2)
        BB = np.tensordot(
            BB, B_dag, axes=([3], [2])
        )  # (a_i, a_i+1, b_i, p_i+1, q_i+1, b_i+2)
        # print BB.shape
        BB = np.tensordot(
            BB, Blist[i], axes=([1, 3, 4], [2, 0, 1])
        )  # (a_i, b_i, b_i+2, a_i+2)
        # print BB.shape
        BB = np.transpose(BB, (0, 3, 1, 2))
        # print BB.shape, i+1
        i = i + 1
    return np.squeeze(BB)


def apply_one_site_op_mix_state(rho, O, i, d=2):
    Mlist = []
    for j in range(len(rho.Ms)):
        if j != i:
            Mlist.append(rho.Ms[j])
        else:
            Mlist.append(np.tensordot(O, rho.Ms[i], (1, 0)))
    rho_new = MPDO(Mlist, rho.Ss, rho.bonds)
    return rho_new


def apply_mixed_gate_mpdo(rho: MPDO, Gs):
    """Apply a 4 inices gate on state rho.

    Args:
        rho (MPDO): _description_
        G (_type_): 4 indices gate. In fact in an MPS, this is a sigle site operator.

    Returns:
        _type_: _description_
    """
    for i in range(len(rho.Ms)):
        rho.Ms[i] = np.tensordot(Gs[i], rho.Ms[i], axes=([2, 3], [0, 1]))
    return rho


def apply_gate_physical_indices(psi: MPDO, Gs):
    return NotImplementedError


def expectation_value_op(rho: MPDO, op: ndarray, j: int, d=2):
    """Compute expectation value of an operator through an MPDO.

    Args:
        rho (MPDO): _description_
        op (ndarray): _description_
        j (int): _description_
        d (int, optional): _description_. Defaults to 2.

    Returns:
        float: _description_
    """
    L = len(rho.Ms)
    i = 0
    T = np.tensordot(op, rho.Ms[j], ([0, 1], [0, 1]))  # (a_j, a_j+1)
    if i == j:
        C = T
        i = i + 1
    else:
        C = np.tensordot(np.eye(d), rho.Ms[i], ([0, 1], [0, 1]))  # (a_i+1, a_i+1)
        i = i + 1
    while i < L:
        if i == j:
            BB = T
            C = np.tensordot(C, BB, ([1], [0]))  # (a_i, a_i+2)
            i = i + 1
        elif i != j:
            BB = np.tensordot(np.eye(d), rho.Ms[i], ([0, 1], [0, 1]))  # (a_i+1, a_i+1)
            C = np.tensordot(C, BB, ([1], [0]))  # (a_i, a_i+2)
            i = i + 1
    return np.squeeze(C)


def apply_one_site_op_mpo(mpo: MPDO, op, site, d=2):
    Mlist = []
    for j in range(len(mpo.Ms)):
        if j != site:
            Mlist.append(mpo.Ms[j])
        else:
            Mlist.append(np.tensordot(op, mpo.Ms[site], (1, 0)))
    mpo = MPDO(Mlist, mpo.Ss, mpo.bonds)
    return mpo
