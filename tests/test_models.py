import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from tensor_networks_simulations.mps.models import BondHamiltonian, HBond, H_bond_choi


@pytest.fixture
def sample_system():
    L = 4
    Jxs = np.ones(L-1)
    Jys = np.ones(L-1) * 0.5
    Jzs = np.ones(L-1) * 1.0
    Hxs = np.ones(L-1) * 0.1
    Hzs = np.ones(L-1) * 0.2
    mus = np.ones(L) * 0.05
    return L, Jxs, Jys, Jzs, Hxs, Hzs, mus

def test_BondHamiltonian_nn_bonds(sample_system):
    L, Jxs, Jys, Jzs, Hxs, Hzs, mus = sample_system
    bh = BondHamiltonian(L, Jxs, Jys, Jzs, Hxs, Hzs, mus)

    # Test bond 0 (left boundary)
    H0 = bh.nn_bonds(0)
    assert H0.shape == (4, 4)
    assert np.allclose(H0, H0.conj().T)  # Should be Hermitian

    # Test middle bond
    H1 = bh.nn_bonds(1)
    assert H1.shape == (4, 4)
    assert np.allclose(H1, H1.conj().T)

    # Test last bond (L-2)
    H_last = bh.nn_bonds(L-2)
    assert H_last.shape == (4, 4)
    assert np.allclose(H_last, H_last.conj().T)

def test_HBond_h_bond(sample_system):
    L, Jxs, Jys, Jzs, Hxs, Hzs, mus = sample_system
    hbond = HBond(L, Jxs, Jys, Jzs, Hxs, Hzs, mus)

    for bond in range(L-1):
        H = hbond.h_bond(bond)
        assert H.shape == (16, 16)
        # Should be anti-Hermitian up to numerical error: H + H^\dagger ≈ 0
        assert_allclose(H + H.conj().T, 0, atol=1e-14)

def test_HBond_lindbladian(sample_system):
    L, Jxs, Jys, Jzs, Hxs, Hzs, mus = sample_system
    hbond = HBond(L, Jxs, Jys, Jzs, Hxs, Hzs, mus)

    op = hbond.sp  # raising operator
    for bond in range(L-1):
        L_op = hbond.lindbladian(op, gamma=0.1)
        assert L_op.shape == (16, 16)
        # Lindbladian should be Hermitian
        assert_allclose(L_op, L_op.conj().T, atol=1e-14)

        # new_lindbladian should match lindbladian (up to boundary factors)
        L_new = hbond.new_lindbladian(op, bond, gamma=0.1)
        assert L_new.shape == (16, 16)
        assert_allclose(L_new, L_new.conj().T, atol=1e-14)

def test_HBond_new_lindbladian_boundary(sample_system):
    L, Jxs, Jys, Jzs, Hxs, Hzs, mus = sample_system
    hbond = HBond(L, Jxs, Jys, Jzs, Hxs, Hzs, mus)
    op = hbond.sp

    # Bond 0: cL = 1.0, cR = 0.5
    L0 = hbond.new_lindbladian(op, 0, gamma=1.0)
    assert L0.shape == (16, 16)

    # Bond L-2: cL = 0.5, cR = 1.0
    Llast = hbond.new_lindbladian(op, L-2, gamma=1.0)
    assert Llast.shape == (16, 16)

def test_H_bond_choi(sample_system):
    L, Jxs, Jys, Jzs, Hxs, Hzs, mus = sample_system
    bh = BondHamiltonian(L, Jxs, Jys, Jzs, Hxs, Hzs, mus)

    for site in range(L-1):
        Hc = H_bond_choi(bh, site, gamma=0.01)
        assert Hc.shape == (16, 16)

        # Choi matrix for CPTP map should have trace <= 1 on partial trace over output
        # But for small gamma, check structure
        iden4 = np.eye(4)
        partial_trace = np partial_trace_over_input(Hc.reshape(4,4,4,4), [2, 2])  # needs helper
        # Instead: check anti-Hermitian + dissipative part
        H_nn = bh.nn_bonds(site)
        expected_antiherm = -1j * np.kron(H_nn, iden4) + 1j * np.kron(iden4, H_nn.conj().T)
        assert_allclose(Hc[:16, :16].real, expected_antiherm.real, atol=1e-14)
        assert_allclose(Hc[:16, :16].imag, expected_antiherm.imag + 0.01 * np.kron(np.eye(4), np.eye(4)), atol=1e-14)

def test_Operator_dag():
    from your_module import Operator
    A = np.random.rand(3, 4) + 1j * np.random.rand(3, 4)
    A_op = A.view(Operator)
    assert_allclose(A_op.dag, A.conj().T)
