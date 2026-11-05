import pytest
from qiskit_aer import AerSimulator


@pytest.fixture(scope="session")
def sim():
    return AerSimulator()


@pytest.fixture(scope="session")
def mps():
    return AerSimulator(method="matrix_product_state")
