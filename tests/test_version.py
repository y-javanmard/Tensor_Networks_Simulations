"""Test file to check that version for package is available."""
import tensor_networks_simulations


def test_get_version():
    """Check that the version variable is accessible."""
    project_version = tensor_networks_simulations.__version__
    assert project_version is not None and project_version != ""
