"""Setup script."""

from setuptools import setup

VERSION_TEMPLATE = '''
"""Automatically generated."""
__version__ = {version!r}
'''.lstrip()

setup(
    use_scm_version={
        "write_to": "src/tensor_networks_simulations/version.py",
        "write_to_template": VERSION_TEMPLATE,
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    }
)
