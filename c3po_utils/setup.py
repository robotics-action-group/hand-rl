"""Installation script for the 'c3po' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# Installation operation
setup(
    name="c3po_utils",
    packages=["c3po_utils"],
    # If you have single-file modules in the root, list them here, e.g.:
    # py_modules=["cli_args", "load_yaml"],
    # author=EXTENSION_TOML_DATA["package"]["author"],
    # maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    # url=EXTENSION_TOML_DATA["package"]["repository"],
    # version=EXTENSION_TOML_DATA["package"]["version"],
    # description=EXTENSION_TOML_DATA["package"]["description"],
    # keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
