# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

from setuptools import setup, find_packages
from pathlib import Path

CURRENT_DIR = Path(__file__).absolute().parent
readme = (CURRENT_DIR / 'README.md').read_text(encoding='utf-8')

setup(
    name='fostool',
    version="0.0.2",
    author="Microsoft",
    author_email="fostool@microsoft.com",
    description='FOST Python Package',
    long_description=readme,
    long_description_content_type="text/markdown",
    # Automatically install 'install_requires' packages from pypi when installing our package
    install_requires=[
        'scikit-learn',
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "addict",
    ],
    # 'extras_require packages' are not normally used, but only when the module is used in depth, and need to be installed manually
    extras_require={},
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    license='The MIT License (Microsoft)',
    url='https://github.com/microsoft/FOST',
    calssifiers=[
        # refer link:https://pypi.org/pypi?%3Aaction=list_classifiers
        # development status:3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researcher/developer',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9'
    ]
)