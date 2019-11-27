#! /usr/bin/env python
#

# Copyright (C) 2015 Jean-Remi King
# <jeanremi.king@gmail.com>
#
# Adapted from MNE-Python

import os
import os.path as op
import setuptools  # noqa
from numpy.distutils.core import setup

DISTNAME = 'torch_ridge'
DESCRIPTION = 'Adapt sklearn RidgeCV with torch, pretraining & multiple alphas'
MAINTAINER = 'Jean-Remi King'
MAINTAINER_EMAIL = 'jeanremi.king@gmail.com'
URL = 'https://github.com/kingjr/torch_ridge/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/kingjr/torch_ridge/'
VERSION = 0.1


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=False,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          zip_safe=False,  # the package can run out of an .egg file
          platforms='any',
          packages=package_tree('torch_ridge'),
          package_data={},
          scripts=[])
