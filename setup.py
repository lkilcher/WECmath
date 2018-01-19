#!/usr/bin/env python

from distutils.core import setup

setup(name='WECmath',
      version='0.1',
      description='Wave Energy Converter (WEC) math toolbox',
      author='Levi Kilcher',
      author_email='levi.kilcher@nrel.gov',
      packages=['WECmath'],
      package_data={'WECmath': ['data/*']}
      )
