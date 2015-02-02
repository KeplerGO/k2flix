#!/usr/bin/env python
from setuptools import setup

entry_points = {'console_scripts': [
    'k2flix = k2flix.core:k2flix_main'
]}

setup(name='k2flix',
      version='1.0',
      description="Creates movies or animated gifs "
                  "from the pixel data obtained "
                  "by NASA's Kepler/K2 spacecraft",
      author='Geert Barentsen',
      license='MIT',
      url='http://barentsen.github.io/k2flix',
      packages=['k2flix'],
      install_requires=['numpy',
                        'matplotlib',
                        'astropy',
                        'imageio',
                        'progressbar'],
      entry_points=entry_points,
      )
