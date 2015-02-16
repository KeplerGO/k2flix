#!/usr/bin/env python
from setuptools import setup


# PyPi requires reStructuredText instead of Markdown,
# so we convert our Markdown README for the long description
try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = open('README.md').read()

# Command-line tools
entry_points = {'console_scripts': [
    'k2flix = k2flix.core:k2flix_main'
]}

setup(name='k2flix',
      version='1.0.0',
      description="Create beautiful quicklook movies "
                  "from the pixel data observed "
                  "by NASA's Kepler spacecraft.",
      long_description=long_description,
      author='Geert Barentsen',
      author_email='geert@barentsen.be',
      license='MIT',
      url='http://barentsen.github.io/k2flix',
      packages=['k2flix'],
      install_requires=['numpy',
                        'matplotlib',
                        'astropy>=0.4',
                        'imageio>=1'],
      entry_points=entry_points,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
