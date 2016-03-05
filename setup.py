#!/usr/bin/env python
import os
import sys
from setuptools import setup


if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()


# PyPi requires reStructuredText instead of Markdown,
# so we convert our Markdown README for the long description
try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = open('README.md').read()

# Load the __version__ variable without importing the package already
exec(open('k2flix/version.py').read())

# Command-line tools
entry_points = {'console_scripts': [
    'k2flix = k2flix.core:k2flix_main'
]}

setup(name='k2flix',
      version=__version__,
      description="Create beautiful quicklook movies "
                  "from the pixel data observed "
                  "by NASA's Kepler spacecraft.",
      long_description=long_description,
      author='Geert Barentsen',
      author_email='hello@geert.io',
      license='MIT',
      url='http://barentsen.github.io/k2flix',
      packages=['k2flix'],
      install_requires=['numpy',
                        'matplotlib',
                        'astropy>=0.4',
                        'imageio>=1',
                        'tqdm'],
      entry_points=entry_points,
      keywords="kepler k2 astrophysics",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
