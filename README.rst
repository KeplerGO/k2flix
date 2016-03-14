K2flix: Kepler pixel data visualizer 
====================================
***Create beautiful quicklook movies from the pixel data observed by NASA's Kepler spacecraft.***

.. image:: http://img.shields.io/pypi/v/k2flix.svg
    :target: https://pypi.python.org/pypi/k2flix/
    :alt: PyPI

.. image:: http://img.shields.io/pypi/dm/k2flix.svg
    :target: https://pypi.python.org/pypi/k2flix/
    :alt: PyPI Downloads

.. image:: http://img.shields.io/travis/barentsen/k2flix/master.svg
    :target: http://travis-ci.org/barentsen/k2flix
    :alt: Travis status

.. image:: http://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/barentsen/k2flix/blob/master/LICENSE
    :alt: MIT license

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.15576.svg
    :target: http://dx.doi.org/10.5281/zenodo.15576
    :alt: DOI

.. image:: https://img.shields.io/badge/NASA%20ADS-2015ascl.soft03001B-blue.svg
    :target: http://adsabs.harvard.edu/abs/2015ascl.soft03001B
    :alt: ADS Bibcode

K2flix makes it easy to inspect the CCD pixel data
obtained by `NASA's Kepler space telescope <http://keplerscience.nasa.gov>`_.
The need for this tool arises from the fact that the two-wheeled extended Kepler mission, K2,
is affected by new sources of noise -- including pointing jitter and foreground asteroids --
which are more easy to spot by eye than by algorithm.
The code takes Kepler's *Target Pixel Files (TPF)* as input
and turns them into contrast-stretched animated gifs or MPEG-4 movies.
These *TPF files* are publically available from the 
`Kepler archive <https://archive.stsci.edu/missions/kepler/target_pixel_files/>`_
and the `K2 archive <https://archive.stsci.edu/missions/k2/target_pixel_files/>`_. 
K2flix can be used both as a command-line tool or using its Python API.

Example
-------
Asteroids commonly pass in front of Kepler/K2 targets. 
How many can you spot in this example 2-day animation? ::

    $ k2flix --start 545 --stop 680 --step 1 --fps 12 http://archive.stsci.edu\
    /missions/k2/target_pixel_files/c1/201500000/72000/ktwo201572338-c01_lpd-targ.fits.gz

.. image:: https://raw.githubusercontent.com/barentsen/k2flix/master/examples/epic-201572338.gif
    :alt: k2flix output example

To see many more examples, follow `@KeplerBot <https://twitter.com/KeplerBot>`_ on Twitter!

Installation
------------
If you have a working installation of Python on your system, you can install k2flix using pip::

  $ pip install k2flix

Alternatively, you can get the latest version by installing from source::

  $ git clone https://github.com/barentsen/k2flix.git
  $ cd k2flix
  $ python setup.py install

K2flix has been tested under Linux.  Get in touch if you encounter issues on OS X or Windows.

Using k2flix
------------
After installation, the `k2flix` tool will be available on the command line. You can then use it as follows.

Converting a Kepler pixel file to an animated gif::

  $ k2flix tpf-file.fits.gz

Converting a Kepler pixel file to an MPEG-4 movie::

  $ k2flix -o movie.mp4 tpf-file.fits.gz

K2flix supports reading from web URLs, so you can generate a movie directly from the data archive::
  
  $ k2flix https://archive.stsci.edu/missions/k2/target_pixel_files/c1/201400000/00000/ktwo201400022-c01_lpd-targ.fits.gz


To see all the options, use the `--help` argument to see the full usage information::

  $ k2flix --help
  usage: k2flix [-h] [-o filename] [--start IDX] [--stop IDX] [--step STEP]
              [--fps FPS] [--dpi DPI] [--min_percent MIN_PERCENT]
              [--max_percent MAX_PERCENT] [--cmap colormap_name] [--raw]
              [--flags] [--jd | --mjd | --bjd | --bkjd | --cadence]
              filename [filename ...]

  Converts a Target Pixel File (TPF) from NASA's Kepler/K2 spacecraft into a
  movie or animated gif.

  positional arguments:
    filename              path to one or more Kepler Target Pixel Files (TPF)

  optional arguments:
    -h, --help            show this help message and exit
    -o filename, --output filename
                          output filename (default: gif with the same name as
                          the input file)
    --start IDX           first frame to show (default: 0)
    --stop IDX            last frame to show (default: -1)
    --step STEP           spacing between frames (default: output 100 frames)
    --fps FPS             frames per second (default: 15)
    --dpi DPI             resolution of the output in dots per K2 pixel
    --min_percent MIN_PERCENT
                          percentile value used to determine the minimum cut
                          level (default: 1.0)
    --max_percent MAX_PERCENT
                          percentile value used to determine the maximum cut
                          level (default: 95.0)
    --cmap colormap_name  matplotlib color map name (default: gray)
    --raw                 show the uncalibrated pixel data ('RAW_CNTS')
    --flags               show quality flags
    --jd                  show the Julian Day
    --mjd                 show the Modified Julian Day
    --bjd                 show the Barycentric Julian Day
    --bkjd                show the Bareycentric Kepler Julian Day
    --cadence             show the cadence number

Citing
------
If you use this tool in an academic publication, please include a citation.
The code has been registered in the Astrophysics Source Code Library `[ascl:1503.001] <http://ascl.net/code/v/1069>`_ and the preferred BibTeX entry is::
  
  @MISC{2015ascl.soft03001B,
    author        = {{Barentsen}, G.},
    title         = "{K2flix: Kepler pixel data visualizer}",
    howpublished  = {Astrophysics Source Code Library},
    year          = 2015,
    month         = mar,
    archivePrefix = "ascl",
    eprint        = {1503.001},
    adsurl        = {http://adsabs.harvard.edu/abs/2015ascl.soft03001B},
    adsnote       = {Provided by the SAO/NASA Astrophysics Data System},
    doi           = {10.5281/zenodo.15576},
    url           = {http://dx.doi.org/10.5281/zenodo.15576}
  }

Contributing
------------
To report bugs and request features, please use the `issue tracker <https://github.com/barentsen/k2flix/issues>`_. Code contributions are very welcome.

License
-------
Copyright 2016 Geert Barentsen. K2flix is free software made available under the MIT License. For details see the LICENSE file.