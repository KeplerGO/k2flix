K2flix: Kepler/K2/TESS pixel data visualizer 
============================================
**Create quicklook movies from the pixels observed by NASA's Kepler/K2/TESS spacecraft.**

.. image:: http://img.shields.io/pypi/v/k2flix.svg
    :target: https://pypi.python.org/pypi/k2flix/
    :alt: PyPI

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
obtained by NASA's `Kepler space telescope <http://keplerscience.nasa.gov>`_,
or simulated data from the future `TESS space telescope <https://tess.gsfc.nasa.gov>`_.

The need for this tool arised from the fact that the two-wheeled extended Kepler mission, K2,
is affected by new sources of noise -- including pointing jitter and foreground asteroids --
which are more easily spotted by eye than by algorithm.

This tool takes Kepler's *Target Pixel Files (TPF)* as input
and turns them into contrast-stretched animated gifs or MPEG-4 movies.
These *TPF files* are publically available from the 
`Kepler archive <https://archive.stsci.edu/missions/kepler/target_pixel_files/>`_
and the `K2 archive <https://archive.stsci.edu/missions/k2/target_pixel_files/>`_. 

K2flix can be used both as a command-line tool or using its Python API.

Example
-------
Asteroids commonly pass in front of Kepler/K2 targets. 
We can use k2flix to create a two-day animation of pixel data to count the number of asteroids whizzing past::

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
    usage: k2flix [-h] [-v] [--output FILENAME] [--start START] [--stop STOP]
                  [--step STEP] [--fps FPS] [--binning BINNING] [--dpi DPI]
                  [--stretch STRETCH] [--min_cut MIN_CUT] [--max_cut MAX_CUT]
                  [--min_percent %] [--max_percent %] [--cmap CMAP] [--flags]
                  [--raw | --background | --cosmic]
                  [--ut | --jd | --mjd | --bkjd | --cadence]
                  tpf_filename [tpf_filename ...]

    Converts a Target Pixel File (TPF) from NASA's Kepler/K2/TESS spacecraft into
    an animated gif or MPEG-4 movie for human inspection.

    positional arguments:
      tpf_filename       path to one or more Target Pixel Files (TPF)

    optional arguments:
      -h, --help         show this help message and exit
      -v, --verbose
      --output FILENAME  .gif or .mp4 output filename (default: gif with the same
                         name as the input file)
      --start START      first frame to show. Give the frame number (default 0),
                         or a Julian Day if --jd/--mjd is set, or a cadence number
                         if --cadence is set.
      --stop STOP        final frame to show. Give the frame number (default: -1),
                         or a Julian Day if --jd/--mjd is set, or a cadence number
                         if --cadence is set.
      --step STEP        spacing between frames (default: show 100 frames)
      --fps FPS          frames per second (default: 15)
      --binning BINNING  number of cadence to co-add per frame (default: 1)
      --dpi DPI          resolution of the output in dots per K2 pixel (default:
                         choose a dpi that produces a 440px-wide image)
      --stretch STRETCH  type of contrast stretching: "linear", "sqrt", "power",
                         "log", or "asinh" (default is "log")
      --min_cut MIN_CUT  minimum cut level (default: use min_percent)
      --max_cut MAX_CUT  maximum cut level (default: use max_percent)
      --min_percent %    minimum cut percentile (default: 1.0)
      --max_percent %    maximum cut percentile (default: 95)
      --cmap CMAP        matplotlib color map name (default: gray)
      --flags            show the quality flags
      --raw              show the uncalibrated pixel counts ('RAW_CNTS')
      --background       show the background flux ('FLUX_BKG')
      --cosmic           show the cosmic rays ('COSMIC_RAYS')
      --ut               use Universal Time
      --jd               use Julian Day for annotation and --start/--stop
      --mjd              use Modified Julian Day for annotation and --start/--stop
      --bkjd             use Kepler Julian Day for annotation and --start/--stop
      --cadence          use Cadence Number for annotation and --start/--stop

Citing
------
This tool was created by Geert Barentsen at NASA's Kepler/K2 Guest Observer Office.
If this tool aided your research, please include a citation.
The code has been registered in the Astrophysics Source Code Library [`ascl:1503.001 <http://ascl.net/code/v/1069>`_] and the preferred BibTeX entry is::
  
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
Copyright 2016 Geert Barentsen.
K2flix is free software made available under the MIT License.
For details see the LICENSE file.
