# K2flix [![build status](http://img.shields.io/travis/barentsen/k2flix/master.svg?style=flat)](http://travis-ci.org/barentsen/k2flix) [![MIT license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/barentsen/k2flix/blob/master/LICENSE) 

***Create beautiful quicklook movies from the pixel data observed by NASA's Kepler spacecraft.***

It is good practice to inspect scientific measurements *by eye*
before applying sophisticated data analysis algorithms.
K2flix makes it easy to convert the CCD pixel data
obtained by [NASA's Kepler space telescope](http://kepler.nasa.gov)
into beautiful movies for human inspection.
K2flix takes Kepler's *Target Pixel Files (TPF)* as input
and turns them into contrast-stretched animated gifs or MPEG-4 movies.
These *TPF files* are publically available from the 
[Kepler archive](https://archive.stsci.edu/missions/kepler/target_pixel_files/)
and the [K2 archive](https://archive.stsci.edu/missions/k2/target_pixel_files/).

### Example
Asteroids commonly pass in front of Kepler/K2 targets.  How many can you spot in this example 2-day animation?
```
$ k2flix --start 545 --stop 680 --step 1 --fps 12 --dpi 25 http://archive.stsci.edu/missions/k2\
/target_pixel_files/c1/201500000/72000/ktwo201572338-c01_lpd-targ.fits.gz
```
<img src="https://raw.githubusercontent.com/barentsen/k2flix/master/examples/epic-201572338.gif" width="220" />

### Installation
If you have a working installation of Python on your system, you can install k2flix using pip:
```
pip install git+https://github.com/barentsen/k2flix
```
Alternatively, you can clone the repository and install from source:
```
$ git clone https://github.com/barentsen/k2flix.git
$ cd k2flix
$ python setup.py install
```
K2flix has only been tested under Linux at present.  Get in touch if you encounter issues on OS X or Windows.

### Using k2flix
Converting a Kepler pixel file to an animated gif:
```
$ k2flix tpf-file.fits.gz
```

Converting a Kepler pixel file to an MPEG-4 movie:
```
$ k2flix -o movie.mp4 tpf-file.fits.gz
```

K2flix supports reading from web URLs, so you can generate a movie directly from the data archive:
```
$ k2flix https://archive.stsci.edu/missions/k2/target_pixel_files/c1/201400000/00000/ktwo201400022-c01_lpd-targ.fits.gz
```

To see all the options, use the `--help` argument to see the full usage information:
```
$ k2flix --help
usage: k2flix [-h] [-o filename] [--start IDX] [--stop IDX] [--step STEP]
              [--fps FPS] [--dpi DPI] [--min_percent MIN_PERCENT]
              [--max_percent MAX_PERCENT] [--cmap colormap_name]
              filename [filename ...]

Converts a Target Pixel File (TPF) from NASA's Kepler/K2 spacecraft into a
movie or animated gif.

positional arguments:
  filename              path to one or more Kepler Target Pixel Files (TPF)

optional arguments:
  -h, --help            show this help message and exit
  -o filename           output filename (default: gif with the same name as
                        the input file)
  --start IDX           first frame to show (default: 0)
  --stop IDX            last frame to show (default: -1)
  --step STEP           spacing between frames (default: output 100 frames)
  --fps FPS             frames per second (default: 5)
  --dpi DPI             resolution of the output in dots per K2 pixel
  --min_percent MIN_PERCENT
                        percentile value used to determine the minimum cut
                        level (default: 1.0)
  --max_percent MAX_PERCENT
                        percentile value used to determine the maximum cut
                        level (default: 95.0)
  --cmap colormap_name  matplotlib color map name (default: gray)
```

### Contributing
To report bugs and request features, please use the [issue tracker](https://github.com/barentsen/k2flix/issues). Code contributions are very welcome.

### License
Copyright 2015 Geert Barentsen.

K2flix is free software made available under the MIT License. For details see the LICENSE file.
