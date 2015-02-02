k2flix
======
Creates beautiful quicklook animated gifs or mp4 movies from the pixel data
("target pixel files") produced by NASA's Kepler/K2 spacecraft.

Installation
------------
If you have a working installation of Python on your system, you can install k2flix using pip:
```
pip install git+https://github.com/barentsen/k2flix
```

Usage
-----
Converting a Kepler pixel file to an animated gif:
```
$ k2flix tpf-filename.fits.gz
```

Converting a Kepler pixel file to an MPEG-4 movie:
```
$ k2flix -o movie.mp4 tpf-filename.fits.gz
```

Printing the full usage information:
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

License
-------
Copyright 2015 Geert Barentsen.

k2flix is free software made available under the MIT License. For details see the LICENSE file.
