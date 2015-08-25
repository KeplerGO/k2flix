#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""Create movies or animated gifs from Kepler Target Pixel Files.

Author: Geert Barentsen
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ["TargetPixelFile"]

import warnings
import imageio
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.image import imsave
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects

from astropy.io import fits
from astropy.time import Time
from astropy.utils.console import ProgressBar
from astropy import log


class BadKeplerFrame(Exception):
    """Raised if a frame is empty."""
    pass

class TargetPixelFile(object):
    """Represent a Target Pixel File (TPC) from the Kepler spacecraft.

    Parameters
    ----------
    filename : str
        Path of the pixel file.

    cache : boolean
        If the file name is a URL, this specifies whether or not to save the
        file locally in Astropy’s download cache. (default: True)
    """
    def __init__(self, filename, cache=True):
        self.filename = filename
        self.hdulist = fits.open(filename, cache=cache)
        self.no_frames = len(self.hdulist[1].data['FLUX'])

    @property
    def target(self):
        return self.hdulist[0].header['OBJECT']

    @property
    def ra(self):
        return self.hdulist[0].header['RA_OBJ']

    @property
    def dec(self):
        return self.hdulist[0].header['DEC_OBJ']

    def timestamp(self, frame):
        """Returns the ISO timestamp for a given frame number.

        Parameters
        ----------
        frame : int
            Index of the image in the file, starting from zero.

        Returns
        -------
        timestamp : str
            ISO-formatted timestamp "YYYY-MM-DD HH:MM:SS"
        """
        # Note: we are using barycentric julian date;
        # see Kepler Archive Manual Sect. 2.3.2
        time_value = self.hdulist[1].data['TIME'][frame]
        if np.isnan(time_value):
            raise BadKeplerFrame('frame {0}: empty time value'.format(frame))
        bjd = (time_value
               + self.hdulist[1].header['BJDREFI']
               + self.hdulist[1].header['BJDREFF'])
        t = Time(bjd, format='jd')
        return t.iso

    def flux(self, frame=0):
        flux = self.hdulist[1].data['FLUX'][frame].copy()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            if np.all(np.isnan(flux) | (flux < 1e-5)):
                raise BadKeplerFrame('frame {0}: bad frame'.format(frame))
        return flux

    def annotated_image(self, frame=0, dpi=None, vmin=1, vmax=5000,
                        cmap='gray'):
        """Returns the visualization (image array) for a single frame.

        Parameters
        ----------
        frame : int
            Image number in the target pixel file.

        dpi : float, optional [dots per inch]
            Resolution of the output in dots per Kepler CCD pixel.
            By default the dpi is chosen such that the image is 440px wide.

        vmin : float, optional
            Minimum cut level (default: 0).

        vmax : float, optional
            Maximum cut level (default: 5000).

        cmap : str, optional
            The matplotlib color map name.  The default is 'gray',
            can also be e.g. 'gist_heat'.

        Returns
        -------
        image : array
            An array of unisgned integers of shape (x, y, 3),
            representing an RBG colour image x px wide and y px high.
        """
        flx = self.flux(frame)
        if dpi is None:
            # Twitter timeline requires dimensions between 440x220 and 1024x512
            dpi = 440 / float(flx.shape[0])
        fontsize = 3. * flx.shape[0]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            flx[np.isnan(flx) | (flx < vmin)] = vmin
        # Create the frame
        fig = pl.figure(figsize=flx.shape, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.matshow(flx, aspect='auto',
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap=cmap, origin='lower',
                   interpolation='nearest')
        # Annotate the frame
        margin = 0.03
        # Target name
        txt = ax.text(margin, margin, self.target,
                      family="monospace", fontsize=fontsize,
                      color='white', transform=ax.transAxes)
        # ISO timestamp
        txt2 = ax.text(1 - margin, margin,
                       self.timestamp(frame)[0:16],
                       family="monospace", fontsize=fontsize,
                       color='white', ha='right',
                       transform=ax.transAxes)
        txt.set_path_effects([path_effects.Stroke(linewidth=fontsize/6.,
                                                  foreground='black'),
                              path_effects.Normal()])
        txt2.set_path_effects([path_effects.Stroke(linewidth=fontsize/6.,
                                                   foreground='black'),
                               path_effects.Normal()])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        fig.canvas.draw()
        # Convert the frame to a numpy array
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pl.close(fig)
        return img

    def save_movie(self, output_fn=None, start=0, stop=-1, step=None, fps=15.,
                   dpi=None, min_percent=1., max_percent=95., cmap='gray',
                   ignore_bad_frames=True):
        """Save an animation.

        Parameters
        ----------
        output_fn : str
            The filename of the output movie.  The type of the movie
            is determined from the filename (e.g. use '.gif' to save
            as an animated gif). The default is a GIF file with the same name
            as the input FITS file.

        start : int
            Number of the first frame in the TPF file to show.  Default is 0.

        stop : int
            Number of the last frame in the TPF file to show.
            Default is -1 (i.e. the last frame).

        step : int
            Spacing between frames.  Default is to set the stepsize
            automatically such that the movie contains 100 frames between
            start and stop.

        fps : float (optional)
            Frames per second.  Default is 15.0.

        dpi : float (optional)
            Resolution of the output in dots per Kepler pixel.
            The default is to produce output which is 440px wide.

        min_percent : float, optional
            The percentile value used to determine the pixel value of
            minimum cut level.  The default is 1.0.

        max_percent : float, optional
            The percentile value used to determine the pixel value of
            maximum cut level.  The default is 95.0.

        cmap : str, optional
            The matplotlib color map name.  The default is 'gray',
            can also be e.g. 'gist_heat'.

        ignore_bad_frames : boolean
             If true, any frames which cannot be rendered will be ignored
             without raising a `BadKeplerFrame` exception.
        """
        if stop < 0:
            stop = self.no_frames + stop
        if step is None:
            step = (stop - start) / 100
        if output_fn is None:
            output_fn = self.filename.split('/')[-1] + '.gif'
        log.info('Writing {0}'.format(output_fn))
        # Determine cut levels for contrast stretching from a sample
        # First we try to use the first and last frame
        try:
            sample = np.concatenate((self.flux(start), self.flux(stop)))
        except BadKeplerFrame:
            # If the first or last frame are no good, then find first good frame
            success = False
            for idx in range(stop):
                try:
                    sample = self.flux(np.random.randint(stop))
                    break
                except BadKeplerFrame:
                    pass
            if not success:
                raise BadKeplerFrame("Could not find a good frame.")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            vmin, vmax = np.percentile(sample[sample > 0],
                                       [min_percent, max_percent])
        # Create the movie frames
        viz = []
        for frameno in ProgressBar(np.arange(start, stop+1, step, dtype=int)):
            try:
                viz.append(self.annotated_image(frame=frameno, dpi=dpi,
                                                vmin=vmin, vmax=vmax,
                                                cmap=cmap))
            except BadKeplerFrame as e:
                log.warning(e)
                if not ignore_bad_frames:
                    raise e
        # Save the output as a movie
        if output_fn.endswith('.gif'):
            kwargs = {'duration': 1. / fps}
        else:
            kwargs = {'fps': fps}
        imageio.mimsave(output_fn, viz, **kwargs)


def k2flix_main(args=None):
    """Script to convert Kepler pixel data (TPF files) to a movie.""" 
    parser = argparse.ArgumentParser(
        description="Converts a Target Pixel File (TPF) from NASA's "
                    "Kepler/K2 spacecraft into a movie or animated gif.")
    parser.add_argument('-o', metavar='filename', type=str, default=None,
                        help='output filename (default: gif with the same name'
                             ' as the input file)')
    parser.add_argument('--start', metavar='IDX', type=int, default=0,
                        help='first frame to show (default: 0)')
    parser.add_argument('--stop', metavar='IDX', type=int, default=-1,
                        help='last frame to show (default: -1)')
    parser.add_argument('--step', type=int, default=None,
                        help='spacing between frames '
                             '(default: output 100 frames)')
    parser.add_argument('--fps', type=float, default=15.,
                        help='frames per second (default: 5)')
    parser.add_argument('--dpi', type=float, default=None,
                        help='resolution of the output in dots per K2 pixel')
    parser.add_argument('--min_percent', type=float, default=1.,
                        help='percentile value used to determine the '
                             'minimum cut level (default: 1.0)')
    parser.add_argument('--max_percent', type=float, default=95.,
                        help='percentile value used to determine the '
                             'maximum cut level (default: 95.0)')
    parser.add_argument('--cmap', metavar='colormap_name', type=str,
                        default='gray', help='matplotlib color map name '
                                             '(default: gray)')
    parser.add_argument('filename', nargs='+',
                        help='path to one or more Kepler '
                             'Target Pixel Files (TPF)')
    args = parser.parse_args(args)

    for fn in args.filename:
        tpf = TargetPixelFile(fn)
        tpf.save_movie(output_fn=args.o,
                       start=args.start,
                       stop=args.stop,
                       step=args.step,
                       fps=args.fps,
                       dpi=args.dpi,
                       min_percent=args.min_percent,
                       max_percent=args.max_percent,
                       cmap=args.cmap)

# Example use
if __name__ == '__main__':
    fn = ('http://archive.stsci.edu/missions/kepler/target_pixel_files/'
          '0007/000757076/kplr000757076-2010174085026_lpd-targ.fits.gz')
    tpf = TargetPixelFile(fn)
    tpf.save_movie('/tmp/animation.mp4')
