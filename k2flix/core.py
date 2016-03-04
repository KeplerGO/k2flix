#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""Create movies or animated gifs from Kepler Target Pixel Files.

Author: Geert Barentsen
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ["TargetPixelFile"]

import argparse
import imageio
import numpy as np
from tqdm import tqdm
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.image import imsave
import matplotlib.patheffects as path_effects

from astropy.io import fits
from astropy.time import Time
from astropy import log
from astropy import visualization


# The meaning of the various flags are described in the Kepler Archive Manual
KEPLER_QUALITY_FLAGS = {
    "1": "Attitude tweak",
    "2": "Safe mode",
    "4": "Coarse point",
    "8": "Earth point",
    "16": "Zero crossing",
    "32": "Desaturation event",
    "64": "Argabrightening",
    "128": "Cosmic ray",
    "256": "Manual exclude",
    "1024": "Sudden sensitivity dropout",
    "2048": "Impulsive outlier",
    "4096": "Argabrightening",
    "8192": "Cosmic ray",
    "16384": "Detector anomaly",
    "32768": "No fine point",
    "65536": "No data",
    "131072": "Rolling band",
    "262144": "Rolling band",
    "545288": "Possible thruster firing",
    "2097152": "Thruster firing"
}


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

    def bjd(self, frameno):
        """Get the Barycentric Julian Date for a given frame."""
        return (self.hdulist[1].data['TIME'][frameno] +
                self.hdulist[1].header['BJDREFI'] +
                self.hdulist[1].header['BJDREFF'])

    def bkjd(self, frameno):
        """Get the Barycentric Kepler Julian Day (BKJD) for a given frame.

        Kepler Barycentric Julian Day is a Julian day minus 2454833.0 (UTC=January
        1, 2009 12:00:00) and corrected to be the arrival times at the barycenter
        of the Solar System. See Section 2.3.2 in the Kepler Archive Manual.
        """
        return self.bjd(frameno) - 2454833.

    def jd(self, frameno):
        """Get the Julian Day for a given frame."""
        return (self.bjd(frameno) -
                self.hdulist[1].data['TIMECORR'][frameno] +
                (0.25 + 0.62 * (5 - self.hdulist[1].header['TIMSLICE'])) / (86400.))

    def mjd(self, frameno):
        """Get the Modified Julian Day for a given frame."""
        return self.jd(frameno) - 2400000.5

    def timestamp(self, frameno, time_format='iso'):
        """Returns the ISO timestamp for a given frame number.

        Parameters
        ----------
        frameno : int
            Index of the image in the file, starting from zero.

        time_format : str
            One of 'iso', 'jd', 'mjd', 'bjd', 'bkjd', or 'cadence'.

        Returns
        -------
        timestamp : str
            Appropriately formatted timestamp.
        """
        if time_format == 'iso':
            return Time(self.jd(frameno), format='jd').iso[0:19]
        elif time_format == 'jd':
            return 'JD {:.2f}'.format(self.jd(frameno))
        elif time_format == 'mjd':
            return 'MJD {:.2f}'.format(self.mjd(frameno))
        elif time_format == 'bjd':
            return 'BJD {:.2f}'.format(self.bjd(frameno))
        elif time_format == 'bkjd':
            return 'BKJD {:.2f}'.format(self.bkjd(frameno))
        elif time_format == 'cadence':
            return 'Cadence {:d}'.format(self.hdulist[1].data['CADENCENO'][frameno])

    def quality_flags(self, frameno):
        """Returns a list of strings describing the quality flags raised."""
        quality = self.hdulist[1].data['QUALITY'][frameno]
        flags = []
        for flag in KEPLER_QUALITY_FLAGS.keys():
            if quality & int(flag) > 0:
                flags.append(KEPLER_QUALITY_FLAGS[flag])
        return flags

    def flux(self, frameno=0, raw=False, pedestal=1000.):
        """Returns the raw or calibrated flux data for a given frame.

        Parameters
        ----------
        frameno : int
            Frame number.

        raw : bool
            If `True` return the raw counts, if `False` return the calibrated
            flux. (Default: `False`.)

        pedestal : float
            Value to add to the pixel counts.  This is useful to help prevent
            the counts from being negative after background subtraction.
            (Default: 1000.)
        """
        if raw:
            flux_column = "RAW_CNTS"
        else:
            flux_column = "FLUX"
        flux = self.hdulist[1].data[flux_column][frameno].copy() + pedestal
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            if np.all(np.isnan(flux)):
                raise BadKeplerFrame('frame {0}: bad frame'.format(frameno))
        return flux

    def cut_levels(self, min_percent=1., max_percent=95., raw=False,
                   sample_start=0, sample_stop=-1, n_samples=3):
        """Determine the cut levels for contrast stretching.

        For speed, the levels are determined using only `n_samples` number
        of frames selected between `sample_start` and `sample_stop`.

        Returns
        -------
        vmin, vmax : float, float
            Min and max cut levels.
        """
        if sample_stop < 0:
            sample_stop = self.no_frames + sample_stop
        # Build a sample of pixels
        try:
            sample = np.concatenate(
                                    [
                                     self.flux(frameno, raw=raw)
                                     for frameno
                                     in np.linspace(sample_start, sample_stop,
                                                    n_samples, dtype=int)
                                     ]
                                    )
        # If we hit a bad frame, then try to find a random set of good frames
        except BadKeplerFrame:
            success = False
            # Try 20 times
            for idx in range(20):
                try:
                    sample = np.concatenate(
                                    [
                                     self.flux(frameno, raw=raw)
                                     for frameno
                                     in np.random.randint(sample_start,
                                                          sample_stop,
                                                           n_samples)
                                     ]
                                    )
                    break  # Leave loop on success
                except BadKeplerFrame:
                    pass  # Try again
            if not success:
                raise BadKeplerFrame("Could not find a good frame.")
        # Finally, determine the cut levels, ignoring invalid value warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            vmin, vmax = np.percentile(sample[sample > 0],
                                       [min_percent, max_percent])
        return vmin, vmax

    def create_figure(self, frameno=0, dpi=None, vmin=1, vmax=5000,
                      cmap='gray', raw=False, annotate=True,
                      time_format='iso', show_flags=False):
        """Returns a matplotlib Figure object that visualizes a frame.

        Parameters
        ----------
        frameno : int
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

        raw : boolean, optional
            If `True`, show the raw pixel counts rather than
            the calibrated flux. Default: `False`.

        annotate : boolean, optional
            Annotate the Figure with a timestamp and target name?
            (Default: `True`.)

        show_flags : boolean, optional
            Show the quality flags?
            (Default: `False`.)

        Returns
        -------
        image : array
            An array of unisgned integers of shape (x, y, 3),
            representing an RBG colour image x px wide and y px high.
        """
        flx = self.flux(frameno, raw=raw)
        shape = list(flx.shape)
        # Determine the figsize and dpi
        if dpi is None:
            # Twitter timeline requires dimensions between 440x220 and 1024x512
            # so we make 440 the default
            dpi = 440 / float(shape[0])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            flx[np.isnan(flx) | (flx < vmin)] = vmin
        # libx264 require the height to be divisible by 2, we ensure this here:
        shape[1] -= ((shape[1] * dpi) % 2) / dpi
        # Create the figureand display the flux image using matshow
        fig = pl.figure(figsize=shape, dpi=dpi)
        # Display the image using matshow
        ax = fig.add_subplot(1, 1, 1)
        transform = (visualization.LogStretch() +
                     visualization.ManualInterval(vmin=vmin, vmax=vmax))
        ax.matshow(transform(flx), aspect='auto',
                   cmap=cmap, origin='lower',
                   interpolation='nearest')
        if annotate:  # Annotate the frame with a timestamp and target name?
            fontsize = 3. * shape[0]
            margin = 0.03
            # Print target name in lower left corner
            txt = ax.text(margin, margin, self.target,
                          family="monospace", fontsize=fontsize,
                          color='white', transform=ax.transAxes)
            txt.set_path_effects([path_effects.Stroke(linewidth=fontsize/6.,
                                                      foreground='black'),
                                  path_effects.Normal()])
            # Print a timestring in the lower right corner
            txt2 = ax.text(1 - margin, margin,
                           self.timestamp(frameno, time_format=time_format),
                           family="monospace", fontsize=fontsize,
                           color='white', ha='right',
                           transform=ax.transAxes)
            txt2.set_path_effects([path_effects.Stroke(linewidth=fontsize/6.,
                                                       foreground='black'),
                                   path_effects.Normal()])
            # Print quality flags in upper right corner
            if show_flags:
                flags = self.quality_flags(frameno)
                if len(flags) > 0:
                    txt3 = ax.text(margin, 1 - margin, '\n'.join(flags),
                                   family="monospace", fontsize=fontsize*1.3,
                                   color='white', ha='left', va='top',
                                   transform=ax.transAxes,
                                   linespacing=1.5, backgroundcolor='red')
                    txt3.set_path_effects([path_effects.Stroke(linewidth=fontsize/6.,
                                                               foreground='black'),
                                           path_effects.Normal()])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        fig.canvas.draw()
        return fig

    def save_movie(self, output_fn=None, start=0, stop=-1, step=None, fps=15.,
                   dpi=None, min_percent=1., max_percent=95., cmap='gray',
                   time_format='iso', show_flags=False, raw=False,
                   ignore_bad_frames=True,):
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

        raw : boolean, optional
            If `True`, show the raw pixel counts rather than
            the calibrated flux. Default: `False`.

        show_flags : boolean, optional
            If `True`, annotate the quality flags if set.

        ignore_bad_frames : boolean, optional
             If `True`, any frames which cannot be rendered will be ignored
             without raising a ``BadKeplerFrame`` exception. Default: `True`.
        """
        if stop < 0:
            stop = self.no_frames + stop
        if step is None:
            step = int((stop - start) / 100)
            if step < 1:
                step = 1
        if output_fn is None:
            output_fn = self.filename.split('/')[-1] + '.gif'
        print('Writing {0}'.format(output_fn))
        # Determine cut levels for contrast stretching from a sample of pixels
        vmin, vmax = self.cut_levels(min_percent=min_percent,
                                     max_percent=max_percent,
                                     raw=raw)
        # Create the movie frames
        viz = []
        for frameno in tqdm(np.arange(start, stop+1, step, dtype=int)):
            try:
                fig = self.create_figure(frameno=frameno, dpi=dpi,
                                         vmin=vmin, vmax=vmax, cmap=cmap,
                                         raw=raw, time_format=time_format,
                                         show_flags=show_flags)
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                pl.close(fig)  # Avoids memory leak!
                viz.append(img)
            except BadKeplerFrame as e:
                log.debug(e)
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
    parser.add_argument('-o', '--output', metavar='filename',
                        type=str, default=None,
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
                        help='frames per second (default: 15)')
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
    parser.add_argument('--raw', action='store_true',
                        help="show the uncalibrated pixel data ('RAW_CNTS')")
    parser.add_argument('--flags', action='store_true',
                        help='show quality flags')
    parser.add_argument('filename', nargs='+',
                        help='path to one or more Kepler '
                             'Target Pixel Files (TPF)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--jd', action='store_true',
                       help='show the Julian Day')
    group.add_argument('--mjd', action='store_true',
                       help='show the Modified Julian Day')
    group.add_argument('--bjd', action='store_true',
                       help='show the Barycentric Julian Day')
    group.add_argument('--bkjd', action='store_true',
                       help='show the Bareycentric Kepler Julian Day')
    group.add_argument('--cadence', action='store_true',
                       help='show the cadence number')

    args = parser.parse_args(args)

    # What is the time format to use?
    if args.jd:
        time_format = 'jd'
    elif args.mjd:
        time_format = 'mjd'
    elif args.bjd:
        time_format = 'bjd'
    elif args.bkjd:
        time_format = 'bkjd'
    elif args.cadence:
        time_format = 'cadence'
    else:
        time_format = 'iso'

    for fn in args.filename:
        tpf = TargetPixelFile(fn)
        tpf.save_movie(output_fn=args.output,
                       start=args.start,
                       stop=args.stop,
                       step=args.step,
                       fps=args.fps,
                       dpi=args.dpi,
                       min_percent=args.min_percent,
                       max_percent=args.max_percent,
                       cmap=args.cmap,
                       time_format=time_format,
                       show_flags=args.flags,
                       raw=args.raw)

# Example use
if __name__ == '__main__':
    fn = ('http://archive.stsci.edu/missions/kepler/target_pixel_files/'
          '0007/000757076/kplr000757076-2010174085026_lpd-targ.fits.gz')
    tpf = TargetPixelFile(fn)
    tpf.save_movie('/tmp/animation.mp4')
