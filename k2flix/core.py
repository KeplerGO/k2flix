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
import matplotlib.patheffects as path_effects
from matplotlib.colors import NoNorm

from astropy.io import fits
from astropy.time import Time
from astropy import log
from astropy import visualization

###
# Kepler constants
###

# Quality flags, from the Kepler Archive Manual
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
    "524288": "Possible thruster firing",
    "1048576": "Thruster firing"
}


###
# Kepler time functions
###

def bkjd_to_bjd(bkjd, offset=2454833.):
    """Barycentric Kepler Julian Date (BKJD) to Barycentric Julian Date (BJD).

    Kepler Barycentric Julian Day is the value recorded in the 'TIME' column
    in the official Kepler/K2 lightcurve and target pixel files.
    It is a Julian day minus 2454833.0 (UTC=January 1, 2009 12:00:00)
    and corrected to be the arrival times at the barycenter
    of the Solar System. See Section 2.3.2 in the Kepler Archive Manual.
    """
    return bkjd + offset


def bkjd_to_jd(bkjd, timecorr, timslice):
    """Convert Barycentric Kepler Julian Date (BKJD) to Julian Date (JD)."""
    return bkjd_to_bjd(bkjd) - timecorr + (0.25 + 0.62 * (5 - timslice)) / 86400.


def bkjd_to_mjd(bkjd, timecorr, timslice):
    """Convert Barycentric Kepler Julian Date (BKJD) to Modified Julian Date (MJD)."""
    return bkjd_to_jd(bkjd, timecorr, timslice) - 2400000.5


###
# Helper classes
###

class BadKeplerFrame(Exception):
    """Raised if a frame is empty."""
    pass


class BadCadenceRange(Exception):
    """Raised if a the range of frames is invalid."""
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

    verbose : boolean
        If True, print debugging output.
    """
    def __init__(self, filename, cache=True, verbose=False):
        self.filename = filename
        self.hdulist = fits.open(filename, cache=cache)
        self.no_frames = len(self.hdulist[1].data['FLUX'])
        self.verbose = verbose

    @property
    def objectname(self):
        try:
            return self.hdulist[0].header['OBJECT']
        except KeyError:
            return ''

    @property
    def ra(self):
        return self.hdulist[0].header['RA_OBJ']

    @property
    def dec(self):
        return self.hdulist[0].header['DEC_OBJ']

    def cadenceno(self, frameno=None):
        """Returns the cadence number for a given frame, or all frames."""
        if frameno is None:
            return self.hdulist[1].data['CADENCENO']
        return self.hdulist[1].data['CADENCENO'][frameno]

    def bkjd(self, frameno=None):
        """Get the Barycentric Kepler Julian Day (BKJD) for a given frame.

        Kepler Barycentric Julian Day is a Julian day minus 2454833.0 (UTC=January
        1, 2009 12:00:00) and corrected to be the arrival times at the barycenter
        of the Solar System. See Section 2.3.2 in the Kepler Archive Manual.
        """
        if frameno is None:
            return self.hdulist[1].data['TIME']
        return self.hdulist[1].data['TIME'][frameno]

    def bjd(self, frameno=None):
        """Get the Barycentric Julian Date for a given frame."""
        return bkjd_to_bjd(self.bkjd(frameno))

    def jd(self, frameno=None):
        """Get the Julian Day for a given frame."""
        if frameno is None:
            timecorr = self.hdulist[1].data['TIMECORR']
        else:
            timecorr = self.hdulist[1].data['TIMECORR'][frameno]
        return bkjd_to_jd(self.bkjd(frameno), timecorr, self.hdulist[1].header['TIMSLICE'])

    def mjd(self, frameno=None):
        """Get the Modified Julian Day for a given frame."""
        if frameno is None:
            timecorr = self.hdulist[1].data['TIMECORR']
        else:
            timecorr = self.hdulist[1].data['TIMECORR'][frameno]
        return bkjd_to_mjd(self.bkjd(frameno), timecorr, self.hdulist[1].header['TIMSLICE'])

    def time(self, time_format='bkjd'):
        if time_format == 'jd':
            return self.jd()
        elif time_format == 'mjd':
            return self.mjd()
        elif time_format == 'bjd':
            return self.bjd()
        elif time_format == 'bkjd':
            return self.bkjd()
        elif time_format == 'cadence':
            return self.cadenceno()

    def timestamp(self, frameno, time_format='ut'):
        """Returns the timestamp for a given frame number.

        Parameters
        ----------
        frameno : int
            Index of the image in the file, starting from zero.

        time_format : str
            One of 'frameno', 'ut', jd', 'mjd', 'bjd', 'bkjd', or 'cadence'.

        Returns
        -------
        timestamp : str
            Appropriately formatted timestamp.
        """
        if time_format == 'frameno':
            return 'Frame {}'.format(frameno)
        elif time_format == 'ut':
            try:
                return Time(self.jd(frameno), format='jd').iso[0:19]
            except ValueError:
                return np.nan
        elif time_format == 'jd':
            return 'JD {:.2f}'.format(self.jd(frameno))
        elif time_format == 'mjd':
            return 'MJD {:.2f}'.format(self.mjd(frameno))
        elif time_format == 'bjd':
            return 'BJD {:.2f}'.format(self.bjd(frameno))
        elif time_format == 'bkjd':
            return 'BKJD {:.2f}'.format(self.bkjd(frameno))
        elif time_format == 'cadence':
            return 'Cadence {:d}'.format(self.cadenceno(frameno))

    def quality_flags(self, frameno):
        """Returns a list of strings describing the quality flags raised."""
        quality = self.hdulist[1].data['QUALITY'][frameno]
        flags = []
        for flag in KEPLER_QUALITY_FLAGS.keys():
            if quality & int(flag) > 0:
                flags.append(KEPLER_QUALITY_FLAGS[flag])
        return flags

    def flux(self, frameno=0, data_col='FLUX', pedestal=1000.):
        """Returns the data for a given frame.

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
        # Quick hack for TESS simulated data:
        if (data_col == 'COSMIC_RAYS') and (data_col not in self.hdulist[1].data.columns.names) and ('COSMICS' in self.hdulist[1].data.columns.names):
            data_col = 'COSMICS'
        flux = self.hdulist[1].data[data_col][frameno].copy() + pedestal
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            if data_col != 'COSMIC_RAYS' and np.all(np.isnan(flux)):
                raise BadKeplerFrame('frame {0}: bad frame'.format(frameno))
        return flux

    def flux_binned(self, frameno=0, binning=1, data_col='FLUX', pedestal=1000):
        """Returns the co-added data centered on a frame.

        Parameters
        ----------
        frameno : int
            Frame number.

        binning : int
            Number of frames to co-add, centered around `frameno`.

        raw : bool
            If `True` return the raw counts, if `False` return the calibrated
            flux. (Default: `False`.)

        pedestal : float
            Value to add to the pixel counts.  This is useful to help prevent
            the counts from being negative after background subtraction.
            (Default: 1000.)
        """
        flx = self.flux(frameno, data_col=data_col, pedestal=pedestal)
        framecount = 1
        # Add additional frames if binning was requested
        if binning > 1:
            for frameno_offset in np.arange(-binning/2, binning/2, 1, dtype=int):
                if frameno_offset == 0:
                    continue  # We already included the center frame above
                frameno_to_add = frameno + frameno_offset
                if frameno_to_add < 0 or frameno_to_add > self.no_frames - 1:
                    continue  # Avoid going out of bounds
                flx += self.flux(frameno_to_add, data_col=data_col, pedestal=pedestal)
                framecount += 1
            flx = flx / framecount
            if self.verbose:
                print('Frame {}: co-adding {} cadences.'.format(frameno, framecount))
        return flx

    def cut_levels(self, min_percent=1., max_percent=95., data_col='FLUX',
                   sample_start=None, sample_stop=None, n_samples=3):
        """Determine the cut levels for contrast stretching.

        For speed, the levels are determined using only `n_samples` number
        of frames selected between `sample_start` and `sample_stop`.

        Returns
        -------
        vmin, vmax : float, float
            Min and max cut levels.
        """
        if sample_start is None:
            sample_start = 0
        if sample_stop is None:
            sample_stop = -1
        if sample_stop < 0:
            sample_stop = self.no_frames + sample_stop
        # Build a sample of pixels
        try:
            sample = np.concatenate(
                                    [
                                     self.flux(frameno, data_col=data_col)
                                     for frameno
                                     in np.linspace(sample_start, sample_stop,
                                                    n_samples).astype(int)
                                     ]
                                    )
        # If we hit a bad frame, then try to find a random set of good frames
        except BadKeplerFrame:
            success = False
            # Try 20 times
            for _ in range(20):
                try:
                    sample = np.concatenate(
                                    [
                                     self.flux(frameno, data_col=data_col)
                                     for frameno
                                     in np.random.choice(np.arange(sample_start,
                                                                   sample_stop+1),
                                                         size=n_samples, replace=False)
                                     ]
                                    )
                    success = True
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

    def create_figure(self, frameno=0, binning=1, dpi=None,
                      stretch='log', vmin=1, vmax=5000,
                      cmap='gray', data_col='FLUX', annotate=True,
                      time_format='ut', show_flags=False):
        """Returns a matplotlib Figure object that visualizes a frame.

        Parameters
        ----------
        frameno : int
            Image number in the target pixel file.

        binning : int
            Number of frames around `frameno` to co-add. (default: 1).

        dpi : float, optional [dots per inch]
            Resolution of the output in dots per Kepler CCD pixel.
            By default the dpi is chosen such that the image is 440px wide.

        vmin : float, optional
            Minimum cut level (default: 1).

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
        # Get the flux data to visualize
        flx = self.flux_binned(frameno=frameno, binning=binning, data_col=data_col)

        # Determine the figsize and dpi
        shape = list(flx.shape)
        if dpi is None:
            # Twitter timeline requires dimensions between 440x220 and 1024x512
            # so we make 440 the default
            dpi = 440 / float(shape[0])

        # libx264 require the height to be divisible by 2, we ensure this here:
        shape[1] -= ((shape[1] * dpi) % 2) / dpi
        # Create the figureand display the flux image using matshow
        fig = pl.figure(figsize=shape, dpi=dpi)
        # Display the image using matshow
        ax = fig.add_subplot(1, 1, 1)
        if self.verbose:
            print('{} vmin/vmax = {}/{} (median={})'.format(data_col, vmin, vmax, np.nanmedian(flx)))

        if stretch == 'linear':
            stretch_fn = visualization.LinearStretch()
        elif stretch == 'sqrt':
            stretch_fn = visualization.SqrtStretch()
        elif stretch == 'power':
            stretch_fn = visualization.PowerStretch(1.0)
        elif stretch == 'log':
            stretch_fn = visualization.LogStretch()
        elif stretch == 'asinh':
            stretch_fn = visualization.AsinhStretch(0.1)
        else:
            raise ValueError('Unknown stretch: {0}.'.format(stretch))

        transform = (stretch_fn +
                     visualization.ManualInterval(vmin=vmin, vmax=vmax))
        flx_transform = 255 * transform(flx)
        # Make sure to remove all NaNs!
        flx_transform[~np.isfinite(flx_transform)] = 0
        ax.imshow(flx_transform.astype(int), aspect='auto',
                  origin='lower', interpolation='nearest',
                  cmap=cmap, norm=NoNorm())
        if annotate:  # Annotate the frame with a timestamp and target name?
            fontsize = 3. * shape[0]
            margin = 0.03
            # Print target name in lower left corner
            txt = ax.text(margin, margin, self.objectname,
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

    def _frameno_range(self, start, stop, time_format):
        """Returns the first and last frame numbers to use, given user input."""
        # If no range was set, or the range is in a trivial format (frameno),
        # or in a format we don't support ranges for (ut), then the range
        # determination is straightforward:
        if (start is None and stop is None) or time_format in ['frameno', 'ut']:
            # Set defaults if necessary
            if start is None:
                start = 0
            if stop is None:
                stop = self.no_frames - 1
            return start, stop

        # Set defaults to cover all possible JD/MJD/BKJD/BJD/Cadenceno values
        if start is None:
            start = -9e99
        if stop is None:
            stop = 9e99
        # Compute the times for all cadences and find the frame number range
        times = self.time(time_format)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="(.*)invalid value(.*)")
            idx = np.where((times >= start) & (times <= stop))[0]
        if len(idx) == 0:
            raise BadCadenceRange('No frames found for {} = [{}, {}]'.format(time_format, start, stop))
        return idx[0], idx[-1]

    def save_movie(self, output_fn=None, start=None, stop=None, step=None,
                   binning=1, fps=15., dpi=None, stretch='log',
                   min_cut=None, max_cut=None, min_percent=1., max_percent=95.,
                   cmap='gray', time_format='ut', show_flags=False, data_col='FLUX',
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
            Number or time of the first frame to show.
            If `None` (default), the first frame will be the start of the data.

        stop : int
            Number or time of the last frame to show.
            If `None` (default), the last frame will be the end of the data.

        step : int
            Spacing between frames.  Default is to set the stepsize
            automatically such that the movie contains 100 frames between
            start and stop.

        binning : int
            Number of frames to co-add per frame.  The default is 1.

        fps : float (optional)
            Frames per second.  Default is 15.0.

        dpi : float (optional)
            Resolution of the output in dots per Kepler pixel.
            The default is to produce output which is 440px wide.

        min_cut : float, optional
            Minimum cut level.  The default is None (i.e. use min_percent).

        max_cut : float, optional
            Minimum cut level.  The default is None (i.e. use max_percent).

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
        if output_fn is None:
            output_fn = self.filename.split('/')[-1] + '.gif'
        # Determine the first/last frame number and the step size
        frameno_start, frameno_stop = self._frameno_range(start, stop, time_format)
        if step is None:
            step = int((frameno_stop - frameno_start) / 100)
            if step < 1:
                step = 1
        # Determine cut levels for contrast stretching from a sample of pixels
        if min_cut is None or max_cut is None:
            vmin, vmax = self.cut_levels(min_percent=min_percent,
                                         max_percent=max_percent,
                                         sample_start=frameno_start,
                                         sample_stop=frameno_stop,
                                         data_col=data_col)
        if min_cut is not None:
            vmin = min_cut
        if max_cut is not None:
            vmax = max_cut
        # Create the movie frames
        print('Creating {0}'.format(output_fn))
        viz = []
        for frameno in tqdm(np.arange(frameno_start, frameno_stop + 1, step, dtype=int)):
            try:
                fig = self.create_figure(frameno=frameno, binning=binning,
                                         dpi=dpi, stretch=stretch,
                                         vmin=vmin, vmax=vmax, cmap=cmap,
                                         data_col=data_col, time_format=time_format,
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
                    "Kepler/K2/TESS spacecraft into an animated gif or MPEG-4 movie for human inspection.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='')
    parser.add_argument('--output', metavar='FILENAME',
                        type=str, default=None,
                        help='.gif or .mp4 output filename (default: gif with the same name as the input file)')
    parser.add_argument('--start', type=float, default=None,
                        help='first frame to show. Give the frame number (default 0), or a Julian Day if --jd/--mjd/--bkjd is set, or a cadence number if --cadence is set.')
    parser.add_argument('--stop', type=float, default=None,
                        help='final frame to show. Give the frame number (default: -1), or a Julian Day if --jd/--mjd/--bkjd is set, or a cadence number if --cadence is set.')
    parser.add_argument('--step', type=int, default=None,
                        help='spacing between frames '
                             '(default: show 100 frames)')
    parser.add_argument('--fps', type=float, default=15.,
                        help='frames per second (default: 15)')
    parser.add_argument('--binning', type=int, default=1,
                        help='number of cadence to co-add per frame '
                             '(default: 1)')
    parser.add_argument('--dpi', type=float, default=None,
                        help='resolution of the output in dots per K2 pixel '
                             '(default: choose a dpi that produces a 440px-wide image)')
    parser.add_argument('--stretch', type=str, default='log',
                        help='type of contrast stretching: "linear", "sqrt", '
                             '"power", "log", or "asinh" (default is "log")')
    parser.add_argument('--min_cut', type=float, default=None,
                        help='minimum cut level (default: use min_percent)')
    parser.add_argument('--max_cut', type=float, default=None,
                        help='maximum cut level (default: use max_percent)')
    parser.add_argument('--min_percent', metavar='%', type=float, default=1.,
                        help='minimum cut percentile (default: 1.0)')
    parser.add_argument('--max_percent', metavar='%', type=float, default=95.,
                        help='maximum cut percentile (default: 95)')
    parser.add_argument('--cmap', type=str,
                        default='gray', help='matplotlib color map name '
                                             '(default: gray)')
    parser.add_argument('--flags', action='store_true',
                        help='show the quality flags')
    parser.add_argument('tpf_filename', nargs='+',
                        help='path to one or more Target Pixel Files (TPF)')

    datagroup = parser.add_mutually_exclusive_group()
    datagroup.add_argument('--raw', action='store_true',
                           help="show the uncalibrated pixel counts ('RAW_CNTS')")
    datagroup.add_argument('--background', action='store_true',
                           help="show the background flux ('FLUX_BKG')")
    datagroup.add_argument('--cosmic', action='store_true',
                           help="show the cosmic rays ('COSMIC_RAYS')")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ut', action='store_true',
                       help='use Universal Time for annotation (default)')
    group.add_argument('--jd', action='store_true',
                       help='use Julian Day for annotation and --start/--stop')
    group.add_argument('--mjd', action='store_true',
                       help='use Modified Julian Day for annotation and --start/--stop')
    group.add_argument('--bkjd', action='store_true',
                       help='use Kepler Julian Day for annotation and --start/--stop')
    group.add_argument('--cadence', action='store_true',
                       help='use Cadence Number  for annotation and --start/--stop')

    args = parser.parse_args(args)

    # What is the data column to show?
    if args.raw:
        data_col = 'RAW_CNTS'
    elif args.background:
        data_col = 'FLUX_BKG'
    elif args.cosmic:
        data_col = 'COSMIC_RAYS'
    else:
        data_col = 'FLUX'

    # What is the time format to use?
    if args.ut:
        time_format = 'ut'
    elif args.jd:
        time_format = 'jd'
    elif args.mjd:
        time_format = 'mjd'
    elif args.bkjd:
        time_format = 'bkjd'
    elif args.cadence:
        time_format = 'cadence'
    else:
        time_format = 'frameno'

    for fn in args.tpf_filename:
        try:
            tpf = TargetPixelFile(fn, verbose=args.verbose)
            tpf.save_movie(output_fn=args.output,
                           start=args.start,
                           stop=args.stop,
                           step=args.step,
                           fps=args.fps,
                           binning=args.binning,
                           dpi=args.dpi,
                           stretch=args.stretch,
                           min_cut=args.min_cut,
                           max_cut=args.max_cut,
                           min_percent=args.min_percent,
                           max_percent=args.max_percent,
                           cmap=args.cmap,
                           time_format=time_format,
                           show_flags=args.flags,
                           data_col=data_col)
        except Exception as e:
            if args.verbose:
                raise e
            else:
                print('ERROR: {}'.format(e))

# Example use
if __name__ == '__main__':
    fn = ('http://archive.stsci.edu/missions/kepler/target_pixel_files/'
          '0007/000757076/kplr000757076-2010174085026_lpd-targ.fits.gz')
    tpf = TargetPixelFile(fn)
    tpf.save_movie('/tmp/animation.mp4')
