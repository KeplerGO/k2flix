#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Posts a random Kepler Target Pixel File (TPF) to Twitter.

Author: Geert Barentsen
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random
from twython import Twython

from astropy import log
from astropy.coordinates import SkyCoord

from k2flix import TargetPixelFile
from k2flix.crawler import KeplerArchiveCrawlerDB

from secrets import *


def generate_tweet(tpf_fn=None, movie_length=96):
    """Generate a status message and animated gif.

    Parameters
    ----------
    tpf_fn : str (optional)
        Path or url to a TPF file. If `None`, a random file will be downloaded.

    move_length : int (optional)
        Number of frames in the animation.

    Returns
    -------
    (status, gif, tpf) : (str, str, `TargetPixelFile`)
    """
    # Open the Target Pixel File
    if tpf_fn is None:  # Get a random url
        db = KeplerArchiveCrawlerDB('c1-urls.txt')
        tpf_fn = db.random_url()
    log.info('Opening {0}'.format(tpf_fn))
    tpf = TargetPixelFile(tpf_fn)
    log.info('KEPMAG = {0}, DIM = {1}'.format(tpf.hdulist[0].header['KEPMAG'],
                                              tpf.hdulist[1].header['TDIM5']))
    # Files contain occasional bad frames, so we make multiple attempts
    # with random starting points
    attempt_no = 0
    while attempt_no < 7:
        attempt_no += 1
        try:
            start = random.randint(0, tpf.no_frames - movie_length)
            crd = SkyCoord(tpf.ra, tpf.dec, unit='deg')
            ra, dec = crd.to_string('hmsdms', sep=':').split()
            try:
                kepmag = ', KepMag {0:.1f}'.format(float(
                         tpf.hdulist[0].header['KEPMAG']))
            except Exception:
                kepmag = ''
            simbad = ("http://simbad.u-strasbg.fr/simbad/sim-coo?"
                      "output.format=HTML&Coord={0}{1}&Radius=0.5".format(ra[0:8], dec[0:9]))
            status = "{0} (RA {1}, Dec {2}{3}) on {4}. {5}".format(
                        tpf.target,
                        ra[0:8],
                        dec[0:9],
                        kepmag,
                        tpf.timestamp(start)[0:10],
                        simbad)
            log.info(status)
            # Creat the animated gif
            #gif_fn = tpf_fn.split('/')[-1] + '.gif'
            gif_fn = '/tmp/keplerbot.gif'
            tpf.save_movie(gif_fn, start=start, stop=start + movie_length,
                           step=1, fps=8, min_percent=0., max_percent=94.,
                           ignore_bad_frames=True)
            return status, gif_fn, tpf
        except Exception as e:
            log.warning(e)
    raise Exception('Tweet failed')


def post_tweet(status, gif):
    """Post an animated gif and associated status message to Twitter."""
    twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    # response = twitter.update_status_with_media(status=status, media=open(gif, 'rb'))
    upload_response = twitter.upload_media(media=open(gif, 'rb'))
    response = twitter.update_status(status=status, media_ids=upload_response['media_id'])
    log.info(response)
    return twitter, response


if __name__ == '__main__':
    attempt_no = 0
    while attempt_no < 10:
        attempt_no += 1
        try:
            status, gif, tpf = generate_tweet()
            #twitter, response = post_tweet(status, gif)
            break
        except Exception as e:
            log.warning(e)
