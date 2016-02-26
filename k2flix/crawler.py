#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""Crawls the K2 archive in search for urls to TPF files.

Author: Geert Barentsen
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
import random
import posixpath
from bs4 import BeautifulSoup
try:
    import urlparse  # Python 2
    from urllib import urlopen
except ImportError:
    import urllib.parse as urlparse  # Python 3
    from urllib.request import urlopen

from astropy import log


class KeplerArchiveCrawler(object):
    """
    Class used to search for *targ.fits.fz files in the Kepler/K2 archive.

    Parameters
    ----------
    baseurl : str
        e.g. "http://archive.stsci.edu/missions/k2/target_pixel_files/c1"

    max_requests : int
        Maximum number of URLs that will be opened.
    """
    def __init__(self, baseurl, max_requests=2e5):
        self.baseurl = baseurl
        self.urlqueue = [baseurl]
        self.tpf_files = []
        self.visited = []
        self.max_requests = max_requests

    def __del__(self):
        self.output.close()

    def crawl(self, output_fn, sleep=0.5):
        """Run the crawler.

        Parameters
        ----------
        output_fn : str
            Path to the text file to which results will be written.

        sleep : int (optional)
            Number of seconds to sleep between HTTP requests. (default: 1)
        """
        self.output = open(output_fn, 'w')
        counter = 0
        while len(self.urlqueue) > 0:
            time.sleep(sleep)  # be nice to the server
            url = self.urlqueue.pop()
            log.info('{0}: visiting {1}'.format(counter, url))
            self.visit_url(url)
            counter += 1
            if counter >= self.max_requests:
                log.warning('Max requests reached '
                            '(visited {0} urls)'.format(counter))
                break
        self.output.close()

    def visit_url(self, url):
        """Searches a single url for TPF files or new subdirs."""
        self.visited.append(url)
        html = urlopen(url)
        soup = BeautifulSoup(html)
        links = [link.get('href') for link in soup.find_all('a', href=True)]
        for l in links:
            new_url = posixpath.join(url, l)
            if new_url.startswith('/'):
                # ignore absolute urls
                continue
            elif new_url.endswith('/'):
                # crawl new relative urls
                log.debug('Will crawl {0}'.format(new_url))
                self.add_url_to_check(new_url)
            elif new_url.endswith('targ.fits.gz') or new_url.endswith('targ.fits'):
                # found a target file!
                self.save_url(new_url)

    def add_url_to_check(self, url):
        """Adds a url to the crawling queue."""
        if url in self.visited:
            log.warning('already visited {0}'.format(url))
        else:
            self.urlqueue.append(url)

    def save_url(self, url):
        """Saves the urls of the FITS files found to a text file."""
        self.output.write(url + '\n')


class KeplerArchiveCrawlerDB():
    """Simple interface to the crawler's results file.

    Parameters
    ----------
    db_fn : str
        Path to the results file produced by `KeplerArchiveCrawler`.
    """
    def __init__(self, db_fn):
        self.db_fn = db_fn
        self.db = open(db_fn, 'r').readlines()

    def random_url(self):
        idx = random.randint(0, len(self.db))
        return self.db[idx]


# Example use
if __name__ == '__main__':
    # Create an index of TPF files
    campaign = 'c6'
    output_fn = '{0}-fits-urls.txt'.format(campaign)
    c = KeplerArchiveCrawler('http://archive.stsci.edu/missions/k2/'
                             'target_pixel_files/' + campaign)
    c.crawl(output_fn)

    # Example: get a random TPF url
    #db = KeplerArchiveCrawlerDB(output_fn)
    #url = db.random_url()
