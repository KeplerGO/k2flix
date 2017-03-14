"""Tests the basic functionality."""
import os
import imageio
from k2flix import TargetPixelFile


class TestCore():

    def setup_class(self):
        url = ('http://archive.stsci.edu/missions/k2/target_pixel_files/c1/'
               '201500000/08000/ktwo201508413-c01_lpd-targ.fits.gz')
        self.tpf = TargetPixelFile(url)

    def test_time(self):
        assert self.tpf.timestamp(0)[0:19] == '2014-05-30 16:09:28'
        assert self.tpf.timestamp(900)[0:19] == '2014-06-18 01:31:25'

    def test_flux(self):
        assert self.tpf.flux(500).shape == (15, 16)

    def test_create_figure(self):
        tmp = self.tpf.create_figure(1000)

    def test_save_gif(self, tmpdir):
        moviepath = tmpdir.join('test.gif')
        self.tpf.save_movie(output_fn=str(moviepath),
                            start=10, stop=20, step=1)
        assert moviepath.check()

    def test_save_mp4(self, tmpdir):
        imageio.plugins.ffmpeg.download()
        moviepath = tmpdir.join('test.mp4')
        self.tpf.save_movie(output_fn=str(moviepath),
                            start=10, stop=20, step=1)
        assert moviepath.check()
