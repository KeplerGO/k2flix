"""Tests the basic functionality."""
import os
from k2flix import TargetPixelFile


class TestCore():

    def setup_class(self):
        data = os.path.join(os.path.dirname(__file__), 'data')
        self.tpf = TargetPixelFile(os.path.join(data, 'tpf-example.fits.gz'))

    def test_time(self):
        assert self.tpf.timestamp(0)[0:19] == '2014-05-30 16:16:51'
        assert self.tpf.timestamp(900)[0:19] == '2014-06-18 01:36:52'

    def test_flux(self):
        assert self.tpf.flux(500).shape == (15, 16)

    def test_annotated_image(self):
        tmp = self.tpf.annotated_image(1000)

    def test_save_gif(self, tmpdir):
        moviepath = tmpdir.join('test.gif')
        self.tpf.save_movie(output_fn=str(moviepath),
                            start=10, stop=20, step=1)
        assert moviepath.check()

    def test_save_mp4(self, tmpdir):
        moviepath = tmpdir.join('test.mp4')
        self.tpf.save_movie(output_fn=str(moviepath),
                            start=10, stop=20, step=1)
        assert moviepath.check()
