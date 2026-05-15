import unittest
import tempfile


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.empty_fladir = tempfile.mkdtemp()
        self.outdir = tempfile.mkdtemp()
        self.model_type = "ols"

    def tearDown(self):
        pass

    def test_parser(self):
        pass
