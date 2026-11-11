"""核对领域原语的业务数值。"""
import unittest
from analysis_data.io import load
from analysis_data.domain import summarize
class DomainTests(unittest.TestCase):
    def test_discount(self):
        from analysis_data.domain import discounted
        self.assertEqual(str(discounted('10',1,'1/4')),'8')
        self.assertEqual(str(discounted('-3/2',0,'1/4')),'-3/2')
        with self.assertRaises(ValueError):discounted('1',1,'-1')
        self.assertEqual(summarize(load('data/normal.json')[0])['nodes'],7)
