"""Summary
"""
from unittest import TextTestRunner, TestSuite, TestLoader
from test_naivebayes import *


def suite():
    """Summary

    Returns:
        TYPE: Description
    """
    loader = TestLoader()
    suite = TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestNaiveBayesModel))
    return suite


if __name__ == '__main__':
    runner = TextTestRunner(verbosity=2)
    runner.run(suite())
