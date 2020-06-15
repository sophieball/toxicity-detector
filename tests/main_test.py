# Lint as: python3
"""Unit tests."""

import unittest
import test_create_features
import test_receive_data

if __name__ == '__main__':
  test_classes_to_run = [test_create_features.TestCreateFeatures,
                         test_receive_data.TestReceiveData]
  loader = unittest.TestLoader()

  suites_list = []
  for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suites_list.append(suite)

  big_suite = unittest.TestSuite(suites_list)

  runner = unittest.TextTestRunner()
  results = runner.run(big_suite)
