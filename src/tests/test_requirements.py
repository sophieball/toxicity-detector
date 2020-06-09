# Lint as: python3
"""Test availability of required packages."""

import unittest

import pkg_resources

from pathlib import Path
_REQUIREMENTS_PATH = Path(__file__).with_name("requirements.txt")

class TestRequirements(unittest.TestCase):
  """Test availability of required packages."""

  def test_requirements(self):
    """Test that each required package is available."""
    # Ref: https://stackoverflow.com/a/45474387/
    requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH.open())
    for requirement in requirements:
      requirement = str(requirement)
      with self.subTest(requirement=requirement):
        pkg_resources.require(requirement)

test_req = TestRequirements()
test_req.test_requirements()
