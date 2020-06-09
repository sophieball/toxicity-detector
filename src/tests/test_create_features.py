# Lint as: python3
"""
test TextParsers
"""

import unittest
#from TextParser import *
from TextParser import *
from text_modifier import percent_uppercase

class TestCreateFeatures(unittest.TestCase):
  def test_percent_uppercase(self):
    text = "aA"
    uppercase = percent_uppercase(text)
    self.assertEqual(0.5, uppercase)

    text = "A"
    uppercase = percent_uppercase(text)
    self.assertEqual(1, uppercase)

    text = "a"
    uppercase = percent_uppercase(text)
    self.assertEqual(0, uppercase)

  def test_count_ref(self):
    text = ">"
    num_ref = count_reference_line(text)
    self.assertEqual(1, num_ref)

    text = "x"
    num_ref = count_reference_line(text)
    self.assertEqual(0, num_ref)

    text = "x>"
    num_ref = count_reference_line(text)
    self.assertEqual(0, num_ref)

    text = "x\n>"
    num_ref = count_reference_line(text)
    self.assertEqual(1, num_ref)

  def test_remove_ref(self):
    text = ">"
    text = remove_reference(text)
    self.assertEqual("", text)

    text = "hello\n>x"
    text = remove_reference(text)
    self.assertEqual("hello\n", text)

    text = "hello\n>x\nhello"
    text = remove_reference(text)
    self.assertEqual("hello\n\nhello", text)

  def test_plus_one(self):
    text = "+1"
    num_plus_one = count_plus_one(text)
    self.assertEqual(1, num_plus_one)

    text = ""
    num_plus_one = count_plus_one(text)
    self.assertEqual(0, num_plus_one)

    text = "+1"
    clean = sub_PlusOne(text)
    print(clean)
    self.assertEqual("plus one", clean)

    text = "hello+1world"
    clean = sub_PlusOne(text)
    self.assertEqual("helloplus oneworld", clean)

if __name__ == '__main__':
    unittest.main()
