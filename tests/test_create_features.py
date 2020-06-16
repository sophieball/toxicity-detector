# Lint as: python3
"""
test TextParsers
"""

import unittest
from src import text_parser

class TestCreateFeatures(unittest.TestCase):
  def test_percent_uppercase(self):
    text = "aA"
    uppercase = text_parser.percent_uppercase(text)
    self.assertEqual(0.5, uppercase)

    text = "A"
    uppercase = text_parser.percent_uppercase(text)
    self.assertEqual(1, uppercase)

    text = "a"
    uppercase = text_parser.percent_uppercase(text)
    self.assertEqual(0, uppercase)

  def test_count_ref(self):
    text = ">"
    num_ref = text_parser.count_reference_line(text)
    self.assertEqual(1, num_ref)

    text = "x"
    num_ref = text_parser.count_reference_line(text)
    self.assertEqual(0, num_ref)

    text = "x>"
    num_ref = text_parser.count_reference_line(text)
    self.assertEqual(0, num_ref)

    text = "x\n>"
    num_ref = text_parser.count_reference_line(text)
    self.assertEqual(1, num_ref)

  def test_remove_ref(self):
    text = ">"
    text = text_parser.remove_reference(text)
    self.assertEqual("", text)

    text = "hello\n>x"
    text = text_parser.remove_reference(text)
    self.assertEqual("hello\n", text)

    text = "hello\n>x\nhello"
    text = text_parser.remove_reference(text)
    self.assertEqual("hello\n\nhello", text)

  def test_plus_one(self):
    text = "+1"
    num_plus_one = text_parser.count_plus_one(text)
    self.assertEqual(1, num_plus_one)

    text = ""
    num_plus_one = text_parser.count_plus_one(text)
    self.assertEqual(0, num_plus_one)

    text = "+1"
    clean = text_parser.sub_PlusOne(text)
    print(clean)
    self.assertEqual("plus one", clean)

    text = "hello+1world"
    clean = text_parser.sub_PlusOne(text)
    self.assertEqual("helloplus oneworld", clean)
