# Lint as: python3
"""Test constructing convokit conversation on some dummy data"""

import unittest
import sys
import io
import pandas as pd
import pickle
import conversation_struct


class TestConvStruct(unittest.TestCase):

  def test_conv(self):
    comments = pd.read_csv("src/data/test_conv_struct_data.csv")
    speakers = conversation_struct.create_speakers(comments)
    corpus = conversation_struct.prepare_corpus(comments, speakers, False)

    self.assertTrue(speakers is not None)
    self.assertTrue(corpus is not None)
    #self.assertEqual(len(corpus.get_speaker_ids()), 3)
    conv_ids = corpus.get_conversation_ids()
    self.assertEqual(len(conv_ids), 2)

    conv_ids = corpus.get_conversation_ids()
    first_convo = corpus.get_conversation(conv_ids[0])
    first_convo_utts = first_convo.get_utterance_ids()
    self.assertEqual(len(first_convo_utts), 5)

    first_convo_2nd_utt = first_convo.get_utterance(first_convo_utts[1])
    self.assertEqual(first_convo_2nd_utt.reply_to, "1_____1_____1")

    first_convo_3rd_utt = first_convo.get_utterance(first_convo_utts[2])
    self.assertEqual(first_convo_3rd_utt.reply_to, "1_____1_____1_____2205")

  def test_G_conv(self):
    # this test dataset contains empty text, the program should be able to
    # handle it
    comments = pd.read_csv("src/data/dummy_conv.csv")
    speakers = conversation_struct.create_speakers(comments)
    corpus = conversation_struct.prepare_corpus(comments, speakers, True)

    corpus.get_conversation("1").check_integrity()
    self.assertTrue(speakers is not None)
    self.assertEqual(len(speakers), 4)
    self.assertEqual(corpus.get_utterance("1_____1").reply_to, "1")


if __name__ == "__main__":
  unittest.main()
