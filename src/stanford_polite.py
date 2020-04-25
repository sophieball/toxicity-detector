import nltk
import sys
sys.path.insert(0, "politeness3")
import politeness3.model

def get_stanford_polite_score(text): # score
	sentences = nltk.sent_tokenize(text)
	return politeness3.model.score(
            {"sentences": sentences,
             "text": text})['polite']
