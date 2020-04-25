import logging
import pickle
import time
import perspective
import stanford_polite
import text_cleaning
from wordfreq import word_frequency
from util import log_odds
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import re

logging.info("loading model")
model = pickle.load(open("pretrained_model.p","rb"))
# import suite


def compute_prediction_scores(text):
	# english? -- fast
	penglish = text_cleaning.probability_of_english(text)
	# call perspective API -- slow
	[perspective_score, perspective_raw] = perspective.get_perspective_score_raw(text)
	# compute stanford politeness score -- fairly fast
	politeness_score = stanford_polite.get_stanford_polite_score(text)
	# predict result
	score = model.predict([[perspective_score,politeness_score]])[0]
	return [score, perspective_score, perspective_raw, politeness_score, penglish]

def compute_prediction_final_score(text):
	[score, a,b,c,d]=compute_prediction_scores(text)
	return score

empty_report = { "score": 0, "reason": "empty"}

def compute_prediction_report(text):
	start = time.time()
	# remove markdown
	text = text_cleaning.remove_markdown(text)
	if text.strip() == "":
		return [empty_report, 0]
	[score, perspective_score, perspective_raw, politeness_score, penglish] = \
		compute_prediction_scores(text)
	# build the report
	result = { 
			"score": score.item(), 
			"orig" : {"score": score.item(), "persp": perspective_score, "polite": politeness_score, "persp_raw": perspective_raw},
			"en" : penglish
			}

	# if toxic, look at alternatives
	if score==1:
		alt_texts = clean_text(text)
		if len(alt_texts)==0:
			print(" == found toxic issue, no alternatives")
		else:
			print(" == found toxic issue, exploring "+str(len(alt_texts))+" alternatives")
			result["alt_tried"]=len(alt_texts)
			isToxic = True
			for a in alt_texts:
				if isToxic:
					[score, per, perr, pol, pe] = compute_prediction_scores(text)
					if score == 0:
						print(" === found nontoxic alternative")
						isToxic=False
						result["score"]=0
						result["alt"]={"text":a,"score": score, "persp": per, "polite": pol}
	return [result, time.time() - start]




counter = pickle.load(open("pickles/github_words.p","rb"))
our_words = dict([(i,word_frequency(i,"en")*10**9) for i in counter])
different_words = log_odds(defaultdict(int,counter),defaultdict(int,our_words))

# postprocessing (usually only done for toxic comments)
# returns list of clean text variants
def clean_text(text):
    result = []
    words = text.split(" ")
    words = [a.strip(',.!?:; ') for a in words]

    words = list(set(words))
    words = [word for word in words if not word.isalpha() or word.lower() in different_words]

    for word in set(words):
        # Maybe unkify?
        result += [re.sub(r'[^a-zA-Z0-9]' + re.escape(word.lower()) + r'[^a-zA-Z0-9]', ' potato ', " "+text.lower()+" ").strip()]

    tokenizer = RegexpTokenizer(r'\w+')
    all_words = tokenizer.tokenize(text)
    # logging.info("all_words "+str(all_words))
    # Try removing all unknown words
    for word in set(all_words):
        if word.lower() not in counter and word_frequency(word.lower(), "en") == 0 and len(word) > 2:
            text = text.replace(word, '')

    result += [text]
    return result
