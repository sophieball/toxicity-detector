import perspective

def test_perspective():
	assert .9 < perspective.get_perspective_score("I hate you")
	assert .1 > perspective.get_perspective_score("I love you")
	assert 0 == perspective.get_perspective_score(" ")

def test_perspective_after_text_processing():	
	import text_cleaning
	assert 0 == perspective.get_perspective_score(text_cleaning.remove_markdown("`foo`"))
	assert .1 > perspective.get_perspective_score(text_cleaning.remove_markdown("I `hate` you"))

def test_perspective_raw():	
	[s,r] = perspective.get_perspective_score_raw("Du bist nicht sehr schlau, oder?")
	assert r["detectedLanguages"]==["de"]
	assert r["attributeScores"]["TOXICITY"]["summaryScore"]["value"]==s

 
