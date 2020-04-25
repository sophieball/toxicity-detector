import stanford_polite

def test_stanford_polite():
	assert 0.5 < stanford_polite.get_stanford_polite_score("Can you help, please?")
	assert 0.5 > stanford_polite.get_stanford_polite_score("Help!")
	assert 0.5 > stanford_polite.get_stanford_polite_score("")
