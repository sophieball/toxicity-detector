import text_cleaning

def test_remove_non_english():
	text = "Du bist sehr schlau"
	clean = text_cleaning.remove_non_english(text)
	assert "" == clean
	text = "This is great"
	clean = text_cleaning.remove_non_english(text)
	assert text == clean

def test_probability_of_english():
	assert 0==text_cleaning.probability_of_english("Du bist sehr schlau")
	assert 1==text_cleaning.probability_of_english("The weather is great")
	assert 1>text_cleaning.probability_of_english("The weather is schlau")

def test_markdown_cleaning():
	assert "foo"==text_cleaning.remove_html("<test>foo</test>")
	assert "#foo\n..."==text_cleaning.remove_html("#foo\n...")
	assert "foo\n..."==text_cleaning.remove_html("#foo\n...",True)
	assert "foo bar x"==text_cleaning.remove_html("foo *bar* x",True)

def test_markdown_code_removal():
	assert "foo  x"==text_cleaning.remove_html("foo `code` x",True)
	assert "foo\n\nx"==text_cleaning.remove_html("foo\n```\ncode\n```\nx\n",True)

def test_probability_of_english():
	assert 0 == text_cleaning.probability_of_english("")
	assert 1 == text_cleaning.probability_of_english("hi there")
	assert .5 == text_cleaning.probability_of_english("Deutschland you")

