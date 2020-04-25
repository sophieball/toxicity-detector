import nltk
import markdown

words = set(nltk.corpus.words.words())
english_vocab = set(w.lower() for w in nltk.corpus.words.words())


try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser


def remove_html(text, md=False):
    if md:
        text = markdown.markdown(text)
    # credit: stackoverflow
    class MLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reset()
            self.in_code = 0
            self.strict = False
            self.convert_charrefs= True
            self.fed = []
        def handle_starttag(self, tag, attr):
            if tag=="code":
                self.in_code += 1
        def handle_endtag(self, tag):
            if tag=="code":
                self.in_code -= 1
        def handle_data(self, d):
            if self.in_code==0:
                self.fed.append(d)
        def get_data(self):
            return ''.join(self.fed)

    s = MLStripper()
    s.feed(text)
    return s.get_data()

def remove_non_english(text):
    text = text.replace('"','').replace("'",'')
    text = (w for w in nltk.wordpunct_tokenize(text) \
         if w.lower() in words and w.isalpha())

    return " ".join(text)

def probability_of_english(text):
    words = nltk.wordpunct_tokenize(text)
    text_vocab = set(w.lower() for w in words if w.lower().isalpha())
    if len(text_vocab)==0:
        return 0
    return 1 - len(text_vocab.difference(english_vocab)) / len(text_vocab)

def remove_markdown(text):
    return remove_html(text, True)
