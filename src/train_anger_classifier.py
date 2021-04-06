import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC

text = open("data/anger.txt").read().split("\n")
label = [i.split("\t")[1] for i in text]
train = [i.split("\t")[-1][1:-1] for i in text]
train = [(train[i], label[i]) for i in range(len(train))]
all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
train = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]
classifier = SklearnClassifier(LinearSVC())
classifier.train(train)
pickle.dump(classifier, open("pickles/anger.p", "wb"))
