# TODO popuniti kodom za problem 4
import re
import pickle
import os
import nltk
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import re
from functools import partial
from collections import defaultdict


def time_it(f):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        funkcija = f(*args, **kwargs)
        print(f'{f.__name__}\t{time.time() - start} sekundi')
        return funkcija

    return wrapper

@time_it
def save_to_disk(path, item):
    with open(path, 'wb') as f:
        pickle.dump(item, f)

@time_it
def load_from_disk(path):
    with open(path, 'rb') as f:
        item = pickle.load(f)
    return item

def otvori(path):
    with open(path, 'r', encoding='utf-8') as review:
        return review.read()

@time_it
def bag_of_words(path, broj_vektora=-1):

    # Cistimo korpus
    print('Cleaning the corpus...')
    clean_corpus = []
    porter = PorterStemmer()

    stop_punc = set(stopwords.words('english')).union(set(punctuation))
    for folder in os.listdir(path):
        for i, filename in enumerate(os.listdir(os.path.join(path,folder))):
            if i == broj_vektora:
                break
            specijalni_karakteri = r'[(.,\*\"\'!?\(\)\[\]\$^@%£\-#:)/\\]'
            doc = re.sub(specijalni_karakteri, '', re.sub('<br />', '', otvori(os.path.join(path+'/'+folder, filename))))
            words = wordpunct_tokenize(doc)
            words_lower = [w.lower() for w in words]
            words_filtered = [w for w in words_lower if w not in stop_punc]
            words_stemmed = [porter.stem(w) for w in words_filtered]
            clean_corpus.append(words_stemmed)

    # Kreiramo vokabular
    print('Creating the vocab...')
    vocab_dict = defaultdict(lambda: 0)
    for doc in clean_corpus:
        for word in doc:
            vocab_dict[word] += 1

    vocab = list(map(lambda x: x[0] , sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)[:10000]))

    print('Vocab:', list(zip(vocab, range(len(vocab)))))
    print('Vocab size: ', len(vocab))

    np.set_printoptions(precision=2, linewidth=200)

    def create_BOW(i):
        return np.array(list(map(partial(numocc_score, i), vocab)))

    def numocc_score(doc, word):
        return doc.count(word)

    # Kreiramo Bag-Of-Words feature vektore
    print('Creating BOW features...')
    X = np.array(list(map(create_BOW, clean_corpus)))

    print('X:')
    print(X)
    print()

    return X


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, alpha):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.alpha = alpha

    @time_it
    def fit(self, X, Y):
        nb_examples = X.shape[0]

        # Racunamo P(Klasa) - priors
        # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
        # broja u intervalu [0, maksimalni broj u listi]
        self.priors = np.bincount(Y) / nb_examples
        print('Priors:')
        print(self.priors)

        # Racunamo broj pojavljivanja svake reci u svakoj klasi
        occs = np.zeros((self.nb_classes, self.nb_words))
        '''
        [[0, 0, .... 0],
        [0, 0, .....0]]
        '''
        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                cnt = X[i][w]
                occs[c][w] += cnt
        print('Occurences:')
        print(occs)
        
        # Racunamo P(Rec_i|Klasa) - likelihoods
        self.like = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.alpha
                down = np.sum(occs[c]) + self.nb_words*self.alpha
                self.like[c][w] = up / down
        print('Likelihoods:')
        print(self.like)

    
    
    @time_it  
    def predict(self, bows):
        predictions = []
        for bow in bows: 
            # Racunamo P(Klasa|bow) za svaku klasu
            probs = np.zeros(self.nb_classes) # [0.7, 0.3]
            for c in range(self.nb_classes):
                prob = np.log(self.priors[c])
                for w in range(self.nb_words):
                    cnt = bow[w]
                    prob += cnt * np.log(self.like[c][w])
                probs[c] = prob
            # Trazimo klasu sa najvecom verovatnocom
            # print('\"Probabilites\" for a test BoW (with log):')
            # print(probs)
            predictions += [np.argmax(probs)]

        return np.asarray(predictions)




def bayes(X):

    def verify(resenje):
        brojac = 0
        for x, y in resenje:
            if x == y:
                brojac += 1
        print(brojac/len(prediction))
        return brojac/len(prediction)

    Y = np.concatenate((np.zeros(1250, dtype=np.int64),np.ones(1250, dtype= np.int64)))

    trening = dict()
    test = dict()
    osamdeset = 2500*80//100

    indices = np.random.permutation(2500)
    X = X[indices]
    Y = Y[indices]

    trening['x'] = X[:osamdeset]
    trening['y'] = Y[:osamdeset]

    test['x'] = X[osamdeset:]
    test['y'] = Y[osamdeset:]

    test_bow = test['x']

    model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(X[0]), alpha=1)
    model.fit(trening['x'], trening['y'])
    prediction = model.predict(test_bow)
    # prediction = model.predict_multiply(test_bow)
    resenje = verify(zip(prediction, test['y']))


    return resenje
    
if __name__ == '__main__':
    # nltk.download()
    # X = bag_of_words(path='data/imdb/imdb', broj_vektora=-1)
    # save_to_disk('X1.txt',X)
    
    X = load_from_disk('X1.txt')
    lista_accuracy = []
    for x in range(3):
        lista_accuracy.append(bayes(X))

    print(lista_accuracy)

    
    

