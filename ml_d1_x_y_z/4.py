# TODO popuniti kodom za problem 4
import re
import pickle
import os
import nltk
import numpy as np
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
from functools import partial, reduce
from collections import defaultdict
from zipfile import ZipFile


# pomocne funkcije

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


# funkcija koja vraca tuple matrice BOW feature vektora i vokabular 
@time_it
def bag_of_words(path, broj_vektora=-1):
    # Cistimo korpus
    print('Cleaning the corpus...')
    clean_corpus = []
    porter = PorterStemmer()

    stop_punc = set(stopwords.words('english')).union(set(punctuation))
    for folder in os.listdir(path):
        for i, filename in enumerate(os.listdir(os.path.join(path, folder))):
            if i == broj_vektora:
                break
            specijalni_karakteri = r'[(.,\*\"\'!?\(\)\[\]\$^@%£\-#:)/\\]'
            doc = re.sub(specijalni_karakteri, '',
                         re.sub('<br />', '', otvori(os.path.join(path + '/' + folder, filename))))
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

    vocab = list(map(lambda x: x[0], sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)[:10000]))

    # print('Vocab:', list(zip(vocab, range(len(vocab)))))
    # print('Vocab size: ', len(vocab))

    np.set_printoptions(precision=2, linewidth=200)

    def create_BOW(i):
        return np.array(list(map(partial(numocc_score, i), vocab)))

    def numocc_score(doc, word):
        return doc.count(word)

    # Kreiramo Bag-Of-Words feature vektore
    print('Creating BOW features...')
    X = np.array(list(map(create_BOW, clean_corpus)))

    # print('X:')
    # print(X)
    # print()

    return X, vocab


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
        # print('Priors:')
        # print(self.priors)

        # Racunamo broj pojavljivanja svake reci u svakoj klasi
        occs = np.zeros((self.nb_classes, self.nb_words))

        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                cnt = X[i][w]
                occs[c][w] += cnt
        # print('Occurences:')
        # print(occs)

        # Racunamo P(Rec_i|Klasa) - likelihoods
        self.like = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.alpha
                down = np.sum(occs[c]) + self.nb_words * self.alpha
                self.like[c][w] = up / down
        # print('Likelihoods:')
        # print(self.like)

    @time_it
    def predict(self, bows):
        predictions = []
        for bow in bows:
            # Racunamo P(Klasa|bow) za svaku klasu
            probs = np.zeros(self.nb_classes)  # [0.7, 0.3]
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

# funkcija koja izmesa matricu feature vektora, fituje model, testira ga i vraca rezultate testiranja
def bayes(X):
    Y = np.concatenate((np.zeros(1250, dtype=np.int32), np.ones(1250, dtype=np.int32)))

    trening = dict()
    test = dict()
    osamdeset = 2500 * 80 // 100

    indices = np.random.permutation(2500)
    X = X[indices]
    Y = Y[indices]

    trening['x'] = X[:osamdeset]
    trening['y'] = Y[:osamdeset]

    test['x'] = X[osamdeset:]
    test['y'] = Y[osamdeset:]

    test_bow = test['x']

    model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(X[0]), alpha=1)
    print('Fitting the model...')
    model.fit(trening['x'], trening['y'])
    print('Testing the model...')
    prediction = model.predict(test_bow)
    # prediction = model.predict_multiply(test_bow)

    return list(zip(prediction, test['y']))

# racuna preciznost testa
def verify(resenje):
    brojac = 0
    for x, y in resenje:
        if x == y:
            brojac += 1
    print(f'accuracy: {brojac / len(resenje)}')
    return brojac / len(resenje)

# pravi matricu konfuzije
def create_confusion_matrix(resenje):
    TP, TN, FP, FN = 0, 0, 0, 0
    for x, y in resenje:
        if x == 1 and y == 1:
            TP += 1
        elif x == 0 and y == 0:
            TN += 1
        elif x == 1 and y == 0:
            FP += 1
        else:
            FN += 1
    return [[TN, FP], [FN, TP]]

# c)
def pod_c(X, vocab):

    negativni = X[:1250]
    pozitivni = X[1250:]

    # print(vocab)

    negativni_vektor = np.zeros(10000, dtype=int)
    for vector in negativni:
        negativni_vektor += vector
    negativni_vektor1 = sorted(list(zip(vocab, negativni_vektor)), key=lambda item: item[1], reverse=True)
    prvih_deset_negativnih = negativni_vektor1[:5]

    pozitivni_vektor = np.zeros(10000, dtype=int)
    for vector in pozitivni:
        pozitivni_vektor += vector
    pozitivni_vektor1 = sorted(list(zip(vocab, pozitivni_vektor)), key=lambda item: item[1], reverse=True)
    prvih_deset_pozitivnih = pozitivni_vektor1[:5]

    for i, item in enumerate(prvih_deset_negativnih):
        print(f'{i + 1} NEGATIVNI:{item}\tPOZITIVNI:{prvih_deset_pozitivnih[i]}')


    def odredi_RL(ulaz):
        if ulaz[0][1] < 10 or ulaz[1][1] < 10:
            return 0, 0
        return ulaz[0][0], round(ulaz[0][1] / ulaz[1][1], 5)

    x = list(zip(list(zip(vocab, pozitivni_vektor)), list(zip(vocab, negativni_vektor))))

    RL_list = list(map(odredi_RL, x))
    # print(RL_list[:4])
    # RL_min = RL_list.sort(key=lambda x,y : y, reverse= False)
    # print(RL_min)
    RL_max = sorted(RL_list, key=lambda item: item[1], reverse=True)[:10]

    def reduce_RL_min(izlaz, ulaz):
        if ulaz == (0, 0):
            return izlaz
        return izlaz + [ulaz]

    RL_min = reduce(reduce_RL_min, sorted(RL_list, key=lambda item: item[1], reverse=False), [])[:10]

    print()
    for i, item in enumerate(RL_max):
        print(f'{i + 1} MAX_RL:{item}\tMIN_RL:{RL_min[i]}')


if __name__ == '__main__':

    if not os.path.exists('data/imdb/'):
        print('Unzipping imdb.zip')
        zf = ZipFile('data/imdb.zip', 'r')
        zf.extractall('data/')
        zf.close()
  
    # nltk.download()
    # X, vocab_dict = bag_of_words(path='data/imdb/imdb', broj_vektora=-1)
    # save_to_disk('X1.txt',X)
    # save_to_disk('vocab_dict', vocab_dict)

    # X = load_from_disk('X1.txt')
    # vocab = load_from_disk(('vocab_dict'))
    # test_results = bayes(X)
    # accuracy = verify(test_results)
    # print(accuracy)

    # a)
    print('a)\n')
    lista_accuracy = []
    for x in range(3):
        print(f'++++++++++++++++++++++++++\nPOKRETANJE BR. {x+1}')
        X, vocab = bag_of_words(path='data/imdb/', broj_vektora=-1)
        results_data = bayes(X)
        lista_accuracy.append(verify(results_data))
    print(f'list_accuracy: {lista_accuracy}')

    # b)
    print('b)\n')
    matrix = create_confusion_matrix(results_data)
    print(f'confusion_matrix: {matrix}')

    # c)
    print('c)\n')
    pod_c(X, vocab)

    '''

    Mozemo primetiti da obe vrste kritika dele najcecih 5 reci, cak im je i broj pojavljivanja slican.

    Mozemo primetiti sta razlikuje negativne od pozitivnih komentara. Rec 'stupid' na primer, se pojavljuje 10 puta vise 
    u negativnim nego u pozitivnim komentarima. Ako bismo posmatrali obrnuto proporcionalnu vrednost MAX_RL, primeticemo da
    se duplo razlikuje vrednost od MIN_RL. To je ili zato sto na reci iz MIN_RL ne nailazimo u pozitivnim komentarima, ili 
    zato sto se reci iz MAX_RL pojavljuju nesto cesce u negativnim. Posmatrajmo izmisljeni komentar u kome koristimo
    MAX_RL reci u negativnom kontekstu: 
    "I strongly disagree! This isnt unique, this movie didnt highlight how amazing and beautiful my hometown is like!"

    '''


