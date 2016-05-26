import sys
import csv
import itertools

import nltk

from datamodel import Sequence
from brainforge.Architecture.NNModel import RNN
from brainforge.Utility.cost import Xent
from brainforge.Utility.activations import *

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
csvsroot = dataroot + "csvs/"
datapath = csvsroot + "reddit.csv"

crossval = 0.2
pca = 0

dataargs = (crossval, pca)

time = 7
neurons = 4
eta = 0.3
cost = Xent
act = Tanh

netargs = (neurons, eta, cost, act)


def getrnn(data, args):
    hiddens, lrate, costfn, hidact = args
    data.standardize()
    net = RNN(hiddens, data, lrate, costfn, hidact)
    return net


def pull_reddit_data(path, args: tuple):

    cross_val, pca_pcs = args

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(path, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times."
          % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print("\nExample sentence: '%s'" % sentences[0])
    print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

    # Create the training data
    X = np.array([np.array([word_to_index[w] for w in sent[:-1]]) for sent in tokenized_sentences])
    Y = np.array([np.array([word_to_index[w] for w in sent[1:]]) for sent in tokenized_sentences])

    data = Sequence((X, Y), vocabulary=word_to_index, cross_val=cross_val)

    return data


def main(dargs, nargs):
    net = getrnn(pull_reddit_data(datapath, dargs), nargs)
    net.learn(time)

if __name__ == '__main__':
    main(dataargs, netargs)
