import numpy as np

from .nputils import floatX

YAY = 1.0
NAY = 0.0

UNKNOWN = "<UNK>"


class _Data:
    """Base class for Data Wrappers"""

    def __init__(self, source, cross_val, indeps_n, header, sep, end,
                 standardize, pca, autoencode):

        def parse_source():
            if isinstance(source, np.ndarray):
                return parsearray(source, header, indeps_n)
            elif isinstance(source, tuple):
                return parselearningtable(source)
            elif isinstance(source, str) and "lt.pkl.gz" in source.lower():
                return parselearningtable(source)
            elif isinstance(source, str) and "mnist" in source.lower():
                return parselearningtable(mnist_to_lt(source))
            elif isinstance(source, str) and (".csv" or ".txt" in source.lower()):
                return parsecsv(source, header, indeps_n, sep, end)
            else:
                raise TypeError("DataWrapper doesn't support supplied data source!")

        def determine_no_testing():
            err = "Invalid value for cross_val! Can either be\nrate (float 0.0-1.0)\n" + \
                  "number of testing examples (0 <= int <= len(data))\n" + \
                  "the literals 'full', 'half' or 'quarter', or the NoneType object, None."
            if isinstance(cross_val, float):
                if not (0.0 <= cross_val <= 1.0):
                    raise ValueError(err)
                return int(data.shape[0] * cross_val)
            elif isinstance(cross_val, int):
                return cross_val
            elif isinstance(cross_val, str):
                cv = cross_val.lower()
                if cv not in ("full", "half", "quarter"):
                    raise ValueError(err)
                return int(data.shape[0] * {"f": 1.0, "h": 0.5, "q": 0.25}[cv[0]])
            elif cross_val is None:
                return 0.0
            else:
                raise TypeError(err)

        def transformations():
            if pca and autoencode:
                print("Warning! You chose to do PCA and autoencoding simultaneously on the data.")
            if standardize:
                self._transformation = self.self_standardize
            if pca:
                self._transformation = lambda: self.fit_pca(pca)
            if autoencode:
                self._transformation = lambda: self.fit_autoencoder(autoencode)

        self.learning = None
        self.testing = None
        self.lindeps = None
        self.tindeps = None
        self._pca = None
        self._autoencoder = None
        self._standardize_factors = None
        self._transformation = None
        self.type = None

        self.N = 0
        self.mean = 0
        self.std = 0

        headers, data, indeps = parse_source()
        self.n_testing = determine_no_testing()
        self._crossval = cross_val

        self.headers = headers
        self.data, self.indeps = data, indeps
        self.data.flags["WRITEABLE"] = False

        transformations()

    def table(self, data):
        """Returns a learning table"""
        dat = {"l": self.learning,
               "t": self.testing,
               "d": self.data}[data[0]]
        ind = {"l": self.lindeps,
               "t": self.tindeps,
               "d": self.indeps}[data[0]]

        return dat, ind

    def batchgen(self, bsize: int, data: str = "learning") -> np.ndarray:
        """Returns a generator that yields batches randomly from the
        specified dataset.

        :param bsize: specifies the size of the batches
        :param data: specifies the dataset (learning, testing or data)
        """
        tab = shuffle(self.table(data))
        tsize = len(tab[0])
        start = 0
        end = start + bsize

        while start < tsize:
            if end > tsize:
                end = tsize

            out = (tab[0][start:end], tab[1][start:end])

            start += bsize
            end += bsize

            yield out

    def reset_data(self, shuff, transform, params=None):

        def do_transformation(prm):
            if isinstance(transform, str):
                tr = transform.lower()[:5]
                if tr in ("std", "stand"):
                    self._transformation = self.self_standardize
                    if prm is not None:
                        print("Warning! Supplied parameters but chose standardization!\
                         Parameters are ignored!")
                elif tr in ("pca", "princ"):
                    assert prm is not None and isinstance(prm, int), \
                        "Please supply parameters for PCA like this: (no_factors: int, whiten: bool)"
                    self._transformation = lambda: self.fit_pca(no_factors=prm)
                elif tr in ("ae", "autoe"):
                    assert prm is not None and isinstance(prm, int), \
                        "Please supply the number of features for autoencoding!"
                    self._transformation = lambda: self.fit_autoencoder(no_features=prm)
                    self._transformation = self.fit_autoencoder
            self._transformation()

        n = self.data.shape[0]
        self.n_testing = int(n * self._crossval)
        if shuffle:
            dat, ind = shuffle((self.data, self.indeps))
        else:
            dat, ind = self.data, self.indeps
        self.learning = dat[self.n_testing:]
        self.lindeps = ind[self.n_testing:]
        self.testing = dat[:self.n_testing]
        self.tindeps = ind[:self.n_testing]
        self.N = self.learning.shape[0]
        if transform and self._transformation is not None:
            do_transformation(params)
        if not transform:
            self._transformation = None

    def fit_pca(self, no_factors: int=None):
        from sklearn.decomposition import PCA

        from csxnet.nputils import ravel_to_matrix as rtm

        if self._transformation is not None:
            print("Warning! Appliing transformation to already transformed data! This is untested!")
        self.learning = rtm(self.learning)
        if self._pca:
            raise Exception("Data already transformed by PCA!")
        if not no_factors or no_factors == "full":
            no_factors = self.learning.shape[-1]
            chain = ""
            if not no_factors:
                chain += "Number of factors is unspecified. "
            print(chain + "Assuming {} factors (full).".format(no_factors))

        self._pca = PCA(n_components=no_factors, whiten=True)
        self._pca.fit(self.learning)
        self.learning = self._pca.transform(self.learning)
        self.testing = self.pca(self.testing, no_factors)
        self._transformation = lambda: self.fit_pca(no_factors)

    def fit_autoencoder(self, no_features: int, epochs: int=5):
        from csxnet.high_utils import autoencode
        self.learning, self._autoencoder = autoencode(self.learning, no_features, epochs=epochs,
                                                      validation=self.testing, get_model=True)
        self.testing = self.autoencode(self.testing)
        self._transformation = lambda: self.fit_autoencoder(no_features=no_features, epochs=epochs)

    def self_standardize(self, no_factors: int=None):
        from csxnet.nputils import standardize
        del no_factors
        self.learning, mean, std = standardize(self.learning, return_factors=True)
        self._standardize_factors = mean, std
        if self._crossval > 0.0:
            self.testing = standardize(self.testing,
                                       mean=self._standardize_factors[0],
                                       std=self._standardize_factors[1])
        self._transformation = self.self_standardize

    def pca(self, X, no_features):
        if not self._pca:
            raise Exception("No PCA fitted to data! First apply fit_pca()!")
        X = np.copy(X)
        X = self._pca.transform(X)
        if X.shape[1] != no_features:
            X = X[..., :no_features]
        return X

    def autoencode(self, X):
        X = np.copy(X)
        if self._autoencoder is None:
            raise Exception("No autoencoder fitted to data! First apply fit_autoencoder()!")
        for weights, biases in self._autoencoder[0]:
            X = X.dot(weights) + biases
        return X

    def standardize(self, X):
        if self.pca or self._autoencoder:
            print("Data is transformed with {}!".format("PCA" if self.pca is not None else "autoencoder"))
        if not self._standardize_factors:
            print("No transformation applied to data! First apply standardize()!")
            return
        mean, std = self._standardize_factors
        return (X - mean) / std

    @property
    def neurons_required(self):
        return None

    @property
    def crossval(self):
        return self._crossval

    @crossval.setter
    def crossval(self, alpha):
        if alpha == 0:
            self._crossval = 0.0
        elif isinstance(alpha, int) and alpha == 1:
            print("Received an integer value of 1. Assuming 1 testing sample!")
            self._crossval = 1 / self.data.shape[0]
        elif isinstance(alpha, int) and alpha > 1:
            self._crossval = alpha / self.data.shape[0]
        elif isinstance(alpha, float) and 0.0 < alpha <= 1.0:
            self._crossval = alpha
        else:
            raise ValueError("Wrong value supplied! Give the ratio (0.0 <= alpha <= 1.0)\n" +
                             "or the number of samples to be used for validation!")
        self.reset_data(shuff=True, transform=True)


class CData(_Data):
    """
    This class is for holding categorical learning myData. The myData is read
    from the supplied source .csv semicolon-separated file. The file should
    contain a table of numbers with headers for columns.
    The elements must be of type integer or float.
    """

    def __init__(self, source, cross_val=.2, header=True, sep="\t", end="\n", pca=0,
                 standardize=False, autoencode=False):
        _Data.__init__(self, source, cross_val, 1, header, sep, end, standardize, pca, autoencode)
        self.reset_data(shuff=True, transform=True)

        # In categorical data, there is only 1 independent categorical variable
        # which is stored in a 1-tuple or 1 long vector. We free it from its misery
        if isinstance(self.indeps[0], tuple) or isinstance(self.indeps[0], np.ndarray):
            self.indeps = np.array([d[0] for d in self.indeps])
            self.lindeps = np.array([d[0] for d in self.lindeps])
            self.tindeps = np.array([d[0] for d in self.tindeps])

        self.categories = list(set(self.indeps))

        targets = np.zeros((len(self.categories), len(self.categories)))

        targets += NAY
        np.fill_diagonal(targets, YAY)

        targets = targets.astype(floatX)

        self._dictionary = {}
        self._dummycodes = {}

        for category, target in zip(self.categories, targets):
            self._dictionary[category] = target
            self._dummycodes[category] = self.categories.index(category)
            self._dummycodes[self.categories.index(category)] = category

        self.type = "classification"
        if self._transformation is not None:
            self._transformation()

    def table(self, data="learning", shuff=True):
        """Returns a learning table"""
        if shuff:
            datum, indep = shuffle(_Data.table(self, data))
        else:
            datum, indep = _Data.table(self, data)
        adep = np.array([self._dictionary[de] for de in indep])

        return datum, adep

    def translate(self, preds, dummy=False):
        """Translates a Brain's predictions to a human-readable answer"""

        if not dummy:
            out = np.array([self.categories[pred] for pred in preds])
        else:
            out = np.array([self._dummycodes[pred] for pred in preds])
        return out

    def dummycode(self, data="testing"):
        d = {"t": self.tindeps, "l": self.lindeps, "d": self.indeps
             }[data[0]][:len(self.tindeps)]
        return np.array([self._dummycodes[x] for x in d])

    @property
    def neurons_required(self):
        """Returns the required number of input and output neurons
         to process this myData.."""
        return self.learning.shape[1:], len(self.categories)

    def average_replications(self):
        if self._pca or self._standardize_factors or self._autoencoder:
            print("Warning! Data is transformed! This method resets your data!")
        replications = {}
        for i, indep in enumerate(self.indeps):
            if indep in replications:
                replications[indep].append(i)
            else:
                replications[indep] = [i]

        newindeps = np.fromiter(replications.keys(), dtype="<U4")
        newdata = {indep: np.mean(self.data[replicas], axis=0)
                   for indep, replicas in replications.items()}
        newdata = np.array([newdata[indep] for indep in newindeps])
        self.indeps = newindeps
        self.data = newdata
        self.reset_data(shuff=True, transform=True)


class RData(_Data):
    """
    Class for holding regression learning myData. The myData is read from the
    supplied source .csv semicolon-separated file. The file should contain
    a table of numbers with headers for columns.
    The elements must be of type integer or float.
    """

    def __init__(self, source, cross_val, indeps_n, header, sep=";", end="\n",
                 standardize=False, autoencode=0, pca=0):
        _Data.__init__(self, source, cross_val, indeps_n, header, sep, end,
                       standardize, pca, autoencode)

        self._indepscopy = np.copy(np.atleast_2d(self.indeps))

        self.type = "regression"
        self._downscaled = False

        # Calculate the scaling factors for the target values and store them as
        # (a, b). Every target value is scaled by ax + b.
        self._oldfctrs = None
        self._newfctrs = None

        self.reset_data()
        self.indeps = np.atleast_2d(self.indeps)

    def reset_data(self, shuff=True, transform=True, params=None):
        _Data.reset_data(self, shuff, transform, params)
        if not self._downscaled:
            from .nputils import featscale
            self.lindeps, self._oldfctrs, self._newfctrs = \
                featscale(self.lindeps, axis=0, ufctr=(0.1, 0.9), getfctrs=True)
            self._downscaled = True
            self.tindeps = self.downscale(self.tindeps)
        self.indeps = self.indeps.astype(floatX)
        self.lindeps = self.lindeps.astype(floatX)
        self.tindeps = self.tindeps.astype(floatX)

    def _scale(self, A, where):
        def sanitize():
            assert self._downscaled, "Scaling factors not yet set!"
            assert where in ("up", "down"), "Something is very weird here..."
            if where == "up":
                return self._newfctrs, self._oldfctrs
            else:
                return self._oldfctrs, self._newfctrs

        from .nputils import featscale
        fctr_list = sanitize()
        return featscale(A, axis=0, dfctr=fctr_list[0], ufctr=fctr_list[1])

    def upscale(self, A):
        return self._scale(A, "up")

    def downscale(self, A):
        return self._scale(A, "down")

    @property
    def neurons_required(self):
        return self.learning[0].shape, self.lindeps.shape[1]


class Sequence:
    def __init__(self, source):
        self._raw = source
        self._vocabulary = dict()
        self.data = None
        self.embedded = False
        self.tokenized = False
        self.N = len(self._raw)

    def embed(self, dims):
        assert not (self.tokenized or self.embedded)
        self._encode("embed", dims)
        self.embedded = True

    def tokenize(self):
        assert not (self.tokenized or self.embedded)
        self._encode("tokenize")
        self.tokenized = True

    def _encode(self, how, dims=0):
        symbols = list(set(self._raw))
        if how == "tokenize":
            embedding = np.eye(len(symbols), len(symbols))
        elif how == "embed":
            assert dims, "Dims unspecified!"
            embedding = np.random.random((len(symbols), dims)).astype(floatX)
        else:
            raise RuntimeError("Something is not right!")
        self._vocabulary = dict(zip(symbols, embedding))
        self.data = np.array([self._vocabulary[x] for x in self._raw])

    def table(self):
        return ([word[1:] for word in self._raw],
                [word[:-1] for word in self._raw])

    def batchgen(self, size):
        assert self.embedded ^ self.tokenized
        for step in range(self.data.shape[0] // size):
            start = step * size
            end = start + size
            if end > self.data.shape[0]:
                end = self.data.shape[0]
            if start >= self.data.shape[0]:
                break

            sentence = self.data[start:end]

            yield sentence[:-1], sentence[1:]

    def neurons_required(self):
        return self.data.shape[-1], self.data.shape[-1]


class Text(Sequence):
    def __init__(self, source, limit=8000, vocabulary=None,
                 embed=False, embeddim=0, tokenize=False):
        Sequence.__init__(self, source, embed, embeddim, tokenize)
        self._raw = source
        self._tokenized = False
        self._embedded = False
        self._util_tokens = {"unk": "<UNK>",
                             "start": "<START>",
                             "end": "<END>"}
        self._dictionary = dict()
        self._vocabulary = vocabulary if vocabulary else {}

        self.data = None

        if embed and tokenize:
            raise RuntimeError("Please choose either embedding or tokenization, not both!")

        if embed or tokenize:
            if embed and not embeddim:
                print("Warning! Embedding vector dimension unspecified, assuming 5!")
                embeddim = 5
            self.initialize(vocabulary, limit, tokenize, embed, embeddim)

    def neurons_required(self):
        return self.data.shape[0]  # TODO: is this right??

    def initialize(self,
                   vocabulary: dict = None,
                   vlimit: int = None,
                   tokenize: bool = True,
                   embed: bool = False,
                   embeddim: int = 5):

        def accept_vocabulary():
            example = list(vocabulary.keys())[0]
            dtype = example.dtype
            emb = example.shape[-1]
            if "float" in dtype:
                self._embedded = True
                self._tokenized = False
            elif "int" in dtype:
                self._embedded = False
                self._tokenized = True
            else:
                raise RuntimeError("Wrong vocabulary format!")
            self._vocabulary = vocabulary
            self.data = build_learning_data(emb)

        def ask_for_input():
            v = None
            while 1:
                v = input("Please select one:\n(E)mbed\n(T)okenize\n> ")
                if v[0].lower() in "et":
                    break
            return v == "t", v == "e"

        def build_vocabulary():
            words = dict()
            for sentence in self._raw:
                for word in sentence:
                    if word in words:
                        words[word] += 1
                    else:
                        words[word] = 1
            words = [word for word in sorted(list(words.items()), key=lambda x: x[1], reverse=True)][:vlimit]

            if tokenize:
                emb = np.eye(len(words), len(words))
            else:
                emb = np.random.random((len(words), embeddim))

            voc = {word: array for word, array in zip(words, emb)}
            dic = {i: word for i, word in enumerate(words)}

            return voc, dic

        def build_learning_data(emb):
            data = []
            for i, sentence in enumerate(self._raw):
                s = []
                for word in sentence:
                    if word in self._vocabulary:
                        s.append(self._vocabulary[word])
                    else:
                        s.append(UNKNOWN)
                data.append(np.array(s))
            return data

        if vocabulary:
            accept_vocabulary()
            return

        if not (tokenize or embed) or (tokenize and embed):
            tokenize, embed = ask_for_input()

        self._vocabulary, self._dictionary = build_vocabulary()

        embdim = embeddim if embed else vlimit
        self.data = build_learning_data(embdim)

    def batchgen(self, size=None):
        sentences = np.copy(self.data)
        np.random.shuffle(sentences)
        for sentence in sentences:
            yield sentence[:-1], sentence[1:]


class Text2:
    def __init__(self, raw: str, dictionary: dict, embed=0):
        self._raw = raw
        self._dictionary = dictionary
        self._emb = embed
        self._keys = None
        self._values = None

        self.data = [self._dictionary[word] for word in self._raw]

    @classmethod
    def characterwise(cls, chain: str, vocabulary=None, embed=0):
        if vocabulary is None:
            vocabulary = ["<WORD_START>"] + list(set(chain)) + ["<WORD_END>"]
        if embed:
            print("Embedding characters into {} dimensional space!".format(embed))
            embedding = sorted(np.random.randn(len(vocabulary), embed).astype(floatX),
                               key=lambda x: x.sum())
        else:
            print("Tokenizing characters...")
            embedding = np.eye(len(vocabulary)).astype(floatX)
        embedding = dict(zip(sorted(vocabulary), embedding))
        return Text2(chain, embedding)

    @classmethod
    def wordwise_NotImplemented(cls, chain: str, vocabulary=None, embed=0):
        raise NotImplementedError("Factory method not yet implemented!")
        if vocabulary is None:
            vocabulary = ["<SENTECE_START>"] + list(set(chain)) + ["<SENTECE_END>"]
        if embed:
            print("Embedding words into {} dimensional space!".format(embed))
            embedding = np.random.randn(len(vocabulary), embed).astype(floatX), vocabulary
        else:
            print("Tokenizing words...")
            embedding = np.eye(len(vocabulary)).astype(floatX)
        embedding = dict(zip(vocabulary, embedding))
        return Text2(chain, embedding)

    @property
    def neurons_required(self):
        embeddim = self._dictionary[0].shape[1]
        return embeddim, embeddim

    @property
    def encoding(self):
        embdim = self._dictionary[0].shape[1]
        return "Embedding ({})".format(embdim) if self._emb else "Tokenization ({})".format(embdim)

    def table(self):
        return [ch for ch in self.data[:-1]], [ch for ch in self.data[1:]]

    def translate(self, output):
        def from_tokenization():
            if self._keys is None:
                keys, values = list(zip(*sorted(
                    [(k, np.argmax(v)) for k, v in self._dictionary.items], key=lambda x: x[1]
                )))
                self._keys = keys
            return self._keys[np.argmax(output)]

        def from_embedding():
            from csxnet.nputils import euclidean
            if self._keys is None or self._values is None:
                self._keys, self._values = list(zip(*list(self._dictionary.items())))
            return self._keys[np.argmin(
                [euclidean([output for _ in range(len(self._values))], self._values)]
            )]

        output = from_embedding() if self.encoding.lower()[0] == "e" else from_tokenization()
        return output


def parsecsv(source: str, header: int, indeps_n: int, sep: str, end: str):
    file = open(source, encoding='utf8')
    text = file.read()
    text = text.replace(",", ".")
    file.close()
    assert sep in text and end in text, \
        "Separator or Endline character is not present in the file!"

    lines = text.split(end)

    if header:
        headers = lines[0]
        headers = headers.split(sep)
        lines = lines[1:-1]
    else:
        lines = lines[:-1]
        headers = None

    lines = np.array([line.split(sep) for line in lines])
    indeps = lines[..., :indeps_n]
    data = lines[..., indeps_n:].astype(floatX)

    return headers, data, indeps


def parsearray(X: np.ndarray, header: int, indeps_n: int):
    headers = X[0] if header else None
    matrix = X[1:] if header else X
    indeps = matrix[:, :indeps_n]
    data = matrix[:, indeps_n:]
    return headers, data, indeps


def parselearningtable(source, coding=None):
    if isinstance(source, str) and source[-7:] == ".pkl.gz":
        source = unpickle_gzip(source, coding)
    if not isinstance(source, tuple):
        raise RuntimeError("Please supply a learning table (tuple) or a *lt.pkl.gz file!")
    if source[0].dtype != floatX:
        print("Warning! dtype differences in datamodel.parselearningtable()! Casting...")
        source = source[0].astype(floatX), source[1]

    return None, source[0], source[1]


def unpickle_gzip(source: str, coding='latin1'):
    import pickle
    import gzip

    f = gzip.open(source)
    if coding:
        with f:
            u = pickle._Unpickler(f)
            u.encoding = coding
            tup = u.load()
        f.close()
    else:
        tup = pickle.load(f)
    return tup


def mnist_to_lt(source: str, reshape=True):
    """The reason of this method's existance is that I'm lazy as ..."""
    tup = unpickle_gzip(source, coding="latin1")
    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype(floatX, copy=False)
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    if reshape:
        questions = questions.reshape((questions.shape[0], 1, 28, 28))
    return questions, targets


def shuffle(learning_table: tuple):
    """Shuffles and recreates the learning table"""
    indices = np.arange(learning_table[0].shape[0])
    np.random.shuffle(indices)
    return learning_table[0][indices], learning_table[1][indices]
