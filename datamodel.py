import numpy as np

YAY = 1.0
NAY = 0.0
REAL = np.float32

UNKNOWN = "<UNK>"


class _Data:
    """Base class for Data Wrappers"""

    def __init__(self, source, cross_val, indeps_n, header, sep, end):

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

        self.learning = None
        self.testing = None
        self.lindeps = None
        self.tindeps = None
        self.pca = None
        self.type = None

        self.N = 0
        self.mean = 0
        self.std = 0

        self.standardized = False

        headers, data, indeps = parse_source()
        self.n_testing = determine_no_testing()

        self.headers = headers
        self.data, self.indeps = data, indeps

    def table(self, data):
        """Returns a learning table"""
        dat = {"l": self.learning,
               "t": self.testing,
               "d": self.data}[data[0]]
        ind = {"l": self.lindeps,
               "t": self.tindeps,
               "d": self.indeps}[data[0]]

        # I wasn't sure if the order gets messed up or not, but it seems it isn't
        # Might be wise to implement this as a test method, because shuffling
        # transposed data might lead to surprises
        return dat, ind

    def batchgen(self, bsize: int, data: str="learning") -> np.ndarray:
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

    def split_data(self, shuff=True):
        if shuffle:
            dat, ind = shuffle((self.data, self.indeps))
        else:
            dat, ind = self.data, self.indeps
        self.learning = dat[self.n_testing:]
        self.lindeps = ind[self.n_testing:]
        self.testing = dat[:self.n_testing]
        self.tindeps = ind[:self.n_testing]
        self.N = self.learning.shape[0]

    def do_pca(self, no_factors):
        self.data = self.data.reshape(self.data.shape[0], np.prod(self.data.shape[1:]))
        if self.pca or self.standardized:
            print("Ignoring attempt to PCA transform already PCA'd or standardized data.")
            return
        if no_factors is "full":
            print("Doing full PCA. No features:", self.data.shape[1])
            no_factors = self.data.shape[1]
        if not self.pca:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=no_factors, whiten=True)
            self.pca.fit(self.data)

            self.data = self.pca.transform(self.data)

    def standardize(self):
        if self.standardized or self.pca:
            print("Ignoring attempt to standardize already standardized or PCA'd data!")
            return
        self.data += 0.1
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.data -= self.mean
        self.data /= self.std
        # self.data = np.nan_to_num(self.data)
        self.standardized = True

    def neurons_required(self):
        pass


class CData(_Data):
    """
    This class is for holding categorical learning myData. The myData is read
    from the supplied source .csv semicolon-separated file. The file should
    contain a table of numbers with headers for columns.
    The elements must be of type integer or float.
    """

    def __init__(self, source, cross_val=.2, header=True, sep="\t", end="\n", pca=0):
        _Data.__init__(self, source, cross_val, 1, header, sep, end)

        if pca:
            self.do_pca(pca)
        self.split_data()

        # In categorical data, there is only 1 independent categorical variable
        # which is stored in a 1-tuple or 1 long vector. We free it from its misery
        if isinstance(self.indeps[0], tuple) or isinstance(self.indeps[0], np.ndarray):
            self.indeps = np.array([d[0] for d in self.indeps])
            self.lindeps = np.array([d[0] for d in self.lindeps])
            self.tindeps = np.array([d[0] for d in self.tindeps])

        # We extract the set of categories. set() removes duplicate values.
        self.categories = list(set(self.indeps))

        # Every category gets associated with a y long array, where y is the
        # number of categories. Every array gets filled with zeros.
        targets = np.zeros((len(self.categories),
                            len(self.categories)))

        # The respective element of the array, corresponding to the associated
        # category is set to YAY. Thus if 10 categories are present, then category
        # No. 3 is represented by the following array:
        # NAY, NAY, YAY, NAY, NAY, NAY, NAY, NAY, NAY, NAY
        targets += NAY
        np.fill_diagonal(targets, YAY)

        targets = targets.astype(REAL)

        self._dictionary = {}
        self._dummycodes = {}

        for category, target in zip(self.categories, targets):
            self._dictionary[category] = target
            self._dummycodes[category] = self.categories.index(category)
            self._dummycodes[self.categories.index(category)] = category

        self.type = "classification"

    def table(self, data="learning", shuff=True):
        """Returns a learning table"""
        if shuff:
            datum, indep = shuffle(_Data.table(self, data))
        else:
            datum, indep = _Data.table(self, data)
        adep = np.array([self._dictionary[de] for de in indep])

        return datum, adep

    def do_pca(self, no_factors):
        _Data.do_pca(self, no_factors)
        self.split_data(shuff=True)

    def standardize(self):
        _Data.standardize(self)
        self.split_data()

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

    def neurons_required(self):
        """Returns the required number of input and output neurons
         to process this myData.."""
        return self.data[0].shape, len(self.categories)


class RData(_Data):
    """
    Class for holding regression learning myData. The myData is read from the
    supplied source .csv semicolon-separated file. The file should contain
    a table of numbers with headers for columns.
    The elements must be of type integer or float.
    """

    def __init__(self, source, cross_val, indeps_n, header, sep=";", end="\n", pca=0):
        _Data.__init__(self, source, cross_val, indeps_n, header, sep, end)

        self._indepscopy = np.copy(np.atleast_2d(self.indeps))

        self.type = "regression"
        self._downscaled = False

        # Calculate the scaling factors for the target values and store them as
        # (a, b). Every target value is scaled by ax + b.
        self._oldfctrs = None
        self._newfctrs = None

        if pca:
            self.do_pca(pca)
        self.indeps = np.atleast_2d(self.indeps)
        self.split_data()

    def split_data(self, shuff=True):
        if not self._downscaled:
            from .nputils import featscale
            self.indeps, self._oldfctrs, self._newfctrs = \
                featscale(self.indeps, axis=0, ufctr=(0.1, 0.9), getfctrs=True)
            self._downscaled = True
        _Data.split_data(self, shuff)
        self.indeps = self.indeps.astype(REAL)
        self.lindeps = self.lindeps.astype(REAL)
        self.tindeps = self.tindeps.astype(REAL)

    def do_pca(self, no_factors):
        _Data.do_pca(self, no_factors)
        self.split_data(shuff=True)

    def standardize(self):
        _Data.standardize(self)
        self.split_data(shuff=True)

    def upscale(self, A):
        from .nputils import featscale
        if not self._downscaled:
            return A
        else:
            return featscale(A, axis=0, dfctr=self._newfctrs, ufctr=self._oldfctrs)

    def downscale(self, A):
        from .nputils import featscale
        if not self._downscaled:
            return A
        else:
            return featscale(A, axis=0, dfctr=self._oldfctrs, ufctr=self._newfctrs)

    def neurons_required(self):
        return self.data[0].shape, self.indeps.shape[1]


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
            embedding = np.random.random((len(symbols), dims))
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
                   vocabulary: dict=None,
                   vlimit: int=None,
                   tokenize: bool=True,
                   embed: bool=False,
                   embeddim: int=5):

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


def parsecsv(source: str, header: int, indeps_n: int, sep: str, end: str):
    file = open(source, encoding='utf8')
    text = file.read()
    text.replace(",", ".")
    file.close()

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
    data = lines[..., indeps_n:].astype(REAL)

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
    if source[0].dtype != REAL:
        print("Warning! dtype differences in datamodel.parselearningtable()! Casting...")
        source = source[0].astype(REAL), source[1]

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
    questions = questions.astype(REAL, copy=False)
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    if reshape:
        questions = questions.reshape((questions.shape[0], 1, 28, 28))
    return questions, targets


def shuffle(learning_table: tuple):
    """Shuffles and recreates the learning table"""
    indices = np.arange(learning_table[0].shape[0])
    np.random.shuffle(indices)
    return learning_table[0][indices], learning_table[1][indices]
