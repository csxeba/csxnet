import numpy as np


YAY = 1.0
NAY = 0.0
REAL = np.float64


class _Data:
    """Base class for Data Wrappers"""
    def __init__(self, source, cross_val, indeps_n, header, sep, end, pca):
        self.learning = None
        self.testing = None
        self.lindeps = None
        self.tindeps = None
        self.pca = None
        self.N = 0
        self.type = None

        if isinstance(source, np.ndarray):
            headers, data, indeps = parsearray(source, header, indeps_n)
        elif isinstance(source, tuple):
            headers, data, indeps = parselearningtable(source)
        elif isinstance(source, str) and "lt.pkl.gz" in source:
            headers, data, indeps = parselearningtable(source)
        else:
            headers, data, indeps = parsecsv(source, header, indeps_n, sep, end)

        self.headers = headers
        self.data, self.indeps = data, indeps
        self._datacopy = data
        self.n_testing = int(len(data) * (1 - cross_val))

        if pca:
            self.do_pca(pca)

        self.split_data()

    def table(self, data):
        """Returns a learning table"""
        dat = {"l": self.learning,
               "t": self.testing,
               "d": self.data}[data[0]]
        ind = {"l": self.lindeps,
               "t": self.tindeps,
               "d": self.data}[data[0]]

        # I wasn't sure if the order gets messed up or not, but it seems it isn't
        # Might be wise to implement thisas a test method, because shuffling
        # transposed myData might lead to surprises
        return dat, ind

    def batchgen(self, bsize, data="learning"):
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

    def do_pca(self, no_factors):
        self.data = self.data.reshape(self.data.shape[0], np.prod(self.data.shape[1:]))
        if no_factors is "full":
            print("Doing full PCA. No features:", self.data.shape[1])
            no_factors = self.data.shape[1]
        if not self.pca:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=no_factors, whiten=True)
            self.pca.fit(self.data)

        self.data = self.pca.transform(self.data)
        self.split_data()

    def restore(self):
        self.data = self._datacopy
        self.split_data()

    def split_data(self):
        dat, ind = shuffle((self.data, self.indeps))
        self.learning = dat[:self.n_testing]
        self.lindeps = ind[:self.n_testing]
        self.testing = dat[self.n_testing:]
        self.tindeps = ind[self.n_testing:]
        self.N = self.learning.shape[0]


class CData(_Data):
    """
    This class is for holding categorical learning myData. The myData is read
    from the supplied source .csv semicolon-separated file. The file should
    contain a table of numbers with headers for columns.
    The elements must be of type integer or float.
    """

    def __init__(self, source, cross_val=.2, header=True, sep="\t", end="\n", pca=0):
        _Data.__init__(self, source, cross_val, 1, header, sep, end, pca)

        # In categorical myData, there is only 1 independent categorical variable
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

        # A dictionary is created, which associates the category names to the target
        # arrays representing them.
        self.dictionary = {category: target for category, target in
                           zip(self.categories, targets)}
        self.type = "classification"

    def table(self, data="learning", shuff=True):
        """Returns a learning table"""
        if shuff:
            datum, indep = shuffle(_Data.table(self, data))
        else:
            datum, indep = _Data.table(self, data)
        adep = np.array([self.dictionary[de] for de in indep])

        return datum, adep

    def translate(self, output):
        """Translates a Brain's output to a human-readable answer"""
        output = output.ravel().tolist()
        answer = output.index(max(output))
        return self.categories[answer]

    def dummycode(self, data="testing"):
        d = {"t": self.tindeps, "l": self.lindeps, "d": self.indeps
             }[data[0]][:len(self.tindeps)]
        return np.array([self.categories.index(x) for x in d])

    def neurons_required(self):
        """Returns the required number of input and output neurons
         to process this myData.."""
        return self.data[0].shape, len(self.categories)

    def test(self, brain, data="testing"):
        """Test a brain against the myData.
        Returns the rate of rights answers"""
        print("Warning! CData.test() is OBSOLETE!")
        ttable = self.table(data)
        thoughts = brain.think(ttable[0])
        answers = [self.translate(thought) for thought in thoughts]
        targets = [self.translate(target) for target in ttable[1]]
        results = [int(x == y) for x, y in zip(answers, targets)]
        return sum(results) / len(results)


class RData(_Data):
    def __init__(self, source, cross_val, indeps_n, header, sep=";", end="\n", pca=0):
        """
        Class for holding regression learning myData. The myData is read from the
        supplied source .csv semicolon-separated file. The file should contain
        a table of numbers with headers for columns.
        The elements must be of type integer or float.
        """
        _Data.__init__(self, source, cross_val, indeps_n, header, sep, end, pca)
        self.type = "regression"

    def split_data(self):
        self.indeps /= np.max(self.indeps, axis=0) * 2
        _Data.split_data(self)
        self.indeps = self.indeps.astype(REAL)
        self.lindeps = self.lindeps.astype(REAL)
        self.tindeps = self.tindeps.astype(REAL)

    def neurons_required(self):
        return self.data[0].shape, self.indeps.shape[1]

    def test(self, brain, data="testing", queue=None):
        ttable = self.table(data)
        thoughts = brain.think(ttable[0])

        assert ttable[1].shape == thoughts.shape, "Targets' shape differ from brain output!"

        if not queue:
            return np.average(np.sqrt(np.square(ttable[1] - thoughts)))
        queue.put(brain)


def parsecsv(source: str, header: int, indeps_n: int, sep: str, end: str):
    file = open(source, encoding='utf8')  # open mind! :D
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
        print("Warning! dtype differences in datamodel.parselearningtable()!")

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


def mnist_to_lt(source: str):
    """The reason of this method's existance is that I'm lazy as ..."""
    tup = unpickle_gzip(source, coding="latin1")
    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype(REAL, copy=False)
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    return questions, targets


def shuffle(learning_table: tuple):
    """Shuffles and recreates the learning table"""
    indices = np.arange(learning_table[0].shape[0])
    np.random.shuffle(indices)
    half = len(indices) // 2
    return learning_table[0][indices], learning_table[1][indices]
