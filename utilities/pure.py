"""Pure python and python stdlib based utilities are here.
This module aims to be PyPy compatible."""


class _Roots:
    def __init__(self):
        import sys
        win = sys.platform == "win32"
        self.dataroot = "D:/Data/" if win else "/data/Prog/data/"
        self.miscroot = self.dataroot + "misc/"
        self.rawroot = self.dataroot + "raw/"
        self.ltroot = self.dataroot + "lts/"
        self.csvroot = self.dataroot + "csvs/"
        self.nirroot = self.rawroot + "nir/"
        self.tmproot = "E:/tmp/" if win else "/run/media/csa/ramdisk/"
        self.hippocrates = self.rawroot + "Project_Hippocrates/"

        self._dict = {"data": self.dataroot,
                      "raw": self.rawroot,
                      "lt": self.ltroot,
                      "lts": self.ltroot,
                      "csv": self.csvroot,
                      "csvs": self.csvroot,
                      "nir": self.nirroot,
                      "tmp": self.tmproot,
                      "temp": self.tmproot,
                      "misc": self.miscroot,
                      "hippocrates": self.hippocrates,
                      "hippo": self.hippocrates}

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        if item not in self._dict:
            raise IndexError("Requested path not in database!")
        return self._dict[item]

    def __call__(self, item):
        return self.__getitem__(item)

roots = _Roots()


def euclidean(itr, target):
    import math
    assert len(itr) == len(target), "Can't perform distance calculation"
    res = math.sqrt(sum([(itr[i]-target[i])**2 for i in range(len(itr))]))
    return res


def chooseN(iterable: list, N=1):
    """Choose N elements randomly from an iterable and remove the element"""
    return [choose(iterable) for _ in range(N)]  #TODO: untested


def choose(iterable: list):
    """Chooses an element randomly from a list, then removes it from the list"""
    import random
    out = random.choice(iterable)
    iterable.remove(out)
    return out  # TODO: untested


def feature_scale(iterable, from_=0, to=1):
    """Scales the elements of a vector between from_ and to uniformly"""
    # TODO: untested
    if max(iterable) + min(iterable) == 0:
        # print("Feature scale warning: every value is 0 in iterable!")
        return type(iterable)([from_ for _ in range(len(iterable))])

    out = []
    for e in iterable:
        try:
            x = ((e - min(iterable)) / (max(iterable) - min(iterable)) * (to - from_)) + from_
        except ZeroDivisionError:
            x = 0
        out.append(x)
    return type(iterable)(out)


def avg(iterable):
    return sum(iterable) / len(iterable)


def pull_table(path, header=True, labels=False, sep="\t", end="\n"):
    """Extracts a data table from a file

    Returns data, header labels"""
    with open(path) as f:
        text = f.read()
        f.close()
    assert sep in text and end in text, "Separator or Endline character not present in file!"
    if "," in text:
        print("Warning! Replacing every ',' character with '.'!")
        text = text.replace(",", ".")
    lines = [l.split(sep) for l in text.split(end) if l]
    if header:
        header, lines = lines[0], lines[1:]
    else:
        header = None
    if labels:
        labels = [ln[0] for ln in lines]
        lines = [[float(d) for d in ln[1:]] for ln in lines]
    else:
        labels = None
        lines = [[float(d) for d in ln] for ln in lines]

    return lines, header, labels


def l1term(eta, lmbd, N):
    return (eta * lmbd) / N


def l2term(eta, lmbd, N):
    return 1 - ((eta * lmbd) / N)