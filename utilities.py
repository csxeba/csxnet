import random

# In case of PyPy
try:
    import numpy as np
except ImportError:
    np = None


def euclidean(itr, target):
    import math
    assert len(itr) == len(target), "Can't perform distance calculation"
    res = math.sqrt(sum([(itr[i]-target[i])**2 for i in range(len(itr))]))
    return res


def haversine(coords1: np.ndarray, coords2: np.ndarray):
    """The distance of two points on Earth given their GPS coords"""
    R = 6367  # radius of Mother Earth in kms
    np.radians(coords1, out=coords1)
    np.radians(coords2, out=coords2)
    lon1, lat1 = coords1[..., 0], coords1[..., 1]
    lon2, lat2 = coords2[..., 0], coords2[..., 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    e = 2 * np.arcsin(np.sqrt(d))
    return e * R


def pull(lst: list, key: callable):
    """Filters elements, than returns (trues, falses)"""
    trues = list(filter(key, lst))
    falses = [elem for elem in lst if elem not in trues]
    return trues, falses
    

def chooseN(iterable, N=1):
    """Choose an element randomly from an iterable and remove the element"""
    return [choose(iterable) for _ in range(N)]


def choose(iterable):
    out = random.choice(iterable)
    iterable.remove(out)
    return out


def feature_scale(iterable):
    """Scales the elements of a vector between 0 and 1"""
    if not any(iterable):
        raise RuntimeError("Every fitness is 0...")
    out = []
    for e in iterable:
        try:
            x = (e - min(iterable)) / (max(iterable) - min(iterable))
        except ZeroDivisionError:
            x = 0
        out.append(x)
    return out


def avg(iterable):
    return sum(iterable) / len(iterable)


def plot(*lsts):
    import matplotlib.pyplot as plt
    for fn, lst in enumerate(lsts):
        plt.subplot(len(lsts), 1, fn + 1)
        plt.plot(lst)
    plt.show()
