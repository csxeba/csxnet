import random


def euclidean(itr, target):
    import math
    assert len(itr) == len(target), "Can't perform distance calculation"
    res = math.sqrt(sum([(itr[i]-target[i])**2 for i in range(len(itr))]))
    return res


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


def feature_scale(iterable, from_=0, to=1):
    """Scales the elements of a vector between from_ and to"""
    if max(iterable) + min(iterable) == 0:
        print("Feature scale warning: every value is 0 in iterable!")
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


def plot(*lsts):
    import matplotlib.pyplot as plt
    for fn, lst in enumerate(lsts):
        plt.subplot(len(lsts), 1, fn + 1)
        plt.plot(lst)
    plt.show()
