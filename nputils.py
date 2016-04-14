import numpy as np


def featscale(X: np.ndarray, axis=0, ufctr=(0, 1), dfctr=None, getfctrs=False):
    if dfctr is None:
        dfctr = (X.min(axis=axis), X.max(axis=axis))
    frm, to = ufctr
    X -= dfctr[0]
    X /= dfctr[1] - dfctr[0]
    X *= (to - frm)
    X += frm
    if not getfctrs:
        return X
    else:
        return X, dfctr, (frm, to)


def euclidean(itr: np.ndarray, target: np.ndarray):
    """Distance of two (or more) vectors in euclidean space"""
    print("Warning, nputils.euclidean() is untested!")
    return np.sqrt(np.square(np.sum(itr - target, axis=0)))


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


