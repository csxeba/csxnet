import numpy as np


def outshape(inshape: tuple, fshape: tuple, stride: int):
    """Calculates the shape of an output matrix if a filter of shape
    <fshape> gets slided along a matrix of shape <fanin> with a
    stride of <stride>.
    Returns x, y sizes of the output matrix"""
    output = [int((x - ins) / stride) + 1 if (x - ins) % stride == 0 else "NaN"
              for x, ins in zip(inshape[1:3], fshape[1:3])]
    if "NaN" in output:
        raise RuntimeError("Shapes not compatible!")
    return tuple(output)


def calcsteps(inshape: tuple, fshape: tuple, stride: int):
    """Calculates the coordinates required to slide
    a filter of shape <fshape> along a matrix of shape <inshape>
    with a stride of <stride>.
    Returns a list of coordinates"""
    xsteps, ysteps = outshape(inshape, fshape, stride)

    startxes = np.arange(xsteps) * stride
    startys = np.arange(ysteps) * stride

    endxes = startxes + fshape[1]
    endys = startys + fshape[2]

    coords = []

    for sy, ey in zip(startys, endys):
        for sx, ex in zip(startxes, endxes):
            coords.append((sx, ex, sy, ey))

    return tuple(coords)


def l1term(eta, lmbd, N):
    return (eta * lmbd) / N


def l2term(eta, lmbd, N):
    return 1 - ((eta * lmbd) / N)
