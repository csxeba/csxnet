import numpy as np


def save(obj, filename='autosave.bro'):
    """Saves the brain/model/population to a .bro file in text format"""
    import pickle
    fl = open(filename, mode="wb")
    pickle.dump(obj, fl)
    fl.close()


def load(filename):
    """Loads a brain/model/population object from a previously saved file"""
    if not filename:
        import tkinter as tk
        import tkinter.filedialog as tkfd
        root = tk.Tk()
        filename = tkfd.askopenfilename(filetypes=(
            ("Brain objects", "*.bro"),
            ("All files", "*.*")))
        root.destroy()
        del root, tkfd, tk

    fl = open(filename, mode="rb")
    import pickle
    obj = pickle.load(fl)
    fl.close()
    return obj


def outshape(inshape: tuple, fshape: tuple, stride: int):
    """Calculates the shape of an output matrix if a filter of shape
    <fshape> gets slided along a matrix of shape <inshape> with a
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

    startxes = np.array(range(xsteps)) * stride
    startys = np.array(range(ysteps)) * stride

    endxes = startxes + fshape[1]
    endys = startys + fshape[2]

    coords = []

    for sy, ey in zip(startys, endys):
        for sx, ex in zip(startxes, endxes):
            coords.append((sx, ex, sy, ey))

    return tuple(coords)


def ravel_to_matrix(A):
    A = np.atleast_2d(A)
    A = A.reshape(A.shape[0], np.prod(A.shape[1:]))
    return A


def l2term(eta, lmbd, N):
    return 1 - ((eta * lmbd) / N)
