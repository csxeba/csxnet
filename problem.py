import sys
from tkinter import *
from tkinter.filedialog import askopenfilename

from datamodel import CData, RData

DATAROOT = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
LTDIR = "learning_tables/"


class Problem(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.data = None
        self.network = None
        self.type = "classification"

    def open_learning_table(self):
        import gzip
        import pickle

        path = askopenfilename(initialdir=DATAROOT+LTDIR, filetypes=[("Learning tables", ".pkl.gz")])
        fl = gzip.open(path, mode="rb")
        dataclass = {"cla": CData, "reg": RData}[self.type[:3]]
        self.data = dataclass(pickle.load(fl))

    def build_gui(self):
        pass
