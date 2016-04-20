"""My crazy expreminets are taking place here"""


class Abomination:
    """
    Experiment #1

    Codename: ABOMINATION

    Description:
    I am constructiong a neural-level simulated brain and one of its layers
    will be built from complete nerual networks instead of neurons...
    """

    def __init__(self, fanin, abolayer, outshape):
        pass


class AboLayer:
    def __init__(self, brain, minionlayout, no_minions):
        self.fanin = brain.layers[-1].fanout
        self.fanout = no_minions * minionlayout[-1]
        self.neurons = []