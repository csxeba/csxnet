"""My crazy expreminets are taking place here"""

from model import Network
from ..Layerdef import _LayerBase


class Abomination:
    """
    Experiment #1

    Codename: ABOMINATION

    Description:
    Constructing a neural-level simulated brain and one of its layers
    will be built from complete nerual networks instead of neurons...
    """

    def __init__(self, data, eta, lmbd, cost):
        self.data = data
        self.eta = eta
        self.lmbd = lmbd
        self.cost = cost
        self.layers = []


class AboLayer(_LayerBase):
    def __init__(self, brain: Abomination, no_minions):
        self.brain = brain
        self.fanin = brain.layers[-1].fanout
        self.neurons = []

        for pos in range(no_minions):
            self.neurons.append(self._forge_minion())

    def _forge_minion(self):
        minion = Network(self.brain.data, self.brain.eta, self.brain.lmbd, self.brain.cost)
        minion.add_fc(10)
        minion.finalize_architecture()
        return minion

    def feedforward(self, inputs):
        """this ain't so simple after all O.O"""
        pass
