"""
Neural Networking and Evolutionary Algorithms Library and Wrappers

Copyright (c) 2015 Csaba Gor

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import Architecture.FFNN as FFNN
from Utility.DataModel import CData
from mylibs.evolution.model import Population

# Evolution hyperparameters
LIMIT = 100
SELECTION = .6
CROSSING_OVER = .2
MUTATION = .2
MAX_OFFSPRINGS = 3

# These FFNN hyperparameters will be evolved...
# They are:  learning rate:  (0.1,2.0,0.1)
#            repeats:        (100,500,50)
#            hidden layers:  (1,3,1)
#            hidden neurons: (5,30,1)

ETA_RANGE = (0.01, 1.0)
LAMBDA_RANGE = (0.1, 1.0)
REPEATS_RANGE = (300, 1000, 50)
HIDDENS_RANGE = (20, 100, 1)

RANGE = [ETA_RANGE,
         LAMBDA_RANGE,
         REPEATS_RANGE,
         HIDDENS_RANGE]


def fitness_func(ind, queue=None):
    if ind.fitness:
        print("Warning! Recalculation of an existing fitness value!")

    data = CData("TestData/Dohany_ANN/full.csv", cross_val=.3, header=True, sep=";")
    repeats = ind.genome[2]
    ind.phenotype = get_brain(ind.genome, data)

    for epoch in range(repeats):
        ind.phenotype.learn(batch_size=20)

    fitness = ind.phenotype.evaluate()
    ind.fitness = fitness

    # If parallelization is involved
    if queue:
        queue.put(ind)


def get_brain(genome, data):
    hiddens = genome[3]
    eta = genome[0]
    lmbd = genome[1]

    candidate = FFNN.FFLayerBrain(hiddens=hiddens, data=data, eta=eta, lmbd=lmbd)
    return candidate


def main():
    pop = Population(LIMIT, SELECTION, CROSSING_OVER, MUTATION, fitness_func,
                     MAX_OFFSPRINGS, RANGE, get_brain)
    print("Population created with size:", len(pop.individuals))
    print("Genomes:", "ETA", "LAMBDA" "REPEATS", "HIDDENS", sep="\t")
    for ind in pop.individuals:
        print(ind)
    log = pop.run(epochs=5, verbose=4, log=1)
    return pop, log


"""
TODO:
- Trained networks should be stored in the Individual objects... Or else the
good weights will be lost and the candidate must be trained again...
"""

if __name__ == '__main__':
    myPop, logz = main()
