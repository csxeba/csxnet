"""Simple Genetic Algorithm from the perspective of a Biologist

Copyright (c) 2016 Gor Csaba

This program is governed by the GPL 3 license and is thus free software:
you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation.

    This program is distributed WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.

See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>..
"""

import time
import random

from csxnet.utilities import avg, feature_scale


# I hereby state that
# this evolutionary algorithm is MINIMIZING the fitness value!

class Population:
    """Model of a population in the ecologigal sense"""

    def __init__(self, limit, survivors_rate, crossing_over_rate, mutation_rate,
                 fitness_function, max_offsprings, ranges, parallel=False):

        self.fitness = fitness_function

        self.limit = limit
        self.survivors = survivors_rate
        self.crossing_over_rate = crossing_over_rate
        self.mutation_rate = mutation_rate
        self.max_offsprings = max_offsprings
        self.ranges = ranges
        self.parallel = parallel

        self.individuals = [Individual(random_genome(ranges)) for _ in
                            range(random.randrange(int(limit / 2), limit))]

        self.update(self.parallel)

    def update(self, parallel):
        if parallel:
            # This branch can make use of multiple CPU cores, to speed things up
            # if the fitness function is computation heavy.
            self._parallel_update()
        else:
            # This branch can be used when the fitness determination is not
            # that complicated.
            # Below line doesn't return anything -> should consider tho
            # self.individuals = list(map(self.fitness, self.individuals))
            for ind in self.individuals:
                if ind.fitness is None:
                    self.fitness(ind, queue=None)

    def _parallel_update(self):
        import multiprocessing as mp

        inds = [ind for ind in self.individuals if not ind.fitness]
        size = len(inds)
        queue = mp.Queue()
        new_inds = []
        jobs = mp.cpu_count() + 1
        while inds:
            procs = []
            some_new_inds = []
            workers = jobs if len(inds) >= jobs else len(inds)

            for _ in range(workers):
                ind = inds.pop()
                procs.append(mp.Process(target=self.fitness, args=(ind, queue)))
            for proc in procs:
                proc.start()
            while len(some_new_inds) != workers:
                some_new_inds.append(queue.get())
                time.sleep(0.1)
            for proc in procs:
                proc.join()

            new_inds.extend(some_new_inds)

        if len(new_inds) != size:
            print("Warning: expected {} individuals, but got {}"
                  .format(size, len(new_inds)))

        self.individuals = list(new_inds)

    def selection(self):
        fitnesses = feature_scale([ind.fitness for ind in self.individuals])
        # Stochastic selection <produces noise!>
        chances = [random.uniform(0.0, f) for f in fitnesses]
        survives = [c < self.survivors for c in chances]
        # Non-stochastic selection
        # survives = [f < self.survivors for f in fitnesses]
        survivors = [ind for srv, ind in zip(survives, self.individuals) if srv]
        self.individuals = survivors

    def reproduction(self):
        """This method generates new individuals by mating existing ones"""

        # A reverse-ordered, feature scaled list is generated from the ind fitnesses
        reproducers = sorted(list(self.individuals), key=lambda ind: ind.fitness, reverse=True)

        # In every round, two individuals with the highest fitnesses reproduce
        while (len(reproducers) > 1) and (len(self.individuals) + self.max_offsprings <= self.limit):
            # Stochastic reproduction
            # ind_a, ind_b = chooseN(reproducers, N=2)
            # Non-stochastic reproduction
            ind_a = reproducers.pop()
            ind_b = reproducers.pop()
            offspr = mate(ind_a, ind_b,  self.crossing_over_rate, self.max_offsprings)
            self.individuals = offspr + self.individuals

    def mutation(self):
        """Generate mutations in the given population. Rate is given by <pop.mutation_rate>"""

        mutations = 0

        # All the loci in the population
        size = len(self.individuals)  # The number of individuals
        loci = len(self.individuals[0].genome)  # The number of loci in a single individual
        all_loci = size * loci  # All loci in the population given the number of chromosomes (T) = 1

        # The chance of mutation_rate is applied to loci and not individuals!
        for i in range(all_loci):
            roll = random.random()
            if roll < (self.mutation_rate / loci):
                no_ind = i // loci
                m_loc = i % loci

                # OK, like, WTF???
                newgenome = list(self.individuals[no_ind].genome)  # the genome gets copied
                newgenome[m_loc] = random_locus(self, m_loc)
                self.individuals[no_ind].genome = newgenome
                # Above snippet is a workaround, because the following won't work:
                # self.individuals[i // loci].genome[m_loc] = random_locus(self, m_loc)
                # It somehow alters the selected individual's genome, but fails to reset
                # its fitness to None. Something earie is going on here...
                self.individuals[no_ind].fitness = None
                self.individuals[no_ind].mutant += 1
                mutations += 1

    def run(self, epochs, verbose, log=False, parallel=None):
        """Runs a given number of epochs: a selection followed by a reproduction"""
        
        if parallel is None:
            parallel = self.parallel

        start = time.time()
        grades = []
        epoch = 0
        for epoch in range(1, epochs + 1):

            self.selection()
            self.reproduction()
            self.mutation()
            self.update(parallel)

            while len(self.individuals) < 4:
                if len(self.individuals) < 2:
                    raise RuntimeError("Population died out. Adjust selection parameters!")
                self.reproduction()
                if len(self.individuals) >= 4:
                    print("Added extra reproduction steps due to low number of individuals!")
                    self.update(parallel)

            if verbose > 0 and epoch % 100 == 0:
                print("Evolutionary epoch:", epoch)
                describe(self, show=1)
            if verbose > 1:
                print("Evolutionary epoch:", epoch)
                describe(self, show=1)

        if verbose:
            print("\n-------------------------------")
            print("Run finished. Epochs:", epoch)
            print("This took", round(time.time() - start, 2), "seconds!")

        print("\n")

        # return the logs
        if log:
            return grades

        return self

    def grade(self):
        """Calculates an average fitness value for the whole population"""
        return avg([ind.fitness for ind in self.individuals])


class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None
        self.mutant = 0


def mate(ind1, ind2, co_chance, max_offsprings):
    """Mate this individual with another"""
    no_offsprings = random.randint(1, max_offsprings + 1)
    offsprings = []
    for _ in range(no_offsprings):
        if random.random() < co_chance:
            newgenomes = crossing_over(ind1.genome, ind2.genome)
            newgenome = random.choice(newgenomes)
        else:
            newgenome = random.choice((ind1.genome, ind2.genome))
        offsprings.append(Individual(newgenome))

    assert any([o.fitness is None for o in offsprings]), "FUUUUUUUUU"

    return offsprings


def crossing_over(chromosome1, chromosome2):
    # Crossing over works similarily to the biological crossing over
    # A point is selected randomly along the loci of the chromosomes,
    # excluding position 0 and the last position (after the last locus).
    # Whether the "head" or "tail" of the chromosome gets swapped is random
    position = random.randrange(len(chromosome1) - 2) + 1
    if random.random() >= 0.5:
        return (chromosome1[:position] + chromosome2[position:],
                chromosome2[:position] + chromosome1[position:])
    else:
        return (chromosome2[:position] + chromosome1[position:],
                chromosome1[:position] + chromosome2[position:])


def random_genome(ranges):
    return [{float: random.uniform, int: random.randrange}[type(t[0])](*t) for t in ranges]  # because python
    # genome = []
    # for t in ranges:
    #     func = {float: random.uniform, int:random.randrange}[type(t[0])]
    #     genome.append(func(*t))
    # return genome


def random_locus(pop, locus):
    ranges = pop.ranges[locus]
    return {float: random.uniform, int: random.randrange}[type(ranges[0])](*ranges)


def mutants(pop):
    holder = [ind.mutant > 0 for ind in pop.individuals]
    return sum(holder) / len(holder)


def gene_loss(pop):
    """Equals the number of different genes divided by
    the initial number of individual genes"""
    lost = round((len(gene_pool(pop)) / len(pop.init_gene_pool)) * 100, 2)
    lost = 100 - lost  # Possibly not?
    return lost


def gene_pool(pop):
    holder = [tuple(x.chromosomeA) for x in pop.individuals] + \
             [tuple(x.chromosomeB) for x in pop.individuals]
    return set(holder)


def describe(pop, show=0):
    """Print out useful information about a population"""
    showme = sorted(pop.individuals, key=lambda i: i.fitness)[:show]
    print("------------------------")
    for ind in showme:
        print("Ind:", str(ind.genome), str(ind.fitness))
    print("Size:\t", len(pop.individuals), sep="\t")
    print("Avg fitness:", pop.grade(), sep="\t")
