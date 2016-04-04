import time

from evolution import Population, describe


def test_evolution():
    def fn1(ind, queue=None):
        from utilities import euclidean
        target = [50] * len(ind.genome)
        ind.fitness = euclidean(ind.genome, target)

    def fn2(ind, queue=None):
        ind.fitness = sum(ind.genome)

    limit = 100
    survivors = 0.4
    crossing_over_rate = 0.1
    mutation_rate = 0.1
    max_offsprings = 3
    epochs = 100
    verbose = 0
    fitness = fn2
    genome_len = 5

    ranges = [(0, 30) for _ in range(genome_len)]

    demo_pop = Population(limit=limit,
                          survivors_rate=survivors,
                          crossing_over_rate=crossing_over_rate,
                          mutation_rate=mutation_rate,
                          fitness_function=fitness,
                          max_offsprings=max_offsprings,
                          ranges=ranges)
    print("Population created with {} individuals".format(len(demo_pop.individuals)))
    describe(demo_pop, 3)
    demo_pop.run(epochs, verbose)
    print("Run done.")
    describe(demo_pop, 3)


if __name__ == '__main__':
    start = time.time()
    test_evolution()
    print("Time elapsed: {} s".format(round(time.time()-start, 2)))
    print("Observation: fitness update is somehow buggy!")
    # TODO: fix above problem in evolution

