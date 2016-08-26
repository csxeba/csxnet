import time

from csxnet.evolution import Population, describe


def fn(ind, queue=None):
    from csxdata.utilities.pure import euclidean
    target = [50] * len(ind.genome)
    ind.fitness = euclidean(ind.genome, target)


def test_evolution():
    limit = 1000
    survivors = 0.4
    crossing_over_rate = 0.2
    mutation_rate = 0.01
    max_offsprings = 3
    epochs = 300
    verbose = 1
    fitness = fn
    genome_len = 10

    ranges = [(0, 100) for _ in range(genome_len)]

    demo_pop = Population(limit=limit,
                          survivors_rate=survivors,
                          crossing_over_rate=crossing_over_rate,
                          mutation_rate=mutation_rate,
                          fitness_function=fitness,
                          max_offsprings=max_offsprings,
                          ranges=ranges,
                          parallel=False)

    print("Population created with {} individuals".format(len(demo_pop.individuals)))
    describe(demo_pop, 3)
    demo_pop.run(epochs, verbose)
    print("Run done.")
    describe(demo_pop, 3)


if __name__ == '__main__':
    start = time.time()
    test_evolution()
    print("Time elapsed: {} s".format(round(time.time()-start, 2)))